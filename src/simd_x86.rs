//! AVX2 implementations of [`crate::encode_vec`] and [`crate::decode_vec`].
//!
//! The scalar encoder and decoder are byte-at-a-time state machines: the next state depends on
//! the current state, which depends on the previous byte. That loop-carried dependency is what
//! stops the compiler from vectorising them, not the width of the registers.
//!
//! Both routines here keep the scalar state machine's structure but move one byte-at-a-time loop
//! into a vector: finding the next sentinel. Everything between two sentinels is copied whole.
//! Keeping the copies at run granularity is the load-bearing part — an earlier version that
//! copied in 32-byte pieces to fuse the scan and the copy into a single pass was 3x slower than
//! this one, because `copy_nonoverlapping` is much cheaper per byte for a 250-byte run than for
//! eight 32-byte ones.
//!
//! Output is byte-for-byte identical to the scalar path, including the error variants and their
//! `decoded_bytes` counts. `tests::differential` asserts that over a corpus of well-formed and
//! malformed frames.

use crate::DecodeError;
use alloc::vec::Vec;
use core::arch::x86_64::*;

/// Whether the AVX2 path can be used.
///
/// With `std` this is a runtime check, so a binary built for baseline x86-64 still gets the fast
/// path on a machine that has AVX2. Without `std` there is no runtime detection available, so it
/// falls back to what the target was compiled for.
#[inline]
pub(crate) fn avx2_available() -> bool {
    #[cfg(feature = "std")]
    {
        std::is_x86_feature_detected!("avx2")
    }
    #[cfg(not(feature = "std"))]
    {
        cfg!(target_feature = "avx2")
    }
}

/// Index of the first zero byte at or after `from`, if any.
///
/// # Safety
/// The caller must have checked for AVX2 support.
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn find_zero(src: *const u8, from: usize, len: usize) -> Option<usize> {
    // SAFETY: the caller guarantees AVX2 support, and every load stays within `len`.
    unsafe {
        let zero = _mm256_setzero_si256();
        let mut i = from;
        while i + 32 <= len {
            let v = _mm256_loadu_si256(src.add(i) as *const __m256i);
            let mask = _mm256_movemask_epi8(_mm256_cmpeq_epi8(v, zero)) as u32;
            if mask != 0 {
                return Some(i + mask.trailing_zeros() as usize);
            }
            i += 32;
        }
        while i < len {
            if *src.add(i) == 0 {
                return Some(i);
            }
            i += 1;
        }
        None
    }
}

/// Index of the first non-zero byte at or after `from`, if any.
///
/// # Safety
/// The caller must have checked for AVX2 support.
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn find_non_zero(src: *const u8, from: usize, len: usize) -> Option<usize> {
    // SAFETY: the caller guarantees AVX2 support, and every load stays within `len`.
    unsafe {
        let zero = _mm256_setzero_si256();
        let mut i = from;
        while i + 32 <= len {
            let v = _mm256_loadu_si256(src.add(i) as *const __m256i);
            let mask = _mm256_movemask_epi8(_mm256_cmpeq_epi8(v, zero)) as u32;
            if mask != u32::MAX {
                return Some(i + (!mask).trailing_zeros() as usize);
            }
            i += 32;
        }
        while i < len {
            if *src.add(i) != 0 {
                return Some(i);
            }
            i += 1;
        }
        None
    }
}

/// Append `len` literal bytes to the group that is currently open, starting a new group whenever
/// one fills up at 254 bytes.
///
/// # Safety
/// `dst` must have room for `len` more bytes plus one code byte per split.
#[inline]
unsafe fn append_literals(
    dst: *mut u8,
    out: &mut usize,
    code_idx: &mut usize,
    run_len: &mut usize,
    mut src: *const u8,
    mut len: usize,
    ended_on_full_run: &mut bool,
) {
    // SAFETY: the caller guarantees `dst` has room for `len` bytes plus one code byte per split.
    unsafe {
        while *run_len + len >= 254 {
            let take = 254 - *run_len;
            core::ptr::copy_nonoverlapping(src, dst.add(*out), take);
            *out += take;
            src = src.add(take);
            len -= take;
            *dst.add(*code_idx) = 0xFF;
            *code_idx = *out;
            *out += 1;
            *run_len = 0;
            *ended_on_full_run = true;
        }
        if len > 0 {
            core::ptr::copy_nonoverlapping(src, dst.add(*out), len);
            *out += len;
            *run_len += len;
            *ended_on_full_run = false;
        }
    }
}

/// AVX2 equivalent of [`crate::encode_vec`].
///
/// # Safety
/// The caller must have checked for AVX2 support.
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn encode_vec(source: &[u8]) -> Vec<u8> {
    // SAFETY: `encoded` was reserved to `max_encoding_length`, which bounds `out`; `src` is read only within `len`.
    unsafe {
        let len = source.len();
        // One code byte per 254 literal bytes, plus the leading one; `max_encoding_length` already
        // accounts for both, and for the empty input encoding to a single `0x01`.
        let mut encoded: Vec<u8> = Vec::with_capacity(crate::max_encoding_length(len));
        let dst = encoded.as_mut_ptr();
        let src = source.as_ptr();

        let mut out = 1;
        let mut code_idx = 0;
        let mut run_len = 0;
        // Whether the last group was closed because it filled up rather than because a zero ended it.
        // A message that ends exactly on a full 254-byte run needs no further group, which is the one
        // byte COBS saves in that case.
        let mut ended_on_full_run = false;
        let mut i = 0;

        loop {
            match find_zero(src, i, len) {
                Some(zero_at) => {
                    append_literals(
                        dst,
                        &mut out,
                        &mut code_idx,
                        &mut run_len,
                        src.add(i),
                        zero_at - i,
                        &mut ended_on_full_run,
                    );
                    *dst.add(code_idx) = (run_len + 1) as u8;
                    code_idx = out;
                    out += 1;
                    run_len = 0;
                    ended_on_full_run = false;
                    i = zero_at + 1;
                }
                None => {
                    append_literals(
                        dst,
                        &mut out,
                        &mut code_idx,
                        &mut run_len,
                        src.add(i),
                        len - i,
                        &mut ended_on_full_run,
                    );
                    break;
                }
            }
        }

        if ended_on_full_run && run_len == 0 {
            out -= 1;
        } else {
            *dst.add(code_idx) = (run_len + 1) as u8;
        }
        // `out` transiently reaches `max_encoding_length(len) + 1` when the message ends exactly
        // on a full 254-byte run, but that only bumps the counter: the speculative code byte is
        // never written, and the branch above takes the byte back.
        debug_assert!(out <= encoded.capacity());
        encoded.set_len(out);
        encoded
    }
}

/// AVX2 equivalent of [`crate::decode_vec`].
///
/// # Safety
/// The caller must have checked for AVX2 support.
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn decode_vec(source: &[u8]) -> Result<Vec<u8>, DecodeError> {
    // SAFETY: `decoded` was reserved to `end - start`, which bounds `out`; every read stays below `end <= len`.
    unsafe {
        let len = source.len();
        if len == 0 {
            return Err(DecodeError::EmptyFrame);
        }
        let src = source.as_ptr();

        // The streaming decoder stays idle on leading sentinels rather than treating them as an
        // end of frame, so a frame that starts with one still decodes.
        let start = match find_non_zero(src, 0, len) {
            Some(s) => s,
            None => return Err(DecodeError::InvalidFrame { decoded_bytes: 0 }),
        };
        let end = find_zero(src, start, len).unwrap_or(len);

        let mut decoded: Vec<u8> = Vec::with_capacity(end - start);
        let dst = decoded.as_mut_ptr();
        let mut out = 0;
        let mut i = start;

        while i < end {
            let code = *src.add(i) as usize;
            let run = code - 1;
            let from = i + 1;
            if from + run > end {
                // Same as the scalar decoder: report what had already been decoded.
                let available = end - from;
                core::ptr::copy_nonoverlapping(src.add(from), dst.add(out), available);
                out += available;
                decoded.set_len(out);
                return Err(DecodeError::InvalidFrame { decoded_bytes: out });
            }
            core::ptr::copy_nonoverlapping(src.add(from), dst.add(out), run);
            out += run;
            i = from + run;
            if code != 0xFF && i < end {
                *dst.add(out) = 0;
                out += 1;
            }
        }

        debug_assert!(out <= decoded.capacity());
        decoded.set_len(out);
        Ok(decoded)
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;
    use alloc::vec::Vec;

    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            let mut x = self.0;
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            self.0 = x;
            x
        }
        fn byte(&mut self) -> u8 {
            (self.next() & 0xFF) as u8
        }
    }

    /// Inputs chosen to cover every shape that has a different code path: the empty message, a
    /// message ending exactly on a full 254-byte run (with and without an earlier zero, which is
    /// what decides whether the final group is elided), sentinels straddling a 32-byte vector
    /// boundary, and all-zero and all-0xFF messages.
    fn corpus() -> Vec<Vec<u8>> {
        let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
        let mut out = Vec::new();
        for n in 0..=600 {
            out.push((0..n).map(|_| rng.byte()).collect());
            out.push(vec![0u8; n]);
            out.push(vec![0xFFu8; n]);
            out.push(
                (0..n)
                    .map(|_| {
                        if rng.next() % 4 == 0 {
                            0
                        } else {
                            rng.byte() | 1
                        }
                    })
                    .collect(),
            );
        }
        for k in 1..=6 {
            out.push(vec![0xABu8; 254 * k]);
            for lead in [1, 2, 7, 100] {
                let mut v = vec![0u8; lead];
                v.extend(core::iter::repeat_n(0xCDu8, 254 * k));
                out.push(v);
            }
        }
        for pos in 0..80 {
            let mut v = vec![7u8; 200];
            v[pos] = 0;
            out.push(v);
        }
        out
    }

    #[test]
    fn differential() {
        if !super::avx2_available() {
            return;
        }
        let mut rng = Rng(0x2545_F491_4F6C_DD1D);
        for source in corpus() {
            let want = crate::enc::encode_vec_scalar(&source);
            let got = unsafe { super::encode_vec(&source) };
            assert_eq!(got, want, "encode diverged for {} bytes", source.len());
            assert!(!got.contains(&0), "encoded frame contains a sentinel");

            assert_eq!(unsafe { super::decode_vec(&got) }, Ok(source.clone()));

            // Malformed frames must fail the same way, including the byte count in the error.
            for _ in 0..4 {
                if got.is_empty() {
                    break;
                }
                let mut broken = got.clone();
                let at = (rng.next() as usize) % broken.len();
                broken[at] = rng.byte();
                assert_eq!(
                    unsafe { super::decode_vec(&broken) },
                    crate::dec::decode_vec_scalar(&broken),
                    "decode diverged for a corrupted {}-byte frame",
                    broken.len()
                );
            }
        }
    }

    #[test]
    fn boundary_frames() {
        if !super::avx2_available() {
            return;
        }
        for frame in [
            vec![],
            vec![0x00],
            vec![0x00, 0x01],
            vec![0x00, 0x03, 1, 2],
            vec![0x03, 1, 2],
            vec![0x03, 1, 2, 0x00],
            vec![0x03, 1, 2, 0x00, 0x03, 3, 4],
            vec![0x05, 1, 2],
            vec![0x01],
            vec![0x01, 0x01],
            vec![0x02, 7],
        ] {
            assert_eq!(
                unsafe { super::decode_vec(&frame) },
                crate::dec::decode_vec_scalar(&frame),
                "decode diverged for {frame:?}"
            );
        }
    }
}
