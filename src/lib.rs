//! A dependency-free, `#![no_std]` decoder for the [QOI image format].
//!
//! # no_std
//!
//! The crate depends only on `core`. It performs no heap allocation and uses
//! no `unsafe`, so it can be used from bare-metal, kernel, and other `no_std`
//! environments.
//!
//! # Usage
//!
//! The caller owns the output buffer. [`decode`] writes tightly packed RGB
//! (`RGBRGBRGB…`) or RGBA (`RGBARGBARGBA…`) bytes into it. The image header
//! declares `width`, `height`, and a channel count of 3 or 4, so a buffer of
//! `width * height * channels` bytes is required. If the provided buffer is
//! too small, [`Error::OutputTooSmall`] is returned.
//!
//! ```
//! let qoi: &[u8] = &[
//!     b'q', b'o', b'i', b'f',
//!     0, 0, 0, 1,
//!     0, 0, 0, 1,
//!     3,
//!     0,
//!     0xFE, 255, 0, 0,
//!     0, 0, 0, 0, 0, 0, 0, 1,
//! ];
//!
//! let mut pixels = [0u8; 1 * 1 * 3];
//! let info = qoi::decode(qoi, &mut pixels)?;
//!
//! assert_eq!(info.width, 1);
//! assert_eq!(info.height, 1);
//! assert_eq!(info.channels, 3);
//! assert_eq!(pixels, [255, 0, 0]);
//! # Ok::<(), qoi::Error>(())
//! ```
//!
//! [QOI image format]: https://qoiformat.org/

#![no_std]
#![forbid(unsafe_code)]

#[cfg(test)]
extern crate std;

use core::fmt;

const QOI_MAGIC: u32 = u32::from_be_bytes(*b"qoif");

const QOI_OP_INDEX: u8 = 0x00;
const QOI_OP_DIFF: u8 = 0x40;
const QOI_OP_LUMA: u8 = 0x80;
const QOI_OP_RUN: u8 = 0xC0;
const QOI_OP_RGB: u8 = 0xFE;
const QOI_OP_RGBA: u8 = 0xFF;

const QOI_END_MARKER: [u8; 8] = [0, 0, 0, 0, 0, 0, 0, 1];

/// Description of a successfully decoded QOI image.
///
/// The channel count is taken from the image header and is either 3 (RGB) or
/// 4 (RGBA).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ImageInfo {
    /// Image width in pixels.
    pub width: u32,
    /// Image height in pixels.
    pub height: u32,
    /// Channel count from the header: 3 for RGB, 4 for RGBA.
    pub channels: u8,
    /// Colorspace from the header: 0 = sRGB with linear alpha, 1 = all linear.
    pub colorspace: u8,
    /// Number of bytes written into the output buffer
    /// (`width * height * channels`).
    pub output_bytes: usize,
}

impl ImageInfo {
    /// Returns true when the image uses 3 channels (RGB).
    pub const fn is_rgb(&self) -> bool {
        self.channels == 3
    }

    /// Returns true when the image uses 4 channels (RGBA).
    pub const fn is_rgba(&self) -> bool {
        self.channels == 4
    }
}

/// Errors returned by [`decode`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Error {
    /// The four magic bytes are not "qoif".
    InvalidMagic,
    /// The image width or height is zero.
    InvalidDimensions,
    /// The header channel count is not 3 or 4.
    InvalidChannelCount(u8),
    /// The header colorspace value is not 0 or 1.
    InvalidColorSpace(u8),
    /// The output buffer is smaller than `width * height * channels`.
    OutputTooSmall,
    /// The input ended before all required bytes were read.
    TruncatedInput,
    /// The eight-byte end marker is missing or does not match the spec.
    InvalidEndMarker,
    /// The encoded stream declares more pixels than the header allows.
    MalformedStream,
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::InvalidMagic => f.write_str("invalid QOI magic bytes"),
            Error::InvalidDimensions => f.write_str("image width and height must be non-zero"),
            Error::InvalidChannelCount(channels) => {
                write!(f, "invalid channel count {channels}, expected 3 or 4")
            }
            Error::InvalidColorSpace(colorspace) => {
                write!(f, "invalid colorspace value {colorspace}, expected 0 or 1")
            }
            Error::OutputTooSmall => f.write_str("output buffer is too small for the image"),
            Error::TruncatedInput => f.write_str("unexpected end of input"),
            Error::InvalidEndMarker => f.write_str("missing or invalid end marker"),
            Error::MalformedStream => f.write_str("malformed QOI stream"),
        }
    }
}

impl core::error::Error for Error {}

/// Decodes a QOI image from `input` into `output`.
///
/// `output` must be at least `width * height * channels` bytes long, where
/// the dimensions and channel count come from the image header. The decoded
/// pixels are written as tightly packed RGB or RGBA bytes. No memory is
/// allocated.
///
/// Returns a [`ImageInfo`] describing the image on success, or an [`Error`]
/// if the stream is malformed or `output` is too small.
pub fn decode(input: &[u8], output: &mut [u8]) -> Result<ImageInfo, Error> {
    let mut reader = Reader::new(input);

    if reader.read_u32_be()? != QOI_MAGIC {
        return Err(Error::InvalidMagic);
    }
    let width = reader.read_u32_be()?;
    let height = reader.read_u32_be()?;
    let header_channels = reader.read_u8()?;
    let colorspace = reader.read_u8()?;

    if width == 0 || height == 0 {
        return Err(Error::InvalidDimensions);
    }
    if header_channels != 3 && header_channels != 4 {
        return Err(Error::InvalidChannelCount(header_channels));
    }
    if colorspace > 1 {
        return Err(Error::InvalidColorSpace(colorspace));
    }

    let pixels = u64::from(width) * u64::from(height);
    let needed = match pixels.checked_mul(u64::from(header_channels)) {
        Some(needed) => needed,
        None => return Err(Error::OutputTooSmall),
    };
    if needed > output.len() as u64 {
        return Err(Error::OutputTooSmall);
    }

    let channels = usize::from(header_channels);
    let mut remaining = pixels;
    let mut out_pos = 0usize;
    let mut index = [Pixel::ZERO; 64];
    let mut px = Pixel::FIRST;

    while remaining > 0 {
        let byte = reader.read_u8()?;
        let mut run = 1u64;

        match byte {
            QOI_OP_RGB => {
                px.r = reader.read_u8()?;
                px.g = reader.read_u8()?;
                px.b = reader.read_u8()?;
            }
            QOI_OP_RGBA => {
                px.r = reader.read_u8()?;
                px.g = reader.read_u8()?;
                px.b = reader.read_u8()?;
                px.a = reader.read_u8()?;
            }
            QOI_OP_INDEX..=0x3F => {
                px = index[byte as usize];
            }
            QOI_OP_DIFF..=0x7F => {
                let dr = ((byte >> 4) & 0x03) as i8 - 2;
                let dg = ((byte >> 2) & 0x03) as i8 - 2;
                let db = (byte & 0x03) as i8 - 2;
                px.r = px.r.wrapping_add(dr as u8);
                px.g = px.g.wrapping_add(dg as u8);
                px.b = px.b.wrapping_add(db as u8);
            }
            QOI_OP_LUMA..=0xBF => {
                let byte2 = reader.read_u8()?;
                let dg = (byte & 0x3F) as i8 - 32;
                let dr = dg + ((byte2 >> 4) as i8 - 8);
                let db = dg + ((byte2 & 0x0F) as i8 - 8);
                px.r = px.r.wrapping_add(dr as u8);
                px.g = px.g.wrapping_add(dg as u8);
                px.b = px.b.wrapping_add(db as u8);
            }
            QOI_OP_RUN..=0xFD => {
                run = u64::from(byte & 0x3F) + 1;
            }
        }

        if run > remaining {
            return Err(Error::MalformedStream);
        }

        index[index_pos(&px)] = px;

        for _ in 0..run {
            write_pixel(output, out_pos, &px, channels);
            out_pos += channels;
        }

        remaining -= run;
    }

    let end_marker = &input[reader.pos..];
    if end_marker.len() < QOI_END_MARKER.len()
        || end_marker[..QOI_END_MARKER.len()] != QOI_END_MARKER
    {
        return Err(Error::InvalidEndMarker);
    }

    Ok(ImageInfo {
        width,
        height,
        channels: header_channels,
        colorspace,
        output_bytes: out_pos,
    })
}

#[derive(Clone, Copy)]
struct Pixel {
    r: u8,
    g: u8,
    b: u8,
    a: u8,
}

impl Pixel {
    const ZERO: Pixel = Pixel {
        r: 0,
        g: 0,
        b: 0,
        a: 0,
    };
    const FIRST: Pixel = Pixel {
        r: 0,
        g: 0,
        b: 0,
        a: 255,
    };
}

fn index_pos(px: &Pixel) -> usize {
    let hash =
        u32::from(px.r) * 3 + u32::from(px.g) * 5 + u32::from(px.b) * 7 + u32::from(px.a) * 11;
    (hash as usize) & 63
}

fn write_pixel(out: &mut [u8], pos: usize, px: &Pixel, channels: usize) {
    out[pos] = px.r;
    out[pos + 1] = px.g;
    out[pos + 2] = px.b;
    if channels == 4 {
        out[pos + 3] = px.a;
    }
}

struct Reader<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Reader { bytes, pos: 0 }
    }

    fn read_u8(&mut self) -> Result<u8, Error> {
        match self.bytes.get(self.pos) {
            Some(&byte) => {
                self.pos += 1;
                Ok(byte)
            }
            None => Err(Error::TruncatedInput),
        }
    }

    fn read_u32_be(&mut self) -> Result<u32, Error> {
        let b0 = self.read_u8()?;
        let b1 = self.read_u8()?;
        let b2 = self.read_u8()?;
        let b3 = self.read_u8()?;
        Ok(u32::from_be_bytes([b0, b1, b2, b3]))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::vec::Vec;

    fn header(width: u32, height: u32, channels: u8, colorspace: u8) -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"qoif");
        bytes.extend_from_slice(&width.to_be_bytes());
        bytes.extend_from_slice(&height.to_be_bytes());
        bytes.push(channels);
        bytes.push(colorspace);
        bytes
    }

    fn image(width: u32, height: u32, channels: u8, colorspace: u8, chunks: &[u8]) -> Vec<u8> {
        let mut bytes = header(width, height, channels, colorspace);
        bytes.extend_from_slice(chunks);
        bytes.extend_from_slice(&[0, 0, 0, 0, 0, 0, 0, 1]);
        bytes
    }

    fn rgb_image(width: u32, height: u32, chunks: &[u8]) -> Vec<u8> {
        image(width, height, 3, 0, chunks)
    }

    fn rgba_image(width: u32, height: u32, chunks: &[u8]) -> Vec<u8> {
        image(width, height, 4, 0, chunks)
    }

    fn decode_into(bytes: &[u8], capacity: usize) -> Result<(ImageInfo, Vec<u8>), Error> {
        let mut out = std::vec![0u8; capacity];
        match decode(bytes, &mut out) {
            Ok(info) => {
                out.truncate(info.output_bytes);
                Ok((info, out))
            }
            Err(error) => Err(error),
        }
    }

    #[test]
    fn parses_header_of_rgb_image() {
        let data = rgb_image(1, 1, &[QOI_OP_RGB, 255, 0, 0]);
        let (info, pixels) = decode_into(&data, 4).unwrap();
        assert_eq!(info.width, 1);
        assert_eq!(info.height, 1);
        assert_eq!(info.channels, 3);
        assert_eq!(info.colorspace, 0);
        assert_eq!(info.output_bytes, 3);
        assert!(info.is_rgb());
        assert!(!info.is_rgba());
        assert_eq!(pixels, [255, 0, 0]);
    }

    #[test]
    fn parses_header_of_rgba_image() {
        let chunks = [QOI_OP_RGBA, 10, 20, 30, 40, 0xC4];
        let data = rgba_image(2, 3, &chunks);
        let (info, pixels) = decode_into(&data, 24).unwrap();
        assert_eq!(info.width, 2);
        assert_eq!(info.height, 3);
        assert_eq!(info.channels, 4);
        assert!(info.is_rgba());
        assert!(!info.is_rgb());
        assert_eq!(pixels.len(), 24);
    }

    #[test]
    fn decodes_rgb_image() {
        let chunks = [
            QOI_OP_RGB, 10, 20, 30, QOI_OP_RGB, 40, 50, 60, QOI_OP_RGB, 70, 80, 90, QOI_OP_RGB,
            100, 110, 120,
        ];
        let data = rgb_image(2, 2, &chunks);
        let (info, pixels) = decode_into(&data, 12).unwrap();
        assert_eq!(info.output_bytes, 12);
        assert_eq!(pixels, [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120]);
    }

    #[test]
    fn decodes_rgba_image() {
        let chunks = [QOI_OP_RGBA, 10, 20, 30, 40, QOI_OP_RGBA, 50, 60, 70, 80];
        let data = rgba_image(2, 1, &chunks);
        let (info, pixels) = decode_into(&data, 8).unwrap();
        assert_eq!(info.output_bytes, 8);
        assert_eq!(pixels, [10, 20, 30, 40, 50, 60, 70, 80]);
    }

    #[test]
    fn op_rgb_preserves_alpha() {
        let chunks = [QOI_OP_RGBA, 10, 20, 30, 40, QOI_OP_RGB, 200, 200, 200];
        let data = rgba_image(2, 1, &chunks);
        let (_, pixels) = decode_into(&data, 8).unwrap();
        assert_eq!(pixels, [10, 20, 30, 40, 200, 200, 200, 40]);
    }

    #[test]
    fn op_index() {
        let data = rgb_image(2, 1, &[QOI_OP_RGB, 100, 100, 100, 0x11]);
        let (_, pixels) = decode_into(&data, 6).unwrap();
        assert_eq!(pixels, [100, 100, 100, 100, 100, 100]);
    }

    #[test]
    fn op_index_initial_slot_is_zero_pixel() {
        let data = rgb_image(1, 1, &[0x00]);
        let (_, pixels) = decode_into(&data, 3).unwrap();
        assert_eq!(pixels, [0, 0, 0]);
    }

    #[test]
    fn op_diff() {
        let data = rgb_image(2, 1, &[QOI_OP_RGB, 128, 128, 128, 0x72]);
        let (_, pixels) = decode_into(&data, 6).unwrap();
        assert_eq!(pixels, [128, 128, 128, 129, 126, 128]);
    }

    #[test]
    fn op_diff_wraps_around() {
        let data = rgb_image(2, 1, &[QOI_OP_RGB, 0, 1, 255, 0x57]);
        let (_, pixels) = decode_into(&data, 6).unwrap();
        assert_eq!(pixels, [0, 1, 255, 255, 0, 0]);
    }

    #[test]
    fn op_luma() {
        let data = rgb_image(2, 1, &[QOI_OP_RGB, 100, 100, 100, 0xAA, 0x23]);
        let (_, pixels) = decode_into(&data, 6).unwrap();
        assert_eq!(pixels, [100, 100, 100, 104, 110, 105]);
    }

    #[test]
    fn op_run_repeats_pixels() {
        let data = rgb_image(5, 1, &[QOI_OP_RGB, 50, 60, 70, 0xC3]);
        let (_, pixels) = decode_into(&data, 15).unwrap();
        assert_eq!(
            pixels,
            [50, 60, 70, 50, 60, 70, 50, 60, 70, 50, 60, 70, 50, 60, 70]
        );
    }

    #[test]
    fn op_run_max_length() {
        let data = rgb_image(63, 1, &[QOI_OP_RGB, 50, 60, 70, 0xFD]);
        let (info, pixels) = decode_into(&data, 63 * 3).unwrap();
        assert_eq!(info.output_bytes, 63 * 3);
        assert_eq!(pixels.len(), 63 * 3);
        assert!(
            pixels
                .iter()
                .enumerate()
                .all(|(i, &v)| v == [50, 60, 70][i % 3])
        );
    }

    #[test]
    fn op_run_exactly_fills_image() {
        let data = rgb_image(3, 1, &[QOI_OP_RGB, 1, 2, 3, 0xC1]);
        let (_, pixels) = decode_into(&data, 9).unwrap();
        assert_eq!(pixels, [1, 2, 3, 1, 2, 3, 1, 2, 3]);
    }

    #[test]
    fn op_run_overrun_is_malformed() {
        let data = rgb_image(2, 1, &[QOI_OP_RGB, 50, 60, 70, 0xFD]);
        let mut out = [0u8; 6];
        assert_eq!(decode(&data, &mut out), Err(Error::MalformedStream));
    }

    #[test]
    fn mixed_opcodes() {
        let chunks = [
            QOI_OP_RGB,
            10,
            20,
            30,
            0x73,
            0xA7,
            0x50,
            0x31,
            0xC2,
            QOI_OP_RGBA,
            1,
            2,
            3,
            4,
        ];
        let data = rgba_image(8, 1, &chunks);
        let (_, pixels) = decode_into(&data, 32).unwrap();
        assert_eq!(
            pixels,
            [
                10, 20, 30, 255, 11, 18, 31, 255, 15, 25, 30, 255, 15, 25, 30, 255, 15, 25, 30,
                255, 15, 25, 30, 255, 15, 25, 30, 255, 1, 2, 3, 4,
            ]
        );
    }

    #[test]
    fn solid_color_image() {
        let data = rgb_image(8, 1, &[QOI_OP_RGB, 200, 100, 50, 0xC6]);
        let (_, pixels) = decode_into(&data, 24).unwrap();
        assert_eq!(
            pixels,
            [
                200, 100, 50, 200, 100, 50, 200, 100, 50, 200, 100, 50, 200, 100, 50, 200, 100, 50,
                200, 100, 50, 200, 100, 50
            ]
        );
    }

    #[test]
    fn small_multi_pixel_image() {
        let data = rgb_image(
            3,
            2,
            &[
                QOI_OP_RGB, 255, 0, 0, QOI_OP_RGB, 0, 255, 0, QOI_OP_RGB, 0, 0, 255, QOI_OP_RGB,
                255, 255, 0, QOI_OP_RGB, 255, 0, 255, QOI_OP_RGB, 0, 255, 255,
            ],
        );
        let (info, pixels) = decode_into(&data, 18).unwrap();
        assert_eq!(info.width, 3);
        assert_eq!(info.height, 2);
        assert_eq!(
            pixels,
            [
                255, 0, 0, 0, 255, 0, 0, 0, 255, 255, 255, 0, 255, 0, 255, 0, 255, 255,
            ]
        );
    }

    #[test]
    fn truncated_header_is_error() {
        let full = header(1, 1, 3, 0);
        for len in 0..full.len() {
            let mut out = [0u8; 16];
            assert_eq!(
                decode(&full[..len], &mut out),
                Err(Error::TruncatedInput),
                "len = {len}"
            );
        }
    }

    #[test]
    fn truncated_pixel_data_is_error() {
        let data = rgb_image(2, 1, &[QOI_OP_RGB, 1, 2, 3, QOI_OP_RGB, 4, 5, 6]);
        let mut out = [0u8; 16];
        for len in 0..data.len() - 8 {
            assert_eq!(
                decode(&data[..len], &mut out),
                Err(Error::TruncatedInput),
                "len = {len}"
            );
        }
    }

    #[test]
    fn invalid_magic_is_error() {
        let mut data = rgb_image(1, 1, &[QOI_OP_RGB, 1, 2, 3]);
        data[0] = b'x';
        let mut out = [0u8; 3];
        assert_eq!(decode(&data, &mut out), Err(Error::InvalidMagic));
    }

    #[test]
    fn invalid_channel_count_is_error() {
        for channels in [2u8, 5, 0, 255] {
            let data = image(1, 1, channels, 0, &[]);
            let mut out = [0u8; 3];
            assert_eq!(
                decode(&data, &mut out),
                Err(Error::InvalidChannelCount(channels)),
                "channels = {channels}"
            );
        }
    }

    #[test]
    fn invalid_colorspace_is_error() {
        let data = image(1, 1, 3, 2, &[]);
        let mut out = [0u8; 3];
        assert_eq!(decode(&data, &mut out), Err(Error::InvalidColorSpace(2)));
    }

    #[test]
    fn zero_dimensions_are_invalid() {
        let data = image(0, 1, 3, 0, &[]);
        let mut out = [0u8; 3];
        assert_eq!(decode(&data, &mut out), Err(Error::InvalidDimensions));

        let data = image(1, 0, 3, 0, &[]);
        let mut out = [0u8; 3];
        assert_eq!(decode(&data, &mut out), Err(Error::InvalidDimensions));
    }

    #[test]
    fn output_buffer_too_small() {
        let data = rgb_image(2, 1, &[QOI_OP_RGB, 1, 2, 3, QOI_OP_RGB, 4, 5, 6]);
        let mut out = [0u8; 5];
        assert_eq!(decode(&data, &mut out), Err(Error::OutputTooSmall));

        let data = rgba_image(1, 1, &[QOI_OP_RGBA, 1, 2, 3, 4]);
        let mut out = [0u8; 3];
        assert_eq!(decode(&data, &mut out), Err(Error::OutputTooSmall));
    }

    #[test]
    fn invalid_end_marker_is_error() {
        let mut data = rgb_image(1, 1, &[QOI_OP_RGB, 255, 0, 0]);
        let last = data.len() - 1;
        data[last] = 2;
        let mut out = [0u8; 3];
        assert_eq!(decode(&data, &mut out), Err(Error::InvalidEndMarker));

        let mut data = rgb_image(1, 1, &[QOI_OP_RGB, 255, 0, 0]);
        let padding_byte = data.len() - 5;
        data[padding_byte] = 1;
        let mut out = [0u8; 3];
        assert_eq!(decode(&data, &mut out), Err(Error::InvalidEndMarker));
    }

    #[test]
    fn missing_end_marker_is_error() {
        let mut data = rgb_image(1, 1, &[QOI_OP_RGB, 255, 0, 0]);
        data.truncate(data.len() - 4);
        let mut out = [0u8; 3];
        assert_eq!(decode(&data, &mut out), Err(Error::InvalidEndMarker));
    }

    #[test]
    fn huge_dimensions_do_not_panic() {
        let data = image(u32::MAX, u32::MAX, 4, 0, &[]);
        let mut out = [0u8; 16];
        assert_eq!(decode(&data, &mut out), Err(Error::OutputTooSmall));

        let data = image(u32::MAX, 2, 3, 0, &[]);
        let mut out = [0u8; 16];
        assert_eq!(decode(&data, &mut out), Err(Error::OutputTooSmall));

        let data = image(1, u32::MAX, 4, 0, &[]);
        let mut out = [0u8; 16];
        assert_eq!(decode(&data, &mut out), Err(Error::OutputTooSmall));
    }

    const REFERENCE_RGBA: &[u8] = &[
        0x71, 0x6F, 0x69, 0x66, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0x04, 0x04, 0x00, 0xFE,
        0x0A, 0x14, 0x1E, 0xA5, 0x52, 0xFE, 0xC8, 0x64, 0x32, 0xC0, 0x09, 0xC1, 0xFF, 0x3C, 0x3C,
        0x3C, 0x80, 0xA3, 0x73, 0xC0, 0x00, 0xC0, 0xFF, 0x64, 0x96, 0xC8, 0xFF, 0x9F, 0xCA, 0xFF,
        0x01, 0x02, 0x03, 0x04, 0xC0, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01,
    ];
    const REFERENCE_RGBA_PIXELS: &[u8] = &[
        0x0A, 0x14, 0x1E, 0xFF, 0x0C, 0x19, 0x1D, 0xFF, 0xC8, 0x64, 0x32, 0xFF, 0xC8, 0x64, 0x32,
        0xFF, 0x0A, 0x14, 0x1E, 0xFF, 0x0A, 0x14, 0x1E, 0xFF, 0x0A, 0x14, 0x1E, 0xFF, 0x3C, 0x3C,
        0x3C, 0x80, 0x3E, 0x3F, 0x3A, 0x80, 0x3E, 0x3F, 0x3A, 0x80, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x64, 0x96, 0xC8, 0xFF, 0x67, 0x95, 0xC9, 0xFF, 0x01, 0x02, 0x03, 0x04,
        0x01, 0x02, 0x03, 0x04,
    ];
    const REFERENCE_RGB: &[u8] = &[
        0x71, 0x6F, 0x69, 0x66, 0x00, 0x00, 0x00, 0x05, 0x00, 0x00, 0x00, 0x03, 0x03, 0x01, 0xFE,
        0x00, 0xC8, 0x00, 0xFE, 0x25, 0xBB, 0x32, 0xFE, 0x4A, 0xAE, 0x64, 0xFE, 0x6F, 0xA1, 0x96,
        0xFE, 0x94, 0x94, 0xC8, 0xFE, 0xB9, 0x87, 0x00, 0xFE, 0xDE, 0x7A, 0x32, 0xFE, 0x03, 0x6D,
        0x64, 0xFE, 0x28, 0x60, 0x96, 0xFE, 0x4D, 0x53, 0xC8, 0xFE, 0x72, 0x46, 0x00, 0xFE, 0x97,
        0x39, 0x32, 0xFE, 0xBC, 0x2C, 0x64, 0xFE, 0xE1, 0x1F, 0x96, 0xFE, 0x06, 0x12, 0xC8, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01,
    ];
    const REFERENCE_RGB_PIXELS: &[u8] = &[
        0x00, 0xC8, 0x00, 0x25, 0xBB, 0x32, 0x4A, 0xAE, 0x64, 0x6F, 0xA1, 0x96, 0x94, 0x94, 0xC8,
        0xB9, 0x87, 0x00, 0xDE, 0x7A, 0x32, 0x03, 0x6D, 0x64, 0x28, 0x60, 0x96, 0x4D, 0x53, 0xC8,
        0x72, 0x46, 0x00, 0x97, 0x39, 0x32, 0xBC, 0x2C, 0x64, 0xE1, 0x1F, 0x96, 0x06, 0x12, 0xC8,
    ];

    #[test]
    fn decodes_reference_encoded_rgba_image() {
        let mut out = [0u8; 4 * 4 * 4];
        let info = decode(REFERENCE_RGBA, &mut out).unwrap();
        assert_eq!(info.width, 4);
        assert_eq!(info.height, 4);
        assert_eq!(info.channels, 4);
        assert_eq!(info.colorspace, 0);
        assert_eq!(&out, REFERENCE_RGBA_PIXELS);
    }

    #[test]
    fn decodes_reference_encoded_rgb_image() {
        let mut out = [0u8; 5 * 3 * 3];
        let info = decode(REFERENCE_RGB, &mut out).unwrap();
        assert_eq!(info.width, 5);
        assert_eq!(info.height, 3);
        assert_eq!(info.channels, 3);
        assert_eq!(info.colorspace, 1);
        assert_eq!(&out, REFERENCE_RGB_PIXELS);
    }
}
