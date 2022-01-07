// Copyright 2020+ Klaus Post. All rights reserved.
// License information can be found in the LICENSE file.

package zstd

import (
	"bytes"
	"errors"
	"io"
)

// HeaderMaxSize is the maximum size of a Frame and Block Header.
// If less is sent to Header.Decode it *may* still contain enough information.
const HeaderMaxSize = 14 + 3

// Header contains information about the first frame and block within that.
type Header struct {
	// Window Size the window of data to keep while decoding.
	// Will only be set if HasFCS is false.
	WindowSize uint64

	// Frame content size.
	// Expected size of the entire frame.
	FrameContentSize uint64

	// Dictionary ID.
	// If 0, no dictionary.
	DictionaryID uint32

	// First block information.
	FirstBlock struct {
		// OK will be set if first block could be decoded.
		OK bool

		// Is this the last block of a frame?
		Last bool

		// Is the data compressed?
		// If true CompressedSize will be populated.
		// Unfortunately DecompressedSize cannot be determined
		// without decoding the blocks.
		Compressed bool

		// DecompressedSize is the expected decompressed size of the block.
		// Will be 0 if it cannot be determined.
		DecompressedSize int

		// CompressedSize of the data in the block.
		// Does not include the block header.
		// Will be equal to DecompressedSize if not Compressed.
		CompressedSize int
	}

	// Skippable will be true if the frame is meant to be skipped.
	// No other information will be populated.
	Skippable bool

	// If set there is a checksum present for the block content.
	HasCheckSum bool

	// If this is true FrameContentSize will have a valid value
	HasFCS bool

	SingleSegment bool
}

// Decode the header from the beginning of the stream.
// This will decode the frame header and the first block header if enough bytes are provided.
// It is recommended to provide at least HeaderMaxSize bytes.
// If the frame header cannot be read an error will be returned.
// If there isn't enough input, io.ErrUnexpectedEOF is returned.
// The FirstBlock.OK will indicate if enough information was available to decode the first block header.
func (h *Header) Decode(in []byte) error {
	if len(in) < 4 {
		return io.ErrUnexpectedEOF
	}
	b, in := in[:4], in[4:]
	if !bytes.Equal(b, frameMagic) {
		if !bytes.Equal(b[1:4], skippableFrameMagic) || b[0]&0xf0 != 0x50 {
			return ErrMagicMismatch
		}
		*h = Header{Skippable: true}
		return nil
	}
	if len(in) < 1 {
		return io.ErrUnexpectedEOF
	}

	// Clear output
	*h = Header{}
	fhd, in := in[0], in[1:]
	h.SingleSegment = fhd&(1<<5) != 0
	h.HasCheckSum = fhd&(1<<2) != 0

	if fhd&(1<<3) != 0 {
		return errors.New("reserved bit set on frame header")
	}

	// Read Window_Descriptor
	// https://github.com/facebook/zstd/blob/dev/doc/zstd_compression_format.md#window_descriptor
	if !h.SingleSegment {
		if len(in) < 1 {
			return io.ErrUnexpectedEOF
		}
		var wd byte
		wd, in = in[0], in[1:]
		windowLog := 10 + (wd >> 3)
		windowBase := uint64(1) << windowLog
		windowAdd := (windowBase / 8) * uint64(wd&0x7)
		h.WindowSize = windowBase + windowAdd
	}

	// Read Dictionary_ID
	// https://github.com/facebook/zstd/blob/dev/doc/zstd_compression_format.md#dictionary_id
	if size := fhd & 3; size != 0 {
		if size == 3 {
			size = 4
		}
		if len(in) < int(size) {
			return io.ErrUnexpectedEOF
		}
		b, in = in[:size], in[size:]
		if b == nil {
			return io.ErrUnexpectedEOF
		}
		switch size {
		case 1:
			h.DictionaryID = uint32(b[0])
		case 2:
			h.DictionaryID = uint32(b[0]) | (uint32(b[1]) << 8)
		case 4:
			h.DictionaryID = uint32(b[0]) | (uint32(b[1]) << 8) | (uint32(b[2]) << 16) | (uint32(b[3]) << 24)
		}
	}

	// Read Frame_Content_Size
	// https://github.com/facebook/zstd/blob/dev/doc/zstd_compression_format.md#frame_content_size
	var fcsSize int
	v := fhd >> 6
	switch v {
	case 0:
		if h.SingleSegment {
			fcsSize = 1
		}
	default:
		fcsSize = 1 << v
	}

	if fcsSize > 0 {
		h.HasFCS = true
		if len(in) < fcsSize {
			return io.ErrUnexpectedEOF
		}
		b, in = in[:fcsSize], in[fcsSize:]
		if b == nil {
			return io.ErrUnexpectedEOF
		}
		switch fcsSize {
		case 1:
			h.FrameContentSize = uint64(b[0])
		case 2:
			// When FCS_Field_Size is 2, the offset of 256 is added.
			h.FrameContentSize = uint64(b[0]) | (uint64(b[1]) << 8) + 256
		case 4:
			h.FrameContentSize = uint64(b[0]) | (uint64(b[1]) << 8) | (uint64(b[2]) << 16) | (uint64(b[3]) << 24)
		case 8:
			d1 := uint32(b[0]) | (uint32(b[1]) << 8) | (uint32(b[2]) << 16) | (uint32(b[3]) << 24)
			d2 := uint32(b[4]) | (uint32(b[5]) << 8) | (uint32(b[6]) << 16) | (uint32(b[7]) << 24)
			h.FrameContentSize = uint64(d1) | (uint64(d2) << 32)
		}
	}

	// Frame Header done, we will not fail from now on.
	if len(in) < 3 {
		return nil
	}
	tmp := in[:3]
	bh := uint32(tmp[0]) | (uint32(tmp[1]) << 8) | (uint32(tmp[2]) << 16)
	h.FirstBlock.Last = bh&1 != 0
	blockType := blockType((bh >> 1) & 3)
	// find size.
	cSize := int(bh >> 3)
	switch blockType {
	case blockTypeReserved:
		return nil
	case blockTypeRLE:
		h.FirstBlock.Compressed = true
		h.FirstBlock.DecompressedSize = cSize
		h.FirstBlock.CompressedSize = 1
	case blockTypeCompressed:
		h.FirstBlock.Compressed = true
		h.FirstBlock.CompressedSize = cSize
	case blockTypeRaw:
		h.FirstBlock.DecompressedSize = cSize
		h.FirstBlock.CompressedSize = cSize
	default:
		panic("Invalid block type")
	}

	h.FirstBlock.OK = true
	return nil
}
