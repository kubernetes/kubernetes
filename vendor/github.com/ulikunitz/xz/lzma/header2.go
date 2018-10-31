// Copyright 2014-2017 Ulrich Kunitz. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lzma

import (
	"errors"
	"fmt"
	"io"
)

const (
	// maximum size of compressed data in a chunk
	maxCompressed = 1 << 16
	// maximum size of uncompressed data in a chunk
	maxUncompressed = 1 << 21
)

// chunkType represents the type of an LZMA2 chunk. Note that this
// value is an internal representation and no actual encoding of a LZMA2
// chunk header.
type chunkType byte

// Possible values for the chunk type.
const (
	// end of stream
	cEOS chunkType = iota
	// uncompressed; reset dictionary
	cUD
	// uncompressed; no reset of dictionary
	cU
	// LZMA compressed; no reset
	cL
	// LZMA compressed; reset state
	cLR
	// LZMA compressed; reset state; new property value
	cLRN
	// LZMA compressed; reset state; new property value; reset dictionary
	cLRND
)

// chunkTypeStrings provide a string representation for the chunk types.
var chunkTypeStrings = [...]string{
	cEOS:  "EOS",
	cU:    "U",
	cUD:   "UD",
	cL:    "L",
	cLR:   "LR",
	cLRN:  "LRN",
	cLRND: "LRND",
}

// String returns a string representation of the chunk type.
func (c chunkType) String() string {
	if !(cEOS <= c && c <= cLRND) {
		return "unknown"
	}
	return chunkTypeStrings[c]
}

// Actual encodings for the chunk types in the value. Note that the high
// uncompressed size bits are stored in the header byte additionally.
const (
	hEOS  = 0
	hUD   = 1
	hU    = 2
	hL    = 1 << 7
	hLR   = 1<<7 | 1<<5
	hLRN  = 1<<7 | 1<<6
	hLRND = 1<<7 | 1<<6 | 1<<5
)

// errHeaderByte indicates an unsupported value for the chunk header
// byte. These bytes starts the variable-length chunk header.
var errHeaderByte = errors.New("lzma: unsupported chunk header byte")

// headerChunkType converts the header byte into a chunk type. It
// ignores the uncompressed size bits in the chunk header byte.
func headerChunkType(h byte) (c chunkType, err error) {
	if h&hL == 0 {
		// no compression
		switch h {
		case hEOS:
			c = cEOS
		case hUD:
			c = cUD
		case hU:
			c = cU
		default:
			return 0, errHeaderByte
		}
		return
	}
	switch h & hLRND {
	case hL:
		c = cL
	case hLR:
		c = cLR
	case hLRN:
		c = cLRN
	case hLRND:
		c = cLRND
	default:
		return 0, errHeaderByte
	}
	return
}

// uncompressedHeaderLen provides the length of an uncompressed header
const uncompressedHeaderLen = 3

// headerLen returns the length of the LZMA2 header for a given chunk
// type.
func headerLen(c chunkType) int {
	switch c {
	case cEOS:
		return 1
	case cU, cUD:
		return uncompressedHeaderLen
	case cL, cLR:
		return 5
	case cLRN, cLRND:
		return 6
	}
	panic(fmt.Errorf("unsupported chunk type %d", c))
}

// chunkHeader represents the contents of a chunk header.
type chunkHeader struct {
	ctype        chunkType
	uncompressed uint32
	compressed   uint16
	props        Properties
}

// String returns a string representation of the chunk header.
func (h *chunkHeader) String() string {
	return fmt.Sprintf("%s %d %d %s", h.ctype, h.uncompressed,
		h.compressed, &h.props)
}

// UnmarshalBinary reads the content of the chunk header from the data
// slice. The slice must have the correct length.
func (h *chunkHeader) UnmarshalBinary(data []byte) error {
	if len(data) == 0 {
		return errors.New("no data")
	}
	c, err := headerChunkType(data[0])
	if err != nil {
		return err
	}

	n := headerLen(c)
	if len(data) < n {
		return errors.New("incomplete data")
	}
	if len(data) > n {
		return errors.New("invalid data length")
	}

	*h = chunkHeader{ctype: c}
	if c == cEOS {
		return nil
	}

	h.uncompressed = uint32(uint16BE(data[1:3]))
	if c <= cU {
		return nil
	}
	h.uncompressed |= uint32(data[0]&^hLRND) << 16

	h.compressed = uint16BE(data[3:5])
	if c <= cLR {
		return nil
	}

	h.props, err = PropertiesForCode(data[5])
	return err
}

// MarshalBinary encodes the chunk header value. The function checks
// whether the content of the chunk header is correct.
func (h *chunkHeader) MarshalBinary() (data []byte, err error) {
	if h.ctype > cLRND {
		return nil, errors.New("invalid chunk type")
	}
	if err = h.props.verify(); err != nil {
		return nil, err
	}

	data = make([]byte, headerLen(h.ctype))

	switch h.ctype {
	case cEOS:
		return data, nil
	case cUD:
		data[0] = hUD
	case cU:
		data[0] = hU
	case cL:
		data[0] = hL
	case cLR:
		data[0] = hLR
	case cLRN:
		data[0] = hLRN
	case cLRND:
		data[0] = hLRND
	}

	putUint16BE(data[1:3], uint16(h.uncompressed))
	if h.ctype <= cU {
		return data, nil
	}
	data[0] |= byte(h.uncompressed>>16) &^ hLRND

	putUint16BE(data[3:5], h.compressed)
	if h.ctype <= cLR {
		return data, nil
	}

	data[5] = h.props.Code()
	return data, nil
}

// readChunkHeader reads the chunk header from the IO reader.
func readChunkHeader(r io.Reader) (h *chunkHeader, err error) {
	p := make([]byte, 1, 6)
	if _, err = io.ReadFull(r, p); err != nil {
		return
	}
	c, err := headerChunkType(p[0])
	if err != nil {
		return
	}
	p = p[:headerLen(c)]
	if _, err = io.ReadFull(r, p[1:]); err != nil {
		return
	}
	h = new(chunkHeader)
	if err = h.UnmarshalBinary(p); err != nil {
		return nil, err
	}
	return h, nil
}

// uint16BE converts a big-endian uint16 representation to an uint16
// value.
func uint16BE(p []byte) uint16 {
	return uint16(p[0])<<8 | uint16(p[1])
}

// putUint16BE puts the big-endian uint16 presentation into the given
// slice.
func putUint16BE(p []byte, x uint16) {
	p[0] = byte(x >> 8)
	p[1] = byte(x)
}

// chunkState is used to manage the state of the chunks
type chunkState byte

// start and stop define the initial and terminating state of the chunk
// state
const (
	start chunkState = 'S'
	stop             = 'T'
)

// errors for the chunk state handling
var (
	errChunkType = errors.New("lzma: unexpected chunk type")
	errState     = errors.New("lzma: wrong chunk state")
)

// next transitions state based on chunk type input
func (c *chunkState) next(ctype chunkType) error {
	switch *c {
	// start state
	case 'S':
		switch ctype {
		case cEOS:
			*c = 'T'
		case cUD:
			*c = 'R'
		case cLRND:
			*c = 'L'
		default:
			return errChunkType
		}
	// normal LZMA mode
	case 'L':
		switch ctype {
		case cEOS:
			*c = 'T'
		case cUD:
			*c = 'R'
		case cU:
			*c = 'U'
		case cL, cLR, cLRN, cLRND:
			break
		default:
			return errChunkType
		}
	// reset required
	case 'R':
		switch ctype {
		case cEOS:
			*c = 'T'
		case cUD, cU:
			break
		case cLRN, cLRND:
			*c = 'L'
		default:
			return errChunkType
		}
	// uncompressed
	case 'U':
		switch ctype {
		case cEOS:
			*c = 'T'
		case cUD:
			*c = 'R'
		case cU:
			break
		case cL, cLR, cLRN, cLRND:
			*c = 'L'
		default:
			return errChunkType
		}
	// terminal state
	case 'T':
		return errChunkType
	default:
		return errState
	}
	return nil
}

// defaultChunkType returns the default chunk type for each chunk state.
func (c chunkState) defaultChunkType() chunkType {
	switch c {
	case 'S':
		return cLRND
	case 'L', 'U':
		return cL
	case 'R':
		return cLRN
	default:
		// no error
		return cEOS
	}
}

// maxDictCap defines the maximum dictionary capacity supported by the
// LZMA2 dictionary capacity encoding.
const maxDictCap = 1<<32 - 1

// maxDictCapCode defines the maximum dictionary capacity code.
const maxDictCapCode = 40

// The function decodes the dictionary capacity byte, but doesn't change
// for the correct range of the given byte.
func decodeDictCap(c byte) int64 {
	return (2 | int64(c)&1) << (11 + (c>>1)&0x1f)
}

// DecodeDictCap decodes the encoded dictionary capacity. The function
// returns an error if the code is out of range.
func DecodeDictCap(c byte) (n int64, err error) {
	if c >= maxDictCapCode {
		if c == maxDictCapCode {
			return maxDictCap, nil
		}
		return 0, errors.New("lzma: invalid dictionary size code")
	}
	return decodeDictCap(c), nil
}

// EncodeDictCap encodes a dictionary capacity. The function returns the
// code for the capacity that is greater or equal n. If n exceeds the
// maximum support dictionary capacity, the maximum value is returned.
func EncodeDictCap(n int64) byte {
	a, b := byte(0), byte(40)
	for a < b {
		c := a + (b-a)>>1
		m := decodeDictCap(c)
		if n <= m {
			if n == m {
				return c
			}
			b = c
		} else {
			a = c + 1
		}
	}
	return a
}
