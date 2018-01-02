// Package lzx implements a decompressor for the the WIM variant of the
// LZX compression algorithm.
//
// The LZX algorithm is an earlier variant of LZX DELTA, which is documented
// at https://msdn.microsoft.com/en-us/library/cc483133(v=exchg.80).aspx.
package lzx

import (
	"bytes"
	"encoding/binary"
	"errors"
	"io"
)

const (
	maincodecount = 496
	maincodesplit = 256
	lencodecount  = 249
	lenshift      = 9
	codemask      = 0x1ff
	tablebits     = 9
	tablesize     = 1 << tablebits

	maxBlockSize = 32768
	windowSize   = 32768

	maxTreePathLen = 16

	e8filesize  = 12000000
	maxe8offset = 0x3fffffff

	verbatimBlock      = 1
	alignedOffsetBlock = 2
	uncompressedBlock  = 3
)

var footerBits = [...]byte{
	0, 0, 0, 0, 1, 1, 2, 2,
	3, 3, 4, 4, 5, 5, 6, 6,
	7, 7, 8, 8, 9, 9, 10, 10,
	11, 11, 12, 12, 13, 13, 14,
}

var basePosition = [...]uint16{
	0, 1, 2, 3, 4, 6, 8, 12,
	16, 24, 32, 48, 64, 96, 128, 192,
	256, 384, 512, 768, 1024, 1536, 2048, 3072,
	4096, 6144, 8192, 12288, 16384, 24576, 32768,
}

var (
	errCorrupt = errors.New("LZX data corrupt")
)

// Reader is an interface used by the decompressor to access
// the input stream. If the provided io.Reader does not implement
// Reader, then a bufio.Reader is used.
type Reader interface {
	io.Reader
	io.ByteReader
}

type decompressor struct {
	r            io.Reader
	err          error
	unaligned    bool
	nbits        byte
	c            uint32
	lru          [3]uint16
	uncompressed int
	windowReader *bytes.Reader
	mainlens     [maincodecount]byte
	lenlens      [lencodecount]byte
	window       [windowSize]byte
	b            []byte
	bv           int
	bo           int
}

//go:noinline
func (f *decompressor) fail(err error) {
	if f.err == nil {
		f.err = err
	}
	f.bo = 0
	f.bv = 0
}

func (f *decompressor) ensureAtLeast(n int) error {
	if f.bv-f.bo >= n {
		return nil
	}

	if f.err != nil {
		return f.err
	}

	if f.bv != f.bo {
		copy(f.b[:f.bv-f.bo], f.b[f.bo:f.bv])
	}
	n, err := io.ReadAtLeast(f.r, f.b[f.bv-f.bo:], n)
	if err != nil {
		if err == io.EOF {
			err = io.ErrUnexpectedEOF
		} else {
			f.fail(err)
		}
		return err
	}
	f.bv = f.bv - f.bo + n
	f.bo = 0
	return nil
}

// feed retrieves another 16-bit word from the stream and consumes
// it into f.c. It returns false if there are no more bytes available.
// Otherwise, on error, it sets f.err.
func (f *decompressor) feed() bool {
	err := f.ensureAtLeast(2)
	if err != nil {
		if err == io.ErrUnexpectedEOF {
			return false
		}
	}
	f.c |= (uint32(f.b[f.bo+1])<<8 | uint32(f.b[f.bo])) << (16 - f.nbits)
	f.nbits += 16
	f.bo += 2
	return true
}

// getBits retrieves the next n bits from the byte stream. n
// must be <= 16. It sets f.err on error.
func (f *decompressor) getBits(n byte) uint16 {
	if f.nbits < n {
		if !f.feed() {
			f.fail(io.ErrUnexpectedEOF)
		}
	}
	c := uint16(f.c >> (32 - n))
	f.c <<= n
	f.nbits -= n
	return c
}

type huffman struct {
	extra   [][]uint16
	maxbits byte
	table   [tablesize]uint16
}

// buildTable builds a huffman decoding table from a slice of code lengths,
// one per code, in order. Each code length must be <= maxTreePathLen.
// See https://en.wikipedia.org/wiki/Canonical_Huffman_code.
func buildTable(codelens []byte) *huffman {
	// Determine the number of codes of each length, and the
	// maximum length.
	var count [maxTreePathLen + 1]uint
	var max byte
	for _, cl := range codelens {
		count[cl]++
		if max < cl {
			max = cl
		}
	}

	if max == 0 {
		return &huffman{}
	}

	// Determine the first code of each length.
	var first [maxTreePathLen + 1]uint
	code := uint(0)
	for i := byte(1); i <= max; i++ {
		code <<= 1
		first[i] = code
		code += count[i]
	}

	if code != 1<<max {
		return nil
	}

	// Build a table for code lookup. For code sizes < max,
	// put all possible suffixes for the code into the table, too.
	// For max > tablebits, split long codes into additional tables
	// of suffixes of max-tablebits length.
	h := &huffman{maxbits: max}
	if max > tablebits {
		core := first[tablebits+1] / 2 // Number of codes that fit without extra tables
		nextra := 1<<tablebits - core  // Number of extra entries
		h.extra = make([][]uint16, nextra)
		for code := core; code < 1<<tablebits; code++ {
			h.table[code] = uint16(code - core)
			h.extra[code-core] = make([]uint16, 1<<(max-tablebits))
		}
	}

	for i, cl := range codelens {
		if cl != 0 {
			code := first[cl]
			first[cl]++
			v := uint16(cl)<<lenshift | uint16(i)
			if cl <= tablebits {
				extendedCode := code << (tablebits - cl)
				for j := uint(0); j < 1<<(tablebits-cl); j++ {
					h.table[extendedCode+j] = v
				}
			} else {
				prefix := code >> (cl - tablebits)
				suffix := code & (1<<(cl-tablebits) - 1)
				extendedCode := suffix << (max - cl)
				for j := uint(0); j < 1<<(max-cl); j++ {
					h.extra[h.table[prefix]][extendedCode+j] = v
				}
			}
		}
	}

	return h
}

// getCode retrieves the next code using the provided
// huffman tree. It sets f.err on error.
func (f *decompressor) getCode(h *huffman) uint16 {
	if h.maxbits > 0 {
		if f.nbits < maxTreePathLen {
			f.feed()
		}

		// For codes with length < tablebits, it doesn't matter
		// what the remainder of the bits used for table lookup
		// are, since entries with all possible suffixes were
		// added to the table.
		c := h.table[f.c>>(32-tablebits)]
		if c >= 1<<lenshift {
			// The code is already in c.
		} else {
			c = h.extra[c][f.c<<tablebits>>(32-(h.maxbits-tablebits))]
		}

		n := byte(c >> lenshift)
		if f.nbits >= n {
			// Only consume the length of the code, not the maximum
			// code length.
			f.c <<= n
			f.nbits -= n
			return c & codemask
		}

		f.fail(io.ErrUnexpectedEOF)
		return 0
	}

	// This is an empty tree. It should not be used.
	f.fail(errCorrupt)
	return 0
}

// readTree updates the huffman tree path lengths in lens by
// reading and decoding lengths from the byte stream. lens
// should be prepopulated with the previous block's tree's path
// lengths. For the first block, lens should be zero.
func (f *decompressor) readTree(lens []byte) error {
	// Get the pre-tree for the main tree.
	var pretreeLen [20]byte
	for i := range pretreeLen {
		pretreeLen[i] = byte(f.getBits(4))
	}
	if f.err != nil {
		return f.err
	}
	h := buildTable(pretreeLen[:])

	// The lengths are encoded as a series of huffman codes
	// encoded by the pre-tree.
	for i := 0; i < len(lens); {
		c := byte(f.getCode(h))
		if f.err != nil {
			return f.err
		}
		switch {
		case c <= 16: // length is delta from previous length
			lens[i] = (lens[i] + 17 - c) % 17
			i++
		case c == 17: // next n + 4 lengths are zero
			zeroes := int(f.getBits(4)) + 4
			if i+zeroes > len(lens) {
				return errCorrupt
			}
			for j := 0; j < zeroes; j++ {
				lens[i+j] = 0
			}
			i += zeroes
		case c == 18: // next n + 20 lengths are zero
			zeroes := int(f.getBits(5)) + 20
			if i+zeroes > len(lens) {
				return errCorrupt
			}
			for j := 0; j < zeroes; j++ {
				lens[i+j] = 0
			}
			i += zeroes
		case c == 19: // next n + 4 lengths all have the same value
			same := int(f.getBits(1)) + 4
			if i+same > len(lens) {
				return errCorrupt
			}
			c = byte(f.getCode(h))
			if c > 16 {
				return errCorrupt
			}
			l := (lens[i] + 17 - c) % 17
			for j := 0; j < same; j++ {
				lens[i+j] = l
			}
			i += same
		default:
			return errCorrupt
		}
	}

	if f.err != nil {
		return f.err
	}
	return nil
}

func (f *decompressor) readBlockHeader() (byte, uint16, error) {
	// If the previous block was an unaligned uncompressed block, restore
	// 2-byte alignment.
	if f.unaligned {
		err := f.ensureAtLeast(1)
		if err != nil {
			return 0, 0, err
		}
		f.bo++
		f.unaligned = false
	}

	blockType := f.getBits(3)
	full := f.getBits(1)
	var blockSize uint16
	if full != 0 {
		blockSize = maxBlockSize
	} else {
		blockSize = f.getBits(16)
		if blockSize > maxBlockSize {
			return 0, 0, errCorrupt
		}
	}

	if f.err != nil {
		return 0, 0, f.err
	}

	switch blockType {
	case verbatimBlock, alignedOffsetBlock:
		// The caller will read the huffman trees.
	case uncompressedBlock:
		if f.nbits > 16 {
			panic("impossible: more than one 16-bit word remains")
		}

		// Drop the remaining bits in the current 16-bit word
		// If there are no bits left, discard a full 16-bit word.
		n := f.nbits
		if n == 0 {
			n = 16
		}

		f.getBits(n)

		// Read the LRU values for the next block.
		err := f.ensureAtLeast(12)
		if err != nil {
			return 0, 0, err
		}

		f.lru[0] = uint16(binary.LittleEndian.Uint32(f.b[f.bo : f.bo+4]))
		f.lru[1] = uint16(binary.LittleEndian.Uint32(f.b[f.bo+4 : f.bo+8]))
		f.lru[2] = uint16(binary.LittleEndian.Uint32(f.b[f.bo+8 : f.bo+12]))
		f.bo += 12

	default:
		return 0, 0, errCorrupt
	}

	return byte(blockType), blockSize, nil
}

// readTrees reads the two or three huffman trees for the current block.
// readAligned specifies whether to read the aligned offset tree.
func (f *decompressor) readTrees(readAligned bool) (main *huffman, length *huffman, aligned *huffman, err error) {
	// Aligned offset blocks start with a small aligned offset tree.
	if readAligned {
		var alignedLen [8]byte
		for i := range alignedLen {
			alignedLen[i] = byte(f.getBits(3))
		}
		aligned = buildTable(alignedLen[:])
		if aligned == nil {
			err = errors.New("corrupt")
			return
		}
	}

	// The main tree is encoded in two parts.
	err = f.readTree(f.mainlens[:maincodesplit])
	if err != nil {
		return
	}
	err = f.readTree(f.mainlens[maincodesplit:])
	if err != nil {
		return
	}

	main = buildTable(f.mainlens[:])
	if main == nil {
		err = errors.New("corrupt")
		return
	}

	// The length tree is encoding in a single part.
	err = f.readTree(f.lenlens[:])
	if err != nil {
		return
	}

	length = buildTable(f.lenlens[:])
	if length == nil {
		err = errors.New("corrupt")
		return
	}

	err = f.err
	return
}

// readCompressedBlock decodes a compressed block, writing into the window
// starting at start and ending at end, and using the provided huffman trees.
func (f *decompressor) readCompressedBlock(start, end uint16, hmain, hlength, haligned *huffman) (int, error) {
	i := start
	for i < end {
		main := f.getCode(hmain)
		if f.err != nil {
			break
		}
		if main < 256 {
			// Literal byte.
			f.window[i] = byte(main)
			i++
			continue
		}

		// This is a match backward in the window. Determine
		// the offset and dlength.
		matchlen := (main - 256) % 8
		slot := (main - 256) / 8

		// The length is either the low bits of the code,
		// or if this is 7, is encoded with the length tree.
		if matchlen == 7 {
			matchlen += f.getCode(hlength)
		}
		matchlen += 2

		var matchoffset uint16
		if slot < 3 {
			// The offset is one of the LRU values.
			matchoffset = f.lru[slot]
			f.lru[slot] = f.lru[0]
			f.lru[0] = matchoffset
		} else {
			// The offset is encoded as a combination of the
			// slot and more bits from the bit stream.
			offsetbits := footerBits[slot]
			var verbatimbits, alignedbits uint16
			if offsetbits > 0 {
				if haligned != nil && offsetbits >= 3 {
					// This is an aligned offset block. Combine
					// the bits written verbatim with the aligned
					// offset tree code.
					verbatimbits = f.getBits(offsetbits-3) * 8
					alignedbits = f.getCode(haligned)
				} else {
					// There are no aligned offset bits to read,
					// only verbatim bits.
					verbatimbits = f.getBits(offsetbits)
					alignedbits = 0
				}
			}
			matchoffset = basePosition[slot] + verbatimbits + alignedbits - 2
			// Update the LRU cache.
			f.lru[2] = f.lru[1]
			f.lru[1] = f.lru[0]
			f.lru[0] = matchoffset
		}

		if matchoffset <= i && matchlen <= end-i {
			copyend := i + matchlen
			for ; i < copyend; i++ {
				f.window[i] = f.window[i-matchoffset]
			}
		} else {
			f.fail(errCorrupt)
			break
		}
	}
	return int(i - start), f.err
}

// readBlock decodes the current block and returns the number of uncompressed bytes.
func (f *decompressor) readBlock(start uint16) (int, error) {
	blockType, size, err := f.readBlockHeader()
	if err != nil {
		return 0, err
	}

	if blockType == uncompressedBlock {
		if size%2 == 1 {
			// Remember to realign the byte stream at the next block.
			f.unaligned = true
		}
		copied := 0
		if f.bo < f.bv {
			copied = int(size)
			s := int(start)
			if copied > f.bv-f.bo {
				copied = f.bv - f.bo
			}
			copy(f.window[s:s+copied], f.b[f.bo:f.bo+copied])
			f.bo += copied
		}
		n, err := io.ReadFull(f.r, f.window[start+uint16(copied):start+size])
		return copied + n, err
	}

	hmain, hlength, haligned, err := f.readTrees(blockType == alignedOffsetBlock)
	if err != nil {
		return 0, err
	}

	return f.readCompressedBlock(start, start+size, hmain, hlength, haligned)
}

// decodeE8 reverses the 0xe8 x86 instruction encoding that was performed
// to the uncompressed data before it was compressed.
func decodeE8(b []byte, off int64) {
	if off > maxe8offset || len(b) < 10 {
		return
	}
	for i := 0; i < len(b)-10; i++ {
		if b[i] == 0xe8 {
			currentPtr := int32(off) + int32(i)
			abs := int32(binary.LittleEndian.Uint32(b[i+1 : i+5]))
			if abs >= -currentPtr && abs < e8filesize {
				var rel int32
				if abs >= 0 {
					rel = abs - currentPtr
				} else {
					rel = abs + e8filesize
				}
				binary.LittleEndian.PutUint32(b[i+1:i+5], uint32(rel))
			}
			i += 4
		}
	}
}

func (f *decompressor) Read(b []byte) (int, error) {
	// Read and uncompress everything.
	if f.windowReader == nil {
		n := 0
		for n < f.uncompressed {
			k, err := f.readBlock(uint16(n))
			if err != nil {
				return 0, err
			}
			n += k
		}
		decodeE8(f.window[:f.uncompressed], 0)
		f.windowReader = bytes.NewReader(f.window[:f.uncompressed])
	}

	// Just read directly from the window.
	return f.windowReader.Read(b)
}

func (f *decompressor) Close() error {
	return nil
}

// NewReader returns a new io.ReadCloser that decompresses a
// WIM LZX stream until uncompressedSize bytes have been returned.
func NewReader(r io.Reader, uncompressedSize int) (io.ReadCloser, error) {
	if uncompressedSize > windowSize {
		return nil, errors.New("uncompressed size is limited to 32KB")
	}
	f := &decompressor{
		lru:          [3]uint16{1, 1, 1},
		uncompressed: uncompressedSize,
		b:            make([]byte, 4096),
		r:            r,
	}
	return f, nil
}
