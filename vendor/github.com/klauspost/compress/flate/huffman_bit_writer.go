// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package flate

import (
	"io"
)

const (
	// The largest offset code.
	offsetCodeCount = 30

	// The special code used to mark the end of a block.
	endBlockMarker = 256

	// The first length code.
	lengthCodesStart = 257

	// The number of codegen codes.
	codegenCodeCount = 19
	badCode          = 255

	// bufferFlushSize indicates the buffer size
	// after which bytes are flushed to the writer.
	// Should preferably be a multiple of 6, since
	// we accumulate 6 bytes between writes to the buffer.
	bufferFlushSize = 240

	// bufferSize is the actual output byte buffer size.
	// It must have additional headroom for a flush
	// which can contain up to 8 bytes.
	bufferSize = bufferFlushSize + 8
)

// The number of extra bits needed by length code X - LENGTH_CODES_START.
var lengthExtraBits = [32]int8{
	/* 257 */ 0, 0, 0,
	/* 260 */ 0, 0, 0, 0, 0, 1, 1, 1, 1, 2,
	/* 270 */ 2, 2, 2, 3, 3, 3, 3, 4, 4, 4,
	/* 280 */ 4, 5, 5, 5, 5, 0,
}

// The length indicated by length code X - LENGTH_CODES_START.
var lengthBase = [32]uint8{
	0, 1, 2, 3, 4, 5, 6, 7, 8, 10,
	12, 14, 16, 20, 24, 28, 32, 40, 48, 56,
	64, 80, 96, 112, 128, 160, 192, 224, 255,
}

// offset code word extra bits.
var offsetExtraBits = [64]int8{
	0, 0, 0, 0, 1, 1, 2, 2, 3, 3,
	4, 4, 5, 5, 6, 6, 7, 7, 8, 8,
	9, 9, 10, 10, 11, 11, 12, 12, 13, 13,
	/* extended window */
	14, 14, 15, 15, 16, 16, 17, 17, 18, 18, 19, 19, 20, 20,
}

var offsetBase = [64]uint32{
	/* normal deflate */
	0x000000, 0x000001, 0x000002, 0x000003, 0x000004,
	0x000006, 0x000008, 0x00000c, 0x000010, 0x000018,
	0x000020, 0x000030, 0x000040, 0x000060, 0x000080,
	0x0000c0, 0x000100, 0x000180, 0x000200, 0x000300,
	0x000400, 0x000600, 0x000800, 0x000c00, 0x001000,
	0x001800, 0x002000, 0x003000, 0x004000, 0x006000,

	/* extended window */
	0x008000, 0x00c000, 0x010000, 0x018000, 0x020000,
	0x030000, 0x040000, 0x060000, 0x080000, 0x0c0000,
	0x100000, 0x180000, 0x200000, 0x300000,
}

// The odd order in which the codegen code sizes are written.
var codegenOrder = []uint32{16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15}

type huffmanBitWriter struct {
	// writer is the underlying writer.
	// Do not use it directly; use the write method, which ensures
	// that Write errors are sticky.
	writer io.Writer

	// Data waiting to be written is bytes[0:nbytes]
	// and then the low nbits of bits.
	bits            uint64
	nbits           uint16
	nbytes          uint8
	literalEncoding *huffmanEncoder
	offsetEncoding  *huffmanEncoder
	codegenEncoding *huffmanEncoder
	err             error
	lastHeader      int
	// Set between 0 (reused block can be up to 2x the size)
	logNewTablePenalty uint
	lastHuffMan        bool
	bytes              [256]byte
	literalFreq        [lengthCodesStart + 32]uint16
	offsetFreq         [32]uint16
	codegenFreq        [codegenCodeCount]uint16

	// codegen must have an extra space for the final symbol.
	codegen [literalCount + offsetCodeCount + 1]uint8
}

// Huffman reuse.
//
// The huffmanBitWriter supports reusing huffman tables and thereby combining block sections.
//
// This is controlled by several variables:
//
// If lastHeader is non-zero the Huffman table can be reused.
// This also indicates that a Huffman table has been generated that can output all
// possible symbols.
// It also indicates that an EOB has not yet been emitted, so if a new tabel is generated
// an EOB with the previous table must be written.
//
// If lastHuffMan is set, a table for outputting literals has been generated and offsets are invalid.
//
// An incoming block estimates the output size of a new table using a 'fresh' by calculating the
// optimal size and adding a penalty in 'logNewTablePenalty'.
// A Huffman table is not optimal, which is why we add a penalty, and generating a new table
// is slower both for compression and decompression.

func newHuffmanBitWriter(w io.Writer) *huffmanBitWriter {
	return &huffmanBitWriter{
		writer:          w,
		literalEncoding: newHuffmanEncoder(literalCount),
		codegenEncoding: newHuffmanEncoder(codegenCodeCount),
		offsetEncoding:  newHuffmanEncoder(offsetCodeCount),
	}
}

func (w *huffmanBitWriter) reset(writer io.Writer) {
	w.writer = writer
	w.bits, w.nbits, w.nbytes, w.err = 0, 0, 0, nil
	w.lastHeader = 0
	w.lastHuffMan = false
}

func (w *huffmanBitWriter) canReuse(t *tokens) (offsets, lits bool) {
	offsets, lits = true, true
	a := t.offHist[:offsetCodeCount]
	b := w.offsetFreq[:len(a)]
	for i := range a {
		if b[i] == 0 && a[i] != 0 {
			offsets = false
			break
		}
	}

	a = t.extraHist[:literalCount-256]
	b = w.literalFreq[256:literalCount]
	b = b[:len(a)]
	for i := range a {
		if b[i] == 0 && a[i] != 0 {
			lits = false
			break
		}
	}
	if lits {
		a = t.litHist[:]
		b = w.literalFreq[:len(a)]
		for i := range a {
			if b[i] == 0 && a[i] != 0 {
				lits = false
				break
			}
		}
	}
	return
}

func (w *huffmanBitWriter) flush() {
	if w.err != nil {
		w.nbits = 0
		return
	}
	if w.lastHeader > 0 {
		// We owe an EOB
		w.writeCode(w.literalEncoding.codes[endBlockMarker])
		w.lastHeader = 0
	}
	n := w.nbytes
	for w.nbits != 0 {
		w.bytes[n] = byte(w.bits)
		w.bits >>= 8
		if w.nbits > 8 { // Avoid underflow
			w.nbits -= 8
		} else {
			w.nbits = 0
		}
		n++
	}
	w.bits = 0
	w.write(w.bytes[:n])
	w.nbytes = 0
}

func (w *huffmanBitWriter) write(b []byte) {
	if w.err != nil {
		return
	}
	_, w.err = w.writer.Write(b)
}

func (w *huffmanBitWriter) writeBits(b int32, nb uint16) {
	w.bits |= uint64(b) << (w.nbits & reg16SizeMask64)
	w.nbits += nb
	if w.nbits >= 48 {
		w.writeOutBits()
	}
}

func (w *huffmanBitWriter) writeBytes(bytes []byte) {
	if w.err != nil {
		return
	}
	n := w.nbytes
	if w.nbits&7 != 0 {
		w.err = InternalError("writeBytes with unfinished bits")
		return
	}
	for w.nbits != 0 {
		w.bytes[n] = byte(w.bits)
		w.bits >>= 8
		w.nbits -= 8
		n++
	}
	if n != 0 {
		w.write(w.bytes[:n])
	}
	w.nbytes = 0
	w.write(bytes)
}

// RFC 1951 3.2.7 specifies a special run-length encoding for specifying
// the literal and offset lengths arrays (which are concatenated into a single
// array).  This method generates that run-length encoding.
//
// The result is written into the codegen array, and the frequencies
// of each code is written into the codegenFreq array.
// Codes 0-15 are single byte codes. Codes 16-18 are followed by additional
// information. Code badCode is an end marker
//
//  numLiterals      The number of literals in literalEncoding
//  numOffsets       The number of offsets in offsetEncoding
//  litenc, offenc   The literal and offset encoder to use
func (w *huffmanBitWriter) generateCodegen(numLiterals int, numOffsets int, litEnc, offEnc *huffmanEncoder) {
	for i := range w.codegenFreq {
		w.codegenFreq[i] = 0
	}
	// Note that we are using codegen both as a temporary variable for holding
	// a copy of the frequencies, and as the place where we put the result.
	// This is fine because the output is always shorter than the input used
	// so far.
	codegen := w.codegen[:] // cache
	// Copy the concatenated code sizes to codegen. Put a marker at the end.
	cgnl := codegen[:numLiterals]
	for i := range cgnl {
		cgnl[i] = uint8(litEnc.codes[i].len)
	}

	cgnl = codegen[numLiterals : numLiterals+numOffsets]
	for i := range cgnl {
		cgnl[i] = uint8(offEnc.codes[i].len)
	}
	codegen[numLiterals+numOffsets] = badCode

	size := codegen[0]
	count := 1
	outIndex := 0
	for inIndex := 1; size != badCode; inIndex++ {
		// INVARIANT: We have seen "count" copies of size that have not yet
		// had output generated for them.
		nextSize := codegen[inIndex]
		if nextSize == size {
			count++
			continue
		}
		// We need to generate codegen indicating "count" of size.
		if size != 0 {
			codegen[outIndex] = size
			outIndex++
			w.codegenFreq[size]++
			count--
			for count >= 3 {
				n := 6
				if n > count {
					n = count
				}
				codegen[outIndex] = 16
				outIndex++
				codegen[outIndex] = uint8(n - 3)
				outIndex++
				w.codegenFreq[16]++
				count -= n
			}
		} else {
			for count >= 11 {
				n := 138
				if n > count {
					n = count
				}
				codegen[outIndex] = 18
				outIndex++
				codegen[outIndex] = uint8(n - 11)
				outIndex++
				w.codegenFreq[18]++
				count -= n
			}
			if count >= 3 {
				// count >= 3 && count <= 10
				codegen[outIndex] = 17
				outIndex++
				codegen[outIndex] = uint8(count - 3)
				outIndex++
				w.codegenFreq[17]++
				count = 0
			}
		}
		count--
		for ; count >= 0; count-- {
			codegen[outIndex] = size
			outIndex++
			w.codegenFreq[size]++
		}
		// Set up invariant for next time through the loop.
		size = nextSize
		count = 1
	}
	// Marker indicating the end of the codegen.
	codegen[outIndex] = badCode
}

func (w *huffmanBitWriter) codegens() int {
	numCodegens := len(w.codegenFreq)
	for numCodegens > 4 && w.codegenFreq[codegenOrder[numCodegens-1]] == 0 {
		numCodegens--
	}
	return numCodegens
}

func (w *huffmanBitWriter) headerSize() (size, numCodegens int) {
	numCodegens = len(w.codegenFreq)
	for numCodegens > 4 && w.codegenFreq[codegenOrder[numCodegens-1]] == 0 {
		numCodegens--
	}
	return 3 + 5 + 5 + 4 + (3 * numCodegens) +
		w.codegenEncoding.bitLength(w.codegenFreq[:]) +
		int(w.codegenFreq[16])*2 +
		int(w.codegenFreq[17])*3 +
		int(w.codegenFreq[18])*7, numCodegens
}

// dynamicSize returns the size of dynamically encoded data in bits.
func (w *huffmanBitWriter) dynamicReuseSize(litEnc, offEnc *huffmanEncoder) (size int) {
	size = litEnc.bitLength(w.literalFreq[:]) +
		offEnc.bitLength(w.offsetFreq[:])
	return size
}

// dynamicSize returns the size of dynamically encoded data in bits.
func (w *huffmanBitWriter) dynamicSize(litEnc, offEnc *huffmanEncoder, extraBits int) (size, numCodegens int) {
	header, numCodegens := w.headerSize()
	size = header +
		litEnc.bitLength(w.literalFreq[:]) +
		offEnc.bitLength(w.offsetFreq[:]) +
		extraBits
	return size, numCodegens
}

// extraBitSize will return the number of bits that will be written
// as "extra" bits on matches.
func (w *huffmanBitWriter) extraBitSize() int {
	total := 0
	for i, n := range w.literalFreq[257:literalCount] {
		total += int(n) * int(lengthExtraBits[i&31])
	}
	for i, n := range w.offsetFreq[:offsetCodeCount] {
		total += int(n) * int(offsetExtraBits[i&31])
	}
	return total
}

// fixedSize returns the size of dynamically encoded data in bits.
func (w *huffmanBitWriter) fixedSize(extraBits int) int {
	return 3 +
		fixedLiteralEncoding.bitLength(w.literalFreq[:]) +
		fixedOffsetEncoding.bitLength(w.offsetFreq[:]) +
		extraBits
}

// storedSize calculates the stored size, including header.
// The function returns the size in bits and whether the block
// fits inside a single block.
func (w *huffmanBitWriter) storedSize(in []byte) (int, bool) {
	if in == nil {
		return 0, false
	}
	if len(in) <= maxStoreBlockSize {
		return (len(in) + 5) * 8, true
	}
	return 0, false
}

func (w *huffmanBitWriter) writeCode(c hcode) {
	// The function does not get inlined if we "& 63" the shift.
	w.bits |= uint64(c.code) << w.nbits
	w.nbits += c.len
	if w.nbits >= 48 {
		w.writeOutBits()
	}
}

// writeOutBits will write bits to the buffer.
func (w *huffmanBitWriter) writeOutBits() {
	bits := w.bits
	w.bits >>= 48
	w.nbits -= 48
	n := w.nbytes
	w.bytes[n] = byte(bits)
	w.bytes[n+1] = byte(bits >> 8)
	w.bytes[n+2] = byte(bits >> 16)
	w.bytes[n+3] = byte(bits >> 24)
	w.bytes[n+4] = byte(bits >> 32)
	w.bytes[n+5] = byte(bits >> 40)
	n += 6
	if n >= bufferFlushSize {
		if w.err != nil {
			n = 0
			return
		}
		w.write(w.bytes[:n])
		n = 0
	}
	w.nbytes = n
}

// Write the header of a dynamic Huffman block to the output stream.
//
//  numLiterals  The number of literals specified in codegen
//  numOffsets   The number of offsets specified in codegen
//  numCodegens  The number of codegens used in codegen
func (w *huffmanBitWriter) writeDynamicHeader(numLiterals int, numOffsets int, numCodegens int, isEof bool) {
	if w.err != nil {
		return
	}
	var firstBits int32 = 4
	if isEof {
		firstBits = 5
	}
	w.writeBits(firstBits, 3)
	w.writeBits(int32(numLiterals-257), 5)
	w.writeBits(int32(numOffsets-1), 5)
	w.writeBits(int32(numCodegens-4), 4)

	for i := 0; i < numCodegens; i++ {
		value := uint(w.codegenEncoding.codes[codegenOrder[i]].len)
		w.writeBits(int32(value), 3)
	}

	i := 0
	for {
		var codeWord = uint32(w.codegen[i])
		i++
		if codeWord == badCode {
			break
		}
		w.writeCode(w.codegenEncoding.codes[codeWord])

		switch codeWord {
		case 16:
			w.writeBits(int32(w.codegen[i]), 2)
			i++
		case 17:
			w.writeBits(int32(w.codegen[i]), 3)
			i++
		case 18:
			w.writeBits(int32(w.codegen[i]), 7)
			i++
		}
	}
}

// writeStoredHeader will write a stored header.
// If the stored block is only used for EOF,
// it is replaced with a fixed huffman block.
func (w *huffmanBitWriter) writeStoredHeader(length int, isEof bool) {
	if w.err != nil {
		return
	}
	if w.lastHeader > 0 {
		// We owe an EOB
		w.writeCode(w.literalEncoding.codes[endBlockMarker])
		w.lastHeader = 0
	}

	// To write EOF, use a fixed encoding block. 10 bits instead of 5 bytes.
	if length == 0 && isEof {
		w.writeFixedHeader(isEof)
		// EOB: 7 bits, value: 0
		w.writeBits(0, 7)
		w.flush()
		return
	}

	var flag int32
	if isEof {
		flag = 1
	}
	w.writeBits(flag, 3)
	w.flush()
	w.writeBits(int32(length), 16)
	w.writeBits(int32(^uint16(length)), 16)
}

func (w *huffmanBitWriter) writeFixedHeader(isEof bool) {
	if w.err != nil {
		return
	}
	if w.lastHeader > 0 {
		// We owe an EOB
		w.writeCode(w.literalEncoding.codes[endBlockMarker])
		w.lastHeader = 0
	}

	// Indicate that we are a fixed Huffman block
	var value int32 = 2
	if isEof {
		value = 3
	}
	w.writeBits(value, 3)
}

// writeBlock will write a block of tokens with the smallest encoding.
// The original input can be supplied, and if the huffman encoded data
// is larger than the original bytes, the data will be written as a
// stored block.
// If the input is nil, the tokens will always be Huffman encoded.
func (w *huffmanBitWriter) writeBlock(tokens *tokens, eof bool, input []byte) {
	if w.err != nil {
		return
	}

	tokens.AddEOB()
	if w.lastHeader > 0 {
		// We owe an EOB
		w.writeCode(w.literalEncoding.codes[endBlockMarker])
		w.lastHeader = 0
	}
	numLiterals, numOffsets := w.indexTokens(tokens, false)
	w.generate(tokens)
	var extraBits int
	storedSize, storable := w.storedSize(input)
	if storable {
		extraBits = w.extraBitSize()
	}

	// Figure out smallest code.
	// Fixed Huffman baseline.
	var literalEncoding = fixedLiteralEncoding
	var offsetEncoding = fixedOffsetEncoding
	var size = w.fixedSize(extraBits)

	// Dynamic Huffman?
	var numCodegens int

	// Generate codegen and codegenFrequencies, which indicates how to encode
	// the literalEncoding and the offsetEncoding.
	w.generateCodegen(numLiterals, numOffsets, w.literalEncoding, w.offsetEncoding)
	w.codegenEncoding.generate(w.codegenFreq[:], 7)
	dynamicSize, numCodegens := w.dynamicSize(w.literalEncoding, w.offsetEncoding, extraBits)

	if dynamicSize < size {
		size = dynamicSize
		literalEncoding = w.literalEncoding
		offsetEncoding = w.offsetEncoding
	}

	// Stored bytes?
	if storable && storedSize < size {
		w.writeStoredHeader(len(input), eof)
		w.writeBytes(input)
		return
	}

	// Huffman.
	if literalEncoding == fixedLiteralEncoding {
		w.writeFixedHeader(eof)
	} else {
		w.writeDynamicHeader(numLiterals, numOffsets, numCodegens, eof)
	}

	// Write the tokens.
	w.writeTokens(tokens.Slice(), literalEncoding.codes, offsetEncoding.codes)
}

// writeBlockDynamic encodes a block using a dynamic Huffman table.
// This should be used if the symbols used have a disproportionate
// histogram distribution.
// If input is supplied and the compression savings are below 1/16th of the
// input size the block is stored.
func (w *huffmanBitWriter) writeBlockDynamic(tokens *tokens, eof bool, input []byte, sync bool) {
	if w.err != nil {
		return
	}

	sync = sync || eof
	if sync {
		tokens.AddEOB()
	}

	// We cannot reuse pure huffman table, and must mark as EOF.
	if (w.lastHuffMan || eof) && w.lastHeader > 0 {
		// We will not try to reuse.
		w.writeCode(w.literalEncoding.codes[endBlockMarker])
		w.lastHeader = 0
		w.lastHuffMan = false
	}
	if !sync {
		tokens.Fill()
	}
	numLiterals, numOffsets := w.indexTokens(tokens, !sync)

	var size int
	// Check if we should reuse.
	if w.lastHeader > 0 {
		// Estimate size for using a new table.
		// Use the previous header size as the best estimate.
		newSize := w.lastHeader + tokens.EstimatedBits()
		newSize += newSize >> w.logNewTablePenalty

		// The estimated size is calculated as an optimal table.
		// We add a penalty to make it more realistic and re-use a bit more.
		reuseSize := w.dynamicReuseSize(w.literalEncoding, w.offsetEncoding) + w.extraBitSize()

		// Check if a new table is better.
		if newSize < reuseSize {
			// Write the EOB we owe.
			w.writeCode(w.literalEncoding.codes[endBlockMarker])
			size = newSize
			w.lastHeader = 0
		} else {
			size = reuseSize
		}
		// Check if we get a reasonable size decrease.
		if ssize, storable := w.storedSize(input); storable && ssize < (size+size>>4) {
			w.writeStoredHeader(len(input), eof)
			w.writeBytes(input)
			w.lastHeader = 0
			return
		}
	}

	// We want a new block/table
	if w.lastHeader == 0 {
		w.generate(tokens)
		// Generate codegen and codegenFrequencies, which indicates how to encode
		// the literalEncoding and the offsetEncoding.
		w.generateCodegen(numLiterals, numOffsets, w.literalEncoding, w.offsetEncoding)
		w.codegenEncoding.generate(w.codegenFreq[:], 7)
		var numCodegens int
		size, numCodegens = w.dynamicSize(w.literalEncoding, w.offsetEncoding, w.extraBitSize())
		// Store bytes, if we don't get a reasonable improvement.
		if ssize, storable := w.storedSize(input); storable && ssize < (size+size>>4) {
			w.writeStoredHeader(len(input), eof)
			w.writeBytes(input)
			w.lastHeader = 0
			return
		}

		// Write Huffman table.
		w.writeDynamicHeader(numLiterals, numOffsets, numCodegens, eof)
		w.lastHeader, _ = w.headerSize()
		w.lastHuffMan = false
	}

	if sync {
		w.lastHeader = 0
	}
	// Write the tokens.
	w.writeTokens(tokens.Slice(), w.literalEncoding.codes, w.offsetEncoding.codes)
}

// indexTokens indexes a slice of tokens, and updates
// literalFreq and offsetFreq, and generates literalEncoding
// and offsetEncoding.
// The number of literal and offset tokens is returned.
func (w *huffmanBitWriter) indexTokens(t *tokens, filled bool) (numLiterals, numOffsets int) {
	copy(w.literalFreq[:], t.litHist[:])
	copy(w.literalFreq[256:], t.extraHist[:])
	copy(w.offsetFreq[:], t.offHist[:offsetCodeCount])

	if t.n == 0 {
		return
	}
	if filled {
		return maxNumLit, maxNumDist
	}
	// get the number of literals
	numLiterals = len(w.literalFreq)
	for w.literalFreq[numLiterals-1] == 0 {
		numLiterals--
	}
	// get the number of offsets
	numOffsets = len(w.offsetFreq)
	for numOffsets > 0 && w.offsetFreq[numOffsets-1] == 0 {
		numOffsets--
	}
	if numOffsets == 0 {
		// We haven't found a single match. If we want to go with the dynamic encoding,
		// we should count at least one offset to be sure that the offset huffman tree could be encoded.
		w.offsetFreq[0] = 1
		numOffsets = 1
	}
	return
}

func (w *huffmanBitWriter) generate(t *tokens) {
	w.literalEncoding.generate(w.literalFreq[:literalCount], 15)
	w.offsetEncoding.generate(w.offsetFreq[:offsetCodeCount], 15)
}

// writeTokens writes a slice of tokens to the output.
// codes for literal and offset encoding must be supplied.
func (w *huffmanBitWriter) writeTokens(tokens []token, leCodes, oeCodes []hcode) {
	if w.err != nil {
		return
	}
	if len(tokens) == 0 {
		return
	}

	// Only last token should be endBlockMarker.
	var deferEOB bool
	if tokens[len(tokens)-1] == endBlockMarker {
		tokens = tokens[:len(tokens)-1]
		deferEOB = true
	}

	// Create slices up to the next power of two to avoid bounds checks.
	lits := leCodes[:256]
	offs := oeCodes[:32]
	lengths := leCodes[lengthCodesStart:]
	lengths = lengths[:32]
	for _, t := range tokens {
		if t < matchType {
			w.writeCode(lits[t.literal()])
			continue
		}

		// Write the length
		length := t.length()
		lengthCode := lengthCode(length)
		if false {
			w.writeCode(lengths[lengthCode&31])
		} else {
			// inlined
			c := lengths[lengthCode&31]
			w.bits |= uint64(c.code) << (w.nbits & reg16SizeMask64)
			w.nbits += c.len
			if w.nbits >= 48 {
				w.writeOutBits()
			}
		}

		extraLengthBits := uint16(lengthExtraBits[lengthCode&31])
		if extraLengthBits > 0 {
			extraLength := int32(length - lengthBase[lengthCode&31])
			w.writeBits(extraLength, extraLengthBits)
		}
		// Write the offset
		offset := t.offset()
		offsetCode := offsetCode(offset)
		if false {
			w.writeCode(offs[offsetCode&31])
		} else {
			// inlined
			c := offs[offsetCode&31]
			w.bits |= uint64(c.code) << (w.nbits & reg16SizeMask64)
			w.nbits += c.len
			if w.nbits >= 48 {
				w.writeOutBits()
			}
		}
		extraOffsetBits := uint16(offsetExtraBits[offsetCode&63])
		if extraOffsetBits > 0 {
			extraOffset := int32(offset - offsetBase[offsetCode&63])
			w.writeBits(extraOffset, extraOffsetBits)
		}
	}
	if deferEOB {
		w.writeCode(leCodes[endBlockMarker])
	}
}

// huffOffset is a static offset encoder used for huffman only encoding.
// It can be reused since we will not be encoding offset values.
var huffOffset *huffmanEncoder

func init() {
	w := newHuffmanBitWriter(nil)
	w.offsetFreq[0] = 1
	huffOffset = newHuffmanEncoder(offsetCodeCount)
	huffOffset.generate(w.offsetFreq[:offsetCodeCount], 15)
}

// writeBlockHuff encodes a block of bytes as either
// Huffman encoded literals or uncompressed bytes if the
// results only gains very little from compression.
func (w *huffmanBitWriter) writeBlockHuff(eof bool, input []byte, sync bool) {
	if w.err != nil {
		return
	}

	// Clear histogram
	for i := range w.literalFreq[:] {
		w.literalFreq[i] = 0
	}
	if !w.lastHuffMan {
		for i := range w.offsetFreq[:] {
			w.offsetFreq[i] = 0
		}
	}

	// Add everything as literals
	// We have to estimate the header size.
	// Assume header is around 70 bytes:
	// https://stackoverflow.com/a/25454430
	const guessHeaderSizeBits = 70 * 8
	estBits, estExtra := histogramSize(input, w.literalFreq[:], !eof && !sync)
	estBits += w.lastHeader + 15
	if w.lastHeader == 0 {
		estBits += guessHeaderSizeBits
	}
	estBits += estBits >> w.logNewTablePenalty

	// Store bytes, if we don't get a reasonable improvement.
	ssize, storable := w.storedSize(input)
	if storable && ssize < estBits {
		w.writeStoredHeader(len(input), eof)
		w.writeBytes(input)
		return
	}

	if w.lastHeader > 0 {
		reuseSize := w.literalEncoding.bitLength(w.literalFreq[:256])
		estBits += estExtra

		if estBits < reuseSize {
			// We owe an EOB
			w.writeCode(w.literalEncoding.codes[endBlockMarker])
			w.lastHeader = 0
		}
	}

	const numLiterals = endBlockMarker + 1
	const numOffsets = 1
	if w.lastHeader == 0 {
		w.literalFreq[endBlockMarker] = 1
		w.literalEncoding.generate(w.literalFreq[:numLiterals], 15)

		// Generate codegen and codegenFrequencies, which indicates how to encode
		// the literalEncoding and the offsetEncoding.
		w.generateCodegen(numLiterals, numOffsets, w.literalEncoding, huffOffset)
		w.codegenEncoding.generate(w.codegenFreq[:], 7)
		numCodegens := w.codegens()

		// Huffman.
		w.writeDynamicHeader(numLiterals, numOffsets, numCodegens, eof)
		w.lastHuffMan = true
		w.lastHeader, _ = w.headerSize()
	}

	encoding := w.literalEncoding.codes[:257]
	for _, t := range input {
		// Bitwriting inlined, ~30% speedup
		c := encoding[t]
		w.bits |= uint64(c.code) << ((w.nbits) & reg16SizeMask64)
		w.nbits += c.len
		if w.nbits >= 48 {
			bits := w.bits
			w.bits >>= 48
			w.nbits -= 48
			n := w.nbytes
			w.bytes[n] = byte(bits)
			w.bytes[n+1] = byte(bits >> 8)
			w.bytes[n+2] = byte(bits >> 16)
			w.bytes[n+3] = byte(bits >> 24)
			w.bytes[n+4] = byte(bits >> 32)
			w.bytes[n+5] = byte(bits >> 40)
			n += 6
			if n >= bufferFlushSize {
				if w.err != nil {
					n = 0
					return
				}
				w.write(w.bytes[:n])
				n = 0
			}
			w.nbytes = n
		}
	}
	if eof || sync {
		w.writeCode(encoding[endBlockMarker])
		w.lastHeader = 0
		w.lastHuffMan = false
	}
}
