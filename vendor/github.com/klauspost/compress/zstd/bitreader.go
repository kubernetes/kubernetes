// Copyright 2019+ Klaus Post. All rights reserved.
// License information can be found in the LICENSE file.
// Based on work by Yann Collet, released under BSD License.

package zstd

import (
	"errors"
	"fmt"
	"io"
	"math/bits"

	"github.com/klauspost/compress/internal/le"
)

// bitReader reads a bitstream in reverse.
// The last set bit indicates the start of the stream and is used
// for aligning the input.
type bitReader struct {
	in       []byte
	value    uint64 // Maybe use [16]byte, but shifting is awkward.
	cursor   int    // offset where next read should end
	bitsRead uint8
}

// init initializes and resets the bit reader.
func (b *bitReader) init(in []byte) error {
	if len(in) < 1 {
		return errors.New("corrupt stream: too short")
	}
	b.in = in
	// The highest bit of the last byte indicates where to start
	v := in[len(in)-1]
	if v == 0 {
		return errors.New("corrupt stream, did not find end of stream")
	}
	b.cursor = len(in)
	b.bitsRead = 64
	b.value = 0
	if len(in) >= 8 {
		b.fillFastStart()
	} else {
		b.fill()
		b.fill()
	}
	b.bitsRead += 8 - uint8(highBits(uint32(v)))
	return nil
}

// getBits will return n bits. n can be 0.
func (b *bitReader) getBits(n uint8) int {
	if n == 0 /*|| b.bitsRead >= 64 */ {
		return 0
	}
	return int(b.get32BitsFast(n))
}

// get32BitsFast requires that at least one bit is requested every time.
// There are no checks if the buffer is filled.
func (b *bitReader) get32BitsFast(n uint8) uint32 {
	const regMask = 64 - 1
	v := uint32((b.value << (b.bitsRead & regMask)) >> ((regMask + 1 - n) & regMask))
	b.bitsRead += n
	return v
}

// fillFast() will make sure at least 32 bits are available.
// There must be at least 4 bytes available.
func (b *bitReader) fillFast() {
	if b.bitsRead < 32 {
		return
	}
	b.cursor -= 4
	b.value = (b.value << 32) | uint64(le.Load32(b.in, b.cursor))
	b.bitsRead -= 32
}

// fillFastStart() assumes the bitreader is empty and there is at least 8 bytes to read.
func (b *bitReader) fillFastStart() {
	b.cursor -= 8
	b.value = le.Load64(b.in, b.cursor)
	b.bitsRead = 0
}

// fill() will make sure at least 32 bits are available.
func (b *bitReader) fill() {
	if b.bitsRead < 32 {
		return
	}
	if b.cursor >= 4 {
		b.cursor -= 4
		b.value = (b.value << 32) | uint64(le.Load32(b.in, b.cursor))
		b.bitsRead -= 32
		return
	}

	b.bitsRead -= uint8(8 * b.cursor)
	for b.cursor > 0 {
		b.cursor -= 1
		b.value = (b.value << 8) | uint64(b.in[b.cursor])
	}
}

// finished returns true if all bits have been read from the bit stream.
func (b *bitReader) finished() bool {
	return b.cursor == 0 && b.bitsRead >= 64
}

// overread returns true if more bits have been requested than is on the stream.
func (b *bitReader) overread() bool {
	return b.bitsRead > 64
}

// remain returns the number of bits remaining.
func (b *bitReader) remain() uint {
	return 8*uint(b.cursor) + 64 - uint(b.bitsRead)
}

// close the bitstream and returns an error if out-of-buffer reads occurred.
func (b *bitReader) close() error {
	// Release reference.
	b.in = nil
	b.cursor = 0
	if !b.finished() {
		return fmt.Errorf("%d extra bits on block, should be 0", b.remain())
	}
	if b.bitsRead > 64 {
		return io.ErrUnexpectedEOF
	}
	return nil
}

func highBits(val uint32) (n uint32) {
	return uint32(bits.Len32(val) - 1)
}
