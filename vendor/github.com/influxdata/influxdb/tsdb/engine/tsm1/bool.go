package tsm1

// boolean encoding uses 1 bit per value.  Each compressed byte slice contains a 1 byte header
// indicating the compression type, followed by a variable byte encoded length indicating
// how many booleans are packed in the slice.  The remaining bytes contains 1 byte for every
// 8 boolean values encoded.

import (
	"encoding/binary"
	"fmt"
)

const (
	// booleanUncompressed is an uncompressed boolean format.
	// Not yet implemented.
	booleanUncompressed = 0

	// booleanCompressedBitPacked is an bit packed format using 1 bit per boolean
	booleanCompressedBitPacked = 1
)

// BooleanEncoder encodes a series of booleans to an in-memory buffer.
type BooleanEncoder struct {
	// The encoded bytes
	bytes []byte

	// The current byte being encoded
	b byte

	// The number of bools packed into b
	i int

	// The total number of bools written
	n int
}

// NewBooleanEncoder returns a new instance of BooleanEncoder.
func NewBooleanEncoder(sz int) BooleanEncoder {
	return BooleanEncoder{
		bytes: make([]byte, 0, (sz+7)/8),
	}
}

func (e *BooleanEncoder) Reset() {
	e.bytes = e.bytes[:0]
	e.b = 0
	e.i = 0
	e.n = 0
}

func (e *BooleanEncoder) Write(b bool) {
	// If we have filled the current byte, flush it
	if e.i >= 8 {
		e.flush()
	}

	// Use 1 bit for each boolean value, shift the current byte
	// by 1 and set the least signficant bit acordingly
	e.b = e.b << 1
	if b {
		e.b |= 1
	}

	// Increment the current boolean count
	e.i++
	// Increment the total boolean count
	e.n++
}

func (e *BooleanEncoder) flush() {
	// Pad remaining byte w/ 0s
	for e.i < 8 {
		e.b = e.b << 1
		e.i++
	}

	// If we have bits set, append them to the byte slice
	if e.i > 0 {
		e.bytes = append(e.bytes, e.b)
		e.b = 0
		e.i = 0
	}
}

func (e *BooleanEncoder) Bytes() ([]byte, error) {
	// Ensure the current byte is flushed
	e.flush()
	b := make([]byte, 10+1)

	// Store the encoding type in the 4 high bits of the first byte
	b[0] = byte(booleanCompressedBitPacked) << 4

	i := 1
	// Encode the number of booleans written
	i += binary.PutUvarint(b[i:], uint64(e.n))

	// Append the packed booleans
	return append(b[:i], e.bytes...), nil
}

// BooleanDecoder decodes a series of booleans from an in-memory buffer.
type BooleanDecoder struct {
	b   []byte
	i   int
	n   int
	err error
}

// SetBytes initializes the decoder with a new set of bytes to read from.
// This must be called before calling any other methods.
func (e *BooleanDecoder) SetBytes(b []byte) {
	if len(b) == 0 {
		return
	}

	// First byte stores the encoding type, only have 1 bit-packet format
	// currently ignore for now.
	b = b[1:]
	count, n := binary.Uvarint(b)
	if n <= 0 {
		e.err = fmt.Errorf("BooleanDecoder: invalid count")
		return
	}

	e.b = b[n:]
	e.i = -1
	e.n = int(count)

	if min := len(e.b) * 8; min < e.n {
		// Shouldn't happen - TSM file was truncated/corrupted
		e.n = min
	}
}

func (e *BooleanDecoder) Next() bool {
	if e.err != nil {
		return false
	}

	e.i++
	return e.i < e.n
}

func (e *BooleanDecoder) Read() bool {
	// Index into the byte slice
	idx := e.i >> 3 // integer division by 8

	// Bit position
	pos := 7 - (e.i & 0x7)

	// The mask to select the bit
	mask := byte(1 << uint(pos))

	// The packed byte
	v := e.b[idx]

	// Returns true if the bit is set
	return v&mask == mask
}

func (e *BooleanDecoder) Error() error {
	return e.err
}
