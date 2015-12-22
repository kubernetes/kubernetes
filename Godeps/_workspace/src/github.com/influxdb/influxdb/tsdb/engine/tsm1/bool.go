package tsm1

// bool encoding uses 1 bit per value.  Each compressed byte slice contains a 1 byte header
// indicating the compression type, followed by a variable byte encoded length indicating
// how many booleans are packed in the slice.  The remaining bytes contains 1 byte for every
// 8 boolean values encoded.

import "encoding/binary"

const (
	// boolUncompressed is an uncompressed boolean format.
	// Not yet implemented.
	boolUncompressed = 0

	// boolCompressedBitPacked is an bit packed format using 1 bit per boolean
	boolCompressedBitPacked = 1
)

// BoolEncoder encodes a series of bools to an in-memory buffer.
type BoolEncoder interface {
	Write(b bool)
	Bytes() ([]byte, error)
}

type boolEncoder struct {
	// The encoded bytes
	bytes []byte

	// The current byte being encoded
	b byte

	// The number of bools packed into b
	i int

	// The total number of bools written
	n int
}

// NewBoolEncoder returns a new instance of BoolEncoder.
func NewBoolEncoder() BoolEncoder {
	return &boolEncoder{}
}

func (e *boolEncoder) Write(b bool) {
	// If we have filled the current byte, flush it
	if e.i >= 8 {
		e.flush()
	}

	// Use 1 bit for each boolen value, shift the current byte
	// by 1 and set the least signficant bit acordingly
	e.b = e.b << 1
	if b {
		e.b |= 1
	}

	// Increment the current bool count
	e.i++
	// Increment the total bool count
	e.n++
}

func (e *boolEncoder) flush() {
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

func (e *boolEncoder) Bytes() ([]byte, error) {
	// Ensure the current byte is flushed
	e.flush()
	b := make([]byte, 10+1)

	// Store the encoding type in the 4 high bits of the first byte
	b[0] = byte(boolCompressedBitPacked) << 4

	i := 1
	// Encode the number of bools written
	i += binary.PutUvarint(b[i:], uint64(e.n))

	// Append the packed booleans
	return append(b[:i], e.bytes...), nil
}

// BoolDecoder decodes a series of bools from an in-memory buffer.
type BoolDecoder interface {
	Next() bool
	Read() bool
	Error() error
}

type boolDecoder struct {
	b   []byte
	i   int
	n   int
	err error
}

// NewBoolDecoder returns a new instance of BoolDecoder.
func NewBoolDecoder(b []byte) BoolDecoder {
	// First byte stores the encoding type, only have 1 bit-packet format
	// currently ignore for now.
	b = b[1:]
	count, n := binary.Uvarint(b)
	return &boolDecoder{b: b[n:], i: -1, n: int(count)}
}

func (e *boolDecoder) Next() bool {
	e.i++
	return e.i < e.n
}

func (e *boolDecoder) Read() bool {
	// Index into the byte slice
	idx := e.i / 8

	// Bit position
	pos := (8 - e.i%8) - 1

	// The mask to select the bit
	mask := byte(1 << uint(pos))

	// The packed byte
	v := e.b[idx]

	// Returns true if the bit is set
	return v&mask == mask
}

func (e *boolDecoder) Error() error {
	return e.err
}
