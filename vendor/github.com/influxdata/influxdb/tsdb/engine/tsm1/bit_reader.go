package tsm1

import "io"

// BitReader reads bits from an io.Reader.
type BitReader struct {
	data []byte

	buf struct {
		v uint64 // bit buffer
		n uint   // available bits
	}
}

// NewBitReader returns a new instance of BitReader that reads from data.
func NewBitReader(data []byte) *BitReader {
	b := new(BitReader)
	b.Reset(data)
	return b
}

// Reset sets the underlying reader on b and reinitializes.
func (r *BitReader) Reset(data []byte) {
	r.data = data
	r.buf.v, r.buf.n = 0, 0
	r.readBuf()
}

// CanReadBitFast returns true if calling ReadBitFast() is allowed.
// Fast bit reads are allowed when at least 2 values are in the buffer.
// This is because it is not required to refilled the buffer and the caller
// can inline the calls.
func (r *BitReader) CanReadBitFast() bool { return r.buf.n > 1 }

// ReadBitFast is an optimized bit read.
// IMPORTANT: Only allowed if CanReadFastBit() is true!
func (r *BitReader) ReadBitFast() bool {
	v := (r.buf.v&(1<<63) != 0)
	r.buf.v <<= 1
	r.buf.n -= 1
	return v
}

// ReadBit returns the next bit from the underlying data.
func (r *BitReader) ReadBit() (bool, error) {
	v, err := r.ReadBits(1)
	return v != 0, err
}

// ReadBits reads nbits from the underlying data.
func (r *BitReader) ReadBits(nbits uint) (uint64, error) {
	// Return EOF if there is no more data.
	if r.buf.n == 0 {
		return 0, io.EOF
	}

	// Return bits from buffer if less than available bits.
	if nbits <= r.buf.n {
		// Return all bits, if requested.
		if nbits == 64 {
			v := r.buf.v
			r.buf.v, r.buf.n = 0, 0
			r.readBuf()
			return v, nil
		}

		// Otherwise mask returned bits.
		v := (r.buf.v >> (64 - nbits))
		r.buf.v <<= nbits
		r.buf.n -= nbits

		if r.buf.n == 0 {
			r.readBuf()
		}
		return v, nil
	}

	// Otherwise read all available bits in current buffer.
	v, n := r.buf.v, r.buf.n

	// Read new buffer.
	r.buf.v, r.buf.n = 0, 0
	r.readBuf()

	// Append new buffer to previous buffer and shift to remove unnecessary bits.
	v |= (r.buf.v >> n)
	v >>= 64 - nbits

	// Remove used bits from new buffer.
	bufN := nbits - n
	if bufN > r.buf.n {
		bufN = r.buf.n
	}
	r.buf.v <<= bufN
	r.buf.n -= bufN

	if r.buf.n == 0 {
		r.readBuf()
	}

	return v, nil
}

func (r *BitReader) readBuf() {
	// Determine number of bytes to read to fill buffer.
	byteN := 8 - (r.buf.n / 8)

	// Limit to the length of our data.
	if n := uint(len(r.data)); byteN > n {
		byteN = n
	}

	// Optimized 8-byte read.
	if byteN == 8 {
		r.buf.v = uint64(r.data[7]) | uint64(r.data[6])<<8 |
			uint64(r.data[5])<<16 | uint64(r.data[4])<<24 |
			uint64(r.data[3])<<32 | uint64(r.data[2])<<40 |
			uint64(r.data[1])<<48 | uint64(r.data[0])<<56
		r.buf.n = 64
		r.data = r.data[8:]
		return
	}

	// Otherwise append bytes to buffer.
	for i := uint(0); i < byteN; i++ {
		r.buf.n += 8
		r.buf.v |= uint64(r.data[i]) << (64 - r.buf.n)
	}

	// Move data forward.
	r.data = r.data[byteN:]
}
