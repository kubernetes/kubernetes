package tsm1

/*
This code is originally from: https://github.com/dgryski/go-tsz and has been modified to remove
the timestamp compression fuctionality.

It implements the float compression as presented in: http://www.vldb.org/pvldb/vol8/p1816-teller.pdf.
This implementation uses a sentinel value of NaN which means that float64 NaN cannot be stored using
this version.
*/

import (
	"bytes"
	"fmt"
	"math"

	"github.com/dgryski/go-bits"
	"github.com/dgryski/go-bitstream"
)

const (
	// floatUncompressed is an uncompressed format using 8 bytes per value.
	// Not yet implemented.
	floatUncompressed = 0

	// floatCompressedGorilla is a compressed format using the gorilla paper encoding
	floatCompressedGorilla = 1
)

// uvnan is the constant returned from math.NaN().
const uvnan = 0x7FF8000000000001

// FloatEncoder encodes multiple float64s into a byte slice
type FloatEncoder struct {
	val float64
	err error

	leading  uint64
	trailing uint64

	buf bytes.Buffer
	bw  *bitstream.BitWriter

	first    bool
	finished bool
}

func NewFloatEncoder() *FloatEncoder {
	s := FloatEncoder{
		first:   true,
		leading: ^uint64(0),
	}

	s.bw = bitstream.NewWriter(&s.buf)
	s.buf.WriteByte(floatCompressedGorilla << 4)

	return &s

}

func (s *FloatEncoder) Reset() {
	s.val = 0
	s.err = nil
	s.leading = ^uint64(0)
	s.trailing = 0
	s.buf.Reset()
	s.buf.WriteByte(floatCompressedGorilla << 4)

	s.bw.Resume(0x0, 8)

	s.finished = false
	s.first = true
}

func (s *FloatEncoder) Bytes() ([]byte, error) {
	return s.buf.Bytes(), s.err
}

func (s *FloatEncoder) Finish() {
	if !s.finished {
		// write an end-of-stream record
		s.finished = true
		s.Push(math.NaN())
		s.bw.Flush(bitstream.Zero)
	}
}

func (s *FloatEncoder) Push(v float64) {
	// Only allow NaN as a sentinel value
	if math.IsNaN(v) && !s.finished {
		s.err = fmt.Errorf("unsupported value: NaN")
		return
	}
	if s.first {
		// first point
		s.val = v
		s.first = false
		s.bw.WriteBits(math.Float64bits(v), 64)
		return
	}

	vDelta := math.Float64bits(v) ^ math.Float64bits(s.val)

	if vDelta == 0 {
		s.bw.WriteBit(bitstream.Zero)
	} else {
		s.bw.WriteBit(bitstream.One)

		leading := bits.Clz(vDelta)
		trailing := bits.Ctz(vDelta)

		// Clamp number of leading zeros to avoid overflow when encoding
		leading &= 0x1F
		if leading >= 32 {
			leading = 31
		}

		// TODO(dgryski): check if it's 'cheaper' to reset the leading/trailing bits instead
		if s.leading != ^uint64(0) && leading >= s.leading && trailing >= s.trailing {
			s.bw.WriteBit(bitstream.Zero)
			s.bw.WriteBits(vDelta>>s.trailing, 64-int(s.leading)-int(s.trailing))
		} else {
			s.leading, s.trailing = leading, trailing

			s.bw.WriteBit(bitstream.One)
			s.bw.WriteBits(leading, 5)

			// Note that if leading == trailing == 0, then sigbits == 64.  But that
			// value doesn't actually fit into the 6 bits we have.
			// Luckily, we never need to encode 0 significant bits, since that would
			// put us in the other case (vdelta == 0).  So instead we write out a 0 and
			// adjust it back to 64 on unpacking.
			sigbits := 64 - leading - trailing
			s.bw.WriteBits(sigbits, 6)
			s.bw.WriteBits(vDelta>>trailing, int(sigbits))
		}
	}

	s.val = v
}

// FloatDecoder decodes a byte slice into multipe float64 values
type FloatDecoder struct {
	val uint64

	leading  uint64
	trailing uint64

	br BitReader
	b  []byte

	first    bool
	finished bool

	err error
}

// SetBytes initializes the decoder with b. Must call before calling Next().
func (it *FloatDecoder) SetBytes(b []byte) error {
	var v uint64
	if len(b) == 0 {
		v = uvnan
	} else {
		// first byte is the compression type.
		// we currently just have gorilla compression.
		it.br.Reset(b[1:])

		var err error
		v, err = it.br.ReadBits(64)
		if err != nil {
			return err
		}
	}

	// Reset all fields.
	it.val = v
	it.leading = 0
	it.trailing = 0
	it.b = b
	it.first = true
	it.finished = false
	it.err = nil

	return nil
}

func (it *FloatDecoder) Next() bool {
	if it.err != nil || it.finished {
		return false
	}

	if it.first {
		it.first = false

		// mark as finished if there were no values.
		if it.val == uvnan { // IsNaN
			it.finished = true
			return false
		}

		return true
	}

	// read compressed value
	var bit bool
	if it.br.CanReadBitFast() {
		bit = it.br.ReadBitFast()
	} else if v, err := it.br.ReadBit(); err != nil {
		it.err = err
		return false
	} else {
		bit = v
	}

	if !bit {
		// it.val = it.val
	} else {
		var bit bool
		if it.br.CanReadBitFast() {
			bit = it.br.ReadBitFast()
		} else if v, err := it.br.ReadBit(); err != nil {
			it.err = err
			return false
		} else {
			bit = v
		}

		if !bit {
			// reuse leading/trailing zero bits
			// it.leading, it.trailing = it.leading, it.trailing
		} else {
			bits, err := it.br.ReadBits(5)
			if err != nil {
				it.err = err
				return false
			}
			it.leading = bits

			bits, err = it.br.ReadBits(6)
			if err != nil {
				it.err = err
				return false
			}
			mbits := bits
			// 0 significant bits here means we overflowed and we actually need 64; see comment in encoder
			if mbits == 0 {
				mbits = 64
			}
			it.trailing = 64 - it.leading - mbits
		}

		mbits := uint(64 - it.leading - it.trailing)
		bits, err := it.br.ReadBits(mbits)
		if err != nil {
			it.err = err
			return false
		}

		vbits := it.val
		vbits ^= (bits << it.trailing)

		if vbits == uvnan { // IsNaN
			it.finished = true
			return false
		}
		it.val = vbits
	}

	return true
}

func (it *FloatDecoder) Values() float64 {
	return math.Float64frombits(it.val)
}

func (it *FloatDecoder) Error() error {
	return it.err
}
