package tsm1

// Timestamp encoding is adaptive and based on structure of the timestamps that are encoded.  It
// uses a combination of delta encoding, scaling and compression using simple8b, run length encoding
// as well as falling back to no compression if needed.
//
// Timestamp values to be encoded should be sorted before encoding.  When encoded, the values are
// first delta-encoded.  The first value is the starting timestamp, subsequent values are the difference.
// from the prior value.
//
// Timestamp resolution can also be in the nanosecond.  Many timestamps are monotonically increasing
// and fall on even boundaries of time such as every 10s.  When the timestamps have this structure,
// they are scaled by the largest common divisor that is also a factor of 10.  This has the effect
// of converting very large integer deltas into very small one that can be reversed by multiplying them
// by the scaling factor.
//
// Using these adjusted values, if all the deltas are the same, the time range is stored using run
// length encoding.  If run length encoding is not possible and all values are less than 1 << 60 - 1
//  (~36.5 yrs in nanosecond resolution), then the timestamps are encoded using simple8b encoding.  If
// any value exceeds the maximum values, the deltas are stored uncompressed using 8b each.
//
// Each compressed byte slice has a 1 byte header indicating the compression type.  The 4 high bits
// indicated the encoding type.  The 4 low bits are used by the encoding type.
//
// For run-length encoding, the 4 low bits store the log10 of the scaling factor.  The next 8 bytes are
// the starting timestamp, next 1-10 bytes is the delta value using variable-length encoding, finally the
// next 1-10 bytes is the count of values.
//
// For simple8b encoding, the 4 low bits store the log10 of the scaling factor.  The next 8 bytes is the
// first delta value stored uncompressed, the remaining bytes are 64bit words containg compressed delta
// values.
//
// For uncompressed encoding, the delta values are stored using 8 bytes each.

import (
	"encoding/binary"
	"fmt"
	"math"
	"time"

	"github.com/jwilder/encoding/simple8b"
)

const (
	// timeUncompressed is a an uncompressed format using 8 bytes per timestamp
	timeUncompressed = 0
	// timeCompressedPackedSimple is a bit-packed format using simple8b encoding
	timeCompressedPackedSimple = 1
	// timeCompressedRLE is a run-length encoding format
	timeCompressedRLE = 2
)

// TimeEncoder encodes time.Time to byte slices.
type TimeEncoder interface {
	Write(t time.Time)
	Bytes() ([]byte, error)
}

// TimeDecoder decodes byte slices to time.Time values.
type TimeDecoder interface {
	Next() bool
	Read() time.Time
	Error() error
}

type encoder struct {
	ts []uint64
}

// NewTimeEncoder returns a TimeEncoder
func NewTimeEncoder() TimeEncoder {
	return &encoder{}
}

// Write adds a time.Time to the compressed stream.
func (e *encoder) Write(t time.Time) {
	e.ts = append(e.ts, uint64(t.UnixNano()))
}

func (e *encoder) reduce() (max, divisor uint64, rle bool, deltas []uint64) {
	// Compute the deltas in place to avoid allocating another slice
	deltas = e.ts
	// Starting values for a max and divisor
	max, divisor = 0, 1e12

	// Indicates whether the the deltas can be run-length encoded
	rle = true

	// Iterate in reverse so we can apply deltas in place
	for i := len(deltas) - 1; i > 0; i-- {

		// First differential encode the values
		deltas[i] = deltas[i] - deltas[i-1]

		// We also need to keep track of the max value and largest common divisor
		v := deltas[i]

		if v > max {
			max = v
		}

		for {
			// If our value is divisible by 10, break.  Otherwise, try the next smallest divisor.
			if v%divisor == 0 {
				break
			}
			divisor /= 10
		}

		// Skip the first value || see if prev = curr.  The deltas can be RLE if the are all equal.
		rle = i == len(deltas)-1 || rle && (deltas[i+1] == deltas[i])
	}
	return
}

// Bytes returns the encoded bytes of all written times.
func (e *encoder) Bytes() ([]byte, error) {
	if len(e.ts) == 0 {
		return []byte{}, nil
	}

	// Maximum and largest common divisor.  rle is true if dts (the delta timestamps),
	// are all the same.
	max, div, rle, dts := e.reduce()

	// The deltas are all the same, so we can run-length encode them
	if rle && len(e.ts) > 1 {
		return e.encodeRLE(e.ts[0], e.ts[1], div, len(e.ts))
	}

	// We can't compress this time-range, the deltas exceed 1 << 60
	if max > simple8b.MaxValue {
		return e.encodeRaw()
	}

	return e.encodePacked(div, dts)
}

func (e *encoder) encodePacked(div uint64, dts []uint64) ([]byte, error) {
	enc := simple8b.NewEncoder()
	for _, v := range dts[1:] {
		enc.Write(uint64(v) / div)
	}

	b := make([]byte, 8+1)

	// 4 high bits used for the encoding type
	b[0] = byte(timeCompressedPackedSimple) << 4
	// 4 low bits are the log10 divisor
	b[0] |= byte(math.Log10(float64(div)))

	// The first delta value
	binary.BigEndian.PutUint64(b[1:9], uint64(dts[0]))

	// The compressed deltas
	deltas, err := enc.Bytes()
	if err != nil {
		return nil, err
	}

	return append(b, deltas...), nil
}

func (e *encoder) encodeRaw() ([]byte, error) {
	b := make([]byte, 1+len(e.ts)*8)
	b[0] = byte(timeUncompressed) << 4
	for i, v := range e.ts {
		binary.BigEndian.PutUint64(b[1+i*8:1+i*8+8], uint64(v))
	}
	return b, nil
}

func (e *encoder) encodeRLE(first, delta, div uint64, n int) ([]byte, error) {
	// Large varints can take up to 10 bytes
	b := make([]byte, 1+10*3)

	// 4 high bits used for the encoding type
	b[0] = byte(timeCompressedRLE) << 4
	// 4 low bits are the log10 divisor
	b[0] |= byte(math.Log10(float64(div)))

	i := 1
	// The first timestamp
	binary.BigEndian.PutUint64(b[i:], uint64(first))
	i += 8
	// The first delta
	i += binary.PutUvarint(b[i:], uint64(delta/div))
	// The number of times the delta is repeated
	i += binary.PutUvarint(b[i:], uint64(n))

	return b[:i], nil
}

type decoder struct {
	v   time.Time
	ts  []uint64
	err error
}

func NewTimeDecoder(b []byte) TimeDecoder {
	d := &decoder{}
	d.decode(b)
	return d
}

func (d *decoder) Next() bool {
	if len(d.ts) == 0 {
		return false
	}
	d.v = time.Unix(0, int64(d.ts[0]))
	d.ts = d.ts[1:]
	return true
}

func (d *decoder) Read() time.Time {
	return d.v
}

func (d *decoder) Error() error {
	return d.err
}

func (d *decoder) decode(b []byte) {
	if len(b) == 0 {
		return
	}

	// Encoding type is stored in the 4 high bits of the first byte
	encoding := b[0] >> 4
	switch encoding {
	case timeUncompressed:
		d.decodeRaw(b[1:])
	case timeCompressedRLE:
		d.decodeRLE(b)
	case timeCompressedPackedSimple:
		d.decodePacked(b)
	default:
		d.err = fmt.Errorf("unknown encoding: %v", encoding)
	}
}

func (d *decoder) decodePacked(b []byte) {
	div := uint64(math.Pow10(int(b[0] & 0xF)))
	first := uint64(binary.BigEndian.Uint64(b[1:9]))

	enc := simple8b.NewDecoder(b[9:])

	deltas := []uint64{first}
	for enc.Next() {
		deltas = append(deltas, enc.Read())
	}

	// Compute the prefix sum and scale the deltas back up
	for i := 1; i < len(deltas); i++ {
		dgap := deltas[i] * div
		deltas[i] = deltas[i-1] + dgap
	}

	d.ts = deltas
}

func (d *decoder) decodeRLE(b []byte) {
	var i, n int

	// Lower 4 bits hold the 10 based exponent so we can scale the values back up
	mod := int64(math.Pow10(int(b[i] & 0xF)))
	i++

	// Next 8 bytes is the starting timestamp
	first := binary.BigEndian.Uint64(b[i : i+8])
	i += 8

	// Next 1-10 bytes is our (scaled down by factor of 10) run length values
	value, n := binary.Uvarint(b[i:])

	// Scale the value back up
	value *= uint64(mod)
	i += n

	// Last 1-10 bytes is how many times the value repeats
	count, _ := binary.Uvarint(b[i:])

	// Rebuild construct the original values now
	deltas := make([]uint64, count)
	for i := range deltas {
		deltas[i] = value
	}

	// Reverse the delta-encoding
	deltas[0] = first
	for i := 1; i < len(deltas); i++ {
		deltas[i] = deltas[i-1] + deltas[i]
	}

	d.ts = deltas
}

func (d *decoder) decodeRaw(b []byte) {
	d.ts = make([]uint64, len(b)/8)
	for i := range d.ts {
		d.ts[i] = binary.BigEndian.Uint64(b[i*8 : i*8+8])

		delta := d.ts[i]
		// Compute the prefix sum and scale the deltas back up
		if i > 0 {
			d.ts[i] = d.ts[i-1] + delta
		}
	}
}
