package tsm1

// String encoding uses snappy compression to compress each string.  Each string is
// appended to byte slice prefixed with a variable byte length followed by the string
// bytes.  The bytes are compressed using snappy compressor and a 1 byte header is used
// to indicate the type of encoding.

import (
	"encoding/binary"
	"fmt"

	"github.com/golang/snappy"
)

const (
	// stringUncompressed is a an uncompressed format encoding strings as raw bytes.
	// Not yet implemented.
	stringUncompressed = 0

	// stringCompressedSnappy is a compressed encoding using Snappy compression
	stringCompressedSnappy = 1
)

type StringEncoder struct {
	// The encoded bytes
	bytes []byte
}

func NewStringEncoder(sz int) StringEncoder {
	return StringEncoder{
		bytes: make([]byte, 0, sz),
	}
}

func (e *StringEncoder) Reset() {
	e.bytes = e.bytes[:0]
}

func (e *StringEncoder) Write(s string) {
	b := make([]byte, 10)
	// Append the length of the string using variable byte encoding
	i := binary.PutUvarint(b, uint64(len(s)))
	e.bytes = append(e.bytes, b[:i]...)

	// Append the string bytes
	e.bytes = append(e.bytes, s...)
}

func (e *StringEncoder) Bytes() ([]byte, error) {
	// Compress the currently appended bytes using snappy and prefix with
	// a 1 byte header for future extension
	data := snappy.Encode(nil, e.bytes)
	return append([]byte{stringCompressedSnappy << 4}, data...), nil
}

type StringDecoder struct {
	b   []byte
	l   int
	i   int
	err error
}

// SetBytes initializes the decoder with bytes to read from.
// This must be called before calling any other method.
func (e *StringDecoder) SetBytes(b []byte) error {
	// First byte stores the encoding type, only have snappy format
	// currently so ignore for now.
	var data []byte
	if len(b) > 0 {
		var err error
		data, err = snappy.Decode(nil, b[1:])
		if err != nil {
			return fmt.Errorf("failed to decode string block: %v", err.Error())
		}
	}

	e.b = data
	e.l = 0
	e.i = 0
	e.err = nil

	return nil
}

func (e *StringDecoder) Next() bool {
	if e.err != nil {
		return false
	}

	e.i += e.l
	return e.i < len(e.b)
}

func (e *StringDecoder) Read() string {
	// Read the length of the string
	length, n := binary.Uvarint(e.b[e.i:])
	if n <= 0 {
		e.err = fmt.Errorf("StringDecoder: invalid encoded string length")
		return ""
	}

	// The length of this string plus the length of the variable byte encoded length
	e.l = int(length) + n

	lower := e.i + n
	upper := lower + int(length)
	if upper < lower {
		e.err = fmt.Errorf("StringDecoder: length overflow")
		return ""
	}
	if upper > len(e.b) {
		e.err = fmt.Errorf("StringDecoder: not enough data to represent encoded string")
		return ""
	}

	return string(e.b[lower:upper])
}

func (e *StringDecoder) Error() error {
	return e.err
}
