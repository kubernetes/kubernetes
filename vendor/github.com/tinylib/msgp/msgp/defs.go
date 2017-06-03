// This package is the support library for the msgp code generator (http://github.com/tinylib/msgp).
//
// This package defines the utilites used by the msgp code generator for encoding and decoding MessagePack
// from []byte and io.Reader/io.Writer types. Much of this package is devoted to helping the msgp code
// generator implement the Marshaler/Unmarshaler and Encodable/Decodable interfaces.
//
// This package defines four "families" of functions:
// 	- AppendXxxx() appends an object to a []byte in MessagePack encoding.
// 	- ReadXxxxBytes() reads an object from a []byte and returns the remaining bytes.
// 	- (*Writer).WriteXxxx() writes an object to the buffered *Writer type.
// 	- (*Reader).ReadXxxx() reads an object from a buffered *Reader type.
//
// Once a type has satisfied the `Encodable` and `Decodable` interfaces,
// it can be written and read from arbitrary `io.Writer`s and `io.Reader`s using
// 		msgp.Encode(io.Writer, msgp.Encodable)
// and
//		msgp.Decode(io.Reader, msgp.Decodable)
//
// There are also methods for converting MessagePack to JSON without
// an explicit de-serialization step.
//
// For additional tips, tricks, and gotchas, please visit
// the wiki at http://github.com/tinylib/msgp
package msgp

const last4 = 0x0f
const first4 = 0xf0
const last5 = 0x1f
const first3 = 0xe0
const last7 = 0x7f

func isfixint(b byte) bool {
	return b>>7 == 0
}

func isnfixint(b byte) bool {
	return b&first3 == mnfixint
}

func isfixmap(b byte) bool {
	return b&first4 == mfixmap
}

func isfixarray(b byte) bool {
	return b&first4 == mfixarray
}

func isfixstr(b byte) bool {
	return b&first3 == mfixstr
}

func wfixint(u uint8) byte {
	return u & last7
}

func rfixint(b byte) uint8 {
	return b
}

func wnfixint(i int8) byte {
	return byte(i) | mnfixint
}

func rnfixint(b byte) int8 {
	return int8(b)
}

func rfixmap(b byte) uint8 {
	return b & last4
}

func wfixmap(u uint8) byte {
	return mfixmap | (u & last4)
}

func rfixstr(b byte) uint8 {
	return b & last5
}

func wfixstr(u uint8) byte {
	return (u & last5) | mfixstr
}

func rfixarray(b byte) uint8 {
	return (b & last4)
}

func wfixarray(u uint8) byte {
	return (u & last4) | mfixarray
}

// These are all the byte
// prefixes defined by the
// msgpack standard
const (
	// 0XXXXXXX
	mfixint uint8 = 0x00

	// 111XXXXX
	mnfixint uint8 = 0xe0

	// 1000XXXX
	mfixmap uint8 = 0x80

	// 1001XXXX
	mfixarray uint8 = 0x90

	// 101XXXXX
	mfixstr uint8 = 0xa0

	mnil      uint8 = 0xc0
	mfalse    uint8 = 0xc2
	mtrue     uint8 = 0xc3
	mbin8     uint8 = 0xc4
	mbin16    uint8 = 0xc5
	mbin32    uint8 = 0xc6
	mext8     uint8 = 0xc7
	mext16    uint8 = 0xc8
	mext32    uint8 = 0xc9
	mfloat32  uint8 = 0xca
	mfloat64  uint8 = 0xcb
	muint8    uint8 = 0xcc
	muint16   uint8 = 0xcd
	muint32   uint8 = 0xce
	muint64   uint8 = 0xcf
	mint8     uint8 = 0xd0
	mint16    uint8 = 0xd1
	mint32    uint8 = 0xd2
	mint64    uint8 = 0xd3
	mfixext1  uint8 = 0xd4
	mfixext2  uint8 = 0xd5
	mfixext4  uint8 = 0xd6
	mfixext8  uint8 = 0xd7
	mfixext16 uint8 = 0xd8
	mstr8     uint8 = 0xd9
	mstr16    uint8 = 0xda
	mstr32    uint8 = 0xdb
	marray16  uint8 = 0xdc
	marray32  uint8 = 0xdd
	mmap16    uint8 = 0xde
	mmap32    uint8 = 0xdf
)
