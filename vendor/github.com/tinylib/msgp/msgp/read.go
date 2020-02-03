package msgp

import (
	"io"
	"math"
	"sync"
	"time"

	"github.com/philhofer/fwd"
)

// where we keep old *Readers
var readerPool = sync.Pool{New: func() interface{} { return &Reader{} }}

// Type is a MessagePack wire type,
// including this package's built-in
// extension types.
type Type byte

// MessagePack Types
//
// The zero value of Type
// is InvalidType.
const (
	InvalidType Type = iota

	// MessagePack built-in types

	StrType
	BinType
	MapType
	ArrayType
	Float64Type
	Float32Type
	BoolType
	IntType
	UintType
	NilType
	ExtensionType

	// pseudo-types provided
	// by extensions

	Complex64Type
	Complex128Type
	TimeType

	_maxtype
)

// String implements fmt.Stringer
func (t Type) String() string {
	switch t {
	case StrType:
		return "str"
	case BinType:
		return "bin"
	case MapType:
		return "map"
	case ArrayType:
		return "array"
	case Float64Type:
		return "float64"
	case Float32Type:
		return "float32"
	case BoolType:
		return "bool"
	case UintType:
		return "uint"
	case IntType:
		return "int"
	case ExtensionType:
		return "ext"
	case NilType:
		return "nil"
	default:
		return "<invalid>"
	}
}

func freeR(m *Reader) {
	readerPool.Put(m)
}

// Unmarshaler is the interface fulfilled
// by objects that know how to unmarshal
// themselves from MessagePack.
// UnmarshalMsg unmarshals the object
// from binary, returing any leftover
// bytes and any errors encountered.
type Unmarshaler interface {
	UnmarshalMsg([]byte) ([]byte, error)
}

// Decodable is the interface fulfilled
// by objects that know how to read
// themselves from a *Reader.
type Decodable interface {
	DecodeMsg(*Reader) error
}

// Decode decodes 'd' from 'r'.
func Decode(r io.Reader, d Decodable) error {
	rd := NewReader(r)
	err := d.DecodeMsg(rd)
	freeR(rd)
	return err
}

// NewReader returns a *Reader that
// reads from the provided reader. The
// reader will be buffered.
func NewReader(r io.Reader) *Reader {
	p := readerPool.Get().(*Reader)
	if p.R == nil {
		p.R = fwd.NewReader(r)
	} else {
		p.R.Reset(r)
	}
	return p
}

// NewReaderSize returns a *Reader with a buffer of the given size.
// (This is vastly preferable to passing the decoder a reader that is already buffered.)
func NewReaderSize(r io.Reader, sz int) *Reader {
	return &Reader{R: fwd.NewReaderSize(r, sz)}
}

// Reader wraps an io.Reader and provides
// methods to read MessagePack-encoded values
// from it. Readers are buffered.
type Reader struct {
	// R is the buffered reader
	// that the Reader uses
	// to decode MessagePack.
	// The Reader itself
	// is stateless; all the
	// buffering is done
	// within R.
	R       *fwd.Reader
	scratch []byte
}

// Read implements `io.Reader`
func (m *Reader) Read(p []byte) (int, error) {
	return m.R.Read(p)
}

// CopyNext reads the next object from m without decoding it and writes it to w.
// It avoids unnecessary copies internally.
func (m *Reader) CopyNext(w io.Writer) (int64, error) {
	sz, o, err := getNextSize(m.R)
	if err != nil {
		return 0, err
	}

	var n int64
	// Opportunistic optimization: if we can fit the whole thing in the m.R
	// buffer, then just get a pointer to that, and pass it to w.Write,
	// avoiding an allocation.
	if int(sz) <= m.R.BufferSize() {
		var nn int
		var buf []byte
		buf, err = m.R.Next(int(sz))
		if err != nil {
			if err == io.ErrUnexpectedEOF {
				err = ErrShortBytes
			}
			return 0, err
		}
		nn, err = w.Write(buf)
		n += int64(nn)
	} else {
		// Fall back to io.CopyN.
		// May avoid allocating if w is a ReaderFrom (e.g. bytes.Buffer)
		n, err = io.CopyN(w, m.R, int64(sz))
		if err == io.ErrUnexpectedEOF {
			err = ErrShortBytes
		}
	}
	if err != nil {
		return n, err
	} else if n < int64(sz) {
		return n, io.ErrShortWrite
	}

	// for maps and slices, read elements
	for x := uintptr(0); x < o; x++ {
		var n2 int64
		n2, err = m.CopyNext(w)
		if err != nil {
			return n, err
		}
		n += n2
	}
	return n, nil
}

// ReadFull implements `io.ReadFull`
func (m *Reader) ReadFull(p []byte) (int, error) {
	return m.R.ReadFull(p)
}

// Reset resets the underlying reader.
func (m *Reader) Reset(r io.Reader) { m.R.Reset(r) }

// Buffered returns the number of bytes currently in the read buffer.
func (m *Reader) Buffered() int { return m.R.Buffered() }

// BufferSize returns the capacity of the read buffer.
func (m *Reader) BufferSize() int { return m.R.BufferSize() }

// NextType returns the next object type to be decoded.
func (m *Reader) NextType() (Type, error) {
	p, err := m.R.Peek(1)
	if err != nil {
		return InvalidType, err
	}
	t := getType(p[0])
	if t == InvalidType {
		return t, InvalidPrefixError(p[0])
	}
	if t == ExtensionType {
		v, err := m.peekExtensionType()
		if err != nil {
			return InvalidType, err
		}
		switch v {
		case Complex64Extension:
			return Complex64Type, nil
		case Complex128Extension:
			return Complex128Type, nil
		case TimeExtension:
			return TimeType, nil
		}
	}
	return t, nil
}

// IsNil returns whether or not
// the next byte is a null messagepack byte
func (m *Reader) IsNil() bool {
	p, err := m.R.Peek(1)
	return err == nil && p[0] == mnil
}

// getNextSize returns the size of the next object on the wire.
// returns (obj size, obj elements, error)
// only maps and arrays have non-zero obj elements
// for maps and arrays, obj size does not include elements
//
// use uintptr b/c it's guaranteed to be large enough
// to hold whatever we can fit in memory.
func getNextSize(r *fwd.Reader) (uintptr, uintptr, error) {
	b, err := r.Peek(1)
	if err != nil {
		return 0, 0, err
	}
	lead := b[0]
	spec := &sizes[lead]
	size, mode := spec.size, spec.extra
	if size == 0 {
		return 0, 0, InvalidPrefixError(lead)
	}
	if mode >= 0 {
		return uintptr(size), uintptr(mode), nil
	}
	b, err = r.Peek(int(size))
	if err != nil {
		return 0, 0, err
	}
	switch mode {
	case extra8:
		return uintptr(size) + uintptr(b[1]), 0, nil
	case extra16:
		return uintptr(size) + uintptr(big.Uint16(b[1:])), 0, nil
	case extra32:
		return uintptr(size) + uintptr(big.Uint32(b[1:])), 0, nil
	case map16v:
		return uintptr(size), 2 * uintptr(big.Uint16(b[1:])), nil
	case map32v:
		return uintptr(size), 2 * uintptr(big.Uint32(b[1:])), nil
	case array16v:
		return uintptr(size), uintptr(big.Uint16(b[1:])), nil
	case array32v:
		return uintptr(size), uintptr(big.Uint32(b[1:])), nil
	default:
		return 0, 0, fatal
	}
}

// Skip skips over the next object, regardless of
// its type. If it is an array or map, the whole array
// or map will be skipped.
func (m *Reader) Skip() error {
	var (
		v   uintptr // bytes
		o   uintptr // objects
		err error
		p   []byte
	)

	// we can use the faster
	// method if we have enough
	// buffered data
	if m.R.Buffered() >= 5 {
		p, err = m.R.Peek(5)
		if err != nil {
			return err
		}
		v, o, err = getSize(p)
		if err != nil {
			return err
		}
	} else {
		v, o, err = getNextSize(m.R)
		if err != nil {
			return err
		}
	}

	// 'v' is always non-zero
	// if err == nil
	_, err = m.R.Skip(int(v))
	if err != nil {
		return err
	}

	// for maps and slices, skip elements
	for x := uintptr(0); x < o; x++ {
		err = m.Skip()
		if err != nil {
			return err
		}
	}
	return nil
}

// ReadMapHeader reads the next object
// as a map header and returns the size
// of the map and the number of bytes written.
// It will return a TypeError{} if the next
// object is not a map.
func (m *Reader) ReadMapHeader() (sz uint32, err error) {
	var p []byte
	var lead byte
	p, err = m.R.Peek(1)
	if err != nil {
		return
	}
	lead = p[0]
	if isfixmap(lead) {
		sz = uint32(rfixmap(lead))
		_, err = m.R.Skip(1)
		return
	}
	switch lead {
	case mmap16:
		p, err = m.R.Next(3)
		if err != nil {
			return
		}
		sz = uint32(big.Uint16(p[1:]))
		return
	case mmap32:
		p, err = m.R.Next(5)
		if err != nil {
			return
		}
		sz = big.Uint32(p[1:])
		return
	default:
		err = badPrefix(MapType, lead)
		return
	}
}

// ReadMapKey reads either a 'str' or 'bin' field from
// the reader and returns the value as a []byte. It uses
// scratch for storage if it is large enough.
func (m *Reader) ReadMapKey(scratch []byte) ([]byte, error) {
	out, err := m.ReadStringAsBytes(scratch)
	if err != nil {
		if tperr, ok := err.(TypeError); ok && tperr.Encoded == BinType {
			return m.ReadBytes(scratch)
		}
		return nil, err
	}
	return out, nil
}

// MapKeyPtr returns a []byte pointing to the contents
// of a valid map key. The key cannot be empty, and it
// must be shorter than the total buffer size of the
// *Reader. Additionally, the returned slice is only
// valid until the next *Reader method call. Users
// should exercise extreme care when using this
// method; writing into the returned slice may
// corrupt future reads.
func (m *Reader) ReadMapKeyPtr() ([]byte, error) {
	p, err := m.R.Peek(1)
	if err != nil {
		return nil, err
	}
	lead := p[0]
	var read int
	if isfixstr(lead) {
		read = int(rfixstr(lead))
		m.R.Skip(1)
		goto fill
	}
	switch lead {
	case mstr8, mbin8:
		p, err = m.R.Next(2)
		if err != nil {
			return nil, err
		}
		read = int(p[1])
	case mstr16, mbin16:
		p, err = m.R.Next(3)
		if err != nil {
			return nil, err
		}
		read = int(big.Uint16(p[1:]))
	case mstr32, mbin32:
		p, err = m.R.Next(5)
		if err != nil {
			return nil, err
		}
		read = int(big.Uint32(p[1:]))
	default:
		return nil, badPrefix(StrType, lead)
	}
fill:
	if read == 0 {
		return nil, ErrShortBytes
	}
	return m.R.Next(read)
}

// ReadArrayHeader reads the next object as an
// array header and returns the size of the array
// and the number of bytes read.
func (m *Reader) ReadArrayHeader() (sz uint32, err error) {
	var lead byte
	var p []byte
	p, err = m.R.Peek(1)
	if err != nil {
		return
	}
	lead = p[0]
	if isfixarray(lead) {
		sz = uint32(rfixarray(lead))
		_, err = m.R.Skip(1)
		return
	}
	switch lead {
	case marray16:
		p, err = m.R.Next(3)
		if err != nil {
			return
		}
		sz = uint32(big.Uint16(p[1:]))
		return

	case marray32:
		p, err = m.R.Next(5)
		if err != nil {
			return
		}
		sz = big.Uint32(p[1:])
		return

	default:
		err = badPrefix(ArrayType, lead)
		return
	}
}

// ReadNil reads a 'nil' MessagePack byte from the reader
func (m *Reader) ReadNil() error {
	p, err := m.R.Peek(1)
	if err != nil {
		return err
	}
	if p[0] != mnil {
		return badPrefix(NilType, p[0])
	}
	_, err = m.R.Skip(1)
	return err
}

// ReadFloat64 reads a float64 from the reader.
// (If the value on the wire is encoded as a float32,
// it will be up-cast to a float64.)
func (m *Reader) ReadFloat64() (f float64, err error) {
	var p []byte
	p, err = m.R.Peek(9)
	if err != nil {
		// we'll allow a coversion from float32 to float64,
		// since we don't lose any precision
		if err == io.EOF && len(p) > 0 && p[0] == mfloat32 {
			ef, err := m.ReadFloat32()
			return float64(ef), err
		}
		return
	}
	if p[0] != mfloat64 {
		// see above
		if p[0] == mfloat32 {
			ef, err := m.ReadFloat32()
			return float64(ef), err
		}
		err = badPrefix(Float64Type, p[0])
		return
	}
	f = math.Float64frombits(getMuint64(p))
	_, err = m.R.Skip(9)
	return
}

// ReadFloat32 reads a float32 from the reader
func (m *Reader) ReadFloat32() (f float32, err error) {
	var p []byte
	p, err = m.R.Peek(5)
	if err != nil {
		return
	}
	if p[0] != mfloat32 {
		err = badPrefix(Float32Type, p[0])
		return
	}
	f = math.Float32frombits(getMuint32(p))
	_, err = m.R.Skip(5)
	return
}

// ReadBool reads a bool from the reader
func (m *Reader) ReadBool() (b bool, err error) {
	var p []byte
	p, err = m.R.Peek(1)
	if err != nil {
		return
	}
	switch p[0] {
	case mtrue:
		b = true
	case mfalse:
	default:
		err = badPrefix(BoolType, p[0])
		return
	}
	_, err = m.R.Skip(1)
	return
}

// ReadInt64 reads an int64 from the reader
func (m *Reader) ReadInt64() (i int64, err error) {
	var p []byte
	var lead byte
	p, err = m.R.Peek(1)
	if err != nil {
		return
	}
	lead = p[0]

	if isfixint(lead) {
		i = int64(rfixint(lead))
		_, err = m.R.Skip(1)
		return
	} else if isnfixint(lead) {
		i = int64(rnfixint(lead))
		_, err = m.R.Skip(1)
		return
	}

	switch lead {
	case mint8:
		p, err = m.R.Next(2)
		if err != nil {
			return
		}
		i = int64(getMint8(p))
		return

	case muint8:
		p, err = m.R.Next(2)
		if err != nil {
			return
		}
		i = int64(getMuint8(p))
		return

	case mint16:
		p, err = m.R.Next(3)
		if err != nil {
			return
		}
		i = int64(getMint16(p))
		return

	case muint16:
		p, err = m.R.Next(3)
		if err != nil {
			return
		}
		i = int64(getMuint16(p))
		return

	case mint32:
		p, err = m.R.Next(5)
		if err != nil {
			return
		}
		i = int64(getMint32(p))
		return

	case muint32:
		p, err = m.R.Next(5)
		if err != nil {
			return
		}
		i = int64(getMuint32(p))
		return

	case mint64:
		p, err = m.R.Next(9)
		if err != nil {
			return
		}
		i = getMint64(p)
		return

	case muint64:
		p, err = m.R.Next(9)
		if err != nil {
			return
		}
		u := getMuint64(p)
		if u > math.MaxInt64 {
			err = UintOverflow{Value: u, FailedBitsize: 64}
			return
		}
		i = int64(u)
		return

	default:
		err = badPrefix(IntType, lead)
		return
	}
}

// ReadInt32 reads an int32 from the reader
func (m *Reader) ReadInt32() (i int32, err error) {
	var in int64
	in, err = m.ReadInt64()
	if in > math.MaxInt32 || in < math.MinInt32 {
		err = IntOverflow{Value: in, FailedBitsize: 32}
		return
	}
	i = int32(in)
	return
}

// ReadInt16 reads an int16 from the reader
func (m *Reader) ReadInt16() (i int16, err error) {
	var in int64
	in, err = m.ReadInt64()
	if in > math.MaxInt16 || in < math.MinInt16 {
		err = IntOverflow{Value: in, FailedBitsize: 16}
		return
	}
	i = int16(in)
	return
}

// ReadInt8 reads an int8 from the reader
func (m *Reader) ReadInt8() (i int8, err error) {
	var in int64
	in, err = m.ReadInt64()
	if in > math.MaxInt8 || in < math.MinInt8 {
		err = IntOverflow{Value: in, FailedBitsize: 8}
		return
	}
	i = int8(in)
	return
}

// ReadInt reads an int from the reader
func (m *Reader) ReadInt() (i int, err error) {
	if smallint {
		var in int32
		in, err = m.ReadInt32()
		i = int(in)
		return
	}
	var in int64
	in, err = m.ReadInt64()
	i = int(in)
	return
}

// ReadUint64 reads a uint64 from the reader
func (m *Reader) ReadUint64() (u uint64, err error) {
	var p []byte
	var lead byte
	p, err = m.R.Peek(1)
	if err != nil {
		return
	}
	lead = p[0]
	if isfixint(lead) {
		u = uint64(rfixint(lead))
		_, err = m.R.Skip(1)
		return
	}
	switch lead {
	case mint8:
		p, err = m.R.Next(2)
		if err != nil {
			return
		}
		v := int64(getMint8(p))
		if v < 0 {
			err = UintBelowZero{Value: v}
			return
		}
		u = uint64(v)
		return

	case muint8:
		p, err = m.R.Next(2)
		if err != nil {
			return
		}
		u = uint64(getMuint8(p))
		return

	case mint16:
		p, err = m.R.Next(3)
		if err != nil {
			return
		}
		v := int64(getMint16(p))
		if v < 0 {
			err = UintBelowZero{Value: v}
			return
		}
		u = uint64(v)
		return

	case muint16:
		p, err = m.R.Next(3)
		if err != nil {
			return
		}
		u = uint64(getMuint16(p))
		return

	case mint32:
		p, err = m.R.Next(5)
		if err != nil {
			return
		}
		v := int64(getMint32(p))
		if v < 0 {
			err = UintBelowZero{Value: v}
			return
		}
		u = uint64(v)
		return

	case muint32:
		p, err = m.R.Next(5)
		if err != nil {
			return
		}
		u = uint64(getMuint32(p))
		return

	case mint64:
		p, err = m.R.Next(9)
		if err != nil {
			return
		}
		v := int64(getMint64(p))
		if v < 0 {
			err = UintBelowZero{Value: v}
			return
		}
		u = uint64(v)
		return

	case muint64:
		p, err = m.R.Next(9)
		if err != nil {
			return
		}
		u = getMuint64(p)
		return

	default:
		if isnfixint(lead) {
			err = UintBelowZero{Value: int64(rnfixint(lead))}
		} else {
			err = badPrefix(UintType, lead)
		}
		return

	}
}

// ReadUint32 reads a uint32 from the reader
func (m *Reader) ReadUint32() (u uint32, err error) {
	var in uint64
	in, err = m.ReadUint64()
	if in > math.MaxUint32 {
		err = UintOverflow{Value: in, FailedBitsize: 32}
		return
	}
	u = uint32(in)
	return
}

// ReadUint16 reads a uint16 from the reader
func (m *Reader) ReadUint16() (u uint16, err error) {
	var in uint64
	in, err = m.ReadUint64()
	if in > math.MaxUint16 {
		err = UintOverflow{Value: in, FailedBitsize: 16}
		return
	}
	u = uint16(in)
	return
}

// ReadUint8 reads a uint8 from the reader
func (m *Reader) ReadUint8() (u uint8, err error) {
	var in uint64
	in, err = m.ReadUint64()
	if in > math.MaxUint8 {
		err = UintOverflow{Value: in, FailedBitsize: 8}
		return
	}
	u = uint8(in)
	return
}

// ReadUint reads a uint from the reader
func (m *Reader) ReadUint() (u uint, err error) {
	if smallint {
		var un uint32
		un, err = m.ReadUint32()
		u = uint(un)
		return
	}
	var un uint64
	un, err = m.ReadUint64()
	u = uint(un)
	return
}

// ReadByte is analogous to ReadUint8.
//
// NOTE: this is *not* an implementation
// of io.ByteReader.
func (m *Reader) ReadByte() (b byte, err error) {
	var in uint64
	in, err = m.ReadUint64()
	if in > math.MaxUint8 {
		err = UintOverflow{Value: in, FailedBitsize: 8}
		return
	}
	b = byte(in)
	return
}

// ReadBytes reads a MessagePack 'bin' object
// from the reader and returns its value. It may
// use 'scratch' for storage if it is non-nil.
func (m *Reader) ReadBytes(scratch []byte) (b []byte, err error) {
	var p []byte
	var lead byte
	p, err = m.R.Peek(2)
	if err != nil {
		return
	}
	lead = p[0]
	var read int64
	switch lead {
	case mbin8:
		read = int64(p[1])
		m.R.Skip(2)
	case mbin16:
		p, err = m.R.Next(3)
		if err != nil {
			return
		}
		read = int64(big.Uint16(p[1:]))
	case mbin32:
		p, err = m.R.Next(5)
		if err != nil {
			return
		}
		read = int64(big.Uint32(p[1:]))
	default:
		err = badPrefix(BinType, lead)
		return
	}
	if int64(cap(scratch)) < read {
		b = make([]byte, read)
	} else {
		b = scratch[0:read]
	}
	_, err = m.R.ReadFull(b)
	return
}

// ReadBytesHeader reads the size header
// of a MessagePack 'bin' object. The user
// is responsible for dealing with the next
// 'sz' bytes from the reader in an application-specific
// way.
func (m *Reader) ReadBytesHeader() (sz uint32, err error) {
	var p []byte
	p, err = m.R.Peek(1)
	if err != nil {
		return
	}
	switch p[0] {
	case mbin8:
		p, err = m.R.Next(2)
		if err != nil {
			return
		}
		sz = uint32(p[1])
		return
	case mbin16:
		p, err = m.R.Next(3)
		if err != nil {
			return
		}
		sz = uint32(big.Uint16(p[1:]))
		return
	case mbin32:
		p, err = m.R.Next(5)
		if err != nil {
			return
		}
		sz = uint32(big.Uint32(p[1:]))
		return
	default:
		err = badPrefix(BinType, p[0])
		return
	}
}

// ReadExactBytes reads a MessagePack 'bin'-encoded
// object off of the wire into the provided slice. An
// ArrayError will be returned if the object is not
// exactly the length of the input slice.
func (m *Reader) ReadExactBytes(into []byte) error {
	p, err := m.R.Peek(2)
	if err != nil {
		return err
	}
	lead := p[0]
	var read int64 // bytes to read
	var skip int   // prefix size to skip
	switch lead {
	case mbin8:
		read = int64(p[1])
		skip = 2
	case mbin16:
		p, err = m.R.Peek(3)
		if err != nil {
			return err
		}
		read = int64(big.Uint16(p[1:]))
		skip = 3
	case mbin32:
		p, err = m.R.Peek(5)
		if err != nil {
			return err
		}
		read = int64(big.Uint32(p[1:]))
		skip = 5
	default:
		return badPrefix(BinType, lead)
	}
	if read != int64(len(into)) {
		return ArrayError{Wanted: uint32(len(into)), Got: uint32(read)}
	}
	m.R.Skip(skip)
	_, err = m.R.ReadFull(into)
	return err
}

// ReadStringAsBytes reads a MessagePack 'str' (utf-8) string
// and returns its value as bytes. It may use 'scratch' for storage
// if it is non-nil.
func (m *Reader) ReadStringAsBytes(scratch []byte) (b []byte, err error) {
	var p []byte
	var lead byte
	p, err = m.R.Peek(1)
	if err != nil {
		return
	}
	lead = p[0]
	var read int64

	if isfixstr(lead) {
		read = int64(rfixstr(lead))
		m.R.Skip(1)
		goto fill
	}

	switch lead {
	case mstr8:
		p, err = m.R.Next(2)
		if err != nil {
			return
		}
		read = int64(uint8(p[1]))
	case mstr16:
		p, err = m.R.Next(3)
		if err != nil {
			return
		}
		read = int64(big.Uint16(p[1:]))
	case mstr32:
		p, err = m.R.Next(5)
		if err != nil {
			return
		}
		read = int64(big.Uint32(p[1:]))
	default:
		err = badPrefix(StrType, lead)
		return
	}
fill:
	if int64(cap(scratch)) < read {
		b = make([]byte, read)
	} else {
		b = scratch[0:read]
	}
	_, err = m.R.ReadFull(b)
	return
}

// ReadStringHeader reads a string header
// off of the wire. The user is then responsible
// for dealing with the next 'sz' bytes from
// the reader in an application-specific manner.
func (m *Reader) ReadStringHeader() (sz uint32, err error) {
	var p []byte
	p, err = m.R.Peek(1)
	if err != nil {
		return
	}
	lead := p[0]
	if isfixstr(lead) {
		sz = uint32(rfixstr(lead))
		m.R.Skip(1)
		return
	}
	switch lead {
	case mstr8:
		p, err = m.R.Next(2)
		if err != nil {
			return
		}
		sz = uint32(p[1])
		return
	case mstr16:
		p, err = m.R.Next(3)
		if err != nil {
			return
		}
		sz = uint32(big.Uint16(p[1:]))
		return
	case mstr32:
		p, err = m.R.Next(5)
		if err != nil {
			return
		}
		sz = big.Uint32(p[1:])
		return
	default:
		err = badPrefix(StrType, lead)
		return
	}
}

// ReadString reads a utf-8 string from the reader
func (m *Reader) ReadString() (s string, err error) {
	var p []byte
	var lead byte
	var read int64
	p, err = m.R.Peek(1)
	if err != nil {
		return
	}
	lead = p[0]

	if isfixstr(lead) {
		read = int64(rfixstr(lead))
		m.R.Skip(1)
		goto fill
	}

	switch lead {
	case mstr8:
		p, err = m.R.Next(2)
		if err != nil {
			return
		}
		read = int64(uint8(p[1]))
	case mstr16:
		p, err = m.R.Next(3)
		if err != nil {
			return
		}
		read = int64(big.Uint16(p[1:]))
	case mstr32:
		p, err = m.R.Next(5)
		if err != nil {
			return
		}
		read = int64(big.Uint32(p[1:]))
	default:
		err = badPrefix(StrType, lead)
		return
	}
fill:
	if read == 0 {
		s, err = "", nil
		return
	}
	// reading into the memory
	// that will become the string
	// itself has vastly superior
	// worst-case performance, because
	// the reader buffer doesn't have
	// to be large enough to hold the string.
	// the idea here is to make it more
	// difficult for someone malicious
	// to cause the system to run out of
	// memory by sending very large strings.
	//
	// NOTE: this works because the argument
	// passed to (*fwd.Reader).ReadFull escapes
	// to the heap; its argument may, in turn,
	// be passed to the underlying reader, and
	// thus escape analysis *must* conclude that
	// 'out' escapes.
	out := make([]byte, read)
	_, err = m.R.ReadFull(out)
	if err != nil {
		return
	}
	s = UnsafeString(out)
	return
}

// ReadComplex64 reads a complex64 from the reader
func (m *Reader) ReadComplex64() (f complex64, err error) {
	var p []byte
	p, err = m.R.Peek(10)
	if err != nil {
		return
	}
	if p[0] != mfixext8 {
		err = badPrefix(Complex64Type, p[0])
		return
	}
	if int8(p[1]) != Complex64Extension {
		err = errExt(int8(p[1]), Complex64Extension)
		return
	}
	f = complex(math.Float32frombits(big.Uint32(p[2:])),
		math.Float32frombits(big.Uint32(p[6:])))
	_, err = m.R.Skip(10)
	return
}

// ReadComplex128 reads a complex128 from the reader
func (m *Reader) ReadComplex128() (f complex128, err error) {
	var p []byte
	p, err = m.R.Peek(18)
	if err != nil {
		return
	}
	if p[0] != mfixext16 {
		err = badPrefix(Complex128Type, p[0])
		return
	}
	if int8(p[1]) != Complex128Extension {
		err = errExt(int8(p[1]), Complex128Extension)
		return
	}
	f = complex(math.Float64frombits(big.Uint64(p[2:])),
		math.Float64frombits(big.Uint64(p[10:])))
	_, err = m.R.Skip(18)
	return
}

// ReadMapStrIntf reads a MessagePack map into a map[string]interface{}.
// (You must pass a non-nil map into the function.)
func (m *Reader) ReadMapStrIntf(mp map[string]interface{}) (err error) {
	var sz uint32
	sz, err = m.ReadMapHeader()
	if err != nil {
		return
	}
	for key := range mp {
		delete(mp, key)
	}
	for i := uint32(0); i < sz; i++ {
		var key string
		var val interface{}
		key, err = m.ReadString()
		if err != nil {
			return
		}
		val, err = m.ReadIntf()
		if err != nil {
			return
		}
		mp[key] = val
	}
	return
}

// ReadTime reads a time.Time object from the reader.
// The returned time's location will be set to time.Local.
func (m *Reader) ReadTime() (t time.Time, err error) {
	var p []byte
	p, err = m.R.Peek(15)
	if err != nil {
		return
	}
	if p[0] != mext8 || p[1] != 12 {
		err = badPrefix(TimeType, p[0])
		return
	}
	if int8(p[2]) != TimeExtension {
		err = errExt(int8(p[2]), TimeExtension)
		return
	}
	sec, nsec := getUnix(p[3:])
	t = time.Unix(sec, int64(nsec)).Local()
	_, err = m.R.Skip(15)
	return
}

// ReadIntf reads out the next object as a raw interface{}.
// Arrays are decoded as []interface{}, and maps are decoded
// as map[string]interface{}. Integers are decoded as int64
// and unsigned integers are decoded as uint64.
func (m *Reader) ReadIntf() (i interface{}, err error) {
	var t Type
	t, err = m.NextType()
	if err != nil {
		return
	}
	switch t {
	case BoolType:
		i, err = m.ReadBool()
		return

	case IntType:
		i, err = m.ReadInt64()
		return

	case UintType:
		i, err = m.ReadUint64()
		return

	case BinType:
		i, err = m.ReadBytes(nil)
		return

	case StrType:
		i, err = m.ReadString()
		return

	case Complex64Type:
		i, err = m.ReadComplex64()
		return

	case Complex128Type:
		i, err = m.ReadComplex128()
		return

	case TimeType:
		i, err = m.ReadTime()
		return

	case ExtensionType:
		var t int8
		t, err = m.peekExtensionType()
		if err != nil {
			return
		}
		f, ok := extensionReg[t]
		if ok {
			e := f()
			err = m.ReadExtension(e)
			i = e
			return
		}
		var e RawExtension
		e.Type = t
		err = m.ReadExtension(&e)
		i = &e
		return

	case MapType:
		mp := make(map[string]interface{})
		err = m.ReadMapStrIntf(mp)
		i = mp
		return

	case NilType:
		err = m.ReadNil()
		i = nil
		return

	case Float32Type:
		i, err = m.ReadFloat32()
		return

	case Float64Type:
		i, err = m.ReadFloat64()
		return

	case ArrayType:
		var sz uint32
		sz, err = m.ReadArrayHeader()

		if err != nil {
			return
		}
		out := make([]interface{}, int(sz))
		for j := range out {
			out[j], err = m.ReadIntf()
			if err != nil {
				return
			}
		}
		i = out
		return

	default:
		return nil, fatal // unreachable
	}
}
