package msgp

import (
	"math"
	"strconv"
)

// The portable parts of the Number implementation

// Number can be
// an int64, uint64, float32,
// or float64 internally.
// It can decode itself
// from any of the native
// messagepack number types.
// The zero-value of Number
// is Int(0). Using the equality
// operator with Number compares
// both the type and the value
// of the number.
type Number struct {
	// internally, this
	// is just a tagged union.
	// the raw bits of the number
	// are stored the same way regardless.
	bits uint64
	typ  Type
}

// AsInt sets the number to an int64.
func (n *Number) AsInt(i int64) {

	// we always store int(0)
	// as {0, InvalidType} in
	// order to preserve
	// the behavior of the == operator
	if i == 0 {
		n.typ = InvalidType
		n.bits = 0
		return
	}

	n.typ = IntType
	n.bits = uint64(i)
}

// AsUint sets the number to a uint64.
func (n *Number) AsUint(u uint64) {
	n.typ = UintType
	n.bits = u
}

// AsFloat32 sets the value of the number
// to a float32.
func (n *Number) AsFloat32(f float32) {
	n.typ = Float32Type
	n.bits = uint64(math.Float32bits(f))
}

// AsFloat64 sets the value of the
// number to a float64.
func (n *Number) AsFloat64(f float64) {
	n.typ = Float64Type
	n.bits = math.Float64bits(f)
}

// Int casts the number as an int64, and
// returns whether or not that was the
// underlying type.
func (n *Number) Int() (int64, bool) {
	return int64(n.bits), n.typ == IntType || n.typ == InvalidType
}

// Uint casts the number as a uint64, and returns
// whether or not that was the underlying type.
func (n *Number) Uint() (uint64, bool) {
	return n.bits, n.typ == UintType
}

// Float casts the number to a float64, and
// returns whether or not that was the underlying
// type (either a float64 or a float32).
func (n *Number) Float() (float64, bool) {
	switch n.typ {
	case Float32Type:
		return float64(math.Float32frombits(uint32(n.bits))), true
	case Float64Type:
		return math.Float64frombits(n.bits), true
	default:
		return 0.0, false
	}
}

// Type will return one of:
// Float64Type, Float32Type, UintType, or IntType.
func (n *Number) Type() Type {
	if n.typ == InvalidType {
		return IntType
	}
	return n.typ
}

// DecodeMsg implements msgp.Decodable
func (n *Number) DecodeMsg(r *Reader) error {
	typ, err := r.NextType()
	if err != nil {
		return err
	}
	switch typ {
	case Float32Type:
		f, err := r.ReadFloat32()
		if err != nil {
			return err
		}
		n.AsFloat32(f)
		return nil
	case Float64Type:
		f, err := r.ReadFloat64()
		if err != nil {
			return err
		}
		n.AsFloat64(f)
		return nil
	case IntType:
		i, err := r.ReadInt64()
		if err != nil {
			return err
		}
		n.AsInt(i)
		return nil
	case UintType:
		u, err := r.ReadUint64()
		if err != nil {
			return err
		}
		n.AsUint(u)
		return nil
	default:
		return TypeError{Encoded: typ, Method: IntType}
	}
}

// UnmarshalMsg implements msgp.Unmarshaler
func (n *Number) UnmarshalMsg(b []byte) ([]byte, error) {
	typ := NextType(b)
	switch typ {
	case IntType:
		i, o, err := ReadInt64Bytes(b)
		if err != nil {
			return b, err
		}
		n.AsInt(i)
		return o, nil
	case UintType:
		u, o, err := ReadUint64Bytes(b)
		if err != nil {
			return b, err
		}
		n.AsUint(u)
		return o, nil
	case Float64Type:
		f, o, err := ReadFloat64Bytes(b)
		if err != nil {
			return b, err
		}
		n.AsFloat64(f)
		return o, nil
	case Float32Type:
		f, o, err := ReadFloat32Bytes(b)
		if err != nil {
			return b, err
		}
		n.AsFloat32(f)
		return o, nil
	default:
		return b, TypeError{Method: IntType, Encoded: typ}
	}
}

// MarshalMsg implements msgp.Marshaler
func (n *Number) MarshalMsg(b []byte) ([]byte, error) {
	switch n.typ {
	case IntType:
		return AppendInt64(b, int64(n.bits)), nil
	case UintType:
		return AppendUint64(b, uint64(n.bits)), nil
	case Float64Type:
		return AppendFloat64(b, math.Float64frombits(n.bits)), nil
	case Float32Type:
		return AppendFloat32(b, math.Float32frombits(uint32(n.bits))), nil
	default:
		return AppendInt64(b, 0), nil
	}
}

// EncodeMsg implements msgp.Encodable
func (n *Number) EncodeMsg(w *Writer) error {
	switch n.typ {
	case IntType:
		return w.WriteInt64(int64(n.bits))
	case UintType:
		return w.WriteUint64(n.bits)
	case Float64Type:
		return w.WriteFloat64(math.Float64frombits(n.bits))
	case Float32Type:
		return w.WriteFloat32(math.Float32frombits(uint32(n.bits)))
	default:
		return w.WriteInt64(0)
	}
}

// Msgsize implements msgp.Sizer
func (n *Number) Msgsize() int {
	switch n.typ {
	case Float32Type:
		return Float32Size
	case Float64Type:
		return Float64Size
	case IntType:
		return Int64Size
	case UintType:
		return Uint64Size
	default:
		return 1 // fixint(0)
	}
}

// MarshalJSON implements json.Marshaler
func (n *Number) MarshalJSON() ([]byte, error) {
	t := n.Type()
	if t == InvalidType {
		return []byte{'0'}, nil
	}
	out := make([]byte, 0, 32)
	switch t {
	case Float32Type, Float64Type:
		f, _ := n.Float()
		return strconv.AppendFloat(out, f, 'f', -1, 64), nil
	case IntType:
		i, _ := n.Int()
		return strconv.AppendInt(out, i, 10), nil
	case UintType:
		u, _ := n.Uint()
		return strconv.AppendUint(out, u, 10), nil
	default:
		panic("(*Number).typ is invalid")
	}
}

// String implements fmt.Stringer
func (n *Number) String() string {
	switch n.typ {
	case InvalidType:
		return "0"
	case Float32Type, Float64Type:
		f, _ := n.Float()
		return strconv.FormatFloat(f, 'f', -1, 64)
	case IntType:
		i, _ := n.Int()
		return strconv.FormatInt(i, 10)
	case UintType:
		u, _ := n.Uint()
		return strconv.FormatUint(u, 10)
	default:
		panic("(*Number).typ is invalid")
	}
}
