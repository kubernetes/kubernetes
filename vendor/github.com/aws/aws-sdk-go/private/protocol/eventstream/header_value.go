package eventstream

import (
	"encoding/base64"
	"encoding/binary"
	"fmt"
	"io"
	"strconv"
	"time"
)

const maxHeaderValueLen = 1<<15 - 1 // 2^15-1 or 32KB - 1

// valueType is the EventStream header value type.
type valueType uint8

// Header value types
const (
	trueValueType valueType = iota
	falseValueType
	int8ValueType  // Byte
	int16ValueType // Short
	int32ValueType // Integer
	int64ValueType // Long
	bytesValueType
	stringValueType
	timestampValueType
	uuidValueType
)

func (t valueType) String() string {
	switch t {
	case trueValueType:
		return "bool"
	case falseValueType:
		return "bool"
	case int8ValueType:
		return "int8"
	case int16ValueType:
		return "int16"
	case int32ValueType:
		return "int32"
	case int64ValueType:
		return "int64"
	case bytesValueType:
		return "byte_array"
	case stringValueType:
		return "string"
	case timestampValueType:
		return "timestamp"
	case uuidValueType:
		return "uuid"
	default:
		return fmt.Sprintf("unknown value type %d", uint8(t))
	}
}

type rawValue struct {
	Type  valueType
	Len   uint16 // Only set for variable length slices
	Value []byte // byte representation of value, BigEndian encoding.
}

func (r rawValue) encodeScalar(w io.Writer, v interface{}) error {
	return binaryWriteFields(w, binary.BigEndian,
		r.Type,
		v,
	)
}

func (r rawValue) encodeFixedSlice(w io.Writer, v []byte) error {
	binary.Write(w, binary.BigEndian, r.Type)

	_, err := w.Write(v)
	return err
}

func (r rawValue) encodeBytes(w io.Writer, v []byte) error {
	if len(v) > maxHeaderValueLen {
		return LengthError{
			Part: "header value",
			Want: maxHeaderValueLen, Have: len(v),
			Value: v,
		}
	}
	r.Len = uint16(len(v))

	err := binaryWriteFields(w, binary.BigEndian,
		r.Type,
		r.Len,
	)
	if err != nil {
		return err
	}

	_, err = w.Write(v)
	return err
}

func (r rawValue) encodeString(w io.Writer, v string) error {
	if len(v) > maxHeaderValueLen {
		return LengthError{
			Part: "header value",
			Want: maxHeaderValueLen, Have: len(v),
			Value: v,
		}
	}
	r.Len = uint16(len(v))

	type stringWriter interface {
		WriteString(string) (int, error)
	}

	err := binaryWriteFields(w, binary.BigEndian,
		r.Type,
		r.Len,
	)
	if err != nil {
		return err
	}

	if sw, ok := w.(stringWriter); ok {
		_, err = sw.WriteString(v)
	} else {
		_, err = w.Write([]byte(v))
	}

	return err
}

func decodeFixedBytesValue(r io.Reader, buf []byte) error {
	_, err := io.ReadFull(r, buf)
	return err
}

func decodeBytesValue(r io.Reader) ([]byte, error) {
	var raw rawValue
	var err error
	raw.Len, err = decodeUint16(r)
	if err != nil {
		return nil, err
	}

	buf := make([]byte, raw.Len)
	_, err = io.ReadFull(r, buf)
	if err != nil {
		return nil, err
	}

	return buf, nil
}

func decodeStringValue(r io.Reader) (string, error) {
	v, err := decodeBytesValue(r)
	return string(v), err
}

// Value represents the abstract header value.
type Value interface {
	Get() interface{}
	String() string
	valueType() valueType
	encode(io.Writer) error
}

// An BoolValue provides eventstream encoding, and representation
// of a Go bool value.
type BoolValue bool

// Get returns the underlying type
func (v BoolValue) Get() interface{} {
	return bool(v)
}

// valueType returns the EventStream header value type value.
func (v BoolValue) valueType() valueType {
	if v {
		return trueValueType
	}
	return falseValueType
}

func (v BoolValue) String() string {
	return strconv.FormatBool(bool(v))
}

// encode encodes the BoolValue into an eventstream binary value
// representation.
func (v BoolValue) encode(w io.Writer) error {
	return binary.Write(w, binary.BigEndian, v.valueType())
}

// An Int8Value provides eventstream encoding, and representation of a Go
// int8 value.
type Int8Value int8

// Get returns the underlying value.
func (v Int8Value) Get() interface{} {
	return int8(v)
}

// valueType returns the EventStream header value type value.
func (Int8Value) valueType() valueType {
	return int8ValueType
}

func (v Int8Value) String() string {
	return fmt.Sprintf("0x%02x", int8(v))
}

// encode encodes the Int8Value into an eventstream binary value
// representation.
func (v Int8Value) encode(w io.Writer) error {
	raw := rawValue{
		Type: v.valueType(),
	}

	return raw.encodeScalar(w, v)
}

func (v *Int8Value) decode(r io.Reader) error {
	n, err := decodeUint8(r)
	if err != nil {
		return err
	}

	*v = Int8Value(n)
	return nil
}

// An Int16Value provides eventstream encoding, and representation of a Go
// int16 value.
type Int16Value int16

// Get returns the underlying value.
func (v Int16Value) Get() interface{} {
	return int16(v)
}

// valueType returns the EventStream header value type value.
func (Int16Value) valueType() valueType {
	return int16ValueType
}

func (v Int16Value) String() string {
	return fmt.Sprintf("0x%04x", int16(v))
}

// encode encodes the Int16Value into an eventstream binary value
// representation.
func (v Int16Value) encode(w io.Writer) error {
	raw := rawValue{
		Type: v.valueType(),
	}
	return raw.encodeScalar(w, v)
}

func (v *Int16Value) decode(r io.Reader) error {
	n, err := decodeUint16(r)
	if err != nil {
		return err
	}

	*v = Int16Value(n)
	return nil
}

// An Int32Value provides eventstream encoding, and representation of a Go
// int32 value.
type Int32Value int32

// Get returns the underlying value.
func (v Int32Value) Get() interface{} {
	return int32(v)
}

// valueType returns the EventStream header value type value.
func (Int32Value) valueType() valueType {
	return int32ValueType
}

func (v Int32Value) String() string {
	return fmt.Sprintf("0x%08x", int32(v))
}

// encode encodes the Int32Value into an eventstream binary value
// representation.
func (v Int32Value) encode(w io.Writer) error {
	raw := rawValue{
		Type: v.valueType(),
	}
	return raw.encodeScalar(w, v)
}

func (v *Int32Value) decode(r io.Reader) error {
	n, err := decodeUint32(r)
	if err != nil {
		return err
	}

	*v = Int32Value(n)
	return nil
}

// An Int64Value provides eventstream encoding, and representation of a Go
// int64 value.
type Int64Value int64

// Get returns the underlying value.
func (v Int64Value) Get() interface{} {
	return int64(v)
}

// valueType returns the EventStream header value type value.
func (Int64Value) valueType() valueType {
	return int64ValueType
}

func (v Int64Value) String() string {
	return fmt.Sprintf("0x%016x", int64(v))
}

// encode encodes the Int64Value into an eventstream binary value
// representation.
func (v Int64Value) encode(w io.Writer) error {
	raw := rawValue{
		Type: v.valueType(),
	}
	return raw.encodeScalar(w, v)
}

func (v *Int64Value) decode(r io.Reader) error {
	n, err := decodeUint64(r)
	if err != nil {
		return err
	}

	*v = Int64Value(n)
	return nil
}

// An BytesValue provides eventstream encoding, and representation of a Go
// byte slice.
type BytesValue []byte

// Get returns the underlying value.
func (v BytesValue) Get() interface{} {
	return []byte(v)
}

// valueType returns the EventStream header value type value.
func (BytesValue) valueType() valueType {
	return bytesValueType
}

func (v BytesValue) String() string {
	return base64.StdEncoding.EncodeToString([]byte(v))
}

// encode encodes the BytesValue into an eventstream binary value
// representation.
func (v BytesValue) encode(w io.Writer) error {
	raw := rawValue{
		Type: v.valueType(),
	}

	return raw.encodeBytes(w, []byte(v))
}

func (v *BytesValue) decode(r io.Reader) error {
	buf, err := decodeBytesValue(r)
	if err != nil {
		return err
	}

	*v = BytesValue(buf)
	return nil
}

// An StringValue provides eventstream encoding, and representation of a Go
// string.
type StringValue string

// Get returns the underlying value.
func (v StringValue) Get() interface{} {
	return string(v)
}

// valueType returns the EventStream header value type value.
func (StringValue) valueType() valueType {
	return stringValueType
}

func (v StringValue) String() string {
	return string(v)
}

// encode encodes the StringValue into an eventstream binary value
// representation.
func (v StringValue) encode(w io.Writer) error {
	raw := rawValue{
		Type: v.valueType(),
	}

	return raw.encodeString(w, string(v))
}

func (v *StringValue) decode(r io.Reader) error {
	s, err := decodeStringValue(r)
	if err != nil {
		return err
	}

	*v = StringValue(s)
	return nil
}

// An TimestampValue provides eventstream encoding, and representation of a Go
// timestamp.
type TimestampValue time.Time

// Get returns the underlying value.
func (v TimestampValue) Get() interface{} {
	return time.Time(v)
}

// valueType returns the EventStream header value type value.
func (TimestampValue) valueType() valueType {
	return timestampValueType
}

func (v TimestampValue) epochMilli() int64 {
	nano := time.Time(v).UnixNano()
	msec := nano / int64(time.Millisecond)
	return msec
}

func (v TimestampValue) String() string {
	msec := v.epochMilli()
	return strconv.FormatInt(msec, 10)
}

// encode encodes the TimestampValue into an eventstream binary value
// representation.
func (v TimestampValue) encode(w io.Writer) error {
	raw := rawValue{
		Type: v.valueType(),
	}

	msec := v.epochMilli()
	return raw.encodeScalar(w, msec)
}

func (v *TimestampValue) decode(r io.Reader) error {
	n, err := decodeUint64(r)
	if err != nil {
		return err
	}

	*v = TimestampValue(timeFromEpochMilli(int64(n)))
	return nil
}

func timeFromEpochMilli(t int64) time.Time {
	secs := t / 1e3
	msec := t % 1e3
	return time.Unix(secs, msec*int64(time.Millisecond))
}

// An UUIDValue provides eventstream encoding, and representation of a UUID
// value.
type UUIDValue [16]byte

// Get returns the underlying value.
func (v UUIDValue) Get() interface{} {
	return v[:]
}

// valueType returns the EventStream header value type value.
func (UUIDValue) valueType() valueType {
	return uuidValueType
}

func (v UUIDValue) String() string {
	return fmt.Sprintf(`%X-%X-%X-%X-%X`, v[0:4], v[4:6], v[6:8], v[8:10], v[10:])
}

// encode encodes the UUIDValue into an eventstream binary value
// representation.
func (v UUIDValue) encode(w io.Writer) error {
	raw := rawValue{
		Type: v.valueType(),
	}

	return raw.encodeFixedSlice(w, v[:])
}

func (v *UUIDValue) decode(r io.Reader) error {
	tv := (*v)[:]
	return decodeFixedBytesValue(r, tv)
}
