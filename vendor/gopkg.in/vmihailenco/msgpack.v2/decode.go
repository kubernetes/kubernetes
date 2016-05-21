package msgpack

import (
	"bufio"
	"bytes"
	"errors"
	"fmt"
	"io"
	"reflect"
	"time"

	"gopkg.in/vmihailenco/msgpack.v2/codes"
)

type bufReader interface {
	Read([]byte) (int, error)
	ReadByte() (byte, error)
	UnreadByte() error
}

func Unmarshal(b []byte, v ...interface{}) error {
	if len(v) == 1 && v[0] != nil {
		unmarshaler, ok := v[0].(Unmarshaler)
		if ok {
			return unmarshaler.UnmarshalMsgpack(b)
		}
	}
	return NewDecoder(bytes.NewReader(b)).Decode(v...)
}

type Decoder struct {
	// TODO: add map len arg
	DecodeMapFunc func(*Decoder) (interface{}, error)

	r   bufReader
	buf []byte
}

func NewDecoder(r io.Reader) *Decoder {
	br, ok := r.(bufReader)
	if !ok {
		br = bufio.NewReader(r)
	}
	return &Decoder{
		DecodeMapFunc: decodeMap,

		r:   br,
		buf: make([]byte, 64),
	}
}

func (d *Decoder) Decode(v ...interface{}) error {
	for _, vv := range v {
		if err := d.decode(vv); err != nil {
			return err
		}
	}
	return nil
}

func (d *Decoder) decode(dst interface{}) error {
	var err error
	switch v := dst.(type) {
	case *string:
		if v != nil {
			*v, err = d.DecodeString()
			return err
		}
	case *[]byte:
		if v != nil {
			*v, err = d.DecodeBytes()
			return err
		}
	case *int:
		if v != nil {
			*v, err = d.DecodeInt()
			return err
		}
	case *int8:
		if v != nil {
			*v, err = d.DecodeInt8()
			return err
		}
	case *int16:
		if v != nil {
			*v, err = d.DecodeInt16()
			return err
		}
	case *int32:
		if v != nil {
			*v, err = d.DecodeInt32()
			return err
		}
	case *int64:
		if v != nil {
			*v, err = d.DecodeInt64()
			return err
		}
	case *uint:
		if v != nil {
			*v, err = d.DecodeUint()
			return err
		}
	case *uint8:
		if v != nil {
			*v, err = d.DecodeUint8()
			return err
		}
	case *uint16:
		if v != nil {
			*v, err = d.DecodeUint16()
			return err
		}
	case *uint32:
		if v != nil {
			*v, err = d.DecodeUint32()
			return err
		}
	case *uint64:
		if v != nil {
			*v, err = d.DecodeUint64()
			return err
		}
	case *bool:
		if v != nil {
			*v, err = d.DecodeBool()
			return err
		}
	case *float32:
		if v != nil {
			*v, err = d.DecodeFloat32()
			return err
		}
	case *float64:
		if v != nil {
			*v, err = d.DecodeFloat64()
			return err
		}
	case *[]string:
		return d.decodeIntoStrings(v)
	case *map[string]string:
		return d.decodeIntoMapStringString(v)
	case *time.Duration:
		if v != nil {
			vv, err := d.DecodeInt64()
			*v = time.Duration(vv)
			return err
		}
	case *time.Time:
		if v != nil {
			*v, err = d.DecodeTime()
			return err
		}
	}

	v := reflect.ValueOf(dst)
	if !v.IsValid() {
		return errors.New("msgpack: Decode(nil)")
	}
	if v.Kind() != reflect.Ptr {
		return fmt.Errorf("msgpack: Decode(nonsettable %T)", dst)
	}
	v = v.Elem()
	if !v.IsValid() {
		return fmt.Errorf("msgpack: Decode(nonsettable %T)", dst)
	}
	return d.DecodeValue(v)
}

func (d *Decoder) DecodeValue(v reflect.Value) error {
	decode := getDecoder(v.Type())
	return decode(d, v)
}

func (d *Decoder) DecodeNil() error {
	c, err := d.r.ReadByte()
	if err != nil {
		return err
	}
	if c != codes.Nil {
		return fmt.Errorf("msgpack: invalid code %x decoding nil", c)
	}
	return nil
}

func (d *Decoder) DecodeBool() (bool, error) {
	c, err := d.r.ReadByte()
	if err != nil {
		return false, err
	}
	return d.bool(c)
}

func (d *Decoder) bool(c byte) (bool, error) {
	if c == codes.False {
		return false, nil
	}
	if c == codes.True {
		return true, nil
	}
	return false, fmt.Errorf("msgpack: invalid code %x decoding bool", c)
}

func (d *Decoder) boolValue(value reflect.Value) error {
	v, err := d.DecodeBool()
	if err != nil {
		return err
	}
	value.SetBool(v)
	return nil
}

func (d *Decoder) interfaceValue(v reflect.Value) error {
	iface, err := d.DecodeInterface()
	if err != nil {
		return err
	}
	if iface != nil {
		v.Set(reflect.ValueOf(iface))
	}
	return nil
}

// DecodeInterface decodes value into interface. Possible value types are:
//   - nil,
//   - int64 for negative numbers,
//   - uint64 for positive numbers,
//   - bool,
//   - float32 and float64,
//   - string,
//   - slices of any of the above,
//   - maps of any of the above.
func (d *Decoder) DecodeInterface() (interface{}, error) {
	c, err := d.r.ReadByte()
	if err != nil {
		return nil, err
	}

	if codes.IsFixedNum(c) {
		if int8(c) < 0 {
			return d.int(c)
		}
		return d.uint(c)
	}
	if codes.IsFixedMap(c) {
		d.r.UnreadByte()
		return d.DecodeMap()
	}
	if codes.IsFixedArray(c) {
		d.r.UnreadByte()
		return d.DecodeSlice()
	}
	if codes.IsFixedString(c) {
		return d.string(c)
	}

	switch c {
	case codes.Nil:
		return nil, nil
	case codes.False, codes.True:
		return d.bool(c)
	case codes.Float:
		return d.float32(c)
	case codes.Double:
		return d.float64(c)
	case codes.Uint8, codes.Uint16, codes.Uint32, codes.Uint64:
		return d.uint(c)
	case codes.Int8, codes.Int16, codes.Int32, codes.Int64:
		return d.int(c)
	case codes.Bin8, codes.Bin16, codes.Bin32:
		return d.bytes(c)
	case codes.Str8, codes.Str16, codes.Str32:
		return d.string(c)
	case codes.Array16, codes.Array32:
		d.r.UnreadByte()
		return d.DecodeSlice()
	case codes.Map16, codes.Map32:
		d.r.UnreadByte()
		return d.DecodeMap()
	case codes.FixExt1, codes.FixExt2, codes.FixExt4, codes.FixExt8, codes.FixExt16, codes.Ext8, codes.Ext16, codes.Ext32:
		return d.ext(c)
	}

	return 0, fmt.Errorf("msgpack: unknown code %x decoding interface{}", c)
}

// Skip skips next value.
func (d *Decoder) Skip() error {
	c, err := d.r.ReadByte()
	if err != nil {
		return err
	}

	if codes.IsFixedNum(c) {
		return nil
	} else if codes.IsFixedMap(c) {
		return d.skipMap(c)
	} else if codes.IsFixedArray(c) {
		return d.skipSlice(c)
	} else if codes.IsFixedString(c) {
		return d.skipBytes(c)
	}

	switch c {
	case codes.Nil, codes.False, codes.True:
		return nil
	case codes.Uint8, codes.Int8:
		return d.skipN(1)
	case codes.Uint16, codes.Int16:
		return d.skipN(2)
	case codes.Uint32, codes.Int32, codes.Float:
		return d.skipN(4)
	case codes.Uint64, codes.Int64, codes.Double:
		return d.skipN(8)
	case codes.Bin8, codes.Bin16, codes.Bin32:
		return d.skipBytes(c)
	case codes.Str8, codes.Str16, codes.Str32:
		return d.skipBytes(c)
	case codes.Array16, codes.Array32:
		return d.skipSlice(c)
	case codes.Map16, codes.Map32:
		return d.skipMap(c)
	case codes.FixExt1, codes.FixExt2, codes.FixExt4, codes.FixExt8, codes.FixExt16, codes.Ext8, codes.Ext16, codes.Ext32:
		return d.skipExt(c)
	}

	return fmt.Errorf("msgpack: unknown code %x", c)
}

// peekCode returns the next Msgpack code. See
// https://github.com/msgpack/msgpack/blob/master/spec.md#formats for details.
func (d *Decoder) PeekCode() (code byte, err error) {
	code, err = d.r.ReadByte()
	if err != nil {
		return 0, err
	}
	return code, d.r.UnreadByte()
}

func (d *Decoder) gotNilCode() bool {
	code, err := d.PeekCode()
	return err == nil && code == codes.Nil
}

func (d *Decoder) readN(n int) ([]byte, error) {
	var b []byte
	if n <= cap(d.buf) {
		b = d.buf[:n]
	} else {
		b = make([]byte, n)
	}
	_, err := io.ReadFull(d.r, b)
	return b, err
}
