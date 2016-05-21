package msgpack

import (
	"fmt"
	"math"
	"reflect"

	"gopkg.in/vmihailenco/msgpack.v2/codes"
)

func (d *Decoder) skipN(n int) error {
	_, err := d.readN(n)
	return err
}

func (d *Decoder) uint8() (uint8, error) {
	c, err := d.r.ReadByte()
	if err != nil {
		return 0, err
	}
	return uint8(c), nil
}

func (d *Decoder) uint16() (uint16, error) {
	b, err := d.readN(2)
	if err != nil {
		return 0, err
	}
	return (uint16(b[0]) << 8) | uint16(b[1]), nil
}

func (d *Decoder) uint32() (uint32, error) {
	b, err := d.readN(4)
	if err != nil {
		return 0, err
	}
	n := (uint32(b[0]) << 24) |
		(uint32(b[1]) << 16) |
		(uint32(b[2]) << 8) |
		uint32(b[3])
	return n, nil
}

func (d *Decoder) uint64() (uint64, error) {
	b, err := d.readN(8)
	if err != nil {
		return 0, err
	}
	n := (uint64(b[0]) << 56) |
		(uint64(b[1]) << 48) |
		(uint64(b[2]) << 40) |
		(uint64(b[3]) << 32) |
		(uint64(b[4]) << 24) |
		(uint64(b[5]) << 16) |
		(uint64(b[6]) << 8) |
		uint64(b[7])
	return n, nil
}

func (d *Decoder) DecodeUint64() (uint64, error) {
	c, err := d.r.ReadByte()
	if err != nil {
		return 0, err
	}
	return d.uint(c)
}

func (d *Decoder) uint(c byte) (uint64, error) {
	if c == codes.Nil {
		return 0, nil
	}
	if codes.IsFixedNum(c) {
		return uint64(int8(c)), nil
	}
	switch c {
	case codes.Uint8:
		n, err := d.uint8()
		return uint64(n), err
	case codes.Int8:
		n, err := d.uint8()
		return uint64(int8(n)), err
	case codes.Uint16:
		n, err := d.uint16()
		return uint64(n), err
	case codes.Int16:
		n, err := d.uint16()
		return uint64(int16(n)), err
	case codes.Uint32:
		n, err := d.uint32()
		return uint64(n), err
	case codes.Int32:
		n, err := d.uint32()
		return uint64(int32(n)), err
	case codes.Uint64, codes.Int64:
		return d.uint64()
	}
	return 0, fmt.Errorf("msgpack: invalid code %x decoding uint64", c)
}

func (d *Decoder) uint64Value(value reflect.Value) error {
	v, err := d.DecodeUint64()
	if err != nil {
		return err
	}
	value.SetUint(v)
	return nil
}

func (d *Decoder) DecodeInt64() (int64, error) {
	c, err := d.r.ReadByte()
	if err != nil {
		return 0, err
	}
	return d.int(c)
}

func (d *Decoder) int(c byte) (int64, error) {
	if c == codes.Nil {
		return 0, nil
	}
	if codes.IsFixedNum(c) {
		return int64(int8(c)), nil
	}
	switch c {
	case codes.Uint8:
		n, err := d.uint8()
		return int64(n), err
	case codes.Int8:
		n, err := d.uint8()
		return int64(int8(n)), err
	case codes.Uint16:
		n, err := d.uint16()
		return int64(n), err
	case codes.Int16:
		n, err := d.uint16()
		return int64(int16(n)), err
	case codes.Uint32:
		n, err := d.uint32()
		return int64(n), err
	case codes.Int32:
		n, err := d.uint32()
		return int64(int32(n)), err
	case codes.Uint64, codes.Int64:
		n, err := d.uint64()
		return int64(n), err
	}
	return 0, fmt.Errorf("msgpack: invalid code %x decoding int64", c)
}

func (d *Decoder) int64Value(v reflect.Value) error {
	n, err := d.DecodeInt64()
	if err != nil {
		return err
	}
	v.SetInt(n)
	return nil
}

func (d *Decoder) DecodeFloat32() (float32, error) {
	c, err := d.r.ReadByte()
	if err != nil {
		return 0, err
	}
	return d.float32(c)
}

func (d *Decoder) float32(c byte) (float32, error) {
	if c != codes.Float {
		return 0, fmt.Errorf("msgpack: invalid code %x decoding float32", c)
	}
	b, err := d.uint32()
	if err != nil {
		return 0, err
	}
	return math.Float32frombits(b), nil
}

func (d *Decoder) float32Value(value reflect.Value) error {
	v, err := d.DecodeFloat32()
	if err != nil {
		return err
	}
	value.SetFloat(float64(v))
	return nil
}

func (d *Decoder) DecodeFloat64() (float64, error) {
	c, err := d.r.ReadByte()
	if err != nil {
		return 0, err
	}
	return d.float64(c)
}

func (d *Decoder) float64(c byte) (float64, error) {
	if c == codes.Float {
		n, err := d.float32(c)
		return float64(n), err
	}
	if c != codes.Double {
		return 0, fmt.Errorf("msgpack: invalid code %x decoding float64", c)
	}
	b, err := d.uint64()
	if err != nil {
		return 0, err
	}
	return math.Float64frombits(b), nil
}

func (d *Decoder) float64Value(value reflect.Value) error {
	v, err := d.DecodeFloat64()
	if err != nil {
		return err
	}
	value.SetFloat(v)
	return nil
}

func (d *Decoder) DecodeUint() (uint, error) {
	n, err := d.DecodeUint64()
	return uint(n), err
}

func (d *Decoder) DecodeUint8() (uint8, error) {
	n, err := d.DecodeUint64()
	return uint8(n), err
}

func (d *Decoder) DecodeUint16() (uint16, error) {
	n, err := d.DecodeUint64()
	return uint16(n), err
}

func (d *Decoder) DecodeUint32() (uint32, error) {
	n, err := d.DecodeUint64()
	return uint32(n), err
}

func (d *Decoder) DecodeInt() (int, error) {
	n, err := d.DecodeInt64()
	return int(n), err
}

func (d *Decoder) DecodeInt8() (int8, error) {
	n, err := d.DecodeInt64()
	return int8(n), err
}

func (d *Decoder) DecodeInt16() (int16, error) {
	n, err := d.DecodeInt64()
	return int16(n), err
}

func (d *Decoder) DecodeInt32() (int32, error) {
	n, err := d.DecodeInt64()
	return int32(n), err
}
