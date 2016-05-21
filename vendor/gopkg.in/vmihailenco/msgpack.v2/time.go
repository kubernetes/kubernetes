package msgpack

import (
	"fmt"
	"reflect"
	"time"
)

var (
	timeType = reflect.TypeOf((*time.Time)(nil)).Elem()
)

func init() {
	Register(timeType, encodeTimeValue, decodeTimeValue)
}

func (e *Encoder) EncodeTime(tm time.Time) error {
	if err := e.w.WriteByte(0x92); err != nil {
		return err
	}
	if err := e.EncodeInt64(tm.Unix()); err != nil {
		return err
	}
	return e.EncodeInt(tm.Nanosecond())
}

func (d *Decoder) DecodeTime() (time.Time, error) {
	b, err := d.r.ReadByte()
	if err != nil {
		return time.Time{}, err
	}
	if b != 0x92 {
		return time.Time{}, fmt.Errorf("msgpack: invalid code %x decoding time", b)
	}

	sec, err := d.DecodeInt64()
	if err != nil {
		return time.Time{}, err
	}
	nsec, err := d.DecodeInt64()
	if err != nil {
		return time.Time{}, err
	}
	return time.Unix(sec, nsec), nil
}

func encodeTimeValue(e *Encoder, v reflect.Value) error {
	tm := v.Interface().(time.Time)
	return e.EncodeTime(tm)
}

func decodeTimeValue(d *Decoder, v reflect.Value) error {
	tm, err := d.DecodeTime()
	if err != nil {
		return err
	}
	v.Set(reflect.ValueOf(tm))
	return nil
}
