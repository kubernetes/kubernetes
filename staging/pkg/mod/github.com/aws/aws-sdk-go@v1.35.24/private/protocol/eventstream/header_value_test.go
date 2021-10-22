package eventstream

import (
	"bytes"
	"encoding/binary"
	"io"
	"reflect"
	"testing"
	"time"
)

func binWrite(v interface{}) []byte {
	var w bytes.Buffer
	binary.Write(&w, binary.BigEndian, v)
	return w.Bytes()
}

var testValueEncodingCases = []struct {
	Val    Value
	Expect []byte
	Decode func(io.Reader) (Value, error)
}{
	{
		BoolValue(true),
		[]byte{byte(trueValueType)},
		nil,
	},
	{
		BoolValue(false),
		[]byte{byte(falseValueType)},
		nil,
	},
	{
		Int8Value(0x0f),
		[]byte{byte(int8ValueType), 0x0f},
		func(r io.Reader) (Value, error) {
			var v Int8Value
			err := v.decode(r)
			return v, err
		},
	},
	{
		Int16Value(0x0f),
		append([]byte{byte(int16ValueType)}, binWrite(int16(0x0f))...),
		func(r io.Reader) (Value, error) {
			var v Int16Value
			err := v.decode(r)
			return v, err
		},
	},
	{
		Int32Value(0x0f),
		append([]byte{byte(int32ValueType)}, binWrite(int32(0x0f))...),
		func(r io.Reader) (Value, error) {
			var v Int32Value
			err := v.decode(r)
			return v, err
		},
	},
	{
		Int64Value(0x0f),
		append([]byte{byte(int64ValueType)}, binWrite(int64(0x0f))...),
		func(r io.Reader) (Value, error) {
			var v Int64Value
			err := v.decode(r)
			return v, err
		},
	},
	{
		BytesValue([]byte{0, 1, 2, 3}),
		[]byte{byte(bytesValueType), 0x00, 0x04, 0, 1, 2, 3},
		func(r io.Reader) (Value, error) {
			var v BytesValue
			err := v.decode(r)
			return v, err
		},
	},
	{
		StringValue("abc123"),
		append([]byte{byte(stringValueType), 0, 6}, []byte("abc123")...),
		func(r io.Reader) (Value, error) {
			var v StringValue
			err := v.decode(r)
			return v, err
		},
	},
	{
		TimestampValue(
			time.Date(2014, 04, 04, 0, 1, 0, 0, time.FixedZone("PDT", -7)),
		),
		append([]byte{byte(timestampValueType)}, binWrite(int64(1396569667000))...),
		func(r io.Reader) (Value, error) {
			var v TimestampValue
			err := v.decode(r)
			return v, err
		},
	},
	{
		UUIDValue(
			[16]byte{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0xa, 0xb, 0xc, 0xd, 0xe, 0xf},
		),
		[]byte{byte(uuidValueType), 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0xa, 0xb, 0xc, 0xd, 0xe, 0xf},
		func(r io.Reader) (Value, error) {
			var v UUIDValue
			err := v.decode(r)
			return v, err
		},
	},
}

func TestValue_MarshalValue(t *testing.T) {
	for i, c := range testValueEncodingCases {
		var w bytes.Buffer

		if err := c.Val.encode(&w); err != nil {
			t.Fatalf("%d, expect no error, got %v", i, err)
		}

		if e, a := c.Expect, w.Bytes(); !reflect.DeepEqual(e, a) {
			t.Errorf("%d, expect %v, got %v", i, e, a)
		}
	}
}

func TestHeader_DecodeValues(t *testing.T) {
	for i, c := range testValueEncodingCases {
		r := bytes.NewBuffer(c.Expect)
		v, err := decodeHeaderValue(r)
		if err != nil {
			t.Fatalf("%d, expect no error, got %v", i, err)
		}

		switch tv := v.(type) {
		case TimestampValue:
			exp := time.Time(c.Val.(TimestampValue))
			if e, a := exp, time.Time(tv); !e.Equal(a) {
				t.Errorf("%d, expect %v, got %v", i, e, a)
			}
		default:
			if e, a := c.Val, v; !reflect.DeepEqual(e, a) {
				t.Errorf("%d, expect %v, got %v", i, e, a)
			}
		}
	}
}

func TestValue_Decode(t *testing.T) {
	for i, c := range testValueEncodingCases {
		if c.Decode == nil {
			continue
		}

		r := bytes.NewBuffer(c.Expect)
		r.ReadByte() // strip off Type field

		v, err := c.Decode(r)
		if err != nil {
			t.Fatalf("%d, expect no error, got %v", i, err)
		}

		switch tv := v.(type) {
		case TimestampValue:
			exp := time.Time(c.Val.(TimestampValue))
			if e, a := exp, time.Time(tv); !e.Equal(a) {
				t.Errorf("%d, expect %v, got %v", i, e, a)
			}
		default:
			if e, a := c.Val, v; !reflect.DeepEqual(e, a) {
				t.Errorf("%d, expect %v, got %v", i, e, a)
			}
		}
	}
}

func TestValue_String(t *testing.T) {
	cases := []struct {
		Val    Value
		Expect string
	}{
		{BoolValue(true), "true"},
		{BoolValue(false), "false"},
		{Int8Value(0x0f), "0x0f"},
		{Int16Value(0x0f), "0x000f"},
		{Int32Value(0x0f), "0x0000000f"},
		{Int64Value(0x0f), "0x000000000000000f"},
		{BytesValue([]byte{0, 1, 2, 3}), "AAECAw=="},
		{StringValue("abc123"), "abc123"},
		{TimestampValue(
			time.Date(2014, 04, 04, 0, 1, 0, 0, time.FixedZone("PDT", -7)),
		),
			"1396569667000",
		},
		{UUIDValue([16]byte{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0xa, 0xb, 0xc, 0xd, 0xe, 0xf}),
			"00010203-0405-0607-0809-0A0B0C0D0E0F",
		},
	}

	for i, c := range cases {
		if e, a := c.Expect, c.Val.String(); e != a {
			t.Errorf("%d, expect %v, got %v", i, e, a)
		}
	}
}
