package jsoniter

import (
	"encoding/json"
	"strconv"
	"testing"
	"time"
	"unsafe"

	"github.com/stretchr/testify/require"
)

func Test_customize_type_decoder(t *testing.T) {
	RegisterTypeDecoderFunc("time.Time", func(ptr unsafe.Pointer, iter *Iterator) {
		t, err := time.ParseInLocation("2006-01-02 15:04:05", iter.ReadString(), time.UTC)
		if err != nil {
			iter.Error = err
			return
		}
		*((*time.Time)(ptr)) = t
	})
	defer ConfigDefault.(*frozenConfig).cleanDecoders()
	val := time.Time{}
	err := Unmarshal([]byte(`"2016-12-05 08:43:28"`), &val)
	if err != nil {
		t.Fatal(err)
	}
	year, month, day := val.Date()
	if year != 2016 || month != 12 || day != 5 {
		t.Fatal(val)
	}
}

func Test_customize_type_encoder(t *testing.T) {
	should := require.New(t)
	RegisterTypeEncoderFunc("time.Time", func(ptr unsafe.Pointer, stream *Stream) {
		t := *((*time.Time)(ptr))
		stream.WriteString(t.UTC().Format("2006-01-02 15:04:05"))
	}, nil)
	defer ConfigDefault.(*frozenConfig).cleanEncoders()
	val := time.Unix(0, 0)
	str, err := MarshalToString(val)
	should.Nil(err)
	should.Equal(`"1970-01-01 00:00:00"`, str)
}

func Test_customize_byte_array_encoder(t *testing.T) {
	ConfigDefault.(*frozenConfig).cleanEncoders()
	should := require.New(t)
	RegisterTypeEncoderFunc("[]uint8", func(ptr unsafe.Pointer, stream *Stream) {
		t := *((*[]byte)(ptr))
		stream.WriteString(string(t))
	}, nil)
	defer ConfigDefault.(*frozenConfig).cleanEncoders()
	val := []byte("abc")
	str, err := MarshalToString(val)
	should.Nil(err)
	should.Equal(`"abc"`, str)
}

func Test_customize_float_marshal(t *testing.T) {
	should := require.New(t)
	json := Config{MarshalFloatWith6Digits: true}.Froze()
	str, err := json.MarshalToString(float32(1.23456789))
	should.Nil(err)
	should.Equal("1.234568", str)
}

type Tom struct {
	field1 string
}

func Test_customize_field_decoder(t *testing.T) {
	RegisterFieldDecoderFunc("jsoniter.Tom", "field1", func(ptr unsafe.Pointer, iter *Iterator) {
		*((*string)(ptr)) = strconv.Itoa(iter.ReadInt())
	})
	defer ConfigDefault.(*frozenConfig).cleanDecoders()
	tom := Tom{}
	err := Unmarshal([]byte(`{"field1": 100}`), &tom)
	if err != nil {
		t.Fatal(err)
	}
}

type TestObject1 struct {
	Field1 string
}

type testExtension struct {
	DummyExtension
}

func (extension *testExtension) UpdateStructDescriptor(structDescriptor *StructDescriptor) {
	if structDescriptor.Type.String() != "jsoniter.TestObject1" {
		return
	}
	binding := structDescriptor.GetField("Field1")
	binding.Encoder = &funcEncoder{fun: func(ptr unsafe.Pointer, stream *Stream) {
		str := *((*string)(ptr))
		val, _ := strconv.Atoi(str)
		stream.WriteInt(val)
	}}
	binding.Decoder = &funcDecoder{func(ptr unsafe.Pointer, iter *Iterator) {
		*((*string)(ptr)) = strconv.Itoa(iter.ReadInt())
	}}
	binding.ToNames = []string{"field-1"}
	binding.FromNames = []string{"field-1"}
}

func Test_customize_field_by_extension(t *testing.T) {
	should := require.New(t)
	RegisterExtension(&testExtension{})
	obj := TestObject1{}
	err := UnmarshalFromString(`{"field-1": 100}`, &obj)
	should.Nil(err)
	should.Equal("100", obj.Field1)
	str, err := MarshalToString(obj)
	should.Nil(err)
	should.Equal(`{"field-1":100}`, str)
}

type timeImplementedMarshaler time.Time

func (obj timeImplementedMarshaler) MarshalJSON() ([]byte, error) {
	seconds := time.Time(obj).Unix()
	return []byte(strconv.FormatInt(seconds, 10)), nil
}

func Test_marshaler(t *testing.T) {
	type TestObject struct {
		Field timeImplementedMarshaler
	}
	should := require.New(t)
	val := timeImplementedMarshaler(time.Unix(123, 0))
	obj := TestObject{val}
	bytes, err := json.Marshal(obj)
	should.Nil(err)
	should.Equal(`{"Field":123}`, string(bytes))
	str, err := MarshalToString(obj)
	should.Nil(err)
	should.Equal(`{"Field":123}`, str)
}

func Test_marshaler_and_encoder(t *testing.T) {
	type TestObject struct {
		Field *timeImplementedMarshaler
	}
	ConfigDefault.(*frozenConfig).cleanEncoders()
	should := require.New(t)
	RegisterTypeEncoderFunc("jsoniter.timeImplementedMarshaler", func(ptr unsafe.Pointer, stream *Stream) {
		stream.WriteString("hello from encoder")
	}, nil)
	val := timeImplementedMarshaler(time.Unix(123, 0))
	obj := TestObject{&val}
	bytes, err := json.Marshal(obj)
	should.Nil(err)
	should.Equal(`{"Field":123}`, string(bytes))
	str, err := MarshalToString(obj)
	should.Nil(err)
	should.Equal(`{"Field":"hello from encoder"}`, str)
}

type ObjectImplementedUnmarshaler int

func (obj *ObjectImplementedUnmarshaler) UnmarshalJSON(s []byte) error {
	val, _ := strconv.ParseInt(string(s[1:len(s)-1]), 10, 64)
	*obj = ObjectImplementedUnmarshaler(val)
	return nil
}

func Test_unmarshaler(t *testing.T) {
	should := require.New(t)
	var obj ObjectImplementedUnmarshaler
	err := json.Unmarshal([]byte(`   "100" `), &obj)
	should.Nil(err)
	should.Equal(100, int(obj))
	iter := ParseString(ConfigDefault, `   "100" `)
	iter.ReadVal(&obj)
	should.Nil(err)
	should.Equal(100, int(obj))
}

func Test_unmarshaler_and_decoder(t *testing.T) {
	type TestObject struct {
		Field  *ObjectImplementedUnmarshaler
		Field2 string
	}
	ConfigDefault.(*frozenConfig).cleanDecoders()
	should := require.New(t)
	RegisterTypeDecoderFunc("jsoniter.ObjectImplementedUnmarshaler", func(ptr unsafe.Pointer, iter *Iterator) {
		*(*ObjectImplementedUnmarshaler)(ptr) = 10
		iter.Skip()
	})
	obj := TestObject{}
	val := ObjectImplementedUnmarshaler(0)
	obj.Field = &val
	err := json.Unmarshal([]byte(`{"Field":"100"}`), &obj)
	should.Nil(err)
	should.Equal(100, int(*obj.Field))
	err = Unmarshal([]byte(`{"Field":"100"}`), &obj)
	should.Nil(err)
	should.Equal(10, int(*obj.Field))
}

type tmString string
type tmStruct struct {
	String tmString
}

func (s tmStruct) MarshalJSON() ([]byte, error) {
	var b []byte
	b = append(b, '"')
	b = append(b, s.String...)
	b = append(b, '"')
	return b, nil
}

func Test_marshaler_on_struct(t *testing.T) {
	fixed := tmStruct{"hello"}
	//json.Marshal(fixed)
	Marshal(fixed)
}

type withChan struct {
	F2 chan []byte
}

func (q withChan) MarshalJSON() ([]byte, error) {
	return []byte(`""`), nil
}

func (q *withChan) UnmarshalJSON(value []byte) error {
	return nil
}

func Test_marshal_json_with_chan(t *testing.T) {
	type TestObject struct {
		F1 withChan
	}
	should := require.New(t)
	output, err := MarshalToString(TestObject{})
	should.Nil(err)
	should.Equal(`{"F1":""}`, output)
}

type withTime struct {
	time.Time
}

func (t *withTime) UnmarshalJSON(b []byte) error {
	return nil
}
func (t withTime) MarshalJSON() ([]byte, error) {
	return []byte(`"fake"`), nil
}

func Test_marshal_json_with_time(t *testing.T) {
	type S1 struct {
		F1 withTime
		F2 *withTime
	}
	type TestObject struct {
		TF1 S1
	}
	should := require.New(t)
	obj := TestObject{
		S1{
			F1: withTime{
				time.Unix(0, 0),
			},
			F2: &withTime{
				time.Unix(0, 0),
			},
		},
	}
	output, err := json.Marshal(obj)
	should.Nil(err)
	should.Equal(`{"TF1":{"F1":"fake","F2":"fake"}}`, string(output))
	output, err = Marshal(obj)
	should.Nil(err)
	should.Equal(`{"TF1":{"F1":"fake","F2":"fake"}}`, string(output))
	obj = TestObject{}
	should.Nil(json.Unmarshal([]byte(`{"TF1":{"F1":"fake","F2":"fake"}}`), &obj))
	should.NotNil(obj.TF1.F2)
	obj = TestObject{}
	should.Nil(Unmarshal([]byte(`{"TF1":{"F1":"fake","F2":"fake"}}`), &obj))
	should.NotNil(obj.TF1.F2)
}

func Test_customize_tag_key(t *testing.T) {

	type TestObject struct {
		Field string `orm:"field"`
	}

	should := require.New(t)
	json := Config{TagKey: "orm"}.Froze()
	str, err := json.MarshalToString(TestObject{"hello"})
	should.Nil(err)
	should.Equal(`{"field":"hello"}`, str)
}

func Test_recursive_empty_interface_customization(t *testing.T) {
	t.Skip()
	var obj interface{}
	RegisterTypeDecoderFunc("interface {}", func(ptr unsafe.Pointer, iter *Iterator) {
		switch iter.WhatIsNext() {
		case NumberValue:
			*(*interface{})(ptr) = iter.ReadInt64()
		default:
			*(*interface{})(ptr) = iter.Read()
		}
	})
	should := require.New(t)
	Unmarshal([]byte("[100]"), &obj)
	should.Equal([]interface{}{int64(100)}, obj)
}

type GeoLocation struct {
	Id string `json:"id,omitempty" db:"id"`
}

func (p *GeoLocation) MarshalJSON() ([]byte, error) {
	return []byte(`{}`), nil
}

func (p *GeoLocation) UnmarshalJSON(input []byte) error {
	p.Id = "hello"
	return nil
}

func Test_marshal_and_unmarshal_on_non_pointer(t *testing.T) {
	should := require.New(t)
	locations := []GeoLocation{{"000"}}
	bytes, err := Marshal(locations)
	should.Nil(err)
	should.Equal("[{}]", string(bytes))
	err = Unmarshal([]byte("[1]"), &locations)
	should.Nil(err)
	should.Equal("hello", locations[0].Id)
}
