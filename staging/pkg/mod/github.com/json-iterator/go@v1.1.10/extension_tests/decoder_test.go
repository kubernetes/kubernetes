package test

import (
	"bytes"
	"fmt"
	"github.com/json-iterator/go"
	"github.com/stretchr/testify/require"
	"strconv"
	"testing"
	"time"
	"unsafe"
)

func Test_customize_type_decoder(t *testing.T) {
	t.Skip()
	jsoniter.RegisterTypeDecoderFunc("time.Time", func(ptr unsafe.Pointer, iter *jsoniter.Iterator) {
		t, err := time.ParseInLocation("2006-01-02 15:04:05", iter.ReadString(), time.UTC)
		if err != nil {
			iter.Error = err
			return
		}
		*((*time.Time)(ptr)) = t
	})
	//defer jsoniter.ConfigDefault.(*frozenConfig).cleanDecoders()
	val := time.Time{}
	err := jsoniter.Unmarshal([]byte(`"2016-12-05 08:43:28"`), &val)
	if err != nil {
		t.Fatal(err)
	}
	year, month, day := val.Date()
	if year != 2016 || month != 12 || day != 5 {
		t.Fatal(val)
	}
}

func Test_customize_byte_array_encoder(t *testing.T) {
	t.Skip()
	//jsoniter.ConfigDefault.(*frozenConfig).cleanEncoders()
	should := require.New(t)
	jsoniter.RegisterTypeEncoderFunc("[]uint8", func(ptr unsafe.Pointer, stream *jsoniter.Stream) {
		t := *((*[]byte)(ptr))
		stream.WriteString(string(t))
	}, nil)
	//defer jsoniter.ConfigDefault.(*frozenConfig).cleanEncoders()
	val := []byte("abc")
	str, err := jsoniter.MarshalToString(val)
	should.Nil(err)
	should.Equal(`"abc"`, str)
}

type CustomEncoderAttachmentTestStruct struct {
	Value int32 `json:"value"`
}

type CustomEncoderAttachmentTestStructEncoder struct {}

func (c *CustomEncoderAttachmentTestStructEncoder) Encode(ptr unsafe.Pointer, stream *jsoniter.Stream) {
	attachVal, ok := stream.Attachment.(int)
	stream.WriteRaw(`"`)
	stream.WriteRaw(fmt.Sprintf("%t %d", ok, attachVal))
	stream.WriteRaw(`"`)
}

func (c *CustomEncoderAttachmentTestStructEncoder) IsEmpty(ptr unsafe.Pointer) bool {
	return false
}

func Test_custom_encoder_attachment(t *testing.T) {

	jsoniter.RegisterTypeEncoder("test.CustomEncoderAttachmentTestStruct", &CustomEncoderAttachmentTestStructEncoder{})
	expectedValue := 17
	should := require.New(t)
	buf := &bytes.Buffer{}
	stream := jsoniter.NewStream(jsoniter.Config{SortMapKeys: true}.Froze(), buf, 4096)
	stream.Attachment = expectedValue
	val := map[string]CustomEncoderAttachmentTestStruct{"a": {}}
	stream.WriteVal(val)
	stream.Flush()
	should.Nil(stream.Error)
	should.Equal("{\"a\":\"true 17\"}", buf.String())
}

func Test_customize_field_decoder(t *testing.T) {
	type Tom struct {
		field1 string
	}
	jsoniter.RegisterFieldDecoderFunc("jsoniter.Tom", "field1", func(ptr unsafe.Pointer, iter *jsoniter.Iterator) {
		*((*string)(ptr)) = strconv.Itoa(iter.ReadInt())
	})
	//defer jsoniter.ConfigDefault.(*frozenConfig).cleanDecoders()
	tom := Tom{}
	err := jsoniter.Unmarshal([]byte(`{"field1": 100}`), &tom)
	if err != nil {
		t.Fatal(err)
	}
}

func Test_recursive_empty_interface_customization(t *testing.T) {
	t.Skip()
	var obj interface{}
	jsoniter.RegisterTypeDecoderFunc("interface {}", func(ptr unsafe.Pointer, iter *jsoniter.Iterator) {
		switch iter.WhatIsNext() {
		case jsoniter.NumberValue:
			*(*interface{})(ptr) = iter.ReadInt64()
		default:
			*(*interface{})(ptr) = iter.Read()
		}
	})
	should := require.New(t)
	jsoniter.Unmarshal([]byte("[100]"), &obj)
	should.Equal([]interface{}{int64(100)}, obj)
}

type MyInterface interface {
	Hello() string
}

type MyString string

func (ms MyString) Hello() string {
	return string(ms)
}

func Test_read_custom_interface(t *testing.T) {
	t.Skip()
	should := require.New(t)
	var val MyInterface
	jsoniter.RegisterTypeDecoderFunc("jsoniter.MyInterface", func(ptr unsafe.Pointer, iter *jsoniter.Iterator) {
		*((*MyInterface)(ptr)) = MyString(iter.ReadString())
	})
	err := jsoniter.UnmarshalFromString(`"hello"`, &val)
	should.Nil(err)
	should.Equal("hello", val.Hello())
}

const flow1 = `
{"A":"hello"}
{"A":"hello"}
{"A":"hello"}
{"A":"hello"}
{"A":"hello"}`

const flow2 = `
{"A":"hello"}
{"A":"hello"}
{"A":"hello"}
{"A":"hello"}
{"A":"hello"}
`

type (
	Type1 struct {
		A string
	}

	Type2 struct {
		A string
	}
)

func (t *Type2) UnmarshalJSON(data []byte) error {
	return nil
}

func (t *Type2) MarshalJSON() ([]byte, error) {
	return nil, nil
}

func TestType1NoFinalLF(t *testing.T) {
	reader := bytes.NewReader([]byte(flow1))
	dec := jsoniter.NewDecoder(reader)

	i := 0
	for dec.More() {
		data := &Type1{}
		if err := dec.Decode(data); err != nil {
			t.Errorf("at %v got %v", i, err)
		}
		i++
	}
}

func TestType1FinalLF(t *testing.T) {
	reader := bytes.NewReader([]byte(flow2))
	dec := jsoniter.NewDecoder(reader)

	i := 0
	for dec.More() {
		data := &Type1{}
		if err := dec.Decode(data); err != nil {
			t.Errorf("at %v got %v", i, err)
		}
		i++
	}
}

func TestType2NoFinalLF(t *testing.T) {
	reader := bytes.NewReader([]byte(flow1))
	dec := jsoniter.NewDecoder(reader)

	i := 0
	for dec.More() {
		data := &Type2{}
		if err := dec.Decode(data); err != nil {
			t.Errorf("at %v got %v", i, err)
		}
		i++
	}
}

func TestType2FinalLF(t *testing.T) {
	reader := bytes.NewReader([]byte(flow2))
	dec := jsoniter.NewDecoder(reader)

	i := 0
	for dec.More() {
		data := &Type2{}
		if err := dec.Decode(data); err != nil {
			t.Errorf("at %v got %v", i, err)
		}
		i++
	}
}
