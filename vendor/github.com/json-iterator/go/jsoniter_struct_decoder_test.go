package jsoniter

import (
	"github.com/stretchr/testify/require"
	"testing"
)

func Test_decode_one_field_struct(t *testing.T) {
	should := require.New(t)
	type TestObject struct {
		Field1 string
	}
	obj := TestObject{}
	should.Nil(UnmarshalFromString(`{}`, &obj))
	should.Equal("", obj.Field1)
	should.Nil(UnmarshalFromString(`{"field1": "hello"}`, &obj))
	should.Equal("hello", obj.Field1)
}

func Test_decode_two_fields_struct(t *testing.T) {
	should := require.New(t)
	type TestObject struct {
		Field1 string
		Field2 string
	}
	obj := TestObject{}
	should.Nil(UnmarshalFromString(`{}`, &obj))
	should.Equal("", obj.Field1)
	should.Nil(UnmarshalFromString(`{"Field1": "a", "Field2": "stream"}`, &obj))
	should.Equal("a", obj.Field1)
	should.Equal("stream", obj.Field2)
}

func Test_decode_three_fields_struct(t *testing.T) {
	should := require.New(t)
	type TestObject struct {
		Field1 string
		Field2 string
		Field3 string
	}
	obj := TestObject{}
	should.Nil(UnmarshalFromString(`{}`, &obj))
	should.Equal("", obj.Field1)
	should.Nil(UnmarshalFromString(`{"Field1": "a", "Field2": "stream", "Field3": "c"}`, &obj))
	should.Equal("a", obj.Field1)
	should.Equal("stream", obj.Field2)
	should.Equal("c", obj.Field3)
}

func Test_decode_four_fields_struct(t *testing.T) {
	should := require.New(t)
	type TestObject struct {
		Field1 string
		Field2 string
		Field3 string
		Field4 string
	}
	obj := TestObject{}
	should.Nil(UnmarshalFromString(`{}`, &obj))
	should.Equal("", obj.Field1)
	should.Nil(UnmarshalFromString(`{"Field1": "a", "Field2": "stream", "Field3": "c", "Field4": "d"}`, &obj))
	should.Equal("a", obj.Field1)
	should.Equal("stream", obj.Field2)
	should.Equal("c", obj.Field3)
	should.Equal("d", obj.Field4)
}

func Test_decode_five_fields_struct(t *testing.T) {
	should := require.New(t)
	type TestObject struct {
		Field1 string
		Field2 string
		Field3 string
		Field4 string
		Field5 string
	}
	obj := TestObject{}
	should.Nil(UnmarshalFromString(`{}`, &obj))
	should.Equal("", obj.Field1)
	should.Nil(UnmarshalFromString(`{"Field1": "a", "Field2": "stream", "Field3": "c", "Field4": "d", "Field5": "e"}`, &obj))
	should.Equal("a", obj.Field1)
	should.Equal("stream", obj.Field2)
	should.Equal("c", obj.Field3)
	should.Equal("d", obj.Field4)
	should.Equal("e", obj.Field5)
}

func Test_decode_six_fields_struct(t *testing.T) {
	should := require.New(t)
	type TestObject struct {
		Field1 string
		Field2 string
		Field3 string
		Field4 string
		Field5 string
		Field6 string
	}
	obj := TestObject{}
	should.Nil(UnmarshalFromString(`{}`, &obj))
	should.Equal("", obj.Field1)
	should.Nil(UnmarshalFromString(`{"Field1": "a", "Field2": "stream", "Field3": "c", "Field4": "d", "Field5": "e", "Field6": "x"}`, &obj))
	should.Equal("a", obj.Field1)
	should.Equal("stream", obj.Field2)
	should.Equal("c", obj.Field3)
	should.Equal("d", obj.Field4)
	should.Equal("e", obj.Field5)
	should.Equal("x", obj.Field6)
}

func Test_decode_seven_fields_struct(t *testing.T) {
	should := require.New(t)
	type TestObject struct {
		Field1 string
		Field2 string
		Field3 string
		Field4 string
		Field5 string
		Field6 string
		Field7 string
	}
	obj := TestObject{}
	should.Nil(UnmarshalFromString(`{}`, &obj))
	should.Equal("", obj.Field1)
	should.Nil(UnmarshalFromString(`{"Field1": "a", "Field2": "stream", "Field3": "c", "Field4": "d", "Field5": "e", "Field6": "x", "Field7":"y"}`, &obj))
	should.Equal("a", obj.Field1)
	should.Equal("stream", obj.Field2)
	should.Equal("c", obj.Field3)
	should.Equal("d", obj.Field4)
	should.Equal("e", obj.Field5)
	should.Equal("x", obj.Field6)
	should.Equal("y", obj.Field7)
}

func Test_decode_eight_fields_struct(t *testing.T) {
	should := require.New(t)
	type TestObject struct {
		Field1 string
		Field2 string
		Field3 string
		Field4 string
		Field5 string
		Field6 string
		Field7 string
		Field8 string
	}
	obj := TestObject{}
	should.Nil(UnmarshalFromString(`{}`, &obj))
	should.Equal("", obj.Field1)
	should.Nil(UnmarshalFromString(`{"Field8":"1", "Field1": "a", "Field2": "stream", "Field3": "c", "Field4": "d", "Field5": "e", "Field6": "x", "Field7":"y"}`, &obj))
	should.Equal("a", obj.Field1)
	should.Equal("stream", obj.Field2)
	should.Equal("c", obj.Field3)
	should.Equal("d", obj.Field4)
	should.Equal("e", obj.Field5)
	should.Equal("x", obj.Field6)
	should.Equal("y", obj.Field7)
	should.Equal("1", obj.Field8)
}

func Test_decode_nine_fields_struct(t *testing.T) {
	should := require.New(t)
	type TestObject struct {
		Field1 string
		Field2 string
		Field3 string
		Field4 string
		Field5 string
		Field6 string
		Field7 string
		Field8 string
		Field9 string
	}
	obj := TestObject{}
	should.Nil(UnmarshalFromString(`{}`, &obj))
	should.Equal("", obj.Field1)
	should.Nil(UnmarshalFromString(`{"Field8" : "zzzzzzzzzzz", "Field7": "zz", "Field6" : "xx", "Field1": "a", "Field2": "stream", "Field3": "c", "Field4": "d", "Field5": "e", "Field9":"f"}`, &obj))
	should.Equal("a", obj.Field1)
	should.Equal("stream", obj.Field2)
	should.Equal("c", obj.Field3)
	should.Equal("d", obj.Field4)
	should.Equal("e", obj.Field5)
	should.Equal("xx", obj.Field6)
	should.Equal("zz", obj.Field7)
	should.Equal("zzzzzzzzzzz", obj.Field8)
	should.Equal("f", obj.Field9)
}

func Test_decode_ten_fields_struct(t *testing.T) {
	should := require.New(t)
	type TestObject struct {
		Field1  string
		Field2  string
		Field3  string
		Field4  string
		Field5  string
		Field6  string
		Field7  string
		Field8  string
		Field9  string
		Field10 string
	}
	obj := TestObject{}
	should.Nil(UnmarshalFromString(`{}`, &obj))
	should.Equal("", obj.Field1)
	should.Nil(UnmarshalFromString(`{"Field10":"x", "Field9": "x", "Field8":"x", "Field7":"x", "Field6":"x", "Field1": "a", "Field2": "stream", "Field3": "c", "Field4": "d", "Field5": "e"}`, &obj))
	should.Equal("a", obj.Field1)
	should.Equal("stream", obj.Field2)
	should.Equal("c", obj.Field3)
	should.Equal("d", obj.Field4)
	should.Equal("e", obj.Field5)
	should.Equal("x", obj.Field6)
	should.Equal("x", obj.Field7)
	should.Equal("x", obj.Field8)
	should.Equal("x", obj.Field9)
	should.Equal("x", obj.Field10)
}

func Test_decode_more_than_ten_fields_struct(t *testing.T) {
	should := require.New(t)
	type TestObject struct {
		Field1  string
		Field2  string
		Field3  string
		Field4  string
		Field5  string
		Field6  string
		Field7  string
		Field8  string
		Field9  string
		Field10 string
		Field11 int
	}
	obj := TestObject{}
	should.Nil(UnmarshalFromString(`{}`, &obj))
	should.Equal("", obj.Field1)
	should.Nil(UnmarshalFromString(`{"field11":1, "field1": "a", "Field2": "stream", "Field3": "c", "Field4": "d", "Field5": "e"}`, &obj))
	should.Equal("a", obj.Field1)
	should.Equal("stream", obj.Field2)
	should.Equal("c", obj.Field3)
	should.Equal("d", obj.Field4)
	should.Equal("e", obj.Field5)
	should.Equal(1, obj.Field11)
}

func Test_decode_struct_field_with_tag(t *testing.T) {
	should := require.New(t)
	type TestObject struct {
		Field1 string `json:"field-1"`
		Field2 string `json:"-"`
		Field3 int    `json:",string"`
	}
	obj := TestObject{Field2: "world"}
	UnmarshalFromString(`{"field-1": "hello", "field2": "", "Field3": "100"}`, &obj)
	should.Equal("hello", obj.Field1)
	should.Equal("world", obj.Field2)
	should.Equal(100, obj.Field3)
}

func Test_decode_struct_field_with_tag_string(t *testing.T) {
	should := require.New(t)
	type TestObject struct {
		Field1 int `json:",string"`
	}
	obj := TestObject{Field1: 100}
	should.Nil(UnmarshalFromString(`{"Field1": "100"}`, &obj))
	should.Equal(100, obj.Field1)
}
