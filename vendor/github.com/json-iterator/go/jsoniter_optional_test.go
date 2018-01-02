package jsoniter

import (
	"github.com/stretchr/testify/require"
	"testing"
)

func Test_encode_optional_int_pointer(t *testing.T) {
	should := require.New(t)
	var ptr *int
	str, err := MarshalToString(ptr)
	should.Nil(err)
	should.Equal("null", str)
	val := 100
	ptr = &val
	str, err = MarshalToString(ptr)
	should.Nil(err)
	should.Equal("100", str)
}

func Test_decode_struct_with_optional_field(t *testing.T) {
	should := require.New(t)
	type TestObject struct {
		Field1 *string
		Field2 *string
	}
	obj := TestObject{}
	UnmarshalFromString(`{"field1": null, "field2": "world"}`, &obj)
	should.Nil(obj.Field1)
	should.Equal("world", *obj.Field2)
}

func Test_encode_struct_with_optional_field(t *testing.T) {
	should := require.New(t)
	type TestObject struct {
		Field1 *string
		Field2 *string
	}
	obj := TestObject{}
	world := "world"
	obj.Field2 = &world
	str, err := MarshalToString(obj)
	should.Nil(err)
	should.Contains(str, `"Field1":null`)
	should.Contains(str, `"Field2":"world"`)
}
