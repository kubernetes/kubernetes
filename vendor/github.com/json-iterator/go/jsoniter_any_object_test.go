package jsoniter

import (
	"testing"

	"github.com/stretchr/testify/require"
)

func Test_read_object_as_any(t *testing.T) {
	should := require.New(t)
	any := Get([]byte(`{"a":"stream","c":"d"}`))
	should.Equal(`{"a":"stream","c":"d"}`, any.ToString())
	// partial parse
	should.Equal("stream", any.Get("a").ToString())
	should.Equal("d", any.Get("c").ToString())
	should.Equal(2, len(any.Keys()))
	any = Get([]byte(`{"a":"stream","c":"d"}`))
	// full parse
	should.Equal(2, len(any.Keys()))
	should.Equal(2, any.Size())
	should.True(any.ToBool())
	should.Equal(0, any.ToInt())
	should.Equal(ObjectValue, any.ValueType())
	should.Nil(any.LastError())
	obj := struct {
		A string
	}{}
	any.ToVal(&obj)
	should.Equal("stream", obj.A)
}

func Test_object_lazy_any_get(t *testing.T) {
	should := require.New(t)
	any := Get([]byte(`{"a":{"stream":{"c":"d"}}}`))
	should.Equal("d", any.Get("a", "stream", "c").ToString())
}

func Test_object_lazy_any_get_all(t *testing.T) {
	should := require.New(t)
	any := Get([]byte(`{"a":[0],"stream":[1]}`))
	should.Contains(any.Get('*', 0).ToString(), `"a":0`)
}

func Test_object_lazy_any_get_invalid(t *testing.T) {
	should := require.New(t)
	any := Get([]byte(`{}`))
	should.Equal(InvalidValue, any.Get("a", "stream", "c").ValueType())
	should.Equal(InvalidValue, any.Get(1).ValueType())
}

func Test_wrap_map_and_convert_to_any(t *testing.T) {
	should := require.New(t)
	any := Wrap(map[string]interface{}{"a": 1})
	should.True(any.ToBool())
	should.Equal(0, any.ToInt())
	should.Equal(int32(0), any.ToInt32())
	should.Equal(int64(0), any.ToInt64())
	should.Equal(float32(0), any.ToFloat32())
	should.Equal(float64(0), any.ToFloat64())
	should.Equal(uint(0), any.ToUint())
	should.Equal(uint32(0), any.ToUint32())
	should.Equal(uint64(0), any.ToUint64())
}

func Test_wrap_object_and_convert_to_any(t *testing.T) {
	should := require.New(t)
	type TestObject struct {
		Field1 string
		field2 string
	}
	any := Wrap(TestObject{"hello", "world"})
	should.Equal("hello", any.Get("Field1").ToString())
	any = Wrap(TestObject{"hello", "world"})
	should.Equal(2, any.Size())
	should.Equal(`{"Field1":"hello"}`, any.Get('*').ToString())

	should.Equal(0, any.ToInt())
	should.Equal(int32(0), any.ToInt32())
	should.Equal(int64(0), any.ToInt64())
	should.Equal(float32(0), any.ToFloat32())
	should.Equal(float64(0), any.ToFloat64())
	should.Equal(uint(0), any.ToUint())
	should.Equal(uint32(0), any.ToUint32())
	should.Equal(uint64(0), any.ToUint64())
	should.True(any.ToBool())
	should.Equal(`{"Field1":"hello"}`, any.ToString())

	// cannot pass!
	//stream := NewStream(ConfigDefault, nil, 32)
	//any.WriteTo(stream)
	//should.Equal(`{"Field1":"hello"}`, string(stream.Buffer()))
	// cannot pass!

}

func Test_any_within_struct(t *testing.T) {
	should := require.New(t)
	type TestObject struct {
		Field1 Any
		Field2 Any
	}
	obj := TestObject{}
	err := UnmarshalFromString(`{"Field1": "hello", "Field2": [1,2,3]}`, &obj)
	should.Nil(err)
	should.Equal("hello", obj.Field1.ToString())
	should.Equal("[1,2,3]", obj.Field2.ToString())
}
