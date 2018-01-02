package jsoniter

import (
	"testing"

	"github.com/stretchr/testify/require"
)

func Test_wrap_and_valuetype_everything(t *testing.T) {
	should := require.New(t)
	var i interface{}
	any := Get([]byte("123"))
	// default of number type is float64
	i = float64(123)
	should.Equal(i, any.GetInterface())

	any = Wrap(int8(10))
	should.Equal(any.ValueType(), NumberValue)
	should.Equal(any.LastError(), nil)
	//  get interface is not int8 interface
	// i = int8(10)
	// should.Equal(i, any.GetInterface())

	any = Wrap(int16(10))
	should.Equal(any.ValueType(), NumberValue)
	should.Equal(any.LastError(), nil)
	//i = int16(10)
	//should.Equal(i, any.GetInterface())

	any = Wrap(int32(10))
	should.Equal(any.ValueType(), NumberValue)
	should.Equal(any.LastError(), nil)
	i = int32(10)
	should.Equal(i, any.GetInterface())
	any = Wrap(int64(10))
	should.Equal(any.ValueType(), NumberValue)
	should.Equal(any.LastError(), nil)
	i = int64(10)
	should.Equal(i, any.GetInterface())

	any = Wrap(uint(10))
	should.Equal(any.ValueType(), NumberValue)
	should.Equal(any.LastError(), nil)
	// not equal
	//i = uint(10)
	//should.Equal(i, any.GetInterface())
	any = Wrap(uint8(10))
	should.Equal(any.ValueType(), NumberValue)
	should.Equal(any.LastError(), nil)
	// not equal
	// i = uint8(10)
	// should.Equal(i, any.GetInterface())
	any = Wrap(uint16(10))
	should.Equal(any.ValueType(), NumberValue)
	should.Equal(any.LastError(), nil)
	any = Wrap(uint32(10))
	should.Equal(any.ValueType(), NumberValue)
	should.Equal(any.LastError(), nil)
	i = uint32(10)
	should.Equal(i, any.GetInterface())
	any = Wrap(uint64(10))
	should.Equal(any.ValueType(), NumberValue)
	should.Equal(any.LastError(), nil)
	i = uint64(10)
	should.Equal(i, any.GetInterface())

	any = Wrap(float32(10))
	should.Equal(any.ValueType(), NumberValue)
	should.Equal(any.LastError(), nil)
	// not equal
	//i = float32(10)
	//should.Equal(i, any.GetInterface())
	any = Wrap(float64(10))
	should.Equal(any.ValueType(), NumberValue)
	should.Equal(any.LastError(), nil)
	i = float64(10)
	should.Equal(i, any.GetInterface())

	any = Wrap(true)
	should.Equal(any.ValueType(), BoolValue)
	should.Equal(any.LastError(), nil)
	i = true
	should.Equal(i, any.GetInterface())
	any = Wrap(false)
	should.Equal(any.ValueType(), BoolValue)
	should.Equal(any.LastError(), nil)
	i = false
	should.Equal(i, any.GetInterface())

	any = Wrap(nil)
	should.Equal(any.ValueType(), NilValue)
	should.Equal(any.LastError(), nil)
	i = nil
	should.Equal(i, any.GetInterface())

	stream := NewStream(ConfigDefault, nil, 32)
	any.WriteTo(stream)
	should.Equal("null", string(stream.Buffer()))
	should.Equal(any.LastError(), nil)

	any = Wrap(struct{ age int }{age: 1})
	should.Equal(any.ValueType(), ObjectValue)
	should.Equal(any.LastError(), nil)
	i = struct{ age int }{age: 1}
	should.Equal(i, any.GetInterface())

	any = Wrap(map[string]interface{}{"abc": 1})
	should.Equal(any.ValueType(), ObjectValue)
	should.Equal(any.LastError(), nil)
	i = map[string]interface{}{"abc": 1}
	should.Equal(i, any.GetInterface())

	any = Wrap("abc")
	i = "abc"
	should.Equal(i, any.GetInterface())
	should.Equal(nil, any.LastError())

}
