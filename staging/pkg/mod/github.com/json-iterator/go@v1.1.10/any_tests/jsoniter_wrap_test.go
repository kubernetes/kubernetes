package any_tests

import (
	"testing"

	"github.com/json-iterator/go"
	"github.com/stretchr/testify/require"
)

func Test_wrap_and_valuetype_everything(t *testing.T) {
	should := require.New(t)
	var i interface{}
	any := jsoniter.Get([]byte("123"))
	// default of number type is float64
	i = float64(123)
	should.Equal(i, any.GetInterface())

	any = jsoniter.Wrap(int8(10))
	should.Equal(any.ValueType(), jsoniter.NumberValue)
	should.Equal(any.LastError(), nil)
	//  get interface is not int8 interface
	// i = int8(10)
	// should.Equal(i, any.GetInterface())

	any = jsoniter.Wrap(int16(10))
	should.Equal(any.ValueType(), jsoniter.NumberValue)
	should.Equal(any.LastError(), nil)
	//i = int16(10)
	//should.Equal(i, any.GetInterface())

	any = jsoniter.Wrap(int32(10))
	should.Equal(any.ValueType(), jsoniter.NumberValue)
	should.Equal(any.LastError(), nil)
	i = int32(10)
	should.Equal(i, any.GetInterface())
	any = jsoniter.Wrap(int64(10))
	should.Equal(any.ValueType(), jsoniter.NumberValue)
	should.Equal(any.LastError(), nil)
	i = int64(10)
	should.Equal(i, any.GetInterface())

	any = jsoniter.Wrap(uint(10))
	should.Equal(any.ValueType(), jsoniter.NumberValue)
	should.Equal(any.LastError(), nil)
	// not equal
	//i = uint(10)
	//should.Equal(i, any.GetInterface())
	any = jsoniter.Wrap(uint8(10))
	should.Equal(any.ValueType(), jsoniter.NumberValue)
	should.Equal(any.LastError(), nil)
	// not equal
	// i = uint8(10)
	// should.Equal(i, any.GetInterface())
	any = jsoniter.Wrap(uint16(10))
	should.Equal(any.ValueType(), jsoniter.NumberValue)
	should.Equal(any.LastError(), nil)
	any = jsoniter.Wrap(uint32(10))
	should.Equal(any.ValueType(), jsoniter.NumberValue)
	should.Equal(any.LastError(), nil)
	i = uint32(10)
	should.Equal(i, any.GetInterface())
	any = jsoniter.Wrap(uint64(10))
	should.Equal(any.ValueType(), jsoniter.NumberValue)
	should.Equal(any.LastError(), nil)
	i = uint64(10)
	should.Equal(i, any.GetInterface())

	any = jsoniter.Wrap(float32(10))
	should.Equal(any.ValueType(), jsoniter.NumberValue)
	should.Equal(any.LastError(), nil)
	// not equal
	//i = float32(10)
	//should.Equal(i, any.GetInterface())
	any = jsoniter.Wrap(float64(10))
	should.Equal(any.ValueType(), jsoniter.NumberValue)
	should.Equal(any.LastError(), nil)
	i = float64(10)
	should.Equal(i, any.GetInterface())

	any = jsoniter.Wrap(true)
	should.Equal(any.ValueType(), jsoniter.BoolValue)
	should.Equal(any.LastError(), nil)
	i = true
	should.Equal(i, any.GetInterface())
	any = jsoniter.Wrap(false)
	should.Equal(any.ValueType(), jsoniter.BoolValue)
	should.Equal(any.LastError(), nil)
	i = false
	should.Equal(i, any.GetInterface())

	any = jsoniter.Wrap(nil)
	should.Equal(any.ValueType(), jsoniter.NilValue)
	should.Equal(any.LastError(), nil)
	i = nil
	should.Equal(i, any.GetInterface())

	stream := jsoniter.NewStream(jsoniter.ConfigDefault, nil, 32)
	any.WriteTo(stream)
	should.Equal("null", string(stream.Buffer()))
	should.Equal(any.LastError(), nil)

	any = jsoniter.Wrap(struct{ age int }{age: 1})
	should.Equal(any.ValueType(), jsoniter.ObjectValue)
	should.Equal(any.LastError(), nil)
	i = struct{ age int }{age: 1}
	should.Equal(i, any.GetInterface())

	any = jsoniter.Wrap(map[string]interface{}{"abc": 1})
	should.Equal(any.ValueType(), jsoniter.ObjectValue)
	should.Equal(any.LastError(), nil)
	i = map[string]interface{}{"abc": 1}
	should.Equal(i, any.GetInterface())

	any = jsoniter.Wrap("abc")
	i = "abc"
	should.Equal(i, any.GetInterface())
	should.Equal(nil, any.LastError())

}
