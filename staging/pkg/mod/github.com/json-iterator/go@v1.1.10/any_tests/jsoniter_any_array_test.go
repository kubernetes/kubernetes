package any_tests

import (
	"testing"

	"github.com/json-iterator/go"
	"github.com/stretchr/testify/require"
)

func Test_read_empty_array_as_any(t *testing.T) {
	should := require.New(t)
	any := jsoniter.Get([]byte("[]"))
	should.Equal(jsoniter.ArrayValue, any.Get().ValueType())
	should.Equal(jsoniter.InvalidValue, any.Get(0.3).ValueType())
	should.Equal(0, any.Size())
	should.Equal(jsoniter.ArrayValue, any.ValueType())
	should.Nil(any.LastError())
	should.Equal(0, any.ToInt())
	should.Equal(int32(0), any.ToInt32())
	should.Equal(int64(0), any.ToInt64())
	should.Equal(uint(0), any.ToUint())
	should.Equal(uint32(0), any.ToUint32())
	should.Equal(uint64(0), any.ToUint64())
	should.Equal(float32(0), any.ToFloat32())
	should.Equal(float64(0), any.ToFloat64())
}

func Test_read_one_element_array_as_any(t *testing.T) {
	should := require.New(t)
	any := jsoniter.Get([]byte("[1]"))
	should.Equal(1, any.Size())
}

func Test_read_two_element_array_as_any(t *testing.T) {
	should := require.New(t)
	any := jsoniter.Get([]byte("[1,2]"))
	should.Equal(1, any.Get(0).ToInt())
	should.Equal(2, any.Size())
	should.True(any.ToBool())
	should.Equal(1, any.ToInt())
	should.Equal([]interface{}{float64(1), float64(2)}, any.GetInterface())
	stream := jsoniter.NewStream(jsoniter.ConfigDefault, nil, 32)
	any.WriteTo(stream)
	should.Equal("[1,2]", string(stream.Buffer()))
	arr := []int{}
	any.ToVal(&arr)
	should.Equal([]int{1, 2}, arr)
}

func Test_wrap_array_and_convert_to_any(t *testing.T) {
	should := require.New(t)
	any := jsoniter.Wrap([]int{1, 2, 3})
	any2 := jsoniter.Wrap([]int{})

	should.Equal("[1,2,3]", any.ToString())
	should.True(any.ToBool())
	should.False(any2.ToBool())

	should.Equal(1, any.ToInt())
	should.Equal(0, any2.ToInt())
	should.Equal(int32(1), any.ToInt32())
	should.Equal(int32(0), any2.ToInt32())
	should.Equal(int64(1), any.ToInt64())
	should.Equal(int64(0), any2.ToInt64())
	should.Equal(uint(1), any.ToUint())
	should.Equal(uint(0), any2.ToUint())
	should.Equal(uint32(1), any.ToUint32())
	should.Equal(uint32(0), any2.ToUint32())
	should.Equal(uint64(1), any.ToUint64())
	should.Equal(uint64(0), any2.ToUint64())
	should.Equal(float32(1), any.ToFloat32())
	should.Equal(float32(0), any2.ToFloat32())
	should.Equal(float64(1), any.ToFloat64())
	should.Equal(float64(0), any2.ToFloat64())
	should.Equal(3, any.Size())
	should.Equal(0, any2.Size())

	var i interface{} = []int{1, 2, 3}
	should.Equal(i, any.GetInterface())
}

func Test_array_lazy_any_get(t *testing.T) {
	should := require.New(t)
	any := jsoniter.Get([]byte("[1,[2,3],4]"))
	should.Equal(3, any.Get(1, 1).ToInt())
	should.Equal("[1,[2,3],4]", any.ToString())
}

func Test_array_lazy_any_get_all(t *testing.T) {
	should := require.New(t)
	any := jsoniter.Get([]byte("[[1],[2],[3,4]]"))
	should.Equal("[1,2,3]", any.Get('*', 0).ToString())
	any = jsoniter.Get([]byte("[[[1],[2],[3,4]]]"), 0, '*', 0)
	should.Equal("[1,2,3]", any.ToString())
}

func Test_array_wrapper_any_get_all(t *testing.T) {
	should := require.New(t)
	any := jsoniter.Wrap([][]int{
		{1, 2},
		{3, 4},
		{5, 6},
	})
	should.Equal("[1,3,5]", any.Get('*', 0).ToString())
	should.Equal(jsoniter.ArrayValue, any.ValueType())
	should.True(any.ToBool())
	should.Equal(1, any.Get(0, 0).ToInt())
}

func Test_array_lazy_any_get_invalid(t *testing.T) {
	should := require.New(t)
	any := jsoniter.Get([]byte("[]"))
	should.Equal(jsoniter.InvalidValue, any.Get(1, 1).ValueType())
	should.NotNil(any.Get(1, 1).LastError())
	should.Equal(jsoniter.InvalidValue, any.Get("1").ValueType())
	should.NotNil(any.Get("1").LastError())
}

func Test_invalid_array(t *testing.T) {
	should := require.New(t)
	any := jsoniter.Get([]byte("["), 0)
	should.Equal(jsoniter.InvalidValue, any.ValueType())
}
