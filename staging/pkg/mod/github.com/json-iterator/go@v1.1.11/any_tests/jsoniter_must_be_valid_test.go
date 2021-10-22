package any_tests

import (
	"testing"

	"github.com/json-iterator/go"
	"github.com/stretchr/testify/require"
)

// if must be valid is useless, just drop this test
func Test_must_be_valid(t *testing.T) {
	should := require.New(t)
	any := jsoniter.Get([]byte("123"))
	should.Equal(any.MustBeValid().ToInt(), 123)

	any = jsoniter.Wrap(int8(10))
	should.Equal(any.MustBeValid().ToInt(), 10)

	any = jsoniter.Wrap(int16(10))
	should.Equal(any.MustBeValid().ToInt(), 10)

	any = jsoniter.Wrap(int32(10))
	should.Equal(any.MustBeValid().ToInt(), 10)

	any = jsoniter.Wrap(int64(10))
	should.Equal(any.MustBeValid().ToInt(), 10)

	any = jsoniter.Wrap(uint(10))
	should.Equal(any.MustBeValid().ToInt(), 10)

	any = jsoniter.Wrap(uint8(10))
	should.Equal(any.MustBeValid().ToInt(), 10)

	any = jsoniter.Wrap(uint16(10))
	should.Equal(any.MustBeValid().ToInt(), 10)

	any = jsoniter.Wrap(uint32(10))
	should.Equal(any.MustBeValid().ToInt(), 10)

	any = jsoniter.Wrap(uint64(10))
	should.Equal(any.MustBeValid().ToInt(), 10)

	any = jsoniter.Wrap(float32(10))
	should.Equal(any.MustBeValid().ToFloat64(), float64(10))

	any = jsoniter.Wrap(float64(10))
	should.Equal(any.MustBeValid().ToFloat64(), float64(10))

	any = jsoniter.Wrap(true)
	should.Equal(any.MustBeValid().ToFloat64(), float64(1))

	any = jsoniter.Wrap(false)
	should.Equal(any.MustBeValid().ToFloat64(), float64(0))

	any = jsoniter.Wrap(nil)
	should.Equal(any.MustBeValid().ToFloat64(), float64(0))

	any = jsoniter.Wrap(struct{ age int }{age: 1})
	should.Equal(any.MustBeValid().ToFloat64(), float64(0))

	any = jsoniter.Wrap(map[string]interface{}{"abc": 1})
	should.Equal(any.MustBeValid().ToFloat64(), float64(0))

	any = jsoniter.Wrap("abc")
	should.Equal(any.MustBeValid().ToFloat64(), float64(0))

	any = jsoniter.Wrap([]int{})
	should.Equal(any.MustBeValid().ToFloat64(), float64(0))

	any = jsoniter.Wrap([]int{1, 2})
	should.Equal(any.MustBeValid().ToFloat64(), float64(1))
}
