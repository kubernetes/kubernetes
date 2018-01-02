package jsoniter

import (
	"testing"

	"github.com/stretchr/testify/require"
)

// if must be valid is useless, just drop this test
func Test_must_be_valid(t *testing.T) {
	should := require.New(t)
	any := Get([]byte("123"))
	should.Equal(any.MustBeValid().ToInt(), 123)

	any = Wrap(int8(10))
	should.Equal(any.MustBeValid().ToInt(), 10)

	any = Wrap(int16(10))
	should.Equal(any.MustBeValid().ToInt(), 10)

	any = Wrap(int32(10))
	should.Equal(any.MustBeValid().ToInt(), 10)

	any = Wrap(int64(10))
	should.Equal(any.MustBeValid().ToInt(), 10)

	any = Wrap(uint(10))
	should.Equal(any.MustBeValid().ToInt(), 10)

	any = Wrap(uint8(10))
	should.Equal(any.MustBeValid().ToInt(), 10)

	any = Wrap(uint16(10))
	should.Equal(any.MustBeValid().ToInt(), 10)

	any = Wrap(uint32(10))
	should.Equal(any.MustBeValid().ToInt(), 10)

	any = Wrap(uint64(10))
	should.Equal(any.MustBeValid().ToInt(), 10)

	any = Wrap(float32(10))
	should.Equal(any.MustBeValid().ToFloat64(), float64(10))

	any = Wrap(float64(10))
	should.Equal(any.MustBeValid().ToFloat64(), float64(10))

	any = Wrap(true)
	should.Equal(any.MustBeValid().ToFloat64(), float64(1))

	any = Wrap(false)
	should.Equal(any.MustBeValid().ToFloat64(), float64(0))

	any = Wrap(nil)
	should.Equal(any.MustBeValid().ToFloat64(), float64(0))

	any = Wrap(struct{ age int }{age: 1})
	should.Equal(any.MustBeValid().ToFloat64(), float64(0))

	any = Wrap(map[string]interface{}{"abc": 1})
	should.Equal(any.MustBeValid().ToFloat64(), float64(0))

	any = Wrap("abc")
	should.Equal(any.MustBeValid().ToFloat64(), float64(0))

	any = Wrap([]int{})
	should.Equal(any.MustBeValid().ToFloat64(), float64(0))

	any = Wrap([]int{1, 2})
	should.Equal(any.MustBeValid().ToFloat64(), float64(1))
}
