package jsoniter

import (
	"testing"

	"github.com/stretchr/testify/require"
)

var floatConvertMap = map[string]float64{
	"null":  0,
	"true":  1,
	"false": 0,

	`"true"`:  0,
	`"false"`: 0,

	"1e1":  10,
	"1e+1": 10,
	"1e-1": .1,
	"1E1":  10,
	"1E+1": 10,
	"1E-1": .1,

	"-1e1":  -10,
	"-1e+1": -10,
	"-1e-1": -.1,
	"-1E1":  -10,
	"-1E+1": -10,
	"-1E-1": -.1,

	`"1e1"`:  10,
	`"1e+1"`: 10,
	`"1e-1"`: .1,
	`"1E1"`:  10,
	`"1E+1"`: 10,
	`"1E-1"`: .1,

	`"-1e1"`:  -10,
	`"-1e+1"`: -10,
	`"-1e-1"`: -.1,
	`"-1E1"`:  -10,
	`"-1E+1"`: -10,
	`"-1E-1"`: -.1,

	"123":       123,
	`"123true"`: 123,
	`"+"`:       0,
	`"-"`:       0,

	`"-123true"`:  -123,
	`"-99.9true"`: -99.9,
	"0":           0,
	`"0"`:         0,
	"-1":          -1,

	"1.1":       1.1,
	"0.0":       0,
	"-1.1":      -1.1,
	`"+1.1"`:    1.1,
	`""`:        0,
	"[1,2]":     1,
	"[]":        0,
	"{}":        0,
	`{"abc":1}`: 0,
}

func Test_read_any_to_float(t *testing.T) {
	should := require.New(t)
	for k, v := range floatConvertMap {
		any := Get([]byte(k))
		should.Equal(float64(v), any.ToFloat64(), "the original val is "+k)
	}

	for k, v := range floatConvertMap {
		any := Get([]byte(k))
		should.Equal(float32(v), any.ToFloat32(), "the original val is "+k)
	}
}

func Test_read_float_to_any(t *testing.T) {
	should := require.New(t)
	any := WrapFloat64(12.3)
	anyFloat64 := float64(12.3)
	//negaAnyFloat64 := float64(-1.1)
	any2 := WrapFloat64(-1.1)
	should.Equal(float64(12.3), any.ToFloat64())
	//should.Equal("12.3", any.ToString())
	should.True(any.ToBool())
	should.Equal(float32(anyFloat64), any.ToFloat32())
	should.Equal(int(anyFloat64), any.ToInt())
	should.Equal(int32(anyFloat64), any.ToInt32())
	should.Equal(int64(anyFloat64), any.ToInt64())
	should.Equal(uint(anyFloat64), any.ToUint())
	should.Equal(uint32(anyFloat64), any.ToUint32())
	should.Equal(uint64(anyFloat64), any.ToUint64())
	should.Equal(uint(0), any2.ToUint())
	should.Equal(uint32(0), any2.ToUint32())
	should.Equal(uint64(0), any2.ToUint64())
	should.Equal(any.ValueType(), NumberValue)

	should.Equal("1.23E+01", any.ToString())
}
