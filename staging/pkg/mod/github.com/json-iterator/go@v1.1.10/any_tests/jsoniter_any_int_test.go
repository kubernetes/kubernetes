package any_tests

import (
	"fmt"
	"testing"

	"github.com/json-iterator/go"
	"github.com/stretchr/testify/require"
)

var intConvertMap = map[string]int{
	"null":       0,
	"321.1":      321,
	"-321.1":     -321,
	`"1.1"`:      1,
	`"-321.1"`:   -321,
	"0.0":        0,
	"0":          0,
	`"0"`:        0,
	`"0.0"`:      0,
	"-1.1":       -1,
	"true":       1,
	"false":      0,
	`"true"`:     0,
	`"false"`:    0,
	`"true123"`:  0,
	`"123true"`:  123,
	`"-123true"`: -123,
	`"1.2332e6"`: 1,
	`""`:         0,
	"+":          0,
	"-":          0,
	"[]":         0,
	"[1,2]":      1,
	`["1","2"]`:  1,
	// object in php cannot convert to int
	"{}": 0,
}

func Test_read_any_to_int(t *testing.T) {
	should := require.New(t)

	// int
	for k, v := range intConvertMap {
		any := jsoniter.Get([]byte(k))
		should.Equal(v, any.ToInt(), fmt.Sprintf("origin val %v", k))
	}

	// int32
	for k, v := range intConvertMap {
		any := jsoniter.Get([]byte(k))
		should.Equal(int32(v), any.ToInt32(), fmt.Sprintf("original val is %v", k))
	}

	// int64
	for k, v := range intConvertMap {
		any := jsoniter.Get([]byte(k))
		should.Equal(int64(v), any.ToInt64(), fmt.Sprintf("original val is %v", k))
	}

}

var uintConvertMap = map[string]int{
	"null":       0,
	"321.1":      321,
	`"1.1"`:      1,
	`"-123.1"`:   0,
	"0.0":        0,
	"0":          0,
	`"0"`:        0,
	`"0.0"`:      0,
	`"00.0"`:     0,
	"true":       1,
	"false":      0,
	`"true"`:     0,
	`"false"`:    0,
	`"true123"`:  0,
	`"+1"`:       1,
	`"123true"`:  123,
	`"-123true"`: 0,
	`"1.2332e6"`: 1,
	`""`:         0,
	"+":          0,
	"-":          0,
	".":          0,
	"[]":         0,
	"[1,2]":      1,
	"{}":         0,
	"{1,2}":      0,
	"-1.1":       0,
	"-321.1":     0,
}

func Test_read_any_to_uint(t *testing.T) {
	should := require.New(t)

	for k, v := range uintConvertMap {
		any := jsoniter.Get([]byte(k))
		should.Equal(uint64(v), any.ToUint64(), fmt.Sprintf("origin val %v", k))
	}

	for k, v := range uintConvertMap {
		any := jsoniter.Get([]byte(k))
		should.Equal(uint32(v), any.ToUint32(), fmt.Sprintf("origin val %v", k))
	}

	for k, v := range uintConvertMap {
		any := jsoniter.Get([]byte(k))
		should.Equal(uint(v), any.ToUint(), fmt.Sprintf("origin val %v", k))
	}

}

func Test_read_int64_to_any(t *testing.T) {
	should := require.New(t)
	any := jsoniter.WrapInt64(12345)
	should.Equal(12345, any.ToInt())
	should.Equal(int32(12345), any.ToInt32())
	should.Equal(int64(12345), any.ToInt64())
	should.Equal(uint(12345), any.ToUint())
	should.Equal(uint32(12345), any.ToUint32())
	should.Equal(uint64(12345), any.ToUint64())
	should.Equal(float32(12345), any.ToFloat32())
	should.Equal(float64(12345), any.ToFloat64())
	should.Equal("12345", any.ToString())
	should.Equal(true, any.ToBool())
	should.Equal(any.ValueType(), jsoniter.NumberValue)
	stream := jsoniter.NewStream(jsoniter.ConfigDefault, nil, 32)
	any.WriteTo(stream)
	should.Equal("12345", string(stream.Buffer()))
}
func Test_read_int32_to_any(t *testing.T) {
	should := require.New(t)
	any := jsoniter.WrapInt32(12345)
	should.Equal(12345, any.ToInt())
	should.Equal(int32(12345), any.ToInt32())
	should.Equal(int64(12345), any.ToInt64())
	should.Equal(uint(12345), any.ToUint())
	should.Equal(uint32(12345), any.ToUint32())
	should.Equal(uint64(12345), any.ToUint64())
	should.Equal(float32(12345), any.ToFloat32())
	should.Equal(float64(12345), any.ToFloat64())
	should.Equal("12345", any.ToString())
	should.Equal(true, any.ToBool())
	should.Equal(any.ValueType(), jsoniter.NumberValue)
	stream := jsoniter.NewStream(jsoniter.ConfigDefault, nil, 32)
	any.WriteTo(stream)
	should.Equal("12345", string(stream.Buffer()))
}

func Test_read_uint32_to_any(t *testing.T) {
	should := require.New(t)
	any := jsoniter.WrapUint32(12345)
	should.Equal(12345, any.ToInt())
	should.Equal(int32(12345), any.ToInt32())
	should.Equal(int64(12345), any.ToInt64())
	should.Equal(uint(12345), any.ToUint())
	should.Equal(uint32(12345), any.ToUint32())
	should.Equal(uint64(12345), any.ToUint64())
	should.Equal(float32(12345), any.ToFloat32())
	should.Equal(float64(12345), any.ToFloat64())
	should.Equal("12345", any.ToString())
	should.Equal(true, any.ToBool())
	should.Equal(any.ValueType(), jsoniter.NumberValue)
	stream := jsoniter.NewStream(jsoniter.ConfigDefault, nil, 32)
	any.WriteTo(stream)
	should.Equal("12345", string(stream.Buffer()))
}

func Test_read_uint64_to_any(t *testing.T) {
	should := require.New(t)
	any := jsoniter.WrapUint64(12345)
	should.Equal(12345, any.ToInt())
	should.Equal(int32(12345), any.ToInt32())
	should.Equal(int64(12345), any.ToInt64())
	should.Equal(uint(12345), any.ToUint())
	should.Equal(uint32(12345), any.ToUint32())
	should.Equal(uint64(12345), any.ToUint64())
	should.Equal(float32(12345), any.ToFloat32())
	should.Equal(float64(12345), any.ToFloat64())
	should.Equal("12345", any.ToString())
	should.Equal(true, any.ToBool())
	should.Equal(any.ValueType(), jsoniter.NumberValue)
	stream := jsoniter.NewStream(jsoniter.ConfigDefault, nil, 32)
	any.WriteTo(stream)
	should.Equal("12345", string(stream.Buffer()))
	stream = jsoniter.NewStream(jsoniter.ConfigDefault, nil, 32)
	stream.WriteUint(uint(123))
	should.Equal("123", string(stream.Buffer()))
}

func Test_int_lazy_any_get(t *testing.T) {
	should := require.New(t)
	any := jsoniter.Get([]byte("1234"))
	// panic!!
	//should.Equal(any.LastError(), io.EOF)
	should.Equal(jsoniter.InvalidValue, any.Get(1, "2").ValueType())
}
