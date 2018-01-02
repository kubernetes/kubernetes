// +build go1.8

package jsoniter

import (
	"bytes"
	"encoding/json"
	"fmt"
	"strconv"
	"testing"

	"github.com/stretchr/testify/require"
)

func Test_read_big_float(t *testing.T) {
	should := require.New(t)
	iter := ParseString(ConfigDefault, `12.3`)
	val := iter.ReadBigFloat()
	val64, _ := val.Float64()
	should.Equal(12.3, val64)
}

func Test_read_big_int(t *testing.T) {
	should := require.New(t)
	iter := ParseString(ConfigDefault, `92233720368547758079223372036854775807`)
	val := iter.ReadBigInt()
	should.NotNil(val)
	should.Equal(`92233720368547758079223372036854775807`, val.String())
}

func Test_read_float(t *testing.T) {
	inputs := []string{
		`1.1`, `1000`, `9223372036854775807`, `12.3`, `-12.3`, `720368.54775807`, `720368.547758075`,
		`1e1`, `1e+1`, `1e-1`, `1E1`, `1E+1`, `1E-1`, `-1e1`, `-1e+1`, `-1e-1`,
	}
	for _, input := range inputs {
		// non-streaming
		t.Run(fmt.Sprintf("%v", input), func(t *testing.T) {
			should := require.New(t)
			iter := ParseString(ConfigDefault, input+",")
			expected, err := strconv.ParseFloat(input, 32)
			should.Nil(err)
			should.Equal(float32(expected), iter.ReadFloat32())
		})
		t.Run(fmt.Sprintf("%v", input), func(t *testing.T) {
			should := require.New(t)
			iter := ParseString(ConfigDefault, input+",")
			expected, err := strconv.ParseFloat(input, 64)
			should.Nil(err)
			should.Equal(expected, iter.ReadFloat64())
		})
		// streaming
		t.Run(fmt.Sprintf("%v", input), func(t *testing.T) {
			should := require.New(t)
			iter := Parse(ConfigDefault, bytes.NewBufferString(input+","), 2)
			expected, err := strconv.ParseFloat(input, 32)
			should.Nil(err)
			should.Equal(float32(expected), iter.ReadFloat32())
		})
		t.Run(fmt.Sprintf("%v", input), func(t *testing.T) {
			should := require.New(t)
			iter := Parse(ConfigDefault, bytes.NewBufferString(input+","), 2)
			val := float64(0)
			err := json.Unmarshal([]byte(input), &val)
			should.Nil(err)
			should.Equal(val, iter.ReadFloat64())
		})
	}
}

func Test_read_float_as_interface(t *testing.T) {
	should := require.New(t)
	iter := ParseString(ConfigDefault, `12.3`)
	should.Equal(float64(12.3), iter.Read())
}

func Test_wrap_float(t *testing.T) {
	should := require.New(t)
	str, err := MarshalToString(WrapFloat64(12.3))
	should.Nil(err)
	should.Equal("12.3", str)
}

func Test_write_float32(t *testing.T) {
	vals := []float32{0, 1, -1, 99, 0xff, 0xfff, 0xffff, 0xfffff, 0xffffff, 0x4ffffff, 0xfffffff,
		-0x4ffffff, -0xfffffff, 1.2345, 1.23456, 1.234567, 1.001}
	for _, val := range vals {
		t.Run(fmt.Sprintf("%v", val), func(t *testing.T) {
			should := require.New(t)
			buf := &bytes.Buffer{}
			stream := NewStream(ConfigDefault, buf, 4096)
			stream.WriteFloat32Lossy(val)
			stream.Flush()
			should.Nil(stream.Error)
			output, err := json.Marshal(val)
			should.Nil(err)
			should.Equal(string(output), buf.String())
		})
		t.Run(fmt.Sprintf("%v", val), func(t *testing.T) {
			should := require.New(t)
			buf := &bytes.Buffer{}
			stream := NewStream(ConfigDefault, buf, 4096)
			stream.WriteVal(val)
			stream.Flush()
			should.Nil(stream.Error)
			output, err := json.Marshal(val)
			should.Nil(err)
			should.Equal(string(output), buf.String())
		})
	}
	should := require.New(t)
	buf := &bytes.Buffer{}
	stream := NewStream(ConfigDefault, buf, 10)
	stream.WriteRaw("abcdefg")
	stream.WriteFloat32Lossy(1.123456)
	stream.Flush()
	should.Nil(stream.Error)
	should.Equal("abcdefg1.123456", buf.String())

	stream = NewStream(ConfigDefault, nil, 0)
	stream.WriteFloat32(float32(0.0000001))
	should.Equal("1e-07", string(stream.buf))
}

func Test_write_float64(t *testing.T) {
	vals := []float64{0, 1, -1, 99, 0xff, 0xfff, 0xffff, 0xfffff, 0xffffff, 0x4ffffff, 0xfffffff,
		-0x4ffffff, -0xfffffff, 1.2345, 1.23456, 1.234567, 1.001}
	for _, val := range vals {
		t.Run(fmt.Sprintf("%v", val), func(t *testing.T) {
			should := require.New(t)
			buf := &bytes.Buffer{}
			stream := NewStream(ConfigDefault, buf, 4096)
			stream.WriteFloat64Lossy(val)
			stream.Flush()
			should.Nil(stream.Error)
			should.Equal(strconv.FormatFloat(val, 'f', -1, 64), buf.String())
		})
		t.Run(fmt.Sprintf("%v", val), func(t *testing.T) {
			should := require.New(t)
			buf := &bytes.Buffer{}
			stream := NewStream(ConfigDefault, buf, 4096)
			stream.WriteVal(val)
			stream.Flush()
			should.Nil(stream.Error)
			should.Equal(strconv.FormatFloat(val, 'f', -1, 64), buf.String())
		})
	}
	should := require.New(t)
	buf := &bytes.Buffer{}
	stream := NewStream(ConfigDefault, buf, 10)
	stream.WriteRaw("abcdefg")
	stream.WriteFloat64Lossy(1.123456)
	stream.Flush()
	should.Nil(stream.Error)
	should.Equal("abcdefg1.123456", buf.String())

	stream = NewStream(ConfigDefault, nil, 0)
	stream.WriteFloat64(float64(0.0000001))
	should.Equal("1e-07", string(stream.buf))
}

func Test_read_float64_cursor(t *testing.T) {
	should := require.New(t)
	iter := ParseString(ConfigDefault, "[1.23456789\n,2,3]")
	should.True(iter.ReadArray())
	should.Equal(1.23456789, iter.Read())
	should.True(iter.ReadArray())
	should.Equal(float64(2), iter.Read())
}

func Test_read_float_scientific(t *testing.T) {
	should := require.New(t)
	var obj interface{}
	should.Nil(UnmarshalFromString(`1e1`, &obj))
	should.Equal(float64(10), obj)
	should.Nil(json.Unmarshal([]byte(`1e1`), &obj))
	should.Equal(float64(10), obj)
	should.Nil(UnmarshalFromString(`1.0e1`, &obj))
	should.Equal(float64(10), obj)
	should.Nil(json.Unmarshal([]byte(`1.0e1`), &obj))
	should.Equal(float64(10), obj)
}

func Test_lossy_float_marshal(t *testing.T) {
	should := require.New(t)
	api := Config{MarshalFloatWith6Digits: true}.Froze()
	output, err := api.MarshalToString(float64(0.1234567))
	should.Nil(err)
	should.Equal("0.123457", output)
	output, err = api.MarshalToString(float32(0.1234567))
	should.Nil(err)
	should.Equal("0.123457", output)
}

func Benchmark_jsoniter_float(b *testing.B) {
	b.ReportAllocs()
	input := []byte(`1.1123,`)
	iter := NewIterator(ConfigDefault)
	for n := 0; n < b.N; n++ {
		iter.ResetBytes(input)
		iter.ReadFloat64()
	}
}

func Benchmark_json_float(b *testing.B) {
	for n := 0; n < b.N; n++ {
		result := float64(0)
		json.Unmarshal([]byte(`1.1`), &result)
	}
}
