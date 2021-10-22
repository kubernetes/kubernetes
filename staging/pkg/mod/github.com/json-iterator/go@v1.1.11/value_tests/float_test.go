package test

import (
	"bytes"
	"encoding/json"
	"fmt"
	"github.com/json-iterator/go"
	"github.com/stretchr/testify/require"
	"strconv"
	"testing"
)

func Test_read_float(t *testing.T) {
	inputs := []string{
		`1.1`, `1000`, `9223372036854775807`, `12.3`, `-12.3`, `720368.54775807`, `720368.547758075`,
		`1e1`, `1e+1`, `1e-1`, `1E1`, `1E+1`, `1E-1`, `-1e1`, `-1e+1`, `-1e-1`,
	}
	for _, input := range inputs {
		// non-streaming
		t.Run(fmt.Sprintf("%v", input), func(t *testing.T) {
			should := require.New(t)
			iter := jsoniter.ParseString(jsoniter.ConfigDefault, input+",")
			expected, err := strconv.ParseFloat(input, 32)
			should.Nil(err)
			should.Equal(float32(expected), iter.ReadFloat32())
		})
		t.Run(fmt.Sprintf("%v", input), func(t *testing.T) {
			should := require.New(t)
			iter := jsoniter.ParseString(jsoniter.ConfigDefault, input+",")
			expected, err := strconv.ParseFloat(input, 64)
			should.Nil(err)
			should.Equal(expected, iter.ReadFloat64())
		})
		// streaming
		t.Run(fmt.Sprintf("%v", input), func(t *testing.T) {
			should := require.New(t)
			iter := jsoniter.Parse(jsoniter.ConfigDefault, bytes.NewBufferString(input+","), 2)
			expected, err := strconv.ParseFloat(input, 32)
			should.Nil(err)
			should.Equal(float32(expected), iter.ReadFloat32())
		})
		t.Run(fmt.Sprintf("%v", input), func(t *testing.T) {
			should := require.New(t)
			iter := jsoniter.Parse(jsoniter.ConfigDefault, bytes.NewBufferString(input+","), 2)
			val := float64(0)
			err := json.Unmarshal([]byte(input), &val)
			should.Nil(err)
			should.Equal(val, iter.ReadFloat64())
		})
	}
}

func Test_write_float32(t *testing.T) {
	vals := []float32{0, 1, -1, 99, 0xff, 0xfff, 0xffff, 0xfffff, 0xffffff, 0x4ffffff, 0xfffffff,
		-0x4ffffff, -0xfffffff, 1.2345, 1.23456, 1.234567, 1.001}
	for _, val := range vals {
		t.Run(fmt.Sprintf("%v", val), func(t *testing.T) {
			should := require.New(t)
			buf := &bytes.Buffer{}
			stream := jsoniter.NewStream(jsoniter.ConfigDefault, buf, 4096)
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
			stream := jsoniter.NewStream(jsoniter.ConfigDefault, buf, 4096)
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
	stream := jsoniter.NewStream(jsoniter.ConfigDefault, buf, 10)
	stream.WriteRaw("abcdefg")
	stream.WriteFloat32Lossy(1.123456)
	stream.Flush()
	should.Nil(stream.Error)
	should.Equal("abcdefg1.123456", buf.String())

	stream = jsoniter.NewStream(jsoniter.ConfigDefault, nil, 0)
	stream.WriteFloat32(float32(0.0000001))
	should.Equal("1e-07", string(stream.Buffer()))
}

func Test_write_float64(t *testing.T) {
	vals := []float64{0, 1, -1, 99, 0xff, 0xfff, 0xffff, 0xfffff, 0xffffff, 0x4ffffff, 0xfffffff,
		-0x4ffffff, -0xfffffff, 1.2345, 1.23456, 1.234567, 1.001}
	for _, val := range vals {
		t.Run(fmt.Sprintf("%v", val), func(t *testing.T) {
			should := require.New(t)
			buf := &bytes.Buffer{}
			stream := jsoniter.NewStream(jsoniter.ConfigDefault, buf, 4096)
			stream.WriteFloat64Lossy(val)
			stream.Flush()
			should.Nil(stream.Error)
			should.Equal(strconv.FormatFloat(val, 'f', -1, 64), buf.String())
		})
		t.Run(fmt.Sprintf("%v", val), func(t *testing.T) {
			should := require.New(t)
			buf := &bytes.Buffer{}
			stream := jsoniter.NewStream(jsoniter.ConfigDefault, buf, 4096)
			stream.WriteVal(val)
			stream.Flush()
			should.Nil(stream.Error)
			should.Equal(strconv.FormatFloat(val, 'f', -1, 64), buf.String())
		})
	}
	should := require.New(t)
	buf := &bytes.Buffer{}
	stream := jsoniter.NewStream(jsoniter.ConfigDefault, buf, 10)
	stream.WriteRaw("abcdefg")
	stream.WriteFloat64Lossy(1.123456)
	stream.Flush()
	should.Nil(stream.Error)
	should.Equal("abcdefg1.123456", buf.String())

	stream = jsoniter.NewStream(jsoniter.ConfigDefault, nil, 0)
	stream.WriteFloat64(float64(0.0000001))
	should.Equal("1e-07", string(stream.Buffer()))
}
