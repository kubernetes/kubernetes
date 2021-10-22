// +build go1.8

package misc_tests

import (
	"bytes"
	"encoding/json"
	"io/ioutil"
	"strconv"
	"testing"

	"github.com/json-iterator/go"
	"github.com/stretchr/testify/require"
)

func Test_read_uint64_invalid(t *testing.T) {
	should := require.New(t)
	iter := jsoniter.ParseString(jsoniter.ConfigDefault, ",")
	iter.ReadUint64()
	should.NotNil(iter.Error)
}

func Test_read_int32_array(t *testing.T) {
	should := require.New(t)
	input := `[123,456,789]`
	val := make([]int32, 0)
	jsoniter.UnmarshalFromString(input, &val)
	should.Equal(3, len(val))
}

func Test_read_int64_array(t *testing.T) {
	should := require.New(t)
	input := `[123,456,789]`
	val := make([]int64, 0)
	jsoniter.UnmarshalFromString(input, &val)
	should.Equal(3, len(val))
}

func Test_wrap_int(t *testing.T) {
	should := require.New(t)
	str, err := jsoniter.MarshalToString(jsoniter.WrapInt64(100))
	should.Nil(err)
	should.Equal("100", str)
}

func Test_write_val_int(t *testing.T) {
	should := require.New(t)
	buf := &bytes.Buffer{}
	stream := jsoniter.NewStream(jsoniter.ConfigDefault, buf, 4096)
	stream.WriteVal(1001)
	stream.Flush()
	should.Nil(stream.Error)
	should.Equal("1001", buf.String())
}

func Test_write_val_int_ptr(t *testing.T) {
	should := require.New(t)
	buf := &bytes.Buffer{}
	stream := jsoniter.NewStream(jsoniter.ConfigDefault, buf, 4096)
	val := 1001
	stream.WriteVal(&val)
	stream.Flush()
	should.Nil(stream.Error)
	should.Equal("1001", buf.String())
}

func Test_float_as_int(t *testing.T) {
	should := require.New(t)
	var i int
	should.NotNil(jsoniter.Unmarshal([]byte(`1.1`), &i))
}

func Benchmark_jsoniter_encode_int(b *testing.B) {
	stream := jsoniter.NewStream(jsoniter.ConfigDefault, ioutil.Discard, 64)
	for n := 0; n < b.N; n++ {
		stream.Reset(nil)
		stream.WriteUint64(0xffffffff)
	}
}

func Benchmark_itoa(b *testing.B) {
	for n := 0; n < b.N; n++ {
		strconv.FormatInt(0xffffffff, 10)
	}
}

func Benchmark_jsoniter_int(b *testing.B) {
	iter := jsoniter.NewIterator(jsoniter.ConfigDefault)
	input := []byte(`100`)
	for n := 0; n < b.N; n++ {
		iter.ResetBytes(input)
		iter.ReadInt64()
	}
}

func Benchmark_json_int(b *testing.B) {
	for n := 0; n < b.N; n++ {
		result := int64(0)
		json.Unmarshal([]byte(`-100`), &result)
	}
}
