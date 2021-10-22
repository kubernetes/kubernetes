// +build go1.8

package misc_tests

import (
	"bytes"
	"encoding/json"
	"io"
	"io/ioutil"
	"math/rand"
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

// chunkedData is io.Reader which returns random amount of data in range [1, chunkedData.chunkSize].
// It simulates chunked data on from HTTP server, which is commonly used by net/http package.
type chunkedData struct {
	chunkSize int
	data      []byte
	head      int
}

// Read is implementation of the io.Reader which returns random amount of data in range [1, chunkedData.chunkSize].
func (c *chunkedData) Read(p []byte) (n int, err error) {
	to := c.head + int(rand.Int31n(int32(c.chunkSize))+1)

	// copy does not copy more data then p can consume
	n = copy(p, c.data[c.head:to])
	c.head = c.head + n
	if c.head >= len(c.data) {
		err = io.EOF
	}
	return n, err
}

// TestIterator_ReadInt_chunkedInput validates the behaviour of Iterator.ReadInt() method in where:
// - it reads data from io.Reader,
// - expected value is 0 (zero)
// - Iterator.tail == Iterator.head
// - Iterator.tail < len(Iterator.buf)
// - value in buffer after Iterator.tail is presented from previous read and has '.' character.
func TestIterator_ReadInt_chunkedInput(t *testing.T) {
	should := require.New(t)

	data := &chunkedData{
		data: jsonFloatIntArray(t, 10),
	}

	// because this test is rely on randomness of chunkedData, we are doing multiple iterations to
	// be sure, that we can hit a required case.
	for data.chunkSize = 3; data.chunkSize <= len(data.data); data.chunkSize++ {
		data.head = 0

		iter := jsoniter.Parse(jsoniter.ConfigDefault, data, data.chunkSize)
		i := 0
		for iter.ReadArray() {
			// every even item is float, let's just skip it.
			if i%2 == 0 {
				iter.Skip()
				i++
				continue
			}

			should.Zero(iter.ReadInt())
			should.NoError(iter.Error)

			i++
		}
	}
}

// jsonFloatIntArray generates JSON array where every
//  - even item is float 0.1
//  - odd item is integer 0
//
//  [0.1, 0, 0.1, 0]
func jsonFloatIntArray(t *testing.T, numberOfItems int) []byte {
	t.Helper()
	numbers := make([]jsoniter.Any, numberOfItems)
	for i := range numbers {
		switch i % 2 {
		case 0:
			numbers[i] = jsoniter.WrapFloat64(0.1)
		default:
			numbers[i] = jsoniter.WrapInt64(0)
		}
	}

	fixture, err := jsoniter.ConfigFastest.Marshal(numbers)
	if err != nil {
		panic(err)
	}

	b := &bytes.Buffer{}

	require.NoError(
		t,
		json.Compact(b, fixture),
		"json should be compactable",
	)
	return b.Bytes()
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
