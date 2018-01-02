package jsoniter

import (
	"github.com/stretchr/testify/require"
	"testing"
)

func Test_writeByte_should_grow_buffer(t *testing.T) {
	should := require.New(t)
	stream := NewStream(ConfigDefault, nil, 1)
	stream.writeByte('1')
	should.Equal("1", string(stream.Buffer()))
	should.Equal(1, len(stream.buf))
	stream.writeByte('2')
	should.Equal("12", string(stream.Buffer()))
	should.Equal(2, len(stream.buf))
	stream.writeThreeBytes('3', '4', '5')
	should.Equal("12345", string(stream.Buffer()))
}

func Test_writeBytes_should_grow_buffer(t *testing.T) {
	should := require.New(t)
	stream := NewStream(ConfigDefault, nil, 1)
	stream.Write([]byte{'1', '2'})
	should.Equal("12", string(stream.Buffer()))
	should.Equal(3, len(stream.buf))
	stream.Write([]byte{'3', '4', '5', '6', '7'})
	should.Equal("1234567", string(stream.Buffer()))
	should.Equal(8, len(stream.buf))
}

func Test_writeIndention_should_grow_buffer(t *testing.T) {
	should := require.New(t)
	stream := NewStream(Config{IndentionStep: 2}.Froze(), nil, 1)
	stream.WriteVal([]int{1, 2, 3})
	should.Equal("[\n  1,\n  2,\n  3\n]", string(stream.Buffer()))
}

func Test_writeRaw_should_grow_buffer(t *testing.T) {
	should := require.New(t)
	stream := NewStream(ConfigDefault, nil, 1)
	stream.WriteRaw("123")
	should.Nil(stream.Error)
	should.Equal("123", string(stream.Buffer()))
}

func Test_writeString_should_grow_buffer(t *testing.T) {
	should := require.New(t)
	stream := NewStream(ConfigDefault, nil, 0)
	stream.WriteString("123")
	should.Nil(stream.Error)
	should.Equal(`"123"`, string(stream.Buffer()))
}

type NopWriter struct {
	bufferSize int
}

func (w *NopWriter) Write(p []byte) (n int, err error) {
	w.bufferSize = cap(p)
	return len(p), nil
}

func Test_flush_buffer_should_stop_grow_buffer(t *testing.T) {
	writer := new(NopWriter)
	NewEncoder(writer).Encode(make([]int, 10000000))
	should := require.New(t)
	should.Equal(512, writer.bufferSize)
}
