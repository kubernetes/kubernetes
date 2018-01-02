package request

import (
	"bytes"
	"io"
	"math/rand"
	"sync"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
)

func TestOffsetReaderRead(t *testing.T) {
	buf := []byte("testData")
	reader := &offsetReader{buf: bytes.NewReader(buf)}

	tempBuf := make([]byte, len(buf))

	n, err := reader.Read(tempBuf)

	assert.Equal(t, n, len(buf))
	assert.Nil(t, err)
	assert.Equal(t, buf, tempBuf)
}

func TestOffsetReaderSeek(t *testing.T) {
	buf := []byte("testData")
	reader := newOffsetReader(bytes.NewReader(buf), 0)

	orig, err := reader.Seek(0, 1)
	assert.NoError(t, err)
	assert.Equal(t, int64(0), orig)

	n, err := reader.Seek(0, 2)
	assert.NoError(t, err)
	assert.Equal(t, int64(len(buf)), n)

	n, err = reader.Seek(orig, 0)
	assert.NoError(t, err)
	assert.Equal(t, int64(0), n)
}

func TestOffsetReaderClose(t *testing.T) {
	buf := []byte("testData")
	reader := &offsetReader{buf: bytes.NewReader(buf)}

	err := reader.Close()
	assert.Nil(t, err)

	tempBuf := make([]byte, len(buf))
	n, err := reader.Read(tempBuf)
	assert.Equal(t, n, 0)
	assert.Equal(t, err, io.EOF)
}

func TestOffsetReaderCloseAndCopy(t *testing.T) {
	buf := []byte("testData")
	tempBuf := make([]byte, len(buf))
	reader := &offsetReader{buf: bytes.NewReader(buf)}

	newReader := reader.CloseAndCopy(0)

	n, err := reader.Read(tempBuf)
	assert.Equal(t, n, 0)
	assert.Equal(t, err, io.EOF)

	n, err = newReader.Read(tempBuf)
	assert.Equal(t, n, len(buf))
	assert.Nil(t, err)
	assert.Equal(t, buf, tempBuf)
}

func TestOffsetReaderCloseAndCopyOffset(t *testing.T) {
	buf := []byte("testData")
	tempBuf := make([]byte, len(buf))
	reader := &offsetReader{buf: bytes.NewReader(buf)}

	newReader := reader.CloseAndCopy(4)
	n, err := newReader.Read(tempBuf)
	assert.Equal(t, n, len(buf)-4)
	assert.Nil(t, err)

	expected := []byte{'D', 'a', 't', 'a', 0, 0, 0, 0}
	assert.Equal(t, expected, tempBuf)
}

func TestOffsetReaderRace(t *testing.T) {
	wg := sync.WaitGroup{}

	f := func(reader *offsetReader) {
		defer wg.Done()
		var err error
		buf := make([]byte, 1)
		_, err = reader.Read(buf)
		for err != io.EOF {
			_, err = reader.Read(buf)
		}

	}

	closeFn := func(reader *offsetReader) {
		defer wg.Done()
		time.Sleep(time.Duration(rand.Intn(20)+1) * time.Millisecond)
		reader.Close()
	}
	for i := 0; i < 50; i++ {
		reader := &offsetReader{buf: bytes.NewReader(make([]byte, 1024*1024))}
		wg.Add(1)
		go f(reader)
		wg.Add(1)
		go closeFn(reader)
	}
	wg.Wait()
}

func BenchmarkOffsetReader(b *testing.B) {
	bufSize := 1024 * 1024 * 100
	buf := make([]byte, bufSize)
	reader := &offsetReader{buf: bytes.NewReader(buf)}

	tempBuf := make([]byte, 1024)

	for i := 0; i < b.N; i++ {
		reader.Read(tempBuf)
	}
}

func BenchmarkBytesReader(b *testing.B) {
	bufSize := 1024 * 1024 * 100
	buf := make([]byte, bufSize)
	reader := bytes.NewReader(buf)

	tempBuf := make([]byte, 1024)

	for i := 0; i < b.N; i++ {
		reader.Read(tempBuf)
	}
}
