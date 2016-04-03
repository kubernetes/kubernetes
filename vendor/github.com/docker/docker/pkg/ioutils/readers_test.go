package ioutils

import (
	"bytes"
	"fmt"
	"io"
	"io/ioutil"
	"strings"
	"testing"
)

// Implement io.Reader
type errorReader struct{}

func (r *errorReader) Read(p []byte) (int, error) {
	return 0, fmt.Errorf("Error reader always fail.")
}

func TestReadCloserWrapperClose(t *testing.T) {
	reader := strings.NewReader("A string reader")
	wrapper := NewReadCloserWrapper(reader, func() error {
		return fmt.Errorf("This will be called when closing")
	})
	err := wrapper.Close()
	if err == nil || !strings.Contains(err.Error(), "This will be called when closing") {
		t.Fatalf("readCloserWrapper should have call the anonymous func and thus, fail.")
	}
}

func TestReaderErrWrapperReadOnError(t *testing.T) {
	called := false
	reader := &errorReader{}
	wrapper := NewReaderErrWrapper(reader, func() {
		called = true
	})
	_, err := wrapper.Read([]byte{})
	if err == nil || !strings.Contains(err.Error(), "Error reader always fail.") {
		t.Fatalf("readErrWrapper should returned an error")
	}
	if !called {
		t.Fatalf("readErrWrapper should have call the anonymous function on failure")
	}
}

func TestReaderErrWrapperRead(t *testing.T) {
	reader := strings.NewReader("a string reader.")
	wrapper := NewReaderErrWrapper(reader, func() {
		t.Fatalf("readErrWrapper should not have called the anonymous function")
	})
	// Read 20 byte (should be ok with the string above)
	num, err := wrapper.Read(make([]byte, 20))
	if err != nil {
		t.Fatal(err)
	}
	if num != 16 {
		t.Fatalf("readerErrWrapper should have read 16 byte, but read %d", num)
	}
}

func TestNewBufReaderWithDrainbufAndBuffer(t *testing.T) {
	reader, writer := io.Pipe()

	drainBuffer := make([]byte, 1024)
	buffer := bytes.Buffer{}
	bufreader := NewBufReaderWithDrainbufAndBuffer(reader, drainBuffer, &buffer)

	// Write everything down to a Pipe
	// Usually, a pipe should block but because of the buffered reader,
	// the writes will go through
	done := make(chan bool)
	go func() {
		writer.Write([]byte("hello world"))
		writer.Close()
		done <- true
	}()

	// Drain the reader *after* everything has been written, just to verify
	// it is indeed buffering
	<-done

	output, err := ioutil.ReadAll(bufreader)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(output, []byte("hello world")) {
		t.Error(string(output))
	}
}

func TestBufReader(t *testing.T) {
	reader, writer := io.Pipe()
	bufreader := NewBufReader(reader)

	// Write everything down to a Pipe
	// Usually, a pipe should block but because of the buffered reader,
	// the writes will go through
	done := make(chan bool)
	go func() {
		writer.Write([]byte("hello world"))
		writer.Close()
		done <- true
	}()

	// Drain the reader *after* everything has been written, just to verify
	// it is indeed buffering
	<-done
	output, err := ioutil.ReadAll(bufreader)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(output, []byte("hello world")) {
		t.Error(string(output))
	}
}

func TestBufReaderCloseWithNonReaderCloser(t *testing.T) {
	reader := strings.NewReader("buffer")
	bufreader := NewBufReader(reader)

	if err := bufreader.Close(); err != nil {
		t.Fatal(err)
	}

}

// implements io.ReadCloser
type simpleReaderCloser struct{}

func (r *simpleReaderCloser) Read(p []byte) (n int, err error) {
	return 0, nil
}

func (r *simpleReaderCloser) Close() error {
	return nil
}

func TestBufReaderCloseWithReaderCloser(t *testing.T) {
	reader := &simpleReaderCloser{}
	bufreader := NewBufReader(reader)

	err := bufreader.Close()
	if err != nil {
		t.Fatal(err)
	}

}

func TestHashData(t *testing.T) {
	reader := strings.NewReader("hash-me")
	actual, err := HashData(reader)
	if err != nil {
		t.Fatal(err)
	}
	expected := "sha256:4d11186aed035cc624d553e10db358492c84a7cd6b9670d92123c144930450aa"
	if actual != expected {
		t.Fatalf("Expecting %s, got %s", expected, actual)
	}
}

type repeatedReader struct {
	readCount int
	maxReads  int
	data      []byte
}

func newRepeatedReader(max int, data []byte) *repeatedReader {
	return &repeatedReader{0, max, data}
}

func (r *repeatedReader) Read(p []byte) (int, error) {
	if r.readCount >= r.maxReads {
		return 0, io.EOF
	}
	r.readCount++
	n := copy(p, r.data)
	return n, nil
}

func testWithData(data []byte, reads int) {
	reader := newRepeatedReader(reads, data)
	bufReader := NewBufReader(reader)
	io.Copy(ioutil.Discard, bufReader)
}

func Benchmark1M10BytesReads(b *testing.B) {
	reads := 1000000
	readSize := int64(10)
	data := make([]byte, readSize)
	b.SetBytes(readSize * int64(reads))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		testWithData(data, reads)
	}
}

func Benchmark1M1024BytesReads(b *testing.B) {
	reads := 1000000
	readSize := int64(1024)
	data := make([]byte, readSize)
	b.SetBytes(readSize * int64(reads))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		testWithData(data, reads)
	}
}

func Benchmark10k32KBytesReads(b *testing.B) {
	reads := 10000
	readSize := int64(32 * 1024)
	data := make([]byte, readSize)
	b.SetBytes(readSize * int64(reads))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		testWithData(data, reads)
	}
}
