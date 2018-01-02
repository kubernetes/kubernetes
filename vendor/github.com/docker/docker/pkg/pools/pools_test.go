package pools

import (
	"bufio"
	"bytes"
	"io"
	"strings"
	"testing"
)

func TestBufioReaderPoolGetWithNoReaderShouldCreateOne(t *testing.T) {
	reader := BufioReader32KPool.Get(nil)
	if reader == nil {
		t.Fatalf("BufioReaderPool should have create a bufio.Reader but did not.")
	}
}

func TestBufioReaderPoolPutAndGet(t *testing.T) {
	sr := bufio.NewReader(strings.NewReader("foobar"))
	reader := BufioReader32KPool.Get(sr)
	if reader == nil {
		t.Fatalf("BufioReaderPool should not return a nil reader.")
	}
	// verify the first 3 byte
	buf1 := make([]byte, 3)
	_, err := reader.Read(buf1)
	if err != nil {
		t.Fatal(err)
	}
	if actual := string(buf1); actual != "foo" {
		t.Fatalf("The first letter should have been 'foo' but was %v", actual)
	}
	BufioReader32KPool.Put(reader)
	// Try to read the next 3 bytes
	_, err = sr.Read(make([]byte, 3))
	if err == nil || err != io.EOF {
		t.Fatalf("The buffer should have been empty, issue an EOF error.")
	}
}

type simpleReaderCloser struct {
	io.Reader
	closed bool
}

func (r *simpleReaderCloser) Close() error {
	r.closed = true
	return nil
}

func TestNewReadCloserWrapperWithAReadCloser(t *testing.T) {
	br := bufio.NewReader(strings.NewReader(""))
	sr := &simpleReaderCloser{
		Reader: strings.NewReader("foobar"),
		closed: false,
	}
	reader := BufioReader32KPool.NewReadCloserWrapper(br, sr)
	if reader == nil {
		t.Fatalf("NewReadCloserWrapper should not return a nil reader.")
	}
	// Verify the content of reader
	buf := make([]byte, 3)
	_, err := reader.Read(buf)
	if err != nil {
		t.Fatal(err)
	}
	if actual := string(buf); actual != "foo" {
		t.Fatalf("The first 3 letter should have been 'foo' but were %v", actual)
	}
	reader.Close()
	// Read 3 more bytes "bar"
	_, err = reader.Read(buf)
	if err != nil {
		t.Fatal(err)
	}
	if actual := string(buf); actual != "bar" {
		t.Fatalf("The first 3 letter should have been 'bar' but were %v", actual)
	}
	if !sr.closed {
		t.Fatalf("The ReaderCloser should have been closed, it is not.")
	}
}

func TestBufioWriterPoolGetWithNoReaderShouldCreateOne(t *testing.T) {
	writer := BufioWriter32KPool.Get(nil)
	if writer == nil {
		t.Fatalf("BufioWriterPool should have create a bufio.Writer but did not.")
	}
}

func TestBufioWriterPoolPutAndGet(t *testing.T) {
	buf := new(bytes.Buffer)
	bw := bufio.NewWriter(buf)
	writer := BufioWriter32KPool.Get(bw)
	if writer == nil {
		t.Fatalf("BufioReaderPool should not return a nil writer.")
	}
	written, err := writer.Write([]byte("foobar"))
	if err != nil {
		t.Fatal(err)
	}
	if written != 6 {
		t.Fatalf("Should have written 6 bytes, but wrote %v bytes", written)
	}
	// Make sure we Flush all the way ?
	writer.Flush()
	bw.Flush()
	if len(buf.Bytes()) != 6 {
		t.Fatalf("The buffer should contain 6 bytes ('foobar') but contains %v ('%v')", buf.Bytes(), string(buf.Bytes()))
	}
	// Reset the buffer
	buf.Reset()
	BufioWriter32KPool.Put(writer)
	// Try to write something
	if _, err = writer.Write([]byte("barfoo")); err != nil {
		t.Fatal(err)
	}
	// If we now try to flush it, it should panic (the writer is nil)
	// recover it
	defer func() {
		if r := recover(); r == nil {
			t.Fatal("Trying to flush the writter should have 'paniced', did not.")
		}
	}()
	writer.Flush()
}

type simpleWriterCloser struct {
	io.Writer
	closed bool
}

func (r *simpleWriterCloser) Close() error {
	r.closed = true
	return nil
}

func TestNewWriteCloserWrapperWithAWriteCloser(t *testing.T) {
	buf := new(bytes.Buffer)
	bw := bufio.NewWriter(buf)
	sw := &simpleWriterCloser{
		Writer: new(bytes.Buffer),
		closed: false,
	}
	bw.Flush()
	writer := BufioWriter32KPool.NewWriteCloserWrapper(bw, sw)
	if writer == nil {
		t.Fatalf("BufioReaderPool should not return a nil writer.")
	}
	written, err := writer.Write([]byte("foobar"))
	if err != nil {
		t.Fatal(err)
	}
	if written != 6 {
		t.Fatalf("Should have written 6 bytes, but wrote %v bytes", written)
	}
	writer.Close()
	if !sw.closed {
		t.Fatalf("The ReaderCloser should have been closed, it is not.")
	}
}

func TestBufferPoolPutAndGet(t *testing.T) {
	buf := buffer32KPool.Get()
	buffer32KPool.Put(buf)
}
