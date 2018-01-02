package ioutils

import (
	"bytes"
	"testing"
)

func TestFixedBufferCap(t *testing.T) {
	buf := &fixedBuffer{buf: make([]byte, 0, 5)}

	n := buf.Cap()
	if n != 5 {
		t.Fatalf("expected buffer capacity to be 5 bytes, got %d", n)
	}
}

func TestFixedBufferLen(t *testing.T) {
	buf := &fixedBuffer{buf: make([]byte, 0, 10)}

	buf.Write([]byte("hello"))
	l := buf.Len()
	if l != 5 {
		t.Fatalf("expected buffer length to be 5 bytes, got %d", l)
	}

	buf.Write([]byte("world"))
	l = buf.Len()
	if l != 10 {
		t.Fatalf("expected buffer length to be 10 bytes, got %d", l)
	}

	// read 5 bytes
	b := make([]byte, 5)
	buf.Read(b)

	l = buf.Len()
	if l != 5 {
		t.Fatalf("expected buffer length to be 5 bytes, got %d", l)
	}

	n, err := buf.Write([]byte("i-wont-fit"))
	if n != 0 {
		t.Fatalf("expected no bytes to be written to buffer, got %d", n)
	}
	if err != errBufferFull {
		t.Fatalf("expected errBufferFull, got %v", err)
	}

	l = buf.Len()
	if l != 5 {
		t.Fatalf("expected buffer length to still be 5 bytes, got %d", l)
	}

	buf.Reset()
	l = buf.Len()
	if l != 0 {
		t.Fatalf("expected buffer length to still be 0 bytes, got %d", l)
	}
}

func TestFixedBufferString(t *testing.T) {
	buf := &fixedBuffer{buf: make([]byte, 0, 10)}

	buf.Write([]byte("hello"))
	buf.Write([]byte("world"))

	out := buf.String()
	if out != "helloworld" {
		t.Fatalf("expected output to be \"helloworld\", got %q", out)
	}

	// read 5 bytes
	b := make([]byte, 5)
	buf.Read(b)

	// test that fixedBuffer.String() only returns the part that hasn't been read
	out = buf.String()
	if out != "world" {
		t.Fatalf("expected output to be \"world\", got %q", out)
	}
}

func TestFixedBufferWrite(t *testing.T) {
	buf := &fixedBuffer{buf: make([]byte, 0, 64)}
	n, err := buf.Write([]byte("hello"))
	if err != nil {
		t.Fatal(err)
	}

	if n != 5 {
		t.Fatalf("expected 5 bytes written, got %d", n)
	}

	if string(buf.buf[:5]) != "hello" {
		t.Fatalf("expected \"hello\", got %q", string(buf.buf[:5]))
	}

	n, err = buf.Write(bytes.Repeat([]byte{1}, 64))
	if n != 59 {
		t.Fatalf("expected 59 bytes written before buffer is full, got %d", n)
	}
	if err != errBufferFull {
		t.Fatalf("expected errBufferFull, got %v - %v", err, buf.buf[:64])
	}
}

func TestFixedBufferRead(t *testing.T) {
	buf := &fixedBuffer{buf: make([]byte, 0, 64)}
	if _, err := buf.Write([]byte("hello world")); err != nil {
		t.Fatal(err)
	}

	b := make([]byte, 5)
	n, err := buf.Read(b)
	if err != nil {
		t.Fatal(err)
	}

	if n != 5 {
		t.Fatalf("expected 5 bytes read, got %d - %s", n, buf.String())
	}

	if string(b) != "hello" {
		t.Fatalf("expected \"hello\", got %q", string(b))
	}

	n, err = buf.Read(b)
	if err != nil {
		t.Fatal(err)
	}

	if n != 5 {
		t.Fatalf("expected 5 bytes read, got %d", n)
	}

	if string(b) != " worl" {
		t.Fatalf("expected \" worl\", got %s", string(b))
	}

	b = b[:1]
	n, err = buf.Read(b)
	if err != nil {
		t.Fatal(err)
	}

	if n != 1 {
		t.Fatalf("expected 1 byte read, got %d - %s", n, buf.String())
	}

	if string(b) != "d" {
		t.Fatalf("expected \"d\", got %s", string(b))
	}
}
