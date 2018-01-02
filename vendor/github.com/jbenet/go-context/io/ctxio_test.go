package ctxio

import (
	"bytes"
	"io"
	"testing"
	"time"

	context "golang.org/x/net/context"
)

func TestReader(t *testing.T) {
	buf := []byte("abcdef")
	buf2 := make([]byte, 3)
	r := NewReader(context.Background(), bytes.NewReader(buf))

	// read first half
	n, err := r.Read(buf2)
	if n != 3 {
		t.Error("n should be 3")
	}
	if err != nil {
		t.Error("should have no error")
	}
	if string(buf2) != string(buf[:3]) {
		t.Error("incorrect contents")
	}

	// read second half
	n, err = r.Read(buf2)
	if n != 3 {
		t.Error("n should be 3")
	}
	if err != nil {
		t.Error("should have no error")
	}
	if string(buf2) != string(buf[3:6]) {
		t.Error("incorrect contents")
	}

	// read more.
	n, err = r.Read(buf2)
	if n != 0 {
		t.Error("n should be 0", n)
	}
	if err != io.EOF {
		t.Error("should be EOF", err)
	}
}

func TestWriter(t *testing.T) {
	var buf bytes.Buffer
	w := NewWriter(context.Background(), &buf)

	// write three
	n, err := w.Write([]byte("abc"))
	if n != 3 {
		t.Error("n should be 3")
	}
	if err != nil {
		t.Error("should have no error")
	}
	if string(buf.Bytes()) != string("abc") {
		t.Error("incorrect contents")
	}

	// write three more
	n, err = w.Write([]byte("def"))
	if n != 3 {
		t.Error("n should be 3")
	}
	if err != nil {
		t.Error("should have no error")
	}
	if string(buf.Bytes()) != string("abcdef") {
		t.Error("incorrect contents")
	}
}

func TestReaderCancel(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	piper, pipew := io.Pipe()
	r := NewReader(ctx, piper)

	buf := make([]byte, 10)
	done := make(chan ioret)

	go func() {
		n, err := r.Read(buf)
		done <- ioret{n, err}
	}()

	pipew.Write([]byte("abcdefghij"))

	select {
	case ret := <-done:
		if ret.n != 10 {
			t.Error("ret.n should be 10", ret.n)
		}
		if ret.err != nil {
			t.Error("ret.err should be nil", ret.err)
		}
		if string(buf) != "abcdefghij" {
			t.Error("read contents differ")
		}
	case <-time.After(20 * time.Millisecond):
		t.Fatal("failed to read")
	}

	go func() {
		n, err := r.Read(buf)
		done <- ioret{n, err}
	}()

	cancel()

	select {
	case ret := <-done:
		if ret.n != 0 {
			t.Error("ret.n should be 0", ret.n)
		}
		if ret.err == nil {
			t.Error("ret.err should be ctx error", ret.err)
		}
	case <-time.After(20 * time.Millisecond):
		t.Fatal("failed to stop reading after cancel")
	}
}

func TestWriterCancel(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	piper, pipew := io.Pipe()
	w := NewWriter(ctx, pipew)

	buf := make([]byte, 10)
	done := make(chan ioret)

	go func() {
		n, err := w.Write([]byte("abcdefghij"))
		done <- ioret{n, err}
	}()

	piper.Read(buf)

	select {
	case ret := <-done:
		if ret.n != 10 {
			t.Error("ret.n should be 10", ret.n)
		}
		if ret.err != nil {
			t.Error("ret.err should be nil", ret.err)
		}
		if string(buf) != "abcdefghij" {
			t.Error("write contents differ")
		}
	case <-time.After(20 * time.Millisecond):
		t.Fatal("failed to write")
	}

	go func() {
		n, err := w.Write([]byte("abcdefghij"))
		done <- ioret{n, err}
	}()

	cancel()

	select {
	case ret := <-done:
		if ret.n != 0 {
			t.Error("ret.n should be 0", ret.n)
		}
		if ret.err == nil {
			t.Error("ret.err should be ctx error", ret.err)
		}
	case <-time.After(20 * time.Millisecond):
		t.Fatal("failed to stop writing after cancel")
	}
}

func TestReadPostCancel(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	piper, pipew := io.Pipe()
	r := NewReader(ctx, piper)

	buf := make([]byte, 10)
	done := make(chan ioret)

	go func() {
		n, err := r.Read(buf)
		done <- ioret{n, err}
	}()

	cancel()

	select {
	case ret := <-done:
		if ret.n != 0 {
			t.Error("ret.n should be 0", ret.n)
		}
		if ret.err == nil {
			t.Error("ret.err should be ctx error", ret.err)
		}
	case <-time.After(20 * time.Millisecond):
		t.Fatal("failed to stop reading after cancel")
	}

	pipew.Write([]byte("abcdefghij"))

	if !bytes.Equal(buf, make([]byte, len(buf))) {
		t.Fatal("buffer should have not been written to")
	}
}

func TestWritePostCancel(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	piper, pipew := io.Pipe()
	w := NewWriter(ctx, pipew)

	buf := []byte("abcdefghij")
	buf2 := make([]byte, 10)
	done := make(chan ioret)

	go func() {
		n, err := w.Write(buf)
		done <- ioret{n, err}
	}()

	piper.Read(buf2)

	select {
	case ret := <-done:
		if ret.n != 10 {
			t.Error("ret.n should be 10", ret.n)
		}
		if ret.err != nil {
			t.Error("ret.err should be nil", ret.err)
		}
		if string(buf2) != "abcdefghij" {
			t.Error("write contents differ")
		}
	case <-time.After(20 * time.Millisecond):
		t.Fatal("failed to write")
	}

	go func() {
		n, err := w.Write(buf)
		done <- ioret{n, err}
	}()

	cancel()

	select {
	case ret := <-done:
		if ret.n != 0 {
			t.Error("ret.n should be 0", ret.n)
		}
		if ret.err == nil {
			t.Error("ret.err should be ctx error", ret.err)
		}
	case <-time.After(20 * time.Millisecond):
		t.Fatal("failed to stop writing after cancel")
	}

	copy(buf, []byte("aaaaaaaaaa"))

	piper.Read(buf2)

	if string(buf2) == "aaaaaaaaaa" {
		t.Error("buffer was read from after ctx cancel")
	} else if string(buf2) != "abcdefghij" {
		t.Error("write contents differ from expected")
	}
}
