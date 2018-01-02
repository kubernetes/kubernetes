package client

import (
	"bytes"
	"io"
	"testing"
)

type mockCloser struct {
	closed bool
}

func (closer *mockCloser) Read(b []byte) (int, error) {
	return 0, io.EOF
}

func (closer *mockCloser) Close() error {
	closer.closed = true
	return nil
}

func TestTeeReaderCloser(t *testing.T) {
	expected := "FOO"
	buf := bytes.NewBuffer([]byte(expected))
	lw := bytes.NewBuffer(nil)
	c := &mockCloser{}
	closer := teeReaderCloser{
		io.TeeReader(buf, lw),
		c,
	}

	b := make([]byte, len(expected))
	_, err := closer.Read(b)
	closer.Close()

	if expected != lw.String() {
		t.Errorf("Expected %q, but received %q", expected, lw.String())
	}

	if err != nil {
		t.Errorf("Expected 'nil', but received %v", err)
	}

	if !c.closed {
		t.Error("Expected 'true', but received 'false'")
	}
}

func TestLogWriter(t *testing.T) {
	expected := "FOO"
	lw := &logWriter{nil, bytes.NewBuffer(nil)}
	lw.Write([]byte(expected))

	if expected != lw.buf.String() {
		t.Errorf("Expected %q, but received %q", expected, lw.buf.String())
	}
}
