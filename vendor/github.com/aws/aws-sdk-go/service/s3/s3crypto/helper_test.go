package s3crypto

import (
	"bytes"
	"testing"
)

func TestBytesReadWriteSeeker_Read(t *testing.T) {
	b := &bytesReadWriteSeeker{[]byte{1, 2, 3}, 0}
	expected := []byte{1, 2, 3}
	buf := make([]byte, 3)
	n, err := b.Read(buf)

	if err != nil {
		t.Errorf("expected no error, but received %v", err)
	}

	if e, a := 3, n; e != a {
		t.Errorf("expected %d, but received %d", e, a)
	}

	if !bytes.Equal(expected, buf) {
		t.Error("expected equivalent byte slices, but received otherwise")
	}
}

func TestBytesReadWriteSeeker_Write(t *testing.T) {
	b := &bytesReadWriteSeeker{}
	expected := []byte{1, 2, 3}
	buf := make([]byte, 3)
	n, err := b.Write([]byte{1, 2, 3})

	if err != nil {
		t.Errorf("expected no error, but received %v", err)
	}

	if e, a := 3, n; e != a {
		t.Errorf("expected %d, but received %d", e, a)
	}

	n, err = b.Read(buf)
	if err != nil {
		t.Errorf("expected no error, but received %v", err)
	}

	if e, a := 3, n; e != a {
		t.Errorf("expected %d, but received %d", e, a)
	}

	if !bytes.Equal(expected, buf) {
		t.Error("expected equivalent byte slices, but received otherwise")
	}
}

func TestBytesReadWriteSeeker_Seek(t *testing.T) {
	b := &bytesReadWriteSeeker{[]byte{1, 2, 3}, 0}
	expected := []byte{2, 3}
	m, err := b.Seek(1, 0)

	if err != nil {
		t.Errorf("expected no error, but received %v", err)
	}

	if e, a := 1, int(m); e != a {
		t.Errorf("expected %d, but received %d", e, a)
	}

	buf := make([]byte, 3)
	n, err := b.Read(buf)

	if err != nil {
		t.Errorf("expected no error, but received %v", err)
	}

	if e, a := 2, n; e != a {
		t.Errorf("expected %d, but received %d", e, a)
	}

	if !bytes.Equal(expected, buf[:n]) {
		t.Error("expected equivalent byte slices, but received otherwise")
	}
}
