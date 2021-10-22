package s3manager

import (
	"bytes"
	"io"
	"testing"

	"github.com/aws/aws-sdk-go/internal/sdkio"
)

func TestBufferedReadSeekerRead(t *testing.T) {
	expected := []byte("testData")

	readSeeker := NewBufferedReadSeeker(bytes.NewReader(expected), make([]byte, 4))

	var (
		actual []byte
		buffer = make([]byte, 2)
	)

	for {
		n, err := readSeeker.Read(buffer)
		actual = append(actual, buffer[:n]...)
		if err != nil && err == io.EOF {
			break
		} else if err != nil {
			t.Fatalf("failed to read from reader: %v", err)
		}
	}

	if !bytes.Equal(expected, actual) {
		t.Errorf("expected %v, got %v", expected, actual)
	}
}

func TestBufferedReadSeekerSeek(t *testing.T) {
	content := []byte("testData")

	readSeeker := NewBufferedReadSeeker(bytes.NewReader(content), make([]byte, 4))

	_, err := readSeeker.Seek(4, sdkio.SeekStart)
	if err != nil {
		t.Fatalf("failed to seek reader: %v", err)
	}

	var (
		actual []byte
		buffer = make([]byte, 4)
	)

	for {
		n, err := readSeeker.Read(buffer)
		actual = append(actual, buffer[:n]...)
		if err != nil && err == io.EOF {
			break
		} else if err != nil {
			t.Fatalf("failed to read from reader: %v", err)
		}
	}

	if e := []byte("Data"); !bytes.Equal(e, actual) {
		t.Errorf("expected %v, got %v", e, actual)
	}
}

func TestBufferedReadSeekerReadAt(t *testing.T) {
	content := []byte("testData")

	readSeeker := NewBufferedReadSeeker(bytes.NewReader(content), make([]byte, 2))

	buffer := make([]byte, 4)

	_, err := readSeeker.ReadAt(buffer, 0)
	if err != nil {
		t.Fatalf("failed to seek reader: %v", err)
	}

	if e := content[:4]; !bytes.Equal(e, buffer) {
		t.Errorf("expected %v, got %v", e, buffer)
	}
}
