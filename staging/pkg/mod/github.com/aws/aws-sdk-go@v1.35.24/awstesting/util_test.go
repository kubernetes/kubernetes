package awstesting_test

import (
	"io"
	"testing"

	"github.com/aws/aws-sdk-go/awstesting"
)

func TestReadCloserClose(t *testing.T) {
	rc := awstesting.ReadCloser{Size: 1}
	err := rc.Close()

	if err != nil {
		t.Errorf("expect nil, got %v", err)
	}
	if !rc.Closed {
		t.Errorf("expect closed, was not")
	}
	if e, a := rc.Size, 1; e != a {
		t.Errorf("expect %v, got %v", e, a)
	}
}

func TestReadCloserRead(t *testing.T) {
	rc := awstesting.ReadCloser{Size: 5}
	b := make([]byte, 2)

	n, err := rc.Read(b)

	if err != nil {
		t.Errorf("expect nil, got %v", err)
	}
	if e, a := n, 2; e != a {
		t.Errorf("expect %v, got %v", e, a)
	}
	if rc.Closed {
		t.Errorf("expect not to be closed")
	}
	if e, a := rc.Size, 3; e != a {
		t.Errorf("expect %v, got %v", e, a)
	}

	err = rc.Close()
	if err != nil {
		t.Errorf("expect nil, got %v", err)
	}
	n, err = rc.Read(b)
	if e, a := err, io.EOF; e != a {
		t.Errorf("expect %v, got %v", e, a)
	}
	if e, a := n, 0; e != a {
		t.Errorf("expect %v, got %v", e, a)
	}
}

func TestReadCloserReadAll(t *testing.T) {
	rc := awstesting.ReadCloser{Size: 5}
	b := make([]byte, 5)

	n, err := rc.Read(b)

	if e, a := err, io.EOF; e != a {
		t.Errorf("expect %v, got %v", e, a)
	}
	if e, a := n, 5; e != a {
		t.Errorf("expect %v, got %v", e, a)
	}
	if rc.Closed {
		t.Errorf("expect not to be closed")
	}
	if e, a := rc.Size, 0; e != a {
		t.Errorf("expect %v, got %v", e, a)
	}
}
