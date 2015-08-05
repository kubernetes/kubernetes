package ioutils

import (
	"bytes"
	"fmt"
	"io"
	"io/ioutil"
	"os"
	"strings"
	"testing"
)

func TestMultiReadSeekerReadAll(t *testing.T) {
	str := "hello world"
	s1 := strings.NewReader(str + " 1")
	s2 := strings.NewReader(str + " 2")
	s3 := strings.NewReader(str + " 3")
	mr := MultiReadSeeker(s1, s2, s3)

	expectedSize := int64(s1.Len() + s2.Len() + s3.Len())

	b, err := ioutil.ReadAll(mr)
	if err != nil {
		t.Fatal(err)
	}

	expected := "hello world 1hello world 2hello world 3"
	if string(b) != expected {
		t.Fatalf("ReadAll failed, got: %q, expected %q", string(b), expected)
	}

	size, err := mr.Seek(0, os.SEEK_END)
	if err != nil {
		t.Fatal(err)
	}
	if size != expectedSize {
		t.Fatalf("reader size does not match, got %d, expected %d", size, expectedSize)
	}

	// Reset the position and read again
	pos, err := mr.Seek(0, os.SEEK_SET)
	if err != nil {
		t.Fatal(err)
	}
	if pos != 0 {
		t.Fatalf("expected position to be set to 0, got %d", pos)
	}

	b, err = ioutil.ReadAll(mr)
	if err != nil {
		t.Fatal(err)
	}

	if string(b) != expected {
		t.Fatalf("ReadAll failed, got: %q, expected %q", string(b), expected)
	}
}

func TestMultiReadSeekerReadEach(t *testing.T) {
	str := "hello world"
	s1 := strings.NewReader(str + " 1")
	s2 := strings.NewReader(str + " 2")
	s3 := strings.NewReader(str + " 3")
	mr := MultiReadSeeker(s1, s2, s3)

	var totalBytes int64
	for i, s := range []*strings.Reader{s1, s2, s3} {
		sLen := int64(s.Len())
		buf := make([]byte, s.Len())
		expected := []byte(fmt.Sprintf("%s %d", str, i+1))

		if _, err := mr.Read(buf); err != nil && err != io.EOF {
			t.Fatal(err)
		}

		if !bytes.Equal(buf, expected) {
			t.Fatalf("expected %q to be %q", string(buf), string(expected))
		}

		pos, err := mr.Seek(0, os.SEEK_CUR)
		if err != nil {
			t.Fatalf("iteration: %d, error: %v", i+1, err)
		}

		// check that the total bytes read is the current position of the seeker
		totalBytes += sLen
		if pos != totalBytes {
			t.Fatalf("expected current position to be: %d, got: %d, iteration: %d", totalBytes, pos, i+1)
		}

		// This tests not only that SEEK_SET and SEEK_CUR give the same values, but that the next iteration is in the expected position as well
		newPos, err := mr.Seek(pos, os.SEEK_SET)
		if err != nil {
			t.Fatal(err)
		}
		if newPos != pos {
			t.Fatalf("expected to get same position when calling SEEK_SET with value from SEEK_CUR, cur: %d, set: %d", pos, newPos)
		}
	}
}

func TestMultiReadSeekerReadSpanningChunks(t *testing.T) {
	str := "hello world"
	s1 := strings.NewReader(str + " 1")
	s2 := strings.NewReader(str + " 2")
	s3 := strings.NewReader(str + " 3")
	mr := MultiReadSeeker(s1, s2, s3)

	buf := make([]byte, s1.Len()+3)
	_, err := mr.Read(buf)
	if err != nil {
		t.Fatal(err)
	}

	// expected is the contents of s1 + 3 bytes from s2, ie, the `hel` at the end of this string
	expected := "hello world 1hel"
	if string(buf) != expected {
		t.Fatalf("expected %s to be %s", string(buf), expected)
	}
}

func TestMultiReadSeekerNegativeSeek(t *testing.T) {
	str := "hello world"
	s1 := strings.NewReader(str + " 1")
	s2 := strings.NewReader(str + " 2")
	s3 := strings.NewReader(str + " 3")
	mr := MultiReadSeeker(s1, s2, s3)

	s1Len := s1.Len()
	s2Len := s2.Len()
	s3Len := s3.Len()

	s, err := mr.Seek(int64(-1*s3.Len()), os.SEEK_END)
	if err != nil {
		t.Fatal(err)
	}
	if s != int64(s1Len+s2Len) {
		t.Fatalf("expected %d to be %d", s, s1.Len()+s2.Len())
	}

	buf := make([]byte, s3Len)
	if _, err := mr.Read(buf); err != nil && err != io.EOF {
		t.Fatal(err)
	}
	expected := fmt.Sprintf("%s %d", str, 3)
	if string(buf) != fmt.Sprintf("%s %d", str, 3) {
		t.Fatalf("expected %q to be %q", string(buf), expected)
	}
}
