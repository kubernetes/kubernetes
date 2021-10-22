package s3crypto

import (
	"bytes"
	"encoding/hex"
	"fmt"
	"io"
	"io/ioutil"
	"strings"
	"testing"
)

// From Go stdlib encoding/sha256 test cases
func TestSHA256(t *testing.T) {
	sha := newSHA256Writer(nil)
	expected, _ := hex.DecodeString("e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855")
	b := sha.GetValue()

	if !bytes.Equal(expected, b) {
		t.Errorf("expected equivalent sha values, but received otherwise")
	}
}

func TestSHA256_Case2(t *testing.T) {
	sha := newSHA256Writer(bytes.NewBuffer([]byte{}))
	sha.Write([]byte("hello"))
	expected, _ := hex.DecodeString("2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824")
	b := sha.GetValue()

	if !bytes.Equal(expected, b) {
		t.Errorf("expected equivalent sha values, but received otherwise")
	}
}

type mockReader struct {
	err error
}

func (m mockReader) Read(p []byte) (int, error) {
	return len(p), m.err
}

func TestContentLengthReader(t *testing.T) {
	cases := []struct {
		reader      io.Reader
		expected    int64
		expectedErr string
	}{
		{
			reader:   bytes.NewReader([]byte("foo bar baz")),
			expected: 11,
		},
		{
			reader:   bytes.NewReader(nil),
			expected: 0,
		},
		{
			reader:      mockReader{err: fmt.Errorf("not an EOF error")},
			expectedErr: "not an EOF error",
		},
	}

	for _, tt := range cases {
		reader := newContentLengthReader(tt.reader)
		_, err := ioutil.ReadAll(reader)
		if err != nil {
			if len(tt.expectedErr) == 0 {
				t.Errorf("expected no error, got %v", err)
			} else if !strings.Contains(err.Error(), tt.expectedErr) {
				t.Errorf("expected error %v, got %v", tt.expectedErr, err.Error())
			}
			continue
		} else if len(tt.expectedErr) > 0 {
			t.Error("expected error, got none")
			continue
		}
		actual := reader.GetContentLength()
		if tt.expected != actual {
			t.Errorf("expected %v, got %v", tt.expected, actual)
		}
	}
}
