package s3manager

import (
	"fmt"
	"io"
	"reflect"
	"testing"
)

type testBufioWriter struct {
	ReadFromN   int64
	ReadFromErr error
	FlushReturn error
}

func (t testBufioWriter) Write(p []byte) (n int, err error) {
	panic("unused")
}

func (t testBufioWriter) ReadFrom(r io.Reader) (n int64, err error) {
	return t.ReadFromN, t.ReadFromErr
}

func (t testBufioWriter) Flush() error {
	return t.FlushReturn
}

func (t *testBufioWriter) Reset(io.Writer) {
	panic("unused")
}

func TestBufferedReadFromFlusher_ReadFrom(t *testing.T) {
	cases := map[string]struct {
		w           testBufioWriter
		expectedErr error
	}{
		"no errors": {},
		"error returned from underlying ReadFrom": {
			w: testBufioWriter{
				ReadFromN:   42,
				ReadFromErr: fmt.Errorf("readfrom"),
			},
			expectedErr: fmt.Errorf("readfrom"),
		},
		"error returned from Flush": {
			w: testBufioWriter{
				ReadFromN:   7,
				FlushReturn: fmt.Errorf("flush"),
			},
			expectedErr: fmt.Errorf("flush"),
		},
		"error returned from ReadFrom and Flush": {
			w: testBufioWriter{
				ReadFromN:   1337,
				ReadFromErr: fmt.Errorf("readfrom"),
				FlushReturn: fmt.Errorf("flush"),
			},
			expectedErr: fmt.Errorf("readfrom"),
		},
	}

	for name, tCase := range cases {
		t.Log(name)
		readFromFlusher := bufferedReadFrom{bufferedWriter: &tCase.w}
		n, err := readFromFlusher.ReadFrom(nil)
		if e, a := tCase.w.ReadFromN, n; e != a {
			t.Errorf("expected %v bytes, got %v", e, a)
		}
		if e, a := tCase.expectedErr, err; !reflect.DeepEqual(e, a) {
			t.Errorf("expected error %v. got %v", e, a)
		}
	}
}
