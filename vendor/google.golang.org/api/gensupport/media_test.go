// Copyright 2015 Google Inc. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gensupport

import (
	"bytes"
	"errors"
	"io"
	"io/ioutil"
	"reflect"
	"testing"
)

// errReader reads out of a buffer until it is empty, then returns the specified error.
type errReader struct {
	buf []byte
	err error
}

var errBang error = errors.New("bang")

func (er *errReader) Read(p []byte) (int, error) {
	if len(er.buf) == 0 {
		if er.err == nil {
			return 0, io.EOF
		}
		return 0, er.err
	}
	n := copy(p, er.buf)
	er.buf = er.buf[n:]
	return n, nil
}

func TestAll(t *testing.T) {
	type testCase struct {
		data     []byte // the data to read from the Reader
		finalErr error  // error to return after data has been read

		wantContentType       string
		wantContentTypeResult bool
	}

	for _, tc := range []testCase{
		{
			data:                  []byte{0, 0, 0, 0},
			finalErr:              nil,
			wantContentType:       "application/octet-stream",
			wantContentTypeResult: true,
		},
		{
			data:                  []byte(""),
			finalErr:              nil,
			wantContentType:       "text/plain; charset=utf-8",
			wantContentTypeResult: true,
		},
		{
			data:                  []byte(""),
			finalErr:              errBang,
			wantContentType:       "text/plain; charset=utf-8",
			wantContentTypeResult: false,
		},
		{
			data:                  []byte("abc"),
			finalErr:              nil,
			wantContentType:       "text/plain; charset=utf-8",
			wantContentTypeResult: true,
		},
		{
			data:                  []byte("abc"),
			finalErr:              errBang,
			wantContentType:       "text/plain; charset=utf-8",
			wantContentTypeResult: false,
		},
		// The following examples contain more bytes than are buffered for sniffing.
		{
			data:                  bytes.Repeat([]byte("a"), 513),
			finalErr:              nil,
			wantContentType:       "text/plain; charset=utf-8",
			wantContentTypeResult: true,
		},
		{
			data:                  bytes.Repeat([]byte("a"), 513),
			finalErr:              errBang,
			wantContentType:       "text/plain; charset=utf-8",
			wantContentTypeResult: true, // true because error is after first 512 bytes.
		},
	} {
		er := &errReader{buf: tc.data, err: tc.finalErr}

		sct := NewContentSniffer(er)

		// Even if was an error during the first 512 bytes, we should still be able to read those bytes.
		buf, err := ioutil.ReadAll(sct)

		if !reflect.DeepEqual(buf, tc.data) {
			t.Fatalf("Failed reading buffer: got: %q; want:%q", buf, tc.data)
		}

		if err != tc.finalErr {
			t.Fatalf("Reading buffer error: got: %v; want: %v", err, tc.finalErr)
		}

		ct, ok := sct.ContentType()
		if ok != tc.wantContentTypeResult {
			t.Fatalf("Content type result got: %v; want: %v", ok, tc.wantContentTypeResult)
		}
		if ok && ct != tc.wantContentType {
			t.Fatalf("Content type got: %q; want: %q", ct, tc.wantContentType)
		}
	}
}
