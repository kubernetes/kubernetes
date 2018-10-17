// Copyright 2015 Google Inc. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gensupport

import (
	"bytes"
	"crypto/rand"
	"io"
	"io/ioutil"
	"net/http"
	"reflect"
	"strings"
	"testing"

	"google.golang.org/api/googleapi"
)

func TestContentSniffing(t *testing.T) {
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
			finalErr:              io.ErrUnexpectedEOF,
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
			finalErr:              io.ErrUnexpectedEOF,
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
			finalErr:              io.ErrUnexpectedEOF,
			wantContentType:       "text/plain; charset=utf-8",
			wantContentTypeResult: true, // true because error is after first 512 bytes.
		},
	} {
		er := &errReader{buf: tc.data, err: tc.finalErr}

		sct := newContentSniffer(er)

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

type staticContentTyper struct {
	io.Reader
}

func (sct staticContentTyper) ContentType() string {
	return "static content type"
}

func TestDetermineContentType(t *testing.T) {
	data := []byte("abc")
	rdr := func() io.Reader {
		return bytes.NewBuffer(data)
	}

	type testCase struct {
		r                  io.Reader
		explicitConentType string
		wantContentType    string
	}

	for _, tc := range []testCase{
		{
			r:               rdr(),
			wantContentType: "text/plain; charset=utf-8",
		},
		{
			r:               staticContentTyper{rdr()},
			wantContentType: "static content type",
		},
		{
			r:                  staticContentTyper{rdr()},
			explicitConentType: "explicit",
			wantContentType:    "explicit",
		},
	} {
		r, ctype := DetermineContentType(tc.r, tc.explicitConentType)
		got, err := ioutil.ReadAll(r)
		if err != nil {
			t.Fatalf("Failed reading buffer: %v", err)
		}
		if !reflect.DeepEqual(got, data) {
			t.Fatalf("Failed reading buffer: got: %q; want:%q", got, data)
		}

		if ctype != tc.wantContentType {
			t.Fatalf("Content type got: %q; want: %q", ctype, tc.wantContentType)
		}
	}
}

func TestNewInfoFromMedia(t *testing.T) {
	const textType = "text/plain; charset=utf-8"
	for _, test := range []struct {
		desc                                   string
		r                                      io.Reader
		opts                                   []googleapi.MediaOption
		wantType                               string
		wantMedia, wantBuffer, wantSingleChunk bool
	}{
		{
			desc:            "an empty reader results in a MediaBuffer with a single, empty chunk",
			r:               new(bytes.Buffer),
			opts:            nil,
			wantType:        textType,
			wantBuffer:      true,
			wantSingleChunk: true,
		},
		{
			desc:            "ContentType is observed",
			r:               new(bytes.Buffer),
			opts:            []googleapi.MediaOption{googleapi.ContentType("xyz")},
			wantType:        "xyz",
			wantBuffer:      true,
			wantSingleChunk: true,
		},
		{
			desc:            "chunk size of zero: don't use a MediaBuffer; upload as a single chunk",
			r:               strings.NewReader("12345"),
			opts:            []googleapi.MediaOption{googleapi.ChunkSize(0)},
			wantType:        textType,
			wantMedia:       true,
			wantSingleChunk: true,
		},
		{
			desc:            "chunk size > data size: MediaBuffer with single chunk",
			r:               strings.NewReader("12345"),
			opts:            []googleapi.MediaOption{googleapi.ChunkSize(100)},
			wantType:        textType,
			wantBuffer:      true,
			wantSingleChunk: true,
		},
		{
			desc:            "chunk size == data size: MediaBuffer with single chunk",
			r:               &nullReader{googleapi.MinUploadChunkSize},
			opts:            []googleapi.MediaOption{googleapi.ChunkSize(1)},
			wantType:        "application/octet-stream",
			wantBuffer:      true,
			wantSingleChunk: true,
		},
		{
			desc: "chunk size < data size: MediaBuffer, not single chunk",
			// Note that ChunkSize = 1 is rounded up to googleapi.MinUploadChunkSize.
			r:               &nullReader{2 * googleapi.MinUploadChunkSize},
			opts:            []googleapi.MediaOption{googleapi.ChunkSize(1)},
			wantType:        "application/octet-stream",
			wantBuffer:      true,
			wantSingleChunk: false,
		},
	} {

		mi := NewInfoFromMedia(test.r, test.opts)
		if got, want := mi.mType, test.wantType; got != want {
			t.Errorf("%s: type: got %q, want %q", test.desc, got, want)
		}
		if got, want := (mi.media != nil), test.wantMedia; got != want {
			t.Errorf("%s: media non-nil: got %t, want %t", test.desc, got, want)
		}
		if got, want := (mi.buffer != nil), test.wantBuffer; got != want {
			t.Errorf("%s: buffer non-nil: got %t, want %t", test.desc, got, want)
		}
		if got, want := mi.singleChunk, test.wantSingleChunk; got != want {
			t.Errorf("%s: singleChunk: got %t, want %t", test.desc, got, want)
		}
	}
}

func TestUploadRequest(t *testing.T) {
	for _, test := range []struct {
		desc            string
		r               io.Reader
		chunkSize       int
		wantContentType string
		wantUploadType  string
	}{
		{
			desc:            "chunk size of zero: don't use a MediaBuffer; upload as a single chunk",
			r:               strings.NewReader("12345"),
			chunkSize:       0,
			wantContentType: "multipart/related;",
		},
		{
			desc:            "chunk size > data size: MediaBuffer with single chunk",
			r:               strings.NewReader("12345"),
			chunkSize:       100,
			wantContentType: "multipart/related;",
		},
		{
			desc:            "chunk size == data size: MediaBuffer with single chunk",
			r:               &nullReader{googleapi.MinUploadChunkSize},
			chunkSize:       1,
			wantContentType: "multipart/related;",
		},
		{
			desc: "chunk size < data size: MediaBuffer, not single chunk",
			// Note that ChunkSize = 1 is rounded up to googleapi.MinUploadChunkSize.
			r:              &nullReader{2 * googleapi.MinUploadChunkSize},
			chunkSize:      1,
			wantUploadType: "application/octet-stream",
		},
	} {
		mi := NewInfoFromMedia(test.r, []googleapi.MediaOption{googleapi.ChunkSize(test.chunkSize)})
		h := http.Header{}
		mi.UploadRequest(h, new(bytes.Buffer))
		if got, want := h.Get("Content-Type"), test.wantContentType; !strings.HasPrefix(got, want) {
			t.Errorf("%s: Content-Type: got %q, want prefix %q", test.desc, got, want)
		}
		if got, want := h.Get("X-Upload-Content-Type"), test.wantUploadType; got != want {
			t.Errorf("%s: X-Upload-Content-Type: got %q, want %q", test.desc, got, want)
		}
	}
}

func TestUploadRequestGetBody(t *testing.T) {
	// Test that a single chunk results in a getBody function that is non-nil, and
	// that produces the same content as the original body.

	// Mock out rand.Reader so we use the same multipart boundary every time.
	rr := rand.Reader
	rand.Reader = &nullReader{1000}
	defer func() {
		rand.Reader = rr
	}()

	for _, test := range []struct {
		desc            string
		r               io.Reader
		chunkSize       int
		wantGetBody     bool
		wantContentType string
		wantUploadType  string
	}{
		{
			desc:        "chunk size of zero: no getBody",
			r:           &nullReader{10},
			chunkSize:   0,
			wantGetBody: false,
		},
		{
			desc:        "chunk size == data size: 1 chunk, getBody",
			r:           &nullReader{googleapi.MinUploadChunkSize},
			chunkSize:   1,
			wantGetBody: true,
		},
		{
			desc: "chunk size < data size: MediaBuffer, >1 chunk, no getBody",
			// No getBody here, because the initial request contains no media data
			// Note that ChunkSize = 1 is rounded up to googleapi.MinUploadChunkSize.
			r:           &nullReader{2 * googleapi.MinUploadChunkSize},
			chunkSize:   1,
			wantGetBody: false,
		},
	} {
		mi := NewInfoFromMedia(test.r, []googleapi.MediaOption{googleapi.ChunkSize(test.chunkSize)})
		r, getBody, _ := mi.UploadRequest(http.Header{}, bytes.NewBuffer([]byte("body")))
		if got, want := (getBody != nil), test.wantGetBody; got != want {
			t.Errorf("%s: getBody: got %t, want %t", test.desc, got, want)
			continue
		}
		if getBody == nil {
			continue
		}
		want, err := ioutil.ReadAll(r)
		if err != nil {
			t.Fatal(err)
		}
		for i := 0; i < 3; i++ {
			rc, err := getBody()
			if err != nil {
				t.Fatal(err)
			}
			got, err := ioutil.ReadAll(rc)
			if err != nil {
				t.Fatal(err)
			}
			if !bytes.Equal(got, want) {
				t.Errorf("%s, %d:\ngot:\n%s\nwant:\n%s", test.desc, i, string(got), string(want))
			}
		}
	}
}

func TestResumableUpload(t *testing.T) {
	for _, test := range []struct {
		desc                string
		r                   io.Reader
		chunkSize           int
		wantUploadType      string
		wantResumableUpload bool
	}{
		{
			desc:                "chunk size of zero: don't use a MediaBuffer; upload as a single chunk",
			r:                   strings.NewReader("12345"),
			chunkSize:           0,
			wantUploadType:      "multipart",
			wantResumableUpload: false,
		},
		{
			desc:                "chunk size > data size: MediaBuffer with single chunk",
			r:                   strings.NewReader("12345"),
			chunkSize:           100,
			wantUploadType:      "multipart",
			wantResumableUpload: false,
		},
		{
			desc: "chunk size == data size: MediaBuffer with single chunk",
			// (Because nullReader returns EOF with the last bytes.)
			r:                   &nullReader{googleapi.MinUploadChunkSize},
			chunkSize:           googleapi.MinUploadChunkSize,
			wantUploadType:      "multipart",
			wantResumableUpload: false,
		},
		{
			desc: "chunk size < data size: MediaBuffer, not single chunk",
			// Note that ChunkSize = 1 is rounded up to googleapi.MinUploadChunkSize.
			r:                   &nullReader{2 * googleapi.MinUploadChunkSize},
			chunkSize:           1,
			wantUploadType:      "resumable",
			wantResumableUpload: true,
		},
	} {
		mi := NewInfoFromMedia(test.r, []googleapi.MediaOption{googleapi.ChunkSize(test.chunkSize)})
		if got, want := mi.UploadType(), test.wantUploadType; got != want {
			t.Errorf("%s: upload type: got %q, want %q", test.desc, got, want)
		}
		if got, want := mi.ResumableUpload("") != nil, test.wantResumableUpload; got != want {
			t.Errorf("%s: resumable upload non-nil: got %t, want %t", test.desc, got, want)
		}
	}
}

// A nullReader simulates reading a fixed number of bytes.
type nullReader struct {
	remain int
}

// Read doesn't touch buf, but it does reduce the amount of bytes remaining
// by len(buf).
func (r *nullReader) Read(buf []byte) (int, error) {
	n := len(buf)
	if r.remain < n {
		n = r.remain
	}
	r.remain -= n
	var err error
	if r.remain == 0 {
		err = io.EOF
	}
	return n, err
}
