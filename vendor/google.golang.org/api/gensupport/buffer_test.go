// Copyright 2015 Google Inc. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gensupport

import (
	"bytes"
	"io"
	"io/ioutil"
	"reflect"
	"testing"
	"testing/iotest"

	"google.golang.org/api/googleapi"
)

// getChunkAsString reads a chunk from mb, but does not call Next.
func getChunkAsString(t *testing.T, mb *MediaBuffer) (string, error) {
	chunk, _, size, err := mb.Chunk()

	buf, e := ioutil.ReadAll(chunk)
	if e != nil {
		t.Fatalf("Failed reading chunk: %v", e)
	}
	if size != len(buf) {
		t.Fatalf("reported chunk size doesn't match actual chunk size: got: %v; want: %v", size, len(buf))
	}
	return string(buf), err
}

func TestChunking(t *testing.T) {
	type testCase struct {
		data       string // the data to read from the Reader
		finalErr   error  // error to return after data has been read
		chunkSize  int
		wantChunks []string
	}

	for _, singleByteReads := range []bool{true, false} {
		for _, tc := range []testCase{
			{
				data:       "abcdefg",
				finalErr:   nil,
				chunkSize:  3,
				wantChunks: []string{"abc", "def", "g"},
			},
			{
				data:       "abcdefg",
				finalErr:   nil,
				chunkSize:  1,
				wantChunks: []string{"a", "b", "c", "d", "e", "f", "g"},
			},
			{
				data:       "abcdefg",
				finalErr:   nil,
				chunkSize:  7,
				wantChunks: []string{"abcdefg"},
			},
			{
				data:       "abcdefg",
				finalErr:   nil,
				chunkSize:  8,
				wantChunks: []string{"abcdefg"},
			},
			{
				data:       "abcdefg",
				finalErr:   io.ErrUnexpectedEOF,
				chunkSize:  3,
				wantChunks: []string{"abc", "def", "g"},
			},
			{
				data:       "abcdefg",
				finalErr:   io.ErrUnexpectedEOF,
				chunkSize:  8,
				wantChunks: []string{"abcdefg"},
			},
		} {
			var r io.Reader = &errReader{buf: []byte(tc.data), err: tc.finalErr}

			if singleByteReads {
				r = iotest.OneByteReader(r)
			}

			mb := NewMediaBuffer(r, tc.chunkSize)
			var gotErr error
			got := []string{}
			for {
				chunk, err := getChunkAsString(t, mb)
				if len(chunk) != 0 {
					got = append(got, string(chunk))
				}
				if err != nil {
					gotErr = err
					break
				}
				mb.Next()
			}

			if !reflect.DeepEqual(got, tc.wantChunks) {
				t.Errorf("Failed reading buffer: got: %v; want:%v", got, tc.wantChunks)
			}

			expectedErr := tc.finalErr
			if expectedErr == nil {
				expectedErr = io.EOF
			}
			if gotErr != expectedErr {
				t.Errorf("Reading buffer error: got: %v; want: %v", gotErr, expectedErr)
			}
		}
	}
}

func TestChunkCanBeReused(t *testing.T) {
	er := &errReader{buf: []byte("abcdefg")}
	mb := NewMediaBuffer(er, 3)

	// expectChunk reads a chunk and checks that it got what was wanted.
	expectChunk := func(want string, wantErr error) {
		got, err := getChunkAsString(t, mb)
		if err != wantErr {
			t.Errorf("error reading buffer: got: %v; want: %v", err, wantErr)
		}
		if !reflect.DeepEqual(got, want) {
			t.Errorf("Failed reading buffer: got: %q; want:%q", got, want)
		}
	}
	expectChunk("abc", nil)
	// On second call, should get same chunk again.
	expectChunk("abc", nil)
	mb.Next()
	expectChunk("def", nil)
	expectChunk("def", nil)
	mb.Next()
	expectChunk("g", io.EOF)
	expectChunk("g", io.EOF)
	mb.Next()
	expectChunk("", io.EOF)
}

func TestPos(t *testing.T) {
	er := &errReader{buf: []byte("abcdefg")}
	mb := NewMediaBuffer(er, 3)

	expectChunkAtOffset := func(want int64, wantErr error) {
		_, off, _, err := mb.Chunk()
		if err != wantErr {
			t.Errorf("error reading buffer: got: %v; want: %v", err, wantErr)
		}
		if got := off; got != want {
			t.Errorf("resumable buffer Pos: got: %v; want: %v", got, want)
		}
	}

	// We expect the first chunk to be at offset 0.
	expectChunkAtOffset(0, nil)
	// Fetching the same chunk should return the same offset.
	expectChunkAtOffset(0, nil)

	// Calling Next multiple times should only cause off to advance by 3, since off is not advanced until
	// the chunk is actually read.
	mb.Next()
	mb.Next()
	expectChunkAtOffset(3, nil)

	mb.Next()

	// Load the final 1-byte chunk.
	expectChunkAtOffset(6, io.EOF)

	// Next will advance 1 byte.  But there are no more chunks, so off will not increase beyond 7.
	mb.Next()
	expectChunkAtOffset(7, io.EOF)
	mb.Next()
	expectChunkAtOffset(7, io.EOF)
}

// bytes.Reader implements both Reader and ReaderAt.  The following types
// implement various combinations of Reader, ReaderAt and ContentTyper, by
// wrapping bytes.Reader.  All implement at least ReaderAt, so they can be
// passed to ReaderAtToReader.  The following table summarizes which types
// implement which interfaces:
//
//                 ReaderAt	Reader	ContentTyper
// reader              x          x
// typerReader         x          x          x
// readerAt            x
// typerReaderAt       x                     x

// reader implements Reader, in addition to ReaderAt.
type reader struct {
	r *bytes.Reader
}

func (r *reader) ReadAt(b []byte, off int64) (n int, err error) {
	return r.r.ReadAt(b, off)
}

func (r *reader) Read(b []byte) (n int, err error) {
	return r.r.Read(b)
}

// typerReader implements Reader and ContentTyper, in addition to ReaderAt.
type typerReader struct {
	r *bytes.Reader
}

func (tr *typerReader) ReadAt(b []byte, off int64) (n int, err error) {
	return tr.r.ReadAt(b, off)
}

func (tr *typerReader) Read(b []byte) (n int, err error) {
	return tr.r.Read(b)
}

func (tr *typerReader) ContentType() string {
	return "ctype"
}

// readerAt implements only ReaderAt.
type readerAt struct {
	r *bytes.Reader
}

func (ra *readerAt) ReadAt(b []byte, off int64) (n int, err error) {
	return ra.r.ReadAt(b, off)
}

// typerReaderAt implements ContentTyper, in addition to ReaderAt.
type typerReaderAt struct {
	r *bytes.Reader
}

func (tra *typerReaderAt) ReadAt(b []byte, off int64) (n int, err error) {
	return tra.r.ReadAt(b, off)
}

func (tra *typerReaderAt) ContentType() string {
	return "ctype"
}

func TestAdapter(t *testing.T) {
	data := "abc"

	checkConversion := func(to io.Reader, wantTyper bool) {
		if _, ok := to.(googleapi.ContentTyper); ok != wantTyper {
			t.Errorf("reader implements typer? got: %v; want: %v", ok, wantTyper)
		}
		if typer, ok := to.(googleapi.ContentTyper); ok && typer.ContentType() != "ctype" {
			t.Errorf("content type: got: %s; want: ctype", typer.ContentType())
		}
		buf, err := ioutil.ReadAll(to)
		if err != nil {
			t.Errorf("error reading data: %v", err)
			return
		}
		if !bytes.Equal(buf, []byte(data)) {
			t.Errorf("failed reading data: got: %s; want: %s", buf, data)
		}
	}

	type testCase struct {
		from      io.ReaderAt
		wantTyper bool
	}
	for _, tc := range []testCase{
		{
			from:      &reader{bytes.NewReader([]byte(data))},
			wantTyper: false,
		},
		{
			// Reader and ContentTyper
			from:      &typerReader{bytes.NewReader([]byte(data))},
			wantTyper: true,
		},
		{
			// ReaderAt
			from:      &readerAt{bytes.NewReader([]byte(data))},
			wantTyper: false,
		},
		{
			// ReaderAt and ContentTyper
			from:      &typerReaderAt{bytes.NewReader([]byte(data))},
			wantTyper: true,
		},
	} {
		to := ReaderAtToReader(tc.from, int64(len(data)))
		checkConversion(to, tc.wantTyper)
		// tc.from is a ReaderAt, and should be treated like one, even
		// if it also implements Reader.  Specifically, it can be
		// reused and read from the beginning.
		to = ReaderAtToReader(tc.from, int64(len(data)))
		checkConversion(to, tc.wantTyper)
	}
}
