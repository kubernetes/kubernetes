// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gensupport

import (
	"fmt"
	"io"
	"io/ioutil"
	"mime/multipart"
	"net/http"
	"net/textproto"

	"google.golang.org/api/googleapi"
)

const sniffBuffSize = 512

func newContentSniffer(r io.Reader) *contentSniffer {
	return &contentSniffer{r: r}
}

// contentSniffer wraps a Reader, and reports the content type determined by sniffing up to 512 bytes from the Reader.
type contentSniffer struct {
	r     io.Reader
	start []byte // buffer for the sniffed bytes.
	err   error  // set to any error encountered while reading bytes to be sniffed.

	ctype   string // set on first sniff.
	sniffed bool   // set to true on first sniff.
}

func (cs *contentSniffer) Read(p []byte) (n int, err error) {
	// Ensure that the content type is sniffed before any data is consumed from Reader.
	_, _ = cs.ContentType()

	if len(cs.start) > 0 {
		n := copy(p, cs.start)
		cs.start = cs.start[n:]
		return n, nil
	}

	// We may have read some bytes into start while sniffing, even if the read ended in an error.
	// We should first return those bytes, then the error.
	if cs.err != nil {
		return 0, cs.err
	}

	// Now we have handled all bytes that were buffered while sniffing.  Now just delegate to the underlying reader.
	return cs.r.Read(p)
}

// ContentType returns the sniffed content type, and whether the content type was succesfully sniffed.
func (cs *contentSniffer) ContentType() (string, bool) {
	if cs.sniffed {
		return cs.ctype, cs.ctype != ""
	}
	cs.sniffed = true
	// If ReadAll hits EOF, it returns err==nil.
	cs.start, cs.err = ioutil.ReadAll(io.LimitReader(cs.r, sniffBuffSize))

	// Don't try to detect the content type based on possibly incomplete data.
	if cs.err != nil {
		return "", false
	}

	cs.ctype = http.DetectContentType(cs.start)
	return cs.ctype, true
}

// DetermineContentType determines the content type of the supplied reader.
// If the content type is already known, it can be specified via ctype.
// Otherwise, the content of media will be sniffed to determine the content type.
// If media implements googleapi.ContentTyper (deprecated), this will be used
// instead of sniffing the content.
// After calling DetectContentType the caller must not perform further reads on
// media, but rather read from the Reader that is returned.
func DetermineContentType(media io.Reader, ctype string) (io.Reader, string) {
	// Note: callers could avoid calling DetectContentType if ctype != "",
	// but doing the check inside this function reduces the amount of
	// generated code.
	if ctype != "" {
		return media, ctype
	}

	// For backwards compatability, allow clients to set content
	// type by providing a ContentTyper for media.
	if typer, ok := media.(googleapi.ContentTyper); ok {
		return media, typer.ContentType()
	}

	sniffer := newContentSniffer(media)
	if ctype, ok := sniffer.ContentType(); ok {
		return sniffer, ctype
	}
	// If content type could not be sniffed, reads from sniffer will eventually fail with an error.
	return sniffer, ""
}

type typeReader struct {
	io.Reader
	typ string
}

// multipartReader combines the contents of multiple readers to creat a multipart/related HTTP body.
// Close must be called if reads from the multipartReader are abandoned before reaching EOF.
type multipartReader struct {
	pr       *io.PipeReader
	pipeOpen bool
	ctype    string
}

func newMultipartReader(parts []typeReader) *multipartReader {
	mp := &multipartReader{pipeOpen: true}
	var pw *io.PipeWriter
	mp.pr, pw = io.Pipe()
	mpw := multipart.NewWriter(pw)
	mp.ctype = "multipart/related; boundary=" + mpw.Boundary()
	go func() {
		for _, part := range parts {
			w, err := mpw.CreatePart(typeHeader(part.typ))
			if err != nil {
				mpw.Close()
				pw.CloseWithError(fmt.Errorf("googleapi: CreatePart failed: %v", err))
				return
			}
			_, err = io.Copy(w, part.Reader)
			if err != nil {
				mpw.Close()
				pw.CloseWithError(fmt.Errorf("googleapi: Copy failed: %v", err))
				return
			}
		}

		mpw.Close()
		pw.Close()
	}()
	return mp
}

func (mp *multipartReader) Read(data []byte) (n int, err error) {
	return mp.pr.Read(data)
}

func (mp *multipartReader) Close() error {
	if !mp.pipeOpen {
		return nil
	}
	mp.pipeOpen = false
	return mp.pr.Close()
}

// CombineBodyMedia combines a json body with media content to create a multipart/related HTTP body.
// It returns a ReadCloser containing the combined body, and the overall "multipart/related" content type, with random boundary.
//
// The caller must call Close on the returned ReadCloser if reads are abandoned before reaching EOF.
func CombineBodyMedia(body io.Reader, bodyContentType string, media io.Reader, mediaContentType string) (io.ReadCloser, string) {
	mp := newMultipartReader([]typeReader{
		{body, bodyContentType},
		{media, mediaContentType},
	})
	return mp, mp.ctype
}

func typeHeader(contentType string) textproto.MIMEHeader {
	h := make(textproto.MIMEHeader)
	if contentType != "" {
		h.Set("Content-Type", contentType)
	}
	return h
}

// PrepareUpload determines whether the data in the supplied reader should be
// uploaded in a single request, or in sequential chunks.
// chunkSize is the size of the chunk that media should be split into.
// If chunkSize is non-zero and the contents of media do not fit in a single
// chunk (or there is an error reading media), then media will be returned as a
// MediaBuffer.  Otherwise, media will be returned as a Reader.
//
// After PrepareUpload has been called, media should no longer be used: the
// media content should be accessed via one of the return values.
func PrepareUpload(media io.Reader, chunkSize int) (io.Reader, *MediaBuffer) {
	if chunkSize == 0 { // do not chunk
		return media, nil
	}

	mb := NewMediaBuffer(media, chunkSize)
	rdr, _, _, err := mb.Chunk()

	if err == io.EOF { // we can upload this in a single request
		return rdr, nil
	}
	// err might be a non-EOF error. If it is, the next call to mb.Chunk will
	// return the same error. Returning a MediaBuffer ensures that this error
	// will be handled at some point.

	return nil, mb
}
