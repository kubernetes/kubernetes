// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gensupport

import (
	"errors"
	"fmt"
	"io"
	"io/ioutil"
	"mime/multipart"
	"net/http"
	"net/textproto"

	"google.golang.org/api/googleapi"
)

const sniffBuffSize = 512

func NewContentSniffer(r io.Reader) *ContentSniffer {
	return &ContentSniffer{r: r}
}

// ContentSniffer wraps a Reader, and reports the content type determined by sniffing up to 512 bytes from the Reader.
type ContentSniffer struct {
	r     io.Reader
	start []byte // buffer for the sniffed bytes.
	err   error  // set to any error encountered while reading bytes to be sniffed.

	ctype   string // set on first sniff.
	sniffed bool   // set to true on first sniff.
}

func (sct *ContentSniffer) Read(p []byte) (n int, err error) {
	// Ensure that the content type is sniffed before any data is consumed from Reader.
	_, _ = sct.ContentType()

	if len(sct.start) > 0 {
		n := copy(p, sct.start)
		sct.start = sct.start[n:]
		return n, nil
	}

	// We may have read some bytes into start while sniffing, even if the read ended in an error.
	// We should first return those bytes, then the error.
	if sct.err != nil {
		return 0, sct.err
	}

	// Now we have handled all bytes that were buffered while sniffing.  Now just delegate to the underlying reader.
	return sct.r.Read(p)
}

// ContentType returns the sniffed content type, and whether the content type was succesfully sniffed.
func (sct *ContentSniffer) ContentType() (string, bool) {
	if sct.sniffed {
		return sct.ctype, sct.ctype != ""
	}
	sct.sniffed = true
	// If ReadAll hits EOF, it returns err==nil.
	sct.start, sct.err = ioutil.ReadAll(io.LimitReader(sct.r, sniffBuffSize))

	// Don't try to detect the content type based on possibly incomplete data.
	if sct.err != nil {
		return "", false
	}

	sct.ctype = http.DetectContentType(sct.start)
	return sct.ctype, true
}

// IncludeMedia combines an existing HTTP body with media content to create a multipart/related HTTP body.
//
// bodyp is an in/out parameter.  It should initially point to the
// reader of the application/json (or whatever) payload to send in the
// API request.  It's updated to point to the multipart body reader.
//
// ctypep is an in/out parameter.  It should initially point to the
// content type of the bodyp, usually "application/json".  It's updated
// to the "multipart/related" content type, with random boundary.
//
// The return value is a function that can be used to close the bodyp Reader with an error.
func IncludeMedia(media io.Reader, bodyp *io.Reader, ctypep *string) func() {
	var mediaType string
	media, mediaType = getMediaType(media)

	body, bodyType := *bodyp, *ctypep

	pr, pw := io.Pipe()
	mpw := multipart.NewWriter(pw)
	*bodyp = pr
	*ctypep = "multipart/related; boundary=" + mpw.Boundary()
	go func() {
		w, err := mpw.CreatePart(typeHeader(bodyType))
		if err != nil {
			mpw.Close()
			pw.CloseWithError(fmt.Errorf("googleapi: body CreatePart failed: %v", err))
			return
		}
		_, err = io.Copy(w, body)
		if err != nil {
			mpw.Close()
			pw.CloseWithError(fmt.Errorf("googleapi: body Copy failed: %v", err))
			return
		}

		w, err = mpw.CreatePart(typeHeader(mediaType))
		if err != nil {
			mpw.Close()
			pw.CloseWithError(fmt.Errorf("googleapi: media CreatePart failed: %v", err))
			return
		}
		_, err = io.Copy(w, media)
		if err != nil {
			mpw.Close()
			pw.CloseWithError(fmt.Errorf("googleapi: media Copy failed: %v", err))
			return
		}
		mpw.Close()
		pw.Close()
	}()
	return func() { pw.CloseWithError(errAborted) }
}

var errAborted = errors.New("googleapi: upload aborted")

func getMediaType(media io.Reader) (io.Reader, string) {
	if typer, ok := media.(googleapi.ContentTyper); ok {
		return media, typer.ContentType()
	}

	sniffer := NewContentSniffer(media)
	typ, ok := sniffer.ContentType()
	if !ok {
		// TODO(mcgreevy): Remove this default.  It maintains the semantics of the existing code,
		// but should not be relied on.
		typ = "application/octet-stream"
	}
	return sniffer, typ
}

// DetectMediaType detects and returns the content type of the provided media.
// If the type can not be determined, "application/octet-stream" is returned.
func DetectMediaType(media io.ReaderAt) string {
	if typer, ok := media.(googleapi.ContentTyper); ok {
		return typer.ContentType()
	}

	typ := "application/octet-stream"
	buf := make([]byte, 1024)
	n, err := media.ReadAt(buf, 0)
	buf = buf[:n]
	if err == nil || err == io.EOF {
		typ = http.DetectContentType(buf)
	}
	return typ
}

func typeHeader(contentType string) textproto.MIMEHeader {
	h := make(textproto.MIMEHeader)
	h.Set("Content-Type", contentType)
	return h
}
