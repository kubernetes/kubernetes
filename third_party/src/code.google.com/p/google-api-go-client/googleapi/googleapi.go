// Copyright 2011 Google Inc. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package googleapi contains the common code shared by all Google API
// libraries.
package googleapi

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"io/ioutil"
	"mime/multipart"
	"net/http"
	"net/textproto"
	"net/url"
	"os"
	"strings"
)

// ContentTyper is an interface for Readers which know (or would like
// to override) their Content-Type. If a media body doesn't implement
// ContentTyper, the type is sniffed from the content using
// http.DetectContentType.
type ContentTyper interface {
	ContentType() string
}

const Version = "0.5"

// Error contains an error response from the server.
type Error struct {
	// Code is the HTTP response status code and will always be populated.
	Code int `json:"code"`
	// Message is the server response message and is only populated when
	// explicitly referenced by the JSON server response.
	Message string `json:"message"`
	// Body is the raw response returned by the server.
	// It is often but not always JSON, depending on how the request fails.
	Body string
}

func (e *Error) Error() string {
	if e.Message != "" {
		return fmt.Sprintf("googleapi: Error %d: %v", e.Code, e.Message)
	}
	return fmt.Sprintf("googleapi: got HTTP response code %d with body: %v", e.Code, e.Body)
}

type errorReply struct {
	Error *Error `json:"error"`
}

// CheckResponse returns an error (of type *Error) if the response
// status code is not 2xx.
func CheckResponse(res *http.Response) error {
	if res.StatusCode >= 200 && res.StatusCode <= 299 {
		return nil
	}
	slurp, err := ioutil.ReadAll(res.Body)
	if err == nil {
		jerr := new(errorReply)
		err = json.Unmarshal(slurp, jerr)
		if err == nil && jerr.Error != nil {
			if jerr.Error.Code == 0 {
				jerr.Error.Code = res.StatusCode
			}
			jerr.Error.Body = string(slurp)
			return jerr.Error
		}
	}
	return &Error{
		Code: res.StatusCode,
		Body: string(slurp),
	}
}

type MarshalStyle bool

var WithDataWrapper = MarshalStyle(true)
var WithoutDataWrapper = MarshalStyle(false)

func (wrap MarshalStyle) JSONReader(v interface{}) (io.Reader, error) {
	buf := new(bytes.Buffer)
	if wrap {
		buf.Write([]byte(`{"data": `))
	}
	err := json.NewEncoder(buf).Encode(v)
	if err != nil {
		return nil, err
	}
	if wrap {
		buf.Write([]byte(`}`))
	}
	return buf, nil
}

func getMediaType(media io.Reader) (io.Reader, string) {
	if typer, ok := media.(ContentTyper); ok {
		return media, typer.ContentType()
	}

	typ := "application/octet-stream"
	buf := make([]byte, 1024)
	n, err := media.Read(buf)
	buf = buf[:n]
	if err == nil {
		typ = http.DetectContentType(buf)
	}
	return io.MultiReader(bytes.NewBuffer(buf), media), typ
}

type Lengther interface {
	Len() int
}

// endingWithErrorReader from r until it returns an error.  If the
// final error from r is os.EOF and e is non-nil, e is used instead.
type endingWithErrorReader struct {
	r io.Reader
	e error
}

func (er endingWithErrorReader) Read(p []byte) (n int, err error) {
	n, err = er.r.Read(p)
	if err == io.EOF && er.e != nil {
		err = er.e
	}
	return
}

func getReaderSize(r io.Reader) (io.Reader, int64) {
	// Ideal case, the reader knows its own size.
	if lr, ok := r.(Lengther); ok {
		return r, int64(lr.Len())
	}

	// But maybe it's a seeker and we can seek to the end to find its size.
	if s, ok := r.(io.Seeker); ok {
		pos0, err := s.Seek(0, os.SEEK_CUR)
		if err == nil {
			posend, err := s.Seek(0, os.SEEK_END)
			if err == nil {
				_, err = s.Seek(pos0, os.SEEK_SET)
				if err == nil {
					return r, posend - pos0
				} else {
					// We moved it forward but can't restore it.
					// Seems unlikely, but can't really restore now.
					return endingWithErrorReader{strings.NewReader(""), err}, posend - pos0
				}
			}
		}
	}

	// Otherwise we have to make a copy to calculate how big the reader is.
	buf := new(bytes.Buffer)
	// TODO(bradfitz): put a cap on this copy? spill to disk after
	// a certain point?
	_, err := io.Copy(buf, r)
	return endingWithErrorReader{buf, err}, int64(buf.Len())
}

func typeHeader(contentType string) textproto.MIMEHeader {
	h := make(textproto.MIMEHeader)
	h.Set("Content-Type", contentType)
	return h
}

// countingWriter counts the number of bytes it receives to write, but
// discards them.
type countingWriter struct {
	n *int64
}

func (w countingWriter) Write(p []byte) (int, error) {
	*w.n += int64(len(p))
	return len(p), nil
}

// ConditionallyIncludeMedia does nothing if media is nil.
//
// bodyp is an in/out parameter.  It should initially point to the
// reader of the application/json (or whatever) payload to send in the
// API request.  It's updated to point to the multipart body reader.
//
// ctypep is an in/out parameter.  It should initially point to the
// content type of the bodyp, usually "application/json".  It's updated
// to the "multipart/related" content type, with random boundary.
//
// The return value is the content-length of the entire multpart body.
func ConditionallyIncludeMedia(media io.Reader, bodyp *io.Reader, ctypep *string) (totalContentLength int64, ok bool) {
	if media == nil {
		return
	}
	// Get the media type and size. The type check might return a
	// different reader instance, so do the size check first,
	// which looks at the specific type of the io.Reader.
	var mediaType string
	if typer, ok := media.(ContentTyper); ok {
		mediaType = typer.ContentType()
	}
	media, mediaSize := getReaderSize(media)
	if mediaType == "" {
		media, mediaType = getMediaType(media)
	}
	body, bodyType := *bodyp, *ctypep
	body, bodySize := getReaderSize(body)

	// Calculate how big the the multipart will be.
	{
		totalContentLength = bodySize + mediaSize
		mpw := multipart.NewWriter(countingWriter{&totalContentLength})
		mpw.CreatePart(typeHeader(bodyType))
		mpw.CreatePart(typeHeader(mediaType))
		mpw.Close()
	}

	pr, pw := io.Pipe()
	mpw := multipart.NewWriter(pw)
	*bodyp = pr
	*ctypep = "multipart/related; boundary=" + mpw.Boundary()
	go func() {
		defer pw.Close()
		defer mpw.Close()

		w, err := mpw.CreatePart(typeHeader(bodyType))
		if err != nil {
			return
		}
		_, err = io.Copy(w, body)
		if err != nil {
			return
		}

		w, err = mpw.CreatePart(typeHeader(mediaType))
		if err != nil {
			return
		}
		_, err = io.Copy(w, media)
		if err != nil {
			return
		}
	}()
	return totalContentLength, true
}

func ResolveRelative(basestr, relstr string) string {
	u, _ := url.Parse(basestr)
	rel, _ := url.Parse(relstr)
	u = u.ResolveReference(rel)
	us := u.String()
	us = strings.Replace(us, "%7B", "{", -1)
	us = strings.Replace(us, "%7D", "}", -1)
	return us
}

// has4860Fix is whether this Go environment contains the fix for
// http://golang.org/issue/4860
var has4860Fix bool

// init initializes has4860Fix by checking the behavior of the net/http package.
func init() {
	r := http.Request{
		URL: &url.URL{
			Scheme: "http",
			Opaque: "//opaque",
		},
	}
	b := &bytes.Buffer{}
	r.Write(b)
	has4860Fix = bytes.HasPrefix(b.Bytes(), []byte("GET http"))
}

// SetOpaque sets u.Opaque from u.Path such that HTTP requests to it
// don't alter any hex-escaped characters in u.Path.
func SetOpaque(u *url.URL) {
	u.Opaque = "//" + u.Host + u.Path
	if !has4860Fix {
		u.Opaque = u.Scheme + ":" + u.Opaque
	}
}

// CloseBody is used to close res.Body.
// Prior to calling Close, it also tries to Read a small amount to see an EOF.
// Not seeing an EOF can prevent HTTP Transports from reusing connections.
func CloseBody(res *http.Response) {
	if res == nil || res.Body == nil {
		return
	}
	// Justification for 3 byte reads: two for up to "\r\n" after
	// a JSON/XML document, and then 1 to see EOF if we haven't yet.
	// TODO(bradfitz): detect Go 1.3+ and skip these reads.
	// See https://codereview.appspot.com/58240043
	// and https://codereview.appspot.com/49570044
	buf := make([]byte, 1)
	for i := 0; i < 3; i++ {
		_, err := res.Body.Read(buf)
		if err != nil {
			break
		}
	}
	res.Body.Close()

}
