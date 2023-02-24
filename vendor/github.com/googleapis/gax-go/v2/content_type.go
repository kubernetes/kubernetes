// Copyright 2022, Google Inc.
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
//     * Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//     * Redistributions in binary form must reproduce the above
// copyright notice, this list of conditions and the following disclaimer
// in the documentation and/or other materials provided with the
// distribution.
//     * Neither the name of Google Inc. nor the names of its
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

package gax

import (
	"io"
	"io/ioutil"
	"net/http"
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

// ContentType returns the sniffed content type, and whether the content type was successfully sniffed.
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
// The content of media will be sniffed to determine the content type.
// After calling DetectContentType the caller must not perform further reads on
// media, but rather read from the Reader that is returned.
func DetermineContentType(media io.Reader) (io.Reader, string) {
	// For backwards compatibility, allow clients to set content
	// type by providing a ContentTyper for media.
	// Note: This is an anonymous interface definition copied from googleapi.ContentTyper.
	if typer, ok := media.(interface {
		ContentType() string
	}); ok {
		return media, typer.ContentType()
	}

	sniffer := newContentSniffer(media)
	if ctype, ok := sniffer.ContentType(); ok {
		return sniffer, ctype
	}
	// If content type could not be sniffed, reads from sniffer will eventually fail with an error.
	return sniffer, ""
}
