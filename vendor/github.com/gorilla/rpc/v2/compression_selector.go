// Copyright 2009 The Go Authors. All rights reserved.
// Copyright 2012 The Gorilla Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package rpc

import (
	"compress/flate"
	"compress/gzip"
	"io"
	"net/http"
	"strings"
	"unicode"
)

// gzipWriter writes and closes the gzip writer.
type gzipWriter struct {
	w *gzip.Writer
}

func (gw *gzipWriter) Write(p []byte) (n int, err error) {
	defer gw.w.Close()
	return gw.w.Write(p)
}

// gzipEncoder implements the gzip compressed http encoder.
type gzipEncoder struct {
}

func (enc *gzipEncoder) Encode(w http.ResponseWriter) io.Writer {
	w.Header().Set("Content-Encoding", "gzip")
	return &gzipWriter{gzip.NewWriter(w)}
}

// flateWriter writes and closes the flate writer.
type flateWriter struct {
	w *flate.Writer
}

func (fw *flateWriter) Write(p []byte) (n int, err error) {
	defer fw.w.Close()
	return fw.w.Write(p)
}

// flateEncoder implements the flate compressed http encoder.
type flateEncoder struct {
}

func (enc *flateEncoder) Encode(w http.ResponseWriter) io.Writer {
	fw, err := flate.NewWriter(w, flate.DefaultCompression)
	if err != nil {
		return w
	}
	w.Header().Set("Content-Encoding", "deflate")
	return &flateWriter{fw}
}

// CompressionSelector generates the compressed http encoder.
type CompressionSelector struct {
}

// acceptedEnc returns the first compression type in "Accept-Encoding" header
// field of the request.
func acceptedEnc(req *http.Request) string {
	encHeader := req.Header.Get("Accept-Encoding")
	if encHeader == "" {
		return ""
	}
	encTypes := strings.FieldsFunc(encHeader, func(r rune) bool {
		return unicode.IsSpace(r) || r == ','
	})
	for _, enc := range encTypes {
		if enc == "gzip" || enc == "deflate" {
			return enc
		}
	}
	return ""
}

// Select method selects the correct compression encoder based on http HEADER.
func (_ *CompressionSelector) Select(r *http.Request) Encoder {
	switch acceptedEnc(r) {
	case "gzip":
		return &gzipEncoder{}
	case "flate":
		return &flateEncoder{}
	}
	return DefaultEncoder
}
