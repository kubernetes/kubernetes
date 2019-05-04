// Copyright 2013 The Gorilla Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package handlers

import (
	"compress/flate"
	"compress/gzip"
	"io"
	"net/http"
	"strings"
)

type compressResponseWriter struct {
	io.Writer
	http.ResponseWriter
	http.Hijacker
	http.Flusher
	http.CloseNotifier
}

func (w *compressResponseWriter) WriteHeader(c int) {
	w.ResponseWriter.Header().Del("Content-Length")
	w.ResponseWriter.WriteHeader(c)
}

func (w *compressResponseWriter) Header() http.Header {
	return w.ResponseWriter.Header()
}

func (w *compressResponseWriter) Write(b []byte) (int, error) {
	h := w.ResponseWriter.Header()
	if h.Get("Content-Type") == "" {
		h.Set("Content-Type", http.DetectContentType(b))
	}
	h.Del("Content-Length")

	return w.Writer.Write(b)
}

type flusher interface {
	Flush() error
}

func (w *compressResponseWriter) Flush() {
	// Flush compressed data if compressor supports it.
	if f, ok := w.Writer.(flusher); ok {
		f.Flush()
	}
	// Flush HTTP response.
	if w.Flusher != nil {
		w.Flusher.Flush()
	}
}

// CompressHandler gzip compresses HTTP responses for clients that support it
// via the 'Accept-Encoding' header.
//
// Compressing TLS traffic may leak the page contents to an attacker if the
// page contains user input: http://security.stackexchange.com/a/102015/12208
func CompressHandler(h http.Handler) http.Handler {
	return CompressHandlerLevel(h, gzip.DefaultCompression)
}

// CompressHandlerLevel gzip compresses HTTP responses with specified compression level
// for clients that support it via the 'Accept-Encoding' header.
//
// The compression level should be gzip.DefaultCompression, gzip.NoCompression,
// or any integer value between gzip.BestSpeed and gzip.BestCompression inclusive.
// gzip.DefaultCompression is used in case of invalid compression level.
func CompressHandlerLevel(h http.Handler, level int) http.Handler {
	if level < gzip.DefaultCompression || level > gzip.BestCompression {
		level = gzip.DefaultCompression
	}

	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
	L:
		for _, enc := range strings.Split(r.Header.Get("Accept-Encoding"), ",") {
			switch strings.TrimSpace(enc) {
			case "gzip":
				w.Header().Set("Content-Encoding", "gzip")
				r.Header.Del("Accept-Encoding")
				w.Header().Add("Vary", "Accept-Encoding")

				gw, _ := gzip.NewWriterLevel(w, level)
				defer gw.Close()

				h, hok := w.(http.Hijacker)
				if !hok { /* w is not Hijacker... oh well... */
					h = nil
				}

				f, fok := w.(http.Flusher)
				if !fok {
					f = nil
				}

				cn, cnok := w.(http.CloseNotifier)
				if !cnok {
					cn = nil
				}

				w = &compressResponseWriter{
					Writer:         gw,
					ResponseWriter: w,
					Hijacker:       h,
					Flusher:        f,
					CloseNotifier:  cn,
				}

				break L
			case "deflate":
				w.Header().Set("Content-Encoding", "deflate")
				r.Header.Del("Accept-Encoding")
				w.Header().Add("Vary", "Accept-Encoding")

				fw, _ := flate.NewWriter(w, level)
				defer fw.Close()

				h, hok := w.(http.Hijacker)
				if !hok { /* w is not Hijacker... oh well... */
					h = nil
				}

				f, fok := w.(http.Flusher)
				if !fok {
					f = nil
				}

				cn, cnok := w.(http.CloseNotifier)
				if !cnok {
					cn = nil
				}

				w = &compressResponseWriter{
					Writer:         fw,
					ResponseWriter: w,
					Hijacker:       h,
					Flusher:        f,
					CloseNotifier:  cn,
				}

				break L
			}
		}

		h.ServeHTTP(w, r)
	})
}
