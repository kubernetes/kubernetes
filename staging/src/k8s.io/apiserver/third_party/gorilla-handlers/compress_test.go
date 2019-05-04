// Copyright 2013 The Gorilla Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package handlers

import (
	"bufio"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"strconv"
	"testing"
)

var contentType = "text/plain; charset=utf-8"

func compressedRequest(w *httptest.ResponseRecorder, compression string) {
	CompressHandler(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Length", strconv.Itoa(9*1024))
		w.Header().Set("Content-Type", contentType)
		for i := 0; i < 1024; i++ {
			io.WriteString(w, "Gorilla!\n")
		}
	})).ServeHTTP(w, &http.Request{
		Method: "GET",
		Header: http.Header{
			"Accept-Encoding": []string{compression},
		},
	})

}

func TestCompressHandlerNoCompression(t *testing.T) {
	w := httptest.NewRecorder()
	compressedRequest(w, "")
	if enc := w.HeaderMap.Get("Content-Encoding"); enc != "" {
		t.Errorf("wrong content encoding, got %q want %q", enc, "")
	}
	if ct := w.HeaderMap.Get("Content-Type"); ct != contentType {
		t.Errorf("wrong content type, got %q want %q", ct, contentType)
	}
	if w.Body.Len() != 1024*9 {
		t.Errorf("wrong len, got %d want %d", w.Body.Len(), 1024*9)
	}
	if l := w.HeaderMap.Get("Content-Length"); l != "9216" {
		t.Errorf("wrong content-length. got %q expected %d", l, 1024*9)
	}
}

func TestAcceptEncodingIsDropped(t *testing.T) {
	tCases := []struct {
		name,
		compression,
		expect string
		isPresent bool
	}{
		{
			"accept-encoding-gzip",
			"gzip",
			"",
			false,
		},
		{
			"accept-encoding-deflate",
			"deflate",
			"",
			false,
		},
		{
			"accept-encoding-gzip,deflate",
			"gzip,deflate",
			"",
			false,
		},
		{
			"accept-encoding-gzip,deflate,something",
			"gzip,deflate,something",
			"",
			false,
		},
		{
			"accept-encoding-unknown",
			"unknown",
			"unknown",
			true,
		},
	}

	for _, tCase := range tCases {
		ch := CompressHandler(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			acceptEnc := r.Header.Get("Accept-Encoding")
			if acceptEnc == "" && tCase.isPresent {
				t.Fatalf("%s: expected 'Accept-Encoding' header to be present but was not", tCase.name)
			}
			if acceptEnc != "" {
				if !tCase.isPresent {
					t.Fatalf("%s: expected 'Accept-Encoding' header to be dropped but was still present having value %q", tCase.name, acceptEnc)
				}
				if acceptEnc != tCase.expect {
					t.Fatalf("%s: expected 'Accept-Encoding' to be %q but was %q", tCase.name, tCase.expect, acceptEnc)
				}
			}
		}))

		w := httptest.NewRecorder()
		ch.ServeHTTP(w, &http.Request{
			Method: "GET",
			Header: http.Header{
				"Accept-Encoding": []string{tCase.compression},
			},
		})
	}
}

func TestCompressHandlerGzip(t *testing.T) {
	w := httptest.NewRecorder()
	compressedRequest(w, "gzip")
	if w.HeaderMap.Get("Content-Encoding") != "gzip" {
		t.Errorf("wrong content encoding, got %q want %q", w.HeaderMap.Get("Content-Encoding"), "gzip")
	}
	if w.HeaderMap.Get("Content-Type") != "text/plain; charset=utf-8" {
		t.Errorf("wrong content type, got %s want %s", w.HeaderMap.Get("Content-Type"), "text/plain; charset=utf-8")
	}
	if w.Body.Len() != 72 {
		t.Errorf("wrong len, got %d want %d", w.Body.Len(), 72)
	}
	if l := w.HeaderMap.Get("Content-Length"); l != "" {
		t.Errorf("wrong content-length. got %q expected %q", l, "")
	}
}

func TestCompressHandlerDeflate(t *testing.T) {
	w := httptest.NewRecorder()
	compressedRequest(w, "deflate")
	if w.HeaderMap.Get("Content-Encoding") != "deflate" {
		t.Fatalf("wrong content encoding, got %q want %q", w.HeaderMap.Get("Content-Encoding"), "deflate")
	}
	if w.HeaderMap.Get("Content-Type") != "text/plain; charset=utf-8" {
		t.Fatalf("wrong content type, got %s want %s", w.HeaderMap.Get("Content-Type"), "text/plain; charset=utf-8")
	}
	if w.Body.Len() != 54 {
		t.Fatalf("wrong len, got %d want %d", w.Body.Len(), 54)
	}
}

func TestCompressHandlerGzipDeflate(t *testing.T) {
	w := httptest.NewRecorder()
	compressedRequest(w, "gzip, deflate ")
	if w.HeaderMap.Get("Content-Encoding") != "gzip" {
		t.Fatalf("wrong content encoding, got %q want %q", w.HeaderMap.Get("Content-Encoding"), "gzip")
	}
	if w.HeaderMap.Get("Content-Type") != "text/plain; charset=utf-8" {
		t.Fatalf("wrong content type, got %s want %s", w.HeaderMap.Get("Content-Type"), "text/plain; charset=utf-8")
	}
}

type fullyFeaturedResponseWriter struct{}

// Header/Write/WriteHeader implement the http.ResponseWriter interface.
func (fullyFeaturedResponseWriter) Header() http.Header {
	return http.Header{}
}
func (fullyFeaturedResponseWriter) Write([]byte) (int, error) {
	return 0, nil
}
func (fullyFeaturedResponseWriter) WriteHeader(int) {}

// Flush implements the http.Flusher interface.
func (fullyFeaturedResponseWriter) Flush() {}

// Hijack implements the http.Hijacker interface.
func (fullyFeaturedResponseWriter) Hijack() (net.Conn, *bufio.ReadWriter, error) {
	return nil, nil, nil
}

// CloseNotify implements the http.CloseNotifier interface.
func (fullyFeaturedResponseWriter) CloseNotify() <-chan bool {
	return nil
}

func TestCompressHandlerPreserveInterfaces(t *testing.T) {
	// Compile time validation fullyFeaturedResponseWriter implements all the
	// interfaces we're asserting in the test case below.
	var (
		_ http.Flusher       = fullyFeaturedResponseWriter{}
		_ http.CloseNotifier = fullyFeaturedResponseWriter{}
		_ http.Hijacker      = fullyFeaturedResponseWriter{}
	)
	var h http.Handler = http.HandlerFunc(func(rw http.ResponseWriter, r *http.Request) {
		comp := r.Header.Get("Accept-Encoding")
		if _, ok := rw.(*compressResponseWriter); !ok {
			t.Fatalf("ResponseWriter wasn't wrapped by compressResponseWriter, got %T type", rw)
		}
		if _, ok := rw.(http.Flusher); !ok {
			t.Errorf("ResponseWriter lost http.Flusher interface for %q", comp)
		}
		if _, ok := rw.(http.CloseNotifier); !ok {
			t.Errorf("ResponseWriter lost http.CloseNotifier interface for %q", comp)
		}
		if _, ok := rw.(http.Hijacker); !ok {
			t.Errorf("ResponseWriter lost http.Hijacker interface for %q", comp)
		}
	})
	h = CompressHandler(h)
	var (
		rw fullyFeaturedResponseWriter
	)
	r, err := http.NewRequest("GET", "/", nil)
	if err != nil {
		t.Fatalf("Failed to create test request: %v", err)
	}
	r.Header.Set("Accept-Encoding", "gzip")
	h.ServeHTTP(rw, r)

	r.Header.Set("Accept-Encoding", "deflate")
	h.ServeHTTP(rw, r)
}
