// Copyright 2020 Google LLC All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Package gzip provides helper functions for interacting with gzipped streams.
package gzip

import (
	"bufio"
	"bytes"
	"compress/gzip"
	"io"

	"github.com/google/go-containerregistry/internal/and"
)

var gzipMagicHeader = []byte{'\x1f', '\x8b'}

// ReadCloser reads uncompressed input data from the io.ReadCloser and
// returns an io.ReadCloser from which compressed data may be read.
// This uses gzip.BestSpeed for the compression level.
func ReadCloser(r io.ReadCloser) io.ReadCloser {
	return ReadCloserLevel(r, gzip.BestSpeed)
}

// ReadCloserLevel reads uncompressed input data from the io.ReadCloser and
// returns an io.ReadCloser from which compressed data may be read.
// Refer to compress/gzip for the level:
// https://golang.org/pkg/compress/gzip/#pkg-constants
func ReadCloserLevel(r io.ReadCloser, level int) io.ReadCloser {
	pr, pw := io.Pipe()

	// For highly compressible layers, gzip.Writer will output a very small
	// number of bytes per Write(). This is normally fine, but when pushing
	// to a registry, we want to ensure that we're taking full advantage of
	// the available bandwidth instead of sending tons of tiny writes over
	// the wire.
	// 64K ought to be small enough for anybody.
	bw := bufio.NewWriterSize(pw, 2<<16)

	// Returns err so we can pw.CloseWithError(err)
	go func() error {
		// TODO(go1.14): Just defer {pw,gw,r}.Close like you'd expect.
		// Context: https://golang.org/issue/24283
		gw, err := gzip.NewWriterLevel(bw, level)
		if err != nil {
			return pw.CloseWithError(err)
		}

		if _, err := io.Copy(gw, r); err != nil {
			defer r.Close()
			defer gw.Close()
			return pw.CloseWithError(err)
		}

		// Close gzip writer to Flush it and write gzip trailers.
		if err := gw.Close(); err != nil {
			return pw.CloseWithError(err)
		}

		// Flush bufio writer to ensure we write out everything.
		if err := bw.Flush(); err != nil {
			return pw.CloseWithError(err)
		}

		// We don't really care if these fail.
		defer pw.Close()
		defer r.Close()

		return nil
	}()

	return pr
}

// UnzipReadCloser reads compressed input data from the io.ReadCloser and
// returns an io.ReadCloser from which uncompessed data may be read.
func UnzipReadCloser(r io.ReadCloser) (io.ReadCloser, error) {
	gr, err := gzip.NewReader(r)
	if err != nil {
		return nil, err
	}
	return &and.ReadCloser{
		Reader: gr,
		CloseFunc: func() error {
			// If the unzip fails, then this seems to return the same
			// error as the read.  We don't want this to interfere with
			// us closing the main ReadCloser, since this could leave
			// an open file descriptor (fails on Windows).
			gr.Close()
			return r.Close()
		},
	}, nil
}

// Is detects whether the input stream is compressed.
func Is(r io.Reader) (bool, error) {
	magicHeader := make([]byte, 2)
	n, err := r.Read(magicHeader)
	if n == 0 && err == io.EOF {
		return false, nil
	}
	if err != nil {
		return false, err
	}
	return bytes.Equal(magicHeader, gzipMagicHeader), nil
}
