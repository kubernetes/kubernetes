package restful

// Copyright 2015 Ernest Micklei. All rights reserved.
// Use of this source code is governed by a license
// that can be found in the LICENSE file.

import (
	"compress/gzip"
	"compress/zlib"
)

// CompressorProvider describes a component that can provider compressors for the std methods.
type CompressorProvider interface {
	// Returns a *gzip.Writer which needs to be released later.
	// Before using it, call Reset().
	AcquireGzipWriter() *gzip.Writer

	// Releases an acquired *gzip.Writer.
	ReleaseGzipWriter(w *gzip.Writer)

	// Returns a *gzip.Reader which needs to be released later.
	AcquireGzipReader() *gzip.Reader

	// Releases an acquired *gzip.Reader.
	ReleaseGzipReader(w *gzip.Reader)

	// Returns a *zlib.Writer which needs to be released later.
	// Before using it, call Reset().
	AcquireZlibWriter() *zlib.Writer

	// Releases an acquired *zlib.Writer.
	ReleaseZlibWriter(w *zlib.Writer)
}

// DefaultCompressorProvider is the actual provider of compressors (zlib or gzip).
var currentCompressorProvider CompressorProvider

func init() {
	currentCompressorProvider = NewSyncPoolCompessors()
}

// CurrentCompressorProvider returns the current CompressorProvider.
// It is initialized using a SyncPoolCompessors.
func CurrentCompressorProvider() CompressorProvider {
	return currentCompressorProvider
}

// SetCompressorProvider sets the actual provider of compressors (zlib or gzip).
func SetCompressorProvider(p CompressorProvider) {
	if p == nil {
		panic("cannot set compressor provider to nil")
	}
	currentCompressorProvider = p
}
