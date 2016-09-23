package restful

// Copyright 2015 Ernest Micklei. All rights reserved.
// Use of this source code is governed by a license
// that can be found in the LICENSE file.

import (
	"bytes"
	"compress/gzip"
	"compress/zlib"
	"sync"
)

// SyncPoolCompessors is a CompressorProvider that use the standard sync.Pool.
type SyncPoolCompessors struct {
	GzipWriterPool *sync.Pool
	GzipReaderPool *sync.Pool
	ZlibWriterPool *sync.Pool
}

// NewSyncPoolCompessors returns a new ("empty") SyncPoolCompessors.
func NewSyncPoolCompessors() *SyncPoolCompessors {
	return &SyncPoolCompessors{
		GzipWriterPool: &sync.Pool{
			New: func() interface{} { return newGzipWriter() },
		},
		GzipReaderPool: &sync.Pool{
			New: func() interface{} { return newGzipReader() },
		},
		ZlibWriterPool: &sync.Pool{
			New: func() interface{} { return newZlibWriter() },
		},
	}
}

func (s *SyncPoolCompessors) AcquireGzipWriter() *gzip.Writer {
	return s.GzipWriterPool.Get().(*gzip.Writer)
}

func (s *SyncPoolCompessors) ReleaseGzipWriter(w *gzip.Writer) {
	s.GzipWriterPool.Put(w)
}

func (s *SyncPoolCompessors) AcquireGzipReader() *gzip.Reader {
	return s.GzipReaderPool.Get().(*gzip.Reader)
}

func (s *SyncPoolCompessors) ReleaseGzipReader(r *gzip.Reader) {
	s.GzipReaderPool.Put(r)
}

func (s *SyncPoolCompessors) AcquireZlibWriter() *zlib.Writer {
	return s.ZlibWriterPool.Get().(*zlib.Writer)
}

func (s *SyncPoolCompessors) ReleaseZlibWriter(w *zlib.Writer) {
	s.ZlibWriterPool.Put(w)
}

func newGzipWriter() *gzip.Writer {
	// create with an empty bytes writer; it will be replaced before using the gzipWriter
	writer, err := gzip.NewWriterLevel(new(bytes.Buffer), gzip.BestSpeed)
	if err != nil {
		panic(err.Error())
	}
	return writer
}

func newGzipReader() *gzip.Reader {
	// create with an empty reader (but with GZIP header); it will be replaced before using the gzipReader
	// we can safely use currentCompressProvider because it is set on package initialization.
	w := currentCompressorProvider.AcquireGzipWriter()
	defer currentCompressorProvider.ReleaseGzipWriter(w)
	b := new(bytes.Buffer)
	w.Reset(b)
	w.Flush()
	w.Close()
	reader, err := gzip.NewReader(bytes.NewReader(b.Bytes()))
	if err != nil {
		panic(err.Error())
	}
	return reader
}

func newZlibWriter() *zlib.Writer {
	writer, err := zlib.NewWriterLevel(new(bytes.Buffer), gzip.BestSpeed)
	if err != nil {
		panic(err.Error())
	}
	return writer
}
