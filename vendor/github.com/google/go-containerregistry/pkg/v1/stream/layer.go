// Copyright 2018 Google LLC All Rights Reserved.
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

// Package stream implements a single-pass streaming v1.Layer.
package stream

import (
	"bufio"
	"compress/gzip"
	"crypto"
	"encoding/hex"
	"errors"
	"hash"
	"io"
	"os"
	"sync"

	v1 "github.com/google/go-containerregistry/pkg/v1"
	"github.com/google/go-containerregistry/pkg/v1/types"
)

var (
	// ErrNotComputed is returned when the requested value is not yet
	// computed because the stream has not been consumed yet.
	ErrNotComputed = errors.New("value not computed until stream is consumed")

	// ErrConsumed is returned by Compressed when the underlying stream has
	// already been consumed and closed.
	ErrConsumed = errors.New("stream was already consumed")
)

// Layer is a streaming implementation of v1.Layer.
type Layer struct {
	blob        io.ReadCloser
	consumed    bool
	compression int

	mu             sync.Mutex
	digest, diffID *v1.Hash
	size           int64
	mediaType      types.MediaType
}

var _ v1.Layer = (*Layer)(nil)

// LayerOption applies options to layer
type LayerOption func(*Layer)

// WithCompressionLevel sets the gzip compression. See `gzip.NewWriterLevel` for possible values.
func WithCompressionLevel(level int) LayerOption {
	return func(l *Layer) {
		l.compression = level
	}
}

// WithMediaType is a functional option for overriding the layer's media type.
func WithMediaType(mt types.MediaType) LayerOption {
	return func(l *Layer) {
		l.mediaType = mt
	}
}

// NewLayer creates a Layer from an io.ReadCloser.
func NewLayer(rc io.ReadCloser, opts ...LayerOption) *Layer {
	layer := &Layer{
		blob:        rc,
		compression: gzip.BestSpeed,
		// We use DockerLayer for now as uncompressed layers
		// are unimplemented
		mediaType: types.DockerLayer,
	}

	for _, opt := range opts {
		opt(layer)
	}

	return layer
}

// Digest implements v1.Layer.
func (l *Layer) Digest() (v1.Hash, error) {
	l.mu.Lock()
	defer l.mu.Unlock()
	if l.digest == nil {
		return v1.Hash{}, ErrNotComputed
	}
	return *l.digest, nil
}

// DiffID implements v1.Layer.
func (l *Layer) DiffID() (v1.Hash, error) {
	l.mu.Lock()
	defer l.mu.Unlock()
	if l.diffID == nil {
		return v1.Hash{}, ErrNotComputed
	}
	return *l.diffID, nil
}

// Size implements v1.Layer.
func (l *Layer) Size() (int64, error) {
	l.mu.Lock()
	defer l.mu.Unlock()
	if l.size == 0 {
		return 0, ErrNotComputed
	}
	return l.size, nil
}

// MediaType implements v1.Layer
func (l *Layer) MediaType() (types.MediaType, error) {
	return l.mediaType, nil
}

// Uncompressed implements v1.Layer.
func (l *Layer) Uncompressed() (io.ReadCloser, error) {
	return nil, errors.New("NYI: stream.Layer.Uncompressed is not implemented")
}

// Compressed implements v1.Layer.
func (l *Layer) Compressed() (io.ReadCloser, error) {
	l.mu.Lock()
	defer l.mu.Unlock()
	if l.consumed {
		return nil, ErrConsumed
	}
	return newCompressedReader(l)
}

// finalize sets the layer to consumed and computes all hash and size values.
func (l *Layer) finalize(uncompressed, compressed hash.Hash, size int64) error {
	l.mu.Lock()
	defer l.mu.Unlock()

	diffID, err := v1.NewHash("sha256:" + hex.EncodeToString(uncompressed.Sum(nil)))
	if err != nil {
		return err
	}
	l.diffID = &diffID

	digest, err := v1.NewHash("sha256:" + hex.EncodeToString(compressed.Sum(nil)))
	if err != nil {
		return err
	}
	l.digest = &digest

	l.size = size
	l.consumed = true
	return nil
}

type compressedReader struct {
	pr     io.Reader
	closer func() error
}

func newCompressedReader(l *Layer) (*compressedReader, error) {
	// Collect digests of compressed and uncompressed stream and size of
	// compressed stream.
	h := crypto.SHA256.New()
	zh := crypto.SHA256.New()
	count := &countWriter{}

	// gzip.Writer writes to the output stream via pipe, a hasher to
	// capture compressed digest, and a countWriter to capture compressed
	// size.
	pr, pw := io.Pipe()

	// Write compressed bytes to be read by the pipe.Reader, hashed by zh, and counted by count.
	mw := io.MultiWriter(pw, zh, count)

	// Buffer the output of the gzip writer so we don't have to wait on pr to keep writing.
	// 64K ought to be small enough for anybody.
	bw := bufio.NewWriterSize(mw, 2<<16)
	zw, err := gzip.NewWriterLevel(bw, l.compression)
	if err != nil {
		return nil, err
	}

	doneDigesting := make(chan struct{})

	cr := &compressedReader{
		pr: pr,
		closer: func() error {
			// Immediately close pw without error. There are three ways to get
			// here.
			//
			// 1. There was a copy error due from the underlying reader, in which
			//    case the error will not be overwritten.
			// 2. Copying from the underlying reader completed successfully.
			// 3. Close has been called before the underlying reader has been
			//    fully consumed. In this case pw must be closed in order to
			//    keep the flush of bw from blocking indefinitely.
			//
			// NOTE: pw.Close never returns an error. The signature is only to
			// implement io.Closer.
			_ = pw.Close()

			// Close the inner ReadCloser.
			//
			// NOTE: net/http will call close on success, so if we've already
			// closed the inner rc, it's not an error.
			if err := l.blob.Close(); err != nil && !errors.Is(err, os.ErrClosed) {
				return err
			}

			// Finalize layer with its digest and size values.
			<-doneDigesting
			return l.finalize(h, zh, count.n)
		},
	}
	go func() {
		// Copy blob into the gzip writer, which also hashes and counts the
		// size of the compressed output, and hasher of the raw contents.
		_, copyErr := io.Copy(io.MultiWriter(h, zw), l.blob)

		// Close the gzip writer once copying is done. If this is done in the
		// Close method of compressedReader instead, then it can cause a panic
		// when the compressedReader is closed before the blob is fully
		// consumed and io.Copy in this goroutine is still blocking.
		closeErr := zw.Close()

		// Check errors from writing and closing streams.
		if copyErr != nil {
			close(doneDigesting)
			pw.CloseWithError(copyErr)
			return
		}
		if closeErr != nil {
			close(doneDigesting)
			pw.CloseWithError(closeErr)
			return
		}

		// Flush the buffer once all writes are complete to the gzip writer.
		if err := bw.Flush(); err != nil {
			close(doneDigesting)
			pw.CloseWithError(err)
			return
		}

		// Notify closer that digests are done being written.
		close(doneDigesting)

		// Close the compressed reader to calculate digest/diffID/size. This
		// will cause pr to return EOF which will cause readers of the
		// Compressed stream to finish reading.
		pw.CloseWithError(cr.Close())
	}()

	return cr, nil
}

func (cr *compressedReader) Read(b []byte) (int, error) { return cr.pr.Read(b) }

func (cr *compressedReader) Close() error { return cr.closer() }

// countWriter counts bytes written to it.
type countWriter struct{ n int64 }

func (c *countWriter) Write(p []byte) (int, error) {
	c.n += int64(len(p))
	return len(p), nil
}
