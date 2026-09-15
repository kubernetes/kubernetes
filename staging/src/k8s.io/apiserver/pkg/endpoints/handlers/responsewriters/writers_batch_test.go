/*
Copyright The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package responsewriters

import (
	"bytes"
	"compress/gzip"
	"context"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

// countingResponseWriter is an httptest.ResponseRecorder that counts the
// Write calls reaching the network-facing response writer (hw), as opposed
// to the encoder's writes into deferredResponseWriter, and can inject write
// failures.
type countingResponseWriter struct {
	*httptest.ResponseRecorder
	writeCount int
	failWrites bool
}

func newCountingResponseWriter() *countingResponseWriter {
	return &countingResponseWriter{ResponseRecorder: httptest.NewRecorder()}
}

func (w *countingResponseWriter) Write(p []byte) (int, error) {
	w.writeCount++
	if w.failWrites {
		return 0, errors.New("connection reset by peer")
	}
	return w.ResponseRecorder.Write(p)
}

func newDeferredWriter(hw http.ResponseWriter, mediaType, contentEncoding string) *deferredResponseWriter {
	return &deferredResponseWriter{
		mediaType:       mediaType,
		statusCode:      http.StatusOK,
		contentEncoding: contentEncoding,
		hw:              hw,
		ctx:             context.Background(),
	}
}

// streamWrite simulates a streaming collection encoder: a small opening
// write, one write per item, and an optional closing write. It returns the
// concatenation the response body must equal.
func streamWrite(t testing.TB, w io.Writer, opener, item, closer []byte, items int) []byte {
	t.Helper()
	var want bytes.Buffer
	writeBoth := func(p []byte) {
		want.Write(p)
		if _, err := w.Write(p); err != nil {
			t.Fatalf("write failed: %v", err)
		}
	}
	writeBoth(opener)
	for range items {
		writeBoth(item)
	}
	if len(closer) > 0 {
		writeBoth(closer)
	}
	return want.Bytes()
}

// varyingItem returns a ~1KiB item whose content differs per index, so a
// stream of them compresses the way real lists do (a few x) instead of
// collapsing to almost nothing.
func varyingItem(i int) []byte {
	var b strings.Builder
	x := uint64(i)*0x9E3779B97F4A7C15 | 1
	for b.Len() < 1024 {
		x ^= x << 13
		x ^= x >> 7
		x ^= x << 17
		fmt.Fprintf(&b, `{"name":"pod-%d","uid":"%016x","phase":"Running"},`, i, x)
	}
	return []byte(b.String())
}

// TestStreamedResponseBatchesWrites pins the lazy engagement contract and
// byte-identity for identity responses: a streamed response (JSON's "{" or
// the protobuf stream header first) reaches the response writer write for
// write until streamingBatchEngageThresholdBytes, then in buffer-sized
// batches; a streamed response under the threshold, and a single-object
// response, are never batched.
func TestStreamedResponseBatchesWrites(t *testing.T) {
	jsonOpener, protobufOpener := []byte("{"), []byte{0x6b, 0x38, 0x73, 0x00}
	const itemSize = 1024
	tests := []struct {
		name    string
		opener  []byte // nil: a single object, written in one call
		items   int
		batched bool
	}{
		{name: "large JSON stream engages after threshold", opener: jsonOpener, items: 3000, batched: true},
		{name: "large protobuf stream engages after threshold", opener: protobufOpener, items: 3000, batched: true},
		{name: "small stream stays direct", opener: jsonOpener, items: streamingBatchEngageThresholdBytes/itemSize - 16},
		{name: "single object stays direct", items: 0},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			rec := newCountingResponseWriter()
			drw := newDeferredWriter(rec, "application/json", "")
			var want []byte
			if tc.opener == nil {
				want = bytes.Repeat([]byte("y"), 512*1024)
				if _, err := drw.Write(want); err != nil {
					t.Fatalf("write failed: %v", err)
				}
			} else {
				want = streamWrite(t, drw, tc.opener, bytes.Repeat([]byte("x"), itemSize), []byte("}"), tc.items)
			}
			if err := drw.Close(); err != nil {
				t.Fatalf("close failed: %v", err)
			}
			if !bytes.Equal(rec.Body.Bytes(), want) {
				t.Errorf("body differs from what was written: got %d bytes, want %d", rec.Body.Len(), len(want))
			}
			// Unbatched, every write reaches the response writer; batched, the
			// writes before the threshold do and the rest arrive in
			// buffer-sized flushes (+2 slack for the opener and closer).
			directWrites := 1
			if tc.opener != nil {
				directWrites = tc.items + 2
			}
			minWrites, maxWrites := directWrites, directWrites
			if tc.batched {
				minWrites = streamingBatchEngageThresholdBytes / itemSize
				maxWrites = minWrites + 2 + tc.items*itemSize/streamingBatchBufferBytes + 2
			}
			if rec.writeCount < minWrites || rec.writeCount > maxWrites {
				t.Errorf("response reached the response writer in %d writes, want %d..%d", rec.writeCount, minWrites, maxWrites)
			}
		})
	}
}

// TestStreamedGzipResponseEngagesLazily pins that batching is lazy under
// gzip too: the first compressed bytes reach the response writer as soon as
// the gzip threshold commits the response, not once a batch buffer of
// compressed output has filled, and the batched body still decompresses to
// what was written.
func TestStreamedGzipResponseEngagesLazily(t *testing.T) {
	rec := newCountingResponseWriter()
	drw := newDeferredWriter(rec, "application/json", "gzip")
	var want bytes.Buffer
	firstByteAt := -1 // plaintext bytes written when compressed output first reached rec
	write := func(p []byte) {
		want.Write(p)
		if _, err := drw.Write(p); err != nil {
			t.Fatalf("write failed: %v", err)
		}
		if firstByteAt < 0 && rec.Body.Len() > 0 {
			firstByteAt = want.Len()
		}
	}
	write([]byte("{"))
	const items = 4096 // ~4MiB of plaintext, ~1.5MiB compressed
	for i := range items {
		write(varyingItem(i))
	}
	if err := drw.Close(); err != nil {
		t.Fatalf("close failed: %v", err)
	}

	if firstByteAt < 0 || firstByteAt > defaultGzipThresholdBytes+2*1024 {
		t.Errorf("first compressed bytes reached the response writer after %d bytes of plaintext, want at commit (~%d)", firstByteAt, defaultGzipThresholdBytes)
	}
	// Past the threshold the compressor's small writes are batched, so far
	// fewer writes reach the response writer than items were written.
	if rec.writeCount >= items {
		t.Errorf("%d writes reached the response writer for %d items, want batching", rec.writeCount, items)
	}
	zr, err := gzip.NewReader(bytes.NewReader(rec.Body.Bytes()))
	if err != nil {
		t.Fatalf("body is not valid gzip: %v", err)
	}
	if got, err := io.ReadAll(zr); err != nil || !bytes.Equal(got, want.Bytes()) {
		t.Fatalf("decompressed body differs from streamed input (err=%v): got %d bytes, want %d", err, len(got), want.Len())
	}
}

// TestStreamedResponseFlushErrorSurfaces pins that once batching is engaged,
// a network write failure discovered at Close's flush is returned to the
// caller (and recorded like any other write error) rather than swallowed,
// under both encodings.
func TestStreamedResponseFlushErrorSurfaces(t *testing.T) {
	for _, contentEncoding := range []string{"", "gzip"} {
		t.Run("contentEncoding="+contentEncoding, func(t *testing.T) {
			rec := newCountingResponseWriter()
			drw := newDeferredWriter(rec, "application/json", contentEncoding)
			if _, err := drw.Write([]byte("{")); err != nil {
				t.Fatalf("write failed: %v", err)
			}
			// Stream while the connection is healthy until batching engages...
			for i := 0; drw.batcher.batch == nil; i++ {
				if i > 8192 {
					t.Fatal("batching never engaged")
				}
				if _, err := drw.Write(varyingItem(i)); err != nil {
					t.Fatalf("write failed: %v", err)
				}
			}
			// ...then break the connection: the next write lands in the batch
			// buffer, and only the Close-time flush can discover the failure.
			rec.failWrites = true
			if _, err := drw.Write(varyingItem(0)); err != nil {
				t.Fatalf("unexpected write error before flush: %v", err)
			}
			if err := drw.Close(); err == nil {
				t.Fatal("Close succeeded despite the underlying writer failing")
			}
			if drw.lastWriteErr == nil {
				t.Error("flush failure not recorded in lastWriteErr")
			}
		})
	}
}
