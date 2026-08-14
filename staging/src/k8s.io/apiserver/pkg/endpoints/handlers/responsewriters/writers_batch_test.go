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
	"crypto/tls"
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

// TestStreamedResponseBatchesWrites pins the lazy engagement contract for
// identity responses: streamed output reaches the response writer directly
// until streamingBatchEngageThresholdBytes, then in large batches; responses below
// the threshold, and single-object responses, are never batched. The body is
// byte-identical throughout.
func TestStreamedResponseBatchesWrites(t *testing.T) {
	t.Run("large streamed engages after threshold", func(t *testing.T) {
		const itemSize, items = 1024, 3000
		rec := newCountingResponseWriter()
		drw := newDeferredWriter(rec, "application/json", "")
		want := streamWrite(t, drw, []byte("{"), bytes.Repeat([]byte("x"), itemSize), []byte("}"), items)
		if err := drw.Close(); err != nil {
			t.Fatalf("close failed: %v", err)
		}
		if !bytes.Equal(rec.Body.Bytes(), want) {
			t.Errorf("batched body differs from streamed input: got %d bytes, want %d", rec.Body.Len(), len(want))
		}
		// Direct writes until the threshold crosses, then buffer-sized
		// flushes for the remainder (+small slack for the opener/closer).
		minWrites := streamingBatchEngageThresholdBytes / itemSize
		maxWrites := minWrites + 2 + items*itemSize/streamingBatchBufferBytes + 2
		if rec.writeCount > maxWrites {
			t.Errorf("large streamed response reached the response writer in %d writes, want batching after the threshold (<=%d)", rec.writeCount, maxWrites)
		}
		if rec.writeCount < minWrites {
			t.Errorf("large streamed response reached the response writer in %d writes, want the threshold prefix unbatched (>=%d)", rec.writeCount, minWrites)
		}
	})
	t.Run("small streamed stays direct", func(t *testing.T) {
		// Comfortably under streamingBatchEngageThresholdBytes: every write goes
		// straight through, nothing is held back for Close.
		const itemSize = 1024
		items := streamingBatchEngageThresholdBytes/itemSize - 16
		wantWrites := items + 2 // opener + items + closer
		rec := newCountingResponseWriter()
		drw := newDeferredWriter(rec, "application/json", "")
		want := streamWrite(t, drw, []byte("{"), bytes.Repeat([]byte("x"), itemSize), []byte("}"), items)
		if rec.writeCount != wantWrites {
			t.Errorf("small streamed response saw %d underlying writes before Close, want %d (fully direct)", rec.writeCount, wantWrites)
		}
		if err := drw.Close(); err != nil {
			t.Fatalf("close failed: %v", err)
		}
		if !bytes.Equal(rec.Body.Bytes(), want) {
			t.Errorf("body differs: got %d bytes, want %d", rec.Body.Len(), len(want))
		}
		if rec.writeCount != wantWrites {
			t.Errorf("small streamed response took %d writes total, want %d (no batching engaged)", rec.writeCount, wantWrites)
		}
	})
	t.Run("single object stays direct", func(t *testing.T) {
		rec := newCountingResponseWriter()
		drw := newDeferredWriter(rec, "application/json", "")
		body := strings.Repeat("y", 512*1024)
		if _, err := io.WriteString(drw, body); err != nil {
			t.Fatalf("write failed: %v", err)
		}
		if err := drw.Close(); err != nil {
			t.Fatalf("close failed: %v", err)
		}
		if rec.Body.String() != body {
			t.Errorf("body altered: got %d bytes, want %d", rec.Body.Len(), len(body))
		}
		if rec.writeCount != 1 {
			t.Errorf("single-object response took %d writes, want 1 (no batching)", rec.writeCount)
		}
	})
}

// TestStreamedProtobufDetection pins that the 4-byte protobuf stream header
// (the first write of streamed protobuf collections) is classified as
// streamed and batched by the same lazy rule as JSON's "{".
func TestStreamedProtobufDetection(t *testing.T) {
	rec := newCountingResponseWriter()
	drw := newDeferredWriter(rec, "application/vnd.kubernetes.protobuf", "")
	magic := []byte{0x6b, 0x38, 0x73, 0x00}
	const itemSize, items = 2048, 1000
	want := streamWrite(t, drw, magic, bytes.Repeat([]byte{0x01}, itemSize), nil, items)
	if err := drw.Close(); err != nil {
		t.Fatalf("close failed: %v", err)
	}
	if !bytes.Equal(rec.Body.Bytes(), want) {
		t.Errorf("batched protobuf body differs from streamed input")
	}
	minWrites := streamingBatchEngageThresholdBytes / itemSize
	maxWrites := minWrites + 2 + items*itemSize/streamingBatchBufferBytes + 2
	if rec.writeCount > maxWrites {
		t.Errorf("streamed protobuf response reached the response writer in %d writes, want batching after the threshold (<=%d)", rec.writeCount, maxWrites)
	}
	if rec.writeCount < minWrites {
		t.Errorf("streamed protobuf response reached the response writer in %d writes, want the threshold prefix unbatched (>=%d)", rec.writeCount, minWrites)
	}
}

// TestStreamedGzipResponseEngagesLazily pins that gzip responses follow the
// same lazy rule as identity ones: the first compressed bytes reach the
// response writer as soon as the gzip threshold commits the response (so
// headers and time-to-first-byte are untouched), the compressor's small
// writes pass straight through until streamingBatchEngageThresholdBytes of them
// have gone out, and batching engages only then, in streamingBatchBufferBytes
// flushes.
func TestStreamedGzipResponseEngagesLazily(t *testing.T) {
	const items = 4096 // ~4MiB of plaintext, ~1.5MiB compressed
	rec := newCountingResponseWriter()
	drw := newDeferredWriter(rec, "application/json", "gzip")
	var want bytes.Buffer
	write := func(p []byte) {
		want.Write(p)
		if _, err := drw.Write(p); err != nil {
			t.Fatalf("write failed: %v", err)
		}
	}
	write([]byte("{"))
	firstBytePlaintext, directWrites, directBytes := -1, -1, 0
	for i := range items {
		write(varyingItem(i))
		if firstBytePlaintext < 0 && rec.Body.Len() > 0 {
			firstBytePlaintext = want.Len()
		}
		if directWrites < 0 && rec.Body.Len() >= streamingBatchEngageThresholdBytes {
			// Everything so far went straight through; the next compressed
			// write engages the batch buffer.
			directWrites, directBytes = rec.writeCount, rec.Body.Len()
		}
	}
	if err := drw.Close(); err != nil {
		t.Fatalf("close failed: %v", err)
	}
	if got := rec.Header().Get("Content-Encoding"); got != "gzip" {
		t.Fatalf("Content-Encoding = %q, want gzip", got)
	}
	zr, err := gzip.NewReader(bytes.NewReader(rec.Body.Bytes()))
	if err != nil {
		t.Fatalf("body is not valid gzip: %v", err)
	}
	if got, err := io.ReadAll(zr); err != nil || !bytes.Equal(got, want.Bytes()) {
		t.Fatalf("decompressed body differs from streamed input (err=%v): got %d bytes, want %d", err, len(got), want.Len())
	}

	// First bytes leave when the gzip threshold commits the response, not
	// when a batch buffer of compressed output has filled.
	if firstBytePlaintext < 0 || firstBytePlaintext > defaultGzipThresholdBytes+2*1024 {
		t.Errorf("first compressed bytes reached the response writer after %d bytes of plaintext, want at commit (~%d)", firstBytePlaintext, defaultGzipThresholdBytes)
	}
	if directWrites < 0 {
		t.Fatalf("compressed output never reached streamingBatchEngageThresholdBytes directly (%d bytes total)", rec.Body.Len())
	}
	// The pass-through prefix arrives as the compressor's own small writes;
	// a batcher engaging early would have delivered it in a flush or two.
	if fewest := directBytes/streamingBatchBufferBytes + 2; directWrites <= fewest {
		t.Errorf("first %d compressed bytes reached the response writer in %d writes, want the compressor's writes passed through unbatched (>%d)", directBytes, directWrites, fewest)
	}
	batchedBytes := rec.Body.Len() - directBytes
	batchedWrites := rec.writeCount - directWrites
	if batchedBytes < 2*streamingBatchBufferBytes {
		t.Fatalf("test body too small to exercise batching: %d compressed bytes after the threshold", batchedBytes)
	}
	if maxWrites := batchedBytes/streamingBatchBufferBytes + 2; batchedWrites > maxWrites {
		t.Errorf("after the threshold, %d compressed bytes reached the response writer in %d writes, want batching (<=%d)", batchedBytes, batchedWrites, maxWrites)
	}
}

// TestStreamedGzipResponseBelowThresholdStaysDirect pins that a gzip response
// whose compressed output stays under streamingBatchEngageThresholdBytes is
// compressed exactly as before and never takes a pooled batch buffer.
func TestStreamedGzipResponseBelowThresholdStaysDirect(t *testing.T) {
	rec := newCountingResponseWriter()
	drw := newDeferredWriter(rec, "application/json", "gzip")
	// ~300KiB streamed as 100 item writes: over defaultGzipThresholdBytes, so
	// compression commits, but trivially compressible, so the compressed
	// output stays far below the batching threshold.
	want := streamWrite(t, drw, []byte("{"), bytes.Repeat([]byte("x"), 3*1024), []byte("}"), 100)
	if drw.batcher.batch != nil {
		t.Error("gzip response under the batching threshold took a pooled batch buffer")
	}
	if err := drw.Close(); err != nil {
		t.Fatalf("close failed: %v", err)
	}
	if got := rec.Header().Get("Content-Encoding"); got != "gzip" {
		t.Fatalf("streamed response lost compression: Content-Encoding = %q, body %d bytes", got, rec.Body.Len())
	}
	if rec.Body.Len() >= streamingBatchEngageThresholdBytes {
		t.Fatalf("test body compressed to %d bytes, want under the %d threshold", rec.Body.Len(), streamingBatchEngageThresholdBytes)
	}
	zr, err := gzip.NewReader(rec.Body)
	if err != nil {
		t.Fatalf("body is not valid gzip: %v", err)
	}
	got, err := io.ReadAll(zr)
	if err != nil {
		t.Fatalf("decompress failed: %v", err)
	}
	if !bytes.Equal(got, want) {
		t.Errorf("decompressed body differs from streamed input: got %d bytes, want %d", len(got), len(want))
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

// BenchmarkStreamedResponseTransport measures a streamed list response end to
// end over a loopback TLS connection, per protocol, with and without
// batching: one ~10MiB response of 10KiB item writes per iteration, served
// through net/http's real HTTP/2 and HTTP/1.1 response writers.
func BenchmarkStreamedResponseTransport(b *testing.B) {
	item := []byte(strings.Repeat("x", 10*1024))
	const items = 1000
	handler := http.HandlerFunc(func(hw http.ResponseWriter, r *http.Request) {
		drw := newDeferredWriter(hw, "application/json", "")
		drw.ctx = r.Context()
		if _, err := drw.Write([]byte("{")); err != nil {
			b.Error(err)
			return
		}
		for range items {
			if _, err := drw.Write(item); err != nil {
				b.Error(err)
				return
			}
		}
		if err := drw.Close(); err != nil {
			b.Error(err)
		}
	})
	for _, proto := range []string{"http2", "http1.1"} {
		for _, batching := range []bool{true, false} {
			name := fmt.Sprintf("%s/batched", proto)
			if !batching {
				name = fmt.Sprintf("%s/unbatched", proto)
			}
			b.Run(name, func(b *testing.B) {
				prev := streamedWriteBatching
				streamedWriteBatching = batching
				defer func() { streamedWriteBatching = prev }()
				srv := httptest.NewUnstartedServer(handler)
				srv.EnableHTTP2 = proto == "http2"
				if !srv.EnableHTTP2 {
					srv.TLS = &tls.Config{NextProtos: []string{"http/1.1"}}
				}
				srv.StartTLS()
				defer srv.Close()
				client := srv.Client()
				b.SetBytes(int64(1 + items*len(item)))
				b.ReportAllocs()
				b.ResetTimer()
				for range b.N {
					resp, err := client.Get(srv.URL)
					if err != nil {
						b.Fatal(err)
					}
					if resp.ProtoMajor == 2 != srv.EnableHTTP2 {
						b.Fatalf("negotiated %s, want %s", resp.Proto, proto)
					}
					if _, err := io.Copy(io.Discard, resp.Body); err != nil {
						b.Fatal(err)
					}
					resp.Body.Close()
				}
			})
		}
	}
}
