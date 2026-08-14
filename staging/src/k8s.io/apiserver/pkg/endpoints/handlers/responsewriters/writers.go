/*
Copyright 2016 The Kubernetes Authors.

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
	"bufio"
	"compress/gzip"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strconv"
	"sync"
	"time"

	"go.opentelemetry.io/otel/attribute"

	"k8s.io/apiserver/pkg/features"

	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apiserver/pkg/audit"
	"k8s.io/apiserver/pkg/endpoints/handlers/negotiation"
	"k8s.io/apiserver/pkg/endpoints/metrics"
	"k8s.io/apiserver/pkg/endpoints/request"
	"k8s.io/apiserver/pkg/registry/rest"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/apiserver/pkg/util/flushwriter"
	"k8s.io/component-base/tracing"
	"k8s.io/streaming/pkg/httpstream/wsstream"
)

// StreamObject performs input stream negotiation from a ResourceStreamer and writes that to the response.
// If the client requests a websocket upgrade, negotiate for a websocket reader protocol (because many
// browser clients cannot easily handle binary streaming protocols).
func StreamObject(statusCode int, gv schema.GroupVersion, s runtime.NegotiatedSerializer, stream rest.ResourceStreamer, w http.ResponseWriter, req *http.Request) {
	out, flush, contentType, err := stream.InputStream(req.Context(), gv.String(), req.Header.Get("Accept"))
	if err != nil {
		ErrorNegotiated(err, s, gv, w, req)
		return
	}
	if out == nil {
		// No output provided - return StatusNoContent
		w.WriteHeader(http.StatusNoContent)
		return
	}
	defer out.Close()

	if wsstream.IsWebSocketRequest(req) {
		r := wsstream.NewReader(out, true, wsstream.NewDefaultReaderProtocols())
		if err := r.Copy(w, req); err != nil {
			utilruntime.HandleError(fmt.Errorf("error encountered while streaming results via websocket: %v", err))
		}
		return
	}

	if len(contentType) == 0 {
		contentType = "application/octet-stream"
	}
	w.Header().Set("Content-Type", contentType)
	w.WriteHeader(statusCode)
	// Flush headers, if possible
	if flusher, ok := w.(http.Flusher); ok {
		flusher.Flush()
	}
	writer := w.(io.Writer)
	if flush {
		writer = flushwriter.Wrap(w)
	}
	io.Copy(writer, out)
}

// SerializeObject renders an object in the content type negotiated by the client using the provided encoder.
// The context is optional and can be nil. This method will perform optional content compression if requested by
// a client and the feature gate for APIResponseCompression is enabled.
func SerializeObject(mediaType string, encoder runtime.Encoder, hw http.ResponseWriter, req *http.Request, statusCode int, object runtime.Object) {
	ctx := req.Context()
	ctx, span := tracing.Start(ctx, "SerializeObject",
		attribute.String("audit-id", audit.GetAuditIDTruncated(ctx)),
		attribute.String("method", req.Method),
		attribute.String("url", req.URL.Path),
		attribute.String("protocol", req.Proto),
		attribute.String("mediaType", mediaType),
		attribute.String("encoder", string(encoder.Identifier())))
	req = req.WithContext(ctx)
	defer span.End(5 * time.Second)

	w := &deferredResponseWriter{
		mediaType:       mediaType,
		statusCode:      statusCode,
		contentEncoding: responseContentEncodingSupported(req),
		hw:              hw,
		ctx:             ctx,
	}

	var memoryAllocator runtime.MemoryAllocator
	if encoderWithAllocator, supportsAllocator := encoder.(runtime.EncoderWithAllocator); supportsAllocator {
		memoryAllocator = runtime.AllocatorPool.Get().(*runtime.Allocator)
		encoder = runtime.NewEncoderWithAllocator(encoderWithAllocator, memoryAllocator)
	}
	if memoryAllocator != nil {
		defer runtime.AllocatorPool.Put(memoryAllocator)
	}

	err := encoder.Encode(object, w)
	if err == nil {
		err = w.Close()
		if err != nil {
			// we cannot write an error to the writer anymore as the Encode call was successful.
			utilruntime.HandleError(fmt.Errorf("apiserver was unable to close cleanly the response writer: %v", err))
		}
		return
	}

	// make a best effort to write the object if a failure is detected
	utilruntime.HandleError(fmt.Errorf("apiserver was unable to write a %s response: %w", w.mediaType, err))
	w.discardBufferedResponse()
	status := ErrorToAPIStatus(err)
	candidateStatusCode := int(status.Code)
	// if the current status code is successful, allow the error's status code to overwrite it
	if statusCode >= http.StatusOK && statusCode < http.StatusBadRequest {
		w.statusCode = candidateStatusCode
	}
	output, err := runtime.Encode(encoder, status)
	if err != nil {
		w.mediaType = "text/plain"
		output = []byte(fmt.Sprintf("%s: %s", status.Reason, status.Message))
	}
	if _, err := w.Write(output); err != nil {
		utilruntime.HandleError(fmt.Errorf("apiserver was unable to write a fallback %s response: %w", w.mediaType, err))
	}
	w.Close()
}

var gzipPool = NewGzipWriterPoolOrDie()

// batchBufferPool recycles lazyBatchWriter's batch buffers across responses.
var batchBufferPool = &sync.Pool{
	New: func() interface{} {
		return bufio.NewWriterSize(nil, streamingBatchBufferBytes)
	},
}

const (
	// defaultGzipThresholdBytes is compared to the size of the first write from the stream
	// (usually the entire object), and if the size is smaller no gzipping will be performed
	// if the client requests it.
	defaultGzipThresholdBytes = 128 * 1024
	// Use the length of the first write to recognize streaming implementations.
	// When streaming JSON first write is "{", while Kubernetes protobuf starts unique 4 byte header.
	firstWriteStreamingThresholdBytes = 4
	// streamingBatchBufferBytes is the size of the buffer that batches a large
	// streamed response's output. A ten-point sweep (4KiB-4MiB) on multi-GB
	// list responses measured a flat performance plateau from 128KiB through
	// 1MiB, degradation below 64KiB (item-sized writes pass through a small
	// bufio unbatched), and regression at 2MiB and above (larger bursts make
	// flow control choppier). 128KiB is the smallest size on the plateau,
	// which keeps pooled memory minimal: at most one buffer per in-flight
	// large streamed response.
	streamingBatchBufferBytes = 128 * 1024
	// streamingBatchEngageThresholdBytes is how many bytes lazyBatchWriter
	// passes straight to the response writer (compressed bytes, under gzip)
	// before it takes a pooled buffer and batches the rest. Responses that
	// stay under it, the small-cluster steady state, never take a pooled
	// buffer and reach the response writer exactly as on the unbatched path;
	// larger responses forfeit batching only for this prefix (item-sized
	// writes under identity, the compressor's small writes under gzip), a
	// sliver of a multi-GB response. 64KiB is where the sweep put ~85% of the
	// achievable benefit: below it, batching has little left to save.
	streamingBatchEngageThresholdBytes = 64 * 1024
)

// firstWriteIsStreaming reports whether a response's first write looks like a
// streaming collection encoder's opening bytes rather than a fully-marshaled
// single object. Gzip negotiation and output batching must agree on this
// classification, so both use this helper.
func firstWriteIsStreaming(p []byte) bool {
	return len(p) <= firstWriteStreamingThresholdBytes
}

// lazyBatchWriter is the last stage above the HTTP response writer for a
// committed streamed response, under both encodings (encoder ->
// lazyBatchWriter, or encoder -> gzip -> lazyBatchWriter).
//
// Streamed list responses arrive as one small write per item, and the
// transports below make each such write expensive: HTTP/2 turns roughly every
// one into a cross-goroutine DATA-frame handoff that parks the handler
// goroutine (its 4KiB buffer only coalesces writes smaller than itself), so a
// multi-GB list pays on the order of a million scheduler round-trips and
// ships mostly sub-full frames; HTTP/1.1 turns each into its own
// chunked-transfer frame, with TLS records fragmented to match. Batching
// turns those into one handoff, or one chunk, per buffer.
//
// The writer passes bytes straight through until the response has proven
// large (streamingBatchEngageThresholdBytes written directly), then takes a
// pooled buffer and batches the remainder, so small responses never hold a
// pooled buffer and reach the response writer exactly as before, whatever the
// encoding. The accepted cost, once engaged: after a client reset or request
// timeout, encoding continues until the next buffer flush fails (under gzip,
// one buffer of compressed output can be over a MiB of plaintext) instead of
// failing within one item or one deflate block. Bounded, and per canceled
// request.
type lazyBatchWriter struct {
	hw     io.Writer     // the response writer
	direct int           // bytes written straight to hw so far; decides engagement
	batch  *bufio.Writer // from batchBufferPool once engaged; nil before, and after close
}

func (b *lazyBatchWriter) Write(p []byte) (int, error) {
	if b.batch == nil {
		if b.direct < streamingBatchEngageThresholdBytes {
			n, err := b.hw.Write(p)
			if err == nil {
				// a failed (possibly partial) write must not count: engaging
				// on the next write would accept it into a fresh buffer
				// instead of failing it too
				b.direct += n
			}
			return n, err
		}
		b.batch = batchBufferPool.Get().(*bufio.Writer)
		b.batch.Reset(b.hw)
	}
	return b.batch.Write(p)
}

// close flushes any batched bytes to the response writer and returns the
// buffer to the pool; a no-op for responses that never engaged batching.
func (b *lazyBatchWriter) close() error {
	if b.batch == nil {
		return nil
	}
	err := b.batch.Flush()
	b.batch.Reset(nil)
	batchBufferPool.Put(b.batch)
	b.batch = nil
	return err
}

type deferredResponseWriter struct {
	mediaType       string
	statusCode      int
	contentEncoding string

	hasBuffered bool
	buffer      []byte
	hasWritten  bool
	hw          http.ResponseWriter
	w           io.Writer
	// batcher carries a committed streamed response's output to hw (below
	// the gzip writer, when compressing); see lazyBatchWriter. Responses are
	// wired through it by shape (a first write of a few bytes), which CBOR's
	// per-object tag also matches: such single objects pass through it
	// unengaged under identity, and batch like a list under gzip once large.
	batcher lazyBatchWriter
	// totalBytes is the number of bytes written to `w` (including bytes still
	// batched below it) and does not include buffered bytes
	totalBytes int
	// lastWriteErr holds the error result (if any) of the last write attempt
	// to `w`, or of Close's final flush
	lastWriteErr error

	ctx context.Context
}

func (w *deferredResponseWriter) Write(p []byte) (n int, err error) {
	switch {
	case w.hasWritten:
		// already written, cannot buffer
		return w.unbufferedWrite(p)

	case w.contentEncoding != "gzip":
		// non-gzip, no need to buffer
		return w.unbufferedWrite(p)

	case !w.hasBuffered && len(p) > defaultGzipThresholdBytes:
		// not yet buffered, first write is long enough to trigger gzip, no need to buffer
		return w.unbufferedWrite(p)

	case !w.hasBuffered && !firstWriteIsStreaming(p):
		// not yet buffered, first write is longer than expected for streaming scenarios that would require buffering, no need to buffer
		return w.unbufferedWrite(p)

	default:
		if !w.hasBuffered {
			w.hasBuffered = true
			// Start at 80 bytes to avoid rapid reallocation of the buffer.
			// The minimum size of a 0-item serialized list object is 80 bytes:
			// {"kind":"List","apiVersion":"v1","metadata":{"resourceVersion":"1"},"items":[]}\n
			w.buffer = make([]byte, 0, max(80, len(p)))
		}
		w.buffer = append(w.buffer, p...)
		var err error
		if len(w.buffer) > defaultGzipThresholdBytes {
			// we've accumulated enough to trigger gzip, write and clear buffer
			_, err = w.unbufferedWrite(w.buffer)
			w.buffer = nil
		}
		return len(p), err
	}
}

func (w *deferredResponseWriter) discardBufferedResponse() {
	if w.hasWritten {
		return
	}
	w.hasBuffered = false
	w.buffer = nil
}

// unbufferedWrite commits the response on its first call (status code and
// content-encoding headers, chosen from what Write accumulated) and writes p.
// "Unbuffered" refers to that gzip-negotiation buffer, not to delivery: a
// streamed response that proves large is batched by lazyBatchWriter below
// this point. Everything SerializeObject writes comes through here; watch and
// ResourceStreamer responses do not (they use their own writers, with
// per-event flushes), so batching never applies to them.
func (w *deferredResponseWriter) unbufferedWrite(p []byte) (n int, err error) {
	defer func() {
		w.totalBytes += n
		w.lastWriteErr = err
	}()

	if w.hasWritten {
		return w.w.Write(p)
	}
	w.hasWritten = true

	hw := w.hw
	header := hw.Header()

	// A response with a streaming encoder's shape (a tiny first write, or
	// the accumulation of one) reaches hw through the lazy batcher; anything
	// else is a fully-marshaled object and goes to hw directly. Misjudging a
	// small response as streamed costs nothing: the batcher is pass-through
	// below streamingBatchEngageThresholdBytes.
	var sink io.Writer = hw
	if w.hasBuffered || firstWriteIsStreaming(p) {
		w.batcher = lazyBatchWriter{hw: hw}
		sink = &w.batcher
	}
	switch {
	case w.contentEncoding == "gzip" && len(p) > defaultGzipThresholdBytes:
		header.Set("Content-Encoding", "gzip")
		header.Add("Vary", "Accept-Encoding")

		gw := gzipPool.Get().(*gzip.Writer)
		gw.Reset(sink)

		w.w = gw
	default:
		w.w = sink
	}

	span := tracing.SpanFromContext(w.ctx)
	span.AddEvent("About to start writing response",
		attribute.String("writer", fmt.Sprintf("%T", w.w)),
		attribute.Int("size", len(p)),
	)

	header.Set("Content-Type", w.mediaType)
	hw.WriteHeader(w.statusCode)
	return w.w.Write(p)
}

func (w *deferredResponseWriter) Close() (err error) {
	defer func() {
		if !w.hasWritten {
			return
		}

		span := tracing.SpanFromContext(w.ctx)

		if w.lastWriteErr != nil {
			span.AddEvent("Write call failed",
				attribute.Int("size", w.totalBytes),
				attribute.String("err", w.lastWriteErr.Error()))
		} else {
			span.AddEvent("Write call succeeded",
				attribute.Int("size", w.totalBytes))
		}
	}()

	if !w.hasWritten {
		if !w.hasBuffered {
			return nil
		}
		// never reached defaultGzipThresholdBytes: commit now, writing the
		// accumulated body uncompressed
		_, err = w.unbufferedWrite(w.buffer)
		w.buffer = nil
	}

	// Release whatever the response engaged; a body drained just above is one
	// uncompressed write and engaged nothing.
	switch t := w.w.(type) {
	case *gzip.Writer:
		err = t.Close()
		t.Reset(nil)
		gzipPool.Put(t)
	}
	if flushErr := w.batcher.close(); flushErr != nil && err == nil {
		err = flushErr
	}
	// A cleanup failure is recorded like a write error so the span reports it.
	if err != nil && w.lastWriteErr == nil {
		w.lastWriteErr = err
	}
	// The pooled writers released above may already serve another response;
	// drop w.w so that a Write after Close (which no caller does) panics on
	// the nil writer instead of reaching them.
	w.w = nil
	return err
}

// WriteObjectNegotiated renders an object in the content type negotiated by the client.
func WriteObjectNegotiated(s runtime.NegotiatedSerializer, restrictions negotiation.EndpointRestrictions, gv schema.GroupVersion, w http.ResponseWriter, req *http.Request, statusCode int, object runtime.Object, listGVKInContentType bool) {
	stream, ok := object.(rest.ResourceStreamer)
	if ok {
		requestInfo, _ := request.RequestInfoFrom(req.Context())
		metrics.RecordLongRunning(req, requestInfo, metrics.APIServerComponent, func() {
			StreamObject(statusCode, gv, s, stream, w, req)
		})
		return
	}

	mediaType, serializer, err := negotiation.NegotiateOutputMediaType(req, s, restrictions)
	if err != nil {
		// if original statusCode was not successful we need to return the original error
		// we cannot hide it behind negotiation problems
		if statusCode < http.StatusOK || statusCode >= http.StatusBadRequest {
			WriteRawJSON(int(statusCode), object, w)
			return
		}
		status := ErrorToAPIStatus(err)
		WriteRawJSON(int(status.Code), status, w)
		return
	}

	audit.LogResponseObject(req.Context(), object, gv, s)

	var encoder runtime.Encoder
	if utilfeature.DefaultFeatureGate.Enabled(features.CBORServingAndStorage) {
		encoder = s.EncoderForVersion(runtime.UseNondeterministicEncoding(serializer.Serializer), gv)
	} else {
		encoder = s.EncoderForVersion(serializer.Serializer, gv)
	}
	request.TrackSerializeResponseObjectLatency(req.Context(), func() {
		if listGVKInContentType {
			SerializeObject(generateMediaTypeWithGVK(serializer.MediaType, mediaType.Convert), encoder, w, req, statusCode, object)
		} else {
			SerializeObject(serializer.MediaType, encoder, w, req, statusCode, object)
		}
	})
}

func generateMediaTypeWithGVK(mediaType string, gvk *schema.GroupVersionKind) string {
	if gvk == nil {
		return mediaType
	}
	if gvk.Group != "" {
		mediaType += ";g=" + gvk.Group
	}
	if gvk.Version != "" {
		mediaType += ";v=" + gvk.Version
	}
	if gvk.Kind != "" {
		mediaType += ";as=" + gvk.Kind
	}
	return mediaType
}

// ErrorNegotiated renders an error to the response. Returns the HTTP status code of the error.
// The context is optional and may be nil.
func ErrorNegotiated(err error, s runtime.NegotiatedSerializer, gv schema.GroupVersion, w http.ResponseWriter, req *http.Request) int {
	status := ErrorToAPIStatus(err)
	code := int(status.Code)
	// when writing an error, check to see if the status indicates a retry after period
	if status.Details != nil && status.Details.RetryAfterSeconds > 0 {
		delay := strconv.Itoa(int(status.Details.RetryAfterSeconds))
		w.Header().Set("Retry-After", delay)
	}

	if code == http.StatusNoContent {
		w.WriteHeader(code)
		return code
	}

	WriteObjectNegotiated(s, negotiation.DefaultEndpointRestrictions, gv, w, req, code, status, false)
	return code
}

// WriteRawJSON writes a non-API object in JSON.
func WriteRawJSON(statusCode int, object interface{}, w http.ResponseWriter) {
	output, err := json.MarshalIndent(object, "", "  ")
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(statusCode)
	w.Write(output)
}
