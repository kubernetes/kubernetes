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

const (
	// defaultGzipThresholdBytes is the response size above which the response
	// is gzipped, if the client accepts gzip.
	defaultGzipThresholdBytes = 128 * 1024
	// responseBufferBytes is the size of the buffer every response is written
	// through, which batches the per-item writes of streamed list responses
	// into writes of this size. It equals defaultGzipThresholdBytes so that
	// the buffer also makes the gzip decision: a response that overflows it
	// exceeds the threshold, and one that fits is flushed once, at Close,
	// uncompressed.
	responseBufferBytes = defaultGzipThresholdBytes
)

// responseBufferPool holds the buffers responses are written through.
// TODO: engage lazily so small responses don't hold a pooled buffer.
var responseBufferPool = &sync.Pool{
	New: func() interface{} {
		return bufio.NewWriterSize(nil, responseBufferBytes)
	},
}

type deferredResponseWriter struct {
	mediaType       string
	statusCode      int
	contentEncoding string

	// buf takes every write; its first flush commits the response (see
	// commit) and from then on it batches writes on their way to w.
	buf        *bufio.Writer
	final      bool // set by Close before it flushes buf: no writes follow
	hasWritten bool // the response is committed
	hw         http.ResponseWriter
	w          io.Writer // below buf once committed: hw, or a gzip writer over hw
	// totalBytes is the number of bytes written to `w` and does not include buffered bytes
	totalBytes int
	// lastWriteErr holds the error result (if any) of the last write attempt to `w`
	lastWriteErr error

	ctx context.Context
}

func (w *deferredResponseWriter) Write(p []byte) (n int, err error) {
	if w.buf == nil {
		w.buf = responseBufferPool.Get().(*bufio.Writer)
		w.buf.Reset(committer{w})
	}
	return w.buf.Write(p)
}

// discardBufferedResponse drops what has been written so far, if the response
// is not committed yet, so that an error response can be written instead.
func (w *deferredResponseWriter) discardBufferedResponse() {
	if w.hasWritten || w.buf == nil {
		return
	}
	w.buf.Reset(committer{w})
}

// committer is the writer below buf: it receives buf's flushes.
type committer struct{ w *deferredResponseWriter }

func (c committer) Write(p []byte) (int, error) { return c.w.commit(p) }

// commit receives buf's flushes. The first one commits the response (decides
// gzip, writes status and headers); p and every later flush go to the chosen
// writer.
func (w *deferredResponseWriter) commit(p []byte) (n int, err error) {
	defer func() {
		w.totalBytes += n
		w.lastWriteErr = err
	}()

	if w.hasWritten {
		return w.w.Write(p)
	}
	w.hasWritten = true

	// buf holds exactly defaultGzipThresholdBytes, so it flushes before Close
	// only if the response exceeds the gzip threshold; if the first flush is
	// Close's, the whole response fit under it.
	exceedsGzipThreshold := !w.final

	hw := w.hw
	header := hw.Header()
	switch {
	case w.contentEncoding == "gzip" && exceedsGzipThreshold:
		header.Set("Content-Encoding", "gzip")
		header.Add("Vary", "Accept-Encoding")

		gw := gzipPool.Get().(*gzip.Writer)
		gw.Reset(hw)

		w.w = gw
	default:
		w.w = hw
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

	if w.buf == nil {
		return nil // nothing was written
	}
	w.final = true
	err = w.buf.Flush() // commits the response, if it fit in buf
	if err == nil && !w.hasWritten {
		_, err = w.commit(nil) // an empty body still gets its status and headers
	}
	w.buf.Reset(nil)
	responseBufferPool.Put(w.buf)
	w.buf = nil
	if gw, ok := w.w.(*gzip.Writer); ok {
		if cerr := gw.Close(); cerr != nil && err == nil {
			err = cerr
		}
		gw.Reset(nil)
		gzipPool.Put(gw)
	}
	w.w = nil
	if err != nil && w.lastWriteErr == nil {
		w.lastWriteErr = err
	}
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
