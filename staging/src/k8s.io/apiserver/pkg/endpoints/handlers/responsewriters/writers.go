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
	"bytes"
	"compress/flate"
	"compress/gzip"
	"context"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"hash/crc32"
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
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/apiserver/pkg/util/flushwriter"
	"k8s.io/component-base/tracing"
	"k8s.io/streaming/pkg/httpstream/wsstream"
)

// resourceStreamer is the method set of k8s.io/apiserver/pkg/registry/rest.ResourceStreamer. It is
// declared here so that this package does not import registry/rest, which pulls the admission and
// CEL packages into every binary that serves /flagz or /statusz. writers_test.go asserts that the
// two interfaces stay assignable to each other.
type resourceStreamer interface {
	InputStream(ctx context.Context, apiVersion, acceptHeader string) (stream io.ReadCloser, flush bool, mimeType string, err error)
}

// StreamObject performs input stream negotiation from a rest.ResourceStreamer and writes that to the response.
// If the client requests a websocket upgrade, negotiate for a websocket reader protocol (because many
// browser clients cannot easily handle binary streaming protocols).
func StreamObject(statusCode int, gv schema.GroupVersion, s runtime.NegotiatedSerializer, stream resourceStreamer, w http.ResponseWriter, req *http.Request) {
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
	contentEncoding := responseContentEncodingSupported(req)
	if contentEncoding == "gzip" {
		w.Header().Set("Content-Encoding", "gzip")
		w.Header().Add("Vary", "Accept-Encoding")
	}
	w.Header().Set("Content-Type", contentType)
	w.WriteHeader(statusCode)
	// Flush headers, if possible
	if flusher, ok := w.(http.Flusher); ok {
		flusher.Flush()
	}
	writer := w.(io.Writer)
	if contentEncoding == "gzip" {
		sw := newGzipStreamWriter(w, flush)
		defer sw.Close()
		writer = sw
	} else if flush {
		writer = flushwriter.Wrap(w)
	}
	io.Copy(writer, out)
}

const (
	// gzipStreamLevel is the flate level for streams. Level 1 does not match
	// against a primed history.
	gzipStreamLevel = 2
	// gzipHistoryBytes is how much of a stream's recent output primes the
	// compressor for the next write.
	gzipHistoryBytes = 2 * 1024
	// gzipBatchBytes is how much output a stream that is not followed holds
	// before compressing it.
	gzipBatchBytes = 32 * 1024
	// gzipMaxPooledBufferBytes is the largest output buffer returned to the pool.
	gzipMaxPooledBufferBytes = 64 * 1024
)

// gzipHeader is the header of a gzip member without a name, comment or
// modification time.
var gzipHeader = []byte{0x1f, 0x8b, 8, 0, 0, 0, 0, 0, 0, 255}

var errGzipStreamClosed = errors.New("gzip stream writer is closed")

// primedFlateWriter is a pooled flate writer whose output can be redirected
// while it is primed with a stream's history.
type primedFlateWriter struct {
	fw  *flate.Writer
	out redirectWriter
}

type redirectWriter struct{ w io.Writer }

func (r *redirectWriter) Write(p []byte) (int, error) { return r.w.Write(p) }

var primedFlateWriterPool = &sync.Pool{
	New: func() interface{} {
		pw := &primedFlateWriter{out: redirectWriter{w: io.Discard}}
		fw, err := flate.NewWriter(&pw.out, gzipStreamLevel)
		if err != nil {
			panic(err)
		}
		pw.fw = fw
		return pw
	},
}

var gzipOutputBufferPool = &sync.Pool{
	New: func() interface{} { return new(bytes.Buffer) },
}

// gzipStreamWriter writes a stream as a single gzip member in which every
// write is compressed and sent at once by a pooled flate writer, primed with
// the last gzipHistoryBytes of the stream so that it can refer back to them.
// The flate writer is returned to the pool before the output is written to
// the response, so a slow client does not hold it.
type gzipStreamWriter struct {
	w       io.Writer
	flusher http.Flusher
	follow  bool

	buf     []byte // pending output of a stream that is not followed
	hist    []byte // last gzipHistoryBytes sent
	started bool
	closed  bool
	crc     uint32
	size    uint32
	err     error
}

func newGzipStreamWriter(w http.ResponseWriter, follow bool) *gzipStreamWriter {
	flusher, _ := w.(http.Flusher)
	return &gzipStreamWriter{w: w, flusher: flusher, follow: follow}
}

func (sw *gzipStreamWriter) Write(p []byte) (int, error) {
	if sw.closed {
		return 0, errGzipStreamClosed
	}
	if sw.err != nil {
		return 0, sw.err
	}
	if sw.follow {
		if err := sw.send(p); err != nil {
			return 0, err
		}
		return len(p), nil
	}
	sw.buf = append(sw.buf, p...)
	if len(sw.buf) >= gzipBatchBytes {
		err := sw.send(sw.buf)
		sw.buf = sw.buf[:0]
		if err != nil {
			return 0, err
		}
	}
	return len(p), nil
}

// send compresses p, primed with the stream's history, and sends it.
func (sw *gzipStreamWriter) send(p []byte) error {
	if len(p) == 0 {
		return nil
	}
	sw.err = sw.compressAndWrite(func(pw *primedFlateWriter, out io.Writer) error {
		if len(sw.hist) > 0 {
			pw.out.w = io.Discard
			if _, err := pw.fw.Write(sw.hist); err != nil {
				return err
			}
			if err := pw.fw.Flush(); err != nil {
				return err
			}
			pw.out.w = out
		}
		if _, err := pw.fw.Write(p); err != nil {
			return err
		}
		return pw.fw.Flush()
	}, nil)
	sw.crc = crc32.Update(sw.crc, crc32.IEEETable, p)
	sw.size += uint32(len(p))
	sw.hist = appendTail(sw.hist, p, gzipHistoryBytes)
	if sw.err == nil && sw.follow && sw.flusher != nil {
		sw.flusher.Flush()
	}
	return sw.err
}

// compressAndWrite runs fn with a pooled flate writer that writes to a pooled
// buffer, returns the flate writer to the pool, then writes the gzip header if
// needed, the buffer and trailer to the response.
func (sw *gzipStreamWriter) compressAndWrite(fn func(*primedFlateWriter, io.Writer) error, trailer []byte) error {
	out := gzipOutputBufferPool.Get().(*bytes.Buffer)
	out.Reset()
	defer func() {
		if out.Cap() <= gzipMaxPooledBufferBytes {
			gzipOutputBufferPool.Put(out)
		}
	}()
	if !sw.started {
		sw.started = true
		out.Write(gzipHeader)
	}
	pw := primedFlateWriterPool.Get().(*primedFlateWriter)
	pw.out.w = out
	pw.fw.Reset(&pw.out)
	err := fn(pw, out)
	pw.out.w = io.Discard
	primedFlateWriterPool.Put(pw)
	if err != nil {
		return err
	}
	out.Write(trailer)
	_, err = sw.w.Write(out.Bytes())
	return err
}

// Close sends the remaining output and ends the gzip member. Nothing is sent
// after Close returns.
func (sw *gzipStreamWriter) Close() error {
	if sw.closed {
		return sw.err
	}
	sw.closed = true
	if sw.err != nil {
		return sw.err
	}
	if err := sw.send(sw.buf); err != nil {
		return err
	}
	sw.buf = nil
	// The final, empty deflate block, then the CRC-32 and size of the data.
	var trailer [8]byte
	binary.LittleEndian.PutUint32(trailer[:4], sw.crc)
	binary.LittleEndian.PutUint32(trailer[4:], sw.size)
	sw.err = sw.compressAndWrite(func(pw *primedFlateWriter, _ io.Writer) error { return pw.fw.Close() }, trailer[:])
	return sw.err
}

// appendTail appends p to hist and keeps the last n bytes, never growing
// hist's capacity beyond n.
func appendTail(hist, p []byte, n int) []byte {
	if len(p) >= n {
		if cap(hist) < n {
			hist = make([]byte, 0, n)
		}
		return append(hist[:0], p[len(p)-n:]...)
	}
	if len(hist)+len(p) > n {
		keep := n - len(p)
		copy(hist, hist[len(hist)-keep:])
		hist = hist[:keep]
	}
	if cap(hist)-len(hist) < len(p) {
		c := min(max(2*cap(hist), len(hist)+len(p)), n)
		hist = append(make([]byte, 0, c), hist...)
	}
	return append(hist, p...)
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
	// defaultGzipThresholdBytes is compared to the size of the first write from the stream
	// (usually the entire object), and if the size is smaller no gzipping will be performed
	// if the client requests it.
	defaultGzipThresholdBytes = 128 * 1024
	// Use the length of the first write to recognize streaming implementations.
	// When streaming JSON first write is "{", while Kubernetes protobuf starts unique 4 byte header.
	firstWriteStreamingThresholdBytes = 4
)

type deferredResponseWriter struct {
	mediaType       string
	statusCode      int
	contentEncoding string

	hasBuffered bool
	buffer      []byte
	hasWritten  bool
	hw          http.ResponseWriter
	w           io.Writer
	// totalBytes is the number of bytes written to `w` and does not include buffered bytes
	totalBytes int
	// lastWriteErr holds the error result (if any) of the last write attempt to `w`
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

	case !w.hasBuffered && len(p) > firstWriteStreamingThresholdBytes:
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
	switch {
	case w.contentEncoding == "gzip" && len(p) > defaultGzipThresholdBytes:
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

	if !w.hasWritten {
		if !w.hasBuffered {
			return nil
		}
		// never reached defaultGzipThresholdBytes, no need to do the gzip writer cleanup
		_, err := w.unbufferedWrite(w.buffer)
		w.buffer = nil
		return err
	}

	switch t := w.w.(type) {
	case *gzip.Writer:
		err = t.Close()
		t.Reset(nil)
		gzipPool.Put(t)
	}
	return err
}

// WriteObjectNegotiated renders an object in the content type negotiated by the client.
func WriteObjectNegotiated(s runtime.NegotiatedSerializer, restrictions negotiation.EndpointRestrictions, gv schema.GroupVersion, w http.ResponseWriter, req *http.Request, statusCode int, object runtime.Object, listGVKInContentType bool) {
	stream, ok := object.(resourceStreamer)
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
