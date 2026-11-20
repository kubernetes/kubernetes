// Copyright 2016-2017 The New York Times Company. Licensed under the Apache License, Version 2.0.
// Copied from github.com/NYTimes/gziphandler v1.1.1 (dd0439581c76).
// Changes: gofmt; the HTTP/2 Push method (gzip_go18.go) and its tests are dropped; the import comment is removed;
// fields and functions used only in this package are privatised; content-type filtering is removed; options are removed;
// a single writer pool is used (for the single compression level supported).

package gziphandler

import (
	"bufio"
	"compress/gzip"
	"fmt"
	"io"
	"net"
	"net/http"
	"strconv"
	"strings"
	"sync"
)

const (
	vary            = "Vary"
	acceptEncoding  = "Accept-Encoding"
	contentEncoding = "Content-Encoding"
	contentType     = "Content-Type"
	contentLength   = "Content-Length"
)

type codings map[string]float64

const (
	// defaultQValue is the default qvalue to assign to an encoding if no explicit qvalue is set.
	// This is actually kind of ambiguous in RFC 2616, so hopefully it's correct.
	// The examples seem to indicate that it is.
	defaultQValue = 1.0

	// defaultMinSize is the default minimum size until we enable gzip compression.
	// 1500 bytes is the MTU size for the internet since that is the largest size allowed at the network layer.
	// If you take a file that is 1300 bytes and compress it to 800 bytes, it’s still transmitted in that same 1500 byte packet regardless, so you’ve gained nothing.
	// That being the case, you should restrict the gzip compression to files with a size greater than a single packet, 1400 bytes (1.4KB) is a safe value.
	defaultMinSize = 1400
)

// gzipWriterPool stores a sync.Pool for the default compression level for reuse of
// gzip.Writers.
var gzipWriterPool = newGzipWriterPool()

func newGzipWriterPool() *sync.Pool {
	return &sync.Pool{
		New: func() any {
			w, _ := gzip.NewWriterLevel(nil, gzip.DefaultCompression)
			return w
		},
	}
}

// gzipResponseWriter provides an http.ResponseWriter interface, which gzips
// bytes before writing them to the underlying response. This doesn't close the
// writers, so don't forget to do that.
// It can be configured to skip response smaller than minSize.
type gzipResponseWriter struct {
	http.ResponseWriter
	gw *gzip.Writer

	code int // Saves the WriteHeader value.

	buf    []byte // Holds the first part of the write before reaching the minSize or the end of the write.
	ignore bool   // If true, then we immediately passthru writes to the underlying ResponseWriter.
}

type gzipResponseWriterWithCloseNotify struct {
	*gzipResponseWriter
}

func (w gzipResponseWriterWithCloseNotify) CloseNotify() <-chan bool {
	return w.ResponseWriter.(http.CloseNotifier).CloseNotify()
}

// Write appends data to the gzip writer.
func (w *gzipResponseWriter) Write(b []byte) (int, error) {
	// GZIP responseWriter is initialized. Use the GZIP responseWriter.
	if w.gw != nil {
		return w.gw.Write(b)
	}

	// If we have already decided not to use GZIP, immediately passthrough.
	if w.ignore {
		return w.ResponseWriter.Write(b)
	}

	// Save the write into a buffer for later use in GZIP responseWriter (if content is long enough) or at close with regular responseWriter.
	// On the first write, w.buf changes from nil to a valid slice
	w.buf = append(w.buf, b...)

	var (
		cl, _ = strconv.Atoi(w.Header().Get(contentLength))
		ct    = w.Header().Get(contentType)
		ce    = w.Header().Get(contentEncoding)
	)
	// Only continue if they didn't already choose an encoding or a known unhandled content length.
	if ce == "" && (cl == 0 || cl >= defaultMinSize) {
		// If the current buffer is less than minSize and a Content-Length isn't set, then wait until we have more data.
		if len(w.buf) < defaultMinSize && cl == 0 {
			return len(b), nil
		}
		// If the Content-Length is larger than minSize or the current buffer is larger than minSize, then continue.
		if cl >= defaultMinSize || len(w.buf) >= defaultMinSize {
			// If a Content-Type wasn't specified, infer it from the current buffer.
			if ct == "" {
				ct = http.DetectContentType(w.buf)
				w.Header().Set(contentType, ct)
			}
			if err := w.startGzip(); err != nil {
				return 0, err
			}
			return len(b), nil
		}
	}
	// If we got here, we should not GZIP this response.
	if err := w.startPlain(); err != nil {
		return 0, err
	}
	return len(b), nil
}

// startGzip initializes a GZIP writer and writes the buffer.
func (w *gzipResponseWriter) startGzip() error {
	// Set the GZIP header.
	w.Header().Set(contentEncoding, "gzip")

	// if the Content-Length is already set, then calls to Write on gzip
	// will fail to set the Content-Length header since its already set
	// See: https://github.com/golang/go/issues/14975.
	w.Header().Del(contentLength)

	// Write the header to gzip response.
	if w.code != 0 {
		w.ResponseWriter.WriteHeader(w.code)
		// Ensure that no other WriteHeader's happen
		w.code = 0
	}

	// Initialize and flush the buffer into the gzip response if there are any bytes.
	// If there aren't any, we shouldn't initialize it yet because on Close it will
	// write the gzip header even if nothing was ever written.
	if len(w.buf) > 0 {
		// Initialize the GZIP response.
		w.init()
		n, err := w.gw.Write(w.buf)

		// This should never happen (per io.Writer docs), but if the write didn't
		// accept the entire buffer but returned no specific error, we have no clue
		// what's going on, so abort just to be safe.
		if err == nil && n < len(w.buf) {
			err = io.ErrShortWrite
		}
		return err
	}
	return nil
}

// startPlain writes to sent bytes and buffer the underlying ResponseWriter without gzip.
func (w *gzipResponseWriter) startPlain() error {
	if w.code != 0 {
		w.ResponseWriter.WriteHeader(w.code)
		// Ensure that no other WriteHeader's happen
		w.code = 0
	}
	w.ignore = true
	// If Write was never called then don't call Write on the underlying ResponseWriter.
	if w.buf == nil {
		return nil
	}
	n, err := w.ResponseWriter.Write(w.buf)
	w.buf = nil
	// This should never happen (per io.Writer docs), but if the write didn't
	// accept the entire buffer but returned no specific error, we have no clue
	// what's going on, so abort just to be safe.
	if err == nil && n < len(w.buf) {
		err = io.ErrShortWrite
	}
	return err
}

// WriteHeader just saves the response code until close or GZIP effective writes.
func (w *gzipResponseWriter) WriteHeader(code int) {
	if w.code == 0 {
		w.code = code
	}
}

// init graps a new gzip writer from the gzipWriterPool and writes the correct
// content encoding header.
func (w *gzipResponseWriter) init() {
	// Bytes written during ServeHTTP are redirected to this gzip writer
	// before being written to the underlying response.
	gzw := gzipWriterPool.Get().(*gzip.Writer)
	gzw.Reset(w.ResponseWriter)
	w.gw = gzw
}

// Close will close the gzip.Writer and will put it back in the gzipWriterPool.
func (w *gzipResponseWriter) Close() error {
	if w.ignore {
		return nil
	}

	if w.gw == nil {
		// GZIP not triggered yet, write out regular response.
		err := w.startPlain()
		// Returns the error if any at write.
		if err != nil {
			err = fmt.Errorf("gziphandler: write to regular responseWriter at close gets error: %q", err.Error())
		}
		return err
	}

	err := w.gw.Close()
	gzipWriterPool.Put(w.gw)
	w.gw = nil
	return err
}

// Flush flushes the underlying *gzip.Writer and then the underlying
// http.ResponseWriter if it is an http.Flusher. This makes gzipResponseWriter
// an http.Flusher.
func (w *gzipResponseWriter) Flush() {
	if w.gw == nil && !w.ignore {
		// Only flush once startGzip or startPlain has been called.
		//
		// Flush is thus a no-op until we're certain whether a plain
		// or gzipped response will be served.
		return
	}

	if w.gw != nil {
		w.gw.Flush()
	}

	if fw, ok := w.ResponseWriter.(http.Flusher); ok {
		fw.Flush()
	}
}

// Hijack implements http.Hijacker. If the underlying ResponseWriter is a
// Hijacker, its Hijack method is returned. Otherwise an error is returned.
func (w *gzipResponseWriter) Hijack() (net.Conn, *bufio.ReadWriter, error) {
	if hj, ok := w.ResponseWriter.(http.Hijacker); ok {
		return hj.Hijack()
	}
	return nil, nil, fmt.Errorf("http.Hijacker interface is not supported")
}

// verify Hijacker interface implementation
var _ http.Hijacker = &gzipResponseWriter{}

// GzipHandler wraps an HTTP handler, to transparently gzip the response body if
// the client supports it (via the Accept-Encoding header). This will compress at
// the default compression level.
func GzipHandler(h http.Handler) http.Handler {
	return func(h http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.Header().Add(vary, acceptEncoding)
			if acceptsGzip(r) {
				gw := &gzipResponseWriter{
					ResponseWriter: w,
				}
				defer gw.Close()

				if _, ok := w.(http.CloseNotifier); ok {
					gwcn := gzipResponseWriterWithCloseNotify{gw}
					h.ServeHTTP(gwcn, r)
				} else {
					h.ServeHTTP(gw, r)
				}

			} else {
				h.ServeHTTP(w, r)
			}
		})
	}(h)
}

// acceptsGzip returns true if the given HTTP request indicates that it will
// accept a gzipped response.
func acceptsGzip(r *http.Request) bool {
	acceptedEncodings, _ := parseEncodings(r.Header.Get(acceptEncoding))
	return acceptedEncodings["gzip"] > 0.0
}

// parseEncodings attempts to parse a list of codings, per RFC 2616, as might
// appear in an Accept-Encoding header. It returns a map of content-codings to
// quality values, and an error containing the errors encountered. It's probably
// safe to ignore those, because silently ignoring errors is how the internet
// works.
//
// See: http://tools.ietf.org/html/rfc2616#section-14.3.
func parseEncodings(s string) (codings, error) {
	c := make(codings)
	var e []string

	for _, ss := range strings.Split(s, ",") {
		coding, qvalue, err := parseCoding(ss)

		if err != nil {
			e = append(e, err.Error())
		} else {
			c[coding] = qvalue
		}
	}

	// TODO (adammck): Use a proper multi-error struct, so the individual errors
	//                 can be extracted if anyone cares.
	if len(e) > 0 {
		return c, fmt.Errorf("errors while parsing encodings: %s", strings.Join(e, ", "))
	}

	return c, nil
}

// parseCoding parses a single conding (content-coding with an optional qvalue),
// as might appear in an Accept-Encoding header. It attempts to forgive minor
// formatting errors.
func parseCoding(s string) (coding string, qvalue float64, err error) {
	for n, part := range strings.Split(s, ";") {
		part = strings.TrimSpace(part)
		qvalue = defaultQValue

		if n == 0 {
			coding = strings.ToLower(part)
		} else if strings.HasPrefix(part, "q=") {
			qvalue, err = strconv.ParseFloat(strings.TrimPrefix(part, "q="), 64)

			if qvalue < 0.0 {
				qvalue = 0.0
			} else if qvalue > 1.0 {
				qvalue = 1.0
			}
		}
	}

	if coding == "" {
		err = fmt.Errorf("empty content-coding")
	}

	return
}
