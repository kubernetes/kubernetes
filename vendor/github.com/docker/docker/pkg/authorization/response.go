package authorization

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"net"
	"net/http"

	"github.com/sirupsen/logrus"
)

// ResponseModifier allows authorization plugins to read and modify the content of the http.response
type ResponseModifier interface {
	http.ResponseWriter
	http.Flusher
	http.CloseNotifier

	// RawBody returns the current http content
	RawBody() []byte

	// RawHeaders returns the current content of the http headers
	RawHeaders() ([]byte, error)

	// StatusCode returns the current status code
	StatusCode() int

	// OverrideBody replaces the body of the HTTP reply
	OverrideBody(b []byte)

	// OverrideHeader replaces the headers of the HTTP reply
	OverrideHeader(b []byte) error

	// OverrideStatusCode replaces the status code of the HTTP reply
	OverrideStatusCode(statusCode int)

	// FlushAll flushes all data to the HTTP response
	FlushAll() error

	// Hijacked indicates the response has been hijacked by the Docker daemon
	Hijacked() bool
}

// NewResponseModifier creates a wrapper to an http.ResponseWriter to allow inspecting and modifying the content
func NewResponseModifier(rw http.ResponseWriter) ResponseModifier {
	return &responseModifier{rw: rw, header: make(http.Header)}
}

// responseModifier is used as an adapter to http.ResponseWriter in order to manipulate and explore
// the http request/response from docker daemon
type responseModifier struct {
	// The original response writer
	rw http.ResponseWriter
	// body holds the response body
	body []byte
	// header holds the response header
	header http.Header
	// statusCode holds the response status code
	statusCode int
	// hijacked indicates the request has been hijacked
	hijacked bool
}

func (rm *responseModifier) Hijacked() bool {
	return rm.hijacked
}

// WriterHeader stores the http status code
func (rm *responseModifier) WriteHeader(s int) {

	// Use original request if hijacked
	if rm.hijacked {
		rm.rw.WriteHeader(s)
		return
	}

	rm.statusCode = s
}

// Header returns the internal http header
func (rm *responseModifier) Header() http.Header {

	// Use original header if hijacked
	if rm.hijacked {
		return rm.rw.Header()
	}

	return rm.header
}

// StatusCode returns the http status code
func (rm *responseModifier) StatusCode() int {
	return rm.statusCode
}

// OverrideBody replaces the body of the HTTP response
func (rm *responseModifier) OverrideBody(b []byte) {
	rm.body = b
}

// OverrideStatusCode replaces the status code of the HTTP response
func (rm *responseModifier) OverrideStatusCode(statusCode int) {
	rm.statusCode = statusCode
}

// OverrideHeader replaces the headers of the HTTP response
func (rm *responseModifier) OverrideHeader(b []byte) error {
	header := http.Header{}
	if err := json.Unmarshal(b, &header); err != nil {
		return err
	}
	rm.header = header
	return nil
}

// Write stores the byte array inside content
func (rm *responseModifier) Write(b []byte) (int, error) {

	if rm.hijacked {
		return rm.rw.Write(b)
	}

	rm.body = append(rm.body, b...)
	return len(b), nil
}

// Body returns the response body
func (rm *responseModifier) RawBody() []byte {
	return rm.body
}

func (rm *responseModifier) RawHeaders() ([]byte, error) {
	var b bytes.Buffer
	if err := rm.header.Write(&b); err != nil {
		return nil, err
	}
	return b.Bytes(), nil
}

// Hijack returns the internal connection of the wrapped http.ResponseWriter
func (rm *responseModifier) Hijack() (net.Conn, *bufio.ReadWriter, error) {

	rm.hijacked = true
	rm.FlushAll()

	hijacker, ok := rm.rw.(http.Hijacker)
	if !ok {
		return nil, nil, fmt.Errorf("Internal response writer doesn't support the Hijacker interface")
	}
	return hijacker.Hijack()
}

// CloseNotify uses the internal close notify API of the wrapped http.ResponseWriter
func (rm *responseModifier) CloseNotify() <-chan bool {
	closeNotifier, ok := rm.rw.(http.CloseNotifier)
	if !ok {
		logrus.Error("Internal response writer doesn't support the CloseNotifier interface")
		return nil
	}
	return closeNotifier.CloseNotify()
}

// Flush uses the internal flush API of the wrapped http.ResponseWriter
func (rm *responseModifier) Flush() {
	flusher, ok := rm.rw.(http.Flusher)
	if !ok {
		logrus.Error("Internal response writer doesn't support the Flusher interface")
		return
	}

	rm.FlushAll()
	flusher.Flush()
}

// FlushAll flushes all data to the HTTP response
func (rm *responseModifier) FlushAll() error {
	// Copy the header
	for k, vv := range rm.header {
		for _, v := range vv {
			rm.rw.Header().Add(k, v)
		}
	}

	// Copy the status code
	// Also WriteHeader needs to be done after all the headers
	// have been copied (above).
	if rm.statusCode > 0 {
		rm.rw.WriteHeader(rm.statusCode)
	}

	var err error
	if len(rm.body) > 0 {
		// Write body
		_, err = rm.rw.Write(rm.body)
	}

	// Clean previous data
	rm.body = nil
	rm.statusCode = 0
	rm.header = http.Header{}
	return err
}
