package http

import (
	"net/http"
)

// TestResponseWriter is a http.ResponseWriter object that keeps track of all activity
// allowing you to make assertions about how it was used.
//
// DEPRECATED: We recommend you use http://golang.org/pkg/net/http/httptest instead.
type TestResponseWriter struct {

	// StatusCode is the last int written by the call to WriteHeader(int)
	StatusCode int

	// Output is a string containing the written bytes using the Write([]byte) func.
	Output string

	// header is the internal storage of the http.Header object
	header http.Header
}

// Header gets the http.Header describing the headers that were set in this response.
func (rw *TestResponseWriter) Header() http.Header {

	if rw.header == nil {
		rw.header = make(http.Header)
	}

	return rw.header
}

// Write writes the specified bytes to Output.
func (rw *TestResponseWriter) Write(bytes []byte) (int, error) {

	// assume 200 success if no header has been set
	if rw.StatusCode == 0 {
		rw.WriteHeader(200)
	}

	// add these bytes to the output string
	rw.Output = rw.Output + string(bytes)

	// return normal values
	return 0, nil

}

// WriteHeader stores the HTTP status code in the StatusCode.
func (rw *TestResponseWriter) WriteHeader(i int) {
	rw.StatusCode = i
}
