// Package easyjson contains marshaler/unmarshaler interfaces and helper functions.
package easyjson

import (
	"io"
	"io/ioutil"
	"net/http"
	"strconv"

	"github.com/mailru/easyjson/jlexer"
	"github.com/mailru/easyjson/jwriter"
)

// Marshaler is an easyjson-compatible marshaler interface.
type Marshaler interface {
	MarshalEasyJSON(w *jwriter.Writer)
}

// Marshaler is an easyjson-compatible unmarshaler interface.
type Unmarshaler interface {
	UnmarshalEasyJSON(w *jlexer.Lexer)
}

// Optional defines an undefined-test method for a type to integrate with 'omitempty' logic.
type Optional interface {
	IsDefined() bool
}

// Marshal returns data as a single byte slice. Method is suboptimal as the data is likely to be copied
// from a chain of smaller chunks.
func Marshal(v Marshaler) ([]byte, error) {
	w := jwriter.Writer{}
	v.MarshalEasyJSON(&w)
	return w.BuildBytes()
}

// MarshalToWriter marshals the data to an io.Writer.
func MarshalToWriter(v Marshaler, w io.Writer) (written int, err error) {
	jw := jwriter.Writer{}
	v.MarshalEasyJSON(&jw)
	return jw.DumpTo(w)
}

// MarshalToHTTPResponseWriter sets Content-Length and Content-Type headers for the
// http.ResponseWriter, and send the data to the writer. started will be equal to
// false if an error occurred before any http.ResponseWriter methods were actually
// invoked (in this case a 500 reply is possible).
func MarshalToHTTPResponseWriter(v Marshaler, w http.ResponseWriter) (started bool, written int, err error) {
	jw := jwriter.Writer{}
	v.MarshalEasyJSON(&jw)
	if jw.Error != nil {
		return false, 0, jw.Error
	}
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("Content-Length", strconv.Itoa(jw.Size()))

	started = true
	written, err = jw.DumpTo(w)
	return
}

// Unmarshal decodes the JSON in data into the object.
func Unmarshal(data []byte, v Unmarshaler) error {
	l := jlexer.Lexer{Data: data}
	v.UnmarshalEasyJSON(&l)
	return l.Error()
}

// UnmarshalFromReader reads all the data in the reader and decodes as JSON into the object.
func UnmarshalFromReader(r io.Reader, v Unmarshaler) error {
	data, err := ioutil.ReadAll(r)
	if err != nil {
		return err
	}
	l := jlexer.Lexer{Data: data}
	v.UnmarshalEasyJSON(&l)
	return l.Error()
}
