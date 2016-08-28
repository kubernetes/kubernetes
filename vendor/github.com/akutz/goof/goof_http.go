package goof

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
)

// HTTPError is the base type for a Goof error that has an HTTP status and
// could possibly be marshalled to JSON.
type HTTPError interface {
	Goof

	// Status returns the HTTP status.
	Status() int

	// Inner returns the inner error.
	Inner() error
}

type httpError struct {
	*goof
}

func (e *httpError) Status() int {
	if v, ok := e.data["status"].(int); ok {
		return v
	}
	return 0
}

func (e *httpError) Inner() error {
	if v, ok := e.data[InnerErrorKey].(error); ok {
		return v
	}
	return nil
}

// MarshalJSON marshals this object to JSON for the encoding/json package.
func (e *httpError) MarshalJSON() ([]byte, error) {

	if len(e.data) == 0 {
		return json.Marshal(e.msg)
	}

	innerErr := withFieldsE(nil, "", nil)
	for k, v := range e.data {
		if k != "status" {
			innerErr.data[k] = v
		}
	}

	var innerInnerErr error
	switch len(innerErr.data) {
	case 0:
		innerErr = nil
	case 1:
		var err error
		for _, v := range innerErr.data {
			if tErr, ok := v.(error); ok {
				err = tErr
				break
			}
		}
		if err != nil {
			innerInnerErr = err
		}
	}

	jsonError := struct {
		Message string `json:"message"`
		Status  int    `json:"status"`
		Inner   error  `json:"error,omitempty"`
	}{
		Message: e.msg,
		Status:  e.Status(),
		Inner:   innerErr,
	}

	if innerInnerErr != nil {
		jsonError.Inner = innerInnerErr
	}

	return json.Marshal(jsonError)
}

// UnmarshalJSON unmarshals JSON data to a Goof error.
func (e *httpError) UnmarshalJSON(data []byte) error {
	e.goof = newGoof("", nil)
	if err := e.goof.UnmarshalJSON(data); err != nil {
		return err
	}
	if v, ok := e.data["status"].(float64); ok {
		e.data["status"] = int(v)
	}
	return nil
}

// NewHTTPError returns a new HTTPError using the provided HTTP status.
func NewHTTPError(err error, status int) HTTPError {

	var innerErr *goof
	switch tErr := err.(type) {
	case nil:
		return &httpError{
			withFieldsE(Fields{
				"status": status,
			}, fmt.Sprintf("%d", status), nil),
		}
	case *goof:
		innerErr = tErr
	case Goof:
		innerErr = withFieldsE(tErr.Fields(), tErr.Error(), nil)
	default:
		innerErr = withFieldsE(nil, tErr.Error(), nil)
	}

	innerErr.IncludeMessageInJSON(false)

	return &httpError{
		withFieldsE(Fields{"status": status}, innerErr.Error(), innerErr),
	}
}

// UnmarshalHTTPError returns a new HTTPError object by unmarshalling the
// contents of the buffer.
func UnmarshalHTTPError(data []byte) (HTTPError, error) {
	return DecodeHTTPError(bytes.NewReader(data))
}

// DecodeHTTPError returns a new HTTPError object by decoding the contents of
// the reader.
func DecodeHTTPError(r io.Reader) (HTTPError, error) {
	d := json.NewDecoder(r)
	e := &httpError{}
	if err := d.Decode(e); err != nil {
		return nil, err
	}
	return e, nil
}
