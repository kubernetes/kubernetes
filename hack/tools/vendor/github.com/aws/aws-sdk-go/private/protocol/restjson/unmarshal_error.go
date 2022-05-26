package restjson

import (
	"bytes"
	"io"
	"io/ioutil"
	"net/http"
	"strings"

	"github.com/aws/aws-sdk-go/aws/awserr"
	"github.com/aws/aws-sdk-go/aws/request"
	"github.com/aws/aws-sdk-go/private/protocol"
	"github.com/aws/aws-sdk-go/private/protocol/json/jsonutil"
	"github.com/aws/aws-sdk-go/private/protocol/rest"
)

const (
	errorTypeHeader    = "X-Amzn-Errortype"
	errorMessageHeader = "X-Amzn-Errormessage"
)

// UnmarshalTypedError provides unmarshaling errors API response errors
// for both typed and untyped errors.
type UnmarshalTypedError struct {
	exceptions map[string]func(protocol.ResponseMetadata) error
}

// NewUnmarshalTypedError returns an UnmarshalTypedError initialized for the
// set of exception names to the error unmarshalers
func NewUnmarshalTypedError(exceptions map[string]func(protocol.ResponseMetadata) error) *UnmarshalTypedError {
	return &UnmarshalTypedError{
		exceptions: exceptions,
	}
}

// UnmarshalError attempts to unmarshal the HTTP response error as a known
// error type. If unable to unmarshal the error type, the generic SDK error
// type will be used.
func (u *UnmarshalTypedError) UnmarshalError(
	resp *http.Response,
	respMeta protocol.ResponseMetadata,
) (error, error) {

	code := resp.Header.Get(errorTypeHeader)
	msg := resp.Header.Get(errorMessageHeader)

	body := resp.Body
	if len(code) == 0 {
		// If unable to get code from HTTP headers have to parse JSON message
		// to determine what kind of exception this will be.
		var buf bytes.Buffer
		var jsonErr jsonErrorResponse
		teeReader := io.TeeReader(resp.Body, &buf)
		err := jsonutil.UnmarshalJSONError(&jsonErr, teeReader)
		if err != nil {
			return nil, err
		}

		body = ioutil.NopCloser(&buf)
		code = jsonErr.Code
		msg = jsonErr.Message
	}

	// If code has colon separators remove them so can compare against modeled
	// exception names.
	code = strings.SplitN(code, ":", 2)[0]

	if fn, ok := u.exceptions[code]; ok {
		// If exception code is know, use associated constructor to get a value
		// for the exception that the JSON body can be unmarshaled into.
		v := fn(respMeta)
		if err := jsonutil.UnmarshalJSONCaseInsensitive(v, body); err != nil {
			return nil, err
		}

		if err := rest.UnmarshalResponse(resp, v, true); err != nil {
			return nil, err
		}

		return v, nil
	}

	// fallback to unmodeled generic exceptions
	return awserr.NewRequestFailure(
		awserr.New(code, msg, nil),
		respMeta.StatusCode,
		respMeta.RequestID,
	), nil
}

// UnmarshalErrorHandler is a named request handler for unmarshaling restjson
// protocol request errors
var UnmarshalErrorHandler = request.NamedHandler{
	Name: "awssdk.restjson.UnmarshalError",
	Fn:   UnmarshalError,
}

// UnmarshalError unmarshals a response error for the REST JSON protocol.
func UnmarshalError(r *request.Request) {
	defer r.HTTPResponse.Body.Close()

	var jsonErr jsonErrorResponse
	err := jsonutil.UnmarshalJSONError(&jsonErr, r.HTTPResponse.Body)
	if err != nil {
		r.Error = awserr.NewRequestFailure(
			awserr.New(request.ErrCodeSerialization,
				"failed to unmarshal response error", err),
			r.HTTPResponse.StatusCode,
			r.RequestID,
		)
		return
	}

	code := r.HTTPResponse.Header.Get(errorTypeHeader)
	if code == "" {
		code = jsonErr.Code
	}
	msg := r.HTTPResponse.Header.Get(errorMessageHeader)
	if msg == "" {
		msg = jsonErr.Message
	}

	code = strings.SplitN(code, ":", 2)[0]
	r.Error = awserr.NewRequestFailure(
		awserr.New(code, jsonErr.Message, nil),
		r.HTTPResponse.StatusCode,
		r.RequestID,
	)
}

type jsonErrorResponse struct {
	Code    string `json:"code"`
	Message string `json:"message"`
}
