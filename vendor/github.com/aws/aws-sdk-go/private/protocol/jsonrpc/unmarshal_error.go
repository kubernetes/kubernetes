package jsonrpc

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

	var buf bytes.Buffer
	var jsonErr jsonErrorResponse
	teeReader := io.TeeReader(resp.Body, &buf)
	err := jsonutil.UnmarshalJSONError(&jsonErr, teeReader)
	if err != nil {
		return nil, err
	}
	body := ioutil.NopCloser(&buf)

	// Code may be separated by hash(#), with the last element being the code
	// used by the SDK.
	codeParts := strings.SplitN(jsonErr.Code, "#", 2)
	code := codeParts[len(codeParts)-1]
	msg := jsonErr.Message

	if fn, ok := u.exceptions[code]; ok {
		// If exception code is know, use associated constructor to get a value
		// for the exception that the JSON body can be unmarshaled into.
		v := fn(respMeta)
		err := jsonutil.UnmarshalJSON(v, body)
		if err != nil {
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

// UnmarshalErrorHandler is a named request handler for unmarshaling jsonrpc
// protocol request errors
var UnmarshalErrorHandler = request.NamedHandler{
	Name: "awssdk.jsonrpc.UnmarshalError",
	Fn:   UnmarshalError,
}

// UnmarshalError unmarshals an error response for a JSON RPC service.
func UnmarshalError(req *request.Request) {
	defer req.HTTPResponse.Body.Close()

	var jsonErr jsonErrorResponse
	err := jsonutil.UnmarshalJSONError(&jsonErr, req.HTTPResponse.Body)
	if err != nil {
		req.Error = awserr.NewRequestFailure(
			awserr.New(request.ErrCodeSerialization,
				"failed to unmarshal error message", err),
			req.HTTPResponse.StatusCode,
			req.RequestID,
		)
		return
	}

	codes := strings.SplitN(jsonErr.Code, "#", 2)
	req.Error = awserr.NewRequestFailure(
		awserr.New(codes[len(codes)-1], jsonErr.Message, nil),
		req.HTTPResponse.StatusCode,
		req.RequestID,
	)
}

type jsonErrorResponse struct {
	Code    string `json:"__type"`
	Message string `json:"message"`
}
