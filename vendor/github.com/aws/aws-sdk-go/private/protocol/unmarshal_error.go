package protocol

import (
	"net/http"

	"github.com/aws/aws-sdk-go/aws/awserr"
	"github.com/aws/aws-sdk-go/aws/request"
)

// UnmarshalErrorHandler provides unmarshaling errors API response errors for
// both typed and untyped errors.
type UnmarshalErrorHandler struct {
	unmarshaler ErrorUnmarshaler
}

// ErrorUnmarshaler is an abstract interface for concrete implementations to
// unmarshal protocol specific response errors.
type ErrorUnmarshaler interface {
	UnmarshalError(*http.Response, ResponseMetadata) (error, error)
}

// NewUnmarshalErrorHandler returns an UnmarshalErrorHandler
// initialized for the set of exception names to the error unmarshalers
func NewUnmarshalErrorHandler(unmarshaler ErrorUnmarshaler) *UnmarshalErrorHandler {
	return &UnmarshalErrorHandler{
		unmarshaler: unmarshaler,
	}
}

// UnmarshalErrorHandlerName is the name of the named handler.
const UnmarshalErrorHandlerName = "awssdk.protocol.UnmarshalError"

// NamedHandler returns a NamedHandler for the unmarshaler using the set of
// errors the unmarshaler was initialized for.
func (u *UnmarshalErrorHandler) NamedHandler() request.NamedHandler {
	return request.NamedHandler{
		Name: UnmarshalErrorHandlerName,
		Fn:   u.UnmarshalError,
	}
}

// UnmarshalError will attempt to unmarshal the API response's error message
// into either a generic SDK error type, or a typed error corresponding to the
// errors exception name.
func (u *UnmarshalErrorHandler) UnmarshalError(r *request.Request) {
	defer r.HTTPResponse.Body.Close()

	respMeta := ResponseMetadata{
		StatusCode: r.HTTPResponse.StatusCode,
		RequestID:  r.RequestID,
	}

	v, err := u.unmarshaler.UnmarshalError(r.HTTPResponse, respMeta)
	if err != nil {
		r.Error = awserr.NewRequestFailure(
			awserr.New(request.ErrCodeSerialization,
				"failed to unmarshal response error", err),
			respMeta.StatusCode,
			respMeta.RequestID,
		)
		return
	}

	r.Error = v
}
