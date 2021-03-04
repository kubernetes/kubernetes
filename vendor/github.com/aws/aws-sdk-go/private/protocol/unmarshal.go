package protocol

import (
	"io"
	"io/ioutil"

	"github.com/aws/aws-sdk-go/aws/request"
)

// UnmarshalDiscardBodyHandler is a named request handler to empty and close a response's body
var UnmarshalDiscardBodyHandler = request.NamedHandler{Name: "awssdk.shared.UnmarshalDiscardBody", Fn: UnmarshalDiscardBody}

// UnmarshalDiscardBody is a request handler to empty a response's body and closing it.
func UnmarshalDiscardBody(r *request.Request) {
	if r.HTTPResponse == nil || r.HTTPResponse.Body == nil {
		return
	}

	io.Copy(ioutil.Discard, r.HTTPResponse.Body)
	r.HTTPResponse.Body.Close()
}

// ResponseMetadata provides the SDK response metadata attributes.
type ResponseMetadata struct {
	StatusCode int
	RequestID  string
}
