package protocol

import (
	"io"
	"io/ioutil"
	"net/http"

	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/aws/client/metadata"
	"github.com/aws/aws-sdk-go/aws/request"
)

// PayloadUnmarshaler provides the interface for unmarshaling a payload's
// reader into a SDK shape.
type PayloadUnmarshaler interface {
	UnmarshalPayload(io.Reader, interface{}) error
}

// HandlerPayloadUnmarshal implements the PayloadUnmarshaler from a
// HandlerList. This provides the support for unmarshaling a payload reader to
// a shape without needing a SDK request first.
type HandlerPayloadUnmarshal struct {
	Unmarshalers request.HandlerList
}

// UnmarshalPayload unmarshals the io.Reader payload into the SDK shape using
// the Unmarshalers HandlerList provided. Returns an error if unable
// unmarshaling fails.
func (h HandlerPayloadUnmarshal) UnmarshalPayload(r io.Reader, v interface{}) error {
	req := &request.Request{
		HTTPRequest: &http.Request{},
		HTTPResponse: &http.Response{
			StatusCode: 200,
			Header:     http.Header{},
			Body:       ioutil.NopCloser(r),
		},
		Data: v,
	}

	h.Unmarshalers.Run(req)

	return req.Error
}

// PayloadMarshaler provides the interface for marshaling a SDK shape into and
// io.Writer.
type PayloadMarshaler interface {
	MarshalPayload(io.Writer, interface{}) error
}

// HandlerPayloadMarshal implements the PayloadMarshaler from a HandlerList.
// This provides support for marshaling a SDK shape into an io.Writer without
// needing a SDK request first.
type HandlerPayloadMarshal struct {
	Marshalers request.HandlerList
}

// MarshalPayload marshals the SDK shape into the io.Writer using the
// Marshalers HandlerList provided. Returns an error if unable if marshal
// fails.
func (h HandlerPayloadMarshal) MarshalPayload(w io.Writer, v interface{}) error {
	req := request.New(
		aws.Config{},
		metadata.ClientInfo{},
		request.Handlers{},
		nil,
		&request.Operation{HTTPMethod: "GET"},
		v,
		nil,
	)

	h.Marshalers.Run(req)

	if req.Error != nil {
		return req.Error
	}

	io.Copy(w, req.GetBody())

	return nil
}
