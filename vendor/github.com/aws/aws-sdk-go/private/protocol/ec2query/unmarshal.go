package ec2query

//go:generate go run -tags codegen ../../../models/protocol_tests/generate.go ../../../models/protocol_tests/output/ec2.json unmarshal_test.go

import (
	"encoding/xml"
	"io"

	"github.com/aws/aws-sdk-go/aws/awserr"
	"github.com/aws/aws-sdk-go/aws/request"
	"github.com/aws/aws-sdk-go/private/protocol/xml/xmlutil"
)

// UnmarshalHandler is a named request handler for unmarshaling ec2query protocol requests
var UnmarshalHandler = request.NamedHandler{Name: "awssdk.ec2query.Unmarshal", Fn: Unmarshal}

// UnmarshalMetaHandler is a named request handler for unmarshaling ec2query protocol request metadata
var UnmarshalMetaHandler = request.NamedHandler{Name: "awssdk.ec2query.UnmarshalMeta", Fn: UnmarshalMeta}

// UnmarshalErrorHandler is a named request handler for unmarshaling ec2query protocol request errors
var UnmarshalErrorHandler = request.NamedHandler{Name: "awssdk.ec2query.UnmarshalError", Fn: UnmarshalError}

// Unmarshal unmarshals a response body for the EC2 protocol.
func Unmarshal(r *request.Request) {
	defer r.HTTPResponse.Body.Close()
	if r.DataFilled() {
		decoder := xml.NewDecoder(r.HTTPResponse.Body)
		err := xmlutil.UnmarshalXML(r.Data, decoder, "")
		if err != nil {
			r.Error = awserr.New("SerializationError", "failed decoding EC2 Query response", err)
			return
		}
	}
}

// UnmarshalMeta unmarshals response headers for the EC2 protocol.
func UnmarshalMeta(r *request.Request) {
	// TODO implement unmarshaling of request IDs
}

type xmlErrorResponse struct {
	XMLName   xml.Name `xml:"Response"`
	Code      string   `xml:"Errors>Error>Code"`
	Message   string   `xml:"Errors>Error>Message"`
	RequestID string   `xml:"RequestID"`
}

// UnmarshalError unmarshals a response error for the EC2 protocol.
func UnmarshalError(r *request.Request) {
	defer r.HTTPResponse.Body.Close()

	resp := &xmlErrorResponse{}
	err := xml.NewDecoder(r.HTTPResponse.Body).Decode(resp)
	if err != nil && err != io.EOF {
		r.Error = awserr.New("SerializationError", "failed decoding EC2 Query error response", err)
	} else {
		r.Error = awserr.NewRequestFailure(
			awserr.New(resp.Code, resp.Message, nil),
			r.HTTPResponse.StatusCode,
			resp.RequestID,
		)
	}
}
