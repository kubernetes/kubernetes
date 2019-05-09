package query

//go:generate go run -tags codegen ../../../models/protocol_tests/generate.go ../../../models/protocol_tests/output/query.json unmarshal_test.go

import (
	"encoding/xml"

	"github.com/aws/aws-sdk-go/aws/awserr"
	"github.com/aws/aws-sdk-go/aws/request"
	"github.com/aws/aws-sdk-go/private/protocol/xml/xmlutil"
)

// UnmarshalHandler is a named request handler for unmarshaling query protocol requests
var UnmarshalHandler = request.NamedHandler{Name: "awssdk.query.Unmarshal", Fn: Unmarshal}

// UnmarshalMetaHandler is a named request handler for unmarshaling query protocol request metadata
var UnmarshalMetaHandler = request.NamedHandler{Name: "awssdk.query.UnmarshalMeta", Fn: UnmarshalMeta}

// Unmarshal unmarshals a response for an AWS Query service.
func Unmarshal(r *request.Request) {
	defer r.HTTPResponse.Body.Close()
	if r.DataFilled() {
		decoder := xml.NewDecoder(r.HTTPResponse.Body)
		err := xmlutil.UnmarshalXML(r.Data, decoder, r.Operation.Name+"Result")
		if err != nil {
			r.Error = awserr.NewRequestFailure(
				awserr.New("SerializationError", "failed decoding Query response", err),
				r.HTTPResponse.StatusCode,
				r.RequestID,
			)
			return
		}
	}
}

// UnmarshalMeta unmarshals header response values for an AWS Query service.
func UnmarshalMeta(r *request.Request) {
	r.RequestID = r.HTTPResponse.Header.Get("X-Amzn-Requestid")
}
