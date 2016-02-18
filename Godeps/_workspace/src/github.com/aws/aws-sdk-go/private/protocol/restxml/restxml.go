// Package restxml provides RESTful XML serialisation of AWS
// requests and responses.
package restxml

//go:generate go run ../../../models/protocol_tests/generate.go ../../../models/protocol_tests/input/rest-xml.json build_test.go
//go:generate go run ../../../models/protocol_tests/generate.go ../../../models/protocol_tests/output/rest-xml.json unmarshal_test.go

import (
	"bytes"
	"encoding/xml"

	"github.com/aws/aws-sdk-go/aws/awserr"
	"github.com/aws/aws-sdk-go/aws/request"
	"github.com/aws/aws-sdk-go/private/protocol/query"
	"github.com/aws/aws-sdk-go/private/protocol/rest"
	"github.com/aws/aws-sdk-go/private/protocol/xml/xmlutil"
)

// Build builds a request payload for the REST XML protocol.
func Build(r *request.Request) {
	rest.Build(r)

	if t := rest.PayloadType(r.Params); t == "structure" || t == "" {
		var buf bytes.Buffer
		err := xmlutil.BuildXML(r.Params, xml.NewEncoder(&buf))
		if err != nil {
			r.Error = awserr.New("SerializationError", "failed to encode rest XML request", err)
			return
		}
		r.SetBufferBody(buf.Bytes())
	}
}

// Unmarshal unmarshals a payload response for the REST XML protocol.
func Unmarshal(r *request.Request) {
	if t := rest.PayloadType(r.Data); t == "structure" || t == "" {
		defer r.HTTPResponse.Body.Close()
		decoder := xml.NewDecoder(r.HTTPResponse.Body)
		err := xmlutil.UnmarshalXML(r.Data, decoder, "")
		if err != nil {
			r.Error = awserr.New("SerializationError", "failed to decode REST XML response", err)
			return
		}
	} else {
		rest.Unmarshal(r)
	}
}

// UnmarshalMeta unmarshals response headers for the REST XML protocol.
func UnmarshalMeta(r *request.Request) {
	rest.UnmarshalMeta(r)
}

// UnmarshalError unmarshals a response error for the REST XML protocol.
func UnmarshalError(r *request.Request) {
	query.UnmarshalError(r)
}
