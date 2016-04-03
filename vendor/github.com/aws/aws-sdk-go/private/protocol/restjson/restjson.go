// Package restjson provides RESTful JSON serialisation of AWS
// requests and responses.
package restjson

//go:generate go run ../../../models/protocol_tests/generate.go ../../../models/protocol_tests/input/rest-json.json build_test.go
//go:generate go run ../../../models/protocol_tests/generate.go ../../../models/protocol_tests/output/rest-json.json unmarshal_test.go

import (
	"encoding/json"
	"io/ioutil"
	"strings"

	"github.com/aws/aws-sdk-go/aws/awserr"
	"github.com/aws/aws-sdk-go/aws/request"
	"github.com/aws/aws-sdk-go/private/protocol/jsonrpc"
	"github.com/aws/aws-sdk-go/private/protocol/rest"
)

// Build builds a request for the REST JSON protocol.
func Build(r *request.Request) {
	rest.Build(r)

	if t := rest.PayloadType(r.Params); t == "structure" || t == "" {
		jsonrpc.Build(r)
	}
}

// Unmarshal unmarshals a response body for the REST JSON protocol.
func Unmarshal(r *request.Request) {
	if t := rest.PayloadType(r.Data); t == "structure" || t == "" {
		jsonrpc.Unmarshal(r)
	} else {
		rest.Unmarshal(r)
	}
}

// UnmarshalMeta unmarshals response headers for the REST JSON protocol.
func UnmarshalMeta(r *request.Request) {
	rest.UnmarshalMeta(r)
}

// UnmarshalError unmarshals a response error for the REST JSON protocol.
func UnmarshalError(r *request.Request) {
	code := r.HTTPResponse.Header.Get("X-Amzn-Errortype")
	bodyBytes, err := ioutil.ReadAll(r.HTTPResponse.Body)
	if err != nil {
		r.Error = awserr.New("SerializationError", "failed reading REST JSON error response", err)
		return
	}
	if len(bodyBytes) == 0 {
		r.Error = awserr.NewRequestFailure(
			awserr.New("SerializationError", r.HTTPResponse.Status, nil),
			r.HTTPResponse.StatusCode,
			"",
		)
		return
	}
	var jsonErr jsonErrorResponse
	if err := json.Unmarshal(bodyBytes, &jsonErr); err != nil {
		r.Error = awserr.New("SerializationError", "failed decoding REST JSON error response", err)
		return
	}

	if code == "" {
		code = jsonErr.Code
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
