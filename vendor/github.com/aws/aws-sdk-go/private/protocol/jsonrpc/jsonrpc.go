// Package jsonrpc provides JSON RPC utilities for serialization of AWS
// requests and responses.
package jsonrpc

//go:generate go run -tags codegen ../../../models/protocol_tests/generate.go ../../../models/protocol_tests/input/json.json build_test.go
//go:generate go run -tags codegen ../../../models/protocol_tests/generate.go ../../../models/protocol_tests/output/json.json unmarshal_test.go

import (
	"encoding/json"
	"io"
	"strings"

	"github.com/aws/aws-sdk-go/aws/awserr"
	"github.com/aws/aws-sdk-go/aws/request"
	"github.com/aws/aws-sdk-go/private/protocol/json/jsonutil"
	"github.com/aws/aws-sdk-go/private/protocol/rest"
)

var emptyJSON = []byte("{}")

// BuildHandler is a named request handler for building jsonrpc protocol requests
var BuildHandler = request.NamedHandler{Name: "awssdk.jsonrpc.Build", Fn: Build}

// UnmarshalHandler is a named request handler for unmarshaling jsonrpc protocol requests
var UnmarshalHandler = request.NamedHandler{Name: "awssdk.jsonrpc.Unmarshal", Fn: Unmarshal}

// UnmarshalMetaHandler is a named request handler for unmarshaling jsonrpc protocol request metadata
var UnmarshalMetaHandler = request.NamedHandler{Name: "awssdk.jsonrpc.UnmarshalMeta", Fn: UnmarshalMeta}

// UnmarshalErrorHandler is a named request handler for unmarshaling jsonrpc protocol request errors
var UnmarshalErrorHandler = request.NamedHandler{Name: "awssdk.jsonrpc.UnmarshalError", Fn: UnmarshalError}

// Build builds a JSON payload for a JSON RPC request.
func Build(req *request.Request) {
	var buf []byte
	var err error
	if req.ParamsFilled() {
		buf, err = jsonutil.BuildJSON(req.Params)
		if err != nil {
			req.Error = awserr.New("SerializationError", "failed encoding JSON RPC request", err)
			return
		}
	} else {
		buf = emptyJSON
	}

	if req.ClientInfo.TargetPrefix != "" || string(buf) != "{}" {
		req.SetBufferBody(buf)
	}

	if req.ClientInfo.TargetPrefix != "" {
		target := req.ClientInfo.TargetPrefix + "." + req.Operation.Name
		req.HTTPRequest.Header.Add("X-Amz-Target", target)
	}
	if req.ClientInfo.JSONVersion != "" {
		jsonVersion := req.ClientInfo.JSONVersion
		req.HTTPRequest.Header.Add("Content-Type", "application/x-amz-json-"+jsonVersion)
	}
}

// Unmarshal unmarshals a response for a JSON RPC service.
func Unmarshal(req *request.Request) {
	defer req.HTTPResponse.Body.Close()
	if req.DataFilled() {
		err := jsonutil.UnmarshalJSON(req.Data, req.HTTPResponse.Body)
		if err != nil {
			req.Error = awserr.NewRequestFailure(
				awserr.New("SerializationError", "failed decoding JSON RPC response", err),
				req.HTTPResponse.StatusCode,
				req.RequestID,
			)
		}
	}
	return
}

// UnmarshalMeta unmarshals headers from a response for a JSON RPC service.
func UnmarshalMeta(req *request.Request) {
	rest.UnmarshalMeta(req)
}

// UnmarshalError unmarshals an error response for a JSON RPC service.
func UnmarshalError(req *request.Request) {
	defer req.HTTPResponse.Body.Close()

	var jsonErr jsonErrorResponse
	err := json.NewDecoder(req.HTTPResponse.Body).Decode(&jsonErr)
	if err == io.EOF {
		req.Error = awserr.NewRequestFailure(
			awserr.New("SerializationError", req.HTTPResponse.Status, nil),
			req.HTTPResponse.StatusCode,
			req.RequestID,
		)
		return
	} else if err != nil {
		req.Error = awserr.NewRequestFailure(
			awserr.New("SerializationError", "failed decoding JSON RPC error response", err),
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
