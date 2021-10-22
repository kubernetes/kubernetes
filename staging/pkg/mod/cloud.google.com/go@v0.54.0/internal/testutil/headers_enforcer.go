// Copyright 2019 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package testutil

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"log"
	"os"
	"strings"

	"google.golang.org/api/option"
	"google.golang.org/grpc"
	"google.golang.org/grpc/metadata"
)

// HeaderChecker defines header checking and validation rules for any outgoing metadata.
type HeaderChecker struct {
	// Key is the header name to be checked against e.g. "x-goog-api-client".
	Key string

	// ValuesValidator validates the header values retrieved from mapping against
	// Key in the Headers.
	ValuesValidator func(values ...string) error
}

// HeadersEnforcer asserts that outgoing RPC headers
// are present and match expectations. If the expected headers
// are not present or don't match expectations, it'll invoke OnFailure
// with the validation error, or instead log.Fatal if OnFailure is nil.
//
// It expects that every declared key will be present in the outgoing
// RPC header and each value will be validated by the validation function.
type HeadersEnforcer struct {
	// Checkers maps header keys that are expected to be sent in the metadata
	// of outgoing gRPC requests, against the values passed into the custom
	// validation functions.
	//
	// If Checkers is nil or empty, only the default header "x-goog-api-client"
	// will be checked for.
	// Otherwise, if you supply Matchers, those keys and their respective
	// validation functions will be checked.
	Checkers []*HeaderChecker

	// OnFailure is the function that will be invoked after all validation
	// failures have been composed. If OnFailure is nil, log.Fatal will be
	// invoked instead.
	OnFailure func(fmt_ string, args ...interface{})
}

// StreamInterceptors returns a list of StreamClientInterceptor functions which
// enforce the presence and validity of expected headers during streaming RPCs.
//
// For client implementations which provide their own StreamClientInterceptor(s)
// these interceptors should be specified as the final elements to
// WithChainStreamInterceptor.
//
// Alternatively, users may apply gPRC options produced from DialOptions to
// apply all applicable gRPC interceptors.
func (h *HeadersEnforcer) StreamInterceptors() []grpc.StreamClientInterceptor {
	return []grpc.StreamClientInterceptor{h.interceptStream}
}

// UnaryInterceptors returns a list of UnaryClientInterceptor functions which
// enforce the presence and validity of expected headers during unary RPCs.
//
// For client implementations which provide their own UnaryClientInterceptor(s)
// these interceptors should be specified as the final elements to
// WithChainUnaryInterceptor.
//
// Alternatively, users may apply gPRC options produced from DialOptions to
// apply all applicable gRPC interceptors.
func (h *HeadersEnforcer) UnaryInterceptors() []grpc.UnaryClientInterceptor {
	return []grpc.UnaryClientInterceptor{h.interceptUnary}
}

// DialOptions returns gRPC DialOptions consisting of unary and stream interceptors
// to enforce the presence and validity of expected headers.
func (h *HeadersEnforcer) DialOptions() []grpc.DialOption {
	return []grpc.DialOption{
		grpc.WithChainStreamInterceptor(h.interceptStream),
		grpc.WithChainUnaryInterceptor(h.interceptUnary),
	}
}

// CallOptions returns ClientOptions consisting of unary and stream interceptors
// to enforce the presence and validity of expected headers.
func (h *HeadersEnforcer) CallOptions() (copts []option.ClientOption) {
	dopts := h.DialOptions()
	for _, dopt := range dopts {
		copts = append(copts, option.WithGRPCDialOption(dopt))
	}
	return
}

func (h *HeadersEnforcer) interceptUnary(ctx context.Context, method string, req, res interface{}, cc *grpc.ClientConn, invoker grpc.UnaryInvoker, opts ...grpc.CallOption) error {
	h.checkMetadata(ctx, method)
	return invoker(ctx, method, req, res, cc, opts...)
}

func (h *HeadersEnforcer) interceptStream(ctx context.Context, desc *grpc.StreamDesc, cc *grpc.ClientConn, method string, streamer grpc.Streamer, opts ...grpc.CallOption) (grpc.ClientStream, error) {
	h.checkMetadata(ctx, method)
	return streamer(ctx, desc, cc, method, opts...)
}

// XGoogClientHeaderChecker is a HeaderChecker that ensures that the "x-goog-api-client"
// header is present on outgoing metadata.
var XGoogClientHeaderChecker = &HeaderChecker{
	Key: "x-goog-api-client",
	ValuesValidator: func(values ...string) error {
		if len(values) == 0 {
			return errors.New("expecting values")
		}
		for _, value := range values {
			switch {
			case strings.Contains(value, "gl-go/"):
				// TODO: check for exact version strings.
				return nil

			default: // Add others here.
			}
		}
		return errors.New("unmatched values")
	},
}

// DefaultHeadersEnforcer returns a HeadersEnforcer that at bare minimum checks that
// the "x-goog-api-client" key is present in the outgoing metadata headers. On any
// validation failure, it will invoke log.Fatalf with the error message.
func DefaultHeadersEnforcer() *HeadersEnforcer {
	return &HeadersEnforcer{
		Checkers: []*HeaderChecker{XGoogClientHeaderChecker},
	}
}

func (h *HeadersEnforcer) checkMetadata(ctx context.Context, method string) {
	onFailure := h.OnFailure
	if onFailure == nil {
		lgr := log.New(os.Stderr, "", 0) // Do not log the time prefix, it is noisy in test failure logs.
		onFailure = func(fmt_ string, args ...interface{}) {
			lgr.Fatalf(fmt_, args...)
		}
	}

	md, ok := metadata.FromOutgoingContext(ctx)
	if !ok {
		onFailure("Missing metadata for method %q", method)
		return
	}
	checkers := h.Checkers
	if len(checkers) == 0 {
		// Instead use the default HeaderChecker.
		checkers = append(checkers, XGoogClientHeaderChecker)
	}

	errBuf := new(bytes.Buffer)
	for _, checker := range checkers {
		hdrKey := checker.Key
		outHdrValues, ok := md[hdrKey]
		if !ok {
			fmt.Fprintf(errBuf, "missing header %q\n", hdrKey)
			continue
		}
		if err := checker.ValuesValidator(outHdrValues...); err != nil {
			fmt.Fprintf(errBuf, "header %q: %v\n", hdrKey, err)
		}
	}

	if errBuf.Len() != 0 {
		onFailure("For method %q, errors:\n%s", method, errBuf)
		return
	}
}
