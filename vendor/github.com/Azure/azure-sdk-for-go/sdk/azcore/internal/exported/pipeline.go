//go:build go1.18
// +build go1.18

// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

package exported

import (
	"errors"
	"fmt"
	"net/http"

	"golang.org/x/net/http/httpguts"
)

// Policy represents an extensibility point for the Pipeline that can mutate the specified
// Request and react to the received Response.
// Exported as policy.Policy.
type Policy interface {
	// Do applies the policy to the specified Request.  When implementing a Policy, mutate the
	// request before calling req.Next() to move on to the next policy, and respond to the result
	// before returning to the caller.
	Do(req *Request) (*http.Response, error)
}

// Pipeline represents a primitive for sending HTTP requests and receiving responses.
// Its behavior can be extended by specifying policies during construction.
// Exported as runtime.Pipeline.
type Pipeline struct {
	policies []Policy
}

// Transporter represents an HTTP pipeline transport used to send HTTP requests and receive responses.
// Exported as policy.Transporter.
type Transporter interface {
	// Do sends the HTTP request and returns the HTTP response or error.
	Do(req *http.Request) (*http.Response, error)
}

// used to adapt a TransportPolicy to a Policy
type transportPolicy struct {
	trans Transporter
}

func (tp transportPolicy) Do(req *Request) (*http.Response, error) {
	if tp.trans == nil {
		return nil, errors.New("missing transporter")
	}
	resp, err := tp.trans.Do(req.Raw())
	if err != nil {
		return nil, err
	} else if resp == nil {
		// there was no response and no error (rare but can happen)
		// this ensures the retry policy will retry the request
		return nil, errors.New("received nil response")
	}
	return resp, nil
}

// NewPipeline creates a new Pipeline object from the specified Policies.
// Not directly exported, but used as part of runtime.NewPipeline().
func NewPipeline(transport Transporter, policies ...Policy) Pipeline {
	// transport policy must always be the last in the slice
	policies = append(policies, transportPolicy{trans: transport})
	return Pipeline{
		policies: policies,
	}
}

// Do is called for each and every HTTP request. It passes the request through all
// the Policy objects (which can transform the Request's URL/query parameters/headers)
// and ultimately sends the transformed HTTP request over the network.
func (p Pipeline) Do(req *Request) (*http.Response, error) {
	if req == nil {
		return nil, errors.New("request cannot be nil")
	}
	// check copied from Transport.roundTrip()
	for k, vv := range req.Raw().Header {
		if !httpguts.ValidHeaderFieldName(k) {
			if req.Raw().Body != nil {
				req.Raw().Body.Close()
			}
			return nil, fmt.Errorf("invalid header field name %q", k)
		}
		for _, v := range vv {
			if !httpguts.ValidHeaderFieldValue(v) {
				if req.Raw().Body != nil {
					req.Raw().Body.Close()
				}
				return nil, fmt.Errorf("invalid header field value %q for key %v", v, k)
			}
		}
	}
	req.policies = p.policies
	return req.Next()
}
