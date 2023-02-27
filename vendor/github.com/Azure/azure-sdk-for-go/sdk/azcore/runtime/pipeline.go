//go:build go1.18
// +build go1.18

// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

package runtime

import (
	"net/http"

	"github.com/Azure/azure-sdk-for-go/sdk/azcore/internal/exported"
	"github.com/Azure/azure-sdk-for-go/sdk/azcore/policy"
)

// PipelineOptions contains Pipeline options for SDK developers
type PipelineOptions struct {
	AllowedHeaders, AllowedQueryParameters []string
	PerCall, PerRetry                      []policy.Policy
}

// Pipeline represents a primitive for sending HTTP requests and receiving responses.
// Its behavior can be extended by specifying policies during construction.
type Pipeline = exported.Pipeline

// NewPipeline creates a pipeline from connection options, with any additional policies as specified.
// Policies from ClientOptions are placed after policies from PipelineOptions.
// The module and version parameters are used by the telemetry policy, when enabled.
func NewPipeline(module, version string, plOpts PipelineOptions, options *policy.ClientOptions) Pipeline {
	cp := policy.ClientOptions{}
	if options != nil {
		cp = *options
	}
	if len(plOpts.AllowedHeaders) > 0 {
		headers := make([]string, 0, len(plOpts.AllowedHeaders)+len(cp.Logging.AllowedHeaders))
		copy(headers, plOpts.AllowedHeaders)
		headers = append(headers, cp.Logging.AllowedHeaders...)
		cp.Logging.AllowedHeaders = headers
	}
	if len(plOpts.AllowedQueryParameters) > 0 {
		qp := make([]string, 0, len(plOpts.AllowedQueryParameters)+len(cp.Logging.AllowedQueryParams))
		copy(qp, plOpts.AllowedQueryParameters)
		qp = append(qp, cp.Logging.AllowedQueryParams...)
		cp.Logging.AllowedQueryParams = qp
	}
	// we put the includeResponsePolicy at the very beginning so that the raw response
	// is populated with the final response (some policies might mutate the response)
	policies := []policy.Policy{policyFunc(includeResponsePolicy)}
	if !cp.Telemetry.Disabled {
		policies = append(policies, NewTelemetryPolicy(module, version, &cp.Telemetry))
	}
	policies = append(policies, plOpts.PerCall...)
	policies = append(policies, cp.PerCallPolicies...)
	policies = append(policies, NewRetryPolicy(&cp.Retry))
	policies = append(policies, plOpts.PerRetry...)
	policies = append(policies, cp.PerRetryPolicies...)
	policies = append(policies, NewLogPolicy(&cp.Logging))
	policies = append(policies, policyFunc(httpHeaderPolicy), policyFunc(bodyDownloadPolicy))
	transport := cp.Transport
	if transport == nil {
		transport = defaultHTTPClient
	}
	return exported.NewPipeline(transport, policies...)
}

// policyFunc is a type that implements the Policy interface.
// Use this type when implementing a stateless policy as a first-class function.
type policyFunc func(*policy.Request) (*http.Response, error)

// Do implements the Policy interface on policyFunc.
func (pf policyFunc) Do(req *policy.Request) (*http.Response, error) {
	return pf(req)
}
