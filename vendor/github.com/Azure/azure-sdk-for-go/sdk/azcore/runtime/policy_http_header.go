//go:build go1.18
// +build go1.18

// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

package runtime

import (
	"context"
	"net/http"

	"github.com/Azure/azure-sdk-for-go/sdk/azcore/internal/shared"
	"github.com/Azure/azure-sdk-for-go/sdk/azcore/policy"
)

// newHTTPHeaderPolicy creates a policy object that adds custom HTTP headers to a request
func httpHeaderPolicy(req *policy.Request) (*http.Response, error) {
	// check if any custom HTTP headers have been specified
	if header := req.Raw().Context().Value(shared.CtxWithHTTPHeaderKey{}); header != nil {
		for k, v := range header.(http.Header) {
			// use Set to replace any existing value
			// it also canonicalizes the header key
			req.Raw().Header.Set(k, v[0])
			// add any remaining values
			for i := 1; i < len(v); i++ {
				req.Raw().Header.Add(k, v[i])
			}
		}
	}
	return req.Next()
}

// WithHTTPHeader adds the specified http.Header to the parent context.
// Use this to specify custom HTTP headers at the API-call level.
// Any overlapping headers will have their values replaced with the values specified here.
func WithHTTPHeader(parent context.Context, header http.Header) context.Context {
	return context.WithValue(parent, shared.CtxWithHTTPHeaderKey{}, header)
}
