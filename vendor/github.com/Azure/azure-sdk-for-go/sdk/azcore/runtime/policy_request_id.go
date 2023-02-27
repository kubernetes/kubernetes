//go:build go1.18
// +build go1.18

// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

package runtime

import (
	"net/http"

	"github.com/Azure/azure-sdk-for-go/sdk/azcore/policy"
	"github.com/Azure/azure-sdk-for-go/sdk/internal/uuid"
)

type requestIDPolicy struct{}

// NewRequestIDPolicy returns a policy that add the x-ms-client-request-id header
func NewRequestIDPolicy() policy.Policy {
	return &requestIDPolicy{}
}

func (r *requestIDPolicy) Do(req *policy.Request) (*http.Response, error) {
	const requestIdHeader = "x-ms-client-request-id"
	if req.Raw().Header.Get(requestIdHeader) == "" {
		id, err := uuid.New()
		if err != nil {
			return nil, err
		}
		req.Raw().Header.Set(requestIdHeader, id.String())
	}

	return req.Next()
}
