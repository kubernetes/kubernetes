//go:build go1.18
// +build go1.18

// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

package runtime

import (
	"fmt"
	"net/http"
	"strings"

	"github.com/Azure/azure-sdk-for-go/sdk/azcore/internal/exported"
	"github.com/Azure/azure-sdk-for-go/sdk/azcore/policy"
	"github.com/Azure/azure-sdk-for-go/sdk/internal/errorinfo"
)

// bodyDownloadPolicy creates a policy object that downloads the response's body to a []byte.
func bodyDownloadPolicy(req *policy.Request) (*http.Response, error) {
	resp, err := req.Next()
	if err != nil {
		return resp, err
	}
	var opValues bodyDownloadPolicyOpValues
	// don't skip downloading error response bodies
	if req.OperationValue(&opValues); opValues.Skip && resp.StatusCode < 400 {
		return resp, err
	}
	// Either bodyDownloadPolicyOpValues was not specified (so skip is false)
	// or it was specified and skip is false: don't skip downloading the body
	_, err = exported.Payload(resp)
	if err != nil {
		return resp, newBodyDownloadError(err, req)
	}
	return resp, err
}

// bodyDownloadPolicyOpValues is the struct containing the per-operation values
type bodyDownloadPolicyOpValues struct {
	Skip bool
}

type bodyDownloadError struct {
	err error
}

func newBodyDownloadError(err error, req *policy.Request) error {
	// on failure, only retry the request for idempotent operations.
	// we currently identify them as DELETE, GET, and PUT requests.
	if m := strings.ToUpper(req.Raw().Method); m == http.MethodDelete || m == http.MethodGet || m == http.MethodPut {
		// error is safe for retry
		return err
	}
	// wrap error to avoid retries
	return &bodyDownloadError{
		err: err,
	}
}

func (b *bodyDownloadError) Error() string {
	return fmt.Sprintf("body download policy: %s", b.err.Error())
}

func (b *bodyDownloadError) NonRetriable() {
	// marker method
}

func (b *bodyDownloadError) Unwrap() error {
	return b.err
}

var _ errorinfo.NonRetriable = (*bodyDownloadError)(nil)
