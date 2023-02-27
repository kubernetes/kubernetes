//go:build go1.18
// +build go1.18

// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

package azcore

import "github.com/Azure/azure-sdk-for-go/sdk/azcore/internal/exported"

// ResponseError is returned when a request is made to a service and
// the service returns a non-success HTTP status code.
// Use errors.As() to access this type in the error chain.
type ResponseError = exported.ResponseError
