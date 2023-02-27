//go:build go1.18
// +build go1.18

// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

package errorinfo

// NonRetriable represents a non-transient error.  This works in
// conjunction with the retry policy, indicating that the error condition
// is idempotent, so no retries will be attempted.
// Use errors.As() to access this interface in the error chain.
type NonRetriable interface {
	error
	NonRetriable()
}
