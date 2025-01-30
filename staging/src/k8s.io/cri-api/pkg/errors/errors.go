/*
Copyright 2020 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package errors

import (
	"errors"

	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

var (
	// ErrRegistryUnavailable - Get http error on the PullImage RPC call.
	ErrRegistryUnavailable = errors.New("RegistryUnavailable")

	// ErrSignatureValidationFailed - Unable to validate the image signature on the PullImage RPC call.
	ErrSignatureValidationFailed = errors.New("SignatureValidationFailed")

	// ErrRROUnsupported - Unable to enforce recursive readonly mounts
	ErrRROUnsupported = errors.New("RROUnsupported")
)

// IsNotFound returns a boolean indicating whether the error
// is grpc not found error.
// See https://github.com/grpc/grpc/blob/master/doc/statuscodes.md
// for a list of grpc status codes.
func IsNotFound(err error) bool {
	s, ok := status.FromError(err)
	if !ok {
		return ok
	}
	if s.Code() == codes.NotFound {
		return true
	}

	return false
}
