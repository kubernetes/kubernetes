/*
Copyright 2016 The Kubernetes Authors.

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

package streaming

import (
	"net/http"
	"strconv"

	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

// NewErrorStreamingDisabled creates an error for disabled streaming method.
func NewErrorStreamingDisabled(method string) error {
	return status.Errorf(codes.NotFound, "streaming method %s disabled", method)
}

// NewErrorTooManyInFlight creates an error for exceeding the maximum number of in-flight requests.
func NewErrorTooManyInFlight() error {
	return status.Error(codes.ResourceExhausted, "maximum number of in-flight requests exceeded")
}

// WriteError translates a CRI streaming error into an appropriate HTTP response.
func WriteError(err error, w http.ResponseWriter) error {
	var status int
	switch grpc.Code(err) {
	case codes.NotFound:
		status = http.StatusNotFound
	case codes.ResourceExhausted:
		// We only expect to hit this if there is a DoS, so we just wait the full TTL.
		// If this is ever hit in steady-state operations, consider increasing the maxInFlight requests,
		// or plumbing through the time to next expiration.
		w.Header().Set("Retry-After", strconv.Itoa(int(cacheTTL.Seconds())))
		status = http.StatusTooManyRequests
	default:
		status = http.StatusInternalServerError
	}
	w.WriteHeader(status)
	_, writeErr := w.Write([]byte(err.Error()))
	return writeErr
}
