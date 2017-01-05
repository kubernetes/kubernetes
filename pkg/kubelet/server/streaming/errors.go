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
	"fmt"
	"net/http"
	"regexp"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
)

var (
	retryAfterRegexp = regexp.MustCompile(`\[Retry-After:([0-9]+)\]`)
)

func ErrorStreamingDisabled(method string) error {
	return grpc.Errorf(codes.NotFound, fmt.Sprintf("streaming method %s disabled", method))
}

func ErrorTimeout(op string, timeout time.Duration) error {
	return grpc.Errorf(codes.DeadlineExceeded, fmt.Sprintf("%s timed out after %s", op, timeout.String()))
}

// The error returned when the maximum number of in-flight requests is exceeded.
func ErrorTooManyInFlight(retryAfter int) error {
	return grpc.Errorf(codes.ResourceExhausted, "maximum number of in-flight requests exceeded [Retry-After:%d]",
		retryAfter)
}

// Translates a CRI streaming error into an appropriate HTTP response.
func WriteError(err error, w http.ResponseWriter) error {
	var status int
	switch grpc.Code(err) {
	case codes.NotFound:
		status = http.StatusNotFound
	case codes.ResourceExhausted:
		// Extract the Retry-After time from the error description.
		retryAfter := retryAfterRegexp.FindStringSubmatch(err.Error())
		if len(retryAfter) == 2 {
			w.Header().Set("Retry-After", retryAfter[1])
		}
		status = http.StatusTooManyRequests
	default:
		status = http.StatusInternalServerError
	}
	w.WriteHeader(status)
	_, writeErr := w.Write([]byte(err.Error()))
	return writeErr
}
