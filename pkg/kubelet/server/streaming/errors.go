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
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
)

func ErrorStreamingDisabled(method string) error {
	return grpc.Errorf(codes.NotFound, fmt.Sprintf("streaming method %s disabled", method))
}

func ErrorTimeout(op string, timeout time.Duration) error {
	return grpc.Errorf(codes.DeadlineExceeded, fmt.Sprintf("%s timed out after %s", op, timeout.String()))
}

// Translates a CRI streaming error into an HTTP status code.
func HTTPStatus(err error) int {
	switch grpc.Code(err) {
	case codes.NotFound:
		return http.StatusNotFound
	default:
		return http.StatusInternalServerError
	}
}
