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
	"time"

	"google.golang.org/grpc/codes"
)

type ResponseError struct {
	Err  string
	Code codes.Code
}

func (e *ResponseError) Error() string {
	return e.Err
}

func ErrorStreamingDisabled(method string) error {
	return &ResponseError{
		Err:  fmt.Sprintf("streaming method %s disabled", method),
		Code: codes.NotFound,
	}
}

func ErrorTimeout(op string, timeout time.Duration) error {
	return &ResponseError{
		Err:  fmt.Sprintf("%s timed out after %s", op, timeout.String()),
		Code: codes.DeadlineExceeded,
	}
}
