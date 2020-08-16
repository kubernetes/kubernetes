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
	"fmt"
	"testing"

	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

func TestErrorIsNotFound(t *testing.T) {
	enf := status.Errorf(codes.NotFound, "container not found")
	if !IsNotFound(enf) {
		t.Errorf("%v expected to pass not found check", enf)
	}
}

func TestSimpleErrorDoesNotTriggerNotFound(t *testing.T) {
	err := fmt.Errorf("Some random error")
	if IsNotFound(err) {
		t.Errorf("%v unexpectedly passed not found check", err)
	}
}

func TestOtherGrpcErrorDoesNotTriggerNotFound(t *testing.T) {
	gerr := status.Errorf(codes.DeadlineExceeded, "timed out")
	if IsNotFound(gerr) {
		t.Errorf("%v unexpectedly passed not found check", gerr)
	}
}
