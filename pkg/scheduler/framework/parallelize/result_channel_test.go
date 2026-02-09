/*
Copyright 2019 The Kubernetes Authors.

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

package parallelize

import (
	"context"
	"errors"
	"testing"

	"k8s.io/klog/v2/ktesting"
)

func TestErrorChannel(t *testing.T) {
	errCh := NewResultChannel[error]()

	if actualErr := errCh.Receive(); actualErr != nil {
		t.Errorf("expect nil from err channel, but got %v", actualErr)
	}

	err := errors.New("unknown error")
	errCh.Send(err)
	if actualErr := errCh.Receive(); !errors.Is(actualErr, err) {
		t.Errorf("expect %v from err channel, but got %v", err, actualErr)
	}

	_, ctx := ktesting.NewTestContext(t)
	ctx, cancel := context.WithCancel(ctx)
	errCh.SendWithCancel(err, cancel)
	if actualErr := errCh.Receive(); !errors.Is(actualErr, err) {
		t.Errorf("expect %v from err channel, but got %v", err, actualErr)
	}

	if ctxErr := ctx.Err(); ctxErr != context.Canceled {
		t.Errorf("expect context canceled, but got %v", ctxErr)
	}
}
