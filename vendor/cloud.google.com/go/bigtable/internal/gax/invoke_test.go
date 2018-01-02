/*
Copyright 2015 Google Inc. All Rights Reserved.

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
package gax

import (
	"testing"
	"time"

	"golang.org/x/net/context"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
)

func TestRandomizedDelays(t *testing.T) {
	max := 200 * time.Millisecond
	settings := []CallOption{
		WithRetryCodes([]codes.Code{codes.Unavailable, codes.DeadlineExceeded}),
		WithDelayTimeoutSettings(10*time.Millisecond, max, 1.5),
	}

	deadline := time.Now().Add(1 * time.Second)
	ctx, _ := context.WithDeadline(context.Background(), deadline)
	var invokeTime time.Time
	Invoke(ctx, func(childCtx context.Context) error {
		// Keep failing, make sure we never slept more than max (plus a fudge factor)
		if !invokeTime.IsZero() {
			if time.Since(invokeTime) > (max + 20*time.Millisecond) {
				t.Fatalf("Slept too long: %v", max)
			}
		}
		invokeTime = time.Now()
		// Workaround for `go vet`: https://github.com/grpc/grpc-go/issues/90
		errf := grpc.Errorf
		return errf(codes.Unavailable, "")
	}, settings...)
}
