/*
Copyright 2022 The Kubernetes Authors.

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

package request

import (
	"context"
	"testing"
)

func TestCompressionDisabled(t *testing.T) {
	ctx := context.Background()

	// Default value is false.
	if got, want := CompressionDisabledFrom(ctx), false; got != want {
		t.Errorf("CompressionDisabledFrom(ctx) = %v; want = %v", got, want)
	}

	// We retrieve stored true.
	ctx = WithCompressionDisabled(ctx, true)
	if got, want := CompressionDisabledFrom(ctx), true; got != want {
		t.Errorf("CompressionDisabledFrom(ctx) = %v; want = %v", got, want)
	}

	// We retrieve stored false.
	ctx = WithCompressionDisabled(ctx, false)
	if got, want := CompressionDisabledFrom(ctx), false; got != want {
		t.Errorf("CompressionDisabledFrom(ctx) = %v; want = %v", got, want)
	}
}
