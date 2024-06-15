/*
Copyright 2024 The Kubernetes Authors.

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

package filters

import (
	"testing"

	"k8s.io/apiserver/pkg/endpoints/responsewriter"
	responsewritertesting "k8s.io/apiserver/pkg/endpoints/responsewriter/testing"
)

func TestLatencyTrackerResponseWriterDecoratorConstruction(t *testing.T) {
	inner := &responsewritertesting.FakeResponseWriter{}
	middle := &writeLatencyTracker{ResponseWriter: inner}
	outer := responsewriter.WrapForHTTP1Or2(middle)

	// FakeResponseWriter does not implement http.Flusher, FlusherError,
	// http.CloseNotifier, or http.Hijacker; so WrapForHTTP1Or2 is not
	// expected to return an outer object.
	if outer != middle {
		t.Errorf("did not expect a new outer object, but got %v", outer)
	}

	decorator, ok := outer.(responsewriter.UserProvidedDecorator)
	if !ok {
		t.Fatal("expected the middle to implement UserProvidedDecorator")
	}
	if want, got := inner, decorator.Unwrap(); want != got {
		t.Errorf("expected the decorator to return the inner http.ResponseWriter object")
	}
}

func TestLatencyTrackerResponseWriterWithFake(t *testing.T) {
	responsewritertesting.VerifyResponseWriterDecoratorWithFake(t, WithLatencyTrackers)
}
