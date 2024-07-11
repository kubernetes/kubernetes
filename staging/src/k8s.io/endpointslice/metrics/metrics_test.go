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

package metrics

import (
	"testing"
)

// TestDoOnce verifies metrics are registered only once.
// legacyregistry will panic if duplicate metrics collector registration attempted.
func TestDoOnce(t *testing.T) {
	endpointSliceMetrics1 := NewEndpointSliceMetrics("subsystem")
	endpointSliceMetrics2 := NewEndpointSliceMetrics("subsystem")
	if endpointSliceMetrics1 != endpointSliceMetrics2 {
		t.Errorf("Expected NewEndpointSliceMetrics to return the same instance for the same subsystem")
	}

	endpointSliceMetrics3 := NewEndpointSliceMetrics("subsystem_b")
	if endpointSliceMetrics1 == endpointSliceMetrics3 {
		t.Errorf("Expected NewEndpointSliceMetrics to return the different instance for a different subsystem")
	}
}
