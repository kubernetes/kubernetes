/*
Copyright 2025 The Kubernetes Authors.

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

package invariants

import "testing"

func TestApiServerMetricInvariantsFieldsAreSet(t *testing.T) {
	// enforce that all fields are set for all registered metrics
	for i, inv := range apiServerMetricInvariants {
		if inv.Metric == "" {
			t.Errorf("apiServerMetricInvariants[%d].Metric is not set", i)
		}
		if inv.SIG == "" {
			t.Errorf("apiServerMetricInvariants[%d].SIG is not set", i)
		}
		if len(inv.Owners) == 0 {
			t.Errorf("apiServerMetricInvariants[%d].Owners is not set", i)
		}
		if inv.IsValid == nil {
			t.Errorf("apiServerMetricInvariants[%d].IsValid is not set", i)
		}
	}
}
