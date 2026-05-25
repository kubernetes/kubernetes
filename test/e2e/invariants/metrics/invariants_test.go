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

package metrics

import (
	"testing"
)

func TestApiServerMetricInvariantsFieldsAreSet(t *testing.T) {
	// enforce that all fields are set for all registered metrics
	for i, inv := range apiServerInvariants {
		if inv.metricName == "" {
			t.Errorf("apiServerInvariants[%d].metricName is not set", i)
		}
		if inv.sig == "" {
			t.Errorf("apiServerInvariants[%d].sig is not set", i)
		}
		if len(inv.owners) == 0 {
			t.Errorf("apiServerInvariants[%d].owners is not set", i)
		}
		if inv.isValid == nil {
			t.Errorf("apiServerInvariants[%d].isValid is not set", i)
		}
	}
}
