/*
Copyright The Kubernetes Authors.

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

package coverage

import "testing"

func TestTargetStatesHasNoDuplicates(t *testing.T) {
	seen := make(map[State]bool)
	for _, state := range TargetStates() {
		if seen[state] {
			t.Errorf("duplicate state in target states: %s", state)
		}
		seen[state] = true
	}
	t.Logf("target states: %d distinct states", len(seen))
}
