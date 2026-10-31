/*
Copyright 2026 The Kubernetes Authors.

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

package v1

import (
	"encoding/json"
	"testing"
)

func TestWithShadowedField(t *testing.T) {
	applyConfig := TestTypeWithShadowedField().WithKind("direct")

	data, err := json.Marshal(applyConfig)
	if err != nil {
		t.Fatalf("failed to marshal apply configuration: %v", err)
	}
	if got, want := string(data), `{"kind":"direct"}`; got != want {
		t.Errorf("unexpected apply configuration: got %s, want %s", got, want)
	}
}
