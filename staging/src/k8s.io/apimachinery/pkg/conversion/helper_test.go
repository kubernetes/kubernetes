/*
Copyright 2014 The Kubernetes Authors.

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

package conversion

import "testing"

func TestInvalidPtrValueKind(t *testing.T) {
	var simple interface{}
	switch obj := simple.(type) {
	default:
		_, err := EnforcePtr(obj)
		if err == nil {
			t.Errorf("Expected error on invalid kind")
		}
	}
}

func TestEnforceNilPtr(t *testing.T) {
	var nilPtr *struct{}
	_, err := EnforcePtr(nilPtr)
	if err == nil {
		t.Errorf("Expected error on nil pointer")
	}
}
