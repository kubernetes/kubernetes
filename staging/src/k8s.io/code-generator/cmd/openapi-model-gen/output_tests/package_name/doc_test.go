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

package package_name

import "testing"

func TestOutput(t *testing.T) {
	t1 := &T1{}
	if t1.OpenAPICanonicalTypeName() != "io.k8s.api.core.v1.T1" {
		t.Errorf("unexpected type name: %s", t1.OpenAPICanonicalTypeName())
	}

	t2 := &T2{}
	if t2.OpenAPICanonicalTypeName() != "io.k8s.api.core.v1.T2" {
		t.Errorf("unexpected type name: %s", t2.OpenAPICanonicalTypeName())
	}
}
