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

package version

import (
	"testing"
)

func TestEffectiveVersionRegistry(t *testing.T) {
	r := NewComponentGlobalsRegistry()
	testComponent := "test"
	ver1 := NewEffectiveVersion("1.31")
	ver2 := NewEffectiveVersion("1.28")

	if r.EffectiveVersionFor(testComponent) != nil {
		t.Fatalf("expected nil EffectiveVersion initially")
	}
	if err := r.Register(testComponent, ver1, nil, false); err != nil {
		t.Fatalf("expected no error to register new component, but got err: %v", err)
	}
	if !r.EffectiveVersionFor(testComponent).EqualTo(ver1) {
		t.Fatalf("expected EffectiveVersionFor to return the version registered")
	}
	// overwrite
	if err := r.Register(testComponent, ver2, nil, false); err == nil {
		t.Fatalf("expected error to register existing component when override is false")
	}
	if err := r.Register(testComponent, ver2, nil, true); err != nil {
		t.Fatalf("expected no error to overriding existing component, but got err: %v", err)
	}
	if !r.EffectiveVersionFor(testComponent).EqualTo(ver2) {
		t.Fatalf("expected EffectiveVersionFor to return the version overridden")
	}
}
