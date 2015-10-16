/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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

package group

import (
	"testing"
)

func TestRunAsAnyGenerate(t *testing.T) {
	s, err := NewRunAsAny()
	if err != nil {
		t.Fatalf("unexpected error initializing NewRunAsAny %v", err)
	}
	groups, err := s.Generate(nil)
	if len(groups) > 0 {
		t.Errorf("expected empty but got %v", groups)
	}
	if err != nil {
		t.Errorf("unexpected error generating groups: %v", err)
	}
}

func TestRunAsAnyGenerateSingle(t *testing.T) {
	s, err := NewRunAsAny()
	if err != nil {
		t.Fatalf("unexpected error initializing NewRunAsAny %v", err)
	}
	group, err := s.GenerateSingle(nil)
	if group != nil {
		t.Errorf("expected empty but got %v", group)
	}
	if err != nil {
		t.Errorf("unexpected error generating groups: %v", err)
	}
}

func TestRunAsAnyValidte(t *testing.T) {
	s, err := NewRunAsAny()
	if err != nil {
		t.Fatalf("unexpected error initializing NewRunAsAny %v", err)
	}
	errs := s.Validate(nil, nil)
	if len(errs) != 0 {
		t.Errorf("unexpected errors: %v", errs)
	}
}
