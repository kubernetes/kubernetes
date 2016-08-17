/*
Copyright 2016 The Kubernetes Authors.

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

package user

import (
	"testing"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/apis/extensions"
)

func TestNonRootOptions(t *testing.T) {
	_, err := NewRunAsNonRoot(nil)
	if err != nil {
		t.Fatalf("unexpected error initializing NewRunAsNonRoot %v", err)
	}
	_, err = NewRunAsNonRoot(&extensions.RunAsUserStrategyOptions{})
	if err != nil {
		t.Errorf("unexpected error initializing NewRunAsNonRoot %v", err)
	}
}

func TestNonRootGenerate(t *testing.T) {
	s, err := NewRunAsNonRoot(&extensions.RunAsUserStrategyOptions{})
	if err != nil {
		t.Fatalf("unexpected error initializing NewRunAsNonRoot %v", err)
	}
	uid, err := s.Generate(nil, nil)
	if uid != nil {
		t.Errorf("expected nil uid but got %d", *uid)
	}
	if err != nil {
		t.Errorf("unexpected error generating uid %v", err)
	}
}

func TestNonRootValidate(t *testing.T) {
	var uid int64 = 1
	var badUID int64 = 0
	s, err := NewRunAsNonRoot(&extensions.RunAsUserStrategyOptions{})
	if err != nil {
		t.Fatalf("unexpected error initializing NewMustRunAs %v", err)
	}
	container := &api.Container{
		SecurityContext: &api.SecurityContext{
			RunAsUser: &badUID,
		},
	}

	errs := s.Validate(nil, container)
	if len(errs) == 0 {
		t.Errorf("expected errors from root uid but got none")
	}

	container.SecurityContext.RunAsUser = &uid
	errs = s.Validate(nil, container)
	if len(errs) != 0 {
		t.Errorf("expected no errors from non-root uid but got %v", errs)
	}

	container.SecurityContext.RunAsUser = nil
	errs = s.Validate(nil, container)
	if len(errs) != 0 {
		t.Errorf("expected no errors from nil uid but got %v", errs)
	}
}
