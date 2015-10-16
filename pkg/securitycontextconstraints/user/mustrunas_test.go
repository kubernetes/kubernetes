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

package user

import (
	"testing"

	"k8s.io/kubernetes/pkg/api"
)

func TestMustRunAsOptions(t *testing.T) {
	var uid int64 = 1
	tests := map[string]struct {
		opts *api.RunAsUserStrategyOptions
		pass bool
	}{
		"invalid opts": {
			opts: &api.RunAsUserStrategyOptions{},
			pass: false,
		},
		"valid opts": {
			opts: &api.RunAsUserStrategyOptions{UID: &uid},
			pass: true,
		},
	}
	for name, tc := range tests {
		_, err := NewMustRunAs(tc.opts)
		if err != nil && tc.pass {
			t.Errorf("%s expected to pass but received error %v", name, err)
		}
		if err == nil && !tc.pass {
			t.Errorf("%s expected to fail but did not receive an error", name)
		}
	}
}

func TestMustRunAsGenerate(t *testing.T) {
	var uid int64 = 1
	opts := &api.RunAsUserStrategyOptions{UID: &uid}
	mustRunAs, err := NewMustRunAs(opts)
	if err != nil {
		t.Fatalf("unexpected error initializing NewMustRunAs %v", err)
	}
	generated, err := mustRunAs.Generate(nil, nil)
	if err != nil {
		t.Fatalf("unexpected error generating uid %v", err)
	}
	if *generated != uid {
		t.Errorf("generated uid does not equal configured uid")
	}
}

func TestMustRunAsValidate(t *testing.T) {
	var uid int64 = 1
	var badUID int64 = 2
	opts := &api.RunAsUserStrategyOptions{UID: &uid}
	mustRunAs, err := NewMustRunAs(opts)
	if err != nil {
		t.Fatalf("unexpected error initializing NewMustRunAs %v", err)
	}
	container := &api.Container{
		SecurityContext: &api.SecurityContext{
			RunAsUser: &badUID,
		},
	}

	errs := mustRunAs.Validate(nil, container)
	if len(errs) == 0 {
		t.Errorf("expected errors from mismatch uid but got none")
	}

	container.SecurityContext.RunAsUser = &uid
	errs = mustRunAs.Validate(nil, container)
	if len(errs) != 0 {
		t.Errorf("expected no errors from matching uid but got %v", errs)
	}
}
