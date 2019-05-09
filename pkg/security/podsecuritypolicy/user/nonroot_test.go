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
	api "k8s.io/api/core/v1"
	policy "k8s.io/api/policy/v1beta1"
	"testing"
)

func TestNonRootOptions(t *testing.T) {
	_, err := NewRunAsNonRoot(nil)
	if err != nil {
		t.Fatalf("unexpected error initializing NewRunAsNonRoot %v", err)
	}
	_, err = NewRunAsNonRoot(&policy.RunAsUserStrategyOptions{})
	if err != nil {
		t.Errorf("unexpected error initializing NewRunAsNonRoot %v", err)
	}
}

func TestNonRootGenerate(t *testing.T) {
	s, err := NewRunAsNonRoot(&policy.RunAsUserStrategyOptions{})
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
	goodUID := int64(1)
	badUID := int64(0)
	untrue := false
	unfalse := true
	s, err := NewRunAsNonRoot(&policy.RunAsUserStrategyOptions{})
	if err != nil {
		t.Fatalf("unexpected error initializing NewMustRunAs %v", err)
	}
	tests := []struct {
		container   *api.Container
		expectedErr bool
		msg         string
	}{
		{
			container: &api.Container{
				SecurityContext: &api.SecurityContext{
					RunAsUser: &badUID,
				},
			},
			expectedErr: true,
			msg:         "in test case %d, expected errors from root uid but got none: %v",
		},
		{
			container: &api.Container{
				SecurityContext: &api.SecurityContext{
					RunAsUser: &goodUID,
				},
			},
			expectedErr: false,
			msg:         "in test case %d, expected no errors from non-root uid but got %v",
		},
		{
			container: &api.Container{
				SecurityContext: &api.SecurityContext{
					RunAsNonRoot: &untrue,
				},
			},
			expectedErr: true,
			msg:         "in test case %d, expected errors from RunAsNonRoot but got none: %v",
		},
		{
			container: &api.Container{
				SecurityContext: &api.SecurityContext{
					RunAsNonRoot: &unfalse,
					RunAsUser:    &goodUID,
				},
			},
			expectedErr: false,
			msg:         "in test case %d, expected no errors from non-root uid but got %v",
		},
		{
			container: &api.Container{
				SecurityContext: &api.SecurityContext{
					RunAsNonRoot: nil,
					RunAsUser:    nil,
				},
			},
			expectedErr: true,
			msg:         "in test case %d, expected errors from nil runAsNonRoot and nil runAsUser but got %v",
		},
	}

	for i, tc := range tests {
		errs := s.Validate(nil, nil, nil, tc.container.SecurityContext.RunAsNonRoot, tc.container.SecurityContext.RunAsUser)
		if (len(errs) == 0) == tc.expectedErr {
			t.Errorf(tc.msg, i, errs)
		}
	}
}
