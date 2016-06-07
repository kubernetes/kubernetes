/*
Copyright 2016 The Kubernetes Authors All rights reserved.

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

package validation

import (
	"testing"

	api "k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/apis/rbac"
	"k8s.io/kubernetes/pkg/util/validation/field"
)

func TestValidateRoleBinding(t *testing.T) {
	errs := validateRoleBinding(
		&rbac.RoleBinding{
			ObjectMeta: api.ObjectMeta{Namespace: api.NamespaceDefault, Name: "master"},
			RoleRef:    api.ObjectReference{Namespace: "master", Name: "valid"},
			Subjects: []rbac.Subject{
				{Name: "validsaname", Kind: rbac.ServiceAccountKind},
				{Name: "valid@username", Kind: rbac.UserKind},
				{Name: "valid@groupname", Kind: rbac.GroupKind},
			},
		},
		true,
	)
	if len(errs) != 0 {
		t.Errorf("expected success: %v", errs)
	}

	errorCases := map[string]struct {
		A rbac.RoleBinding
		T field.ErrorType
		F string
	}{
		"zero-length namespace": {
			A: rbac.RoleBinding{
				ObjectMeta: api.ObjectMeta{Name: "default"},
				RoleRef:    api.ObjectReference{Namespace: "master", Name: "valid"},
			},
			T: field.ErrorTypeRequired,
			F: "metadata.namespace",
		},
		"zero-length name": {
			A: rbac.RoleBinding{
				ObjectMeta: api.ObjectMeta{Namespace: api.NamespaceDefault},
				RoleRef:    api.ObjectReference{Namespace: "master", Name: "valid"},
			},
			T: field.ErrorTypeRequired,
			F: "metadata.name",
		},
		"invalid ref": {
			A: rbac.RoleBinding{
				ObjectMeta: api.ObjectMeta{Namespace: api.NamespaceDefault, Name: "name"},
				RoleRef:    api.ObjectReference{Namespace: "-192083", Name: "valid"},
			},
			T: field.ErrorTypeInvalid,
			F: "roleRef.namespace",
		},
		"bad role": {
			A: rbac.RoleBinding{
				ObjectMeta: api.ObjectMeta{Namespace: api.NamespaceDefault, Name: "default"},
				RoleRef:    api.ObjectReference{Namespace: "default"},
			},
			T: field.ErrorTypeRequired,
			F: "roleRef.name",
		},
		"bad subject kind": {
			A: rbac.RoleBinding{
				ObjectMeta: api.ObjectMeta{Namespace: api.NamespaceDefault, Name: "master"},
				RoleRef:    api.ObjectReference{Namespace: "master", Name: "valid"},
				Subjects:   []rbac.Subject{{Name: "subject"}},
			},
			T: field.ErrorTypeNotSupported,
			F: "subjects[0].kind",
		},
		"bad subject name": {
			A: rbac.RoleBinding{
				ObjectMeta: api.ObjectMeta{Namespace: api.NamespaceDefault, Name: "master"},
				RoleRef:    api.ObjectReference{Namespace: "master", Name: "valid"},
				Subjects:   []rbac.Subject{{Name: "subject:bad", Kind: rbac.ServiceAccountKind}},
			},
			T: field.ErrorTypeInvalid,
			F: "subjects[0].name",
		},
		"missing subject name": {
			A: rbac.RoleBinding{
				ObjectMeta: api.ObjectMeta{Namespace: api.NamespaceDefault, Name: "master"},
				RoleRef:    api.ObjectReference{Namespace: "master", Name: "valid"},
				Subjects:   []rbac.Subject{{Kind: rbac.ServiceAccountKind}},
			},
			T: field.ErrorTypeRequired,
			F: "subjects[0].name",
		},
	}
	for k, v := range errorCases {
		errs := validateRoleBinding(&v.A, true)
		if len(errs) == 0 {
			t.Errorf("expected failure %s for %v", k, v.A)
			continue
		}
		for i := range errs {
			if errs[i].Type != v.T {
				t.Errorf("%s: expected errors to have type %s: %v", k, v.T, errs[i])
			}
			if errs[i].Field != v.F {
				t.Errorf("%s: expected errors to have field %s: %v", k, v.F, errs[i])
			}
		}
	}
}

func TestValidateRoleBindingUpdate(t *testing.T) {
	old := &rbac.RoleBinding{
		ObjectMeta: api.ObjectMeta{Namespace: api.NamespaceDefault, Name: "master", ResourceVersion: "1"},
		RoleRef:    api.ObjectReference{Namespace: "master", Name: "valid"},
	}

	errs := validateRoleBindingUpdate(
		&rbac.RoleBinding{
			ObjectMeta: api.ObjectMeta{Namespace: api.NamespaceDefault, Name: "master", ResourceVersion: "1"},
			RoleRef:    api.ObjectReference{Namespace: "master", Name: "valid"},
		},
		old,
		true,
	)
	if len(errs) != 0 {
		t.Errorf("expected success: %v", errs)
	}

	errorCases := map[string]struct {
		A rbac.RoleBinding
		T field.ErrorType
		F string
	}{
		"changedRef": {
			A: rbac.RoleBinding{
				ObjectMeta: api.ObjectMeta{Namespace: api.NamespaceDefault, Name: "master", ResourceVersion: "1"},
				RoleRef:    api.ObjectReference{Namespace: "master", Name: "changed"},
			},
			T: field.ErrorTypeInvalid,
			F: "roleRef",
		},
	}
	for k, v := range errorCases {
		errs := validateRoleBindingUpdate(&v.A, old, true)
		if len(errs) == 0 {
			t.Errorf("expected failure %s for %v", k, v.A)
			continue
		}
		for i := range errs {
			if errs[i].Type != v.T {
				t.Errorf("%s: expected errors to have type %s: %v", k, v.T, errs[i])
			}
			if errs[i].Field != v.F {
				t.Errorf("%s: expected errors to have field %s: %v", k, v.F, errs[i])
			}
		}
	}
}

func TestValidateRole(t *testing.T) {
	errs := validateRole(
		&rbac.Role{
			ObjectMeta: api.ObjectMeta{Namespace: api.NamespaceDefault, Name: "master"},
		},
		true,
	)
	if len(errs) != 0 {
		t.Errorf("expected success: %v", errs)
	}

	errorCases := map[string]struct {
		A rbac.Role
		T field.ErrorType
		F string
	}{
		"zero-length namespace": {
			A: rbac.Role{
				ObjectMeta: api.ObjectMeta{Name: "default"},
			},
			T: field.ErrorTypeRequired,
			F: "metadata.namespace",
		},
		"zero-length name": {
			A: rbac.Role{
				ObjectMeta: api.ObjectMeta{Namespace: api.NamespaceDefault},
			},
			T: field.ErrorTypeRequired,
			F: "metadata.name",
		},
	}
	for k, v := range errorCases {
		errs := validateRole(&v.A, true)
		if len(errs) == 0 {
			t.Errorf("expected failure %s for %v", k, v.A)
			continue
		}
		for i := range errs {
			if errs[i].Type != v.T {
				t.Errorf("%s: expected errors to have type %s: %v", k, v.T, errs[i])
			}
			if errs[i].Field != v.F {
				t.Errorf("%s: expected errors to have field %s: %v", k, v.F, errs[i])
			}
		}
	}
}
