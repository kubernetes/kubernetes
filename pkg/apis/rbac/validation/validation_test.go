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

func TestNonResourceURLCovers(t *testing.T) {
	tests := []struct {
		owner     string
		requested string
		want      bool
	}{
		{"*", "/api", true},
		{"/api", "/api", true},
		{"/apis", "/api", false},
		{"/api/v1", "/api", false},
		{"/api/v1", "/api/v1", true},
		{"/api/*", "/api/v1", true},
		{"/api/*", "/api", false},
		{"/api/*/*", "/api/v1", false},
		{"/*/v1/*", "/api/v1", false},
	}

	for _, tc := range tests {
		got := nonResourceURLCovers(tc.owner, tc.requested)
		if got != tc.want {
			t.Errorf("nonResourceURLCovers(%q, %q): want=(%t), got=(%t)", tc.owner, tc.requested, tc.want, got)
		}
	}
}

type validateRoleTest struct {
	role         rbac.Role
	isNamespaced bool
	wantErr      bool
	errType      field.ErrorType
	field        string
}

func (v validateRoleTest) test(t *testing.T) {
	errs := validateRole(&v.role, v.isNamespaced)
	if len(errs) == 0 {
		if v.wantErr {
			t.Fatal("expected validation error")
		}
		return
	}
	if !v.wantErr {
		t.Errorf("didn't expect error, got %v", errs)
		return
	}
	for i := range errs {
		if errs[i].Type != v.errType {
			t.Errorf("expected errors to have type %s: %v", v.errType, errs[i])
		}
		if errs[i].Field != v.field {
			t.Errorf("expected errors to have field %s: %v", v.field, errs[i])
		}
	}
}

func TestValidateRoleZeroLengthNamespace(t *testing.T) {
	validateRoleTest{
		role: rbac.Role{
			ObjectMeta: api.ObjectMeta{Name: "default"},
		},
		isNamespaced: true,
		wantErr:      true,
		errType:      field.ErrorTypeRequired,
		field:        "metadata.namespace",
	}.test(t)
}

func TestValidateRoleZeroLengthName(t *testing.T) {
	validateRoleTest{
		role: rbac.Role{
			ObjectMeta: api.ObjectMeta{Namespace: "default"},
		},
		isNamespaced: true,
		wantErr:      true,
		errType:      field.ErrorTypeRequired,
		field:        "metadata.name",
	}.test(t)
}

func TestValidateRoleValidRole(t *testing.T) {
	validateRoleTest{
		role: rbac.Role{
			ObjectMeta: api.ObjectMeta{
				Namespace: "default",
				Name:      "default",
			},
		},
		isNamespaced: true,
		wantErr:      false,
	}.test(t)
}

func TestValidateRoleValidRoleNoNamespace(t *testing.T) {
	validateRoleTest{
		role: rbac.Role{
			ObjectMeta: api.ObjectMeta{
				Name: "default",
			},
		},
		isNamespaced: false,
		wantErr:      false,
	}.test(t)
}

func TestValidateRoleNonResourceURL(t *testing.T) {
	validateRoleTest{
		role: rbac.Role{
			ObjectMeta: api.ObjectMeta{
				Name: "default",
			},
			Rules: []rbac.PolicyRule{
				{
					Verbs:           []string{"get"},
					NonResourceURLs: []string{"/*"},
				},
			},
		},
		isNamespaced: false,
		wantErr:      false,
	}.test(t)
}

func TestValidateRoleNamespacedNonResourceURL(t *testing.T) {
	validateRoleTest{
		role: rbac.Role{
			ObjectMeta: api.ObjectMeta{
				Namespace: "default",
				Name:      "default",
			},
			Rules: []rbac.PolicyRule{
				{
					// non-resource URLs are invalid for namespaced rules
					Verbs:           []string{"get"},
					NonResourceURLs: []string{"/*"},
				},
			},
		},
		isNamespaced: true,
		wantErr:      true,
		errType:      field.ErrorTypeInvalid,
		field:        "rules[0].nonResourceURLs",
	}.test(t)
}

func TestValidateRoleNonResourceURLNoVerbs(t *testing.T) {
	validateRoleTest{
		role: rbac.Role{
			ObjectMeta: api.ObjectMeta{
				Name: "default",
			},
			Rules: []rbac.PolicyRule{
				{
					Verbs:           []string{},
					NonResourceURLs: []string{"/*"},
				},
			},
		},
		isNamespaced: false,
		wantErr:      true,
		errType:      field.ErrorTypeRequired,
		field:        "rules[0].verbs",
	}.test(t)
}

func TestValidateRoleMixedNonResourceAndResource(t *testing.T) {
	validateRoleTest{
		role: rbac.Role{
			ObjectMeta: api.ObjectMeta{
				Name: "default",
			},
			Rules: []rbac.PolicyRule{
				{
					Verbs:           []string{"get"},
					NonResourceURLs: []string{"/*"},
					APIGroups:       []string{"v1"},
					Resources:       []string{"pods"},
				},
			},
		},
		wantErr: true,
		errType: field.ErrorTypeInvalid,
		field:   "rules[0].nonResourceURLs",
	}.test(t)
}

func TestValidateRoleValidResource(t *testing.T) {
	validateRoleTest{
		role: rbac.Role{
			ObjectMeta: api.ObjectMeta{
				Name: "default",
			},
			Rules: []rbac.PolicyRule{
				{
					Verbs:     []string{"get"},
					APIGroups: []string{"v1"},
					Resources: []string{"pods"},
				},
			},
		},
		wantErr: false,
	}.test(t)
}

func TestValidateRoleNoAPIGroup(t *testing.T) {
	validateRoleTest{
		role: rbac.Role{
			ObjectMeta: api.ObjectMeta{
				Name: "default",
			},
			Rules: []rbac.PolicyRule{
				{
					Verbs:     []string{"get"},
					Resources: []string{"pods"},
				},
			},
		},
		wantErr: true,
		errType: field.ErrorTypeRequired,
		field:   "rules[0].apiGroups",
	}.test(t)
}

func TestValidateRoleNoResources(t *testing.T) {
	validateRoleTest{
		role: rbac.Role{
			ObjectMeta: api.ObjectMeta{
				Name: "default",
			},
			Rules: []rbac.PolicyRule{
				{
					Verbs:     []string{"get"},
					APIGroups: []string{"v1"},
				},
			},
		},
		wantErr: true,
		errType: field.ErrorTypeRequired,
		field:   "rules[0].resources",
	}.test(t)
}
