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

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/kubernetes/pkg/apis/rbac"
)

func TestValidateClusterRoleBinding(t *testing.T) {
	errs := ValidateClusterRoleBinding(
		&rbac.ClusterRoleBinding{
			ObjectMeta: metav1.ObjectMeta{Name: "master"},
			RoleRef:    rbac.RoleRef{APIGroup: rbac.GroupName, Kind: "ClusterRole", Name: "valid"},
			Subjects: []rbac.Subject{
				{Name: "validsaname", APIGroup: "", Namespace: "foo", Kind: rbac.ServiceAccountKind},
				{Name: "valid@username", APIGroup: rbac.GroupName, Kind: rbac.UserKind},
				{Name: "valid@groupname", APIGroup: rbac.GroupName, Kind: rbac.GroupKind},
			},
		},
	)
	if len(errs) != 0 {
		t.Errorf("expected success: %v", errs)
	}

	errorCases := map[string]struct {
		A rbac.ClusterRoleBinding
		T field.ErrorType
		F string
	}{
		"bad group": {
			A: rbac.ClusterRoleBinding{
				ObjectMeta: metav1.ObjectMeta{Name: "default"},
				RoleRef:    rbac.RoleRef{APIGroup: "rbac.GroupName", Kind: "ClusterRole", Name: "valid"},
			},
			T: field.ErrorTypeNotSupported,
			F: "roleRef.apiGroup",
		},
		"bad kind": {
			A: rbac.ClusterRoleBinding{
				ObjectMeta: metav1.ObjectMeta{Name: "default"},
				RoleRef:    rbac.RoleRef{APIGroup: rbac.GroupName, Kind: "Type", Name: "valid"},
			},
			T: field.ErrorTypeNotSupported,
			F: "roleRef.kind",
		},
		"reference role": {
			A: rbac.ClusterRoleBinding{
				ObjectMeta: metav1.ObjectMeta{Name: "default"},
				RoleRef:    rbac.RoleRef{APIGroup: rbac.GroupName, Kind: "Role", Name: "valid"},
			},
			T: field.ErrorTypeNotSupported,
			F: "roleRef.kind",
		},
		"zero-length name": {
			A: rbac.ClusterRoleBinding{
				ObjectMeta: metav1.ObjectMeta{},
				RoleRef:    rbac.RoleRef{APIGroup: rbac.GroupName, Kind: "ClusterRole", Name: "valid"},
			},
			T: field.ErrorTypeRequired,
			F: "metadata.name",
		},
		"bad role": {
			A: rbac.ClusterRoleBinding{
				ObjectMeta: metav1.ObjectMeta{Name: "default"},
				RoleRef:    rbac.RoleRef{APIGroup: rbac.GroupName, Kind: "ClusterRole"},
			},
			T: field.ErrorTypeRequired,
			F: "roleRef.name",
		},
		"bad subject kind": {
			A: rbac.ClusterRoleBinding{
				ObjectMeta: metav1.ObjectMeta{Name: "master"},
				RoleRef:    rbac.RoleRef{APIGroup: rbac.GroupName, Kind: "ClusterRole", Name: "valid"},
				Subjects:   []rbac.Subject{{Name: "subject"}},
			},
			T: field.ErrorTypeNotSupported,
			F: "subjects[0].kind",
		},
		"bad subject name": {
			A: rbac.ClusterRoleBinding{
				ObjectMeta: metav1.ObjectMeta{Name: "master"},
				RoleRef:    rbac.RoleRef{APIGroup: rbac.GroupName, Kind: "ClusterRole", Name: "valid"},
				Subjects:   []rbac.Subject{{Namespace: "foo", Name: "subject:bad", Kind: rbac.ServiceAccountKind}},
			},
			T: field.ErrorTypeInvalid,
			F: "subjects[0].name",
		},
		"missing SA namespace": {
			A: rbac.ClusterRoleBinding{
				ObjectMeta: metav1.ObjectMeta{Name: "master"},
				RoleRef:    rbac.RoleRef{APIGroup: rbac.GroupName, Kind: "ClusterRole", Name: "valid"},
				Subjects:   []rbac.Subject{{Name: "good", Kind: rbac.ServiceAccountKind}},
			},
			T: field.ErrorTypeRequired,
			F: "subjects[0].namespace",
		},
		"missing subject name": {
			A: rbac.ClusterRoleBinding{
				ObjectMeta: metav1.ObjectMeta{Name: "master"},
				RoleRef:    rbac.RoleRef{APIGroup: rbac.GroupName, Kind: "ClusterRole", Name: "valid"},
				Subjects:   []rbac.Subject{{Namespace: "foo", Kind: rbac.ServiceAccountKind}},
			},
			T: field.ErrorTypeRequired,
			F: "subjects[0].name",
		},
	}
	for k, v := range errorCases {
		errs := ValidateClusterRoleBinding(&v.A)
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

func TestValidateRoleBinding(t *testing.T) {
	errs := ValidateRoleBinding(
		&rbac.RoleBinding{
			ObjectMeta: metav1.ObjectMeta{Namespace: metav1.NamespaceDefault, Name: "master"},
			RoleRef:    rbac.RoleRef{APIGroup: rbac.GroupName, Kind: "Role", Name: "valid"},
			Subjects: []rbac.Subject{
				{Name: "validsaname", APIGroup: "", Kind: rbac.ServiceAccountKind},
				{Name: "valid@username", APIGroup: rbac.GroupName, Kind: rbac.UserKind},
				{Name: "valid@groupname", APIGroup: rbac.GroupName, Kind: rbac.GroupKind},
			},
		},
	)
	if len(errs) != 0 {
		t.Errorf("expected success: %v", errs)
	}

	errorCases := map[string]struct {
		A rbac.RoleBinding
		T field.ErrorType
		F string
	}{
		"bad group": {
			A: rbac.RoleBinding{
				ObjectMeta: metav1.ObjectMeta{Namespace: metav1.NamespaceDefault, Name: "default"},
				RoleRef:    rbac.RoleRef{APIGroup: "rbac.GroupName", Kind: "ClusterRole", Name: "valid"},
			},
			T: field.ErrorTypeNotSupported,
			F: "roleRef.apiGroup",
		},
		"bad kind": {
			A: rbac.RoleBinding{
				ObjectMeta: metav1.ObjectMeta{Namespace: metav1.NamespaceDefault, Name: "default"},
				RoleRef:    rbac.RoleRef{APIGroup: rbac.GroupName, Kind: "Type", Name: "valid"},
			},
			T: field.ErrorTypeNotSupported,
			F: "roleRef.kind",
		},
		"zero-length namespace": {
			A: rbac.RoleBinding{
				ObjectMeta: metav1.ObjectMeta{Name: "default"},
				RoleRef:    rbac.RoleRef{APIGroup: rbac.GroupName, Kind: "Role", Name: "valid"},
			},
			T: field.ErrorTypeRequired,
			F: "metadata.namespace",
		},
		"zero-length name": {
			A: rbac.RoleBinding{
				ObjectMeta: metav1.ObjectMeta{Namespace: metav1.NamespaceDefault},
				RoleRef:    rbac.RoleRef{APIGroup: rbac.GroupName, Kind: "Role", Name: "valid"},
			},
			T: field.ErrorTypeRequired,
			F: "metadata.name",
		},
		"bad role": {
			A: rbac.RoleBinding{
				ObjectMeta: metav1.ObjectMeta{Namespace: metav1.NamespaceDefault, Name: "default"},
				RoleRef:    rbac.RoleRef{APIGroup: rbac.GroupName, Kind: "Role"},
			},
			T: field.ErrorTypeRequired,
			F: "roleRef.name",
		},
		"bad subject kind": {
			A: rbac.RoleBinding{
				ObjectMeta: metav1.ObjectMeta{Namespace: metav1.NamespaceDefault, Name: "master"},
				RoleRef:    rbac.RoleRef{APIGroup: rbac.GroupName, Kind: "Role", Name: "valid"},
				Subjects:   []rbac.Subject{{Name: "subject"}},
			},
			T: field.ErrorTypeNotSupported,
			F: "subjects[0].kind",
		},
		"bad subject name": {
			A: rbac.RoleBinding{
				ObjectMeta: metav1.ObjectMeta{Namespace: metav1.NamespaceDefault, Name: "master"},
				RoleRef:    rbac.RoleRef{APIGroup: rbac.GroupName, Kind: "Role", Name: "valid"},
				Subjects:   []rbac.Subject{{Name: "subject:bad", Kind: rbac.ServiceAccountKind}},
			},
			T: field.ErrorTypeInvalid,
			F: "subjects[0].name",
		},
		"missing subject name": {
			A: rbac.RoleBinding{
				ObjectMeta: metav1.ObjectMeta{Namespace: metav1.NamespaceDefault, Name: "master"},
				RoleRef:    rbac.RoleRef{APIGroup: rbac.GroupName, Kind: "Role", Name: "valid"},
				Subjects:   []rbac.Subject{{Kind: rbac.ServiceAccountKind}},
			},
			T: field.ErrorTypeRequired,
			F: "subjects[0].name",
		},
	}
	for k, v := range errorCases {
		errs := ValidateRoleBinding(&v.A)
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
		ObjectMeta: metav1.ObjectMeta{Namespace: metav1.NamespaceDefault, Name: "master", ResourceVersion: "1"},
		RoleRef:    rbac.RoleRef{APIGroup: rbac.GroupName, Kind: "Role", Name: "valid"},
	}

	errs := ValidateRoleBindingUpdate(
		&rbac.RoleBinding{
			ObjectMeta: metav1.ObjectMeta{Namespace: metav1.NamespaceDefault, Name: "master", ResourceVersion: "1"},
			RoleRef:    rbac.RoleRef{APIGroup: rbac.GroupName, Kind: "Role", Name: "valid"},
		},
		old,
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
				ObjectMeta: metav1.ObjectMeta{Namespace: metav1.NamespaceDefault, Name: "master", ResourceVersion: "1"},
				RoleRef:    rbac.RoleRef{APIGroup: rbac.GroupName, Kind: "Role", Name: "changed"},
			},
			T: field.ErrorTypeInvalid,
			F: "roleRef",
		},
	}
	for k, v := range errorCases {
		errs := ValidateRoleBindingUpdate(&v.A, old)
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

type ValidateRoleTest struct {
	role    rbac.Role
	wantErr bool
	errType field.ErrorType
	field   string
}

func (v ValidateRoleTest) test(t *testing.T) {
	errs := ValidateRole(&v.role)
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

type ValidateClusterRoleTest struct {
	role    rbac.ClusterRole
	wantErr bool
	errType field.ErrorType
	field   string
}

func (v ValidateClusterRoleTest) test(t *testing.T) {
	errs := ValidateClusterRole(&v.role, ClusterRoleValidationOptions{false})
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
	ValidateRoleTest{
		role: rbac.Role{
			ObjectMeta: metav1.ObjectMeta{Name: "default"},
		},
		wantErr: true,
		errType: field.ErrorTypeRequired,
		field:   "metadata.namespace",
	}.test(t)
}

func TestValidateRoleZeroLengthName(t *testing.T) {
	ValidateRoleTest{
		role: rbac.Role{
			ObjectMeta: metav1.ObjectMeta{Namespace: "default"},
		},
		wantErr: true,
		errType: field.ErrorTypeRequired,
		field:   "metadata.name",
	}.test(t)
}

func TestValidateRoleValidRole(t *testing.T) {
	ValidateRoleTest{
		role: rbac.Role{
			ObjectMeta: metav1.ObjectMeta{
				Namespace: "default",
				Name:      "default",
			},
		},
		wantErr: false,
	}.test(t)
}

func TestValidateRoleValidRoleNoNamespace(t *testing.T) {
	ValidateClusterRoleTest{
		role: rbac.ClusterRole{
			ObjectMeta: metav1.ObjectMeta{
				Name: "default",
			},
		},
		wantErr: false,
	}.test(t)
}

func TestValidateRoleNonResourceURL(t *testing.T) {
	ValidateClusterRoleTest{
		role: rbac.ClusterRole{
			ObjectMeta: metav1.ObjectMeta{
				Name: "default",
			},
			Rules: []rbac.PolicyRule{{
				Verbs:           []string{"get"},
				NonResourceURLs: []string{"/*"},
			}},
		},
		wantErr: false,
	}.test(t)
}

func TestValidateRoleNamespacedNonResourceURL(t *testing.T) {
	ValidateRoleTest{
		role: rbac.Role{
			ObjectMeta: metav1.ObjectMeta{
				Namespace: "default",
				Name:      "default",
			},
			Rules: []rbac.PolicyRule{{
				// non-resource URLs are invalid for namespaced rules
				Verbs:           []string{"get"},
				NonResourceURLs: []string{"/*"},
			}},
		},
		wantErr: true,
		errType: field.ErrorTypeInvalid,
		field:   "rules[0].nonResourceURLs",
	}.test(t)
}

func TestValidateRoleNonResourceURLNoVerbs(t *testing.T) {
	ValidateClusterRoleTest{
		role: rbac.ClusterRole{
			ObjectMeta: metav1.ObjectMeta{
				Name: "default",
			},
			Rules: []rbac.PolicyRule{{
				Verbs:           []string{},
				NonResourceURLs: []string{"/*"},
			}},
		},
		wantErr: true,
		errType: field.ErrorTypeRequired,
		field:   "rules[0].verbs",
	}.test(t)
}

func TestValidateRoleMixedNonResourceAndResource(t *testing.T) {
	ValidateRoleTest{
		role: rbac.Role{
			ObjectMeta: metav1.ObjectMeta{
				Name:      "default",
				Namespace: "default",
			},
			Rules: []rbac.PolicyRule{{
				Verbs:           []string{"get"},
				NonResourceURLs: []string{"/*"},
				APIGroups:       []string{"v1"},
				Resources:       []string{"pods"},
			}},
		},
		wantErr: true,
		errType: field.ErrorTypeInvalid,
		field:   "rules[0].nonResourceURLs",
	}.test(t)
}

func TestValidateRoleValidResource(t *testing.T) {
	ValidateRoleTest{
		role: rbac.Role{
			ObjectMeta: metav1.ObjectMeta{
				Name:      "default",
				Namespace: "default",
			},
			Rules: []rbac.PolicyRule{{
				Verbs:     []string{"get"},
				APIGroups: []string{"v1"},
				Resources: []string{"pods"},
			}},
		},
		wantErr: false,
	}.test(t)
}

func TestValidateRoleNoAPIGroup(t *testing.T) {
	ValidateRoleTest{
		role: rbac.Role{
			ObjectMeta: metav1.ObjectMeta{
				Name:      "default",
				Namespace: "default",
			},
			Rules: []rbac.PolicyRule{{
				Verbs:     []string{"get"},
				Resources: []string{"pods"},
			}},
		},
		wantErr: true,
		errType: field.ErrorTypeRequired,
		field:   "rules[0].apiGroups",
	}.test(t)
}

func TestValidateRoleNoResources(t *testing.T) {
	ValidateRoleTest{
		role: rbac.Role{
			ObjectMeta: metav1.ObjectMeta{
				Name:      "default",
				Namespace: "default",
			},
			Rules: []rbac.PolicyRule{{
				Verbs:     []string{"get"},
				APIGroups: []string{"v1"},
			}},
		},
		wantErr: true,
		errType: field.ErrorTypeRequired,
		field:   "rules[0].resources",
	}.test(t)
}
