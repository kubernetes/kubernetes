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
	"strings"
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

func TestValidateProtectedAttribute(t *testing.T) {
	goodOnes := []*rbac.ProtectedAttribute{
		&rbac.ProtectedAttribute{
			ObjectMeta: api.ObjectMeta{
				Namespace: api.NamespaceDefault,
				Name:      "attr-1",
			},
			AttributeKind: rbac.LabelKind,
			AttributeName: "environment",
			RoleRef: api.ObjectReference{
				Kind:      "Role",
				Namespace: api.NamespaceDefault,
				Name:      "admin",
			},
			ProtectedValues: []string{"prod", "dev"},
		},
		&rbac.ProtectedAttribute{
			ObjectMeta: api.ObjectMeta{
				Namespace: api.NamespaceDefault,
				Name:      "attr-1",
			},
			AttributeKind: rbac.AnnotationKind,
			AttributeName: "k8s.io/annotation",
			RoleRef: api.ObjectReference{
				Kind: "ClusterRole",
				Name: "admin",
			},
		},
	}

	for _, v := range goodOnes {
		errs := ValidateProtectedAttribute(v)
		if len(errs) != 0 {
			t.Errorf("expected success: %v", errs)
		}
	}

	errorCases := map[string]struct {
		attr rbac.ProtectedAttribute
		msg  string
	}{
		"empty namespace": {
			rbac.ProtectedAttribute{
				ObjectMeta: api.ObjectMeta{Name: "default"},
			},
			"metadata.namespace: Required value",
		},
		"empty metadata.name": {
			rbac.ProtectedAttribute{
				ObjectMeta: api.ObjectMeta{Namespace: api.NamespaceDefault},
			},
			"metadata.name: Required value",
		},
		"unsupported kind": {
			rbac.ProtectedAttribute{
				ObjectMeta: api.ObjectMeta{
					Namespace: api.NamespaceDefault,
					Name:      "test",
				},
				AttributeKind: "foo1",
			},
			"attributeKind: Unsupported value",
		},
		"empty name": {
			rbac.ProtectedAttribute{
				ObjectMeta: api.ObjectMeta{
					Namespace: api.NamespaceDefault,
					Name:      "test",
				},
				AttributeKind: "Label",
			},
			"attributeName: Required value",
		},
		"roleRef namespace mismatch": {
			rbac.ProtectedAttribute{
				ObjectMeta: api.ObjectMeta{
					Namespace: api.NamespaceDefault,
					Name:      "test",
				},
				AttributeKind: "Label",
				AttributeName: "env",
				RoleRef: api.ObjectReference{
					Kind:      "Role",
					Namespace: "test",
					Name:      "admin",
				},
			},
			"role reference namespace mismatch",
		},
	}

	for k, v := range errorCases {
		if errs := ValidateProtectedAttribute(&v.attr); len(errs) == 0 {
			t.Errorf("expected failure on %s for %v", k, v.attr)
		} else if !strings.Contains(errs[0].Error(), v.msg) {
			t.Errorf("unexpected error on %s: %q, expected: %q", k, errs[0], v.msg)
		}
	}
}

func TestValidateProtectedAttributeUpdate(t *testing.T) {
	oldAttr := &rbac.ProtectedAttribute{
		ObjectMeta: api.ObjectMeta{
			Namespace:       api.NamespaceDefault,
			Name:            "attr-1",
			ResourceVersion: "17",
		},
		AttributeKind: rbac.LabelKind,
		AttributeName: "environment",
		RoleRef: api.ObjectReference{
			Kind:      "Role",
			Namespace: api.NamespaceDefault,
			Name:      "admin",
		},
		ProtectedValues: []string{"prod", "dev"},
	}

	newAttr := &rbac.ProtectedAttribute{
		ObjectMeta: api.ObjectMeta{
			Namespace:       api.NamespaceDefault,
			Name:            "attr-1",
			ResourceVersion: "17",
		},
		AttributeKind: rbac.AnnotationKind,
		AttributeName: "ks8.io/test",
		RoleRef: api.ObjectReference{
			Kind: "ClusterRole",
			Name: "operator",
		},
	}

	errs := ValidateProtectedAttributeUpdate(newAttr, oldAttr)
	if len(errs) != 0 {
		t.Errorf("expected success: %v", errs)
	}
}

func TestValidateClusterProtectedAttribute(t *testing.T) {
	goodOnes := []rbac.ClusterProtectedAttribute{
		rbac.ClusterProtectedAttribute{
			ObjectMeta:    api.ObjectMeta{Name: "attr-1"},
			AttributeKind: rbac.LabelKind,
			AttributeName: "environment",
			RoleRef: api.ObjectReference{
				Kind: "ClusterRole",
				Name: "admin",
			},
			ProtectedValues: []string{"prod", "dev"},
		},
		rbac.ClusterProtectedAttribute{
			ObjectMeta:    api.ObjectMeta{Name: "attr-2"},
			AttributeKind: rbac.AnnotationKind,
			AttributeName: "k8s.io/annotation",
			RoleRef: api.ObjectReference{
				Kind: "ClusterRole",
				Name: "admin",
			},
		},
	}

	for _, v := range goodOnes {
		errs := ValidateClusterProtectedAttribute(&v)
		if len(errs) != 0 {
			t.Errorf("expected success: %v", errs)
		}
	}

	errorCases := map[string]struct {
		attr rbac.ClusterProtectedAttribute
		msg  string
	}{
		"metadata.namespace is present": {
			rbac.ClusterProtectedAttribute{
				ObjectMeta: api.ObjectMeta{
					Namespace: "test",
					Name:      "foo1",
				},
			},
			"metadata.namespace: Forbidden",
		},
		"empty metadata.name": {
			rbac.ClusterProtectedAttribute{
				ObjectMeta: api.ObjectMeta{},
			},
			"metadata.name: Required value",
		},
		"unsupported kind": {
			rbac.ClusterProtectedAttribute{
				ObjectMeta:    api.ObjectMeta{Name: "test"},
				AttributeKind: "foo1",
			},
			"attributeKind: Unsupported value",
		},
		"empty name": {
			rbac.ClusterProtectedAttribute{
				ObjectMeta: api.ObjectMeta{
					Name: "test",
				},
				AttributeKind: "Label",
			},
			"attributeName: Required value",
		},
		"roleRef with namespace": {
			rbac.ClusterProtectedAttribute{
				ObjectMeta:    api.ObjectMeta{Name: "test"},
				AttributeKind: "Label",
				AttributeName: "env",
				RoleRef: api.ObjectReference{
					Kind:      "ClusterRole",
					Namespace: "test",
					Name:      "admin",
				},
			},
			"role reference namespace mismatch",
		},
	}

	for k, v := range errorCases {
		if errs := ValidateClusterProtectedAttribute(&v.attr); len(errs) == 0 {
			t.Errorf("expected failure on %s for %v", k, v.attr)
		} else if !strings.Contains(errs[0].Error(), v.msg) {
			t.Errorf("unexpected error on %s: %q, expected: %q", k, errs[0], v.msg)
		}
	}
}

func TestValidateClusterProtectedAttributeUpdate(t *testing.T) {
	oldAttr := &rbac.ClusterProtectedAttribute{
		ObjectMeta: api.ObjectMeta{
			Name:            "attr-1",
			ResourceVersion: "17",
		},
		AttributeKind: rbac.LabelKind,
		AttributeName: "environment",
		RoleRef: api.ObjectReference{
			Kind: "ClusterRole",
			Name: "admin",
		},
		ProtectedValues: []string{"prod", "dev"},
	}

	newAttr := &rbac.ClusterProtectedAttribute{
		ObjectMeta: api.ObjectMeta{
			Name:            "attr-1",
			ResourceVersion: "17",
		},
		AttributeKind: rbac.AnnotationKind,
		AttributeName: "ks8.io/test",
		RoleRef: api.ObjectReference{
			Kind: "ClusterRole",
			Name: "operator",
		},
	}

	errs := ValidateClusterProtectedAttributeUpdate(newAttr, oldAttr)
	if len(errs) != 0 {
		t.Errorf("expected success: %v", errs)
	}
}
