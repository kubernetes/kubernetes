/*
Copyright 2017 The Kubernetes Authors.

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

package reconciliation

import (
	"testing"

	rbacv1 "k8s.io/api/rbac/v1"
	apiequality "k8s.io/apimachinery/pkg/api/equality"
)

func binding(roleRef rbacv1.RoleRef, subjects []rbacv1.Subject) *rbacv1.ClusterRoleBinding {
	return &rbacv1.ClusterRoleBinding{RoleRef: roleRef, Subjects: subjects}
}

func ref(name string) rbacv1.RoleRef {
	return rbacv1.RoleRef{Name: name}
}

func subject(name string) rbacv1.Subject {
	return rbacv1.Subject{Name: name}
}

func subjects(names ...string) []rbacv1.Subject {
	r := []rbacv1.Subject{}
	for _, name := range names {
		r = append(r, subject(name))
	}
	return r
}

func TestDiffObjectReferenceLists(t *testing.T) {
	tests := map[string]struct {
		A             []rbacv1.Subject
		B             []rbacv1.Subject
		ExpectedOnlyA []rbacv1.Subject
		ExpectedOnlyB []rbacv1.Subject
	}{
		"empty": {},

		"matching, order-independent": {
			A: subjects("foo", "bar"),
			B: subjects("bar", "foo"),
		},

		"partial match": {
			A:             subjects("foo", "bar"),
			B:             subjects("foo", "baz"),
			ExpectedOnlyA: subjects("bar"),
			ExpectedOnlyB: subjects("baz"),
		},

		"missing": {
			A:             subjects("foo"),
			B:             subjects("bar"),
			ExpectedOnlyA: subjects("foo"),
			ExpectedOnlyB: subjects("bar"),
		},

		"remove duplicates": {
			A:             subjects("foo", "foo"),
			B:             subjects("bar", "bar"),
			ExpectedOnlyA: subjects("foo"),
			ExpectedOnlyB: subjects("bar"),
		},
	}

	for k, tc := range tests {
		onlyA, onlyB := diffSubjectLists(tc.A, tc.B)
		if !apiequality.Semantic.DeepEqual(onlyA, tc.ExpectedOnlyA) {
			t.Errorf("%s: Expected %#v, got %#v", k, tc.ExpectedOnlyA, onlyA)
		}
		if !apiequality.Semantic.DeepEqual(onlyB, tc.ExpectedOnlyB) {
			t.Errorf("%s: Expected %#v, got %#v", k, tc.ExpectedOnlyB, onlyB)
		}
	}
}

func TestComputeUpdate(t *testing.T) {
	tests := map[string]struct {
		ExpectedBinding     *rbacv1.ClusterRoleBinding
		ActualBinding       *rbacv1.ClusterRoleBinding
		RemoveExtraSubjects bool

		ExpectedUpdatedBinding *rbacv1.ClusterRoleBinding
		ExpectedUpdateNeeded   bool
	}{
		"match without union": {
			ExpectedBinding:     binding(ref("role"), subjects("a")),
			ActualBinding:       binding(ref("role"), subjects("a")),
			RemoveExtraSubjects: true,

			ExpectedUpdatedBinding: nil,
			ExpectedUpdateNeeded:   false,
		},
		"match with union": {
			ExpectedBinding:     binding(ref("role"), subjects("a")),
			ActualBinding:       binding(ref("role"), subjects("a")),
			RemoveExtraSubjects: false,

			ExpectedUpdatedBinding: nil,
			ExpectedUpdateNeeded:   false,
		},

		"different roleref with identical subjects": {
			ExpectedBinding:     binding(ref("role"), subjects("a")),
			ActualBinding:       binding(ref("differentRole"), subjects("a")),
			RemoveExtraSubjects: false,

			ExpectedUpdatedBinding: binding(ref("role"), subjects("a")),
			ExpectedUpdateNeeded:   true,
		},

		"extra subjects without union": {
			ExpectedBinding:     binding(ref("role"), subjects("a")),
			ActualBinding:       binding(ref("role"), subjects("a", "b")),
			RemoveExtraSubjects: true,

			ExpectedUpdatedBinding: binding(ref("role"), subjects("a")),
			ExpectedUpdateNeeded:   true,
		},
		"extra subjects with union": {
			ExpectedBinding:     binding(ref("role"), subjects("a")),
			ActualBinding:       binding(ref("role"), subjects("a", "b")),
			RemoveExtraSubjects: false,

			ExpectedUpdatedBinding: nil,
			ExpectedUpdateNeeded:   false,
		},

		"missing subjects without union": {
			ExpectedBinding:     binding(ref("role"), subjects("a", "c")),
			ActualBinding:       binding(ref("role"), subjects("a", "b")),
			RemoveExtraSubjects: true,

			ExpectedUpdatedBinding: binding(ref("role"), subjects("a", "c")),
			ExpectedUpdateNeeded:   true,
		},
		"missing subjects with union": {
			ExpectedBinding:     binding(ref("role"), subjects("a", "c")),
			ActualBinding:       binding(ref("role"), subjects("a", "b")),
			RemoveExtraSubjects: false,

			ExpectedUpdatedBinding: binding(ref("role"), subjects("a", "b", "c")),
			ExpectedUpdateNeeded:   true,
		},
	}

	for k, tc := range tests {
		actualRoleBinding := ClusterRoleBindingAdapter{ClusterRoleBinding: tc.ActualBinding}
		expectedRoleBinding := ClusterRoleBindingAdapter{ClusterRoleBinding: tc.ExpectedBinding}
		result, err := computeReconciledRoleBinding(actualRoleBinding, expectedRoleBinding, tc.RemoveExtraSubjects)
		if err != nil {
			t.Errorf("%s: %v", k, err)
			continue
		}
		updateNeeded := result.Operation != ReconcileNone
		updatedBinding := result.RoleBinding.(ClusterRoleBindingAdapter).ClusterRoleBinding
		if updateNeeded != tc.ExpectedUpdateNeeded {
			t.Errorf("%s: Expected\n\t%v\ngot\n\t%v (%v)", k, tc.ExpectedUpdateNeeded, updateNeeded, result.Operation)
			continue
		}
		if updateNeeded && !apiequality.Semantic.DeepEqual(updatedBinding, tc.ExpectedUpdatedBinding) {
			t.Errorf("%s: Expected\n\t%v %v\ngot\n\t%v %v", k, tc.ExpectedUpdatedBinding.RoleRef, tc.ExpectedUpdatedBinding.Subjects, updatedBinding.RoleRef, updatedBinding.Subjects)
		}
	}
}
