/*
Copyright 2017 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY Kind, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package set

import (
	"reflect"
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/kubernetes/pkg/apis/rbac"
	cmdtesting "k8s.io/kubernetes/pkg/kubectl/cmd/testing"
)

func TestValidate(t *testing.T) {
	f, tf, _, _ := cmdtesting.NewAPIFactory()
	tf.Namespace = "test"

	tests := map[string]struct {
		options   *SubjectOptions
		expectErr bool
	}{
		"test-missing-subjects": {
			options: &SubjectOptions{
				users:           []string{},
				groups:          []string{},
				serviceaccounts: []string{},
			},
			expectErr: true,
		},
		"test-invalid-serviceaccounts": {
			options: &SubjectOptions{
				users:           []string{},
				groups:          []string{},
				serviceaccounts: []string{"foo"},
			},
			expectErr: true,
		},
		"test-valid-case": {
			options: &SubjectOptions{
				users:           []string{"foo"},
				groups:          []string{"foo"},
				serviceaccounts: []string{"ns:foo"},
			},
			expectErr: false,
		},
	}

	for name, test := range tests {
		test.options.Mapper, _ = f.Object()
		err := test.options.Validate()
		if test.expectErr && err != nil {
			continue
		}
		if !test.expectErr && err != nil {
			t.Errorf("%s: unexpected error: %v", name, err)
		}
	}
}

func TestAddSubjectForObject(t *testing.T) {
	tests := []struct {
		Name     string
		obj      runtime.Object
		subjects []rbac.Subject
		expected []rbac.Subject
		wantErr  bool
	}{
		{
			Name: "invalid object type",
			obj: &rbac.Role{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "role",
					Namespace: "one",
				},
			},
			wantErr: true,
		},
		{
			Name: "add resource with users in rolebinding",
			obj: &rbac.RoleBinding{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "rolebinding",
					Namespace: "one",
				},
				Subjects: []rbac.Subject{
					{
						APIGroup: "rbac.authorization.k8s.io",
						Kind:     "User",
						Name:     "a",
					},
				},
			},
			subjects: []rbac.Subject{
				{
					APIGroup: "rbac.authorization.k8s.io",
					Kind:     "User",
					Name:     "a",
				},
				{
					APIGroup: "rbac.authorization.k8s.io",
					Kind:     "User",
					Name:     "b",
				},
			},
			expected: []rbac.Subject{
				{
					APIGroup: "rbac.authorization.k8s.io",
					Kind:     "User",
					Name:     "a",
				},
				{
					APIGroup: "rbac.authorization.k8s.io",
					Kind:     "User",
					Name:     "b",
				},
			},
			wantErr: false,
		},
		{
			Name: "add resource with groups in rolebinding",
			obj: &rbac.RoleBinding{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "rolebinding",
					Namespace: "one",
				},
				Subjects: []rbac.Subject{
					{
						APIGroup: "rbac.authorization.k8s.io",
						Kind:     "Group",
						Name:     "a",
					},
				},
			},
			subjects: []rbac.Subject{
				{
					APIGroup: "rbac.authorization.k8s.io",
					Kind:     "Group",
					Name:     "a",
				},
				{
					APIGroup: "rbac.authorization.k8s.io",
					Kind:     "Group",
					Name:     "b",
				},
			},
			expected: []rbac.Subject{
				{
					APIGroup: "rbac.authorization.k8s.io",
					Kind:     "Group",
					Name:     "a",
				},
				{
					APIGroup: "rbac.authorization.k8s.io",
					Kind:     "Group",
					Name:     "b",
				},
			},
			wantErr: false,
		},
		{
			Name: "add resource with serviceaccounts in rolebinding",
			obj: &rbac.RoleBinding{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "rolebinding",
					Namespace: "one",
				},
				Subjects: []rbac.Subject{
					{
						Kind:      "ServiceAccount",
						Namespace: "one",
						Name:      "a",
					},
				},
			},
			subjects: []rbac.Subject{
				{
					Kind:      "ServiceAccount",
					Namespace: "one",
					Name:      "a",
				},
				{
					Kind:      "ServiceAccount",
					Namespace: "one",
					Name:      "b",
				},
			},
			expected: []rbac.Subject{
				{
					Kind:      "ServiceAccount",
					Namespace: "one",
					Name:      "a",
				},
				{
					Kind:      "ServiceAccount",
					Namespace: "one",
					Name:      "b",
				},
			},
			wantErr: false,
		},
		{
			Name: "add resource with serviceaccounts in clusterrolebinding",
			obj: &rbac.ClusterRoleBinding{
				ObjectMeta: metav1.ObjectMeta{
					Name: "clusterrolebinding",
				},
				Subjects: []rbac.Subject{
					{
						APIGroup: "rbac.authorization.k8s.io",
						Kind:     "User",
						Name:     "a",
					},
					{
						APIGroup: "rbac.authorization.k8s.io",
						Kind:     "Group",
						Name:     "a",
					},
				},
			},
			subjects: []rbac.Subject{
				{
					Kind:      "ServiceAccount",
					Namespace: "one",
					Name:      "a",
				},
			},
			expected: []rbac.Subject{
				{
					APIGroup: "rbac.authorization.k8s.io",
					Kind:     "User",
					Name:     "a",
				},
				{
					APIGroup: "rbac.authorization.k8s.io",
					Kind:     "Group",
					Name:     "a",
				},
				{
					Kind:      "ServiceAccount",
					Namespace: "one",
					Name:      "a",
				},
			},
			wantErr: false,
		},
	}
	for _, tt := range tests {
		if _, err := updateSubjectForObject(tt.obj, tt.subjects, false); (err != nil) != tt.wantErr {
			t.Errorf("%q. updateSubjectForObject() error = %v, wantErr %v", tt.Name, err, tt.wantErr)
		}

		want := tt.expected
		var got []rbac.Subject
		switch t := tt.obj.(type) {
		case *rbac.RoleBinding:
			got = t.Subjects
		case *rbac.ClusterRoleBinding:
			got = t.Subjects
		}
		if !reflect.DeepEqual(got, want) {
			t.Errorf("%q. updateSubjectForObject() failed", tt.Name)
			t.Errorf("Got: %v", got)
			t.Errorf("Want: %v", want)
		}
	}
}

func TestRemoveSubjectForObject(t *testing.T) {
	tests := []struct {
		Name     string
		obj      runtime.Object
		subjects []rbac.Subject
		expected []rbac.Subject
		wantErr  bool
	}{
		{
			Name: "delete resource with users in rolebinding",
			obj: &rbac.RoleBinding{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "rolebinding",
					Namespace: "one",
				},
				Subjects: []rbac.Subject{
					{
						APIGroup: "rbac.authorization.k8s.io",
						Kind:     "User",
						Name:     "a",
					},
					{
						APIGroup: "rbac.authorization.k8s.io",
						Kind:     "User",
						Name:     "b",
					},
				},
			},
			subjects: []rbac.Subject{
				{
					APIGroup: "rbac.authorization.k8s.io",
					Kind:     "User",
					Name:     "a",
				},
				{
					APIGroup: "rbac.authorization.k8s.io",
					Kind:     "User",
					Name:     "c",
				},
			},
			expected: []rbac.Subject{
				{
					APIGroup: "rbac.authorization.k8s.io",
					Kind:     "User",
					Name:     "b",
				},
			},
			wantErr: false,
		},
		{
			Name: "delete resource with groups in rolebinding",
			obj: &rbac.RoleBinding{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "rolebinding",
					Namespace: "one",
				},
				Subjects: []rbac.Subject{
					{
						APIGroup: "rbac.authorization.k8s.io",
						Kind:     "Group",
						Name:     "a",
					},
					{
						APIGroup: "rbac.authorization.k8s.io",
						Kind:     "Group",
						Name:     "b",
					},
				},
			},
			subjects: []rbac.Subject{
				{
					APIGroup: "rbac.authorization.k8s.io",
					Kind:     "Group",
					Name:     "a",
				},
				{
					APIGroup: "rbac.authorization.k8s.io",
					Kind:     "Group",
					Name:     "c",
				},
			},
			expected: []rbac.Subject{
				{
					APIGroup: "rbac.authorization.k8s.io",
					Kind:     "Group",
					Name:     "b",
				},
			},
			wantErr: false,
		},
		{
			Name: "delete resource with serviceaccounts in rolebinding",
			obj: &rbac.RoleBinding{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "rolebinding",
					Namespace: "one",
				},
				Subjects: []rbac.Subject{
					{
						Kind:      "ServiceAccount",
						Namespace: "one",
						Name:      "a",
					},
					{
						Kind:      "ServiceAccount",
						Namespace: "one",
						Name:      "b",
					},
				},
			},
			subjects: []rbac.Subject{
				{
					Kind:      "ServiceAccount",
					Namespace: "one",
					Name:      "a",
				},
				{
					Kind:      "ServiceAccount",
					Namespace: "one",
					Name:      "c",
				},
			},
			expected: []rbac.Subject{
				{
					Kind:      "ServiceAccount",
					Namespace: "one",
					Name:      "b",
				},
			},
			wantErr: false,
		},
		{
			Name: "delete resource with serviceaccounts in clusterrolebinding",
			obj: &rbac.ClusterRoleBinding{
				ObjectMeta: metav1.ObjectMeta{
					Name: "clusterrolebinding",
				},
				Subjects: []rbac.Subject{
					{
						APIGroup: "rbac.authorization.k8s.io",
						Kind:     "User",
						Name:     "a",
					},
					{
						APIGroup: "rbac.authorization.k8s.io",
						Kind:     "Group",
						Name:     "a",
					},
					{
						Kind:      "ServiceAccount",
						Namespace: "one",
						Name:      "a",
					},
				},
			},
			subjects: []rbac.Subject{
				{
					Kind:      "ServiceAccount",
					Namespace: "one",
					Name:      "a",
				},
			},
			expected: []rbac.Subject{
				{
					APIGroup: "rbac.authorization.k8s.io",
					Kind:     "User",
					Name:     "a",
				},
				{
					APIGroup: "rbac.authorization.k8s.io",
					Kind:     "Group",
					Name:     "a",
				},
			},
			wantErr: false,
		},
	}
	for _, tt := range tests {
		if _, err := updateSubjectForObject(tt.obj, tt.subjects, true); (err != nil) != tt.wantErr {
			t.Errorf("%q. updateSubjectForObject() error = %v, wantErr %v", tt.Name, err, tt.wantErr)
		}

		want := tt.expected
		var got []rbac.Subject
		switch t := tt.obj.(type) {
		case *rbac.RoleBinding:
			got = t.Subjects
		case *rbac.ClusterRoleBinding:
			got = t.Subjects
		}
		if !reflect.DeepEqual(got, want) {
			t.Errorf("%q. updateSubjectForObject() failed", tt.Name)
			t.Errorf("Got: %v", got)
			t.Errorf("Want: %v", want)
		}
	}
}
