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

package set

import (
	"reflect"
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/kubernetes/pkg/apis/rbac"
	cmdtesting "k8s.io/kubernetes/pkg/kubectl/cmd/testing"
	"k8s.io/kubernetes/pkg/kubectl/resource"
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
				Users:           []string{},
				Groups:          []string{},
				ServiceAccounts: []string{},
			},
			expectErr: true,
		},
		"test-invalid-serviceaccounts": {
			options: &SubjectOptions{
				Users:           []string{},
				Groups:          []string{},
				ServiceAccounts: []string{"foo"},
			},
			expectErr: true,
		},
		"test-missing-serviceaccounts-name": {
			options: &SubjectOptions{
				Users:           []string{},
				Groups:          []string{},
				ServiceAccounts: []string{"foo:"},
			},
			expectErr: true,
		},
		"test-missing-serviceaccounts-namespace": {
			options: &SubjectOptions{
				Infos: []*resource.Info{
					{
						Object: &rbac.ClusterRoleBinding{
							ObjectMeta: metav1.ObjectMeta{
								Name: "clusterrolebinding",
							},
							RoleRef: rbac.RoleRef{
								APIGroup: "rbac.authorization.k8s.io",
								Kind:     "ClusterRole",
								Name:     "role",
							},
						},
					},
				},
				Users:           []string{},
				Groups:          []string{},
				ServiceAccounts: []string{":foo"},
			},
			expectErr: true,
		},
		"test-valid-case": {
			options: &SubjectOptions{
				Infos: []*resource.Info{
					{
						Object: &rbac.RoleBinding{
							ObjectMeta: metav1.ObjectMeta{
								Name:      "rolebinding",
								Namespace: "one",
							},
							RoleRef: rbac.RoleRef{
								APIGroup: "rbac.authorization.k8s.io",
								Kind:     "ClusterRole",
								Name:     "role",
							},
						},
					},
				},
				Users:           []string{"foo"},
				Groups:          []string{"foo"},
				ServiceAccounts: []string{"ns:foo"},
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

func TestUpdateSubjectForObject(t *testing.T) {
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
		if _, err := updateSubjectForObject(tt.obj, tt.subjects, addSubjects); (err != nil) != tt.wantErr {
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

func TestAddSubject(t *testing.T) {
	tests := []struct {
		Name       string
		existing   []rbac.Subject
		subjects   []rbac.Subject
		expected   []rbac.Subject
		wantChange bool
	}{
		{
			Name: "add resource with users",
			existing: []rbac.Subject{
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
			subjects: []rbac.Subject{
				{
					APIGroup: "rbac.authorization.k8s.io",
					Kind:     "User",
					Name:     "a",
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
			wantChange: false,
		},
		{
			Name: "add resource with serviceaccounts",
			existing: []rbac.Subject{
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
			subjects: []rbac.Subject{
				{
					Kind:      "ServiceAccount",
					Namespace: "two",
					Name:      "a",
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
				{
					Kind:      "ServiceAccount",
					Namespace: "two",
					Name:      "a",
				},
			},
			wantChange: true,
		},
	}
	for _, tt := range tests {
		changed := false
		got := []rbac.Subject{}
		if changed, got = addSubjects(tt.existing, tt.subjects); (changed != false) != tt.wantChange {
			t.Errorf("%q. addSubjects() changed = %v, wantChange = %v", tt.Name, changed, tt.wantChange)
		}

		want := tt.expected
		if !reflect.DeepEqual(got, want) {
			t.Errorf("%q. addSubjects() failed", tt.Name)
			t.Errorf("Got: %v", got)
			t.Errorf("Want: %v", want)
		}
	}
}
