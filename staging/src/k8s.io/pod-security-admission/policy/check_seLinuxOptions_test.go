/*
Copyright 2021 The Kubernetes Authors.

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

package policy

import (
	"testing"

	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/util/validation/field"

	"github.com/google/go-cmp/cmp"
	"github.com/google/go-cmp/cmp/cmpopts"
)

func TestSELinuxOptions(t *testing.T) {
	tests := []struct {
		name          string
		pod           *corev1.Pod
		opts          options
		expectReason  string
		expectDetail  string
		expectErrList field.ErrorList
	}{
		{
			name: "invalid pod and containers",
			pod: &corev1.Pod{Spec: corev1.PodSpec{
				SecurityContext: &corev1.PodSecurityContext{
					SELinuxOptions: &corev1.SELinuxOptions{
						Type: "foo",
						User: "bar",
						Role: "baz",
					},
				},
				Containers: []corev1.Container{
					{Name: "a", SecurityContext: &corev1.SecurityContext{SELinuxOptions: &corev1.SELinuxOptions{
						Type: "container_t",
					}}},
					{Name: "b", SecurityContext: &corev1.SecurityContext{SELinuxOptions: &corev1.SELinuxOptions{
						Type: "container_init_t",
					}}},
					{Name: "c", SecurityContext: &corev1.SecurityContext{SELinuxOptions: &corev1.SELinuxOptions{
						Type: "container_kvm_t",
					}}},
					{Name: "d", SecurityContext: &corev1.SecurityContext{SELinuxOptions: &corev1.SELinuxOptions{
						Type: "bar",
					}}},
					{Name: "e", SecurityContext: &corev1.SecurityContext{SELinuxOptions: &corev1.SELinuxOptions{
						User: "bar",
					}}},
					{Name: "f", SecurityContext: &corev1.SecurityContext{SELinuxOptions: &corev1.SELinuxOptions{
						Role: "baz",
					}}},
				},
			}},
			expectReason: `seLinuxOptions`,
			expectDetail: `pod and containers "d", "e", "f" set forbidden securityContext.seLinuxOptions: types "bar", "foo"; user may not be set; role may not be set`,
		},
		{
			name: "invalid pod and containers, enable field error list",
			pod: &corev1.Pod{Spec: corev1.PodSpec{
				SecurityContext: &corev1.PodSecurityContext{
					SELinuxOptions: &corev1.SELinuxOptions{
						Type: "foo",
						User: "bar",
						Role: "baz",
					},
				},
				Containers: []corev1.Container{
					{Name: "a", SecurityContext: &corev1.SecurityContext{SELinuxOptions: &corev1.SELinuxOptions{
						Type: "container_t",
					}}},
					{Name: "b", SecurityContext: &corev1.SecurityContext{SELinuxOptions: &corev1.SELinuxOptions{
						Type: "container_init_t",
					}}},
					{Name: "c", SecurityContext: &corev1.SecurityContext{SELinuxOptions: &corev1.SELinuxOptions{
						Type: "container_kvm_t",
					}}},
					{Name: "d", SecurityContext: &corev1.SecurityContext{SELinuxOptions: &corev1.SELinuxOptions{
						Type: "bar",
					}}},
					{Name: "e", SecurityContext: &corev1.SecurityContext{SELinuxOptions: &corev1.SELinuxOptions{
						User: "bar",
					}}},
					{Name: "f", SecurityContext: &corev1.SecurityContext{SELinuxOptions: &corev1.SELinuxOptions{
						Role: "baz",
					}}},
				},
			}},
			opts: options{
				withFieldErrors: true,
			},
			expectReason: `seLinuxOptions`,
			expectDetail: `pod and containers "d", "e", "f" set forbidden securityContext.seLinuxOptions: types "bar", "foo"; user may not be set; role may not be set`,
			expectErrList: field.ErrorList{
				{Type: field.ErrorTypeForbidden, Field: "spec.securityContext.seLinuxOptions.type", BadValue: "foo"},
				{Type: field.ErrorTypeForbidden, Field: "spec.securityContext.seLinuxOptions.user", BadValue: "bar"},
				{Type: field.ErrorTypeForbidden, Field: "spec.securityContext.seLinuxOptions.role", BadValue: "baz"},
				{Type: field.ErrorTypeForbidden, Field: "spec.containers[3].securityContext.seLinuxOptions.type", BadValue: "bar"},
				{Type: field.ErrorTypeForbidden, Field: "spec.containers[4].securityContext.seLinuxOptions.user", BadValue: "bar"},
				{Type: field.ErrorTypeForbidden, Field: "spec.containers[5].securityContext.seLinuxOptions.role", BadValue: "baz"},
			},
		},
		{
			name: "invalid pod",
			pod: &corev1.Pod{Spec: corev1.PodSpec{
				SecurityContext: &corev1.PodSecurityContext{
					SELinuxOptions: &corev1.SELinuxOptions{
						Type: "foo",
						User: "bar",
						Role: "baz",
					},
				},
				Containers: []corev1.Container{
					{Name: "a", SecurityContext: &corev1.SecurityContext{SELinuxOptions: &corev1.SELinuxOptions{
						Type: "container_t",
					}}},
					{Name: "b", SecurityContext: &corev1.SecurityContext{SELinuxOptions: &corev1.SELinuxOptions{
						Type: "container_init_t",
					}}},
					{Name: "c", SecurityContext: &corev1.SecurityContext{SELinuxOptions: &corev1.SELinuxOptions{
						Type: "container_kvm_t",
					}}},
				},
			}},
			expectReason: `seLinuxOptions`,
			expectDetail: `pod set forbidden securityContext.seLinuxOptions: type "foo"; user may not be set; role may not be set`,
		},
		{
			name: "invalid pod, enable field error list",
			pod: &corev1.Pod{Spec: corev1.PodSpec{
				SecurityContext: &corev1.PodSecurityContext{
					SELinuxOptions: &corev1.SELinuxOptions{
						Type: "foo",
						User: "bar",
						Role: "baz",
					},
				},
				Containers: []corev1.Container{
					{Name: "a", SecurityContext: &corev1.SecurityContext{SELinuxOptions: &corev1.SELinuxOptions{
						Type: "container_t",
					}}},
					{Name: "b", SecurityContext: &corev1.SecurityContext{SELinuxOptions: &corev1.SELinuxOptions{
						Type: "container_init_t",
					}}},
					{Name: "c", SecurityContext: &corev1.SecurityContext{SELinuxOptions: &corev1.SELinuxOptions{
						Type: "container_kvm_t",
					}}},
				},
			}},
			opts: options{
				withFieldErrors: true,
			},
			expectReason: `seLinuxOptions`,
			expectDetail: `pod set forbidden securityContext.seLinuxOptions: type "foo"; user may not be set; role may not be set`,
			expectErrList: field.ErrorList{
				{Type: field.ErrorTypeForbidden, Field: "spec.securityContext.seLinuxOptions.type", BadValue: "foo"},
				{Type: field.ErrorTypeForbidden, Field: "spec.securityContext.seLinuxOptions.user", BadValue: "bar"},
				{Type: field.ErrorTypeForbidden, Field: "spec.securityContext.seLinuxOptions.role", BadValue: "baz"},
			},
		},
		{
			name: "invalid containers",
			pod: &corev1.Pod{Spec: corev1.PodSpec{
				SecurityContext: &corev1.PodSecurityContext{
					SELinuxOptions: &corev1.SELinuxOptions{},
				},
				Containers: []corev1.Container{
					{Name: "a", SecurityContext: &corev1.SecurityContext{SELinuxOptions: &corev1.SELinuxOptions{
						Type: "container_t",
					}}},
					{Name: "b", SecurityContext: &corev1.SecurityContext{SELinuxOptions: &corev1.SELinuxOptions{
						Type: "container_init_t",
					}}},
					{Name: "c", SecurityContext: &corev1.SecurityContext{SELinuxOptions: &corev1.SELinuxOptions{
						Type: "container_kvm_t",
					}}},
					{Name: "d", SecurityContext: &corev1.SecurityContext{SELinuxOptions: &corev1.SELinuxOptions{
						Type: "bar",
					}}},
					{Name: "e", SecurityContext: &corev1.SecurityContext{SELinuxOptions: &corev1.SELinuxOptions{
						User: "bar",
					}}},
					{Name: "f", SecurityContext: &corev1.SecurityContext{SELinuxOptions: &corev1.SELinuxOptions{
						Role: "baz",
					}}},
				},
			}},
			expectReason: `seLinuxOptions`,
			expectDetail: `containers "d", "e", "f" set forbidden securityContext.seLinuxOptions: type "bar"; user may not be set; role may not be set`,
		},
		{
			name: "invalid containers, enable field error list",
			pod: &corev1.Pod{Spec: corev1.PodSpec{
				SecurityContext: &corev1.PodSecurityContext{
					SELinuxOptions: &corev1.SELinuxOptions{},
				},
				Containers: []corev1.Container{
					{Name: "a", SecurityContext: &corev1.SecurityContext{SELinuxOptions: &corev1.SELinuxOptions{
						Type: "container_t",
					}}},
					{Name: "b", SecurityContext: &corev1.SecurityContext{SELinuxOptions: &corev1.SELinuxOptions{
						Type: "container_init_t",
					}}},
					{Name: "c", SecurityContext: &corev1.SecurityContext{SELinuxOptions: &corev1.SELinuxOptions{
						Type: "container_kvm_t",
					}}},
					{Name: "d", SecurityContext: &corev1.SecurityContext{SELinuxOptions: &corev1.SELinuxOptions{
						Type: "bar",
					}}},
					{Name: "e", SecurityContext: &corev1.SecurityContext{SELinuxOptions: &corev1.SELinuxOptions{
						User: "bar",
					}}},
					{Name: "f", SecurityContext: &corev1.SecurityContext{SELinuxOptions: &corev1.SELinuxOptions{
						Role: "baz",
					}}},
				},
			}},
			opts: options{
				withFieldErrors: true,
			},
			expectReason: `seLinuxOptions`,
			expectDetail: `containers "d", "e", "f" set forbidden securityContext.seLinuxOptions: type "bar"; user may not be set; role may not be set`,
			expectErrList: field.ErrorList{
				{Type: field.ErrorTypeForbidden, Field: "spec.containers[3].securityContext.seLinuxOptions.type", BadValue: "bar"},
				{Type: field.ErrorTypeForbidden, Field: "spec.containers[4].securityContext.seLinuxOptions.user", BadValue: "bar"},
				{Type: field.ErrorTypeForbidden, Field: "spec.containers[5].securityContext.seLinuxOptions.role", BadValue: "baz"},
			},
		},
		{
			name: "bad type",
			pod: &corev1.Pod{Spec: corev1.PodSpec{
				SecurityContext: &corev1.PodSecurityContext{
					SELinuxOptions: &corev1.SELinuxOptions{
						Type: "bad",
					},
				},
			}},
			expectReason: `seLinuxOptions`,
			expectDetail: `pod set forbidden securityContext.seLinuxOptions: type "bad"`,
		},
		{
			name: "bad type, enable field error list",
			pod: &corev1.Pod{Spec: corev1.PodSpec{
				SecurityContext: &corev1.PodSecurityContext{
					SELinuxOptions: &corev1.SELinuxOptions{
						Type: "bad",
					},
				},
			}},
			opts: options{
				withFieldErrors: true,
			},
			expectReason: `seLinuxOptions`,
			expectDetail: `pod set forbidden securityContext.seLinuxOptions: type "bad"`,
			expectErrList: field.ErrorList{
				{Type: field.ErrorTypeForbidden, Field: "spec.securityContext.seLinuxOptions.type", BadValue: "bad"},
			},
		},
		{
			name: "bad user",
			pod: &corev1.Pod{Spec: corev1.PodSpec{
				SecurityContext: &corev1.PodSecurityContext{
					SELinuxOptions: &corev1.SELinuxOptions{
						User: "bad",
					},
				},
			}},
			expectReason: `seLinuxOptions`,
			expectDetail: `pod set forbidden securityContext.seLinuxOptions: user may not be set`,
		},
		{
			name: "bad user, enable field error list",
			pod: &corev1.Pod{Spec: corev1.PodSpec{
				SecurityContext: &corev1.PodSecurityContext{
					SELinuxOptions: &corev1.SELinuxOptions{
						User: "bad",
					},
				},
			}},
			opts: options{
				withFieldErrors: true,
			},
			expectReason: `seLinuxOptions`,
			expectDetail: `pod set forbidden securityContext.seLinuxOptions: user may not be set`,
			expectErrList: field.ErrorList{
				{Type: field.ErrorTypeForbidden, Field: "spec.securityContext.seLinuxOptions.user", BadValue: "bad"},
			},
		},
		{
			name: "bad role",
			pod: &corev1.Pod{Spec: corev1.PodSpec{
				SecurityContext: &corev1.PodSecurityContext{
					SELinuxOptions: &corev1.SELinuxOptions{
						Role: "bad",
					},
				},
			}},
			expectReason: `seLinuxOptions`,
			expectDetail: `pod set forbidden securityContext.seLinuxOptions: role may not be set`,
		},
		{
			name: "bad role, enable field error list",
			pod: &corev1.Pod{Spec: corev1.PodSpec{
				SecurityContext: &corev1.PodSecurityContext{
					SELinuxOptions: &corev1.SELinuxOptions{
						Role: "bad",
					},
				},
			}},
			opts: options{
				withFieldErrors: true,
			},
			expectReason: `seLinuxOptions`,
			expectDetail: `pod set forbidden securityContext.seLinuxOptions: role may not be set`,
			expectErrList: field.ErrorList{
				{Type: field.ErrorTypeForbidden, Field: "spec.securityContext.seLinuxOptions.role", BadValue: "bad"},
			},
		},
	}

	cmpOpts := []cmp.Option{cmpopts.IgnoreFields(field.Error{}, "Detail"), cmpopts.SortSlices(func(a, b *field.Error) bool { return a.Error() < b.Error() })}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			result := seLinuxOptionsV1Dot0(&tc.pod.ObjectMeta, &tc.pod.Spec, tc.opts)
			if result.Allowed {
				t.Fatal("expected disallowed")
			}
			if e, a := tc.expectReason, result.ForbiddenReason; e != a {
				t.Errorf("expected\n%s\ngot\n%s", e, a)
			}
			if e, a := tc.expectDetail, result.ForbiddenDetail; e != a {
				t.Errorf("expected\n%s\ngot\n%s", e, a)
			}
			if result.ErrList != nil {
				if diff := cmp.Diff(tc.expectErrList, *result.ErrList, cmpOpts...); diff != "" {
					t.Errorf("unexpected field errors (-want,+got):\n%s", diff)
				}
			}
		})
	}
}
