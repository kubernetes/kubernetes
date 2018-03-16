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

package securitycontext

import (
	"reflect"
	"testing"

	"k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

func TestParseSELinuxOptions(t *testing.T) {
	cases := []struct {
		name     string
		input    string
		expected *v1.SELinuxOptions
	}{
		{
			name:  "simple",
			input: "user_t:role_t:type_t:s0",
			expected: &v1.SELinuxOptions{
				User:  "user_t",
				Role:  "role_t",
				Type:  "type_t",
				Level: "s0",
			},
		},
		{
			name:  "simple + categories",
			input: "user_t:role_t:type_t:s0:c0",
			expected: &v1.SELinuxOptions{
				User:  "user_t",
				Role:  "role_t",
				Type:  "type_t",
				Level: "s0:c0",
			},
		},
		{
			name:  "not enough fields",
			input: "type_t:s0:c0",
		},
	}

	for _, tc := range cases {
		result, err := ParseSELinuxOptions(tc.input)

		if err != nil {
			if tc.expected == nil {
				continue
			} else {
				t.Errorf("%v: unexpected error: %v", tc.name, err)
			}
		}

		compareContexts(tc.name, tc.expected, result, t)
	}
}

func compareContexts(name string, ex, ac *v1.SELinuxOptions, t *testing.T) {
	if e, a := ex.User, ac.User; e != a {
		t.Errorf("%v: expected user: %v, got: %v", name, e, a)
	}
	if e, a := ex.Role, ac.Role; e != a {
		t.Errorf("%v: expected role: %v, got: %v", name, e, a)
	}
	if e, a := ex.Type, ac.Type; e != a {
		t.Errorf("%v: expected type: %v, got: %v", name, e, a)
	}
	if e, a := ex.Level, ac.Level; e != a {
		t.Errorf("%v: expected level: %v, got: %v", name, e, a)
	}
}

func containerWithUser(ptr *int64) *v1.Container {
	return &v1.Container{SecurityContext: &v1.SecurityContext{RunAsUser: ptr}}
}

func TestHaRootUID(t *testing.T) {
	nonRoot := int64(1)
	root := int64(0)

	tests := map[string]struct {
		container *v1.Container
		expect    bool
	}{
		"nil sc": {
			container: &v1.Container{SecurityContext: nil},
		},
		"nil runAsuser": {
			container: containerWithUser(nil),
		},
		"runAsUser non-root": {
			container: containerWithUser(&nonRoot),
		},
		"runAsUser root": {
			container: containerWithUser(&root),
			expect:    true,
		},
	}

	for k, v := range tests {
		actual := HasRootUID(v.container)
		if actual != v.expect {
			t.Errorf("%s failed, expected %t but received %t", k, v.expect, actual)
		}
	}
}

func TestHasRunAsUser(t *testing.T) {
	runAsUser := int64(0)

	tests := map[string]struct {
		container *v1.Container
		expect    bool
	}{
		"nil sc": {
			container: &v1.Container{SecurityContext: nil},
		},
		"nil runAsUser": {
			container: containerWithUser(nil),
		},
		"valid runAsUser": {
			container: containerWithUser(&runAsUser),
			expect:    true,
		},
	}

	for k, v := range tests {
		actual := HasRunAsUser(v.container)
		if actual != v.expect {
			t.Errorf("%s failed, expected %t but received %t", k, v.expect, actual)
		}
	}
}

func TestHasRootRunAsUser(t *testing.T) {
	nonRoot := int64(1)
	root := int64(0)

	tests := map[string]struct {
		container *v1.Container
		expect    bool
	}{
		"nil sc": {
			container: &v1.Container{SecurityContext: nil},
		},
		"nil runAsuser": {
			container: containerWithUser(nil),
		},
		"runAsUser non-root": {
			container: containerWithUser(&nonRoot),
		},
		"runAsUser root": {
			container: containerWithUser(&root),
			expect:    true,
		},
	}

	for k, v := range tests {
		actual := HasRootRunAsUser(v.container)
		if actual != v.expect {
			t.Errorf("%s failed, expected %t but received %t", k, v.expect, actual)
		}
	}
}

func TestAddNoNewPrivileges(t *testing.T) {
	pfalse := false
	ptrue := true

	tests := map[string]struct {
		sc     *v1.SecurityContext
		expect bool
	}{
		"allowPrivilegeEscalation nil security context nil": {
			sc:     nil,
			expect: false,
		},
		"allowPrivilegeEscalation nil": {
			sc: &v1.SecurityContext{
				AllowPrivilegeEscalation: nil,
			},
			expect: false,
		},
		"allowPrivilegeEscalation false": {
			sc: &v1.SecurityContext{
				AllowPrivilegeEscalation: &pfalse,
			},
			expect: true,
		},
		"allowPrivilegeEscalation true": {
			sc: &v1.SecurityContext{
				AllowPrivilegeEscalation: &ptrue,
			},
			expect: false,
		},
	}

	for k, v := range tests {
		actual := AddNoNewPrivileges(v.sc)
		if actual != v.expect {
			t.Errorf("%s failed, expected %t but received %t", k, v.expect, actual)
		}
	}
}

func TestDetermineEffectiveSecurityContext(t *testing.T) {
	privileged := true
	runAsUser := int64(1)
	runAsUserPod := int64(12)
	runAsGroup := int64(123)
	runAsNonRoot := true
	runAsNonRootPod := false
	readOnlyRootFilesystem := true
	allowPrivilegeEscalation := true

	containerWithoutSc := &v1.Container{Name: "containerWithoutSc"}

	containerWithSc := &v1.Container{
		Name: "containerWithSc",
		SecurityContext: &v1.SecurityContext{
			SELinuxOptions: &v1.SELinuxOptions{
				User:  "user_c",
				Role:  "role_c",
				Type:  "type_c",
				Level: "s0_c",
			},
			Capabilities: &v1.Capabilities{
				Add:  []v1.Capability{"foo_c"},
				Drop: []v1.Capability{"bar_c"},
			},
			Privileged:               &privileged,
			ReadOnlyRootFilesystem:   &readOnlyRootFilesystem,
			RunAsUser:                &runAsUser,
			RunAsNonRoot:             &runAsNonRoot,
			AllowPrivilegeEscalation: &allowPrivilegeEscalation,
		},
	}

	podSc := &v1.PodSecurityContext{
		SELinuxOptions: &v1.SELinuxOptions{
			User:  "user_p",
			Role:  "role_p",
			Type:  "type_p",
			Level: "s0_p",
		},
		RunAsGroup:   &runAsGroup,
		RunAsUser:    &runAsUserPod,
		RunAsNonRoot: &runAsNonRootPod,
	}

	podWithoutSc := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:        "podWithoutSc",
			Namespace:   "namespace",
			Annotations: map[string]string{},
		},
		Spec: v1.PodSpec{
			ServiceAccountName: "default",
		},
	}

	podWithSc := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:        "podWithSc",
			Namespace:   "namespace",
			Annotations: map[string]string{},
		},
		Spec: v1.PodSpec{
			ServiceAccountName: "default",
			SecurityContext:    podSc,
		},
	}

	tests := []struct {
		name      string
		pod       *v1.Pod
		container *v1.Container
		expect    *v1.SecurityContext
	}{
		{
			name:      "none SecurityContext",
			pod:       podWithoutSc,
			container: containerWithoutSc,
			expect:    nil,
		},
		{
			name:      "pod SecurityContext",
			pod:       podWithSc,
			container: containerWithoutSc,
			expect: &v1.SecurityContext{
				SELinuxOptions: &v1.SELinuxOptions{
					User:  "user_p",
					Role:  "role_p",
					Type:  "type_p",
					Level: "s0_p",
				},
				RunAsGroup:   &runAsGroup,
				RunAsUser:    &runAsUserPod,
				RunAsNonRoot: &runAsNonRootPod,
			},
		},
		{
			name:      "container SecurityContext",
			pod:       podWithoutSc,
			container: containerWithSc,
			expect: &v1.SecurityContext{
				SELinuxOptions: &v1.SELinuxOptions{
					User:  "user_c",
					Role:  "role_c",
					Type:  "type_c",
					Level: "s0_c",
				},
				Capabilities: &v1.Capabilities{
					Add:  []v1.Capability{"foo_c"},
					Drop: []v1.Capability{"bar_c"},
				},
				Privileged:               &privileged,
				ReadOnlyRootFilesystem:   &readOnlyRootFilesystem,
				RunAsUser:                &runAsUser,
				RunAsNonRoot:             &runAsNonRoot,
				AllowPrivilegeEscalation: &allowPrivilegeEscalation,
			},
		},
		{
			name:      "pod SecurityContext and container SecurityContext",
			pod:       podWithSc,
			container: containerWithSc,
			expect: &v1.SecurityContext{
				SELinuxOptions: &v1.SELinuxOptions{
					User:  "user_c",
					Role:  "role_c",
					Type:  "type_c",
					Level: "s0_c",
				},
				Capabilities: &v1.Capabilities{
					Add:  []v1.Capability{"foo_c"},
					Drop: []v1.Capability{"bar_c"},
				},
				Privileged:               &privileged,
				ReadOnlyRootFilesystem:   &readOnlyRootFilesystem,
				RunAsGroup:               &runAsGroup,
				RunAsUser:                &runAsUser,
				RunAsNonRoot:             &runAsNonRoot,
				AllowPrivilegeEscalation: &allowPrivilegeEscalation,
			},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			result := DetermineEffectiveSecurityContext(test.pod, test.container)
			if !reflect.DeepEqual(result, test.expect) {
				t.Errorf("Expected SecurityContext: %v, Got SecurityContext: %v", test.expect, result)
			}
		})
	}
}
