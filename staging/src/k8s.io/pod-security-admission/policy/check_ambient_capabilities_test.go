/*
Copyright The Kubernetes Authors.

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
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

func TestAmbientCapabilitiesAdmission(t *testing.T) {
	for _, tc := range []struct {
		name    string
		check   func(*metav1.ObjectMeta, *corev1.PodSpec) CheckResult
		cap     corev1.Capability
		allowed bool
	}{
		{"baseline rejects all", capabilitiesBaseline_1_0, "ALL", false},
		{"restricted rejects all", capabilitiesRestricted_1_22, "ALL", false},
		{"restricted rejects lowercase", capabilitiesRestricted_1_22, "net_bind_service", false},
		{"baseline rejects sys admin", capabilitiesBaseline_1_0, "SYS_ADMIN", false},
		{"baseline permits chown", capabilitiesBaseline_1_0, "CHOWN", true},
		{"restricted rejects chown", capabilitiesRestricted_1_22, "CHOWN", false},
		{"restricted permits bind service", capabilitiesRestricted_1_22, "NET_BIND_SERVICE", true},
	} {
		for _, kind := range []string{"container", "init", "ephemeral"} {
			t.Run(tc.name+"/"+kind, func(t *testing.T) {
				c := corev1.Container{Name: "test", SecurityContext: &corev1.SecurityContext{Capabilities: &corev1.Capabilities{Drop: []corev1.Capability{"ALL"}, Ambient: []corev1.Capability{tc.cap}}}}
				pod := &corev1.Pod{}
				switch kind {
				case "container":
					pod.Spec.Containers = []corev1.Container{c}
				case "init":
					pod.Spec.InitContainers = []corev1.Container{c}
				case "ephemeral":
					pod.Spec.EphemeralContainers = []corev1.EphemeralContainer{{EphemeralContainerCommon: corev1.EphemeralContainerCommon(c)}}
				}
				got := tc.check(&pod.ObjectMeta, &pod.Spec)
				if got.Allowed != tc.allowed {
					t.Fatalf("allowed=%v, want %v: %s", got.Allowed, tc.allowed, got.ForbiddenDetail)
				}
				if !got.Allowed {
					want := `container "test" must not include "` + string(tc.cap) + `" in securityContext.capabilities.ambient`
					if got.ForbiddenDetail != want {
						t.Fatalf("rejection=%q, want %q", got.ForbiddenDetail, want)
					}
				}
			})
		}
	}
}

func TestAmbientCapabilitiesAdmissionFieldNames(t *testing.T) {
	for name, check := range map[string]func(*metav1.ObjectMeta, *corev1.PodSpec) CheckResult{
		"baseline":   capabilitiesBaseline_1_0,
		"restricted": capabilitiesRestricted_1_22,
	} {
		for _, tc := range []struct {
			name         string
			add, ambient []corev1.Capability
			fields       string
		}{
			{"add only", []corev1.Capability{"SYS_ADMIN"}, nil, "securityContext.capabilities.add"},
			{"ambient only", nil, []corev1.Capability{"SYS_ADMIN"}, "securityContext.capabilities.ambient"},
			{"allowed add", []corev1.Capability{"NET_BIND_SERVICE"}, []corev1.Capability{"SYS_ADMIN"}, "securityContext.capabilities.ambient"},
			{"allowed ambient", []corev1.Capability{"SYS_ADMIN"}, []corev1.Capability{"NET_BIND_SERVICE"}, "securityContext.capabilities.add"},
			{"both", []corev1.Capability{"SYS_ADMIN"}, []corev1.Capability{"SYS_ADMIN"}, "securityContext.capabilities.add or securityContext.capabilities.ambient"},
		} {
			t.Run(name+"/"+tc.name, func(t *testing.T) {
				pod := &corev1.Pod{Spec: corev1.PodSpec{Containers: []corev1.Container{{Name: "test", SecurityContext: &corev1.SecurityContext{Capabilities: &corev1.Capabilities{Add: tc.add, Ambient: tc.ambient, Drop: []corev1.Capability{"ALL"}}}}}}}
				got := check(&pod.ObjectMeta, &pod.Spec)
				want := `container "test" must not include "SYS_ADMIN" in ` + tc.fields
				if got.Allowed || got.ForbiddenDetail != want {
					t.Fatalf("allowed=%v rejection=%q, want disallowed with %q", got.Allowed, got.ForbiddenDetail, want)
				}
			})
		}
	}
}
