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

package securityprofileoci

import (
	"testing"

	v1 "k8s.io/api/core/v1"
	"k8s.io/component-helpers/nodedeclaredfeatures/types"
)

type fakeFeatureGate map[string]bool

func (f fakeFeatureGate) Enabled(key string) bool {
	return f[key]
}

func TestDiscover(t *testing.T) {
	for _, enabled := range []bool{true, false} {
		cfg := &types.NodeConfiguration{FeatureGates: fakeFeatureGate{SecurityProfileOCIFeatureGate: enabled}}
		if got := Feature.Discover(cfg); got != enabled {
			t.Errorf("Discover() with gate enabled=%v = %v", enabled, got)
		}
	}
}

func TestRequirements(t *testing.T) {
	reqs := Feature.Requirements()
	if reqs == nil || len(reqs.EnabledFeatureGates) != 1 || reqs.EnabledFeatureGates[0] != SecurityProfileOCIFeatureGate {
		t.Fatalf("unexpected requirements: %+v", reqs)
	}
	if reqs.RequiredRuntimeFeatures != nil {
		t.Errorf("expected no runtime feature requirements, got %+v", reqs.RequiredRuntimeFeatures)
	}
}

func TestName(t *testing.T) {
	if Feature.Name() != SecurityProfileOCI {
		t.Fatalf("expected Name to be %s, got %s", SecurityProfileOCI, Feature.Name())
	}
}

func TestMaxVersion(t *testing.T) {
	if v := Feature.MaxVersion(); v != nil {
		t.Errorf("expected no MaxVersion before GA, got %v", v)
	}
}

func TestInferForScheduling(t *testing.T) {
	oci := &v1.SeccompProfile{Type: v1.SeccompProfileTypeOCI, OCI: &v1.SecurityProfileOCI{Ref: "registry.example.com/profile@sha256:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"}}
	runtimeDefault := &v1.SeccompProfile{Type: v1.SeccompProfileTypeRuntimeDefault}
	withProfile := func(profile *v1.SeccompProfile) *v1.SecurityContext {
		return &v1.SecurityContext{SeccompProfile: profile}
	}

	tests := []struct {
		name     string
		spec     *v1.PodSpec
		expected bool
	}{{
		name: "no security context",
		spec: &v1.PodSpec{Containers: []v1.Container{{Name: "c"}}},
	}, {
		name: "RuntimeDefault profiles",
		spec: &v1.PodSpec{
			SecurityContext: &v1.PodSecurityContext{SeccompProfile: runtimeDefault},
			Containers:      []v1.Container{{Name: "c", SecurityContext: withProfile(runtimeDefault)}},
		},
	}, {
		name:     "pod-level OCI profile",
		spec:     &v1.PodSpec{SecurityContext: &v1.PodSecurityContext{SeccompProfile: oci}, Containers: []v1.Container{{Name: "c"}}},
		expected: true,
	}, {
		name:     "container OCI profile",
		spec:     &v1.PodSpec{Containers: []v1.Container{{Name: "a"}, {Name: "b", SecurityContext: withProfile(oci)}}},
		expected: true,
	}, {
		name:     "init container OCI profile",
		spec:     &v1.PodSpec{InitContainers: []v1.Container{{Name: "init", SecurityContext: withProfile(oci)}}, Containers: []v1.Container{{Name: "c"}}},
		expected: true,
	}, {
		name: "ephemeral container OCI profile",
		spec: &v1.PodSpec{
			Containers: []v1.Container{{Name: "c"}},
			EphemeralContainers: []v1.EphemeralContainer{{
				EphemeralContainerCommon: v1.EphemeralContainerCommon{Name: "debug", SecurityContext: withProfile(oci)},
			}},
		},
		expected: true,
	}}

	noOCI := &types.PodInfo{Spec: &v1.PodSpec{Containers: []v1.Container{{Name: "c"}}}}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			podInfo := &types.PodInfo{Spec: tc.spec}
			if got := Feature.InferForScheduling(podInfo); got != tc.expected {
				t.Errorf("InferForScheduling() = %v, want %v", got, tc.expected)
			}
			if got := Feature.InferForUpdate(noOCI, podInfo); got != tc.expected {
				t.Errorf("InferForUpdate() from a pod without OCI profiles = %v, want %v", got, tc.expected)
			}
			if Feature.InferForUpdate(podInfo, podInfo) {
				t.Error("InferForUpdate() without a change = true, want false")
			}
		})
	}
}
