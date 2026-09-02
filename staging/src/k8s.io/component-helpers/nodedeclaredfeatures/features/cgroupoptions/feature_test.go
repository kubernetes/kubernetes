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

package cgroupoptions

import (
	"testing"

	v1 "k8s.io/api/core/v1"
	"k8s.io/component-helpers/nodedeclaredfeatures/types"
)

func TestRequirements(t *testing.T) {
	reqs := Feature.Requirements()
	if reqs == nil {
		t.Fatalf("Requirements returned nil")
	}
	if len(reqs.EnabledFeatureGates) != 1 || reqs.EnabledFeatureGates[0] != CgroupOptionsFeatureGate {
		t.Fatalf("unexpected required feature gates: %v", reqs.EnabledFeatureGates)
	}
	if reqs.RequiredRuntimeFeatures == nil || !reqs.RequiredRuntimeFeatures.CgroupMountMode {
		t.Fatalf("unexpected required runtime features: %v", reqs.RequiredRuntimeFeatures)
	}
	if reqs.RequiredStaticConfig == nil || !reqs.RequiredStaticConfig.Cgroup2UnifiedMode || !reqs.RequiredStaticConfig.CgroupsPerQOS || !reqs.RequiredStaticConfig.CgroupNsdelegate {
		t.Fatalf("unexpected required static config: %v", reqs.RequiredStaticConfig)
	}
}

func TestName(t *testing.T) {
	if Feature.Name() != CgroupOptionsFeatureName {
		t.Fatalf("expected Name to be %s, got %s", CgroupOptionsFeatureName, Feature.Name())
	}
}

func TestDiscoverFeature(t *testing.T) {
	tests := []struct {
		name               string
		featureGate        bool
		cgroup2UnifiedMode bool
		cgroupsPerQOS      bool
		cgroupNsdelegate   bool
		runtimeSupport     bool
		expected           bool
	}{
		{
			name:               "feature gate disabled",
			featureGate:        false,
			cgroup2UnifiedMode: true,
			cgroupsPerQOS:      true,
			cgroupNsdelegate:   true,
			runtimeSupport:     true,
			expected:           false,
		},
		{
			name:               "node runs cgroup v1",
			featureGate:        true,
			cgroup2UnifiedMode: false,
			cgroupsPerQOS:      true,
			cgroupNsdelegate:   true,
			runtimeSupport:     true,
			expected:           false,
		},
		{
			name:               "runtime does not support cgroup mount mode",
			featureGate:        true,
			cgroup2UnifiedMode: true,
			cgroupsPerQOS:      true,
			cgroupNsdelegate:   true,
			runtimeSupport:     false,
			expected:           false,
		},
		{
			name:               "kubelet does not manage per-QoS cgroups",
			featureGate:        true,
			cgroup2UnifiedMode: true,
			cgroupsPerQOS:      false,
			cgroupNsdelegate:   true,
			runtimeSupport:     true,
			expected:           false,
		},
		{
			name:               "cgroup hierarchy is not mounted with nsdelegate",
			featureGate:        true,
			cgroup2UnifiedMode: true,
			cgroupsPerQOS:      true,
			cgroupNsdelegate:   false,
			runtimeSupport:     true,
			expected:           false,
		},
		{
			name:               "all requirements met",
			featureGate:        true,
			cgroup2UnifiedMode: true,
			cgroupsPerQOS:      true,
			cgroupNsdelegate:   true,
			runtimeSupport:     true,
			expected:           true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cfg := &types.NodeConfiguration{
				FeatureGates: types.FeatureGateMap{CgroupOptionsFeatureGate: tt.featureGate},
				StaticConfig: types.StaticConfiguration{
					Cgroup2UnifiedMode: tt.cgroup2UnifiedMode,
					CgroupsPerQOS:      tt.cgroupsPerQOS,
					CgroupNsdelegate:   tt.cgroupNsdelegate,
				},
				RuntimeFeatures: types.RuntimeFeatures{CgroupMountMode: tt.runtimeSupport},
			}
			if got := Feature.Discover(cfg); got != tt.expected {
				t.Errorf("Discover() = %v, want %v", got, tt.expected)
			}
		})
	}
}

func writableCgroupsSecurityContext() *v1.SecurityContext {
	mountMode := v1.CgroupMountModeWritable
	return &v1.SecurityContext{CgroupOptions: &v1.CgroupOptions{MountMode: &mountMode}}
}

func readOnlyCgroupsSecurityContext() *v1.SecurityContext {
	mountMode := v1.CgroupMountModeReadOnly
	return &v1.SecurityContext{CgroupOptions: &v1.CgroupOptions{MountMode: &mountMode}}
}

func TestInferForScheduling(t *testing.T) {
	tests := []struct {
		name     string
		spec     v1.PodSpec
		expected bool
	}{
		{
			name:     "no security context",
			spec:     v1.PodSpec{Containers: []v1.Container{{Name: "a"}}},
			expected: false,
		},
		{
			name:     "read-only mount mode",
			spec:     v1.PodSpec{Containers: []v1.Container{{Name: "a", SecurityContext: readOnlyCgroupsSecurityContext()}}},
			expected: true,
		},
		{
			name:     "empty cgroupOptions",
			spec:     v1.PodSpec{Containers: []v1.Container{{Name: "a", SecurityContext: &v1.SecurityContext{CgroupOptions: &v1.CgroupOptions{}}}}},
			expected: false,
		},
		{
			name:     "writable container",
			spec:     v1.PodSpec{Containers: []v1.Container{{Name: "a", SecurityContext: writableCgroupsSecurityContext()}}},
			expected: true,
		},
		{
			name:     "writable init container",
			spec:     v1.PodSpec{InitContainers: []v1.Container{{Name: "a", SecurityContext: writableCgroupsSecurityContext()}}},
			expected: true,
		},
		{
			name:     "read-only init container",
			spec:     v1.PodSpec{InitContainers: []v1.Container{{Name: "a", SecurityContext: readOnlyCgroupsSecurityContext()}}},
			expected: true,
		},
		{
			name: "ephemeral containers do not require the feature",
			spec: v1.PodSpec{EphemeralContainers: []v1.EphemeralContainer{{
				EphemeralContainerCommon: v1.EphemeralContainerCommon{Name: "a", SecurityContext: writableCgroupsSecurityContext()},
			}}},
			expected: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			spec := tt.spec
			if got := Feature.InferForScheduling(&types.PodInfo{Spec: &spec}); got != tt.expected {
				t.Errorf("InferForScheduling() = %v, want %v", got, tt.expected)
			}
		})
	}
}

func TestInferForUpdate(t *testing.T) {
	plain := v1.PodSpec{Containers: []v1.Container{{Name: "a"}}}
	withWritable := v1.PodSpec{
		Containers: []v1.Container{{Name: "a", SecurityContext: writableCgroupsSecurityContext()}},
	}
	withReadOnly := v1.PodSpec{
		Containers: []v1.Container{{Name: "a", SecurityContext: readOnlyCgroupsSecurityContext()}},
	}

	tests := []struct {
		name     string
		old, new v1.PodSpec
		expected bool
	}{
		{name: "no change", old: plain, new: plain, expected: false},
		{name: "static Pod update requests writable cgroups", old: plain, new: withWritable, expected: true},
		{name: "static Pod update requests read-only cgroups", old: plain, new: withReadOnly, expected: true},
		{name: "read-only to writable already requires the feature", old: withReadOnly, new: withWritable, expected: false},
		{name: "writable to read-only already requires the feature", old: withWritable, new: withReadOnly, expected: false},
		{name: "already required", old: withWritable, new: withWritable, expected: false},
		{name: "no longer required", old: withWritable, new: plain, expected: false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			oldSpec, newSpec := tt.old, tt.new
			got := Feature.InferForUpdate(&types.PodInfo{Spec: &oldSpec}, &types.PodInfo{Spec: &newSpec})
			if got != tt.expected {
				t.Errorf("InferForUpdate() = %v, want %v", got, tt.expected)
			}
		})
	}
}
