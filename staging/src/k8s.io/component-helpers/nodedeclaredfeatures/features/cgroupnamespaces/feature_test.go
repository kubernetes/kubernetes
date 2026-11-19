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

package cgroupnamespaces

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
	for _, tc := range []struct {
		name           string
		featureGate    bool
		runtimeSupport bool
		expected       bool
	}{
		{name: "feature gate disabled", featureGate: false, runtimeSupport: true, expected: false},
		{name: "feature gate enabled but no runtime support", featureGate: true, runtimeSupport: false, expected: false},
		{name: "feature gate enabled and runtime supports it", featureGate: true, runtimeSupport: true, expected: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			cfg := &types.NodeConfiguration{
				FeatureGates:    fakeFeatureGate{CgroupNamespacesSupportFeatureGate: tc.featureGate},
				RuntimeFeatures: types.RuntimeFeatures{CgroupNamespaces: tc.runtimeSupport},
			}
			if got := Feature.Discover(cfg); got != tc.expected {
				t.Errorf("Discover() = %v, expected %v", got, tc.expected)
			}
		})
	}
}

func TestRequirements(t *testing.T) {
	reqs := Feature.Requirements()
	if reqs == nil || len(reqs.EnabledFeatureGates) != 1 || reqs.EnabledFeatureGates[0] != CgroupNamespacesSupportFeatureGate {
		t.Fatalf("unexpected requirements: %+v", reqs)
	}
	if reqs.RequiredRuntimeFeatures == nil || !reqs.RequiredRuntimeFeatures.CgroupNamespaces {
		t.Errorf("expected the CgroupNamespaces runtime feature to be required, got %+v", reqs.RequiredRuntimeFeatures)
	}
}

func TestName(t *testing.T) {
	if Feature.Name() != CgroupNamespacesSupport {
		t.Fatalf("expected Name to be %s, got %s", CgroupNamespacesSupport, Feature.Name())
	}
}

func TestInferForScheduling(t *testing.T) {
	tests := []struct {
		name     string
		spec     *v1.PodSpec
		expected bool
	}{
		{name: "nil spec", spec: nil, expected: false},
		{name: "unset", spec: &v1.PodSpec{InitContainers: []v1.Container{{Name: "init"}}, Containers: []v1.Container{{Name: "ctr"}}}, expected: false},
		{name: "container sets true", spec: &v1.PodSpec{Containers: []v1.Container{{Name: "ctr", HostCgroupNamespace: new(true)}}}, expected: true},
		{name: "container sets false", spec: &v1.PodSpec{Containers: []v1.Container{{Name: "ctr", HostCgroupNamespace: new(false)}}}, expected: true},
		{name: "init container sets false", spec: &v1.PodSpec{InitContainers: []v1.Container{{Name: "init", HostCgroupNamespace: new(false)}}, Containers: []v1.Container{{Name: "ctr"}}}, expected: true},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if got := Feature.InferForScheduling(&types.PodInfo{Spec: tc.spec}); got != tc.expected {
				t.Errorf("InferForScheduling() = %v, expected %v", got, tc.expected)
			}
		})
	}
}

func TestInferForUpdate(t *testing.T) {
	withField := &types.PodInfo{Spec: &v1.PodSpec{Containers: []v1.Container{{Name: "ctr", HostCgroupNamespace: new(true)}}}}
	withoutField := &types.PodInfo{Spec: &v1.PodSpec{Containers: []v1.Container{{Name: "ctr"}}}}
	if Feature.InferForUpdate(withoutField, withField) {
		t.Errorf("InferForUpdate() = true, expected false as the field is immutable")
	}
}
