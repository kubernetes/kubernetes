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

package testing

import (
	"testing"

	"k8s.io/apimachinery/pkg/util/version"
	"k8s.io/component-helpers/nodedeclaredfeatures"
	"k8s.io/component-helpers/nodedeclaredfeatures/types"
)

var _ = types.FeatureGate((*MockFeatureGate)(nil))

type MockFeatureGate struct {
	t        *testing.T
	features map[string]bool
}

func (m *MockFeatureGate) SetEnabled(gate string, enabled bool) {
	m.features[gate] = enabled
}
func (m *MockFeatureGate) Enabled(gate string) bool {
	if enabled, known := m.features[gate]; known {
		return enabled
	}
	m.t.Errorf("unknown gate requested: %v", gate)
	return false
}

func NewMockFeatureGate(t *testing.T) *MockFeatureGate {
	return &MockFeatureGate{
		t:        t,
		features: map[string]bool{},
	}
}

var _ = types.Feature((*MockFeature)(nil))

type MockFeature struct {
	t                  testing.TB
	name               *string
	discover           func(cfg *types.NodeConfiguration) bool
	inferForScheduling func(podInfo *types.PodInfo) bool
	inferForUpdate     func(oldPodInfo, newPodInfo *types.PodInfo) bool
	maxVersion         **version.Version
	requirements       **types.FeatureRequirements
}

func (m *MockFeature) Name() string {
	if m.name == nil {
		m.t.Errorf("unexpected call to Name")
		return ""
	}
	return *m.name
}
func (m *MockFeature) SetName(name string) {
	m.name = &name
}
func (m *MockFeature) Discover(cfg *types.NodeConfiguration) bool {
	if m.discover == nil {
		m.t.Errorf("unexpected call to Discover")
		return false
	}
	return m.discover(cfg)
}
func (m *MockFeature) SetDiscover(discover func(cfg *types.NodeConfiguration) bool) {
	m.discover = discover
}
func (m *MockFeature) InferForScheduling(podInfo *types.PodInfo) bool {
	if m.inferForScheduling == nil {
		m.t.Errorf("unexpected call to InferForScheduling")
		return false
	}
	return m.inferForScheduling(podInfo)
}
func (m *MockFeature) SetInferForScheduling(inferForScheduling func(podInfo *types.PodInfo) bool) {
	m.inferForScheduling = inferForScheduling
}
func (m *MockFeature) InferForUpdate(oldPodInfo, newPodInfo *types.PodInfo) bool {
	if m.inferForUpdate == nil {
		m.t.Errorf("unexpected call to InferForUpdate")
		return false
	}
	return m.inferForUpdate(oldPodInfo, newPodInfo)
}
func (m *MockFeature) SetInferForUpdate(inferForUpdate func(oldPodInfo, newPodInfo *types.PodInfo) bool) {
	m.inferForUpdate = inferForUpdate
}
func (m *MockFeature) MaxVersion() *version.Version {
	if m.maxVersion == nil {
		m.t.Errorf("unexpected call to MaxVersion")
		return nil
	}
	return *m.maxVersion
}
func (m *MockFeature) SetMaxVersion(maxVersion *version.Version) {
	m.maxVersion = &maxVersion
}
func (m *MockFeature) Requirements() *nodedeclaredfeatures.FeatureRequirements {
	if m.requirements == nil {
		m.t.Errorf("unexpected call to Requirements")
		return nil
	}
	return *m.requirements
}
func (m *MockFeature) SetRequirements(req *nodedeclaredfeatures.FeatureRequirements) {
	m.requirements = &req
}

func NewMockFeature(t testing.TB) *MockFeature {
	return &MockFeature{
		t: t,
	}
}
