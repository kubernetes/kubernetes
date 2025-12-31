/*
Copyright 2025 The Kubernetes Authors.

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

package nodedeclaredfeatures

import (
	"fmt"
	"os"
	"testing"

	"k8s.io/apimachinery/pkg/util/version"
	_ "k8s.io/component-base/logs/json/register"
	ndf "k8s.io/component-helpers/nodedeclaredfeatures"
	ndffeatures "k8s.io/component-helpers/nodedeclaredfeatures/features"
	perf "k8s.io/kubernetes/test/integration/scheduler_perf"
)

func TestMain(m *testing.M) {
	if err := perf.InitTests(); err != nil {
		fmt.Fprintf(os.Stderr, "%v\n", err)
		os.Exit(1)
	}

	m.Run()
}

// mockFeature is a mock implementation of the Feature interface for testing.
type mockFeature struct {
	name       string
	maxVersion *version.Version
}

func (f *mockFeature) Name() string {
	return f.name
}

func (f *mockFeature) Discover(cfg *ndf.NodeConfiguration) bool {
	return true
}

func (f *mockFeature) InferForScheduling(podInfo *ndf.PodInfo) bool {
	// Check if any container has an env var matching the feature name.
	if podInfo.Spec == nil {
		return false
	}
	for _, container := range podInfo.Spec.Containers {
		for _, envVar := range container.Env {
			if envVar.Value == f.name {
				return true
			}
		}
	}
	return false
}

func (f *mockFeature) InferForUpdate(oldPodInfo, newPodInfo *ndf.PodInfo) bool {
	return false
}

func (f *mockFeature) MaxVersion() *version.Version {
	return f.maxVersion
}

func createMockFeature(name string, maxVersionStr string) ndf.Feature {
	var v *version.Version
	if maxVersionStr != "" {
		v = version.MustParseSemantic(maxVersionStr)
	}
	return &mockFeature{
		name:       name,
		maxVersion: v,
	}
}

// maxDeclaredFeaturesTest is set based on the maximum number of features
// declared by tests in performance-config.yaml
const maxDeclaredFeaturesTest = 20

// featurePrefix is the prefix used for feature name during declaration in performance-config.yaml
const featurePrefix = "Feature"

func setupFeatures(numFeatures int) func() {
	nodeFeatures := make([]ndf.Feature, 0, numFeatures)
	for i := 1; i <= numFeatures; i++ {
		featureName := fmt.Sprintf("%s%d", featurePrefix, i)
		nodeFeatures = append(nodeFeatures, createMockFeature(featureName, ""))
	}
	originalAllFeatures := ndffeatures.AllFeatures
	ndffeatures.AllFeatures = nodeFeatures
	return func() {
		ndffeatures.AllFeatures = originalAllFeatures
	}
}

func TestSchedulerPerf(t *testing.T) {
	cleanup := setupFeatures(maxDeclaredFeaturesTest)
	defer cleanup()
	perf.RunIntegrationPerfScheduling(t, "performance-config.yaml")
}

func BenchmarkPerfScheduling(b *testing.B) {
	cleanup := setupFeatures(maxDeclaredFeaturesTest)
	defer cleanup()
	perf.RunBenchmarkPerfScheduling(b, "performance-config.yaml", "nodedeclaredfeatures", nil)
}
