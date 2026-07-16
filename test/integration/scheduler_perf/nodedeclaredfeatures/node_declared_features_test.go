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

package nodedeclaredfeatures

import (
	"context"
	"fmt"
	"os"
	"slices"
	"sort"
	"testing"
	"time"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/version"
	"k8s.io/apimachinery/pkg/util/wait"
	_ "k8s.io/component-base/logs/json/register"
	ndf "k8s.io/component-helpers/nodedeclaredfeatures"
	ndffeatures "k8s.io/component-helpers/nodedeclaredfeatures/features"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/scheduler"
	perf "k8s.io/kubernetes/test/integration/scheduler_perf"
	"k8s.io/kubernetes/test/utils/ktesting"
)

// This test benchmarks the NodeDeclaredFeatures features performance impact.
//
// It mocks a variable number of node features registered (via 'numNodeDeclaredFeatures' param) but the
// inference function of all the registered features look for pod level resouces ('spec.resources').
// The tests creates a pod with pod level resources to ensure that all features are inferred
// during scheduling.

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
	// This function is called by kubelet to discover features on the node and report them in the node status.
	// In the test context, feature discovery is bypassed and we update node status directly after
	// the node is created (see updateNodesWithDeclaredFeatures()), so this function is a no-op.
	return true
}

func (f *mockFeature) InferForScheduling(podInfo *ndf.PodInfo) bool {
	// Check if the pod is using pod level resources.
	return podInfo.Spec != nil && podInfo.Spec.Resources != nil
}

func (f *mockFeature) InferForUpdate(oldPodInfo, newPodInfo *ndf.PodInfo) bool {
	return false
}

func (f *mockFeature) Requirements() *ndf.FeatureRequirements {
	return nil
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

func setupFeatures(numNodeDeclaredFeatures int) func() {
	nodeFeatures := make([]ndf.Feature, 0, numNodeDeclaredFeatures)
	for i := 1; i <= numNodeDeclaredFeatures; i++ {
		featureName := fmt.Sprintf("Feature%d", i)
		nodeFeatures = append(nodeFeatures, createMockFeature(featureName, ""))
	}
	originalAllFeatures := ndffeatures.AllFeatures
	ndffeatures.AllFeatures = nodeFeatures
	return func() {
		ndffeatures.AllFeatures = originalAllFeatures
	}
}

func preInitNodeDeclaredFeatures(t ktesting.TContext, w *perf.Workload) (func(), error) {
	if !w.FeatureGates[features.NodeDeclaredFeatures] {
		t.Logf("Skipping NodeDeclaredFeatures pre-init as the feature gate is disabled")
		return func() {}, nil
	}

	numNodeDeclaredFeatures, err := w.GetParam("numNodeDeclaredFeatures")
	if err != nil {
		t.Logf("numNodeDeclaredFeatures param not specified in workload config")
		numNodeDeclaredFeatures = 0
	}

	t.Logf("PreInit: Setting up %d mock features for %s", numNodeDeclaredFeatures, w.Name)
	// Setup mock features to be registered in the NDF library.
	cleanupMockFeatures := setupFeatures(numNodeDeclaredFeatures)

	return cleanupMockFeatures, nil
}

func updateNodesWithDeclaredFeatures(tCtx ktesting.TContext, scheduler *scheduler.Scheduler, w *perf.Workload, nodes *v1.NodeList) error {
	if !w.FeatureGates[features.NodeDeclaredFeatures] {
		return nil
	}
	numNodeDeclaredFeatures, err := w.GetParam("numNodeDeclaredFeatures")
	if err != nil {
		tCtx.Logf("numNodeDeclaredFeatures param not specified in workload config")
		numNodeDeclaredFeatures = 0
	}

	var featureNames []string
	for i := 1; i <= numNodeDeclaredFeatures; i++ {
		featureNames = append(featureNames, fmt.Sprintf("Feature%d", i))
	}
	sort.Strings(featureNames)

	for _, node := range nodes.Items {
		nodeToUpdate := node.DeepCopy()
		nodeToUpdate.Status.DeclaredFeatures = featureNames
		_, err := tCtx.Client().CoreV1().Nodes().UpdateStatus(tCtx.Context, nodeToUpdate, metav1.UpdateOptions{})
		if err != nil {
			return fmt.Errorf("failed to update node %s status: %w", node.Name, err)
		}
		tCtx.Logf("Updated node %s status with %d features", node.Name, len(featureNames))
	}

	schedulerCache := scheduler.Cache
	err = wait.PollUntilContextTimeout(tCtx.Context, 100*time.Millisecond, 60*time.Second, true, func(ctx context.Context) (bool, error) {
		for _, n := range nodes.Items {
			nodeInfo, err := schedulerCache.GetNode(n.Name)
			if err != nil {
				return false, nil

			}
			cachedNode := nodeInfo.Node()
			if !slices.Equal(cachedNode.Status.DeclaredFeatures, featureNames) {
				return false, nil
			}
		}
		return true, nil
	})

	if err != nil {
		return fmt.Errorf("timeout waiting for all nodes to reflect status update: %w", err)
	}

	return nil
}

func TestSchedulerPerf(t *testing.T) {
	perf.RunIntegrationPerfScheduling(t, "performance-config.yaml", perf.WithPreRunFn(preInitNodeDeclaredFeatures), perf.WithNodeUpdateFn(updateNodesWithDeclaredFeatures))
}

func BenchmarkPerfScheduling(b *testing.B) {
	perf.RunBenchmarkPerfScheduling(b, "performance-config.yaml", "nodedeclaredfeatures", nil, perf.WithPreRunFn(preInitNodeDeclaredFeatures), perf.WithNodeUpdateFn(updateNodesWithDeclaredFeatures))
}
