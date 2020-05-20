/*
Copyright 2020 The Kubernetes Authors.

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

package scheduler

import (
	"testing"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/kubernetes/pkg/features"
	testutils "k8s.io/kubernetes/test/integration/util"
)

func TestNodeResourceLimits(t *testing.T) {
	defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.ResourceLimitsPriorityFunction, true)()

	testCtx := initTest(t, "node-resource-limits")
	defer testutils.CleanupTest(t, testCtx)

	// Add one node
	expectedNode, err := createNode(testCtx.ClientSet, "test-node1", &v1.ResourceList{
		v1.ResourcePods:   *resource.NewQuantity(32, resource.DecimalSI),
		v1.ResourceCPU:    *resource.NewMilliQuantity(2000, resource.DecimalSI),
		v1.ResourceMemory: *resource.NewQuantity(2000, resource.DecimalSI),
	})
	if err != nil {
		t.Fatalf("Cannot create node: %v", err)
	}

	// Add another node with less resource
	_, err = createNode(testCtx.ClientSet, "test-node2", &v1.ResourceList{
		v1.ResourcePods:   *resource.NewQuantity(32, resource.DecimalSI),
		v1.ResourceCPU:    *resource.NewMilliQuantity(1000, resource.DecimalSI),
		v1.ResourceMemory: *resource.NewQuantity(1000, resource.DecimalSI),
	})
	if err != nil {
		t.Fatalf("Cannot create node: %v", err)
	}

	podName := "pod-with-resource-limits"
	pod, err := runPausePod(testCtx.ClientSet, initPausePod(&pausePodConfig{
		Name:      podName,
		Namespace: testCtx.NS.Name,
		Resources: &v1.ResourceRequirements{Requests: v1.ResourceList{
			v1.ResourceCPU:    *resource.NewMilliQuantity(500, resource.DecimalSI),
			v1.ResourceMemory: *resource.NewQuantity(500, resource.DecimalSI)},
		},
	}))
	if err != nil {
		t.Fatalf("Error running pause pod: %v", err)
	}

	if pod.Spec.NodeName != expectedNode.Name {
		t.Errorf("pod %v got scheduled on an unexpected node: %v. Expected node: %v.", podName, pod.Spec.NodeName, expectedNode.Name)
	} else {
		t.Logf("pod %v got successfully scheduled on node %v.", podName, pod.Spec.NodeName)
	}
}
