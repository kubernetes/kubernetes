/*
Copyright 2017 The Kubernetes Authors.

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

package e2e_node

import (
	"fmt"
	"time"

	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	nodeutil "k8s.io/kubernetes/pkg/api/v1/node"
	"k8s.io/kubernetes/pkg/kubelet/apis/kubeletconfig"
	"k8s.io/kubernetes/pkg/kubelet/cm"
	"k8s.io/kubernetes/test/e2e/framework"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
)

// Eviction Policy is described here:
// https://github.com/kubernetes/kubernetes/blob/master/docs/proposals/kubelet-eviction.md

var _ = framework.KubeDescribe("MemoryAllocatableEviction [Slow] [Serial] [Disruptive] [Flaky]", func() {
	f := framework.NewDefaultFramework("memory-allocatable-eviction-test")

	podTestSpecs := []podTestSpec{
		{
			evictionPriority: 1, // This pod should be evicted before the innocent pod
			pod:              getMemhogPod("memory-hog-pod", "memory-hog", v1.ResourceRequirements{}),
		},
		{
			evictionPriority: 0, // This pod should never be evicted
			pod:              getInnocentPod(),
		},
	}
	evictionTestTimeout := 10 * time.Minute
	testCondition := "Memory Pressure"

	Context(fmt.Sprintf("when we run containers that should cause %s", testCondition), func() {
		tempSetCurrentKubeletConfig(f, func(initialConfig *kubeletconfig.KubeletConfiguration) {
			// Set large system and kube reserved values to trigger allocatable thresholds far before hard eviction thresholds.
			kubeReserved := getNodeCPUAndMemoryCapacity(f)[v1.ResourceMemory]
			// The default hard eviction threshold is 250Mb, so Allocatable = Capacity - Reserved - 250Mb
			// We want Allocatable = 50Mb, so set Reserved = Capacity - Allocatable - 250Mb = Capacity - 300Mb
			kubeReserved.Sub(resource.MustParse("300Mi"))
			initialConfig.KubeReserved = kubeletconfig.ConfigurationMap(map[string]string{"memory": kubeReserved.String()})
			initialConfig.EnforceNodeAllocatable = []string{cm.NodeAllocatableEnforcementKey}
			initialConfig.ExperimentalNodeAllocatableIgnoreEvictionThreshold = false
			initialConfig.CgroupsPerQOS = true
		})
		// Place the remainder of the test within a context so that the kubelet config is set before and after the test.
		Context("With kubeconfig updated", func() {
			runEvictionTest(f, testCondition, podTestSpecs, evictionTestTimeout, hasMemoryPressure)
		})
	})
})

// Returns TRUE if the node has Memory Pressure, FALSE otherwise
func hasMemoryPressure(f *framework.Framework, testCondition string) (bool, error) {
	localNodeStatus := getLocalNode(f).Status
	_, pressure := nodeutil.GetNodeCondition(&localNodeStatus, v1.NodeMemoryPressure)
	Expect(pressure).NotTo(BeNil())
	hasPressure := pressure.Status == v1.ConditionTrue
	By(fmt.Sprintf("checking if pod has %s: %v", testCondition, hasPressure))

	// Additional Logging relating to Memory
	summary, err := getNodeSummary()
	if err != nil {
		return false, err
	}
	if summary.Node.Memory != nil && summary.Node.Memory.WorkingSetBytes != nil && summary.Node.Memory.AvailableBytes != nil {
		framework.Logf("Node.Memory.WorkingSetBytes: %d, summary.Node.Memory.AvailableBytes: %d", *summary.Node.Memory.WorkingSetBytes, *summary.Node.Memory.AvailableBytes)
	}
	for _, pod := range summary.Pods {
		framework.Logf("Pod: %s", pod.PodRef.Name)
		for _, container := range pod.Containers {
			if container.Memory != nil && container.Memory.WorkingSetBytes != nil {
				framework.Logf("--- summary Container: %s WorkingSetBytes: %d", container.Name, *container.Memory.WorkingSetBytes)
			}
		}
	}
	return hasPressure, nil
}
