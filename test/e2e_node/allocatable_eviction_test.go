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

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/apis/componentconfig"
	"k8s.io/kubernetes/pkg/kubelet/cm"
	"k8s.io/kubernetes/test/e2e/framework"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
)

// Eviction Policy is described here:
// https://github.com/kubernetes/kubernetes/blob/master/docs/proposals/kubelet-eviction.md

var _ = framework.KubeDescribe("AllocatableEviction [Slow] [Serial] [Disruptive] [Flaky]", func() {
	f := framework.NewDefaultFramework("allocatable-eviction-test")

	podTestSpecs := []podTestSpec{
		{
			evictionPriority: 1, // This pod should be evicted before the innocent pod
			pod:              *getMemhogPod("memory-hog-pod", "memory-hog", v1.ResourceRequirements{}),
		},
		{
			evictionPriority: 0, // This pod should never be evicted
			pod: v1.Pod{
				ObjectMeta: metav1.ObjectMeta{Name: "innocent-pod"},
				Spec: v1.PodSpec{
					RestartPolicy: v1.RestartPolicyNever,
					Containers: []v1.Container{
						{
							Image: "gcr.io/google_containers/busybox:1.24",
							Name:  "normal-memory-usage-container",
							Command: []string{
								"sh",
								"-c", //make one big (5 Gb) file
								"dd if=/dev/urandom of=largefile bs=5000000000 count=1; while true; do sleep 5; done",
							},
						},
					},
				},
			},
		},
	}
	evictionTestTimeout := 40 * time.Minute
	testCondition := "Memory Pressure"
	kubeletConfigUpdate := func(initialConfig *componentconfig.KubeletConfiguration) {
		initialConfig.EvictionHard = "memory.available<10%"
		// Set large system and kube reserved values to trigger allocatable thresholds far before hard eviction thresholds.
		initialConfig.SystemReserved = componentconfig.ConfigurationMap(map[string]string{"memory": "1Gi"})
		initialConfig.KubeReserved = componentconfig.ConfigurationMap(map[string]string{"memory": "1Gi"})
		initialConfig.EnforceNodeAllocatable = []string{cm.NodeAllocatableEnforcementKey}
		initialConfig.ExperimentalNodeAllocatableIgnoreEvictionThreshold = false
		initialConfig.CgroupsPerQOS = true
	}
	runEvictionTest(f, testCondition, podTestSpecs, evictionTestTimeout, hasMemoryPressure, kubeletConfigUpdate)
})

// Returns TRUE if the node has Memory Pressure, FALSE otherwise
func hasMemoryPressure(f *framework.Framework, testCondition string) (bool, error) {
	localNodeStatus := getLocalNode(f).Status
	_, pressure := v1.GetNodeCondition(&localNodeStatus, v1.NodeMemoryPressure)
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
