/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

package e2e

import (
	"time"

	"k8s.io/kubernetes/pkg/client"
	"k8s.io/kubernetes/pkg/fields"
	"k8s.io/kubernetes/pkg/labels"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
)

const datapointAmount = 5

type resourceUsagePerContainer map[string]*containerResourceUsage

//TODO tweak those values.
var allowedUsage = resourceUsagePerContainer{
	"/": &containerResourceUsage{
		CPUUsageInCores:         0.3,
		MemoryUsageInBytes:      4000000000,
		MemoryWorkingSetInBytes: 2000000000,
	},
	"/docker-daemon": &containerResourceUsage{
		CPUUsageInCores:         0.05,
		MemoryUsageInBytes:      2500000000,
		MemoryWorkingSetInBytes: 1500000000,
	},
	"/kubelet": &containerResourceUsage{
		CPUUsageInCores:         0.1,
		MemoryUsageInBytes:      80000000,
		MemoryWorkingSetInBytes: 80000000,
	},
	"/kube-proxy": &containerResourceUsage{
		CPUUsageInCores:         0.02,
		MemoryUsageInBytes:      12000000,
		MemoryWorkingSetInBytes: 12000000,
	},
	"/system": &containerResourceUsage{
		CPUUsageInCores:         0.01,
		MemoryUsageInBytes:      70000000,
		MemoryWorkingSetInBytes: 35000000,
	},
}

func computeAverage(sliceOfUsages []resourceUsagePerContainer) (result resourceUsagePerContainer) {
	result = make(resourceUsagePerContainer)
	for _, container := range targetContainers {
		result[container] = &containerResourceUsage{}
	}
	for _, usage := range sliceOfUsages {
		for _, container := range targetContainers {
			singleResult := &containerResourceUsage{
				CPUUsageInCores:         result[container].CPUUsageInCores + usage[container].CPUUsageInCores,
				MemoryUsageInBytes:      result[container].MemoryUsageInBytes + usage[container].MemoryUsageInBytes,
				MemoryWorkingSetInBytes: result[container].MemoryWorkingSetInBytes + usage[container].MemoryWorkingSetInBytes,
			}
			result[container] = singleResult
		}
	}
	for _, container := range targetContainers {
		singleResult := &containerResourceUsage{
			CPUUsageInCores:         result[container].CPUUsageInCores / float64(len(sliceOfUsages)),
			MemoryUsageInBytes:      result[container].MemoryUsageInBytes / int64(len(sliceOfUsages)),
			MemoryWorkingSetInBytes: result[container].MemoryWorkingSetInBytes / int64(len(sliceOfUsages)),
		}
		result[container] = singleResult
	}
	return
}

// This tests does nothing except checking current resource usage of containers defained in kubelet_stats targetContainers variable.
// Test fails if an average container resource consumption over datapointAmount tries exceeds amount defined in allowedUsage.
var _ = Describe("ResourceUsage", func() {
	var c *client.Client
	BeforeEach(func() {
		var err error
		c, err = loadClient()
		expectNoError(err)
	})

	It("should not exceed expected amount.", func() {
		By("Getting ResourceConsumption on all nodes")
		nodeList, err := c.Nodes().List(labels.Everything(), fields.Everything())
		expectNoError(err)

		resourceUsagePerNode := make(map[string][]resourceUsagePerContainer)

		for i := 0; i < datapointAmount; i++ {
			for _, node := range nodeList.Items {
				resourceUsage, err := getOneTimeResourceUsageOnNode(c, node.Name, 5*time.Second)
				expectNoError(err)
				resourceUsagePerNode[node.Name] = append(resourceUsagePerNode[node.Name], resourceUsage)
			}
			time.Sleep(3 * time.Second)
		}

		averageResourceUsagePerNode := make(map[string]resourceUsagePerContainer)
		for _, node := range nodeList.Items {
			averageResourceUsagePerNode[node.Name] = computeAverage(resourceUsagePerNode[node.Name])
		}

		violating := make(map[string]resourceUsagePerContainer)
		for node, usage := range averageResourceUsagePerNode {
			for container, cUsage := range usage {
				Logf("%v on %v usage: %#v", container, node, cUsage)
				if !allowedUsage[container].isStrictlyGreaterThan(cUsage) {
					violating[node] = usage
				}
			}
		}

		Expect(violating).To(BeEmpty())
	})
})
