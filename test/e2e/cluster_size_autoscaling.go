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
	"fmt"
	"time"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/test/e2e/framework"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
)

const (
	scaleTimeout = 5 * time.Minute
)

var _ = framework.KubeDescribe("Cluster size autoscaling [Feature:ClusterSizeAutoscaling] [Slow]", func() {
	f := framework.NewDefaultFramework("autoscaling")
	var nodeCount int
	var coresPerNode int
	var memCapacityMb int

	BeforeEach(func() {
		framework.SkipUnlessProviderIs("gce")

		nodes := framework.GetReadySchedulableNodesOrDie(f.Client)
		nodeCount = len(nodes.Items)
		Expect(nodeCount).NotTo(BeZero())
		cpu := nodes.Items[0].Status.Capacity[api.ResourceCPU]
		mem := nodes.Items[0].Status.Capacity[api.ResourceMemory]
		coresPerNode = int((&cpu).MilliValue() / 1000)
		memCapacityMb = int((&mem).Value() / 1024 / 1024)
	})

	It("Should correctly handle pending pods", func() {
		By("Too large pending pod does not increase cluster size")
		ReserveMemory(f, "memory-reservation", 1, memCapacityMb, false)
		// Verify, that cluster size is not changed.
		// TODO: find a better way of verification that the cluster size will remain unchanged.
		time.Sleep(scaleTimeout)
		framework.ExpectNoError(framework.WaitForClusterSize(f.Client, nodeCount, scaleTimeout))

		framework.ExpectNoError(framework.DeleteRC(f.Client, f.Namespace.Name, "memory-reservation"))
		framework.ExpectNoError(framework.WaitForClusterSize(f.Client, nodeCount, scaleTimeout))

		By("Small pending pods increase cluster size")
		ReserveMemory(f, "memory-reservation", 100, nodeCount*memCapacityMb, false)
		// Verify, that cluster size is increased
		framework.ExpectNoError(framework.WaitForClusterSize(f.Client, nodeCount+1, scaleTimeout))
		framework.ExpectNoError(framework.DeleteRC(f.Client, f.Namespace.Name, "memory-reservation"))
		// TODO(jsz): Disable the line bellow when scale down is implemented.
		framework.ExpectNoError(ResizeGroup(int32(nodeCount)))
		framework.ExpectNoError(framework.WaitForClusterSize(f.Client, nodeCount, scaleTimeout))

		By("Handling node port pods")
		CreateHostPortPods(f, "host-port", nodeCount+2, false)
		framework.ExpectNoError(framework.WaitForClusterSize(f.Client, nodeCount+2, scaleTimeout))
		framework.ExpectNoError(framework.DeleteRC(f.Client, f.Namespace.Name, "host-port"))
		// TODO(jsz): Disable the line bellow when scale down is implemented.
		framework.ExpectNoError(ResizeGroup(int32(nodeCount)))
		framework.ExpectNoError(framework.WaitForClusterSize(f.Client, nodeCount, scaleTimeout))
	})
})

func CreateHostPortPods(f *framework.Framework, id string, replicas int, expectRunning bool) {
	By(fmt.Sprintf("Running RC which reserves host port"))
	config := &framework.RCConfig{
		Client:    f.Client,
		Name:      id,
		Namespace: f.Namespace.Name,
		Timeout:   scaleTimeout,
		Image:     "gcr.io/google_containers/pause-amd64:3.0",
		Replicas:  replicas,
		HostPorts: map[string]int{"port1": 4321},
	}
	err := framework.RunRC(*config)
	if expectRunning {
		framework.ExpectNoError(err)
	}

}

func ReserveCpu(f *framework.Framework, id string, replicas, millicores int) {
	By(fmt.Sprintf("Running RC which reserves %v millicores", millicores))
	request := int64(millicores / replicas)
	config := &framework.RCConfig{
		Client:     f.Client,
		Name:       id,
		Namespace:  f.Namespace.Name,
		Timeout:    scaleTimeout,
		Image:      "gcr.io/google_containers/pause-amd64:3.0",
		Replicas:   replicas,
		CpuRequest: request,
	}
	framework.ExpectNoError(framework.RunRC(*config))
}

func ReserveMemory(f *framework.Framework, id string, replicas, megabytes int, expectRunning bool) {
	By(fmt.Sprintf("Running RC which reserves %v MB of memory", megabytes))
	request := int64(1024 * 1024 * megabytes / replicas)
	config := &framework.RCConfig{
		Client:     f.Client,
		Name:       id,
		Namespace:  f.Namespace.Name,
		Timeout:    scaleTimeout,
		Image:      "gcr.io/google_containers/pause-amd64:3.0",
		Replicas:   replicas,
		MemRequest: request,
	}
	err := framework.RunRC(*config)
	if expectRunning {
		framework.ExpectNoError(err)
	}
}
