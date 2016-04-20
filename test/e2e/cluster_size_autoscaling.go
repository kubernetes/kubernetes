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
	"os/exec"
	"time"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/test/e2e/framework"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
)

const (
	scaleUpTimeout   = 20 * time.Minute
	scaleDownTimeout = 30 * time.Minute
)

// [Feature:ClusterSizeAutoscaling]: Cluster size autoscaling is experimental
// and require Google Cloud Monitoring to be enabled, so these tests are not
// run by default.
//
// These tests take ~20 minutes to run each.
var _ = framework.KubeDescribe("Cluster size autoscaling [Feature:ClusterSizeAutoscaling] [Slow]", func() {
	f := framework.NewDefaultFramework("autoscaling")
	var nodeCount int
	var coresPerNode int
	var memCapacityMb int

	BeforeEach(func() {
		framework.SkipUnlessProviderIs("gce")

		nodes := framework.ListSchedulableNodesOrDie(f.Client)
		nodeCount = len(nodes.Items)
		Expect(nodeCount).NotTo(BeZero())
		cpu := nodes.Items[0].Status.Capacity[api.ResourceCPU]
		mem := nodes.Items[0].Status.Capacity[api.ResourceMemory]
		coresPerNode = int((&cpu).MilliValue() / 1000)
		memCapacityMb = int((&mem).Value() / 1024 / 1024)
	})

	AfterEach(func() {
		cleanUpAutoscaler()
	})

	It("Should scale cluster size based on cpu utilization", func() {
		setUpAutoscaler("cpu/node_utilization", 0.4, nodeCount, nodeCount+1)

		// Consume 50% CPU
		rcs := createConsumingRCs(f, "cpu-utilization", nodeCount*coresPerNode, 500, 0)
		err := framework.WaitForClusterSize(f.Client, nodeCount+1, scaleUpTimeout)
		for _, rc := range rcs {
			rc.CleanUp()
		}
		framework.ExpectNoError(err)

		framework.ExpectNoError(framework.WaitForClusterSize(f.Client, nodeCount, scaleDownTimeout))
	})

	It("Should scale cluster size based on cpu reservation", func() {
		setUpAutoscaler("cpu/node_reservation", 0.5, nodeCount, nodeCount+1)

		ReserveCpu(f, "cpu-reservation", 600*nodeCount*coresPerNode)
		framework.ExpectNoError(framework.WaitForClusterSize(f.Client, nodeCount+1, scaleUpTimeout))

		framework.ExpectNoError(framework.DeleteRC(f.Client, f.Namespace.Name, "cpu-reservation"))
		framework.ExpectNoError(framework.WaitForClusterSize(f.Client, nodeCount, scaleDownTimeout))
	})

	It("Should scale cluster size based on memory utilization", func() {
		setUpAutoscaler("memory/node_utilization", 0.6, nodeCount, nodeCount+1)

		// Consume 60% of total memory capacity
		megabytesPerReplica := int(memCapacityMb * 6 / 10 / coresPerNode)
		rcs := createConsumingRCs(f, "mem-utilization", nodeCount*coresPerNode, 0, megabytesPerReplica)
		err := framework.WaitForClusterSize(f.Client, nodeCount+1, scaleUpTimeout)
		for _, rc := range rcs {
			rc.CleanUp()
		}
		framework.ExpectNoError(err)

		framework.ExpectNoError(framework.WaitForClusterSize(f.Client, nodeCount, scaleDownTimeout))
	})

	It("Should scale cluster size based on memory reservation", func() {
		setUpAutoscaler("memory/node_reservation", 0.5, nodeCount, nodeCount+1)

		ReserveMemory(f, "memory-reservation", nodeCount*memCapacityMb*6/10)
		framework.ExpectNoError(framework.WaitForClusterSize(f.Client, nodeCount+1, scaleUpTimeout))

		framework.ExpectNoError(framework.DeleteRC(f.Client, f.Namespace.Name, "memory-reservation"))
		framework.ExpectNoError(framework.WaitForClusterSize(f.Client, nodeCount, scaleDownTimeout))
	})
})

func setUpAutoscaler(metric string, target float64, min, max int) {
	// TODO integrate with kube-up.sh script once it will support autoscaler setup.
	By("Setting up autoscaler to scale based on " + metric)
	out, err := exec.Command("gcloud", "compute", "instance-groups", "managed", "set-autoscaling",
		framework.TestContext.CloudConfig.NodeInstanceGroup,
		"--project="+framework.TestContext.CloudConfig.ProjectID,
		"--zone="+framework.TestContext.CloudConfig.Zone,
		"--custom-metric-utilization=metric=custom.cloudmonitoring.googleapis.com/kubernetes.io/"+metric+fmt.Sprintf(",utilization-target=%v", target)+",utilization-target-type=GAUGE",
		fmt.Sprintf("--min-num-replicas=%v", min),
		fmt.Sprintf("--max-num-replicas=%v", max),
	).CombinedOutput()
	framework.ExpectNoError(err, "Output: "+string(out))
}

func createConsumingRCs(f *framework.Framework, name string, count, cpuPerReplica, memPerReplica int) []*ResourceConsumer {
	var res []*ResourceConsumer
	for i := 1; i <= count; i++ {
		name := fmt.Sprintf("%s-%d", name, i)
		res = append(res, NewStaticResourceConsumer(name, 1, cpuPerReplica, memPerReplica, 0, int64(cpuPerReplica), int64(memPerReplica+100), f))
	}
	return res
}

func cleanUpAutoscaler() {
	By("Removing autoscaler")
	out, err := exec.Command("gcloud", "compute", "instance-groups", "managed", "stop-autoscaling",
		framework.TestContext.CloudConfig.NodeInstanceGroup,
		"--project="+framework.TestContext.CloudConfig.ProjectID,
		"--zone="+framework.TestContext.CloudConfig.Zone,
	).CombinedOutput()
	framework.ExpectNoError(err, "Output: "+string(out))
}

func ReserveCpu(f *framework.Framework, id string, millicores int) {
	By(fmt.Sprintf("Running RC which reserves %v millicores", millicores))
	config := &framework.RCConfig{
		Client:     f.Client,
		Name:       id,
		Namespace:  f.Namespace.Name,
		Timeout:    10 * time.Minute,
		Image:      "gcr.io/google_containers/pause-amd64:3.0",
		Replicas:   millicores / 100,
		CpuRequest: 100,
	}
	framework.ExpectNoError(framework.RunRC(*config))
}

func ReserveMemory(f *framework.Framework, id string, megabytes int) {
	By(fmt.Sprintf("Running RC which reserves %v MB of memory", megabytes))
	config := &framework.RCConfig{
		Client:     f.Client,
		Name:       id,
		Namespace:  f.Namespace.Name,
		Timeout:    10 * time.Minute,
		Image:      "gcr.io/google_containers/pause-amd64:3.0",
		Replicas:   megabytes / 500,
		MemRequest: 500 * 1024 * 1024,
	}
	framework.ExpectNoError(framework.RunRC(*config))
}
