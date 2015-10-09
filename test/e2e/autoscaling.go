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
	"k8s.io/kubernetes/pkg/fields"
	"k8s.io/kubernetes/pkg/labels"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
)

const (
	scaleUpTimeout   = 20 * time.Minute
	scaleDownTimeout = 30 * time.Minute
)

var _ = Describe("Autoscaling", func() {
	f := NewFramework("autoscaling")
	var nodeCount int
	var coresPerNode int
	var memCapacityMb int

	BeforeEach(func() {
		SkipUnlessProviderIs("gce")

		nodes, err := f.Client.Nodes().List(labels.Everything(), fields.Everything())
		expectNoError(err)
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

	It("[Skipped][Autoscaling Suite] should scale cluster size based on cpu utilization", func() {
		setUpAutoscaler("cpu/node_utilization", 0.4, nodeCount, nodeCount+1)

		// Consume 50% CPU
		millicoresPerReplica := 500
		rc := NewStaticResourceConsumer("cpu-utilization", nodeCount*coresPerNode, millicoresPerReplica*nodeCount*coresPerNode, 0, int64(millicoresPerReplica), 100, f)
		expectNoError(waitForClusterSize(f.Client, nodeCount+1, scaleUpTimeout))

		rc.CleanUp()
		expectNoError(waitForClusterSize(f.Client, nodeCount, scaleDownTimeout))
	})

	It("[Skipped][Autoscaling Suite] should scale cluster size based on cpu reservation", func() {
		setUpAutoscaler("cpu/node_reservation", 0.5, nodeCount, nodeCount+1)

		ReserveCpu(f, "cpu-reservation", 600*nodeCount*coresPerNode)
		expectNoError(waitForClusterSize(f.Client, nodeCount+1, scaleUpTimeout))

		expectNoError(DeleteRC(f.Client, f.Namespace.Name, "cpu-reservation"))
		expectNoError(waitForClusterSize(f.Client, nodeCount, scaleDownTimeout))
	})

	It("[Skipped][Autoscaling Suite] should scale cluster size based on memory utilization", func() {
		setUpAutoscaler("memory/node_utilization", 0.6, nodeCount, nodeCount+1)

		// Consume 60% of total memory capacity
		megabytesPerReplica := int(memCapacityMb * 6 / 10 / coresPerNode)
		rc := NewStaticResourceConsumer("mem-utilization", nodeCount*coresPerNode, 0, megabytesPerReplica*nodeCount*coresPerNode, 100, int64(megabytesPerReplica+100), f)
		expectNoError(waitForClusterSize(f.Client, nodeCount+1, scaleUpTimeout))

		rc.CleanUp()
		expectNoError(waitForClusterSize(f.Client, nodeCount, scaleDownTimeout))
	})

	It("[Skipped][Autoscaling Suite] should scale cluster size based on memory reservation", func() {
		setUpAutoscaler("memory/node_reservation", 0.5, nodeCount, nodeCount+1)

		ReserveMemory(f, "memory-reservation", nodeCount*memCapacityMb*6/10)
		expectNoError(waitForClusterSize(f.Client, nodeCount+1, scaleUpTimeout))

		expectNoError(DeleteRC(f.Client, f.Namespace.Name, "memory-reservation"))
		expectNoError(waitForClusterSize(f.Client, nodeCount, scaleDownTimeout))
	})
})

func setUpAutoscaler(metric string, target float64, min, max int) {
	// TODO integrate with kube-up.sh script once it will support autoscaler setup.
	By("Setting up autoscaler to scale based on " + metric)
	out, err := exec.Command("gcloud", "compute", "instance-groups", "managed", "set-autoscaling",
		testContext.CloudConfig.NodeInstanceGroup,
		"--project="+testContext.CloudConfig.ProjectID,
		"--zone="+testContext.CloudConfig.Zone,
		"--custom-metric-utilization=metric=custom.cloudmonitoring.googleapis.com/kubernetes.io/"+metric+fmt.Sprintf(",utilization-target=%v", target)+",utilization-target-type=GAUGE",
		fmt.Sprintf("--min-num-replicas=%v", min),
		fmt.Sprintf("--max-num-replicas=%v", max),
	).CombinedOutput()
	expectNoError(err, "Output: "+string(out))
}

func cleanUpAutoscaler() {
	By("Removing autoscaler")
	out, err := exec.Command("gcloud", "compute", "instance-groups", "managed", "stop-autoscaling",
		testContext.CloudConfig.NodeInstanceGroup,
		"--project="+testContext.CloudConfig.ProjectID,
		"--zone="+testContext.CloudConfig.Zone,
	).CombinedOutput()
	expectNoError(err, "Output: "+string(out))
}

func ReserveCpu(f *Framework, id string, millicores int) {
	By(fmt.Sprintf("Running RC which reserves %v millicores", millicores))
	config := &RCConfig{
		Client:     f.Client,
		Name:       id,
		Namespace:  f.Namespace.Name,
		Timeout:    10 * time.Minute,
		Image:      "gcr.io/google_containers/pause",
		Replicas:   millicores / 100,
		CpuRequest: 100,
	}
	expectNoError(RunRC(*config))
}

func ReserveMemory(f *Framework, id string, megabytes int) {
	By(fmt.Sprintf("Running RC which reserves %v MB of memory", megabytes))
	config := &RCConfig{
		Client:     f.Client,
		Name:       id,
		Namespace:  f.Namespace.Name,
		Timeout:    10 * time.Minute,
		Image:      "gcr.io/google_containers/pause",
		Replicas:   megabytes / 500,
		MemRequest: 500 * 1024 * 1024,
	}
	expectNoError(RunRC(*config))
}
