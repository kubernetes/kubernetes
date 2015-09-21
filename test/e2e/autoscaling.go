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
	"k8s.io/kubernetes/pkg/util"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
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
		setUpAutoscaler("cpu/node_utilization", 0.7, nodeCount, nodeCount+1)

		ConsumeCpu(f, "cpu-utilization", nodeCount*coresPerNode)
		expectNoError(waitForClusterSize(f.Client, nodeCount+1, 20*time.Minute))

		StopConsuming(f, "cpu-utilization")
		expectNoError(waitForClusterSize(f.Client, nodeCount, 20*time.Minute))
	})

	It("[Skipped] should scale cluster size based on cpu reservation", func() {
		setUpAutoscaler("cpu/node_reservation", 0.7, 1, 10)

		ReserveCpu(f, "cpu-reservation", 800)
		expectNoError(waitForClusterSize(f.Client, 2, 20*time.Minute))

		StopConsuming(f, "cpu-reservation")
		expectNoError(waitForClusterSize(f.Client, 1, 20*time.Minute))
	})

	It("[Skipped][Autoscaling Suite] should scale cluster size based on memory utilization", func() {
		setUpAutoscaler("memory/node_utilization", 0.5, nodeCount, nodeCount+1)

		// Consume 60% of total memory capacity in 256MB chunks.
		chunks := memCapacityMb * nodeCount * 6 / 10 / 256
		ConsumeMemory(f, "memory-utilization", chunks)
		expectNoError(waitForClusterSize(f.Client, nodeCount+1, 20*time.Minute))

		StopConsuming(f, "memory-utilization")
		expectNoError(waitForClusterSize(f.Client, nodeCount, 20*time.Minute))
	})

	It("[Skipped] should scale cluster size based on memory reservation", func() {
		setUpAutoscaler("memory/node_reservation", 0.5, 1, 10)

		ReserveMemory(f, "memory-reservation", 2)
		expectNoError(waitForClusterSize(f.Client, 2, 20*time.Minute))

		StopConsuming(f, "memory-reservation")
		expectNoError(waitForClusterSize(f.Client, 1, 20*time.Minute))
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

func CreateService(f *Framework, name string) {
	By("Running sevice" + name)
	service := &api.Service{
		ObjectMeta: api.ObjectMeta{
			Name: name,
		},
		Spec: api.ServiceSpec{
			Selector: map[string]string{
				"name": name,
			},
			Ports: []api.ServicePort{{
				Port:       8080,
				TargetPort: util.NewIntOrStringFromInt(8080),
			}},
		},
	}
	_, err := f.Client.Services(f.Namespace.Name).Create(service)
	Expect(err).NotTo(HaveOccurred())
}

func ConsumeCpu(f *Framework, id string, cores int) {
	CreateService(f, id)
	By(fmt.Sprintf("Running RC which consumes %v cores", cores))
	config := &RCConfig{
		Client:     f.Client,
		Name:       id,
		Namespace:  f.Namespace.Name,
		Timeout:    10 * time.Minute,
		Image:      "jess/stress",
		Command:    []string{"stress", "-c", "1"},
		Replicas:   cores,
		CpuRequest: 500,
		CpuLimit:   1000,
	}
	expectNoError(RunRC(*config))
}

// Consume <chunks> chunks of size 256MB.
func ConsumeMemory(f *Framework, id string, chunks int) {
	CreateService(f, id)
	By(fmt.Sprintf("Running RC which consumes %v MB of memory in 256MB chunks", chunks*256))
	config := &RCConfig{
		Client:    f.Client,
		Name:      id,
		Namespace: f.Namespace.Name,
		Timeout:   10 * time.Minute,
		Image:     "jess/stress",
		Command:   []string{"stress", "-m", "1", "--vm-hang", "0"},
		Replicas:  chunks,
	}
	expectNoError(RunRC(*config))
}

func ReserveCpu(f *Framework, id string, millicores int) {
	By(fmt.Sprintf("Running RC which reserves %v millicores", millicores))
	config := &RCConfig{
		Client:    f.Client,
		Name:      id,
		Namespace: f.Namespace.Name,
		Timeout:   10 * time.Minute,
		Image:     "gcr.io/google_containers/pause",
		Replicas:  millicores / 100,
		CpuLimit:  100,
	}
	expectNoError(RunRC(*config))
}

func ReserveMemory(f *Framework, id string, gigabytes int) {
	By(fmt.Sprintf("Running RC which reserves %v GB of memory", gigabytes))
	config := &RCConfig{
		Client:    f.Client,
		Name:      id,
		Namespace: f.Namespace.Name,
		Timeout:   10 * time.Minute,
		Image:     "gcr.io/google_containers/pause",
		Replicas:  5 * gigabytes,
		MemLimit:  200 * 1024 * 1024,
	}
	expectNoError(RunRC(*config))
}

func StopConsuming(f *Framework, id string) {
	By("Stopping service " + id)
	err := f.Client.Services(f.Namespace.Name).Delete(id)
	Expect(err).NotTo(HaveOccurred())
	By("Stopping RC " + id)
	expectNoError(DeleteRC(f.Client, f.Namespace.Name, id))
}
