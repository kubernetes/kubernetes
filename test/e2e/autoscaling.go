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

	. "github.com/onsi/ginkgo"
)

var _ = Describe("Autoscaling", func() {
	f := NewFramework("autoscaling")

	BeforeEach(func() {
		// Ensure cluster size is equal to 1.
		expectNoError(waitForClusterSize(f.Client, 1))
	})

	AfterEach(func() {
		cleanUpAutoscaler()
	})

	It("[Skipped] [Autoscaling] should scale cluster size based on cpu utilization", func() {
		setUpAutoscaler("cpu/node_utilization", 0.8, 1, 10)

		ConsumeCpu(f, "cpu-utilization", 1)
		expectNoError(waitForClusterSize(f.Client, 2))

		StopConsuming(f, "cpu-utilization")
		expectNoError(waitForClusterSize(f.Client, 1))
	})

	It("[Skipped] [Autoscaling] should scale cluster size based on cpu reservation", func() {
		setUpAutoscaler("cpu/node_reservation", 0.7, 1, 10)

		ReserveCpu(f, "cpu-reservation", 800)
		expectNoError(waitForClusterSize(f.Client, 2))

		StopConsuming(f, "cpu-reservation")
		expectNoError(waitForClusterSize(f.Client, 1))
	})

	It("[Skipped] [Autoscaling] should scale cluster size based on memory utilization", func() {
		setUpAutoscaler("memory/node_utilization", 0.5, 1, 10)

		ConsumeMemory(f, "memory-utilization", 2)
		expectNoError(waitForClusterSize(f.Client, 2))

		StopConsuming(f, "memory-utilization")
		expectNoError(waitForClusterSize(f.Client, 1))
	})

	It("[Skipped] [Autoscaling] should scale cluster size based on memory reservation", func() {
		setUpAutoscaler("memory/node_reservation", 0.5, 1, 10)

		ReserveMemory(f, "memory-reservation", 2)
		expectNoError(waitForClusterSize(f.Client, 2))

		StopConsuming(f, "memory-reservation")
		expectNoError(waitForClusterSize(f.Client, 1))
	})
})

func setUpAutoscaler(metric string, target float64, min, max int64) {
	// TODO integrate with kube-up.sh script once it will support autoscaler setup.
	By("Setting up autoscaler to scale based on " + metric)
	_, err := exec.Command("gcloud", "preview", "autoscaler",
		"--zone="+testContext.CloudConfig.Zone,
		"create", "e2e-test-autoscaler",
		"--project="+testContext.CloudConfig.ProjectID,
		"--target="+testContext.CloudConfig.NodeInstanceGroup,
		"--custom-metric=custom.cloudmonitoring.googleapis.com/kubernetes.io/"+metric,
		fmt.Sprintf("--target-custom-metric-utilization=%v", target),
		"--custom-metric-utilization-target-type=GAUGE",
		fmt.Sprintf("--min-num-replicas=%v", min),
		fmt.Sprintf("--max-num-replicas=%v", max),
	).CombinedOutput()
	expectNoError(err)
}

func cleanUpAutoscaler() {
	By("Removing autoscaler")
	_, err := exec.Command("gcloud", "preview", "autoscaler", "--zone="+testContext.CloudConfig.Zone, "delete", "e2e-test-autoscaler").CombinedOutput()
	expectNoError(err)
}

func ConsumeCpu(f *Framework, id string, cores int) {
	By(fmt.Sprintf("Running RC which consumes %v cores", cores))
	config := &RCConfig{
		Client:    f.Client,
		Name:      id,
		Namespace: f.Namespace.Name,
		Timeout:   10 * time.Minute,
		Image:     "jess/stress",
		Command:   []string{"stress", "-c", "1"},
		Replicas:  cores,
	}
	expectNoError(RunRC(*config))
}

func ConsumeMemory(f *Framework, id string, gigabytes int) {
	By(fmt.Sprintf("Running RC which consumes %v GB of memory", gigabytes))
	config := &RCConfig{
		Client:    f.Client,
		Name:      id,
		Namespace: f.Namespace.Name,
		Timeout:   10 * time.Minute,
		Image:     "jess/stress",
		Command:   []string{"stress", "-m", "1"},
		Replicas:  4 * gigabytes,
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
	By("Stopping RC " + id)
	expectNoError(DeleteRC(f.Client, f.Namespace.Name, id))
}
