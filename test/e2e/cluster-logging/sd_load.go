/*
Copyright 2016 The Kubernetes Authors.

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

	meta_v1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/kubernetes/test/e2e/framework"

	. "github.com/onsi/ginkgo"
)

const (
	loadTestMaxAllowedLostFraction    = 0.01
	loadTestMaxAllowedFluentdRestarts = 1
)

var _ = framework.KubeDescribe("Cluster level logging using GCL [Feature:StackdriverLogging]", func() {
	f := framework.NewDefaultFramework("gcl-logging-load")

	It("should create a constant load with long-living pods and ensure logs delivery", func() {
		gclLogsProvider, err := newGclLogsProvider(f)
		framework.ExpectNoError(err, "Failed to create GCL logs provider")

		err = gclLogsProvider.Init()
		defer gclLogsProvider.Cleanup()
		framework.ExpectNoError(err, "Failed to init GCL logs provider")

		nodes := framework.GetReadySchedulableNodesOrDie(f.ClientSet).Items
		nodeCount := len(nodes)
		podCount := 30 * nodeCount
		loggingDuration := 10 * time.Minute
		linesPerSecond := 1000 * nodeCount
		linesPerPod := linesPerSecond * int(loggingDuration.Seconds()) / podCount
		ingestionTimeout := 20 * time.Minute

		By("Running logs generator pods")
		pods := []*loggingPod{}
		for podIdx := 0; podIdx < podCount; podIdx++ {
			node := nodes[podIdx%len(nodes)]
			podName := fmt.Sprintf("logs-generator-%d-%d", linesPerPod, podIdx)
			pods = append(pods, createLoggingPod(f, podName, node.Name, linesPerPod, loggingDuration))

			defer f.PodClient().Delete(podName, &meta_v1.DeleteOptions{})
		}

		By("Waiting for all log lines to be ingested")
		config := &loggingTestConfig{
			LogsProvider:              gclLogsProvider,
			Pods:                      pods,
			IngestionTimeout:          ingestionTimeout,
			MaxAllowedLostFraction:    loadTestMaxAllowedLostFraction,
			MaxAllowedFluentdRestarts: loadTestMaxAllowedFluentdRestarts,
		}
		err = waitForFullLogsIngestion(f, config)
		if err != nil {
			framework.Failf("Failed to ingest logs: %v", err)
		} else {
			framework.Logf("Successfully ingested all logs")
		}
	})

	It("should create a constant load with short-living pods and ensure logs delivery", func() {
		gclLogsProvider, err := newGclLogsProvider(f)
		framework.ExpectNoError(err, "Failed to create GCL logs provider")

		err = gclLogsProvider.Init()
		defer gclLogsProvider.Cleanup()
		framework.ExpectNoError(err, "Failed to init GCL logs provider")

		nodes := framework.GetReadySchedulableNodesOrDie(f.ClientSet).Items
		maxPodCount := 10
		jobDuration := 1 * time.Minute
		linesPerPodPerSecond := 100
		testDuration := 10 * time.Minute
		ingestionTimeout := 20 * time.Minute

		podRunDelay := time.Duration(int64(jobDuration) / int64(maxPodCount))
		podRunCount := int(testDuration.Seconds())/int(podRunDelay.Seconds()) - 1
		linesPerPod := linesPerPodPerSecond * int(jobDuration.Seconds())

		By("Running short-living pods")
		pods := []*loggingPod{}
		for runIdx := 0; runIdx < podRunCount; runIdx++ {
			for nodeIdx, node := range nodes {
				podName := fmt.Sprintf("job-logs-generator-%d-%d-%d-%d", maxPodCount, linesPerPod, runIdx, nodeIdx)
				pods = append(pods, createLoggingPod(f, podName, node.Name, linesPerPod, jobDuration))

				defer f.PodClient().Delete(podName, &meta_v1.DeleteOptions{})
			}
			time.Sleep(podRunDelay)
		}

		By("Waiting for all log lines to be ingested")
		config := &loggingTestConfig{
			LogsProvider:              gclLogsProvider,
			Pods:                      pods,
			IngestionTimeout:          ingestionTimeout,
			MaxAllowedLostFraction:    loadTestMaxAllowedLostFraction,
			MaxAllowedFluentdRestarts: loadTestMaxAllowedFluentdRestarts,
		}
		err = waitForFullLogsIngestion(f, config)
		if err != nil {
			framework.Failf("Failed to ingest logs: %v", err)
		} else {
			framework.Logf("Successfully ingested all logs")
		}
	})
})
