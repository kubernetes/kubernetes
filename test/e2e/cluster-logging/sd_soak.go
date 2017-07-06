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
	"math"
	"time"

	meta_v1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/kubernetes/test/e2e/framework"

	. "github.com/onsi/ginkgo"
)

const (
	// maxAllowedLostFraction is the fraction of lost logs considered acceptable.
	maxAllowedLostFraction = 0.01
	// maxAllowedRestartsPerHour is the number of fluentd container restarts
	// considered acceptable. Once per hour is fine for now, as long as it
	// doesn't loose too much logs.
	maxAllowedRestartsPerHour = 1.0
	// lastPodIngestionSlack is the amount of time to wait for the last pod's
	// logs to be ingested by the logging agent.
	lastPodIngestionSlack = 5 * time.Minute
)

var _ = framework.KubeDescribe("Cluster level logging implemented by Stackdriver [Feature:StackdriverLogging] [Soak]", func() {
	f := framework.NewDefaultFramework("sd-logging-load")

	It("should ingest logs from applications running for a prolonged amount of time", func() {
		sdLogsProvider, err := newSdLogsProvider(f)
		framework.ExpectNoError(err, "Failed to create Stackdriver logs provider")

		err = sdLogsProvider.Init()
		defer sdLogsProvider.Cleanup()
		framework.ExpectNoError(err, "Failed to init Stackdriver logs provider")

		nodes := framework.GetReadySchedulableNodesOrDie(f.ClientSet).Items
		maxPodCount := 10
		jobDuration := 1 * time.Hour
		linesPerPodPerSecond := 100
		testDuration := 21 * time.Hour
		ingestionTimeout := testDuration + 30*time.Minute
		allowedRestarts := int(math.Ceil(float64(testDuration) /
			float64(time.Hour) * maxAllowedRestartsPerHour))

		podRunDelay := time.Duration(int64(jobDuration) / int64(maxPodCount))
		podRunCount := maxPodCount*(int(testDuration/jobDuration)-1) + 1
		linesPerPod := linesPerPodPerSecond * int(jobDuration.Seconds())

		pods := []*loggingPod{}
		for runIdx := 0; runIdx < podRunCount; runIdx++ {
			for nodeIdx, node := range nodes {
				podName := fmt.Sprintf("job-logs-generator-%d-%d-%d-%d", maxPodCount, linesPerPod, runIdx, nodeIdx)
				pods = append(pods, newLoggingPod(podName, node.Name, linesPerPod, jobDuration))
			}
		}

		By("Running short-living pods")
		go func() {
			for _, pod := range pods {
				pod.Start(f)
				time.Sleep(podRunDelay)
				defer f.PodClient().Delete(pod.Name, &meta_v1.DeleteOptions{})
			}
			// Waiting until the last pod has completed
			time.Sleep(jobDuration - podRunDelay + lastPodIngestionSlack)
		}()

		By("Waiting for all log lines to be ingested")
		config := &loggingTestConfig{
			LogsProvider:              sdLogsProvider,
			Pods:                      pods,
			IngestionTimeout:          ingestionTimeout,
			MaxAllowedLostFraction:    maxAllowedLostFraction,
			MaxAllowedFluentdRestarts: allowedRestarts,
		}
		err = waitForFullLogsIngestion(f, config)
		if err != nil {
			framework.Failf("Failed to ingest logs: %v", err)
		} else {
			framework.Logf("Successfully ingested all logs")
		}
	})
})
