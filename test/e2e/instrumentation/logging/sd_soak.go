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

package logging

import (
	"fmt"
	"math"
	"time"

	"k8s.io/kubernetes/test/e2e/framework"
	"k8s.io/kubernetes/test/e2e/instrumentation"

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

var _ = instrumentation.SIGDescribe("Cluster level logging implemented by Stackdriver [Feature:StackdriverLogging] [Soak]", func() {
	f := framework.NewDefaultFramework("sd-logging-load")

	It("should ingest logs from applications running for a prolonged amount of time", func() {
		sdLogsProvider, err := newSdLogsProvider(f)
		framework.ExpectNoError(err, "Failed to create Stackdriver logs provider")

		err = sdLogsProvider.Init()
		defer sdLogsProvider.Cleanup()
		framework.ExpectNoError(err, "Failed to init Stackdriver logs provider")

		nodes := framework.GetReadySchedulableNodesOrDie(f.ClientSet).Items
		maxPodCount := 10
		jobDuration := 30 * time.Minute
		linesPerPodPerSecond := 100
		// TODO(crassirostris): Increase to 21 hrs
		testDuration := 3 * time.Hour
		ingestionTimeout := testDuration + 30*time.Minute
		allowedRestarts := int(math.Ceil(float64(testDuration) /
			float64(time.Hour) * maxAllowedRestartsPerHour))

		podRunDelay := time.Duration(int64(jobDuration) / int64(maxPodCount))
		podRunCount := maxPodCount*(int(testDuration/jobDuration)-1) + 1
		linesPerPod := linesPerPodPerSecond * int(jobDuration.Seconds())

		// pods is a flat array of all pods to be run and to expect in Stackdriver.
		pods := []*loggingPod{}
		// podsByRun is a two-dimensional array of pods, first dimension is the run
		// index, the second dimension is the node index. Since we want to create
		// an equal load on all nodes, for the same run we have one pod per node.
		podsByRun := [][]*loggingPod{}
		for runIdx := 0; runIdx < podRunCount; runIdx++ {
			podsInRun := []*loggingPod{}
			for nodeIdx, node := range nodes {
				podName := fmt.Sprintf("job-logs-generator-%d-%d-%d-%d", maxPodCount, linesPerPod, runIdx, nodeIdx)
				pod := newLoggingPod(podName, node.Name, linesPerPod, jobDuration)
				pods = append(pods, pod)
				podsInRun = append(podsInRun, pod)
			}
			podsByRun = append(podsByRun, podsInRun)
		}

		By("Running short-living pods")
		go func() {
			for runIdx := 0; runIdx < podRunCount; runIdx++ {
				// Starting one pod on each node.
				for _, pod := range podsByRun[runIdx] {
					pod.Start(f)
				}
				time.Sleep(podRunDelay)
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
