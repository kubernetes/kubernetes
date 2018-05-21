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

package stackdriver

import (
	"fmt"
	"math"
	"time"

	"k8s.io/kubernetes/test/e2e/framework"
	instrumentation "k8s.io/kubernetes/test/e2e/instrumentation/common"
	"k8s.io/kubernetes/test/e2e/instrumentation/logging/utils"

	"github.com/onsi/ginkgo"
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

	ginkgo.It("should ingest logs from applications running for a prolonged amount of time", func() {
		withLogProviderForScope(f, podsScope, func(p *sdLogProvider) {
			nodes := framework.GetReadySchedulableNodesOrDie(f.ClientSet).Items
			maxPodCount := 10
			jobDuration := 30 * time.Minute
			linesPerPodPerSecond := 100
			// TODO(instrumentation): Increase to 21 hrs
			testDuration := 3 * time.Hour
			ingestionInterval := 1 * time.Minute
			ingestionTimeout := testDuration + 30*time.Minute
			allowedRestarts := int(math.Ceil(float64(testDuration) /
				float64(time.Hour) * maxAllowedRestartsPerHour))

			podRunDelay := time.Duration(int64(jobDuration) / int64(maxPodCount))
			podRunCount := maxPodCount*(int(testDuration/jobDuration)-1) + 1
			linesPerPod := linesPerPodPerSecond * int(jobDuration.Seconds())

			// pods is a flat array of all pods to be run and to expect in Stackdriver.
			pods := []utils.FiniteLoggingPod{}
			// podsByRun is a two-dimensional array of pods, first dimension is the run
			// index, the second dimension is the node index. Since we want to create
			// an equal load on all nodes, for the same run we have one pod per node.
			podsByRun := [][]utils.FiniteLoggingPod{}
			for runIdx := 0; runIdx < podRunCount; runIdx++ {
				podsInRun := []utils.FiniteLoggingPod{}
				for nodeIdx, node := range nodes {
					podName := fmt.Sprintf("job-logs-generator-%d-%d-%d-%d", maxPodCount, linesPerPod, runIdx, nodeIdx)
					pod := utils.NewLoadLoggingPod(podName, node.Name, linesPerPod, jobDuration)
					pods = append(pods, pod)
					podsInRun = append(podsInRun, pod)
				}
				podsByRun = append(podsByRun, podsInRun)
			}

			ginkgo.By("Running short-living pods")
			go func() {
				t := time.NewTicker(podRunDelay)
				defer t.Stop()
				for runIdx := 0; runIdx < podRunCount; runIdx++ {
					// Starting one pod on each node.
					for _, pod := range podsByRun[runIdx] {
						if err := pod.Start(f); err != nil {
							framework.Logf("Failed to start pod: %v", err)
						}
					}
					<-t.C
				}
			}()

			checker := utils.NewFullIngestionPodLogChecker(p, maxAllowedLostFraction, pods...)
			err := utils.WaitForLogs(checker, ingestionInterval, ingestionTimeout)
			framework.ExpectNoError(err)

			utils.EnsureLoggingAgentRestartsCount(f, p.LoggingAgentName(), allowedRestarts)
		})
	})
})
