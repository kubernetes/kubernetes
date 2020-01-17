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
	"time"

	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/kubernetes/test/e2e/framework"
	instrumentation "k8s.io/kubernetes/test/e2e/instrumentation/common"
	"k8s.io/kubernetes/test/e2e/instrumentation/logging/utils"

	"github.com/onsi/ginkgo"
	"k8s.io/apimachinery/pkg/util/json"
	"k8s.io/apimachinery/pkg/util/uuid"
)

const (
	ingestionInterval = 10 * time.Second
	ingestionTimeout  = 10 * time.Minute
)

var _ = instrumentation.SIGDescribe("Cluster level logging implemented by Stackdriver", func() {
	f := framework.NewDefaultFramework("sd-logging")

	ginkgo.BeforeEach(func() {
		framework.SkipUnlessProviderIs("gce", "gke")
	})

	ginkgo.It("should ingest logs [Feature:StackdriverLogging]", func() {
		withLogProviderForScope(f, podsScope, func(p *sdLogProvider) {
			ginkgo.By("Checking ingesting text logs", func() {
				pod, err := utils.StartAndReturnSelf(utils.NewRepeatingLoggingPod("synthlogger-1", "hey"), f)
				framework.ExpectNoError(err, "Failed to start a pod")

				ginkgo.By("Waiting for logs to ingest")
				c := utils.NewLogChecker(p, utils.UntilFirstEntry, utils.JustTimeout, pod.Name())
				err = utils.WaitForLogs(c, ingestionInterval, ingestionTimeout)
				framework.ExpectNoError(err)
			})

			ginkgo.By("Checking ingesting json logs", func() {
				logRaw := "{\"a\":\"b\"}"
				pod, err := utils.StartAndReturnSelf(utils.NewRepeatingLoggingPod("synthlogger-2", logRaw), f)
				framework.ExpectNoError(err, "Failed to start a pod")

				ginkgo.By("Waiting for logs to ingest")
				c := utils.NewLogChecker(p, func(_ string, logEntries []utils.LogEntry) (bool, error) {
					if len(logEntries) == 0 {
						return false, nil
					}
					log := logEntries[0]
					if log.JSONPayload == nil {
						return false, fmt.Errorf("log entry unexpectedly is not json: %s", log.TextPayload)
					}
					if log.JSONPayload["a"] != "b" {
						bytes, err := json.Marshal(log.JSONPayload)
						if err != nil {
							return false, fmt.Errorf("log entry ingested incorrectly, failed to marshal: %v", err)
						}
						return false, fmt.Errorf("log entry ingested incorrectly, got %v, want %s",
							string(bytes), logRaw)
					}
					return true, nil
				}, utils.JustTimeout, pod.Name())
				err = utils.WaitForLogs(c, ingestionInterval, ingestionTimeout)
				framework.ExpectNoError(err)
			})

			ginkgo.By("Checking ingesting logs in glog format", func() {
				logUnformatted := "Text"
				logRaw := fmt.Sprintf("I0101 00:00:00.000000       1 main.go:1] %s", logUnformatted)
				pod, err := utils.StartAndReturnSelf(utils.NewRepeatingLoggingPod("synthlogger-3", logRaw), f)
				framework.ExpectNoError(err, "Failed to start a pod")

				ginkgo.By("Waiting for logs to ingest")
				c := utils.NewLogChecker(p, func(_ string, logEntries []utils.LogEntry) (bool, error) {
					if len(logEntries) == 0 {
						return false, nil
					}
					log := logEntries[0]
					if log.TextPayload == "" {
						return false, fmt.Errorf("log entry is unexpectedly json: %v", log.JSONPayload)
					}
					if log.TextPayload != logUnformatted {
						return false, fmt.Errorf("log entry ingested incorrectly, got %s, want %s",
							log.TextPayload, logUnformatted)
					}
					return true, nil
				}, utils.JustTimeout, pod.Name())
				err = utils.WaitForLogs(c, ingestionInterval, ingestionTimeout)
				framework.ExpectNoError(err)
			})

			ginkgo.By("Checking that too long lines are trimmed", func() {
				maxLength := 100 * 1024
				cmd := []string{
					"/bin/sh",
					"-c",
					fmt.Sprintf("while :; do printf '%%*s' %d | tr ' ' 'A'; echo; sleep 60; done", maxLength+1),
				}

				pod, err := utils.StartAndReturnSelf(utils.NewExecLoggingPod("synthlogger-4", cmd), f)
				framework.ExpectNoError(err, "Failed to start a pod")

				ginkgo.By("Waiting for logs to ingest")
				c := utils.NewLogChecker(p, func(_ string, logEntries []utils.LogEntry) (bool, error) {
					if len(logEntries) == 0 {
						return false, nil
					}
					log := logEntries[0]
					if log.JSONPayload != nil {
						return false, fmt.Errorf("got json log entry %v, wanted plain text", log.JSONPayload)
					}
					if len(log.TextPayload) > maxLength {
						return false, fmt.Errorf("got too long entry of length %d", len(log.TextPayload))
					}
					return true, nil
				}, utils.JustTimeout, pod.Name())
				err = utils.WaitForLogs(c, ingestionInterval, ingestionTimeout)
				framework.ExpectNoError(err)
			})
		})
	})

	ginkgo.It("should ingest events [Feature:StackdriverLogging]", func() {
		eventCreationInterval := 10 * time.Second

		withLogProviderForScope(f, eventsScope, func(p *sdLogProvider) {
			ginkgo.By("Running pods to generate events while waiting for some of them to be ingested")
			stopCh := make(chan struct{})
			cleanupCh := make(chan struct{})
			defer func() { <-cleanupCh }()
			defer close(stopCh)
			go func() {
				defer ginkgo.GinkgoRecover()
				defer close(cleanupCh)

				wait.PollUntil(eventCreationInterval, func() (bool, error) {
					podName := fmt.Sprintf("synthlogger-%s", string(uuid.NewUUID()))
					err := utils.NewLoadLoggingPod(podName, "", 1, 1*time.Second).Start(f)
					if err != nil {
						framework.Logf("Failed to create a logging pod: %v", err)
					}
					return false, nil
				}, stopCh)
			}()

			ginkgo.By("Waiting for events to ingest")
			location := framework.TestContext.CloudConfig.Zone
			if framework.TestContext.CloudConfig.MultiMaster {
				location = framework.TestContext.CloudConfig.Region
			}
			c := utils.NewLogChecker(p, utils.UntilFirstEntryFromLocation(location), utils.JustTimeout, "")
			err := utils.WaitForLogs(c, ingestionInterval, ingestionTimeout)
			framework.ExpectNoError(err)
		})
	})

	ginkgo.It("should ingest system logs from all nodes [Feature:StackdriverLogging]", func() {
		withLogProviderForScope(f, systemScope, func(p *sdLogProvider) {
			ginkgo.By("Waiting for some kubelet logs to be ingested from each node", func() {
				nodeIds := utils.GetNodeIds(f.ClientSet)
				log := fmt.Sprintf("projects/%s/logs/kubelet", framework.TestContext.CloudConfig.ProjectID)
				c := utils.NewLogChecker(p, utils.UntilFirstEntryFromLog(log), utils.JustTimeout, nodeIds...)
				err := utils.WaitForLogs(c, ingestionInterval, ingestionTimeout)
				framework.ExpectNoError(err)
			})

			ginkgo.By("Waiting for some container runtime logs to be ingested from each node", func() {
				nodeIds := utils.GetNodeIds(f.ClientSet)
				log := fmt.Sprintf("projects/%s/logs/container-runtime", framework.TestContext.CloudConfig.ProjectID)
				c := utils.NewLogChecker(p, utils.UntilFirstEntryFromLog(log), utils.JustTimeout, nodeIds...)
				err := utils.WaitForLogs(c, ingestionInterval, ingestionTimeout)
				framework.ExpectNoError(err)
			})
		})
	})
})
