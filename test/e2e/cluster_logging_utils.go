/*
Copyright 2015 The Kubernetes Authors.

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
	"regexp"
	"strconv"
	"time"

	"k8s.io/apimachinery/pkg/api/resource"
	meta_v1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/client-go/util/integer"
	"k8s.io/kubernetes/pkg/api"
	api_v1 "k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/test/e2e/framework"
)

const (
	// Duration of delay between any two attempts to check if all logs are ingested
	ingestionRetryDelay = 10 * time.Second

	// Amount of requested cores for logging container in millicores
	loggingContainerCpuRequest = 10

	// Amount of requested memory for logging container in bytes
	loggingContainerMemoryRequest = 10 * 1024 * 1024
)

var (
	// Regexp, matching the contents of log entries, parsed or not
	logEntryMessageRegex = regexp.MustCompile("(?:I\\d+ \\d+:\\d+:\\d+.\\d+       \\d+ logs_generator.go:67] )?(\\d+) .*")
)

// Type to track the progress of logs generating pod
type loggingPod struct {
	// Name of the pod
	Name string
	// If we didn't read some log entries, their
	// timestamps should be no less than this timestamp.
	// Effectively, timestamp of the last ingested entry
	// for which there's no missing entry before it
	LastTimestamp time.Time
	// Cache of ingested and read entries
	Occurrences map[int]*logEntry
	// Number of lines expected to be ingested from this pod
	ExpectedLinesNumber int
}

type logEntry struct {
	Payload   string
	Timestamp time.Time
}

type logsProvider interface {
	FluentdApplicationName() string
	EnsureWorking() error
	ReadEntries(*loggingPod) []*logEntry
}

type loggingTestConfig struct {
	LogsProvider              logsProvider
	Pods                      []*loggingPod
	IngestionTimeout          time.Duration
	MaxAllowedLostFraction    float64
	MaxAllowedFluentdRestarts int
}

func (entry *logEntry) getLogEntryNumber() (int, bool) {
	submatch := logEntryMessageRegex.FindStringSubmatch(entry.Payload)
	if submatch == nil || len(submatch) < 2 {
		return 0, false
	}

	lineNumber, err := strconv.Atoi(submatch[1])
	return lineNumber, err == nil
}

func createLoggingPod(f *framework.Framework, podName string, totalLines int, loggingDuration time.Duration) *loggingPod {
	framework.Logf("Starting pod %s", podName)
	createLogsGeneratorPod(f, podName, totalLines, loggingDuration)

	return &loggingPod{
		Name: podName,
		// It's used to avoid querying logs from before the pod was started
		LastTimestamp:       time.Now(),
		Occurrences:         make(map[int]*logEntry),
		ExpectedLinesNumber: totalLines,
	}
}

func createLogsGeneratorPod(f *framework.Framework, podName string, linesCount int, duration time.Duration) {
	f.PodClient().Create(&api_v1.Pod{
		ObjectMeta: meta_v1.ObjectMeta{
			Name: podName,
		},
		Spec: api_v1.PodSpec{
			RestartPolicy: api_v1.RestartPolicyNever,
			Containers: []api_v1.Container{
				{
					Name:  podName,
					Image: "gcr.io/google_containers/logs-generator:v0.1.0",
					Env: []api_v1.EnvVar{
						{
							Name:  "LOGS_GENERATOR_LINES_TOTAL",
							Value: strconv.Itoa(linesCount),
						},
						{
							Name:  "LOGS_GENERATOR_DURATION",
							Value: duration.String(),
						},
					},
					Resources: api_v1.ResourceRequirements{
						Requests: api_v1.ResourceList{
							api_v1.ResourceCPU: *resource.NewMilliQuantity(
								loggingContainerCpuRequest,
								resource.DecimalSI),
							api_v1.ResourceMemory: *resource.NewQuantity(
								loggingContainerMemoryRequest,
								resource.BinarySI),
						},
					},
				},
			},
		},
	})
}

func waitForLogsIngestion(f *framework.Framework, config *loggingTestConfig) error {
	expectedLinesNumber := 0
	for _, pod := range config.Pods {
		expectedLinesNumber += pod.ExpectedLinesNumber
	}

	totalMissing := expectedLinesNumber

	missingByPod := make([]int, len(config.Pods))
	for podIdx, pod := range config.Pods {
		missingByPod[podIdx] = pod.ExpectedLinesNumber
	}

	for start := time.Now(); totalMissing > 0 && time.Since(start) < config.IngestionTimeout; time.Sleep(ingestionRetryDelay) {
		missing := 0
		for podIdx, pod := range config.Pods {
			if missingByPod[podIdx] == 0 {
				continue
			}

			missingByPod[podIdx] = pullMissingLogsCount(config.LogsProvider, pod)
			missing += missingByPod[podIdx]
		}

		totalMissing = missing
		if totalMissing > 0 {
			framework.Logf("Still missing %d lines in total", totalMissing)
		}
	}

	lostFraction := float64(totalMissing) / float64(expectedLinesNumber)

	if totalMissing > 0 {
		framework.Logf("After %v still missing %d lines, %.2f%% of total number of lines",
			config.IngestionTimeout, totalMissing, lostFraction*100)
	}

	if lostFraction > config.MaxAllowedLostFraction {
		return fmt.Errorf("lost %.2f%% of lines, but only loss of %.2f%% can be tolerated",
			lostFraction*100, config.MaxAllowedLostFraction*100)
	}

	fluentdPods, err := getFluentdPods(f, config.LogsProvider.FluentdApplicationName())
	if err != nil {
		return fmt.Errorf("failed to get fluentd pods due to %v", err)
	}

	maxRestartCount := 0
	for _, fluentdPod := range fluentdPods.Items {
		restartCount := int(fluentdPod.Status.ContainerStatuses[0].RestartCount)
		maxRestartCount = integer.IntMax(maxRestartCount, restartCount)

		framework.Logf("Fluentd pod %s on node %s was restarted %d times",
			fluentdPod.Name, fluentdPod.Spec.NodeName, restartCount)
	}

	if maxRestartCount > config.MaxAllowedFluentdRestarts {
		return fmt.Errorf("max fluentd pod restarts was %d, which is more than allowed %d",
			maxRestartCount, config.MaxAllowedFluentdRestarts)
	}

	return nil
}

func pullMissingLogsCount(logsProvider logsProvider, pod *loggingPod) int {
	missingOnPod, err := getMissingLinesCount(logsProvider, pod)
	if err != nil {
		framework.Logf("Failed to get missing lines count from pod %s due to %v", pod.Name, err)
		return pod.ExpectedLinesNumber
	} else if missingOnPod > 0 {
		framework.Logf("Pod %s is missing %d lines", pod.Name, missingOnPod)
	} else {
		framework.Logf("All logs from pod %s are ingested", pod.Name)
	}
	return missingOnPod
}

func getMissingLinesCount(logsProvider logsProvider, pod *loggingPod) (int, error) {
	entries := logsProvider.ReadEntries(pod)

	framework.Logf("Got %d entries from provider", len(entries))

	for _, entry := range entries {
		lineNumber, ok := entry.getLogEntryNumber()
		if !ok {
			continue
		}

		if lineNumber < 0 || lineNumber >= pod.ExpectedLinesNumber {
			framework.Logf("Unexpected line number: %d", lineNumber)
		} else {
			pod.Occurrences[lineNumber] = entry
		}
	}

	for i := 0; i < pod.ExpectedLinesNumber; i++ {
		entry, ok := pod.Occurrences[i]
		if !ok {
			break
		}

		if entry.Timestamp.After(pod.LastTimestamp) {
			pod.LastTimestamp = entry.Timestamp
		}
	}

	return pod.ExpectedLinesNumber - len(pod.Occurrences), nil
}

func getFluentdPods(f *framework.Framework, fluentdApplicationName string) (*api_v1.PodList, error) {
	label := labels.SelectorFromSet(labels.Set(map[string]string{"k8s-app": fluentdApplicationName}))
	options := meta_v1.ListOptions{LabelSelector: label.String()}
	return f.ClientSet.Core().Pods(api.NamespaceSystem).List(options)
}
