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
	"os/exec"
	"strconv"
	"strings"
	"time"

	"k8s.io/apimachinery/pkg/util/json"
	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/test/e2e/framework"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
)

var _ = framework.KubeDescribe("Cluster level logging using GCL", func() {
	f := framework.NewDefaultFramework("gcl-logging")

	BeforeEach(func() {
		framework.SkipUnlessProviderIs("gce", "gke")
	})

	It("should check that logs from containers are ingested in GCL", func() {
		By("Running synthetic logger")
		createSynthLogger(f, expectedLinesCount)
		defer f.PodClient().Delete(synthLoggerPodName, &v1.DeleteOptions{})
		err := framework.WaitForPodSuccessInNamespace(f.ClientSet, synthLoggerPodName, f.Namespace.Name)
		framework.ExpectNoError(err, fmt.Sprintf("Should've successfully waited for pod %s to succeed", synthLoggerPodName))

		By("Waiting for logs to ingest")
		totalMissing := expectedLinesCount
		for start := time.Now(); time.Since(start) < ingestionTimeout; time.Sleep(ingestionRetryDelay) {
			var err error
			totalMissing, err = getMissingLinesCountGcl(f, synthLoggerPodName, expectedLinesCount)
			if err != nil {
				framework.Logf("Failed to get missing lines count due to %v", err)
				totalMissing = expectedLinesCount
			} else if totalMissing > 0 {
				framework.Logf("Still missing %d lines", totalMissing)
			}

			if totalMissing == 0 {
				break
			}
		}

		if totalMissing > 0 {
			if err := reportLogsFromFluentdPod(f); err != nil {
				framework.Logf("Failed to report logs from fluentd pod due to %v", err)
			}
		}

		Expect(totalMissing).To(Equal(0), "Some log lines are still missing")
	})
})

func getMissingLinesCountGcl(f *framework.Framework, podName string, expectedCount int) (int, error) {
	gclFilter := fmt.Sprintf("resource.labels.pod_id:%s AND resource.labels.namespace_id:%s", podName, f.Namespace.Name)
	entries, err := readFilteredEntriesFromGcl(gclFilter)
	if err != nil {
		return 0, err
	}

	occurrences := make(map[int]int)
	for _, entry := range entries {
		lineNumber, err := strconv.Atoi(strings.TrimSpace(entry))
		if err != nil {
			continue
		}
		if lineNumber < 0 || lineNumber >= expectedCount {
			framework.Logf("Unexpected line number: %d", lineNumber)
		} else {
			// Duplicates are possible and fine, fluentd has at-least-once delivery
			occurrences[lineNumber]++
		}
	}

	return expectedCount - len(occurrences), nil
}

type LogEntry struct {
	TextPayload string
}

// Since GCL API is not easily available from the outside of cluster
// we use gcloud command to perform search with filter
func readFilteredEntriesFromGcl(filter string) ([]string, error) {
	framework.Logf("Reading entries from GCL with filter '%v'", filter)
	argList := []string{"beta",
		"logging",
		"read",
		filter,
		"--format",
		"json",
		"--project",
		framework.TestContext.CloudConfig.ProjectID,
	}
	output, err := exec.Command("gcloud", argList...).CombinedOutput()
	if err != nil {
		return nil, err
	}

	var entries []*LogEntry
	if err = json.Unmarshal(output, &entries); err != nil {
		return nil, err
	}
	framework.Logf("Read %d entries from GCL", len(entries))

	var result []string
	for _, entry := range entries {
		if entry.TextPayload != "" {
			result = append(result, entry.TextPayload)
		}
	}

	return result, nil
}
