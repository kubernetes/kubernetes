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
	"os/exec"
	"strconv"
	"strings"
	"time"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/util/json"
	"k8s.io/kubernetes/test/e2e/framework"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
)

var _ = framework.KubeDescribe("Cluster level logging using GCL", func() {
	f := framework.NewDefaultFramework("gcl-logging")

	BeforeEach(func() {
		// TODO (crassirostris): Expand to GKE once the test is stable
		framework.SkipUnlessProviderIs("gce")
	})

	It("should check that logs from containers are ingested in GCL", func() {
		synthLoggerPodName := f.Namespace.Name + "-synthlogger"

		By("Creating synthetic logger")
		createSynthLogger(f, synthLoggerPodName, expectedLinesCount)
		defer f.PodClient().Delete(synthLoggerPodName, &api.DeleteOptions{})

		By("Waiting for logs to ingest")
		totalMissing := expectedLinesCount
		for start := time.Now(); totalMissing > 0 && time.Since(start) < ingestionTimeout; time.Sleep(25 * time.Second) {
			var err error
			totalMissing, err = getMissingLinesCountGcl(synthLoggerPodName, expectedLinesCount)
			if err != nil {
				framework.Logf("Failed to get missing lines count due to %v", err)
				totalMissing = expectedLinesCount
			} else if totalMissing > 0 {
				framework.Logf("Still missing %d lines", totalMissing)
			}
		}

		Expect(totalMissing).To(Equal(0), "Some log lines are still missing")
	})
})

func getMissingLinesCountGcl(podName string, expectedCount int) (int, error) {
	gclFilter := fmt.Sprintf("resource.labels.pod_id:%s", podName)
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
			// Duplicates are fine, exactly once delivery is OK
			occurrences[lineNumber]++
		}
	}

	return expectedCount - len(occurrences), nil
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

	var jsonArray []interface{}
	if err = json.Unmarshal(output, &jsonArray); err != nil {
		return nil, err
	}
	framework.Logf("Read %d entries from GCL", len(jsonArray))

	var entries []string
	for _, obj := range jsonArray {
		jsonObject, ok := obj.(map[string]interface{})

		if !ok {
			// All elements in returned array are expected to be objects
			continue
		}

		textPayloadObj, ok := jsonObject["textPayload"]
		if !ok {
			// Entry does not contain textPayload field
			// In this test, we don't deal with jsonPayload or structPayload
			continue
		}

		textPayload, ok := textPayloadObj.(string)
		if !ok {
			// Text payload should be string
			continue
		}

		entries = append(entries, textPayload)
	}

	return entries, nil
}
