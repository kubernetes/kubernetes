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
		framework.SkipUnlessProviderIs("gce")
	})

	It("should check that logs from pods on all nodes are ingested into GCL", func() {
		ClusterLevelLoggingWithGcl(f)
	})
})

type SynthLoggerEntry struct {
	NodeNo  int
	EntryNo int
}

type LogEntry struct {
	TextPayload string
}

const (
	testLogName             = "test-log"
	testLogIngestionTimeout = 3 * time.Minute
)

// ClusterLevelLoggingWithGcl is an end to end test for cluster level logging.
func ClusterLevelLoggingWithGcl(f *framework.Framework) {
	By("Checking GCL API readiness")
	// Write an event to the GCL API to ensure it is working
	payload := "test_message_" + f.Namespace.Name
	err := writeEventsToGcl(testLogName, payload)
	Expect(err).NotTo(HaveOccurred(), "Failed to write event to GCL API")

	// Read an event written earlier to ensure reading from GCL API also works
	var entries []*LogEntry
	for start := time.Now(); len(entries) == 0 && time.Since(start) < testLogIngestionTimeout; time.Sleep(5 * time.Second) {
		entries, err = readFilteredEntriesFromGcl(payload)
		Expect(err).NotTo(HaveOccurred(), "Failed to read filtered event from GCL API")
	}
	Expect(entries).To(HaveLen(1))

	// Wait for the Fluentd pods to enter the running state.
	By("Checking to make sure the Fluentd pod are running on each healthy node")
	// Obtain a list of healthy nodes so we can place one synthetic logger on each node.
	nodes := getHealthyNodes(f)
	fluentdPods, err := getFluentdPods(f)
	Expect(err).NotTo(HaveOccurred(), "Failed to obtain fluentd pods")
	err = waitForFluentdPods(f, nodes, fluentdPods)
	Expect(err).NotTo(HaveOccurred(), "Failed to wait for fluentd pods entering running state")

	By("Creating dummy loggers")
	taintName, podNames, err := createSynthLoggers(f, nodes)
	Expect(err).NotTo(HaveOccurred(), "Failed to create dummy loggers")

	// Cleanup the pods when we are done.
	defer cleanupLoggingPods(f, podNames)

	// Wait for the synthetic logging pods to finish.
	By("Waiting for the pods to succeed.")
	err = waitForPodsToSucceed(f, podNames)
	Expect(err).NotTo(HaveOccurred())

	// Make several attempts to observe the logs ingested into GCL
	By("Checking all the log lines were ingested into GCL")
	totalMissing, missingPerNode := waitForLogsToIngest(len(nodes.Items), taintName)

	for n := range missingPerNode {
		if missingPerNode[n] > 0 {
			framework.Logf("Node %d %s is missing %d logs", n, nodes.Items[n].Name, missingPerNode[n])
			opts := &api.PodLogOptions{}
			body, err := f.Client.Pods(f.Namespace.Name).GetLogs(podNames[n], opts).DoRaw()
			if err != nil {
				framework.Logf("Cannot get logs from pod %v", podNames[n])
				continue
			}
			framework.Logf("Pod %s has the following logs: %s", podNames[n], body)

			for _, pod := range fluentdPods.Items {
				if pod.Spec.NodeName == nodes.Items[n].Name {
					body, err = f.Client.Pods(api.NamespaceSystem).GetLogs(pod.Name, opts).DoRaw()
					if err != nil {
						framework.Logf("Cannot get logs from pod %v", pod.Name)
						break
					}
					framework.Logf("Fluentd Pod %s on node %s has the following logs: %s", pod.Name, nodes.Items[n].Name, body)
					break
				}
			}
		}
	}

	if totalMissing != 0 {
		framework.Failf("Failed to find all %d log lines", len(nodes.Items)*countTo)
	}
}

func waitForLogsToIngest(nodeCount int, taintName string) (totalMissing int, missingPerNode []int) {
	missingPerNode = make([]int, nodeCount)
	for i := range missingPerNode {
		missingPerNode[i] = countTo
	}

	gclFilter := fmt.Sprintf("textPayload:%s", taintName)
	for start := time.Now(); time.Since(start) < ingestionTimeout; time.Sleep(25 * time.Second) {
		entries, err := readFilteredEntriesFromGcl(gclFilter)
		if err != nil {
			framework.Logf("Failed to read events from gcl after %v due to %v", time.Since(start), err)
			continue
		}

		observed := make(map[int]map[int]int)
		for n := 0; n < nodeCount; n++ {
			observed[n] = make(map[int]int)
		}

		for _, entry := range entries {
			synthLoggerEntry, err := parseSynthLoggerEntry(entry.TextPayload)
			if err != nil {
				framework.Logf("Failed to parse log entry \"%s\" due to %v", entry.TextPayload, err)
				continue
			}

			observedCounts, ok := observed[synthLoggerEntry.NodeNo]
			if !ok {
				framework.Logf("Unexpected node name: %d", synthLoggerEntry.NodeNo)
				continue
			}

			observedCounts[synthLoggerEntry.EntryNo]++
		}

		var duplicatesFound bool
		totalMissing, missingPerNode, duplicatesFound = analyzeObservedEntries(observed, countTo)

		if totalMissing > 0 {
			framework.Logf("Still missing %d entries", totalMissing)
			continue
		}

		if duplicatesFound {
			framework.Logf("Duplaces found, waiting for deduplication to happen")
			continue
		}

		break
	}

	return
}

func analyzeObservedEntries(observedCounts map[int]map[int]int, expectedEntriesCount int) (totalMissing int, missingPerNode []int, duplicatesFound bool) {
	missingPerNode = make([]int, len(observedCounts))

	for n, counts := range observedCounts {
		for i := 0; i < expectedEntriesCount; i++ {
			count, ok := counts[i]

			if !ok {
				totalMissing++
				missingPerNode[n]++
			}

			if count > 1 {
				duplicatesFound = true
			}
		}
	}

	return
}

func parseSynthLoggerEntry(entry string) (*SynthLoggerEntry, error) {
	chunks := strings.Split(entry, " ")

	if len(chunks) < 3 {
		return nil, fmt.Errorf("\"%s\" is not a correct synth logger entry, it should contain at least 3 space-separated words", entry)
	}

	nodeNo, nodeParseErr := strconv.Atoi(chunks[0])
	if nodeParseErr != nil {
		return nil, fmt.Errorf("Failed to parse node number due to %v", nodeParseErr)
	}

	entryNo, entryParseErr := strconv.Atoi(chunks[2])
	if entryParseErr != nil {
		return nil, fmt.Errorf("Failed to parse entry number due to %v", entryParseErr)
	}

	return &SynthLoggerEntry{NodeNo: nodeNo, EntryNo: entryNo}, nil
}

func writeEventsToGcl(logName string, entry string) error {
	framework.Logf("Writing entry '%s' to log '%s' in GCL", entry, logName)
	argList := []string{
		"beta",
		"logging",
		"write",
		logName,
		entry,
		"--project",
		framework.TestContext.CloudConfig.ProjectID,
	}
	_, err := exec.Command("gcloud", argList...).CombinedOutput()
	return err
}

func readFilteredEntriesFromGcl(filter string) ([]*LogEntry, error) {
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

	var entries []*LogEntry
	for i, obj := range jsonArray {
		jsonObject, ok := obj.(map[string]interface{})

		if !ok {
			// All elements in returned array are expected to be objects
			framework.Logf("Element at position %d is not an object", i)
			continue
		}

		textPayloadObj, ok := jsonObject["textPayload"]
		if !ok {
			// Entry does not contain textPayload field
			// In this test, we don't deal with jsonPayload or structPayload
			framework.Logf("Element at position %d does not contain text payload", i)
			continue
		}

		textPayload, ok := textPayloadObj.(string)
		if !ok {
			// Text payload should be string
			framework.Logf("Element at position %d has non-string textPayload", i)
			continue
		}

		entries = append(entries, &LogEntry{TextPayload: textPayload})
	}

	return entries, nil
}
