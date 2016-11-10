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
	"encoding/json"
	"fmt"
	"strconv"
	"strings"
	"time"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/labels"
	"k8s.io/kubernetes/test/e2e/framework"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
)

const (
	// graceTime is how long to keep retrying requesting elasticsearch for status information.
	graceTime = 5 * time.Minute
)

var _ = framework.KubeDescribe("Cluster level logging using Elasticsearch [Feature:Elasticsearch]", func() {
	f := framework.NewDefaultFramework("es-logging")

	BeforeEach(func() {
		// TODO: For now assume we are only testing cluster logging with Elasticsearch
		// on GCE. Once we are sure that Elasticsearch cluster level logging
		// works for other providers we should widen this scope of this test.
		framework.SkipUnlessProviderIs("gce")
	})

	It("should check that logs from containers are ingested into Elasticsearch", func() {
		err := checkElasticsearchReadiness(f)
		framework.ExpectNoError(err, "Elasticsearch failed to start")

		By("Running synthetic logger")
		createSynthLogger(f, expectedLinesCount)
		defer f.PodClient().Delete(synthLoggerPodName, &v1.DeleteOptions{})
		err = framework.WaitForPodSuccessInNamespace(f.ClientSet, synthLoggerPodName, f.Namespace.Name)
		framework.ExpectNoError(err, fmt.Sprintf("Should've successfully waited for pod %s to succeed", synthLoggerPodName))

		By("Waiting for logs to ingest")
		totalMissing := expectedLinesCount
		for start := time.Now(); time.Since(start) < ingestionTimeout; time.Sleep(ingestionRetryDelay) {
			totalMissing, err = getMissingLinesCountElasticsearch(f, expectedLinesCount)
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

// Ensures that elasticsearch is running and ready to serve requests
func checkElasticsearchReadiness(f *framework.Framework) error {
	// Check for the existence of the Elasticsearch service.
	By("Checking the Elasticsearch service exists.")
	s := f.ClientSet.Core().Services(api.NamespaceSystem)
	// Make a few attempts to connect. This makes the test robust against
	// being run as the first e2e test just after the e2e cluster has been created.
	var err error
	for start := time.Now(); time.Since(start) < graceTime; time.Sleep(5 * time.Second) {
		if _, err = s.Get("elasticsearch-logging"); err == nil {
			break
		}
		framework.Logf("Attempt to check for the existence of the Elasticsearch service failed after %v", time.Since(start))
	}
	Expect(err).NotTo(HaveOccurred())

	// Wait for the Elasticsearch pods to enter the running state.
	By("Checking to make sure the Elasticsearch pods are running")
	label := labels.SelectorFromSet(labels.Set(map[string]string{"k8s-app": "elasticsearch-logging"}))
	options := v1.ListOptions{LabelSelector: label.String()}
	pods, err := f.ClientSet.Core().Pods(api.NamespaceSystem).List(options)
	Expect(err).NotTo(HaveOccurred())
	for _, pod := range pods.Items {
		err = framework.WaitForPodRunningInNamespace(f.ClientSet, &pod)
		Expect(err).NotTo(HaveOccurred())
	}

	By("Checking to make sure we are talking to an Elasticsearch service.")
	// Perform a few checks to make sure this looks like an Elasticsearch cluster.
	var statusCode int
	err = nil
	var body []byte
	for start := time.Now(); time.Since(start) < graceTime; time.Sleep(10 * time.Second) {
		proxyRequest, errProxy := framework.GetServicesProxyRequest(f.ClientSet, f.ClientSet.Core().RESTClient().Get())
		if errProxy != nil {
			framework.Logf("After %v failed to get services proxy request: %v", time.Since(start), errProxy)
			continue
		}
		// Query against the root URL for Elasticsearch.
		response := proxyRequest.Namespace(api.NamespaceSystem).
			Name("elasticsearch-logging").
			Do()
		err = response.Error()
		response.StatusCode(&statusCode)

		if err != nil {
			framework.Logf("After %v proxy call to elasticsearch-loigging failed: %v", time.Since(start), err)
			continue
		}
		if int(statusCode) != 200 {
			framework.Logf("After %v Elasticsearch cluster has a bad status: %v", time.Since(start), statusCode)
			continue
		}
		break
	}
	Expect(err).NotTo(HaveOccurred())
	if int(statusCode) != 200 {
		framework.Failf("Elasticsearch cluster has a bad status: %v", statusCode)
	}

	// Now assume we really are talking to an Elasticsearch instance.
	// Check the cluster health.
	By("Checking health of Elasticsearch service.")
	healthy := false
	for start := time.Now(); time.Since(start) < graceTime; time.Sleep(5 * time.Second) {
		proxyRequest, errProxy := framework.GetServicesProxyRequest(f.ClientSet, f.ClientSet.Core().RESTClient().Get())
		if errProxy != nil {
			framework.Logf("After %v failed to get services proxy request: %v", time.Since(start), errProxy)
			continue
		}
		body, err = proxyRequest.Namespace(api.NamespaceSystem).
			Name("elasticsearch-logging").
			Suffix("_cluster/health").
			Param("level", "indices").
			DoRaw()
		if err != nil {
			continue
		}
		health := make(map[string]interface{})
		err := json.Unmarshal(body, &health)
		if err != nil {
			framework.Logf("Bad json response from elasticsearch: %v", err)
			continue
		}
		statusIntf, ok := health["status"]
		if !ok {
			framework.Logf("No status field found in cluster health response: %v", health)
			continue
		}
		status := statusIntf.(string)
		if status != "green" && status != "yellow" {
			framework.Logf("Cluster health has bad status: %v", health)
			continue
		}
		if err == nil && ok {
			healthy = true
			break
		}
	}
	if !healthy {
		return fmt.Errorf("After %v elasticsearch cluster is not healthy", graceTime)
	}

	return nil
}

func getMissingLinesCountElasticsearch(f *framework.Framework, expectedCount int) (int, error) {
	proxyRequest, errProxy := framework.GetServicesProxyRequest(f.ClientSet, f.ClientSet.Core().RESTClient().Get())
	if errProxy != nil {
		return 0, fmt.Errorf("Failed to get services proxy request: %v", errProxy)
	}

	// Ask Elasticsearch to return all the log lines that were tagged with the
	// pod name. Ask for ten times as many log lines because duplication is possible.
	body, err := proxyRequest.Namespace(api.NamespaceSystem).
		Name("elasticsearch-logging").
		Suffix("_search").
		// TODO: Change filter to only match records from current test run
		// after fluent-plugin-kubernetes_metadata_filter is enabled
		// and optimize current query
		Param("q", fmt.Sprintf("tag:*%s*", synthLoggerPodName)).
		Param("size", strconv.Itoa(expectedCount*10)).
		DoRaw()
	if err != nil {
		return 0, fmt.Errorf("Failed to make proxy call to elasticsearch-logging: %v", err)
	}

	var response map[string]interface{}
	err = json.Unmarshal(body, &response)
	if err != nil {
		return 0, fmt.Errorf("Failed to unmarshal response: %v", err)
	}

	hits, ok := response["hits"].(map[string]interface{})
	if !ok {
		return 0, fmt.Errorf("response[hits] not of the expected type: %T", response["hits"])
	}

	h, ok := hits["hits"].([]interface{})
	if !ok {
		return 0, fmt.Errorf("Hits not of the expected type: %T", hits["hits"])
	}

	// Initialize data-structure for observing counts.
	counts := make(map[int]int)

	// Iterate over the hits and populate the observed array.
	for _, e := range h {
		l, ok := e.(map[string]interface{})
		if !ok {
			framework.Logf("Element of hit not of expected type: %T", e)
			continue
		}
		source, ok := l["_source"].(map[string]interface{})
		if !ok {
			framework.Logf("_source not of the expected type: %T", l["_source"])
			continue
		}
		msg, ok := source["log"].(string)
		if !ok {
			framework.Logf("Log not of the expected type: %T", source["log"])
			continue
		}
		lineNumber, err := strconv.Atoi(strings.TrimSpace(msg))
		if err != nil {
			framework.Logf("Log line %s is not a number", msg)
			continue
		}
		if lineNumber < 0 || lineNumber >= expectedCount {
			framework.Logf("Number %d is not valid, expected number from range [0, %d)", lineNumber, expectedCount)
			continue
		}
		// Record the observation of a log line
		// Duplicates are possible and fine, fluentd has at-least-once delivery
		counts[lineNumber]++
	}

	return expectedCount - len(counts), nil
}
