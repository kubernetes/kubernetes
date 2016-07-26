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
	"encoding/json"
	"fmt"
	"strconv"
	"strings"
	"time"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/labels"
	"k8s.io/kubernetes/test/e2e/framework"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
)

var _ = framework.KubeDescribe("Cluster level logging using Elasticsearch [Feature:Elasticsearch]", func() {
	f := framework.NewDefaultFramework("es-logging")

	BeforeEach(func() {
		// TODO: For now assume we are only testing cluster logging with Elasticsearch
		// on GCE. Once we are sure that Elasticsearch cluster level logging
		// works for other providers we should widen this scope of this test.
		framework.SkipUnlessProviderIs("gce")
	})

	It("should check that logs from pods on all nodes are ingested into Elasticsearch", func() {
		ClusterLevelLoggingWithElasticsearch(f)
	})
})

const (
	k8sAppKey    = "k8s-app"
	esValue      = "elasticsearch-logging"
	fluentdValue = "fluentd-logging"
)

func bodyToJSON(body []byte) (map[string]interface{}, error) {
	var r map[string]interface{}
	if err := json.Unmarshal(body, &r); err != nil {
		framework.Logf("Bad JSON: %s", string(body))
		return nil, fmt.Errorf("failed to unmarshal Elasticsearch response: %v", err)
	}
	return r, nil
}

func nodeInNodeList(nodeName string, nodeList *api.NodeList) bool {
	for _, node := range nodeList.Items {
		if nodeName == node.Name {
			return true
		}
	}
	return false
}

// ClusterLevelLoggingWithElasticsearch is an end to end test for cluster level logging.
func ClusterLevelLoggingWithElasticsearch(f *framework.Framework) {
	// graceTime is how long to keep retrying requests for status information.
	const graceTime = 5 * time.Minute
	// ingestionTimeout is how long to keep retrying to wait for all the
	// logs to be ingested.
	const ingestionTimeout = 10 * time.Minute

	// Check for the existence of the Elasticsearch service.
	By("Checking the Elasticsearch service exists.")
	s := f.Client.Services(api.NamespaceSystem)
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
	label := labels.SelectorFromSet(labels.Set(map[string]string{k8sAppKey: esValue}))
	options := api.ListOptions{LabelSelector: label}
	pods, err := f.Client.Pods(api.NamespaceSystem).List(options)
	Expect(err).NotTo(HaveOccurred())
	for _, pod := range pods.Items {
		err = framework.WaitForPodRunningInNamespace(f.Client, &pod)
		Expect(err).NotTo(HaveOccurred())
	}

	By("Checking to make sure we are talking to an Elasticsearch service.")
	// Perform a few checks to make sure this looks like an Elasticsearch cluster.
	var statusCode float64
	var esResponse map[string]interface{}
	err = nil
	var body []byte
	for start := time.Now(); time.Since(start) < graceTime; time.Sleep(10 * time.Second) {
		proxyRequest, errProxy := framework.GetServicesProxyRequest(f.Client, f.Client.Get())
		if errProxy != nil {
			framework.Logf("After %v failed to get services proxy request: %v", time.Since(start), errProxy)
			continue
		}
		// Query against the root URL for Elasticsearch.
		body, err = proxyRequest.Namespace(api.NamespaceSystem).
			Name("elasticsearch-logging").
			DoRaw()
		if err != nil {
			framework.Logf("After %v proxy call to elasticsearch-loigging failed: %v", time.Since(start), err)
			continue
		}
		esResponse, err = bodyToJSON(body)
		if err != nil {
			framework.Logf("After %v failed to convert Elasticsearch JSON response %v to map[string]interface{}: %v", time.Since(start), string(body), err)
			continue
		}
		statusIntf, ok := esResponse["status"]
		if !ok {
			framework.Logf("After %v Elasticsearch response has no status field: %v", time.Since(start), esResponse)
			continue
		}
		statusCode, ok = statusIntf.(float64)
		if !ok {
			// Assume this is a string returning Failure. Retry.
			framework.Logf("After %v expected status to be a float64 but got %v of type %T", time.Since(start), statusIntf, statusIntf)
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
	// Check to see if have a cluster_name field.
	clusterName, ok := esResponse["cluster_name"]
	if !ok {
		framework.Failf("No cluster_name field in Elasticsearch response: %v", esResponse)
	}
	if clusterName != "kubernetes-logging" {
		framework.Failf("Connected to wrong cluster %q (expecting kubernetes_logging)", clusterName)
	}

	// Now assume we really are talking to an Elasticsearch instance.
	// Check the cluster health.
	By("Checking health of Elasticsearch service.")
	healthy := false
	for start := time.Now(); time.Since(start) < graceTime; time.Sleep(5 * time.Second) {
		proxyRequest, errProxy := framework.GetServicesProxyRequest(f.Client, f.Client.Get())
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
		health, err := bodyToJSON(body)
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
		framework.Failf("After %v elasticsearch cluster is not healthy", graceTime)
	}

	// Obtain a list of nodes so we can place one synthetic logger on each node.
	nodes := framework.GetReadySchedulableNodesOrDie(f.Client)
	nodeCount := len(nodes.Items)
	if nodeCount == 0 {
		framework.Failf("Failed to find any nodes")
	}
	framework.Logf("Found %d nodes.", len(nodes.Items))

	// Filter out unhealthy nodes.
	// Previous tests may have cause failures of some nodes. Let's skip
	// 'Not Ready' nodes, just in case (there is no need to fail the test).
	framework.FilterNodes(nodes, func(node api.Node) bool {
		return framework.IsNodeConditionSetAsExpected(&node, api.NodeReady, true)
	})
	if len(nodes.Items) < 2 {
		framework.Failf("Less than two nodes were found Ready: %d", len(nodes.Items))
	}
	framework.Logf("Found %d healthy nodes.", len(nodes.Items))

	// Wait for the Fluentd pods to enter the running state.
	By("Checking to make sure the Fluentd pod are running on each healthy node")
	label = labels.SelectorFromSet(labels.Set(map[string]string{k8sAppKey: fluentdValue}))
	options = api.ListOptions{LabelSelector: label}
	fluentdPods, err := f.Client.Pods(api.NamespaceSystem).List(options)
	Expect(err).NotTo(HaveOccurred())
	for _, pod := range fluentdPods.Items {
		if nodeInNodeList(pod.Spec.NodeName, nodes) {
			err = framework.WaitForPodRunningInNamespace(f.Client, &pod)
			Expect(err).NotTo(HaveOccurred())
		}
	}

	// Check if each healthy node has fluentd running on it
	for _, node := range nodes.Items {
		exists := false
		for _, pod := range fluentdPods.Items {
			if pod.Spec.NodeName == node.Name {
				exists = true
				break
			}
		}
		if !exists {
			framework.Failf("Node %v does not have fluentd pod running on it.", node.Name)
		}
	}

	// Create a unique root name for the resources in this test to permit
	// parallel executions of this test.
	// Use a unique namespace for the resources created in this test.
	ns := f.Namespace.Name
	name := "synthlogger"
	// Form a unique name to taint log lines to be collected.
	// Replace '-' characters with '_' to prevent the analyzer from breaking apart names.
	taintName := strings.Replace(ns+name, "-", "_", -1)
	framework.Logf("Tainting log lines with %v", taintName)
	// podNames records the names of the synthetic logging pods that are created in the
	// loop below.
	var podNames []string
	// countTo is the number of log lines emitted (and checked) for each synthetic logging pod.
	const countTo = 100
	// Instantiate a synthetic logger pod on each node.
	for i, node := range nodes.Items {
		podName := fmt.Sprintf("%s-%d", name, i)
		_, err := f.Client.Pods(ns).Create(&api.Pod{
			ObjectMeta: api.ObjectMeta{
				Name:   podName,
				Labels: map[string]string{"name": name},
			},
			Spec: api.PodSpec{
				Containers: []api.Container{
					{
						Name:  "synth-logger",
						Image: "gcr.io/google_containers/ubuntu:14.04",
						// notice: the subshell syntax is escaped with `$$`
						Command: []string{"bash", "-c", fmt.Sprintf("i=0; while ((i < %d)); do echo \"%d %s $i %s\"; i=$$(($i+1)); done", countTo, i, taintName, podName)},
					},
				},
				NodeName:      node.Name,
				RestartPolicy: api.RestartPolicyNever,
			},
		})
		Expect(err).NotTo(HaveOccurred())
		podNames = append(podNames, podName)
	}

	// Cleanup the pods when we are done.
	defer func() {
		for _, pod := range podNames {
			if err = f.Client.Pods(ns).Delete(pod, nil); err != nil {
				framework.Logf("Failed to delete pod %s: %v", pod, err)
			}
		}
	}()

	// Wait for the synthetic logging pods to finish.
	By("Waiting for the pods to succeed.")
	for _, pod := range podNames {
		err = framework.WaitForPodSuccessInNamespace(f.Client, pod, "synth-logger", ns)
		Expect(err).NotTo(HaveOccurred())
	}

	// Wait a bit for the log information to make it into Elasticsearch.
	time.Sleep(30 * time.Second)

	// Make several attempts to observe the logs ingested into Elasticsearch.
	By("Checking all the log lines were ingested into Elasticsearch")
	totalMissing := 0
	expected := nodeCount * countTo
	missingPerNode := []int{}
	for start := time.Now(); time.Since(start) < ingestionTimeout; time.Sleep(25 * time.Second) {

		// Debugging code to report the status of the elasticsearch logging endpoints.
		selector := labels.Set{k8sAppKey: esValue}.AsSelector()
		options := api.ListOptions{LabelSelector: selector}
		esPods, err := f.Client.Pods(api.NamespaceSystem).List(options)
		if err != nil {
			framework.Logf("Attempt to list Elasticsearch nodes encountered a problem -- may retry: %v", err)
			continue
		} else {
			for i, pod := range esPods.Items {
				framework.Logf("pod %d: %s PodIP %s phase %s condition %+v", i, pod.Name, pod.Status.PodIP, pod.Status.Phase,
					pod.Status.Conditions)
			}
		}

		proxyRequest, errProxy := framework.GetServicesProxyRequest(f.Client, f.Client.Get())
		if errProxy != nil {
			framework.Logf("After %v failed to get services proxy request: %v", time.Since(start), errProxy)
			continue
		}
		// Ask Elasticsearch to return all the log lines that were tagged with the underscore
		// version of the name. Ask for twice as many log lines as we expect to check for
		// duplication bugs.
		body, err = proxyRequest.Namespace(api.NamespaceSystem).
			Name("elasticsearch-logging").
			Suffix("_search").
			Param("q", fmt.Sprintf("log:%s", taintName)).
			Param("size", strconv.Itoa(2*expected)).
			DoRaw()
		if err != nil {
			framework.Logf("After %v failed to make proxy call to elasticsearch-logging: %v", time.Since(start), err)
			continue
		}

		response, err := bodyToJSON(body)
		if err != nil {
			framework.Logf("After %v failed to unmarshal response: %v", time.Since(start), err)
			framework.Logf("Body: %s", string(body))
			continue
		}
		hits, ok := response["hits"].(map[string]interface{})
		if !ok {
			framework.Logf("response[hits] not of the expected type: %T", response["hits"])
			continue
		}
		totalF, ok := hits["total"].(float64)
		if !ok {
			framework.Logf("After %v hits[total] not of the expected type: %T", time.Since(start), hits["total"])
			continue
		}
		total := int(totalF)
		if total != expected {
			framework.Logf("After %v expecting to find %d log lines but saw %d", time.Since(start), expected, total)
		}
		h, ok := hits["hits"].([]interface{})
		if !ok {
			framework.Logf("After %v hits not of the expected type: %T", time.Since(start), hits["hits"])
			continue
		}
		// Initialize data-structure for observing counts.
		observed := make([][]int, nodeCount)
		for i := range observed {
			observed[i] = make([]int, countTo)
		}
		// Iterate over the hits and populate the observed array.
		for _, e := range h {
			l, ok := e.(map[string]interface{})
			if !ok {
				framework.Logf("element of hit not of expected type: %T", e)
				continue
			}
			source, ok := l["_source"].(map[string]interface{})
			if !ok {
				framework.Logf("_source not of the expected type: %T", l["_source"])
				continue
			}
			msg, ok := source["log"].(string)
			if !ok {
				framework.Logf("log not of the expected type: %T", source["log"])
				continue
			}
			words := strings.Split(msg, " ")
			if len(words) != 4 {
				framework.Logf("Malformed log line: %s", msg)
				continue
			}
			n, err := strconv.ParseUint(words[0], 10, 0)
			if err != nil {
				framework.Logf("Expecting numer of node as first field of %s", msg)
				continue
			}
			if n < 0 || int(n) >= nodeCount {
				framework.Logf("Node count index out of range: %d", nodeCount)
				continue
			}
			index, err := strconv.ParseUint(words[2], 10, 0)
			if err != nil {
				framework.Logf("Expecting number as third field of %s", msg)
				continue
			}
			if index < 0 || index >= countTo {
				framework.Logf("Index value out of range: %d", index)
				continue
			}
			if words[1] != taintName {
				framework.Logf("Elasticsearch query return unexpected log line: %s", msg)
				continue
			}
			// Record the observation of a log line from node n at the given index.
			observed[n][index]++
		}
		// Make sure we correctly observed the expected log lines from each node.
		totalMissing = 0
		missingPerNode = make([]int, nodeCount)
		incorrectCount := false
		for n := range observed {
			for i, c := range observed[n] {
				if c == 0 {
					totalMissing++
					missingPerNode[n]++
				}
				if c < 0 || c > 1 {
					framework.Logf("Got incorrect count for node %d index %d: %d", n, i, c)
					incorrectCount = true
				}
			}
		}
		if incorrectCount {
			framework.Logf("After %v es still return duplicated log lines", time.Since(start))
			continue
		}
		if totalMissing != 0 {
			framework.Logf("After %v still missing %d log lines", time.Since(start), totalMissing)
			continue
		}
		framework.Logf("After %s found all %d log lines", time.Since(start), expected)
		return
	}
	for n := range missingPerNode {
		if missingPerNode[n] > 0 {
			framework.Logf("Node %d %s is missing %d logs", n, nodes.Items[n].Name, missingPerNode[n])
			opts := &api.PodLogOptions{}
			body, err = f.Client.Pods(ns).GetLogs(podNames[n], opts).DoRaw()
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
	framework.Failf("Failed to find all %d log lines", expected)
}
