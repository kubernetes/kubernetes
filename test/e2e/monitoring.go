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
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"time"

	influxdb "github.com/influxdata/influxdb/client/v2"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/kubernetes/pkg/client/clientset_generated/clientset"
	"k8s.io/kubernetes/test/e2e/framework"

	. "github.com/onsi/ginkgo"
)

var _ = framework.KubeDescribe("Monitoring", func() {
	f := framework.NewDefaultFramework("monitoring")

	BeforeEach(func() {
		framework.SkipUnlessProviderIs("gce")
	})

	It("should verify monitoring pods and all cluster nodes are available on influxdb using heapster.", func() {
		testMonitoringUsingHeapsterInfluxdb(f.ClientSet)
	})
})

const (
	influxdbService       = "monitoring-influxdb"
	influxdbDatabaseName  = "k8s"
	podlistQuery          = "show tag values from \"cpu/usage\" with key = pod_name"
	nodelistQuery         = "show tag values from \"cpu/usage\" with key = nodename"
	sleepBetweenAttempts  = 5 * time.Second
	testTimeout           = 5 * time.Minute
	initializationTimeout = 5 * time.Minute
)

var (
	rcLabels         = []string{"heapster", "influxGrafana"}
	expectedServices = map[string]bool{
		influxdbService:      false,
		"monitoring-grafana": false,
	}
)

// Query sends a command to the server and returns the Response
func Query(c clientset.Interface, query string) (*influxdb.Response, error) {
	subResourceProxyAvailable, err := framework.ServerVersionGTE(framework.SubResourceServiceAndNodeProxyVersion, c.Discovery())
	if err != nil {
		return nil, err
	}

	ctx, cancel := context.WithTimeout(context.Background(), framework.SingleCallTimeout)
	defer cancel()

	var result []byte
	if subResourceProxyAvailable {
		result, err = c.Core().RESTClient().Get().
			Context(ctx).
			Namespace("kube-system").
			Resource("services").
			Name(influxdbService+":api").
			SubResource("proxy").
			Suffix("query").
			Param("q", query).
			Param("db", influxdbDatabaseName).
			Param("epoch", "s").
			Do().
			Raw()
	} else {
		result, err = c.Core().RESTClient().Get().
			Context(ctx).
			Prefix("proxy").
			Namespace("kube-system").
			Resource("services").
			Name(influxdbService+":api").
			Suffix("query").
			Param("q", query).
			Param("db", influxdbDatabaseName).
			Param("epoch", "s").
			Do().
			Raw()
	}

	if err != nil {
		if ctx.Err() != nil {
			framework.Failf("Failed to query influx db: %v", err)
		}
		return nil, err
	}

	var response influxdb.Response
	dec := json.NewDecoder(bytes.NewReader(result))
	dec.UseNumber()
	err = dec.Decode(&response)

	if err != nil {
		return nil, err
	}
	return &response, nil
}

func verifyExpectedRcsExistAndGetExpectedPods(c clientset.Interface) ([]string, error) {
	expectedPods := []string{}
	// Iterate over the labels that identify the replication controllers that we
	// want to check. The rcLabels contains the value values for the k8s-app key
	// that identify the replication controllers that we want to check. Using a label
	// rather than an explicit name is preferred because the names will typically have
	// a version suffix e.g. heapster-monitoring-v1 and this will change after a rolling
	// update e.g. to heapster-monitoring-v2. By using a label query we can check for the
	// situation when a heapster-monitoring-v1 and heapster-monitoring-v2 replication controller
	// is running (which would be an error except during a rolling update).
	for _, rcLabel := range rcLabels {
		selector := labels.Set{"k8s-app": rcLabel}.AsSelector()
		options := metav1.ListOptions{LabelSelector: selector.String()}
		deploymentList, err := c.Extensions().Deployments(metav1.NamespaceSystem).List(options)
		if err != nil {
			return nil, err
		}
		rcList, err := c.Core().ReplicationControllers(metav1.NamespaceSystem).List(options)
		if err != nil {
			return nil, err
		}
		psList, err := c.Apps().StatefulSets(metav1.NamespaceSystem).List(options)
		if err != nil {
			return nil, err
		}
		if (len(rcList.Items) + len(deploymentList.Items) + len(psList.Items)) != 1 {
			return nil, fmt.Errorf("expected to find one replica for RC or deployment with label %s but got %d",
				rcLabel, len(rcList.Items))
		}
		// Check all the replication controllers.
		for _, rc := range rcList.Items {
			selector := labels.Set(rc.Spec.Selector).AsSelector()
			options := metav1.ListOptions{LabelSelector: selector.String()}
			podList, err := c.Core().Pods(metav1.NamespaceSystem).List(options)
			if err != nil {
				return nil, err
			}
			for _, pod := range podList.Items {
				if pod.DeletionTimestamp != nil {
					continue
				}
				expectedPods = append(expectedPods, pod.Name)
			}
		}
		// Do the same for all deployments.
		for _, rc := range deploymentList.Items {
			selector := labels.Set(rc.Spec.Selector.MatchLabels).AsSelector()
			options := metav1.ListOptions{LabelSelector: selector.String()}
			podList, err := c.Core().Pods(metav1.NamespaceSystem).List(options)
			if err != nil {
				return nil, err
			}
			for _, pod := range podList.Items {
				if pod.DeletionTimestamp != nil {
					continue
				}
				expectedPods = append(expectedPods, pod.Name)
			}
		}
		// And for pet sets.
		for _, ps := range psList.Items {
			selector := labels.Set(ps.Spec.Selector.MatchLabels).AsSelector()
			options := metav1.ListOptions{LabelSelector: selector.String()}
			podList, err := c.Core().Pods(metav1.NamespaceSystem).List(options)
			if err != nil {
				return nil, err
			}
			for _, pod := range podList.Items {
				if pod.DeletionTimestamp != nil {
					continue
				}
				expectedPods = append(expectedPods, pod.Name)
			}
		}
	}
	return expectedPods, nil
}

func expectedServicesExist(c clientset.Interface) error {
	serviceList, err := c.Core().Services(metav1.NamespaceSystem).List(metav1.ListOptions{})
	if err != nil {
		return err
	}
	for _, service := range serviceList.Items {
		if _, ok := expectedServices[service.Name]; ok {
			expectedServices[service.Name] = true
		}
	}
	for service, found := range expectedServices {
		if !found {
			return fmt.Errorf("Service %q not found", service)
		}
	}
	return nil
}

func getAllNodesInCluster(c clientset.Interface) ([]string, error) {
	// It should be OK to list unschedulable Nodes here.
	nodeList, err := c.Core().Nodes().List(metav1.ListOptions{})
	if err != nil {
		return nil, err
	}
	result := []string{}
	for _, node := range nodeList.Items {
		result = append(result, node.Name)
	}
	return result, nil
}

func getInfluxdbData(c clientset.Interface, query string, tag string) (map[string]bool, error) {
	response, err := Query(c, query)
	if err != nil {
		return nil, err
	}
	if len(response.Results) != 1 {
		return nil, fmt.Errorf("expected only one result from Influxdb for query %q. Got %+v", query, response)
	}
	if len(response.Results[0].Series) != 1 {
		return nil, fmt.Errorf("expected exactly one series for query %q.", query)
	}
	if len(response.Results[0].Series[0].Columns) != 2 {
		framework.Failf("Expected two columns for query %q. Found %v", query, response.Results[0].Series[0].Columns)
	}
	result := map[string]bool{}
	for _, value := range response.Results[0].Series[0].Values {
		name := value[1].(string)
		result[name] = true
	}
	return result, nil
}

func expectedItemsExist(expectedItems []string, actualItems map[string]bool) bool {
	if len(actualItems) < len(expectedItems) {
		return false
	}
	for _, item := range expectedItems {
		if _, found := actualItems[item]; !found {
			return false
		}
	}
	return true
}

func validatePodsAndNodes(c clientset.Interface, expectedPods, expectedNodes []string) bool {
	pods, err := getInfluxdbData(c, podlistQuery, "pod_id")
	if err != nil {
		// We don't fail the test here because the influxdb service might still not be running.
		framework.Logf("failed to query list of pods from influxdb. Query: %q, Err: %v", podlistQuery, err)
		return false
	}
	nodes, err := getInfluxdbData(c, nodelistQuery, "hostname")
	if err != nil {
		framework.Logf("failed to query list of nodes from influxdb. Query: %q, Err: %v", nodelistQuery, err)
		return false
	}
	if !expectedItemsExist(expectedPods, pods) {
		framework.Logf("failed to find all expected Pods.\nExpected: %v\nActual: %v", expectedPods, pods)
		return false
	}
	if !expectedItemsExist(expectedNodes, nodes) {
		framework.Logf("failed to find all expected Nodes.\nExpected: %v\nActual: %v", expectedNodes, nodes)
		return false
	}
	return true
}

func testMonitoringUsingHeapsterInfluxdb(c clientset.Interface) {
	// Check if heapster pods and services are up.
	var expectedPods []string
	rcErr := fmt.Errorf("failed to verify expected RCs within timeout")
	serviceErr := fmt.Errorf("failed to verify expected services within timeout")
	err := wait.PollImmediate(sleepBetweenAttempts, initializationTimeout, func() (bool, error) {
		expectedPods, rcErr = verifyExpectedRcsExistAndGetExpectedPods(c)
		if rcErr != nil {
			framework.Logf("Waiting for expected RCs (got error: %v)", rcErr)
			return false, nil
		}
		serviceErr = expectedServicesExist(c)
		if serviceErr != nil {
			framework.Logf("Waiting for expected services (got error: %v)", serviceErr)
			return false, nil
		}
		return true, nil
	})
	if err != nil {
		framework.ExpectNoError(rcErr)
		framework.ExpectNoError(serviceErr)
		framework.Failf("Failed to verify RCs and services within timeout: %v", err)
	}

	expectedNodes, err := getAllNodesInCluster(c)
	framework.ExpectNoError(err)
	startTime := time.Now()
	for {
		if validatePodsAndNodes(c, expectedPods, expectedNodes) {
			return
		}
		if time.Since(startTime) >= testTimeout {
			// temporary workaround to help debug issue #12765
			printDebugInfo(c)
			break
		}
		time.Sleep(sleepBetweenAttempts)
	}
	framework.Failf("monitoring using heapster and influxdb test failed")
}

func printDebugInfo(c clientset.Interface) {
	set := labels.Set{"k8s-app": "heapster"}
	options := metav1.ListOptions{LabelSelector: set.AsSelector().String()}
	podList, err := c.Core().Pods(metav1.NamespaceSystem).List(options)
	if err != nil {
		framework.Logf("Error while listing pods %v", err)
		return
	}
	for _, pod := range podList.Items {
		framework.Logf("Kubectl output:\n%v",
			framework.RunKubectlOrDie("log", pod.Name, "--namespace=kube-system", "--container=heapster"))
	}
}
