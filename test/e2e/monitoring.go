/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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
	"net/http"
	"net/url"
	"time"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/fields"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/labels"
	influxdb "github.com/influxdb/influxdb/client"

	. "github.com/onsi/ginkgo"
)

// TODO: quinton: debug issue #6541 and then remove Pending flag here.
var _ = Describe("Monitoring", func() {
	var c *client.Client

	BeforeEach(func() {
		var err error
		c, err = loadClient()
		expectNoError(err)
	})

	It("verify monitoring pods and all cluster nodes are available on influxdb using heapster.", func() {
		if !providerIs("gce") {
			By(fmt.Sprintf("Skipping Monitoring test, which is only supported for provider gce (not %s)",
				testContext.Provider))
			return
		}
		testMonitoringUsingHeapsterInfluxdb(c)
	})
})

const (
	influxdbService      = "monitoring-influxdb"
	influxdbDatabaseName = "k8s"
	influxdbUser         = "root"
	influxdbPW           = "root"
	podlistQuery         = "select distinct(pod_id) from /cpu.*/"
	nodelistQuery        = "select distinct(hostname) from /cpu.*/"
	sleepBetweenAttempts = 5 * time.Second
	testTimeout          = 5 * time.Minute
)

var (
	expectedRcs = map[string]bool{
		"monitoring-heapster-controller":       false,
		"monitoring-influx-grafana-controller": false,
	}
	expectedServices = map[string]bool{
		influxdbService:      false,
		"monitoring-grafana": false,
	}
)

func verifyExpectedRcsExistAndGetExpectedPods(c *client.Client) ([]string, error) {
	rcList, err := c.ReplicationControllers(api.NamespaceDefault).List(labels.Everything())
	if err != nil {
		return nil, err
	}
	expectedPods := []string{}
	for _, rc := range rcList.Items {
		if _, ok := expectedRcs[rc.Name]; ok {
			if rc.Status.Replicas != 1 {
				return nil, fmt.Errorf("expected to find only one replica for rc %q, found %d", rc.Name, rc.Status.Replicas)
			}
			expectedRcs[rc.Name] = true
			podList, err := c.Pods(api.NamespaceDefault).List(labels.Set(rc.Spec.Selector).AsSelector(), fields.Everything())
			if err != nil {
				return nil, err
			}
			for _, pod := range podList.Items {
				expectedPods = append(expectedPods, string(pod.UID))
			}
		}
	}
	for rc, found := range expectedRcs {
		if !found {
			return nil, fmt.Errorf("Replication Controller %q not found.", rc)
		}
	}
	return expectedPods, nil
}

func expectedServicesExist(c *client.Client) error {
	serviceList, err := c.Services(api.NamespaceDefault).List(labels.Everything())
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

func getAllNodesInCluster(c *client.Client) ([]string, error) {
	nodeList, err := c.Nodes().List(labels.Everything(), fields.Everything())
	if err != nil {
		return nil, err
	}
	result := []string{}
	for _, node := range nodeList.Items {
		result = append(result, node.Name)
	}
	return result, nil
}

func getInfluxdbData(c *influxdb.Client, query string) (map[string]bool, error) {
	series, err := c.Query(query, influxdb.Second)
	if err != nil {
		return nil, err
	}
	if len(series) != 1 {
		return nil, fmt.Errorf("expected only one series from Influxdb for query %q. Got %+v", query, series)
	}
	if len(series[0].GetColumns()) != 2 {
		Failf("Expected two columns for query %q. Found %v", query, series[0].GetColumns())
	}
	result := map[string]bool{}
	for _, point := range series[0].GetPoints() {
		if len(point) != 2 {
			Failf("Expected only two entries in a point for query %q. Got %v", query, point)
		}
		name, ok := point[1].(string)
		if !ok {
			Failf("expected %v to be a string, but it is %T", point[1], point[1])
		}
		result[name] = false
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

func validatePodsAndNodes(influxdbClient *influxdb.Client, expectedPods, expectedNodes []string) bool {
	pods, err := getInfluxdbData(influxdbClient, podlistQuery)
	if err != nil {
		// We don't fail the test here because the influxdb service might still not be running.
		Logf("failed to query list of pods from influxdb. Query: %q, Err: %v", podlistQuery, err)
		return false
	}
	nodes, err := getInfluxdbData(influxdbClient, nodelistQuery)
	if err != nil {
		Logf("failed to query list of nodes from influxdb. Query: %q, Err: %v", nodelistQuery, err)
		return false
	}
	if !expectedItemsExist(expectedPods, pods) {
		Logf("failed to find all expected Pods.\nExpected: %v\nActual: %v", expectedPods, pods)
		return false
	}
	if !expectedItemsExist(expectedNodes, nodes) {
		Logf("failed to find all expected Nodes.\nExpected: %v\nActual: %v", expectedNodes, nodes)
		return false
	}
	return true
}

func getMasterHost() string {
	masterUrl, err := url.Parse(testContext.Host)
	expectNoError(err)
	return masterUrl.Host
}

func testMonitoringUsingHeapsterInfluxdb(c *client.Client) {
	// Check if heapster pods and services are up.
	expectedPods, err := verifyExpectedRcsExistAndGetExpectedPods(c)
	expectNoError(err)
	expectNoError(expectedServicesExist(c))
	// TODO: Wait for all pods and services to be running.
	kubeMasterHttpClient, ok := c.Client.(*http.Client)
	if !ok {
		Failf("failed to get master http client")
	}
	proxyUrl := fmt.Sprintf("%s/api/v1beta3/proxy/namespaces/default/services/%s:api/", getMasterHost(), influxdbService)
	config := &influxdb.ClientConfig{
		Host: proxyUrl,
		// TODO(vishh): Infer username and pw from the Pod spec.
		Username:   influxdbUser,
		Password:   influxdbPW,
		Database:   influxdbDatabaseName,
		HttpClient: kubeMasterHttpClient,
		IsSecure:   true,
	}
	influxdbClient, err := influxdb.NewClient(config)
	expectNoError(err, "failed to create influxdb client")

	expectedNodes, err := getAllNodesInCluster(c)
	expectNoError(err)
	startTime := time.Now()
	for {
		if validatePodsAndNodes(influxdbClient, expectedPods, expectedNodes) {
			return
		}
		if time.Since(startTime) >= testTimeout {
			break
		}
		time.Sleep(sleepBetweenAttempts)
	}
	Failf("monitoring using heapster and influxdb test failed")
}
