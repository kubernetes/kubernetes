/*
Copyright 2015 Google Inc. All rights reserved.

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
	"github.com/GoogleCloudPlatform/kubernetes/pkg/labels"
	influxdb "github.com/influxdb/influxdb/client"

	. "github.com/onsi/ginkgo"
)

var _ = Describe("Monitoring", func() {
	var c *client.Client

	BeforeEach(func() {
		var err error
		c, err = loadClient()
		expectNoError(err)
	})

	It("pod and node resource usage metrics are available on influxdb using heapster.", func() {
		testMonitoringUsingHeapsterInfluxdb(c)
	})
})

const (
	influxdbService      = "monitoring-influxdb"
	influxdbDatabaseName = "k8s"
	influxdbUser         = "root"
	influxdbPW           = "root"
	podlistQuery         = "select distinct(pod) from stats"
	nodelistQuery        = "select distinct(hostname) from machine"
	sleepBetweenAttempts = 30 * time.Second
	maxAttempts          = 10 // Total sleep time of 5 minutes for this test.
)

var (
	expectedRcs = map[string]bool{
		"monitoring-heapster-controller":       false,
		"monitoring-influx-grafana-controller": false,
	}
	expectedServices = map[string]bool{
		influxdbService:       false,
		"monitoring-heapster": false,
		"monitoring-grafana":  false,
	}
)

func expectedRcsExist(c *client.Client) {
	rcList, err := c.ReplicationControllers(api.NamespaceDefault).List(labels.Everything())
	expectNoError(err)
	for _, rc := range rcList.Items {
		if _, ok := expectedRcs[rc.Name]; ok {
			if rc.Status.Replicas != 1 {
				Failf("expected to find only one replica for rc %q, found %d", rc.Name, rc.Status.Replicas)
			}
			expectedRcs[rc.Name] = true
		}
	}
	for rc, found := range expectedRcs {
		if !found {
			Failf("Replication Controller %q not found.", rc)
		}
	}
}

func expectedServicesExist(c *client.Client) {
	serviceList, err := c.Services(api.NamespaceDefault).List(labels.Everything())
	expectNoError(err)
	for _, service := range serviceList.Items {
		if _, ok := expectedServices[service.Name]; ok {
			expectedServices[service.Name] = true
		}
	}
	for service, found := range expectedServices {
		if !found {
			Failf("Service %q not found", service)
		}
	}
}

func getAllPodsInCluster(c *client.Client) []string {
	podList, err := c.Pods(api.NamespaceAll).List(labels.Everything())
	expectNoError(err)
	result := []string{}
	for _, pod := range podList.Items {
		result = append(result, pod.Name)
	}
	return result
}

func getAllNodesInCluster(c *client.Client) []string {
	nodeList, err := c.Nodes().List()
	expectNoError(err)
	result := []string{}
	for _, node := range nodeList.Items {
		result = append(result, node.Name)
	}
	return result
}

func getInfluxdbData(c *influxdb.Client, query string) (map[string]bool, error) {
	series, err := c.Query(query, influxdb.Second)
	if err != nil {
		return nil, err
	}
	if len(series) != 1 {
		Failf("expected only one series from Influxdb for query %q. Got %+v", query, series)
	}
	if len(series[0].GetColumns()) != 2 {
		Failf("Expected two columns for query %q. Found %v", query, series[0].GetColumns())
	}
	result := map[string]bool{}
	for _, point := range series[0].GetPoints() {
		name, ok := point[1].(string)
		if !ok {
			Failf("expected %v to be a string, but it is %T", point[1], point[1])
		}
		result[name] = false
	}
	return result, nil
}

func expectedItemsExist(expectedItems []string, actualItems map[string]bool) bool {
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
	masterUrl, err := url.Parse(testContext.host)
	expectNoError(err)
	return masterUrl.Host
}

func testMonitoringUsingHeapsterInfluxdb(c *client.Client) {
	// Check if heapster pods and services are up.
	expectedRcsExist(c)
	expectedServicesExist(c)
	// TODO: Wait for all pods and services to be running.
	kubeMasterHttpClient, ok := c.Client.(*http.Client)
	if !ok {
		Failf("failed to get master http client")
	}
	proxyUrl := fmt.Sprintf("%s/api/v1beta1/proxy/services/%s/", getMasterHost(), influxdbService)
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

	expectedPods := getAllPodsInCluster(c)
	expectedNodes := getAllNodesInCluster(c)
	attempt := maxAttempts
	for {
		if validatePodsAndNodes(influxdbClient, expectedPods, expectedNodes) {
			return
		}
		if attempt--; attempt <= 0 {
			break
		}
		time.Sleep(sleepBetweenAttempts)
	}
	Failf("monitoring using heapster and influxdb test failed")
}
