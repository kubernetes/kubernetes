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

package elasticsearch

import (
	"encoding/json"
	"fmt"
	"strconv"
	"time"

	meta_v1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/fields"
	api "k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/kubernetes/test/e2e/framework"
	e2elog "k8s.io/kubernetes/test/e2e/framework/log"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	e2eservice "k8s.io/kubernetes/test/e2e/framework/service"
	"k8s.io/kubernetes/test/e2e/instrumentation/logging/utils"
)

const (
	// esRetryTimeout is how long to keep retrying requesting elasticsearch for status information.
	esRetryTimeout = 5 * time.Minute

	// esRetryDelay is how much time to wait between two attempts to send a request to elasticsearch
	esRetryDelay = 5 * time.Second

	// searchPageSize is how many entries to search for in Elasticsearch.
	searchPageSize = 1000
)

var _ utils.LogProvider = &esLogProvider{}

type esLogProvider struct {
	Framework *framework.Framework
}

func newEsLogProvider(f *framework.Framework) (*esLogProvider, error) {
	return &esLogProvider{Framework: f}, nil
}

// Ensures that elasticsearch is running and ready to serve requests
func (p *esLogProvider) Init() error {
	f := p.Framework
	// Check for the existence of the Elasticsearch service.
	e2elog.Logf("Checking the Elasticsearch service exists.")
	s := f.ClientSet.CoreV1().Services(api.NamespaceSystem)
	// Make a few attempts to connect. This makes the test robust against
	// being run as the first e2e test just after the e2e cluster has been created.
	var err error
	for start := time.Now(); time.Since(start) < esRetryTimeout; time.Sleep(esRetryDelay) {
		if _, err = s.Get("elasticsearch-logging", meta_v1.GetOptions{}); err == nil {
			break
		}
		e2elog.Logf("Attempt to check for the existence of the Elasticsearch service failed after %v", time.Since(start))
	}
	if err != nil {
		return err
	}

	// Wait for the Elasticsearch pods to enter the running state.
	e2elog.Logf("Checking to make sure the Elasticsearch pods are running")
	labelSelector := fields.SelectorFromSet(fields.Set(map[string]string{"k8s-app": "elasticsearch-logging"})).String()
	options := meta_v1.ListOptions{LabelSelector: labelSelector}
	pods, err := f.ClientSet.CoreV1().Pods(api.NamespaceSystem).List(options)
	if err != nil {
		return err
	}
	for _, pod := range pods.Items {
		err = e2epod.WaitForPodRunningInNamespace(f.ClientSet, &pod)
		if err != nil {
			return err
		}
	}

	e2elog.Logf("Checking to make sure we are talking to an Elasticsearch service.")
	// Perform a few checks to make sure this looks like an Elasticsearch cluster.
	var statusCode int
	err = nil
	var body []byte
	for start := time.Now(); time.Since(start) < esRetryTimeout; time.Sleep(esRetryDelay) {
		proxyRequest, errProxy := e2eservice.GetServicesProxyRequest(f.ClientSet, f.ClientSet.CoreV1().RESTClient().Get())
		if errProxy != nil {
			e2elog.Logf("After %v failed to get services proxy request: %v", time.Since(start), errProxy)
			continue
		}
		// Query against the root URL for Elasticsearch.
		response := proxyRequest.Namespace(api.NamespaceSystem).
			Name("elasticsearch-logging").
			Do()
		err = response.Error()
		response.StatusCode(&statusCode)

		if err != nil {
			e2elog.Logf("After %v proxy call to elasticsearch-loigging failed: %v", time.Since(start), err)
			continue
		}
		if int(statusCode) != 200 {
			e2elog.Logf("After %v Elasticsearch cluster has a bad status: %v", time.Since(start), statusCode)
			continue
		}
		break
	}
	if err != nil {
		return err
	}
	if int(statusCode) != 200 {
		e2elog.Failf("Elasticsearch cluster has a bad status: %v", statusCode)
	}

	// Now assume we really are talking to an Elasticsearch instance.
	// Check the cluster health.
	e2elog.Logf("Checking health of Elasticsearch service.")
	healthy := false
	for start := time.Now(); time.Since(start) < esRetryTimeout; time.Sleep(esRetryDelay) {
		proxyRequest, errProxy := e2eservice.GetServicesProxyRequest(f.ClientSet, f.ClientSet.CoreV1().RESTClient().Get())
		if errProxy != nil {
			e2elog.Logf("After %v failed to get services proxy request: %v", time.Since(start), errProxy)
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
			e2elog.Logf("Bad json response from elasticsearch: %v", err)
			continue
		}
		statusIntf, ok := health["status"]
		if !ok {
			e2elog.Logf("No status field found in cluster health response: %v", health)
			continue
		}
		status := statusIntf.(string)
		if status != "green" && status != "yellow" {
			e2elog.Logf("Cluster health has bad status: %v", health)
			continue
		}
		if err == nil && ok {
			healthy = true
			break
		}
	}
	if !healthy {
		return fmt.Errorf("after %v elasticsearch cluster is not healthy", esRetryTimeout)
	}

	return nil
}

func (p *esLogProvider) Cleanup() {
	// Nothing to do
}

func (p *esLogProvider) ReadEntries(name string) []utils.LogEntry {
	f := p.Framework

	proxyRequest, errProxy := e2eservice.GetServicesProxyRequest(f.ClientSet, f.ClientSet.CoreV1().RESTClient().Get())
	if errProxy != nil {
		e2elog.Logf("Failed to get services proxy request: %v", errProxy)
		return nil
	}

	query := fmt.Sprintf("kubernetes.pod_name:%s AND kubernetes.namespace_name:%s", name, f.Namespace.Name)
	e2elog.Logf("Sending a search request to Elasticsearch with the following query: %s", query)

	// Ask Elasticsearch to return all the log lines that were tagged with the
	// pod name. Ask for ten times as many log lines because duplication is possible.
	body, err := proxyRequest.Namespace(api.NamespaceSystem).
		Name("elasticsearch-logging").
		Suffix("_search").
		Param("q", query).
		// Ask for more in case we included some unrelated records in our query
		Param("size", strconv.Itoa(searchPageSize)).
		DoRaw()
	if err != nil {
		e2elog.Logf("Failed to make proxy call to elasticsearch-logging: %v", err)
		return nil
	}

	var response map[string]interface{}
	err = json.Unmarshal(body, &response)
	if err != nil {
		e2elog.Logf("Failed to unmarshal response: %v", err)
		return nil
	}

	hits, ok := response["hits"].(map[string]interface{})
	if !ok {
		e2elog.Logf("response[hits] not of the expected type: %T", response["hits"])
		return nil
	}

	h, ok := hits["hits"].([]interface{})
	if !ok {
		e2elog.Logf("Hits not of the expected type: %T", hits["hits"])
		return nil
	}

	entries := []utils.LogEntry{}
	// Iterate over the hits and populate the observed array.
	for _, e := range h {
		l, ok := e.(map[string]interface{})
		if !ok {
			e2elog.Logf("Element of hit not of expected type: %T", e)
			continue
		}

		source, ok := l["_source"].(map[string]interface{})
		if !ok {
			e2elog.Logf("_source not of the expected type: %T", l["_source"])
			continue
		}

		msg, ok := source["log"].(string)
		if ok {
			entries = append(entries, utils.LogEntry{TextPayload: msg})
			continue
		}

		obj, ok := source["log"].(map[string]interface{})
		if ok {
			entries = append(entries, utils.LogEntry{JSONPayload: obj})
			continue
		}

		e2elog.Logf("Log is of unknown type, got %v, want string or object in field 'log'", source)
	}

	return entries
}

func (p *esLogProvider) LoggingAgentName() string {
	return "fluentd-es"
}
