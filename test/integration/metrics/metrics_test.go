// +build integration,!no-etcd,linux

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

package metrics

import (
	"bufio"
	"fmt"
	"net/http"
	"net/http/httptest"
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	restclient "k8s.io/client-go/rest"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/client/clientset_generated/clientset"
	"k8s.io/kubernetes/test/integration/framework"

	"github.com/golang/glog"
	"github.com/golang/protobuf/proto"
	prometheuspb "github.com/prometheus/client_model/go"
)

const scrapeRequestHeader = "application/vnd.google.protobuf;proto=io.prometheus.client.MetricFamily;encoding=compact-text"

func scrapeMetrics(s *httptest.Server) ([]*prometheuspb.MetricFamily, error) {
	req, err := http.NewRequest("GET", s.URL+"/metrics", nil)
	if err != nil {
		return nil, fmt.Errorf("Unable to create http request: %v", err)
	}
	// Ask the prometheus exporter for its text protocol buffer format, since it's
	// much easier to parse than its plain-text format. Don't use the serialized
	// proto representation since it uses a non-standard varint delimiter between
	// metric families.
	req.Header.Add("Accept", scrapeRequestHeader)

	client := &http.Client{}
	resp, err := client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("Unable to contact metrics endpoint of master: %v", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != 200 {
		return nil, fmt.Errorf("Non-200 response trying to scrape metrics from master: %v", resp)
	}

	// Each line in the response body should contain all the data for a single metric.
	var metrics []*prometheuspb.MetricFamily
	scanner := bufio.NewScanner(resp.Body)
	for scanner.Scan() {
		var metric prometheuspb.MetricFamily
		if err := proto.UnmarshalText(scanner.Text(), &metric); err != nil {
			return nil, fmt.Errorf("Failed to unmarshal line of metrics response: %v", err)
		}
		glog.V(4).Infof("Got metric %q", metric.GetName())
		metrics = append(metrics, &metric)
	}
	return metrics, nil
}

func checkForExpectedMetrics(t *testing.T, metrics []*prometheuspb.MetricFamily, expectedMetrics []string) {
	foundMetrics := make(map[string]bool)
	for _, metric := range metrics {
		foundMetrics[metric.GetName()] = true
	}
	for _, expected := range expectedMetrics {
		if _, found := foundMetrics[expected]; !found {
			t.Errorf("Master metrics did not include expected metric %q", expected)
		}
	}
}

func TestMasterProcessMetrics(t *testing.T) {
	_, s := framework.RunAMaster(nil)
	defer s.Close()

	metrics, err := scrapeMetrics(s)
	if err != nil {
		t.Fatal(err)
	}
	checkForExpectedMetrics(t, metrics, []string{
		"process_start_time_seconds",
		"process_cpu_seconds_total",
		"go_goroutines",
		"process_open_fds",
		"process_resident_memory_bytes",
	})
}

func TestApiserverMetrics(t *testing.T) {
	_, s := framework.RunAMaster(nil)
	defer s.Close()

	// Make a request to the apiserver to ensure there's at least one data point
	// for the metrics we're expecting -- otherwise, they won't be exported.
	client := clientset.NewForConfigOrDie(&restclient.Config{Host: s.URL, ContentConfig: restclient.ContentConfig{GroupVersion: &api.Registry.GroupOrDie(v1.GroupName).GroupVersion}})
	if _, err := client.Core().Pods(metav1.NamespaceDefault).List(metav1.ListOptions{}); err != nil {
		t.Fatalf("unexpected error getting pods: %v", err)
	}

	metrics, err := scrapeMetrics(s)
	if err != nil {
		t.Fatal(err)
	}
	checkForExpectedMetrics(t, metrics, []string{
		"apiserver_request_count",
		"apiserver_request_latencies",
	})
}
