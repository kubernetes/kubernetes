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

package main

import (
	"encoding/json"
	goflag "flag"
	"fmt"
	"net/http"
	"time"

	"github.com/golang/glog"
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/common/expfmt"
	"github.com/spf13/pflag"
)

// Initialize the prometheus instrumentation and client related flags.
var (
	listenAddress        string
	metricsPath          string
	etcdVersionScrapeURI string
	etcdMetricsScrapeURI string
	scrapeTimeout        time.Duration
)

func registerFlags(fs *pflag.FlagSet) {
	fs.StringVar(&listenAddress, "listen-address", "localhost:9101", "Address to listen on for serving prometheus metrics")
	fs.StringVar(&metricsPath, "metrics-path", "/metrics", "Path under which prometheus metrics are to be served")
	fs.StringVar(&etcdVersionScrapeURI, "etcd-version-scrape-uri", "http://localhost:2379/version", "URI to scrape etcd version info")
	fs.StringVar(&etcdMetricsScrapeURI, "etcd-metrics-scrape-uri", "http://localhost:2379/metrics", "URI to scrape etcd metrics")
	fs.DurationVar(&scrapeTimeout, "scrape-timeout", 15*time.Second, "Timeout for trying to get stats from etcd")
}

const (
	namespace = "etcd" // For prefixing prometheus metrics
)

// Initialize prometheus metrics to be exported.
var (
	etcdVersion = prometheus.NewGaugeVec(
		prometheus.GaugeOpts{
			Namespace: namespace,
			Name:      "version_info",
			Help:      "Etcd server's binary version",
		},
		[]string{"binary_version"})
	etcdGRPCRequestsTotal = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Namespace: namespace,
			Name:      "grpc_requests_total",
			Help:      "Counter of received grpc requests, labeled by the grpc method and service names",
		},
		[]string{"method", "service"})
)

// Struct for unmarshalling the json response from etcd's /version endpoint.
type EtcdVersion struct {
	BinaryVersion  string `json:"etcdserver"`
	ClusterVersion string `json:"etcdcluster"`
}

// Function for fetching etcd version info and feeding it to the prometheus metric.
func getVersion(lastSeenBinaryVersion *string) error {
	// Create the get request for the etcd version endpoint.
	req, err := http.NewRequest("GET", etcdVersionScrapeURI, nil)
	if err != nil {
		return fmt.Errorf("Failed to create GET request for etcd version: %v", err)
	}

	// Send the get request and receive a response.
	client := &http.Client{}
	resp, err := client.Do(req)
	if err != nil {
		return fmt.Errorf("Failed to receive GET response for etcd version: %v", err)
	}
	defer resp.Body.Close()

	// Obtain EtcdVersion from the JSON response.
	var version EtcdVersion
	if err := json.NewDecoder(resp.Body).Decode(&version); err != nil {
		return fmt.Errorf("Failed to decode etcd version JSON: %v", err)
	}

	// Return without updating the version if it stayed the same since last time.
	if *lastSeenBinaryVersion == version.BinaryVersion {
		return nil
	}

	// Delete the metric for the previous version.
	if *lastSeenBinaryVersion != "" {
		deleted := etcdVersion.Delete(prometheus.Labels{"binary_version": *lastSeenBinaryVersion})
		if !deleted {
			return fmt.Errorf("Failed to delete previous version's metric")
		}
	}

	// Record the new version in a metric.
	etcdVersion.With(prometheus.Labels{
		"binary_version": version.BinaryVersion,
	}).Set(0)
	*lastSeenBinaryVersion = version.BinaryVersion
	return nil
}

// Periodically fetches etcd version info.
func getVersionPeriodically(stopCh <-chan struct{}) {
	lastSeenBinaryVersion := ""
	for {
		if err := getVersion(&lastSeenBinaryVersion); err != nil {
			glog.Errorf("Failed to fetch etcd version: %v", err)
		}
		select {
		case <-stopCh:
			break
		case <-time.After(scrapeTimeout):
		}
	}
}

// Struct for storing labels for gRPC request types.
type GRPCRequestLabels struct {
	Method  string
	Service string
}

// Function for fetching etcd grpc request counts and feeding it to the prometheus metric.
func getGRPCRequestCount(lastRecordedCount *map[GRPCRequestLabels]float64) error {
	// Create the get request for the etcd metrics endpoint.
	req, err := http.NewRequest("GET", etcdMetricsScrapeURI, nil)
	if err != nil {
		return fmt.Errorf("Failed to create GET request for etcd metrics: %v", err)
	}

	// Send the get request and receive a response.
	client := &http.Client{}
	resp, err := client.Do(req)
	if err != nil {
		return fmt.Errorf("Failed to receive GET response for etcd metrics: %v", err)
	}
	defer resp.Body.Close()

	// Parse the metrics in text format to a MetricFamily struct.
	var textParser expfmt.TextParser
	metricFamilies, err := textParser.TextToMetricFamilies(resp.Body)
	if err != nil {
		return fmt.Errorf("Failed to parse etcd metrics: %v", err)
	}

	// Look through the grpc requests metric family and update our promotheus metric.
	for _, metric := range metricFamilies["etcd_grpc_requests_total"].GetMetric() {
		var grpcRequestLabels GRPCRequestLabels
		for _, label := range metric.GetLabel() {
			if label.GetName() == "grpc_method" {
				grpcRequestLabels.Method = label.GetValue()
			}
			if label.GetName() == "grpc_service" {
				grpcRequestLabels.Service = label.GetValue()
			}
		}
		if grpcRequestLabels.Method == "" || grpcRequestLabels.Service == "" {
			return fmt.Errorf("Could not get value for grpc_method and/or grpc_service label")
		}

		// Get last recorded value and new value of the metric and update it suitably.
		previousMetricValue := 0.0
		if value, ok := (*lastRecordedCount)[grpcRequestLabels]; ok {
			previousMetricValue = value
		}
		newMetricValue := metric.GetCounter().GetValue()
		(*lastRecordedCount)[grpcRequestLabels] = newMetricValue
		if newMetricValue >= previousMetricValue {
			etcdGRPCRequestsTotal.With(prometheus.Labels{
				"method":  grpcRequestLabels.Method,
				"service": grpcRequestLabels.Service,
			}).Add(newMetricValue - previousMetricValue)
		}
	}
	return nil
}

// Function for periodically fetching etcd GRPC request counts.
func getGRPCRequestCountPeriodically(stopCh <-chan struct{}) {
	// This map stores last recorded count for a given grpc request type.
	lastRecordedCount := make(map[GRPCRequestLabels]float64)
	for {
		if err := getGRPCRequestCount(&lastRecordedCount); err != nil {
			glog.Errorf("Failed to fetch etcd grpc request counts: %v", err)
		}
		select {
		case <-stopCh:
			break
		case <-time.After(scrapeTimeout):
		}
	}
}

func main() {
	// Register the commandline flags passed to the tool.
	registerFlags(pflag.CommandLine)
	pflag.CommandLine.AddGoFlagSet(goflag.CommandLine)
	pflag.Parse()

	// Register the metrics we defined above with prometheus.
	prometheus.MustRegister(etcdVersion)
	prometheus.MustRegister(etcdGRPCRequestsTotal)
	prometheus.Unregister(prometheus.NewGoCollector())

	// Spawn threads for periodically scraping etcd version metrics.
	stopCh := make(chan struct{})
	defer close(stopCh)
	go getVersionPeriodically(stopCh)
	go getGRPCRequestCountPeriodically(stopCh)

	// Serve our metrics on listenAddress/metricsPath.
	glog.Infof("Listening on: %v", listenAddress)
	http.Handle(metricsPath, prometheus.UninstrumentedHandler())
	glog.Errorf("Stopped listening/serving metrics: %v", http.ListenAndServe(listenAddress, nil))
}
