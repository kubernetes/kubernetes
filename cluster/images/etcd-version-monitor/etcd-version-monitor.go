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
	"io/ioutil"
	"net/http"
	"os"
	"os/exec"
	"strconv"
	"strings"
	"time"

	"github.com/golang/glog"
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/common/expfmt"
	"github.com/spf13/pflag"
)

// Initialize the prometheus instrumentation and client related flags.
var (
	listenAddress			string
	metricsPath				string
	etcdVersionScrapeURI	string
	etcdMetricsScrapeURI	string
	scrapeTimeout			time.Duration
)

func registerFlags(fs *pflag.FlagSet) {
	fs.StringVar(&listenAddress, "listen-address", ":9101", "Address to listen on for serving prometheus metrics")
	fs.StringVar(&metricsPath, "metrics-path", "/metrics", "Path under which prometheus metrics are to be served")
	fs.StringVar(&etcdVersionScrapeURI, "etcd-version-scrape-uri", "http://localhost:2379/version", "URI to scrape etcd version info")
	fs.StringVar(&etcdMetricsScrapeURI, "etcd-metrics-scrape-uri", "http://localhost:2379/metrics", "URI to scrape etcd metrics")
	fs.DurationVar(&scrapeTimeout, "scrape-timeout", 15*time.Second, "Timeout for trying to get stats from etcd")
}

const (
	namespace = "etcd"	// For prefixing prometheus metrics
)

// Initialize prometheus metrics to be exported.
var (
	etcdVersionFetchCount 	= prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Namespace: namespace,
			Name: "version_info_fetch_count",
			Help: "Number of times etcd's version info was fetched, labeled by etcd's server binary and cluster version",
		},
		[]string{"server_version", "cluster_version"})
	etcdVersionedDataSize 	= prometheus.NewGaugeVec(
		prometheus.GaugeOpts{
			Namespace: namespace,
			Name: "versioned_data_size",
			Help: "Size of etcd data in a given backend version (>=3), labeled by backend version and unit for size (KB, MB,..)",
		},
		[]string{"backend_version", "unit_for_size"})
	etcdGRPCRequestsTotal	= prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Namespace: namespace,
			Name: "grpc_requests_total",
			Help: "Counter of received grpc requests, labeled by grpc method and grpc service names",
		},
		[]string{"grpc_method", "grpc_service"})
)

// Struct for unmarshalling the json response from etcd's /version endpoint.
type EtcdVersion struct {
	ServerVersion	string `json:"etcdserver"`
	ClusterVersion	string `json:"etcdcluster"`
}

// Function for fetching etcd version info and feeding it to the prometheus metric.
func getVersion() error {
	// Create the get request for the etcd version endpoint.
	req, err := http.NewRequest("GET", etcdVersionScrapeURI, nil)
	if err != nil {
		return fmt.Errorf("Failed to create GET request for etcd version: %v", err)
	}

	// Send the get request and receive a response.
	client := &http.Client{}
	resp, err := client.Do(req)
	defer resp.Body.Close()
	if err != nil {
		return fmt.Errorf("Failed to receive GET response for etcd version: %v", err)
	}

	// Obtain EtcdVersion from the JSON response.
	var version EtcdVersion
	if err := json.NewDecoder(resp.Body).Decode(&version); err != nil {
		return fmt.Errorf("Failed to decode etcd version JSON: %v", err)
	}

	// Record the info in the prometheus metric through labels.
	etcdVersionFetchCount.With(prometheus.Labels{
		"server_version": version.ServerVersion,
		"cluster_version": version.ClusterVersion,
	}).Inc()
	return nil
}

// Periodically fetches etcd version info.
func getVersionPeriodically() {
	for {
		if err := getVersion(); err != nil {
			glog.Errorf("Failed to fetch etcd version: %v", err)
		}
		time.Sleep(scrapeTimeout)
	}
}

// Function for periodically fetching etcd versioned data size from etcdctl and feeding
// it to the prometheus metric.
func getVersionedDataSize() error {
	// Prepare the etcdctl command to be executed.
	cmd := exec.Command("/etcdctl", "--endpoints=localhost:2379", "endpoint", "status")
	cmd.Env = append(os.Environ(), fmt.Sprintf("ETCDCTL_API=3"))
	cmdOut, err := cmd.StdoutPipe()
	if err != nil {
		return fmt.Errorf("Failed to create Stdout pipe for etcdctl command: %v", err)
	}

	// Run the etcdctl command and get the output.
	if err := cmd.Start(); err != nil {
		return fmt.Errorf("Failed to start etcdctl command: %v", err)
	}
	out, _ := ioutil.ReadAll(cmdOut)
	if err := cmd.Wait(); err != nil {
		return fmt.Errorf("Failed to finish etcdctl command: %v", err)
	}

	// Obtain the versioned data size from the comma-seperated output.
	params := strings.Split(string(out), ",")
	size_params := strings.Split(strings.TrimSpace(params[3]), " ")
	value, err := strconv.ParseFloat(size_params[0], 64)
	if err != nil {
		return fmt.Errorf("Failed to parse etcd data size as a float: %v", err)
	}	
	unit := size_params[1]

	// Record the versioned data size in the prometheus metric with labels.
	etcdVersionedDataSize.With(prometheus.Labels{
		"backend_version": "3",
		"unit_for_size": unit,
	}).Set(value)
	return nil
}

// Periodically fetches etcd versioned data size.
func getVersionedDataSizePeriodically() {
	for {
		if err := getVersionedDataSize(); err != nil {
			glog.Errorf("Failed to fetch etcd versioned data size: %v", err)
		}
		time.Sleep(scrapeTimeout)
	}
}

// Struct for storing labels for gRPC request types.
type GRPCRequestLabels struct {
	GrpcMethod	string
	GrpcService	string
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
	defer resp.Body.Close()
	if err != nil {
		return fmt.Errorf("Failed to receive GET response for etcd metrics: %v", err)
	}

	// Parse the metrics in text format to a MetricFamily struct.
	var textParser expfmt.TextParser
	metricFamilies, err := textParser.TextToMetricFamilies(resp.Body)
	if err != nil {
		return fmt.Errorf("Failed to parse etcd metrics: %v", err)
	}

	// Look through the grpc requests metric family and update our promotheus metric.
	for _ , metric := range metricFamilies["etcd_grpc_requests_total"].GetMetric() {
		var grpcRequestLabels GRPCRequestLabels
		for _ , label := range metric.GetLabel() {
			if label.GetName() == "grpc_method" {
				grpcRequestLabels.GrpcMethod = label.GetValue()
			}
			if label.GetName() == "grpc_service" {
				grpcRequestLabels.GrpcService = label.GetValue()
			}
		}
		if grpcRequestLabels.GrpcMethod == "" || grpcRequestLabels.GrpcService == "" {
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
				"grpc_method": grpcRequestLabels.GrpcMethod,
				"grpc_service": grpcRequestLabels.GrpcService,
			}).Add(newMetricValue - previousMetricValue)
		}
	}
	return nil
}

// Function for periodically fetching etcd GRPC request counts.
func getGRPCRequestCountPeriodically() {
	// This map stores last recorded count for a given grpc request type.
	lastRecordedCount := make(map[GRPCRequestLabels]float64)
	for {
		if err := getGRPCRequestCount(&lastRecordedCount); err != nil {
			glog.Errorf("Failed to fetch etcd grpc request counts: %v", err)
		}
		time.Sleep(scrapeTimeout)
	}
}

func main() {
	// Register the commandline flags passed to the tool.
	registerFlags(pflag.CommandLine)
	pflag.CommandLine.AddGoFlagSet(goflag.CommandLine)
	pflag.Parse()

	// Register the metrics we defined above with prometheus.
	prometheus.MustRegister(etcdVersionFetchCount)
	prometheus.MustRegister(etcdVersionedDataSize)
	prometheus.MustRegister(etcdGRPCRequestsTotal)
	prometheus.Unregister(prometheus.NewGoCollector())	// Unregister go metrics.

	// Spawn threads for periodically scraping etcd version metrics.
	go getVersionPeriodically()
	go getVersionedDataSizePeriodically()
	go getGRPCRequestCountPeriodically()

	// Serve our metrics on listenAddress/metricsPath.
	glog.Infof("Listening on: %v", listenAddress)
	http.Handle(metricsPath, prometheus.Handler())
	http.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte(`<html>
             <head><title>Etcd version metrics exporter</title></head>
             <body>
             <h1>Etcd version metrics exporter</h1>
             <p><a href='` + metricsPath + `'>Go here for the etcd metrics</a></p>
             </body>
             </html>`))
	})
	glog.Fatal("%v", http.ListenAndServe(listenAddress, nil))
}
