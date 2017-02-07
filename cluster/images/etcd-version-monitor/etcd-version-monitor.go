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
	"fmt"
	"io"
	"io/ioutil"
	"net/http"
	"os"
	"os/exec"
	"strconv"
	"strings"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/common/log"
	"github.com/spf13/pflag"
)

// Initialize the prometheus instrumentation and client related flags.
var (
	listenAddress				string
	metricsPath					string
	etcdVersionScrapeURI		string
	etcdMetricsScrapeURI		string
	scrapeTimeout				time.Duration
)

func registerFlags(fs *pflag.FlagSet) {
	fs.StringVar(&listenAddress, "listen-address", ":9101", "Address to listen on for serving prometheus formatted metrics")
	fs.StringVar(&metricsPath, "metrics-path", "/metrics", "Path under which prometheus formatted metrics are to be served")
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
			Help: "Size of etcd data in a given backend version (>=3), labeled by backend version, unit for size and etcdctl version used",
		},
		[]string{"backend_version", "unit", "etcdctl_version"})
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
	ServerVersion		string `json:"etcdserver"`
	ClusterVersion		string `json:"etcdcluster"`
}

// Function for periodically fetching etcd version info.
func getVersion() {
	// Create the get request for the etcd endpoint.
	req, err := http.NewRequest("GET", etcdVersionScrapeURI, nil)
	if err != nil {
		log.Fatal("NewRequest: ", err)
		return
	}

	for {
		// Send the get request and receive a response.
		client := &http.Client{}
		resp, err := client.Do(req)
		if err != nil {
			log.Fatal("Do: ", err)
			time.Sleep(scrapeTimeout)
			continue
		}

		// Obtain EtcdVersion from the JSON response.
		var version EtcdVersion
		if err := json.NewDecoder(resp.Body).Decode(&version); err != nil {
			log.Fatal("JSON decoder for etcd version: ", err)
			time.Sleep(scrapeTimeout)
			continue
		}
		resp.Body.Close()

		// Record the info in the prometheus metric through labels.
		etcdVersionFetchCount.With(prometheus.Labels{
			"server_version": version.ServerVersion,
			"cluster_version": version.ClusterVersion,
		}).Inc()
		time.Sleep(scrapeTimeout)
	}
}

// Function for periodically fetching etcd versioned data size from etcdctl.
func getVersionedDataSize() {
	for {
		// Prepare the etcdctl command to be executed.
		cmd := exec.Command("/bin/etcdctl", "--endpoints=localhost:2379", "endpoint", "status")
		env := os.Environ()
		env = append(env, fmt.Sprintf("ETCDCTL_API=3"))
		cmd.Env = env
		cmdOut, _ := cmd.StdoutPipe()

		// Run the etcdctl command and get the output.
		if err := cmd.Start(); err != nil {
			log.Fatal("Failed to start etcdctl command: ", err)
			time.Sleep(scrapeTimeout)
			continue
		}
		out, _ := ioutil.ReadAll(cmdOut)
		if err := cmd.Wait(); err != nil {
			log.Fatal("Failed to finish etcdctl command: ", err)
			time.Sleep(scrapeTimeout)
			continue
		}

		// Obtain the versioned data size from the comma-seperated output.
		params := strings.Split(string(out), ",")
		version := strings.TrimSpace(params[2])
		size_params := strings.Split(strings.TrimSpace(params[3]), " ")
		value, _ := strconv.ParseFloat(size_params[0], 64)
		unit := size_params[1]

		// Record the versioned data size in the prometheus metric with labels.
		etcdVersionedDataSize.With(prometheus.Labels{
			"backend_version": "3",
			"unit": unit,
			"etcdctl_version": version,
		}).Set(value)
		time.Sleep(scrapeTimeout)
	}
}

// Struct for unmarshalling the json for etcd grpc request labels.
type GRPCLabels struct {
	GrpcMethod			string `json:"grpc_method"`
	GrpcService			string `json:"grpc_service"`
}

// Struct for unmarshalling the json for etcd grpc requests metric.
type EtcdGRPCRequestCount struct {
	GrpcLabels			GRPCLabels 	`json:"labels"`
	Count				string 		`json:"value"`
}

// Function for periodically fetching etcd GRPC request counts.
func getGRPCRequestCount() {
	// Map for storing last obtained request count for a given grpc request type.
	lastCount := make(map[GRPCLabels]float64)

	for {
		// Prepare the prom2json and jq commands to be executed in a pipe.
		cmd1 := exec.Command("/bin/prom2json", etcdMetricsScrapeURI)
		cmd2 := exec.Command("jq", ".[]|select(.name==\"etcd_grpc_requests_total\").metrics")
		r, w := io.Pipe()
		cmd1.Stdout = w
		cmd2.Stdin = r
		cmd2Out, _ := cmd2.StdoutPipe()

		// Run the prom2json command.
		if err := cmd1.Start(); err != nil {
			log.Fatal("Failed to start prom2json command: ", err)
			time.Sleep(scrapeTimeout)
			continue
		}
		// Run the jq command.
		if err := cmd2.Start(); err != nil {
			log.Fatal("Failed to start jq command: ", err)
			time.Sleep(scrapeTimeout)
			continue
		}
		// Wait for prom2json to finish and close w.
		if err := cmd1.Wait(); err != nil {
			log.Fatal("Failed to finish prom2json command: ", err)
			time.Sleep(scrapeTimeout)
			continue
		}
		w.Close()
		// Read output from jq command and wait for it to finish.
		out, _ := ioutil.ReadAll(cmd2Out)
		if err := cmd2.Wait(); err != nil {
			log.Fatal("Failed to finish jq command: ", err)
			time.Sleep(scrapeTimeout)
			continue
		}

		// Parse the grpc request counts from the json response.
		var requestCounts []EtcdGRPCRequestCount
		if err := json.Unmarshal(out, &requestCounts); err != nil {
			log.Fatal("JSON decoder for grpc request count: ", err)
			time.Sleep(scrapeTimeout)
			continue
		}

		// Increment the request count metrics based on current and previous value.
		for _ , requestCount := range requestCounts {
			requestType := requestCount.GrpcLabels
			newCount, _ := strconv.ParseFloat(requestCount.Count, 64)
			oldCount := 0.0
			if val, ok := lastCount[requestType]; ok {
				oldCount = val	
			}
			lastCount[requestType] = newCount
			if newCount > oldCount {
				etcdGRPCRequestsTotal.With(prometheus.Labels{
					"grpc_method": requestType.GrpcMethod,
					"grpc_service": requestType.GrpcService,
				}).Add(newCount - oldCount)
			}
		}
		time.Sleep(scrapeTimeout)	
	}
}

func main() {
	// Register the commandline flags passed to the tool.
	registerFlags(pflag.CommandLine)
	pflag.Parse()

	// Register the metrics we defined above with prometheus.
	prometheus.MustRegister(etcdVersionFetchCount)
	prometheus.MustRegister(etcdVersionedDataSize)
	prometheus.MustRegister(etcdGRPCRequestsTotal)
	prometheus.Unregister(prometheus.NewGoCollector())	// Unregister go metrics.

	// Spawn threads for periodically scraping etcd version metrics and populating our metrics.
	go getVersion()
	go getVersionedDataSize()
	go getGRPCRequestCount()

	// Serve our metrics on listenAddress/metricsPath.
	log.Infoln("Listening on: ", listenAddress)
	http.Handle(metricsPath, prometheus.Handler())
	http.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte(`<html>
             <head><title>Etcd Exporter</title></head>
             <body>
             <h1>Etcd Exporter</h1>
             <p><a href='` + metricsPath + `'>Go here for the etcd metrics</a></p>
             </body>
             </html>`))
	})
	log.Fatal(http.ListenAndServe(listenAddress, nil))
}
