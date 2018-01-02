// Copyright 2014 Google Inc. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package main

import (
	"flag"
	"fmt"
	"net/http"
	"net/http/pprof"
	"os"
	"os/signal"
	"runtime"
	"strings"
	"syscall"
	"time"

	"github.com/google/cadvisor/container"
	cadvisorhttp "github.com/google/cadvisor/http"
	"github.com/google/cadvisor/manager"
	"github.com/google/cadvisor/utils/sysfs"
	"github.com/google/cadvisor/version"

	"crypto/tls"

	"github.com/golang/glog"
)

var argIp = flag.String("listen_ip", "", "IP to listen on, defaults to all IPs")
var argPort = flag.Int("port", 8080, "port to listen")
var maxProcs = flag.Int("max_procs", 0, "max number of CPUs that can be used simultaneously. Less than 1 for default (number of cores).")

var versionFlag = flag.Bool("version", false, "print cAdvisor version and exit")

var httpAuthFile = flag.String("http_auth_file", "", "HTTP auth file for the web UI")
var httpAuthRealm = flag.String("http_auth_realm", "localhost", "HTTP auth realm for the web UI")
var httpDigestFile = flag.String("http_digest_file", "", "HTTP digest file for the web UI")
var httpDigestRealm = flag.String("http_digest_realm", "localhost", "HTTP digest file for the web UI")

var prometheusEndpoint = flag.String("prometheus_endpoint", "/metrics", "Endpoint to expose Prometheus metrics on")

var maxHousekeepingInterval = flag.Duration("max_housekeeping_interval", 60*time.Second, "Largest interval to allow between container housekeepings")
var allowDynamicHousekeeping = flag.Bool("allow_dynamic_housekeeping", true, "Whether to allow the housekeeping interval to be dynamic")

var enableProfiling = flag.Bool("profiling", false, "Enable profiling via web interface host:port/debug/pprof/")

var collectorCert = flag.String("collector_cert", "", "Collector's certificate, exposed to endpoints for certificate based authentication.")
var collectorKey = flag.String("collector_key", "", "Key for the collector's certificate")

var (
	// Metrics to be ignored.
	// Tcp metrics are ignored by default.
	ignoreMetrics metricSetValue = metricSetValue{container.MetricSet{
		container.NetworkTcpUsageMetrics: struct{}{},
		container.NetworkUdpUsageMetrics: struct{}{},
	}}

	// List of metrics that can be ignored.
	ignoreWhitelist = container.MetricSet{
		container.DiskUsageMetrics:       struct{}{},
		container.NetworkUsageMetrics:    struct{}{},
		container.NetworkTcpUsageMetrics: struct{}{},
		container.NetworkUdpUsageMetrics: struct{}{},
	}
)

type metricSetValue struct {
	container.MetricSet
}

func (ml *metricSetValue) String() string {
	var values []string
	for metric, _ := range ml.MetricSet {
		values = append(values, string(metric))
	}
	return strings.Join(values, ",")
}

func (ml *metricSetValue) Set(value string) error {
	ml.MetricSet = container.MetricSet{}
	if value == "" {
		return nil
	}
	for _, metric := range strings.Split(value, ",") {
		if ignoreWhitelist.Has(container.MetricKind(metric)) {
			(*ml).Add(container.MetricKind(metric))
		} else {
			return fmt.Errorf("unsupported metric %q specified in disable_metrics", metric)
		}
	}
	return nil
}

func init() {
	flag.Var(&ignoreMetrics, "disable_metrics", "comma-separated list of `metrics` to be disabled. Options are 'disk', 'network', 'tcp', 'udp'. Note: tcp and udp are disabled by default due to high CPU usage.")

	// Default logging verbosity to V(2)
	flag.Set("v", "2")
}

func main() {
	defer glog.Flush()
	flag.Parse()

	if *versionFlag {
		fmt.Printf("cAdvisor version %s (%s)\n", version.Info["version"], version.Info["revision"])
		os.Exit(0)
	}

	setMaxProcs()

	memoryStorage, err := NewMemoryStorage()
	if err != nil {
		glog.Fatalf("Failed to initialize storage driver: %s", err)
	}

	sysFs := sysfs.NewRealSysFs()

	collectorHttpClient := createCollectorHttpClient(*collectorCert, *collectorKey)

	containerManager, err := manager.New(memoryStorage, sysFs, *maxHousekeepingInterval, *allowDynamicHousekeeping, ignoreMetrics.MetricSet, &collectorHttpClient)
	if err != nil {
		glog.Fatalf("Failed to create a Container Manager: %s", err)
	}

	mux := http.NewServeMux()

	if *enableProfiling {
		mux.HandleFunc("/debug/pprof/", pprof.Index)
		mux.HandleFunc("/debug/pprof/cmdline", pprof.Cmdline)
		mux.HandleFunc("/debug/pprof/profile", pprof.Profile)
		mux.HandleFunc("/debug/pprof/symbol", pprof.Symbol)
	}

	// Register all HTTP handlers.
	err = cadvisorhttp.RegisterHandlers(mux, containerManager, *httpAuthFile, *httpAuthRealm, *httpDigestFile, *httpDigestRealm)
	if err != nil {
		glog.Fatalf("Failed to register HTTP handlers: %v", err)
	}

	cadvisorhttp.RegisterPrometheusHandler(mux, containerManager, *prometheusEndpoint, nil)

	// Start the manager.
	if err := containerManager.Start(); err != nil {
		glog.Fatalf("Failed to start container manager: %v", err)
	}

	// Install signal handler.
	installSignalHandler(containerManager)

	glog.V(1).Infof("Starting cAdvisor version: %s-%s on port %d", version.Info["version"], version.Info["revision"], *argPort)

	addr := fmt.Sprintf("%s:%d", *argIp, *argPort)
	glog.Fatal(http.ListenAndServe(addr, mux))
}

func setMaxProcs() {
	// TODO(vmarmol): Consider limiting if we have a CPU mask in effect.
	// Allow as many threads as we have cores unless the user specified a value.
	var numProcs int
	if *maxProcs < 1 {
		numProcs = runtime.NumCPU()
	} else {
		numProcs = *maxProcs
	}
	runtime.GOMAXPROCS(numProcs)

	// Check if the setting was successful.
	actualNumProcs := runtime.GOMAXPROCS(0)
	if actualNumProcs != numProcs {
		glog.Warningf("Specified max procs of %v but using %v", numProcs, actualNumProcs)
	}
}

func installSignalHandler(containerManager manager.Manager) {
	c := make(chan os.Signal, 1)
	signal.Notify(c, os.Interrupt, os.Kill, syscall.SIGTERM)

	// Block until a signal is received.
	go func() {
		sig := <-c
		if err := containerManager.Stop(); err != nil {
			glog.Errorf("Failed to stop container manager: %v", err)
		}
		glog.Infof("Exiting given signal: %v", sig)
		os.Exit(0)
	}()
}

func createCollectorHttpClient(collectorCert, collectorKey string) http.Client {
	//Enable accessing insecure endpoints. We should be able to access metrics from any endpoint
	tlsConfig := &tls.Config{
		InsecureSkipVerify: true,
	}

	if collectorCert != "" {
		if collectorKey == "" {
			glog.Fatal("The collector_key value must be specified if the collector_cert value is set.")
		}
		cert, err := tls.LoadX509KeyPair(collectorCert, collectorKey)
		if err != nil {
			glog.Fatalf("Failed to use the collector certificate and key: %s", err)
		}

		tlsConfig.Certificates = []tls.Certificate{cert}
		tlsConfig.BuildNameToCertificate()
	}

	transport := &http.Transport{
		TLSClientConfig: tlsConfig,
	}

	return http.Client{Transport: transport}
}
