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
	"bytes"
	"encoding/json"
	"errors"
	goflag "flag"
	"fmt"
	"net/http"
	"time"

	"github.com/gogo/protobuf/proto"
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promhttp"
	dto "github.com/prometheus/client_model/go"
	"github.com/prometheus/common/expfmt"
	"github.com/spf13/pflag"
	"k8s.io/klog"
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
	// Register all custom metrics with a dedicated registry to keep them separate.
	customMetricRegistry = prometheus.NewRegistry()

	// Custom etcd version metric since etcd 3.2- does not export one.
	// This will be replaced by https://github.com/coreos/etcd/pull/8960 in etcd 3.3.
	etcdVersion = prometheus.NewGaugeVec(
		prometheus.GaugeOpts{
			Namespace: namespace,
			Name:      "version_info",
			Help:      "Etcd server's binary version",
		},
		[]string{"binary_version"})

	gatherer = &monitorGatherer{
		// Rewrite rules for etcd metrics that are exported by default.
		exported: map[string]*exportedMetric{
			// etcd 3.0 metric format for total grpc requests with renamed method and service labels.
			"etcd_grpc_requests_total": {
				rewriters: []rewriteFunc{
					func(mf *dto.MetricFamily) (*dto.MetricFamily, error) {
						mf = deepCopyMetricFamily(mf)
						renameLabels(mf, map[string]string{
							"grpc_method":  "method",
							"grpc_service": "service",
						})
						return mf, nil
					},
				},
			},
			// etcd 3.1+ metric format for total grpc requests.
			"grpc_server_handled_total": {
				rewriters: []rewriteFunc{
					// Export the metric exactly as-is. For 3.1+ metrics, we will
					// pass all metrics directly through.
					identity,
					// Write to the etcd 3.0 metric format for backward compatibility.
					func(mf *dto.MetricFamily) (*dto.MetricFamily, error) {
						mf = deepCopyMetricFamily(mf)
						renameMetric(mf, "etcd_grpc_requests_total")
						renameLabels(mf, map[string]string{
							"grpc_method":  "method",
							"grpc_service": "service",
						})
						filterMetricsByLabels(mf, map[string]string{
							"grpc_type": "unary",
						})
						groupCounterMetricsByLabels(mf, map[string]bool{
							"grpc_type": true,
							"grpc_code": true,
						})
						return mf, nil
					},
				},
			},

			// etcd 3.0 metric format for grpc request latencies,
			// rewritten to the etcd 3.1+ format.
			"etcd_grpc_unary_requests_duration_seconds": {
				rewriters: []rewriteFunc{
					func(mf *dto.MetricFamily) (*dto.MetricFamily, error) {
						mf = deepCopyMetricFamily(mf)
						renameMetric(mf, "grpc_server_handling_seconds")
						tpeName := "grpc_type"
						tpeVal := "unary"
						for _, m := range mf.Metric {
							m.Label = append(m.Label, &dto.LabelPair{Name: &tpeName, Value: &tpeVal})
						}
						return mf, nil
					},
				},
			},
			// etcd 3.1+ metric format for total grpc requests.
			"grpc_server_handling_seconds": {},
		},
	}
)

// monitorGatherer is a custom metric gatherer for prometheus that exports custom metrics
// defined by this monitor as well as rewritten etcd metrics.
type monitorGatherer struct {
	exported map[string]*exportedMetric
}

// exportedMetric identifies a metric that is exported and defines how it is rewritten before
// it is exported.
type exportedMetric struct {
	rewriters []rewriteFunc
}

// rewriteFunc rewrites metrics before they are exported.
type rewriteFunc func(mf *dto.MetricFamily) (*dto.MetricFamily, error)

func (m *monitorGatherer) Gather() ([]*dto.MetricFamily, error) {
	etcdMetrics, err := scrapeMetrics()
	if err != nil {
		return nil, err
	}
	exported, err := m.rewriteExportedMetrics(etcdMetrics)
	if err != nil {
		return nil, err
	}
	custom, err := customMetricRegistry.Gather()
	if err != nil {
		return nil, err
	}
	result := make([]*dto.MetricFamily, 0, len(exported)+len(custom))
	result = append(result, exported...)
	result = append(result, custom...)
	return result, nil
}

func (m *monitorGatherer) rewriteExportedMetrics(metrics map[string]*dto.MetricFamily) ([]*dto.MetricFamily, error) {
	results := make([]*dto.MetricFamily, 0, len(metrics))
	for n, mf := range metrics {
		if e, ok := m.exported[n]; ok {
			// Apply rewrite rules for metrics that have them.
			if e.rewriters == nil {
				results = append(results, mf)
			} else {
				for _, rewriter := range e.rewriters {
					new, err := rewriter(mf)
					if err != nil {
						return nil, err
					}
					results = append(results, new)
				}
			}
		} else {
			// Proxy all metrics without any rewrite rules directly.
			results = append(results, mf)
		}
	}
	return results, nil
}

// EtcdVersion struct for unmarshalling the json response from etcd's /version endpoint.
type EtcdVersion struct {
	BinaryVersion  string `json:"etcdserver"`
	ClusterVersion string `json:"etcdcluster"`
}

// Function for fetching etcd version info and feeding it to the prometheus metric.
func getVersion(lastSeenBinaryVersion *string) error {
	// Create the get request for the etcd version endpoint.
	req, err := http.NewRequest("GET", etcdVersionScrapeURI, nil)
	if err != nil {
		return fmt.Errorf("failed to create GET request for etcd version: %v", err)
	}

	// Send the get request and receive a response.
	client := &http.Client{}
	resp, err := client.Do(req)
	if err != nil {
		return fmt.Errorf("failed to receive GET response for etcd version: %v", err)
	}
	defer resp.Body.Close()

	// Obtain EtcdVersion from the JSON response.
	var version EtcdVersion
	if err := json.NewDecoder(resp.Body).Decode(&version); err != nil {
		return fmt.Errorf("failed to decode etcd version JSON: %v", err)
	}

	// Return without updating the version if it stayed the same since last time.
	if *lastSeenBinaryVersion == version.BinaryVersion {
		return nil
	}

	// Delete the metric for the previous version.
	if *lastSeenBinaryVersion != "" {
		deleted := etcdVersion.Delete(prometheus.Labels{"binary_version": *lastSeenBinaryVersion})
		if !deleted {
			return errors.New("failed to delete previous version's metric")
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
			klog.Errorf("Failed to fetch etcd version: %v", err)
		}
		select {
		case <-stopCh:
			break
		case <-time.After(scrapeTimeout):
		}
	}
}

// scrapeMetrics scrapes the prometheus metrics from the etcd metrics URI.
func scrapeMetrics() (map[string]*dto.MetricFamily, error) {
	req, err := http.NewRequest("GET", etcdMetricsScrapeURI, nil)
	if err != nil {
		return nil, fmt.Errorf("failed to create GET request for etcd metrics: %v", err)
	}

	// Send the get request and receive a response.
	client := &http.Client{}
	resp, err := client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("failed to receive GET response for etcd metrics: %v", err)
	}
	defer resp.Body.Close()

	// Parse the metrics in text format to a MetricFamily struct.
	var textParser expfmt.TextParser
	return textParser.TextToMetricFamilies(resp.Body)
}

func renameMetric(mf *dto.MetricFamily, name string) {
	mf.Name = &name
}

func renameLabels(mf *dto.MetricFamily, nameMapping map[string]string) {
	for _, m := range mf.Metric {
		for _, lbl := range m.Label {
			if alias, ok := nameMapping[*lbl.Name]; ok {
				lbl.Name = &alias
			}
		}
	}
}

func filterMetricsByLabels(mf *dto.MetricFamily, labelValues map[string]string) {
	buf := mf.Metric[:0]
	for _, m := range mf.Metric {
		shouldRemove := false
		for _, lbl := range m.Label {
			if val, ok := labelValues[*lbl.Name]; ok && val != *lbl.Value {
				shouldRemove = true
				break
			}
		}
		if !shouldRemove {
			buf = append(buf, m)
		}
	}
	mf.Metric = buf
}

func groupCounterMetricsByLabels(mf *dto.MetricFamily, names map[string]bool) {
	buf := mf.Metric[:0]
	deleteLabels(mf, names)
	byLabels := map[string]*dto.Metric{}
	for _, m := range mf.Metric {
		if metric, ok := byLabels[labelsKey(m.Label)]; ok {
			metric.Counter.Value = proto.Float64(*metric.Counter.Value + *m.Counter.Value)
		} else {
			byLabels[labelsKey(m.Label)] = m
			buf = append(buf, m)
		}
	}
	mf.Metric = buf
}

func labelsKey(lbls []*dto.LabelPair) string {
	var buf bytes.Buffer
	for i, lbl := range lbls {
		buf.WriteString(lbl.String())
		if i < len(lbls)-1 {
			buf.WriteString(",")
		}
	}
	return buf.String()
}

func deleteLabels(mf *dto.MetricFamily, names map[string]bool) {
	for _, m := range mf.Metric {
		buf := m.Label[:0]
		for _, lbl := range m.Label {
			shouldRemove := names[*lbl.Name]
			if !shouldRemove {
				buf = append(buf, lbl)
			}
		}
		m.Label = buf
	}
}

func identity(mf *dto.MetricFamily) (*dto.MetricFamily, error) {
	return mf, nil
}

func deepCopyMetricFamily(mf *dto.MetricFamily) *dto.MetricFamily {
	r := &dto.MetricFamily{}
	r.Name = mf.Name
	r.Help = mf.Help
	r.Type = mf.Type
	r.Metric = make([]*dto.Metric, len(mf.Metric))
	for i, m := range mf.Metric {
		r.Metric[i] = deepCopyMetric(m)
	}
	return r
}

func deepCopyMetric(m *dto.Metric) *dto.Metric {
	r := &dto.Metric{}
	r.Label = make([]*dto.LabelPair, len(m.Label))
	for i, lp := range m.Label {
		r.Label[i] = deepCopyLabelPair(lp)
	}
	r.Gauge = m.Gauge
	r.Counter = m.Counter
	r.Summary = m.Summary
	r.Untyped = m.Untyped
	r.Histogram = m.Histogram
	r.TimestampMs = m.TimestampMs
	return r
}

func deepCopyLabelPair(lp *dto.LabelPair) *dto.LabelPair {
	r := &dto.LabelPair{}
	r.Name = lp.Name
	r.Value = lp.Value
	return r
}

func main() {
	// Register the commandline flags passed to the tool.
	registerFlags(pflag.CommandLine)
	pflag.CommandLine.AddGoFlagSet(goflag.CommandLine)
	pflag.Parse()

	// Register the metrics we defined above with prometheus.
	customMetricRegistry.MustRegister(etcdVersion)
	customMetricRegistry.Unregister(prometheus.NewGoCollector())

	// Spawn threads for periodically scraping etcd version metrics.
	stopCh := make(chan struct{})
	defer close(stopCh)
	go getVersionPeriodically(stopCh)

	// Serve our metrics on listenAddress/metricsPath.
	klog.Infof("Listening on: %v", listenAddress)
	http.Handle(metricsPath, promhttp.HandlerFor(gatherer, promhttp.HandlerOpts{}))
	klog.Errorf("Stopped listening/serving metrics: %v", http.ListenAndServe(listenAddress, nil))
}
