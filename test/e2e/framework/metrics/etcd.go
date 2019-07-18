/*
Copyright 2019 The Kubernetes Authors.

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
	"fmt"
	"io"
	"math"
	"reflect"
	"strings"
	"sync"
	"time"

	e2elog "k8s.io/kubernetes/test/e2e/framework/log"
	e2essh "k8s.io/kubernetes/test/e2e/framework/ssh"

	"github.com/prometheus/common/expfmt"
	"github.com/prometheus/common/model"
)

// Histogram is a struct for managing histogram.
type Histogram struct {
	Labels  map[string]string `json:"labels"`
	Buckets map[string]int    `json:"buckets"`
}

// HistogramVec is an array of Histogram.
type HistogramVec []Histogram

func newHistogram(labels map[string]string) *Histogram {
	return &Histogram{
		Labels:  labels,
		Buckets: make(map[string]int),
	}
}

// EtcdMetrics is a struct for managing etcd metrics.
type EtcdMetrics struct {
	BackendCommitDuration     HistogramVec `json:"backendCommitDuration"`
	SnapshotSaveTotalDuration HistogramVec `json:"snapshotSaveTotalDuration"`
	PeerRoundTripTime         HistogramVec `json:"peerRoundTripTime"`
	WalFsyncDuration          HistogramVec `json:"walFsyncDuration"`
	MaxDatabaseSize           float64      `json:"maxDatabaseSize"`
}

func newEtcdMetrics() *EtcdMetrics {
	return &EtcdMetrics{
		BackendCommitDuration:     make(HistogramVec, 0),
		SnapshotSaveTotalDuration: make(HistogramVec, 0),
		PeerRoundTripTime:         make(HistogramVec, 0),
		WalFsyncDuration:          make(HistogramVec, 0),
	}
}

// SummaryKind returns the summary of etcd metrics.
func (l *EtcdMetrics) SummaryKind() string {
	return "EtcdMetrics"
}

// PrintHumanReadable returns etcd metrics with JSON format.
func (l *EtcdMetrics) PrintHumanReadable() string {
	return PrettyPrintJSON(l)
}

// PrintJSON returns etcd metrics with JSON format.
func (l *EtcdMetrics) PrintJSON() string {
	return PrettyPrintJSON(l)
}

// EtcdMetricsCollector is a struct for managing etcd metrics collector.
type EtcdMetricsCollector struct {
	stopCh  chan struct{}
	wg      *sync.WaitGroup
	metrics *EtcdMetrics
}

// NewEtcdMetricsCollector creates a new etcd metrics collector.
func NewEtcdMetricsCollector() *EtcdMetricsCollector {
	return &EtcdMetricsCollector{
		stopCh:  make(chan struct{}),
		wg:      &sync.WaitGroup{},
		metrics: newEtcdMetrics(),
	}
}

// extractMetricSamples parses the prometheus metric samples from the input string.
func extractMetricSamples(metricsBlob string) ([]*model.Sample, error) {
	dec := expfmt.NewDecoder(strings.NewReader(metricsBlob), expfmt.FmtText)
	decoder := expfmt.SampleDecoder{
		Dec:  dec,
		Opts: &expfmt.DecodeOptions{},
	}

	var samples []*model.Sample
	for {
		var v model.Vector
		if err := decoder.Decode(&v); err != nil {
			if err == io.EOF {
				// Expected loop termination condition.
				return samples, nil
			}
			return nil, err
		}
		samples = append(samples, v...)
	}
}

func getEtcdMetrics(provider string, masterHostname string) ([]*model.Sample, error) {
	// Etcd is only exposed on localhost level. We are using ssh method
	if provider == "gke" || provider == "eks" {
		e2elog.Logf("Not grabbing etcd metrics through master SSH: unsupported for %s", provider)
		return nil, nil
	}

	cmd := "curl http://localhost:2379/metrics"
	sshResult, err := e2essh.SSH(cmd, masterHostname+":22", provider)
	if err != nil || sshResult.Code != 0 {
		return nil, fmt.Errorf("unexpected error (code: %d) in ssh connection to master: %#v", sshResult.Code, err)
	}
	data := sshResult.Stdout

	return extractMetricSamples(data)
}

func getEtcdDatabaseSize(provider string, masterHostname string) (float64, error) {
	samples, err := getEtcdMetrics(provider, masterHostname)
	if err != nil {
		return 0, err
	}
	for _, sample := range samples {
		if sample.Metric[model.MetricNameLabel] == "etcd_debugging_mvcc_db_total_size_in_bytes" {
			return float64(sample.Value), nil
		}
	}
	return 0, fmt.Errorf("Couldn't find etcd database size metric")
}

// StartCollecting starts to collect etcd db size metric periodically
// and updates MaxDatabaseSize accordingly.
func (mc *EtcdMetricsCollector) StartCollecting(interval time.Duration, provider string, masterHostname string) {
	mc.wg.Add(1)
	go func() {
		defer mc.wg.Done()
		for {
			select {
			case <-time.After(interval):
				dbSize, err := getEtcdDatabaseSize(provider, masterHostname)
				if err != nil {
					e2elog.Logf("Failed to collect etcd database size")
					continue
				}
				mc.metrics.MaxDatabaseSize = math.Max(mc.metrics.MaxDatabaseSize, dbSize)
			case <-mc.stopCh:
				return
			}
		}
	}()
}

func convertSampleToBucket(sample *model.Sample, h *HistogramVec) {
	labels := make(map[string]string)
	for k, v := range sample.Metric {
		if k != "le" {
			labels[string(k)] = string(v)
		}
	}
	var hist *Histogram
	for i := range *h {
		if reflect.DeepEqual(labels, (*h)[i].Labels) {
			hist = &((*h)[i])
			break
		}
	}
	if hist == nil {
		hist = newHistogram(labels)
		*h = append(*h, *hist)
	}
	hist.Buckets[string(sample.Metric["le"])] = int(sample.Value)
}

// StopAndSummarize stops etcd metrics collector and summarizes the metrics.
func (mc *EtcdMetricsCollector) StopAndSummarize(provider string, masterHostname string) error {
	close(mc.stopCh)
	mc.wg.Wait()

	// Do some one-off collection of metrics.
	samples, err := getEtcdMetrics(provider, masterHostname)
	if err != nil {
		return err
	}
	for _, sample := range samples {
		switch sample.Metric[model.MetricNameLabel] {
		case "etcd_disk_backend_commit_duration_seconds_bucket":
			convertSampleToBucket(sample, &mc.metrics.BackendCommitDuration)
		case "etcd_debugging_snap_save_total_duration_seconds_bucket":
			convertSampleToBucket(sample, &mc.metrics.SnapshotSaveTotalDuration)
		case "etcd_disk_wal_fsync_duration_seconds_bucket":
			convertSampleToBucket(sample, &mc.metrics.WalFsyncDuration)
		case "etcd_network_peer_round_trip_time_seconds_bucket":
			convertSampleToBucket(sample, &mc.metrics.PeerRoundTripTime)
		}
	}
	return nil
}

// GetMetrics returns metrics of etcd metrics collector.
func (mc *EtcdMetricsCollector) GetMetrics() *EtcdMetrics {
	return mc.metrics
}
