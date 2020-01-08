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

package benchmark

import (
	"encoding/json"
	"flag"
	"fmt"
	"io/ioutil"
	"math"
	"path"
	"sort"
	"time"

	dto "github.com/prometheus/client_model/go"
	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime/schema"
	coreinformers "k8s.io/client-go/informers/core/v1"
	clientset "k8s.io/client-go/kubernetes"
	restclient "k8s.io/client-go/rest"
	"k8s.io/component-base/metrics/legacyregistry"
	"k8s.io/klog"
	"k8s.io/kubernetes/test/integration/util"
)

const (
	dateFormat                = "2006-01-02T15:04:05Z"
	throughputSampleFrequency = time.Second
)

var dataItemsDir = flag.String("data-items-dir", "", "destination directory for storing generated data items for perf dashboard")

// mustSetupScheduler starts the following components:
// - k8s api server (a.k.a. master)
// - scheduler
// It returns clientset and destroyFunc which should be used to
// remove resources after finished.
// Notes on rate limiter:
//   - client rate limit is set to 5000.
func mustSetupScheduler() (util.ShutdownFunc, coreinformers.PodInformer, clientset.Interface) {
	apiURL, apiShutdown := util.StartApiserver()
	clientSet := clientset.NewForConfigOrDie(&restclient.Config{
		Host:          apiURL,
		ContentConfig: restclient.ContentConfig{GroupVersion: &schema.GroupVersion{Group: "", Version: "v1"}},
		QPS:           5000.0,
		Burst:         5000,
	})
	_, podInformer, schedulerShutdown := util.StartScheduler(clientSet)

	shutdownFunc := func() {
		schedulerShutdown()
		apiShutdown()
	}

	return shutdownFunc, podInformer, clientSet
}

func getScheduledPods(podInformer coreinformers.PodInformer) ([]*v1.Pod, error) {
	pods, err := podInformer.Lister().List(labels.Everything())
	if err != nil {
		return nil, err
	}

	scheduled := make([]*v1.Pod, 0, len(pods))
	for i := range pods {
		pod := pods[i]
		if len(pod.Spec.NodeName) > 0 {
			scheduled = append(scheduled, pod)
		}
	}
	return scheduled, nil
}

// DataItem is the data point.
type DataItem struct {
	// Data is a map from bucket to real data point (e.g. "Perc90" -> 23.5). Notice
	// that all data items with the same label combination should have the same buckets.
	Data map[string]float64 `json:"data"`
	// Unit is the data unit. Notice that all data items with the same label combination
	// should have the same unit.
	Unit string `json:"unit"`
	// Labels is the labels of the data item.
	Labels map[string]string `json:"labels,omitempty"`
}

// DataItems is the data point set. It is the struct that perf dashboard expects.
type DataItems struct {
	Version   string     `json:"version"`
	DataItems []DataItem `json:"dataItems"`
}

func dataItems2JSONFile(dataItems DataItems, namePrefix string) error {
	b, err := json.Marshal(dataItems)
	if err != nil {
		return err
	}

	destFile := fmt.Sprintf("%v_%v.json", namePrefix, time.Now().Format(dateFormat))
	if *dataItemsDir != "" {
		destFile = path.Join(*dataItemsDir, destFile)
	}

	return ioutil.WriteFile(destFile, b, 0644)
}

// prometheusCollector collects metrics from legacyregistry.DefaultGatherer.Gather() endpoint.
// Currently only Histrogram metrics are supported.
type prometheusCollector struct {
	metric string
	cache  *dto.MetricFamily
}

func newPrometheusCollector(metric string) *prometheusCollector {
	return &prometheusCollector{
		metric: metric,
	}
}

func (pc *prometheusCollector) collect() *DataItem {
	var metricFamily *dto.MetricFamily
	m, err := legacyregistry.DefaultGatherer.Gather()
	if err != nil {
		klog.Error(err)
		return nil
	}
	for _, mFamily := range m {
		if mFamily.Name != nil && *mFamily.Name == pc.metric {
			metricFamily = mFamily
			break
		}
	}

	if metricFamily == nil {
		klog.Infof("Metric %q not found", pc.metric)
		return nil
	}

	if metricFamily.GetMetric() == nil {
		klog.Infof("Metric %q is empty", pc.metric)
		return nil
	}

	if len(metricFamily.GetMetric()) == 0 {
		klog.Infof("Metric %q is empty", pc.metric)
		return nil
	}

	// Histograms are stored under the first index (based on observation).
	// Given there's only one histogram registered per each metric name, accessaing
	// the first index is sufficient.
	dataItem := pc.promHist2Summary(metricFamily.GetMetric()[0].GetHistogram())
	if dataItem.Data == nil {
		return nil
	}

	// clear the metrics so that next test always starts with empty prometheus
	// metrics (since the metrics are shared among all tests run inside the same binary)
	clearPromHistogram(metricFamily.GetMetric()[0].GetHistogram())

	return dataItem
}

// Bucket of a histogram
type bucket struct {
	upperBound float64
	count      float64
}

func bucketQuantile(q float64, buckets []bucket) float64 {
	if q < 0 {
		return math.Inf(-1)
	}
	if q > 1 {
		return math.Inf(+1)
	}

	if len(buckets) < 2 {
		return math.NaN()
	}

	rank := q * buckets[len(buckets)-1].count
	b := sort.Search(len(buckets)-1, func(i int) bool { return buckets[i].count >= rank })

	if b == 0 {
		return buckets[0].upperBound * (rank / buckets[0].count)
	}

	// linear approximation of b-th bucket
	brank := rank - buckets[b-1].count
	bSize := buckets[b].upperBound - buckets[b-1].upperBound
	bCount := buckets[b].count - buckets[b-1].count

	return buckets[b-1].upperBound + bSize*(brank/bCount)
}

func (pc *prometheusCollector) promHist2Summary(hist *dto.Histogram) *DataItem {
	buckets := []bucket{}

	if hist.SampleCount == nil || *hist.SampleCount == 0 {
		return &DataItem{}
	}

	if hist.SampleSum == nil || *hist.SampleSum == 0 {
		return &DataItem{}
	}

	for _, bckt := range hist.Bucket {
		if bckt == nil {
			return &DataItem{}
		}
		if bckt.UpperBound == nil || *bckt.UpperBound < 0 {
			return &DataItem{}
		}
		buckets = append(buckets, bucket{
			count:      float64(*bckt.CumulativeCount),
			upperBound: *bckt.UpperBound,
		})
	}

	// bucketQuantile expects the upper bound of the last bucket to be +inf
	buckets[len(buckets)-1].upperBound = math.Inf(+1)

	q50 := bucketQuantile(0.50, buckets)
	q90 := bucketQuantile(0.90, buckets)
	q99 := bucketQuantile(0.95, buckets)

	msFactor := float64(time.Second) / float64(time.Millisecond)

	return &DataItem{
		Labels: map[string]string{
			"Metric": pc.metric,
		},
		Data: map[string]float64{
			"Perc50":  q50 * msFactor,
			"Perc90":  q90 * msFactor,
			"Perc99":  q99 * msFactor,
			"Average": (*hist.SampleSum / float64(*hist.SampleCount)) * msFactor,
		},
		Unit: "ms",
	}
}

func clearPromHistogram(hist *dto.Histogram) {
	if hist.SampleCount != nil {
		*hist.SampleCount = 0
	}
	if hist.SampleSum != nil {
		*hist.SampleSum = 0
	}
	for _, b := range hist.Bucket {
		if b.CumulativeCount != nil {
			*b.CumulativeCount = 0
		}
		if b.UpperBound != nil {
			*b.UpperBound = 0
		}
	}
}

type throughputCollector struct {
	podInformer           coreinformers.PodInformer
	schedulingThroughputs []float64
}

func newThroughputCollector(podInformer coreinformers.PodInformer) *throughputCollector {
	return &throughputCollector{
		podInformer: podInformer,
	}
}

func (tc *throughputCollector) run(stopCh chan struct{}) {
	podsScheduled, err := getScheduledPods(tc.podInformer)
	if err != nil {
		klog.Fatalf("%v", err)
	}
	lastScheduledCount := len(podsScheduled)
	for {
		select {
		case <-stopCh:
			return
		case <-time.After(throughputSampleFrequency):
			podsScheduled, err := getScheduledPods(tc.podInformer)
			if err != nil {
				klog.Fatalf("%v", err)
			}

			scheduled := len(podsScheduled)
			samplingRatioSeconds := float64(throughputSampleFrequency) / float64(time.Second)
			throughput := float64(scheduled-lastScheduledCount) / samplingRatioSeconds
			tc.schedulingThroughputs = append(tc.schedulingThroughputs, throughput)
			lastScheduledCount = scheduled

			klog.Infof("%d pods scheduled", lastScheduledCount)
		}
	}
}

func (tc *throughputCollector) collect() *DataItem {
	throughputSummary := &DataItem{}
	if length := len(tc.schedulingThroughputs); length > 0 {
		sort.Float64s(tc.schedulingThroughputs)
		sum := 0.0
		for i := range tc.schedulingThroughputs {
			sum += tc.schedulingThroughputs[i]
		}

		throughputSummary.Labels = map[string]string{
			"Metric": "SchedulingThroughput",
		}
		throughputSummary.Data = map[string]float64{
			"Average": sum / float64(length),
			"Perc50":  tc.schedulingThroughputs[int(math.Ceil(float64(length*50)/100))-1],
			"Perc90":  tc.schedulingThroughputs[int(math.Ceil(float64(length*90)/100))-1],
			"Perc99":  tc.schedulingThroughputs[int(math.Ceil(float64(length*99)/100))-1],
		}
		throughputSummary.Unit = "pods/s"
	}
	return throughputSummary
}
