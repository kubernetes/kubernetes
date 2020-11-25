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
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"io/ioutil"
	"math"
	"path"
	"sort"
	"time"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/sets"
	coreinformers "k8s.io/client-go/informers/core/v1"
	clientset "k8s.io/client-go/kubernetes"
	restclient "k8s.io/client-go/rest"
	"k8s.io/component-base/metrics/legacyregistry"
	"k8s.io/component-base/metrics/testutil"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/test/integration/util"
	testutils "k8s.io/kubernetes/test/utils"
)

const (
	dateFormat                = "2006-01-02T15:04:05Z"
	testNamespace             = "sched-test"
	setupNamespace            = "sched-setup"
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
	fakePVControllerShutdown := util.StartFakePVController(clientSet)

	shutdownFunc := func() {
		fakePVControllerShutdown()
		schedulerShutdown()
		apiShutdown()
	}

	return shutdownFunc, podInformer, clientSet
}

// Returns the list of scheduled pods in the specified namespaces.
// Note that no namespces specified matches all namespaces.
func getScheduledPods(podInformer coreinformers.PodInformer, namespaces ...string) ([]*v1.Pod, error) {
	pods, err := podInformer.Lister().List(labels.Everything())
	if err != nil {
		return nil, err
	}

	s := sets.NewString(namespaces...)
	scheduled := make([]*v1.Pod, 0, len(pods))
	for i := range pods {
		pod := pods[i]
		if len(pod.Spec.NodeName) > 0 && (len(s) == 0 || s.Has(pod.Namespace)) {
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

// makeBasePod creates a Pod object to be used as a template.
func makeBasePod() *v1.Pod {
	basePod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			GenerateName: "pod-",
		},
		Spec: testutils.MakePodSpec(),
	}
	return basePod
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

// metricsCollectorConfig is the config to be marshalled to YAML config file.
type metricsCollectorConfig struct {
	Metrics []string
}

// metricsCollector collects metrics from legacyregistry.DefaultGatherer.Gather() endpoint.
// Currently only Histrogram metrics are supported.
type metricsCollector struct {
	*metricsCollectorConfig
	labels map[string]string
}

func newMetricsCollector(config *metricsCollectorConfig, labels map[string]string) *metricsCollector {
	return &metricsCollector{
		metricsCollectorConfig: config,
		labels:                 labels,
	}
}

func (*metricsCollector) run(ctx context.Context) {
	// metricCollector doesn't need to start before the tests, so nothing to do here.
}

func (pc *metricsCollector) collect() []DataItem {
	var dataItems []DataItem
	for _, metric := range pc.Metrics {
		dataItem := collectHistogram(metric, pc.labels)
		if dataItem != nil {
			dataItems = append(dataItems, *dataItem)
		}
	}
	return dataItems
}

func collectHistogram(metric string, labels map[string]string) *DataItem {
	hist, err := testutil.GetHistogramFromGatherer(legacyregistry.DefaultGatherer, metric)
	if err != nil {
		klog.Error(err)
		return nil
	}
	if hist.Histogram == nil {
		klog.Errorf("metric %q is not a Histogram metric", metric)
		return nil
	}
	if err := hist.Validate(); err != nil {
		klog.Error(err)
		return nil
	}

	q50 := hist.Quantile(0.50)
	q90 := hist.Quantile(0.90)
	q99 := hist.Quantile(0.95)
	avg := hist.Average()

	// clear the metrics so that next test always starts with empty prometheus
	// metrics (since the metrics are shared among all tests run inside the same binary)
	hist.Clear()

	msFactor := float64(time.Second) / float64(time.Millisecond)

	// Copy labels and add "Metric" label for this metric.
	labelMap := map[string]string{"Metric": metric}
	for k, v := range labels {
		labelMap[k] = v
	}
	return &DataItem{
		Labels: labelMap,
		Data: map[string]float64{
			"Perc50":  q50 * msFactor,
			"Perc90":  q90 * msFactor,
			"Perc99":  q99 * msFactor,
			"Average": avg * msFactor,
		},
		Unit: "ms",
	}
}

type throughputCollector struct {
	podInformer           coreinformers.PodInformer
	schedulingThroughputs []float64
	labels                map[string]string
	namespaces            []string
}

func newThroughputCollector(podInformer coreinformers.PodInformer, labels map[string]string, namespaces []string) *throughputCollector {
	return &throughputCollector{
		podInformer: podInformer,
		labels:      labels,
		namespaces:  namespaces,
	}
}

func (tc *throughputCollector) run(ctx context.Context) {
	podsScheduled, err := getScheduledPods(tc.podInformer, tc.namespaces...)
	if err != nil {
		klog.Fatalf("%v", err)
	}
	lastScheduledCount := len(podsScheduled)
	ticker := time.NewTicker(throughputSampleFrequency)
	defer ticker.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			podsScheduled, err := getScheduledPods(tc.podInformer, tc.namespaces...)
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

func (tc *throughputCollector) collect() []DataItem {
	throughputSummary := DataItem{Labels: tc.labels}
	if length := len(tc.schedulingThroughputs); length > 0 {
		sort.Float64s(tc.schedulingThroughputs)
		sum := 0.0
		for i := range tc.schedulingThroughputs {
			sum += tc.schedulingThroughputs[i]
		}

		throughputSummary.Labels["Metric"] = "SchedulingThroughput"
		throughputSummary.Data = map[string]float64{
			"Average": sum / float64(length),
			"Perc50":  tc.schedulingThroughputs[int(math.Ceil(float64(length*50)/100))-1],
			"Perc90":  tc.schedulingThroughputs[int(math.Ceil(float64(length*90)/100))-1],
			"Perc99":  tc.schedulingThroughputs[int(math.Ceil(float64(length*99)/100))-1],
		}
		throughputSummary.Unit = "pods/s"
	}

	return []DataItem{throughputSummary}
}
