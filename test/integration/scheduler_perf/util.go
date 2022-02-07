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
	"bytes"
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"io/ioutil"
	"math"
	"os"
	"path"
	"sort"
	"time"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/client-go/dynamic"
	coreinformers "k8s.io/client-go/informers/core/v1"
	clientset "k8s.io/client-go/kubernetes"
	restclient "k8s.io/client-go/rest"
	"k8s.io/component-base/metrics/legacyregistry"
	"k8s.io/component-base/metrics/testutil"
	"k8s.io/klog/v2"
	"k8s.io/kube-scheduler/config/v1beta2"
	"k8s.io/kubernetes/pkg/scheduler/apis/config"
	kubeschedulerscheme "k8s.io/kubernetes/pkg/scheduler/apis/config/scheme"
	"k8s.io/kubernetes/test/integration/framework"
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

func newDefaultComponentConfig() (*config.KubeSchedulerConfiguration, error) {
	gvk := v1beta2.SchemeGroupVersion.WithKind("KubeSchedulerConfiguration")
	cfg := config.KubeSchedulerConfiguration{}
	_, _, err := kubeschedulerscheme.Codecs.UniversalDecoder().Decode(nil, &gvk, &cfg)
	if err != nil {
		return nil, err
	}
	return &cfg, nil
}

// mustSetupScheduler starts the following components:
// - k8s api server
// - scheduler
// It returns regular and dynamic clients, and destroyFunc which should be used to
// remove resources after finished.
// Notes on rate limiter:
//   - client rate limit is set to 5000.
func mustSetupScheduler(config *config.KubeSchedulerConfiguration) (util.ShutdownFunc, coreinformers.PodInformer, clientset.Interface, dynamic.Interface) {
	// Run API server with minimimal logging by default. Can be raised with -v.
	framework.MinVerbosity = 0
	apiURL, apiShutdown := util.StartApiserver()
	var err error

	// TODO: client connection configuration, such as QPS or Burst is configurable in theory, this could be derived from the `config`, need to
	// support this when there is any testcase that depends on such configuration.
	cfg := &restclient.Config{
		Host:          apiURL,
		ContentConfig: restclient.ContentConfig{GroupVersion: &schema.GroupVersion{Group: "", Version: "v1"}},
		QPS:           5000.0,
		Burst:         5000,
	}

	// use default component config if config here is nil
	if config == nil {
		config, err = newDefaultComponentConfig()
		if err != nil {
			klog.Fatalf("Error creating default component config: %v", err)
		}
	}

	client := clientset.NewForConfigOrDie(cfg)
	dynClient := dynamic.NewForConfigOrDie(cfg)

	// Not all config options will be effective but only those mostly related with scheduler performance will
	// be applied to start a scheduler, most of them are defined in `scheduler.schedulerOptions`.
	_, podInformer, schedulerShutdown := util.StartScheduler(client, cfg, config)
	fakePVControllerShutdown := util.StartFakePVController(client)

	shutdownFunc := func() {
		fakePVControllerShutdown()
		schedulerShutdown()
		apiShutdown()
	}

	return shutdownFunc, podInformer, client, dynClient
}

// Returns the list of scheduled pods in the specified namespaces.
// Note that no namespaces specified matches all namespaces.
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
		// Ensure the "dataItemsDir" path to be valid.
		if err := os.MkdirAll(*dataItemsDir, 0750); err != nil {
			return fmt.Errorf("dataItemsDir path %v does not exist and cannot be created: %v", *dataItemsDir, err)
		}
		destFile = path.Join(*dataItemsDir, destFile)
	}
	formatted := &bytes.Buffer{}
	if err := json.Indent(formatted, b, "", "  "); err != nil {
		return fmt.Errorf("indenting error: %v", err)
	}
	return ioutil.WriteFile(destFile, formatted.Bytes(), 0644)
}

type labelValues struct {
	label  string
	values []string
}

// metricsCollectorConfig is the config to be marshalled to YAML config file.
// NOTE: The mapping here means only one filter is supported, either value in the list of `values` is able to be collected.
type metricsCollectorConfig struct {
	Metrics map[string]*labelValues
}

// metricsCollector collects metrics from legacyregistry.DefaultGatherer.Gather() endpoint.
// Currently only Histogram metrics are supported.
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
	for metric, labelVals := range pc.Metrics {
		// no filter is specified, aggregate all the metrics within the same metricFamily.
		if labelVals == nil {
			dataItem := collectHistogramVec(metric, pc.labels, nil)
			if dataItem != nil {
				dataItems = append(dataItems, *dataItem)
			}
		} else {
			// fetch the metric from metricFamily which match each of the lvMap.
			for _, value := range labelVals.values {
				lvMap := map[string]string{labelVals.label: value}
				dataItem := collectHistogramVec(metric, pc.labels, lvMap)
				if dataItem != nil {
					dataItems = append(dataItems, *dataItem)
				}
			}
		}
	}
	return dataItems
}

func collectHistogramVec(metric string, labels map[string]string, lvMap map[string]string) *DataItem {
	vec, err := testutil.GetHistogramVecFromGatherer(legacyregistry.DefaultGatherer, metric, lvMap)
	if err != nil {
		klog.Error(err)
		return nil
	}

	if err := vec.Validate(); err != nil {
		klog.Error(err)
		return nil
	}

	q50 := vec.Quantile(0.50)
	q90 := vec.Quantile(0.90)
	q95 := vec.Quantile(0.95)
	q99 := vec.Quantile(0.99)
	avg := vec.Average()

	msFactor := float64(time.Second) / float64(time.Millisecond)

	// Copy labels and add "Metric" label for this metric.
	labelMap := map[string]string{"Metric": metric}
	for k, v := range labels {
		labelMap[k] = v
	}
	for k, v := range lvMap {
		labelMap[k] = v
	}
	return &DataItem{
		Labels: labelMap,
		Data: map[string]float64{
			"Perc50":  q50 * msFactor,
			"Perc90":  q90 * msFactor,
			"Perc95":  q95 * msFactor,
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
			// Only do sampling if number of scheduled pods is greater than zero
			if scheduled > 0 {
				samplingRatioSeconds := float64(throughputSampleFrequency) / float64(time.Second)
				throughput := float64(scheduled-lastScheduledCount) / samplingRatioSeconds
				tc.schedulingThroughputs = append(tc.schedulingThroughputs, throughput)
				lastScheduledCount = scheduled
				klog.Infof("%d pods scheduled", lastScheduledCount)
			}

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
			"Perc95":  tc.schedulingThroughputs[int(math.Ceil(float64(length*95)/100))-1],
			"Perc99":  tc.schedulingThroughputs[int(math.Ceil(float64(length*99)/100))-1],
		}
		throughputSummary.Unit = "pods/s"
	}

	return []DataItem{throughputSummary}
}
