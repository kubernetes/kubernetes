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
	"encoding/json"
	"flag"
	"fmt"
	"math"
	"os"
	"path"
	"sort"
	"strings"
	"time"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/client-go/informers"
	coreinformers "k8s.io/client-go/informers/core/v1"
	restclient "k8s.io/client-go/rest"
	"k8s.io/component-base/featuregate"
	"k8s.io/component-base/metrics/legacyregistry"
	"k8s.io/component-base/metrics/testutil"
	"k8s.io/klog/v2"
	kubeschedulerconfigv1 "k8s.io/kube-scheduler/config/v1"
	apiservertesting "k8s.io/kubernetes/cmd/kube-apiserver/app/testing"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/scheduler/apis/config"
	kubeschedulerscheme "k8s.io/kubernetes/pkg/scheduler/apis/config/scheme"
	frameworkruntime "k8s.io/kubernetes/pkg/scheduler/framework/runtime"
	"k8s.io/kubernetes/test/integration/framework"
	"k8s.io/kubernetes/test/integration/util"
	testutils "k8s.io/kubernetes/test/utils"
	"k8s.io/kubernetes/test/utils/ktesting"
)

const (
	dateFormat               = "2006-01-02T15:04:05Z"
	testNamespace            = "sched-test"
	setupNamespace           = "sched-setup"
	throughputSampleInterval = time.Second
)

var dataItemsDir = flag.String("data-items-dir", "", "destination directory for storing generated data items for perf dashboard")

func newDefaultComponentConfig() (*config.KubeSchedulerConfiguration, error) {
	gvk := kubeschedulerconfigv1.SchemeGroupVersion.WithKind("KubeSchedulerConfiguration")
	cfg := config.KubeSchedulerConfiguration{}
	_, _, err := kubeschedulerscheme.Codecs.UniversalDecoder().Decode(nil, &gvk, &cfg)
	if err != nil {
		return nil, err
	}
	return &cfg, nil
}

// mustSetupCluster starts the following components:
// - k8s api server
// - scheduler
// - some of the kube-controller-manager controllers
//
// It returns regular and dynamic clients, and destroyFunc which should be used to
// remove resources after finished.
// Notes on rate limiter:
//   - client rate limit is set to 5000.
func mustSetupCluster(tCtx ktesting.TContext, config *config.KubeSchedulerConfiguration, enabledFeatures map[featuregate.Feature]bool, outOfTreePluginRegistry frameworkruntime.Registry) (informers.SharedInformerFactory, ktesting.TContext) {
	// No alpha APIs (overrides api/all=true in https://github.com/kubernetes/kubernetes/blob/d647d19f6aef811bace300eec96a67644ff303d4/staging/src/k8s.io/apiextensions-apiserver/pkg/cmd/server/testing/testserver.go#L136),
	// except for DRA API group when needed.
	runtimeConfig := []string{"api/alpha=false"}
	if enabledFeatures[features.DynamicResourceAllocation] {
		runtimeConfig = append(runtimeConfig, "resource.k8s.io/v1alpha3=true")
	}
	customFlags := []string{
		// Disable ServiceAccount admission plugin as we don't have serviceaccount controller running.
		"--disable-admission-plugins=ServiceAccount,TaintNodesByCondition,Priority",
		"--runtime-config=" + strings.Join(runtimeConfig, ","),
	}
	serverOpts := apiservertesting.NewDefaultTestServerOptions()
	// Timeout sufficiently long to handle deleting pods of the largest test cases.
	serverOpts.RequestTimeout = 10 * time.Minute
	server, err := apiservertesting.StartTestServer(tCtx, serverOpts, customFlags, framework.SharedEtcd())
	if err != nil {
		tCtx.Fatalf("start apiserver: %v", err)
	}
	// Cleanup will be in reverse order: first the clients by canceling the
	// child context (happens automatically), then the server.
	tCtx.Cleanup(server.TearDownFn)
	tCtx = ktesting.WithCancel(tCtx)

	// TODO: client connection configuration, such as QPS or Burst is configurable in theory, this could be derived from the `config`, need to
	// support this when there is any testcase that depends on such configuration.
	cfg := restclient.CopyConfig(server.ClientConfig)
	cfg.QPS = 5000.0
	cfg.Burst = 5000

	// use default component config if config here is nil
	if config == nil {
		var err error
		config, err = newDefaultComponentConfig()
		if err != nil {
			tCtx.Fatalf("Error creating default component config: %v", err)
		}
	}

	tCtx = ktesting.WithRESTConfig(tCtx, cfg)

	// Not all config options will be effective but only those mostly related with scheduler performance will
	// be applied to start a scheduler, most of them are defined in `scheduler.schedulerOptions`.
	_, informerFactory := util.StartScheduler(tCtx, tCtx.Client(), cfg, config, outOfTreePluginRegistry)
	util.StartFakePVController(tCtx, tCtx.Client(), informerFactory)
	runGC := util.CreateGCController(tCtx, tCtx, *cfg, informerFactory)
	runNS := util.CreateNamespaceController(tCtx, tCtx, *cfg, informerFactory)

	runResourceClaimController := func() {}
	if enabledFeatures[features.DynamicResourceAllocation] {
		// Testing of DRA with inline resource claims depends on this
		// controller for creating and removing ResourceClaims.
		runResourceClaimController = util.CreateResourceClaimController(tCtx, tCtx, tCtx.Client(), informerFactory)
	}

	informerFactory.Start(tCtx.Done())
	informerFactory.WaitForCacheSync(tCtx.Done())
	go runGC()
	go runNS()
	go runResourceClaimController()

	return informerFactory, tCtx
}

// Returns the list of scheduled and unscheduled pods in the specified namespaces.
// Note that no namespaces specified matches all namespaces.
func getScheduledPods(podInformer coreinformers.PodInformer, namespaces ...string) ([]*v1.Pod, []*v1.Pod, error) {
	pods, err := podInformer.Lister().List(labels.Everything())
	if err != nil {
		return nil, nil, err
	}

	s := sets.New(namespaces...)
	scheduled := make([]*v1.Pod, 0, len(pods))
	unscheduled := make([]*v1.Pod, 0, len(pods))
	for i := range pods {
		pod := pods[i]
		if len(s) == 0 || s.Has(pod.Namespace) {
			if len(pod.Spec.NodeName) > 0 {
				scheduled = append(scheduled, pod)
			} else {
				unscheduled = append(unscheduled, pod)
			}
		}
	}
	return scheduled, unscheduled, nil
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
	// perfdash expects all data items to have the same set of labels.  It
	// then renders drop-down buttons for each label with all values found
	// for each label. If we were to store data items that don't have a
	// certain label, then perfdash will never show those data items
	// because it will only show data items that have the currently
	// selected label value. To avoid that, we collect all labels used
	// anywhere and then add missing labels with "not applicable" as value.
	labels := sets.New[string]()
	for _, item := range dataItems.DataItems {
		for label := range item.Labels {
			labels.Insert(label)
		}
	}
	for _, item := range dataItems.DataItems {
		for label := range labels {
			if _, ok := item.Labels[label]; !ok {
				item.Labels[label] = "not applicable"
			}
		}
	}

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
	return os.WriteFile(destFile, formatted.Bytes(), 0644)
}

type labelValues struct {
	label  string
	values []string
}

// metricsCollectorConfig is the config to be marshalled to YAML config file.
// NOTE: The mapping here means only one filter is supported, either value in the list of `values` is able to be collected.
type metricsCollectorConfig struct {
	Metrics map[string][]*labelValues
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

func (mc *metricsCollector) init() error {
	// Reset the metrics so that the measurements do not interfere with those collected during the previous steps.
	m, err := legacyregistry.DefaultGatherer.Gather()
	if err != nil {
		return fmt.Errorf("failed to gather metrics to reset: %w", err)
	}
	for _, mFamily := range m {
		// Reset only metrics defined in the collector.
		if _, ok := mc.Metrics[mFamily.GetName()]; ok {
			mFamily.Reset()
		}
	}
	return nil
}

func (*metricsCollector) run(tCtx ktesting.TContext) {
	// metricCollector doesn't need to start before the tests, so nothing to do here.
}

func (mc *metricsCollector) collect() []DataItem {
	var dataItems []DataItem
	for metric, labelValsSlice := range mc.Metrics {
		// no filter is specified, aggregate all the metrics within the same metricFamily.
		if labelValsSlice == nil {
			dataItem := collectHistogramVec(metric, mc.labels, nil)
			if dataItem != nil {
				dataItems = append(dataItems, *dataItem)
			}
		} else {
			for _, lvMap := range uniqueLVCombos(labelValsSlice) {
				dataItem := collectHistogramVec(metric, mc.labels, lvMap)
				if dataItem != nil {
					dataItems = append(dataItems, *dataItem)
				}
			}
		}
	}
	return dataItems
}

// uniqueLVCombos lists up all possible label values combinations.
// e.g., if there are 3 labelValues, each of which has 2 values,
// the result would be {A: a1, B: b1, C: c1}, {A: a2, B: b1, C: c1}, {A: a1, B: b2, C: c1}, ... (2^3 = 8 combinations).
func uniqueLVCombos(lvs []*labelValues) []map[string]string {
	if len(lvs) == 0 {
		return []map[string]string{{}}
	}

	remainingCombos := uniqueLVCombos(lvs[1:])

	results := make([]map[string]string, 0)

	current := lvs[0]
	for _, value := range current.values {
		for _, combo := range remainingCombos {
			newCombo := make(map[string]string, len(combo)+1)
			for k, v := range combo {
				newCombo[k] = v
			}
			newCombo[current.label] = value
			results = append(results, newCombo)
		}
	}
	return results
}

func collectHistogramVec(metric string, labels map[string]string, lvMap map[string]string) *DataItem {
	vec, err := testutil.GetHistogramVecFromGatherer(legacyregistry.DefaultGatherer, metric, lvMap)
	if err != nil {
		klog.Error(err)
		return nil
	}

	if err := vec.Validate(); err != nil {
		klog.ErrorS(err, "the validation for HistogramVec is failed. The data for this metric won't be stored in a benchmark result file", "metric", metric, "labels", labels)
		return nil
	}

	if vec.GetAggregatedSampleCount() == 0 {
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
	errorMargin           float64
}

func newThroughputCollector(podInformer coreinformers.PodInformer, labels map[string]string, namespaces []string, errorMargin float64) *throughputCollector {
	return &throughputCollector{
		podInformer: podInformer,
		labels:      labels,
		namespaces:  namespaces,
		errorMargin: errorMargin,
	}
}

func (tc *throughputCollector) init() error {
	return nil
}

func (tc *throughputCollector) run(tCtx ktesting.TContext) {
	podsScheduled, _, err := getScheduledPods(tc.podInformer, tc.namespaces...)
	if err != nil {
		klog.Fatalf("%v", err)
	}
	lastScheduledCount := len(podsScheduled)
	ticker := time.NewTicker(throughputSampleInterval)
	defer ticker.Stop()
	lastSampleTime := time.Now()
	started := false
	skipped := 0

	for {
		select {
		case <-tCtx.Done():
			return
		case <-ticker.C:
			now := time.Now()
			podsScheduled, _, err := getScheduledPods(tc.podInformer, tc.namespaces...)
			if err != nil {
				klog.Fatalf("%v", err)
			}

			scheduled := len(podsScheduled)
			// Only do sampling if number of scheduled pods is greater than zero.
			if scheduled == 0 {
				continue
			}
			if !started {
				started = true
				// Skip the initial sample. It's likely to be an outlier because
				// sampling and creating pods get started independently.
				lastScheduledCount = scheduled
				lastSampleTime = now
				continue
			}

			newScheduled := scheduled - lastScheduledCount
			if newScheduled == 0 {
				// Throughput would be zero for the interval.
				// Instead of recording 0 pods/s, keep waiting
				// until we see at least one additional pod
				// being scheduled.
				skipped++
				continue
			}

			// This should be roughly equal to
			// throughputSampleInterval * (skipped + 1), but we
			// don't count on that because the goroutine might not
			// be scheduled immediately when the timer
			// triggers. Instead we track the actual time stamps.
			duration := now.Sub(lastSampleTime)
			expectedDuration := throughputSampleInterval * time.Duration(skipped+1)
			errorMargin := (duration - expectedDuration).Seconds() / expectedDuration.Seconds() * 100
			if tc.errorMargin > 0 && math.Abs(errorMargin) > tc.errorMargin {
				// This might affect the result, report it.
				klog.Infof("WARNING: Expected throughput collector to sample at regular time intervals. The %d most recent intervals took %s instead of %s, a difference of %0.1f%%.", skipped+1, duration, expectedDuration, errorMargin)
			}

			// To keep percentiles accurate, we have to record multiple samples with the same
			// throughput value if we skipped some intervals.
			throughput := float64(newScheduled) / duration.Seconds()
			for i := 0; i <= skipped; i++ {
				tc.schedulingThroughputs = append(tc.schedulingThroughputs, throughput)
			}
			lastScheduledCount = scheduled
			klog.Infof("%d pods have been scheduled successfully", lastScheduledCount)
			skipped = 0
			lastSampleTime = now
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
