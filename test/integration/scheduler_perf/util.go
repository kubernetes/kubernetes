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
	"maps"
	"math"
	"os"
	"path"
	"runtime"
	"sort"
	"strings"
	"sync"
	"time"

	v1 "k8s.io/api/core/v1"
	resourceapi "k8s.io/api/resource/v1"
	resourcealpha "k8s.io/api/resource/v1alpha3"
	resourcev1beta1 "k8s.io/api/resource/v1beta1"
	resourcev1beta2 "k8s.io/api/resource/v1beta2"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/client-go/informers"
	coreinformers "k8s.io/client-go/informers/core/v1"
	restclient "k8s.io/client-go/rest"
	"k8s.io/client-go/tools/cache"
	"k8s.io/component-base/featuregate"
	"k8s.io/component-base/metrics/legacyregistry"
	"k8s.io/component-base/metrics/testutil"
	"k8s.io/klog/v2"
	kubeschedulerconfigv1 "k8s.io/kube-scheduler/config/v1"
	apiservertesting "k8s.io/kubernetes/cmd/kube-apiserver/app/testing"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/scheduler"
	"k8s.io/kubernetes/pkg/scheduler/apis/config"
	kubeschedulerscheme "k8s.io/kubernetes/pkg/scheduler/apis/config/scheme"
	frameworkruntime "k8s.io/kubernetes/pkg/scheduler/framework/runtime"
	schedutil "k8s.io/kubernetes/pkg/scheduler/util"
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

var runID = time.Now().Format(dateFormat)

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
func mustSetupCluster(tCtx ktesting.TContext, config *config.KubeSchedulerConfiguration, enabledFeatures map[featuregate.Feature]bool, outOfTreePluginRegistry frameworkruntime.Registry) (*scheduler.Scheduler, informers.SharedInformerFactory, ktesting.TContext) {
	var runtimeConfig []string
	if enabledFeatures[features.DynamicResourceAllocation] {
		runtimeConfig = append(runtimeConfig, fmt.Sprintf("%s=true", resourceapi.SchemeGroupVersion))
		runtimeConfig = append(runtimeConfig, fmt.Sprintf("%s=true", resourcev1beta2.SchemeGroupVersion))
		runtimeConfig = append(runtimeConfig, fmt.Sprintf("%s=true", resourcev1beta1.SchemeGroupVersion))
		runtimeConfig = append(runtimeConfig, fmt.Sprintf("%s=true", resourcealpha.SchemeGroupVersion))
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
	scheduler, informerFactory := util.StartScheduler(tCtx, config, outOfTreePluginRegistry)
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

	return scheduler, informerFactory, tCtx
}

func isAttempted(pod *v1.Pod) bool {
	for _, cond := range pod.Status.Conditions {
		if cond.Type == v1.PodScheduled {
			return true
		}
	}
	return false
}

// getScheduledPods returns the list of scheduled, attempted but unschedulable
// and unattempted pods in the specified namespaces.
// Label selector can be used to filter the pods.
// Note that no namespaces specified matches all namespaces.
func getScheduledPods(podInformer coreinformers.PodInformer, labelSelector map[string]string, namespaces ...string) ([]*v1.Pod, []*v1.Pod, []*v1.Pod, error) {
	pods, err := podInformer.Lister().List(labels.Everything())
	if err != nil {
		return nil, nil, nil, err
	}

	s := sets.New(namespaces...)
	scheduled := make([]*v1.Pod, 0, len(pods))
	attempted := make([]*v1.Pod, 0, len(pods))
	unattempted := make([]*v1.Pod, 0, len(pods))
	for i := range pods {
		pod := pods[i]
		if (len(s) == 0 || s.Has(pod.Namespace)) && labelsMatch(pod.Labels, labelSelector) {
			if len(pod.Spec.NodeName) > 0 {
				scheduled = append(scheduled, pod)
			} else if isAttempted(pod) {
				attempted = append(attempted, pod)
			} else {
				unattempted = append(unattempted, pod)
			}
		}
	}
	return scheduled, attempted, unattempted, nil
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

	// progress contains number of scheduled pods over time.
	progress []podScheduling
	start    time.Time
}

// DataItems is the data point set. It is the struct that perf dashboard expects.
type DataItems struct {
	Version   string     `json:"version"`
	DataItems []DataItem `json:"dataItems"`
}

type podScheduling struct {
	ts            time.Time
	attempts      int
	completed     int
	observedTotal int
	observedRate  float64
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

// makeBaseNode creates a Node object with given nodeNamePrefix to be used as a template.
func makeBaseNode(nodeNamePrefix string) *v1.Node {
	return &v1.Node{
		ObjectMeta: metav1.ObjectMeta{
			GenerateName: nodeNamePrefix,
		},
		Status: v1.NodeStatus{
			Capacity: v1.ResourceList{
				v1.ResourcePods:   *resource.NewQuantity(110, resource.DecimalSI),
				v1.ResourceCPU:    resource.MustParse("4"),
				v1.ResourceMemory: resource.MustParse("32Gi"),
			},
			Allocatable: v1.ResourceList{
				v1.ResourcePods:   *resource.NewQuantity(110, resource.DecimalSI),
				v1.ResourceCPU:    resource.MustParse("4"),
				v1.ResourceMemory: resource.MustParse("32Gi"),
			},
			Phase: v1.NodeRunning,
			Conditions: []v1.NodeCondition{
				{Type: v1.NodeReady, Status: v1.ConditionTrue},
			},
		},
	}

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

func dataFilename(destFile string) (string, error) {
	if *dataItemsDir != "" {
		// Ensure the "dataItemsDir" path is valid.
		if err := os.MkdirAll(*dataItemsDir, 0750); err != nil {
			return "", fmt.Errorf("dataItemsDir path %v does not exist and cannot be created: %w", *dataItemsDir, err)
		}
		destFile = path.Join(*dataItemsDir, destFile)
	}
	return destFile, nil
}

type labelValues struct {
	Label  string
	Values []string
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
	for _, value := range current.Values {
		for _, combo := range remainingCombos {
			newCombo := make(map[string]string, len(combo)+1)
			for k, v := range combo {
				newCombo[k] = v
			}
			newCombo[current.Label] = value
			results = append(results, newCombo)
		}
	}
	return results
}

func collectHistogramVec(metric string, labels map[string]string, lvMap map[string]string) *DataItem {
	vec, err := testutil.GetHistogramVecFromGatherer(legacyregistry.DefaultGatherer, metric, lvMap)
	if err != nil {
		// "metric ... not found" is pretty normal. Don't spam the output with it!
		if !strings.HasSuffix(err.Error(), "not found") {
			klog.Error(err)
		}
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
	labelSelector         map[string]string
	resultLabels          map[string]string
	namespaces            sets.Set[string]
	errorMargin           float64

	progress []podScheduling
	start    time.Time
}

func newThroughputCollector(podInformer coreinformers.PodInformer, resultLabels map[string]string, labelSelector map[string]string, namespaces []string, errorMargin float64) *throughputCollector {
	return &throughputCollector{
		podInformer:   podInformer,
		labelSelector: labelSelector,
		resultLabels:  resultLabels,
		namespaces:    sets.New(namespaces...),
		errorMargin:   errorMargin,
	}
}

func (tc *throughputCollector) init() error {
	return nil
}

func (tc *throughputCollector) run(tCtx ktesting.TContext) {
	// The collector is based on informer cache events instead of periodically listing pods because:
	// - polling causes more overhead
	// - it does not work when pods get created, scheduled and deleted quickly
	//
	// Normally, informers cannot be used to observe state changes reliably.
	// They only guarantee that the *some* updates get reported, but not *all*.
	// But in scheduler_perf, the scheduler and the test share the same informer,
	// therefore we are guaranteed to see a new pod without NodeName (because
	// that is what the scheduler needs to see to schedule it) and then the updated
	// pod with NodeName (because nothing makes further changes to it).
	var mutex sync.Mutex
	scheduledPods := 0
	getScheduledPods := func() int {
		mutex.Lock()
		defer mutex.Unlock()
		return scheduledPods
	}
	onPodChange := func(oldObj, newObj any) {
		oldPod, newPod, err := schedutil.As[*v1.Pod](oldObj, newObj)
		if err != nil {
			tCtx.Errorf("unexpected pod events: %v", err)
			return
		}

		if !tc.namespaces.Has(newPod.Namespace) || !labelsMatch(newPod.Labels, tc.labelSelector) {
			return
		}

		mutex.Lock()
		defer mutex.Unlock()
		if (oldPod == nil || oldPod.Spec.NodeName == "") && newPod.Spec.NodeName != "" {
			// Got scheduled.
			scheduledPods++
		}
	}
	handle, err := tc.podInformer.Informer().AddEventHandler(cache.ResourceEventHandlerFuncs{
		AddFunc: func(obj any) {
			onPodChange(nil, obj)
		},
		UpdateFunc: func(oldObj, newObj any) {
			onPodChange(oldObj, newObj)
		},
	})
	if err != nil {
		tCtx.Fatalf("register pod event handler: %v", err)
	}
	defer func() {
		tCtx.ExpectNoError(tc.podInformer.Informer().RemoveEventHandler(handle), "remove event handler")
	}()

	// Waiting for the initial sync didn't work, `handle.HasSynced` always returned
	// false - perhaps because the event handlers get added to a running informer.
	// That's okay(ish), throughput is typically measured within an empty namespace.
	//
	// syncTicker := time.NewTicker(time.Millisecond)
	// defer syncTicker.Stop()
	// for {
	// 	select {
	// 	case <-syncTicker.C:
	// 		if handle.HasSynced() {
	// 			break
	// 		}
	// 	case <-tCtx.Done():
	// 		return
	// 	}
	// }
	tCtx.Logf("Started pod throughput collector for namespace(s) %s, %d pods scheduled so far", sets.List(tc.namespaces), getScheduledPods())

	lastScheduledCount := getScheduledPods()
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

			scheduled := getScheduledPods()
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
				tc.start = now
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

			// Record the metric sample.
			counters, err := testutil.GetCounterValuesFromGatherer(legacyregistry.DefaultGatherer, "scheduler_schedule_attempts_total", map[string]string{"profile": "default-scheduler"}, "result")
			if err != nil {
				klog.Error(err)
			}
			tc.progress = append(tc.progress, podScheduling{
				ts:            now,
				attempts:      int(counters["unschedulable"] + counters["error"] + counters["scheduled"]),
				completed:     int(counters["scheduled"]),
				observedTotal: scheduled,
				observedRate:  throughput,
			})

			lastScheduledCount = scheduled
			klog.Infof("%d pods have been scheduled successfully", lastScheduledCount)
			skipped = 0
			lastSampleTime = now
		}
	}
}

func (tc *throughputCollector) collect() []DataItem {
	throughputSummary := DataItem{
		Labels:   tc.resultLabels,
		progress: tc.progress,
		start:    tc.start,
	}

	// tc.schedulingThroughputs can be empty if the scenario doesn't have
	// enough number of pods and nodes to take more than throughputSampleInterval (i.e. 1 second).
	length := len(tc.schedulingThroughputs)
	if length == 0 {
		klog.Warningf("Failed to measure SchedulingThroughput for %s. Increase pods and/or nodes to make scheduling take longer", tc.resultLabels["Name"])
		return []DataItem{throughputSummary}
	}

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

	return []DataItem{throughputSummary}
}

// memoryCollector collects memory usage metrics during the test
type memoryCollector struct {
	samples      []memorySample
	resultLabels map[string]string
	interval     time.Duration
	mu           sync.RWMutex
}

type memorySample struct {
	timestamp   time.Time
	heapInuseMB float64
}

func newMemoryCollector(resultLabels map[string]string, interval time.Duration) *memoryCollector {
	return &memoryCollector{
		resultLabels: resultLabels,
		interval:     interval,
	}
}

func (mc *memoryCollector) init() error {
	mc.collectSample()
	return nil
}

func (mc *memoryCollector) run(tCtx ktesting.TContext) {
	ticker := time.NewTicker(mc.interval)
	defer ticker.Stop()

	for {
		select {
		case <-tCtx.Done():
			return
		case <-ticker.C:
			mc.collectSample()
		}
	}
}

func (mc *memoryCollector) collectSample() {
	var m runtime.MemStats
	runtime.ReadMemStats(&m)

	sample := memorySample{
		timestamp:   time.Now(),
		heapInuseMB: float64(m.HeapInuse) / 1024 / 1024,
	}

	mc.mu.Lock()
	mc.samples = append(mc.samples, sample)
	mc.mu.Unlock()
}

func (mc *memoryCollector) createMetricDataItem(values []float64, unit, metricName string) DataItem {
	sort.Float64s(values)

	sum := 0.0
	for _, v := range values {
		sum += v
	}

	labels := maps.Clone(mc.resultLabels)
	labels["Metric"] = metricName

	return DataItem{
		Labels: labels,
		Data: map[string]float64{
			"Perc50":  values[int(math.Ceil(float64(len(values)*50)/100))-1],
			"Perc90":  values[int(math.Ceil(float64(len(values)*90)/100))-1],
			"Perc95":  values[int(math.Ceil(float64(len(values)*95)/100))-1],
			"Perc99":  values[int(math.Ceil(float64(len(values)*99)/100))-1],
			"Average": sum / float64(len(values)),
			"Max":     values[len(values)-1],
		},
		Unit: unit,
	}
}

func (mc *memoryCollector) collect() []DataItem {
	mc.mu.RLock()
	defer mc.mu.RUnlock()

	length := len(mc.samples)
	if length == 0 {
		return nil
	}

	firstSample := mc.samples[0]
	lastSample := firstSample
	if length >= 2 {
		lastSample = mc.samples[length-1]
	}
	durationMin := lastSample.timestamp.Sub(firstSample.timestamp).Minutes()
	growthRateMBPerMin := 0.0
	if durationMin > 0 {
		growthRateMBPerMin = (lastSample.heapInuseMB - firstSample.heapInuseMB) / durationMin
	}
	growthItem := DataItem{
		Labels: maps.Clone(mc.resultLabels),
		Data: map[string]float64{
			"GrowthRate": growthRateMBPerMin,
		},
		Unit: "MB/min",
	}
	growthItem.Labels["Metric"] = "memory_growth_rate"

	heapValues := make([]float64, len(mc.samples))
	for i, s := range mc.samples {
		heapValues[i] = s.heapInuseMB
	}

	return []DataItem{
		mc.createMetricDataItem(heapValues, "MB", "heap_memory_usage"),
		growthItem,
	}
}

// schedulingDurationCollector calculates the total duration of the scheduling phase, including pod creation.
type schedulingDurationCollector struct {
	resultLabels map[string]string
	duration     time.Duration
}

func newSchedulingDurationCollector(resultLabels map[string]string) *schedulingDurationCollector {
	return &schedulingDurationCollector{
		resultLabels: resultLabels,
	}
}

func (sdc *schedulingDurationCollector) init() error {
	return nil
}

func (sdc *schedulingDurationCollector) run(tCtx ktesting.TContext) {
	start := time.Now()
	// Wait for the scheduling to finish
	<-tCtx.Done()
	sdc.duration = time.Since(start)
}

func (sdc *schedulingDurationCollector) collect() []DataItem {
	labels := maps.Clone(sdc.resultLabels)
	labels["Metric"] = "SchedulingDuration"
	return []DataItem{{
		Labels: labels,
		Data: map[string]float64{
			"Duration": sdc.duration.Seconds(),
		},
		Unit: "s",
	}}
}
