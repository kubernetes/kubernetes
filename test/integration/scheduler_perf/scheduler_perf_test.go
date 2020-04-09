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

package benchmark

import (
	"fmt"
	"io/ioutil"
	"testing"
	"time"

	v1 "k8s.io/api/core/v1"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	coreinformers "k8s.io/client-go/informers/core/v1"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/component-base/featuregate"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/test/integration/framework"
	testutils "k8s.io/kubernetes/test/utils"
	"sigs.k8s.io/yaml"
)

const (
	configFile = "config/performance-config.yaml"
)

var (
	defaultMetricsCollectorConfig = metricsCollectorConfig{
		Metrics: []string{
			"scheduler_scheduling_algorithm_predicate_evaluation_seconds",
			"scheduler_scheduling_algorithm_priority_evaluation_seconds",
			"scheduler_binding_duration_seconds",
			"scheduler_e2e_scheduling_duration_seconds",
		},
	}
)

// testCase configures a test case to run the scheduler performance test. Users should be able to
// provide this via a YAML file.
//
// It specifies nodes and pods in the cluster before running the test. It also specifies the pods to
// schedule during the test. The config can be as simple as just specify number of nodes/pods, where
// default spec will be applied. It also allows the user to specify a pod spec template for more
// complicated test cases.
//
// It also specifies the metrics to be collected after the test. If nothing is specified, default metrics
// such as scheduling throughput and latencies will be collected.
type testCase struct {
	// description of the test case
	Desc string
	// configures nodes in the cluster
	Nodes nodeCase
	// configures pods in the cluster before running the tests
	InitPods []podCase
	// pods to be scheduled during the test.
	PodsToSchedule podCase
	// optional, feature gates to set before running the test
	FeatureGates map[featuregate.Feature]bool
	// optional, replaces default defaultMetricsCollectorConfig if supplied.
	MetricsCollectorConfig *metricsCollectorConfig
}

type nodeCase struct {
	Num              int
	NodeTemplatePath *string
	// At most one of the following strategies can be defined. If not specified, default to TrivialNodePrepareStrategy.
	NodeAllocatableStrategy  *testutils.NodeAllocatableStrategy
	LabelNodePrepareStrategy *testutils.LabelNodePrepareStrategy
	UniqueNodeLabelStrategy  *testutils.UniqueNodeLabelStrategy
}

type podCase struct {
	Num                               int
	PodTemplatePath                   *string
	PersistentVolumeTemplatePath      *string
	PersistentVolumeClaimTemplatePath *string
}

// simpleTestCases defines a set of test cases that share the same template (node spec, pod spec, etc)
// with testParams(e.g., NumNodes) being overridden. This provides a convenient way to define multiple tests
// with various sizes.
type simpleTestCases struct {
	Template testCase
	Params   []testParams
}

type testParams struct {
	NumNodes          int
	NumInitPods       []int
	NumPodsToSchedule int
}

type testDataCollector interface {
	run(stopCh chan struct{})
	collect() []DataItem
}

func BenchmarkPerfScheduling(b *testing.B) {
	dataItems := DataItems{Version: "v1"}
	tests, err := parseTestCases(configFile)
	if err != nil {
		b.Fatal(err)
	}

	for _, test := range tests {
		initPods := 0
		for _, p := range test.InitPods {
			initPods += p.Num
		}
		name := fmt.Sprintf("%v/%vNodes/%vInitPods/%vPodsToSchedule", test.Desc, test.Nodes.Num, initPods, test.PodsToSchedule.Num)
		b.Run(name, func(b *testing.B) {
			for feature, flag := range test.FeatureGates {
				defer featuregatetesting.SetFeatureGateDuringTest(b, utilfeature.DefaultFeatureGate, feature, flag)()
			}
			dataItems.DataItems = append(dataItems.DataItems, perfScheduling(test, b)...)
		})
	}
	if err := dataItems2JSONFile(dataItems, b.Name()); err != nil {
		klog.Fatalf("%v: unable to write measured data: %v", b.Name(), err)
	}
}

func perfScheduling(test testCase, b *testing.B) []DataItem {
	finalFunc, podInformer, clientset := mustSetupScheduler()
	defer finalFunc()

	nodePreparer, err := getNodePreparer(test.Nodes, clientset)
	if err != nil {
		b.Fatal(err)
	}
	if err := nodePreparer.PrepareNodes(); err != nil {
		b.Fatal(err)
	}
	defer nodePreparer.CleanupNodes()

	total := 0
	for _, p := range test.InitPods {
		if err := createPods(setupNamespace, p, clientset); err != nil {
			b.Fatal(err)
		}
		total += p.Num
	}
	if err := waitNumPodsScheduled(b, total, podInformer); err != nil {
		b.Fatal(err)
	}

	// start benchmark
	b.ResetTimer()

	// Start test data collectors.
	stopCh := make(chan struct{})
	collectors := getTestDataCollectors(test, podInformer, b)
	for _, collector := range collectors {
		go collector.run(stopCh)
	}

	// Schedule the main workload
	if err := createPods(testNamespace, test.PodsToSchedule, clientset); err != nil {
		b.Fatal(err)
	}
	if err := waitNumPodsScheduled(b, total+test.PodsToSchedule.Num, podInformer); err != nil {
		b.Fatal(err)
	}

	close(stopCh)
	// Note: without this line we're taking the overhead of defer() into account.
	b.StopTimer()

	var dataItems []DataItem
	for _, collector := range collectors {
		dataItems = append(dataItems, collector.collect()...)
	}
	return dataItems
}

func waitNumPodsScheduled(b *testing.B, num int, podInformer coreinformers.PodInformer) error {
	for {
		scheduled, err := getScheduledPods(podInformer)
		if err != nil {
			return err
		}
		if len(scheduled) >= num {
			break
		}
		klog.Infof("%s: got %d existing pods, required: %d", b.Name(), len(scheduled), num)
		time.Sleep(1 * time.Second)
	}
	return nil
}

func getTestDataCollectors(tc testCase, podInformer coreinformers.PodInformer, b *testing.B) []testDataCollector {
	collectors := []testDataCollector{newThroughputCollector(podInformer, map[string]string{"Name": b.Name()})}
	metricsCollectorConfig := defaultMetricsCollectorConfig
	if tc.MetricsCollectorConfig != nil {
		metricsCollectorConfig = *tc.MetricsCollectorConfig
	}
	collectors = append(collectors, newMetricsCollector(metricsCollectorConfig, map[string]string{"Name": b.Name()}))
	return collectors
}

func getNodePreparer(nc nodeCase, clientset clientset.Interface) (testutils.TestNodePreparer, error) {
	var nodeStrategy testutils.PrepareNodeStrategy = &testutils.TrivialNodePrepareStrategy{}
	if nc.NodeAllocatableStrategy != nil {
		nodeStrategy = nc.NodeAllocatableStrategy
	} else if nc.LabelNodePrepareStrategy != nil {
		nodeStrategy = nc.LabelNodePrepareStrategy
	} else if nc.UniqueNodeLabelStrategy != nil {
		nodeStrategy = nc.UniqueNodeLabelStrategy
	}

	if nc.NodeTemplatePath != nil {
		node, err := getNodeSpecFromFile(nc.NodeTemplatePath)
		if err != nil {
			return nil, err
		}
		return framework.NewIntegrationTestNodePreparerWithNodeSpec(
			clientset,
			[]testutils.CountToStrategy{{Count: nc.Num, Strategy: nodeStrategy}},
			node,
		), nil
	}
	return framework.NewIntegrationTestNodePreparer(
		clientset,
		[]testutils.CountToStrategy{{Count: nc.Num, Strategy: nodeStrategy}},
		"scheduler-perf-",
	), nil
}

func createPods(ns string, pc podCase, clientset clientset.Interface) error {
	strategy, err := getPodStrategy(pc)
	if err != nil {
		return err
	}
	config := testutils.NewTestPodCreatorConfig()
	config.AddStrategy(ns, pc.Num, strategy)
	podCreator := testutils.NewTestPodCreator(clientset, config)
	return podCreator.CreatePods()
}

func getPodStrategy(pc podCase) (testutils.TestPodCreateStrategy, error) {
	basePod := makeBasePod()
	if pc.PodTemplatePath != nil {
		var err error
		basePod, err = getPodSpecFromFile(pc.PodTemplatePath)
		if err != nil {
			return nil, err
		}
	}
	if pc.PersistentVolumeClaimTemplatePath == nil {
		return testutils.NewCustomCreatePodStrategy(basePod), nil
	}

	pvTemplate, err := getPersistentVolumeSpecFromFile(pc.PersistentVolumeTemplatePath)
	if err != nil {
		return nil, err
	}
	pvcTemplate, err := getPersistentVolumeClaimSpecFromFile(pc.PersistentVolumeClaimTemplatePath)
	if err != nil {
		return nil, err
	}
	return testutils.NewCreatePodWithPersistentVolumeStrategy(pvcTemplate, getCustomVolumeFactory(pvTemplate), basePod), nil
}

func parseTestCases(path string) ([]testCase, error) {
	var simpleTests []simpleTestCases
	if err := getSpecFromFile(&path, &simpleTests); err != nil {
		return nil, fmt.Errorf("parsing test cases: %v", err)
	}

	testCases := make([]testCase, 0)
	for _, s := range simpleTests {
		testCase := s.Template
		for _, p := range s.Params {
			testCase.Nodes.Num = p.NumNodes
			testCase.InitPods = append([]podCase(nil), testCase.InitPods...)
			for i, v := range p.NumInitPods {
				testCase.InitPods[i].Num = v
			}
			testCase.PodsToSchedule.Num = p.NumPodsToSchedule
			testCases = append(testCases, testCase)
		}
	}

	return testCases, nil
}

func getNodeSpecFromFile(path *string) (*v1.Node, error) {
	nodeSpec := &v1.Node{}
	if err := getSpecFromFile(path, nodeSpec); err != nil {
		return nil, fmt.Errorf("parsing Node: %v", err)
	}
	return nodeSpec, nil
}

func getPodSpecFromFile(path *string) (*v1.Pod, error) {
	podSpec := &v1.Pod{}
	if err := getSpecFromFile(path, podSpec); err != nil {
		return nil, fmt.Errorf("parsing Pod: %v", err)
	}
	return podSpec, nil
}

func getPersistentVolumeSpecFromFile(path *string) (*v1.PersistentVolume, error) {
	persistentVolumeSpec := &v1.PersistentVolume{}
	if err := getSpecFromFile(path, persistentVolumeSpec); err != nil {
		return nil, fmt.Errorf("parsing PersistentVolume: %v", err)
	}
	return persistentVolumeSpec, nil
}

func getPersistentVolumeClaimSpecFromFile(path *string) (*v1.PersistentVolumeClaim, error) {
	persistentVolumeClaimSpec := &v1.PersistentVolumeClaim{}
	if err := getSpecFromFile(path, persistentVolumeClaimSpec); err != nil {
		return nil, fmt.Errorf("parsing PersistentVolumeClaim: %v", err)
	}
	return persistentVolumeClaimSpec, nil
}

func getSpecFromFile(path *string, spec interface{}) error {
	bytes, err := ioutil.ReadFile(*path)
	if err != nil {
		return err
	}
	return yaml.Unmarshal(bytes, spec)
}

func getCustomVolumeFactory(pvTemplate *v1.PersistentVolume) func(id int) *v1.PersistentVolume {
	return func(id int) *v1.PersistentVolume {
		pv := pvTemplate.DeepCopy()
		volumeID := fmt.Sprintf("vol-%d", id)
		pv.ObjectMeta.Name = volumeID
		pvs := pv.Spec.PersistentVolumeSource
		if pvs.CSI != nil {
			pvs.CSI.VolumeHandle = volumeID
		} else if pvs.AWSElasticBlockStore != nil {
			pvs.AWSElasticBlockStore.VolumeID = volumeID
		}
		return pv
	}
}
