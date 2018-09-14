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

package scalability

import (
	"fmt"
	"math"
	"os"
	"sort"
	"strconv"
	"sync"
	"time"

	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	utiluuid "k8s.io/apimachinery/pkg/util/uuid"
	"k8s.io/apimachinery/pkg/watch"
	clientset "k8s.io/client-go/kubernetes"
	scaleclient "k8s.io/client-go/scale"
	"k8s.io/client-go/tools/cache"
	"k8s.io/client-go/util/workqueue"
	"k8s.io/kubernetes/pkg/apis/batch"
	api "k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/kubernetes/pkg/apis/extensions"
	"k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset"
	"k8s.io/kubernetes/test/e2e/framework"
	"k8s.io/kubernetes/test/e2e/framework/timer"
	testutils "k8s.io/kubernetes/test/utils"
	imageutils "k8s.io/kubernetes/test/utils/image"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
)

const (
	PodStartupLatencyThreshold = 5 * time.Second
	MinSaturationThreshold     = 2 * time.Minute
	MinPodsPerSecondThroughput = 8
	DensityPollInterval        = 10 * time.Second
	MinPodStartupMeasurements  = 500
)

// Maximum container failures this test tolerates before failing.
var MaxContainerFailures = 0

// Maximum no. of missing measurements related to pod-startup that the test tolerates.
var MaxMissingPodStartupMeasurements = 0

// Number of nodes in the cluster (computed inside BeforeEach).
var nodeCount = 0

type DensityTestConfig struct {
	Configs            []testutils.RunObjectConfig
	ClientSets         []clientset.Interface
	InternalClientsets []internalclientset.Interface
	ScaleClients       []scaleclient.ScalesGetter
	PollInterval       time.Duration
	PodCount           int
	// What kind of resource we want to create
	kind             schema.GroupKind
	SecretConfigs    []*testutils.SecretConfig
	ConfigMapConfigs []*testutils.ConfigMapConfig
	DaemonConfigs    []*testutils.DaemonConfig
}

func (dtc *DensityTestConfig) runSecretConfigs(testPhase *timer.Phase) {
	defer testPhase.End()
	for _, sc := range dtc.SecretConfigs {
		sc.Run()
	}
}

func (dtc *DensityTestConfig) runConfigMapConfigs(testPhase *timer.Phase) {
	defer testPhase.End()
	for _, cmc := range dtc.ConfigMapConfigs {
		cmc.Run()
	}
}

func (dtc *DensityTestConfig) runDaemonConfigs(testPhase *timer.Phase) {
	defer testPhase.End()
	for _, dc := range dtc.DaemonConfigs {
		dc.Run()
	}
}

func (dtc *DensityTestConfig) deleteSecrets(testPhase *timer.Phase) {
	defer testPhase.End()
	for i := range dtc.SecretConfigs {
		dtc.SecretConfigs[i].Stop()
	}
}

func (dtc *DensityTestConfig) deleteConfigMaps(testPhase *timer.Phase) {
	defer testPhase.End()
	for i := range dtc.ConfigMapConfigs {
		dtc.ConfigMapConfigs[i].Stop()
	}
}

func (dtc *DensityTestConfig) deleteDaemonSets(numberOfClients int, testPhase *timer.Phase) {
	defer testPhase.End()
	for i := range dtc.DaemonConfigs {
		framework.ExpectNoError(framework.DeleteResourceAndWaitForGC(
			dtc.ClientSets[i%numberOfClients],
			extensions.Kind("DaemonSet"),
			dtc.DaemonConfigs[i].Namespace,
			dtc.DaemonConfigs[i].Name,
		))
	}
}

func density30AddonResourceVerifier(numNodes int) map[string]framework.ResourceConstraint {
	var apiserverMem uint64
	var controllerMem uint64
	var schedulerMem uint64
	apiserverCPU := math.MaxFloat32
	apiserverMem = math.MaxUint64
	controllerCPU := math.MaxFloat32
	controllerMem = math.MaxUint64
	schedulerCPU := math.MaxFloat32
	schedulerMem = math.MaxUint64
	framework.Logf("Setting resource constraints for provider: %s", framework.TestContext.Provider)
	if framework.ProviderIs("kubemark") {
		if numNodes <= 5 {
			apiserverCPU = 0.35
			apiserverMem = 150 * (1024 * 1024)
			controllerCPU = 0.15
			controllerMem = 100 * (1024 * 1024)
			schedulerCPU = 0.05
			schedulerMem = 50 * (1024 * 1024)
		} else if numNodes <= 100 {
			apiserverCPU = 1.5
			apiserverMem = 1500 * (1024 * 1024)
			controllerCPU = 0.5
			controllerMem = 500 * (1024 * 1024)
			schedulerCPU = 0.4
			schedulerMem = 180 * (1024 * 1024)
		} else if numNodes <= 500 {
			apiserverCPU = 3.5
			apiserverMem = 3400 * (1024 * 1024)
			controllerCPU = 1.3
			controllerMem = 1100 * (1024 * 1024)
			schedulerCPU = 1.5
			schedulerMem = 500 * (1024 * 1024)
		} else if numNodes <= 1000 {
			apiserverCPU = 5.5
			apiserverMem = 4000 * (1024 * 1024)
			controllerCPU = 3
			controllerMem = 2000 * (1024 * 1024)
			schedulerCPU = 1.5
			schedulerMem = 750 * (1024 * 1024)
		}
	} else {
		if numNodes <= 100 {
			apiserverCPU = 2.2
			apiserverMem = 1700 * (1024 * 1024)
			controllerCPU = 0.8
			controllerMem = 530 * (1024 * 1024)
			schedulerCPU = 0.4
			schedulerMem = 180 * (1024 * 1024)
		}
	}

	constraints := make(map[string]framework.ResourceConstraint)
	constraints["fluentd-elasticsearch"] = framework.ResourceConstraint{
		CPUConstraint:    0.2,
		MemoryConstraint: 250 * (1024 * 1024),
	}
	constraints["elasticsearch-logging"] = framework.ResourceConstraint{
		CPUConstraint: 2,
		// TODO: bring it down to 750MB again, when we lower Kubelet verbosity level. I.e. revert #19164
		MemoryConstraint: 5000 * (1024 * 1024),
	}
	constraints["heapster"] = framework.ResourceConstraint{
		CPUConstraint:    2,
		MemoryConstraint: 1800 * (1024 * 1024),
	}
	constraints["kibana-logging"] = framework.ResourceConstraint{
		CPUConstraint:    0.2,
		MemoryConstraint: 100 * (1024 * 1024),
	}
	constraints["kube-proxy"] = framework.ResourceConstraint{
		CPUConstraint:    0.15,
		MemoryConstraint: 100 * (1024 * 1024),
	}
	constraints["l7-lb-controller"] = framework.ResourceConstraint{
		CPUConstraint:    0.2 + 0.00015*float64(numNodes),
		MemoryConstraint: (75 + uint64(math.Ceil(0.8*float64(numNodes)))) * (1024 * 1024),
	}
	constraints["influxdb"] = framework.ResourceConstraint{
		CPUConstraint:    2,
		MemoryConstraint: 500 * (1024 * 1024),
	}
	constraints["kube-apiserver"] = framework.ResourceConstraint{
		CPUConstraint:    apiserverCPU,
		MemoryConstraint: apiserverMem,
	}
	constraints["kube-controller-manager"] = framework.ResourceConstraint{
		CPUConstraint:    controllerCPU,
		MemoryConstraint: controllerMem,
	}
	constraints["kube-scheduler"] = framework.ResourceConstraint{
		CPUConstraint:    schedulerCPU,
		MemoryConstraint: schedulerMem,
	}
	return constraints
}

func computeAverage(sample []float64) float64 {
	sum := 0.0
	for _, value := range sample {
		sum += value
	}
	return sum / float64(len(sample))
}

func computeQuantile(sample []float64, quantile float64) float64 {
	Expect(sort.Float64sAreSorted(sample)).To(Equal(true))
	Expect(quantile >= 0.0 && quantile <= 1.0).To(Equal(true))
	index := int(quantile*float64(len(sample))) - 1
	if index < 0 {
		return math.NaN()
	}
	return sample[index]
}

func logPodStartupStatus(
	c clientset.Interface,
	expectedPods int,
	observedLabels map[string]string,
	period time.Duration,
	scheduleThroughputs *[]float64,
	stopCh chan struct{}) {

	label := labels.SelectorFromSet(labels.Set(observedLabels))
	podStore, err := testutils.NewPodStore(c, metav1.NamespaceAll, label, fields.Everything())
	framework.ExpectNoError(err)
	defer podStore.Stop()

	ticker := time.NewTicker(period)
	startupStatus := testutils.ComputeRCStartupStatus(podStore.List(), expectedPods)
	lastScheduledCount := startupStatus.Scheduled
	defer ticker.Stop()
	for {
		select {
		case <-ticker.C:
		case <-stopCh:
			return
		}
		// Log status of the pods.
		startupStatus := testutils.ComputeRCStartupStatus(podStore.List(), expectedPods)
		framework.Logf(startupStatus.String("Density"))
		// Compute scheduling throughput for the latest time period.
		throughput := float64(startupStatus.Scheduled-lastScheduledCount) / float64(period/time.Second)
		*scheduleThroughputs = append(*scheduleThroughputs, throughput)
		lastScheduledCount = startupStatus.Scheduled
	}
}

// runDensityTest will perform a density test and return the time it took for
// all pods to start
func runDensityTest(dtc DensityTestConfig, testPhaseDurations *timer.TestPhaseTimer, scheduleThroughputs *[]float64) time.Duration {
	defer GinkgoRecover()

	// Create all secrets, configmaps and daemons.
	dtc.runSecretConfigs(testPhaseDurations.StartPhase(250, "secrets creation"))
	dtc.runConfigMapConfigs(testPhaseDurations.StartPhase(260, "configmaps creation"))
	dtc.runDaemonConfigs(testPhaseDurations.StartPhase(270, "daemonsets creation"))

	replicationCtrlStartupPhase := testPhaseDurations.StartPhase(300, "saturation pods creation")
	defer replicationCtrlStartupPhase.End()

	// Start scheduler CPU profile-gatherer before we begin cluster saturation.
	profileGatheringDelay := time.Duration(1+nodeCount/100) * time.Minute
	schedulerProfilingStopCh := framework.StartCPUProfileGatherer("kube-scheduler", "density", profileGatheringDelay)

	// Start all replication controllers.
	startTime := time.Now()
	wg := sync.WaitGroup{}
	wg.Add(len(dtc.Configs))
	for i := range dtc.Configs {
		config := dtc.Configs[i]
		go func() {
			defer GinkgoRecover()
			// Call wg.Done() in defer to avoid blocking whole test
			// in case of error from RunRC.
			defer wg.Done()
			framework.ExpectNoError(config.Run())
		}()
	}
	logStopCh := make(chan struct{})
	go logPodStartupStatus(dtc.ClientSets[0], dtc.PodCount, map[string]string{"type": "densityPod"}, dtc.PollInterval, scheduleThroughputs, logStopCh)
	wg.Wait()
	startupTime := time.Since(startTime)
	close(logStopCh)
	close(schedulerProfilingStopCh)
	framework.Logf("E2E startup time for %d pods: %v", dtc.PodCount, startupTime)
	framework.Logf("Throughput (pods/s) during cluster saturation phase: %v", float32(dtc.PodCount)/float32(startupTime/time.Second))
	replicationCtrlStartupPhase.End()

	// Grabbing scheduler memory profile after cluster saturation finished.
	wg.Add(1)
	framework.GatherMemoryProfile("kube-scheduler", "density", &wg)
	wg.Wait()

	printPodAllocationPhase := testPhaseDurations.StartPhase(400, "printing pod allocation")
	defer printPodAllocationPhase.End()
	// Print some data about Pod to Node allocation
	By("Printing Pod to Node allocation data")
	podList, err := dtc.ClientSets[0].CoreV1().Pods(metav1.NamespaceAll).List(metav1.ListOptions{})
	framework.ExpectNoError(err)
	pausePodAllocation := make(map[string]int)
	systemPodAllocation := make(map[string][]string)
	for _, pod := range podList.Items {
		if pod.Namespace == metav1.NamespaceSystem {
			systemPodAllocation[pod.Spec.NodeName] = append(systemPodAllocation[pod.Spec.NodeName], pod.Name)
		} else {
			pausePodAllocation[pod.Spec.NodeName]++
		}
	}
	nodeNames := make([]string, 0)
	for k := range pausePodAllocation {
		nodeNames = append(nodeNames, k)
	}
	sort.Strings(nodeNames)
	for _, node := range nodeNames {
		framework.Logf("%v: %v pause pods, system pods: %v", node, pausePodAllocation[node], systemPodAllocation[node])
	}
	defer printPodAllocationPhase.End()
	return startupTime
}

func cleanupDensityTest(dtc DensityTestConfig, testPhaseDurations *timer.TestPhaseTimer) {
	defer GinkgoRecover()
	podCleanupPhase := testPhaseDurations.StartPhase(900, "latency pods deletion")
	defer podCleanupPhase.End()
	By("Deleting created Collections")
	numberOfClients := len(dtc.ClientSets)
	// We explicitly delete all pods to have API calls necessary for deletion accounted in metrics.
	for i := range dtc.Configs {
		name := dtc.Configs[i].GetName()
		namespace := dtc.Configs[i].GetNamespace()
		kind := dtc.Configs[i].GetKind()
		By(fmt.Sprintf("Cleaning up only the %v, garbage collector will clean up the pods", kind))
		err := framework.DeleteResourceAndWaitForGC(dtc.ClientSets[i%numberOfClients], kind, namespace, name)
		framework.ExpectNoError(err)
	}
	podCleanupPhase.End()

	dtc.deleteSecrets(testPhaseDurations.StartPhase(910, "secrets deletion"))
	dtc.deleteConfigMaps(testPhaseDurations.StartPhase(920, "configmaps deletion"))
	dtc.deleteDaemonSets(numberOfClients, testPhaseDurations.StartPhase(930, "daemonsets deletion"))
}

// This test suite can take a long time to run, and can affect or be affected by other tests.
// So by default it is added to the ginkgo.skip list (see driver.go).
// To run this suite you must explicitly ask for it by setting the
// -t/--test flag or ginkgo.focus flag.
// IMPORTANT: This test is designed to work on large (>= 100 Nodes) clusters. For smaller ones
// results will not be representative for control-plane performance as we'll start hitting
// limits on Docker's concurrent container startup.
var _ = SIGDescribe("Density", func() {
	var c clientset.Interface
	var additionalPodsPrefix string
	var ns string
	var uuid string
	var e2eStartupTime time.Duration
	var totalPods int
	var nodeCpuCapacity int64
	var nodeMemCapacity int64
	var nodes *v1.NodeList
	var scheduleThroughputs []float64

	testCaseBaseName := "density"
	missingMeasurements := 0
	var testPhaseDurations *timer.TestPhaseTimer
	var profileGathererStopCh chan struct{}
	var etcdMetricsCollector *framework.EtcdMetricsCollector

	// Gathers data prior to framework namespace teardown
	AfterEach(func() {
		// Stop apiserver CPU profile gatherer and gather memory allocations profile.
		close(profileGathererStopCh)
		wg := sync.WaitGroup{}
		wg.Add(1)
		framework.GatherMemoryProfile("kube-apiserver", "density", &wg)
		wg.Wait()

		saturationThreshold := time.Duration((totalPods / MinPodsPerSecondThroughput)) * time.Second
		if saturationThreshold < MinSaturationThreshold {
			saturationThreshold = MinSaturationThreshold
		}
		Expect(e2eStartupTime).NotTo(BeNumerically(">", saturationThreshold))
		saturationData := framework.SaturationTime{
			TimeToSaturate: e2eStartupTime,
			NumberOfNodes:  nodeCount,
			NumberOfPods:   totalPods,
			Throughput:     float32(totalPods) / float32(e2eStartupTime/time.Second),
		}
		framework.Logf("Cluster saturation time: %s", framework.PrettyPrintJSON(saturationData))

		summaries := make([]framework.TestDataSummary, 0, 2)
		// Verify latency metrics.
		highLatencyRequests, metrics, err := framework.HighLatencyRequests(c, nodeCount)
		framework.ExpectNoError(err)
		if err == nil {
			summaries = append(summaries, metrics)
		}

		// Summarize scheduler metrics.
		latency, err := framework.VerifySchedulerLatency(c)
		framework.ExpectNoError(err)
		if err == nil {
			// Compute avg and quantiles of throughput (excluding last element, that's usually an outlier).
			sampleSize := len(scheduleThroughputs)
			if sampleSize > 1 {
				scheduleThroughputs = scheduleThroughputs[:sampleSize-1]
				sort.Float64s(scheduleThroughputs)
				latency.ThroughputAverage = computeAverage(scheduleThroughputs)
				latency.ThroughputPerc50 = computeQuantile(scheduleThroughputs, 0.5)
				latency.ThroughputPerc90 = computeQuantile(scheduleThroughputs, 0.9)
				latency.ThroughputPerc99 = computeQuantile(scheduleThroughputs, 0.99)
			}
			summaries = append(summaries, latency)
		}

		// Summarize etcd metrics.
		err = etcdMetricsCollector.StopAndSummarize()
		framework.ExpectNoError(err)
		if err == nil {
			summaries = append(summaries, etcdMetricsCollector.GetMetrics())
		}

		summaries = append(summaries, testPhaseDurations)

		framework.PrintSummaries(summaries, testCaseBaseName)

		// Fail if there were some high-latency requests.
		Expect(highLatencyRequests).NotTo(BeNumerically(">", 0), "There should be no high-latency requests")
		// Fail if more than the allowed threshold of measurements were missing in the latencyTest.
		Expect(missingMeasurements <= MaxMissingPodStartupMeasurements).To(Equal(true))
	})

	options := framework.FrameworkOptions{
		ClientQPS:   50.0,
		ClientBurst: 100,
	}
	// Explicitly put here, to delete namespace at the end of the test
	// (after measuring latency metrics, etc.).
	f := framework.NewFramework(testCaseBaseName, options, nil)
	f.NamespaceDeletionTimeout = time.Hour

	BeforeEach(func() {
		c = f.ClientSet
		ns = f.Namespace.Name
		testPhaseDurations = timer.NewTestPhaseTimer()

		_, nodes = framework.GetMasterAndWorkerNodesOrDie(c)
		nodeCount = len(nodes.Items)
		Expect(nodeCount).NotTo(BeZero())

		nodeCpuCapacity = nodes.Items[0].Status.Allocatable.Cpu().MilliValue()
		nodeMemCapacity = nodes.Items[0].Status.Allocatable.Memory().Value()

		// Terminating a namespace (deleting the remaining objects from it - which
		// generally means events) can affect the current run. Thus we wait for all
		// terminating namespace to be finally deleted before starting this test.
		err := framework.CheckTestingNSDeletedExcept(c, ns)
		framework.ExpectNoError(err)

		uuid = string(utiluuid.NewUUID())

		framework.ExpectNoError(framework.ResetSchedulerMetrics(c))
		framework.ExpectNoError(framework.ResetMetrics(c))
		framework.ExpectNoError(os.Mkdir(fmt.Sprintf(framework.TestContext.OutputDir+"/%s", uuid), 0777))

		framework.Logf("Listing nodes for easy debugging:\n")
		for _, node := range nodes.Items {
			var internalIP, externalIP string
			for _, address := range node.Status.Addresses {
				if address.Type == v1.NodeInternalIP {
					internalIP = address.Address
				}
				if address.Type == v1.NodeExternalIP {
					externalIP = address.Address
				}
			}
			framework.Logf("Name: %v, clusterIP: %v, externalIP: %v", node.ObjectMeta.Name, internalIP, externalIP)
		}

		// Start apiserver CPU profile gatherer with frequency based on cluster size.
		profileGatheringDelay := time.Duration(5+nodeCount/100) * time.Minute
		profileGathererStopCh = framework.StartCPUProfileGatherer("kube-apiserver", "density", profileGatheringDelay)

		// Start etcs metrics collection.
		etcdMetricsCollector = framework.NewEtcdMetricsCollector()
		etcdMetricsCollector.StartCollecting(time.Minute)
	})

	type Density struct {
		// Controls if e2e latency tests should be run (they are slow)
		runLatencyTest bool
		podsPerNode    int
		// Controls how often the apiserver is polled for pods
		interval time.Duration
		// What kind of resource we should be creating. Default: ReplicationController
		kind             schema.GroupKind
		secretsPerPod    int
		configMapsPerPod int
		daemonsPerNode   int
		quotas           bool
	}

	densityTests := []Density{
		// TODO: Expose runLatencyTest as ginkgo flag.
		{podsPerNode: 3, runLatencyTest: false, kind: api.Kind("ReplicationController")},
		{podsPerNode: 30, runLatencyTest: true, kind: api.Kind("ReplicationController")},
		{podsPerNode: 50, runLatencyTest: false, kind: api.Kind("ReplicationController")},
		{podsPerNode: 95, runLatencyTest: true, kind: api.Kind("ReplicationController")},
		{podsPerNode: 100, runLatencyTest: false, kind: api.Kind("ReplicationController")},
		// Tests for other resource types:
		{podsPerNode: 30, runLatencyTest: true, kind: extensions.Kind("Deployment")},
		{podsPerNode: 30, runLatencyTest: true, kind: batch.Kind("Job")},
		// Test scheduling when daemons are preset
		{podsPerNode: 30, runLatencyTest: true, kind: api.Kind("ReplicationController"), daemonsPerNode: 2},
		// Test with secrets
		{podsPerNode: 30, runLatencyTest: true, kind: extensions.Kind("Deployment"), secretsPerPod: 2},
		// Test with configmaps
		{podsPerNode: 30, runLatencyTest: true, kind: extensions.Kind("Deployment"), configMapsPerPod: 2},
		// Test with quotas
		{podsPerNode: 30, runLatencyTest: true, kind: api.Kind("ReplicationController"), quotas: true},
	}

	isCanonical := func(test *Density) bool {
		return test.kind == api.Kind("ReplicationController") && test.daemonsPerNode == 0 && test.secretsPerPod == 0 && test.configMapsPerPod == 0 && !test.quotas
	}

	for _, testArg := range densityTests {
		feature := "ManualPerformance"
		switch testArg.podsPerNode {
		case 30:
			if isCanonical(&testArg) {
				feature = "Performance"
			}
		case 95:
			feature = "HighDensityPerformance"
		}

		name := fmt.Sprintf("[Feature:%s] should allow starting %d pods per node using %v with %v secrets, %v configmaps and %v daemons",
			feature,
			testArg.podsPerNode,
			testArg.kind,
			testArg.secretsPerPod,
			testArg.configMapsPerPod,
			testArg.daemonsPerNode,
		)
		if testArg.quotas {
			name += " with quotas"
		}
		itArg := testArg
		It(name, func() {
			nodePrepPhase := testPhaseDurations.StartPhase(100, "node preparation")
			defer nodePrepPhase.End()
			nodePreparer := framework.NewE2ETestNodePreparer(
				f.ClientSet,
				[]testutils.CountToStrategy{{Count: nodeCount, Strategy: &testutils.TrivialNodePrepareStrategy{}}},
			)
			framework.ExpectNoError(nodePreparer.PrepareNodes())
			defer nodePreparer.CleanupNodes()

			podsPerNode := itArg.podsPerNode
			if podsPerNode == 30 {
				f.AddonResourceConstraints = func() map[string]framework.ResourceConstraint { return density30AddonResourceVerifier(nodeCount) }()
			}
			totalPods = (podsPerNode - itArg.daemonsPerNode) * nodeCount
			fileHndl, err := os.Create(fmt.Sprintf(framework.TestContext.OutputDir+"/%s/pod_states.csv", uuid))
			framework.ExpectNoError(err)
			defer fileHndl.Close()
			nodePrepPhase.End()

			// nodeCountPerNamespace and CreateNamespaces are defined in load.go
			numberOfCollections := (nodeCount + nodeCountPerNamespace - 1) / nodeCountPerNamespace
			namespaces, err := CreateNamespaces(f, numberOfCollections, fmt.Sprintf("density-%v", testArg.podsPerNode), testPhaseDurations.StartPhase(200, "namespace creation"))
			framework.ExpectNoError(err)
			if itArg.quotas {
				framework.ExpectNoError(CreateQuotas(f, namespaces, totalPods+nodeCount, testPhaseDurations.StartPhase(210, "quota creation")))
			}

			configs := make([]testutils.RunObjectConfig, numberOfCollections)
			secretConfigs := make([]*testutils.SecretConfig, 0, numberOfCollections*itArg.secretsPerPod)
			configMapConfigs := make([]*testutils.ConfigMapConfig, 0, numberOfCollections*itArg.configMapsPerPod)
			// Since all RCs are created at the same time, timeout for each config
			// has to assume that it will be run at the very end.
			podThroughput := 20
			timeout := time.Duration(totalPods/podThroughput)*time.Second + 3*time.Minute
			// createClients is defined in load.go
			clients, internalClients, scalesClients, err := createClients(numberOfCollections)
			framework.ExpectNoError(err)
			for i := 0; i < numberOfCollections; i++ {
				nsName := namespaces[i].Name
				secretNames := []string{}
				for j := 0; j < itArg.secretsPerPod; j++ {
					secretName := fmt.Sprintf("density-secret-%v-%v", i, j)
					secretConfigs = append(secretConfigs, &testutils.SecretConfig{
						Content:   map[string]string{"foo": "bar"},
						Client:    clients[i],
						Name:      secretName,
						Namespace: nsName,
						LogFunc:   framework.Logf,
					})
					secretNames = append(secretNames, secretName)
				}
				configMapNames := []string{}
				for j := 0; j < itArg.configMapsPerPod; j++ {
					configMapName := fmt.Sprintf("density-configmap-%v-%v", i, j)
					configMapConfigs = append(configMapConfigs, &testutils.ConfigMapConfig{
						Content:   map[string]string{"foo": "bar"},
						Client:    clients[i],
						Name:      configMapName,
						Namespace: nsName,
						LogFunc:   framework.Logf,
					})
					configMapNames = append(configMapNames, configMapName)
				}
				name := fmt.Sprintf("density%v-%v-%v", totalPods, i, uuid)
				baseConfig := &testutils.RCConfig{
					Client:               clients[i],
					InternalClient:       internalClients[i],
					ScalesGetter:         scalesClients[i],
					Image:                imageutils.GetPauseImageName(),
					Name:                 name,
					Namespace:            nsName,
					Labels:               map[string]string{"type": "densityPod"},
					PollInterval:         DensityPollInterval,
					Timeout:              timeout,
					PodStatusFile:        fileHndl,
					Replicas:             (totalPods + numberOfCollections - 1) / numberOfCollections,
					CpuRequest:           nodeCpuCapacity / 100,
					MemRequest:           nodeMemCapacity / 100,
					MaxContainerFailures: &MaxContainerFailures,
					Silent:               true,
					LogFunc:              framework.Logf,
					SecretNames:          secretNames,
					ConfigMapNames:       configMapNames,
				}
				switch itArg.kind {
				case api.Kind("ReplicationController"):
					configs[i] = baseConfig
				case extensions.Kind("ReplicaSet"):
					configs[i] = &testutils.ReplicaSetConfig{RCConfig: *baseConfig}
				case extensions.Kind("Deployment"):
					configs[i] = &testutils.DeploymentConfig{RCConfig: *baseConfig}
				case batch.Kind("Job"):
					configs[i] = &testutils.JobConfig{RCConfig: *baseConfig}
				default:
					framework.Failf("Unsupported kind: %v", itArg.kind)
				}
			}

			// Single client is running out of http2 connections in delete phase, hence we need more.
			clients, internalClients, scalesClients, err = createClients(2)
			framework.ExpectNoError(err)
			dConfig := DensityTestConfig{
				ClientSets:         clients,
				InternalClientsets: internalClients,
				ScaleClients:       scalesClients,
				Configs:            configs,
				PodCount:           totalPods,
				PollInterval:       DensityPollInterval,
				kind:               itArg.kind,
				SecretConfigs:      secretConfigs,
				ConfigMapConfigs:   configMapConfigs,
			}

			for i := 0; i < itArg.daemonsPerNode; i++ {
				dConfig.DaemonConfigs = append(dConfig.DaemonConfigs,
					&testutils.DaemonConfig{
						Client:    f.ClientSet,
						Name:      fmt.Sprintf("density-daemon-%v", i),
						Namespace: f.Namespace.Name,
						LogFunc:   framework.Logf,
					})
			}
			e2eStartupTime = runDensityTest(dConfig, testPhaseDurations, &scheduleThroughputs)
			if itArg.runLatencyTest {
				// Pick latencyPodsIterations so that:
				// latencyPodsIterations * nodeCount >= MinPodStartupMeasurements.
				latencyPodsIterations := (MinPodStartupMeasurements + nodeCount - 1) / nodeCount
				By(fmt.Sprintf("Scheduling additional %d Pods to measure startup latencies", latencyPodsIterations*nodeCount))

				createTimes := make(map[string]metav1.Time, 0)
				nodeNames := make(map[string]string, 0)
				scheduleTimes := make(map[string]metav1.Time, 0)
				runTimes := make(map[string]metav1.Time, 0)
				watchTimes := make(map[string]metav1.Time, 0)

				var mutex sync.Mutex
				checkPod := func(p *v1.Pod) {
					mutex.Lock()
					defer mutex.Unlock()
					defer GinkgoRecover()

					if p.Status.Phase == v1.PodRunning {
						if _, found := watchTimes[p.Name]; !found {
							watchTimes[p.Name] = metav1.Now()
							createTimes[p.Name] = p.CreationTimestamp
							nodeNames[p.Name] = p.Spec.NodeName
							var startTime metav1.Time
							for _, cs := range p.Status.ContainerStatuses {
								if cs.State.Running != nil {
									if startTime.Before(&cs.State.Running.StartedAt) {
										startTime = cs.State.Running.StartedAt
									}
								}
							}
							if startTime != metav1.NewTime(time.Time{}) {
								runTimes[p.Name] = startTime
							} else {
								framework.Failf("Pod %v is reported to be running, but none of its containers is", p.Name)
							}
						}
					}
				}

				additionalPodsPrefix = "density-latency-pod"
				stopCh := make(chan struct{})

				latencyPodStores := make([]cache.Store, len(namespaces))
				for i := 0; i < len(namespaces); i++ {
					nsName := namespaces[i].Name
					latencyPodsStore, controller := cache.NewInformer(
						&cache.ListWatch{
							ListFunc: func(options metav1.ListOptions) (runtime.Object, error) {
								options.LabelSelector = labels.SelectorFromSet(labels.Set{"type": additionalPodsPrefix}).String()
								obj, err := c.CoreV1().Pods(nsName).List(options)
								return runtime.Object(obj), err
							},
							WatchFunc: func(options metav1.ListOptions) (watch.Interface, error) {
								options.LabelSelector = labels.SelectorFromSet(labels.Set{"type": additionalPodsPrefix}).String()
								return c.CoreV1().Pods(nsName).Watch(options)
							},
						},
						&v1.Pod{},
						0,
						cache.ResourceEventHandlerFuncs{
							AddFunc: func(obj interface{}) {
								p, ok := obj.(*v1.Pod)
								if !ok {
									framework.Logf("Failed to cast observed object to *v1.Pod.")
								}
								Expect(ok).To(Equal(true))
								go checkPod(p)
							},
							UpdateFunc: func(oldObj, newObj interface{}) {
								p, ok := newObj.(*v1.Pod)
								if !ok {
									framework.Logf("Failed to cast observed object to *v1.Pod.")
								}
								Expect(ok).To(Equal(true))
								go checkPod(p)
							},
						},
					)
					latencyPodStores[i] = latencyPodsStore

					go controller.Run(stopCh)
				}
				for latencyPodsIteration := 0; latencyPodsIteration < latencyPodsIterations; latencyPodsIteration++ {
					podIndexOffset := latencyPodsIteration * nodeCount
					framework.Logf("Creating %d latency pods in range [%d, %d]", nodeCount, podIndexOffset+1, podIndexOffset+nodeCount)

					watchTimesLen := len(watchTimes)

					// Create some additional pods with throughput ~5 pods/sec.
					latencyPodStartupPhase := testPhaseDurations.StartPhase(800+latencyPodsIteration*10, "latency pods creation")
					defer latencyPodStartupPhase.End()
					var wg sync.WaitGroup
					wg.Add(nodeCount)
					// Explicitly set requests here.
					// Thanks to it we trigger increasing priority function by scheduling
					// a pod to a node, which in turn will result in spreading latency pods
					// more evenly between nodes.
					cpuRequest := *resource.NewMilliQuantity(nodeCpuCapacity/5, resource.DecimalSI)
					memRequest := *resource.NewQuantity(nodeMemCapacity/5, resource.DecimalSI)
					if podsPerNode > 30 {
						// This is to make them schedulable on high-density tests
						// (e.g. 100 pods/node kubemark).
						cpuRequest = *resource.NewMilliQuantity(0, resource.DecimalSI)
						memRequest = *resource.NewQuantity(0, resource.DecimalSI)
					}
					rcNameToNsMap := map[string]string{}
					for i := 1; i <= nodeCount; i++ {
						name := additionalPodsPrefix + "-" + strconv.Itoa(podIndexOffset+i)
						nsName := namespaces[i%len(namespaces)].Name
						rcNameToNsMap[name] = nsName
						go createRunningPodFromRC(&wg, c, name, nsName, imageutils.GetPauseImageName(), additionalPodsPrefix, cpuRequest, memRequest)
						time.Sleep(200 * time.Millisecond)
					}
					wg.Wait()
					latencyPodStartupPhase.End()

					latencyMeasurementPhase := testPhaseDurations.StartPhase(801+latencyPodsIteration*10, "pod startup latencies measurement")
					defer latencyMeasurementPhase.End()
					By("Waiting for all Pods begin observed by the watch...")
					waitTimeout := 10 * time.Minute
					for start := time.Now(); len(watchTimes) < watchTimesLen+nodeCount; time.Sleep(10 * time.Second) {
						if time.Since(start) < waitTimeout {
							framework.Failf("Timeout reached waiting for all Pods being observed by the watch.")
						}
					}

					nodeToLatencyPods := make(map[string]int)
					for i := range latencyPodStores {
						for _, item := range latencyPodStores[i].List() {
							pod := item.(*v1.Pod)
							nodeToLatencyPods[pod.Spec.NodeName]++
						}
						for node, count := range nodeToLatencyPods {
							if count > 1 {
								framework.Logf("%d latency pods scheduled on %s", count, node)
							}
						}
					}
					latencyMeasurementPhase.End()

					By("Removing additional replication controllers")
					podDeletionPhase := testPhaseDurations.StartPhase(802+latencyPodsIteration*10, "latency pods deletion")
					defer podDeletionPhase.End()
					deleteRC := func(i int) {
						defer GinkgoRecover()
						name := additionalPodsPrefix + "-" + strconv.Itoa(podIndexOffset+i+1)
						framework.ExpectNoError(framework.DeleteRCAndWaitForGC(c, rcNameToNsMap[name], name))
					}
					workqueue.Parallelize(25, nodeCount, deleteRC)
					podDeletionPhase.End()
				}
				close(stopCh)

				for i := 0; i < len(namespaces); i++ {
					nsName := namespaces[i].Name
					selector := fields.Set{
						"involvedObject.kind":      "Pod",
						"involvedObject.namespace": nsName,
						"source":                   v1.DefaultSchedulerName,
					}.AsSelector().String()
					options := metav1.ListOptions{FieldSelector: selector}
					schedEvents, err := c.CoreV1().Events(nsName).List(options)
					framework.ExpectNoError(err)
					for k := range createTimes {
						for _, event := range schedEvents.Items {
							if event.InvolvedObject.Name == k {
								scheduleTimes[k] = event.FirstTimestamp
								break
							}
						}
					}
				}

				scheduleLag := make([]framework.PodLatencyData, 0)
				startupLag := make([]framework.PodLatencyData, 0)
				watchLag := make([]framework.PodLatencyData, 0)
				schedToWatchLag := make([]framework.PodLatencyData, 0)
				e2eLag := make([]framework.PodLatencyData, 0)

				for name, create := range createTimes {
					sched, ok := scheduleTimes[name]
					if !ok {
						framework.Logf("Failed to find schedule time for %v", name)
						missingMeasurements++
					}
					run, ok := runTimes[name]
					if !ok {
						framework.Logf("Failed to find run time for %v", name)
						missingMeasurements++
					}
					watch, ok := watchTimes[name]
					if !ok {
						framework.Logf("Failed to find watch time for %v", name)
						missingMeasurements++
					}
					node, ok := nodeNames[name]
					if !ok {
						framework.Logf("Failed to find node for %v", name)
						missingMeasurements++
					}

					scheduleLag = append(scheduleLag, framework.PodLatencyData{Name: name, Node: node, Latency: sched.Time.Sub(create.Time)})
					startupLag = append(startupLag, framework.PodLatencyData{Name: name, Node: node, Latency: run.Time.Sub(sched.Time)})
					watchLag = append(watchLag, framework.PodLatencyData{Name: name, Node: node, Latency: watch.Time.Sub(run.Time)})
					schedToWatchLag = append(schedToWatchLag, framework.PodLatencyData{Name: name, Node: node, Latency: watch.Time.Sub(sched.Time)})
					e2eLag = append(e2eLag, framework.PodLatencyData{Name: name, Node: node, Latency: watch.Time.Sub(create.Time)})
				}

				sort.Sort(framework.LatencySlice(scheduleLag))
				sort.Sort(framework.LatencySlice(startupLag))
				sort.Sort(framework.LatencySlice(watchLag))
				sort.Sort(framework.LatencySlice(schedToWatchLag))
				sort.Sort(framework.LatencySlice(e2eLag))

				framework.PrintLatencies(scheduleLag, "worst create-to-schedule latencies")
				framework.PrintLatencies(startupLag, "worst schedule-to-run latencies")
				framework.PrintLatencies(watchLag, "worst run-to-watch latencies")
				framework.PrintLatencies(schedToWatchLag, "worst schedule-to-watch latencies")
				framework.PrintLatencies(e2eLag, "worst e2e latencies")

				// Capture latency metrics related to pod-startup.
				podStartupLatency := &framework.PodStartupLatency{
					CreateToScheduleLatency: framework.ExtractLatencyMetrics(scheduleLag),
					ScheduleToRunLatency:    framework.ExtractLatencyMetrics(startupLag),
					RunToWatchLatency:       framework.ExtractLatencyMetrics(watchLag),
					ScheduleToWatchLatency:  framework.ExtractLatencyMetrics(schedToWatchLag),
					E2ELatency:              framework.ExtractLatencyMetrics(e2eLag),
				}
				f.TestSummaries = append(f.TestSummaries, podStartupLatency)

				// Test whether e2e pod startup time is acceptable.
				podStartupLatencyThreshold := framework.LatencyMetric{
					Perc50: PodStartupLatencyThreshold,
					Perc90: PodStartupLatencyThreshold,
					Perc99: PodStartupLatencyThreshold,
				}
				framework.ExpectNoError(framework.VerifyLatencyWithinThreshold(podStartupLatencyThreshold, podStartupLatency.E2ELatency, "pod startup"))

				framework.LogSuspiciousLatency(startupLag, e2eLag, nodeCount, c)
			}
			cleanupDensityTest(dConfig, testPhaseDurations)
		})
	}
})

func createRunningPodFromRC(wg *sync.WaitGroup, c clientset.Interface, name, ns, image, podType string, cpuRequest, memRequest resource.Quantity) {
	defer GinkgoRecover()
	defer wg.Done()
	labels := map[string]string{
		"type": podType,
		"name": name,
	}
	rc := &v1.ReplicationController{
		ObjectMeta: metav1.ObjectMeta{
			Name:   name,
			Labels: labels,
		},
		Spec: v1.ReplicationControllerSpec{
			Replicas: func(i int) *int32 { x := int32(i); return &x }(1),
			Selector: labels,
			Template: &v1.PodTemplateSpec{
				ObjectMeta: metav1.ObjectMeta{
					Labels: labels,
				},
				Spec: v1.PodSpec{
					Containers: []v1.Container{
						{
							Name:  name,
							Image: image,
							Resources: v1.ResourceRequirements{
								Requests: v1.ResourceList{
									v1.ResourceCPU:    cpuRequest,
									v1.ResourceMemory: memRequest,
								},
							},
						},
					},
					DNSPolicy: v1.DNSDefault,
				},
			},
		},
	}
	framework.ExpectNoError(testutils.CreateRCWithRetries(c, ns, rc))
	framework.ExpectNoError(framework.WaitForControlledPodsRunning(c, ns, name, api.Kind("ReplicationController")))
	framework.Logf("Found pod '%s' running", name)
}
