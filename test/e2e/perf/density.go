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

package perf

import (
	"fmt"
	"math"
	"os"
	"sort"
	"strconv"
	"sync"
	"time"

	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/sets"
	utiluuid "k8s.io/apimachinery/pkg/util/uuid"
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/client-go/tools/cache"
	"k8s.io/client-go/util/workqueue"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/apis/batch"
	"k8s.io/kubernetes/pkg/apis/extensions"
	"k8s.io/kubernetes/pkg/client/clientset_generated/clientset"
	"k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset"
	"k8s.io/kubernetes/test/e2e/framework"
	testutils "k8s.io/kubernetes/test/utils"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
)

const (
	MinSaturationThreshold     = 2 * time.Minute
	MinPodsPerSecondThroughput = 8
	DensityPollInterval        = 10 * time.Second
)

// Maximum container failures this test tolerates before failing.
var MaxContainerFailures = 0

type DensityTestConfig struct {
	Configs           []testutils.RunObjectConfig
	ClientSet         clientset.Interface
	InternalClientset internalclientset.Interface
	PollInterval      time.Duration
	PodCount          int
	// What kind of resource we want to create
	kind          schema.GroupKind
	SecretConfigs []*testutils.SecretConfig
	DaemonConfigs []*testutils.DaemonConfig
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
	framework.Logf("Setting resource constraings for provider: %s", framework.TestContext.Provider)
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
			controllerCPU = 0.75
			controllerMem = 750 * (1024 * 1024)
			schedulerCPU = 0.75
			schedulerMem = 500 * (1024 * 1024)
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
			// TODO: Investigate higher apiserver consumption and
			// potentially revert to 1.5cpu and 1.3GB - see #30871
			apiserverCPU = 1.8
			apiserverMem = 2200 * (1024 * 1024)
			controllerCPU = 0.5
			controllerMem = 300 * (1024 * 1024)
			schedulerCPU = 0.4
			schedulerMem = 150 * (1024 * 1024)
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
		CPUConstraint: 0.15,
		// When we are running purely density test, 30MB seems to be enough.
		// However, we are usually running Density together with Load test.
		// Thus, if Density is running after Load (which is creating and
		// propagating a bunch of services), kubeproxy is using much more
		// memory and not releasing it afterwards.
		MemoryConstraint: 60 * (1024 * 1024),
	}
	constraints["l7-lb-controller"] = framework.ResourceConstraint{
		CPUConstraint:    0.15,
		MemoryConstraint: 60 * (1024 * 1024),
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

func logPodStartupStatus(c clientset.Interface, expectedPods int, observedLabels map[string]string, period time.Duration, stopCh chan struct{}) {
	label := labels.SelectorFromSet(labels.Set(observedLabels))
	podStore := testutils.NewPodStore(c, metav1.NamespaceAll, label, fields.Everything())
	defer podStore.Stop()
	ticker := time.NewTicker(period)
	defer ticker.Stop()
	for {
		select {
		case <-ticker.C:
			pods := podStore.List()
			startupStatus := testutils.ComputeRCStartupStatus(pods, expectedPods)
			framework.Logf(startupStatus.String("Density"))
		case <-stopCh:
			pods := podStore.List()
			startupStatus := testutils.ComputeRCStartupStatus(pods, expectedPods)
			framework.Logf(startupStatus.String("Density"))
			return
		}
	}
}

// runDensityTest will perform a density test and return the time it took for
// all pods to start
func runDensityTest(dtc DensityTestConfig) time.Duration {
	defer GinkgoRecover()

	// Create all secrets
	for i := range dtc.SecretConfigs {
		dtc.SecretConfigs[i].Run()
	}

	for i := range dtc.DaemonConfigs {
		dtc.DaemonConfigs[i].Run()
	}

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
	go logPodStartupStatus(dtc.ClientSet, dtc.PodCount, map[string]string{"type": "densityPod"}, dtc.PollInterval, logStopCh)
	wg.Wait()
	startupTime := time.Now().Sub(startTime)
	close(logStopCh)
	framework.Logf("E2E startup time for %d pods: %v", dtc.PodCount, startupTime)
	framework.Logf("Throughput (pods/s) during cluster saturation phase: %v", float32(dtc.PodCount)/float32(startupTime/time.Second))

	// Print some data about Pod to Node allocation
	By("Printing Pod to Node allocation data")
	podList, err := dtc.ClientSet.Core().Pods(metav1.NamespaceAll).List(metav1.ListOptions{})
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
	return startupTime
}

func cleanupDensityTest(dtc DensityTestConfig) {
	defer GinkgoRecover()
	By("Deleting created Collections")
	// We explicitly delete all pods to have API calls necessary for deletion accounted in metrics.
	for i := range dtc.Configs {
		name := dtc.Configs[i].GetName()
		namespace := dtc.Configs[i].GetNamespace()
		kind := dtc.Configs[i].GetKind()
		if framework.TestContext.GarbageCollectorEnabled && kindSupportsGarbageCollector(kind) {
			By(fmt.Sprintf("Cleaning up only the %v, garbage collector will clean up the pods", kind))
			err := framework.DeleteResourceAndWaitForGC(dtc.ClientSet, kind, namespace, name)
			framework.ExpectNoError(err)
		} else {
			By(fmt.Sprintf("Cleaning up the %v and pods", kind))
			err := framework.DeleteResourceAndPods(dtc.ClientSet, dtc.InternalClientset, kind, namespace, name)
			framework.ExpectNoError(err)
		}
	}

	// Delete all secrets
	for i := range dtc.SecretConfigs {
		dtc.SecretConfigs[i].Stop()
	}

	for i := range dtc.DaemonConfigs {
		framework.ExpectNoError(framework.DeleteResourceAndPods(
			dtc.ClientSet,
			dtc.InternalClientset,
			extensions.Kind("DaemonSet"),
			dtc.DaemonConfigs[i].Namespace,
			dtc.DaemonConfigs[i].Name,
		))
	}
}

// This test suite can take a long time to run, and can affect or be affected by other tests.
// So by default it is added to the ginkgo.skip list (see driver.go).
// To run this suite you must explicitly ask for it by setting the
// -t/--test flag or ginkgo.focus flag.
// IMPORTANT: This test is designed to work on large (>= 100 Nodes) clusters. For smaller ones
// results will not be representative for control-plane performance as we'll start hitting
// limits on Docker's concurrent container startup.
var _ = framework.KubeDescribe("Density", func() {
	var c clientset.Interface
	var nodeCount int
	var name string
	var additionalPodsPrefix string
	var ns string
	var uuid string
	var e2eStartupTime time.Duration
	var totalPods int
	var nodeCpuCapacity int64
	var nodeMemCapacity int64
	var nodes *v1.NodeList
	var masters sets.String

	// Gathers data prior to framework namespace teardown
	AfterEach(func() {
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

		// Verify latency metrics.
		highLatencyRequests, err := framework.HighLatencyRequests(c)
		framework.ExpectNoError(err)
		Expect(highLatencyRequests).NotTo(BeNumerically(">", 0), "There should be no high-latency requests")

		// Verify scheduler metrics.
		// TODO: Reset metrics at the beginning of the test.
		// We should do something similar to how we do it for APIserver.
		if err = framework.VerifySchedulerLatency(c); err != nil {
			framework.Logf("Warning: Scheduler latency not calculated, %v", err)
		}
	})

	options := framework.FrameworkOptions{
		ClientQPS:   50.0,
		ClientBurst: 100,
	}
	// Explicitly put here, to delete namespace at the end of the test
	// (after measuring latency metrics, etc.).
	f := framework.NewFramework("density", options, nil)
	f.NamespaceDeletionTimeout = time.Hour

	BeforeEach(func() {
		c = f.ClientSet
		ns = f.Namespace.Name

		masters, nodes = framework.GetMasterAndWorkerNodesOrDie(c)
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
	})

	type Density struct {
		// Controls if e2e latency tests should be run (they are slow)
		runLatencyTest bool
		podsPerNode    int
		// Controls how often the apiserver is polled for pods
		interval time.Duration
		// What kind of resource we should be creating. Default: ReplicationController
		kind           schema.GroupKind
		secretsPerPod  int
		daemonsPerNode int
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
	}

	for _, testArg := range densityTests {
		feature := "ManualPerformance"
		switch testArg.podsPerNode {
		case 30:
			if testArg.kind == api.Kind("ReplicationController") && testArg.daemonsPerNode == 0 && testArg.secretsPerPod == 0 {
				feature = "Performance"
			}
		case 95:
			feature = "HighDensityPerformance"
		}

		name := fmt.Sprintf("[Feature:%s] should allow starting %d pods per node using %v with %v secrets and %v daemons",
			feature,
			testArg.podsPerNode,
			testArg.kind,
			testArg.secretsPerPod,
			testArg.daemonsPerNode,
		)
		itArg := testArg
		It(name, func() {
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

			// nodeCountPerNamespace and CreateNamespaces are defined in load.go
			numberOfCollections := (nodeCount + nodeCountPerNamespace - 1) / nodeCountPerNamespace
			namespaces, err := CreateNamespaces(f, numberOfCollections, fmt.Sprintf("density-%v", testArg.podsPerNode))
			framework.ExpectNoError(err)

			configs := make([]testutils.RunObjectConfig, numberOfCollections)
			secretConfigs := make([]*testutils.SecretConfig, 0, numberOfCollections*itArg.secretsPerPod)
			// Since all RCs are created at the same time, timeout for each config
			// has to assume that it will be run at the very end.
			podThroughput := 20
			timeout := time.Duration(totalPods/podThroughput)*time.Second + 3*time.Minute
			// createClients is defined in load.go
			clients, internalClients, err := createClients(numberOfCollections)
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
				name := fmt.Sprintf("density%v-%v-%v", totalPods, i, uuid)
				baseConfig := &testutils.RCConfig{
					Client:               clients[i],
					InternalClient:       internalClients[i],
					Image:                framework.GetPauseImageName(f.ClientSet),
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

			dConfig := DensityTestConfig{
				ClientSet:         f.ClientSet,
				InternalClientset: f.InternalClientset,
				Configs:           configs,
				PodCount:          totalPods,
				PollInterval:      DensityPollInterval,
				kind:              itArg.kind,
				SecretConfigs:     secretConfigs,
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
			e2eStartupTime = runDensityTest(dConfig)
			if itArg.runLatencyTest {
				By("Scheduling additional Pods to measure startup latencies")

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
									if startTime.Before(cs.State.Running.StartedAt) {
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
								obj, err := c.Core().Pods(nsName).List(options)
								return runtime.Object(obj), err
							},
							WatchFunc: func(options metav1.ListOptions) (watch.Interface, error) {
								options.LabelSelector = labels.SelectorFromSet(labels.Set{"type": additionalPodsPrefix}).String()
								return c.Core().Pods(nsName).Watch(options)
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

				// Create some additional pods with throughput ~5 pods/sec.
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
					name := additionalPodsPrefix + "-" + strconv.Itoa(i)
					nsName := namespaces[i%len(namespaces)].Name
					rcNameToNsMap[name] = nsName
					go createRunningPodFromRC(&wg, c, name, nsName, framework.GetPauseImageName(f.ClientSet), additionalPodsPrefix, cpuRequest, memRequest)
					time.Sleep(200 * time.Millisecond)
				}
				wg.Wait()

				By("Waiting for all Pods begin observed by the watch...")
				waitTimeout := 10 * time.Minute
				for start := time.Now(); len(watchTimes) < nodeCount; time.Sleep(10 * time.Second) {
					if time.Since(start) < waitTimeout {
						framework.Failf("Timeout reached waiting for all Pods being observed by the watch.")
					}
				}
				close(stopCh)

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

				for i := 0; i < len(namespaces); i++ {
					nsName := namespaces[i].Name
					selector := fields.Set{
						"involvedObject.kind":      "Pod",
						"involvedObject.namespace": nsName,
						"source":                   v1.DefaultSchedulerName,
					}.AsSelector().String()
					options := metav1.ListOptions{FieldSelector: selector}
					schedEvents, err := c.Core().Events(nsName).List(options)
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
					}
					Expect(ok).To(Equal(true))
					run, ok := runTimes[name]
					if !ok {
						framework.Logf("Failed to find run time for %v", name)
					}
					Expect(ok).To(Equal(true))
					watch, ok := watchTimes[name]
					if !ok {
						framework.Logf("Failed to find watch time for %v", name)
					}
					Expect(ok).To(Equal(true))
					node, ok := nodeNames[name]
					if !ok {
						framework.Logf("Failed to find node for %v", name)
					}
					Expect(ok).To(Equal(true))

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

				framework.PrintLatencies(scheduleLag, "worst schedule latencies")
				framework.PrintLatencies(startupLag, "worst run-after-schedule latencies")
				framework.PrintLatencies(watchLag, "worst watch latencies")
				framework.PrintLatencies(schedToWatchLag, "worst scheduled-to-end total latencies")
				framework.PrintLatencies(e2eLag, "worst e2e total latencies")

				// Test whether e2e pod startup time is acceptable.
				podStartupLatency := framework.PodStartupLatency{Latency: framework.ExtractLatencyMetrics(e2eLag)}
				framework.ExpectNoError(framework.VerifyPodStartupLatency(podStartupLatency))

				framework.LogSuspiciousLatency(startupLag, e2eLag, nodeCount, c)

				By("Removing additional replication controllers")
				deleteRC := func(i int) {
					defer GinkgoRecover()
					name := additionalPodsPrefix + "-" + strconv.Itoa(i+1)
					framework.ExpectNoError(framework.DeleteRCAndWaitForGC(c, rcNameToNsMap[name], name))
				}
				workqueue.Parallelize(25, nodeCount, deleteRC)
			}

			cleanupDensityTest(dConfig)
		})
	}

	// Calculate total number of pods from each node's max-pod
	It("[Feature:ManualPerformance] should allow running maximum capacity pods on nodes", func() {
		totalPods = 0
		for _, n := range nodes.Items {
			totalPods += int(n.Status.Capacity.Pods().Value())
		}
		totalPods -= framework.WaitForStableCluster(c, masters)

		fileHndl, err := os.Create(fmt.Sprintf(framework.TestContext.OutputDir+"/%s/pod_states.csv", uuid))
		framework.ExpectNoError(err)
		defer fileHndl.Close()
		collectionCount := 1
		configs := make([]testutils.RunObjectConfig, collectionCount)
		podsPerCollection := int(totalPods / collectionCount)
		for i := 0; i < collectionCount; i++ {
			if i == collectionCount-1 {
				podsPerCollection += int(math.Mod(float64(totalPods), float64(collectionCount)))
			}
			name = "density" + strconv.Itoa(totalPods) + "-" + strconv.Itoa(i) + "-" + uuid
			configs[i] = &testutils.RCConfig{Client: c,
				Image:                framework.GetPauseImageName(f.ClientSet),
				Name:                 name,
				Namespace:            ns,
				Labels:               map[string]string{"type": "densityPod"},
				PollInterval:         DensityPollInterval,
				PodStatusFile:        fileHndl,
				Replicas:             podsPerCollection,
				MaxContainerFailures: &MaxContainerFailures,
				Silent:               true,
				LogFunc:              framework.Logf,
			}
		}
		dConfig := DensityTestConfig{
			ClientSet:    f.ClientSet,
			Configs:      configs,
			PodCount:     totalPods,
			PollInterval: DensityPollInterval,
		}
		e2eStartupTime = runDensityTest(dConfig)
		cleanupDensityTest(dConfig)
	})
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
	_, err := c.Core().ReplicationControllers(ns).Create(rc)
	framework.ExpectNoError(err)
	framework.ExpectNoError(framework.WaitForControlledPodsRunning(c, ns, name, api.Kind("ReplicationController")))
	framework.Logf("Found pod '%s' running", name)
}

func kindSupportsGarbageCollector(kind schema.GroupKind) bool {
	return kind != extensions.Kind("Deployment") && kind != batch.Kind("Job")
}
