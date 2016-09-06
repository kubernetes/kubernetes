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

package e2e

import (
	"fmt"
	"math"
	"math/rand"
	"os"
	"strconv"
	"sync"
	"time"

	"k8s.io/kubernetes/pkg/api"
	client "k8s.io/kubernetes/pkg/client/unversioned"
	"k8s.io/kubernetes/pkg/labels"
	"k8s.io/kubernetes/pkg/util/intstr"
	"k8s.io/kubernetes/test/e2e/framework"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
)

const (
	smallRCSize       = 5
	mediumRCSize      = 30
	bigRCSize         = 250
	smallRCGroupName  = "load-small-rc"
	mediumRCGroupName = "load-medium-rc"
	bigRCGroupName    = "load-big-rc"
	smallRCBatchSize  = 30
	mediumRCBatchSize = 5
	bigRCBatchSize    = 1
	// We start RCs/Services/pods/... in different namespace in this test.
	// nodeCountPerNamespace determines how many namespaces we will be using
	// depending on the number of nodes in the underlying cluster.
	nodeCountPerNamespace = 100
)

// This test suite can take a long time to run, so by default it is added to
// the ginkgo.skip list (see driver.go).
// To run this suite you must explicitly ask for it by setting the
// -t/--test flag or ginkgo.focus flag.
var _ = framework.KubeDescribe("Load capacity", func() {
	var c *client.Client
	var nodeCount int
	var ns string
	var configs []*framework.RCConfig
	var namespaces []*api.Namespace

	// Gathers metrics before teardown
	// TODO add flag that allows to skip cleanup on failure
	AfterEach(func() {
		// Verify latency metrics
		highLatencyRequests, err := framework.HighLatencyRequests(c)
		framework.ExpectNoError(err, "Too many instances metrics above the threshold")
		Expect(highLatencyRequests).NotTo(BeNumerically(">", 0))
	})

	// We assume a default throughput of 10 pods/second throughput.
	// We may want to revisit it in the future.
	// However, this can be overriden by LOAD_TEST_THROUGHPUT env var.
	throughput := 10
	if throughputEnv := os.Getenv("LOAD_TEST_THROUGHPUT"); throughputEnv != "" {
		if newThroughput, err := strconv.Atoi(throughputEnv); err == nil {
			throughput = newThroughput
		}
	}

	// Explicitly put here, to delete namespace at the end of the test
	// (after measuring latency metrics, etc.).
	options := framework.FrameworkOptions{
		ClientQPS:   float32(math.Max(50.0, float64(2*throughput))),
		ClientBurst: int(math.Max(100.0, float64(4*throughput))),
	}
	f := framework.NewFramework("load", options, nil)
	f.NamespaceDeletionTimeout = time.Hour

	BeforeEach(func() {
		c = f.Client

		// In large clusters we may get to this point but still have a bunch
		// of nodes without Routes created. Since this would make a node
		// unschedulable, we need to wait until all of them are schedulable.
		framework.ExpectNoError(framework.WaitForAllNodesSchedulable(c))

		ns = f.Namespace.Name
		nodes := framework.GetReadySchedulableNodesOrDie(c)
		nodeCount = len(nodes.Items)
		Expect(nodeCount).NotTo(BeZero())

		// Terminating a namespace (deleting the remaining objects from it - which
		// generally means events) can affect the current run. Thus we wait for all
		// terminating namespace to be finally deleted before starting this test.
		err := framework.CheckTestingNSDeletedExcept(c, ns)
		framework.ExpectNoError(err)

		framework.ExpectNoError(framework.ResetMetrics(c))
	})

	type Load struct {
		podsPerNode int
		image       string
		command     []string
	}

	loadTests := []Load{
		// The container will consume 1 cpu and 512mb of memory.
		{podsPerNode: 3, image: "jess/stress", command: []string{"stress", "-c", "1", "-m", "2"}},
		{podsPerNode: 30, image: "gcr.io/google_containers/serve_hostname:v1.4"},
	}

	for _, testArg := range loadTests {
		name := fmt.Sprintf("should be able to handle %v pods per node", testArg.podsPerNode)
		if testArg.podsPerNode == 30 {
			name = "[Feature:Performance] " + name
		} else {
			name = "[Feature:ManualPerformance] " + name
		}
		itArg := testArg

		It(name, func() {
			// Create a number of namespaces.
			namespaces = createNamespaces(f, nodeCount, itArg.podsPerNode)

			totalPods := itArg.podsPerNode * nodeCount
			configs = generateRCConfigs(totalPods, itArg.image, itArg.command, c, namespaces)
			var services []*api.Service
			// Read the environment variable to see if we want to create services
			createServices := os.Getenv("CREATE_SERVICES")
			if createServices == "true" {
				framework.Logf("Creating services")
				services := generateServicesForConfigs(configs)
				for _, service := range services {
					_, err := c.Services(service.Namespace).Create(service)
					framework.ExpectNoError(err)
				}
				framework.Logf("%v Services created.", len(services))
			} else {
				framework.Logf("Skipping service creation")
			}

			// Simulate lifetime of RC:
			//  * create with initial size
			//  * scale RC to a random size and list all pods
			//  * scale RC to a random size and list all pods
			//  * delete it
			//
			// This will generate ~5 creations/deletions per second assuming:
			//  - X small RCs each 5 pods   [ 5 * X = totalPods / 2 ]
			//  - Y medium RCs each 30 pods [ 30 * Y = totalPods / 4 ]
			//  - Z big RCs each 250 pods   [ 250 * Z = totalPods / 4]

			// We would like to spread creating replication controllers over time
			// to make it possible to create/schedule them in the meantime.
			// Currently we assume <throughput> pods/second average throughput.
			// We may want to revisit it in the future.
			framework.Logf("Starting to create ReplicationControllers...")
			creatingTime := time.Duration(totalPods/throughput) * time.Second
			createAllRC(configs, creatingTime)
			By("============================================================================")

			// We would like to spread scaling replication controllers over time
			// to make it possible to create/schedule & delete them in the meantime.
			// Currently we assume that <throughput> pods/second average throughput.
			// The expected number of created/deleted pods is less than totalPods/3.
			scalingTime := time.Duration(totalPods/(3*throughput)) * time.Second
			framework.Logf("Starting to scale ReplicationControllers first time...")
			scaleAllRC(configs, scalingTime)
			By("============================================================================")

			framework.Logf("Starting to scale ReplicationControllers second time...")
			scaleAllRC(configs, scalingTime)
			By("============================================================================")

			// Cleanup all created replication controllers.
			// Currently we assume <throughput> pods/second average deletion throughput.
			// We may want to revisit it in the future.
			deletingTime := time.Duration(totalPods/throughput) * time.Second
			framework.Logf("Starting to delete ReplicationControllers...")
			deleteAllRC(configs, deletingTime)
			if createServices == "true" {
				framework.Logf("Starting to delete services...")
				for _, service := range services {
					err := c.Services(ns).Delete(service.Name)
					framework.ExpectNoError(err)
				}
				framework.Logf("Services deleted")
			}
		})
	}
})

func createNamespaces(f *framework.Framework, nodeCount, podsPerNode int) []*api.Namespace {
	namespaceCount := (nodeCount + nodeCountPerNamespace - 1) / nodeCountPerNamespace
	namespaces := []*api.Namespace{}
	for i := 1; i <= namespaceCount; i++ {
		namespace, err := f.CreateNamespace(fmt.Sprintf("load-%d-nodepods-%d", podsPerNode, i), nil)
		framework.ExpectNoError(err)
		namespaces = append(namespaces, namespace)
	}
	return namespaces
}

func computeRCCounts(total int) (int, int, int) {
	// Small RCs owns ~0.5 of total number of pods, medium and big RCs ~0.25 each.
	// For example for 3000 pods (100 nodes, 30 pods per node) there are:
	//  - 300 small RCs each 5 pods
	//  - 25 medium RCs each 30 pods
	//  - 3 big RCs each 250 pods
	bigRCCount := total / 4 / bigRCSize
	total -= bigRCCount * bigRCSize
	mediumRCCount := total / 3 / mediumRCSize
	total -= mediumRCCount * mediumRCSize
	smallRCCount := total / smallRCSize
	return smallRCCount, mediumRCCount, bigRCCount
}

func generateRCConfigs(totalPods int, image string, command []string, c *client.Client, nss []*api.Namespace) []*framework.RCConfig {
	configs := make([]*framework.RCConfig, 0)

	smallRCCount, mediumRCCount, bigRCCount := computeRCCounts(totalPods)
	configs = append(configs, generateRCConfigsForGroup(c, nss, smallRCGroupName, smallRCSize, smallRCCount, image, command)...)
	configs = append(configs, generateRCConfigsForGroup(c, nss, mediumRCGroupName, mediumRCSize, mediumRCCount, image, command)...)
	configs = append(configs, generateRCConfigsForGroup(c, nss, bigRCGroupName, bigRCSize, bigRCCount, image, command)...)

	return configs
}

func generateRCConfigsForGroup(c *client.Client, nss []*api.Namespace, groupName string, size, count int, image string, command []string) []*framework.RCConfig {
	configs := make([]*framework.RCConfig, 0, count)
	for i := 1; i <= count; i++ {
		config := &framework.RCConfig{
			Client:     c,
			Name:       groupName + "-" + strconv.Itoa(i),
			Namespace:  nss[i%len(nss)].Name,
			Timeout:    10 * time.Minute,
			Image:      image,
			Command:    command,
			Replicas:   size,
			CpuRequest: 10,       // 0.01 core
			MemRequest: 26214400, // 25MB
		}
		configs = append(configs, config)
	}
	return configs
}

func generateServicesForConfigs(configs []*framework.RCConfig) []*api.Service {
	services := make([]*api.Service, 0, len(configs))
	for _, config := range configs {
		serviceName := config.Name + "-svc"
		labels := map[string]string{"name": config.Name}
		service := &api.Service{
			ObjectMeta: api.ObjectMeta{
				Name:      serviceName,
				Namespace: config.Namespace,
			},
			Spec: api.ServiceSpec{
				Selector: labels,
				Ports: []api.ServicePort{{
					Port:       80,
					TargetPort: intstr.FromInt(80),
				}},
			},
		}
		services = append(services, service)
	}
	return services
}

func sleepUpTo(d time.Duration) {
	time.Sleep(time.Duration(rand.Int63n(d.Nanoseconds())))
}

func createAllRC(configs []*framework.RCConfig, creatingTime time.Duration) {
	var wg sync.WaitGroup
	wg.Add(len(configs))
	for _, config := range configs {
		go createRC(&wg, config, creatingTime)
	}
	wg.Wait()
}

func createRC(wg *sync.WaitGroup, config *framework.RCConfig, creatingTime time.Duration) {
	defer GinkgoRecover()
	defer wg.Done()

	sleepUpTo(creatingTime)
	framework.ExpectNoError(framework.RunRC(*config), fmt.Sprintf("creating rc %s", config.Name))
}

func scaleAllRC(configs []*framework.RCConfig, scalingTime time.Duration) {
	var wg sync.WaitGroup
	wg.Add(len(configs))
	for _, config := range configs {
		go scaleRC(&wg, config, scalingTime)
	}
	wg.Wait()
}

// Scales RC to a random size within [0.5*size, 1.5*size] and lists all the pods afterwards.
// Scaling happens always based on original size, not the current size.
func scaleRC(wg *sync.WaitGroup, config *framework.RCConfig, scalingTime time.Duration) {
	defer GinkgoRecover()
	defer wg.Done()

	sleepUpTo(scalingTime)
	newSize := uint(rand.Intn(config.Replicas) + config.Replicas/2)
	framework.ExpectNoError(framework.ScaleRC(config.Client, config.Namespace, config.Name, newSize, true),
		fmt.Sprintf("scaling rc %s for the first time", config.Name))
	selector := labels.SelectorFromSet(labels.Set(map[string]string{"name": config.Name}))
	options := api.ListOptions{
		LabelSelector:   selector,
		ResourceVersion: "0",
	}
	_, err := config.Client.Pods(config.Namespace).List(options)
	framework.ExpectNoError(err, fmt.Sprintf("listing pods from rc %v", config.Name))
}

func deleteAllRC(configs []*framework.RCConfig, deletingTime time.Duration) {
	var wg sync.WaitGroup
	wg.Add(len(configs))
	for _, config := range configs {
		go deleteRC(&wg, config, deletingTime)
	}
	wg.Wait()
}

func deleteRC(wg *sync.WaitGroup, config *framework.RCConfig, deletingTime time.Duration) {
	defer GinkgoRecover()
	defer wg.Done()

	sleepUpTo(deletingTime)
	if framework.TestContext.GarbageCollectorEnabled {
		framework.ExpectNoError(framework.DeleteRCAndWaitForGC(config.Client, config.Namespace, config.Name), fmt.Sprintf("deleting rc %s", config.Name))
	} else {
		framework.ExpectNoError(framework.DeleteRCAndPods(config.Client, config.Namespace, config.Name), fmt.Sprintf("deleting rc %s", config.Name))
	}
}
