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
	"io/ioutil"
	"os"
	"path"
	"sync"
	"testing"
	"time"

	"github.com/golang/glog"
	"github.com/onsi/ginkgo"
	"github.com/onsi/ginkgo/config"
	"github.com/onsi/ginkgo/reporters"
	"github.com/onsi/gomega"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	runtimeutils "k8s.io/apimachinery/pkg/util/runtime"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/kubernetes/pkg/cloudprovider/providers/azure"
	gcecloud "k8s.io/kubernetes/pkg/cloudprovider/providers/gce"
	"k8s.io/kubernetes/pkg/kubectl/util/logs"
	"k8s.io/kubernetes/pkg/version"
	commontest "k8s.io/kubernetes/test/e2e/common"
	"k8s.io/kubernetes/test/e2e/framework"
	"k8s.io/kubernetes/test/e2e/framework/ginkgowrapper"
	"k8s.io/kubernetes/test/e2e/framework/metrics"
	"k8s.io/kubernetes/test/e2e/manifest"
	testutils "k8s.io/kubernetes/test/utils"
)

var (
	cloudConfig = &framework.TestContext.CloudConfig
)

// setupProviderConfig validates and sets up cloudConfig based on framework.TestContext.Provider.
func setupProviderConfig() error {
	switch framework.TestContext.Provider {
	case "":
		glog.Info("The --provider flag is not set.  Treating as a conformance test.  Some tests may not be run.")

	case "gce", "gke":
		framework.Logf("Fetching cloud provider for %q\r\n", framework.TestContext.Provider)
		zone := framework.TestContext.CloudConfig.Zone
		region := framework.TestContext.CloudConfig.Region

		var err error
		if region == "" {
			region, err = gcecloud.GetGCERegion(zone)
			if err != nil {
				return fmt.Errorf("error parsing GCE/GKE region from zone %q: %v", zone, err)
			}
		}
		managedZones := []string{} // Manage all zones in the region
		if !framework.TestContext.CloudConfig.MultiZone {
			managedZones = []string{zone}
		}

		gceAlphaFeatureGate, err := gcecloud.NewAlphaFeatureGate([]string{gcecloud.AlphaFeatureNetworkEndpointGroup})
		if err != nil {
			glog.Errorf("Encountered error for creating alpha feature gate: %v", err)
		}

		gceCloud, err := gcecloud.CreateGCECloud(&gcecloud.CloudConfig{
			ApiEndpoint:        framework.TestContext.CloudConfig.ApiEndpoint,
			ProjectID:          framework.TestContext.CloudConfig.ProjectID,
			Region:             region,
			Zone:               zone,
			ManagedZones:       managedZones,
			NetworkName:        "", // TODO: Change this to use framework.TestContext.CloudConfig.Network?
			SubnetworkName:     "",
			NodeTags:           nil,
			NodeInstancePrefix: "",
			TokenSource:        nil,
			UseMetadataServer:  false,
			AlphaFeatureGate:   gceAlphaFeatureGate})

		if err != nil {
			return fmt.Errorf("Error building GCE/GKE provider: %v", err)
		}

		cloudConfig.Provider = gceCloud

		if cloudConfig.Zone == "" && framework.TestContext.CloudConfig.MultiZone {
			zones, err := gceCloud.GetAllZones()
			if err != nil {
				return err
			}

			cloudConfig.Zone, _ = zones.PopAny()
		}

	case "aws":
		if cloudConfig.Zone == "" {
			return fmt.Errorf("gce-zone must be specified for AWS")
		}
	case "azure":
		if cloudConfig.ConfigFile == "" {
			return fmt.Errorf("config-file must be specified for Azure")
		}
		config, err := os.Open(cloudConfig.ConfigFile)
		if err != nil {
			framework.Logf("Couldn't open cloud provider configuration %s: %#v",
				cloudConfig.ConfigFile, err)
		}
		defer config.Close()
		cloudConfig.Provider, err = azure.NewCloud(config)
	}

	return nil
}

// There are certain operations we only want to run once per overall test invocation
// (such as deleting old namespaces, or verifying that all system pods are running.
// Because of the way Ginkgo runs tests in parallel, we must use SynchronizedBeforeSuite
// to ensure that these operations only run on the first parallel Ginkgo node.
//
// This function takes two parameters: one function which runs on only the first Ginkgo node,
// returning an opaque byte array, and then a second function which runs on all Ginkgo nodes,
// accepting the byte array.
var _ = ginkgo.SynchronizedBeforeSuite(func() []byte {
	// Run only on Ginkgo node 1

	if err := setupProviderConfig(); err != nil {
		framework.Failf("Failed to setup provider config: %v", err)
	}

	switch framework.TestContext.Provider {
	case "gce", "gke":
		framework.LogClusterImageSources()
	}

	c, err := framework.LoadClientset()
	if err != nil {
		glog.Fatal("Error loading client: ", err)
	}

	// Delete any namespaces except those created by the system. This ensures no
	// lingering resources are left over from a previous test run.
	if framework.TestContext.CleanStart {
		deleted, err := framework.DeleteNamespaces(c, nil, /* deleteFilter */
			[]string{
				metav1.NamespaceSystem,
				metav1.NamespaceDefault,
				metav1.NamespacePublic,
			})
		if err != nil {
			framework.Failf("Error deleting orphaned namespaces: %v", err)
		}
		glog.Infof("Waiting for deletion of the following namespaces: %v", deleted)
		if err := framework.WaitForNamespacesDeleted(c, deleted, framework.NamespaceCleanupTimeout); err != nil {
			framework.Failf("Failed to delete orphaned namespaces %v: %v", deleted, err)
		}
	}

	// In large clusters we may get to this point but still have a bunch
	// of nodes without Routes created. Since this would make a node
	// unschedulable, we need to wait until all of them are schedulable.
	framework.ExpectNoError(framework.WaitForAllNodesSchedulable(c, framework.TestContext.NodeSchedulableTimeout))

	// Ensure all pods are running and ready before starting tests (otherwise,
	// cluster infrastructure pods that are being pulled or started can block
	// test pods from running, and tests that ensure all pods are running and
	// ready will fail).
	podStartupTimeout := framework.TestContext.SystemPodsStartupTimeout
	// TODO: In large clusters, we often observe a non-starting pods due to
	// #41007. To avoid those pods preventing the whole test runs (and just
	// wasting the whole run), we allow for some not-ready pods (with the
	// number equal to the number of allowed not-ready nodes).
	if err := framework.WaitForPodsRunningReady(c, metav1.NamespaceSystem, int32(framework.TestContext.MinStartupPods), int32(framework.TestContext.AllowedNotReadyNodes), podStartupTimeout, framework.ImagePullerLabels); err != nil {
		framework.DumpAllNamespaceInfo(c, metav1.NamespaceSystem)
		framework.LogFailedContainers(c, metav1.NamespaceSystem, framework.Logf)
		runKubernetesServiceTestContainer(c, metav1.NamespaceDefault)
		framework.Failf("Error waiting for all pods to be running and ready: %v", err)
	}

	if err := framework.WaitForPodsSuccess(c, metav1.NamespaceSystem, framework.ImagePullerLabels, framework.ImagePrePullingTimeout); err != nil {
		// There is no guarantee that the image pulling will succeed in 3 minutes
		// and we don't even run the image puller on all platforms (including GKE).
		// We wait for it so we get an indication of failures in the logs, and to
		// maximize benefit of image pre-pulling.
		framework.Logf("WARNING: Image pulling pods failed to enter success in %v: %v", framework.ImagePrePullingTimeout, err)
	}

	// Dump the output of the nethealth containers only once per run
	if framework.TestContext.DumpLogsOnFailure {
		logFunc := framework.Logf
		if framework.TestContext.ReportDir != "" {
			filePath := path.Join(framework.TestContext.ReportDir, "nethealth.txt")
			file, err := os.Create(filePath)
			if err != nil {
				framework.Logf("Failed to create a file with network health data %v: %v\nPrinting to stdout", filePath, err)
			} else {
				defer file.Close()
				if err = file.Chmod(0644); err != nil {
					framework.Logf("Failed to chmod to 644 of %v: %v", filePath, err)
				}
				logFunc = framework.GetLogToFileFunc(file)
				framework.Logf("Dumping network health container logs from all nodes to file %v", filePath)
			}
		} else {
			framework.Logf("Dumping network health container logs from all nodes...")
		}
		framework.LogContainersInPodsWithLabels(c, metav1.NamespaceSystem, framework.ImagePullerLabels, "nethealth", logFunc)
	}

	// Log the version of the server and this client.
	framework.Logf("Client version: %s", version.Get().GitVersion)

	dc := c.DiscoveryClient

	serverVersion, serverErr := dc.ServerVersion()
	if serverErr != nil {
		framework.Logf("Unexpected server error retrieving version: %v", serverErr)
	}
	if serverVersion != nil {
		framework.Logf("Server version: %s", serverVersion.GitVersion)
	}

	// Reference common test to make the import valid.
	commontest.CurrentSuite = commontest.E2E

	return nil

}, func(data []byte) {
	// Run on all Ginkgo nodes

	if cloudConfig.Provider == nil {
		if err := setupProviderConfig(); err != nil {
			framework.Failf("Failed to setup provider config: %v", err)
		}
	}
})

type CleanupActionHandle *int

var cleanupActionsLock sync.Mutex
var cleanupActions = map[CleanupActionHandle]func(){}

// AddCleanupAction installs a function that will be called in the event of the
// whole test being terminated.  This allows arbitrary pieces of the overall
// test to hook into SynchronizedAfterSuite().
func AddCleanupAction(fn func()) CleanupActionHandle {
	p := CleanupActionHandle(new(int))
	cleanupActionsLock.Lock()
	defer cleanupActionsLock.Unlock()
	cleanupActions[p] = fn
	return p
}

// RemoveCleanupAction removes a function that was installed by
// AddCleanupAction.
func RemoveCleanupAction(p CleanupActionHandle) {
	cleanupActionsLock.Lock()
	defer cleanupActionsLock.Unlock()
	delete(cleanupActions, p)
}

// RunCleanupActions runs all functions installed by AddCleanupAction.  It does
// not remove them (see RemoveCleanupAction) but it does run unlocked, so they
// may remove themselves.
func RunCleanupActions() {
	list := []func(){}
	func() {
		cleanupActionsLock.Lock()
		defer cleanupActionsLock.Unlock()
		for _, fn := range cleanupActions {
			list = append(list, fn)
		}
	}()
	// Run unlocked.
	for _, fn := range list {
		fn()
	}
}

// Similar to SynchornizedBeforeSuite, we want to run some operations only once (such as collecting cluster logs).
// Here, the order of functions is reversed; first, the function which runs everywhere,
// and then the function that only runs on the first Ginkgo node.
var _ = ginkgo.SynchronizedAfterSuite(func() {
	// Run on all Ginkgo nodes
	framework.Logf("Running AfterSuite actions on all node")
	RunCleanupActions()
}, func() {
	// Run only Ginkgo on node 1
	framework.Logf("Running AfterSuite actions on node 1")
	if framework.TestContext.ReportDir != "" {
		framework.CoreDump(framework.TestContext.ReportDir)
	}
	if framework.TestContext.GatherSuiteMetricsAfterTest {
		if err := gatherTestSuiteMetrics(); err != nil {
			framework.Logf("Error gathering metrics: %v", err)
		}
	}
})

func gatherTestSuiteMetrics() error {
	framework.Logf("Gathering metrics")
	c, err := framework.LoadClientset()
	if err != nil {
		return fmt.Errorf("error loading client: %v", err)
	}

	// Grab metrics for apiserver, scheduler, controller-manager, kubelet (for non-kubemark case) and cluster autoscaler (optionally).
	grabber, err := metrics.NewMetricsGrabber(c, nil, !framework.ProviderIs("kubemark"), true, true, true, framework.TestContext.IncludeClusterAutoscalerMetrics)
	if err != nil {
		return fmt.Errorf("failed to create MetricsGrabber: %v", err)
	}

	received, err := grabber.Grab()
	if err != nil {
		return fmt.Errorf("failed to grab metrics: %v", err)
	}

	metricsForE2E := (*framework.MetricsForE2E)(&received)
	metricsJson := metricsForE2E.PrintJSON()
	if framework.TestContext.ReportDir != "" {
		filePath := path.Join(framework.TestContext.ReportDir, "MetricsForE2ESuite_"+time.Now().Format(time.RFC3339)+".json")
		if err := ioutil.WriteFile(filePath, []byte(metricsJson), 0644); err != nil {
			return fmt.Errorf("error writing to %q: %v", filePath, err)
		}
	} else {
		framework.Logf("\n\nTest Suite Metrics:\n%s\n\n", metricsJson)
	}

	return nil
}

// TestE2E checks configuration parameters (specified through flags) and then runs
// E2E tests using the Ginkgo runner.
// If a "report directory" is specified, one or more JUnit test reports will be
// generated in this directory, and cluster logs will also be saved.
// This function is called on each Ginkgo node in parallel mode.
func RunE2ETests(t *testing.T) {
	runtimeutils.ReallyCrash = true
	logs.InitLogs()
	defer logs.FlushLogs()

	gomega.RegisterFailHandler(ginkgowrapper.Fail)
	// Disable skipped tests unless they are explicitly requested.
	if config.GinkgoConfig.FocusString == "" && config.GinkgoConfig.SkipString == "" {
		config.GinkgoConfig.SkipString = `\[Flaky\]|\[Feature:.+\]`
	}

	// Run tests through the Ginkgo runner with output to console + JUnit for Jenkins
	var r []ginkgo.Reporter
	if framework.TestContext.ReportDir != "" {
		// TODO: we should probably only be trying to create this directory once
		// rather than once-per-Ginkgo-node.
		if err := os.MkdirAll(framework.TestContext.ReportDir, 0755); err != nil {
			glog.Errorf("Failed creating report directory: %v", err)
		} else {
			r = append(r, reporters.NewJUnitReporter(path.Join(framework.TestContext.ReportDir, fmt.Sprintf("junit_%v%02d.xml", framework.TestContext.ReportPrefix, config.GinkgoConfig.ParallelNode))))
		}
	}
	glog.Infof("Starting e2e run %q on Ginkgo node %d", framework.RunId, config.GinkgoConfig.ParallelNode)

	ginkgo.RunSpecsWithDefaultAndCustomReporters(t, "Kubernetes e2e suite", r)
}

// Run a test container to try and contact the Kubernetes api-server from a pod, wait for it
// to flip to Ready, log its output and delete it.
func runKubernetesServiceTestContainer(c clientset.Interface, ns string) {
	path := "test/images/clusterapi-tester/pod.yaml"
	framework.Logf("Parsing pod from %v", path)
	p, err := manifest.PodFromManifest(path)
	if err != nil {
		framework.Logf("Failed to parse clusterapi-tester from manifest %v: %v", path, err)
		return
	}
	p.Namespace = ns
	if _, err := c.CoreV1().Pods(ns).Create(p); err != nil {
		framework.Logf("Failed to create %v: %v", p.Name, err)
		return
	}
	defer func() {
		if err := c.CoreV1().Pods(ns).Delete(p.Name, nil); err != nil {
			framework.Logf("Failed to delete pod %v: %v", p.Name, err)
		}
	}()
	timeout := 5 * time.Minute
	if err := framework.WaitForPodCondition(c, ns, p.Name, "clusterapi-tester", timeout, testutils.PodRunningReady); err != nil {
		framework.Logf("Pod %v took longer than %v to enter running/ready: %v", p.Name, timeout, err)
		return
	}
	logs, err := framework.GetPodLogs(c, ns, p.Name, p.Spec.Containers[0].Name)
	if err != nil {
		framework.Logf("Failed to retrieve logs from %v: %v", p.Name, err)
	} else {
		framework.Logf("Output of clusterapi-tester:\n%v", logs)
	}
}
