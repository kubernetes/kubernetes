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

package framework

import (
	"fmt"
	"io/ioutil"
	"path"
	"time"

	"k8s.io/klog"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/kubernetes/pkg/version"
	e2elog "k8s.io/kubernetes/test/e2e/framework/log"
	e2emetrics "k8s.io/kubernetes/test/e2e/framework/metrics"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
)

var (
	cloudConfig = &TestContext.CloudConfig
)

// SetupSuite is the boilerplate that can be used to setup ginkgo test suites, on the SynchronizedBeforeSuite step.
// There are certain operations we only want to run once per overall test invocation
// (such as deleting old namespaces, or verifying that all system pods are running.
// Because of the way Ginkgo runs tests in parallel, we must use SynchronizedBeforeSuite
// to ensure that these operations only run on the first parallel Ginkgo node.
//
// This function takes two parameters: one function which runs on only the first Ginkgo node,
// returning an opaque byte array, and then a second function which runs on all Ginkgo nodes,
// accepting the byte array.
func SetupSuite() {
	// Run only on Ginkgo node 1

	switch TestContext.Provider {
	case "gce", "gke":
		LogClusterImageSources()
	}

	c, err := LoadClientset()
	if err != nil {
		klog.Fatal("Error loading client: ", err)
	}

	// Delete any namespaces except those created by the system. This ensures no
	// lingering resources are left over from a previous test run.
	if TestContext.CleanStart {
		deleted, err := DeleteNamespaces(c, nil, /* deleteFilter */
			[]string{
				metav1.NamespaceSystem,
				metav1.NamespaceDefault,
				metav1.NamespacePublic,
			})
		if err != nil {
			e2elog.Failf("Error deleting orphaned namespaces: %v", err)
		}
		klog.Infof("Waiting for deletion of the following namespaces: %v", deleted)
		if err := WaitForNamespacesDeleted(c, deleted, NamespaceCleanupTimeout); err != nil {
			e2elog.Failf("Failed to delete orphaned namespaces %v: %v", deleted, err)
		}
	}

	// In large clusters we may get to this point but still have a bunch
	// of nodes without Routes created. Since this would make a node
	// unschedulable, we need to wait until all of them are schedulable.
	ExpectNoError(WaitForAllNodesSchedulable(c, TestContext.NodeSchedulableTimeout))

	// If NumNodes is not specified then auto-detect how many are scheduleable and not tainted
	if TestContext.CloudConfig.NumNodes == DefaultNumNodes {
		TestContext.CloudConfig.NumNodes = len(GetReadySchedulableNodesOrDie(c).Items)
	}

	// Ensure all pods are running and ready before starting tests (otherwise,
	// cluster infrastructure pods that are being pulled or started can block
	// test pods from running, and tests that ensure all pods are running and
	// ready will fail).
	podStartupTimeout := TestContext.SystemPodsStartupTimeout
	// TODO: In large clusters, we often observe a non-starting pods due to
	// #41007. To avoid those pods preventing the whole test runs (and just
	// wasting the whole run), we allow for some not-ready pods (with the
	// number equal to the number of allowed not-ready nodes).
	if err := e2epod.WaitForPodsRunningReady(c, metav1.NamespaceSystem, int32(TestContext.MinStartupPods), int32(TestContext.AllowedNotReadyNodes), podStartupTimeout, map[string]string{}); err != nil {
		DumpAllNamespaceInfo(c, metav1.NamespaceSystem)
		LogFailedContainers(c, metav1.NamespaceSystem, e2elog.Logf)
		runKubernetesServiceTestContainer(c, metav1.NamespaceDefault)
		e2elog.Failf("Error waiting for all pods to be running and ready: %v", err)
	}

	if err := WaitForDaemonSets(c, metav1.NamespaceSystem, int32(TestContext.AllowedNotReadyNodes), TestContext.SystemDaemonsetStartupTimeout); err != nil {
		e2elog.Logf("WARNING: Waiting for all daemonsets to be ready failed: %v", err)
	}

	// Log the version of the server and this client.
	e2elog.Logf("e2e test version: %s", version.Get().GitVersion)

	dc := c.DiscoveryClient

	serverVersion, serverErr := dc.ServerVersion()
	if serverErr != nil {
		e2elog.Logf("Unexpected server error retrieving version: %v", serverErr)
	}
	if serverVersion != nil {
		e2elog.Logf("kube-apiserver version: %s", serverVersion.GitVersion)
	}

	// Obtain the default IP family of the cluster
	// Some e2e test are designed to work on IPv4 only, this global variable
	// allows to adapt those tests to work on both IPv4 and IPv6
	// TODO(dual-stack): dual stack clusters should pass full e2e testing at least with the primary IP family
	// the dual stack clusters can be ipv4-ipv6 or ipv6-ipv4, order matters,
	// and services use the primary IP family by default
	// If we´ll need to provide additional context for dual-stack, we can detect it
	// because pods have two addresses (one per family)
	TestContext.IPFamily = getDefaultClusterIPFamily(c)
	e2elog.Logf("Cluster IP family: %s", TestContext.IPFamily)

	if TestContext.NodeKiller.Enabled {
		nodeKiller := NewNodeKiller(TestContext.NodeKiller, c, TestContext.Provider)
		go nodeKiller.Run(TestContext.NodeKiller.NodeKillerStopCh)
	}
}

// CleanupSuite is the boilerplate that can be used after tests on ginkgo were run, on the SynchronizedAfterSuite step.
// Similar to SynchronizedBeforeSuite, we want to run some operations only once (such as collecting cluster logs).
// Here, the order of functions is reversed; first, the function which runs everywhere,
// and then the function that only runs on the first Ginkgo node.
func CleanupSuite() {
	// Run on all Ginkgo nodes
	e2elog.Logf("Running AfterSuite actions on all nodes")
	RunCleanupActions()
}

// AfterSuiteActions are actions that are run on ginkgo's SynchronizedAfterSuite
func AfterSuiteActions() {
	// Run only Ginkgo on node 1
	e2elog.Logf("Running AfterSuite actions on node 1")
	if TestContext.ReportDir != "" {
		CoreDump(TestContext.ReportDir)
	}
	if TestContext.GatherSuiteMetricsAfterTest {
		if err := gatherTestSuiteMetrics(); err != nil {
			e2elog.Logf("Error gathering metrics: %v", err)
		}
	}
	if TestContext.NodeKiller.Enabled {
		close(TestContext.NodeKiller.NodeKillerStopCh)
	}
}

func gatherTestSuiteMetrics() error {
	e2elog.Logf("Gathering metrics")
	c, err := LoadClientset()
	if err != nil {
		return fmt.Errorf("error loading client: %v", err)
	}

	// Grab metrics for apiserver, scheduler, controller-manager, kubelet (for non-kubemark case) and cluster autoscaler (optionally).
	grabber, err := e2emetrics.NewMetricsGrabber(c, nil, !ProviderIs("kubemark"), true, true, true, TestContext.IncludeClusterAutoscalerMetrics)
	if err != nil {
		return fmt.Errorf("failed to create MetricsGrabber: %v", err)
	}

	received, err := grabber.Grab()
	if err != nil {
		return fmt.Errorf("failed to grab metrics: %v", err)
	}

	metricsForE2E := (*e2emetrics.ComponentCollection)(&received)
	metricsJSON := metricsForE2E.PrintJSON()
	if TestContext.ReportDir != "" {
		filePath := path.Join(TestContext.ReportDir, "MetricsForE2ESuite_"+time.Now().Format(time.RFC3339)+".json")
		if err := ioutil.WriteFile(filePath, []byte(metricsJSON), 0644); err != nil {
			return fmt.Errorf("error writing to %q: %v", filePath, err)
		}
	} else {
		e2elog.Logf("\n\nTest Suite Metrics:\n%s\n", metricsJSON)
	}

	return nil
}
