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
	"context"
	"encoding/json"
	"fmt"
	"io/ioutil"
	"os"
	"os/exec"
	"path"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"k8s.io/klog/v2"

	"github.com/onsi/ginkgo"
	"github.com/onsi/ginkgo/config"
	"github.com/onsi/ginkgo/reporters"
	"github.com/onsi/gomega"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	runtimeutils "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/component-base/logs"
	"k8s.io/component-base/version"
	commontest "k8s.io/kubernetes/test/e2e/common"
	"k8s.io/kubernetes/test/e2e/framework"
	e2ekubectl "k8s.io/kubernetes/test/e2e/framework/kubectl"
	e2emanifest "k8s.io/kubernetes/test/e2e/framework/manifest"
	e2enode "k8s.io/kubernetes/test/e2e/framework/node"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	e2ereporters "k8s.io/kubernetes/test/e2e/reporters"
	testutils "k8s.io/kubernetes/test/utils"
	utilnet "k8s.io/utils/net"

	clientset "k8s.io/client-go/kubernetes"
	// ensure auth plugins are loaded
	_ "k8s.io/client-go/plugin/pkg/client/auth"

	// ensure that cloud providers are loaded
	_ "k8s.io/kubernetes/test/e2e/framework/providers/aws"
	_ "k8s.io/kubernetes/test/e2e/framework/providers/azure"
	_ "k8s.io/kubernetes/test/e2e/framework/providers/gce"
	_ "k8s.io/kubernetes/test/e2e/framework/providers/kubemark"
	_ "k8s.io/kubernetes/test/e2e/framework/providers/openstack"
	_ "k8s.io/kubernetes/test/e2e/framework/providers/vsphere"
)

const (
	// namespaceCleanupTimeout is how long to wait for the namespace to be deleted.
	// If there are any orphaned namespaces to clean up, this test is running
	// on a long lived cluster. A long wait here is preferably to spurious test
	// failures caused by leaked resources from a previous test run.
	namespaceCleanupTimeout = 15 * time.Minute
)

var _ = ginkgo.SynchronizedBeforeSuite(func() []byte {
	// Reference common test to make the import valid.
	commontest.CurrentSuite = commontest.E2E
	setupSuite()
	return nil
}, func(data []byte) {
	// Run on all Ginkgo nodes
	setupSuitePerGinkgoNode()
})

var _ = ginkgo.SynchronizedAfterSuite(func() {
	CleanupSuite()
}, func() {
	AfterSuiteActions()
})

// RunE2ETests checks configuration parameters (specified through flags) and then runs
// E2E tests using the Ginkgo runner.
// If a "report directory" is specified, one or more JUnit test reports will be
// generated in this directory, and cluster logs will also be saved.
// This function is called on each Ginkgo node in parallel mode.
func RunE2ETests(t *testing.T) {
	runtimeutils.ReallyCrash = true
	logs.InitLogs()
	defer logs.FlushLogs()

	gomega.RegisterFailHandler(framework.Fail)
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
			klog.Errorf("Failed creating report directory: %v", err)
		} else {
			r = append(r, reporters.NewJUnitReporter(path.Join(framework.TestContext.ReportDir, fmt.Sprintf("junit_%v%02d.xml", framework.TestContext.ReportPrefix, config.GinkgoConfig.ParallelNode))))
		}
	}

	// Stream the progress to stdout and optionally a URL accepting progress updates.
	r = append(r, e2ereporters.NewProgressReporter(framework.TestContext.ProgressReportURL))

	// The DetailsRepoerter will output details about every test (name, files, lines, etc) which helps
	// when documenting our tests.
	if len(framework.TestContext.SpecSummaryOutput) > 0 {
		r = append(r, e2ereporters.NewDetailsReporterFile(framework.TestContext.SpecSummaryOutput))
	}

	klog.Infof("Starting e2e run %q on Ginkgo node %d", framework.RunID, config.GinkgoConfig.ParallelNode)
	ginkgo.RunSpecsWithDefaultAndCustomReporters(t, "Kubernetes e2e suite", r)
}

// Run a test container to try and contact the Kubernetes api-server from a pod, wait for it
// to flip to Ready, log its output and delete it.
func runKubernetesServiceTestContainer(c clientset.Interface, ns string) {
	path := "test/images/clusterapi-tester/pod.yaml"
	framework.Logf("Parsing pod from %v", path)
	p, err := e2emanifest.PodFromManifest(path)
	if err != nil {
		framework.Logf("Failed to parse clusterapi-tester from manifest %v: %v", path, err)
		return
	}
	p.Namespace = ns
	if _, err := c.CoreV1().Pods(ns).Create(context.TODO(), p, metav1.CreateOptions{}); err != nil {
		framework.Logf("Failed to create %v: %v", p.Name, err)
		return
	}
	defer func() {
		if err := c.CoreV1().Pods(ns).Delete(context.TODO(), p.Name, metav1.DeleteOptions{}); err != nil {
			framework.Logf("Failed to delete pod %v: %v", p.Name, err)
		}
	}()
	timeout := 5 * time.Minute
	if err := e2epod.WaitForPodCondition(c, ns, p.Name, "clusterapi-tester", timeout, testutils.PodRunningReady); err != nil {
		framework.Logf("Pod %v took longer than %v to enter running/ready: %v", p.Name, timeout, err)
		return
	}
	logs, err := e2epod.GetPodLogs(c, ns, p.Name, p.Spec.Containers[0].Name)
	if err != nil {
		framework.Logf("Failed to retrieve logs from %v: %v", p.Name, err)
	} else {
		framework.Logf("Output of clusterapi-tester:\n%v", logs)
	}
}

// getDefaultClusterIPFamily obtains the default IP family of the cluster
// using the Cluster IP address of the kubernetes service created in the default namespace
// This unequivocally identifies the default IP family because services are single family
// TODO: dual-stack may support multiple families per service
// but we can detect if a cluster is dual stack because pods have two addresses (one per family)
func getDefaultClusterIPFamily(c clientset.Interface) string {
	// Get the ClusterIP of the kubernetes service created in the default namespace
	svc, err := c.CoreV1().Services(metav1.NamespaceDefault).Get(context.TODO(), "kubernetes", metav1.GetOptions{})
	if err != nil {
		framework.Failf("Failed to get kubernetes service ClusterIP: %v", err)
	}

	if utilnet.IsIPv6String(svc.Spec.ClusterIP) {
		return "ipv6"
	}
	return "ipv4"
}

// waitForDaemonSets for all daemonsets in the given namespace to be ready
// (defined as all but 'allowedNotReadyNodes' pods associated with that
// daemonset are ready).
func waitForDaemonSets(c clientset.Interface, ns string, allowedNotReadyNodes int32, timeout time.Duration) error {
	start := time.Now()
	framework.Logf("Waiting up to %v for all daemonsets in namespace '%s' to start",
		timeout, ns)

	return wait.PollImmediate(framework.Poll, timeout, func() (bool, error) {
		dsList, err := c.AppsV1().DaemonSets(ns).List(context.TODO(), metav1.ListOptions{})
		if err != nil {
			framework.Logf("Error getting daemonsets in namespace: '%s': %v", ns, err)
			return false, err
		}
		var notReadyDaemonSets []string
		for _, ds := range dsList.Items {
			framework.Logf("%d / %d pods ready in namespace '%s' in daemonset '%s' (%d seconds elapsed)", ds.Status.NumberReady, ds.Status.DesiredNumberScheduled, ns, ds.ObjectMeta.Name, int(time.Since(start).Seconds()))
			if ds.Status.DesiredNumberScheduled-ds.Status.NumberReady > allowedNotReadyNodes {
				notReadyDaemonSets = append(notReadyDaemonSets, ds.ObjectMeta.Name)
			}
		}

		if len(notReadyDaemonSets) > 0 {
			framework.Logf("there are not ready daemonsets: %v", notReadyDaemonSets)
			return false, nil
		}

		return true, nil
	})
}

// setupSuite is the boilerplate that can be used to setup ginkgo test suites, on the SynchronizedBeforeSuite step.
// There are certain operations we only want to run once per overall test invocation
// (such as deleting old namespaces, or verifying that all system pods are running.
// Because of the way Ginkgo runs tests in parallel, we must use SynchronizedBeforeSuite
// to ensure that these operations only run on the first parallel Ginkgo node.
//
// This function takes two parameters: one function which runs on only the first Ginkgo node,
// returning an opaque byte array, and then a second function which runs on all Ginkgo nodes,
// accepting the byte array.
func setupSuite() {
	// Run only on Ginkgo node 1

	switch framework.TestContext.Provider {
	case "gce", "gke":
		logClusterImageSources()
	}

	c, err := framework.LoadClientset()
	if err != nil {
		klog.Fatal("Error loading client: ", err)
	}

	// Delete any namespaces except those created by the system. This ensures no
	// lingering resources are left over from a previous test run.
	if framework.TestContext.CleanStart {
		deleted, err := framework.DeleteNamespaces(c, nil, /* deleteFilter */
			[]string{
				metav1.NamespaceSystem,
				metav1.NamespaceDefault,
				metav1.NamespacePublic,
				v1.NamespaceNodeLease,
			})
		if err != nil {
			framework.Failf("Error deleting orphaned namespaces: %v", err)
		}
		klog.Infof("Waiting for deletion of the following namespaces: %v", deleted)
		if err := framework.WaitForNamespacesDeleted(c, deleted, namespaceCleanupTimeout); err != nil {
			framework.Failf("Failed to delete orphaned namespaces %v: %v", deleted, err)
		}
	}

	// In large clusters we may get to this point but still have a bunch
	// of nodes without Routes created. Since this would make a node
	// unschedulable, we need to wait until all of them are schedulable.
	framework.ExpectNoError(framework.WaitForAllNodesSchedulable(c, framework.TestContext.NodeSchedulableTimeout))

	// If NumNodes is not specified then auto-detect how many are scheduleable and not tainted
	if framework.TestContext.CloudConfig.NumNodes == framework.DefaultNumNodes {
		nodes, err := e2enode.GetReadySchedulableNodes(c)
		framework.ExpectNoError(err)
		framework.TestContext.CloudConfig.NumNodes = len(nodes.Items)
	}

	// Ensure all pods are running and ready before starting tests (otherwise,
	// cluster infrastructure pods that are being pulled or started can block
	// test pods from running, and tests that ensure all pods are running and
	// ready will fail).
	podStartupTimeout := framework.TestContext.SystemPodsStartupTimeout
	// TODO: In large clusters, we often observe a non-starting pods due to
	// #41007. To avoid those pods preventing the whole test runs (and just
	// wasting the whole run), we allow for some not-ready pods (with the
	// number equal to the number of allowed not-ready nodes).
	if err := e2epod.WaitForPodsRunningReady(c, metav1.NamespaceSystem, int32(framework.TestContext.MinStartupPods), int32(framework.TestContext.AllowedNotReadyNodes), podStartupTimeout, map[string]string{}); err != nil {
		framework.DumpAllNamespaceInfo(c, metav1.NamespaceSystem)
		e2ekubectl.LogFailedContainers(c, metav1.NamespaceSystem, framework.Logf)
		runKubernetesServiceTestContainer(c, metav1.NamespaceDefault)
		framework.Failf("Error waiting for all pods to be running and ready: %v", err)
	}

	if err := waitForDaemonSets(c, metav1.NamespaceSystem, int32(framework.TestContext.AllowedNotReadyNodes), framework.TestContext.SystemDaemonsetStartupTimeout); err != nil {
		framework.Logf("WARNING: Waiting for all daemonsets to be ready failed: %v", err)
	}

	// Log the version of the server and this client.
	framework.Logf("e2e test version: %s", version.Get().GitVersion)

	dc := c.DiscoveryClient

	serverVersion, serverErr := dc.ServerVersion()
	if serverErr != nil {
		framework.Logf("Unexpected server error retrieving version: %v", serverErr)
	}
	if serverVersion != nil {
		framework.Logf("kube-apiserver version: %s", serverVersion.GitVersion)
	}

	if framework.TestContext.NodeKiller.Enabled {
		nodeKiller := framework.NewNodeKiller(framework.TestContext.NodeKiller, c, framework.TestContext.Provider)
		go nodeKiller.Run(framework.TestContext.NodeKiller.NodeKillerStopCh)
	}
}

// logClusterImageSources writes out cluster image sources.
func logClusterImageSources() {
	masterImg, nodeImg, err := lookupClusterImageSources()
	if err != nil {
		framework.Logf("Cluster image sources lookup failed: %v\n", err)
		return
	}
	framework.Logf("cluster-master-image: %s", masterImg)
	framework.Logf("cluster-node-image: %s", nodeImg)

	images := map[string]string{
		"master_os_image": masterImg,
		"node_os_image":   nodeImg,
	}

	outputBytes, _ := json.MarshalIndent(images, "", "  ")
	filePath := filepath.Join(framework.TestContext.ReportDir, "images.json")
	if err := ioutil.WriteFile(filePath, outputBytes, 0644); err != nil {
		framework.Logf("cluster images sources, could not write to %q: %v", filePath, err)
	}
}

// TODO: These should really just use the GCE API client library or at least use
// better formatted output from the --format flag.

// Returns master & node image string, or error
func lookupClusterImageSources() (string, string, error) {
	// Given args for a gcloud compute command, run it with other args, and return the values,
	// whether separated by newlines, commas or semicolons.
	gcloudf := func(argv ...string) ([]string, error) {
		args := []string{"compute"}
		args = append(args, argv...)
		args = append(args, "--project", framework.TestContext.CloudConfig.ProjectID)
		if framework.TestContext.CloudConfig.MultiMaster {
			args = append(args, "--region", framework.TestContext.CloudConfig.Region)
		} else {
			args = append(args, "--zone", framework.TestContext.CloudConfig.Zone)
		}
		outputBytes, err := exec.Command("gcloud", args...).CombinedOutput()
		str := strings.Replace(string(outputBytes), ",", "\n", -1)
		str = strings.Replace(str, ";", "\n", -1)
		lines := strings.Split(str, "\n")
		if err != nil {
			framework.Logf("lookupDiskImageSources: gcloud error with [%#v]; err:%v", argv, err)
			for _, l := range lines {
				framework.Logf(" > %s", l)
			}
		}
		return lines, err
	}

	// Given a GCE instance, look through its disks, finding one that has a sourceImage
	host2image := func(instance string) (string, error) {
		// gcloud compute instances describe {INSTANCE} --format="get(disks[].source)"
		// gcloud compute disks describe {DISKURL} --format="get(sourceImage)"
		disks, err := gcloudf("instances", "describe", instance, "--format=get(disks[].source)")
		if err != nil {
			return "", err
		} else if len(disks) == 0 {
			return "", fmt.Errorf("instance %q had no findable disks", instance)
		}
		// Loop over disks, looking for the boot disk
		for _, disk := range disks {
			lines, err := gcloudf("disks", "describe", disk, "--format=get(sourceImage)")
			if err != nil {
				return "", err
			} else if len(lines) > 0 && lines[0] != "" {
				return lines[0], nil // break, we're done
			}
		}
		return "", fmt.Errorf("instance %q had no disk with a sourceImage", instance)
	}

	// gcloud compute instance-groups list-instances {GROUPNAME} --format="get(instance)"
	nodeName := ""
	instGroupName := strings.Split(framework.TestContext.CloudConfig.NodeInstanceGroup, ",")[0]
	if lines, err := gcloudf("instance-groups", "list-instances", instGroupName, "--format=get(instance)"); err != nil {
		return "", "", err
	} else if len(lines) == 0 {
		return "", "", fmt.Errorf("no instances inside instance-group %q", instGroupName)
	} else {
		nodeName = lines[0]
	}

	nodeImg, err := host2image(nodeName)
	if err != nil {
		return "", "", err
	}
	frags := strings.Split(nodeImg, "/")
	nodeImg = frags[len(frags)-1]

	// For GKE clusters, MasterName will not be defined; we just leave masterImg blank.
	masterImg := ""
	if masterName := framework.TestContext.CloudConfig.MasterName; masterName != "" {
		img, err := host2image(masterName)
		if err != nil {
			return "", "", err
		}
		frags = strings.Split(img, "/")
		masterImg = frags[len(frags)-1]
	}

	return masterImg, nodeImg, nil
}

// setupSuitePerGinkgoNode is the boilerplate that can be used to setup ginkgo test suites, on the SynchronizedBeforeSuite step.
// There are certain operations we only want to run once per overall test invocation on each Ginkgo node
// such as making some global variables accessible to all parallel executions
// Because of the way Ginkgo runs tests in parallel, we must use SynchronizedBeforeSuite
// Ref: https://onsi.github.io/ginkgo/#parallel-specs
func setupSuitePerGinkgoNode() {
	// Obtain the default IP family of the cluster
	// Some e2e test are designed to work on IPv4 only, this global variable
	// allows to adapt those tests to work on both IPv4 and IPv6
	// TODO: dual-stack
	// the dual stack clusters can be ipv4-ipv6 or ipv6-ipv4, order matters,
	// and services use the primary IP family by default
	c, err := framework.LoadClientset()
	if err != nil {
		klog.Fatal("Error loading client: ", err)
	}
	framework.TestContext.IPFamily = getDefaultClusterIPFamily(c)
	framework.Logf("Cluster IP family: %s", framework.TestContext.IPFamily)
}
