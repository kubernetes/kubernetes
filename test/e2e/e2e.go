/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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
	"flag"
	"fmt"
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
	"golang.org/x/oauth2"
	"golang.org/x/oauth2/google"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/client/unversioned/clientcmd"
	gcecloud "k8s.io/kubernetes/pkg/cloudprovider/providers/gce"
	"k8s.io/kubernetes/pkg/util"
	"k8s.io/kubernetes/pkg/util/runtime"
)

const (
	// podStartupTimeout is the time to allow all pods in the cluster to become
	// running and ready before any e2e tests run. It includes pulling all of
	// the pods (as of 5/18/15 this is 8 pods).
	podStartupTimeout = 10 * time.Minute
)

var (
	cloudConfig = &testContext.CloudConfig
)

func RegisterFlags() {
	// Turn on verbose by default to get spec names
	config.DefaultReporterConfig.Verbose = true

	// Turn on EmitSpecProgress to get spec progress (especially on interrupt)
	config.GinkgoConfig.EmitSpecProgress = true

	// Randomize specs as well as suites
	config.GinkgoConfig.RandomizeAllSpecs = true

	flag.StringVar(&testContext.KubeConfig, clientcmd.RecommendedConfigPathFlag, os.Getenv(clientcmd.RecommendedConfigPathEnvVar), "Path to kubeconfig containing embedded authinfo.")
	flag.StringVar(&testContext.KubeContext, clientcmd.FlagContext, "", "kubeconfig context to use/override. If unset, will use value from 'current-context'")
	flag.StringVar(&testContext.KubeVolumeDir, "volume-dir", "/var/lib/kubelet", "Path to the directory containing the kubelet volumes.")
	flag.StringVar(&testContext.CertDir, "cert-dir", "", "Path to the directory containing the certs. Default is empty, which doesn't use certs.")
	flag.StringVar(&testContext.Host, "host", "", "The host, or apiserver, to connect to")
	flag.StringVar(&testContext.RepoRoot, "repo-root", "../../", "Root directory of kubernetes repository, for finding test files.")
	flag.StringVar(&testContext.Provider, "provider", "", "The name of the Kubernetes provider (gce, gke, local, vagrant, etc.)")
	flag.StringVar(&testContext.KubectlPath, "kubectl-path", "kubectl", "The kubectl binary to use. For development, you might use 'cluster/kubectl.sh' here.")
	flag.StringVar(&testContext.OutputDir, "e2e-output-dir", "/tmp", "Output directory for interesting/useful test data, like performance data, benchmarks, and other metrics.")
	flag.StringVar(&testContext.ReportDir, "report-dir", "", "Path to the directory where the JUnit XML reports should be saved. Default is empty, which doesn't generate these reports.")
	flag.StringVar(&testContext.ReportPrefix, "report-prefix", "", "Optional prefix for JUnit XML reports. Default is empty, which doesn't prepend anything to the default name.")
	flag.StringVar(&testContext.prefix, "prefix", "e2e", "A prefix to be added to cloud resources created during testing.")
	flag.StringVar(&testContext.OSDistro, "os-distro", "debian", "The OS distribution of cluster VM instances (debian, trusty, or coreos).")

	// TODO: Flags per provider?  Rename gce-project/gce-zone?
	flag.StringVar(&cloudConfig.MasterName, "kube-master", "", "Name of the kubernetes master. Only required if provider is gce or gke")
	flag.StringVar(&cloudConfig.ProjectID, "gce-project", "", "The GCE project being used, if applicable")
	flag.StringVar(&cloudConfig.Zone, "gce-zone", "", "GCE zone being used, if applicable")
	flag.StringVar(&cloudConfig.ServiceAccount, "gce-service-account", "", "GCE service account to use for GCE API calls, if applicable")
	flag.StringVar(&cloudConfig.Cluster, "gke-cluster", "", "GKE name of cluster being used, if applicable")
	flag.StringVar(&cloudConfig.NodeInstanceGroup, "node-instance-group", "", "Name of the managed instance group for nodes. Valid only for gce, gke or aws")
	flag.IntVar(&cloudConfig.NumNodes, "num-nodes", -1, "Number of nodes in the cluster")

	flag.StringVar(&cloudConfig.ClusterTag, "cluster-tag", "", "Tag used to identify resources.  Only required if provider is aws.")
	flag.IntVar(&testContext.MinStartupPods, "minStartupPods", 0, "The number of pods which we need to see in 'Running' state with a 'Ready' condition of true, before we try running tests. This is useful in any cluster which needs some base pod-based services running before it can be used.")
	flag.StringVar(&testContext.UpgradeTarget, "upgrade-target", "ci/latest", "Version to upgrade to (e.g. 'release/stable', 'release/latest', 'ci/latest', '0.19.1', '0.19.1-669-gabac8c8') if doing an upgrade test.")
	flag.StringVar(&testContext.UpgradeImage, "upgrade-image", "", "Image to upgrade to (e.g. 'container_vm' or 'gci') if doing an upgrade test.")
	flag.StringVar(&testContext.PrometheusPushGateway, "prom-push-gateway", "", "The URL to prometheus gateway, so that metrics can be pushed during e2es and scraped by prometheus. Typically something like 127.0.0.1:9091.")
	flag.BoolVar(&testContext.VerifyServiceAccount, "e2e-verify-service-account", true, "If true tests will verify the service account before running.")
	flag.BoolVar(&testContext.DeleteNamespace, "delete-namespace", true, "If true tests will delete namespace after completion. It is only designed to make debugging easier, DO NOT turn it off by default.")
	flag.BoolVar(&testContext.CleanStart, "clean-start", false, "If true, purge all namespaces except default and system before running tests. This serves to cleanup test namespaces from failed/interrupted e2e runs in a long-lived cluster.")
	flag.BoolVar(&testContext.GatherKubeSystemResourceUsageData, "gather-resource-usage", false, "If set to true framework will be monitoring resource usage of system add-ons in (some) e2e tests.")
	flag.BoolVar(&testContext.GatherLogsSizes, "gather-logs-sizes", false, "If set to true framework will be monitoring logs sizes on all machines running e2e tests.")
	flag.BoolVar(&testContext.GatherMetricsAfterTest, "gather-metrics-at-teardown", false, "If set to true framwork will gather metrics from all components after each test.")
	flag.StringVar(&testContext.OutputPrintType, "output-print-type", "hr", "Comma separated list: 'hr' for human readable summaries 'json' for JSON ones.")
}

// setupProviderConfig validates and sets up cloudConfig based on testContext.Provider.
func setupProviderConfig() error {
	switch testContext.Provider {
	case "":
		glog.Info("The --provider flag is not set.  Treating as a conformance test.  Some tests may not be run.")

	case "gce", "gke":
		var err error
		Logf("Fetching cloud provider for %q\r\n", testContext.Provider)
		var tokenSource oauth2.TokenSource
		tokenSource = nil
		if cloudConfig.ServiceAccount != "" {
			// Use specified service account for auth
			Logf("Using service account %q as token source.", cloudConfig.ServiceAccount)
			tokenSource = google.ComputeTokenSource(cloudConfig.ServiceAccount)
		}
		zone := testContext.CloudConfig.Zone
		region, err := gcecloud.GetGCERegion(zone)
		if err != nil {
			return fmt.Errorf("error parsing GCE/GKE region from zone %q: %v", zone, err)
		}
		managedZones := []string{zone} // Only single-zone for now
		cloudConfig.Provider, err = gcecloud.CreateGCECloud(testContext.CloudConfig.ProjectID, region, zone, managedZones, "" /* networkUrl */, nil /* nodeTags */, "" /* nodeInstancePerfix */, tokenSource, false /* useMetadataServer */)
		if err != nil {
			return fmt.Errorf("Error building GCE/GKE provider: %v", err)
		}

	case "aws":
		if cloudConfig.Zone == "" {
			return fmt.Errorf("gce-zone must be specified for AWS")
		}

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

	// Delete any namespaces except default and kube-system. This ensures no
	// lingering resources are left over from a previous test run.
	if testContext.CleanStart {
		c, err := loadClient()
		if err != nil {
			glog.Fatal("Error loading client: ", err)
		}

		deleted, err := deleteNamespaces(c, nil /* deleteFilter */, []string{api.NamespaceSystem, api.NamespaceDefault})
		if err != nil {
			Failf("Error deleting orphaned namespaces: %v", err)
		}
		glog.Infof("Waiting for deletion of the following namespaces: %v", deleted)
		if err := waitForNamespacesDeleted(c, deleted, namespaceCleanupTimeout); err != nil {
			Failf("Failed to delete orphaned namespaces %v: %v", deleted, err)
		}
	}

	// Ensure all pods are running and ready before starting tests (otherwise,
	// cluster infrastructure pods that are being pulled or started can block
	// test pods from running, and tests that ensure all pods are running and
	// ready will fail).
	if err := waitForPodsRunningReady(api.NamespaceSystem, testContext.MinStartupPods, podStartupTimeout); err != nil {
		if c, errClient := loadClient(); errClient != nil {
			Logf("Unable to dump cluster information because: %v", errClient)
		} else {
			dumpAllNamespaceInfo(c, api.NamespaceSystem)
		}
		logFailedContainers(api.NamespaceSystem)
		runKubernetesServiceTestContainer(testContext.RepoRoot, api.NamespaceDefault)
		Failf("Error waiting for all pods to be running and ready: %v", err)
	}

	return nil

}, func(data []byte) {
	// Run on all Ginkgo nodes

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
	RunCleanupActions()
}, func() {
	// Run only Ginkgo on node 1
	if testContext.ReportDir != "" {
		CoreDump(testContext.ReportDir)
	}
})

// TestE2E checks configuration parameters (specified through flags) and then runs
// E2E tests using the Ginkgo runner.
// If a "report directory" is specified, one or more JUnit test reports will be
// generated in this directory, and cluster logs will also be saved.
// This function is called on each Ginkgo node in parallel mode.
func RunE2ETests(t *testing.T) {
	runtime.ReallyCrash = true
	util.InitLogs()
	defer util.FlushLogs()

	// We must call setupProviderConfig first since SynchronizedBeforeSuite needs
	// cloudConfig to be set up already.
	if err := setupProviderConfig(); err != nil {
		glog.Fatalf(err.Error())
	}

	gomega.RegisterFailHandler(ginkgo.Fail)
	// Disable skipped tests unless they are explicitly requested.
	if config.GinkgoConfig.FocusString == "" && config.GinkgoConfig.SkipString == "" {
		config.GinkgoConfig.SkipString = `\[Flaky\]|\[Feature:.+\]`
	}

	// Run tests through the Ginkgo runner with output to console + JUnit for Jenkins
	var r []ginkgo.Reporter
	if testContext.ReportDir != "" {
		// TODO: we should probably only be trying to create this directory once
		// rather than once-per-Ginkgo-node.
		if err := os.MkdirAll(testContext.ReportDir, 0755); err != nil {
			glog.Errorf("Failed creating report directory: %v", err)
		} else {
			r = append(r, reporters.NewJUnitReporter(path.Join(testContext.ReportDir, fmt.Sprintf("junit_%v%02d.xml", testContext.ReportPrefix, config.GinkgoConfig.ParallelNode))))
		}
	}
	glog.Infof("Starting e2e run %q on Ginkgo node %d", runId, config.GinkgoConfig.ParallelNode)
	ginkgo.RunSpecsWithDefaultAndCustomReporters(t, "Kubernetes e2e suite", r)
}
