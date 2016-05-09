/*
Copyright 2016 The Kubernetes Authors All rights reserved.

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
	"flag"
	"os"

	"github.com/onsi/ginkgo/config"
	"k8s.io/kubernetes/pkg/client/unversioned/clientcmd"
	"k8s.io/kubernetes/pkg/cloudprovider"
)

type TestContextType struct {
	KubeConfig            string
	KubeContext           string
	KubeAPIContentType    string
	KubeVolumeDir         string
	CertDir               string
	Host                  string
	RepoRoot              string
	Provider              string
	CloudConfig           CloudConfig
	KubectlPath           string
	OutputDir             string
	ReportDir             string
	ReportPrefix          string
	Prefix                string
	MinStartupPods        int
	UpgradeTarget         string
	PrometheusPushGateway string
	OSDistro              string
	VerifyServiceAccount  bool
	DeleteNamespace       bool
	CleanStart            bool
	// If set to true framework will start a goroutine monitoring resource usage of system add-ons.
	// It will read the data every 30 seconds from all Nodes and print summary during afterEach.
	GatherKubeSystemResourceUsageData bool
	GatherLogsSizes                   bool
	GatherMetricsAfterTest            bool
	// Currently supported values are 'hr' for human-readable and 'json'. It's a comma separated list.
	OutputPrintType string
	// CreateTestingNS is responsible for creating namespace used for executing e2e tests.
	// It accepts namespace base name, which will be prepended with e2e prefix, kube client
	// and labels to be applied to a namespace.
	CreateTestingNS CreateTestingNSFn
}

type CloudConfig struct {
	ProjectID         string
	Zone              string
	Cluster           string
	MasterName        string
	NodeInstanceGroup string
	NumNodes          int
	ClusterTag        string
	ServiceAccount    string

	Provider cloudprovider.Interface
}

var TestContext TestContextType

func SetTestContext(t TestContextType) {
	TestContext = t
}

func RegisterFlags() {
	// Turn on verbose by default to get spec names
	config.DefaultReporterConfig.Verbose = true

	// Turn on EmitSpecProgress to get spec progress (especially on interrupt)
	config.GinkgoConfig.EmitSpecProgress = true

	// Randomize specs as well as suites
	config.GinkgoConfig.RandomizeAllSpecs = true

	flag.StringVar(&TestContext.KubeConfig, clientcmd.RecommendedConfigPathFlag, os.Getenv(clientcmd.RecommendedConfigPathEnvVar), "Path to kubeconfig containing embedded authinfo.")
	flag.StringVar(&TestContext.KubeContext, clientcmd.FlagContext, "", "kubeconfig context to use/override. If unset, will use value from 'current-context'")
	flag.StringVar(&TestContext.KubeAPIContentType, "kube-api-content-type", "", "ContentType used to communicate with apiserver")
	flag.StringVar(&TestContext.KubeVolumeDir, "volume-dir", "/var/lib/kubelet", "Path to the directory containing the kubelet volumes.")
	flag.StringVar(&TestContext.CertDir, "cert-dir", "", "Path to the directory containing the certs. Default is empty, which doesn't use certs.")
	flag.StringVar(&TestContext.Host, "host", "", "The host, or apiserver, to connect to")
	flag.StringVar(&TestContext.RepoRoot, "repo-root", "../../", "Root directory of kubernetes repository, for finding test files.")
	flag.StringVar(&TestContext.Provider, "provider", "", "The name of the Kubernetes provider (gce, gke, local, vagrant, etc.)")
	flag.StringVar(&TestContext.KubectlPath, "kubectl-path", "kubectl", "The kubectl binary to use. For development, you might use 'cluster/kubectl.sh' here.")
	flag.StringVar(&TestContext.OutputDir, "e2e-output-dir", "/tmp", "Output directory for interesting/useful test data, like performance data, benchmarks, and other metrics.")
	flag.StringVar(&TestContext.ReportDir, "report-dir", "", "Path to the directory where the JUnit XML reports should be saved. Default is empty, which doesn't generate these reports.")
	flag.StringVar(&TestContext.ReportPrefix, "report-prefix", "", "Optional prefix for JUnit XML reports. Default is empty, which doesn't prepend anything to the default name.")
	flag.StringVar(&TestContext.Prefix, "prefix", "e2e", "A prefix to be added to cloud resources created during testing.")
	flag.StringVar(&TestContext.OSDistro, "os-distro", "debian", "The OS distribution of cluster VM instances (debian, trusty, or coreos).")

	// TODO: Flags per provider?  Rename gce-project/gce-zone?
	cloudConfig := &TestContext.CloudConfig
	flag.StringVar(&cloudConfig.MasterName, "kube-master", "", "Name of the kubernetes master. Only required if provider is gce or gke")
	flag.StringVar(&cloudConfig.ProjectID, "gce-project", "", "The GCE project being used, if applicable")
	flag.StringVar(&cloudConfig.Zone, "gce-zone", "", "GCE zone being used, if applicable")
	flag.StringVar(&cloudConfig.ServiceAccount, "gce-service-account", "", "GCE service account to use for GCE API calls, if applicable")
	flag.StringVar(&cloudConfig.Cluster, "gke-cluster", "", "GKE name of cluster being used, if applicable")
	flag.StringVar(&cloudConfig.NodeInstanceGroup, "node-instance-group", "", "Name of the managed instance group for nodes. Valid only for gce, gke or aws")
	flag.IntVar(&cloudConfig.NumNodes, "num-nodes", -1, "Number of nodes in the cluster")

	flag.StringVar(&cloudConfig.ClusterTag, "cluster-tag", "", "Tag used to identify resources.  Only required if provider is aws.")
	flag.IntVar(&TestContext.MinStartupPods, "minStartupPods", 0, "The number of pods which we need to see in 'Running' state with a 'Ready' condition of true, before we try running tests. This is useful in any cluster which needs some base pod-based services running before it can be used.")
	flag.StringVar(&TestContext.UpgradeTarget, "upgrade-target", "ci/latest", "Version to upgrade to (e.g. 'release/stable', 'release/latest', 'ci/latest', '0.19.1', '0.19.1-669-gabac8c8') if doing an upgrade test.")
	flag.StringVar(&TestContext.PrometheusPushGateway, "prom-push-gateway", "", "The URL to prometheus gateway, so that metrics can be pushed during e2es and scraped by prometheus. Typically something like 127.0.0.1:9091.")
	flag.BoolVar(&TestContext.VerifyServiceAccount, "e2e-verify-service-account", true, "If true tests will verify the service account before running.")
	flag.BoolVar(&TestContext.DeleteNamespace, "delete-namespace", true, "If true tests will delete namespace after completion. It is only designed to make debugging easier, DO NOT turn it off by default.")
	flag.BoolVar(&TestContext.CleanStart, "clean-start", false, "If true, purge all namespaces except default and system before running tests. This serves to Cleanup test namespaces from failed/interrupted e2e runs in a long-lived cluster.")
	flag.BoolVar(&TestContext.GatherKubeSystemResourceUsageData, "gather-resource-usage", false, "If set to true framework will be monitoring resource usage of system add-ons in (some) e2e tests.")
	flag.BoolVar(&TestContext.GatherLogsSizes, "gather-logs-sizes", false, "If set to true framework will be monitoring logs sizes on all machines running e2e tests.")
	flag.BoolVar(&TestContext.GatherMetricsAfterTest, "gather-metrics-at-teardown", false, "If set to true framwork will gather metrics from all components after each test.")
	flag.StringVar(&TestContext.OutputPrintType, "output-print-type", "hr", "Comma separated list: 'hr' for human readable summaries 'json' for JSON ones.")
}
