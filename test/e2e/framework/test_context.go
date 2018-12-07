/*
Copyright 2016 The Kubernetes Authors.

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
	"fmt"
	"io/ioutil"
	"os"
	"time"

	"github.com/onsi/ginkgo/config"
	"github.com/pkg/errors"
	utilflag "k8s.io/apiserver/pkg/util/flag"
	restclient "k8s.io/client-go/rest"
	"k8s.io/client-go/tools/clientcmd"
	clientcmdapi "k8s.io/client-go/tools/clientcmd/api"
	"k8s.io/klog"
	kubeletconfig "k8s.io/kubernetes/pkg/kubelet/apis/config"
)

const defaultHost = "http://127.0.0.1:8080"

// TestContextType contains test settings and global state. Due to
// historic reasons, it is a mixture of items managed by the test
// framework itself, cloud providers and individual tests.
// The goal is to move anything not required by the framework
// into the code which uses the settings.
//
// The recommendation for those settings is:
// - They are stored in their own context structure or local
//   variables.
// - The standard `flag` package is used to register them.
//   The flag name should follow the pattern <part1>.<part2>....<partn>
//   where the prefix is unlikely to conflict with other tests or
//   standard packages and each part is in lower camel case. For
//   example, test/e2e/storage/csi/context.go could define
//   storage.csi.numIterations.
// - framework/config can be used to simplify the registration of
//   multiple options with a single function call:
//   var storageCSI {
//       NumIterations `default:"1" usage:"number of iterations"`
//   }
//   _ config.AddOptions(&storageCSI, "storage.csi")
// - The direct use Viper in tests is possible, but discouraged because
//   it only works in test suites which use Viper (which is not
//   required) and the supported options cannot be
//   discovered by a test suite user.
//
// Test suite authors can use framework/viper to make all command line
// parameters also configurable via a configuration file.
type TestContextType struct {
	KubeConfig         string
	KubeContext        string
	KubeAPIContentType string
	KubeVolumeDir      string
	CertDir            string
	Host               string
	// TODO: Deprecating this over time... instead just use gobindata_util.go , see #23987.
	RepoRoot                string
	DockershimCheckpointDir string

	// Provider identifies the infrastructure provider (gce, gke, aws)
	Provider string

	// Tooling is the tooling in use (e.g. kops, gke).  Provider is the cloud provider and might not uniquely identify the tooling.
	Tooling string

	CloudConfig    CloudConfig
	KubectlPath    string
	OutputDir      string
	ReportDir      string
	ReportPrefix   string
	Prefix         string
	MinStartupPods int
	// Timeout for waiting for system pods to be running
	SystemPodsStartupTimeout    time.Duration
	EtcdUpgradeStorage          string
	EtcdUpgradeVersion          string
	IngressUpgradeImage         string
	GCEUpgradeScript            string
	ContainerRuntime            string
	ContainerRuntimeEndpoint    string
	ContainerRuntimeProcessName string
	ContainerRuntimePidFile     string
	// SystemdServices are comma separated list of systemd services the test framework
	// will dump logs for.
	SystemdServices          string
	ImageServiceEndpoint     string
	MasterOSDistro           string
	NodeOSDistro             string
	VerifyServiceAccount     bool
	DeleteNamespace          bool
	DeleteNamespaceOnFailure bool
	AllowedNotReadyNodes     int
	CleanStart               bool
	// If set to 'true' or 'all' framework will start a goroutine monitoring resource usage of system add-ons.
	// It will read the data every 30 seconds from all Nodes and print summary during afterEach. If set to 'master'
	// only master Node will be monitored.
	GatherKubeSystemResourceUsageData string
	GatherLogsSizes                   bool
	GatherMetricsAfterTest            string
	GatherSuiteMetricsAfterTest       bool
	AllowGatheringProfiles            bool
	// If set to 'true' framework will gather ClusterAutoscaler metrics when gathering them for other components.
	IncludeClusterAutoscalerMetrics bool
	// Currently supported values are 'hr' for human-readable and 'json'. It's a comma separated list.
	OutputPrintType string
	// NodeSchedulableTimeout is the timeout for waiting for all nodes to be schedulable.
	NodeSchedulableTimeout time.Duration
	// SystemDaemonsetStartupTimeout is the timeout for waiting for all system daemonsets to be ready.
	SystemDaemonsetStartupTimeout time.Duration
	// CreateTestingNS is responsible for creating namespace used for executing e2e tests.
	// It accepts namespace base name, which will be prepended with e2e prefix, kube client
	// and labels to be applied to a namespace.
	CreateTestingNS CreateTestingNSFn
	// If set to true test will dump data about the namespace in which test was running.
	DumpLogsOnFailure bool
	// Disables dumping cluster log from master and nodes after all tests.
	DisableLogDump bool
	// Path to the GCS artifacts directory to dump logs from nodes. Logexporter gets enabled if this is non-empty.
	LogexporterGCSPath string
	// featureGates is a map of feature names to bools that enable or disable alpha/experimental features.
	FeatureGates map[string]bool
	// Node e2e specific test context
	NodeTestContextType
	// Monitoring solution that is used in current cluster.
	ClusterMonitoringMode string
	// Separate Prometheus monitoring deployed in cluster
	EnablePrometheusMonitoring bool

	// Indicates what path the kubernetes-anywhere is installed on
	KubernetesAnywherePath string

	// The DNS Domain of the cluster.
	ClusterDNSDomain string
}

// NodeTestContextType is part of TestContextType, it is shared by all node e2e test.
type NodeTestContextType struct {
	// NodeE2E indicates whether it is running node e2e.
	NodeE2E bool
	// Name of the node to run tests on.
	NodeName string
	// NodeConformance indicates whether the test is running in node conformance mode.
	NodeConformance bool
	// PrepullImages indicates whether node e2e framework should prepull images.
	PrepullImages bool
	// KubeletConfig is the kubelet configuration the test is running against.
	KubeletConfig kubeletconfig.KubeletConfiguration
	// ImageDescription is the description of the image on which the test is running.
	ImageDescription string
	// SystemSpecName is the name of the system spec (e.g., gke) that's used in
	// the node e2e test. If empty, the default one (system.DefaultSpec) is
	// used. The system specs are in test/e2e_node/system/specs/.
	SystemSpecName string
}

type CloudConfig struct {
	ApiEndpoint       string
	ProjectID         string
	Zone              string // for multizone tests, arbitrarily chosen zone
	Region            string
	MultiZone         bool
	MultiMaster       bool
	Cluster           string
	MasterName        string
	NodeInstanceGroup string // comma-delimited list of groups' names
	NumNodes          int
	ClusterIPRange    string
	ClusterTag        string
	Network           string
	ConfigFile        string // for azure and openstack
	NodeTag           string
	MasterTag         string

	Provider ProviderInterface
}

var TestContext TestContextType

// Register flags common to all e2e test suites.
func RegisterCommonFlags() {
	// Turn on verbose by default to get spec names
	config.DefaultReporterConfig.Verbose = true

	// Turn on EmitSpecProgress to get spec progress (especially on interrupt)
	config.GinkgoConfig.EmitSpecProgress = true

	// Randomize specs as well as suites
	config.GinkgoConfig.RandomizeAllSpecs = true

	flag.StringVar(&TestContext.GatherKubeSystemResourceUsageData, "gather-resource-usage", "false", "If set to 'true' or 'all' framework will be monitoring resource usage of system all add-ons in (some) e2e tests, if set to 'master' framework will be monitoring master node only, if set to 'none' of 'false' monitoring will be turned off.")
	flag.BoolVar(&TestContext.GatherLogsSizes, "gather-logs-sizes", false, "If set to true framework will be monitoring logs sizes on all machines running e2e tests.")
	flag.StringVar(&TestContext.GatherMetricsAfterTest, "gather-metrics-at-teardown", "false", "If set to 'true' framework will gather metrics from all components after each test. If set to 'master' only master component metrics would be gathered.")
	flag.BoolVar(&TestContext.GatherSuiteMetricsAfterTest, "gather-suite-metrics-at-teardown", false, "If set to true framwork will gather metrics from all components after the whole test suite completes.")
	flag.BoolVar(&TestContext.AllowGatheringProfiles, "allow-gathering-profiles", true, "If set to true framework will allow to gather CPU/memory allocation pprof profiles from the master.")
	flag.BoolVar(&TestContext.IncludeClusterAutoscalerMetrics, "include-cluster-autoscaler", false, "If set to true, framework will include Cluster Autoscaler when gathering metrics.")
	flag.StringVar(&TestContext.OutputPrintType, "output-print-type", "json", "Format in which summaries should be printed: 'hr' for human readable, 'json' for JSON ones.")
	flag.BoolVar(&TestContext.DumpLogsOnFailure, "dump-logs-on-failure", true, "If set to true test will dump data about the namespace in which test was running.")
	flag.BoolVar(&TestContext.DisableLogDump, "disable-log-dump", false, "If set to true, logs from master and nodes won't be gathered after test run.")
	flag.StringVar(&TestContext.LogexporterGCSPath, "logexporter-gcs-path", "", "Path to the GCS artifacts directory to dump logs from nodes. Logexporter gets enabled if this is non-empty.")
	flag.BoolVar(&TestContext.DeleteNamespace, "delete-namespace", true, "If true tests will delete namespace after completion. It is only designed to make debugging easier, DO NOT turn it off by default.")
	flag.BoolVar(&TestContext.DeleteNamespaceOnFailure, "delete-namespace-on-failure", true, "If true, framework will delete test namespace on failure. Used only during test debugging.")
	flag.IntVar(&TestContext.AllowedNotReadyNodes, "allowed-not-ready-nodes", 0, "If non-zero, framework will allow for that many non-ready nodes when checking for all ready nodes.")

	flag.StringVar(&TestContext.Host, "host", "", fmt.Sprintf("The host, or apiserver, to connect to. Will default to %s if this argument and --kubeconfig are not set", defaultHost))
	flag.StringVar(&TestContext.ReportPrefix, "report-prefix", "", "Optional prefix for JUnit XML reports. Default is empty, which doesn't prepend anything to the default name.")
	flag.StringVar(&TestContext.ReportDir, "report-dir", "", "Path to the directory where the JUnit XML reports should be saved. Default is empty, which doesn't generate these reports.")
	flag.Var(utilflag.NewMapStringBool(&TestContext.FeatureGates), "feature-gates", "A set of key=value pairs that describe feature gates for alpha/experimental features.")
	flag.StringVar(&TestContext.ContainerRuntime, "container-runtime", "docker", "The container runtime of cluster VM instances (docker/remote).")
	flag.StringVar(&TestContext.ContainerRuntimeEndpoint, "container-runtime-endpoint", "unix:///var/run/dockershim.sock", "The container runtime endpoint of cluster VM instances.")
	flag.StringVar(&TestContext.ContainerRuntimeProcessName, "container-runtime-process-name", "dockerd", "The name of the container runtime process.")
	flag.StringVar(&TestContext.ContainerRuntimePidFile, "container-runtime-pid-file", "/var/run/docker.pid", "The pid file of the container runtime.")
	flag.StringVar(&TestContext.SystemdServices, "systemd-services", "docker", "The comma separated list of systemd services the framework will dump logs for.")
	flag.StringVar(&TestContext.ImageServiceEndpoint, "image-service-endpoint", "", "The image service endpoint of cluster VM instances.")
	flag.StringVar(&TestContext.DockershimCheckpointDir, "dockershim-checkpoint-dir", "/var/lib/dockershim/sandbox", "The directory for dockershim to store sandbox checkpoints.")
	flag.StringVar(&TestContext.KubernetesAnywherePath, "kubernetes-anywhere-path", "/workspace/k8s.io/kubernetes-anywhere", "Which directory kubernetes-anywhere is installed to.")
}

// Register flags specific to the cluster e2e test suite.
func RegisterClusterFlags() {
	flag.BoolVar(&TestContext.VerifyServiceAccount, "e2e-verify-service-account", true, "If true tests will verify the service account before running.")
	flag.StringVar(&TestContext.KubeConfig, clientcmd.RecommendedConfigPathFlag, os.Getenv(clientcmd.RecommendedConfigPathEnvVar), "Path to kubeconfig containing embedded authinfo.")
	flag.StringVar(&TestContext.KubeContext, clientcmd.FlagContext, "", "kubeconfig context to use/override. If unset, will use value from 'current-context'")
	flag.StringVar(&TestContext.KubeAPIContentType, "kube-api-content-type", "application/vnd.kubernetes.protobuf", "ContentType used to communicate with apiserver")

	flag.StringVar(&TestContext.KubeVolumeDir, "volume-dir", "/var/lib/kubelet", "Path to the directory containing the kubelet volumes.")
	flag.StringVar(&TestContext.CertDir, "cert-dir", "", "Path to the directory containing the certs. Default is empty, which doesn't use certs.")
	flag.StringVar(&TestContext.RepoRoot, "repo-root", "../../", "Root directory of kubernetes repository, for finding test files.")
	flag.StringVar(&TestContext.Provider, "provider", "", "The name of the Kubernetes provider (gce, gke, local, etc.)")
	flag.StringVar(&TestContext.Tooling, "tooling", "", "The tooling in use (kops, gke, etc.)")
	flag.StringVar(&TestContext.KubectlPath, "kubectl-path", "kubectl", "The kubectl binary to use. For development, you might use 'cluster/kubectl.sh' here.")
	flag.StringVar(&TestContext.OutputDir, "e2e-output-dir", "/tmp", "Output directory for interesting/useful test data, like performance data, benchmarks, and other metrics.")
	flag.StringVar(&TestContext.Prefix, "prefix", "e2e", "A prefix to be added to cloud resources created during testing.")
	flag.StringVar(&TestContext.MasterOSDistro, "master-os-distro", "debian", "The OS distribution of cluster master (debian, ubuntu, gci, coreos, or custom).")
	flag.StringVar(&TestContext.NodeOSDistro, "node-os-distro", "debian", "The OS distribution of cluster VM instances (debian, ubuntu, gci, coreos, or custom).")
	flag.StringVar(&TestContext.ClusterMonitoringMode, "cluster-monitoring-mode", "standalone", "The monitoring solution that is used in the cluster.")
	flag.BoolVar(&TestContext.EnablePrometheusMonitoring, "prometheus-monitoring", false, "Separate Prometheus monitoring deployed in cluster.")
	flag.StringVar(&TestContext.ClusterDNSDomain, "dns-domain", "cluster.local", "The DNS Domain of the cluster.")

	// TODO: Flags per provider?  Rename gce-project/gce-zone?
	cloudConfig := &TestContext.CloudConfig
	flag.StringVar(&cloudConfig.MasterName, "kube-master", "", "Name of the kubernetes master. Only required if provider is gce or gke")
	flag.StringVar(&cloudConfig.ApiEndpoint, "gce-api-endpoint", "", "The GCE APIEndpoint being used, if applicable")
	flag.StringVar(&cloudConfig.ProjectID, "gce-project", "", "The GCE project being used, if applicable")
	flag.StringVar(&cloudConfig.Zone, "gce-zone", "", "GCE zone being used, if applicable")
	flag.StringVar(&cloudConfig.Region, "gce-region", "", "GCE region being used, if applicable")
	flag.BoolVar(&cloudConfig.MultiZone, "gce-multizone", false, "If true, start GCE cloud provider with multizone support.")
	flag.BoolVar(&cloudConfig.MultiMaster, "gce-multimaster", false, "If true, the underlying GCE/GKE cluster is assumed to be multi-master.")
	flag.StringVar(&cloudConfig.Cluster, "gke-cluster", "", "GKE name of cluster being used, if applicable")
	flag.StringVar(&cloudConfig.NodeInstanceGroup, "node-instance-group", "", "Name of the managed instance group for nodes. Valid only for gce, gke or aws. If there is more than one group: comma separated list of groups.")
	flag.StringVar(&cloudConfig.Network, "network", "e2e", "The cloud provider network for this e2e cluster.")
	flag.IntVar(&cloudConfig.NumNodes, "num-nodes", -1, "Number of nodes in the cluster")
	flag.StringVar(&cloudConfig.ClusterIPRange, "cluster-ip-range", "10.64.0.0/14", "A CIDR notation IP range from which to assign IPs in the cluster.")
	flag.StringVar(&cloudConfig.NodeTag, "node-tag", "", "Network tags used on node instances. Valid only for gce, gke")
	flag.StringVar(&cloudConfig.MasterTag, "master-tag", "", "Network tags used on master instances. Valid only for gce, gke")

	flag.StringVar(&cloudConfig.ClusterTag, "cluster-tag", "", "Tag used to identify resources.  Only required if provider is aws.")
	flag.StringVar(&cloudConfig.ConfigFile, "cloud-config-file", "", "Cloud config file.  Only required if provider is azure.")
	flag.IntVar(&TestContext.MinStartupPods, "minStartupPods", 0, "The number of pods which we need to see in 'Running' state with a 'Ready' condition of true, before we try running tests. This is useful in any cluster which needs some base pod-based services running before it can be used.")
	flag.DurationVar(&TestContext.SystemPodsStartupTimeout, "system-pods-startup-timeout", 10*time.Minute, "Timeout for waiting for all system pods to be running before starting tests.")
	flag.DurationVar(&TestContext.NodeSchedulableTimeout, "node-schedulable-timeout", 30*time.Minute, "Timeout for waiting for all nodes to be schedulable.")
	flag.DurationVar(&TestContext.SystemDaemonsetStartupTimeout, "system-daemonsets-startup-timeout", 5*time.Minute, "Timeout for waiting for all system daemonsets to be ready.")
	flag.StringVar(&TestContext.EtcdUpgradeStorage, "etcd-upgrade-storage", "", "The storage version to upgrade to (either 'etcdv2' or 'etcdv3') if doing an etcd upgrade test.")
	flag.StringVar(&TestContext.EtcdUpgradeVersion, "etcd-upgrade-version", "", "The etcd binary version to upgrade to (e.g., '3.0.14', '2.3.7') if doing an etcd upgrade test.")
	flag.StringVar(&TestContext.IngressUpgradeImage, "ingress-upgrade-image", "", "Image to upgrade to if doing an upgrade test for ingress.")
	flag.StringVar(&TestContext.GCEUpgradeScript, "gce-upgrade-script", "", "Script to use to upgrade a GCE cluster.")
	flag.BoolVar(&TestContext.CleanStart, "clean-start", false, "If true, purge all namespaces except default and system before running tests. This serves to Cleanup test namespaces from failed/interrupted e2e runs in a long-lived cluster.")
}

// Register flags specific to the node e2e test suite.
func RegisterNodeFlags() {
	// Mark the test as node e2e when node flags are api.Registry.
	TestContext.NodeE2E = true
	flag.StringVar(&TestContext.NodeName, "node-name", "", "Name of the node to run tests on.")
	// TODO(random-liu): Move kubelet start logic out of the test.
	// TODO(random-liu): Move log fetch logic out of the test.
	// There are different ways to start kubelet (systemd, initd, docker, manually started etc.)
	// and manage logs (journald, upstart etc.).
	// For different situation we need to mount different things into the container, run different commands.
	// It is hard and unnecessary to deal with the complexity inside the test suite.
	flag.BoolVar(&TestContext.NodeConformance, "conformance", false, "If true, the test suite will not start kubelet, and fetch system log (kernel, docker, kubelet log etc.) to the report directory.")
	flag.BoolVar(&TestContext.PrepullImages, "prepull-images", true, "If true, prepull images so image pull failures do not cause test failures.")
	flag.StringVar(&TestContext.ImageDescription, "image-description", "", "The description of the image which the test will be running on.")
	flag.StringVar(&TestContext.SystemSpecName, "system-spec-name", "", "The name of the system spec (e.g., gke) that's used in the node e2e test. The system specs are in test/e2e_node/system/specs/. This is used by the test framework to determine which tests to run for validating the system requirements.")
}

// HandleFlags sets up all flags and parses the command line.
func HandleFlags() {
	RegisterCommonFlags()
	RegisterClusterFlags()
	flag.Parse()
}

func createKubeConfig(clientCfg *restclient.Config) *clientcmdapi.Config {
	clusterNick := "cluster"
	userNick := "user"
	contextNick := "context"

	config := clientcmdapi.NewConfig()

	credentials := clientcmdapi.NewAuthInfo()
	credentials.Token = clientCfg.BearerToken
	credentials.TokenFile = clientCfg.BearerTokenFile
	credentials.ClientCertificate = clientCfg.TLSClientConfig.CertFile
	if len(credentials.ClientCertificate) == 0 {
		credentials.ClientCertificateData = clientCfg.TLSClientConfig.CertData
	}
	credentials.ClientKey = clientCfg.TLSClientConfig.KeyFile
	if len(credentials.ClientKey) == 0 {
		credentials.ClientKeyData = clientCfg.TLSClientConfig.KeyData
	}
	config.AuthInfos[userNick] = credentials

	cluster := clientcmdapi.NewCluster()
	cluster.Server = clientCfg.Host
	cluster.CertificateAuthority = clientCfg.CAFile
	if len(cluster.CertificateAuthority) == 0 {
		cluster.CertificateAuthorityData = clientCfg.CAData
	}
	cluster.InsecureSkipTLSVerify = clientCfg.Insecure
	config.Clusters[clusterNick] = cluster

	context := clientcmdapi.NewContext()
	context.Cluster = clusterNick
	context.AuthInfo = userNick
	config.Contexts[contextNick] = context
	config.CurrentContext = contextNick

	return config
}

// AfterReadingAllFlags makes changes to the context after all flags
// have been read.
func AfterReadingAllFlags(t *TestContextType) {
	// Only set a default host if one won't be supplied via kubeconfig
	if len(t.Host) == 0 && len(t.KubeConfig) == 0 {
		// Check if we can use the in-cluster config
		if clusterConfig, err := restclient.InClusterConfig(); err == nil {
			if tempFile, err := ioutil.TempFile(os.TempDir(), "kubeconfig-"); err == nil {
				kubeConfig := createKubeConfig(clusterConfig)
				clientcmd.WriteToFile(*kubeConfig, tempFile.Name())
				t.KubeConfig = tempFile.Name()
				klog.Infof("Using a temporary kubeconfig file from in-cluster config : %s", tempFile.Name())
			}
		}
		if len(t.KubeConfig) == 0 {
			klog.Warningf("Unable to find in-cluster config, using default host : %s", defaultHost)
			t.Host = defaultHost
		}
	}
	// Allow 1% of nodes to be unready (statistically) - relevant for large clusters.
	if t.AllowedNotReadyNodes == 0 {
		t.AllowedNotReadyNodes = t.CloudConfig.NumNodes / 100
	}

	// Make sure that all test runs have a valid TestContext.CloudConfig.Provider.
	var err error
	TestContext.CloudConfig.Provider, err = SetupProviderConfig(TestContext.Provider)
	if err == nil {
		return
	}
	if !os.IsNotExist(errors.Cause(err)) {
		Failf("Failed to setup provider config: %v", err)
	}
	// We allow unknown provider parameters for historic reasons. At least log a
	// warning to catch typos.
	// TODO (https://github.com/kubernetes/kubernetes/issues/70200):
	// - remove the fallback for unknown providers
	// - proper error message instead of Failf (which panics)
	klog.Warningf("Unknown provider %q, proceeding as for --provider=skeleton.", TestContext.Provider)
	TestContext.CloudConfig.Provider, err = SetupProviderConfig("skeleton")
	if err != nil {
		Failf("Failed to setup fallback skeleton provider config: %v", err)
	}
}
