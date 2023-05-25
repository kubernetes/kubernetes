package main

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"

	corev1 "k8s.io/api/core/v1"
	kclientset "k8s.io/client-go/kubernetes"
	"k8s.io/client-go/tools/clientcmd"
	"k8s.io/kubernetes/openshift-hack/e2e"
	conformancetestdata "k8s.io/kubernetes/test/conformance/testdata"
	"k8s.io/kubernetes/test/e2e/framework"
	"k8s.io/kubernetes/test/e2e/framework/testfiles"
	"k8s.io/kubernetes/test/e2e/storage/external"
	e2etestingmanifests "k8s.io/kubernetes/test/e2e/testing-manifests"
	testfixtures "k8s.io/kubernetes/test/fixtures"

	// this appears to inexplicably auto-register global flags.
	_ "k8s.io/kubernetes/test/e2e/storage/drivers"

	// these are loading important global flags that we need to get and set
	_ "k8s.io/kubernetes/test/e2e"
	_ "k8s.io/kubernetes/test/e2e/lifecycle"
)

// copied directly from github.com/openshift/origin/cmd/openshift-tests/provider.go
// and github.com/openshift/origin/test/extended/util/test.go
func initializeTestFramework(provider string) error {
	providerInfo := &ClusterConfiguration{}
	if err := json.Unmarshal([]byte(provider), &providerInfo); err != nil {
		return fmt.Errorf("provider must be a JSON object with the 'type' key at a minimum: %v", err)
	}
	if len(providerInfo.ProviderName) == 0 {
		return fmt.Errorf("provider must be a JSON object with the 'type' key")
	}
	config := &ClusterConfiguration{}
	if err := json.Unmarshal([]byte(provider), config); err != nil {
		return fmt.Errorf("provider must decode into the ClusterConfig object: %v", err)
	}

	// update testContext with loaded config
	testContext := &framework.TestContext
	testContext.Provider = config.ProviderName
	testContext.CloudConfig = framework.CloudConfig{
		ProjectID:   config.ProjectID,
		Region:      config.Region,
		Zone:        config.Zone,
		Zones:       config.Zones,
		NumNodes:    config.NumNodes,
		MultiMaster: config.MultiMaster,
		MultiZone:   config.MultiZone,
		ConfigFile:  config.ConfigFile,
	}
	testContext.AllowedNotReadyNodes = -1
	testContext.MinStartupPods = -1
	testContext.MaxNodesToGather = 0
	testContext.KubeConfig = os.Getenv("KUBECONFIG")

	// allow the CSI tests to access test data, but only briefly
	// TODO: ideally CSI would not use any of these test methods
	// var err error
	// exutil.WithCleanup(func() { err = initCSITests(dryRun) })
	// TODO: for now I'm only initializing CSI directly, but we probably need that
	// WithCleanup here as well
	if err := initCSITests(); err != nil {
		return err
	}

	if ad := os.Getenv("ARTIFACT_DIR"); len(strings.TrimSpace(ad)) == 0 {
		os.Setenv("ARTIFACT_DIR", filepath.Join(os.TempDir(), "artifacts"))
	}

	testContext.DeleteNamespace = os.Getenv("DELETE_NAMESPACE") != "false"
	testContext.VerifyServiceAccount = true
	testfiles.AddFileSource(e2etestingmanifests.GetE2ETestingManifestsFS())
	testfiles.AddFileSource(testfixtures.GetTestFixturesFS())
	testfiles.AddFileSource(conformancetestdata.GetConformanceTestdataFS())
	testContext.KubectlPath = "kubectl"
	// context.KubeConfig = KubeConfigPath()
	testContext.KubeConfig = os.Getenv("KUBECONFIG")

	// "debian" is used when not set. At least GlusterFS tests need "custom".
	// (There is no option for "rhel" or "centos".)
	testContext.NodeOSDistro = "custom"
	testContext.MasterOSDistro = "custom"

	// load and set the host variable for kubectl
	clientConfig := clientcmd.NewNonInteractiveDeferredLoadingClientConfig(&clientcmd.ClientConfigLoadingRules{ExplicitPath: testContext.KubeConfig}, &clientcmd.ConfigOverrides{})
	cfg, err := clientConfig.ClientConfig()
	if err != nil {
		return err
	}
	testContext.Host = cfg.Host

	// Ensure that Kube tests run privileged (like they do upstream)
	testContext.CreateTestingNS = func(ctx context.Context, baseName string, c kclientset.Interface, labels map[string]string) (*corev1.Namespace, error) {
		return e2e.CreateTestingNS(ctx, baseName, c, labels, true)
	}

	gomega.RegisterFailHandler(ginkgo.Fail)

	framework.AfterReadingAllFlags(testContext)
	testContext.DumpLogsOnFailure = true

	// these constants are taken from kube e2e and used by tests
	testContext.IPFamily = "ipv4"
	if config.HasIPv6 && !config.HasIPv4 {
		testContext.IPFamily = "ipv6"
	}

	testContext.ReportDir = os.Getenv("TEST_JUNIT_DIR")

	return nil
}

const (
	manifestEnvVar = "TEST_CSI_DRIVER_FILES"
)

// copied directly from github.com/openshift/origin/cmd/openshift-tests/csi.go
// Initialize openshift/csi suite, i.e. define CSI tests from TEST_CSI_DRIVER_FILES.
func initCSITests() error {
	manifestList := os.Getenv(manifestEnvVar)
	if manifestList != "" {
		manifests := strings.Split(manifestList, ",")
		for _, manifest := range manifests {
			if err := external.AddDriverDefinition(manifest); err != nil {
				return fmt.Errorf("failed to load manifest from %q: %s", manifest, err)
			}
			// Register the base dir of the manifest file as a file source.
			// With this we can reference the CSI driver's storageClass
			// in the manifest file (FromFile field).
			testfiles.AddFileSource(testfiles.RootFileSource{
				Root: filepath.Dir(manifest),
			})
		}
	}

	return nil
}
