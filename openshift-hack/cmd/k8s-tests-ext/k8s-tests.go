package main

import (
	"flag"
	"os"
	"reflect"

	"k8s.io/kubernetes/test/e2e/framework"

	"github.com/spf13/cobra"
	"github.com/spf13/pflag"

	"github.com/openshift-eng/openshift-tests-extension/pkg/cmd"
	"github.com/openshift-eng/openshift-tests-extension/pkg/extension"
	e "github.com/openshift-eng/openshift-tests-extension/pkg/extension"
	"github.com/openshift-eng/openshift-tests-extension/pkg/extension/extensiontests"
	g "github.com/openshift-eng/openshift-tests-extension/pkg/ginkgo"
	v "github.com/openshift-eng/openshift-tests-extension/pkg/version"

	"k8s.io/client-go/pkg/version"
	"k8s.io/client-go/tools/clientcmd"
	utilflag "k8s.io/component-base/cli/flag"
	"k8s.io/component-base/logs"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/openshift-hack/e2e/annotate/generated"
	"k8s.io/kubernetes/test/utils/image"

	// initialize framework extensions
	_ "k8s.io/kubernetes/test/e2e/framework/debug/init"
	_ "k8s.io/kubernetes/test/e2e/framework/metrics/init"
)

func main() {
	logs.InitLogs()
	defer logs.FlushLogs()
	pflag.CommandLine.SetNormalizeFunc(utilflag.WordSepNormalizeFunc)

	// These flags are used to pull in the default values to test context - required
	// so tests run correctly, even if the underlying flags aren't used.
	framework.RegisterCommonFlags(flag.CommandLine)
	framework.RegisterClusterFlags(flag.CommandLine)

	// Get version info from kube
	kubeVersion := version.Get()
	v.GitTreeState = kubeVersion.GitTreeState
	v.BuildDate = kubeVersion.BuildDate
	v.CommitFromGit = kubeVersion.GitCommit

	// Create our registry of openshift-tests extensions
	extensionRegistry := e.NewRegistry()
	kubeTestsExtension := e.NewExtension("openshift", "payload", "hyperkube")
	extensionRegistry.Register(kubeTestsExtension)

	providerJSON := os.Getenv("TEST_PROVIDER")
	if providerJSON == "" {
		klog.Fatal("TEST_PROVIDER must be set (example: export TEST_PROVIDER='{\"type\":\"local\"}')")
	}

	// Initialization for kube ginkgo test framework needs to run before all tests are discovered.
	// Some tests use the testContext to generate e2e tests.
	if err := initializeTestFramework(providerJSON); err != nil {
		if clientcmd.IsEmptyConfig(err) {
			klog.Fatalf("Failed to initialize Kubernetes client. Is KUBECONFIG set? Full error: %v", err)
		}
		klog.Fatalf("Failed to initialize test framework: %v", err)
	}

	// Carve up the kube tests into our openshift suites...
	kubeTestsExtension.AddSuite(e.Suite{
		Name: "kubernetes/conformance/parallel",
		Parents: []string{
			"openshift/conformance/parallel",
			"openshift/conformance/parallel/minimal",
		},
		Qualifiers: []string{`!labels.exists(l, l == "Serial") && labels.exists(l, l == "Conformance")`},
	})

	kubeTestsExtension.AddSuite(e.Suite{
		Name: "kubernetes/conformance/serial",
		Parents: []string{
			"openshift/conformance/serial",
			"openshift/conformance/serial/minimal",
		},
		Qualifiers: []string{`labels.exists(l, l == "Serial") && labels.exists(l, l == "Conformance")`},
	})

	for k, v := range image.GetOriginalImageConfigs() {
		image := convertToImage(v)
		image.Index = int(k)
		kubeTestsExtension.RegisterImage(image)
	}

	//FIXME(stbenjam): what other suites does k8s-test contribute to?

	// Build our specs from ginkgo
	specs, err := g.BuildExtensionTestSpecsFromOpenShiftGinkgoSuite()
	if err != nil {
		klog.Fatalf("Failed to build test specs: %v", err)
	}

	// Annotations get appended to test names, these are additions to upstream
	// tests for controlling skips, suite membership, etc.
	//
	// TODO:
	//		- Remove this annotation code, and migrate to Labels/Tags and
	//		  the environmental skip code from the enhancement once its implemented.
	//		- Make sure to account for test renames that occur because of removal of these
	//		  annotations
	specs.Walk(func(spec *extensiontests.ExtensionTestSpec) {
		if annotations, ok := generated.Annotations[spec.Name]; ok {
			spec.Name += annotations
		}
	})

	kubeTestsExtension.AddSpecs(specs)

	// Cobra stuff
	root := &cobra.Command{
		Long: "Kubernetes tests extension for OpenShift",
	}

	root.AddCommand(
		cmd.DefaultExtensionCommands(extensionRegistry)...,
	)

	if err := func() error {
		return root.Execute()
	}(); err != nil {
		os.Exit(1)
	}
}

// convertToImages converts an image.Config to an extension.Image, which
// can easily be serialized to JSON. Since image.Config has unexported fields,
// reflection is used to read its values.
func convertToImage(obj interface{}) extension.Image {
	image := extension.Image{}
	val := reflect.ValueOf(obj)
	typ := reflect.TypeOf(obj)
	for i := 0; i < val.NumField(); i++ {
		structField := typ.Field(i)
		fieldValue := val.Field(i)
		switch structField.Name {
		case "registry":
			image.Registry = fieldValue.String()
		case "name":
			image.Name = fieldValue.String()
		case "version":
			image.Version = fieldValue.String()
		}
	}
	return image
}
