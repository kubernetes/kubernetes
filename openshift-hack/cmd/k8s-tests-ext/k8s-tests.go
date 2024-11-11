package main

import (
	"os"

	"github.com/spf13/cobra"
	"github.com/spf13/pflag"

	"github.com/openshift-eng/openshift-tests-extension/pkg/cmd"
	e "github.com/openshift-eng/openshift-tests-extension/pkg/extension"
	"github.com/openshift-eng/openshift-tests-extension/pkg/extension/extensiontests"
	g "github.com/openshift-eng/openshift-tests-extension/pkg/ginkgo"
	v "github.com/openshift-eng/openshift-tests-extension/pkg/version"

	"k8s.io/client-go/pkg/version"
	utilflag "k8s.io/component-base/cli/flag"
	"k8s.io/component-base/logs"
	"k8s.io/kubernetes/openshift-hack/e2e/annotate/generated"

	// initialize framework extensions
	_ "k8s.io/kubernetes/test/e2e/framework/debug/init"
	_ "k8s.io/kubernetes/test/e2e/framework/metrics/init"
)

func main() {
	logs.InitLogs()
	defer logs.FlushLogs()
	pflag.CommandLine.SetNormalizeFunc(utilflag.WordSepNormalizeFunc)

	// Get version info from kube
	kubeVersion := version.Get()
	v.GitTreeState = kubeVersion.GitTreeState
	v.BuildDate = kubeVersion.BuildDate
	v.CommitFromGit = kubeVersion.GitCommit

	// Create our registry of openshift-tests extensions
	extensionRegistry := e.NewRegistry()
	kubeTestsExtension := e.NewExtension("openshift", "payload", "hyperkube")
	extensionRegistry.Register(kubeTestsExtension)

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

	//FIXME(stbenjam): what other suites does k8s-test contribute to?

	// Build our specs from ginkgo
	specs, err := g.BuildExtensionTestSpecsFromOpenShiftGinkgoSuite()
	if err != nil {
		panic(err)
	}

	// Initialization for kube ginkgo test framework needs to run before all tests execute
	specs.AddBeforeAll(func() {
		if err := initializeTestFramework(os.Getenv("TEST_PROVIDER")); err != nil {
			panic(err)
		}
	})

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
