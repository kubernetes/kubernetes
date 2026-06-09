package main

import (
	"flag"
	"fmt"
	"os"
	"reflect"
	"time"

	"k8s.io/kubernetes/test/e2e/framework"

	"github.com/spf13/cobra"
	"github.com/spf13/pflag"

	"github.com/openshift-eng/openshift-tests-extension/pkg/cmd"
	e "github.com/openshift-eng/openshift-tests-extension/pkg/extension"
	ext "github.com/openshift-eng/openshift-tests-extension/pkg/extension/extensiontests"
	g "github.com/openshift-eng/openshift-tests-extension/pkg/ginkgo"
	v "github.com/openshift-eng/openshift-tests-extension/pkg/version"

	"k8s.io/client-go/pkg/version"
	utilflag "k8s.io/component-base/cli/flag"
	"k8s.io/component-base/logs"
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

	if err := initializeCommonTestFramework(); err != nil {
		panic(err)
	}

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
		Name: "kubernetes/conformance/parallel/minimal",
		Parents: []string{
			"openshift/conformance/parallel/minimal",
		},
		Qualifiers: []string{withExcludedTestsFilter(`(!name.contains('[Serial]') && !labels.exists(l, l == '[Serial]')) && labels.exists(l, l == "Conformance")`)},
	})

	kubeTestsExtension.AddSuite(e.Suite{
		Name: "kubernetes/conformance/serial/minimal",
		Parents: []string{
			"openshift/conformance/serial/minimal",
		},
		Qualifiers: []string{withExcludedTestsFilter(`(name.contains('[Serial]') || labels.exists(l, l == '[Serial]')) && labels.exists(l, l == "Conformance")`)},
	})

	// AddGlobalSuite so the umbrella starts with zero qualifiers and inherits
	// exclusively from its children via mergeParentQualifiers in origin.
	kubeTestsExtension.AddGlobalSuite(e.Suite{
		Name: "kubernetes/conformance",
	})

	kubeTestsExtension.AddSuite(e.Suite{
		Name: "kubernetes/conformance/parallel",
		Parents: []string{
			"kubernetes/conformance",
			"openshift/conformance/parallel",
		},
		Qualifiers: []string{withExcludedTestsFilter(`(!name.contains('[Serial]') && !labels.exists(l, l == '[Serial]'))`)},
	})

	kubeTestsExtension.AddSuite(e.Suite{
		Name: "kubernetes/conformance/serial",
		Parents: []string{
			"kubernetes/conformance",
			"openshift/conformance/serial",
		},
		Qualifiers: []string{withExcludedTestsFilter(`(name.contains('[Serial]') || labels.exists(l, l == '[Serial]'))`)},
	})

	hpaTestTimeout := time.Minute * 30
	kubeTestsExtension.AddSuite(e.Suite{
		Name:        "kubernetes/autoscaling/hpa",
		Qualifiers:  []string{"name.contains('[Feature:HPA]') || name.contains('[Feature:HPAConfigurableTolerance]')"}, // Note that this does not use withExcludedTestsFilter to be able to run DedicatedJob labelled tests.
		Parallelism: 3,                                                                                                 // HPA tests have high CPU + memory usage, so we cannot have a high level of parallelism here.
		TestTimeout: &hpaTestTimeout,
	})

	for k, v := range image.GetOriginalImageConfigs() {
		image := convertToImage(v)
		image.Index = int(k)
		kubeTestsExtension.RegisterImage(image)
	}

	// FIXME(stbenjam): what other suites does k8s-test contribute to?

	// Build our specs from ginkgo
	specs, err := g.BuildExtensionTestSpecsFromOpenShiftGinkgoSuite(ext.AllTestsIncludingVendored())
	if err != nil {
		panic(err)
	}

	// Initialization for kube ginkgo test framework needs to run before all tests execute
	specs.AddBeforeAll(func() {
		if err := updateTestFrameworkForTests(os.Getenv("TEST_PROVIDER")); err != nil {
			panic(err)
		}
	})

	specs = filterOutDisabledSpecs(specs)
	addLabelsToSpecs(specs)

	// EnvironmentSelectors added to the appropriate specs to facilitate including or excluding them
	// based on attributes of the cluster they are running on
	addEnvironmentSelectors(specs)

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
func convertToImage(obj interface{}) e.Image {
	image := e.Image{}
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

func withExcludedTestsFilter(baseExpr string) string {
	excluded := []string{
		"[Disabled:",
		"[Disruptive]",
		"[Skipped]",
		"[Slow]",
		"[Flaky]",
		"[Local]",
		"[DedicatedJob]",
	}

	filter := ""
	for i, s := range excluded {
		if i > 0 {
			filter += " && "
		}
		filter += fmt.Sprintf("!name.contains('%s') && !labels.exists(l, l == '%s')", s, s)
	}

	if baseExpr != "" {
		return fmt.Sprintf("(%s) && (%s)", baseExpr, filter)
	}
	return filter
}
