package e2e

//go:generate go run -mod vendor ./annotate/cmd -- ./annotate/generated/zz_generated.annotations.go

// This file duplicates most of test/e2e/e2e_test.go but limits the included
// tests (via include.go) to tests that are relevant to openshift.

import (
	"flag"
	"fmt"
	"math/rand"
	"os"
	"testing"
	"time"

	"gopkg.in/yaml.v2"

	// Never, ever remove the line with "/ginkgo". Without it,
	// the ginkgo test runner will not detect that this
	// directory contains a Ginkgo test suite.
	// See https://github.com/kubernetes/kubernetes/issues/74827
	// "github.com/onsi/ginkgo"

	"k8s.io/component-base/version"
	"k8s.io/kubernetes/test/e2e"
	"k8s.io/kubernetes/test/e2e/framework"
	"k8s.io/kubernetes/test/e2e/framework/testfiles"
	"k8s.io/kubernetes/test/e2e/generated"
	"k8s.io/kubernetes/test/utils/image"

	// Ensure test annotation
	_ "k8s.io/kubernetes/openshift-hack/e2e/annotate/generated"
)

var viperConfig = flag.String("viper-config", "", "The name of a viper config file (https://github.com/spf13/viper#what-is-viper). All e2e command line parameters can also be configured in such a file. May contain a path and may or may not contain the file suffix. The default is to look for an optional file with `e2e` as base name. If a file is specified explicitly, it must be present.")

func TestMain(m *testing.M) {
	var versionFlag bool
	flag.CommandLine.BoolVar(&versionFlag, "version", false, "Displays version information.")

	// Register test flags, then parse flags.
	e2e.HandleFlags()

	// Now that we know which Viper config (if any) was chosen,
	// parse it and update those options which weren't already set via command line flags
	// (which have higher priority).
	if err := e2e.ViperizeFlagsForKubeE2E(*viperConfig, "e2e", flag.CommandLine); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}

	if framework.TestContext.ListImages {
		for _, v := range image.GetImageConfigs() {
			fmt.Println(v.GetE2EImage())
		}
		os.Exit(0)
	}
	if versionFlag {
		fmt.Printf("%s\n", version.Get())
		os.Exit(0)
	}

	// Enable bindata file lookup as fallback.
	testfiles.AddFileSource(testfiles.BindataFileSource{
		Asset:      generated.Asset,
		AssetNames: generated.AssetNames,
	})
	if framework.TestContext.ListConformanceTests {
		var tests []struct {
			Testname    string `yaml:"testname"`
			Codename    string `yaml:"codename"`
			Description string `yaml:"description"`
			Release     string `yaml:"release"`
			File        string `yaml:"file"`
		}

		data, err := testfiles.Read("test/conformance/testdata/conformance.yaml")
		if err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
		if err := yaml.Unmarshal(data, &tests); err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
		if err := yaml.NewEncoder(os.Stdout).Encode(tests); err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
		os.Exit(0)
	}

	framework.AfterReadingAllFlags(&framework.TestContext)

	// TODO: Deprecating repo-root over time... instead just use gobindata_util.go , see #23987.
	// Right now it is still needed, for example by
	// test/e2e/framework/ingress/ingress_utils.go
	// for providing the optional secret.yaml file and by
	// test/e2e/framework/util.go for cluster/log-dump.
	if framework.TestContext.RepoRoot != "" {
		testfiles.AddFileSource(testfiles.RootFileSource{Root: framework.TestContext.RepoRoot})
	}

	rand.Seed(time.Now().UnixNano())
	os.Exit(m.Run())
}

func TestE2E(t *testing.T) {
	e2e.RunE2ETests(t)
}
