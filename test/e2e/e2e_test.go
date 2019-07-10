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
	"flag"
	"fmt"
	"math/rand"
	"os"
	"path"
	"testing"
	"time"

	// Never, ever remove the line with "/ginkgo". Without it,
	// the ginkgo test runner will not detect that this
	// directory contains a Ginkgo test suite.
	// See https://github.com/kubernetes/kubernetes/issues/74827
	"github.com/DATA-DOG/godog"
	"github.com/onsi/ginkgo"
	"k8s.io/kubernetes/test/e2e/framework/ginkgowrapper"

	// "github.com/ess/jamaica"

	"github.com/onsi/ginkgo/reporters"
	"github.com/onsi/gomega"
	"k8s.io/component-base/logs"
	"k8s.io/klog"
	"k8s.io/kubernetes/test/e2e/features/steps"
	"k8s.io/kubernetes/test/e2e/framework"
	e2elog "k8s.io/kubernetes/test/e2e/framework/log"
	"k8s.io/kubernetes/test/e2e/framework/testfiles"
	"k8s.io/kubernetes/test/e2e/framework/viperconfig"
	"k8s.io/kubernetes/test/e2e/generated"
	"k8s.io/kubernetes/test/utils/image"

	// test sources
	_ "k8s.io/kubernetes/test/e2e/apimachinery"
	_ "k8s.io/kubernetes/test/e2e/apps"
	_ "k8s.io/kubernetes/test/e2e/auth"
	_ "k8s.io/kubernetes/test/e2e/autoscaling"
	_ "k8s.io/kubernetes/test/e2e/cloud"
	_ "k8s.io/kubernetes/test/e2e/common"
	_ "k8s.io/kubernetes/test/e2e/instrumentation"
	_ "k8s.io/kubernetes/test/e2e/kubectl"
	_ "k8s.io/kubernetes/test/e2e/lifecycle"
	_ "k8s.io/kubernetes/test/e2e/lifecycle/bootstrap"
	_ "k8s.io/kubernetes/test/e2e/network"
	_ "k8s.io/kubernetes/test/e2e/node"
	_ "k8s.io/kubernetes/test/e2e/scalability"
	_ "k8s.io/kubernetes/test/e2e/scheduling"
	_ "k8s.io/kubernetes/test/e2e/servicecatalog"
	_ "k8s.io/kubernetes/test/e2e/storage"
	_ "k8s.io/kubernetes/test/e2e/storage/external"
	_ "k8s.io/kubernetes/test/e2e/ui"
	_ "k8s.io/kubernetes/test/e2e/windows"
)

var viperConfig = flag.String("viper-config", "", "The name of a viper config file (https://github.com/spf13/viper#what-is-viper). All e2e command line parameters can also be configured in such a file. May contain a path and may or may not contain the file suffix. The default is to look for an optional file with `e2e` as base name. If a file is specified explicitly, it must be present.")

var (
	runGoDogTests bool
	stopOnFailure bool
)

func init() {
	e2elog.Logf("The FIRST thing that runs")
	// Parse our godog flags... TODO: move to framework.HandleFlags()
	flag.BoolVar(&runGoDogTests, "godog", false, "Set this flag is you want to run godog BDD tests")
	flag.BoolVar(&stopOnFailure, "stop-on-failure", false, "Stop processing on first failed scenario.. Flag is passed to godog")
	flag.Parse()
	e2elog.Logf("runGoDogTests to %v", runGoDogTests)
	// Register framework flags, then handle flags and Viper config.
	framework.HandleFlags()
	if err := viperconfig.ViperizeFlags(*viperConfig, "e2e"); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}

	if framework.TestContext.ListImages {
		for _, v := range image.GetImageConfigs() {
			fmt.Println(v.GetE2EImage())
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

	// Enable bindata file lookup as fallback.
	testfiles.AddFileSource(testfiles.BindataFileSource{
		Asset:      generated.Asset,
		AssetNames: generated.AssetNames,
	})
}

func TestingMainFeatureContext(s *godog.Suite, m *testing.M) {
	e2elog.Logf("Adding Before Suite")
	s.BeforeSuite(func() {
		e2elog.Logf("Running Before Suite")
		// Just in case the standard logs are used
		logs.InitLogs()
		defer logs.FlushLogs()

		/////// We needed to register the Fail Handler otherwise we get this:
		// panic: You are trying to make an assertion, but Gomega's fail handler is nil.
		// If you're using Ginkgo then you probably forgot to put your assertion in an It().
		// Alternatively, you may have forgotten to register a fail handler with RegisterFailHandler() or RegisterTestingT().
		// Depending on your vendoring solution you may be inadvertently importing gomega and subpackages (e.g. ghhtp, gexec,...) from different locations.
		gomega.RegisterFailHandler(ginkgowrapper.Fail)
		BeforeSuiteOnce()
	})
	e2elog.Logf("Adding After Suite")
	s.AfterSuite(func() {
		e2elog.Logf("Running After Suite Once")
		AfterSuiteOnce()
		e2elog.Logf("Running After Suite Many")
		AfterSuiteMany() // not sure how per node werks here
		// Run tests through the Ginkgo runner with output to console + JUnit for Jenkins
		var r []ginkgo.Reporter
		if framework.TestContext.ReportDir != "" {
			// TODO: we should probably only be trying to create this directory once
			// rather than once-per-Ginkgo-node.
			if err := os.MkdirAll(framework.TestContext.ReportDir, 0755); err != nil {
				klog.Errorf("Failed creating report directory: %v", err)
			} else {
				r = append(r, reporters.NewJUnitReporter(
					path.Join(framework.TestContext.ReportDir,
						fmt.Sprintf("junit_%v-godog.xml",
							framework.TestContext.ReportPrefix))))
				// framework.TestContext.ReportPrefix,
				// config.GinkgoConfig.ParallelNode))))
			}
		}
		// ginkgo.globalSuite.beforeSuiteNode = nil
		// ginkgo.globalSuite.afterSuiteNode = nil
		// 	SynchronizedBeforeSuite(
		// 	func() []byte {
		// 		// Do nothing since go-ds handling the suite setup
		// 		return nil
		// 	},
		// 	func(data []byte) {
		// 		// Do nothing since go-dog is handling the suite setup
		// 	})
		// ginkgo.SynchronizedAfterSuite(func() {
		// 	// Do nothing since go-dog is handling the suite teardown
		// }, func() {
		// 	// Do nothing since go-dog is handling the suite teardown
		// })
	})
	// jamaica.SetRootCmd("echo") cmd needs to be cobra.cmd
	// jamaica.StepUp(s)
	e2elog.Logf("Adding FirstSteps FeatureContext")
	steps.FirstStepsFeatureContext(s, m)
	// Do something with Suite s !!
}

func TestMain(m *testing.M) {
	// e2elog.Logf("TestMain being called with Testing.M: %v", m.tests)

	// if not with with --godog, use our existing e2e framework as is
	if !runGoDogTests {
		rand.Seed(time.Now().UnixNano())
		os.Exit(m.Run())
	}
	// otherwise let's use godog
	status := godog.RunWithOptions("e2e", func(s *godog.Suite) {
		// if we can pass m, we can call m.Run()
		TestingMainFeatureContext(s, m)
	}, godog.Options{
		Format: "pretty",
		Paths:  []string{"test/e2e/features"},
		// Randomize:     time.Now().UTC().UnixNano(),
		StopOnFailure: stopOnFailure,
	})

	if st := m.Run(); st > status {
		status = st
	}

	os.Exit(status)
}

func TestE2E(t *testing.T) {
	e2elog.Logf("TestE2E being called with testing.T")
	if !runGoDogTests {
		e2elog.Logf("Running existing E2E Tests")
		RunE2ETests(t)
	}
}
