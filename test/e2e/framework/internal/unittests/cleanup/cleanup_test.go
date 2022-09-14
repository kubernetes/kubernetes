/*
Copyright 2022 The Kubernetes Authors.

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

package cleanup

import (
	"flag"
	"regexp"
	"testing"

	"github.com/onsi/ginkgo/v2"

	"k8s.io/klog/v2"
	"k8s.io/klog/v2/ktesting"
	"k8s.io/kubernetes/test/e2e/framework"
	"k8s.io/kubernetes/test/e2e/framework/internal/output"
	testapiserver "k8s.io/kubernetes/test/utils/apiserver"
)

// The line number of the following code is checked in TestFailureOutput below.
// Be careful when moving it around or changing the import statements above.
// Here are some intentionally blank lines that can be removed to compensate
// for future additional import statements.
//
//
//
//
//
//
//
//
//
//
//
//
//
// This must be line #50.

var _ = ginkgo.Describe("e2e", func() {
	ginkgo.BeforeEach(func() {
		framework.Logf("before")
	})

	f := framework.NewDefaultFramework("test-namespace")

	ginkgo.AfterEach(func() {
		framework.Logf("after")
		if f.ClientSet == nil {
			framework.Fail("Wrong order of cleanup operations: framework.AfterEach already ran and cleared f.ClientSet.")
		}
	})

	ginkgo.It("works", func() {
		// DeferCleanup invokes in first-in-last-out order
		ginkgo.DeferCleanup(func() {
			framework.Logf("cleanup last")
		})
		ginkgo.DeferCleanup(func() {
			framework.Logf("cleanup first")
		})
	})
})

const (
	ginkgoOutput = `[BeforeEach] e2e
  cleanup_test.go:53
INFO: before
[BeforeEach] e2e
  set up framework | framework.go:xxx
STEP: Creating a kubernetes client
INFO: >>> kubeConfig: yyy/kube.config
STEP: Building a namespace api object, basename test-namespace
INFO: Skipping waiting for service account
[It] works
  cleanup_test.go:66
[AfterEach] e2e
  cleanup_test.go:59
INFO: after
[DeferCleanup] e2e
  cleanup_test.go:71
INFO: cleanup first
[DeferCleanup] e2e
  cleanup_test.go:68
INFO: cleanup last
[DeferCleanup] e2e
  dump namespaces | framework.go:xxx
[DeferCleanup] e2e
  tear down framework | framework.go:xxx
STEP: Destroying namespace "test-namespace-zzz" for this suite.
`
)

func TestCleanup(t *testing.T) {
	// The control plane is noisy and randomly logs through klog, for example:
	// E0912 07:08:46.100164   75466 controller.go:254] unable to sync kubernetes service: Endpoints "kubernetes" is invalid: subsets[0].addresses[0].ip: Invalid value: "127.0.0.1": may not be in the loopback range (127.0.0.0/8, ::1/128)
	//
	// By creating a ktesting logger and registering that as global
	// default logger we get the control plane output into the
	// "go test" output in case of a failure (useful for debugging!)
	// while keeping it out of the captured Ginkgo output that
	// the test is comparing below.
	//
	// There are some small drawbacks:
	// - The source code location for control plane log messages
	//   is shown as klog.go because klog does not properly
	//   skip its own helper functions. That's okay, normally
	//   ktesting should not be installed as logging backend like this.
	// - klog.Infof messages are printed with an extra newline.
	logger, _ := ktesting.NewTestContext(t)
	klog.SetLogger(logger)

	apiServer := testapiserver.StartAPITestServer(t)

	// This simulates how test/e2e uses the framework and how users
	// invoke test/e2e.
	framework.RegisterCommonFlags(flag.CommandLine)
	framework.RegisterClusterFlags(flag.CommandLine)
	for flagname, value := range map[string]string{
		"kubeconfig": apiServer.KubeConfigFile,
		// Some features are not supported by the fake cluster.
		"e2e-verify-service-account": "false",
		"allowed-not-ready-nodes":    "-1",
		// This simplifies the text comparison.
		"ginkgo.no-color": "true",
	} {
		if err := flag.Set(flagname, value); err != nil {
			t.Fatalf("set %s: %v", flagname, err)
		}
	}
	framework.AfterReadingAllFlags(&framework.TestContext)
	suiteConfig, reporterConfig := framework.CreateGinkgoConfig()

	expected := output.SuiteResults{
		output.TestResult{
			Name:            "e2e works",
			NormalizeOutput: normalizeOutput,
			Output:          ginkgoOutput,
		},
	}

	output.TestGinkgoOutput(t, expected, suiteConfig, reporterConfig)
}

func normalizeOutput(output string) string {
	for exp, replacement := range map[string]string{
		// Ignore line numbers inside framework source code (likely to change).
		`framework\.go:\d+`: `framework.go:xxx`,
		// Config file name varies for each run.
		`kubeConfig: .*/kube.config`: `kubeConfig: yyy/kube.config`,
		// Random suffix for namespace.
		`test-namespace-\d+`: `test-namespace-zzz`,
	} {
		output = regexp.MustCompile(exp).ReplaceAllString(output, replacement)
	}
	return output
}
