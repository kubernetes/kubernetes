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
//
//
// This must be line #50.

var _ = ginkgo.Describe("framework", func() {
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
	ginkgoOutput = `[BeforeEach] framework
  cleanup_test.go:53
INFO: before
[BeforeEach] framework
  framework.go:xxx
STEP: Creating a kubernetes client
INFO: >>> kubeConfig: yyy/kube.config
STEP: Building a namespace api object, basename test-namespace
INFO: Skipping waiting for service account
[It] works
  cleanup_test.go:66
[AfterEach] framework
  cleanup_test.go:59
INFO: after
[DeferCleanup] framework
  cleanup_test.go:71
INFO: cleanup first
[DeferCleanup] framework
  cleanup_test.go:68
INFO: cleanup last
[DeferCleanup] framework
  framework.go:xxx
[DeferCleanup] framework
  framework.go:xxx
STEP: Destroying namespace "test-namespace-zzz" for this suite.
`
)

func TestCleanup(t *testing.T) {
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
			Name:            "framework works",
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
