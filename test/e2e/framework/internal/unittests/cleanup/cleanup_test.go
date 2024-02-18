//go:build linux && amd64

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

// This test uses etcd that is only fully supported for AMD64 and Linux
// https://etcd.io/docs/v3.5/op-guide/supported-platform/#support-tiers

package cleanup

import (
	"context"
	"flag"
	"fmt"
	"regexp"
	"testing"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/ginkgo/v2/reporters"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/test/e2e/framework"
	"k8s.io/kubernetes/test/e2e/framework/internal/output"
	testapiserver "k8s.io/kubernetes/test/utils/apiserver"
	"k8s.io/kubernetes/test/utils/ktesting"
)

// The line number of the following code is checked in TestFailureOutput below.
// Be careful when moving it around or changing the import statements above.
// Here are some intentionally blank lines that can be removed to compensate
// for future additional import statements.
//
//
//
//
// This must be line #50.

func init() {
	framework.NewFrameworkExtensions = append(framework.NewFrameworkExtensions,
		// This callback runs directly after NewDefaultFramework is done.
		func(f *framework.Framework) {
			ginkgo.BeforeEach(func() { framework.Logf("extension before") })
			ginkgo.AfterEach(func() { framework.Logf("extension after") })
		},
	)
}

var _ = ginkgo.Describe("e2e", func() {
	ginkgo.BeforeEach(func() {
		logBeforeHelper()
	})

	f := framework.NewDefaultFramework("test-namespace")

	// BeforeEach/AfterEach run in first-in-first-out order.

	ginkgo.BeforeEach(func() {
		framework.Logf("before #1")
	})

	ginkgo.BeforeEach(func() {
		framework.Logf("before #2")
	})

	ginkgo.AfterEach(func() {
		framework.Logf("after #1")
		if f.ClientSet == nil {
			framework.Fail("Wrong order of cleanup operations: framework.AfterEach already ran and cleared f.ClientSet.")
		}
	})

	ginkgo.AfterEach(func() {
		framework.Logf("after #2")
	})

	ginkgo.It("works", func(ctx context.Context) {
		// DeferCleanup invokes in first-in-last-out order
		ginkgo.DeferCleanup(func() {
			framework.Logf("cleanup last")
		})
		ginkgo.DeferCleanup(func() {
			framework.Logf("cleanup first")
		})

		ginkgo.DeferCleanup(framework.IgnoreNotFound(f.ClientSet.CoreV1().PersistentVolumes().Delete), "simple", metav1.DeleteOptions{})
		fail := func(ctx context.Context, name string) error {
			return fmt.Errorf("fake error for %q", name)
		}
		ginkgo.DeferCleanup(framework.IgnoreNotFound(fail), "failure") // Without a failure the output would not be shown in JUnit.

		tCtx := f.TContext(ctx)
		tCtx.Log("log", "hello", "world")
		tCtx.Logger().Info("info hello world") //
		var discardLogger klog.Logger
		tCtx = ktesting.WithLogger(tCtx, discardLogger)
		oldCtx := tCtx
		tCtx.CleanupCtx(func(tCtx ktesting.TContext) {
			if tCtx.Logger() != discardLogger {
				tCtx.Errorf("expected discard logger in context, got %+v", tCtx.Logger())
			}
			_, ok := tCtx.Value("GINKGO_SPEC_CONTEXT").(ginkgo.SpecContext)
			if !ok {
				tCtx.Errorf("expected Ginkgo context, got %+v", tCtx)
			}
			if oldCtx.Err() == nil {
				tCtx.Error("Ginkgo.It context should be canceled but isn't")
			}
			if tCtx.Err() != nil {
				tCtx.Errorf("Ginkgo.DeferCleanup context should not be canceled but is: %v", tCtx.Err())
			}
		})

		// More test cases can be added here without affeccting line numbering
		// of existing tests.
	})
})

// logBeforeHelper must be skipped when doing stack unwinding in the logging
// implementation.
func logBeforeHelper() {
	ginkgo.GinkgoHelper()
	framework.Logf("before")
}

const (
	ginkgoOutput = `> Enter [BeforeEach] e2e - cleanup_test.go:63 <time>
<klog> cleanup_test.go:64] before
< Exit [BeforeEach] e2e - cleanup_test.go:63 <time>
> Enter [BeforeEach] e2e - set up framework | framework.go:xxx <time>
STEP: Creating a kubernetes client - framework.go:xxx <time>
<klog> util.go:xxx] >>> kubeConfig: yyy/kube.config
STEP: Building a namespace api object, basename test-namespace - framework.go:xxx <time>
<klog> framework.go:xxx] Skipping waiting for service account
< Exit [BeforeEach] e2e - set up framework | framework.go:xxx <time>
> Enter [BeforeEach] e2e - cleanup_test.go:56 <time>
<klog> cleanup_test.go:56] extension before
< Exit [BeforeEach] e2e - cleanup_test.go:56 <time>
> Enter [BeforeEach] e2e - cleanup_test.go:71 <time>
<klog> cleanup_test.go:72] before #1
< Exit [BeforeEach] e2e - cleanup_test.go:71 <time>
> Enter [BeforeEach] e2e - cleanup_test.go:75 <time>
<klog> cleanup_test.go:76] before #2
< Exit [BeforeEach] e2e - cleanup_test.go:75 <time>
> Enter [It] works - cleanup_test.go:90 <time>
log hello world
< Exit [It] works - cleanup_test.go:90 <time>
> Enter [AfterEach] e2e - cleanup_test.go:57 <time>
<klog> cleanup_test.go:57] extension after
< Exit [AfterEach] e2e - cleanup_test.go:57 <time>
> Enter [AfterEach] e2e - cleanup_test.go:79 <time>
<klog> cleanup_test.go:80] after #1
< Exit [AfterEach] e2e - cleanup_test.go:79 <time>
> Enter [AfterEach] e2e - cleanup_test.go:86 <time>
<klog> cleanup_test.go:87] after #2
< Exit [AfterEach] e2e - cleanup_test.go:86 <time>
> Enter [DeferCleanup (Each)] e2e - cleanup_test.go:111 <time>
< Exit [DeferCleanup (Each)] e2e - cleanup_test.go:111 <time>
> Enter [DeferCleanup (Each)] e2e - cleanup_test.go:103 <time>
[FAILED] DeferCleanup callback returned error: fake error for "failure"
In [DeferCleanup (Each)] at: cleanup_test.go:103 <time>
< Exit [DeferCleanup (Each)] e2e - cleanup_test.go:103 <time>
> Enter [DeferCleanup (Each)] e2e - cleanup_test.go:99 <time>
< Exit [DeferCleanup (Each)] e2e - cleanup_test.go:99 <time>
> Enter [DeferCleanup (Each)] e2e - cleanup_test.go:95 <time>
<klog> cleanup_test.go:96] cleanup first
< Exit [DeferCleanup (Each)] e2e - cleanup_test.go:95 <time>
> Enter [DeferCleanup (Each)] e2e - cleanup_test.go:92 <time>
<klog> cleanup_test.go:93] cleanup last
< Exit [DeferCleanup (Each)] e2e - cleanup_test.go:92 <time>
> Enter [DeferCleanup (Each)] e2e - dump namespaces | framework.go:xxx <time>
< Exit [DeferCleanup (Each)] e2e - dump namespaces | framework.go:xxx <time>
> Enter [DeferCleanup (Each)] e2e - tear down framework | framework.go:xxx <time>
STEP: Destroying namespace "test-namespace-zzz" for this suite. - framework.go:xxx <time>
< Exit [DeferCleanup (Each)] e2e - tear down framework | framework.go:xxx <time>
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

	expected := output.TestResult{
		NormalizeOutput: normalizeOutput,
		Suite: reporters.JUnitTestSuite{
			Tests:    1,
			Failures: 1,
			Errors:   0,
			Disabled: 0,
			Skipped:  0,

			TestCases: []reporters.JUnitTestCase{
				{
					Name:   "[It] e2e works",
					Status: "failed",
					Failure: &reporters.JUnitFailure{
						Type: "failed",
						Description: `[FAILED] DeferCleanup callback returned error: fake error for "failure"
In [DeferCleanup (Each)] at: cleanup_test.go:103 <time>
`,
					},
					SystemErr: ginkgoOutput,
				},
			},
		},
	}

	output.TestGinkgoOutput(t, expected, suiteConfig, reporterConfig)
}

func normalizeOutput(output string) string {
	for exp, replacement := range map[string]string{
		// Ignore line numbers inside framework source code (likely to change).
		`(framework|util)\.go:\d+`: `$1.go:xxx`,
		// Config file name varies for each run.
		`kubeConfig: .*/kube.config`: `kubeConfig: yyy/kube.config`,
		// Random suffix for namespace.
		`test-namespace-\d+`: `test-namespace-zzz`,
	} {
		output = regexp.MustCompile(exp).ReplaceAllString(output, replacement)
	}
	return output
}
