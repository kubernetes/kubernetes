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

package skipper_test

import (
	"flag"
	"testing"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/ginkgo/v2/reporters"

	"k8s.io/kubernetes/test/e2e/framework"
	"k8s.io/kubernetes/test/e2e/framework/internal/output"
	e2eskipper "k8s.io/kubernetes/test/e2e/framework/skipper"
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
//
// This must be line #50.

var _ = ginkgo.Describe("e2e", func() {
	ginkgo.It("skips", func() {
		e2eskipper.Skipf("skipping %d, %d, %d", 1, 3, 4)
	})
})

func TestSkip(t *testing.T) {
	// This simulates how test/e2e uses the framework and how users
	// invoke test/e2e.
	framework.RegisterCommonFlags(flag.CommandLine)
	framework.RegisterClusterFlags(flag.CommandLine)
	for flagname, value := range map[string]string{
		// This simplifies the text comparison.
		"ginkgo.no-color": "true",
	} {
		if err := flag.Set(flagname, value); err != nil {
			t.Fatalf("set %s: %v", flagname, err)
		}
	}
	framework.AfterReadingAllFlags(&framework.TestContext)
	suiteConfig, reporterConfig := framework.CreateGinkgoConfig()
	reporterConfig.Verbose = true

	expected := output.TestResult{
		Suite: reporters.JUnitTestSuite{
			Tests:    1,
			Failures: 0,
			Errors:   0,
			Disabled: 0,
			Skipped:  1,
			TestCases: []reporters.JUnitTestCase{
				{
					Name:   "[It] e2e skips",
					Status: "skipped",
					Skipped: &reporters.JUnitSkipped{
						Message: `skipped - skipping 1, 3, 4`,
					},
				},
			},
		},
	}

	output.TestGinkgoOutput(t, expected, suiteConfig, reporterConfig)
}
