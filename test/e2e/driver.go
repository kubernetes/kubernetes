/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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
	"fmt"
	"path"
	"regexp"
	"strings"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/cloudprovider"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
	"github.com/golang/glog"
	"github.com/onsi/ginkgo"
	"github.com/onsi/ginkgo/config"
	"github.com/onsi/ginkgo/reporters"
	"github.com/onsi/gomega"
)

type testResult bool

type CloudConfig struct {
	ProjectID  string
	Zone       string
	MasterName string

	Provider cloudprovider.Interface
}

func init() {
	// Turn on verbose by default to get spec names
	config.DefaultReporterConfig.Verbose = true

	// Turn on EmitSpecProgress to get spec progress (especially on interrupt)
	config.GinkgoConfig.EmitSpecProgress = true

	// Randomize specs as well as suites
	config.GinkgoConfig.RandomizeAllSpecs = true
}

func (t *testResult) Fail() { *t = false }

// Run each Go end-to-end-test. This function assumes the
// creation of a test cluster.
func RunE2ETests(context *TestContextType, orderseed int64, times int, reportDir string, testList []string) {
	testContext = *context
	util.ReallyCrash = true
	util.InitLogs()
	defer util.FlushLogs()

	if len(testList) != 0 {
		if config.GinkgoConfig.FocusString != "" || config.GinkgoConfig.SkipString != "" {
			glog.Fatal("Either specify --test/-t or --ginkgo.focus/--ginkgo.skip but not both.")
		}
		var testRegexps []string
		for _, t := range testList {
			testRegexps = append(testRegexps, regexp.QuoteMeta(t))
		}
		config.GinkgoConfig.FocusString = `\b(` + strings.Join(testRegexps, "|") + `)\b`
	}

	// Disable density test unless it's explicitly requested.
	if config.GinkgoConfig.FocusString == "" && config.GinkgoConfig.SkipString == "" {
		config.GinkgoConfig.SkipString = "Skipped"
	}

	// TODO: Make orderseed work again.
	var passed testResult = true
	gomega.RegisterFailHandler(ginkgo.Fail)
	// Run the existing tests with output to console + JUnit for Jenkins
	for i := 0; i < times && passed; i++ {
		var r []ginkgo.Reporter
		if reportDir != "" {
			r = append(r, reporters.NewJUnitReporter(path.Join(reportDir, fmt.Sprintf("junit_%d.xml", i+1))))
		}
		ginkgo.RunSpecsWithDefaultAndCustomReporters(&passed, fmt.Sprintf("Kubernetes e2e Suite run %d of %d", i+1, times), r)
	}

	if !passed {
		glog.Fatalf("At least one test failed")
	} else {
		glog.Infof("All tests pass")
	}
}
