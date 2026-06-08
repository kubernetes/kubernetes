/*
Copyright 2025 The Kubernetes Authors.

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

// please speak to SIG-Testing leads before adding anything to this package
// see: https://git.k8s.io/enhancements/keps/sig-testing/5468-invariant-testing
package invariants

import (
	"context"

	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/kubernetes/test/e2e/framework"
	"k8s.io/kubernetes/test/e2e/invariants/metrics"

	"github.com/onsi/ginkgo/v2"
	ginkgotypes "github.com/onsi/ginkgo/v2/types"
)

// checks for api-server metrics that indicate an internal bug has occurred
const invariantMetricsLeafText = "should enable checking invariant metrics"

var _ = framework.SIGDescribe("testing")("Invariant Metrics", func() {
	// this test is a sentinel for selecting the report after suite logic
	//
	// this allows us to run it by default in most jobs, but it can be opted-out,
	// does not run when selecting Conformance, and it can be tagged Flaky
	// if we encounter issues with it
	ginkgo.It(invariantMetricsLeafText, func() {})
})

var _ = ginkgo.ReportAfterSuite("Invariant Metrics", func(ctx ginkgo.SpecContext, report ginkgo.Report) {
	// skip early if we are in dry-run mode and didn't really run any tests
	if report.SuiteConfig.DryRun {
		return
	}
	// check if we ran the 'should enabled checking invariants' "test"
	invariantsSelected := false
	for _, spec := range report.SpecReports {
		if spec.LeafNodeText == invariantMetricsLeafText {
			invariantsSelected = spec.State.Is(ginkgotypes.SpecStatePassed)
			break
		}
	}
	// skip if the associated "test" was skipped
	if !invariantsSelected {
		return
	}
	// otherwise actually check invariants now
	checkInvariantMetrics(ctx)
})

func checkInvariantMetrics(ctx context.Context) {
	// Grab metrics
	config, err := framework.LoadConfig()
	if err != nil {
		framework.Failf("error loading client config: %v", err)
	}
	c, err := clientset.NewForConfig(config)
	if err != nil {
		framework.Failf("error loading client config: %v", err)
	}

	// Check invariant metrics.
	if err := metrics.CheckMetricInvariants(ctx, c, false); err != nil {
		framework.Failf("Invariant check failed: %v", err)
	}
}
