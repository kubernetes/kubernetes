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
	"strings"

	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/component-base/metrics/testutil"
	"k8s.io/kubernetes/test/e2e/framework"
	e2emetrics "k8s.io/kubernetes/test/e2e/framework/metrics"

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

// apiServerMetricInvariant represents an api-server metric invariant, all
// fields must be specified
type apiServerMetricInvariant struct {
	// Metric is the metric name
	Metric string
	// SIG associated with the invariant
	// Bugs related to this invariant check failing should be labeled with
	// this SIG.
	SIG string
	// Owners are the individuals responsible for the invariant
	// Bugs related to this invariant check failing should be assigned to these
	// GitHub handles
	Owners []string
	// IsValid should return false if the metric samples violate the invariant
	IsValid func(testutil.Samples) bool
}

// Please speak to SIG-Testing leads before adding anything here.
// All fields must be specified
var apiServerMetricInvariants = []apiServerMetricInvariant{
	{
		// TODO: Migrate to apiserver_validation_declarative_validation_parity_discrepancies_total
		// when it reaches beta / when this metric is deprecated.
		// For now we uare using the previous beta metric.
		Metric: "apiserver_validation_declarative_validation_mismatch_total",
		SIG:    "api-machinery",
		Owners: []string{"aaron-prindle", "jpbetz", "thockin"},
		IsValid: func(samples testutil.Samples) bool {
			// declarative validation mismatch should never be non-zero
			for _, sample := range samples {
				if sample.Value != 0 {
					return false
				}
			}
			return true
		},
	},
	{
		// TODO: Migrate to apiserver_validation_declarative_validation_panics_total
		// when it reaches beta / when this metric is deprecated.
		// For now we uare using the previous beta metric.
		Metric: "apiserver_validation_declarative_validation_panic_total",
		SIG:    "api-machinery",
		Owners: []string{"aaron-prindle", "jpbetz", "thockin"},
		IsValid: func(samples testutil.Samples) bool {
			// declarative validation panics should never be non-zero
			for _, sample := range samples {
				if sample.Value != 0 {
					return false
				}
			}
			return true
		},
	},
}

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
	grabber, err := e2emetrics.NewMetricsGrabber(ctx, c, nil, config, false, false, false, true, false, false)
	if err != nil {
		framework.Failf("error creating metrics grabber: %v", err)
	}
	apiserverMetrics, err := grabber.GrabFromAPIServer(ctx)
	if err != nil {
		framework.Failf("error grabbing api-server metrics: %v", err)
	}

	// Check invariant metrics.
	//
	//
	// Please speak to SIG-Testing leads before adding anything here.
	for _, invariant := range apiServerMetricInvariants {
		checkAPIServerInvariant(apiserverMetrics, &invariant)
	}
}

func checkAPIServerInvariant(metrics e2emetrics.APIServerMetrics, invariant *apiServerMetricInvariant) {
	metric := invariant.Metric
	samples, ok := metrics[metric]
	if !ok || len(samples) == 0 {
		framework.Failf(
			`Invariant failed for missing metric: %v

If this failed on a pull request, please check if the PR changes may be related to the failure.
If not, you can also search for an existing GitHub issue before filing a new issue.

If this failed in a periodic CI job, please file a bug and /assign the owners.

Owners for this metric: %v
Associated SIG: %v`,
			metric, strings.Join(invariant.Owners, " "), invariant.SIG,
		)
	}
	if !invariant.IsValid(samples) {
		framework.Failf(
			`Invariant failed for metric: %v

If this failed on a pull request, please check if the PR changes may be related to the failure.
If not, you can also search for an existing GitHub issue before filing a new issue.

If this failed in a periodic CI job, please file a bug and /assign the owners.

Owners for this metric: %v
Associated SIG: %v`,
			metric, strings.Join(invariant.Owners, " "), invariant.SIG,
		)
	}
}
