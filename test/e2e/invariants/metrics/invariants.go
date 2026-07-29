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

package metrics

import (
	"context"
	"fmt"
	"strings"
	"time"

	utilerrors "k8s.io/apimachinery/pkg/util/errors"
	"k8s.io/client-go/kubernetes"
	metricstestutil "k8s.io/component-base/metrics/testutil"
)

const (
	// apiServerValidationDeclarativeValidationMismatchTotal is the metric name for declarative validation mismatches.
	apiServerValidationDeclarativeValidationMismatchTotal = "apiserver_validation_declarative_validation_mismatch_total"
	// apiServerValidationDeclarativeValidationPanicTotal is the metric name for declarative validation panics.
	apiServerValidationDeclarativeValidationPanicTotal = "apiserver_validation_declarative_validation_panic_total"
)

// metricInvariant defines an invariant check for a metric.
type metricInvariant struct {
	metricName string
	// sig associated with the invariant
	// Bugs related to this invariant check failing should be labeled with
	// this sig.
	sig string
	// owners are the individuals responsible for the invariant
	// Bugs related to this invariant check failing should be assigned to these
	// GitHub handles
	owners  []string
	isValid func(metricstestutil.Samples) bool
}

// apiServerInvariants are the standard invariants checked for the API server.
//
// Please speak to SIG-Testing leads before adding anything here.
var apiServerInvariants = []metricInvariant{
	{
		metricName: apiServerValidationDeclarativeValidationMismatchTotal,
		sig:        "api-machinery",
		owners:     []string{"aaron-prindle", "jpbetz", "thockin"},
		isValid:    allSamplesZero,
	},
	{
		metricName: apiServerValidationDeclarativeValidationPanicTotal,
		sig:        "api-machinery",
		owners:     []string{"aaron-prindle", "jpbetz", "thockin"},
		isValid:    allSamplesZero,
	},
}

// checkInvariants checks the provided metrics against a list of invariants.
// If failOnMissing is true, it returns an error if a metric listed in invariants is missing from the metrics map.
func checkInvariants(metrics map[string]metricstestutil.Samples, invariants []metricInvariant, failOnMissing bool) error {
	var errs []error
	for _, inv := range invariants {
		samples, ok := metrics[inv.metricName]
		if !ok || len(samples) == 0 {
			if failOnMissing {
				errs = append(errs, fmt.Errorf(`metric %s is missing from scrape (SIG: %s, Owners: %s)

If this failed on a pull request, please check if the PR changes may be related to the failure.
If not, you can also search for an existing GitHub issue before filing a new issue.

If this failed in a periodic CI job, please file a bug and /assign the owners`, inv.metricName, inv.sig, strings.Join(inv.owners, ", ")))
			}
			continue
		}
		if !inv.isValid(samples) {
			errs = append(errs, fmt.Errorf(`metric %s invariant failed (SIG: %s, Owners: %s)

If this failed on a pull request, please check if the PR changes may be related to the failure.
If not, you can also search for an existing GitHub issue before filing a new issue.

If this failed in a periodic CI job, please file a bug and /assign the owners`, inv.metricName, inv.sig, strings.Join(inv.owners, ", ")))
		}
	}
	return utilerrors.NewAggregate(errs)
}

// allSamplesZero is a helper function that returns true if all samples have a value of zero.
func allSamplesZero(samples metricstestutil.Samples) bool {
	for _, sample := range samples {
		if sample.Value != 0 {
			return false
		}
	}
	return true
}

// CheckMetricInvariants scrapes metrics from the API server and checks that
// invariant metrics have the expected values.
func CheckMetricInvariants(ctx context.Context, client kubernetes.Interface, failOnMissing bool) error {
	scrapeCtx, cancelScrape := context.WithTimeout(ctx, 5*time.Second)
	defer cancelScrape()
	body, err := client.Discovery().RESTClient().Get().AbsPath("metrics").DoRaw(scrapeCtx)
	if err != nil {
		return fmt.Errorf("failed to scrape metrics: %w", err)
	}

	scrapedMetrics := metricstestutil.NewMetrics()
	if err := metricstestutil.ParseMetrics(string(body), &scrapedMetrics); err != nil {
		return fmt.Errorf("failed to parse metrics: %w", err)
	}

	return checkInvariants(scrapedMetrics, apiServerInvariants, failOnMissing)
}
