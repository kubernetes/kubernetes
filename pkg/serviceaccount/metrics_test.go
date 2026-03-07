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

package serviceaccount

import (
	"strings"
	"testing"

	"k8s.io/component-base/metrics/legacyregistry"
	"k8s.io/component-base/metrics/testutil"
)

func TestLegacyTokensMetric(t *testing.T) {
	RegisterMetrics()
	legacyTokensTotal.Reset()
	defer legacyTokensTotal.Reset()

	want := `
		# HELP serviceaccount_legacy_tokens_total [BETA] Cumulative legacy service account tokens used
		# TYPE serviceaccount_legacy_tokens_total counter
		serviceaccount_legacy_tokens_total 1
	`

	legacyTokensTotal.Inc()
	if err := testutil.GatherAndCompare(legacyregistry.DefaultGatherer, strings.NewReader(want), "serviceaccount_legacy_tokens_total"); err != nil {
		t.Fatal(err)
	}
}

func TestStaleTokensMetric(t *testing.T) {
	RegisterMetrics()
	staleTokensTotal.Reset()
	defer staleTokensTotal.Reset()

	want := `
		# HELP serviceaccount_stale_tokens_total [BETA] Cumulative stale projected service account tokens used
		# TYPE serviceaccount_stale_tokens_total counter
		serviceaccount_stale_tokens_total 1
	`

	staleTokensTotal.Inc()
	if err := testutil.GatherAndCompare(legacyregistry.DefaultGatherer, strings.NewReader(want), "serviceaccount_stale_tokens_total"); err != nil {
		t.Fatal(err)
	}
}

func TestValidTokensMetric(t *testing.T) {
	RegisterMetrics()
	validTokensTotal.Reset()
	defer validTokensTotal.Reset()

	want := `
		# HELP serviceaccount_valid_tokens_total [BETA] Cumulative valid projected service account tokens used
		# TYPE serviceaccount_valid_tokens_total counter
		serviceaccount_valid_tokens_total 1
	`

	validTokensTotal.Inc()
	if err := testutil.GatherAndCompare(legacyregistry.DefaultGatherer, strings.NewReader(want), "serviceaccount_valid_tokens_total"); err != nil {
		t.Fatal(err)
	}
}
