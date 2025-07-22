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

package aggregated_test

import (
	"fmt"
	"io"
	"strings"
	"testing"

	"k8s.io/component-base/metrics/legacyregistry"
	"k8s.io/component-base/metrics/testutil"

	discoveryendpoint "k8s.io/apiserver/pkg/endpoints/discovery/aggregated"
)

func formatExpectedMetrics(aggregationCount int) io.Reader {
	expected := ``
	if aggregationCount > 0 {
		expected = expected + `# HELP aggregator_discovery_aggregation_count_total [ALPHA] Counter of number of times discovery was aggregated
# TYPE aggregator_discovery_aggregation_count_total counter
aggregator_discovery_aggregation_count_total %d
`
	}
	args := []any{}
	if aggregationCount > 0 {
		args = append(args, aggregationCount)
	}
	return strings.NewReader(fmt.Sprintf(expected, args...))
}

func TestBasicMetrics(t *testing.T) {
	legacyregistry.Reset()
	manager := discoveryendpoint.NewResourceManager("apis")

	apis := fuzzAPIGroups(1, 3, 10)
	manager.SetGroups(apis.Items)

	interests := []string{"aggregator_discovery_aggregation_count_total"}

	_, _, _ = fetchPath(manager, "application/json", discoveryPath, "")
	// A single fetch should aggregate and increment regeneration counter.
	if err := testutil.GatherAndCompare(legacyregistry.DefaultGatherer, formatExpectedMetrics(1), interests...); err != nil {
		t.Fatal(err)
	}
	_, _, _ = fetchPath(manager, "application/json", discoveryPath, "")
	// Subsequent fetches should not reaggregate discovery.
	if err := testutil.GatherAndCompare(legacyregistry.DefaultGatherer, formatExpectedMetrics(1), interests...); err != nil {
		t.Fatal(err)
	}
}

func TestMetricsModified(t *testing.T) {
	legacyregistry.Reset()
	manager := discoveryendpoint.NewResourceManager("apis")

	apis := fuzzAPIGroups(1, 3, 10)
	manager.SetGroups(apis.Items)

	interests := []string{"aggregator_discovery_aggregation_count_total"}

	_, _, _ = fetchPath(manager, "application/json", discoveryPath, "")
	// A single fetch should aggregate and increment regeneration counter.
	if err := testutil.GatherAndCompare(legacyregistry.DefaultGatherer, formatExpectedMetrics(1), interests...); err != nil {
		t.Fatal(err)
	}

	// Update discovery document.
	manager.SetGroups(fuzzAPIGroups(1, 3, 10).Items)
	_, _, _ = fetchPath(manager, "application/json", discoveryPath, "")
	// If the discovery content has changed, reaggregation should be performed.
	if err := testutil.GatherAndCompare(legacyregistry.DefaultGatherer, formatExpectedMetrics(2), interests...); err != nil {
		t.Fatal(err)
	}
}
