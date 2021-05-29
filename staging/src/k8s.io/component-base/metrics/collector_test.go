/*
Copyright 2019 The Kubernetes Authors.

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
	"fmt"
	"reflect"
	"strings"
	"testing"

	"github.com/prometheus/client_golang/prometheus/testutil"
	"github.com/stretchr/testify/assert"

	apimachineryversion "k8s.io/apimachinery/pkg/version"
)

type testCustomCollector struct {
	BaseStableCollector

	descriptors []*Desc
}

func newTestCustomCollector(ds ...*Desc) *testCustomCollector {
	c := &testCustomCollector{}
	c.descriptors = append(c.descriptors, ds...)

	return c
}

func (tc *testCustomCollector) DescribeWithStability(ch chan<- *Desc) {
	for i := range tc.descriptors {
		ch <- tc.descriptors[i]
	}
}

func (tc *testCustomCollector) CollectWithStability(ch chan<- Metric) {
	for i := range tc.descriptors {
		ch <- NewLazyConstMetric(tc.descriptors[i], GaugeValue, 1, "value")
	}
}

func TestBaseCustomCollector(t *testing.T) {
	var currentVersion = apimachineryversion.Info{
		Major:      "1",
		Minor:      "17",
		GitVersion: "v1.17.0-alpha-1.12345",
	}

	var (
		alphaDesc = NewDesc("metric_alpha", "alpha metric", []string{"name"}, nil,
			ALPHA, "")
		stableDesc = NewDesc("metric_stable", "stable metrics", []string{"name"}, nil,
			STABLE, "")
		deprecatedDesc = NewDesc("metric_deprecated", "stable deprecated metrics", []string{"name"}, nil,
			STABLE, "1.17.0")
		hiddenDesc = NewDesc("metric_hidden", "stable hidden metrics", []string{"name"}, nil,
			STABLE, "1.16.0")
	)

	registry := newKubeRegistry(currentVersion)
	customCollector := newTestCustomCollector(alphaDesc, stableDesc, deprecatedDesc, hiddenDesc)

	if err := registry.CustomRegister(customCollector); err != nil {
		t.Fatalf("register collector failed with err: %v", err)
	}

	expectedMetrics := `
        # HELP metric_alpha [ALPHA] alpha metric
        # TYPE metric_alpha gauge
        metric_alpha{name="value"} 1
        # HELP metric_stable [STABLE] stable metrics
        # TYPE metric_stable gauge
        metric_stable{name="value"} 1
        # HELP metric_deprecated [STABLE] (Deprecated since 1.17.0) stable deprecated metrics
        # TYPE metric_deprecated gauge
        metric_deprecated{name="value"} 1
	`

	err := testutil.GatherAndCompare(registry, strings.NewReader(expectedMetrics), alphaDesc.fqName,
		stableDesc.fqName, deprecatedDesc.fqName, hiddenDesc.fqName)
	if err != nil {
		t.Fatal(err)
	}
}

func TestInvalidCustomCollector(t *testing.T) {
	var currentVersion = apimachineryversion.Info{
		Major:      "1",
		Minor:      "17",
		GitVersion: "v1.17.0-alpha-1.12345",
	}
	var namelessDesc = NewDesc("", "this is a nameless metric", nil, nil, ALPHA, "")
	var duplicatedDescA = NewDesc("test_duplicated_metric", "this is a duplicated metric A", nil, nil, ALPHA, "")
	var duplicatedDescB = NewDesc("test_duplicated_metric", "this is a duplicated metric B", nil, nil, ALPHA, "")

	var tests = []struct {
		name        string
		descriptors []*Desc
		panicStr    string
	}{
		{
			name:        "nameless metric will be not allowed",
			descriptors: []*Desc{namelessDesc},
			panicStr:    "nameless metrics will be not allowed",
		},
		{
			name:        "duplicated metric will be not allowed",
			descriptors: []*Desc{duplicatedDescA, duplicatedDescB},
			panicStr:    fmt.Sprintf("duplicate metrics (%s) will be not allowed", duplicatedDescA.fqName),
		},
	}

	for _, test := range tests {
		tc := test
		t.Run(tc.name, func(t *testing.T) {
			registry := newKubeRegistry(currentVersion)
			customCollector := newTestCustomCollector(tc.descriptors...)
			assert.Panics(t, func() {
				registry.CustomMustRegister(customCollector)
			}, tc.panicStr)
		})
	}
}

// TestCustomCollectorClearState guarantees `ClearState()` will fully clear a collector.
// It is necessary because we may forget to clear some new-added fields in the future.
func TestCustomCollectorClearState(t *testing.T) {
	var currentVersion = parseVersion(apimachineryversion.Info{
		Major:      "1",
		Minor:      "17",
		GitVersion: "v1.17.0-alpha-1.12345",
	})

	var (
		alphaDesc = NewDesc("metric_alpha", "alpha metric", []string{"name"}, nil,
			ALPHA, "")
		stableDesc = NewDesc("metric_stable", "stable metrics", []string{"name"}, nil,
			STABLE, "")
		deprecatedDesc = NewDesc("metric_deprecated", "stable deprecated metrics", []string{"name"}, nil,
			STABLE, "1.17.0")
		hiddenDesc = NewDesc("metric_hidden", "stable hidden metrics", []string{"name"}, nil,
			STABLE, "1.16.0")
	)

	benchmarkA := newTestCustomCollector(alphaDesc, stableDesc, deprecatedDesc, hiddenDesc)
	benchmarkB := newTestCustomCollector(alphaDesc, stableDesc, deprecatedDesc, hiddenDesc)

	if benchmarkA.Create(&currentVersion, benchmarkA) == false {
		t.Fatal("collector should be created")
	}
	benchmarkA.ClearState()

	if !reflect.DeepEqual(*benchmarkA, *benchmarkB) {
		t.Fatal("custom collector state hasn't be fully cleared")
	}
}
