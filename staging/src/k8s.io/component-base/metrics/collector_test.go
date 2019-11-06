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
	"testing"

	dto "github.com/prometheus/client_model/go"

	apimachineryversion "k8s.io/apimachinery/pkg/version"
)

type testCustomCollector struct {
	BaseStableCollector
}

var (
	currentVersion = apimachineryversion.Info{
		Major:      "1",
		Minor:      "17",
		GitVersion: "v1.17.0-alpha-1.12345",
	}
	alphaDesc = NewDesc("metric_alpha", "alpha metric", []string{"name"}, nil,
		ALPHA, "")
	stableDesc = NewDesc("metric_stable", "stable metrics", []string{"name"}, nil,
		STABLE, "")
	deprecatedDesc = NewDesc("metric_deprecated", "stable deprecated metrics", []string{"name"}, nil,
		STABLE, "1.17.0")
	hiddenDesc = NewDesc("metric_hidden", "stable hidden metrics", []string{"name"}, nil,
		STABLE, "1.16.0")
)

func (tc *testCustomCollector) DescribeWithStability(ch chan<- *Desc) {
	ch <- alphaDesc
	ch <- stableDesc
	ch <- deprecatedDesc
	ch <- hiddenDesc
}

func (tc *testCustomCollector) CollectWithStability(ch chan<- Metric) {
	ch <- NewLazyConstMetric(
		alphaDesc,
		GaugeValue,
		1,
		"value",
	)
	ch <- NewLazyConstMetric(
		stableDesc,
		GaugeValue,
		1,
		"value",
	)
	ch <- NewLazyConstMetric(
		deprecatedDesc,
		GaugeValue,
		1,
		"value",
	)
	ch <- NewLazyConstMetric(
		hiddenDesc,
		GaugeValue,
		1,
		"value",
	)

}

func getMetric(metrics []*dto.MetricFamily, fqName string) *dto.MetricFamily {
	for _, m := range metrics {
		if *m.Name == fqName {
			return m
		}
	}

	return nil
}

func TestBaseCustomCollector(t *testing.T) {
	var tests = []struct {
		name         string
		d            *Desc
		shouldHidden bool
		expectedHelp string
	}{
		{
			name:         "alpha metric should contains stability metadata",
			d:            alphaDesc,
			shouldHidden: false,
			expectedHelp: "[ALPHA] alpha metric",
		},
		{
			name:         "stable metric should contains stability metadata",
			d:            stableDesc,
			shouldHidden: false,
			expectedHelp: "[STABLE] stable metrics",
		},
		{
			name:         "deprecated metric should contains stability metadata",
			d:            deprecatedDesc,
			shouldHidden: false,
			expectedHelp: "[STABLE] (Deprecated since 1.17.0) stable deprecated metrics",
		},
		{
			name:         "hidden metric should be ignored",
			d:            hiddenDesc,
			shouldHidden: true,
			expectedHelp: "[STABLE] stable hidden metrics",
		},
	}

	registry := newKubeRegistry(currentVersion)
	customCollector := &testCustomCollector{}

	if err := registry.CustomRegister(customCollector); err != nil {
		t.Fatalf("register collector failed with err: %v", err)
	}

	metrics, err := registry.Gather()
	if err != nil {
		t.Fatalf("failed to get metrics from collector, %v", err)
	}

	for _, test := range tests {
		tc := test
		t.Run(tc.name, func(t *testing.T) {
			m := getMetric(metrics, tc.d.fqName)
			if m == nil {
				if !tc.shouldHidden {
					t.Fatalf("Want metric: %s", tc.d.fqName)
				}
			} else {
				if m.GetHelp() != tc.expectedHelp {
					t.Fatalf("Metric(%s) HELP(%s) not contains: %s", tc.d.fqName, *m.Help, tc.expectedHelp)
				}
			}

		})
	}
}
