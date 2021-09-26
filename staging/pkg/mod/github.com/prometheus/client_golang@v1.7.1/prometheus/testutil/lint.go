// Copyright 2020 The Prometheus Authors
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package testutil

import (
	"fmt"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/testutil/promlint"
)

// CollectAndLint registers the provided Collector with a newly created pedantic
// Registry. It then calls GatherAndLint with that Registry and with the
// provided metricNames.
func CollectAndLint(c prometheus.Collector, metricNames ...string) ([]promlint.Problem, error) {
	reg := prometheus.NewPedanticRegistry()
	if err := reg.Register(c); err != nil {
		return nil, fmt.Errorf("registering collector failed: %s", err)
	}
	return GatherAndLint(reg, metricNames...)
}

// GatherAndLint gathers all metrics from the provided Gatherer and checks them
// with the linter in the promlint package. If any metricNames are provided,
// only metrics with those names are checked.
func GatherAndLint(g prometheus.Gatherer, metricNames ...string) ([]promlint.Problem, error) {
	got, err := g.Gather()
	if err != nil {
		return nil, fmt.Errorf("gathering metrics failed: %s", err)
	}
	if metricNames != nil {
		got = filterMetrics(got, metricNames)
	}
	return promlint.NewWithMetricFamilies(got).Lint()
}
