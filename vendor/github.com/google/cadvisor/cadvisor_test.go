// Copyright 2014 Google Inc. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package main

import (
	"flag"
	"testing"

	"github.com/google/cadvisor/container"
	"github.com/stretchr/testify/assert"
)

func TestTcpMetricsAreDisabledByDefault(t *testing.T) {
	assert.True(t, ignoreMetrics.Has(container.NetworkTcpUsageMetrics))
	flag.Parse()
	assert.True(t, ignoreMetrics.Has(container.NetworkTcpUsageMetrics))
}

func TestUdpMetricsAreDisabledByDefault(t *testing.T) {
	assert.True(t, ignoreMetrics.Has(container.NetworkUdpUsageMetrics))
	flag.Parse()
	assert.True(t, ignoreMetrics.Has(container.NetworkUdpUsageMetrics))
}

func TestIgnoreMetrics(t *testing.T) {
	tests := []struct {
		value    string
		expected []container.MetricKind
	}{
		{"", []container.MetricKind{}},
		{"disk", []container.MetricKind{container.DiskUsageMetrics}},
		{"disk,tcp,network", []container.MetricKind{container.DiskUsageMetrics, container.NetworkTcpUsageMetrics, container.NetworkUsageMetrics}},
	}

	for _, test := range tests {
		assert.NoError(t, ignoreMetrics.Set(test.value))

		assert.Equal(t, len(test.expected), len(ignoreMetrics.MetricSet))
		for _, expected := range test.expected {
			assert.True(t, ignoreMetrics.Has(expected), "Missing %s", expected)
		}
	}
}
