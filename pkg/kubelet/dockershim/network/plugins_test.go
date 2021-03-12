// +build !dockerless

/*
Copyright 2020 The Kubernetes Authors.

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

package network

import (
	"strings"
	"testing"
	"time"

	"k8s.io/component-base/metrics/legacyregistry"
	"k8s.io/component-base/metrics/testutil"
	"k8s.io/kubernetes/pkg/kubelet/dockershim/network/metrics"
)

func TestNetworkPluginManagerMetrics(t *testing.T) {
	metrics.Register()

	operation := "test_operation"
	recordOperation(operation, time.Now())
	recordError(operation)

	cases := []struct {
		metricName string
		want       string
	}{
		{
			metricName: "kubelet_network_plugin_operations_total",
			want: `
# HELP kubelet_network_plugin_operations_total [ALPHA] Cumulative number of network plugin operations by operation type.
# TYPE kubelet_network_plugin_operations_total counter
kubelet_network_plugin_operations_total{operation_type="test_operation"} 1
`,
		},
		{
			metricName: "kubelet_network_plugin_operations_errors_total",
			want: `
# HELP kubelet_network_plugin_operations_errors_total [ALPHA] Cumulative number of network plugin operation errors by operation type.
# TYPE kubelet_network_plugin_operations_errors_total counter
kubelet_network_plugin_operations_errors_total{operation_type="test_operation"} 1
`,
		},
	}

	for _, tc := range cases {
		t.Run(tc.metricName, func(t *testing.T) {
			if err := testutil.GatherAndCompare(legacyregistry.DefaultGatherer, strings.NewReader(tc.want), tc.metricName); err != nil {
				t.Fatal(err)
			}
		})
	}
}
