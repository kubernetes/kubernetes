/*
Copyright The Kubernetes Authors.

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
	"strings"
	"testing"

	"k8s.io/component-base/metrics/legacyregistry"
	"k8s.io/component-base/metrics/testutil"
)

func TestIncWebSocketStreamingRequest(t *testing.T) {
	Register()
	ResetForTest()
	t.Cleanup(ResetForTest)

	IncWebSocketStreamingRequest("exec")
	IncWebSocketStreamingRequest("attach")
	IncWebSocketStreamingRequest("portforward")

	expected := `
# HELP kubelet_websocket_streaming_requests_total [ALPHA] Total number of WebSocket streaming requests (exec/attach/portforward) received by the kubelet.
# TYPE kubelet_websocket_streaming_requests_total counter
kubelet_websocket_streaming_requests_total{subresource="attach"} 1
kubelet_websocket_streaming_requests_total{subresource="exec"} 1
kubelet_websocket_streaming_requests_total{subresource="portforward"} 1
`

	if err := testutil.GatherAndCompare(legacyregistry.DefaultGatherer, strings.NewReader(expected),
		"kubelet_websocket_streaming_requests_total"); err != nil {
		t.Errorf("unexpected metric output: %v", err)
	}
}

func TestSetMetricsProvider(t *testing.T) {
	Register()
	MetricsProvider.Reset()

	tests := []struct {
		name     string
		provider MetricsProviderType
		want     string
	}{
		{
			name:     "cadvisor provider",
			provider: CAdvisorMetricsProvider,
			want: `
# HELP kubelet_metrics_provider [ALPHA] Metrics provider used by kubelet to collect container stats. Values can be 'cadvisor' and 'cri'
# TYPE kubelet_metrics_provider gauge
kubelet_metrics_provider{provider="cadvisor"} 1
`,
		},
		{
			name:     "cri provider",
			provider: CRIMetricsProvider,
			want: `
# HELP kubelet_metrics_provider [ALPHA] Metrics provider used by kubelet to collect container stats. Values can be 'cadvisor' and 'cri'
# TYPE kubelet_metrics_provider gauge
kubelet_metrics_provider{provider="cri"} 1
`,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			SetMetricsProvider(tt.provider)

			if err := testutil.GatherAndCompare(legacyregistry.DefaultGatherer, strings.NewReader(tt.want), "kubelet_metrics_provider"); err != nil {
				t.Errorf("unexpected metric output: %v", err)
			}
		})
	}
}
