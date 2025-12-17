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
	"context"
	"strings"
	"testing"

	"k8s.io/component-base/metrics/legacyregistry"
	"k8s.io/component-base/metrics/testutil"
)

func TestIncWebSocketStreamingRequest(t *testing.T) {
	Register()
	ResetForTest()
	t.Cleanup(ResetForTest)

	ctx := context.Background()
	IncWebSocketStreamingRequest(ctx, "exec", "proxied_to_kubelet")
	IncWebSocketStreamingRequest(ctx, "exec", "translated_at_apiserver")
	IncWebSocketStreamingRequest(ctx, "portforward", "tunneled_at_apiserver")

	expected := `
# HELP apiserver_websocket_streaming_requests_total [ALPHA] Total number of streaming requests (exec/attach/portforward) routed by the API server, labeled by subresource and proxy_type. proxy_type is proxied_to_kubelet when ExtendWebSocketsToKubelet is enabled and the kubelet advertises support; otherwise it is translated_at_apiserver (exec/attach) or tunneled_at_apiserver (portforward).
# TYPE apiserver_websocket_streaming_requests_total counter
apiserver_websocket_streaming_requests_total{proxy_type="proxied_to_kubelet",subresource="exec"} 1
apiserver_websocket_streaming_requests_total{proxy_type="translated_at_apiserver",subresource="exec"} 1
apiserver_websocket_streaming_requests_total{proxy_type="tunneled_at_apiserver",subresource="portforward"} 1
`

	if err := testutil.GatherAndCompare(legacyregistry.DefaultGatherer, strings.NewReader(expected),
		"apiserver_websocket_streaming_requests_total"); err != nil {
		t.Errorf("unexpected metric output: %v", err)
	}
}
