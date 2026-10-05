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

package rest

import (
	"context"
	"fmt"
	"net/http"
	"reflect"
	"strings"
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/proxy"
	"k8s.io/apiserver/pkg/authorization/authorizer"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	proxymetrics "k8s.io/apiserver/pkg/util/proxy/metrics"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/component-base/metrics/legacyregistry"
	"k8s.io/component-base/metrics/testutil"
	api "k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/kubelet/client"
	pod "k8s.io/kubernetes/pkg/registry/core/pod"
)

type fakeKubeletConn struct {
	features []string
	err      error
}

func (f *fakeKubeletConn) GetConnectionInfo(ctx context.Context, nodeName types.NodeName) (*client.ConnectionInfo, error) {
	if f.err != nil {
		return nil, f.err
	}
	return &client.ConnectionInfo{
		Scheme:       "http",
		Hostname:     "localhost",
		Port:         "1234",
		Transport:    http.DefaultTransport,
		NodeFeatures: f.features,
	}, nil
}

type fakePodGetter struct {
	pod runtime.Object
	err error
}

func (f *fakePodGetter) Get(_ context.Context, _ string, _ *metav1.GetOptions) (runtime.Object, error) {
	return f.pod, f.err
}

type fakeAuthorizer struct{}

func (f *fakeAuthorizer) Authorize(ctx context.Context, a authorizer.Attributes) (authorizer.Decision, string, error) {
	return authorizer.DecisionAllow, "", nil
}

type fakeResponder struct{}

func (f *fakeResponder) Error(err error) {}

func (f *fakeResponder) Object(statusCode int, obj runtime.Object) {}

func TestCheckNodeSupportsWebsockets(t *testing.T) {
	tests := []struct {
		name         string
		nodeFeatures []string
		expected     bool
	}{
		{
			name:         "feature supported",
			nodeFeatures: []string{string(features.ExtendWebSocketsToKubelet)},
			expected:     true,
		},
		{
			name:         "feature not supported",
			nodeFeatures: []string{},
			expected:     false,
		},
		{
			name:         "multiple features, one supported",
			nodeFeatures: []string{"SomeOtherFeature", string(features.ExtendWebSocketsToKubelet)},
			expected:     true,
		},
		{
			name:         "multiple features, none supported",
			nodeFeatures: []string{"SomeOtherFeature", "AnotherFeature"},
			expected:     false,
		},
		{
			name:         "nil features",
			nodeFeatures: nil,
			expected:     false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := checkNodeSupportsWebsockets(tt.nodeFeatures); got != tt.expected {
				t.Errorf("checkNodeSupportsWebsockets() = %v, want %v", got, tt.expected)
			}
		})
	}
}

func TestExecRESTConnect(t *testing.T) {
	testPod := &api.Pod{
		ObjectMeta: metav1.ObjectMeta{Name: "test-pod", Namespace: "default"},
		Spec: api.PodSpec{
			NodeName:   "test-node",
			Containers: []api.Container{{Name: "test-container", Image: "test-image"}},
		},
	}

	tests := []struct {
		name                      string
		store                     pod.ResourceGetter
		kubeletConn               client.ConnectionInfoGetter
		nodeFeatures              []string
		enableWebSocketsKubelet   bool
		enableTranslateWebSockets bool
		expectProxy               bool
		expectErr                 bool
		expectedProxyType         string // empty means no metric expected
	}{
		// Scenario 1: Both feature gates disabled, node does not support websockets.
		// Expect UpgradeAwareHandler as no translation or extension is active.
		{
			name:                      "Kubelet disabled, Translate disabled, Node no websocket",
			store:                     &fakePodGetter{pod: testPod},
			nodeFeatures:              []string{},
			enableWebSocketsKubelet:   false,
			enableTranslateWebSockets: false,
			expectProxy:               true,
		},
		// Scenario 2: ExtendWebSocketsToKubelet enabled, TranslateStreamCloseWebsocketRequests disabled,
		// node supports websockets. Expect UpgradeAwareHandler as translation is disabled.
		{
			name:                      "Kubelet enabled, Translate disabled, Node with websocket",
			store:                     &fakePodGetter{pod: testPod},
			nodeFeatures:              []string{string(features.ExtendWebSocketsToKubelet)},
			enableWebSocketsKubelet:   true,
			enableTranslateWebSockets: false,
			expectProxy:               true,
		},
		// Scenario 3: ExtendWebSocketsToKubelet disabled, TranslateStreamCloseWebsocketRequests enabled,
		// node does not support websockets. Expect translatingHandler as translation is active but extension is not.
		{
			name:                      "Kubelet disabled, Translate enabled, Node no websocket",
			store:                     &fakePodGetter{pod: testPod},
			nodeFeatures:              []string{},
			enableWebSocketsKubelet:   false,
			enableTranslateWebSockets: true,
			expectProxy:               false,
			expectedProxyType:         "translated_at_apiserver",
		},
		// Scenario 4: ExtendWebSocketsToKubelet enabled, TranslateStreamCloseWebsocketRequests enabled,
		// node supports websockets. Expect UpgradeAwareHandler as extension is active and node supports it.
		{
			name:                      "Kubelet enabled, Translate enabled, Node with websocket",
			store:                     &fakePodGetter{pod: testPod},
			nodeFeatures:              []string{string(features.ExtendWebSocketsToKubelet)},
			enableWebSocketsKubelet:   true,
			enableTranslateWebSockets: true,
			expectProxy:               true,
			expectedProxyType:         "proxied_to_kubelet",
		},
		// Scenario 5: ExtendWebSocketsToKubelet enabled, TranslateStreamCloseWebsocketRequests enabled,
		// node does not support websockets. Expect translatingHandler as extension is active but node does not support it.
		{
			name:                      "Kubelet enabled, Translate enabled, Node no websocket",
			store:                     &fakePodGetter{pod: testPod},
			nodeFeatures:              []string{},
			enableWebSocketsKubelet:   true,
			enableTranslateWebSockets: true,
			expectProxy:               false,
			expectedProxyType:         "translated_at_apiserver",
		},
		// Scenario 6: Pod getter returns error; Connect must propagate it.
		{
			name:      "pod getter error",
			store:     &fakePodGetter{err: fmt.Errorf("pod not found")},
			expectErr: true,
		},
		// Scenario 7: ConnectionInfo getter returns error; Connect must propagate it.
		{
			name:        "connection info getter error",
			store:       &fakePodGetter{pod: testPod},
			kubeletConn: &fakeKubeletConn{err: fmt.Errorf("connection error")},
			expectErr:   true,
		},
	}

	proxymetrics.Register()
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			proxymetrics.ResetForTest()
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.ExtendWebSocketsToKubelet, tt.enableWebSocketsKubelet)
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.TranslateStreamCloseWebsocketRequests, tt.enableTranslateWebSockets)

			info := &genericapirequest.RequestInfo{Verb: "create", Name: "test-pod", Namespace: "default"}
			ctx := genericapirequest.WithRequestInfo(context.Background(), info)

			kubeletConn := tt.kubeletConn
			if kubeletConn == nil {
				kubeletConn = &fakeKubeletConn{features: tt.nodeFeatures}
			}
			execRest := &ExecREST{
				Store:       tt.store,
				KubeletConn: kubeletConn,
				Authorizer:  &fakeAuthorizer{},
			}

			handler, err := execRest.Connect(ctx, "test-pod", &api.PodExecOptions{}, &fakeResponder{})
			if tt.expectErr {
				if err == nil {
					t.Error("expected error but got nil")
				}
				return
			}
			if err != nil {
				t.Fatalf("Unexpected error: %v", err)
			}
			if handler == nil {
				t.Fatal("Handler should not be nil")
			}

			if tt.expectProxy {
				if _, ok := handler.(*proxy.UpgradeAwareHandler); !ok {
					t.Errorf("expected handler of type *proxy.UpgradeAwareHandler, got %T", handler)
				}
			} else {
				expected := "*proxy.translatingHandler"
				got := reflect.TypeOf(handler).String()
				if got != expected {
					t.Errorf("expected handler of type %s, got %s", expected, got)
				}
			}

			if tt.expectedProxyType != "" {
				expectedMetric := fmt.Sprintf(`
# HELP apiserver_websocket_streaming_requests_total [ALPHA] Total number of WebSocket streaming requests (exec/attach/portforward) routed by the API server, labeled by subresource and proxy_type. proxy_type is proxied_to_kubelet when the kubelet handles the request directly; otherwise translated_at_apiserver.
# TYPE apiserver_websocket_streaming_requests_total counter
apiserver_websocket_streaming_requests_total{proxy_type=%q,subresource="exec"} 1
`, tt.expectedProxyType)
				if err := testutil.GatherAndCompare(legacyregistry.DefaultGatherer, strings.NewReader(expectedMetric),
					"apiserver_websocket_streaming_requests_total"); err != nil {
					t.Errorf("unexpected metric output: %v", err)
				}
			}
		})
	}
}

func TestAttachRESTConnect(t *testing.T) {
	testPod := &api.Pod{
		ObjectMeta: metav1.ObjectMeta{Name: "test-pod", Namespace: "default"},
		Spec: api.PodSpec{
			NodeName:   "test-node",
			Containers: []api.Container{{Name: "test-container", Image: "test-image"}},
		},
	}

	tests := []struct {
		name                      string
		store                     pod.ResourceGetter
		kubeletConn               client.ConnectionInfoGetter
		nodeFeatures              []string
		enableWebSocketsKubelet   bool
		enableTranslateWebSockets bool
		expectProxy               bool
		expectErr                 bool
		expectedProxyType         string // empty means no metric expected
	}{
		// Scenario 1: Both feature gates disabled, node does not support websockets.
		// Expect UpgradeAwareHandler as no translation or extension is active.
		{
			name:                      "Kubelet disabled, Translate disabled, Node no websocket",
			store:                     &fakePodGetter{pod: testPod},
			nodeFeatures:              []string{},
			enableWebSocketsKubelet:   false,
			enableTranslateWebSockets: false,
			expectProxy:               true,
		},
		// Scenario 2: ExtendWebSocketsToKubelet enabled, TranslateStreamCloseWebsocketRequests disabled,
		// node supports websockets. Expect UpgradeAwareHandler as translation is disabled.
		{
			name:                      "Kubelet enabled, Translate disabled, Node with websocket",
			store:                     &fakePodGetter{pod: testPod},
			nodeFeatures:              []string{string(features.ExtendWebSocketsToKubelet)},
			enableWebSocketsKubelet:   true,
			enableTranslateWebSockets: false,
			expectProxy:               true,
		},
		// Scenario 3: ExtendWebSocketsToKubelet disabled, TranslateStreamCloseWebsocketRequests enabled,
		// node does not support websockets. Expect translatingHandler as translation is active but extension is not.
		{
			name:                      "Kubelet disabled, Translate enabled, Node no websocket",
			store:                     &fakePodGetter{pod: testPod},
			nodeFeatures:              []string{},
			enableWebSocketsKubelet:   false,
			enableTranslateWebSockets: true,
			expectProxy:               false,
			expectedProxyType:         "translated_at_apiserver",
		},
		// Scenario 4: ExtendWebSocketsToKubelet enabled, TranslateStreamCloseWebsocketRequests enabled,
		// node supports websockets. Expect UpgradeAwareHandler as extension is active and node supports it.
		{
			name:                      "Kubelet enabled, Translate enabled, Node with websocket",
			store:                     &fakePodGetter{pod: testPod},
			nodeFeatures:              []string{string(features.ExtendWebSocketsToKubelet)},
			enableWebSocketsKubelet:   true,
			enableTranslateWebSockets: true,
			expectProxy:               true,
			expectedProxyType:         "proxied_to_kubelet",
		},
		// Scenario 5: ExtendWebSocketsToKubelet enabled, TranslateStreamCloseWebsocketRequests enabled,
		// node does not support websockets. Expect translatingHandler as extension is active but node does not support it.
		{
			name:                      "Kubelet enabled, Translate enabled, Node no websocket",
			store:                     &fakePodGetter{pod: testPod},
			nodeFeatures:              []string{},
			enableWebSocketsKubelet:   true,
			enableTranslateWebSockets: true,
			expectProxy:               false,
			expectedProxyType:         "translated_at_apiserver",
		},
		// Scenario 6: Pod getter returns error; Connect must propagate it.
		{
			name:      "pod getter error",
			store:     &fakePodGetter{err: fmt.Errorf("pod not found")},
			expectErr: true,
		},
		// Scenario 7: ConnectionInfo getter returns error; Connect must propagate it.
		{
			name:        "connection info getter error",
			store:       &fakePodGetter{pod: testPod},
			kubeletConn: &fakeKubeletConn{err: fmt.Errorf("connection error")},
			expectErr:   true,
		},
	}

	proxymetrics.Register()
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			proxymetrics.ResetForTest()
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.ExtendWebSocketsToKubelet, tt.enableWebSocketsKubelet)
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.TranslateStreamCloseWebsocketRequests, tt.enableTranslateWebSockets)

			info := &genericapirequest.RequestInfo{Verb: "create", Name: "test-pod", Namespace: "default"}
			ctx := genericapirequest.WithRequestInfo(context.Background(), info)

			kubeletConn := tt.kubeletConn
			if kubeletConn == nil {
				kubeletConn = &fakeKubeletConn{features: tt.nodeFeatures}
			}
			attachRest := &AttachREST{
				Store:       tt.store,
				KubeletConn: kubeletConn,
				Authorizer:  &fakeAuthorizer{},
			}

			handler, err := attachRest.Connect(ctx, "test-pod", &api.PodAttachOptions{}, &fakeResponder{})
			if tt.expectErr {
				if err == nil {
					t.Error("expected error but got nil")
				}
				return
			}
			if err != nil {
				t.Fatalf("Unexpected error: %v", err)
			}
			if handler == nil {
				t.Fatal("Handler should not be nil")
			}

			if tt.expectProxy {
				if _, ok := handler.(*proxy.UpgradeAwareHandler); !ok {
					t.Errorf("expected handler of type *proxy.UpgradeAwareHandler, got %T", handler)
				}
			} else {
				expected := "*proxy.translatingHandler"
				got := reflect.TypeOf(handler).String()
				if got != expected {
					t.Errorf("expected handler of type %s, got %s", expected, got)
				}
			}

			if tt.expectedProxyType != "" {
				expectedMetric := fmt.Sprintf(`
# HELP apiserver_websocket_streaming_requests_total [ALPHA] Total number of WebSocket streaming requests (exec/attach/portforward) routed by the API server, labeled by subresource and proxy_type. proxy_type is proxied_to_kubelet when the kubelet handles the request directly; otherwise translated_at_apiserver.
# TYPE apiserver_websocket_streaming_requests_total counter
apiserver_websocket_streaming_requests_total{proxy_type=%q,subresource="attach"} 1
`, tt.expectedProxyType)
				if err := testutil.GatherAndCompare(legacyregistry.DefaultGatherer, strings.NewReader(expectedMetric),
					"apiserver_websocket_streaming_requests_total"); err != nil {
					t.Errorf("unexpected metric output: %v", err)
				}
			}
		})
	}
}

func TestPortForwardRESTConnect(t *testing.T) {
	testPod := &api.Pod{
		ObjectMeta: metav1.ObjectMeta{Name: "test-pod", Namespace: "default"},
		Spec: api.PodSpec{
			NodeName:   "test-node",
			Containers: []api.Container{{Name: "test-container", Image: "test-image"}},
		},
	}

	tests := []struct {
		name                    string
		store                   pod.ResourceGetter
		kubeletConn             client.ConnectionInfoGetter
		nodeFeatures            []string
		enableWebSocketsKubelet bool
		enablePortForward       bool
		expectProxy             bool
		expectErr               bool
		expectedProxyType       string // empty means no metric expected
	}{
		// Scenario 1: Both feature gates disabled, node does not support websockets.
		// Expect UpgradeAwareHandler as no tunneling or extension is active.
		{
			name:                    "Kubelet disabled, PortForward disabled, Node no websocket",
			store:                   &fakePodGetter{pod: testPod},
			nodeFeatures:            []string{},
			enableWebSocketsKubelet: false,
			enablePortForward:       false,
			expectProxy:             true,
		},
		// Scenario 2: ExtendWebSocketsToKubelet enabled, PortForwardWebsockets disabled,
		// node supports websockets. Expect UpgradeAwareHandler as tunneling is disabled.
		{
			name:                    "Kubelet enabled, PortForward disabled, Node with websocket",
			store:                   &fakePodGetter{pod: testPod},
			nodeFeatures:            []string{string(features.ExtendWebSocketsToKubelet)},
			enableWebSocketsKubelet: true,
			enablePortForward:       false,
			expectProxy:             true,
		},
		// Scenario 3: ExtendWebSocketsToKubelet disabled, PortForwardWebsockets enabled,
		// node does not support websockets. Expect translatingHandler as tunneling is active but extension is not.
		{
			name:                    "Kubelet disabled, PortForward enabled, Node no websocket",
			store:                   &fakePodGetter{pod: testPod},
			nodeFeatures:            []string{},
			enableWebSocketsKubelet: false,
			enablePortForward:       true,
			expectProxy:             false,
			expectedProxyType:       "translated_at_apiserver",
		},
		// Scenario 4: ExtendWebSocketsToKubelet enabled, PortForwardWebsockets enabled,
		// node supports websockets. Expect UpgradeAwareHandler as extension is active and node supports it.
		{
			name:                    "Kubelet enabled, PortForward enabled, Node with websocket",
			store:                   &fakePodGetter{pod: testPod},
			nodeFeatures:            []string{string(features.ExtendWebSocketsToKubelet)},
			enableWebSocketsKubelet: true,
			enablePortForward:       true,
			expectProxy:             true,
			expectedProxyType:       "proxied_to_kubelet",
		},
		// Scenario 5: ExtendWebSocketsToKubelet enabled, PortForwardWebsockets enabled,
		// node does not support websockets. Expect translatingHandler as extension is active but node does not support it.
		{
			name:                    "Kubelet enabled, PortForward enabled, Node no websocket",
			store:                   &fakePodGetter{pod: testPod},
			nodeFeatures:            []string{},
			enableWebSocketsKubelet: true,
			enablePortForward:       true,
			expectProxy:             false,
			expectedProxyType:       "translated_at_apiserver",
		},
		// Scenario 6: Pod getter returns error; Connect must propagate it.
		{
			name:      "pod getter error",
			store:     &fakePodGetter{err: fmt.Errorf("pod not found")},
			expectErr: true,
		},
		// Scenario 7: ConnectionInfo getter returns error; Connect must propagate it.
		{
			name:        "connection info getter error",
			store:       &fakePodGetter{pod: testPod},
			kubeletConn: &fakeKubeletConn{err: fmt.Errorf("connection error")},
			expectErr:   true,
		},
	}

	proxymetrics.Register()
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			proxymetrics.ResetForTest()
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.ExtendWebSocketsToKubelet, tt.enableWebSocketsKubelet)
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.PortForwardWebsockets, tt.enablePortForward)

			info := &genericapirequest.RequestInfo{Verb: "create", Name: "test-pod", Namespace: "default"}
			ctx := genericapirequest.WithRequestInfo(context.Background(), info)

			kubeletConn := tt.kubeletConn
			if kubeletConn == nil {
				kubeletConn = &fakeKubeletConn{features: tt.nodeFeatures}
			}
			portForwardRest := &PortForwardREST{
				Store:       tt.store,
				KubeletConn: kubeletConn,
				Authorizer:  &fakeAuthorizer{},
			}

			handler, err := portForwardRest.Connect(ctx, "test-pod", &api.PodPortForwardOptions{}, &fakeResponder{})
			if tt.expectErr {
				if err == nil {
					t.Error("expected error but got nil")
				}
				return
			}
			if err != nil {
				t.Fatalf("Unexpected error: %v", err)
			}
			if handler == nil {
				t.Fatal("Handler should not be nil")
			}

			if tt.expectProxy {
				if _, ok := handler.(*proxy.UpgradeAwareHandler); !ok {
					t.Errorf("expected handler of type *proxy.UpgradeAwareHandler, got %T", handler)
				}
			} else {
				expected := "*proxy.translatingHandler"
				got := reflect.TypeOf(handler).String()
				if got != expected {
					t.Errorf("expected handler of type %s, got %s", expected, got)
				}
			}

			if tt.expectedProxyType != "" {
				expectedMetric := fmt.Sprintf(`
# HELP apiserver_websocket_streaming_requests_total [ALPHA] Total number of WebSocket streaming requests (exec/attach/portforward) routed by the API server, labeled by subresource and proxy_type. proxy_type is proxied_to_kubelet when the kubelet handles the request directly; otherwise translated_at_apiserver.
# TYPE apiserver_websocket_streaming_requests_total counter
apiserver_websocket_streaming_requests_total{proxy_type=%q,subresource="portforward"} 1
`, tt.expectedProxyType)
				if err := testutil.GatherAndCompare(legacyregistry.DefaultGatherer, strings.NewReader(expectedMetric),
					"apiserver_websocket_streaming_requests_total"); err != nil {
					t.Errorf("unexpected metric output: %v", err)
				}
			}
		})
	}
}
