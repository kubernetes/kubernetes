package nodeselectoradjuster

import (
	"context"
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apiserver/pkg/admission"
	"k8s.io/apiserver/pkg/authentication/user"
	coreapi "k8s.io/kubernetes/pkg/apis/core"
)

func TestAdmit(t *testing.T) {
	tests := []struct {
		name                 string
		pod                  *coreapi.Pod
		resource             schema.GroupVersionResource
		subresource          string
		expectedNodeSelector map[string]string
	}{
		{
			name: "VPA operator pod: control-plane node selector is added",
			pod: makePod(
				withNamespace(vpaOperatorNamespace),
				withLabels(map[string]string{vpaOperatorLabelKey: vpaOperatorLabelValue}),
			),
			resource:             coreapi.Resource("pods").WithVersion("v1"),
			expectedNodeSelector: map[string]string{controlPlaneRoleKey: ""},
		},
		{
			name: "VPA operator pod: control-plane node selector added alongside existing selectors",
			pod: makePod(
				withNamespace(vpaOperatorNamespace),
				withLabels(map[string]string{vpaOperatorLabelKey: vpaOperatorLabelValue}),
				withNodeSelector(map[string]string{"topology.kubernetes.io/zone": "us-east-1a"}),
			),
			resource: coreapi.Resource("pods").WithVersion("v1"),
			expectedNodeSelector: map[string]string{
				controlPlaneRoleKey:           "",
				"topology.kubernetes.io/zone": "us-east-1a",
			},
		},
		{
			name: "VPA operator pod: control-plane node selector already present is left unchanged",
			pod: makePod(
				withNamespace(vpaOperatorNamespace),
				withLabels(map[string]string{vpaOperatorLabelKey: vpaOperatorLabelValue}),
				withNodeSelector(map[string]string{controlPlaneRoleKey: ""}),
			),
			resource:             coreapi.Resource("pods").WithVersion("v1"),
			expectedNodeSelector: map[string]string{controlPlaneRoleKey: ""},
		},
		{
			name: "non-qualifying pod: not modified",
			pod: makePod(
				withLabels(map[string]string{"app": "some-other-app"}),
			),
			resource:             coreapi.Resource("pods").WithVersion("v1"),
			expectedNodeSelector: nil,
		},
		{
			name:                 "pod with no labels: not modified",
			pod:                  makePod(),
			resource:             coreapi.Resource("pods").WithVersion("v1"),
			expectedNodeSelector: nil,
		},
		{
			name: "VPA operator label with wrong value: not modified",
			pod: makePod(
				withNamespace(vpaOperatorNamespace),
				withLabels(map[string]string{vpaOperatorLabelKey: "some-other-operator"}),
			),
			resource:             coreapi.Resource("pods").WithVersion("v1"),
			expectedNodeSelector: nil,
		},
		{
			name: "VPA operator label in wrong namespace: not modified",
			pod: makePod(
				withNamespace("other-namespace"),
				withLabels(map[string]string{vpaOperatorLabelKey: vpaOperatorLabelValue}),
			),
			resource:             coreapi.Resource("pods").WithVersion("v1"),
			expectedNodeSelector: nil,
		},
		{
			name: "non-pod resource: request is ignored",
			pod: makePod(
				withLabels(map[string]string{vpaOperatorLabelKey: vpaOperatorLabelValue}),
			),
			resource:             coreapi.Resource("nodes").WithVersion("v1"),
			expectedNodeSelector: nil,
		},
		{
			name: "pod subresource: request is ignored",
			pod: makePod(
				withLabels(map[string]string{vpaOperatorLabelKey: vpaOperatorLabelValue}),
			),
			resource:             coreapi.Resource("pods").WithVersion("v1"),
			subresource:          "exec",
			expectedNodeSelector: nil,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			plugin := &nodeSelectorAdjuster{
				Handler: admission.NewHandler(admission.Create),
			}

			attrs := admission.NewAttributesRecord(
				tc.pod,
				nil,
				schema.GroupVersionKind{},
				tc.pod.Namespace,
				tc.pod.Name,
				tc.resource,
				tc.subresource,
				admission.Create,
				nil,
				false,
				fakeUser(),
			)

			if err := plugin.Admit(context.TODO(), attrs, nil); err != nil {
				t.Fatalf("unexpected error: %v", err)
			}

			// Verify node selector.
			gotSelector := tc.pod.Spec.NodeSelector
			if len(gotSelector) != len(tc.expectedNodeSelector) {
				t.Errorf("node selector: expected %v, got %v", tc.expectedNodeSelector, gotSelector)
			} else {
				for k, v := range tc.expectedNodeSelector {
					if gotSelector[k] != v {
						t.Errorf("node selector key %q: expected %q, got %q", k, v, gotSelector[k])
					}
				}
			}
		})
	}
}

func TestRequiresNodeSelectorAdjustment(t *testing.T) {
	tests := []struct {
		name     string
		pod      *coreapi.Pod
		expected bool
	}{
		{
			name:     "VPA operator label in correct namespace: match",
			pod:      makePod(withNamespace(vpaOperatorNamespace), withLabels(map[string]string{vpaOperatorLabelKey: vpaOperatorLabelValue})),
			expected: true,
		},
		{
			name:     "VPA operator label in wrong namespace: no match",
			pod:      makePod(withNamespace("other-namespace"), withLabels(map[string]string{vpaOperatorLabelKey: vpaOperatorLabelValue})),
			expected: false,
		},
		{
			name:     "no labels: no match",
			pod:      makePod(),
			expected: false,
		},
		{
			name:     "unrelated labels: no match",
			pod:      makePod(withNamespace(vpaOperatorNamespace), withLabels(map[string]string{"app": "foo", "version": "v1"})),
			expected: false,
		},
		{
			name:     "VPA operator label with wrong value: no match",
			pod:      makePod(withNamespace(vpaOperatorNamespace), withLabels(map[string]string{vpaOperatorLabelKey: "some-other-operator"})),
			expected: false,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := requiresNodeSelectorAdjustment(tc.pod)
			if got != tc.expected {
				t.Errorf("expected %v, got %v", tc.expected, got)
			}
		})
	}
}

func TestIsStandalone(t *testing.T) {
	tests := []struct {
		name     string
		envValue string
		expected bool
	}{
		{
			name:     "env var set to 'openshift-kube-apiserver': IsStandalone returns true",
			envValue: "openshift-kube-apiserver",
			expected: true,
		},
		{
			name:     "env var set to empty string: IsStandalone returns false",
			envValue: "",
			expected: false,
		},
		{
			name:     "env var set to another namespace: IsStandalone returns false",
			envValue: "clusters-my-cluster",
			expected: false,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Setenv(standaloneEnvVar, tc.envValue)
			got := IsStandalone()
			if got != tc.expected {
				t.Errorf("expected %v, got %v", tc.expected, got)
			}
		})
	}
}

// makePod constructs a coreapi.Pod, applying each option in order.
func makePod(opts ...func(*coreapi.Pod)) *coreapi.Pod {
	p := &coreapi.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test-pod",
			Namespace: "test-namespace",
		},
	}
	for _, opt := range opts {
		opt(p)
	}
	return p
}

func withNamespace(ns string) func(*coreapi.Pod) {
	return func(p *coreapi.Pod) {
		p.Namespace = ns
	}
}

func withLabels(labels map[string]string) func(*coreapi.Pod) {
	return func(p *coreapi.Pod) {
		p.Labels = labels
	}
}

func withNodeSelector(selector map[string]string) func(*coreapi.Pod) {
	return func(p *coreapi.Pod) {
		p.Spec.NodeSelector = selector
	}
}

func fakeUser() user.Info {
	return &user.DefaultInfo{Name: "testuser"}
}
