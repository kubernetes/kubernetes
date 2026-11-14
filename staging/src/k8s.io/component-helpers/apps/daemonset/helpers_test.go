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

package daemonset

import (
	"testing"

	appsv1 "k8s.io/api/apps/v1"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/component-helpers/scheduling/corev1/nodeaffinity"
	"k8s.io/klog/v2/ktesting"
)

func requiredAffinity(terms ...v1.NodeSelectorTerm) *v1.Affinity {
	return &v1.Affinity{NodeAffinity: &v1.NodeAffinity{
		RequiredDuringSchedulingIgnoredDuringExecution: &v1.NodeSelector{NodeSelectorTerms: terms},
	}}
}

func TestNodeShouldRunDaemonPodWithPrecomputed(t *testing.T) {
	gpuNoSchedule := v1.Taint{Key: "dedicated", Value: "gpu", Effect: v1.TaintEffectNoSchedule}
	gpuNoExecute := v1.Taint{Key: "dedicated", Value: "gpu", Effect: v1.TaintEffectNoExecute}
	tierNoExecute := v1.Taint{Key: "tier", Value: "5", Effect: v1.TaintEffectNoExecute}
	tierGreaterThan3 := v1.Toleration{Key: "tier", Operator: v1.TolerationOpGt, Value: "3"}

	testCases := []struct {
		name                      string
		templateSpec              v1.PodSpec
		nodeLabels                map[string]string
		nodeTaints                []v1.Taint
		enableComparisonOperators bool
		wantShouldRun             bool
		wantShouldContinueRunning bool
	}{
		{
			name:                      "no constraints",
			wantShouldRun:             true,
			wantShouldContinueRunning: true,
		},
		{
			name:                      "node name matches",
			templateSpec:              v1.PodSpec{NodeName: "node"},
			wantShouldRun:             true,
			wantShouldContinueRunning: true,
		},
		{
			name:                      "node name does not match",
			templateSpec:              v1.PodSpec{NodeName: "other-node"},
			wantShouldRun:             false,
			wantShouldContinueRunning: false,
		},
		{
			name:                      "node selector matches",
			templateSpec:              v1.PodSpec{NodeSelector: map[string]string{"pool": "a"}},
			nodeLabels:                map[string]string{"pool": "a"},
			wantShouldRun:             true,
			wantShouldContinueRunning: true,
		},
		{
			name:                      "node selector does not match",
			templateSpec:              v1.PodSpec{NodeSelector: map[string]string{"pool": "a"}},
			nodeLabels:                map[string]string{"pool": "b"},
			wantShouldRun:             false,
			wantShouldContinueRunning: false,
		},
		{
			name: "affinity matchExpressions match",
			templateSpec: v1.PodSpec{Affinity: requiredAffinity(v1.NodeSelectorTerm{
				MatchExpressions: []v1.NodeSelectorRequirement{{Key: "pool", Operator: v1.NodeSelectorOpIn, Values: []string{"a", "b"}}},
			})},
			nodeLabels:                map[string]string{"pool": "b"},
			wantShouldRun:             true,
			wantShouldContinueRunning: true,
		},
		{
			name: "affinity matchExpressions do not match",
			templateSpec: v1.PodSpec{Affinity: requiredAffinity(v1.NodeSelectorTerm{
				MatchExpressions: []v1.NodeSelectorRequirement{{Key: "pool", Operator: v1.NodeSelectorOpNotIn, Values: []string{"a"}}},
			})},
			nodeLabels:                map[string]string{"pool": "a"},
			wantShouldRun:             false,
			wantShouldContinueRunning: false,
		},
		{
			name: "affinity matchFields on node name match",
			templateSpec: v1.PodSpec{Affinity: requiredAffinity(v1.NodeSelectorTerm{
				MatchFields: []v1.NodeSelectorRequirement{{Key: metav1.ObjectNameField, Operator: v1.NodeSelectorOpIn, Values: []string{"node"}}},
			})},
			wantShouldRun:             true,
			wantShouldContinueRunning: true,
		},
		{
			name: "affinity matchFields on node name do not match",
			templateSpec: v1.PodSpec{Affinity: requiredAffinity(v1.NodeSelectorTerm{
				MatchFields: []v1.NodeSelectorRequirement{{Key: metav1.ObjectNameField, Operator: v1.NodeSelectorOpIn, Values: []string{"other-node"}}},
			})},
			wantShouldRun:             false,
			wantShouldContinueRunning: false,
		},
		{
			name: "affinity term that fails to parse does not match",
			templateSpec: v1.PodSpec{Affinity: requiredAffinity(v1.NodeSelectorTerm{
				MatchExpressions: []v1.NodeSelectorRequirement{{Key: "pool", Operator: "Invalid", Values: []string{"a"}}},
			})},
			nodeLabels:                map[string]string{"pool": "a"},
			wantShouldRun:             false,
			wantShouldContinueRunning: false,
		},
		{
			name: "affinity term that fails to parse is ignored when another term matches",
			templateSpec: v1.PodSpec{Affinity: requiredAffinity(
				v1.NodeSelectorTerm{MatchExpressions: []v1.NodeSelectorRequirement{{Key: "pool", Operator: "Invalid", Values: []string{"a"}}}},
				v1.NodeSelectorTerm{MatchExpressions: []v1.NodeSelectorRequirement{{Key: "pool", Operator: v1.NodeSelectorOpExists}}},
			)},
			nodeLabels:                map[string]string{"pool": "a"},
			wantShouldRun:             true,
			wantShouldContinueRunning: true,
		},
		{
			name:                      "untolerated NoSchedule taint keeps running pods",
			nodeTaints:                []v1.Taint{gpuNoSchedule},
			wantShouldRun:             false,
			wantShouldContinueRunning: true,
		},
		{
			name:                      "untolerated NoExecute taint evicts running pods",
			nodeTaints:                []v1.Taint{gpuNoExecute},
			wantShouldRun:             false,
			wantShouldContinueRunning: false,
		},
		{
			name:                      "tolerated NoExecute taint",
			templateSpec:              v1.PodSpec{Tolerations: []v1.Toleration{{Key: "dedicated", Operator: v1.TolerationOpExists}}},
			nodeTaints:                []v1.Taint{gpuNoExecute},
			wantShouldRun:             true,
			wantShouldContinueRunning: true,
		},
		{
			name:                      "untolerated NoSchedule taint with tolerated NoExecute taint keeps running pods",
			templateSpec:              v1.PodSpec{Tolerations: []v1.Toleration{{Key: "dedicated", Operator: v1.TolerationOpExists, Effect: v1.TaintEffectNoExecute}}},
			nodeTaints:                []v1.Taint{gpuNoSchedule, gpuNoExecute},
			wantShouldRun:             false,
			wantShouldContinueRunning: true,
		},
		{
			name:                      "PreferNoSchedule taint is ignored",
			nodeTaints:                []v1.Taint{{Key: "dedicated", Value: "gpu", Effect: v1.TaintEffectPreferNoSchedule}},
			wantShouldRun:             true,
			wantShouldContinueRunning: true,
		},
		{
			name:                      "not-ready taint without the default daemon pod tolerations",
			nodeTaints:                []v1.Taint{{Key: v1.TaintNodeNotReady, Effect: v1.TaintEffectNoExecute}},
			wantShouldRun:             false,
			wantShouldContinueRunning: false,
		},
		{
			name:                      "not-ready taint with the default daemon pod toleration",
			templateSpec:              v1.PodSpec{Tolerations: []v1.Toleration{{Key: v1.TaintNodeNotReady, Operator: v1.TolerationOpExists, Effect: v1.TaintEffectNoExecute}}},
			nodeTaints:                []v1.Taint{{Key: v1.TaintNodeNotReady, Effect: v1.TaintEffectNoExecute}},
			wantShouldRun:             true,
			wantShouldContinueRunning: true,
		},
		{
			name:                      "Gt toleration is ignored when comparison operators are disabled",
			templateSpec:              v1.PodSpec{Tolerations: []v1.Toleration{tierGreaterThan3}},
			nodeTaints:                []v1.Taint{tierNoExecute},
			enableComparisonOperators: false,
			wantShouldRun:             false,
			wantShouldContinueRunning: false,
		},
		{
			name:                      "Gt toleration tolerates a larger taint value when comparison operators are enabled",
			templateSpec:              v1.PodSpec{Tolerations: []v1.Toleration{tierGreaterThan3}},
			nodeTaints:                []v1.Taint{tierNoExecute},
			enableComparisonOperators: true,
			wantShouldRun:             true,
			wantShouldContinueRunning: true,
		},
		{
			name:                      "Lt toleration does not tolerate a larger taint value",
			templateSpec:              v1.PodSpec{Tolerations: []v1.Toleration{{Key: "tier", Operator: v1.TolerationOpLt, Value: "3"}}},
			nodeTaints:                []v1.Taint{tierNoExecute},
			enableComparisonOperators: true,
			wantShouldRun:             false,
			wantShouldContinueRunning: false,
		},
		{
			// The untolerated NoSchedule taint fails the first check, so the result depends on the
			// second, NoExecute-only check also honoring enableComparisonOperators.
			name:                      "Gt toleration keeps running pods when comparison operators are enabled",
			templateSpec:              v1.PodSpec{Tolerations: []v1.Toleration{tierGreaterThan3}},
			nodeTaints:                []v1.Taint{gpuNoSchedule, tierNoExecute},
			enableComparisonOperators: true,
			wantShouldRun:             false,
			wantShouldContinueRunning: true,
		},
		{
			name:                      "Gt toleration does not keep running pods when comparison operators are disabled",
			templateSpec:              v1.PodSpec{Tolerations: []v1.Toleration{tierGreaterThan3}},
			nodeTaints:                []v1.Taint{gpuNoSchedule, tierNoExecute},
			enableComparisonOperators: false,
			wantShouldRun:             false,
			wantShouldContinueRunning: false,
		},
	}
	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			logger, _ := ktesting.NewTestContext(t)
			ds := &appsv1.DaemonSet{
				ObjectMeta: metav1.ObjectMeta{Name: "ds", Namespace: metav1.NamespaceDefault},
				Spec:       appsv1.DaemonSetSpec{Template: v1.PodTemplateSpec{Spec: tc.templateSpec}},
			}
			node := &v1.Node{
				ObjectMeta: metav1.ObjectMeta{Name: "node", Labels: tc.nodeLabels},
				Spec:       v1.NodeSpec{Taints: tc.nodeTaints},
			}
			requiredNodeAffinity := nodeaffinity.NewRequiredNodeAffinity(tc.templateSpec.NodeSelector, tc.templateSpec.Affinity)

			shouldRun, shouldContinueRunning := NodeShouldRunDaemonPodWithPrecomputed(logger, node, ds, tc.templateSpec.Tolerations, requiredNodeAffinity, tc.enableComparisonOperators)

			if shouldRun != tc.wantShouldRun {
				t.Errorf("expected shouldRun: %v, got %v", tc.wantShouldRun, shouldRun)
			}
			if shouldContinueRunning != tc.wantShouldContinueRunning {
				t.Errorf("expected shouldContinueRunning: %v, got %v", tc.wantShouldContinueRunning, shouldContinueRunning)
			}
		})
	}
}
