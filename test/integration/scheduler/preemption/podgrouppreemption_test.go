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

package preemption

import (
	"context"
	"errors"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"

	v1 "k8s.io/api/core/v1"
	policyv1 "k8s.io/api/policy/v1"
	schedulingv1alpha3 "k8s.io/api/scheduling/v1alpha3"
	schedulingv1beta1 "k8s.io/api/scheduling/v1beta1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	apimeta "k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	types "k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/intstr"
	"k8s.io/apimachinery/pkg/util/wait"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	clientset "k8s.io/client-go/kubernetes"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	configv1 "k8s.io/kube-scheduler/config/v1"
	fwk "k8s.io/kube-scheduler/framework"
	podutil "k8s.io/kubernetes/pkg/api/v1/pod"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/scheduler"
	config "k8s.io/kubernetes/pkg/scheduler/apis/config"
	configtesting "k8s.io/kubernetes/pkg/scheduler/apis/config/testing"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/defaultbinder"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/names"
	frameworkruntime "k8s.io/kubernetes/pkg/scheduler/framework/runtime"
	st "k8s.io/kubernetes/pkg/scheduler/testing"
	testutils "k8s.io/kubernetes/test/integration/util"
	"k8s.io/utils/ptr"
)

// TestPodGroupPreemption tests preemption scenarios involving pod groups.
func TestPodGroupPreemption(t *testing.T) {
	featuregatetesting.SetFeatureGatesDuringTest(t, utilfeature.DefaultFeatureGate, featuregatetesting.FeatureOverrides{
		features.GenericWorkload:   true,
		features.PodLevelResources: true,
	})
	tests := []struct {
		name          string
		nodes         []*v1.Node
		podGroups     []*schedulingv1beta1.PodGroup
		initialPods   []*v1.Pod // pods that should be scheduled before preemption starts
		preemptorPods []*v1.Pod // pods that belong to a group and should trigger preemption
		// the order may be important to ensure deterministic scheduling result, where only some of the preemptor pods will get scheduled.
		preemptorPodsQueuedInCreationOrder bool
		pdb                                *policyv1.PodDisruptionBudget
		expectedScheduled                  []string
		expectedPreempted                  []string
		expectedUnschedulable              []string
		expectedToHaveNNNInfo              []string
		expectedPodsPreemptedByWAP         int
		enablePodGroupPreemptionPolicy     bool
		customPluginName                   string
		customPluginFunc                   frameworkruntime.PluginFactory
		// tempRemovePG, if true, temporarily removes PodGroups for the time of creating preemptor pods
		// - but after initial pods have been scheduled.
		// This ensures that the initial pods get scheduled before preemptor pods are created AND
		// all preemptor pods are created and kept in incompletePodGroupPods.
		// Once the PodGroup is recreated, all pods become schedulable simultaneously and
		// are guaranteed to be evaluated together in the next PodGroup scheduling cycle.
		// This avoids test flakiness caused by running multiple PodGroup scheduling cycles with a partial set of preemptor pods.
		tempRemovePG       bool
		expectedEventOrder []string
	}{
		{
			name: "Full PodGroup Preemption",
			nodes: []*v1.Node{
				st.MakeNode().Name("node1").Capacity(map[v1.ResourceName]string{v1.ResourceCPU: "3", v1.ResourceMemory: "4Gi", v1.ResourcePods: "32"}).Obj(),
			},
			podGroups: []*schedulingv1beta1.PodGroup{
				st.MakePodGroup().Name("pg1").Namespace("default").Priority(100).MinCount(3).Obj(),
			},
			initialPods: []*v1.Pod{
				st.MakePod().Name("low-1").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").ZeroTerminationGracePeriod().Priority(10).Obj(),
				st.MakePod().Name("low-2").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").ZeroTerminationGracePeriod().Priority(10).Obj(),
				st.MakePod().Name("low-3").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").ZeroTerminationGracePeriod().Priority(10).Obj(),
			},
			preemptorPods: []*v1.Pod{
				st.MakePod().Name("high-1").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("pg1").ZeroTerminationGracePeriod().Priority(100).Obj(),
				st.MakePod().Name("high-2").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("pg1").ZeroTerminationGracePeriod().Priority(100).Obj(),
				st.MakePod().Name("high-3").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("pg1").ZeroTerminationGracePeriod().Priority(100).Obj(),
			},
			expectedScheduled:          []string{"high-1", "high-2", "high-3"},
			expectedPreempted:          []string{"low-1", "low-2", "low-3"},
			expectedToHaveNNNInfo:      []string{"high-1", "high-2", "high-3"},
			expectedPodsPreemptedByWAP: 3,
		},
		{
			name: "Full PodGroup Preemption for basic policy",
			nodes: []*v1.Node{
				st.MakeNode().Name("node1").Capacity(map[v1.ResourceName]string{v1.ResourceCPU: "3", v1.ResourceMemory: "4Gi", v1.ResourcePods: "32"}).Obj(),
			},
			podGroups: []*schedulingv1beta1.PodGroup{
				st.MakePodGroup().Name("pg1").Namespace("default").Priority(100).BasicPolicy().Obj(),
			},
			initialPods: []*v1.Pod{
				st.MakePod().Name("low-1").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").ZeroTerminationGracePeriod().Priority(10).Obj(),
				st.MakePod().Name("low-2").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").ZeroTerminationGracePeriod().Priority(10).Obj(),
				st.MakePod().Name("low-3").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").ZeroTerminationGracePeriod().Priority(10).Obj(),
			},
			preemptorPods: []*v1.Pod{
				st.MakePod().Name("high-1").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("pg1").ZeroTerminationGracePeriod().Priority(100).Obj(),
				st.MakePod().Name("high-2").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("pg1").ZeroTerminationGracePeriod().Priority(100).Obj(),
				st.MakePod().Name("high-3").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("pg1").ZeroTerminationGracePeriod().Priority(100).Obj(),
			},
			expectedScheduled:          []string{"high-1", "high-2", "high-3"},
			expectedPreempted:          []string{"low-1", "low-2", "low-3"},
			expectedToHaveNNNInfo:      []string{"high-1", "high-2", "high-3"},
			expectedPodsPreemptedByWAP: 3,
		},
		{
			name: "Partial Preemption",
			nodes: []*v1.Node{
				st.MakeNode().Name("node1").Capacity(map[v1.ResourceName]string{v1.ResourceCPU: "2", v1.ResourceMemory: "4Gi", v1.ResourcePods: "32"}).Obj(),
				st.MakeNode().Name("node2").Capacity(map[v1.ResourceName]string{v1.ResourceCPU: "2", v1.ResourceMemory: "4Gi", v1.ResourcePods: "32"}).Obj(),
			},
			podGroups: []*schedulingv1beta1.PodGroup{
				st.MakePodGroup().Name("pg1").Namespace("default").Priority(100).MinCount(3).Obj(),
			},
			initialPods: []*v1.Pod{
				// low-1 takes all CPU on node1
				st.MakePod().Name("low-1").Req(map[v1.ResourceName]string{v1.ResourceCPU: "2"}).Container("image").ZeroTerminationGracePeriod().Priority(10).Node("node1").Obj(),
				// low-2 takes half CPU on node2
				st.MakePod().Name("low-2").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").ZeroTerminationGracePeriod().Priority(10).Node("node2").Obj(),
			},
			preemptorPods: []*v1.Pod{
				st.MakePod().Name("high-1").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("pg1").ZeroTerminationGracePeriod().Priority(100).Obj(),
				st.MakePod().Name("high-2").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("pg1").ZeroTerminationGracePeriod().Priority(100).Obj(),
				st.MakePod().Name("high-3").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("pg1").ZeroTerminationGracePeriod().Priority(100).Obj(),
			},
			// With custom scoring, preemptor pods will prefer on high-1 node
			// which will force preemption of low-1 pod.
			expectedScheduled:          []string{"high-1", "high-2", "high-3", "low-2"},
			expectedPreempted:          []string{"low-1"},
			expectedToHaveNNNInfo:      []string{"high-1", "high-2", "high-3"},
			expectedPodsPreemptedByWAP: 1,
			customPluginName:           "mockScorePlugin",
			customPluginFunc:           newPresetScorePlugin(map[string]int64{"node1": 100, "node2": 0}),
		},
		{
			name: "Partial Preemption with basic policy",
			nodes: []*v1.Node{
				st.MakeNode().Name("node1").Capacity(map[v1.ResourceName]string{v1.ResourceCPU: "2", v1.ResourceMemory: "4Gi", v1.ResourcePods: "32"}).Obj(),
				st.MakeNode().Name("node2").Capacity(map[v1.ResourceName]string{v1.ResourceCPU: "2", v1.ResourceMemory: "4Gi", v1.ResourcePods: "32"}).Obj(),
			},
			podGroups: []*schedulingv1beta1.PodGroup{
				st.MakePodGroup().Name("pg1").Namespace("default").Priority(100).BasicPolicy().Obj(),
			},
			initialPods: []*v1.Pod{
				// low-1 takes half CPU on node1
				st.MakePod().Name("low-1").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").ZeroTerminationGracePeriod().Priority(10).Node("node1").Obj(),
				// very-low-1 takes all CPU on node2
				st.MakePod().Name("very-low-1").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").ZeroTerminationGracePeriod().Priority(5).Node("node2").Obj(),
			},
			preemptorPods: []*v1.Pod{
				st.MakePod().Name("high-1").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("pg1").ZeroTerminationGracePeriod().Priority(100).Obj(),
				st.MakePod().Name("high-2").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("pg1").ZeroTerminationGracePeriod().Priority(100).Obj(),
				st.MakePod().Name("high-3").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("pg1").ZeroTerminationGracePeriod().Priority(100).Obj(),
			},
			// With custom scoring, preemptor pods will prefer high-1 node
			// which will force preemption of low-1 pod.
			expectedScheduled:          []string{"high-1", "high-2", "high-3", "very-low-1"},
			expectedPreempted:          []string{"low-1"},
			expectedToHaveNNNInfo:      []string{},
			expectedPodsPreemptedByWAP: 1,
			customPluginName:           "mockScorePlugin",
			customPluginFunc:           newPresetScorePlugin(map[string]int64{"node1": 100, "node2": 0}),
		},
		{
			name: "PDB Violation Handling (Reprieve)",
			nodes: []*v1.Node{
				st.MakeNode().Name("node1").Capacity(map[v1.ResourceName]string{v1.ResourceCPU: "3", v1.ResourceMemory: "4Gi", v1.ResourcePods: "32"}).Obj(),
			},
			podGroups: []*schedulingv1beta1.PodGroup{
				st.MakePodGroup().Name("pg1").Namespace("default").Priority(100).MinCount(2).Obj(),
			},
			initialPods: []*v1.Pod{
				st.MakePod().Name("low-1").Label("app", "foo").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").ZeroTerminationGracePeriod().Priority(10).Obj(),
				st.MakePod().Name("low-2").Label("app", "foo").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").ZeroTerminationGracePeriod().Priority(10).Obj(),
				st.MakePod().Name("low-3").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").ZeroTerminationGracePeriod().Priority(10).Obj(),
			},
			pdb: &policyv1.PodDisruptionBudget{
				ObjectMeta: metav1.ObjectMeta{Name: "foo-pdb"},
				Spec: policyv1.PodDisruptionBudgetSpec{
					MinAvailable: &intstr.IntOrString{IntVal: 2},
					Selector:     &metav1.LabelSelector{MatchLabels: map[string]string{"app": "foo"}},
				},
			},
			preemptorPods: []*v1.Pod{
				st.MakePod().Name("high-1").Req(map[v1.ResourceName]string{v1.ResourceCPU: "0.5"}).Container("image").PodGroupName("pg1").ZeroTerminationGracePeriod().Priority(100).Obj(),
				st.MakePod().Name("high-2").Req(map[v1.ResourceName]string{v1.ResourceCPU: "0.5"}).Container("image").PodGroupName("pg1").ZeroTerminationGracePeriod().Priority(100).Obj(),
			},
			expectedScheduled:          []string{"high-1", "high-2"},
			expectedPreempted:          []string{"low-3"},
			expectedToHaveNNNInfo:      []string{"high-1", "high-2"},
			expectedPodsPreemptedByWAP: 1,
		},
		{
			name: "Multi-node Preemption",
			nodes: []*v1.Node{
				st.MakeNode().Name("node1").Capacity(map[v1.ResourceName]string{v1.ResourceCPU: "2", v1.ResourceMemory: "4Gi", v1.ResourcePods: "32"}).Obj(),
				st.MakeNode().Name("node2").Capacity(map[v1.ResourceName]string{v1.ResourceCPU: "2", v1.ResourceMemory: "4Gi", v1.ResourcePods: "32"}).Obj(),
			},
			podGroups: []*schedulingv1beta1.PodGroup{
				st.MakePodGroup().Name("pg1").Namespace("default").Priority(100).MinCount(4).Obj(),
			},
			initialPods: []*v1.Pod{
				st.MakePod().Name("low-1").Node("node1").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").ZeroTerminationGracePeriod().Priority(10).Obj(),
				st.MakePod().Name("low-2").Node("node1").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").ZeroTerminationGracePeriod().Priority(10).Obj(),
				st.MakePod().Name("low-3").Node("node2").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").ZeroTerminationGracePeriod().Priority(10).Obj(),
				st.MakePod().Name("low-4").Node("node2").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").ZeroTerminationGracePeriod().Priority(10).Obj(),
			},
			preemptorPods: []*v1.Pod{
				st.MakePod().Name("high-1").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("pg1").ZeroTerminationGracePeriod().Priority(100).Obj(),
				st.MakePod().Name("high-2").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("pg1").ZeroTerminationGracePeriod().Priority(100).Obj(),
				st.MakePod().Name("high-3").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("pg1").ZeroTerminationGracePeriod().Priority(100).Obj(),
				st.MakePod().Name("high-4").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("pg1").ZeroTerminationGracePeriod().Priority(100).Obj(),
			},
			expectedScheduled:          []string{"high-1", "high-2", "high-3", "high-4"},
			expectedPreempted:          []string{"low-1", "low-2", "low-3", "low-4"},
			expectedToHaveNNNInfo:      []string{"high-1", "high-2", "high-3", "high-4"},
			expectedPodsPreemptedByWAP: 4,
		},
		{
			name: "Insufficient Resources (No Preemption)",
			nodes: []*v1.Node{
				st.MakeNode().Name("node1").Capacity(map[v1.ResourceName]string{v1.ResourceCPU: "3", v1.ResourceMemory: "4Gi", v1.ResourcePods: "32"}).Obj(),
			},
			podGroups: []*schedulingv1beta1.PodGroup{
				st.MakePodGroup().Name("pg1").Namespace("default").Priority(100).MinCount(3).Obj(),
			},
			initialPods: []*v1.Pod{
				st.MakePod().Name("mid-1").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").ZeroTerminationGracePeriod().Priority(500).Obj(),
				st.MakePod().Name("low-1").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").ZeroTerminationGracePeriod().Priority(10).Obj(),
				st.MakePod().Name("low-2").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").ZeroTerminationGracePeriod().Priority(10).Obj(),
			},
			preemptorPods: []*v1.Pod{
				st.MakePod().Name("high-1").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("pg1").ZeroTerminationGracePeriod().Priority(100).Obj(),
				st.MakePod().Name("high-2").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("pg1").ZeroTerminationGracePeriod().Priority(100).Obj(),
				st.MakePod().Name("high-3").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("pg1").ZeroTerminationGracePeriod().Priority(100).Obj(),
			},
			expectedScheduled:          []string{"mid-1", "low-1", "low-2"},
			expectedPreempted:          []string{},
			expectedUnschedulable:      []string{"high-1", "high-2", "high-3"},
			expectedToHaveNNNInfo:      []string{},
			expectedPodsPreemptedByWAP: 0,
		},
		{
			name: "Priority-based Victim Selection",
			nodes: []*v1.Node{
				st.MakeNode().Name("node1").Capacity(map[v1.ResourceName]string{v1.ResourceCPU: "3", v1.ResourceMemory: "4Gi", v1.ResourcePods: "32"}).Obj(),
			},
			podGroups: []*schedulingv1beta1.PodGroup{
				st.MakePodGroup().Name("pg1").Namespace("default").Priority(100).MinCount(2).Obj(),
			},
			initialPods: []*v1.Pod{
				st.MakePod().Name("mid-1").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").ZeroTerminationGracePeriod().Priority(50).Obj(),
				st.MakePod().Name("low-1").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").ZeroTerminationGracePeriod().Priority(10).Obj(),
				st.MakePod().Name("low-2").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").ZeroTerminationGracePeriod().Priority(10).Obj(),
			},
			preemptorPods: []*v1.Pod{
				st.MakePod().Name("high-1").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("pg1").ZeroTerminationGracePeriod().Priority(100).Obj(),
				st.MakePod().Name("high-2").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("pg1").ZeroTerminationGracePeriod().Priority(100).Obj(),
			},
			expectedScheduled:          []string{"high-1", "high-2", "mid-1"},
			expectedPreempted:          []string{"low-1", "low-2"},
			expectedToHaveNNNInfo:      []string{"high-1", "high-2"},
			expectedPodsPreemptedByWAP: 2,
		},
		{
			name: "Preempt the whole PodGroup even if preempting a single Pod would suffice",
			nodes: []*v1.Node{
				st.MakeNode().Name("node1").Capacity(map[v1.ResourceName]string{v1.ResourceCPU: "3", v1.ResourceMemory: "4Gi", v1.ResourcePods: "32"}).Obj(),
			},
			podGroups: []*schedulingv1beta1.PodGroup{
				st.MakePodGroup().Name("pg1").Namespace("default").Priority(100).MinCount(1).Obj(),
				st.MakePodGroup().Name("pg2").Namespace("default").DisruptionModeAll().Priority(10).MinCount(3).Obj(),
			},
			initialPods: []*v1.Pod{
				st.MakePod().Name("low-1").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("pg2").ZeroTerminationGracePeriod().Priority(10).Obj(),
				st.MakePod().Name("low-2").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("pg2").ZeroTerminationGracePeriod().Priority(10).Obj(),
				st.MakePod().Name("low-3").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("pg2").ZeroTerminationGracePeriod().Priority(10).Obj(),
			},
			preemptorPods: []*v1.Pod{
				st.MakePod().Name("high-1").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("pg1").ZeroTerminationGracePeriod().Priority(100).Obj(),
			},
			expectedScheduled:          []string{"high-1"},
			expectedPreempted:          []string{"low-1", "low-2", "low-3"},
			expectedToHaveNNNInfo:      []string{"high-1"},
			expectedPodsPreemptedByWAP: 3,
		},
		{
			name: "Preempt the whole basic PodGroup with a PodGroup disruption mode",
			nodes: []*v1.Node{
				st.MakeNode().Name("node1").Capacity(map[v1.ResourceName]string{v1.ResourceCPU: "3", v1.ResourceMemory: "4Gi", v1.ResourcePods: "32"}).Obj(),
			},
			podGroups: []*schedulingv1beta1.PodGroup{
				st.MakePodGroup().Name("pg1").Namespace("default").Priority(100).MinCount(1).Obj(),
				st.MakePodGroup().Name("pg2").Namespace("default").DisruptionModeAll().Priority(10).BasicPolicy().Obj(),
			},
			initialPods: []*v1.Pod{
				st.MakePod().Name("low-1").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("pg2").ZeroTerminationGracePeriod().Priority(10).Obj(),
				st.MakePod().Name("low-2").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("pg2").ZeroTerminationGracePeriod().Priority(10).Obj(),
				st.MakePod().Name("low-3").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("pg2").ZeroTerminationGracePeriod().Priority(10).Obj(),
			},
			preemptorPods: []*v1.Pod{
				st.MakePod().Name("high-1").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("pg1").ZeroTerminationGracePeriod().Priority(100).Obj(),
			},
			expectedScheduled:          []string{"high-1"},
			expectedPreempted:          []string{"low-1", "low-2", "low-3"},
			expectedToHaveNNNInfo:      []string{"high-1"},
			expectedPodsPreemptedByWAP: 3,
		},
		{
			name: "Gang scheduling: schedule as many pods as possible without preempting higher priority pods, but still more than minCount",
			nodes: []*v1.Node{
				st.MakeNode().Name("node1").Capacity(map[v1.ResourceName]string{v1.ResourceCPU: "1", v1.ResourceMemory: "4Gi", v1.ResourcePods: "32"}).Obj(),
				st.MakeNode().Name("node2").Capacity(map[v1.ResourceName]string{v1.ResourceCPU: "1", v1.ResourceMemory: "4Gi", v1.ResourcePods: "32"}).Obj(),
				st.MakeNode().Name("node3").Capacity(map[v1.ResourceName]string{v1.ResourceCPU: "1", v1.ResourceMemory: "4Gi", v1.ResourcePods: "32"}).Obj(),
				st.MakeNode().Name("node4").Capacity(map[v1.ResourceName]string{v1.ResourceCPU: "1", v1.ResourceMemory: "4Gi", v1.ResourcePods: "32"}).Obj(),
			},
			podGroups: []*schedulingv1beta1.PodGroup{
				st.MakePodGroup().Name("preemptor-pg").Namespace("default").Priority(100).MinCount(1).Obj(),
			},
			initialPods: []*v1.Pod{
				st.MakePod().Name("p1").Node("node1").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").ZeroTerminationGracePeriod().Priority(10).Obj(),
				st.MakePod().Name("p2").Node("node2").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").ZeroTerminationGracePeriod().Priority(10).Obj(),
				st.MakePod().Name("p3").Node("node3").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").ZeroTerminationGracePeriod().Priority(200).Obj(),
				st.MakePod().Name("p4").Node("node4").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").ZeroTerminationGracePeriod().Priority(200).Obj(),
			},
			preemptorPods: []*v1.Pod{
				st.MakePod().Name("p-a").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("preemptor-pg").ZeroTerminationGracePeriod().Priority(100).Obj(),
				st.MakePod().Name("p-b").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("preemptor-pg").ZeroTerminationGracePeriod().Priority(100).Obj(),
				st.MakePod().Name("p-c").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("preemptor-pg").ZeroTerminationGracePeriod().Priority(100).Obj(),
			},
			preemptorPodsQueuedInCreationOrder: true,
			expectedScheduled:                  []string{"p-a", "p-b", "p3", "p4"},
			expectedPreempted:                  []string{"p1", "p2"},
			expectedUnschedulable:              []string{"p-c"},
			expectedToHaveNNNInfo:              []string{"p-a", "p-b"},
			expectedPodsPreemptedByWAP:         2,
		},
		{
			name: "Gang scheduling: do not reprieve victim pod group of lower priority",
			nodes: []*v1.Node{
				st.MakeNode().Name("node1").Capacity(map[v1.ResourceName]string{v1.ResourceCPU: "1", v1.ResourceMemory: "4Gi", v1.ResourcePods: "32"}).Obj(),
				st.MakeNode().Name("node2").Capacity(map[v1.ResourceName]string{v1.ResourceCPU: "1", v1.ResourceMemory: "4Gi", v1.ResourcePods: "32"}).Obj(),
				st.MakeNode().Name("node3").Capacity(map[v1.ResourceName]string{v1.ResourceCPU: "1", v1.ResourceMemory: "4Gi", v1.ResourcePods: "32"}).Obj(),
			},
			podGroups: []*schedulingv1beta1.PodGroup{
				st.MakePodGroup().Name("victim-pg").Namespace("default").Priority(50).DisruptionModeAll().MinCount(1).Obj(),
				st.MakePodGroup().Name("preemptor-pg").Namespace("default").Priority(100).MinCount(3).Obj(),
			},
			initialPods: []*v1.Pod{
				st.MakePod().Name("v1").Node("node1").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("victim-pg").ZeroTerminationGracePeriod().Priority(50).Obj(),
				st.MakePod().Name("v2").Node("node2").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("victim-pg").ZeroTerminationGracePeriod().Priority(50).Obj(),
				st.MakePod().Name("v3").Node("node3").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").ZeroTerminationGracePeriod().Priority(10).Obj(),
			},
			preemptorPods: []*v1.Pod{
				st.MakePod().Name("p-a").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("preemptor-pg").ZeroTerminationGracePeriod().Priority(100).Obj(),
				st.MakePod().Name("p-b").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("preemptor-pg").ZeroTerminationGracePeriod().Priority(100).Obj(),
				st.MakePod().Name("p-c").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("preemptor-pg").ZeroTerminationGracePeriod().Priority(100).Obj(),
			},
			expectedScheduled:          []string{"p-a", "p-b", "p-c"},
			expectedPreempted:          []string{"v1", "v2", "v3"},
			expectedToHaveNNNInfo:      []string{"p-a", "p-b", "p-c"},
			expectedPodsPreemptedByWAP: 3,
		},
		{
			name: "Gang scheduling: preempt a pod group victim but do not schedule full pod group",
			nodes: []*v1.Node{
				st.MakeNode().Name("node1").Capacity(map[v1.ResourceName]string{v1.ResourceCPU: "1", v1.ResourceMemory: "4Gi", v1.ResourcePods: "32"}).Obj(),
				st.MakeNode().Name("node2").Capacity(map[v1.ResourceName]string{v1.ResourceCPU: "1", v1.ResourceMemory: "4Gi", v1.ResourcePods: "32"}).Obj(),
				st.MakeNode().Name("node3").Capacity(map[v1.ResourceName]string{v1.ResourceCPU: "1", v1.ResourceMemory: "4Gi", v1.ResourcePods: "32"}).Obj(),
				st.MakeNode().Name("node4").Capacity(map[v1.ResourceName]string{v1.ResourceCPU: "1", v1.ResourceMemory: "4Gi", v1.ResourcePods: "32"}).Obj(),
			},
			podGroups: []*schedulingv1beta1.PodGroup{
				st.MakePodGroup().Name("victim-pg").Namespace("default").Priority(50).DisruptionModeAll().MinCount(2).Obj(),
				st.MakePodGroup().Name("victim-pg2").Namespace("default").Priority(200).DisruptionModeAll().MinCount(2).Obj(),
				st.MakePodGroup().Name("preemptor-pg").Namespace("default").Priority(100).MinCount(1).DisruptionModeAll().Obj(),
			},
			initialPods: []*v1.Pod{
				st.MakePod().Name("v1").Node("node1").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("victim-pg").ZeroTerminationGracePeriod().Priority(50).Obj(),
				st.MakePod().Name("v2").Node("node2").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("victim-pg").ZeroTerminationGracePeriod().Priority(50).Obj(),
				st.MakePod().Name("v3").Node("node3").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("victim-pg2").ZeroTerminationGracePeriod().Priority(200).Obj(),
				st.MakePod().Name("v4").Node("node4").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("victim-pg2").ZeroTerminationGracePeriod().Priority(200).Obj(),
			},
			preemptorPods: []*v1.Pod{
				st.MakePod().Name("p-a").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("preemptor-pg").ZeroTerminationGracePeriod().Priority(100).Obj(),
				st.MakePod().Name("p-b").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("preemptor-pg").ZeroTerminationGracePeriod().Priority(100).Obj(),
				st.MakePod().Name("p-c").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("preemptor-pg").ZeroTerminationGracePeriod().Priority(100).Obj(),
			},
			preemptorPodsQueuedInCreationOrder: true,
			// p-a will preempt victim-pg, p-b will schedule to empty space, so only p-a will have NNN info.
			expectedScheduled:          []string{"p-a", "p-b", "v3", "v4"},
			expectedPreempted:          []string{"v1", "v2"},
			expectedUnschedulable:      []string{"p-c"},
			expectedToHaveNNNInfo:      []string{"p-a"},
			expectedPodsPreemptedByWAP: 2,
		},
		{
			name: "Basic scheduling: schedule as many pods as possible without preempting higher priority pods",
			nodes: []*v1.Node{
				st.MakeNode().Name("node1").Capacity(map[v1.ResourceName]string{v1.ResourceCPU: "1", v1.ResourceMemory: "4Gi", v1.ResourcePods: "32"}).Obj(),
				st.MakeNode().Name("node2").Capacity(map[v1.ResourceName]string{v1.ResourceCPU: "1", v1.ResourceMemory: "4Gi", v1.ResourcePods: "32"}).Obj(),
				st.MakeNode().Name("node3").Capacity(map[v1.ResourceName]string{v1.ResourceCPU: "1", v1.ResourceMemory: "4Gi", v1.ResourcePods: "32"}).Obj(),
				st.MakeNode().Name("node4").Capacity(map[v1.ResourceName]string{v1.ResourceCPU: "1", v1.ResourceMemory: "4Gi", v1.ResourcePods: "32"}).Obj(),
			},
			podGroups: []*schedulingv1beta1.PodGroup{
				st.MakePodGroup().Name("preemptor-pg").Namespace("default").Priority(100).BasicPolicy().Obj(),
			},
			initialPods: []*v1.Pod{
				st.MakePod().Name("p1").Node("node1").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").ZeroTerminationGracePeriod().Priority(10).Obj(),
				st.MakePod().Name("p2").Node("node2").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").ZeroTerminationGracePeriod().Priority(10).Obj(),
				st.MakePod().Name("p3").Node("node3").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").ZeroTerminationGracePeriod().Priority(200).Obj(),
				st.MakePod().Name("p4").Node("node4").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").ZeroTerminationGracePeriod().Priority(200).Obj(),
			},
			preemptorPods: []*v1.Pod{
				st.MakePod().Name("p-a").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("preemptor-pg").ZeroTerminationGracePeriod().Priority(100).Obj(),
				st.MakePod().Name("p-b").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("preemptor-pg").ZeroTerminationGracePeriod().Priority(100).Obj(),
				st.MakePod().Name("p-c").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("preemptor-pg").ZeroTerminationGracePeriod().Priority(100).Obj(),
			},
			preemptorPodsQueuedInCreationOrder: true,
			expectedScheduled:                  []string{"p-a", "p-b", "p3", "p4"},
			expectedPreempted:                  []string{"p1", "p2"},
			expectedUnschedulable:              []string{"p-c"},
			expectedToHaveNNNInfo:              []string{"p-a", "p-b"},
			expectedPodsPreemptedByWAP:         2,
		},
		{
			name: "Basic scheduling: do not reprieve victim pod group of lower priority",
			nodes: []*v1.Node{
				st.MakeNode().Name("node1").Capacity(map[v1.ResourceName]string{v1.ResourceCPU: "1", v1.ResourceMemory: "4Gi", v1.ResourcePods: "32"}).Obj(),
				st.MakeNode().Name("node2").Capacity(map[v1.ResourceName]string{v1.ResourceCPU: "1", v1.ResourceMemory: "4Gi", v1.ResourcePods: "32"}).Obj(),
				st.MakeNode().Name("node3").Capacity(map[v1.ResourceName]string{v1.ResourceCPU: "1", v1.ResourceMemory: "4Gi", v1.ResourcePods: "32"}).Obj(),
			},
			podGroups: []*schedulingv1beta1.PodGroup{
				st.MakePodGroup().Name("victim-pg").Namespace("default").Priority(50).DisruptionModeAll().MinCount(1).Obj(),
				st.MakePodGroup().Name("preemptor-pg").Namespace("default").Priority(100).BasicPolicy().Obj(),
			},
			initialPods: []*v1.Pod{
				st.MakePod().Name("v1").Node("node1").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("victim-pg").ZeroTerminationGracePeriod().Priority(50).Obj(),
				st.MakePod().Name("v2").Node("node2").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("victim-pg").ZeroTerminationGracePeriod().Priority(50).Obj(),
				st.MakePod().Name("v3").Node("node3").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").ZeroTerminationGracePeriod().Priority(10).Obj(),
			},
			preemptorPods: []*v1.Pod{
				st.MakePod().Name("p-a").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("preemptor-pg").ZeroTerminationGracePeriod().Priority(100).Obj(),
				st.MakePod().Name("p-b").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("preemptor-pg").ZeroTerminationGracePeriod().Priority(100).Obj(),
				st.MakePod().Name("p-c").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("preemptor-pg").ZeroTerminationGracePeriod().Priority(100).Obj(),
			},
			expectedScheduled: []string{"p-a", "p-b", "p-c"},
			expectedPreempted: []string{"v1", "v2", "v3"},
			// There are no guarantees about NNN,
			// depending on the number of queued pods in WAS cycle
			// WAP can preempt different number of pods
			// It's also possible that WAP will preempt enough pods
			// so the further WAS cycle (after observing more pods)
			// will no longer need to run WAP.
			// In that case it's possible that none of the pods will have NNN set.
			expectedToHaveNNNInfo:      []string{},
			expectedPodsPreemptedByWAP: 3,
		},
		{
			name: "Basic scheduling: preempt a pod group victim but do not schedule full pod group",
			nodes: []*v1.Node{
				st.MakeNode().Name("node1").Capacity(map[v1.ResourceName]string{v1.ResourceCPU: "1", v1.ResourceMemory: "4Gi", v1.ResourcePods: "32"}).Obj(),
				st.MakeNode().Name("node2").Capacity(map[v1.ResourceName]string{v1.ResourceCPU: "1", v1.ResourceMemory: "4Gi", v1.ResourcePods: "32"}).Obj(),
				st.MakeNode().Name("node3").Capacity(map[v1.ResourceName]string{v1.ResourceCPU: "1", v1.ResourceMemory: "4Gi", v1.ResourcePods: "32"}).Obj(),
				st.MakeNode().Name("node4").Capacity(map[v1.ResourceName]string{v1.ResourceCPU: "1", v1.ResourceMemory: "4Gi", v1.ResourcePods: "32"}).Obj(),
			},
			podGroups: []*schedulingv1beta1.PodGroup{
				st.MakePodGroup().Name("victim-pg").Namespace("default").Priority(50).DisruptionModeAll().MinCount(2).Obj(),
				st.MakePodGroup().Name("victim-pg2").Namespace("default").Priority(200).DisruptionModeAll().MinCount(2).Obj(),
				st.MakePodGroup().Name("preemptor-pg").Namespace("default").Priority(100).BasicPolicy().DisruptionModeAll().Obj(),
			},
			initialPods: []*v1.Pod{
				st.MakePod().Name("v1").Node("node1").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("victim-pg").ZeroTerminationGracePeriod().Priority(50).Obj(),
				st.MakePod().Name("v2").Node("node2").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("victim-pg").ZeroTerminationGracePeriod().Priority(50).Obj(),
				st.MakePod().Name("v3").Node("node3").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("victim-pg2").ZeroTerminationGracePeriod().Priority(200).Obj(),
				st.MakePod().Name("v4").Node("node4").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("victim-pg2").ZeroTerminationGracePeriod().Priority(200).Obj(),
			},
			preemptorPods: []*v1.Pod{
				st.MakePod().Name("p-a").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("preemptor-pg").ZeroTerminationGracePeriod().Priority(100).Obj(),
				st.MakePod().Name("p-b").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("preemptor-pg").ZeroTerminationGracePeriod().Priority(100).Obj(),
				st.MakePod().Name("p-c").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("preemptor-pg").ZeroTerminationGracePeriod().Priority(100).Obj(),
			},
			preemptorPodsQueuedInCreationOrder: true,
			// p-a will preempt "victim-pg" and p-b will schedule to empty space, so only p-a will have NNN info.
			expectedScheduled:          []string{"p-a", "p-b", "v3", "v4"},
			expectedPreempted:          []string{"v1", "v2"},
			expectedUnschedulable:      []string{"p-c"},
			expectedToHaveNNNInfo:      []string{"p-a"},
			expectedPodsPreemptedByWAP: 2,
		},

		{
			name: "PodGroup with PreemptNever preemption policy does not perform preemption, with PodGroupPreemptionPolicy enabled",
			nodes: []*v1.Node{
				st.MakeNode().Name("node1").Capacity(map[v1.ResourceName]string{v1.ResourceCPU: "3", v1.ResourceMemory: "4Gi", v1.ResourcePods: "32"}).Obj(),
			},
			podGroups: []*schedulingv1beta1.PodGroup{
				st.MakePodGroup().Name("pg1").Namespace("default").Priority(100).MinCount(3).PreemptionPolicy(schedulingv1beta1.PreemptNever).Obj(),
			},
			initialPods: []*v1.Pod{
				st.MakePod().Name("low-1").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").ZeroTerminationGracePeriod().Priority(10).Obj(),
				st.MakePod().Name("low-2").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").ZeroTerminationGracePeriod().Priority(10).Obj(),
				st.MakePod().Name("low-3").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").ZeroTerminationGracePeriod().Priority(10).Obj(),
			},
			preemptorPods: []*v1.Pod{
				st.MakePod().Name("high-1").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("pg1").ZeroTerminationGracePeriod().Priority(100).PreemptionPolicy(v1.PreemptNever).Obj(),
				st.MakePod().Name("high-2").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("pg1").ZeroTerminationGracePeriod().Priority(100).PreemptionPolicy(v1.PreemptNever).Obj(),
				st.MakePod().Name("high-3").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("pg1").ZeroTerminationGracePeriod().Priority(100).PreemptionPolicy(v1.PreemptNever).Obj(),
			},
			expectedScheduled:              []string{"low-1", "low-2", "low-3"},
			expectedPreempted:              []string{},
			expectedUnschedulable:          []string{"high-1", "high-2", "high-3"},
			expectedPodsPreemptedByWAP:     0,
			enablePodGroupPreemptionPolicy: true,
		},
		{
			name: "PodGroup with PreemptLowerPriority preemption policy performs preemption, with PodGroupPreemptionPolicy enabled",
			nodes: []*v1.Node{
				st.MakeNode().Name("node1").Capacity(map[v1.ResourceName]string{v1.ResourceCPU: "3", v1.ResourceMemory: "4Gi", v1.ResourcePods: "32"}).Obj(),
			},
			podGroups: []*schedulingv1beta1.PodGroup{
				st.MakePodGroup().Name("pg1").Namespace("default").Priority(100).MinCount(3).PreemptionPolicy(schedulingv1beta1.PreemptLowerPriority).Obj(),
			},
			initialPods: []*v1.Pod{
				st.MakePod().Name("low-1").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").ZeroTerminationGracePeriod().Priority(10).Obj(),
				st.MakePod().Name("low-2").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").ZeroTerminationGracePeriod().Priority(10).Obj(),
				st.MakePod().Name("low-3").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").ZeroTerminationGracePeriod().Priority(10).Obj(),
			},
			preemptorPods: []*v1.Pod{
				st.MakePod().Name("high-1").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("pg1").ZeroTerminationGracePeriod().Priority(100).Obj(),
				st.MakePod().Name("high-2").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("pg1").ZeroTerminationGracePeriod().Priority(100).Obj(),
				st.MakePod().Name("high-3").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("pg1").ZeroTerminationGracePeriod().Priority(100).Obj(),
			},
			expectedScheduled:              []string{"high-1", "high-2", "high-3"},
			expectedPreempted:              []string{"low-1", "low-2", "low-3"},
			expectedPodsPreemptedByWAP:     3,
			enablePodGroupPreemptionPolicy: true,
		},
		{
			name: "PodGroup with default preemption policy performs preemption, with PodGroupPreemptionPolicy enabled",
			nodes: []*v1.Node{
				st.MakeNode().Name("node1").Capacity(map[v1.ResourceName]string{v1.ResourceCPU: "3", v1.ResourceMemory: "4Gi", v1.ResourcePods: "32"}).Obj(),
			},
			podGroups: []*schedulingv1beta1.PodGroup{
				st.MakePodGroup().Name("pg1").Namespace("default").Priority(100).MinCount(3).Obj(),
			},
			initialPods: []*v1.Pod{
				st.MakePod().Name("low-1").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").ZeroTerminationGracePeriod().Priority(10).Obj(),
				st.MakePod().Name("low-2").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").ZeroTerminationGracePeriod().Priority(10).Obj(),
				st.MakePod().Name("low-3").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").ZeroTerminationGracePeriod().Priority(10).Obj(),
			},
			preemptorPods: []*v1.Pod{
				st.MakePod().Name("high-1").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("pg1").ZeroTerminationGracePeriod().Priority(100).Obj(),
				st.MakePod().Name("high-2").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("pg1").ZeroTerminationGracePeriod().Priority(100).Obj(),
				st.MakePod().Name("high-3").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("pg1").ZeroTerminationGracePeriod().Priority(100).Obj(),
			},
			expectedScheduled:              []string{"high-1", "high-2", "high-3"},
			expectedPreempted:              []string{"low-1", "low-2", "low-3"},
			expectedPodsPreemptedByWAP:     3,
			enablePodGroupPreemptionPolicy: true,
		},
		{
			name: "PodGroup with PreemptNever preemption policy in all pods does not perform preemption, with PodGroupPreemptionPolicy disabled",
			nodes: []*v1.Node{
				st.MakeNode().Name("node1").Capacity(map[v1.ResourceName]string{v1.ResourceCPU: "3", v1.ResourceMemory: "4Gi", v1.ResourcePods: "32"}).Obj(),
			},
			podGroups: []*schedulingv1beta1.PodGroup{
				st.MakePodGroup().Name("pg1").Namespace("default").Priority(100).MinCount(3).Obj(),
			},
			initialPods: []*v1.Pod{
				st.MakePod().Name("low-1").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").ZeroTerminationGracePeriod().Priority(10).Obj(),
				st.MakePod().Name("low-2").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").ZeroTerminationGracePeriod().Priority(10).Obj(),
				st.MakePod().Name("low-3").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").ZeroTerminationGracePeriod().Priority(10).Obj(),
			},
			preemptorPods: []*v1.Pod{
				st.MakePod().Name("high-1").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("pg1").ZeroTerminationGracePeriod().Priority(100).PreemptionPolicy(v1.PreemptNever).Obj(),
				st.MakePod().Name("high-2").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("pg1").ZeroTerminationGracePeriod().Priority(100).PreemptionPolicy(v1.PreemptNever).Obj(),
				st.MakePod().Name("high-3").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("pg1").ZeroTerminationGracePeriod().Priority(100).PreemptionPolicy(v1.PreemptNever).Obj(),
			},
			expectedScheduled:          []string{"low-1", "low-2", "low-3"},
			expectedPreempted:          []string{},
			expectedUnschedulable:      []string{"high-1", "high-2", "high-3"},
			expectedPodsPreemptedByWAP: 0,
		},
		{
			name: "Gang scheduling: preemption with node resources",
			nodes: []*v1.Node{
				st.MakeNode().Name("node1").Label("kubernetes.io/hostname", "node1").Capacity(map[v1.ResourceName]string{v1.ResourceCPU: "2", v1.ResourceMemory: "4Gi", v1.ResourcePods: "32"}).Obj(),
				st.MakeNode().Name("node2").Label("kubernetes.io/hostname", "node2").Capacity(map[v1.ResourceName]string{v1.ResourceCPU: "2", v1.ResourceMemory: "4Gi", v1.ResourcePods: "32"}).Obj(),
			},
			podGroups: []*schedulingv1beta1.PodGroup{
				st.MakePodGroup().Name("preemptor-pg").Namespace("default").Priority(100).MinCount(2).Obj(),
			},
			initialPods: []*v1.Pod{
				st.MakePod().Name("initial-pod").Label("app", "initial").Node("node1").Req(map[v1.ResourceName]string{v1.ResourceCPU: "2"}).Container("image").ZeroTerminationGracePeriod().Priority(10).Obj(),
			},
			preemptorPods: []*v1.Pod{
				st.MakePod().Name("preemptor-1").Req(map[v1.ResourceName]string{v1.ResourceCPU: "2"}).Container("image").PodGroupName("preemptor-pg").ZeroTerminationGracePeriod().Priority(100).Obj(),
				st.MakePod().Name("preemptor-2").Req(map[v1.ResourceName]string{v1.ResourceCPU: "2"}).Container("image").PodGroupName("preemptor-pg").ZeroTerminationGracePeriod().Priority(100).Obj(),
			},
			expectedScheduled:          []string{"preemptor-1", "preemptor-2"},
			expectedPreempted:          []string{"initial-pod"},
			expectedToHaveNNNInfo:      []string{"preemptor-1", "preemptor-2"},
			expectedPodsPreemptedByWAP: 1,
		},
		{
			name: "Gang scheduling: preemption with node resources, prioritizes reprieval of higher priority pods",
			nodes: []*v1.Node{
				st.MakeNode().Name("node1").Label("kubernetes.io/hostname", "node1").Capacity(map[v1.ResourceName]string{v1.ResourceCPU: "2", v1.ResourceMemory: "4Gi", v1.ResourcePods: "32"}).Obj(),
				st.MakeNode().Name("node2").Label("kubernetes.io/hostname", "node2").Capacity(map[v1.ResourceName]string{v1.ResourceCPU: "2", v1.ResourceMemory: "4Gi", v1.ResourcePods: "32"}).Obj(),
			},
			podGroups: []*schedulingv1beta1.PodGroup{
				st.MakePodGroup().Name("preemptor-pg").Namespace("default").Priority(100).MinCount(2).Obj(),
			},
			initialPods: []*v1.Pod{
				st.MakePod().Name("initial-pod-1").Label("app", "initial").Node("node1").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").ZeroTerminationGracePeriod().Priority(5).Obj(),
				st.MakePod().Name("initial-pod-2").Label("app", "initial").Node("node1").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").ZeroTerminationGracePeriod().Priority(10).Obj(),
			},
			preemptorPods: []*v1.Pod{
				st.MakePod().Name("preemptor-1").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("preemptor-pg").ZeroTerminationGracePeriod().Priority(100).
					NodeSelector(map[string]string{"kubernetes.io/hostname": "node1"}).Obj(),
				st.MakePod().Name("preemptor-2").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("preemptor-pg").ZeroTerminationGracePeriod().Priority(100).
					NodeSelector(map[string]string{"kubernetes.io/hostname": "node2"}).Obj(),
			},
			expectedScheduled:          []string{"preemptor-1", "preemptor-2"},
			expectedPreempted:          []string{"initial-pod-1"},
			expectedToHaveNNNInfo:      []string{"preemptor-1", "preemptor-2"},
			expectedPodsPreemptedByWAP: 1,
		},
		{
			name: "Gang scheduling: preemption with pod level resources",
			nodes: []*v1.Node{
				st.MakeNode().Name("node1").Label("kubernetes.io/hostname", "node1").Capacity(map[v1.ResourceName]string{v1.ResourceCPU: "2", v1.ResourceMemory: "4Gi", v1.ResourcePods: "32"}).Obj(),
				st.MakeNode().Name("node2").Label("kubernetes.io/hostname", "node2").Capacity(map[v1.ResourceName]string{v1.ResourceCPU: "2", v1.ResourceMemory: "4Gi", v1.ResourcePods: "32"}).Obj(),
			},
			podGroups: []*schedulingv1beta1.PodGroup{
				st.MakePodGroup().Name("preemptor-pg").Namespace("default").Priority(100).MinCount(2).Obj(),
			},
			initialPods: []*v1.Pod{
				st.MakePod().Name("initial-pod").Label("app", "initial").Node("node1").PodLevelResourceRequests(map[v1.ResourceName]string{v1.ResourceCPU: "2"}).Container("image").ZeroTerminationGracePeriod().Priority(10).Obj(),
			},
			preemptorPods: []*v1.Pod{
				st.MakePod().Name("preemptor-1").PodLevelResourceRequests(map[v1.ResourceName]string{v1.ResourceCPU: "2"}).Container("image").PodGroupName("preemptor-pg").ZeroTerminationGracePeriod().Priority(100).
					PodAntiAffinityExists("app", "kubernetes.io/hostname", st.PodAntiAffinityWithRequiredReq).Obj(),
				st.MakePod().Name("preemptor-2").PodLevelResourceRequests(map[v1.ResourceName]string{v1.ResourceCPU: "2"}).Container("image").PodGroupName("preemptor-pg").ZeroTerminationGracePeriod().Priority(100).
					PodAntiAffinityExists("app", "kubernetes.io/hostname", st.PodAntiAffinityWithRequiredReq).Obj(),
			},
			expectedScheduled:          []string{"preemptor-1", "preemptor-2"},
			expectedPreempted:          []string{"initial-pod"},
			expectedToHaveNNNInfo:      []string{"preemptor-1", "preemptor-2"},
			expectedPodsPreemptedByWAP: 1,
		},
		{
			// Even though there is enough resources to keep initial pod when scheduling preemptor
			// due to the pod anti affinity it cannot be reprieved.
			name: "Gang scheduling: preemption with pod anti-affinity constraints",
			nodes: []*v1.Node{
				st.MakeNode().Name("node1").Label("kubernetes.io/hostname", "node1").Capacity(map[v1.ResourceName]string{v1.ResourceCPU: "2", v1.ResourceMemory: "4Gi", v1.ResourcePods: "32"}).Obj(),
				st.MakeNode().Name("node2").Label("kubernetes.io/hostname", "node2").Capacity(map[v1.ResourceName]string{v1.ResourceCPU: "2", v1.ResourceMemory: "4Gi", v1.ResourcePods: "32"}).Obj(),
			},
			podGroups: []*schedulingv1beta1.PodGroup{
				st.MakePodGroup().Name("preemptor-pg").Namespace("default").Priority(100).MinCount(2).Obj(),
			},
			initialPods: []*v1.Pod{
				st.MakePod().Name("initial-pod").Label("app", "initial").Node("node1").Req(map[v1.ResourceName]string{v1.ResourceCPU: "0.25"}).Container("image").ZeroTerminationGracePeriod().Priority(10).Obj(),
			},
			preemptorPods: []*v1.Pod{
				st.MakePod().Name("preemptor-1").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1.5"}).Container("image").PodGroupName("preemptor-pg").ZeroTerminationGracePeriod().Priority(100).
					PodAntiAffinityExists("app", "kubernetes.io/hostname", st.PodAntiAffinityWithRequiredReq).Obj(),
				st.MakePod().Name("preemptor-2").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1.5"}).Container("image").PodGroupName("preemptor-pg").ZeroTerminationGracePeriod().Priority(100).
					PodAntiAffinityExists("app", "kubernetes.io/hostname", st.PodAntiAffinityWithRequiredReq).Obj(),
			},
			expectedScheduled:          []string{"preemptor-1", "preemptor-2"},
			expectedPreempted:          []string{"initial-pod"},
			expectedToHaveNNNInfo:      []string{"preemptor-1", "preemptor-2"},
			expectedPodsPreemptedByWAP: 1,
		},
		{
			// Even though there is enough resources to keep initial pod when scheduling preemptor
			// due to the pod node port it cannot be reprieved.
			name: "Gang scheduling: preemption with pod node port",
			nodes: []*v1.Node{
				st.MakeNode().Name("node1").Capacity(map[v1.ResourceName]string{v1.ResourceCPU: "2", v1.ResourceMemory: "4Gi", v1.ResourcePods: "32"}).Obj(),
				st.MakeNode().Name("node2").Capacity(map[v1.ResourceName]string{v1.ResourceCPU: "2", v1.ResourceMemory: "4Gi", v1.ResourcePods: "32"}).Obj(),
			},
			podGroups: []*schedulingv1beta1.PodGroup{
				st.MakePodGroup().Name("preemptor-pg").Namespace("default").Priority(100).MinCount(2).Obj(),
			},
			initialPods: []*v1.Pod{
				st.MakePod().Name("initial-pod").ContainerPort([]v1.ContainerPort{{ContainerPort: 8080, HostPort: 8080}}).Node("node1").Req(map[v1.ResourceName]string{v1.ResourceCPU: "0.25"}).Container("image").ZeroTerminationGracePeriod().Priority(10).Obj(),
			},
			preemptorPods: []*v1.Pod{
				st.MakePod().Name("preemptor-1").ContainerPort([]v1.ContainerPort{{ContainerPort: 8080, HostPort: 8080}}).Req(map[v1.ResourceName]string{v1.ResourceCPU: "1.5"}).Container("image").PodGroupName("preemptor-pg").ZeroTerminationGracePeriod().Priority(100).Obj(),
				st.MakePod().Name("preemptor-2").ContainerPort([]v1.ContainerPort{{ContainerPort: 8080, HostPort: 8080}}).Req(map[v1.ResourceName]string{v1.ResourceCPU: "1.5"}).Container("image").PodGroupName("preemptor-pg").ZeroTerminationGracePeriod().Priority(100).Obj(),
			},
			expectedScheduled:          []string{"preemptor-1", "preemptor-2"},
			expectedPreempted:          []string{"initial-pod"},
			expectedToHaveNNNInfo:      []string{"preemptor-1", "preemptor-2"},
			expectedPodsPreemptedByWAP: 1,
		},
		{
			// Even though there is enough resources to keep initial pod when scheduling preemptor
			// due to the pod topolgy spread it cannot be reprieved.
			name: "Gang scheduling: preemption with pod topology spread constraints",
			nodes: []*v1.Node{
				st.MakeNode().Name("node1").Label("kubernetes.io/hostname", "node1").Capacity(map[v1.ResourceName]string{v1.ResourceCPU: "16", v1.ResourceMemory: "4Gi", v1.ResourcePods: "32"}).Obj(),
				st.MakeNode().Name("node2").Label("kubernetes.io/hostname", "node2").Capacity(map[v1.ResourceName]string{v1.ResourceCPU: "16", v1.ResourceMemory: "4Gi", v1.ResourcePods: "32"}).Obj(),
			},
			podGroups: []*schedulingv1beta1.PodGroup{
				st.MakePodGroup().Name("preemptor-pg").Namespace("default").Priority(100).MinCount(2).Obj(),
			},
			initialPods: []*v1.Pod{
				st.MakePod().Name("initial-pod").Label("app", "foo").Node("node1").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").ZeroTerminationGracePeriod().Priority(10).Obj(),
			},
			preemptorPods: []*v1.Pod{
				func() *v1.Pod {
					p := st.MakePod().Name("preemptor-1").Label("app", "foo").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("preemptor-pg").ZeroTerminationGracePeriod().Priority(100).
						NodeSelector(map[string]string{"kubernetes.io/hostname": "node1"}).Obj()
					p.Spec.TopologySpreadConstraints = []v1.TopologySpreadConstraint{
						{
							MaxSkew:           2,
							TopologyKey:       "kubernetes.io/hostname",
							WhenUnsatisfiable: v1.DoNotSchedule,
							LabelSelector:     &metav1.LabelSelector{MatchLabels: map[string]string{"app": "foo"}},
							MinDomains:        new(int32(10)),
						},
					}
					return p
				}(),
				func() *v1.Pod {
					p := st.MakePod().Name("preemptor-2").Label("app", "foo").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("preemptor-pg").ZeroTerminationGracePeriod().Priority(100).
						NodeSelector(map[string]string{"kubernetes.io/hostname": "node1"}).Obj()
					p.Spec.TopologySpreadConstraints = []v1.TopologySpreadConstraint{
						{
							MaxSkew:           2,
							TopologyKey:       "kubernetes.io/hostname",
							WhenUnsatisfiable: v1.DoNotSchedule,
							LabelSelector:     &metav1.LabelSelector{MatchLabels: map[string]string{"app": "foo"}},
							MinDomains:        new(int32(10)),
						},
					}
					return p
				}(),
			},
			expectedScheduled:          []string{"preemptor-1", "preemptor-2"},
			expectedPreempted:          []string{"initial-pod"},
			expectedToHaveNNNInfo:      []string{"preemptor-1", "preemptor-2"},
			expectedPodsPreemptedByWAP: 1,
		},
		{
			name: "Gang scheduling: preemption with pod topology spread constraints, single reprieve",
			nodes: []*v1.Node{
				st.MakeNode().Name("node1").Label("kubernetes.io/hostname", "node1").Capacity(map[v1.ResourceName]string{v1.ResourceCPU: "16", v1.ResourceMemory: "4Gi", v1.ResourcePods: "32"}).Obj(),
				st.MakeNode().Name("node2").Label("kubernetes.io/hostname", "node2").Capacity(map[v1.ResourceName]string{v1.ResourceCPU: "16", v1.ResourceMemory: "4Gi", v1.ResourcePods: "32"}).Obj(),
			},
			podGroups: []*schedulingv1beta1.PodGroup{
				st.MakePodGroup().Name("preemptor-pg").Namespace("default").Priority(100).MinCount(2).Obj(),
			},
			initialPods: []*v1.Pod{
				st.MakePod().Name("initial-pod").Label("app", "foo").Node("node1").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").ZeroTerminationGracePeriod().Priority(10).Obj(),
				// initial-pod-2 can be reprieved even though it has lower priority, because it won't cause skew
				st.MakePod().Name("initial-pod-2").Label("app", "foo").Node("node2").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").ZeroTerminationGracePeriod().Priority(5).Obj(),
			},
			preemptorPods: []*v1.Pod{
				func() *v1.Pod {
					p := st.MakePod().Name("preemptor-1").Label("app", "foo").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("preemptor-pg").ZeroTerminationGracePeriod().Priority(100).
						NodeSelector(map[string]string{"kubernetes.io/hostname": "node1"}).Obj()
					p.Spec.TopologySpreadConstraints = []v1.TopologySpreadConstraint{
						{
							MaxSkew:           2,
							TopologyKey:       "kubernetes.io/hostname",
							WhenUnsatisfiable: v1.DoNotSchedule,
							LabelSelector:     &metav1.LabelSelector{MatchLabels: map[string]string{"app": "foo"}},
							MinDomains:        new(int32(10)),
						},
					}
					return p
				}(),
				func() *v1.Pod {
					p := st.MakePod().Name("preemptor-2").Label("app", "foo").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("preemptor-pg").ZeroTerminationGracePeriod().Priority(100).
						NodeSelector(map[string]string{"kubernetes.io/hostname": "node1"}).Obj()
					p.Spec.TopologySpreadConstraints = []v1.TopologySpreadConstraint{
						{
							MaxSkew:           2,
							TopologyKey:       "kubernetes.io/hostname",
							WhenUnsatisfiable: v1.DoNotSchedule,
							LabelSelector:     &metav1.LabelSelector{MatchLabels: map[string]string{"app": "foo"}},
							MinDomains:        new(int32(10)),
						},
					}
					return p
				}(),
			},
			expectedScheduled:          []string{"preemptor-1", "preemptor-2"},
			expectedPreempted:          []string{"initial-pod"},
			expectedToHaveNNNInfo:      []string{"preemptor-1", "preemptor-2"},
			expectedPodsPreemptedByWAP: 1,
		},
		{
			// This scenario verifies that during reprieval we respect Reserve plugins.
			// The number of reserved pods + pods with "resource-taken" is at max 2.
			name: "Reserve plugins are called during preemption simulation, so second pod fails",
			nodes: []*v1.Node{
				st.MakeNode().Name("node1").Capacity(map[v1.ResourceName]string{v1.ResourceCPU: "3", v1.ResourceMemory: "4Gi", v1.ResourcePods: "32"}).Obj(),
			},
			podGroups: []*schedulingv1beta1.PodGroup{
				st.MakePodGroup().Name("preemptor-pg").Priority(100).MinCount(2).Obj(),
			},
			initialPods: []*v1.Pod{
				st.MakePod().Name("v1").Node("node1").Label("resource-taken", "true").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").ZeroTerminationGracePeriod().Priority(15).Obj(),
				st.MakePod().Name("v2").Node("node1").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").ZeroTerminationGracePeriod().Priority(10).Obj(),
			},
			preemptorPods: []*v1.Pod{
				st.MakePod().Name("p-a").Label("test-plugin", "true").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("preemptor-pg").ZeroTerminationGracePeriod().Priority(100).Obj(),
				st.MakePod().Name("p-b").Label("test-plugin", "true").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("preemptor-pg").ZeroTerminationGracePeriod().Priority(100).Obj(),
			},
			expectedScheduled:          []string{"p-a", "p-b", "v2"},
			expectedPreempted:          []string{"v1"},
			expectedUnschedulable:      []string{},
			expectedToHaveNNNInfo:      []string{},
			expectedPodsPreemptedByWAP: 1,
			customPluginName:           "mockReservePlugin",
			customPluginFunc: func(_ context.Context, _ runtime.Object, _ fwk.Handle) (fwk.Plugin, error) {
				return &mockReservePlugin{maxPods: 2}, nil
			},
		},
		{
			name: "Binding first before preemption for gang policy",
			nodes: []*v1.Node{
				st.MakeNode().Name("node1").Capacity(map[v1.ResourceName]string{v1.ResourceCPU: "4", v1.ResourceMemory: "4Gi", v1.ResourcePods: "32"}).Obj(),
			},
			podGroups: []*schedulingv1beta1.PodGroup{
				st.MakePodGroup().Name("pg1").Namespace("default").Priority(100).MinCount(2).Obj(),
			},
			initialPods: []*v1.Pod{
				st.MakePod().Name("pg-pod-1").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("pg1").ZeroTerminationGracePeriod().Priority(100).Obj(),
				st.MakePod().Name("pg-pod-2").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("pg1").ZeroTerminationGracePeriod().Priority(100).Obj(),
				st.MakePod().Name("low-1").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").ZeroTerminationGracePeriod().Priority(10).Obj(),
			},
			preemptorPods: []*v1.Pod{
				st.MakePod().Name("high-1").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("pg1").ZeroTerminationGracePeriod().Priority(100).Obj(),
				st.MakePod().Name("high-2").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("pg1").ZeroTerminationGracePeriod().Priority(100).Obj(),
			},
			expectedScheduled:          []string{"pg-pod-1", "pg-pod-2", "high-1", "high-2"},
			expectedPreempted:          []string{"low-1"},
			expectedPodsPreemptedByWAP: 1,
			tempRemovePG:               true,
			// both preemptor pods will become schedulable at once, but there will be only place for 1 pod without preemption
			// the scheduling cycle should prefer binding this pod over preempting to make room for both pods
			// preemption will be called in the subsequent cycle to make room for the second pod.
			expectedEventOrder: []string{"Bind:high-1", "PodGroupPostFilter:pg1", "Bind:high-2"},
		},
		{
			name: "Binding first before preemption for basic policy",
			nodes: []*v1.Node{
				st.MakeNode().Name("node1").Capacity(map[v1.ResourceName]string{v1.ResourceCPU: "4", v1.ResourceMemory: "4Gi", v1.ResourcePods: "32"}).Obj(),
			},
			podGroups: []*schedulingv1beta1.PodGroup{
				st.MakePodGroup().Name("pg1").Namespace("default").Priority(100).BasicPolicy().Obj(),
			},
			initialPods: []*v1.Pod{
				st.MakePod().Name("pg-pod-1").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("pg1").ZeroTerminationGracePeriod().Priority(100).Obj(),
				st.MakePod().Name("pg-pod-2").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("pg1").ZeroTerminationGracePeriod().Priority(100).Obj(),
				st.MakePod().Name("low-1").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").ZeroTerminationGracePeriod().Priority(10).Obj(),
			},
			preemptorPods: []*v1.Pod{
				st.MakePod().Name("high-1").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("pg1").ZeroTerminationGracePeriod().Priority(100).Obj(),
				st.MakePod().Name("high-2").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("pg1").ZeroTerminationGracePeriod().Priority(100).Obj(),
			},
			expectedScheduled:          []string{"pg-pod-1", "pg-pod-2", "high-1", "high-2"},
			expectedPreempted:          []string{"low-1"},
			expectedPodsPreemptedByWAP: 1,
			tempRemovePG:               true,
			// both preemptor pods will become schedulable at once, but there will be only place for 1 pod without preemption
			// the scheduling cycle should prefer binding this pod over preempting to make room for both pods
			// preemption will be called in the subsequent cycle to make room for the second pod.
			expectedEventOrder: []string{"Bind:high-1", "PodGroupPostFilter:pg1", "Bind:high-2"},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			featuregatetesting.SetFeatureGatesDuringTest(t, utilfeature.DefaultFeatureGate, featuregatetesting.FeatureOverrides{
				features.GenericWorkload:          true,
				features.PodGroupPreemptionPolicy: tt.enablePodGroupPreemptionPolicy,
			})
			recorder := eventRecorder{}
			registry := make(frameworkruntime.Registry)

			// Register mock bind plugin that will register NNN information during binding.
			mockBindPluginName := "mockBindPlugin"
			var bindPlugin = mockBindPlugin{
				name:       mockBindPluginName,
				realPlugin: nil,
				nnnInfo:    sync.Map{},
				recorder:   &recorder,
			}
			err := registry.Register(mockBindPluginName, func(ctx context.Context, o runtime.Object, fh fwk.Handle) (fwk.Plugin, error) {
				db, err := defaultbinder.New(ctx, o, fh)
				if err != nil {
					t.Fatalf("Error creating a default binder plugin: %v", err)
				}
				bindPlugin.realPlugin = db.(fwk.BindPlugin)
				return &bindPlugin, nil
			})
			if err != nil {
				t.Fatalf("Error registering a bind plugin: %v", err)
			}

			mockPGPostFilterPluginName := "mockPGPostFilterPlugin"
			var pgPostFilterPlugin = mockPodGroupPostFilterPlugin{
				name:     mockPGPostFilterPluginName,
				recorder: &recorder,
			}
			err = registry.Register(mockPGPostFilterPluginName, func(ctx context.Context, o runtime.Object, fh fwk.Handle) (fwk.Plugin, error) {
				return &pgPostFilterPlugin, nil
			})
			if err != nil {
				t.Fatalf("Error registering a pg post filter plugin: %v", err)
			}

			cfg := configtesting.V1ToInternalWithDefaults(t, configv1.KubeSchedulerConfiguration{
				Profiles: []configv1.KubeSchedulerProfile{{
					SchedulerName: ptr.To(v1.DefaultSchedulerName),
					Plugins: &configv1.Plugins{
						MultiPoint: configv1.PluginSet{
							Enabled: []configv1.Plugin{
								{Name: mockBindPluginName},
								{Name: mockPGPostFilterPluginName},
								{Name: names.DefaultPreemption},
							},
							Disabled: []configv1.Plugin{
								{Name: names.DefaultBinder},
								// Disable DefaultPreemption from its default position to allow explicit ordering.
								// If not disabled, it runs as an override first and terminates the post-filter chain,
								// preventing our mock plugins from recording events.
								{Name: names.DefaultPreemption},
							},
						},
					},
				}},
			})

			if tt.customPluginName != "" {
				err := registry.Register(tt.customPluginName, tt.customPluginFunc)
				if err != nil {
					t.Fatalf("Error registering custom plugin: %v", err)
				}
				cfg.Profiles[0].Plugins.MultiPoint.Enabled = append(cfg.Profiles[0].Plugins.MultiPoint.Enabled, config.Plugin{Name: tt.customPluginName})
			}

			// Set PodMaxBackoff to 1 second to turn on backoff and allow apiCacher to get information about
			// pod NNN. Without this we might have a race between starting binding and update of apiCacher.
			testCtx := testutils.InitTestSchedulerWithNS(t, "podgroup-preemption",
				scheduler.WithProfiles(cfg.Profiles...),
				scheduler.WithFrameworkOutOfTreeRegistry(registry),
				scheduler.WithPodMaxBackoffSeconds(1),
				scheduler.WithPodInitialBackoffSeconds(0))
			cs, ns := testCtx.ClientSet, testCtx.NS.Name

			// Create nodes
			for _, n := range tt.nodes {
				if _, err := cs.CoreV1().Nodes().Create(testCtx.Ctx, n, metav1.CreateOptions{}); err != nil {
					t.Fatalf("Failed to create node %s: %v", n.Name, err)
				}
			}

			// Create PDB if specified
			if tt.pdb != nil {
				tt.pdb.Namespace = ns
				if _, err := cs.PolicyV1().PodDisruptionBudgets(ns).Create(testCtx.Ctx, tt.pdb, metav1.CreateOptions{}); err != nil {
					t.Fatalf("Failed to create PDB: %v", err)
				}
			}

			// 1. Create PodGroups
			for _, pg := range tt.podGroups {
				pg.Namespace = ns
				if _, err := cs.SchedulingV1beta1().PodGroups(ns).Create(testCtx.Ctx, pg, metav1.CreateOptions{}); err != nil {
					t.Fatalf("Failed to create PodGroup %s: %v", pg.Name, err)
				}
			}

			// 2. Create initial pods
			for _, p := range tt.initialPods {
				p.Namespace = ns
				if _, err := cs.CoreV1().Pods(ns).Create(testCtx.Ctx, p, metav1.CreateOptions{}); err != nil {
					t.Fatalf("Failed to create pod %s: %v", p.Name, err)
				}
			}

			// 3. Wait for initial pods to be scheduled
			for _, p := range tt.initialPods {
				if err := wait.PollUntilContextTimeout(testCtx.Ctx, 100*time.Millisecond, 10*time.Second, false,
					testutils.PodScheduled(cs, ns, p.Name)); err != nil {
					t.Errorf("Failed to wait for pod %s to be scheduled: %v", p.Name, err)
				}
			}

			recorder.Clear()

			// 4. Create preemptor pods
			if tt.tempRemovePG {
				// Temporarily remove PodGroups. This is a trick to ensure that all preemptor pods
				// are created and queued as unschedulable first, and then become schedulable at once
				// when the PodGroup is recreated.
				pgNames := make([]string, len(tt.podGroups))
				for i, pg := range tt.podGroups {
					pgNames[i] = pg.Name
				}
				if err := deletePodGroups(testCtx.Ctx, cs, ns, pgNames); err != nil {
					t.Fatalf("Failed to delete PodGroups: %v", err)
				}
			}

			for _, p := range tt.preemptorPods {
				p.Namespace = ns
				if _, err := cs.CoreV1().Pods(ns).Create(testCtx.Ctx, p, metav1.CreateOptions{}); err != nil {
					t.Fatalf("Failed to create pod %s: %v", p.Name, err)
				}
				if !tt.tempRemovePG && tt.preemptorPodsQueuedInCreationOrder {
					podScheduledFn := testutils.PodScheduled(cs, ns, p.Name)
					err := wait.PollUntilContextTimeout(testCtx.Ctx, 100*time.Millisecond, 10*time.Second, false, func(ctx context.Context) (bool, error) {
						_, ok := testCtx.Scheduler.SchedulingQueue.GetPod(p.Name, p.Namespace, p.Spec.SchedulingGroup)
						if ok {
							return true, nil
						}
						// pod may have gotten queued and scheduled between the polls
						return podScheduledFn(ctx)
					})
					if err != nil {
						t.Fatalf("Failed to ensure order of pod %s: %v", p.Name, err)
					}
				}
			}

			if tt.tempRemovePG {
				// Wait for preemptor pods to be unschedulable
				for _, p := range tt.preemptorPods {
					if err := wait.PollUntilContextTimeout(testCtx.Ctx, 100*time.Millisecond, 10*time.Second, false,
						func(ctx context.Context) (bool, error) {
							return isPodInUnschedulableQueue(testCtx.Scheduler, p.Name, ns), nil
						}); err != nil {
						t.Fatalf("Failed to wait for pod %s to be unschedulable: %v", p.Name, err)
					}
				}

				// Recreate PodGroups
				for _, pg := range tt.podGroups {
					pgCopy := pg.DeepCopy()
					pgCopy.ResourceVersion = ""
					if _, err := cs.SchedulingV1beta1().PodGroups(ns).Create(testCtx.Ctx, pgCopy, metav1.CreateOptions{}); err != nil {
						t.Fatalf("Failed to recreate PodGroup %s: %v", pg.Name, err)
					}
				}
			}

			// 5. Wait for preemption to complete if WAP calls are expected
			if tt.expectedPodsPreemptedByWAP > 0 {
				wapCalls := 0
				err := wait.PollUntilContextTimeout(testCtx.Ctx, 100*time.Millisecond, 10*time.Second, false, func(ctx context.Context) (bool, error) {
					wapCalls = 0
					for _, podName := range tt.expectedPreempted {
						events, err := cs.CoreV1().Events(ns).List(ctx, metav1.ListOptions{
							FieldSelector: "involvedObject.name=" + podName,
						})
						if err != nil {
							return false, err
						}
						for _, event := range events.Items {
							if event.Reason == "Preempted" && strings.HasPrefix(event.Message, "Preempted by podgroup") {
								wapCalls++
								break
							}
						}
					}
					return wapCalls == tt.expectedPodsPreemptedByWAP, nil
				})
				if err != nil {
					t.Errorf("WorkloadAwarePreemption was not called expected times within timeout: want=%d, got=%d", wapCalls, tt.expectedPodsPreemptedByWAP)
				}
			}

			// 6. Verify unschedulable pods
			for _, podName := range tt.expectedUnschedulable {
				if err := wait.PollUntilContextTimeout(testCtx.Ctx, 100*time.Millisecond, 10*time.Second, false,
					testutils.PodUnschedulable(cs, ns, podName)); err != nil {
					t.Errorf("Pod %s was expected to be unschedulable but wasn't: %v", podName, err)
				}
			}

			// 7. Verify scheduled pods
			for _, podName := range tt.expectedScheduled {
				if err := wait.PollUntilContextTimeout(testCtx.Ctx, 100*time.Millisecond, 10*time.Second, false,
					testutils.PodScheduled(cs, ns, podName)); err != nil {
					t.Errorf("Pod %s was expected to be scheduled but wasn't: %v", podName, err)
				}
			}

			// 8. Verify preempted pods
			for _, podName := range tt.expectedPreempted {
				if err := wait.PollUntilContextTimeout(testCtx.Ctx, 200*time.Millisecond, 5*time.Second, false,
					func(ctx context.Context) (bool, error) {
						pod, err := cs.CoreV1().Pods(ns).Get(ctx, podName, metav1.GetOptions{})
						if err != nil {
							return apierrors.IsNotFound(err), nil
						}
						if pod.DeletionTimestamp != nil {
							return true, nil
						}
						_, cond := podutil.GetPodCondition(&pod.Status, v1.DisruptionTarget)
						return cond != nil, nil
					}); err != nil {
					t.Errorf("Pod %s was expected to be preempted but wasn't", podName)
				}
			}

			// 9. Verify preemptor pods have nominated node name
			for _, podName := range tt.expectedToHaveNNNInfo {
				if node, ok := bindPlugin.nnnInfo.Load(podName); !ok || node.(string) == "" {
					t.Errorf("Pod %s was expected to have nominated node name but didn't", podName)
				}
			}

			// Verify event order
			if len(tt.expectedEventOrder) > 0 {
				actualEvents := recorder.GetEvents()
				if diff := cmp.Diff(tt.expectedEventOrder, actualEvents); diff != "" {
					t.Errorf("Unexpected event order (-want,+got):\n%s", diff)
				}
			}

			// 10. Dump the state of pods to ease debugging failed runs.
			if t.Failed() {
				t.Log("Dumping states of initial and preemptor pods:")
				var allPods []string
				for _, p := range tt.initialPods {
					allPods = append(allPods, p.Name)
				}
				for _, p := range tt.preemptorPods {
					allPods = append(allPods, p.Name)
				}
				for _, podName := range allPods {
					pod, err := cs.CoreV1().Pods(ns).Get(testCtx.Ctx, podName, metav1.GetOptions{})
					if err != nil {
						if apierrors.IsNotFound(err) {
							t.Logf("Pod %q: not present in cluster", podName)
						} else {
							t.Logf("Pod %q: failed to get: %v", podName, err)
						}
						continue
					}

					var statusStr string
					if pod.Spec.NodeName != "" {
						statusStr = "scheduled on node " + pod.Spec.NodeName
					} else {
						_, cond := podutil.GetPodCondition(&pod.Status, v1.PodScheduled)
						if cond != nil && cond.Status == v1.ConditionFalse && cond.Reason == v1.PodReasonUnschedulable {
							statusStr = "unschedulable"
						} else {
							statusStr = "pending"
						}
					}
					t.Logf("Pod %q: status=%s, phase=%s", podName, statusStr, pod.Status.Phase)
				}
			}
		})
	}
}

func TestPodGroupPreemptionStatus(t *testing.T) {
	featuregatetesting.SetFeatureGatesDuringTest(t, utilfeature.DefaultFeatureGate, featuregatetesting.FeatureOverrides{
		features.GenericWorkload: true,
	})
	testCtx := testutils.InitTestSchedulerWithNS(t, "podgroup-preemption-status")

	cs := testCtx.ClientSet
	ns := testCtx.NS.Name

	// Create a node.
	node := st.MakeNode().Name("node-1").Capacity(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Obj()
	if _, err := cs.CoreV1().Nodes().Create(testCtx.Ctx, node, metav1.CreateOptions{}); err != nil {
		t.Fatalf("Failed to create node: %v", err)
	}
	// Create a low-priority pod low-1 taking whole node
	lowPod := st.MakePod().Name("low-1").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").Priority(10).TerminationGracePeriodSeconds(30).Obj()
	if _, err := cs.CoreV1().Pods(ns).Create(testCtx.Ctx, lowPod, metav1.CreateOptions{}); err != nil {
		t.Fatalf("Failed to create low-priority pod: %v", err)
	}
	// Wait for low-priority pod to be scheduled
	if err := wait.PollUntilContextTimeout(testCtx.Ctx, 100*time.Millisecond, 10*time.Second, false,
		testutils.PodScheduled(cs, ns, lowPod.Name)); err != nil {
		t.Fatalf("Failed to wait for low-priority pod to be scheduled: %v", err)
	}
	// Create a high-priority pod high-1 belonging to pg1 (priority=100)
	pg := st.MakePodGroup().Name("pg1").Namespace(ns).MinCount(1).Priority(100).Obj()
	if _, err := cs.SchedulingV1beta1().PodGroups(ns).Create(testCtx.Ctx, pg, metav1.CreateOptions{}); err != nil {
		t.Fatalf("Failed to create PodGroup: %v", err)
	}
	highPod := st.MakePod().Name("high-1").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("pg1").Priority(100).Obj()
	if _, err := cs.CoreV1().Pods(ns).Create(testCtx.Ctx, highPod, metav1.CreateOptions{}); err != nil {
		t.Fatalf("Failed to create high-priority pod: %v", err)
	}
	// Poll until the low-priority pod gets DeletionTimestamp set (which indicates preemption is triggered)
	err := wait.PollUntilContextTimeout(testCtx.Ctx, 100*time.Millisecond, 10*time.Second, false, func(ctx context.Context) (bool, error) {
		pod, err := cs.CoreV1().Pods(ns).Get(ctx, lowPod.Name, metav1.GetOptions{})
		if err != nil {
			return false, err
		}
		return pod.DeletionTimestamp != nil, nil
	})
	if err != nil {
		t.Fatalf("Failed to wait for low-priority pod to get DeletionTimestamp set: %v", err)
	}
	// Verify the PodGroup condition.
	// We want PodGroupInitiallyScheduled status to be False, Reason to be Unschedulable, and Message to contain
	// both "minCount (1) cannot be satisfied" and "pod group preemption found a placement for podgroup"
	var cond *metav1.Condition
	err = wait.PollUntilContextTimeout(testCtx.Ctx, 100*time.Millisecond, 5*time.Second, false, func(ctx context.Context) (bool, error) {
		currentPG, err := cs.SchedulingV1beta1().PodGroups(ns).Get(ctx, pg.Name, metav1.GetOptions{})
		if err != nil {
			return false, err
		}
		cond = apimeta.FindStatusCondition(currentPG.Status.Conditions, schedulingv1beta1.PodGroupInitiallyScheduled)
		if cond != nil &&
			cond.Status == metav1.ConditionFalse &&
			cond.Reason == schedulingv1beta1.PodGroupReasonUnschedulable &&
			strings.Contains(cond.Message, "minCount (1) cannot be satisfied") &&
			strings.Contains(cond.Message, "pod group preemption: found a placement for podgroup, preempting 1 victims") {
			return true, nil
		}
		return false, nil
	})
	if err != nil {
		t.Logf("Failed to verify PodGroup condition: %v", err)
		t.Fatalf("Last observed podGroup condition: Status=%s, Reason=%s, Message=%q", cond.Status, cond.Reason, cond.Message)
	}
}

func TestPodGroupPreemption_NominatedNodeNameRespected(t *testing.T) {
	featuregatetesting.SetFeatureGatesDuringTest(t, utilfeature.DefaultFeatureGate, featuregatetesting.FeatureOverrides{
		features.GenericWorkload: true,
	})

	mockScorePlugin := "mockScorePlugin"
	registry := make(frameworkruntime.Registry)
	err := registry.Register(mockScorePlugin, newPresetScorePlugin(map[string]int64{"node1": 100, "node2": 0}))
	if err != nil {
		t.Fatalf("Failed to register custom score plugin: %v", err)
	}

	cfg := configtesting.V1ToInternalWithDefaults(t, configv1.KubeSchedulerConfiguration{
		Profiles: []configv1.KubeSchedulerProfile{{
			SchedulerName: ptr.To(v1.DefaultSchedulerName),
			Plugins: &configv1.Plugins{
				Score: configv1.PluginSet{
					Enabled: []configv1.Plugin{
						{Name: mockScorePlugin},
					},
				},
			},
		}},
	})

	testCtx := testutils.InitTestSchedulerWithNS(t, "pg-preemption-nnn",
		scheduler.WithProfiles(cfg.Profiles...),
		scheduler.WithFrameworkOutOfTreeRegistry(registry),
		scheduler.WithPodMaxBackoffSeconds(1),
		scheduler.WithPodInitialBackoffSeconds(0),
	)
	cs, ns := testCtx.ClientSet, testCtx.NS.Name

	// 1. Create 3 nodes
	nodes := []*v1.Node{
		st.MakeNode().Name("node1").Capacity(map[v1.ResourceName]string{v1.ResourceCPU: "1", v1.ResourceMemory: "4Gi", v1.ResourcePods: "32"}).Obj(),
		st.MakeNode().Name("node2").Capacity(map[v1.ResourceName]string{v1.ResourceCPU: "1", v1.ResourceMemory: "4Gi", v1.ResourcePods: "32"}).Obj(),
		st.MakeNode().Name("node3").Capacity(map[v1.ResourceName]string{v1.ResourceCPU: "1", v1.ResourceMemory: "4Gi", v1.ResourcePods: "32"}).Obj(),
	}
	for _, node := range nodes {
		if _, err := cs.CoreV1().Nodes().Create(testCtx.Ctx, node, metav1.CreateOptions{}); err != nil {
			t.Fatalf("Failed to create node %s: %v", node.Name, err)
		}
	}

	// 2. Create PodGroup with minCount=2
	pg := st.MakePodGroup().Name("pg1").Namespace(ns).Priority(50).MinCount(2).Obj()
	if _, err := cs.SchedulingV1beta1().PodGroups(ns).Create(testCtx.Ctx, pg, metav1.CreateOptions{}); err != nil {
		t.Fatalf("Failed to create pod group pg1: %v", err)
	}

	// 3. Create initial pods: pod1 on node1 (high priority), pod2 on node2 & pod3 on node3 (low priority with default grace period)
	initialPods := []*v1.Pod{
		st.MakePod().Name("pod1").Namespace(ns).Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").Priority(100).ZeroTerminationGracePeriod().Node("node1").Obj(),
		st.MakePod().Name("pod2").Namespace(ns).Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").Priority(10).Node("node2").Obj(),
		st.MakePod().Name("pod3").Namespace(ns).Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").Priority(10).Node("node3").Obj(),
	}
	for _, pod := range initialPods {
		if _, err := cs.CoreV1().Pods(ns).Create(testCtx.Ctx, pod, metav1.CreateOptions{}); err != nil {
			t.Fatalf("Failed to create pod %s: %v", pod.Name, err)
		}
	}

	// Wait for initial pods to be scheduled
	for _, pod := range initialPods {
		if err := wait.PollUntilContextTimeout(testCtx.Ctx, 100*time.Millisecond, 10*time.Second, false, testutils.PodScheduled(cs, ns, pod.Name)); err != nil {
			t.Fatalf("Failed to wait for initial pod %s to schedule: %v", pod.Name, err)
		}
	}

	// 4. Create preemptor pods belonging to pg1
	preemptorPods := []*v1.Pod{
		st.MakePod().Name("preemptor-1").Namespace(ns).Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("pg1").Priority(50).Obj(),
		st.MakePod().Name("preemptor-2").Namespace(ns).Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("pg1").Priority(50).Obj(),
	}
	for _, pod := range preemptorPods {
		if _, err := cs.CoreV1().Pods(ns).Create(testCtx.Ctx, pod, metav1.CreateOptions{}); err != nil {
			t.Fatalf("Failed to create preemptor pod %s: %v", pod.Name, err)
		}
	}

	// 5. Wait for preemption to occur and verify that NominatedNodeName is set on both preemptor pods
	initialNNNs := make(map[string]string)
	for _, pod := range preemptorPods {
		podName := pod.Name
		err := wait.PollUntilContextTimeout(testCtx.Ctx, 100*time.Millisecond, 10*time.Second, false, func(ctx context.Context) (bool, error) {
			p, err := cs.CoreV1().Pods(ns).Get(ctx, podName, metav1.GetOptions{})
			if err != nil {
				return false, err
			}
			if p.Status.NominatedNodeName != "" {
				initialNNNs[podName] = p.Status.NominatedNodeName
				return true, nil
			}
			return false, nil
		})
		if err != nil {
			t.Fatalf("Timed out waiting for NominatedNodeName on %s: %v", podName, err)
		}
	}

	for podName, nodeName := range initialNNNs {
		if nodeName == "node1" {
			t.Errorf("Expected preemptor pod %s NNN to be node2 or node3, got %s", podName, nodeName)
		}
	}

	// 6. Remove pod1 from node1
	if err := cs.CoreV1().Pods(ns).Delete(testCtx.Ctx, "pod1", metav1.DeleteOptions{GracePeriodSeconds: new(int64(0))}); err != nil {
		t.Fatalf("Failed to delete pod1: %v", err)
	}

	// Wait for pod1 to be completely removed from API server
	err = wait.PollUntilContextTimeout(testCtx.Ctx, 100*time.Millisecond, 10*time.Second, false, func(ctx context.Context) (bool, error) {
		_, err := cs.CoreV1().Pods(ns).Get(ctx, "pod1", metav1.GetOptions{})
		if apierrors.IsNotFound(err) {
			return true, nil
		}
		return false, nil
	})
	if err != nil {
		t.Fatalf("Timed out waiting for pod1 to be deleted: %v", err)
	}

	// 7. Verify that nominated node names did not change to node1 despite node1 having free space and higher score from preferNode1ScorePlugin
	err = wait.PollUntilContextTimeout(testCtx.Ctx, 100*time.Millisecond, 3*time.Second, false, func(ctx context.Context) (bool, error) {
		_, err := cs.CoreV1().Pods(ns).Get(ctx, "pod2", metav1.GetOptions{})
		if apierrors.IsNotFound(err) {
			t.Fatalf("pod2 got removed")
		}
		_, err = cs.CoreV1().Pods(ns).Get(ctx, "pod3", metav1.GetOptions{})
		if apierrors.IsNotFound(err) {
			t.Fatalf("pod3 got removed")
		}

		for _, pod := range preemptorPods {
			events, err := cs.CoreV1().Events(ns).List(ctx, metav1.ListOptions{
				FieldSelector: "involvedObject.name=" + pod.Name,
			})
			if err != nil {
				return false, err
			}
			for _, event := range events.Items {
				t.Logf("Event: %v", event.Message)
			}

			p, err := cs.CoreV1().Pods(ns).Get(ctx, pod.Name, metav1.GetOptions{})
			if err != nil {
				return false, err
			}
			if p.Spec.NodeName == "node1" {
				t.Errorf("Pod %s was incorrectly scheduled to node1", pod.Name)
				return false, nil
			}
			if p.Status.NominatedNodeName != initialNNNs[pod.Name] {
				t.Errorf("Pod %s NominatedNodeName changed from %s to %s. NodeName = %s", pod.Name, initialNNNs[pod.Name], p.Status.NominatedNodeName, p.Spec.NodeName)
				return false, nil
			}
		}
		return false, nil
	})
	if err != nil && !errors.Is(err, context.DeadlineExceeded) && !wait.Interrupted(err) {
		t.Fatalf("Unexpected error while checking NNN persistence: %v", err)
	}
}

// TestCompositePodGroupPreemption tests preemption scenarios involving composite pod groups.
func TestCompositePodGroupPreemption(t *testing.T) {
	featuregatetesting.SetFeatureGatesDuringTest(t, utilfeature.DefaultFeatureGate, featuregatetesting.FeatureOverrides{
		features.GenericWorkload:                 true,
		features.CompositePodGroup:               true,
		features.TopologyAwareWorkloadScheduling: true,
	})

	tests := []struct {
		name                           string
		nodes                          []*v1.Node
		compositePodGroups             []*schedulingv1alpha3.CompositePodGroup
		podGroups                      []*schedulingv1beta1.PodGroup
		initialPods                    []*v1.Pod // pods that should be scheduled before preemption starts
		preemptorPods                  []*v1.Pod // pods that belong to a CPG hierarchy and should trigger preemption
		pdb                            *policyv1.PodDisruptionBudget
		expectedScheduled              []string
		expectedPreempted              []string
		expectedUnschedulable          []string
		expectedToHaveNNNInfo          []string
		expectedPodsPreemptedByWAP     int
		enablePodGroupPreemptionPolicy bool
		customPluginName               string
		customPluginFunc               frameworkruntime.PluginFactory
		// tempRemoveCPG, if true, temporarily removes CompositePodGroups and PodGroups for the time
		// of creating preemptor pods - but after initial pods have been scheduled.
		// This ensures that all preemptor pods are created and kept in incompletePodGroupPods.
		// Once the groups are recreated, all pods become schedulable simultaneously and
		// are guaranteed to be evaluated together in the next PodGroup scheduling cycle.
		// This avoids test flakiness caused by running multiple scheduling cycles with a partial set of preemptor pods.
		tempRemoveCPG bool
		// removeCPGNameBeforePreemption removes the specified CPG from the cluster after initial pods are scheduled,
		// but before preemptors are created. This simulates scenarios where a hierarchy is broken
		// mid-operation to ensure the victim selection logic correctly falls back when parents are missing.
		removeCPGNameBeforePreemption string
	}{

		{
			name: "CPG Partial Preemption",
			nodes: []*v1.Node{
				st.MakeNode().Name("node1").Capacity(map[v1.ResourceName]string{v1.ResourceCPU: "2", v1.ResourceMemory: "4Gi", v1.ResourcePods: "32"}).Obj(),
				st.MakeNode().Name("node2").Capacity(map[v1.ResourceName]string{v1.ResourceCPU: "2", v1.ResourceMemory: "4Gi", v1.ResourcePods: "32"}).Obj(),
			},
			compositePodGroups: []*schedulingv1alpha3.CompositePodGroup{
				st.MakeCompositePodGroup().Name("cpg1").Namespace("default").Priority(100).BasicPolicy().WorkloadRef("wl1", "t1").Obj(),
			},
			podGroups: []*schedulingv1beta1.PodGroup{
				st.MakePodGroup().Name("pg1").Namespace("default").Priority(100).MinCount(3).ParentCompositePodGroup("cpg1").WorkloadRef("t1", "wl1").Obj(),
			},
			initialPods: []*v1.Pod{
				st.MakePod().Name("low-1").Req(map[v1.ResourceName]string{v1.ResourceCPU: "2"}).Container("image").ZeroTerminationGracePeriod().Priority(10).Node("node1").Obj(),
				st.MakePod().Name("low-2").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").ZeroTerminationGracePeriod().Priority(10).Node("node2").Obj(),
			},
			preemptorPods: []*v1.Pod{
				st.MakePod().Name("high-1").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("pg1").ZeroTerminationGracePeriod().Priority(100).Obj(),
				st.MakePod().Name("high-2").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("pg1").ZeroTerminationGracePeriod().Priority(100).Obj(),
				st.MakePod().Name("high-3").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("pg1").ZeroTerminationGracePeriod().Priority(100).Obj(),
			},
			expectedScheduled:              []string{"high-1", "high-2", "high-3", "low-2"},
			expectedPreempted:              []string{"low-1"},
			expectedPodsPreemptedByWAP:     1,
			customPluginName:               "mockScorePlugin",
			customPluginFunc:               newPresetScorePlugin(map[string]int64{"node1": 100, "node2": 0}),
			enablePodGroupPreemptionPolicy: true,
		},
		{
			name: "CPG Victim Across Multiple Nodes, CPG DisruptionModeAll",
			nodes: []*v1.Node{
				st.MakeNode().Name("node1").Label("kubernetes.io/hostname", "node1").Capacity(map[v1.ResourceName]string{v1.ResourceCPU: "2", v1.ResourceMemory: "4Gi", v1.ResourcePods: "32"}).Obj(),
				st.MakeNode().Name("node2").Label("kubernetes.io/hostname", "node2").Capacity(map[v1.ResourceName]string{v1.ResourceCPU: "2", v1.ResourceMemory: "4Gi", v1.ResourcePods: "32"}).Obj(),
			},
			compositePodGroups: []*schedulingv1alpha3.CompositePodGroup{
				st.MakeCompositePodGroup().Name("cpg1").Namespace("default").Priority(10).BasicPolicy().DisruptionModeAll().WorkloadRef("wl1", "t1").Obj(),
			},
			podGroups: []*schedulingv1beta1.PodGroup{
				st.MakePodGroup().Name("pg1").Namespace("default").Priority(10).MinCount(2).ParentCompositePodGroup("cpg1").WorkloadRef("t1", "wl1").Obj(),
				st.MakePodGroup().Name("pg2").Namespace("default").Priority(10).MinCount(1).ParentCompositePodGroup("cpg1").WorkloadRef("t1", "wl1").Obj(),
			},
			initialPods: []*v1.Pod{
				st.MakePod().Name("low-1").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("pg1").Node("node1").ZeroTerminationGracePeriod().Priority(10).Obj(),
				st.MakePod().Name("low-2").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("pg1").Node("node2").ZeroTerminationGracePeriod().Priority(10).Obj(),
				st.MakePod().Name("low-3").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("pg2").Node("node2").ZeroTerminationGracePeriod().Priority(10).Obj(),
			},
			preemptorPods: []*v1.Pod{
				st.MakePod().Name("high-1").Req(map[v1.ResourceName]string{v1.ResourceCPU: "2"}).Container("image").NodeSelector(map[string]string{"kubernetes.io/hostname": "node1"}).ZeroTerminationGracePeriod().Priority(100).Obj(),
			},
			expectedScheduled:              []string{"high-1"},
			expectedPreempted:              []string{"low-1", "low-2", "low-3"},
			expectedPodsPreemptedByWAP:     3,
			enablePodGroupPreemptionPolicy: false,
		},
		{
			name: "CPG Victim Across Multiple Nodes, CPG DisruptionModeSingle",
			nodes: []*v1.Node{
				st.MakeNode().Name("node1").Label("kubernetes.io/hostname", "node1").Capacity(map[v1.ResourceName]string{v1.ResourceCPU: "2", v1.ResourceMemory: "4Gi", v1.ResourcePods: "32"}).Obj(),
				st.MakeNode().Name("node2").Label("kubernetes.io/hostname", "node2").Capacity(map[v1.ResourceName]string{v1.ResourceCPU: "2", v1.ResourceMemory: "4Gi", v1.ResourcePods: "32"}).Obj(),
			},
			compositePodGroups: []*schedulingv1alpha3.CompositePodGroup{
				st.MakeCompositePodGroup().Name("cpg1").Namespace("default").Priority(10).BasicPolicy().DisruptionModeSingle().WorkloadRef("wl1", "t1").Obj(),
			},
			podGroups: []*schedulingv1beta1.PodGroup{
				st.MakePodGroup().Name("pg1").Namespace("default").Priority(10).MinCount(2).DisruptionModeSingle().ParentCompositePodGroup("cpg1").WorkloadRef("t1", "wl1").Obj(),
				st.MakePodGroup().Name("pg2").Namespace("default").Priority(10).MinCount(1).DisruptionModeSingle().ParentCompositePodGroup("cpg1").WorkloadRef("t1", "wl1").Obj(),
			},
			initialPods: []*v1.Pod{
				st.MakePod().Name("low-1").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("pg1").Node("node1").ZeroTerminationGracePeriod().Priority(10).Obj(),
				st.MakePod().Name("low-2").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("pg1").Node("node2").ZeroTerminationGracePeriod().Priority(10).Obj(),
				st.MakePod().Name("low-3").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("pg2").Node("node2").ZeroTerminationGracePeriod().Priority(10).Obj(),
			},
			preemptorPods: []*v1.Pod{
				st.MakePod().Name("high-1").Req(map[v1.ResourceName]string{v1.ResourceCPU: "2"}).Container("image").NodeSelector(map[string]string{"kubernetes.io/hostname": "node1"}).ZeroTerminationGracePeriod().Priority(100).Obj(),
			},
			expectedScheduled:              []string{"high-1", "low-2", "low-3"},
			expectedPreempted:              []string{"low-1"},
			expectedPodsPreemptedByWAP:     1,
			enablePodGroupPreemptionPolicy: false,
		},
		{
			name: "CPG Preemption aborted if victim yields insufficient resources",
			nodes: []*v1.Node{
				st.MakeNode().Name("node1").Capacity(map[v1.ResourceName]string{v1.ResourceCPU: "2", v1.ResourceMemory: "4Gi", v1.ResourcePods: "32"}).Obj(),
			},
			compositePodGroups: []*schedulingv1alpha3.CompositePodGroup{
				st.MakeCompositePodGroup().Name("cpg-victim").Namespace("default").Priority(10).BasicPolicy().WorkloadRef("wl1", "t1").Obj(),
			},
			podGroups: []*schedulingv1beta1.PodGroup{
				st.MakePodGroup().Name("pg-victim").Namespace("default").Priority(10).MinCount(1).ParentCompositePodGroup("cpg-victim").WorkloadRef("t1", "wl1").Obj(),
			},
			initialPods: []*v1.Pod{
				st.MakePod().Name("low-pod").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("pg-victim").Node("node1").ZeroTerminationGracePeriod().Priority(10).Obj(),
			},
			preemptorPods: []*v1.Pod{
				st.MakePod().Name("high-preemptor").Req(map[v1.ResourceName]string{v1.ResourceCPU: "3"}).Container("image").ZeroTerminationGracePeriod().Priority(100).Obj(),
			},
			expectedScheduled:              []string{},
			expectedPreempted:              []string{},
			expectedUnschedulable:          []string{"high-preemptor"},
			expectedPodsPreemptedByWAP:     0,
			enablePodGroupPreemptionPolicy: false,
		},
		{
			name: "CPG Tie-Breaking: PodGroup chosen over CompositePodGroup",
			nodes: []*v1.Node{
				st.MakeNode().Name("node1").Capacity(map[v1.ResourceName]string{v1.ResourceCPU: "2", v1.ResourceMemory: "4Gi", v1.ResourcePods: "32"}).Obj(),
			},
			compositePodGroups: []*schedulingv1alpha3.CompositePodGroup{
				st.MakeCompositePodGroup().Name("cpg-victim").Namespace("default").Priority(10).BasicPolicy().DisruptionModeAll().WorkloadRef("wl1", "t1").Obj(),
			},
			podGroups: []*schedulingv1beta1.PodGroup{
				// Workload 1: A standalone PodGroup (Rank 2)
				st.MakePodGroup().Name("pg-victim-standalone").Namespace("default").Priority(10).MinCount(1).WorkloadRef("t1", "wl2").Obj(),
				// Workload 2: A PodGroup under a CompositePodGroup (Rank 3)
				st.MakePodGroup().Name("pg-victim-child").Namespace("default").Priority(10).MinCount(1).ParentCompositePodGroup("cpg-victim").WorkloadRef("t1", "wl1").Obj(),
				// Preemptor Workload
				st.MakePodGroup().Name("pg-preemptor").Namespace("default").Priority(100).MinCount(1).WorkloadRef("t1", "wl3").Obj(),
			},
			initialPods: []*v1.Pod{
				// Standalone PG Pod
				st.MakePod().Name("low-pg").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("pg-victim-standalone").ZeroTerminationGracePeriod().Priority(10).Node("node1").Obj(),
				// CPG Child PG Pod
				st.MakePod().Name("low-cpg").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("pg-victim-child").ZeroTerminationGracePeriod().Priority(10).Node("node1").Obj(),
			},
			preemptorPods: []*v1.Pod{
				st.MakePod().Name("high-preemptor").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("pg-preemptor").ZeroTerminationGracePeriod().Priority(100).Obj(),
			},
			// The scheduler needs 1 CPU. It has two options: low-pg (Rank 2) or low-cpg (Rank 3).
			// Since Rank 2 < Rank 3, it will choose low-pg (standalone PG) to reprieve the more important CPG structure.
			expectedScheduled:              []string{"high-preemptor"},
			expectedPreempted:              []string{"low-pg"},
			expectedPodsPreemptedByWAP:     1,
			enablePodGroupPreemptionPolicy: false,
		},
		{
			name: "Scheduler targets lower-priority CPG over higher-priority CPG",
			nodes: []*v1.Node{
				st.MakeNode().Name("node1").Capacity(map[v1.ResourceName]string{v1.ResourceCPU: "3", v1.ResourceMemory: "4Gi", v1.ResourcePods: "32"}).Obj(),
			},
			compositePodGroups: []*schedulingv1alpha3.CompositePodGroup{
				st.MakeCompositePodGroup().Name("cpg-lower").Namespace("default").Priority(10).BasicPolicy().WorkloadRef("wl1", "t1").Obj(),
				st.MakeCompositePodGroup().Name("cpg-higher").Namespace("default").Priority(20).BasicPolicy().WorkloadRef("wl2", "t1").Obj(),
			},
			podGroups: []*schedulingv1beta1.PodGroup{
				st.MakePodGroup().Name("pg-lower").Namespace("default").Priority(10).MinCount(1).ParentCompositePodGroup("cpg-lower").WorkloadRef("t1", "wl1").Obj(),
				st.MakePodGroup().Name("pg-higher").Namespace("default").Priority(20).MinCount(1).ParentCompositePodGroup("cpg-higher").WorkloadRef("t1", "wl2").Obj(),
			},
			initialPods: []*v1.Pod{
				st.MakePod().Name("lower-pod").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("pg-lower").Node("node1").ZeroTerminationGracePeriod().Priority(10).Obj(),
				st.MakePod().Name("higher-pod").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("pg-higher").Node("node1").ZeroTerminationGracePeriod().Priority(20).Obj(),
			},
			preemptorPods: []*v1.Pod{
				st.MakePod().Name("high-preemptor").Req(map[v1.ResourceName]string{v1.ResourceCPU: "2"}).Container("image").ZeroTerminationGracePeriod().Priority(100).Obj(),
			},
			expectedScheduled:              []string{"high-preemptor", "higher-pod"},
			expectedPreempted:              []string{"lower-pod"},
			expectedPodsPreemptedByWAP:     1,
			enablePodGroupPreemptionPolicy: false,
		},
		{
			name: "PDB Violation reprieves the entire CPG victim (DisruptionModeAll)",
			nodes: []*v1.Node{
				st.MakeNode().Name("node1").Capacity(map[v1.ResourceName]string{v1.ResourceCPU: "3", v1.ResourceMemory: "4Gi", v1.ResourcePods: "32"}).Obj(),
			},
			compositePodGroups: []*schedulingv1alpha3.CompositePodGroup{
				st.MakeCompositePodGroup().Name("cpg-pdb").Namespace("default").Priority(10).BasicPolicy().DisruptionModeAll().WorkloadRef("wl1", "t1").Obj(),
				st.MakeCompositePodGroup().Name("cpg-no-pdb").Namespace("default").Priority(10).BasicPolicy().DisruptionModeAll().WorkloadRef("wl2", "t1").Obj(),
			},
			podGroups: []*schedulingv1beta1.PodGroup{
				st.MakePodGroup().Name("pg-pdb").Namespace("default").Priority(10).MinCount(2).ParentCompositePodGroup("cpg-pdb").WorkloadRef("t1", "wl1").Obj(),
				st.MakePodGroup().Name("pg-no-pdb").Namespace("default").Priority(10).MinCount(1).ParentCompositePodGroup("cpg-no-pdb").WorkloadRef("t1", "wl2").Obj(),
			},
			initialPods: []*v1.Pod{
				st.MakePod().Name("pod-pdb-1").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("pg-pdb").Node("node1").Label("app", "foo").ZeroTerminationGracePeriod().Priority(10).Obj(),
				st.MakePod().Name("pod-pdb-2").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("pg-pdb").Node("node1").Label("app", "foo").ZeroTerminationGracePeriod().Priority(10).Obj(),
				st.MakePod().Name("pod-no-pdb-1").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("pg-no-pdb").Node("node1").ZeroTerminationGracePeriod().Priority(10).Obj(),
			},
			pdb: &policyv1.PodDisruptionBudget{
				ObjectMeta: metav1.ObjectMeta{Name: "pdb"},
				Spec: policyv1.PodDisruptionBudgetSpec{
					MinAvailable: &intstr.IntOrString{IntVal: 2},
					Selector:     &metav1.LabelSelector{MatchLabels: map[string]string{"app": "foo"}},
				},
			},
			preemptorPods: []*v1.Pod{
				st.MakePod().Name("high-preemptor").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").ZeroTerminationGracePeriod().Priority(100).Obj(),
			},
			expectedScheduled:              []string{"high-preemptor", "pod-pdb-1", "pod-pdb-2"},
			expectedPreempted:              []string{"pod-no-pdb-1"},
			expectedPodsPreemptedByWAP:     1,
			enablePodGroupPreemptionPolicy: false,
		},
		{
			name: "PDB Violation reprieves only the child PG (DisruptionModeSingle)",
			nodes: []*v1.Node{
				st.MakeNode().Name("node1").Capacity(map[v1.ResourceName]string{v1.ResourceCPU: "3", v1.ResourceMemory: "4Gi", v1.ResourcePods: "32"}).Obj(),
			},
			compositePodGroups: []*schedulingv1alpha3.CompositePodGroup{
				st.MakeCompositePodGroup().Name("cpg-single").Namespace("default").Priority(10).BasicPolicy().DisruptionModeSingle().WorkloadRef("wl1", "t1").Obj(),
			},
			podGroups: []*schedulingv1beta1.PodGroup{
				st.MakePodGroup().Name("pg-pdb").Namespace("default").Priority(10).MinCount(2).DisruptionModeSingle().ParentCompositePodGroup("cpg-single").WorkloadRef("t1", "wl1").Obj(),
				st.MakePodGroup().Name("pg-no-pdb").Namespace("default").Priority(10).MinCount(1).DisruptionModeSingle().ParentCompositePodGroup("cpg-single").WorkloadRef("t1", "wl1").Obj(),
			},
			initialPods: []*v1.Pod{
				st.MakePod().Name("pod-pdb-1").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("pg-pdb").Node("node1").Label("app", "foo").ZeroTerminationGracePeriod().Priority(10).Obj(),
				st.MakePod().Name("pod-pdb-2").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("pg-pdb").Node("node1").Label("app", "foo").ZeroTerminationGracePeriod().Priority(10).Obj(),
				st.MakePod().Name("pod-no-pdb-1").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("pg-no-pdb").Node("node1").ZeroTerminationGracePeriod().Priority(10).Obj(),
			},
			pdb: &policyv1.PodDisruptionBudget{
				ObjectMeta: metav1.ObjectMeta{Name: "pdb"},
				Spec: policyv1.PodDisruptionBudgetSpec{
					MinAvailable: &intstr.IntOrString{IntVal: 2},
					Selector:     &metav1.LabelSelector{MatchLabels: map[string]string{"app": "foo"}},
				},
			},
			preemptorPods: []*v1.Pod{
				st.MakePod().Name("high-preemptor").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").ZeroTerminationGracePeriod().Priority(100).Obj(),
			},
			expectedScheduled:              []string{"high-preemptor", "pod-pdb-1", "pod-pdb-2"},
			expectedPreempted:              []string{"pod-no-pdb-1"},
			expectedPodsPreemptedByWAP:     1,
			enablePodGroupPreemptionPolicy: false,
		},
		{
			name: "3-Level Nested CPG resolves DisruptionMode up to the grandparent",
			nodes: []*v1.Node{
				st.MakeNode().Name("node1").Label("kubernetes.io/hostname", "node1").Capacity(map[v1.ResourceName]string{v1.ResourceCPU: "2", v1.ResourceMemory: "4Gi", v1.ResourcePods: "32"}).Obj(),
				st.MakeNode().Name("node2").Label("kubernetes.io/hostname", "node2").Capacity(map[v1.ResourceName]string{v1.ResourceCPU: "2", v1.ResourceMemory: "4Gi", v1.ResourcePods: "32"}).Obj(),
			},
			compositePodGroups: []*schedulingv1alpha3.CompositePodGroup{
				st.MakeCompositePodGroup().Name("cpg-grandparent").Namespace("default").Priority(10).BasicPolicy().DisruptionModeAll().WorkloadRef("wl1", "t1").Obj(),
				st.MakeCompositePodGroup().Name("cpg-parent").Namespace("default").Priority(10).BasicPolicy().DisruptionModeSingle().ParentCompositePodGroup("cpg-grandparent").WorkloadRef("wl1", "t1").Obj(),
			},
			podGroups: []*schedulingv1beta1.PodGroup{
				st.MakePodGroup().Name("pg1").Namespace("default").Priority(10).MinCount(2).DisruptionModeSingle().ParentCompositePodGroup("cpg-parent").WorkloadRef("t1", "wl1").Obj(),
				st.MakePodGroup().Name("pg2").Namespace("default").Priority(10).MinCount(1).DisruptionModeSingle().ParentCompositePodGroup("cpg-parent").WorkloadRef("t1", "wl1").Obj(),
			},
			initialPods: []*v1.Pod{
				st.MakePod().Name("low-1").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("pg1").Node("node1").ZeroTerminationGracePeriod().Priority(10).Obj(),
				st.MakePod().Name("low-2").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("pg1").Node("node2").ZeroTerminationGracePeriod().Priority(10).Obj(),
				st.MakePod().Name("low-3").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("pg2").Node("node2").ZeroTerminationGracePeriod().Priority(10).Obj(),
			},
			preemptorPods: []*v1.Pod{
				st.MakePod().Name("high-1").Req(map[v1.ResourceName]string{v1.ResourceCPU: "2"}).Container("image").NodeSelector(map[string]string{"kubernetes.io/hostname": "node1"}).ZeroTerminationGracePeriod().Priority(100).Obj(),
			},
			expectedScheduled:              []string{"high-1"},
			expectedPreempted:              []string{"low-1", "low-2", "low-3"},
			expectedPodsPreemptedByWAP:     3,
			enablePodGroupPreemptionPolicy: false,
		},
		{
			name: "Hierarchical DisruptionMode - Grandparent Single, Parent All",
			nodes: []*v1.Node{
				st.MakeNode().Name("node1").Label("kubernetes.io/hostname", "node1").Capacity(map[v1.ResourceName]string{v1.ResourceCPU: "2", v1.ResourceMemory: "4Gi", v1.ResourcePods: "32"}).Obj(),
				st.MakeNode().Name("node2").Label("kubernetes.io/hostname", "node2").Capacity(map[v1.ResourceName]string{v1.ResourceCPU: "2", v1.ResourceMemory: "4Gi", v1.ResourcePods: "32"}).Obj(),
			},
			compositePodGroups: []*schedulingv1alpha3.CompositePodGroup{
				st.MakeCompositePodGroup().Name("cpg-grandparent").Namespace("default").Priority(10).BasicPolicy().DisruptionModeSingle().WorkloadRef("wl1", "t1").Obj(),
				st.MakeCompositePodGroup().Name("cpg-parent").Namespace("default").Priority(10).BasicPolicy().DisruptionModeAll().ParentCompositePodGroup("cpg-grandparent").WorkloadRef("wl1", "t1").Obj(),
			},
			podGroups: []*schedulingv1beta1.PodGroup{
				st.MakePodGroup().Name("pg1").Namespace("default").Priority(10).MinCount(2).DisruptionModeSingle().ParentCompositePodGroup("cpg-parent").WorkloadRef("t1", "wl1").Obj(),
				st.MakePodGroup().Name("pg2").Namespace("default").Priority(10).MinCount(1).DisruptionModeSingle().ParentCompositePodGroup("cpg-grandparent").WorkloadRef("t1", "wl1").Obj(),
			},
			initialPods: []*v1.Pod{
				st.MakePod().Name("low-1").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("pg1").Node("node1").ZeroTerminationGracePeriod().Priority(10).Obj(),
				st.MakePod().Name("low-2").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("pg1").Node("node2").ZeroTerminationGracePeriod().Priority(10).Obj(),
				st.MakePod().Name("low-3").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("pg2").Node("node2").ZeroTerminationGracePeriod().Priority(10).Obj(),
			},
			preemptorPods: []*v1.Pod{
				st.MakePod().Name("high-1").Req(map[v1.ResourceName]string{v1.ResourceCPU: "2"}).Container("image").NodeSelector(map[string]string{"kubernetes.io/hostname": "node1"}).ZeroTerminationGracePeriod().Priority(100).Obj(),
			},
			expectedScheduled:              []string{"high-1", "low-3"},
			expectedPreempted:              []string{"low-1", "low-2"},
			expectedPodsPreemptedByWAP:     2,
			enablePodGroupPreemptionPolicy: false,
		},
		{
			name: "CPG with default preemption policy performs preemption, with PodGroupPreemptionPolicy enabled",
			nodes: []*v1.Node{
				st.MakeNode().Name("node1").Capacity(map[v1.ResourceName]string{v1.ResourceCPU: "3", v1.ResourceMemory: "4Gi", v1.ResourcePods: "32"}).Obj(),
			},
			compositePodGroups: []*schedulingv1alpha3.CompositePodGroup{
				st.MakeCompositePodGroup().Name("cpg1").Namespace("default").Priority(100).BasicPolicy().WorkloadRef("wl1", "t1").Obj(),
			},
			podGroups: []*schedulingv1beta1.PodGroup{
				st.MakePodGroup().Name("pg1").Namespace("default").Priority(100).MinCount(3).ParentCompositePodGroup("cpg1").WorkloadRef("t1", "wl1").Obj(),
			},
			initialPods: []*v1.Pod{
				st.MakePod().Name("low-1").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").ZeroTerminationGracePeriod().Priority(10).Obj(),
				st.MakePod().Name("low-2").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").ZeroTerminationGracePeriod().Priority(10).Obj(),
				st.MakePod().Name("low-3").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").ZeroTerminationGracePeriod().Priority(10).Obj(),
			},
			preemptorPods: []*v1.Pod{
				st.MakePod().Name("high-1").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("pg1").ZeroTerminationGracePeriod().Priority(100).Obj(),
				st.MakePod().Name("high-2").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("pg1").ZeroTerminationGracePeriod().Priority(100).Obj(),
				st.MakePod().Name("high-3").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("pg1").ZeroTerminationGracePeriod().Priority(100).Obj(),
			},
			expectedScheduled:              []string{"high-1", "high-2", "high-3"},
			expectedPreempted:              []string{"low-1", "low-2", "low-3"},
			expectedPodsPreemptedByWAP:     3,
			enablePodGroupPreemptionPolicy: true,
		},
		{
			name: "CPG with PreemptNever Policy",
			nodes: []*v1.Node{
				st.MakeNode().Name("node1").Capacity(map[v1.ResourceName]string{v1.ResourceCPU: "3", v1.ResourceMemory: "4Gi", v1.ResourcePods: "32"}).Obj(),
			},
			compositePodGroups: []*schedulingv1alpha3.CompositePodGroup{
				st.MakeCompositePodGroup().Name("cpg1").Namespace("default").Priority(100).BasicPolicy().WorkloadRef("wl1", "t1").PreemptionPolicy(schedulingv1alpha3.PreemptNever).Obj(),
			},
			podGroups: []*schedulingv1beta1.PodGroup{
				st.MakePodGroup().Name("pg1").Namespace("default").Priority(100).MinCount(2).ParentCompositePodGroup("cpg1").WorkloadRef("t1", "wl1").PreemptionPolicy(schedulingv1beta1.PreemptNever).Obj(),
				st.MakePodGroup().Name("pg2").Namespace("default").Priority(100).MinCount(1).ParentCompositePodGroup("cpg1").WorkloadRef("t1", "wl1").PreemptionPolicy(schedulingv1beta1.PreemptNever).Obj(),
			},
			initialPods: []*v1.Pod{
				st.MakePod().Name("low-1").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").ZeroTerminationGracePeriod().Priority(10).Obj(),
				st.MakePod().Name("low-2").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").ZeroTerminationGracePeriod().Priority(10).Obj(),
				st.MakePod().Name("low-3").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").ZeroTerminationGracePeriod().Priority(10).Obj(),
			},
			preemptorPods: []*v1.Pod{
				st.MakePod().Name("high-1").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("pg1").ZeroTerminationGracePeriod().Priority(100).PreemptionPolicy(v1.PreemptNever).Obj(),
				st.MakePod().Name("high-2").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("pg1").ZeroTerminationGracePeriod().Priority(100).PreemptionPolicy(v1.PreemptNever).Obj(),
				st.MakePod().Name("high-3").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("pg2").ZeroTerminationGracePeriod().Priority(100).PreemptionPolicy(v1.PreemptNever).Obj(),
			},
			expectedScheduled:              []string{"low-1", "low-2", "low-3"},
			expectedPreempted:              []string{},
			expectedUnschedulable:          []string{"high-1", "high-2", "high-3"},
			expectedPodsPreemptedByWAP:     0,
			enablePodGroupPreemptionPolicy: true,
		},
		{
			name: "CPG with PreemptLowerPriority Policy",
			nodes: []*v1.Node{
				st.MakeNode().Name("node1").Capacity(map[v1.ResourceName]string{v1.ResourceCPU: "3", v1.ResourceMemory: "4Gi", v1.ResourcePods: "32"}).Obj(),
			},
			compositePodGroups: []*schedulingv1alpha3.CompositePodGroup{
				st.MakeCompositePodGroup().Name("cpg1").Namespace("default").Priority(100).BasicPolicy().WorkloadRef("wl1", "t1").PreemptionPolicy(schedulingv1alpha3.PreemptLowerPriority).Obj(),
			},
			podGroups: []*schedulingv1beta1.PodGroup{
				st.MakePodGroup().Name("pg1").Namespace("default").Priority(100).MinCount(2).ParentCompositePodGroup("cpg1").WorkloadRef("t1", "wl1").PreemptionPolicy(schedulingv1beta1.PreemptLowerPriority).Obj(),
				st.MakePodGroup().Name("pg2").Namespace("default").Priority(100).MinCount(1).ParentCompositePodGroup("cpg1").WorkloadRef("t1", "wl1").PreemptionPolicy(schedulingv1beta1.PreemptLowerPriority).Obj(),
			},
			initialPods: []*v1.Pod{
				st.MakePod().Name("low-1").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").ZeroTerminationGracePeriod().Priority(10).Obj(),
				st.MakePod().Name("low-2").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").ZeroTerminationGracePeriod().Priority(10).Obj(),
				st.MakePod().Name("low-3").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").ZeroTerminationGracePeriod().Priority(10).Obj(),
			},
			preemptorPods: []*v1.Pod{
				st.MakePod().Name("high-1").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("pg1").ZeroTerminationGracePeriod().Priority(100).PreemptionPolicy(v1.PreemptLowerPriority).Obj(),
				st.MakePod().Name("high-2").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("pg1").ZeroTerminationGracePeriod().Priority(100).PreemptionPolicy(v1.PreemptLowerPriority).Obj(),
				st.MakePod().Name("high-3").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("pg2").ZeroTerminationGracePeriod().Priority(100).PreemptionPolicy(v1.PreemptLowerPriority).Obj(),
			},
			expectedScheduled:              []string{"high-1", "high-2", "high-3"},
			expectedPreempted:              []string{"low-1", "low-2", "low-3"},
			expectedToHaveNNNInfo:          []string{"high-1", "high-2", "high-3"},
			expectedPodsPreemptedByWAP:     3,
			enablePodGroupPreemptionPolicy: true,
		},
		{
			name: "CPG PreemptionPolicy PreemptNever prevents preemption",
			nodes: []*v1.Node{
				st.MakeNode().Name("node1").Capacity(map[v1.ResourceName]string{v1.ResourceCPU: "3", v1.ResourceMemory: "4Gi", v1.ResourcePods: "32"}).Obj(),
			},
			compositePodGroups: []*schedulingv1alpha3.CompositePodGroup{
				st.MakeCompositePodGroup().Name("cpg-preempt-never").Namespace("default").Priority(100).BasicPolicy().WorkloadRef("wl1", "t1").PreemptionPolicy(schedulingv1alpha3.PreemptNever).Obj(),
			},
			podGroups: []*schedulingv1beta1.PodGroup{
				st.MakePodGroup().Name("pg-preempt-never").Namespace("default").Priority(100).MinCount(2).ParentCompositePodGroup("cpg-preempt-never").WorkloadRef("t1", "wl1").PreemptionPolicy(schedulingv1beta1.PreemptNever).Obj(),
			},
			initialPods: []*v1.Pod{
				st.MakePod().Name("low-1").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").ZeroTerminationGracePeriod().Priority(10).Node("node1").Obj(),
				st.MakePod().Name("low-2").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").ZeroTerminationGracePeriod().Priority(10).Node("node1").Obj(),
				st.MakePod().Name("low-3").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").ZeroTerminationGracePeriod().Priority(10).Node("node1").Obj(),
			},
			preemptorPods: []*v1.Pod{
				st.MakePod().Name("high-1").Req(map[v1.ResourceName]string{v1.ResourceCPU: "2"}).Container("image").PodGroupName("pg-preempt-never").ZeroTerminationGracePeriod().Priority(100).PreemptionPolicy(v1.PreemptNever).Obj(),
				st.MakePod().Name("high-2").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("pg-preempt-never").ZeroTerminationGracePeriod().Priority(100).PreemptionPolicy(v1.PreemptNever).Obj(),
			},
			expectedScheduled:              []string{"low-1", "low-2", "low-3"},
			expectedPreempted:              []string{},
			expectedUnschedulable:          []string{"high-1", "high-2"},
			expectedPodsPreemptedByWAP:     0,
			enablePodGroupPreemptionPolicy: true,
		},
		{
			name: "CPG PreemptionPolicy PreemptLowerPriority allows preemption",
			nodes: []*v1.Node{
				st.MakeNode().Name("node1").Capacity(map[v1.ResourceName]string{v1.ResourceCPU: "3", v1.ResourceMemory: "4Gi", v1.ResourcePods: "32"}).Obj(),
			},
			compositePodGroups: []*schedulingv1alpha3.CompositePodGroup{
				st.MakeCompositePodGroup().Name("cpg-preempt-lower").Namespace("default").Priority(100).BasicPolicy().WorkloadRef("wl1", "t1").PreemptionPolicy(schedulingv1alpha3.PreemptLowerPriority).Obj(),
			},
			podGroups: []*schedulingv1beta1.PodGroup{
				st.MakePodGroup().Name("pg-preempt-lower").Namespace("default").Priority(100).MinCount(2).ParentCompositePodGroup("cpg-preempt-lower").WorkloadRef("t1", "wl1").PreemptionPolicy(schedulingv1beta1.PreemptLowerPriority).Obj(),
			},
			initialPods: []*v1.Pod{
				st.MakePod().Name("low-1").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").ZeroTerminationGracePeriod().Priority(10).Node("node1").Obj(),
				st.MakePod().Name("low-2").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").ZeroTerminationGracePeriod().Priority(10).Node("node1").Obj(),
				st.MakePod().Name("low-3").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").ZeroTerminationGracePeriod().Priority(10).Node("node1").Obj(),
			},
			preemptorPods: []*v1.Pod{
				st.MakePod().Name("high-1").Req(map[v1.ResourceName]string{v1.ResourceCPU: "2"}).Container("image").PodGroupName("pg-preempt-lower").ZeroTerminationGracePeriod().Priority(100).PreemptionPolicy(v1.PreemptLowerPriority).Obj(),
				st.MakePod().Name("high-2").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("pg-preempt-lower").ZeroTerminationGracePeriod().Priority(100).PreemptionPolicy(v1.PreemptLowerPriority).Obj(),
			},
			expectedScheduled:              []string{"high-1", "high-2"},
			expectedPreempted:              []string{"low-1", "low-2", "low-3"},
			expectedToHaveNNNInfo:          []string{"high-1", "high-2"},
			expectedPodsPreemptedByWAP:     3,
			enablePodGroupPreemptionPolicy: true,
		},
		{
			name: "CPG with PreemptNever preemption policy in all pods does not perform preemption, with PodGroupPreemptionPolicy disabled",
			nodes: []*v1.Node{
				st.MakeNode().Name("node1").Capacity(map[v1.ResourceName]string{v1.ResourceCPU: "3", v1.ResourceMemory: "4Gi", v1.ResourcePods: "32"}).Obj(),
			},
			compositePodGroups: []*schedulingv1alpha3.CompositePodGroup{
				st.MakeCompositePodGroup().Name("cpg1").Namespace("default").Priority(100).BasicPolicy().WorkloadRef("wl1", "t1").Obj(),
			},
			podGroups: []*schedulingv1beta1.PodGroup{
				st.MakePodGroup().Name("pg1").Namespace("default").Priority(100).MinCount(3).ParentCompositePodGroup("cpg1").WorkloadRef("t1", "wl1").Obj(),
			},
			initialPods: []*v1.Pod{
				st.MakePod().Name("low-1").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").ZeroTerminationGracePeriod().Priority(10).Obj(),
				st.MakePod().Name("low-2").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").ZeroTerminationGracePeriod().Priority(10).Obj(),
				st.MakePod().Name("low-3").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").ZeroTerminationGracePeriod().Priority(10).Obj(),
			},
			preemptorPods: []*v1.Pod{
				st.MakePod().Name("high-1").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("pg1").ZeroTerminationGracePeriod().Priority(100).PreemptionPolicy(v1.PreemptNever).Obj(),
				st.MakePod().Name("high-2").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("pg1").ZeroTerminationGracePeriod().Priority(100).PreemptionPolicy(v1.PreemptNever).Obj(),
				st.MakePod().Name("high-3").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("pg1").ZeroTerminationGracePeriod().Priority(100).PreemptionPolicy(v1.PreemptNever).Obj(),
			},
			expectedScheduled:              []string{"low-1", "low-2", "low-3"},
			expectedPreempted:              []string{},
			expectedUnschedulable:          []string{"high-1", "high-2", "high-3"},
			expectedPodsPreemptedByWAP:     0,
			enablePodGroupPreemptionPolicy: false,
		},
		{
			name: "CPG Gang scheduling: preemption with pod anti-affinity constraints",
			nodes: []*v1.Node{
				st.MakeNode().Name("node1").Label("kubernetes.io/hostname", "node1").Capacity(map[v1.ResourceName]string{v1.ResourceCPU: "2", v1.ResourceMemory: "4Gi", v1.ResourcePods: "32"}).Obj(),
				st.MakeNode().Name("node2").Label("kubernetes.io/hostname", "node2").Capacity(map[v1.ResourceName]string{v1.ResourceCPU: "2", v1.ResourceMemory: "4Gi", v1.ResourcePods: "32"}).Obj(),
			},
			compositePodGroups: []*schedulingv1alpha3.CompositePodGroup{
				st.MakeCompositePodGroup().Name("preemptor-cpg").Namespace("default").Priority(100).BasicPolicy().WorkloadRef("wl1", "t1").Obj(),
			},
			podGroups: []*schedulingv1beta1.PodGroup{
				st.MakePodGroup().Name("preemptor-pg").Namespace("default").Priority(100).MinCount(2).ParentCompositePodGroup("preemptor-cpg").WorkloadRef("t1", "wl1").Obj(),
			},
			initialPods: []*v1.Pod{
				st.MakePod().Name("initial-pod").Label("app", "initial").Node("node1").Req(map[v1.ResourceName]string{v1.ResourceCPU: "0.25"}).Container("image").ZeroTerminationGracePeriod().Priority(10).Obj(),
			},
			preemptorPods: []*v1.Pod{
				st.MakePod().Name("preemptor-1").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1.5"}).Container("image").PodGroupName("preemptor-pg").ZeroTerminationGracePeriod().Priority(100).
					PodAntiAffinityExists("app", "kubernetes.io/hostname", st.PodAntiAffinityWithRequiredReq).Obj(),
				st.MakePod().Name("preemptor-2").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1.5"}).Container("image").PodGroupName("preemptor-pg").ZeroTerminationGracePeriod().Priority(100).
					PodAntiAffinityExists("app", "kubernetes.io/hostname", st.PodAntiAffinityWithRequiredReq).Obj(),
			},
			expectedScheduled:              []string{"preemptor-1", "preemptor-2"},
			expectedPreempted:              []string{"initial-pod"},
			expectedPodsPreemptedByWAP:     1,
			enablePodGroupPreemptionPolicy: true,
		},
		{
			name: "CPG Gang scheduling: preemption with pod node port",
			nodes: []*v1.Node{
				st.MakeNode().Name("node1").Capacity(map[v1.ResourceName]string{v1.ResourceCPU: "2", v1.ResourceMemory: "4Gi", v1.ResourcePods: "32"}).Obj(),
				st.MakeNode().Name("node2").Capacity(map[v1.ResourceName]string{v1.ResourceCPU: "2", v1.ResourceMemory: "4Gi", v1.ResourcePods: "32"}).Obj(),
			},
			compositePodGroups: []*schedulingv1alpha3.CompositePodGroup{
				st.MakeCompositePodGroup().Name("preemptor-cpg").Namespace("default").Priority(100).BasicPolicy().WorkloadRef("wl1", "t1").Obj(),
			},
			podGroups: []*schedulingv1beta1.PodGroup{
				st.MakePodGroup().Name("preemptor-pg").Namespace("default").Priority(100).MinCount(2).ParentCompositePodGroup("preemptor-cpg").WorkloadRef("t1", "wl1").Obj(),
			},
			initialPods: []*v1.Pod{
				st.MakePod().Name("initial-pod").ContainerPort([]v1.ContainerPort{{ContainerPort: 8080, HostPort: 8080}}).Node("node1").Req(map[v1.ResourceName]string{v1.ResourceCPU: "0.25"}).Container("image").ZeroTerminationGracePeriod().Priority(10).Obj(),
			},
			preemptorPods: []*v1.Pod{
				st.MakePod().Name("preemptor-1").ContainerPort([]v1.ContainerPort{{ContainerPort: 8080, HostPort: 8080}}).Req(map[v1.ResourceName]string{v1.ResourceCPU: "1.5"}).Container("image").PodGroupName("preemptor-pg").ZeroTerminationGracePeriod().Priority(100).Obj(),
				st.MakePod().Name("preemptor-2").ContainerPort([]v1.ContainerPort{{ContainerPort: 8080, HostPort: 8080}}).Req(map[v1.ResourceName]string{v1.ResourceCPU: "1.5"}).Container("image").PodGroupName("preemptor-pg").ZeroTerminationGracePeriod().Priority(100).Obj(),
			},
			expectedScheduled:              []string{"preemptor-1", "preemptor-2"},
			expectedPreempted:              []string{"initial-pod"},
			expectedPodsPreemptedByWAP:     1,
			enablePodGroupPreemptionPolicy: true,
		},
		{
			name: "Reserve plugins are called during preemption simulation, so second pod fails",
			nodes: []*v1.Node{
				st.MakeNode().Name("node1").Capacity(map[v1.ResourceName]string{v1.ResourceCPU: "2", v1.ResourceMemory: "4Gi", v1.ResourcePods: "32"}).Obj(),
			},
			compositePodGroups: []*schedulingv1alpha3.CompositePodGroup{
				st.MakeCompositePodGroup().Name("cpg-victim").Namespace("default").Priority(10).BasicPolicy().WorkloadRef("wl1", "t1").Obj(),
				st.MakeCompositePodGroup().Name("cpg-preemptor").Namespace("default").Priority(100).BasicPolicy().WorkloadRef("wl2", "t2").Obj(),
			},
			podGroups: []*schedulingv1beta1.PodGroup{
				st.MakePodGroup().Name("pg-victim").Namespace("default").Priority(10).MinCount(2).ParentCompositePodGroup("cpg-victim").WorkloadRef("t1", "wl1").Obj(),
				st.MakePodGroup().Name("pg-preemptor").Namespace("default").Priority(100).MinCount(2).ParentCompositePodGroup("cpg-preemptor").WorkloadRef("t2", "wl2").Obj(),
			},
			initialPods: []*v1.Pod{
				st.MakePod().Name("low-1").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("pg-victim").Node("node1").ZeroTerminationGracePeriod().Priority(10).Obj(),
				st.MakePod().Name("low-2").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("pg-victim").Node("node1").ZeroTerminationGracePeriod().Priority(10).Obj(),
			},
			preemptorPods: []*v1.Pod{
				st.MakePod().Name("high-1").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("pg-preemptor").Label("test-plugin", "true").ZeroTerminationGracePeriod().Priority(100).Obj(),
				st.MakePod().Name("high-2").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("pg-preemptor").Label("test-plugin", "true").ZeroTerminationGracePeriod().Priority(100).Obj(),
			},
			expectedScheduled:     []string{},
			expectedPreempted:     []string{},
			expectedUnschedulable: []string{"high-1", "high-2"},
			customPluginName:      "mockReservePlugin",
			customPluginFunc: func(ctx context.Context, _ runtime.Object, _ fwk.Handle) (fwk.Plugin, error) {
				return &mockReservePlugin{maxPods: 1}, nil
			},
		},
		{
			name: "Binding first before preemption for gang policy",
			nodes: []*v1.Node{
				st.MakeNode().Name("node1").Capacity(map[v1.ResourceName]string{v1.ResourceCPU: "2", v1.ResourceMemory: "4Gi", v1.ResourcePods: "32"}).Obj(),
				st.MakeNode().Name("node2").Capacity(map[v1.ResourceName]string{v1.ResourceCPU: "2", v1.ResourceMemory: "4Gi", v1.ResourcePods: "32"}).Obj(),
			},
			compositePodGroups: []*schedulingv1alpha3.CompositePodGroup{
				st.MakeCompositePodGroup().Name("cpg-victim").Namespace("default").Priority(10).MinGroupCount(1).WorkloadRef("wl1", "t1").Obj(),
				st.MakeCompositePodGroup().Name("cpg-preemptor").Namespace("default").Priority(100).MinGroupCount(1).WorkloadRef("wl2", "t2").Obj(),
			},
			podGroups: []*schedulingv1beta1.PodGroup{
				st.MakePodGroup().Name("pg-victim").Namespace("default").Priority(10).MinCount(4).ParentCompositePodGroup("cpg-victim").WorkloadRef("t1", "wl1").Obj(),
				st.MakePodGroup().Name("pg-preemptor").Namespace("default").Priority(100).MinCount(4).ParentCompositePodGroup("cpg-preemptor").WorkloadRef("t2", "wl2").Obj(),
			},
			initialPods: []*v1.Pod{
				st.MakePod().Name("low-1").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("pg-victim").Node("node1").ZeroTerminationGracePeriod().Priority(10).Obj(),
				st.MakePod().Name("low-2").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("pg-victim").Node("node2").ZeroTerminationGracePeriod().Priority(10).Obj(),
			},
			preemptorPods: []*v1.Pod{
				st.MakePod().Name("high-1").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("pg-preemptor").ZeroTerminationGracePeriod().Priority(100).Obj(),
				st.MakePod().Name("high-2").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("pg-preemptor").ZeroTerminationGracePeriod().Priority(100).Obj(),
				st.MakePod().Name("high-3").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("pg-preemptor").ZeroTerminationGracePeriod().Priority(100).Obj(),
				st.MakePod().Name("high-4").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("pg-preemptor").ZeroTerminationGracePeriod().Priority(100).Obj(),
			},
			expectedScheduled:              []string{"high-1", "high-2", "high-3", "high-4"},
			expectedPreempted:              []string{"low-1", "low-2"},
			expectedPodsPreemptedByWAP:     2,
			enablePodGroupPreemptionPolicy: false,
			tempRemoveCPG:                  true,
		},
		{
			name: "Binding first before preemption for CPG",
			nodes: []*v1.Node{
				st.MakeNode().Name("node1").Capacity(map[v1.ResourceName]string{v1.ResourceCPU: "2", v1.ResourceMemory: "4Gi", v1.ResourcePods: "32"}).Obj(),
				st.MakeNode().Name("node2").Capacity(map[v1.ResourceName]string{v1.ResourceCPU: "2", v1.ResourceMemory: "4Gi", v1.ResourcePods: "32"}).Obj(),
			},
			compositePodGroups: []*schedulingv1alpha3.CompositePodGroup{
				st.MakeCompositePodGroup().Name("cpg-victim").Namespace("default").Priority(10).BasicPolicy().WorkloadRef("wl1", "t1").Obj(),
				st.MakeCompositePodGroup().Name("cpg-preemptor").Namespace("default").Priority(100).BasicPolicy().WorkloadRef("wl2", "t2").Obj(),
			},
			podGroups: []*schedulingv1beta1.PodGroup{
				st.MakePodGroup().Name("pg-victim").Namespace("default").Priority(10).MinCount(4).ParentCompositePodGroup("cpg-victim").WorkloadRef("t1", "wl1").Obj(),
				st.MakePodGroup().Name("pg-preemptor").Namespace("default").Priority(100).MinCount(4).ParentCompositePodGroup("cpg-preemptor").WorkloadRef("t2", "wl2").Obj(),
			},
			initialPods: []*v1.Pod{
				st.MakePod().Name("low-1").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("pg-victim").Node("node1").ZeroTerminationGracePeriod().Priority(10).Obj(),
				st.MakePod().Name("low-2").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("pg-victim").Node("node2").ZeroTerminationGracePeriod().Priority(10).Obj(),
			},
			preemptorPods: []*v1.Pod{
				st.MakePod().Name("high-1").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("pg-preemptor").ZeroTerminationGracePeriod().Priority(100).Obj(),
				st.MakePod().Name("high-2").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("pg-preemptor").ZeroTerminationGracePeriod().Priority(100).Obj(),
				st.MakePod().Name("high-3").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("pg-preemptor").ZeroTerminationGracePeriod().Priority(100).Obj(),
				st.MakePod().Name("high-4").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("pg-preemptor").ZeroTerminationGracePeriod().Priority(100).Obj(),
			},
			expectedScheduled:              []string{"high-1", "high-2", "high-3", "high-4"},
			expectedPreempted:              []string{"low-1", "low-2"},
			expectedPodsPreemptedByWAP:     2,
			enablePodGroupPreemptionPolicy: false,
		},
		{
			name: "CPG Missing Parent Reference Fallback",
			nodes: []*v1.Node{
				st.MakeNode().Name("node1").Capacity(map[v1.ResourceName]string{v1.ResourceCPU: "2", v1.ResourceMemory: "4Gi", v1.ResourcePods: "32"}).Obj(),
			},
			compositePodGroups: []*schedulingv1alpha3.CompositePodGroup{
				st.MakeCompositePodGroup().Name("cpg-victim").Namespace("default").Priority(10).BasicPolicy().WorkloadRef("wl1", "t1").Obj(),
				st.MakeCompositePodGroup().Name("cpg-preemptor").Namespace("default").Priority(100).BasicPolicy().WorkloadRef("wl2", "t1").Obj(),
			},
			podGroups: []*schedulingv1beta1.PodGroup{
				st.MakePodGroup().Name("pg-victim").Namespace("default").Priority(10).MinCount(2).ParentCompositePodGroup("cpg-victim").WorkloadRef("t1", "wl1").Obj(),
				st.MakePodGroup().Name("pg-preemptor").Namespace("default").Priority(100).MinCount(2).ParentCompositePodGroup("cpg-preemptor").WorkloadRef("t1", "wl2").Obj(),
			},
			initialPods: []*v1.Pod{
				st.MakePod().Name("low-1").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("pg-victim").ZeroTerminationGracePeriod().Priority(10).Node("node1").Obj(),
				st.MakePod().Name("low-2").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("pg-victim").ZeroTerminationGracePeriod().Priority(10).Node("node1").Obj(),
			},
			preemptorPods: []*v1.Pod{
				st.MakePod().Name("high-1").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("pg-preemptor").ZeroTerminationGracePeriod().Priority(100).Obj(),
				st.MakePod().Name("high-2").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("pg-preemptor").ZeroTerminationGracePeriod().Priority(100).Obj(),
			},
			expectedScheduled:             []string{"high-1", "high-2"},
			expectedPreempted:             []string{"low-1", "low-2"},
			expectedPodsPreemptedByWAP:    2,
			removeCPGNameBeforePreemption: "cpg-victim",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			featuregatetesting.SetFeatureGatesDuringTest(t, utilfeature.DefaultFeatureGate, featuregatetesting.FeatureOverrides{
				features.GenericWorkload:                 true,
				features.CompositePodGroup:               true,
				features.TopologyAwareWorkloadScheduling: true,
				features.PodGroupPreemptionPolicy:        tt.enablePodGroupPreemptionPolicy,
			})
			registry := make(frameworkruntime.Registry)

			// Register mock bind plugin that will register NNN information during binding.
			mockBindPluginName := "mockBindPlugin"
			var bindPlugin = mockBindPlugin{
				name:       mockBindPluginName,
				realPlugin: nil,
				nnnInfo:    sync.Map{},
			}
			err := registry.Register(mockBindPluginName, func(ctx context.Context, o runtime.Object, fh fwk.Handle) (fwk.Plugin, error) {
				db, err := defaultbinder.New(ctx, o, fh)
				if err != nil {
					t.Fatalf("Error creating a default binder plugin: %v", err)
				}
				bindPlugin.realPlugin = db.(fwk.BindPlugin)
				return &bindPlugin, nil
			})
			if err != nil {
				t.Fatalf("Error registering a bind plugin: %v", err)
			}

			mockPGPostFilterPluginName := "mockPGPostFilterPlugin"
			var pgPostFilterPlugin = mockPodGroupPostFilterPlugin{
				name: mockPGPostFilterPluginName,
			}
			err = registry.Register(mockPGPostFilterPluginName, func(ctx context.Context, o runtime.Object, fh fwk.Handle) (fwk.Plugin, error) {
				return &pgPostFilterPlugin, nil
			})
			if err != nil {
				t.Fatalf("Error registering a pg post filter plugin: %v", err)
			}

			if tt.customPluginFunc != nil {
				err = registry.Register(tt.customPluginName, tt.customPluginFunc)
				if err != nil {
					t.Fatalf("Error registering custom plugin: %v", err)
				}
			}

			cfgV1 := configv1.KubeSchedulerConfiguration{
				Profiles: []configv1.KubeSchedulerProfile{{
					SchedulerName: new(v1.DefaultSchedulerName),
					Plugins: &configv1.Plugins{
						MultiPoint: configv1.PluginSet{
							Enabled: []configv1.Plugin{
								{Name: mockBindPluginName},
								{Name: mockPGPostFilterPluginName},
								{Name: names.DefaultPreemption},
							},
							Disabled: []configv1.Plugin{
								{Name: names.DefaultBinder},
								{Name: names.DefaultPreemption},
							},
						},
					},
				}},
			}
			if tt.customPluginFunc != nil {
				cfgV1.Profiles[0].Plugins.MultiPoint.Enabled = append(cfgV1.Profiles[0].Plugins.MultiPoint.Enabled, configv1.Plugin{Name: tt.customPluginName})
			}
			cfg := configtesting.V1ToInternalWithDefaults(t, cfgV1)

			// Set PodMaxBackoff to 1 second to turn on backoff and allow apiCacher to get information about
			// pod NNN. Without this we might have a race between starting binding and update of apiCacher.
			testCtx := testutils.InitTestSchedulerWithNS(t, "cpg-preemption",
				scheduler.WithProfiles(cfg.Profiles...),
				scheduler.WithFrameworkOutOfTreeRegistry(registry),
				scheduler.WithPodMaxBackoffSeconds(1),
				scheduler.WithPodInitialBackoffSeconds(0))
			cs, ns := testCtx.ClientSet, testCtx.NS.Name

			// Create nodes
			for _, n := range tt.nodes {
				if _, err := cs.CoreV1().Nodes().Create(testCtx.Ctx, n, metav1.CreateOptions{}); err != nil {
					t.Fatalf("Failed to create node %s: %v", n.Name, err)
				}
			}

			// 1. Create CompositePodGroups
			for _, cpg := range tt.compositePodGroups {
				cpg.Namespace = ns
				if _, err := cs.SchedulingV1alpha3().CompositePodGroups(ns).Create(testCtx.Ctx, cpg, metav1.CreateOptions{}); err != nil {
					t.Fatalf("Failed to create CompositePodGroup %s: %v", cpg.Name, err)
				}
			}

			// 2. Create PodGroups
			for _, pg := range tt.podGroups {
				pg.Namespace = ns
				if _, err := cs.SchedulingV1beta1().PodGroups(ns).Create(testCtx.Ctx, pg, metav1.CreateOptions{}); err != nil {
					t.Fatalf("Failed to create PodGroup %s: %v", pg.Name, err)
				}
			}

			// 3. Create PodDisruptionBudget if provided
			if tt.pdb != nil {
				tt.pdb.Namespace = ns
				if _, err := cs.PolicyV1().PodDisruptionBudgets(ns).Create(testCtx.Ctx, tt.pdb, metav1.CreateOptions{}); err != nil {
					t.Fatalf("Failed to create PDB: %v", err)
				}
			}

			// 4. Create initial pods
			for _, p := range tt.initialPods {
				p.Namespace = ns
				if _, err := cs.CoreV1().Pods(ns).Create(testCtx.Ctx, p, metav1.CreateOptions{}); err != nil {
					t.Fatalf("Failed to create pod %s: %v", p.Name, err)
				}
			}
			for _, p := range tt.initialPods {
				if err := wait.PollUntilContextTimeout(testCtx.Ctx, 100*time.Millisecond, 10*time.Second, false,
					testutils.PodScheduled(cs, ns, p.Name)); err != nil {
					t.Errorf("Failed to wait for pod %s to be scheduled: %v", p.Name, err)
				}
			}

			// 5. Create preemptor pods
			if tt.tempRemoveCPG {
				// Temporarily remove CPGs and PGs. This is a trick to ensure that all preemptor pods
				// are created and queued as unschedulable first, and then become schedulable at once
				// when the CPG is recreated.
				cpgNames := make([]string, len(tt.compositePodGroups))
				for i, cpg := range tt.compositePodGroups {
					cpgNames[i] = cpg.Name
				}
				if err := deleteCompositePodGroups(testCtx.Ctx, cs, ns, cpgNames); err != nil {
					t.Fatalf("Failed to delete CompositePodGroups: %v", err)
				}
				pgNames := make([]string, len(tt.podGroups))
				for i, pg := range tt.podGroups {
					pgNames[i] = pg.Name
				}
				if err := deletePodGroups(testCtx.Ctx, cs, ns, pgNames); err != nil {
					t.Fatalf("Failed to delete PodGroups: %v", err)
				}
			}

			if tt.removeCPGNameBeforePreemption != "" {
				if err := cs.SchedulingV1alpha3().CompositePodGroups(ns).Delete(testCtx.Ctx, tt.removeCPGNameBeforePreemption, metav1.DeleteOptions{}); err != nil {
					t.Fatalf("Failed to delete CompositePodGroup %s: %v", tt.removeCPGNameBeforePreemption, err)
				}
			}

			for _, p := range tt.preemptorPods {
				p.Namespace = ns
				if _, err := cs.CoreV1().Pods(ns).Create(testCtx.Ctx, p, metav1.CreateOptions{}); err != nil {
					t.Fatalf("Failed to create pod %s: %v", p.Name, err)
				}
			}

			if tt.tempRemoveCPG {
				// Wait for preemptor pods to be unschedulable
				for _, p := range tt.preemptorPods {
					if err := wait.PollUntilContextTimeout(testCtx.Ctx, 100*time.Millisecond, 10*time.Second, false,
						func(ctx context.Context) (bool, error) {
							return isPodInUnschedulableQueue(testCtx.Scheduler, p.Name, ns), nil
						}); err != nil {
						t.Fatalf("Failed to wait for pod %s to be unschedulable: %v", p.Name, err)
					}
				}

				// Recreate CPGs and PGs
				for _, cpg := range tt.compositePodGroups {
					cpgCopy := cpg.DeepCopy()
					cpgCopy.ResourceVersion = ""
					if _, err := cs.SchedulingV1alpha3().CompositePodGroups(ns).Create(testCtx.Ctx, cpgCopy, metav1.CreateOptions{}); err != nil {
						t.Fatalf("Failed to recreate CompositePodGroup %s: %v", cpg.Name, err)
					}
				}
				for _, pg := range tt.podGroups {
					pgCopy := pg.DeepCopy()
					pgCopy.ResourceVersion = ""
					if _, err := cs.SchedulingV1beta1().PodGroups(ns).Create(testCtx.Ctx, pgCopy, metav1.CreateOptions{}); err != nil {
						t.Fatalf("Failed to recreate PodGroup %s: %v", pg.Name, err)
					}
				}
			}

			// 6. Wait for preemption to complete if WAP calls are expected
			if tt.expectedPodsPreemptedByWAP > 0 {
				wapCalls := 0
				err := wait.PollUntilContextTimeout(testCtx.Ctx, 100*time.Millisecond, 10*time.Second, false, func(ctx context.Context) (bool, error) {
					wapCalls = 0
					for _, podName := range tt.expectedPreempted {
						events, err := cs.CoreV1().Events(ns).List(ctx, metav1.ListOptions{
							FieldSelector: "involvedObject.name=" + podName,
						})
						if err != nil {
							return false, err
						}
						for _, event := range events.Items {
							if event.Reason == "Preempted" && (strings.HasPrefix(event.Message, "Preempted by compositepodgroup") || strings.HasPrefix(event.Message, "Preempted by podgroup") || strings.HasPrefix(event.Message, "Preempted by pod")) {
								wapCalls++
								break
							}
						}
					}
					return wapCalls == tt.expectedPodsPreemptedByWAP, nil
				})
				if err != nil {
					t.Errorf("WorkloadAwarePreemption was not called expected times within timeout: want=%d, got=%d", tt.expectedPodsPreemptedByWAP, wapCalls)
				}
			}

			// 7. Verify unschedulable pods
			for _, podName := range tt.expectedUnschedulable {
				if err := wait.PollUntilContextTimeout(testCtx.Ctx, 100*time.Millisecond, 10*time.Second, false,
					testutils.PodUnschedulable(cs, ns, podName)); err != nil {
					t.Errorf("Pod %s was expected to be unschedulable but wasn't: %v", podName, err)
				}
			}

			// 8. Verify scheduled pods
			for _, podName := range tt.expectedScheduled {
				if err := wait.PollUntilContextTimeout(testCtx.Ctx, 100*time.Millisecond, 10*time.Second, false,
					testutils.PodScheduled(cs, ns, podName)); err != nil {
					t.Errorf("Pod %s was expected to be scheduled but wasn't: %v", podName, err)
				}
			}

			// 9. Verify preempted pods
			for _, podName := range tt.expectedPreempted {
				if err := wait.PollUntilContextTimeout(testCtx.Ctx, 200*time.Millisecond, 5*time.Second, false,
					func(ctx context.Context) (bool, error) {
						pod, err := cs.CoreV1().Pods(ns).Get(ctx, podName, metav1.GetOptions{})
						if err != nil {
							return apierrors.IsNotFound(err), nil
						}
						if pod.DeletionTimestamp != nil {
							return true, nil
						}
						_, cond := podutil.GetPodCondition(&pod.Status, v1.DisruptionTarget)
						return cond != nil, nil
					}); err != nil {
					t.Errorf("Pod %s was expected to be preempted but wasn't", podName)
				}
			}

			// 10. Verify preemptor pods have nominated node name
			for _, podName := range tt.expectedToHaveNNNInfo {
				if node, ok := bindPlugin.nnnInfo.Load(podName); !ok || node.(string) == "" {
					t.Errorf("Pod %s was expected to have nominated node name but didn't", podName)
				}
			}

			// 11. Dump the state of pods to ease debugging failed runs.
			if t.Failed() {
				t.Log("Dumping states of initial and preemptor pods:")
				var allPods []string
				for _, p := range tt.initialPods {
					allPods = append(allPods, p.Name)
				}
				for _, p := range tt.preemptorPods {
					allPods = append(allPods, p.Name)
				}
				for _, podName := range allPods {
					pod, err := cs.CoreV1().Pods(ns).Get(testCtx.Ctx, podName, metav1.GetOptions{})
					if err != nil {
						if apierrors.IsNotFound(err) {
							t.Logf("Pod %q: not present in cluster", podName)
						} else {
							t.Logf("Pod %q: failed to get: %v", podName, err)
						}
						continue
					}

					var statusStr string
					if pod.Spec.NodeName != "" {
						statusStr = "scheduled on node " + pod.Spec.NodeName
					} else {
						_, cond := podutil.GetPodCondition(&pod.Status, v1.PodScheduled)
						if cond != nil && cond.Status == v1.ConditionFalse && cond.Reason == v1.PodReasonUnschedulable {
							statusStr = "unschedulable"
						} else {
							statusStr = "pending"
						}
					}
					t.Logf("Pod %q: status=%s, phase=%s", podName, statusStr, pod.Status.Phase)
				}
			}

		})
	}
}

// mockBindPlugin is a fake plugin that registers NNN information during binding.
type mockBindPlugin struct {
	name       string
	realPlugin fwk.BindPlugin
	nnnInfo    sync.Map
	recorder   *eventRecorder
}

func (bp *mockBindPlugin) Name() string {
	return bp.name
}

func (bp *mockBindPlugin) Bind(ctx context.Context, state fwk.CycleState, p *v1.Pod, nodeName string) *fwk.Status {
	if p.Status.NominatedNodeName != "" {
		bp.nnnInfo.Store(p.Name, p.Status.NominatedNodeName)
	}
	if bp.recorder != nil {
		bp.recorder.Record("Bind:" + p.Name)
	}
	return bp.realPlugin.Bind(ctx, state, p, nodeName)
}

var _ fwk.BindPlugin = &mockBindPlugin{}

type eventRecorder struct {
	lock   sync.Mutex
	events []string
}

func (er *eventRecorder) Record(event string) {
	er.lock.Lock()
	defer er.lock.Unlock()
	er.events = append(er.events, event)
}

func (er *eventRecorder) GetEvents() []string {
	er.lock.Lock()
	defer er.lock.Unlock()
	return append([]string(nil), er.events...)
}

func (er *eventRecorder) Clear() {
	er.lock.Lock()
	defer er.lock.Unlock()
	er.events = nil
}

type mockPodGroupPostFilterPlugin struct {
	name     string
	recorder *eventRecorder
}

func (p *mockPodGroupPostFilterPlugin) Name() string {
	return p.name
}

func (p *mockPodGroupPostFilterPlugin) PodGroupPostFilter(ctx context.Context, state fwk.PodGroupCycleState, pgInfo fwk.PodGroupInfo, pgSchedulingFunc fwk.PodGroupSchedulingFunc) (*fwk.PodGroupPostFilterResult, *fwk.Status) {
	if p.recorder != nil {
		p.recorder.Record("PodGroupPostFilter:" + pgInfo.GetName())
	}
	return nil, fwk.NewStatus(fwk.Unschedulable, "injected PodGroupPostFilter log")
}

var _ fwk.PodGroupPostFilterPlugin = &mockPodGroupPostFilterPlugin{}

type mockReservePlugin struct {
	lock          sync.Mutex
	reservedCount int
	maxPods       int
}

func (p *mockReservePlugin) Name() string {
	return "mockReservePlugin"
}

func (p *mockReservePlugin) Reserve(ctx context.Context, state fwk.CycleState, pod *v1.Pod, nodeName string) *fwk.Status {
	if pod.Labels["test-plugin"] != "true" {
		return nil
	}
	p.lock.Lock()
	defer p.lock.Unlock()
	p.reservedCount++
	return nil
}

func (p *mockReservePlugin) Unreserve(ctx context.Context, state fwk.CycleState, pod *v1.Pod, nodeName string) {
	if pod.Labels["test-plugin"] != "true" {
		return
	}
	p.lock.Lock()
	defer p.lock.Unlock()
	p.reservedCount--
}

func (p *mockReservePlugin) Filter(ctx context.Context, state fwk.CycleState, pod *v1.Pod, nodeInfo fwk.NodeInfo) *fwk.Status {
	if pod.Labels["test-plugin"] != "true" {
		return nil
	}
	takenCount := 0
	for _, p := range nodeInfo.GetPods() {
		if p.GetPod().Labels["resource-taken"] == "true" {
			takenCount++
		}
	}

	p.lock.Lock()
	defer p.lock.Unlock()
	if p.reservedCount+takenCount >= p.maxPods {
		return fwk.NewStatus(fwk.Unschedulable, "already reserved")
	}
	return nil
}

var _ fwk.ReservePlugin = &mockReservePlugin{}
var _ fwk.FilterPlugin = &mockReservePlugin{}
var _ fwk.ScorePlugin = &mockScorePlugin{}

type mockScorePlugin struct {
	scores map[string]int64
}

func (p *mockScorePlugin) Name() string {
	return "mockScorePlugin"
}

func (p *mockScorePlugin) Score(ctx context.Context, state fwk.CycleState, pod *v1.Pod, nodeInfo fwk.NodeInfo) (int64, *fwk.Status) {
	if score, ok := p.scores[nodeInfo.Node().Name]; ok {
		return score, nil
	}
	return 0, nil
}

func (p *mockScorePlugin) ScoreExtensions() fwk.ScoreExtensions {
	return nil
}

func newPresetScorePlugin(scores map[string]int64) frameworkruntime.PluginFactory {
	return func(_ context.Context, _ runtime.Object, _ fwk.Handle) (fwk.Plugin, error) {
		return &mockScorePlugin{scores: scores}, nil
	}
}

func isPodInUnschedulableQueue(sched *scheduler.Scheduler, name, namespace string) bool {
	for _, p := range sched.SchedulingQueue.UnschedulablePods() {
		if p.Name == name && p.Namespace == namespace {
			return true
		}
	}
	for _, p := range sched.SchedulingQueue.IncompletePodGroupPodsPods() {
		if p.Name == name && p.Namespace == namespace {
			return true
		}
	}
	return false
}

func deletePodGroups(ctx context.Context, cs clientset.Interface, ns string, pgNames []string) error {
	for _, name := range pgNames {
		patch := []byte(`{"metadata":{"finalizers":null}}`)
		if _, err := cs.SchedulingV1beta1().PodGroups(ns).Patch(ctx, name, types.MergePatchType, patch, metav1.PatchOptions{}); err != nil && !apierrors.IsNotFound(err) {
			return err
		}
		if err := cs.SchedulingV1beta1().PodGroups(ns).Delete(ctx, name, metav1.DeleteOptions{}); err != nil && !apierrors.IsNotFound(err) {
			return err
		}
	}
	// Wait for the pod groups to be deleted.
	for _, name := range pgNames {
		err := wait.PollUntilContextTimeout(ctx, 100*time.Millisecond, 10*time.Second, false, func(ctx context.Context) (bool, error) {
			_, err := cs.SchedulingV1beta1().PodGroups(ns).Get(ctx, name, metav1.GetOptions{})
			return apierrors.IsNotFound(err), nil
		})
		if err != nil {
			return err
		}
	}
	return nil
}

func deleteCompositePodGroups(ctx context.Context, cs clientset.Interface, ns string, cpgNames []string) error {
	for _, name := range cpgNames {
		patch := []byte(`{"metadata":{"finalizers":null}}`)
		if _, err := cs.SchedulingV1alpha3().CompositePodGroups(ns).Patch(ctx, name, types.MergePatchType, patch, metav1.PatchOptions{}); err != nil && !apierrors.IsNotFound(err) {
			return err
		}
		if err := cs.SchedulingV1alpha3().CompositePodGroups(ns).Delete(ctx, name, metav1.DeleteOptions{}); err != nil && !apierrors.IsNotFound(err) {
			return err
		}
	}
	// Wait for the composite pod groups to be deleted.
	for _, name := range cpgNames {
		err := wait.PollUntilContextTimeout(ctx, 100*time.Millisecond, 10*time.Second, false, func(ctx context.Context) (bool, error) {
			_, err := cs.SchedulingV1alpha3().CompositePodGroups(ns).Get(ctx, name, metav1.GetOptions{})
			return apierrors.IsNotFound(err), nil
		})
		if err != nil {
			return err
		}
	}
	return nil
}
