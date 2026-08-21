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

package podgrouppreemption

import (
	"context"
	"errors"
	"fmt"
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
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	types "k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/intstr"
	"k8s.io/apimachinery/pkg/util/sets"
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
	"k8s.io/kubernetes/test/integration/scheduler/preemption/asyncframework"
	testutils "k8s.io/kubernetes/test/integration/util"
	"k8s.io/kubernetes/test/utils/client-go/ktesting"
)

// TestPodGroupPreemption tests preemption scenarios involving pod groups.
func TestPodGroupPreemption(t *testing.T) {
	featuregatetesting.SetFeatureGatesDuringTest(t, utilfeature.DefaultFeatureGate, featuregatetesting.FeatureOverrides{
		features.GenericWorkload:                 true,
		features.PodLevelResources:               true,
		features.TopologyAwareWorkloadScheduling: true,
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
		expectedCandidatesForPreemption    []string
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
			expectedScheduled:               []string{"high-1", "high-2", "high-3"},
			expectedCandidatesForPreemption: []string{"low-1", "low-2", "low-3"},
			expectedToHaveNNNInfo:           []string{"high-1", "high-2", "high-3"},
			expectedPodsPreemptedByWAP:      3,
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
			expectedScheduled:               []string{"high-1", "high-2", "high-3"},
			expectedCandidatesForPreemption: []string{"low-1", "low-2", "low-3"},
			expectedToHaveNNNInfo:           []string{"high-1", "high-2", "high-3"},
			expectedPodsPreemptedByWAP:      3,
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
			expectedScheduled:               []string{"high-1", "high-2", "high-3", "low-2"},
			expectedCandidatesForPreemption: []string{"low-1"},
			expectedToHaveNNNInfo:           []string{"high-1", "high-2", "high-3"},
			expectedPodsPreemptedByWAP:      1,
			customPluginName:                "mockScorePlugin",
			customPluginFunc:                newPresetScorePlugin(map[string]int64{"node1": 100, "node2": 0}),
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
			expectedScheduled:               []string{"high-1", "high-2", "high-3", "very-low-1"},
			expectedCandidatesForPreemption: []string{"low-1"},
			expectedToHaveNNNInfo:           []string{},
			expectedPodsPreemptedByWAP:      1,
			customPluginName:                "mockScorePlugin",
			customPluginFunc:                newPresetScorePlugin(map[string]int64{"node1": 100, "node2": 0}),
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
			expectedScheduled:               []string{"high-1", "high-2"},
			expectedCandidatesForPreemption: []string{"low-3"},
			expectedToHaveNNNInfo:           []string{"high-1", "high-2"},
			expectedPodsPreemptedByWAP:      1,
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
			expectedScheduled:               []string{"high-1", "high-2", "high-3", "high-4"},
			expectedCandidatesForPreemption: []string{"low-1", "low-2", "low-3", "low-4"},
			expectedToHaveNNNInfo:           []string{"high-1", "high-2", "high-3", "high-4"},
			expectedPodsPreemptedByWAP:      4,
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
			expectedScheduled:               []string{"mid-1", "low-1", "low-2"},
			expectedCandidatesForPreemption: []string{},
			expectedUnschedulable:           []string{"high-1", "high-2", "high-3"},
			expectedToHaveNNNInfo:           []string{},
			expectedPodsPreemptedByWAP:      0,
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
			expectedScheduled:               []string{"high-1", "high-2", "mid-1"},
			expectedCandidatesForPreemption: []string{"low-1", "low-2"},
			expectedToHaveNNNInfo:           []string{"high-1", "high-2"},
			expectedPodsPreemptedByWAP:      2,
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
			expectedScheduled:               []string{"high-1"},
			expectedCandidatesForPreemption: []string{"low-1", "low-2", "low-3"},
			expectedToHaveNNNInfo:           []string{"high-1"},
			expectedPodsPreemptedByWAP:      3,
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
			expectedScheduled:               []string{"high-1"},
			expectedCandidatesForPreemption: []string{"low-1", "low-2", "low-3"},
			expectedToHaveNNNInfo:           []string{"high-1"},
			expectedPodsPreemptedByWAP:      3,
		},
		{
			name: "Pods from a single gang PodGroup with DisruptionModeSingle can be preempted individually by the higher priority gang PodGroup",
			nodes: []*v1.Node{
				st.MakeNode().Name("node1").Capacity(map[v1.ResourceName]string{v1.ResourceCPU: "3", v1.ResourceMemory: "4Gi", v1.ResourcePods: "32"}).Obj(),
			},
			podGroups: []*schedulingv1beta1.PodGroup{
				st.MakePodGroup().Name("preemptor-pg").Namespace("default").Priority(100).MinCount(1).Obj(),
				st.MakePodGroup().Name("victim-pg").Namespace("default").Priority(10).MinCount(2).DisruptionModeSingle().Obj(),
			},
			initialPods: []*v1.Pod{
				st.MakePod().Name("low-1").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("victim-pg").ZeroTerminationGracePeriod().Priority(10).Obj(),
				st.MakePod().Name("low-2").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("victim-pg").ZeroTerminationGracePeriod().Priority(10).Obj(),
				st.MakePod().Name("low-3").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("victim-pg").ZeroTerminationGracePeriod().Priority(10).Obj(),
			},
			preemptorPods: []*v1.Pod{
				st.MakePod().Name("high-1").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("preemptor-pg").ZeroTerminationGracePeriod().Priority(100).Obj(),
			},
			// We expect one of the pods from victim-pg to be preempted, but do not choose specific pod.
			expectedScheduled:               []string{"high-1"},
			expectedCandidatesForPreemption: []string{"low-1", "low-2", "low-3"},
			expectedPodsPreemptedByWAP:      1,
		},
		{
			name: "Pods from a single gang PodGroup with DisruptionModeSingle can be preempted individually by the higher priority gang PodGroup even below mincount",
			nodes: []*v1.Node{
				st.MakeNode().Name("node1").Capacity(map[v1.ResourceName]string{v1.ResourceCPU: "3", v1.ResourceMemory: "4Gi", v1.ResourcePods: "32"}).Obj(),
			},
			podGroups: []*schedulingv1beta1.PodGroup{
				st.MakePodGroup().Name("preemptor-pg").Namespace("default").Priority(100).MinCount(1).Obj(),
				st.MakePodGroup().Name("victim-pg").Namespace("default").Priority(10).MinCount(3).DisruptionModeSingle().Obj(),
			},
			initialPods: []*v1.Pod{
				st.MakePod().Name("low-1").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("victim-pg").ZeroTerminationGracePeriod().Priority(10).Obj(),
				st.MakePod().Name("low-2").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("victim-pg").ZeroTerminationGracePeriod().Priority(10).Obj(),
				st.MakePod().Name("low-3").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("victim-pg").ZeroTerminationGracePeriod().Priority(10).Obj(),
			},
			preemptorPods: []*v1.Pod{
				st.MakePod().Name("high-1").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("preemptor-pg").ZeroTerminationGracePeriod().Priority(100).Obj(),
			},
			// We expect one of the pods from victim-pg to be preempted, but do not choose specific pod.
			expectedScheduled:               []string{"high-1"},
			expectedCandidatesForPreemption: []string{"low-1", "low-2", "low-3"},
			expectedPodsPreemptedByWAP:      1,
		},
		{
			name: "Pods from a single gang PodGroup with DisruptionModeSingle can be preempted individually by the higher priority basic PodGroup",
			nodes: []*v1.Node{
				st.MakeNode().Name("node1").Capacity(map[v1.ResourceName]string{v1.ResourceCPU: "3", v1.ResourceMemory: "4Gi", v1.ResourcePods: "32"}).Obj(),
			},
			podGroups: []*schedulingv1beta1.PodGroup{
				st.MakePodGroup().Name("preemptor-pg").Namespace("default").Priority(100).BasicPolicy().Obj(),
				st.MakePodGroup().Name("victim-pg").Namespace("default").Priority(10).MinCount(2).DisruptionModeSingle().Obj(),
			},
			initialPods: []*v1.Pod{
				st.MakePod().Name("low-1").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("victim-pg").ZeroTerminationGracePeriod().Priority(10).Obj(),
				st.MakePod().Name("low-2").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("victim-pg").ZeroTerminationGracePeriod().Priority(10).Obj(),
				st.MakePod().Name("low-3").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("victim-pg").ZeroTerminationGracePeriod().Priority(10).Obj(),
			},
			preemptorPods: []*v1.Pod{
				st.MakePod().Name("high-1").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("preemptor-pg").ZeroTerminationGracePeriod().Priority(100).Obj(),
			},
			// We expect one of the pods from victim-pg to be preempted, but do not choose specific pod.
			expectedScheduled:               []string{"high-1"},
			expectedCandidatesForPreemption: []string{"low-1", "low-2", "low-3"},
			expectedPodsPreemptedByWAP:      1,
		},
		{
			name: "Pods from a single basic PodGroup with DisruptionModeSingle can be preempted individually by the higher priority gang PodGroup",
			nodes: []*v1.Node{
				st.MakeNode().Name("node1").Capacity(map[v1.ResourceName]string{v1.ResourceCPU: "3", v1.ResourceMemory: "4Gi", v1.ResourcePods: "32"}).Obj(),
			},
			podGroups: []*schedulingv1beta1.PodGroup{
				st.MakePodGroup().Name("pg1").Namespace("default").Priority(100).MinCount(1).Obj(),
				st.MakePodGroup().Name("pg2").Namespace("default").Priority(10).BasicPolicy().DisruptionModeSingle().Obj(),
			},
			initialPods: []*v1.Pod{
				st.MakePod().Name("low-1").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("pg2").ZeroTerminationGracePeriod().Priority(10).Obj(),
				st.MakePod().Name("low-2").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("pg2").ZeroTerminationGracePeriod().Priority(10).Obj(),
				st.MakePod().Name("low-3").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("pg2").ZeroTerminationGracePeriod().Priority(10).Obj(),
			},
			preemptorPods: []*v1.Pod{
				st.MakePod().Name("high-1").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("pg1").ZeroTerminationGracePeriod().Priority(100).Obj(),
			},
			// We expect one of the pods from victim-pg to be preempted, but do not choose specific pod.
			expectedScheduled:               []string{"high-1"},
			expectedCandidatesForPreemption: []string{"low-1", "low-2", "low-3"},
			expectedPodsPreemptedByWAP:      1,
		},
		{
			name: "Pods from a single basic PodGroup with DisruptionModeSingle can be preempted individually by the higher priority basic PodGroup",
			nodes: []*v1.Node{
				st.MakeNode().Name("node1").Capacity(map[v1.ResourceName]string{v1.ResourceCPU: "3", v1.ResourceMemory: "4Gi", v1.ResourcePods: "32"}).Obj(),
			},
			podGroups: []*schedulingv1beta1.PodGroup{
				st.MakePodGroup().Name("pg1").Namespace("default").Priority(100).BasicPolicy().Obj(),
				st.MakePodGroup().Name("pg2").Namespace("default").Priority(10).BasicPolicy().DisruptionModeSingle().Obj(),
			},
			initialPods: []*v1.Pod{
				st.MakePod().Name("low-1").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("pg2").ZeroTerminationGracePeriod().Priority(10).Obj(),
				st.MakePod().Name("low-2").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("pg2").ZeroTerminationGracePeriod().Priority(10).Obj(),
				st.MakePod().Name("low-3").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("pg2").ZeroTerminationGracePeriod().Priority(10).Obj(),
			},
			preemptorPods: []*v1.Pod{
				st.MakePod().Name("high-1").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("pg1").ZeroTerminationGracePeriod().Priority(100).Obj(),
			},
			// We expect one of the pods from victim-pg to be preempted, but do not choose specific pod.
			expectedScheduled:               []string{"high-1"},
			expectedCandidatesForPreemption: []string{"low-1", "low-2", "low-3"},
			expectedPodsPreemptedByWAP:      1,
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
			expectedCandidatesForPreemption:    []string{"p1", "p2"},
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
			expectedScheduled:               []string{"p-a", "p-b", "p-c"},
			expectedCandidatesForPreemption: []string{"v1", "v2", "v3"},
			expectedToHaveNNNInfo:           []string{"p-a", "p-b", "p-c"},
			expectedPodsPreemptedByWAP:      3,
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
			expectedScheduled:               []string{"p-a", "p-b", "v3", "v4"},
			expectedCandidatesForPreemption: []string{"v1", "v2"},
			expectedUnschedulable:           []string{"p-c"},
			expectedToHaveNNNInfo:           []string{"p-a"},
			expectedPodsPreemptedByWAP:      2,
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
			expectedCandidatesForPreemption:    []string{"p1", "p2"},
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
			expectedScheduled:               []string{"p-a", "p-b", "p-c"},
			expectedCandidatesForPreemption: []string{"v1", "v2", "v3"},
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
			expectedScheduled:               []string{"p-a", "p-b", "v3", "v4"},
			expectedCandidatesForPreemption: []string{"v1", "v2"},
			expectedUnschedulable:           []string{"p-c"},
			expectedToHaveNNNInfo:           []string{"p-a"},
			expectedPodsPreemptedByWAP:      2,
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
			expectedScheduled:               []string{"low-1", "low-2", "low-3"},
			expectedCandidatesForPreemption: []string{},
			expectedUnschedulable:           []string{"high-1", "high-2", "high-3"},
			expectedPodsPreemptedByWAP:      0,
			enablePodGroupPreemptionPolicy:  true,
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
			expectedScheduled:               []string{"high-1", "high-2", "high-3"},
			expectedCandidatesForPreemption: []string{"low-1", "low-2", "low-3"},
			expectedPodsPreemptedByWAP:      3,
			enablePodGroupPreemptionPolicy:  true,
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
			expectedScheduled:               []string{"high-1", "high-2", "high-3"},
			expectedCandidatesForPreemption: []string{"low-1", "low-2", "low-3"},
			expectedPodsPreemptedByWAP:      3,
			enablePodGroupPreemptionPolicy:  true,
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
			expectedScheduled:               []string{"low-1", "low-2", "low-3"},
			expectedCandidatesForPreemption: []string{},
			expectedUnschedulable:           []string{"high-1", "high-2", "high-3"},
			expectedPodsPreemptedByWAP:      0,
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
			expectedScheduled:               []string{"preemptor-1", "preemptor-2"},
			expectedCandidatesForPreemption: []string{"initial-pod"},
			expectedToHaveNNNInfo:           []string{"preemptor-1", "preemptor-2"},
			expectedPodsPreemptedByWAP:      1,
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
			expectedScheduled:               []string{"preemptor-1", "preemptor-2"},
			expectedCandidatesForPreemption: []string{"initial-pod-1"},
			expectedToHaveNNNInfo:           []string{"preemptor-1", "preemptor-2"},
			expectedPodsPreemptedByWAP:      1,
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
			expectedScheduled:               []string{"preemptor-1", "preemptor-2"},
			expectedCandidatesForPreemption: []string{"initial-pod"},
			expectedToHaveNNNInfo:           []string{"preemptor-1", "preemptor-2"},
			expectedPodsPreemptedByWAP:      1,
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
			expectedScheduled:               []string{"preemptor-1", "preemptor-2"},
			expectedCandidatesForPreemption: []string{"initial-pod"},
			expectedToHaveNNNInfo:           []string{"preemptor-1", "preemptor-2"},
			expectedPodsPreemptedByWAP:      1,
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
			expectedScheduled:               []string{"preemptor-1", "preemptor-2"},
			expectedCandidatesForPreemption: []string{"initial-pod"},
			expectedToHaveNNNInfo:           []string{"preemptor-1", "preemptor-2"},
			expectedPodsPreemptedByWAP:      1,
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
			expectedScheduled:               []string{"preemptor-1", "preemptor-2"},
			expectedCandidatesForPreemption: []string{"initial-pod"},
			expectedToHaveNNNInfo:           []string{"preemptor-1", "preemptor-2"},
			expectedPodsPreemptedByWAP:      1,
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
			expectedScheduled:               []string{"preemptor-1", "preemptor-2"},
			expectedCandidatesForPreemption: []string{"initial-pod"},
			expectedToHaveNNNInfo:           []string{"preemptor-1", "preemptor-2"},
			expectedPodsPreemptedByWAP:      1,
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
			expectedScheduled:               []string{"p-a", "p-b", "v2"},
			expectedCandidatesForPreemption: []string{"v1"},
			expectedUnschedulable:           []string{},
			expectedToHaveNNNInfo:           []string{},
			expectedPodsPreemptedByWAP:      1,
			customPluginName:                "mockReservePlugin",
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
			expectedScheduled:               []string{"pg-pod-1", "pg-pod-2", "high-1", "high-2"},
			expectedCandidatesForPreemption: []string{"low-1"},
			expectedPodsPreemptedByWAP:      1,
			tempRemovePG:                    true,
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
			expectedScheduled:               []string{"pg-pod-1", "pg-pod-2", "high-1", "high-2"},
			expectedCandidatesForPreemption: []string{"low-1"},
			expectedPodsPreemptedByWAP:      1,
			tempRemovePG:                    true,
			// both preemptor pods will become schedulable at once, but there will be only place for 1 pod without preemption
			// the scheduling cycle should prefer binding this pod over preempting to make room for both pods
			// preemption will be called in the subsequent cycle to make room for the second pod.
			expectedEventOrder: []string{"Bind:high-1", "PodGroupPostFilter:pg1", "Bind:high-2"},
		},
		{
			name: "Topology-Aware Preemption: single topology domain",
			nodes: []*v1.Node{
				st.MakeNode().Name("node1").Label("topology-key", "zone1").Capacity(map[v1.ResourceName]string{v1.ResourceCPU: "1", v1.ResourceMemory: "4Gi", v1.ResourcePods: "32"}).Obj(),
				st.MakeNode().Name("node2").Label("topology-key", "zone1").Capacity(map[v1.ResourceName]string{v1.ResourceCPU: "1", v1.ResourceMemory: "4Gi", v1.ResourcePods: "32"}).Obj(),
			},
			podGroups: []*schedulingv1beta1.PodGroup{
				st.MakePodGroup().Name("pg1").Namespace("default").Priority(100).MinCount(2).TopologyKey("topology-key").Obj(),
			},
			initialPods: []*v1.Pod{
				st.MakePod().Name("low-1").Node("node1").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").ZeroTerminationGracePeriod().Priority(10).Obj(),
				st.MakePod().Name("low-2").Node("node2").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").ZeroTerminationGracePeriod().Priority(10).Obj(),
			},
			preemptorPods: []*v1.Pod{
				st.MakePod().Name("high-1").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("pg1").ZeroTerminationGracePeriod().Priority(100).Obj(),
				st.MakePod().Name("high-2").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("pg1").ZeroTerminationGracePeriod().Priority(100).Obj(),
			},
			expectedScheduled:               []string{"high-1", "high-2"},
			expectedCandidatesForPreemption: []string{"low-1", "low-2"},
			expectedToHaveNNNInfo:           []string{"high-1", "high-2"},
			expectedPodsPreemptedByWAP:      2,
			tempRemovePG:                    true,
		},
		{
			name: "Topology-Aware Preemption: two topologies, only one eligible",
			nodes: []*v1.Node{
				st.MakeNode().Name("node1").Label("topology-key", "zone1").Capacity(map[v1.ResourceName]string{v1.ResourceCPU: "1", v1.ResourceMemory: "4Gi", v1.ResourcePods: "32"}).Obj(),
				st.MakeNode().Name("node2").Label("topology-key", "zone1").Capacity(map[v1.ResourceName]string{v1.ResourceCPU: "1", v1.ResourceMemory: "4Gi", v1.ResourcePods: "32"}).Obj(),
				st.MakeNode().Name("node3").Label("topology-key", "zone2").Capacity(map[v1.ResourceName]string{v1.ResourceCPU: "1", v1.ResourceMemory: "4Gi", v1.ResourcePods: "32"}).Obj(),
				st.MakeNode().Name("node4").Label("topology-key", "zone2").Capacity(map[v1.ResourceName]string{v1.ResourceCPU: "1", v1.ResourceMemory: "4Gi", v1.ResourcePods: "32"}).Obj(),
			},
			podGroups: []*schedulingv1beta1.PodGroup{
				st.MakePodGroup().Name("pg1").Namespace("default").Priority(100).MinCount(2).TopologyKey("topology-key").Obj(),
			},
			initialPods: []*v1.Pod{
				st.MakePod().Name("low-1").Node("node1").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").ZeroTerminationGracePeriod().Priority(10).Obj(),
				st.MakePod().Name("low-2").Node("node2").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").ZeroTerminationGracePeriod().Priority(10).Obj(),
				st.MakePod().Name("high-priority-initial-1").Node("node3").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").ZeroTerminationGracePeriod().Priority(200).Obj(),
				st.MakePod().Name("high-priority-initial-2").Node("node4").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").ZeroTerminationGracePeriod().Priority(200).Obj(),
			},
			preemptorPods: []*v1.Pod{
				st.MakePod().Name("high-1").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("pg1").ZeroTerminationGracePeriod().Priority(100).Obj(),
				st.MakePod().Name("high-2").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("pg1").ZeroTerminationGracePeriod().Priority(100).Obj(),
			},
			expectedScheduled:               []string{"high-1", "high-2", "high-priority-initial-1", "high-priority-initial-2"},
			expectedCandidatesForPreemption: []string{"low-1", "low-2"},
			expectedToHaveNNNInfo:           []string{"high-1", "high-2"},
			expectedPodsPreemptedByWAP:      2,
			tempRemovePG:                    true,
		},
		{
			name: "Topology-Aware Preemption: two topologies, both eligible, selects higher scored topology",
			nodes: []*v1.Node{
				st.MakeNode().Name("node1").Label("topology-key", "zone1").Capacity(map[v1.ResourceName]string{v1.ResourceCPU: "1", v1.ResourceMemory: "4Gi", v1.ResourcePods: "32"}).Obj(),
				st.MakeNode().Name("node2").Label("topology-key", "zone1").Capacity(map[v1.ResourceName]string{v1.ResourceCPU: "1", v1.ResourceMemory: "4Gi", v1.ResourcePods: "32"}).Obj(),
				st.MakeNode().Name("node3").Label("topology-key", "zone2").Capacity(map[v1.ResourceName]string{v1.ResourceCPU: "1", v1.ResourceMemory: "4Gi", v1.ResourcePods: "32"}).Obj(),
				st.MakeNode().Name("node4").Label("topology-key", "zone2").Capacity(map[v1.ResourceName]string{v1.ResourceCPU: "1", v1.ResourceMemory: "4Gi", v1.ResourcePods: "32"}).Obj(),
			},
			podGroups: []*schedulingv1beta1.PodGroup{
				st.MakePodGroup().Name("pg1").Namespace("default").Priority(100).MinCount(2).TopologyKey("topology-key").Obj(),
			},
			initialPods: []*v1.Pod{
				st.MakePod().Name("low-1").Node("node1").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").ZeroTerminationGracePeriod().Priority(10).Obj(),
				st.MakePod().Name("low-2").Node("node2").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").ZeroTerminationGracePeriod().Priority(10).Obj(),
				st.MakePod().Name("low-3").Node("node3").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").ZeroTerminationGracePeriod().Priority(10).Obj(),
				st.MakePod().Name("low-4").Node("node4").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").ZeroTerminationGracePeriod().Priority(10).Obj(),
			},
			preemptorPods: []*v1.Pod{
				st.MakePod().Name("high-1").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("pg1").ZeroTerminationGracePeriod().Priority(100).Obj(),
				st.MakePod().Name("high-2").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("pg1").ZeroTerminationGracePeriod().Priority(100).Obj(),
			},
			expectedScheduled:               []string{"high-1", "high-2", "low-1", "low-2"},
			expectedCandidatesForPreemption: []string{"low-3", "low-4"},
			expectedToHaveNNNInfo:           []string{"high-1", "high-2"},
			expectedPodsPreemptedByWAP:      2,
			tempRemovePG:                    true,
			customPluginName:                "mockScorePlugin",
			customPluginFunc:                newPresetScorePlugin(map[string]int64{"node1": 0, "node2": 0, "node3": 100, "node4": 100}),
		},
		{
			// Even after removing low prio pods in each topology, none of them becomes available.
			name: "Topology-Aware Preemption: insufficient capacity across all topologies, no preemption performed",
			nodes: []*v1.Node{
				st.MakeNode().Name("node1").Label("topology-key", "zone1").Capacity(map[v1.ResourceName]string{v1.ResourceCPU: "1", v1.ResourceMemory: "4Gi", v1.ResourcePods: "32"}).Obj(),
				st.MakeNode().Name("node2").Label("topology-key", "zone1").Capacity(map[v1.ResourceName]string{v1.ResourceCPU: "1", v1.ResourceMemory: "4Gi", v1.ResourcePods: "32"}).Obj(),
				st.MakeNode().Name("node3").Label("topology-key", "zone2").Capacity(map[v1.ResourceName]string{v1.ResourceCPU: "1", v1.ResourceMemory: "4Gi", v1.ResourcePods: "32"}).Obj(),
				st.MakeNode().Name("node4").Label("topology-key", "zone2").Capacity(map[v1.ResourceName]string{v1.ResourceCPU: "1", v1.ResourceMemory: "4Gi", v1.ResourcePods: "32"}).Obj(),
			},
			podGroups: []*schedulingv1beta1.PodGroup{
				st.MakePodGroup().Name("pg1").Namespace("default").Priority(100).MinCount(2).TopologyKey("topology-key").Obj(),
			},
			initialPods: []*v1.Pod{
				st.MakePod().Name("low-1").Node("node1").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").ZeroTerminationGracePeriod().Priority(10).Obj(),
				st.MakePod().Name("high-priority-initial-1").Node("node2").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").ZeroTerminationGracePeriod().Priority(200).Obj(),
				st.MakePod().Name("low-2").Node("node3").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").ZeroTerminationGracePeriod().Priority(10).Obj(),
				st.MakePod().Name("high-priority-initial-2").Node("node4").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").ZeroTerminationGracePeriod().Priority(200).Obj(),
			},
			preemptorPods: []*v1.Pod{
				st.MakePod().Name("high-1").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("pg1").ZeroTerminationGracePeriod().Priority(100).Obj(),
				st.MakePod().Name("high-2").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("pg1").ZeroTerminationGracePeriod().Priority(100).Obj(),
			},
			expectedScheduled:               []string{"low-1", "low-2", "high-priority-initial-1", "high-priority-initial-2"},
			expectedCandidatesForPreemption: nil,
			expectedToHaveNNNInfo:           nil,
			expectedPodsPreemptedByWAP:      0,
			tempRemovePG:                    true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			featuregatetesting.SetFeatureGatesDuringTest(t, utilfeature.DefaultFeatureGate, featuregatetesting.FeatureOverrides{
				features.GenericWorkload:                 true,
				features.PodGroupPreemptionPolicy:        tt.enablePodGroupPreemptionPolicy,
				features.TopologyAwareWorkloadScheduling: true,
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
					for _, podName := range tt.expectedCandidatesForPreemption {
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
			if len(tt.expectedCandidatesForPreemption) > 0 {
				var preemptedCount int
				var notPreemptedPods []string
				// Subgroup of pods (might be all) in candidatesForPreemption is expected to be preempted.
				// Preemption has finished, because all expected pods were scheduled - checked in step 7.
				// Retry will be performed when there is an error or number of preempted pod do not match expectedPodsPreemptedByWAP.
				err := wait.PollUntilContextTimeout(testCtx.Ctx, 200*time.Millisecond, 5*time.Second, false,
					func(ctx context.Context) (bool, error) {
						preemptedCount = 0
						notPreemptedPods = nil
						for _, podName := range tt.expectedCandidatesForPreemption {
							pod, err := cs.CoreV1().Pods(ns).Get(ctx, podName, metav1.GetOptions{})
							if err != nil {
								if apierrors.IsNotFound(err) {
									preemptedCount++
									continue
								}
								return false, err
							}
							if pod.DeletionTimestamp != nil {
								preemptedCount++
								continue
							}
							if _, cond := podutil.GetPodCondition(&pod.Status, v1.DisruptionTarget); cond != nil {
								preemptedCount++
								continue
							}
							notPreemptedPods = append(notPreemptedPods, podName)
						}
						return preemptedCount == tt.expectedPodsPreemptedByWAP, nil
					})
				if err != nil {
					t.Errorf("Expected exactly %d pods from %v to be preempted, but only %d pods were preempted, not preempted pods: %v. Error: %v", tt.expectedPodsPreemptedByWAP, tt.expectedCandidatesForPreemption, preemptedCount, notPreemptedPods, err)
				}
			}

			// 9. Verify preemptor pods have nominated node name
			for _, podName := range tt.expectedToHaveNNNInfo {
				if node, ok := bindPlugin.nnnInfo.Load(podName); !ok || node.(string) == "" {
					t.Errorf("Pod %s was expected to have nominated node name but didn't", podName)
				}
			}

			// 10. Verify event order
			if len(tt.expectedEventOrder) > 0 {
				actualEvents := recorder.GetEvents()
				if diff := cmp.Diff(tt.expectedEventOrder, actualEvents); diff != "" {
					t.Errorf("Unexpected event order (-want,+got):\n%s", diff)
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
			SchedulerName: new(v1.DefaultSchedulerName),
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

func TestPodGroupCycleStatePreserved(t *testing.T) {
	featuregatetesting.SetFeatureGatesDuringTest(t, utilfeature.DefaultFeatureGate, featuregatetesting.FeatureOverrides{
		features.GenericWorkload: true,
	})

	stateSaverPluginName := "stateSaverPreFilterPlugin"
	stateVerifierPluginName := "stateVerifierPostFilterPlugin"

	var stateSaver stateSaverPreFilterPlugin
	var stateVerifier stateVerifierPostFilterPlugin

	registry := make(frameworkruntime.Registry)
	err := registry.Register(stateSaverPluginName, func(ctx context.Context, o runtime.Object, fh fwk.Handle) (fwk.Plugin, error) {
		stateSaver = stateSaverPreFilterPlugin{
			name:   stateSaverPluginName,
			handle: fh,
		}
		return &stateSaver, nil
	})
	if err != nil {
		t.Fatalf("Failed to register stateSaverPreFilterPlugin: %v", err)
	}

	err = registry.Register(stateVerifierPluginName, func(ctx context.Context, o runtime.Object, fh fwk.Handle) (fwk.Plugin, error) {
		stateVerifier = stateVerifierPostFilterPlugin{
			name: stateVerifierPluginName,
		}
		return &stateVerifier, nil
	})
	if err != nil {
		t.Fatalf("Failed to register stateVerifierPostFilterPlugin: %v", err)
	}

	cfg := configtesting.V1ToInternalWithDefaults(t, configv1.KubeSchedulerConfiguration{
		Profiles: []configv1.KubeSchedulerProfile{{
			SchedulerName: new(v1.DefaultSchedulerName),
			Plugins: &configv1.Plugins{
				MultiPoint: configv1.PluginSet{
					Enabled: []configv1.Plugin{
						{Name: stateSaverPluginName},
						{Name: names.DefaultPreemption},
						{Name: stateVerifierPluginName},
					},
					Disabled: []configv1.Plugin{
						{Name: names.DefaultPreemption},
					},
				},
			},
		}},
	})

	testCtx := testutils.InitTestSchedulerWithNS(t, "state-preserved",
		scheduler.WithProfiles(cfg.Profiles...),
		scheduler.WithFrameworkOutOfTreeRegistry(registry),
		scheduler.WithPodMaxBackoffSeconds(1),
		scheduler.WithPodInitialBackoffSeconds(0),
	)
	cs, ns := testCtx.ClientSet, testCtx.NS.Name

	// 1. Create node with 3 CPU capacity
	node := st.MakeNode().Name("node1").Capacity(map[v1.ResourceName]string{v1.ResourceCPU: "3", v1.ResourceMemory: "4Gi", v1.ResourcePods: "32"}).Obj()
	if _, err := cs.CoreV1().Nodes().Create(testCtx.Ctx, node, metav1.CreateOptions{}); err != nil {
		t.Fatalf("Failed to create node: %v", err)
	}

	// 2. Create high-pod (priority 500, CPU 2) and low-pod (priority 10, CPU 1)
	highPod := st.MakePod().Name("high-pod").Namespace(ns).Req(map[v1.ResourceName]string{v1.ResourceCPU: "2"}).Container("image").Priority(500).ZeroTerminationGracePeriod().Node("node1").Obj()
	lowPod := st.MakePod().Name("low-pod").Namespace(ns).Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").Priority(10).ZeroTerminationGracePeriod().Node("node1").Obj()

	for _, p := range []*v1.Pod{highPod, lowPod} {
		if _, err := cs.CoreV1().Pods(ns).Create(testCtx.Ctx, p, metav1.CreateOptions{}); err != nil {
			t.Fatalf("Failed to create pod %s: %v", p.Name, err)
		}
	}

	// Wait for initial pods to be scheduled
	for _, p := range []*v1.Pod{highPod, lowPod} {
		if err := wait.PollUntilContextTimeout(testCtx.Ctx, 100*time.Millisecond, 10*time.Second, false, testutils.PodScheduled(cs, ns, p.Name)); err != nil {
			t.Fatalf("Failed to wait for initial pod %s to schedule: %v", p.Name, err)
		}
	}

	// 3. Create preemptor pods belonging to pg1 (each needing CPU 2)
	preemptor1 := st.MakePod().Name("preemptor-1").Namespace(ns).Req(map[v1.ResourceName]string{v1.ResourceCPU: "2"}).Container("image").PodGroupName("pg1").Priority(100).ZeroTerminationGracePeriod().Obj()
	preemptor2 := st.MakePod().Name("preemptor-2").Namespace(ns).Req(map[v1.ResourceName]string{v1.ResourceCPU: "2"}).Container("image").PodGroupName("pg1").Priority(100).ZeroTerminationGracePeriod().Obj()

	for _, p := range []*v1.Pod{preemptor1, preemptor2} {
		if _, err := cs.CoreV1().Pods(ns).Create(testCtx.Ctx, p, metav1.CreateOptions{}); err != nil {
			t.Fatalf("Failed to create preemptor pod %s: %v", p.Name, err)
		}
	}

	// 4. Create PodGroup with minCount=2 and priority=100
	pg := st.MakePodGroup().Name("pg1").Namespace(ns).Priority(100).MinCount(2).Obj()
	if _, err := cs.SchedulingV1beta1().PodGroups(ns).Create(testCtx.Ctx, pg, metav1.CreateOptions{}); err != nil {
		t.Fatalf("Failed to create PodGroup pg1: %v", err)
	}

	var recordedPods sets.Set[string]

	// 5. Verify that stateVerifierPostFilterPlugin was called and state was preserved
	err = wait.PollUntilContextTimeout(testCtx.Ctx, 100*time.Millisecond, 10*time.Second, false, func(ctx context.Context) (bool, error) {
		stateVerifier.lock.Lock()
		recordedPods = sets.New(stateVerifier.recordedPodNames...)
		defer stateVerifier.lock.Unlock()
		return stateVerifier.called, nil
	})
	if err != nil {
		t.Fatalf("Timed out waiting for stateVerifierPostFilterPlugin to be called: %v", err)
	}

	if stateVerifier.readErr != nil {
		t.Errorf("Expected no error reading stateSaverKey from PodGroupCycleState, got: %v", stateVerifier.readErr)
	}

	expectedPods := sets.New("high-pod", "low-pod")

	if !recordedPods.IsSuperset(expectedPods) {
		t.Errorf("PodGroupCycleState lost pod information! Expected pods %v to be in state, got %v", sets.List(expectedPods), sets.List(recordedPods))
	}
}

// TestPodGroupAsyncPreemption is equivalent for TestAsyncPreemption
// in test/integration/scheduler/preemption/preemption_test.go
// When adding test here, add also test there.
func TestPodGroupAsyncPreemption(t *testing.T) {

	tests := []struct {
		Name  string
		Steps []asyncframework.Step
	}{
		{
			// Very base test case: if it fails, the base scenario is broken somewhere.
			Name: "base: async preemption happens expectedly",
			Steps: []asyncframework.Step{
				{
					Name:       "create Node",
					CreateNode: "node",
				},
				{
					Name: "create scheduled Pod",
					CreatePod: &asyncframework.CreatePod{
						Pod:   st.MakePod().GenerateName("victim-").Req(map[v1.ResourceName]string{v1.ResourceCPU: "2"}).Node("node").Container("image").ZeroTerminationGracePeriod().Priority(1).Obj(),
						Count: new(2),
					},
				},
				{
					Name: "create pod group for preemptor",
					CreatePodGroup: &asyncframework.CreatePodGroup{
						PodGroup: st.MakePodGroup().Name("pg-preemptor").MinCount(2).Priority(100).Obj(),
					},
				},
				{
					Name: "create a preemptor Pods",
					CreatePod: &asyncframework.CreatePod{
						Pod: st.MakePod().Name("preemptor-1").Req(map[v1.ResourceName]string{v1.ResourceCPU: "2"}).Container("image").Priority(100).PodGroupName("pg-preemptor").Obj(),
					},
				},
				{
					Name: "create a preemptor Pods",
					CreatePod: &asyncframework.CreatePod{
						Pod: st.MakePod().Name("preemptor-2").Req(map[v1.ResourceName]string{v1.ResourceCPU: "2"}).Container("image").Priority(100).PodGroupName("pg-preemptor").Obj(),
					},
				},
				{
					Name: "schedule the preemptor Pod",
					SchedulePodGroup: &asyncframework.SchedulePodGroup{
						PodGroupName:        "pg-preemptor",
						ExpectUnschedulable: true,
					},
				},
				{
					Name:            "check the pod is in the queue and gated",
					PodGatedInQueue: "preemptor-1",
				},
				{
					Name:            "check the pod is in the queue and gated",
					PodGatedInQueue: "preemptor-2",
				},
				{
					Name:                 "check the preemptor Pod making the preemption API calls",
					PodRunningPreemption: new(2),
				},
				{
					Name:               "complete the preemption API calls",
					CompletePreemption: "pg-preemptor",
				},
				{
					Name: "schedule the preemptor Pod after the preemption",
					SchedulePodGroup: &asyncframework.SchedulePodGroup{
						PodGroupName:  "pg-preemptor",
						ExpectSuccess: true,
					},
				},
			},
		},
		{
			Name: "base async preemption with 1 victim, preemptor gated until preemption API call finishes",
			Steps: []asyncframework.Step{
				{
					Name:       "create Node",
					CreateNode: "node",
				},
				{
					Name: "create victim",
					CreatePod: &asyncframework.CreatePod{
						Pod: st.MakePod().GenerateName("victim-").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Node("node").Container("image").ZeroTerminationGracePeriod().Priority(1).Obj(),
					},
				},
				{
					Name: "create pod group for preemptor",
					CreatePodGroup: &asyncframework.CreatePodGroup{
						PodGroup: st.MakePodGroup().Name("pg-preemptor").MinCount(1).Priority(100).Obj(),
					},
				},
				{
					Name: "create a preemptor Pod",
					CreatePod: &asyncframework.CreatePod{
						Pod: st.MakePod().Name("preemptor").Req(map[v1.ResourceName]string{v1.ResourceCPU: "4"}).Container("image").Priority(100).PodGroupName("pg-preemptor").Obj(),
					},
				},
				{
					Name: "schedule the preemptor Pod",
					SchedulePodGroup: &asyncframework.SchedulePodGroup{
						PodGroupName:        "pg-preemptor",
						ExpectUnschedulable: true,
					},
				},
				{
					Name:            "check the preemptor Pod is in the queue and gated",
					PodGatedInQueue: "preemptor",
				},
				{
					Name:                 "check the preemptor Pod making the preemption API calls",
					PodRunningPreemption: new(1),
				},
				{
					Name:               "complete the preemption API call",
					CompletePreemption: "pg-preemptor",
				},
				{
					Name: "schedule the preemptor Pod again and expect it to be scheduled",
					SchedulePodGroup: &asyncframework.SchedulePodGroup{
						PodGroupName:  "pg-preemptor",
						ExpectSuccess: true,
					},
				},
			},
		},
		{
			Name: "Lower priority Pod doesn't take over the place for higher priority Pod that is running the preemption",
			Steps: []asyncframework.Step{
				{
					Name:       "create Node",
					CreateNode: "node",
				},
				{
					Name: "create scheduled Pod",
					CreatePod: &asyncframework.CreatePod{
						Pod:   st.MakePod().GenerateName("victim-").Req(map[v1.ResourceName]string{v1.ResourceCPU: "2"}).Node("node").Container("image").ZeroTerminationGracePeriod().Priority(1).Obj(),
						Count: new(2),
					},
				},
				{
					Name: "create pod group for preemptor",
					CreatePodGroup: &asyncframework.CreatePodGroup{
						PodGroup: st.MakePodGroup().Name("pg-preemptor").MinCount(1).Priority(100).Obj(),
					},
				},
				{
					Name: "create a preemptor Pod",
					CreatePod: &asyncframework.CreatePod{
						Pod: st.MakePod().Name("preemptor-high-priority").Req(map[v1.ResourceName]string{v1.ResourceCPU: "4"}).Container("image").Priority(100).PodGroupName("pg-preemptor").Obj(),
					},
				},
				{
					Name: "schedule the preemptor Pod",
					SchedulePodGroup: &asyncframework.SchedulePodGroup{
						PodGroupName:        "pg-preemptor",
						ExpectUnschedulable: true,
					},
				},
				{
					Name:            "check the pod is in the queue and gated",
					PodGatedInQueue: "preemptor-high-priority",
				},
				{
					Name:                 "check the preemptor Pod making the preemption API calls",
					PodRunningPreemption: new(2),
				},
				{
					Name: "create a lower priority pod group for a lower priority pod",
					CreatePodGroup: &asyncframework.CreatePodGroup{
						PodGroup: st.MakePodGroup().Name("pg-mid-priority").MinCount(1).Priority(50).Obj(),
					},
				},
				{
					// This Pod is lower priority than the preemptor Pod.
					// Given the preemptor Pod is nominated to the node, this Pod should be unschedulable.
					Name: "create a second Pod that is lower priority than the first preemptor Pod",
					CreatePod: &asyncframework.CreatePod{
						Pod: st.MakePod().Name("pod-mid-priority").Req(map[v1.ResourceName]string{v1.ResourceCPU: "4"}).Container("image").PodGroupName("pg-mid-priority").Priority(50).Obj(),
					},
				},
				{
					Name: "schedule the mid-priority Pod",
					SchedulePodGroup: &asyncframework.SchedulePodGroup{
						PodGroupName:  "pg-mid-priority",
						ExpectInQueue: true,
					},
				},
				{
					Name:               "complete the preemption API calls",
					CompletePreemption: "pg-preemptor",
				},
				{
					// the preemptor pod should be popped from the queue before the mid-priority pod.
					Name: "schedule the preemptor Pod again",
					SchedulePodGroup: &asyncframework.SchedulePodGroup{
						PodGroupName:  "pg-preemptor",
						ExpectSuccess: true,
					},
				},
				{
					Name: "schedule the mid-priority Pod again",
					SchedulePodGroup: &asyncframework.SchedulePodGroup{
						PodGroupName:  "pg-mid-priority",
						ExpectInQueue: true,
					},
				},
			},
		},
		{
			Name: "Higher priority Pod takes over the place for lower priority Pod that is running the preemption",
			Steps: []asyncframework.Step{
				{
					Name:       "create Node",
					CreateNode: "node",
				},
				{
					Name: "create scheduled Pod",
					CreatePod: &asyncframework.CreatePod{
						Pod:   st.MakePod().GenerateName("victim-").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Node("node").Container("image").ZeroTerminationGracePeriod().Priority(1).Obj(),
						Count: new(4),
					},
				},
				{
					Name: "create pod group for preemptor",
					CreatePodGroup: &asyncframework.CreatePodGroup{
						PodGroup: st.MakePodGroup().Name("pg-preemptor").MinCount(1).Priority(100).Obj(),
					},
				},
				{
					Name: "create a preemptor Pod",
					CreatePod: &asyncframework.CreatePod{
						Pod: st.MakePod().Name("preemptor-high-priority").Req(map[v1.ResourceName]string{v1.ResourceCPU: "4"}).Container("image").Priority(100).PodGroupName("pg-preemptor").Obj(),
					},
				},
				{
					Name: "schedule the preemptor Pod",
					SchedulePodGroup: &asyncframework.SchedulePodGroup{
						PodGroupName:        "pg-preemptor",
						ExpectUnschedulable: true,
					},
				},
				{
					Name:            "check the pod is in the queue and gated",
					PodGatedInQueue: "preemptor-high-priority",
				},
				{
					Name:                 "check the preemptor Pod making the preemption API calls",
					PodRunningPreemption: new(4),
				},
				{
					// This Pod is higher priority than the preemptor Pod.
					// Even though the preemptor Pod is nominated to the node, this Pod can take over the place.
					Name: "create pod group for super-high-priority preemptor",
					CreatePodGroup: &asyncframework.CreatePodGroup{
						PodGroup: st.MakePodGroup().Name("pg-preemptor-super-high-priority").MinCount(1).Priority(200).Obj(),
					},
				},
				{
					Name: "create a second Pod that is higher priority than the first preemptor Pod",
					CreatePod: &asyncframework.CreatePod{
						Pod: st.MakePod().Name("preemptor-super-high-priority").Req(map[v1.ResourceName]string{v1.ResourceCPU: "4"}).Container("image").Priority(200).PodGroupName("pg-preemptor-super-high-priority").Obj(),
					},
				},
				{
					Name: "schedule the super-high-priority Pod",
					SchedulePodGroup: &asyncframework.SchedulePodGroup{
						PodGroupName:        "pg-preemptor-super-high-priority",
						ExpectUnschedulable: true,
					},
				},
				{
					Name:                 "check the super-high-priority Pod making the preemption API calls",
					PodRunningPreemption: new(5),
				},
				{
					// the super-high-priority preemptor should enter the preemption
					// and select the place where the preemptor-high-priority selected.
					// So, basically both goroutines are preempting the same Pods.
					Name:            "check the super-high-priority pod is in the queue and gated",
					PodGatedInQueue: "preemptor-super-high-priority",
				},
				{
					Name:               "complete the preemption API calls of super-high-priority",
					CompletePreemption: "pg-preemptor-super-high-priority",
				},
				{
					Name:               "complete the preemption API calls of high-priority",
					CompletePreemption: "pg-preemptor",
				},
				{
					Name: "schedule the super-high-priority Pod",
					SchedulePodGroup: &asyncframework.SchedulePodGroup{
						PodGroupName:  "pg-preemptor-super-high-priority",
						ExpectSuccess: true,
					},
				},
				// In WAP we send the unschedulable pod straight to the queue with backoff time.
				// By default it's set to 0s, making the pod jump straight to activeQ.
				// We set the time to 1s to give some time for pod to be considered unschedulable.
				{
					Name: "schedule the high-priority Pod",
					SchedulePodGroup: &asyncframework.SchedulePodGroup{
						PodGroupName:  "pg-preemptor",
						ExpectInQueue: true,
					},
				},
			},
		},
		{
			Name: "Lower priority Pod can select the same place where the higher priority Pod is preempting if the node is big enough",
			Steps: []asyncframework.Step{
				{
					Name:       "create Node",
					CreateNode: "node",
				},
				{
					Name: "create scheduled Pod",
					CreatePod: &asyncframework.CreatePod{
						Pod:   st.MakePod().GenerateName("victim-").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Node("node").Container("image").ZeroTerminationGracePeriod().Priority(1).Obj(),
						Count: new(4),
					},
				},
				{
					Name: "create pod group for preemptor",
					CreatePodGroup: &asyncframework.CreatePodGroup{
						PodGroup: st.MakePodGroup().Name("pg-preemptor").MinCount(1).Priority(100).Obj(),
					},
				},
				{
					// It will preempt two victims.
					Name: "create a preemptor Pod",
					CreatePod: &asyncframework.CreatePod{
						Pod: st.MakePod().Name("preemptor-high-priority").Req(map[v1.ResourceName]string{v1.ResourceCPU: "2"}).Container("image").Priority(100).PodGroupName("pg-preemptor").Obj(),
					},
				},
				{
					Name: "schedule the preemptor Pod",
					SchedulePodGroup: &asyncframework.SchedulePodGroup{
						PodGroupName:        "pg-preemptor",
						ExpectUnschedulable: true,
					},
				},
				{
					Name:            "check the pod is in the queue and gated",
					PodGatedInQueue: "preemptor-high-priority",
				},
				{
					Name:                 "check the preemptor Pod making the preemption API calls",
					PodRunningPreemption: new(4),
				},
				{
					Name: "create pod group for preemptor",
					CreatePodGroup: &asyncframework.CreatePodGroup{
						PodGroup: st.MakePodGroup().Name("pg-second-preemptor").MinCount(1).Priority(50).Obj(),
					},
				},
				{
					// This Pod is lower priority than the preemptor Pod.
					// Given the preemptor PodGroup don't support nominated node names yet, this Pod should be unschedulable.
					// This Pod will trigger the preemption to target the any of the 4 victims.
					Name: "create a second Pod that is lower priority than the first preemptor Pod",
					CreatePod: &asyncframework.CreatePod{
						Pod: st.MakePod().Name("preemptor-mid-priority").Req(map[v1.ResourceName]string{v1.ResourceCPU: "2"}).Container("image").Priority(50).PodGroupName("pg-second-preemptor").Obj(),
					},
				},
				{
					Name: "schedule the mid-priority Pod",
					SchedulePodGroup: &asyncframework.SchedulePodGroup{
						PodGroupName:        "pg-second-preemptor",
						ExpectUnschedulable: true,
					},
				},
				{
					Name:            "check the mid-priority pod is in the queue and gated",
					PodGatedInQueue: "preemptor-mid-priority",
				},
				{
					Name:                 "check the mid-priority Pod making the preemption API calls",
					PodRunningPreemption: new(5),
				},
				{
					Name:               "complete the preemption API calls",
					CompletePreemption: "pg-second-preemptor",
				},
				{
					Name:               "complete the preemption API calls",
					CompletePreemption: "pg-preemptor",
				},
				{
					// the preemptor pod should be popped from the queue before the mid-priority pod.
					Name: "schedule the preemptor Pod again",
					SchedulePodGroup: &asyncframework.SchedulePodGroup{
						PodGroupName:  "pg-preemptor",
						ExpectSuccess: true,
					},
				},
				{
					Name: "schedule the mid-priority Pod again",
					SchedulePodGroup: &asyncframework.SchedulePodGroup{
						PodGroupName:  "pg-second-preemptor",
						ExpectSuccess: true,
					},
				},
			},
		},
		{
			// This scenario verifies the fix for https://github.com/kubernetes/kubernetes/issues/134217
			// Scenario reproduces the issue:
			// Victim pod takes long in binding. Preemptor pod attempts preemption, goes to unschedulable, then the victim is deleted.
			// Preemptor pod is woken up by the Pod/Delete event and is being scheduled, even before the victim binding is terminated.
			Name: "victim blocked in binding, preemptor pod gets scheduled after victim-in-binding is deleted",
			Steps: []asyncframework.Step{
				{
					Name:       "create Node",
					CreateNode: "node",
				},
				{
					Name: "create victim Pod that is going to be blocked in binding",
					CreatePod: &asyncframework.CreatePod{
						Pod: st.MakePod().Name(asyncframework.PodBlockedInBindingName).Req(map[v1.ResourceName]string{v1.ResourceCPU: "2"}).Container("image").ZeroTerminationGracePeriod().Priority(1).Obj(),
					},
				},
				{
					Name: "schedule victim Pod",
					SchedulePod: &asyncframework.SchedulePod{
						PodName: asyncframework.PodBlockedInBindingName,
					},
				},
				{
					Name: "create pod group for preemptor",
					CreatePodGroup: &asyncframework.CreatePodGroup{
						PodGroup: st.MakePodGroup().Name("pg-preemptor").MinCount(1).Priority(100).Obj(),
					},
				},
				{
					Name: "create a preemptor Pod",
					CreatePod: &asyncframework.CreatePod{
						Pod: st.MakePod().Name("preemptor").Req(map[v1.ResourceName]string{v1.ResourceCPU: "4"}).Container("image").Priority(100).PodGroupName("pg-preemptor").Obj(),
					},
				},
				{
					Name: "schedule the preemptor Pod",
					SchedulePodGroup: &asyncframework.SchedulePodGroup{
						PodGroupName:        "pg-preemptor",
						ExpectUnschedulable: true,
					},
				},
				{
					Name:               "complete the preemption API call",
					CompletePreemption: "pg-preemptor",
				},
				{
					Name: "schedule the preemptor Pod again and expect it to be scheduled (assumed victim pod was forgotten)",
					SchedulePodGroup: &asyncframework.SchedulePodGroup{
						PodGroupName:  "pg-preemptor",
						ExpectSuccess: true,
					},
				},
				{
					Name:       "resume binding of the blocked pod",
					ResumeBind: true,
				},
			},
		},
		{
			// This scenario verifies the fix for https://github.com/kubernetes/kubernetes/issues/134217
			// Scenario reproduces the issue, but with a victim that is under graceful termination:
			// Victim pod takes long in binding. Preemptor pod attempts preemption, goes to unschedulable, then the victim's graceful termination is initiated.
			// Preemptor pod is woken up by the Pod/Update event (working like AssignedPodDeleted) and is being scheduled, even before the victim binding is terminated.
			Name: "victim blocked in binding, preemptor pod gets scheduled when victim-in-binding is under graceful termination",
			Steps: []asyncframework.Step{
				{
					Name:       "create Node",
					CreateNode: "node",
				},
				{
					Name: "create victim Pod with long termination grace period that is going to be blocked in binding",
					CreatePod: &asyncframework.CreatePod{
						Pod: st.MakePod().Name(asyncframework.PodBlockedInBindingName).Req(map[v1.ResourceName]string{v1.ResourceCPU: "2"}).Container("image").TerminationGracePeriodSeconds(1000).Priority(1).Obj(),
					},
				},
				{
					Name: "schedule victim Pod",
					SchedulePod: &asyncframework.SchedulePod{
						PodName: asyncframework.PodBlockedInBindingName,
					},
				},
				{
					Name: "create pod group for preemptor",
					CreatePodGroup: &asyncframework.CreatePodGroup{
						PodGroup: st.MakePodGroup().Name("pg-preemptor").MinCount(1).Priority(100).Obj(),
					},
				},
				{
					Name: "create a preemptor Pod",
					CreatePod: &asyncframework.CreatePod{
						Pod: st.MakePod().Name("preemptor").Req(map[v1.ResourceName]string{v1.ResourceCPU: "4"}).Container("image").Priority(100).PodGroupName("pg-preemptor").Obj(),
					},
				},
				{
					Name: "schedule the preemptor Pod",
					SchedulePodGroup: &asyncframework.SchedulePodGroup{
						PodGroupName:        "pg-preemptor",
						ExpectUnschedulable: true,
					},
				},
				{
					Name:               "complete the preemption API call",
					CompletePreemption: "pg-preemptor",
				},
				{
					Name: "schedule the preemptor Pod again and expect it to be scheduled (assumed victim pod was forgotten)",
					SchedulePodGroup: &asyncframework.SchedulePodGroup{
						PodGroupName:  "pg-preemptor",
						ExpectSuccess: true,
					},
				},
				{
					Name:       "resume binding of the blocked pod",
					ResumeBind: true,
				},
			},
		},
		{
			// This scenario verifies the fix for https://github.com/kubernetes/kubernetes/issues/134217
			// Scenario reproduces the issue, but with a victim that is reserving some resources required by the preemptor:
			// Victim pod takes long in binding. Preemptor pod attempts preemption, goes to unschedulable, then the victim is deleted.
			// Preemptor pod is woken up by the Pod/Update event (working like AssignedPodDeleted), but is still unschedulable, because victim has to unreserve its resources.
			// After resuming binding for a victim, it releases the resources in its failure handler, preemptor is woken up again and ultimately scheduled.
			Name: "victim blocked in binding, preemptor pod gets scheduled after victim-in-binding is deleted and its resources are unreserved",
			Steps: []asyncframework.Step{
				{
					Name:       "create Node",
					CreateNode: "node",
				},
				{
					Name: "create victim Pod that is going to be blocked in binding",
					CreatePod: &asyncframework.CreatePod{
						Pod: st.MakePod().Name(asyncframework.PodBlockedInBindingName + asyncframework.ReservingPodName).Req(map[v1.ResourceName]string{v1.ResourceCPU: "2"}).Container("image").ZeroTerminationGracePeriod().Priority(1).Obj(),
					},
				},
				{
					Name: "schedule victim Pod",
					SchedulePod: &asyncframework.SchedulePod{
						PodName: asyncframework.PodBlockedInBindingName + asyncframework.ReservingPodName,
					},
				},
				{
					Name: "create pod group for preemptor",
					CreatePodGroup: &asyncframework.CreatePodGroup{
						PodGroup: st.MakePodGroup().Name("pg-preemptor").MinCount(1).Priority(100).Obj(),
					},
				},
				{
					Name: "create a preemptor Pod",
					CreatePod: &asyncframework.CreatePod{
						Pod: st.MakePod().Name("preemptor").Req(map[v1.ResourceName]string{v1.ResourceCPU: "4"}).Container("image").Priority(100).PodGroupName("pg-preemptor").Obj(),
					},
				},
				{
					Name: "schedule the preemptor Pod",
					SchedulePodGroup: &asyncframework.SchedulePodGroup{
						PodGroupName:        "pg-preemptor",
						ExpectUnschedulable: true,
					},
				},
				{
					Name:               "complete the preemption API call",
					CompletePreemption: "pg-preemptor",
				},
				{
					Name: "schedule the preemptor Pod again and expect it to be unschedulable (resources are still reserved by the victim)",
					SchedulePodGroup: &asyncframework.SchedulePodGroup{
						PodGroupName:  "pg-preemptor",
						ExpectInQueue: true,
					},
				},
				{
					Name:       "resume binding of the blocked pod",
					ResumeBind: true,
				},
				{
					Name: "schedule the preemptor Pod again and expect it to be scheduled (victim pod unreserved its resources)",
					SchedulePodGroup: &asyncframework.SchedulePodGroup{
						PodGroupName:  "pg-preemptor",
						ExpectSuccess: true,
					},
				},
			},
		},
		{
			Name: "gated preemptor is eventually scheduled even if victim deletion doesn't raise queue hints",
			Steps: []asyncframework.Step{
				{
					Name:       "create Node",
					CreateNode: "node",
				},
				{
					Name: "create victim pods",
					CreatePod: &asyncframework.CreatePod{
						Pod:   st.MakePod().GenerateName(fmt.Sprintf("victim-%s-", asyncframework.BlockingPodName)).Node("node").Priority(1).Container("image").ZeroTerminationGracePeriod().Obj(),
						Count: new(2),
					},
				},
				{
					Name: "create pod group for preemptor",
					CreatePodGroup: &asyncframework.CreatePodGroup{
						PodGroup: st.MakePodGroup().Name("pg-preemptor").MinCount(1).Priority(100).Obj(),
					},
				},
				{
					Name: "create preemptor",
					CreatePod: &asyncframework.CreatePod{
						Pod: st.MakePod().Name("preemptor").Priority(100).Container("image").PodGroupName("pg-preemptor").Obj(),
					},
				},
				{
					Name: "schedule preemptor",
					SchedulePodGroup: &asyncframework.SchedulePodGroup{
						PodGroupName:        "pg-preemptor",
						ExpectUnschedulable: true,
					},
				},
				{
					Name:                 "verify preemptor running preemption",
					PodRunningPreemption: new(2),
				},
				{
					Name:            "gate preemptor",
					PodGatedInQueue: "preemptor",
				},
				{
					Name:               "complete preemption",
					CompletePreemption: "pg-preemptor",
				},
				{
					Name:               "wait for victims to be deleted",
					WaitForPodsDeleted: []int{0, 1},
				},
				{
					Name:                     "verify preemptor is still in unschedulable queue",
					VerifyPodInUnschedulable: "preemptor",
				},
				{
					Name:               "flush scheduling queue",
					FlushUnschedulable: true,
				},
				{
					Name: "verify preemptor scheduled",
					SchedulePodGroup: &asyncframework.SchedulePodGroup{
						PodGroupName:  "pg-preemptor",
						ExpectSuccess: true,
					},
				},
			},
		},
		{
			// This scenario verifies the fix for https://github.com/kubernetes/kubernetes/issues/134217
			// Scenario reproduces the issue, but with a victim that is under graceful termination and sis reserving some resources required by the preemptor:
			// Victim pod takes long in binding. Preemptor pod attempts preemption, goes to unschedulable, then the victim's graceful termination is initiated.
			// Preemptor pod is woken up by the Pod/Update event (working like AssignedPodDeleted), but is still unschedulable, because victim has to unreserve its resources.
			// After resuming binding for a victim, it releases the resources in its failure handler, preemptor is woken up again and ultimately scheduled.
			Name: "victim blocked in binding, preemptor pod gets scheduled after victim-in-binding is under graceful termination and its resources are unreserved",
			Steps: []asyncframework.Step{
				{
					Name:       "create Node",
					CreateNode: "node",
				},
				{
					Name: "create victim Pod that is going to be blocked in binding",
					CreatePod: &asyncframework.CreatePod{
						Pod: st.MakePod().Name(asyncframework.PodBlockedInBindingName + asyncframework.ReservingPodName).Req(map[v1.ResourceName]string{v1.ResourceCPU: "2"}).Container("image").TerminationGracePeriodSeconds(1000).Priority(1).Obj(),
					},
				},
				{
					Name: "schedule victim Pod",
					SchedulePod: &asyncframework.SchedulePod{
						PodName: asyncframework.PodBlockedInBindingName + asyncframework.ReservingPodName,
					},
				},
				{
					Name: "create pod group for preemptor",
					CreatePodGroup: &asyncframework.CreatePodGroup{
						PodGroup: st.MakePodGroup().Name("pg-preemptor").MinCount(1).Priority(100).Obj(),
					},
				},
				{
					Name: "create a preemptor Pod",
					CreatePod: &asyncframework.CreatePod{
						Pod: st.MakePod().Name("preemptor").Req(map[v1.ResourceName]string{v1.ResourceCPU: "4"}).Container("image").Priority(100).PodGroupName("pg-preemptor").Obj(),
					},
				},
				{
					Name: "schedule the preemptor Pod",
					SchedulePodGroup: &asyncframework.SchedulePodGroup{
						PodGroupName:        "pg-preemptor",
						ExpectUnschedulable: true,
					},
				},
				{
					Name:               "complete the preemption API call",
					CompletePreemption: "pg-preemptor",
				},
				{
					Name: "schedule the preemptor Pod again and expect it to be unschedulable (resources are still reserved by the victim)",
					SchedulePodGroup: &asyncframework.SchedulePodGroup{
						PodGroupName:  "pg-preemptor",
						ExpectInQueue: true,
					},
				},
				{
					Name:       "resume binding of the blocked pod",
					ResumeBind: true,
				},
				{
					Name: "schedule the preemptor Pod again and expect it to be scheduled (victim pod unreserved its resources)",
					SchedulePodGroup: &asyncframework.SchedulePodGroup{
						PodGroupName:  "pg-preemptor",
						ExpectSuccess: true,
					},
				},
			},
		},
		{
			// This scenario verifies that when preemption is in progress and a higher priotiy pod comes it will take place created for lower priority preemptor.
			// The lower priority Pod switches to another node, does not get stuck in unschedulable queue forever.
			Name: "While lower priority Pod is waiting for preemption, higher priority Pod takes its place on the node",
			Steps: []asyncframework.Step{
				{
					Name:       "create Node",
					CreateNode: "node",
				},
				{
					Name: "create N-1 victim Pods on the first node",
					CreatePod: &asyncframework.CreatePod{
						Pod:   st.MakePod().GenerateName("victim-").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Node("node").Container("image").ZeroTerminationGracePeriod().Priority(1).Obj(),
						Count: new(3),
					},
				},
				{
					Name: "create the last victim Pod on the first node, that is going to be blocked in binding",
					CreatePod: &asyncframework.CreatePod{
						Pod: st.MakePod().Name(asyncframework.PodBlockedInBindingName).Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").ZeroTerminationGracePeriod().Priority(1).Obj(),
					},
				},
				{
					Name: "schedule the last victim Pod",
					SchedulePod: &asyncframework.SchedulePod{
						PodName: asyncframework.PodBlockedInBindingName,
					},
				},
				{
					Name: "create pod group for preemptor",
					CreatePodGroup: &asyncframework.CreatePodGroup{
						PodGroup: st.MakePodGroup().Name("pg-preemptor").MinCount(1).Priority(50).Obj(),
					},
				},
				{
					Name: "create a mid-priority preemptor Pod",
					CreatePod: &asyncframework.CreatePod{
						Pod: st.MakePod().Name("preemptor-mid-priority").Req(map[v1.ResourceName]string{v1.ResourceCPU: "4"}).Container("image").Priority(50).PodGroupName("pg-preemptor").Obj(),
					},
				},
				{
					Name: "schedule the mid-priority preemptor Pod",
					SchedulePodGroup: &asyncframework.SchedulePodGroup{
						PodGroupName: "pg-preemptor",
					},
				},
				{
					Name:               "complete the preemption API calls",
					CompletePreemption: "pg-preemptor",
				},
				{
					Name:            "check the mid-priority preemptor Pod is gated, waiting for the last victim to be preempted",
					PodGatedInQueue: "preemptor-mid-priority",
				},
				{
					Name:       "create node2",
					CreateNode: "node2",
				},
				{
					Name: "create victim Pods on node2",
					CreatePod: &asyncframework.CreatePod{
						Pod:   st.MakePod().GenerateName("victim-").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Node("node2").Container("image").ZeroTerminationGracePeriod().Priority(1).Obj(),
						Count: new(4),
					},
				},
				{
					Name: "create pod group for high priority preemptor",
					CreatePodGroup: &asyncframework.CreatePodGroup{
						PodGroup: st.MakePodGroup().Name("pg-high-priority-preemptor").MinCount(1).Priority(100).Obj(),
					},
				},
				{
					Name: "create a high-priority preemptor Pod",
					CreatePod: &asyncframework.CreatePod{
						Pod: st.MakePod().Name("preemptor-high-priority").Req(map[v1.ResourceName]string{v1.ResourceCPU: "2"}).Container("image").Priority(100).PodGroupName("pg-high-priority-preemptor").Obj(),
					},
				},
				{
					Name: "schedule the high-priority preemptor Pod and expect it to get scheduled on node1",
					// While we don't check explicitly that Pod is scheduled on node1, we can assume that because
					// Pod won't fit on node2 without preemption and there are enough resources on node1.
					SchedulePodGroup: &asyncframework.SchedulePodGroup{
						PodGroupName:  "pg-high-priority-preemptor",
						ExpectSuccess: true,
					},
				},
				{
					Name:       "allow the preemption of the last victim Pod on node1 to finish",
					ResumeBind: true,
				},
				{
					Name: "check that mid-priority preemptor Pod got activated by completed preemption and try scheduling it again",
					SchedulePodGroup: &asyncframework.SchedulePodGroup{
						PodGroupName: "pg-preemptor",
						// Pod won't fit on node1 anymore and should trigger preemptions on node2.
						ExpectUnschedulable: true,
					},
				},
				{
					Name:               "complete the preemption API calls on node2",
					CompletePreemption: "pg-preemptor",
				},
				{
					Name: "check that mid-priority Pod got activated, schedule it on node2",
					SchedulePodGroup: &asyncframework.SchedulePodGroup{
						PodGroupName:  "pg-preemptor",
						ExpectSuccess: true,
					},
				},
			},
		},
	}
	for _, test := range tests {
		t.Run(test.Name, func(t *testing.T) {
			// map[string]chan struct{}
			preemptionDoneChannels := &sync.Map{}
			defer func() {
				preemptionDoneChannels.Range(func(key, value any) bool {
					ch := value.(chan struct{})
					select {
					case <-ch:
					default:
						close(ch)
					}
					return true
				})
			}()

			blockBindingChannel := make(chan struct{})
			defer close(blockBindingChannel)
			preemptionConfig := asyncframework.AsyncPreemptionTestConfig{
				EnableGenericWorkload:  true,
				PreemptionDoneChannels: preemptionDoneChannels,
				BlockBindingChannel:    blockBindingChannel,
			}
			testCtx, preemptionPlugin, cs := asyncframework.InitTestForAsyncPreemption(t, preemptionConfig)

			logger, _ := ktesting.NewTestContext(t)
			if testCtx.Scheduler.APIDispatcher != nil {
				testCtx.Scheduler.APIDispatcher.Run(logger)
				defer testCtx.Scheduler.APIDispatcher.Close()
			}
			testCtx.Scheduler.SchedulingQueue.Run(logger)
			defer testCtx.Scheduler.SchedulingQueue.Close()

			createdPods := []*v1.Pod{}
			defer func() {
				testutils.CleanupPods(testCtx.Ctx, cs, t, createdPods)
			}()

			config := asyncframework.AsyncPreemptionStepRunnerConfig{
				CreatedPods:            createdPods,
				ClientSet:              cs,
				PreemptionDoneChannels: preemptionDoneChannels,
				Logger:                 logger,
				PreemptionPlugin:       preemptionPlugin,
				BlockBindingChannel:    blockBindingChannel,
			}
			asyncframework.RunAsyncPreemptionSteps(testCtx, t, test.Steps, config)
		})
	}
}

// TestDisablePodGroupPreemption verifies that podgroup preemption does not happen if default preemption plugin is disabled.
func TestDisablePodGroupPreemption(t *testing.T) {
	for _, asyncPreemptionEnabled := range []bool{true, false} {
		t.Run(fmt.Sprintf("TestDisablePodGroupPreemption (Async preemption enabled: %v)", asyncPreemptionEnabled), func(t *testing.T) {
			featuregatetesting.SetFeatureGatesDuringTest(t, utilfeature.DefaultFeatureGate, featuregatetesting.FeatureOverrides{
				features.GenericWorkload:          true,
				features.SchedulerAsyncPreemption: asyncPreemptionEnabled,
			})

			// Initialize scheduler, and disable preemption.
			testCtx := testutils.InitTestDisablePreemption(t, "disable-preemption")
			cs := testCtx.ClientSet

			// Create a node with some resources
			nodeRes := map[v1.ResourceName]string{
				v1.ResourcePods:   "32",
				v1.ResourceCPU:    "500m",
				v1.ResourceMemory: "500",
			}
			_, err := testutils.CreateNode(cs, st.MakeNode().Name("node1").Capacity(nodeRes).Obj())
			if err != nil {
				t.Fatalf("Error creating nodes: %v", err)
			}

			// Create and run existingPod.
			existingPod := testutils.InitPausePod(&testutils.PausePodConfig{
				Name:      "victim-pod",
				Namespace: testCtx.NS.Name,
				Priority:  &asyncframework.LowPriority,
				Resources: &v1.ResourceRequirements{Requests: v1.ResourceList{
					v1.ResourceCPU:    *resource.NewMilliQuantity(400, resource.DecimalSI),
					v1.ResourceMemory: *resource.NewQuantity(200, resource.DecimalSI)},
				},
			})
			_, err = testutils.RunPausePod(cs, existingPod)
			if err != nil {
				t.Fatalf("TestDisablePodGroupPreemption (Async preemption enabled: %v): Error running pause pod: %v", asyncPreemptionEnabled, err)
			}

			// Create pod group.
			podGroup := st.MakePodGroup().Namespace(testCtx.NS.Name).Name("pg-preemptor").MinCount(1).Priority(asyncframework.HighPriority).Obj()
			if _, err := cs.SchedulingV1beta1().PodGroups(testCtx.NS.Name).Create(testCtx.Ctx, podGroup, metav1.CreateOptions{}); err != nil {
				t.Fatalf("Error creating pod group: %v", err)
			}

			// Create the preemptor pod.
			preemptor := testutils.InitPausePod(&testutils.PausePodConfig{
				Name:         "preemptor-pod",
				Namespace:    testCtx.NS.Name,
				Priority:     &asyncframework.HighPriority,
				PodGroupName: "pg-preemptor",
				Resources: &v1.ResourceRequirements{Requests: v1.ResourceList{
					v1.ResourceCPU:    *resource.NewMilliQuantity(300, resource.DecimalSI),
					v1.ResourceMemory: *resource.NewQuantity(200, resource.DecimalSI)},
				},
			})
			preemptor, err = testutils.CreatePausePod(cs, preemptor)
			if err != nil {
				t.Errorf("Error while creating high priority pod: %v", err)
			}
			// Ensure preemption did not happened.
			// For preemption to happened, preemptor should be scheduled or unscheduled with nominated node name.
			// So if the preemptor is unscheduled and does not have nominated node name, it means preemption did not happened.
			if err := testutils.WaitForPodUnschedulable(testCtx.Ctx, cs, preemptor); err != nil {
				t.Errorf("Preemptor %v should not become scheduled", preemptor.Name)
			}
			if err := testutils.WaitForNominatedNodeNameWithTimeout(testCtx.Ctx, cs, preemptor, 5*time.Second); err == nil {
				t.Errorf("Preemptor %v should not be nominated to any node", preemptor.Name)
			}
		})
	}
}

// TestPodGroupPreemptionRespectsWaitingPod tests that preemption respects pods that are waiting in the Permit phase
// (WaitOnPermit), simulating putting a pod in the waiting pods map with a custom permit plugin.
// There is equivalent test for pod by pod preemption at: test/integration/scheduler/preemption/preemption_test.go
// When adding new test cases for pod group preemption with waiting pods, add them to this test.
func TestPodGroupPreemptionRespectsWaitingPod(t *testing.T) {
	featuregatetesting.SetFeatureGatesDuringTest(t, utilfeature.DefaultFeatureGate, featuregatetesting.FeatureOverrides{
		features.GenericWorkload: true,
	})
	tests := []struct {
		name      string
		podGroups []*schedulingv1beta1.PodGroup
		victims   []*v1.Pod
		preemptor *v1.Pod
	}{
		{
			name: "preemptor without podGroup and victims in podGroup",
			podGroups: []*schedulingv1beta1.PodGroup{
				st.MakePodGroup().Name("pg1").Priority(10).MinCount(2).Obj(),
			},
			victims: []*v1.Pod{
				st.MakePod().Name("victim-1").Priority(10).PodGroupName("pg1").
					Req(map[v1.ResourceName]string{v1.ResourceCPU: "1", v1.ResourceMemory: "1Gi"}).
					ZeroTerminationGracePeriod().Obj(),
				st.MakePod().Name("victim-2").Priority(10).PodGroupName("pg1").
					Req(map[v1.ResourceName]string{v1.ResourceCPU: "1", v1.ResourceMemory: "1Gi"}).
					ZeroTerminationGracePeriod().Obj(),
			},
			preemptor: st.MakePod().Name("preemptor").Priority(100).
				Req(map[v1.ResourceName]string{v1.ResourceCPU: "2", v1.ResourceMemory: "1.5Gi"}).
				ZeroTerminationGracePeriod().Obj(),
		},
		{
			name: "preemptor in podGroup and victim without podGroup",
			podGroups: []*schedulingv1beta1.PodGroup{
				st.MakePodGroup().Name("preemptor-pg").Priority(100).MinCount(1).Obj(),
			},
			victims: []*v1.Pod{
				st.MakePod().Name("victim-1").Priority(10).
					Req(map[v1.ResourceName]string{v1.ResourceCPU: "1", v1.ResourceMemory: "1Gi"}).
					ZeroTerminationGracePeriod().Obj(),
				st.MakePod().Name("victim-2").Priority(10).
					Req(map[v1.ResourceName]string{v1.ResourceCPU: "1", v1.ResourceMemory: "1Gi"}).
					ZeroTerminationGracePeriod().Obj(),
			},
			preemptor: st.MakePod().Name("preemptor").Priority(100).PodGroupName("preemptor-pg").
				Req(map[v1.ResourceName]string{v1.ResourceCPU: "2", v1.ResourceMemory: "1.5Gi"}).
				ZeroTerminationGracePeriod().Obj(),
		},
		{
			name: "both preemptor and victims in podGroups",
			podGroups: []*schedulingv1beta1.PodGroup{
				st.MakePodGroup().Name("victim-pg").Priority(10).MinCount(2).Obj(),
				st.MakePodGroup().Name("preemptor-pg").Priority(100).MinCount(1).Obj(),
			},
			victims: []*v1.Pod{
				st.MakePod().Name("victim-1").Priority(10).PodGroupName("victim-pg").
					Req(map[v1.ResourceName]string{v1.ResourceCPU: "1", v1.ResourceMemory: "1Gi"}).
					ZeroTerminationGracePeriod().Obj(),
				st.MakePod().Name("victim-2").Priority(10).PodGroupName("victim-pg").
					Req(map[v1.ResourceName]string{v1.ResourceCPU: "1", v1.ResourceMemory: "1Gi"}).
					ZeroTerminationGracePeriod().Obj(),
			},
			preemptor: st.MakePod().Name("preemptor").Priority(100).PodGroupName("preemptor-pg").
				Req(map[v1.ResourceName]string{v1.ResourceCPU: "2", v1.ResourceMemory: "1.5Gi"}).
				ZeroTerminationGracePeriod().Obj(),
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			podsToBlock := make(map[string]*asyncframework.BlockedPod, len(tt.victims))
			for _, v := range tt.victims {
				podsToBlock[v.Name] = &asyncframework.BlockedPod{
					Blocked: make(chan struct{}, 1),
				}
			}

			registry := make(frameworkruntime.Registry)
			err := registry.Register(PersistentBlockingPermitPluginName, func(ctx context.Context, obj runtime.Object, fh fwk.Handle) (fwk.Plugin, error) {
				return &persistentBlockingPermitPlugin{podsToBlock: podsToBlock}, nil
			})
			if err != nil {
				t.Fatalf("Error registering plugin: %v", err)
			}

			cfg := configtesting.V1ToInternalWithDefaults(t, configv1.KubeSchedulerConfiguration{
				Profiles: []configv1.KubeSchedulerProfile{{
					SchedulerName: new(v1.DefaultSchedulerName),
					Plugins: &configv1.Plugins{
						Permit: configv1.PluginSet{
							Enabled: []configv1.Plugin{
								{Name: PersistentBlockingPermitPluginName},
							},
						},
					},
				}},
			})

			testCtx := testutils.InitTestSchedulerWithNS(t, "podgroup-waiting-preemption",
				scheduler.WithProfiles(cfg.Profiles...),
				scheduler.WithFrameworkOutOfTreeRegistry(registry),
				scheduler.WithPodMaxBackoffSeconds(0),
				scheduler.WithPodInitialBackoffSeconds(0))
			cs, ns := testCtx.ClientSet, testCtx.NS.Name

			// Create a node with resources for only one set of pods (victims or preemptor).
			nodeRes := map[v1.ResourceName]string{
				v1.ResourceCPU:    "2",
				v1.ResourceMemory: "2Gi",
			}
			node := st.MakeNode().Name("big-node").Capacity(nodeRes).Obj()
			if _, err := cs.CoreV1().Nodes().Create(testCtx.Ctx, node, metav1.CreateOptions{}); err != nil {
				t.Fatalf("Error creating node: %v", err)
			}

			for _, pg := range tt.podGroups {
				pg := pg.DeepCopy()
				pg.Namespace = ns
				if _, err := cs.SchedulingV1beta1().PodGroups(ns).Create(testCtx.Ctx, pg, metav1.CreateOptions{}); err != nil {
					t.Fatalf("Failed to create PodGroup %s: %v", pg.Name, err)
				}
			}

			createdVictims := make([]*v1.Pod, len(tt.victims))
			for i, v := range tt.victims {
				v := v.DeepCopy()
				v.Namespace = ns
				createdVictim, err := cs.CoreV1().Pods(ns).Create(testCtx.Ctx, v, metav1.CreateOptions{})
				if err != nil {
					t.Fatalf("Error creating victim %s: %v", v.Name, err)
				}
				createdVictims[i] = createdVictim
			}

			for _, v := range tt.victims {
				select {
				case <-podsToBlock[v.Name].Blocked:
					t.Logf("%s reached WaitOnPermit", v.Name)
				case <-time.After(15 * time.Second):
					t.Fatalf("Timed out waiting for %s to reach WaitOnPermit", v.Name)
				}
			}
			if err := wait.PollUntilContextTimeout(testCtx.Ctx, 100*time.Millisecond, 15*time.Second, false, func(ctx context.Context) (bool, error) {
				for _, cv := range createdVictims {
					if wp := testCtx.Scheduler.Profiles[v1.DefaultSchedulerName].GetWaitingPod(cv.UID); wp == nil {
						return false, nil
					}
				}
				return true, nil
			}); err != nil {
				t.Fatalf("Timed out waiting for victim pods to be recorded as waiting pods: %v", err)
			}

			preemptor := tt.preemptor.DeepCopy()
			preemptor.Namespace = ns
			t.Logf("Creating preemptor pod")
			if _, err := cs.CoreV1().Pods(ns).Create(testCtx.Ctx, preemptor, metav1.CreateOptions{}); err != nil {
				t.Fatalf("Error creating preemptor: %v", err)
			}

			t.Logf("Waiting for preemptor to be scheduled")
			err = wait.PollUntilContextTimeout(testCtx.Ctx, 100*time.Millisecond, 15*time.Second, false, testutils.PodScheduled(cs, ns, preemptor.Name))
			if err != nil {
				t.Fatalf("Failed waiting for preemptor to schedule: %v", err)
			}

			p, err := cs.CoreV1().Pods(ns).Get(testCtx.Ctx, preemptor.Name, metav1.GetOptions{})
			if err != nil {
				t.Fatalf("Error getting preemptor: %v", err)
			}
			if p.Spec.NodeName != "big-node" {
				t.Fatalf("Preemptor should be scheduled on big-node, but was scheduled on %s", p.Spec.NodeName)
			}

			for _, v := range tt.victims {
				gotVictim, err := cs.CoreV1().Pods(ns).Get(testCtx.Ctx, v.Name, metav1.GetOptions{})
				if err != nil {
					t.Fatalf("Error getting %s at the end: %v", v.Name, err)
				}
				if gotVictim.Spec.NodeName != "" {
					t.Fatalf("%s's NodeName should be empty, but it is %s", v.Name, gotVictim.Spec.NodeName)
				}
			}
		})
	}
}

// mockBindPlugin is a fake BindPlugin that registers NNN information during binding.
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

// eventRecorder is a simple thread-safe recorder of events for testing purposes.
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

// mockPodGroupPostFilterPlugin is a fake PodGroupPostFilterPlugin that registers pod group information during PodGroupPostFilter phase.
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

// mockReservePlugin is a fake ReservePlugin and FilterPlugin it records the number of reserved pods with "test-plugin" label.
// And returns unschedulable status ("already reserved") in Filter stage, if reserved pods count >= maxPods
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

// mockScorePlugin is a fake ScorePlugin and PlacementScorePlugin it returns the score for a node
// (based on a predefine map of node names to scores) for the given pod.
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

func (p *mockScorePlugin) PlacementScoreExtensions() fwk.PlacementScoreExtensions {
	return nil
}

func (p *mockScorePlugin) ScorePlacement(ctx context.Context, state fwk.PlacementCycleState, podGroup fwk.PodGroupInfo, placement *fwk.PodGroupAssignments) (int64, *fwk.Status) {
	if len(placement.ProposedAssignments) == 0 {
		return 0, nil
	}
	var total int64
	for _, pa := range placement.ProposedAssignments {
		if score, ok := p.scores[pa.GetNodeName()]; ok {
			total += score
		}
	}
	return total / int64(len(placement.ProposedAssignments)), nil
}

var _ fwk.ScorePlugin = &mockScorePlugin{}
var _ fwk.PlacementScorePlugin = &mockScorePlugin{}

// stateSaverKey is a key used by stateSaverPreFilterPlugin and stateVerifierPostFilterPlugin to store and retrieve pod names.
const stateSaverKey fwk.StateKey = "stateSaverKey"

// podGroupStateData is a mock struct used to save and retrieve pod names.
type podGroupStateData struct {
	podNames []string
}

func (d *podGroupStateData) Clone() fwk.StateData {
	return &podGroupStateData{
		podNames: append([]string(nil), d.podNames...),
	}
}

// stateSaverPreFilterPlugin is a mock PreFilterPlugin that saves the pod names of all pods in the cluster to the pod group cycle state.
type stateSaverPreFilterPlugin struct {
	name   string
	handle fwk.Handle
}

func (p *stateSaverPreFilterPlugin) Name() string {
	return p.name
}

func (p *stateSaverPreFilterPlugin) PreFilter(ctx context.Context, state fwk.CycleState, pod *v1.Pod, nodes []fwk.NodeInfo) (*fwk.PreFilterResult, *fwk.Status) {
	if !state.IsPodGroupSchedulingCycle() {
		return nil, nil
	}
	pgState := state.GetPodGroupSchedulingCycle()
	if pgState == nil {
		return nil, nil
	}

	var podNames []string
	if p.handle != nil && p.handle.SnapshotSharedLister() != nil {
		nodeInfos, err := p.handle.SnapshotSharedLister().NodeInfos().List()
		if err == nil {
			for _, nodeInfo := range nodeInfos {
				for _, podInfo := range nodeInfo.GetPods() {
					if podInfo != nil && podInfo.GetPod() != nil {
						podNames = append(podNames, podInfo.GetPod().Name)
					}
				}
			}
		}
	}

	pgState.Write(stateSaverKey, &podGroupStateData{podNames: podNames})
	return nil, nil
}

func (p *stateSaverPreFilterPlugin) PreFilterExtensions() fwk.PreFilterExtensions {
	return nil
}

var _ fwk.PreFilterPlugin = &stateSaverPreFilterPlugin{}

// stateVerifierPostFilterPlugin is a mock PodGroupPostFilterPlugin that reads the pod names from pod group cycle state.
type stateVerifierPostFilterPlugin struct {
	name             string
	called           bool
	recordedPodNames []string
	readErr          error
	lock             sync.Mutex
}

func (p *stateVerifierPostFilterPlugin) Name() string {
	return p.name
}

func (p *stateVerifierPostFilterPlugin) PodGroupPostFilter(ctx context.Context, state fwk.PodGroupCycleState, pgInfo fwk.PodGroupInfo, pgSchedulingFunc fwk.PodGroupSchedulingFunc) (*fwk.PodGroupPostFilterResult, *fwk.Status) {
	p.lock.Lock()
	defer p.lock.Unlock()
	p.called = true

	data, err := state.Read(stateSaverKey)
	p.readErr = err
	if err == nil {
		if stateData, ok := data.(*podGroupStateData); ok {
			p.recordedPodNames = append([]string(nil), stateData.podNames...)
		}
	}
	return nil, fwk.NewStatus(fwk.Unschedulable, "injected verifier PostFilter")
}

var _ fwk.PodGroupPostFilterPlugin = &stateVerifierPostFilterPlugin{}

// persistentBlockingPermitPlugin is a Permit plugin that returns Wait on every Permit call for blocked pods without deleting them.
// This is necessary for PodGroup scheduling where Permit is called twice (during evaluation and prepareForBindingCycle).
type persistentBlockingPermitPlugin struct {
	podsToBlock map[string]*asyncframework.BlockedPod
}

const PersistentBlockingPermitPluginName = "persistent-blocking-permit-plugin"

func (pl *persistentBlockingPermitPlugin) Name() string {
	return PersistentBlockingPermitPluginName
}

func (pl *persistentBlockingPermitPlugin) Permit(ctx context.Context, state fwk.CycleState, pod *v1.Pod, nodeName string) (*fwk.Status, time.Duration) {
	if p, ok := pl.podsToBlock[pod.Name]; ok {
		select {
		case p.Blocked <- struct{}{}:
		default:
		}
		return fwk.NewStatus(fwk.Wait, "waiting"), time.Minute
	}
	return nil, 0
}

var _ fwk.PermitPlugin = &persistentBlockingPermitPlugin{}

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
