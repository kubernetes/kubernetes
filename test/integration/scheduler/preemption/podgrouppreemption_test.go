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
	"strings"
	"sync"
	"testing"
	"time"

	v1 "k8s.io/api/core/v1"
	policyv1 "k8s.io/api/policy/v1"
	schedulingapi "k8s.io/api/scheduling/v1alpha3"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/intstr"
	"k8s.io/apimachinery/pkg/util/wait"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
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
		podGroups     []*schedulingapi.PodGroup
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
	}{
		{
			name: "Full PodGroup Preemption",
			nodes: []*v1.Node{
				st.MakeNode().Name("node1").Capacity(map[v1.ResourceName]string{v1.ResourceCPU: "3", v1.ResourceMemory: "4Gi", v1.ResourcePods: "32"}).Obj(),
			},
			podGroups: []*schedulingapi.PodGroup{
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
			podGroups: []*schedulingapi.PodGroup{
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
			podGroups: []*schedulingapi.PodGroup{
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
			// high-3 will fit on node2 (it has 1 CPU free).
			// high-1 and high-2 will fit on node1 if low-1 is preempted.
			expectedScheduled:          []string{"high-1", "high-2", "high-3", "low-2"},
			expectedPreempted:          []string{"low-1"},
			expectedToHaveNNNInfo:      []string{"high-1", "high-2", "high-3"},
			expectedPodsPreemptedByWAP: 1,
		},
		{
			name: "Partial Preemption with basic policy",
			nodes: []*v1.Node{
				st.MakeNode().Name("node1").Capacity(map[v1.ResourceName]string{v1.ResourceCPU: "2", v1.ResourceMemory: "4Gi", v1.ResourcePods: "32"}).Obj(),
				st.MakeNode().Name("node2").Capacity(map[v1.ResourceName]string{v1.ResourceCPU: "2", v1.ResourceMemory: "4Gi", v1.ResourcePods: "32"}).Obj(),
			},
			podGroups: []*schedulingapi.PodGroup{
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
			// with default scoring the assignments without victims will be
			// high-1, high-3 -> node1
			// high-2 -> node2
			// very-low-1 can be reprieved on node2
			expectedScheduled:          []string{"high-1", "high-2", "high-3", "very-low-1"},
			expectedPreempted:          []string{"low-1"},
			expectedToHaveNNNInfo:      []string{},
			expectedPodsPreemptedByWAP: 1,
		},
		{
			name: "PDB Violation Handling (Reprieve)",
			nodes: []*v1.Node{
				st.MakeNode().Name("node1").Capacity(map[v1.ResourceName]string{v1.ResourceCPU: "3", v1.ResourceMemory: "4Gi", v1.ResourcePods: "32"}).Obj(),
			},
			podGroups: []*schedulingapi.PodGroup{
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
			podGroups: []*schedulingapi.PodGroup{
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
			podGroups: []*schedulingapi.PodGroup{
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
			podGroups: []*schedulingapi.PodGroup{
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
			podGroups: []*schedulingapi.PodGroup{
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
			podGroups: []*schedulingapi.PodGroup{
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
			podGroups: []*schedulingapi.PodGroup{
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
			podGroups: []*schedulingapi.PodGroup{
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
			podGroups: []*schedulingapi.PodGroup{
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
			name: "Basic scheduling: do not reprieve if it reduces scheduled pods below max possible",
			nodes: []*v1.Node{
				st.MakeNode().Name("node1").Capacity(map[v1.ResourceName]string{v1.ResourceCPU: "1", v1.ResourceMemory: "4Gi", v1.ResourcePods: "32"}).Obj(),
				st.MakeNode().Name("node2").Capacity(map[v1.ResourceName]string{v1.ResourceCPU: "1", v1.ResourceMemory: "4Gi", v1.ResourcePods: "32"}).Obj(),
				st.MakeNode().Name("node3").Capacity(map[v1.ResourceName]string{v1.ResourceCPU: "1", v1.ResourceMemory: "4Gi", v1.ResourcePods: "32"}).Obj(),
			},
			podGroups: []*schedulingapi.PodGroup{
				st.MakePodGroup().Name("preemptor-pg").Namespace("default").Priority(100).BasicPolicy().Obj(),
			},
			initialPods: []*v1.Pod{
				st.MakePod().Name("p1").Node("node1").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").ZeroTerminationGracePeriod().Priority(10).Obj(),
				st.MakePod().Name("p2").Node("node2").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").ZeroTerminationGracePeriod().Priority(50).Obj(),
				st.MakePod().Name("p3").Node("node3").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").ZeroTerminationGracePeriod().Priority(50).Obj(),
			},
			preemptorPods: []*v1.Pod{
				st.MakePod().Name("p-a").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("preemptor-pg").ZeroTerminationGracePeriod().Priority(100).Obj(),
				st.MakePod().Name("p-b").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("preemptor-pg").ZeroTerminationGracePeriod().Priority(100).Obj(),
				st.MakePod().Name("p-c").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("preemptor-pg").ZeroTerminationGracePeriod().Priority(100).Obj(),
			},
			expectedScheduled:          []string{"p-a", "p-b", "p-c"},
			expectedPreempted:          []string{"p1", "p2", "p3"},
			expectedToHaveNNNInfo:      []string{"p-a", "p-b", "p-c"},
			expectedPodsPreemptedByWAP: 3,
		},
		{
			name: "Basic scheduling: reprieve if it does not reduce scheduled pods below max possible",
			nodes: []*v1.Node{
				st.MakeNode().Name("node1").Capacity(map[v1.ResourceName]string{v1.ResourceCPU: "1", v1.ResourceMemory: "4Gi", v1.ResourcePods: "32"}).Obj(),
				st.MakeNode().Name("node2").Capacity(map[v1.ResourceName]string{v1.ResourceCPU: "1", v1.ResourceMemory: "4Gi", v1.ResourcePods: "32"}).Obj(),
				st.MakeNode().Name("node3").Capacity(map[v1.ResourceName]string{v1.ResourceCPU: "1", v1.ResourceMemory: "4Gi", v1.ResourcePods: "32"}).Obj(),
				st.MakeNode().Name("node4").Capacity(map[v1.ResourceName]string{v1.ResourceCPU: "1", v1.ResourceMemory: "4Gi", v1.ResourcePods: "32"}).Obj(),
			},
			podGroups: []*schedulingapi.PodGroup{
				st.MakePodGroup().Name("preemptor-pg").Namespace("default").Priority(100).BasicPolicy().Obj(),
			},
			initialPods: []*v1.Pod{
				st.MakePod().Name("p1").Node("node1").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").ZeroTerminationGracePeriod().Priority(10).Obj(),
				st.MakePod().Name("p2").Node("node2").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").ZeroTerminationGracePeriod().Priority(10).Obj(),
				st.MakePod().Name("p3").Node("node3").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").ZeroTerminationGracePeriod().Priority(10).Obj(),
				st.MakePod().Name("p4").Node("node4").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").ZeroTerminationGracePeriod().Priority(50).Obj(),
			},
			preemptorPods: []*v1.Pod{
				st.MakePod().Name("p-a").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("preemptor-pg").ZeroTerminationGracePeriod().Priority(100).Obj(),
				st.MakePod().Name("p-b").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("preemptor-pg").ZeroTerminationGracePeriod().Priority(100).Obj(),
				st.MakePod().Name("p-c").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("preemptor-pg").ZeroTerminationGracePeriod().Priority(100).Obj(),
			},
			expectedScheduled:          []string{"p-a", "p-b", "p-c", "p4"},
			expectedPreempted:          []string{"p1", "p2", "p3"},
			expectedToHaveNNNInfo:      []string{"p-a", "p-b", "p-c"},
			expectedPodsPreemptedByWAP: 3,
		},
		{
			name: "Basic scheduling: schedule as many pods as possible without preempting higher priority pods",
			nodes: []*v1.Node{
				st.MakeNode().Name("node1").Capacity(map[v1.ResourceName]string{v1.ResourceCPU: "1", v1.ResourceMemory: "4Gi", v1.ResourcePods: "32"}).Obj(),
				st.MakeNode().Name("node2").Capacity(map[v1.ResourceName]string{v1.ResourceCPU: "1", v1.ResourceMemory: "4Gi", v1.ResourcePods: "32"}).Obj(),
				st.MakeNode().Name("node3").Capacity(map[v1.ResourceName]string{v1.ResourceCPU: "1", v1.ResourceMemory: "4Gi", v1.ResourcePods: "32"}).Obj(),
				st.MakeNode().Name("node4").Capacity(map[v1.ResourceName]string{v1.ResourceCPU: "1", v1.ResourceMemory: "4Gi", v1.ResourcePods: "32"}).Obj(),
			},
			podGroups: []*schedulingapi.PodGroup{
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
			podGroups: []*schedulingapi.PodGroup{
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
			podGroups: []*schedulingapi.PodGroup{
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
			name: "Reprieval allows more pods to schedule than initial maxScheduledCount due to greedy placement",
			nodes: []*v1.Node{
				st.MakeNode().Name("nodea").Label("topology.kubernetes.io/zone", "zoneA").Capacity(map[v1.ResourceName]string{v1.ResourceCPU: "2", v1.ResourceMemory: "4Gi", v1.ResourcePods: "32"}).Obj(),
				st.MakeNode().Name("nodeb").Label("topology.kubernetes.io/zone", "zoneB").Capacity(map[v1.ResourceName]string{v1.ResourceCPU: "3", v1.ResourceMemory: "4Gi", v1.ResourcePods: "32"}).Obj(),
			},
			podGroups: []*schedulingapi.PodGroup{
				st.MakePodGroup().Name("preemptor-pg").Namespace("default").Priority(100).BasicPolicy().Obj(),
			},
			initialPods: []*v1.Pod{
				st.MakePod().Name("va").Node("nodea").Req(map[v1.ResourceName]string{v1.ResourceCPU: "2"}).Container("image").ZeroTerminationGracePeriod().Priority(50).NodeAffinityIn("topology.kubernetes.io/zone",
					[]string{"zoneA"}, st.NodeSelectorTypeMatchExpressions).Obj(),
				st.MakePod().Name("vb").Node("nodeb").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").ZeroTerminationGracePeriod().Priority(10).NodeAffinityIn("topology.kubernetes.io/zone",
					[]string{"zoneB"}, st.NodeSelectorTypeMatchExpressions).Obj(),
			},
			preemptorPods: []*v1.Pod{
				st.MakePod().Name("p1").Label("pod", "preemptor").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("preemptor-pg").ZeroTerminationGracePeriod().Priority(100).
					NodeAffinity(&v1.NodeAffinity{
						PreferredDuringSchedulingIgnoredDuringExecution: []v1.PreferredSchedulingTerm{
							{Weight: 100, Preference: v1.NodeSelectorTerm{MatchExpressions: []v1.NodeSelectorRequirement{{Key: "topology.kubernetes.io/zone", Operator: v1.NodeSelectorOpIn, Values: []string{"zoneA"}}}}},
						},
					}).
					PodAffinityExists("pod", "topology.kubernetes.io/zone", st.PodAffinityWithRequiredReq).Obj(),
				st.MakePod().Name("p2").Label("pod", "preemptor").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("preemptor-pg").ZeroTerminationGracePeriod().Priority(100).
					NodeAffinity(&v1.NodeAffinity{
						PreferredDuringSchedulingIgnoredDuringExecution: []v1.PreferredSchedulingTerm{
							{Weight: 100, Preference: v1.NodeSelectorTerm{MatchExpressions: []v1.NodeSelectorRequirement{{Key: "topology.kubernetes.io/zone", Operator: v1.NodeSelectorOpIn, Values: []string{"zoneA"}}}}},
						},
					}).
					PodAffinityExists("pod", "topology.kubernetes.io/zone", st.PodAffinityWithRequiredReq).Obj(),
				st.MakePod().Name("p3").Label("pod", "preemptor").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("preemptor-pg").ZeroTerminationGracePeriod().Priority(100).
					NodeAffinity(&v1.NodeAffinity{
						PreferredDuringSchedulingIgnoredDuringExecution: []v1.PreferredSchedulingTerm{
							{Weight: 100, Preference: v1.NodeSelectorTerm{MatchExpressions: []v1.NodeSelectorRequirement{{Key: "topology.kubernetes.io/zone", Operator: v1.NodeSelectorOpIn, Values: []string{"zoneA"}}}}},
						},
					}).
					PodAffinityExists("pod", "topology.kubernetes.io/zone", st.PodAffinityWithRequiredReq).Obj(),
			},
			expectedScheduled:          []string{"p1", "p2", "p3", "va"},
			expectedPreempted:          []string{"vb"},
			expectedPodsPreemptedByWAP: 1,
		},
		{
			name: "Reprieval allows more pods to schedule than initial maxScheduledCount due to greedy placement (gang > minCount)",
			nodes: []*v1.Node{
				st.MakeNode().Name("nodea").Label("topology.kubernetes.io/zone", "zoneA").Capacity(map[v1.ResourceName]string{v1.ResourceCPU: "3", v1.ResourceMemory: "4Gi", v1.ResourcePods: "32"}).Obj(),
				st.MakeNode().Name("nodeb").Label("topology.kubernetes.io/zone", "zoneB").Capacity(map[v1.ResourceName]string{v1.ResourceCPU: "4", v1.ResourceMemory: "4Gi", v1.ResourcePods: "32"}).Obj(),
			},
			podGroups: []*schedulingapi.PodGroup{
				st.MakePodGroup().Name("preemptor-pg").Namespace("default").Priority(100).MinCount(3).Obj(),
			},
			initialPods: []*v1.Pod{
				st.MakePod().Name("va").Node("nodea").Req(map[v1.ResourceName]string{v1.ResourceCPU: "3"}).Container("image").ZeroTerminationGracePeriod().Priority(50).NodeAffinityIn("topology.kubernetes.io/zone", []string{"zoneA"}, st.NodeSelectorTypeMatchExpressions).Obj(),
				st.MakePod().Name("vb").Node("nodeb").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").ZeroTerminationGracePeriod().Priority(10).NodeAffinityIn("topology.kubernetes.io/zone", []string{"zoneB"}, st.NodeSelectorTypeMatchExpressions).Obj(),
			},
			preemptorPods: []*v1.Pod{
				st.MakePod().Name("p1").Label("pod", "p1").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("preemptor-pg").ZeroTerminationGracePeriod().Priority(100).
					NodeAffinity(&v1.NodeAffinity{
						PreferredDuringSchedulingIgnoredDuringExecution: []v1.PreferredSchedulingTerm{
							{
								Weight: 100,
								Preference: v1.NodeSelectorTerm{
									MatchExpressions: []v1.NodeSelectorRequirement{
										{Key: "topology.kubernetes.io/zone", Operator: v1.NodeSelectorOpIn, Values: []string{"zoneA"}},
									},
								},
							},
						},
					}).Obj(),
				st.MakePod().Name("p2").Label("pod", "p2").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("preemptor-pg").ZeroTerminationGracePeriod().Priority(100).
					PodAffinityExists("pod", "topology.kubernetes.io/zone", st.PodAffinityWithRequiredReq).Obj(),
				st.MakePod().Name("p3").Label("pod", "p3").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("preemptor-pg").ZeroTerminationGracePeriod().Priority(100).
					PodAffinityExists("pod", "topology.kubernetes.io/zone", st.PodAffinityWithRequiredReq).Obj(),
				st.MakePod().Name("p4").Label("pod", "p4").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("preemptor-pg").ZeroTerminationGracePeriod().Priority(100).
					PodAffinityExists("pod", "topology.kubernetes.io/zone", st.PodAffinityWithRequiredReq).Obj(),
			},
			expectedScheduled:          []string{"p1", "p2", "p3", "p4", "va"},
			expectedPreempted:          []string{"vb"},
			expectedPodsPreemptedByWAP: 1,
		},
		{
			name: "PodGroup with PreemptNever preemption policy does not perform preemption, with PodGroupPreemptionPolicy enabled",
			nodes: []*v1.Node{
				st.MakeNode().Name("node1").Capacity(map[v1.ResourceName]string{v1.ResourceCPU: "3", v1.ResourceMemory: "4Gi", v1.ResourcePods: "32"}).Obj(),
			},
			podGroups: []*schedulingapi.PodGroup{
				st.MakePodGroup().Name("pg1").Namespace("default").Priority(100).MinCount(3).PreemptionPolicy(schedulingapi.PreemptNever).Obj(),
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
			podGroups: []*schedulingapi.PodGroup{
				st.MakePodGroup().Name("pg1").Namespace("default").Priority(100).MinCount(3).PreemptionPolicy(schedulingapi.PreemptLowerPriority).Obj(),
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
			podGroups: []*schedulingapi.PodGroup{
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
			name: "PodGroup with PreemptNever preemption policy in one of the pods does not perform preemption, with PodGroupPreemptionPolicy disabled",
			nodes: []*v1.Node{
				st.MakeNode().Name("node1").Capacity(map[v1.ResourceName]string{v1.ResourceCPU: "3", v1.ResourceMemory: "4Gi", v1.ResourcePods: "32"}).Obj(),
			},
			podGroups: []*schedulingapi.PodGroup{
				st.MakePodGroup().Name("pg1").Namespace("default").Priority(100).MinCount(3).Obj(),
			},
			initialPods: []*v1.Pod{
				st.MakePod().Name("low-1").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").ZeroTerminationGracePeriod().Priority(10).Obj(),
				st.MakePod().Name("low-2").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").ZeroTerminationGracePeriod().Priority(10).Obj(),
				st.MakePod().Name("low-3").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").ZeroTerminationGracePeriod().Priority(10).Obj(),
			},
			preemptorPods: []*v1.Pod{
				st.MakePod().Name("high-1").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("pg1").ZeroTerminationGracePeriod().Priority(100).Obj(),
				st.MakePod().Name("high-2").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("pg1").ZeroTerminationGracePeriod().Priority(100).PreemptionPolicy(v1.PreemptNever).Obj(),
				st.MakePod().Name("high-3").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("pg1").ZeroTerminationGracePeriod().Priority(100).Obj(),
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
			podGroups: []*schedulingapi.PodGroup{
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
			podGroups: []*schedulingapi.PodGroup{
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
			podGroups: []*schedulingapi.PodGroup{
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
			podGroups: []*schedulingapi.PodGroup{
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
			podGroups: []*schedulingapi.PodGroup{
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
			podGroups: []*schedulingapi.PodGroup{
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
			podGroups: []*schedulingapi.PodGroup{
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
			podGroups: []*schedulingapi.PodGroup{
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
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			featuregatetesting.SetFeatureGatesDuringTest(t, utilfeature.DefaultFeatureGate, featuregatetesting.FeatureOverrides{
				features.GenericWorkload:          true,
				features.PodGroupPreemptionPolicy: tt.enablePodGroupPreemptionPolicy,
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

			cfg := configtesting.V1ToInternalWithDefaults(t, configv1.KubeSchedulerConfiguration{
				Profiles: []configv1.KubeSchedulerProfile{{
					SchedulerName: ptr.To(v1.DefaultSchedulerName),
					Plugins: &configv1.Plugins{
						MultiPoint: configv1.PluginSet{
							Enabled: []configv1.Plugin{
								{Name: mockBindPluginName},
							},
							Disabled: []configv1.Plugin{
								{Name: names.DefaultBinder},
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
				if _, err := cs.SchedulingV1alpha3().PodGroups(ns).Create(testCtx.Ctx, pg, metav1.CreateOptions{}); err != nil {
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

			// 4. Create preemptor pods
			for _, p := range tt.preemptorPods {
				p.Namespace = ns
				if _, err := cs.CoreV1().Pods(ns).Create(testCtx.Ctx, p, metav1.CreateOptions{}); err != nil {
					t.Fatalf("Failed to create pod %s: %v", p.Name, err)
				}
				if tt.preemptorPodsQueuedInCreationOrder {
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

// mockBindPlugin is a fake plugin that registers NNN information during binding.
type mockBindPlugin struct {
	name       string
	realPlugin fwk.BindPlugin
	nnnInfo    sync.Map
}

func (bp *mockBindPlugin) Name() string {
	return bp.name
}

func (bp *mockBindPlugin) Bind(ctx context.Context, state fwk.CycleState, p *v1.Pod, nodeName string) *fwk.Status {
	if p.Status.NominatedNodeName != "" {
		bp.nnnInfo.Store(p.Name, p.Status.NominatedNodeName)
	}
	return bp.realPlugin.Bind(ctx, state, p, nodeName)
}

var _ fwk.BindPlugin = &mockBindPlugin{}

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
