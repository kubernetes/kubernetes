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

package queueing

import (
	"context"
	"fmt"
	"testing"
	"time"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/wait"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	configv1 "k8s.io/kube-scheduler/config/v1"
	fwk "k8s.io/kube-scheduler/framework"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/scheduler"
	configtesting "k8s.io/kubernetes/pkg/scheduler/apis/config/testing"
	internalqueue "k8s.io/kubernetes/pkg/scheduler/backend/queue"
	frameworkruntime "k8s.io/kubernetes/pkg/scheduler/framework/runtime"
	st "k8s.io/kubernetes/pkg/scheduler/testing"
	stepsframework "k8s.io/kubernetes/test/integration/scheduler/podgroup/stepsframework"
	testutils "k8s.io/kubernetes/test/integration/util"
	"k8s.io/utils/ptr"
)

type podInjectorPlugin struct {
	handle          fwk.Handle
	schedulingQueue internalqueue.SchedulingQueue
	watchPod        string
	podToInject     *v1.Pod
	injected        bool
}

func newPodInjectorPlugin(handle fwk.Handle, watchPod string, podToInject *v1.Pod) *podInjectorPlugin {
	return &podInjectorPlugin{
		handle:      handle,
		watchPod:    watchPod,
		podToInject: podToInject,
	}
}

var _ fwk.PreFilterPlugin = &podInjectorPlugin{}

func (p *podInjectorPlugin) Name() string {
	return "PodInjectorPlugin"
}

func (p *podInjectorPlugin) PreFilter(ctx context.Context, state fwk.CycleState, pod *v1.Pod, nodes []fwk.NodeInfo) (*fwk.PreFilterResult, *fwk.Status) {
	if pod.Name == p.watchPod && !p.injected {
		p.injected = true
		p3 := p.podToInject.DeepCopy()
		p3.Namespace = pod.Namespace
		_, err := p.handle.ClientSet().CoreV1().Pods(pod.Namespace).Create(ctx, p3, metav1.CreateOptions{})
		if err != nil {
			return nil, fwk.AsStatus(fmt.Errorf("failed to create pod %s: %w", p3.Name, err))
		}
		err = wait.PollUntilContextTimeout(ctx, 100*time.Millisecond, 10*time.Second, false, func(ctx context.Context) (bool, error) {
			for _, p := range p.schedulingQueue.PendingPodGroupPods() {
				if p.Name == p3.Name {
					return true, nil
				}
			}
			return false, nil
		})
		if err != nil {
			return nil, fwk.AsStatus(fmt.Errorf("failed to wait for pod %s to be pending: %w", p3.Name, err))
		}
	}
	return nil, nil
}

func (p *podInjectorPlugin) PreFilterExtensions() fwk.PreFilterExtensions {
	return nil
}

type bindingFailingPlugin struct {
	watchPod string
	failed   bool
}

func newBindingFailingPlugin(watchPod string) *bindingFailingPlugin {
	return &bindingFailingPlugin{
		watchPod: watchPod,
	}
}

var _ fwk.PreBindPlugin = &bindingFailingPlugin{}

func (p *bindingFailingPlugin) Name() string {
	return "BindingFailingPlugin"
}

func (p *bindingFailingPlugin) PreBind(ctx context.Context, state fwk.CycleState, pod *v1.Pod, nodeName string) *fwk.Status {
	if pod.Name == p.watchPod && !p.failed {
		p.failed = true
		return fwk.NewStatus(fwk.Error, "mock binding failure")
	}
	return nil
}

func (p *bindingFailingPlugin) PreBindPreFlight(ctx context.Context, state fwk.CycleState, pod *v1.Pod, nodeName string) (*fwk.PreBindPreFlightResult, *fwk.Status) {
	return nil, nil
}

func TestPodGroupQueueing(t *testing.T) {
	node := st.MakeNode().Name("node").Capacity(map[v1.ResourceName]string{v1.ResourceCPU: "2"}).Obj()
	node2 := st.MakeNode().Name("node2").Capacity(map[v1.ResourceName]string{v1.ResourceCPU: "6"}).Obj()

	podGroupGang := st.MakePodGroup().Name("pg").MinCount(2).Obj()
	podGroupGangSingle := st.MakePodGroup().Name("pg").MinCount(1).Obj()
	podGroupGangTriple := st.MakePodGroup().Name("pg").MinCount(3).Obj()
	podGroupBasic := st.MakePodGroup().Name("pg").BasicPolicy().Obj()

	p1 := st.MakePod().Name("p1").Req(map[v1.ResourceName]string{v1.ResourceCPU: "2"}).Container("image").PodGroupName("pg").Obj()
	p2 := st.MakePod().Name("p2").Req(map[v1.ResourceName]string{v1.ResourceCPU: "2"}).Container("image").PodGroupName("pg").Obj()
	p3 := st.MakePod().Name("p3").Req(map[v1.ResourceName]string{v1.ResourceCPU: "2"}).Container("image").PodGroupName("pg").Obj()
	p4 := st.MakePod().Name("p4").Req(map[v1.ResourceName]string{v1.ResourceCPU: "2"}).Container("image").PodGroupName("pg").Obj()

	smallP1 := st.MakePod().Name("p1").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("pg").Obj()
	smallP2 := st.MakePod().Name("p2").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("pg").Obj()

	p1Gated := st.MakePod().Name("p1").Req(map[v1.ResourceName]string{v1.ResourceCPU: "2"}).Container("image").PodGroupName("pg").SchedulingGates([]string{"foo-gate"}).Obj()
	p3Gated := st.MakePod().Name("p3").Req(map[v1.ResourceName]string{v1.ResourceCPU: "2"}).Container("image").PodGroupName("pg").SchedulingGates([]string{"foo-gate"}).Obj()

	p2Large := st.MakePod().Name("p2").Req(map[v1.ResourceName]string{v1.ResourceCPU: "4"}).Container("image").PodGroupName("pg").Obj()

	type podInjector struct {
		watchPod    string
		podToInject *v1.Pod
	}

	type bindingFailing struct {
		watchPod string
	}

	tests := []struct {
		name string
		// podInjector injects the given pod when the watched pod is during PreFilter phase.
		// nil means that no pod is injected during scheduling.
		podInjector    *podInjector
		bindingFailing *bindingFailing
		steps          []stepsframework.Step
	}{
		{
			name: "Gang scheduling pod group queueing when new member pods are added",
			steps: []stepsframework.Step{
				{
					Name:           "Create Gang Policy PodGroup",
					CreatePodGroup: podGroupGang,
				},
				{
					Name:       "Create unschedulable member pods p1 and p2",
					CreatePods: []*v1.Pod{p1, p2},
				},
				{
					Name:                     "Verify p1 and p2 are unschedulable",
					WaitForPodsUnschedulable: []string{"p1", "p2"},
				},
				{
					Name:       "Add new pod p3 to the unschedulable pod group",
					CreatePods: []*v1.Pod{p3},
				},
				{
					Name:                     "Verify p1, p2 and p3 are unschedulable",
					WaitForPodsUnschedulable: []string{"p1", "p2", "p3"},
				},
				{
					Name:        "Add node2 to expand capacity",
					CreateNodes: []*v1.Node{node2},
				},
				{
					Name:                 "Verify all pods scheduled successfully",
					WaitForPodsScheduled: []string{"p1", "p2", "p3"},
				},
				{
					Name:       "Add new pod p4 to the scheduled pod group",
					CreatePods: []*v1.Pod{p4},
				},
				{
					Name:                 "Verify all pods scheduled successfully",
					WaitForPodsScheduled: []string{"p1", "p2", "p3", "p4"},
				},
			},
		},
		{
			name: "Basic policy pod group queueing when new member pods are added",
			steps: []stepsframework.Step{
				{
					Name:           "Create Basic Policy PodGroup",
					CreatePodGroup: podGroupBasic,
				},
				{
					Name:       "Create member pod p1",
					CreatePods: []*v1.Pod{p1},
				},
				{
					Name:                 "Verify p1 is scheduled",
					WaitForPodsScheduled: []string{"p1"},
				},
				{
					Name:       "Add new pod p2 to the scheduled pod group",
					CreatePods: []*v1.Pod{p2},
				},
				{
					Name:                     "Verify p2 is unschedulable",
					WaitForPodsUnschedulable: []string{"p2"},
				},
				{
					Name:       "Add new pod p3 to the unschedulable group",
					CreatePods: []*v1.Pod{p3},
				},
				{
					Name:                     "Verify p2 and p3 are unschedulable",
					WaitForPodsUnschedulable: []string{"p2", "p3"},
				},
				{
					Name:        "Add node2 to expand capacity",
					CreateNodes: []*v1.Node{node2},
				},
				{
					Name:                 "Verify all pods scheduled successfully",
					WaitForPodsScheduled: []string{"p1", "p2", "p3"},
				},
			},
		},
		{
			name: "Gang scheduling pod group member pod added mid-cycle is correctly buffered and retried after unschedulable attempt",
			podInjector: &podInjector{
				watchPod:    "p1",
				podToInject: p3,
			},
			steps: []stepsframework.Step{
				{
					Name:           "Create Gang Policy PodGroup",
					CreatePodGroup: podGroupGang,
				},
				{
					Name:       "Create member pods to trigger the cycle and inject p3 mid-cycle",
					CreatePods: []*v1.Pod{p1, p2},
				},
				{
					Name:                     "Verify p1, p2 and p3 are unschedulable",
					WaitForPodsUnschedulable: []string{"p1", "p2", "p3"},
				},
				{
					Name:        "Add node2 to expand capacity",
					CreateNodes: []*v1.Node{node2},
				},
				{
					Name:                 "Verify all pods scheduled successfully",
					WaitForPodsScheduled: []string{"p1", "p2", "p3"},
				},
			},
		},
		{
			name: "Gang scheduling pod group member pod added mid-cycle is correctly buffered and retried after schedulable attempt",
			podInjector: &podInjector{
				watchPod:    "p1",
				podToInject: p3,
			},
			steps: []stepsframework.Step{
				{
					Name:        "Add node2 to expand capacity",
					CreateNodes: []*v1.Node{node2},
				},
				{
					Name:           "Create Gang Policy PodGroup",
					CreatePodGroup: podGroupGang,
				},
				{
					Name:       "Create member pods to trigger the cycle and inject p3 mid-cycle",
					CreatePods: []*v1.Pod{p1, p2},
				},
				{
					Name:                 "Verify all pods scheduled successfully",
					WaitForPodsScheduled: []string{"p1", "p2", "p3"},
				},
			},
		},
		{
			name: "Basic policy pod group member pod added mid-cycle is correctly buffered and retried after unschedulable attempt",
			podInjector: &podInjector{
				watchPod:    "p2",
				podToInject: p3,
			},
			steps: []stepsframework.Step{
				{
					Name:           "Create Basic Policy PodGroup",
					CreatePodGroup: podGroupBasic,
				},
				{
					Name:       "Create member pod p1",
					CreatePods: []*v1.Pod{p1},
				},
				{
					Name:                 "Verify p1 is scheduled",
					WaitForPodsScheduled: []string{"p1"},
				},
				{
					Name:       "Create member pod p2 to trigger the cycle and inject p3 mid-cycle",
					CreatePods: []*v1.Pod{p2},
				},
				{
					Name:                     "Verify p2 and p3 are unschedulable",
					WaitForPodsUnschedulable: []string{"p2", "p3"},
				},
				{
					Name:        "Add node2 to expand capacity",
					CreateNodes: []*v1.Node{node2},
				},
				{
					Name:                 "Verify all pods scheduled successfully",
					WaitForPodsScheduled: []string{"p1", "p2", "p3"},
				},
			},
		},
		{
			name: "Basic policy pod group member pod added mid-cycle is correctly buffered and retried after schedulable attempt",
			podInjector: &podInjector{
				watchPod:    "p2",
				podToInject: p3,
			},
			steps: []stepsframework.Step{
				{
					Name:        "Add node2 to expand capacity",
					CreateNodes: []*v1.Node{node2},
				},
				{
					Name:           "Create Basic Policy PodGroup",
					CreatePodGroup: podGroupBasic,
				},
				{
					Name:       "Create member pods to trigger the cycle and inject p3 mid-cycle",
					CreatePods: []*v1.Pod{p1, p2},
				},
				{
					Name:                 "Verify all pods scheduled successfully",
					WaitForPodsScheduled: []string{"p1", "p2", "p3"},
				},
			},
		},
		{
			name: "Single gated pod blocks gang group from being scheduled until pod is ungated",
			steps: []stepsframework.Step{
				{
					Name:        "Add node2 to expand capacity",
					CreateNodes: []*v1.Node{node2},
				},
				{
					Name:           "Create Gang Policy PodGroup",
					CreatePodGroup: podGroupGang,
				},
				{
					Name:       "Create gated pod p1",
					CreatePods: []*v1.Pod{p1Gated},
				},
				{
					Name:                               "Verify p1 is gated on PreEnqueue",
					WaitForPodsInUnschedulableEntities: []string{"p1"},
				},
				{
					Name:       "Create schedulable pods p2 and p3",
					CreatePods: []*v1.Pod{p2, p3},
				},
				{
					Name:                               "Verify p2 and p3 are gated on PreEnqueue",
					WaitForPodsInUnschedulableEntities: []string{"p2", "p3"},
				},
				{
					Name: "Ungate p1 to trigger requeueing of the whole group",
					UpdatePod: &stepsframework.UpdatePod{
						PodName: "p1",
						ModifyFn: func(p *v1.Pod) {
							p.Spec.SchedulingGates = nil
						},
					},
				},
				{
					Name:                 "Verify all pods scheduled successfully",
					WaitForPodsScheduled: []string{"p1", "p2", "p3"},
				},
			},
		},
		{
			name: "Single gated pod blocks basic group from being scheduled until pod is ungated",
			steps: []stepsframework.Step{
				{
					Name:        "Add node2 to expand capacity",
					CreateNodes: []*v1.Node{node2},
				},
				{
					Name:           "Create Basic Policy PodGroup",
					CreatePodGroup: podGroupBasic,
				},
				{
					Name:       "Create gated pod p1",
					CreatePods: []*v1.Pod{p1Gated},
				},
				{
					Name:                               "Verify p1 is gated on PreEnqueue",
					WaitForPodsInUnschedulableEntities: []string{"p1"},
				},
				{
					Name:       "Create schedulable pods p2 and p3",
					CreatePods: []*v1.Pod{p2, p3},
				},
				{
					Name:                               "Verify p2 and p3 are gated on PreEnqueue",
					WaitForPodsInUnschedulableEntities: []string{"p2", "p3"},
				},
				{
					Name: "Ungate p1 to trigger requeueing of the whole group",
					UpdatePod: &stepsframework.UpdatePod{
						PodName: "p1",
						ModifyFn: func(p *v1.Pod) {
							p.Spec.SchedulingGates = nil
						},
					},
				},
				{
					Name:                 "Verify all pods scheduled successfully",
					WaitForPodsScheduled: []string{"p1", "p2", "p3"},
				},
			},
		},
		{
			name: "Gang pod group in active queue becomes gated when a new gated member pod is added",
			steps: []stepsframework.Step{
				{
					Name:           "Create Gang Policy PodGroup",
					CreatePodGroup: podGroupGang,
				},
				{
					Name:       "Create unschedulable pods p1 and p2",
					CreatePods: []*v1.Pod{p1, p2},
				},
				{
					Name:                     "Verify p1 and p2 are unschedulable",
					WaitForPodsUnschedulable: []string{"p1", "p2"},
				},
				{
					Name:       "Add new gated pod p3 to the pod group",
					CreatePods: []*v1.Pod{p3Gated},
				},
				{
					Name:                               "Verify p1, p2 and p3 are all gated on PreEnqueue",
					WaitForPodsInUnschedulableEntities: []string{"p1", "p2", "p3"},
				},
				{
					Name:        "Add node2 to expand capacity",
					CreateNodes: []*v1.Node{node2},
				},
				{
					Name: "Ungate p3 to trigger requeueing of the whole group",
					UpdatePod: &stepsframework.UpdatePod{
						PodName: "p3",
						ModifyFn: func(p *v1.Pod) {
							p.Spec.SchedulingGates = nil
						},
					},
				},
				{
					Name:                 "Verify all pods scheduled successfully",
					WaitForPodsScheduled: []string{"p1", "p2", "p3"},
				},
			},
		},
		{
			name: "Basic pod group in active queue becomes gated when a new gated member pod is added",
			steps: []stepsframework.Step{
				{
					Name:           "Create Basic Policy PodGroup",
					CreatePodGroup: podGroupBasic,
				},
				{
					Name:       "Create member pod p1",
					CreatePods: []*v1.Pod{p1},
				},
				{
					Name:                 "Verify p1 is scheduled",
					WaitForPodsScheduled: []string{"p1"},
				},
				{
					Name:       "Add new pod p2 to the scheduled pod group",
					CreatePods: []*v1.Pod{p2},
				},
				{
					Name:                     "Verify p2 is unschedulable",
					WaitForPodsUnschedulable: []string{"p2"},
				},
				{
					Name:       "Add new gated pod p3 to the pod group",
					CreatePods: []*v1.Pod{p3Gated},
				},
				{
					Name:                               "Verify p2 and p3 are gated on PreEnqueue",
					WaitForPodsInUnschedulableEntities: []string{"p2", "p3"},
				},
				{
					Name:        "Add node2 to expand capacity",
					CreateNodes: []*v1.Node{node2},
				},
				{
					Name: "Ungate p3 to trigger requeueing of the whole group",
					UpdatePod: &stepsframework.UpdatePod{
						PodName: "p3",
						ModifyFn: func(p *v1.Pod) {
							p.Spec.SchedulingGates = nil
						},
					},
				},
				{
					Name:                 "Verify all pods scheduled successfully",
					WaitForPodsScheduled: []string{"p1", "p2", "p3"},
				},
			},
		},
		{
			name: "Gang scheduling pod group member pod fails PreBind, is correctly retried and scheduled",
			bindingFailing: &bindingFailing{
				watchPod: "p1",
			},
			steps: []stepsframework.Step{
				{
					Name:        "Add node2 to expand capacity",
					CreateNodes: []*v1.Node{node2},
				},
				{
					Name:           "Create Gang Policy PodGroup",
					CreatePodGroup: podGroupGang,
				},
				{
					Name:       "Create member pods p1 and p2",
					CreatePods: []*v1.Pod{p1, p2},
				},
				{
					Name:                 "Verify all pods scheduled successfully on retry",
					WaitForPodsScheduled: []string{"p1", "p2"},
				},
			},
		},
		{
			name: "Basic policy pod group member pod fails PreBind, is correctly retried and scheduled",
			bindingFailing: &bindingFailing{
				watchPod: "p1",
			},
			steps: []stepsframework.Step{
				{
					Name:        "Add node2 to expand capacity",
					CreateNodes: []*v1.Node{node2},
				},
				{
					Name:           "Create Basic Policy PodGroup",
					CreatePodGroup: podGroupBasic,
				},
				{
					Name:       "Create member pods p1 and p2",
					CreatePods: []*v1.Pod{p1, p2},
				},
				{
					Name:                 "Verify all pods scheduled successfully on retry",
					WaitForPodsScheduled: []string{"p1", "p2"},
				},
			},
		},
		{
			name: "Single member gang pod group fails PreBind, is correctly retried and scheduled",
			bindingFailing: &bindingFailing{
				watchPod: "p1",
			},
			steps: []stepsframework.Step{
				{
					Name:           "Create Gang Policy PodGroup with MinCount 1",
					CreatePodGroup: podGroupGangSingle,
				},
				{
					Name:       "Create member pod p1",
					CreatePods: []*v1.Pod{p1},
				},
				{
					Name:                 "Verify p1 is scheduled successfully on retry",
					WaitForPodsScheduled: []string{"p1"},
				},
			},
		},
		{
			name: "Single member basic policy pod group fails PreBind, is correctly retried and scheduled",
			bindingFailing: &bindingFailing{
				watchPod: "p1",
			},
			steps: []stepsframework.Step{
				{
					Name:           "Create Basic Policy PodGroup",
					CreatePodGroup: podGroupBasic,
				},
				{
					Name:       "Create member pod p1",
					CreatePods: []*v1.Pod{p1},
				},
				{
					Name:                 "Verify p1 is scheduled successfully on retry",
					WaitForPodsScheduled: []string{"p1"},
				},
			},
		},
		{
			name: "Gang scheduling pod group member pod fails PreBind while other is unschedulable, is correctly retried and scheduled",
			bindingFailing: &bindingFailing{
				watchPod: "p1",
			},
			steps: []stepsframework.Step{
				{
					Name:       "Create member pods p1 and p2 (p1 succeeds and fails on PreBind, p2 fails due to capacity)",
					CreatePods: []*v1.Pod{p1, p2Large},
				},
				{
					Name:                                "Verify pods are gated at PreEnqueue (no PodGroup object)",
					WaitForPodsInIncompletePodGroupPods: []string{"p1", "p2"},
				},
				{
					Name:           "Create Gang Policy PodGroup with MinCount 1",
					CreatePodGroup: podGroupGangSingle,
				},
				{
					Name:                 "Verify p1 is scheduled successfully",
					WaitForPodsScheduled: []string{"p1"},
				},
				{
					Name:                     "Verify p2 is unschedulable",
					WaitForPodsUnschedulable: []string{"p2"},
				},
				{
					Name:        "Add node2 to expand capacity",
					CreateNodes: []*v1.Node{node2},
				},
				{
					Name:                 "Verify all pods scheduled successfully",
					WaitForPodsScheduled: []string{"p1", "p2"},
				},
			},
		},
		{
			name: "Basic policy pod group member pod fails PreBind while other is unschedulable, is correctly retried and scheduled",
			bindingFailing: &bindingFailing{
				watchPod: "p1",
			},
			steps: []stepsframework.Step{
				{
					Name:           "Create Basic Policy PodGroup",
					CreatePodGroup: podGroupBasic,
				},
				{
					Name:       "Create member pods p1 and p2 (p1 succeeds and fails on PreBind, p2 fails due to capacity)",
					CreatePods: []*v1.Pod{p1, p2Large},
				},
				{
					Name:                 "Verify p1 is scheduled successfully",
					WaitForPodsScheduled: []string{"p1"},
				},
				{
					Name:                     "Verify p2 is unschedulable",
					WaitForPodsUnschedulable: []string{"p2"},
				},
				{
					Name:        "Add node2 to expand capacity",
					CreateNodes: []*v1.Node{node2},
				},
				{
					Name:                 "Verify all pods scheduled successfully",
					WaitForPodsScheduled: []string{"p1", "p2"},
				},
			},
		},
		{
			name: "gang is scheduled, then pod group is deleted, and new enqueued pods belong to it end up in incompletePodGroupPods",
			steps: []stepsframework.Step{
				{
					Name:           "Create the PodGroup object",
					CreatePodGroup: podGroupGang,
				},
				{
					Name:       "Create gang pods",
					CreatePods: []*v1.Pod{smallP1, smallP2},
				},
				{
					Name:                 "Verify gang pods are scheduled successfully",
					WaitForPodsScheduled: []string{"p1", "p2"},
				},
				{
					Name:           "Delete the PodGroup",
					DeletePodGroup: "pg",
				},
				{
					Name:       "Create a new pod belonging to the deleted gang",
					CreatePods: []*v1.Pod{p3},
				},
				{
					Name:                                "Verify the new pod ends up in incompletePodGroupPods",
					WaitForPodsInIncompletePodGroupPods: []string{"p3"},
				},
			},
		},
		{
			name: "basic group is scheduled, then pod group is deleted, and new enqueued pods belong to it end up in incompletePodGroupPods",
			steps: []stepsframework.Step{
				{
					Name:           "Create the PodGroup object with Basic policy",
					CreatePodGroup: podGroupBasic,
				},
				{
					Name:       "Create basic group pods",
					CreatePods: []*v1.Pod{smallP1, smallP2},
				},
				{
					Name:                 "Verify group pods are scheduled successfully",
					WaitForPodsScheduled: []string{"p1", "p2"},
				},
				{
					Name:           "Delete the PodGroup",
					DeletePodGroup: "pg",
				},
				{
					Name:       "Create a new pod belonging to the deleted basic group",
					CreatePods: []*v1.Pod{p3},
				},
				{
					Name:                                "Verify the new pod ends up in incompletePodGroupPods",
					WaitForPodsInIncompletePodGroupPods: []string{"p3"},
				},
			},
		},
		{
			name: "gang is partially scheduled, then pod group is deleted, unschedulable pods end up in incompletePodGroupPods",
			steps: []stepsframework.Step{
				{
					Name:           "Create the PodGroup object",
					CreatePodGroup: podGroupGang,
				},
				{
					Name:       "Create gang pods",
					CreatePods: []*v1.Pod{smallP1, smallP2, p3},
				},
				{
					Name:                 "Verify gang pods are scheduled successfully",
					WaitForPodsScheduled: []string{"p1", "p2"},
				},
				{
					Name:                     "Verify p3 is unschedulable",
					WaitForPodsUnschedulable: []string{"p3"},
				},
				{
					Name:           "Delete the PodGroup",
					DeletePodGroup: "pg",
				},
				{
					Name:                                "Verify p3 ends up in incompletePodGroupPods",
					WaitForPodsInIncompletePodGroupPods: []string{"p3"},
				},
			},
		},
		{
			name: "basic group is partially scheduled, then pod group is deleted, unschedulable pods end up in incompletePodGroupPods",
			steps: []stepsframework.Step{
				{
					Name:           "Create the PodGroup object with Basic policy",
					CreatePodGroup: podGroupBasic,
				},
				{
					Name:       "Create basic group pods",
					CreatePods: []*v1.Pod{smallP1, smallP2, p3},
				},
				{
					Name:                 "Verify group pods are scheduled successfully",
					WaitForPodsScheduled: []string{"p1", "p2"},
				},
				{
					Name:                     "Verify p3 is unschedulable",
					WaitForPodsUnschedulable: []string{"p3"},
				},
				{
					Name:           "Delete the PodGroup",
					DeletePodGroup: "pg",
				},
				{
					Name:                                "Verify p3 ends up in incompletePodGroupPods",
					WaitForPodsInIncompletePodGroupPods: []string{"p3"},
				},
			},
		},
		{
			name: "gang is unschedulable, then pod group is deleted, unschedulable pods end up in incompletePodGroupPods",
			steps: []stepsframework.Step{
				{
					Name:           "Create the PodGroup object",
					CreatePodGroup: podGroupGangTriple,
				},
				{
					Name:       "Create gang pods",
					CreatePods: []*v1.Pod{smallP1, smallP2, p3},
				},
				{
					Name:                     "Verify gang pods are unschedulable",
					WaitForPodsUnschedulable: []string{"p1", "p2", "p3"},
				},
				{
					Name:           "Delete the PodGroup",
					DeletePodGroup: "pg",
				},
				{
					Name:                                "Verify gang pods end up in incompletePodGroupPods",
					WaitForPodsInIncompletePodGroupPods: []string{"p1", "p2", "p3"},
				},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			featuregatetesting.SetFeatureGatesDuringTest(t, utilfeature.DefaultFeatureGate, featuregatetesting.FeatureOverrides{
				features.GenericWorkload: true,
			})

			var testCtx *testutils.TestContext
			if tt.podInjector != nil || tt.bindingFailing != nil {
				var piPlugin *podInjectorPlugin
				var bfPlugin *bindingFailingPlugin
				registry := frameworkruntime.Registry{
					"PodInjectorPlugin": func(ctx context.Context, obj runtime.Object, handle fwk.Handle) (fwk.Plugin, error) {
						piPlugin = newPodInjectorPlugin(handle, tt.podInjector.watchPod, tt.podInjector.podToInject)
						return piPlugin, nil
					},
					"BindingFailingPlugin": func(ctx context.Context, obj runtime.Object, handle fwk.Handle) (fwk.Plugin, error) {
						bfPlugin = newBindingFailingPlugin(tt.bindingFailing.watchPod)
						return bfPlugin, nil
					},
				}
				var plugins configv1.Plugins
				if tt.podInjector != nil {
					plugins.PreFilter = configv1.PluginSet{
						Enabled: []configv1.Plugin{{Name: piPlugin.Name()}},
					}
				}
				if tt.bindingFailing != nil {
					plugins.PreBind = configv1.PluginSet{
						Enabled: []configv1.Plugin{{Name: bfPlugin.Name()}},
					}
				}
				cfg := configtesting.V1ToInternalWithDefaults(t, configv1.KubeSchedulerConfiguration{
					Profiles: []configv1.KubeSchedulerProfile{{
						SchedulerName: ptr.To(v1.DefaultSchedulerName),
						Plugins:       &plugins,
					}},
				})
				testCtx = testutils.InitTestSchedulerWithNS(t, "podgroup-queueing",
					scheduler.WithFrameworkOutOfTreeRegistry(registry),
					scheduler.WithProfiles(cfg.Profiles...),
					scheduler.WithPodMaxBackoffSeconds(0),
					scheduler.WithPodInitialBackoffSeconds(0))
				if piPlugin != nil {
					piPlugin.schedulingQueue = testCtx.Scheduler.SchedulingQueue
				}
			} else {
				testCtx = testutils.InitTestSchedulerWithNS(t, "podgroup-queueing",
					scheduler.WithPodMaxBackoffSeconds(0),
					scheduler.WithPodInitialBackoffSeconds(0))
			}
			ns := testCtx.NS.Name

			steps := append([]stepsframework.Step{
				{
					Name:        "Create initial node",
					CreateNodes: []*v1.Node{node},
				},
			}, tt.steps...)

			if err := stepsframework.RunSteps(testCtx, t, ns, steps); err != nil {
				t.Fatal(err)
			}
		})
	}
}

func TestPodGroupSequentialQueueing(t *testing.T) {
	node := st.MakeNode().Name("node").Capacity(map[v1.ResourceName]string{v1.ResourceCPU: "2"}).Obj()

	podGroupGang := st.MakePodGroup().Name("pg").MinCount(2).Obj()
	podGroupBasic := st.MakePodGroup().Name("pg").BasicPolicy().Obj()

	p1 := st.MakePod().Name("p1").Req(map[v1.ResourceName]string{v1.ResourceCPU: "2"}).Container("image").PodGroupName("pg").Obj()
	p2 := st.MakePod().Name("p2").Req(map[v1.ResourceName]string{v1.ResourceCPU: "2"}).Container("image").PodGroupName("pg").Obj()

	tests := []struct {
		name  string
		steps []stepsframework.Step
	}{
		{
			name: "Gang scheduling pod group fails scheduling and retries without any cluster change",
			steps: []stepsframework.Step{
				{
					Name:           "Create Gang Policy PodGroup",
					CreatePodGroup: podGroupGang,
				},
				{
					Name:       "Create unschedulable pods",
					CreatePods: []*v1.Pod{p1, p2},
				},
				{
					Name:                 "Verify p1 and p2 are in active queue",
					WaitForPodsInActiveQ: []string{"p1", "p2"},
				},
				{
					Name:           "Run scheduling attempt",
					RunScheduleOne: true,
				},
				{
					Name:                     "Verify p1 and p2 are unschedulable",
					WaitForPodsUnschedulable: []string{"p1", "p2"},
				},
				{
					Name:                 "Verify p1 and p2 are in active queue",
					WaitForPodsInActiveQ: []string{"p1", "p2"},
				},
			},
		},
		{
			name: "Basic policy pod group partially fails scheduling and retries without any cluster change",
			steps: []stepsframework.Step{
				{
					Name:           "Create Basic Policy PodGroup",
					CreatePodGroup: podGroupBasic,
				},
				{
					Name:       "Create member pods p1 and p2",
					CreatePods: []*v1.Pod{p1, p2},
				},
				{
					Name:                 "Verify p1 and p2 are in active queue",
					WaitForPodsInActiveQ: []string{"p1", "p2"},
				},
				{
					Name:           "Run scheduling attempt",
					RunScheduleOne: true,
				},
				{
					Name:                 "Verify p1 is scheduled",
					WaitForPodsScheduled: []string{"p1"},
				},
				{
					Name:                     "Verify p2 is unschedulable",
					WaitForPodsUnschedulable: []string{"p2"},
				},
				{
					Name:                 "Verify p2 is in active queue",
					WaitForPodsInActiveQ: []string{"p2"},
				},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			featuregatetesting.SetFeatureGatesDuringTest(t, utilfeature.DefaultFeatureGate, featuregatetesting.FeatureOverrides{
				features.GenericWorkload: true,
			})

			testCtx := testutils.InitTestSchedulerWithOptions(
				t,
				testutils.InitTestAPIServer(t, "podgroup-queueing", nil),
				0,
				scheduler.WithPodInitialBackoffSeconds(0),
				scheduler.WithPodMaxBackoffSeconds(0),
			)
			testutils.SyncSchedulerInformerFactory(testCtx)
			ns := testCtx.NS.Name

			steps := append([]stepsframework.Step{
				{
					Name:        "Create initial node",
					CreateNodes: []*v1.Node{node},
				},
			}, tt.steps...)

			if err := stepsframework.RunSteps(testCtx, t, ns, steps); err != nil {
				t.Fatal(err)
			}
		})
	}
}

func TestPodGroupRequeueRemainingOnSchedulingSuccess(t *testing.T) {
	type podInjector struct {
		watchPod    string
		podToInject *v1.Pod
	}

	node1 := st.MakeNode().Name("node").Capacity(map[v1.ResourceName]string{v1.ResourceCPU: "4"}).Obj()
	node2 := st.MakeNode().Name("node2").Capacity(map[v1.ResourceName]string{v1.ResourceCPU: "4"}).Obj()
	node3 := st.MakeNode().Name("node3").Capacity(map[v1.ResourceName]string{v1.ResourceCPU: "2"}).Obj()

	pg := st.MakePodGroup().Name("pg").BasicPolicy().Obj()
	pgP1 := st.MakePod().Name("pg-p1").Req(map[v1.ResourceName]string{v1.ResourceCPU: "2"}).Container("image").Priority(0).PodGroupName("pg").Obj()
	pgP2 := st.MakePod().Name("pg-p2").Req(map[v1.ResourceName]string{v1.ResourceCPU: "2"}).Container("image").Priority(0).PodGroupName("pg").Obj()
	pgP3 := st.MakePod().Name("pg-p3").Req(map[v1.ResourceName]string{v1.ResourceCPU: "2"}).Container("image").Priority(0).PodGroupName("pg").Obj()

	highPriorityPod := st.MakePod().Name("high-priority-pod").Req(map[v1.ResourceName]string{v1.ResourceCPU: "4"}).Container("image").Priority(1000).Obj()
	normalPriorityPod := st.MakePod().Name("normal-priority-pod").Req(map[v1.ResourceName]string{v1.ResourceCPU: "2"}).Container("image").Priority(0).Obj()

	tests := []struct {
		name        string
		podInjector *podInjector
		steps       []stepsframework.Step
	}{
		{
			name: "After successfully scheduling a podgroup and requeuing a remaining unscheduled pod, the next scheduling attempt prioritizes the higher priority pod",
			steps: []stepsframework.Step{
				{
					Name:           "Create PodGroup pg",
					CreatePodGroup: pg,
				},
				{
					Name:       "Create member pods pg-p1, pg-p2 and pg-p3",
					CreatePods: []*v1.Pod{pgP1, pgP2, pgP3},
				},
				{
					Name:                 "Verify member pods are in active queue",
					WaitForPodsInActiveQ: []string{"pg-p1", "pg-p2", "pg-p3"},
				},
				{
					Name:           "Run scheduling attempt",
					RunScheduleOne: true,
				},
				{
					Name:                 "Verify pg-p1 and pg-p2 are scheduled",
					WaitForPodsScheduled: []string{"pg-p1", "pg-p2"},
				},
				{
					Name:                     "Verify pg-p3 is unschedulable",
					WaitForPodsUnschedulable: []string{"pg-p3"},
				},
				{
					Name:                 "Verify pg-p3 is back in active queue",
					WaitForPodsInActiveQ: []string{"pg-p3"},
				},
				{
					Name:       "Create individual pod with higher priority",
					CreatePods: []*v1.Pod{highPriorityPod},
				},
				{
					Name:                 "Verify high priority pod is in active queue",
					WaitForPodsInActiveQ: []string{"high-priority-pod"},
				},
				{
					Name:        "Add node2 to expand capacity",
					CreateNodes: []*v1.Node{node2},
				},
				{
					Name:           "Run scheduling attempt",
					RunScheduleOne: true,
				},
				{
					Name:                 "Verify high priority pod is scheduled",
					WaitForPodsScheduled: []string{"high-priority-pod"},
				},
				{
					Name:                 "Verify pg-p3 is still in active queue",
					WaitForPodsInActiveQ: []string{"pg-p3"},
				},
			},
		},
		{
			name: "After successfully scheduling a podgroup and requeuing a remaining unscheduled pod, the next scheduling attempt prioritizes the old pod if priorities are equal",
			steps: []stepsframework.Step{
				{
					Name:           "Create PodGroup pg",
					CreatePodGroup: pg,
				},
				{
					Name:       "Create member pods pg-p1, pg-p2 and pg-p3",
					CreatePods: []*v1.Pod{pgP1, pgP2, pgP3},
				},
				{
					Name:                 "Verify member pods are in active queue",
					WaitForPodsInActiveQ: []string{"pg-p1", "pg-p2", "pg-p3"},
				},
				{
					Name:       "Create an individual pod with the same priority",
					CreatePods: []*v1.Pod{normalPriorityPod},
				},
				{
					Name:                 "Verify normal-priority-pod is in active queue",
					WaitForPodsInActiveQ: []string{"normal-priority-pod"},
				},
				{
					Name:           "Run scheduling attempt",
					RunScheduleOne: true,
				},
				{
					Name:                 "Verify pg-p1 and pg-p2 are scheduled",
					WaitForPodsScheduled: []string{"pg-p1", "pg-p2"},
				},
				{
					Name:                     "Verify pg-p3 is unschedulable",
					WaitForPodsUnschedulable: []string{"pg-p3"},
				},
				{
					Name:                 "Verify pg-p3 is back in active queue with preserved older timestamp",
					WaitForPodsInActiveQ: []string{"pg-p3"},
				},
				{
					Name:        "Add node3 to expand capacity",
					CreateNodes: []*v1.Node{node3},
				},
				{
					Name:           "Run scheduling attempt",
					RunScheduleOne: true,
				},
				{
					Name:                 "Verify pg-p3 is scheduled",
					WaitForPodsScheduled: []string{"pg-p3"},
				},
				{
					Name:           "Run scheduling attempt",
					RunScheduleOne: true,
				},
				{
					Name:                     "Verify normal-priority-pod remains unschedulable",
					WaitForPodsUnschedulable: []string{"normal-priority-pod"},
				},
			},
		},
		{
			name: "After successfully scheduling a podgroup with a new pending pod added mid-cycle, an older individual pod in queue is prioritized if priorities are equal",
			podInjector: &podInjector{
				watchPod:    "pg-p1",
				podToInject: pgP3,
			},
			steps: []stepsframework.Step{
				{
					Name:           "Create PodGroup pg",
					CreatePodGroup: pg,
				},
				{
					Name:       "Create member pods pg-p1 and pg-p2",
					CreatePods: []*v1.Pod{pgP1, pgP2},
				},
				{
					Name:                 "Verify member pods are in active queue",
					WaitForPodsInActiveQ: []string{"pg-p1", "pg-p2"},
				},
				{
					Name:       "Create an individual pod with the same priority",
					CreatePods: []*v1.Pod{normalPriorityPod},
				},
				{
					Name:                 "Verify normal-priority-pod is in active queue",
					WaitForPodsInActiveQ: []string{"normal-priority-pod"},
				},
				{
					Name:           "Run scheduling attempt for podgroup (injects pg-p3 mid-cycle)",
					RunScheduleOne: true,
				},
				{
					Name:                 "Verify pg-p1 and pg-p2 are scheduled",
					WaitForPodsScheduled: []string{"pg-p1", "pg-p2"},
				},
				{
					Name:                 "Verify pg-p3 is in active queue with new timestamp",
					WaitForPodsInActiveQ: []string{"pg-p3"},
				},
				{
					Name:        "Add node3 to expand capacity",
					CreateNodes: []*v1.Node{node3},
				},
				{
					Name:           "Run scheduling attempt",
					RunScheduleOne: true,
				},
				{
					Name:                 "Verify normal-priority-pod is scheduled before pg-p3 due to older timestamp",
					WaitForPodsScheduled: []string{"normal-priority-pod"},
				},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.GenericWorkload, true)
			var piPlugin *podInjectorPlugin
			var opts []scheduler.Option
			if tt.podInjector != nil {
				registry := frameworkruntime.Registry{
					"PodInjectorPlugin": func(ctx context.Context, obj runtime.Object, handle fwk.Handle) (fwk.Plugin, error) {
						piPlugin = newPodInjectorPlugin(handle, tt.podInjector.watchPod, tt.podInjector.podToInject)
						return piPlugin, nil
					},
				}
				plugins := configv1.Plugins{
					PreFilter: configv1.PluginSet{
						Enabled: []configv1.Plugin{{Name: "PodInjectorPlugin"}},
					},
				}
				cfg := configtesting.V1ToInternalWithDefaults(t, configv1.KubeSchedulerConfiguration{
					Profiles: []configv1.KubeSchedulerProfile{{
						SchedulerName: ptr.To(v1.DefaultSchedulerName),
						Plugins:       &plugins,
					}},
				})
				opts = append(opts,
					scheduler.WithFrameworkOutOfTreeRegistry(registry),
					scheduler.WithProfiles(cfg.Profiles...),
				)
			}
			opts = append(opts,
				scheduler.WithPodInitialBackoffSeconds(10),
				scheduler.WithPodMaxBackoffSeconds(20),
			)
			testCtx := testutils.InitTestSchedulerWithOptions(
				t,
				testutils.InitTestAPIServer(t, "podgroup-ordering", nil),
				0,
				opts...,
			)
			if piPlugin != nil {
				piPlugin.schedulingQueue = testCtx.Scheduler.SchedulingQueue
			}
			testutils.SyncSchedulerInformerFactory(testCtx)
			ns := testCtx.NS.Name

			steps := append([]stepsframework.Step{
				{
					Name:        "Create initial node",
					CreateNodes: []*v1.Node{node1},
				},
			}, tt.steps...)

			if err := stepsframework.RunSteps(testCtx, t, ns, steps); err != nil {
				t.Fatal(err)
			}
		})
	}
}
