/*
Copyright 2025 The Kubernetes Authors.

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

package gangscheduling

import (
	"testing"

	"github.com/google/go-cmp/cmp"

	v1 "k8s.io/api/core/v1"
	schedulingapi "k8s.io/api/scheduling/v1alpha1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/informers"
	"k8s.io/client-go/kubernetes/fake"
	"k8s.io/klog/v2"
	"k8s.io/klog/v2/ktesting"
	fwk "k8s.io/kube-scheduler/framework"
	"k8s.io/kubernetes/pkg/scheduler/backend/workloadmanager"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/feature"
	frameworkruntime "k8s.io/kubernetes/pkg/scheduler/framework/runtime"
	st "k8s.io/kubernetes/pkg/scheduler/testing"
)

func Test_isSchedulableAfterPodAdded(t *testing.T) {
	tests := []struct {
		name         string
		pod          *v1.Pod
		newPod       *v1.Pod
		expectedHint fwk.QueueingHint
	}{
		{
			name:         "add a newPod which matches the pod's workload and pod group",
			pod:          st.MakePod().Name("p").WorkloadRef(&v1.WorkloadReference{Name: "w1", PodGroup: "pg"}).Obj(),
			newPod:       st.MakePod().WorkloadRef(&v1.WorkloadReference{Name: "w1", PodGroup: "pg"}).Obj(),
			expectedHint: fwk.Queue,
		},
		{
			name:         "add a newPod which matches the pod's workload, pod group and replica key",
			pod:          st.MakePod().Name("p").WorkloadRef(&v1.WorkloadReference{Name: "w1", PodGroup: "pg", PodGroupReplicaKey: "3"}).Obj(),
			newPod:       st.MakePod().WorkloadRef(&v1.WorkloadReference{Name: "w1", PodGroup: "pg", PodGroupReplicaKey: "3"}).Obj(),
			expectedHint: fwk.Queue,
		},
		{
			name:         "add a newPod which doesn't match the pod's namespace",
			pod:          st.MakePod().Name("p").WorkloadRef(&v1.WorkloadReference{Name: "w1", PodGroup: "pg"}).Obj(),
			newPod:       st.MakePod().Namespace("foo").WorkloadRef(&v1.WorkloadReference{Name: "w1", PodGroup: "pg"}).Obj(),
			expectedHint: fwk.QueueSkip,
		},
		{
			name:         "add a newPod which doesn't match the pod's workload name",
			pod:          st.MakePod().Name("p").WorkloadRef(&v1.WorkloadReference{Name: "w1", PodGroup: "pg"}).Obj(),
			newPod:       st.MakePod().WorkloadRef(&v1.WorkloadReference{Name: "w2", PodGroup: "pg"}).Obj(),
			expectedHint: fwk.QueueSkip,
		},
		{
			name:         "add a newPod which doesn't match the pod's pod group name",
			pod:          st.MakePod().Name("p").WorkloadRef(&v1.WorkloadReference{Name: "w1", PodGroup: "pg"}).Obj(),
			newPod:       st.MakePod().WorkloadRef(&v1.WorkloadReference{Name: "w1", PodGroup: "pg2"}).Obj(),
			expectedHint: fwk.QueueSkip,
		},
		{
			name:         "add a newPod which doesn't match the pod's replica key",
			pod:          st.MakePod().Name("p").WorkloadRef(&v1.WorkloadReference{Name: "w1", PodGroup: "pg", PodGroupReplicaKey: "3"}).Obj(),
			newPod:       st.MakePod().WorkloadRef(&v1.WorkloadReference{Name: "w1", PodGroup: "pg", PodGroupReplicaKey: "4"}).Obj(),
			expectedHint: fwk.QueueSkip,
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			logger, ctx := ktesting.NewTestContext(t)

			informerFactory := informers.NewSharedInformerFactory(fake.NewClientset(), 0)
			fh, err := frameworkruntime.NewFramework(ctx, nil, nil,
				frameworkruntime.WithInformerFactory(informerFactory),
			)
			if err != nil {
				t.Fatalf("Failed to create framework: %v", err)
			}

			p, err := New(ctx, nil, fh, feature.Features{})
			if err != nil {
				t.Fatal(err)
			}
			actualHint, err := p.(*GangScheduling).isSchedulableAfterPodAdded(logger, tc.pod, nil, tc.newPod)
			if err != nil {
				t.Errorf("Unexpected error: %v", err)
			}
			if diff := cmp.Diff(tc.expectedHint, actualHint); diff != "" {
				t.Errorf("Expected QueuingHint doesn't match (-want,+got):\n%s", diff)
			}
		})
	}
}

func Test_isSchedulableAfterWorkloadAdded(t *testing.T) {
	tests := []struct {
		name         string
		pod          *v1.Pod
		newWorkload  *schedulingapi.Workload
		expectedHint fwk.QueueingHint
	}{
		{
			name:         "add a workload which matches the pod's workload name",
			pod:          st.MakePod().Name("p").WorkloadRef(&v1.WorkloadReference{Name: "w1", PodGroup: "pg"}).Obj(),
			newWorkload:  st.MakeWorkload().Name("w1").PodGroup(st.MakePodGroup().Name("pg").MinCount(1).Obj()).Obj(),
			expectedHint: fwk.Queue,
		},
		{
			name:         "add a workload which doesn't match the pod's workload name",
			pod:          st.MakePod().Name("p").WorkloadRef(&v1.WorkloadReference{Name: "w2", PodGroup: "pg"}).Obj(),
			newWorkload:  st.MakeWorkload().Name("w1").PodGroup(st.MakePodGroup().Name("pg").MinCount(1).Obj()).Obj(),
			expectedHint: fwk.QueueSkip,
		},
		{
			name:         "add a workload which doesn't match the pod's workload namespace",
			pod:          st.MakePod().Namespace("ns1").Name("p").WorkloadRef(&v1.WorkloadReference{Name: "w1", PodGroup: "pg"}).Obj(),
			newWorkload:  st.MakeWorkload().Namespace("ns2").Name("w1").PodGroup(st.MakePodGroup().Name("pg").MinCount(1).Obj()).Obj(),
			expectedHint: fwk.QueueSkip,
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			logger, ctx := ktesting.NewTestContext(t)

			informerFactory := informers.NewSharedInformerFactory(fake.NewClientset(), 0)
			fh, err := frameworkruntime.NewFramework(ctx, nil, nil,
				frameworkruntime.WithInformerFactory(informerFactory),
			)
			if err != nil {
				t.Fatalf("Failed to create framework: %v", err)
			}

			p, err := New(ctx, nil, fh, feature.Features{})
			if err != nil {
				t.Fatal(err)
			}
			actualHint, err := p.(*GangScheduling).isSchedulableAfterWorkloadAdded(logger, tc.pod, nil, tc.newWorkload)
			if err != nil {
				t.Errorf("Unexpected error: %v", err)
			}
			if diff := cmp.Diff(tc.expectedHint, actualHint); diff != "" {
				t.Errorf("Expected QueuingHint doesn't match (-want,+got):\n%s", diff)
			}
		})
	}
}

type podActivatorMock struct {
	activatedPods []*v1.Pod
}

func (pam *podActivatorMock) Activate(_ klog.Logger, pods map[string]*v1.Pod) {
	for _, pod := range pods {
		pam.activatedPods = append(pam.activatedPods, pod)
	}
}

func TestGangSchedulingFlow(t *testing.T) {
	workload := st.MakeWorkload().Namespace("ns1").Name("gang-wl").
		PodGroup(st.MakePodGroup().Name("pg1").MinCount(3).Obj()).
		PodGroup(st.MakePodGroup().Name("pg2").MinCount(4).Obj()).Obj()

	basicPolicyWorkload := st.MakeWorkload().Namespace("ns1").Name("basic-wl").
		PodGroup(st.MakePodGroup().Name("pg1").BasicPolicy().Obj()).Obj()

	p1 := st.MakePod().Namespace("ns1").Name("p1").UID("p1").
		WorkloadRef(&v1.WorkloadReference{Name: "gang-wl", PodGroup: "pg1"}).Obj()
	p2 := st.MakePod().Namespace("ns1").Name("p2").UID("p2").
		WorkloadRef(&v1.WorkloadReference{Name: "gang-wl", PodGroup: "pg1"}).Obj()
	p3 := st.MakePod().Namespace("ns1").Name("p3").UID("p3").
		WorkloadRef(&v1.WorkloadReference{Name: "gang-wl", PodGroup: "pg1"}).Obj()

	p4 := st.MakePod().Namespace("ns1").Name("p4").UID("p4").
		WorkloadRef(&v1.WorkloadReference{Name: "gang-wl", PodGroup: "pg2"}).Obj()

	p5 := st.MakePod().Namespace("ns1").Name("p5").UID("p5").
		WorkloadRef(&v1.WorkloadReference{Name: "gang-wl", PodGroup: "pg1", PodGroupReplicaKey: "2"}).Obj()

	basicPolicyPod := st.MakePod().Namespace("ns1").Name("basic-pod").UID("basic-pod").
		WorkloadRef(&v1.WorkloadReference{Name: "basic-wl", PodGroup: "pg1"}).Obj()

	nonGangPod := st.MakePod().Namespace("ns1").Name("non-gang").UID("non-gang").Obj()

	tests := []struct {
		name                 string
		pod                  *v1.Pod
		initialPods          []*v1.Pod
		initialWorkloads     []*schedulingapi.Workload
		podsWaitingOnPermit  []*v1.Pod
		wantPreEnqueueStatus *fwk.Status
		wantPermitStatus     *fwk.Status
		wantActivatedPods    []*v1.Pod
		wantAllowedPods      []types.UID
	}{
		{
			name:                 "non-gang pod succeeds immediately",
			pod:                  nonGangPod,
			initialWorkloads:     []*schedulingapi.Workload{workload, basicPolicyWorkload},
			wantPreEnqueueStatus: nil,
			wantPermitStatus:     nil,
		},
		{
			name:                 "basic policy pod succeeds immediately",
			pod:                  basicPolicyPod,
			initialWorkloads:     []*schedulingapi.Workload{workload, basicPolicyWorkload},
			wantPreEnqueueStatus: nil,
			wantPermitStatus:     nil,
		},
		{
			name:                 "gang pod fails PreEnqueue when workload is not yet created",
			pod:                  p1,
			initialPods:          []*v1.Pod{p2, p3, p4, p5},
			initialWorkloads:     []*schedulingapi.Workload{},
			wantPreEnqueueStatus: fwk.NewStatus(fwk.UnschedulableAndUnresolvable, "waiting for pods's workload \"gang-wl\" to appear in scheduling queue"),
		},
		{
			name:                 "gang pod fails PreEnqueue when quorum is not met",
			pod:                  p1,
			initialPods:          []*v1.Pod{p2, p4, p5}, // Only p1 and p2 exist from their gang, minCount is 3.
			initialWorkloads:     []*schedulingapi.Workload{workload},
			wantPreEnqueueStatus: fwk.NewStatus(fwk.UnschedulableAndUnresolvable, "waiting for minCount pods from a gang to appear in scheduling queue"),
		},
		{
			name:                 "gang pod passes PreEnqueue, but waits at Permit",
			pod:                  p1,
			initialPods:          []*v1.Pod{p2, p3, p4, p5}, // All pods are available.
			initialWorkloads:     []*schedulingapi.Workload{workload},
			podsWaitingOnPermit:  []*v1.Pod{p2, p4, p5},
			wantPreEnqueueStatus: nil,
			wantActivatedPods:    []*v1.Pod{p3},
			// At Permit, p1 will be assumed, but the count (2) is less than the quorum (3), so it must wait.
			wantPermitStatus: fwk.NewStatus(fwk.Wait, "waiting for minCount pods from a gang to be scheduled"),
		},
		{
			name:                 "final gang pod arrives at Permit and allows all waiting pods from a gang",
			pod:                  p1, // p3 is the pod being scheduled in this cycle.
			initialPods:          []*v1.Pod{p2, p3, p4, p5},
			initialWorkloads:     []*schedulingapi.Workload{workload},
			podsWaitingOnPermit:  []*v1.Pod{p2, p3, p4, p5},
			wantPreEnqueueStatus: nil,
			wantPermitStatus:     nil,
			wantAllowedPods:      []types.UID{"p1", "p2", "p3"},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			logger, ctx := ktesting.NewTestContext(t)
			manager := workloadmanager.New(logger)

			informerFactory := informers.NewSharedInformerFactory(fake.NewClientset(), 0)
			workloadInformer := informerFactory.Scheduling().V1alpha1().Workloads()

			fakeActivator := &podActivatorMock{}

			fh, err := frameworkruntime.NewFramework(ctx, nil, nil,
				frameworkruntime.WithInformerFactory(informerFactory),
				frameworkruntime.WithWorkloadManager(manager),
				frameworkruntime.WithWaitingPods(frameworkruntime.NewWaitingPodsMap()),
				frameworkruntime.WithPodActivator(fakeActivator),
			)
			if err != nil {
				t.Fatalf("Failed to create framework: %v", err)
			}

			// Populate informers and manager state for the test case.
			for _, wl := range tt.initialWorkloads {
				err := workloadInformer.Informer().GetStore().Add(wl)
				if err != nil {
					t.Fatalf("Failed to add workload %s to store: %v", wl.Name, err)
				}
			}
			for _, p := range tt.initialPods {
				manager.AddPod(p)
			}
			manager.AddPod(tt.pod)

			p, err := New(ctx, nil, fh, feature.Features{EnableGangScheduling: true})
			if err != nil {
				t.Fatalf("Failed to create plugin: %v", err)
			}
			pl := p.(*GangScheduling)

			gotPreEnqueueStatus := pl.PreEnqueue(ctx, tt.pod)
			if diff := cmp.Diff(tt.wantPreEnqueueStatus, gotPreEnqueueStatus); diff != "" {
				t.Fatalf("Unexpected PreEnqueue status (-want,+got):\n%s", diff)
			}
			if !gotPreEnqueueStatus.IsSuccess() {
				// Pod is rejected.
				return
			}

			// Simulate that other pods have already hit Permit and are now waiting.
			for _, p := range tt.podsWaitingOnPermit {
				// Run Reserve and Permit for these pods to get them into the "assumed" state inside the manager.
				status := pl.Reserve(ctx, nil, p, "some-node")
				if !status.IsSuccess() {
					t.Fatalf("Unexpected Reserve status for pod %q: %v", p.Name, status)
				}
				status, _ = pl.Permit(ctx, nil, p, "some-node")
				if status.Code() != fwk.Wait {
					t.Fatalf("Expected Wait status while permitting a pod %q: %v", p.Name, status)
				}
			}

			status := pl.Reserve(ctx, nil, tt.pod, "some-node")
			if !status.IsSuccess() {
				t.Fatalf("Unexpected Reserve status: %v", status)
			}

			// Clear activated pods to assert those activated in tt.pod Permit.
			fakeActivator.activatedPods = nil

			gotPermitStatus, _ := pl.Permit(ctx, nil, tt.pod, "some-node")
			if diff := cmp.Diff(tt.wantPermitStatus, gotPermitStatus); diff != "" {
				t.Fatalf("Unexpected Permit status (-want, +got):\n%s", diff)
			}
			if gotPermitStatus.Code() == fwk.Wait {
				// Pod waits for others from a gang. Simulate its eventual Unreserve.
				pl.Unreserve(ctx, nil, tt.pod, "some-node")
				return
			}

			if diff := cmp.Diff(tt.wantActivatedPods, fakeActivator.activatedPods); diff != "" {
				t.Errorf("Unexpected activated pods (-want, +got):\n%s", diff)
			}
			for _, p := range tt.wantAllowedPods {
				if wp := fh.GetWaitingPod(p); wp != nil {
					t.Errorf("Expected pod %q to be allowed", p)
				}
			}
		})
	}
}
