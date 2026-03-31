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
	schedulingapi "k8s.io/api/scheduling/v1alpha2"
	"k8s.io/apimachinery/pkg/types"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/client-go/informers"
	"k8s.io/client-go/kubernetes/fake"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/klog/v2"
	"k8s.io/klog/v2/ktesting"
	fwk "k8s.io/kube-scheduler/framework"
	"k8s.io/kubernetes/pkg/features"
	internalcache "k8s.io/kubernetes/pkg/scheduler/backend/cache"
	schedulerframework "k8s.io/kubernetes/pkg/scheduler/framework"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/feature"
	frameworkruntime "k8s.io/kubernetes/pkg/scheduler/framework/runtime"
	"k8s.io/kubernetes/pkg/scheduler/metrics"
	st "k8s.io/kubernetes/pkg/scheduler/testing"
)

func init() {
	// This is required for tests where cache is initialized, and cache attempts to update metrics.
	metrics.Register()
}

func Test_isSchedulableAfterPodAdded(t *testing.T) {
	tests := []struct {
		name         string
		pod          *v1.Pod
		newPod       *v1.Pod
		expectedHint fwk.QueueingHint
	}{
		{
			name:         "add a newPod which matches the pod's scheduling group",
			pod:          st.MakePod().Name("p").PodGroupName("pg").Obj(),
			newPod:       st.MakePod().PodGroupName("pg").Obj(),
			expectedHint: fwk.Queue,
		},
		{
			name:         "add a newPod which doesn't match the pod's namespace",
			pod:          st.MakePod().Name("p").PodGroupName("pg").Obj(),
			newPod:       st.MakePod().Namespace("foo").PodGroupName("pg").Obj(),
			expectedHint: fwk.QueueSkip,
		},
		{
			name:         "add a newPod which doesn't match the pod's pod group name",
			pod:          st.MakePod().Name("p").PodGroupName("pg1").Obj(),
			newPod:       st.MakePod().PodGroupName("pg2").Obj(),
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

func Test_isSchedulableAfterPodGroupAdded(t *testing.T) {
	tests := []struct {
		name         string
		pod          *v1.Pod
		newPodGroup  *schedulingapi.PodGroup
		expectedHint fwk.QueueingHint
	}{
		{
			name:         "add a pod group which matches the pod's pod group name",
			pod:          st.MakePod().Name("p").PodGroupName("pg").Obj(),
			newPodGroup:  st.MakePodGroup().Name("pg").MinCount(1).TemplateRef("t", "w").Obj(),
			expectedHint: fwk.Queue,
		},
		{
			name:         "add a pod group which doesn't match the pod's scheduling group name",
			pod:          st.MakePod().Name("p").PodGroupName("pg1").Obj(),
			newPodGroup:  st.MakePodGroup().Name("pg2").MinCount(1).TemplateRef("t", "w").Obj(),
			expectedHint: fwk.QueueSkip,
		},
		{
			name:         "add a pod group which doesn't match the pod's scheduling group namespace",
			pod:          st.MakePod().Namespace("ns1").Name("p").PodGroupName("pg").Obj(),
			newPodGroup:  st.MakePodGroup().Namespace("ns2").Name("pg").MinCount(1).TemplateRef("t", "w").Obj(),
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
			actualHint, err := p.(*GangScheduling).isSchedulableAfterPodGroupAdded(logger, tc.pod, nil, tc.newPodGroup)
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
	gangPodGroup1 := st.MakePodGroup().Namespace("ns1").Name("pg1").TemplateRef("t1", "gang-wl").MinCount(3).Obj()
	gangPodGroup2 := st.MakePodGroup().Namespace("ns1").Name("pg2").TemplateRef("t2", "gang-wl").MinCount(4).Obj()
	basicPodGroup := st.MakePodGroup().Namespace("ns1").Name("pg3").TemplateRef("1", "basic-wl").BasicPolicy().Obj()

	p1 := st.MakePod().Namespace("ns1").Name("p1").UID("p1").PodGroupName("pg1").Obj()
	p2 := st.MakePod().Namespace("ns1").Name("p2").UID("p2").PodGroupName("pg1").Obj()
	p3 := st.MakePod().Namespace("ns1").Name("p3").UID("p3").PodGroupName("pg1").Obj()

	p4 := st.MakePod().Namespace("ns1").Name("p4").UID("p4").PodGroupName("pg2").Obj()
	p5 := st.MakePod().Namespace("ns1").Name("p5").UID("p5").PodGroupName("pg2").Obj()

	basicPolicyPod := st.MakePod().Namespace("ns1").Name("basic-pod").UID("basic-pod").PodGroupName("pg3").Obj()

	nonGangPod := st.MakePod().Namespace("ns1").Name("non-gang").UID("non-gang").Obj()

	tests := []struct {
		name                            string
		pod                             *v1.Pod
		initialPods                     []*v1.Pod
		initialPodGroups                []*schedulingapi.PodGroup
		podsWaitingOnPermit             []*v1.Pod
		isDuringPodGroupSchedulingCycle bool
		wantPreEnqueueStatus            *fwk.Status
		wantPermitStatus                *fwk.Status
		wantActivatedPods               []*v1.Pod
		wantAllowedPods                 []types.UID
	}{
		{
			name:                 "non-gang pod succeeds immediately",
			pod:                  nonGangPod,
			initialPodGroups:     []*schedulingapi.PodGroup{gangPodGroup1, gangPodGroup2, basicPodGroup},
			wantPreEnqueueStatus: nil,
			wantPermitStatus:     nil,
		},
		{
			name:                 "basic policy pod succeeds immediately",
			pod:                  basicPolicyPod,
			initialPodGroups:     []*schedulingapi.PodGroup{gangPodGroup1, gangPodGroup2, basicPodGroup},
			wantPreEnqueueStatus: nil,
			wantPermitStatus:     nil,
		},
		{
			name:                 "gang pod fails PreEnqueue when pod group is not yet created",
			pod:                  p1,
			initialPods:          []*v1.Pod{p2, p3, p4, p5},
			initialPodGroups:     []*schedulingapi.PodGroup{},
			wantPreEnqueueStatus: fwk.NewStatus(fwk.UnschedulableAndUnresolvable, "waiting for pods's pod group \"pg1\" to appear in scheduling queue"),
		},
		{
			name:                 "gang pod fails PreEnqueue when quorum is not met",
			pod:                  p1,
			initialPods:          []*v1.Pod{p2, p4, p5}, // Only p1 and p2 exist from their gang, minCount is 3.
			initialPodGroups:     []*schedulingapi.PodGroup{gangPodGroup1, gangPodGroup2},
			wantPreEnqueueStatus: fwk.NewStatus(fwk.UnschedulableAndUnresolvable, "waiting for minCount pods from a gang to appear in scheduling queue"),
		},
		{
			name:                 "gang pod passes PreEnqueue, but waits at Permit",
			pod:                  p1,
			initialPods:          []*v1.Pod{p2, p3, p4, p5}, // All pods are available.
			initialPodGroups:     []*schedulingapi.PodGroup{gangPodGroup1, gangPodGroup2},
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
			initialPodGroups:     []*schedulingapi.PodGroup{gangPodGroup1, gangPodGroup2},
			podsWaitingOnPermit:  []*v1.Pod{p2, p3, p4, p5},
			wantPreEnqueueStatus: nil,
			wantPermitStatus:     nil,
			wantAllowedPods:      []types.UID{"p1", "p2", "p3"},
		},
		{
			name:                            "final gang pod arrives at Permit during pod group scheduling cycle",
			pod:                             p1,
			initialPods:                     []*v1.Pod{p2, p3, p4, p5},
			initialPodGroups:                []*schedulingapi.PodGroup{gangPodGroup1, gangPodGroup2},
			podsWaitingOnPermit:             []*v1.Pod{p2, p3, p4, p5},
			isDuringPodGroupSchedulingCycle: true,
			wantPreEnqueueStatus:            nil,
			wantPermitStatus:                nil,
			wantAllowedPods:                 []types.UID{"p1", "p2", "p3"},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.GenericWorkload, true)
			logger, ctx := ktesting.NewTestContext(t)
			cache := internalcache.New(ctx, nil, true)

			informerFactory := informers.NewSharedInformerFactory(fake.NewClientset(), 0)
			podGroupInformer := informerFactory.Scheduling().V1alpha2().PodGroups()
			fakeActivator := &podActivatorMock{}
			snapshot := internalcache.NewEmptySnapshot()
			fh, err := frameworkruntime.NewFramework(ctx, nil, nil,
				frameworkruntime.WithInformerFactory(informerFactory),
				frameworkruntime.WithPodGroupManager(cache),
				frameworkruntime.WithWaitingPods(frameworkruntime.NewWaitingPodsMap()),
				frameworkruntime.WithPodActivator(fakeActivator),
				frameworkruntime.WithSnapshotSharedLister(snapshot),
			)
			if err != nil {
				t.Fatalf("Failed to create framework: %v", err)
			}

			// Populate informers and manager state for the test case.
			for _, wl := range tt.initialPodGroups {
				err := podGroupInformer.Informer().GetStore().Add(wl)
				if err != nil {
					t.Fatalf("Failed to add podGroup %s to store: %v", wl.Name, err)
				}
			}

			for _, p := range tt.initialPods {
				cache.AddPodGroupMember(p)
			}
			cache.AddPodGroupMember(tt.pod)

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
				pod := p.DeepCopy()
				pod.Spec.NodeName = "some-node"
				if err := cache.AssumePod(logger, pod); err != nil {
					t.Fatalf("Failed to assume pod %q: %v", pod.Name, err)
				}
				status, _ := pl.Permit(ctx, schedulerframework.NewCycleState(), pod, "some-node")
				if status.Code() != fwk.Wait {
					t.Fatalf("Expected Wait status while permitting a pod %q: %v", pod.Name, status)
				}
			}

			// Clear activated pods to assert those activated in tt.pod Permit.
			fakeActivator.activatedPods = nil

			cycleState := schedulerframework.NewCycleState()
			if tt.isDuringPodGroupSchedulingCycle {
				cycleState.SetPodGroupSchedulingCycle(cycleState)
			}

			pod := tt.pod.DeepCopy()
			pod.Spec.NodeName = "some-node"

			// In a pod group scheduling cycle, a snapshot is taken after all
			// waiting pods are assumed, so that Permit can read from it.
			if tt.isDuringPodGroupSchedulingCycle {
				if err := cache.UpdateSnapshot(logger, snapshot); err != nil {
					t.Fatalf("Failed to update snapshot: %v", err)
				}
				podInfo, err := schedulerframework.NewPodInfo(pod)
				if err != nil {
					t.Fatalf("Failed to create pod info for %q: %v", pod.Name, err)
				}
				// Assume pod in the snapshot, as in a pod group scheduling cycle.
				if err := snapshot.AssumePod(podInfo); err != nil {
					t.Fatalf("Failed to assume pod %q in snapshot: %v", pod.Name, err)
				}
			} else {
				// Assume pod in the cache, as in a pod-by-pod scheduling cycle, where Permit reads from cache.
				if err := cache.AssumePod(logger, pod); err != nil {
					t.Fatalf("Failed to assume pod %q in cache: %v", pod.Name, err)
				}
			}

			gotPermitStatus, _ := pl.Permit(ctx, cycleState, pod, "some-node")
			if diff := cmp.Diff(tt.wantPermitStatus, gotPermitStatus); diff != "" {
				t.Fatalf("Unexpected Permit status (-want, +got):\n%s", diff)
			}
			if gotPermitStatus.Code() == fwk.Wait {
				// Pod waits for others from a gang. Simulate its eventual forget.
				if tt.isDuringPodGroupSchedulingCycle {
					if err := snapshot.ForgetPod(logger, pod); err != nil {
						t.Fatalf("Failed to forget pod %q from snapshot: %v", pod.Name, err)
					}
				} else {
					if err := cache.ForgetPod(logger, pod); err != nil {
						t.Fatalf("Failed to forget pod %q from cache: %v", pod.Name, err)
					}
				}
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
