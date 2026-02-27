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

package coscheduling

import (
	"context"
	"testing"

	"github.com/google/go-cmp/cmp"
	v1 "k8s.io/api/core/v1"
	schedulingapi "k8s.io/api/scheduling/v1alpha2"
	"k8s.io/client-go/informers"
	"k8s.io/client-go/kubernetes/fake"
	"k8s.io/klog/v2"
	"k8s.io/klog/v2/ktesting"
	fwk "k8s.io/kube-scheduler/framework"
	"k8s.io/kubernetes/pkg/scheduler/backend/podgroupmanager"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/feature"
	frameworkruntime "k8s.io/kubernetes/pkg/scheduler/framework/runtime"
	st "k8s.io/kubernetes/pkg/scheduler/testing"
)

type podActivatorMock struct {
	activatedPods []*v1.Pod
}

func (pam *podActivatorMock) Activate(_ klog.Logger, pods map[string]*v1.Pod) {
	for _, pod := range pods {
		pam.activatedPods = append(pam.activatedPods, pod)
	}
}

func TestCoschedulingFlow(t *testing.T) {
	gangPodGroup1 := st.MakePodGroup().Namespace("ns1").Name("pg1").TemplateRef("t1", "gang-wl").MinCount(3).Obj()
	gangPodGroup2 := st.MakePodGroup().Namespace("ns1").Name("pg2").TemplateRef("t2", "gang-wl").MinCount(4).Obj()

	basicPodGroupDesired := st.MakePodGroup().Namespace("ns1").Name("pg3").TemplateRef("1", "basic-wl-desired").BasicPolicy().DesiredCount(4).Obj()
	basicPodGroupNoDesired := st.MakePodGroup().Namespace("ns1").Name("pg4").TemplateRef("2", "basic-wl-no-desired").BasicPolicy().Obj()
	gangPodGroupDesired := st.MakePodGroup().Namespace("ns1").Name("pg5").TemplateRef("3", "gang-wl-desired").MinCount(2).DesiredCount(3).Obj()

	p1 := st.MakePod().Namespace("ns1").Name("p1").UID("p1").PodGroupName("pg1").Obj()
	p2 := st.MakePod().Namespace("ns1").Name("p2").UID("p2").PodGroupName("pg3").Obj()
	p3 := st.MakePod().Namespace("ns1").Name("p3").UID("p3").PodGroupName("pg3").Obj()
	p4 := st.MakePod().Namespace("ns1").Name("p4").UID("p4").PodGroupName("pg3").Obj()
	p5 := st.MakePod().Namespace("ns1").Name("p5").UID("p5").PodGroupName("pg3").Obj()

	gp1 := st.MakePod().Namespace("ns1").Name("gp1").UID("gp1").PodGroupName("pg5").Obj()
	gp2 := st.MakePod().Namespace("ns1").Name("gp2").UID("gp2").PodGroupName("pg5").Obj()
	gp3 := st.MakePod().Namespace("ns1").Name("gp3").UID("gp3").PodGroupName("pg5").Obj()

	nonGangPod := st.MakePod().Namespace("ns1").Name("non-gang").UID("non-gang").Obj()
	wronPgPod := st.MakePod().Namespace("ns1").Name("wrong-pg").UID("wrong-pg").PodGroupName("wrong").Obj()

	tests := []struct {
		name                       string
		enablePodGroupDesiredCount bool
		initialPodGroups           []*schedulingapi.PodGroup
		initialPods                []*v1.Pod
		pod                        *v1.Pod
		wantPreEnqueueStatus       *fwk.Status
	}{
		{
			name:                       "non-gang pod succeeds immediately",
			enablePodGroupDesiredCount: true,
			initialPodGroups:           []*schedulingapi.PodGroup{gangPodGroup1, gangPodGroup2, basicPodGroupDesired, basicPodGroupNoDesired, gangPodGroupDesired},
			pod:                        nonGangPod,
			wantPreEnqueueStatus:       nil,
		},
		{
			name:                       "gang pod fails PreEnqueue when workload is not yet created",
			enablePodGroupDesiredCount: true,
			pod:                        p1,
			wantPreEnqueueStatus:       fwk.NewStatus(fwk.UnschedulableAndUnresolvable, "waiting for pods's pod group \"pg1\" to appear in scheduling queue"),
		},
		{
			name:                       "gang pod fails PreEnqueue when PodGroup doesn't exist",
			pod:                        wronPgPod,
			enablePodGroupDesiredCount: true,
			initialPodGroups:           []*schedulingapi.PodGroup{gangPodGroup1, gangPodGroup2, basicPodGroupDesired, basicPodGroupNoDesired, gangPodGroupDesired},
			wantPreEnqueueStatus:       fwk.NewStatus(fwk.UnschedulableAndUnresolvable, "waiting for pods's pod group \"wrong\" to appear in scheduling queue"),
		},
		{
			name:                       "non basic policy gang pod succeeds immediately",
			enablePodGroupDesiredCount: true,
			initialPodGroups:           []*schedulingapi.PodGroup{gangPodGroup1, gangPodGroup2, basicPodGroupDesired, basicPodGroupNoDesired, gangPodGroupDesired},
			pod:                        p1,
			wantPreEnqueueStatus:       nil,
		},
		{
			name:                       "gang pod fails PreEnqueue when DesiredCount Not Met (feature Enabled)",
			enablePodGroupDesiredCount: true,
			initialPodGroups:           []*schedulingapi.PodGroup{gangPodGroup1, gangPodGroup2, basicPodGroupDesired, basicPodGroupNoDesired, gangPodGroupDesired},
			pod:                        p2,
			wantPreEnqueueStatus:       fwk.NewStatus(fwk.UnschedulableAndUnresolvable, "introducing delay while all pods count: 1 doesn't satisfy desired count requirement: 4"),
		},
		{
			name:                       "gang pod succeeds PreEnqueue when DesiredCount Met (feature Enabled)",
			enablePodGroupDesiredCount: true,
			initialPodGroups:           []*schedulingapi.PodGroup{gangPodGroup1, gangPodGroup2, basicPodGroupDesired, basicPodGroupNoDesired, gangPodGroupDesired},
			initialPods:                []*v1.Pod{p2, p3, p4},
			pod:                        p5,
			wantPreEnqueueStatus:       nil,
		},
		{
			name:                       "gang pod successds PreEnqeueu whe DesiredCount Not Met (feature Disabled)",
			enablePodGroupDesiredCount: false,
			initialPodGroups:           []*schedulingapi.PodGroup{gangPodGroup1, gangPodGroup2, basicPodGroupDesired, basicPodGroupNoDesired, gangPodGroupDesired},
			pod:                        p2,
			wantPreEnqueueStatus:       nil,
		},
		{
			name:                       "gang pod (Gang Policy) fails PreEnqueue when DesiredCount Not Met",
			enablePodGroupDesiredCount: true,
			initialPodGroups:           []*schedulingapi.PodGroup{gangPodGroup1, gangPodGroup2, basicPodGroupDesired, basicPodGroupNoDesired, gangPodGroupDesired},
			initialPods:                []*v1.Pod{gp1},
			pod:                        gp2,
			wantPreEnqueueStatus:       fwk.NewStatus(fwk.UnschedulableAndUnresolvable, "introducing delay while all pods count: 2 doesn't satisfy desired count requirement: 3"),
		},
		{
			name:                       "gang pod (Gang Policy) succeeds PreEnqueue when DesiredCount Met",
			enablePodGroupDesiredCount: true,
			initialPodGroups:           []*schedulingapi.PodGroup{gangPodGroup1, gangPodGroup2, basicPodGroupDesired, basicPodGroupNoDesired, gangPodGroupDesired},
			initialPods:                []*v1.Pod{gp1, gp2},
			pod:                        gp3,
			wantPreEnqueueStatus:       nil,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			logger, ctx := ktesting.NewTestContext(t)
			ctx, cancel := context.WithCancel(ctx)
			defer cancel()

			manager := podgroupmanager.New(logger)

			informerFactory := informers.NewSharedInformerFactory(fake.NewClientset(), 0)
			podGroupInformer := informerFactory.Scheduling().V1alpha2().PodGroups()

			fakeActivator := &podActivatorMock{}

			fh, err := frameworkruntime.NewFramework(ctx, nil, nil,
				frameworkruntime.WithInformerFactory(informerFactory),
				frameworkruntime.WithPodGroupManager(manager),
				frameworkruntime.WithWaitingPods(frameworkruntime.NewWaitingPodsMap()),
				frameworkruntime.WithPodActivator(fakeActivator),
			)
			if err != nil {
				t.Fatalf("Failed to create framework: %v", err)
			}

			// Populate informers and manager state for the test case.
			for _, wl := range tt.initialPodGroups {
				err := podGroupInformer.Informer().GetStore().Add(wl)
				if err != nil {
					t.Fatalf("Failed to add workload %s to store: %v", wl.Name, err)
				}
			}
			for _, p := range tt.initialPods {
				manager.AddPod(p)
			}
			manager.AddPod(tt.pod)

			p, err := New(ctx, nil, fh, feature.Features{
				EnableGangScheduling:       true,
				EnablePodGroupDesiredCount: tt.enablePodGroupDesiredCount,
			})
			if err != nil {
				t.Fatalf("Failed to create plugin: %v", err)
			}
			pl := p.(*Coscheduling)

			gotPreEnqueueStatus := pl.PreEnqueue(ctx, tt.pod)
			if diff := cmp.Diff(tt.wantPreEnqueueStatus, gotPreEnqueueStatus); diff != "" {
				t.Fatalf("Unexpected PreEnqueue status (-want,+got):\n%s", diff)
			}
		})
	}
}
