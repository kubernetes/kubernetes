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

package defaultpreemption

import (
	"fmt"
	"sort"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	"github.com/google/go-cmp/cmp/cmpopts"
	v1 "k8s.io/api/core/v1"
	scheduling "k8s.io/api/scheduling/v1alpha1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/client-go/informers"
	clientsetfake "k8s.io/client-go/kubernetes/fake"
	"k8s.io/klog/v2/ktesting"
	fwk "k8s.io/kube-scheduler/framework"
	"k8s.io/kubernetes/pkg/scheduler/framework"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/feature"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/helper"
	st "k8s.io/kubernetes/pkg/scheduler/testing"
	"k8s.io/kubernetes/pkg/scheduler/util"
	"k8s.io/utils/ptr"
)

type fakeWorkloadManager struct {
	fwk.WorkloadManager
	podGroupStates map[helper.PodGroupKey]fwk.PodGroupState
}

func (f *fakeWorkloadManager) PodGroupState(namespace string, ref *v1.WorkloadReference) (fwk.PodGroupState, error) {
	sig := helper.NewPodGroupKey(namespace, ref)
	if pgs, ok := f.podGroupStates[sig]; ok {
		return pgs, nil
	}
	return nil, fmt.Errorf("not found")
}

type fakePodGroupState struct {
	fwk.PodGroupState
	pods      []*v1.Pod
	startTime *metav1.Time
}

func (f *fakePodGroupState) ScheduledPods() sets.Set[*v1.Pod] {
	return sets.New(f.pods...)
}

func (f *fakePodGroupState) StartTime() *metav1.Time {
	return f.startTime
}

func TestPreparePotentialVictimPodGroups(t *testing.T) {
	epochTime := metav1.NewTime(time.Unix(0, 0))
	epochTime1 := metav1.NewTime(time.Unix(0, 1))
	pg1 := &scheduling.PodGroup{
		Name: "pg1",
		Policy: scheduling.PodGroupPolicy{
			Gang: &scheduling.GangSchedulingPolicy{
				DisruptionMode: ptr.To(scheduling.DisruptionModePodGroup),
			},
		},
	}
	wl1 := st.MakeWorkload().Name("wl1").Namespace("default").PodGroup(pg1).Obj()
	ref1 := &v1.WorkloadReference{Name: "wl1", PodGroup: "pg1", PodGroupReplicaKey: "rk1"}

	tests := []struct {
		name                          string
		enableWorkloadAwarePreemption bool
		potentialVictims              []*v1.Pod
		workloads                     []*scheduling.Workload
		podGroupStates                map[helper.PodGroupKey]fwk.PodGroupState
		want                          []*util.VictimGroup
	}{
		{
			name:                          "workload aware preemption disabled",
			enableWorkloadAwarePreemption: false,
			potentialVictims: []*v1.Pod{
				st.MakePod().Name("p1").UID("p1").Priority(10).StartTime(epochTime).Obj(),
				st.MakePod().Name("p2").UID("p2").Priority(20).StartTime(epochTime1).Obj(),
			},
			want: []*util.VictimGroup{
				{Pods: []*v1.Pod{st.MakePod().Name("p2").UID("p2").Priority(20).StartTime(epochTime1).Obj()}, Priority: 20, StartTime: &epochTime1},
				{Pods: []*v1.Pod{st.MakePod().Name("p1").UID("p1").Priority(10).StartTime(epochTime).Obj()}, Priority: 10, StartTime: &epochTime},
			},
		},
		{
			name:                          "workload aware preemption enabled, no gang",
			enableWorkloadAwarePreemption: true,
			potentialVictims: []*v1.Pod{
				st.MakePod().Name("p1").UID("p1").Priority(10).StartTime(epochTime).Obj(),
			},
			want: []*util.VictimGroup{
				{Pods: []*v1.Pod{st.MakePod().Name("p1").UID("p1").Priority(10).StartTime(epochTime).Obj()}, Priority: 10, StartTime: &epochTime},
			},
		},
		{
			name:                          "workload aware preemption enabled, gang with disruption mode",
			enableWorkloadAwarePreemption: true,
			potentialVictims: []*v1.Pod{
				st.MakePod().Name("p1").UID("p1").Namespace("default").Priority(10).WorkloadRef(ref1).StartTime(epochTime).Obj(),
			},
			workloads: []*scheduling.Workload{wl1},
			podGroupStates: map[helper.PodGroupKey]fwk.PodGroupState{
				helper.NewPodGroupKey("default", ref1): &fakePodGroupState{
					pods: []*v1.Pod{
						st.MakePod().Name("p1").UID("p1").Namespace("default").Priority(10).WorkloadRef(ref1).StartTime(epochTime).Obj(),
						st.MakePod().Name("p2").UID("p2").Namespace("default").Priority(10).WorkloadRef(ref1).StartTime(epochTime1).Obj(),
					},
					startTime: &epochTime,
				},
			},
			want: []*util.VictimGroup{
				{
					Pods: []*v1.Pod{
						st.MakePod().Name("p1").UID("p1").Namespace("default").Priority(10).WorkloadRef(ref1).StartTime(epochTime).Obj(),
						st.MakePod().Name("p2").UID("p2").Namespace("default").Priority(10).WorkloadRef(ref1).StartTime(epochTime1).Obj(),
					},
					Priority:  10,
					StartTime: &epochTime,
					IsGang:    true,
				},
			},
		},
		{
			name:                          "workload aware preemption enabled, gang but missing state",
			enableWorkloadAwarePreemption: true,
			potentialVictims: []*v1.Pod{
				st.MakePod().Name("p1").UID("p1").Namespace("default").Priority(10).WorkloadRef(ref1).StartTime(epochTime).Obj(),
			},
			workloads: []*scheduling.Workload{wl1},
			// Missing podGroupStates
			want: []*util.VictimGroup{}, // It skips if state is missing
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cs := clientsetfake.NewClientset()
			informerFactory := informers.NewSharedInformerFactory(cs, 0)
			wlInformer := informerFactory.Scheduling().V1alpha1().Workloads().Informer()
			for _, wl := range tt.workloads {
				if err := wlInformer.GetStore().Add(wl); err != nil {
					t.Fatal(err)
				}
			}
			pl := &DefaultPreemption{
				fts:                   feature.Features{EnableWorkloadAwarePreemption: tt.enableWorkloadAwarePreemption},
				wl:                    informerFactory.Scheduling().V1alpha1().Workloads().Lister(),
				wm:                    &fakeWorkloadManager{podGroupStates: tt.podGroupStates},
				MoreImportantPodGroup: util.MoreImportantPodGroup,
			}

			potentialVictimsOnNode := make(map[types.UID]fwk.PodInfo)
			for _, p := range tt.potentialVictims {
				// We only need the Pod object for this method
				potentialVictimsOnNode[p.UID] = &framework.PodInfo{Pod: p}
			}

			got := pl.preparePotentialVictimPodGroups(potentialVictimsOnNode, ktesting.NewLogger(t, ktesting.NewConfig()))
			// Sort victim groups inside the obtained result for deterministic checks.
			sort.Slice(got, func(i, j int) bool {
				return pl.MoreImportantPodGroup(got[i], got[j], tt.enableWorkloadAwarePreemption)
			})

			if diff := cmp.Diff(tt.want, got, cmpopts.EquateEmpty()); diff != "" {
				t.Errorf("preparePotentialVictimPodGroups() mismatch (-want +got):\n%s", diff)
			}
		})
	}
}
