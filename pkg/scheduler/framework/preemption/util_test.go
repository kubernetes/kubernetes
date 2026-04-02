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
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"

	v1 "k8s.io/api/core/v1"
	policy "k8s.io/api/policy/v1"
	schedulingapi "k8s.io/api/scheduling/v1alpha3"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	fwk "k8s.io/kube-scheduler/framework"
	"k8s.io/kubernetes/pkg/scheduler/framework"
	st "k8s.io/kubernetes/pkg/scheduler/testing"
)

func TestFilterVictimsWithPDBViolation(t *testing.T) {
	newPodInfo := func(p *v1.Pod) fwk.PodInfo {
		pi, _ := framework.NewPodInfo(p)
		return pi
	}

	viNoPDBMatch := &victim{pods: []fwk.PodInfo{newPodInfo(st.MakePod().Name("p1").Label("app", "foo").Obj())}}
	viMatchPDB := &victim{pods: []fwk.PodInfo{newPodInfo(st.MakePod().Name("p1").Namespace(metav1.NamespaceDefault).Label("app", "foo").Obj())}}
	viMatchPDB2 := &victim{pods: []fwk.PodInfo{newPodInfo(st.MakePod().Name("p2").Namespace(metav1.NamespaceDefault).Label("app", "foo").Obj())}}
	viPodGroup := &victim{
		pods: []fwk.PodInfo{
			newPodInfo(st.MakePod().Name("p1").Namespace(metav1.NamespaceDefault).Label("app", "foo").Obj()),
			newPodInfo(st.MakePod().Name("p2").Namespace(metav1.NamespaceDefault).Label("app", "foo").Obj()),
		},
	}
	viMatchMultiplePDBs := &victim{pods: []fwk.PodInfo{newPodInfo(st.MakePod().Name("p1").Namespace(metav1.NamespaceDefault).Label("app", "foo").Label("tier", "backend").Obj())}}

	tests := []struct {
		name                 string
		victims              []*victim
		pdbs                 []*policy.PodDisruptionBudget
		expectedViolating    []ViolatingVictim[*victim]
		expectedNonViolating []*victim
	}{
		{
			name:                 "no victims, no PDBs",
			victims:              nil,
			pdbs:                 nil,
			expectedViolating:    nil,
			expectedNonViolating: nil,
		},
		{
			name:    "victim with no matching PDBs",
			victims: []*victim{viNoPDBMatch},
			pdbs: []*policy.PodDisruptionBudget{
				{
					ObjectMeta: metav1.ObjectMeta{Namespace: metav1.NamespaceDefault},
					Spec: policy.PodDisruptionBudgetSpec{
						Selector: &metav1.LabelSelector{MatchLabels: map[string]string{"app": "bar"}},
					},
				},
			},
			expectedViolating:    nil,
			expectedNonViolating: []*victim{viNoPDBMatch},
		},
		{
			name:    "victim matching PDB, adequate DisruptionsAllowed",
			victims: []*victim{viMatchPDB},
			pdbs: []*policy.PodDisruptionBudget{
				{
					ObjectMeta: metav1.ObjectMeta{Namespace: metav1.NamespaceDefault},
					Spec: policy.PodDisruptionBudgetSpec{
						Selector: &metav1.LabelSelector{MatchLabels: map[string]string{"app": "foo"}},
					},
					Status: policy.PodDisruptionBudgetStatus{
						DisruptionsAllowed: 1,
					},
				},
			},
			expectedViolating:    nil,
			expectedNonViolating: []*victim{viMatchPDB},
		},
		{
			name:    "victim matching PDB, no DisruptionsAllowed",
			victims: []*victim{viMatchPDB},
			pdbs: []*policy.PodDisruptionBudget{
				{
					ObjectMeta: metav1.ObjectMeta{Namespace: metav1.NamespaceDefault},
					Spec: policy.PodDisruptionBudgetSpec{
						Selector: &metav1.LabelSelector{MatchLabels: map[string]string{"app": "foo"}},
					},
					Status: policy.PodDisruptionBudgetStatus{
						DisruptionsAllowed: 0,
					},
				},
			},
			expectedViolating: []ViolatingVictim[*victim]{
				{
					Victim:       viMatchPDB,
					ViolateCount: 1,
				},
			},
			expectedNonViolating: nil,
		},
		{
			name:    "podgroup victim with multiple pods, all violating",
			victims: []*victim{viPodGroup},
			pdbs: []*policy.PodDisruptionBudget{
				{
					ObjectMeta: metav1.ObjectMeta{Namespace: metav1.NamespaceDefault},
					Spec: policy.PodDisruptionBudgetSpec{
						Selector: &metav1.LabelSelector{MatchLabels: map[string]string{"app": "foo"}},
					},
					Status: policy.PodDisruptionBudgetStatus{
						DisruptionsAllowed: 0,
					},
				},
			},
			expectedViolating: []ViolatingVictim[*victim]{
				{
					Victim:       viPodGroup,
					ViolateCount: 2,
				},
			},
			expectedNonViolating: nil,
		},
		{
			name:    "podgroup victim with multiple pods, none violating",
			victims: []*victim{viPodGroup},
			pdbs: []*policy.PodDisruptionBudget{
				{
					ObjectMeta: metav1.ObjectMeta{Namespace: metav1.NamespaceDefault},
					Spec: policy.PodDisruptionBudgetSpec{
						Selector: &metav1.LabelSelector{MatchLabels: map[string]string{"app": "foo"}},
					},
					Status: policy.PodDisruptionBudgetStatus{
						DisruptionsAllowed: 2,
					},
				},
			},
			expectedViolating:    nil,
			expectedNonViolating: []*victim{viPodGroup},
		},
		{
			name:    "multiple victims matching the same PDB",
			victims: []*victim{viMatchPDB, viMatchPDB2},
			pdbs: []*policy.PodDisruptionBudget{
				{
					ObjectMeta: metav1.ObjectMeta{Namespace: metav1.NamespaceDefault},
					Spec: policy.PodDisruptionBudgetSpec{
						Selector: &metav1.LabelSelector{MatchLabels: map[string]string{"app": "foo"}},
					},
					Status: policy.PodDisruptionBudgetStatus{
						DisruptionsAllowed: 1,
					},
				},
			},
			expectedViolating: []ViolatingVictim[*victim]{
				{
					Victim:       viMatchPDB2,
					ViolateCount: 1,
				},
			},
			expectedNonViolating: []*victim{viMatchPDB},
		},
		{
			name:    "pod in DisruptedPods is ignored",
			victims: []*victim{viMatchPDB},
			pdbs: []*policy.PodDisruptionBudget{
				{
					ObjectMeta: metav1.ObjectMeta{Namespace: metav1.NamespaceDefault},
					Spec: policy.PodDisruptionBudgetSpec{
						Selector: &metav1.LabelSelector{MatchLabels: map[string]string{"app": "foo"}},
					},
					Status: policy.PodDisruptionBudgetStatus{
						DisruptionsAllowed: 0,
						DisruptedPods: map[string]metav1.Time{
							"p1": {Time: time.Now()},
						},
					},
				},
			},
			expectedViolating:    nil,
			expectedNonViolating: []*victim{viMatchPDB},
		},
		{
			name:    "PDB with empty selector",
			victims: []*victim{viMatchPDB},
			pdbs: []*policy.PodDisruptionBudget{
				{
					ObjectMeta: metav1.ObjectMeta{Namespace: metav1.NamespaceDefault},
					Spec: policy.PodDisruptionBudgetSpec{
						Selector: &metav1.LabelSelector{}, // matches nothing
					},
					Status: policy.PodDisruptionBudgetStatus{
						DisruptionsAllowed: 0,
					},
				},
			},
			expectedViolating:    nil,
			expectedNonViolating: []*victim{viMatchPDB},
		},
		{
			name:    "Multiple PDBs",
			victims: []*victim{viMatchMultiplePDBs},
			pdbs: []*policy.PodDisruptionBudget{
				{
					ObjectMeta: metav1.ObjectMeta{Namespace: metav1.NamespaceDefault},
					Spec: policy.PodDisruptionBudgetSpec{
						Selector: &metav1.LabelSelector{MatchLabels: map[string]string{"app": "foo"}},
					},
					Status: policy.PodDisruptionBudgetStatus{
						DisruptionsAllowed: 1,
					},
				},
				{
					ObjectMeta: metav1.ObjectMeta{Namespace: metav1.NamespaceDefault},
					Spec: policy.PodDisruptionBudgetSpec{
						Selector: &metav1.LabelSelector{MatchLabels: map[string]string{"tier": "backend"}},
					},
					Status: policy.PodDisruptionBudgetStatus{
						DisruptionsAllowed: 0,
					},
				},
			},
			expectedViolating: []ViolatingVictim[*victim]{
				{
					Victim:       viMatchMultiplePDBs,
					ViolateCount: 1,
				},
			},
			expectedNonViolating: nil,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			violating, nonViolating := FilterVictimsWithPDBViolation(tt.victims, tt.pdbs)

			if diff := cmp.Diff(tt.expectedViolating, violating, cmp.Comparer(func(a, b *victim) bool { return a == b })); diff != "" {
				t.Errorf("violating victims mismatch (-want, +got):\n%s", diff)
			}

			if diff := cmp.Diff(tt.expectedNonViolating, nonViolating, cmp.Comparer(func(a, b *victim) bool { return a == b })); diff != "" {
				t.Errorf("nonViolating victims mismatch (-want, +got):\n%s", diff)
			}
		})
	}
}

func TestMoreImportantVictim(t *testing.T) {
	newPodInfo := func(p *v1.Pod) fwk.PodInfo {
		pi, _ := framework.NewPodInfo(p)
		return pi
	}

	now := &metav1.Time{Time: time.Unix(1000, 0)}
	before := &metav1.Time{Time: time.Unix(500, 0)}

	tests := []struct {
		name                   string
		vi1                    *victim
		vi2                    *victim
		genericWorkloadEnabled bool
		want                   bool
	}{
		{
			name:                   "vi1 has higher priority",
			vi1:                    &victim{priority: 20},
			vi2:                    &victim{priority: 10},
			genericWorkloadEnabled: true,
			want:                   true,
		},
		{
			name:                   "vi2 has higher priority",
			vi1:                    &victim{priority: 10},
			vi2:                    &victim{priority: 20},
			genericWorkloadEnabled: true,
			want:                   false,
		},
		{
			name:                   "vi1 is PG, vi2 is Pod, same priority",
			vi1:                    &victim{priority: 10, pods: []fwk.PodInfo{newPodInfo(st.MakePod().PodGroupName("pg").Obj())}},
			vi2:                    &victim{priority: 10, pods: []fwk.PodInfo{newPodInfo(st.MakePod().Obj())}},
			genericWorkloadEnabled: true,
			want:                   true,
		},
		{
			name:                   "vi1 is Pod, vi2 is PG, same priority",
			vi1:                    &victim{priority: 10, pods: []fwk.PodInfo{newPodInfo(st.MakePod().Obj())}},
			vi2:                    &victim{priority: 10, pods: []fwk.PodInfo{newPodInfo(st.MakePod().PodGroupName("pg").Obj())}},
			genericWorkloadEnabled: true,
			want:                   false,
		},
		{
			name:                   "both Pods, vi1 older",
			vi1:                    &victim{priority: 10, pods: []fwk.PodInfo{newPodInfo(st.MakePod().Obj())}, earliestStartTime: before},
			vi2:                    &victim{priority: 10, pods: []fwk.PodInfo{newPodInfo(st.MakePod().Obj())}, earliestStartTime: now},
			genericWorkloadEnabled: true,
			want:                   true,
		},
		{
			name:                   "both Pods, vi2 older",
			vi1:                    &victim{priority: 10, pods: []fwk.PodInfo{newPodInfo(st.MakePod().Obj())}, earliestStartTime: now},
			vi2:                    &victim{priority: 10, pods: []fwk.PodInfo{newPodInfo(st.MakePod().Obj())}, earliestStartTime: before},
			genericWorkloadEnabled: true,
			want:                   false,
		},
		{
			name: "both PGs, vi1 larger",
			vi1: &victim{
				priority: 10,
				pods: []fwk.PodInfo{
					newPodInfo(st.MakePod().PodGroupName("pg").Obj()),
					newPodInfo(st.MakePod().PodGroupName("pg").Obj()),
				},
			},
			vi2: &victim{
				priority: 10,
				pods: []fwk.PodInfo{
					newPodInfo(st.MakePod().PodGroupName("pg").Obj()),
				},
			},
			genericWorkloadEnabled: true,
			want:                   true,
		},
		{
			name: "both PGs, vi2 larger",
			vi1: &victim{
				priority: 10,
				pods: []fwk.PodInfo{
					newPodInfo(st.MakePod().PodGroupName("pg").Obj()),
				},
			},
			vi2: &victim{
				priority: 10,
				pods: []fwk.PodInfo{
					newPodInfo(st.MakePod().PodGroupName("pg").Obj()),
					newPodInfo(st.MakePod().PodGroupName("pg").Obj()),
				},
			},
			genericWorkloadEnabled: true,
			want:                   false,
		},
		{
			name: "both PGs, same size, vi1 older",
			vi1: &victim{
				priority: 10,
				pods: []fwk.PodInfo{
					newPodInfo(st.MakePod().PodGroupName("pg").Obj()),
					newPodInfo(st.MakePod().PodGroupName("pg").Obj()),
				},
				earliestStartTime: before,
			},
			vi2: &victim{
				priority: 10,
				pods: []fwk.PodInfo{
					newPodInfo(st.MakePod().PodGroupName("pg").Obj()),
					newPodInfo(st.MakePod().PodGroupName("pg").Obj()),
				},
				earliestStartTime: now,
			},
			genericWorkloadEnabled: true,
			want:                   true,
		},
		{
			name: "both PGs, same size, vi2 older",
			vi1: &victim{
				priority: 10,
				pods: []fwk.PodInfo{
					newPodInfo(st.MakePod().PodGroupName("pg").Obj()),
					newPodInfo(st.MakePod().PodGroupName("pg").Obj()),
				},
				earliestStartTime: now,
			},
			vi2: &victim{
				priority: 10,
				pods: []fwk.PodInfo{
					newPodInfo(st.MakePod().PodGroupName("pg").Obj()),
					newPodInfo(st.MakePod().PodGroupName("pg").Obj()),
				},
				earliestStartTime: before,
			},
			genericWorkloadEnabled: true,
			want:                   false,
		},
		{
			name: "GenereicWorkload disabled: vi1 is larger PodGroup but newer, vi2 is older Pod — start time wins",
			vi1: &victim{
				priority:          10,
				pods:              []fwk.PodInfo{newPodInfo(st.MakePod().PodGroupName("pg").Obj()), newPodInfo(st.MakePod().PodGroupName("pg").Obj())},
				earliestStartTime: now,
			},
			vi2:                    &victim{priority: 10, pods: []fwk.PodInfo{newPodInfo(st.MakePod().Obj())}, earliestStartTime: before},
			genericWorkloadEnabled: false,
			want:                   false,
		},
		{
			name: "GenereicWorkload disabled: both PGs, vi1 larger but newer — start time wins",
			vi1: &victim{
				priority:          10,
				pods:              []fwk.PodInfo{newPodInfo(st.MakePod().PodGroupName("pg").Obj()), newPodInfo(st.MakePod().PodGroupName("pg").Obj())},
				earliestStartTime: now,
			},
			vi2: &victim{
				priority:          10,
				pods:              []fwk.PodInfo{newPodInfo(st.MakePod().PodGroupName("pg").Obj())},
				earliestStartTime: before,
			},
			genericWorkloadEnabled: false,
			want:                   false,
		},
		{
			name: "GenereicWorkload disabled: both PGs, vi1 larger and older — start time wins",
			vi1: &victim{
				priority:          10,
				pods:              []fwk.PodInfo{newPodInfo(st.MakePod().PodGroupName("pg").Obj()), newPodInfo(st.MakePod().PodGroupName("pg").Obj())},
				earliestStartTime: before,
			},
			vi2: &victim{
				priority:          10,
				pods:              []fwk.PodInfo{newPodInfo(st.MakePod().PodGroupName("pg").Obj())},
				earliestStartTime: now,
			},
			genericWorkloadEnabled: false,
			want:                   true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := MoreImportantVictim(tt.vi1, tt.vi2, tt.genericWorkloadEnabled)
			if got != tt.want {
				t.Errorf("MoreImportantVictim() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestPodTerminatingByPreemption(t *testing.T) {
	tests := []struct {
		name string
		pod  *v1.Pod
		want bool
	}{
		{
			name: "pod without DeletionTimestamp",
			pod:  st.MakePod().Name("p1").Obj(),
			want: false,
		},
		{
			name: "pod with DeletionTimestamp but no DisruptionTarget condition",
			pod:  st.MakePod().Name("p1").Terminating().Obj(),
			want: false,
		},
		{
			name: "pod with DeletionTimestamp, DisruptionTarget condition status False",
			pod: st.MakePod().Name("p1").
				Terminating().
				Condition(v1.DisruptionTarget, v1.ConditionFalse, v1.PodReasonPreemptionByScheduler).
				Obj(),
			want: false,
		},
		{
			name: "pod with DeletionTimestamp, DisruptionTarget condition reason not PreemptionByScheduler",
			pod: st.MakePod().Name("p1").
				Terminating().
				Condition(v1.DisruptionTarget, v1.ConditionTrue, "SomeOtherReason").
				Obj(),
			want: false,
		},
		{
			name: "pod with DeletionTimestamp, DisruptionTarget condition status True and reason PreemptionByScheduler",
			pod: st.MakePod().Name("p1").
				Terminating().
				Condition(v1.DisruptionTarget, v1.ConditionTrue, v1.PodReasonPreemptionByScheduler).
				Obj(),
			want: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := PodTerminatingByPreemption(tt.pod)
			if got != tt.want {
				t.Errorf("PodTerminatingByPreemption() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestGetPodPriority(t *testing.T) {
	tests := []struct {
		name             string
		pod              *v1.Pod
		podGroupLister   fwk.PodGroupLister
		expectedPriority int32
	}{
		{
			name:             "pod without PodGroup returns pod priority",
			pod:              st.MakePod().Name("p1").Priority(10).Obj(),
			podGroupLister:   &mockPodGroupLister{},
			expectedPriority: 10,
		},
		{
			name: "pod with PodGroup returns PodGroup priority instead of pod priority",
			pod:  st.MakePod().Name("p2").Priority(10).PodGroupName("pg1").Obj(),
			podGroupLister: &mockPodGroupLister{
				podGroups: map[string]*schedulingapi.PodGroup{
					"pg1": st.MakePodGroup().Name("pg1").Priority(50).Obj(),
				},
			},
			expectedPriority: 50,
		},
		{
			name:             "pod with PodGroup but nil PodGroupLister falls back to pod priority",
			pod:              st.MakePod().Name("p4").Priority(20).PodGroupName("pg1").Obj(),
			podGroupLister:   nil,
			expectedPriority: 20,
		},
		{
			name:             "pod with PodGroup but PodGroup not found in lister falls back to pod priority",
			pod:              st.MakePod().Name("p5").Priority(30).PodGroupName("missing-pg").Obj(),
			podGroupLister:   &mockPodGroupLister{podGroups: map[string]*schedulingapi.PodGroup{}},
			expectedPriority: 30,
		},
		{
			name:             "pod with nil priority and without PodGroup returns zero priority",
			pod:              st.MakePod().Name("p6").Obj(),
			podGroupLister:   &mockPodGroupLister{},
			expectedPriority: 0,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := GetPodPriority(tt.pod, tt.podGroupLister)
			if got != tt.expectedPriority {
				t.Errorf("GetPodPriority() = %v, want %v", got, tt.expectedPriority)
			}
		})
	}
}
