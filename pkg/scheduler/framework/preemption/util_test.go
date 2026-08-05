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
	schedulingv1alpha3 "k8s.io/api/scheduling/v1alpha3"
	schedulingv1beta1 "k8s.io/api/scheduling/v1beta1"
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

	viNoPDBMatch := &victim{pods: []fwk.PodInfo{newPodInfo(st.MakePod().Name("p1").Label("app", "foo").Obj())}, keyType: fwk.PodKeyType}
	viMatchPDB := &victim{pods: []fwk.PodInfo{newPodInfo(st.MakePod().Name("p1").Namespace(metav1.NamespaceDefault).Label("app", "foo").Obj())}, keyType: fwk.PodKeyType}
	viMatchPDB2 := &victim{pods: []fwk.PodInfo{newPodInfo(st.MakePod().Name("p2").Namespace(metav1.NamespaceDefault).Label("app", "foo").Obj())}, keyType: fwk.PodKeyType}
	viPodGroup := &victim{
		pods: []fwk.PodInfo{
			newPodInfo(st.MakePod().Name("p1").Namespace(metav1.NamespaceDefault).Label("app", "foo").Obj()),
			newPodInfo(st.MakePod().Name("p2").Namespace(metav1.NamespaceDefault).Label("app", "foo").Obj()),
		},
		keyType: fwk.PodKeyType,
	}
	viMatchMultiplePDBs := &victim{pods: []fwk.PodInfo{newPodInfo(st.MakePod().Name("p1").Namespace(metav1.NamespaceDefault).Label("app", "foo").Label("tier", "backend").Obj())}, keyType: fwk.PodKeyType}

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
		name string
		vi1  *victim
		vi2  *victim
		want bool
	}{
		{
			name: "vi1 has higher priority",
			vi1:  &victim{priority: 20, keyType: fwk.PodKeyType},
			vi2:  &victim{priority: 10, keyType: fwk.PodKeyType},
			want: true,
		},
		{
			name: "vi2 has higher priority",
			vi1:  &victim{priority: 10, keyType: fwk.PodKeyType},
			vi2:  &victim{priority: 20, keyType: fwk.PodKeyType},
			want: false,
		},
		{
			name: "vi1 is PG, vi2 is Pod, same priority",
			vi1:  &victim{priority: 10, pods: []fwk.PodInfo{newPodInfo(st.MakePod().PodGroupName("pg").Obj())}, keyType: fwk.PodGroupKeyType},
			vi2:  &victim{priority: 10, pods: []fwk.PodInfo{newPodInfo(st.MakePod().Obj())}, keyType: fwk.PodKeyType},
			want: true,
		},
		{
			name: "vi1 is Pod, vi2 is PG, same priority",
			vi1:  &victim{priority: 10, pods: []fwk.PodInfo{newPodInfo(st.MakePod().Obj())}, keyType: fwk.PodKeyType},
			vi2:  &victim{priority: 10, pods: []fwk.PodInfo{newPodInfo(st.MakePod().PodGroupName("pg").Obj())}, keyType: fwk.PodGroupKeyType},
			want: false,
		},
		{
			name: "both Pods, vi1 older",
			vi1:  &victim{priority: 10, pods: []fwk.PodInfo{newPodInfo(st.MakePod().Obj())}, earliestStartTime: before, keyType: fwk.PodKeyType},
			vi2:  &victim{priority: 10, pods: []fwk.PodInfo{newPodInfo(st.MakePod().Obj())}, earliestStartTime: now, keyType: fwk.PodKeyType},
			want: true,
		},
		{
			name: "both Pods, vi2 older",
			vi1:  &victim{priority: 10, pods: []fwk.PodInfo{newPodInfo(st.MakePod().Obj())}, earliestStartTime: now, keyType: fwk.PodKeyType},
			vi2:  &victim{priority: 10, pods: []fwk.PodInfo{newPodInfo(st.MakePod().Obj())}, earliestStartTime: before, keyType: fwk.PodKeyType},
			want: false,
		},
		{
			name: "both PGs, vi1 larger",
			vi1: &victim{
				priority: 10,
				pods: []fwk.PodInfo{
					newPodInfo(st.MakePod().PodGroupName("pg").Obj()),
					newPodInfo(st.MakePod().PodGroupName("pg").Obj()),
				},
				keyType: fwk.PodGroupKeyType,
			},
			vi2: &victim{
				priority: 10,
				pods: []fwk.PodInfo{
					newPodInfo(st.MakePod().PodGroupName("pg").Obj()),
				},
				keyType: fwk.PodGroupKeyType,
			},
			want: true,
		},
		{
			name: "both PGs, vi2 larger",
			vi1: &victim{
				priority: 10,
				pods: []fwk.PodInfo{
					newPodInfo(st.MakePod().PodGroupName("pg").Obj()),
				},
				keyType: fwk.PodGroupKeyType,
			},
			vi2: &victim{
				priority: 10,
				pods: []fwk.PodInfo{
					newPodInfo(st.MakePod().PodGroupName("pg").Obj()),
					newPodInfo(st.MakePod().PodGroupName("pg").Obj()),
				},
				keyType: fwk.PodGroupKeyType,
			},
			want: false,
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
				keyType:           fwk.PodGroupKeyType,
			},
			vi2: &victim{
				priority: 10,
				pods: []fwk.PodInfo{
					newPodInfo(st.MakePod().PodGroupName("pg").Obj()),
					newPodInfo(st.MakePod().PodGroupName("pg").Obj()),
				},
				earliestStartTime: now,
				keyType:           fwk.PodGroupKeyType,
			},
			want: true,
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
				keyType:           fwk.PodGroupKeyType,
			},
			vi2: &victim{
				priority: 10,
				pods: []fwk.PodInfo{
					newPodInfo(st.MakePod().PodGroupName("pg").Obj()),
					newPodInfo(st.MakePod().PodGroupName("pg").Obj()),
				},
				earliestStartTime: before,
				keyType:           fwk.PodGroupKeyType,
			},
			want: false,
		},
		{
			name: "CPG vs Pod, same priority (CPG wins)",
			vi1: &victim{
				priority: 10,
				keyType:  fwk.CompositePodGroupKeyType,
				pods:     []fwk.PodInfo{newPodInfo(st.MakePod().PodGroupName("pg").Obj())},
			},
			vi2: &victim{
				priority: 10,
				keyType:  fwk.PodKeyType,
				pods:     []fwk.PodInfo{newPodInfo(st.MakePod().Obj())},
			},
			want: true,
		},
		{
			name: "Pod vs CPG, same priority (CPG wins)",
			vi1: &victim{
				priority: 10,
				keyType:  fwk.PodKeyType,
				pods:     []fwk.PodInfo{newPodInfo(st.MakePod().Obj())},
			},
			vi2: &victim{
				priority: 10,
				keyType:  fwk.CompositePodGroupKeyType,
				pods:     []fwk.PodInfo{newPodInfo(st.MakePod().PodGroupName("pg").Obj())},
			},
			want: false,
		},
		{
			name: "CPG vs PG, same priority (CPG wins)",
			vi1: &victim{
				priority: 10,
				keyType:  fwk.CompositePodGroupKeyType,
				pods:     []fwk.PodInfo{newPodInfo(st.MakePod().PodGroupName("pg").Obj())},
			},
			vi2: &victim{
				priority: 10,
				keyType:  fwk.PodGroupKeyType,
				pods:     []fwk.PodInfo{newPodInfo(st.MakePod().PodGroupName("pg").Obj())},
			},
			want: true,
		},
		{
			name: "PG vs CPG, same priority (CPG wins)",
			vi1: &victim{
				priority: 10,
				keyType:  fwk.PodGroupKeyType,
				pods:     []fwk.PodInfo{newPodInfo(st.MakePod().PodGroupName("pg").Obj())},
			},
			vi2: &victim{
				priority: 10,
				keyType:  fwk.CompositePodGroupKeyType,
				pods:     []fwk.PodInfo{newPodInfo(st.MakePod().PodGroupName("pg").Obj())},
			},
			want: false,
		},
		{
			name: "CPG vs PG, PG higher priority (higher priority wins)",
			vi1: &victim{
				priority: 10,
				keyType:  fwk.CompositePodGroupKeyType,
				pods:     []fwk.PodInfo{newPodInfo(st.MakePod().PodGroupName("pg").Obj())},
			},
			vi2: &victim{
				priority: 20,
				keyType:  fwk.PodGroupKeyType,
				pods:     []fwk.PodInfo{newPodInfo(st.MakePod().PodGroupName("pg").Obj())},
			},
			want: false,
		},
		{
			name: "PG vs CPG, PG higher priority (higher priority wins)",
			vi1: &victim{
				priority: 20,
				keyType:  fwk.PodGroupKeyType,
				pods:     []fwk.PodInfo{newPodInfo(st.MakePod().PodGroupName("pg").Obj())},
			},
			vi2: &victim{
				priority: 10,
				keyType:  fwk.CompositePodGroupKeyType,
				pods:     []fwk.PodInfo{newPodInfo(st.MakePod().PodGroupName("pg").Obj())},
			},
			want: true,
		},
		{
			name: "CPG vs PG, CPG smaller size, same priority (CPG rank wins over size)",
			vi1: &victim{
				priority: 10,
				keyType:  fwk.CompositePodGroupKeyType,
				pods:     []fwk.PodInfo{newPodInfo(st.MakePod().PodGroupName("pg").Obj())},
			},
			vi2: &victim{
				priority: 10,
				keyType:  fwk.PodGroupKeyType,
				pods: []fwk.PodInfo{
					newPodInfo(st.MakePod().PodGroupName("pg").Obj()),
					newPodInfo(st.MakePod().PodGroupName("pg").Obj()),
				},
			},
			want: true,
		},
		{
			name: "CPG vs PG, CPG newer start time, same priority (CPG rank wins over start time)",
			vi1: &victim{
				priority:          10,
				keyType:           fwk.CompositePodGroupKeyType,
				pods:              []fwk.PodInfo{newPodInfo(st.MakePod().PodGroupName("pg").Obj())},
				earliestStartTime: now,
			},
			vi2: &victim{
				priority:          10,
				keyType:           fwk.PodGroupKeyType,
				pods:              []fwk.PodInfo{newPodInfo(st.MakePod().PodGroupName("pg").Obj())},
				earliestStartTime: before,
			},
			want: true,
		},
		{
			name: "both CPGs, vi1 larger (larger size wins)",
			vi1: &victim{
				priority: 10,
				pods: []fwk.PodInfo{
					newPodInfo(st.MakePod().PodGroupName("pg").Obj()),
					newPodInfo(st.MakePod().PodGroupName("pg").Obj()),
				},
				keyType: fwk.CompositePodGroupKeyType,
			},
			vi2: &victim{
				priority: 10,
				pods: []fwk.PodInfo{
					newPodInfo(st.MakePod().PodGroupName("pg").Obj()),
				},
				keyType: fwk.CompositePodGroupKeyType,
			},
			want: true,
		},
		{
			name: "both CPGs, vi2 larger (larger size wins)",
			vi1: &victim{
				priority: 10,
				pods: []fwk.PodInfo{
					newPodInfo(st.MakePod().PodGroupName("pg").Obj()),
				},
				keyType: fwk.CompositePodGroupKeyType,
			},
			vi2: &victim{
				priority: 10,
				pods: []fwk.PodInfo{
					newPodInfo(st.MakePod().PodGroupName("pg").Obj()),
					newPodInfo(st.MakePod().PodGroupName("pg").Obj()),
				},
				keyType: fwk.CompositePodGroupKeyType,
			},
			want: false,
		},
		{
			name: "both CPGs, same size, vi1 older (earlier start time wins)",
			vi1: &victim{
				priority: 10,
				pods: []fwk.PodInfo{
					newPodInfo(st.MakePod().PodGroupName("pg").Obj()),
				},
				earliestStartTime: before,
				keyType:           fwk.CompositePodGroupKeyType,
			},
			vi2: &victim{
				priority: 10,
				pods: []fwk.PodInfo{
					newPodInfo(st.MakePod().PodGroupName("pg").Obj()),
				},
				earliestStartTime: now,
				keyType:           fwk.CompositePodGroupKeyType,
			},
			want: true,
		},
		{
			name: "both CPGs, same size, vi2 older (earlier start time wins)",
			vi1: &victim{
				priority: 10,
				pods: []fwk.PodInfo{
					newPodInfo(st.MakePod().PodGroupName("pg").Obj()),
				},
				earliestStartTime: now,
				keyType:           fwk.CompositePodGroupKeyType,
			},
			vi2: &victim{
				priority: 10,
				pods: []fwk.PodInfo{
					newPodInfo(st.MakePod().PodGroupName("pg").Obj()),
				},
				earliestStartTime: before,
				keyType:           fwk.CompositePodGroupKeyType,
			},
			want: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := MoreImportantVictim(tt.vi1, tt.vi2)
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
		name                    string
		pod                     *v1.Pod
		podGroupLister          fwk.PodGroupLister
		compositePodGroupLister fwk.CompositePodGroupLister
		expectedPriority        int32
	}{
		{
			name:                    "pod without PodGroup returns pod priority",
			pod:                     st.MakePod().Name("p1").Priority(10).Obj(),
			podGroupLister:          nil,
			compositePodGroupLister: nil,
			expectedPriority:        10,
		},
		{
			name: "pod with PodGroup returns PodGroup priority instead of pod priority",
			pod:  st.MakePod().Name("p2").Priority(10).PodGroupName("pg1").Obj(),
			podGroupLister: &mockPodGroupLister{
				podGroups: map[string]*schedulingv1beta1.PodGroup{
					"pg1": st.MakePodGroup().Name("pg1").Priority(50).Obj(),
				},
			},
			compositePodGroupLister: nil,
			expectedPriority:        50,
		},
		{
			name:                    "pod with PodGroup but nil PodGroupLister falls back to pod priority",
			pod:                     st.MakePod().Name("p4").Priority(20).PodGroupName("pg1").Obj(),
			podGroupLister:          nil,
			compositePodGroupLister: nil,
			expectedPriority:        20,
		},
		{
			name:                    "pod with PodGroup but PodGroup not found in lister falls back to pod priority",
			pod:                     st.MakePod().Name("p5").Priority(30).PodGroupName("missing-pg").Obj(),
			podGroupLister:          &mockPodGroupLister{podGroups: map[string]*schedulingv1beta1.PodGroup{}},
			compositePodGroupLister: nil,
			expectedPriority:        30,
		},
		{
			name:                    "pod with nil priority and without PodGroup returns zero priority",
			pod:                     st.MakePod().Name("p6").Obj(),
			podGroupLister:          nil,
			compositePodGroupLister: nil,
			expectedPriority:        0,
		},
		{
			name: "pod with PodGroup and parent CPG, returns CPG priority",
			pod:  st.MakePod().Name("p7").Priority(10).PodGroupName("pg1").Obj(),
			podGroupLister: &mockPodGroupLister{
				podGroups: map[string]*schedulingv1beta1.PodGroup{
					"pg1": {
						ObjectMeta: metav1.ObjectMeta{Name: "pg1"},
						Spec: schedulingv1beta1.PodGroupSpec{
							ParentCompositePodGroupName: new("cpg1"),
						},
					},
				},
			},
			compositePodGroupLister: &mockCompositePodGroupLister{
				compositePodGroups: map[string]*schedulingv1alpha3.CompositePodGroup{
					"cpg1": {
						ObjectMeta: metav1.ObjectMeta{Name: "cpg1"},
						Spec: schedulingv1alpha3.CompositePodGroupSpec{
							Priority: new(int32(150)),
						},
					},
				},
			},
			expectedPriority: 150,
		},
		{
			name: "pod with PodGroup and grandparent CPG, returns grandparent CPG priority",
			pod:  st.MakePod().Name("p8").Priority(10).PodGroupName("pg1").Obj(),
			podGroupLister: &mockPodGroupLister{
				podGroups: map[string]*schedulingv1beta1.PodGroup{
					"pg1": {
						ObjectMeta: metav1.ObjectMeta{Name: "pg1"},
						Spec: schedulingv1beta1.PodGroupSpec{
							ParentCompositePodGroupName: new("cpg1"),
						},
					},
				},
			},
			compositePodGroupLister: &mockCompositePodGroupLister{
				compositePodGroups: map[string]*schedulingv1alpha3.CompositePodGroup{
					"cpg1": {
						ObjectMeta: metav1.ObjectMeta{Name: "cpg1"},
						Spec: schedulingv1alpha3.CompositePodGroupSpec{
							ParentCompositePodGroupName: new("cpg2"),
						},
					},
					"cpg2": {
						ObjectMeta: metav1.ObjectMeta{Name: "cpg2"},
						Spec: schedulingv1alpha3.CompositePodGroupSpec{
							Priority: new(int32(250)),
						},
					},
				},
			},
			expectedPriority: 250,
		},
		{
			name: "pod with PodGroup and parent CPG, but nil compositePodGroupLister, falls back to PodGroup priority",
			pod:  st.MakePod().Name("p9").Priority(10).PodGroupName("pg1").Obj(),
			podGroupLister: &mockPodGroupLister{
				podGroups: map[string]*schedulingv1beta1.PodGroup{
					"pg1": st.MakePodGroup().Name("pg1").Priority(50).ParentCompositePodGroup("cpg1").Obj(),
				},
			},
			compositePodGroupLister: nil,
			expectedPriority:        50,
		},
		{
			name: "pod with PodGroup and parent CPG, but parent CPG not found in lister, falls back to PodGroup priority",
			pod:  st.MakePod().Name("p10").Priority(10).PodGroupName("pg1").Obj(),
			podGroupLister: &mockPodGroupLister{
				podGroups: map[string]*schedulingv1beta1.PodGroup{
					"pg1": st.MakePodGroup().Name("pg1").Priority(50).ParentCompositePodGroup("cpg1").Obj(),
				},
			},
			compositePodGroupLister: &mockCompositePodGroupLister{
				compositePodGroups: map[string]*schedulingv1alpha3.CompositePodGroup{},
			},
			expectedPriority: 50,
		},
		{
			name: "pod with PodGroup, parent CPG and grandparent CPG, but grandparent CPG not found in lister, falls back to parent CPG priority",
			pod:  st.MakePod().Name("p11").Priority(10).PodGroupName("pg1").Obj(),
			podGroupLister: &mockPodGroupLister{
				podGroups: map[string]*schedulingv1beta1.PodGroup{
					"pg1": st.MakePodGroup().Name("pg1").Priority(50).ParentCompositePodGroup("cpg1").Obj(),
				},
			},
			compositePodGroupLister: &mockCompositePodGroupLister{
				compositePodGroups: map[string]*schedulingv1alpha3.CompositePodGroup{
					"cpg1": {
						ObjectMeta: metav1.ObjectMeta{Name: "cpg1"},
						Spec: schedulingv1alpha3.CompositePodGroupSpec{
							Priority:                    new(int32(150)),
							ParentCompositePodGroupName: new("missing-cpg2"),
						},
					},
				},
			},
			expectedPriority: 150,
		},
		{
			name: "pod with PodGroup, but PodGroup not found in lister and non-nil compositePodGroupLister, falls back to pod priority",
			pod:  st.MakePod().Name("p12").Priority(50).PodGroupName("missing-pg").Obj(),
			podGroupLister: &mockPodGroupLister{
				podGroups: map[string]*schedulingv1beta1.PodGroup{},
			},
			compositePodGroupLister: &mockCompositePodGroupLister{
				compositePodGroups: map[string]*schedulingv1alpha3.CompositePodGroup{},
			},
			expectedPriority: 50,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := GetPodPriority(tt.pod, tt.podGroupLister, tt.compositePodGroupLister)
			if got != tt.expectedPriority {
				t.Errorf("GetPodPriority() = %v, want %v", got, tt.expectedPriority)
			}
		})
	}
}

func TestTraverseHierarchyUp(t *testing.T) {
	namespace := metav1.NamespaceDefault

	tests := []struct {
		name                    string
		startKey                fwk.EntityKey
		podGroupLister          fwk.PodGroupLister
		compositePodGroupLister fwk.CompositePodGroupLister
		stopAt                  string
		expectedVisitedKeys     []fwk.EntityKey
	}{
		{
			name:                    "nil podGroupLister",
			startKey:                fwk.PodGroupKey(namespace, "pg1"),
			podGroupLister:          nil,
			compositePodGroupLister: nil,
			expectedVisitedKeys:     nil,
		},
		{
			name:     "nil compositePodGroupLister",
			startKey: fwk.PodGroupKey(namespace, "pg1"),
			podGroupLister: &mockPodGroupLister{
				podGroups: map[string]*schedulingv1beta1.PodGroup{
					"pg1": st.MakePodGroup().Name("pg1").Obj(),
				},
			},
			compositePodGroupLister: nil,
			expectedVisitedKeys:     nil,
		},
		{
			name:                    "unsupported key type",
			startKey:                fwk.PodKey(namespace, "p1"),
			podGroupLister:          &mockPodGroupLister{podGroups: map[string]*schedulingv1beta1.PodGroup{}},
			compositePodGroupLister: &mockCompositePodGroupLister{compositePodGroups: map[string]*schedulingv1alpha3.CompositePodGroup{}},
			expectedVisitedKeys:     nil,
		},
		{
			name:     "single PG without parent CPG",
			startKey: fwk.PodGroupKey(namespace, "pg1"),
			podGroupLister: &mockPodGroupLister{
				podGroups: map[string]*schedulingv1beta1.PodGroup{
					"pg1": st.MakePodGroup().Namespace(namespace).Name("pg1").Obj(),
				},
			},
			compositePodGroupLister: &mockCompositePodGroupLister{compositePodGroups: map[string]*schedulingv1alpha3.CompositePodGroup{}},
			expectedVisitedKeys: []fwk.EntityKey{
				fwk.PodGroupKey(namespace, "pg1"),
			},
		},
		{
			name:     "PG -> parent CPG -> grandparent CPG",
			startKey: fwk.PodGroupKey(namespace, "pg1"),
			podGroupLister: &mockPodGroupLister{
				podGroups: map[string]*schedulingv1beta1.PodGroup{
					"pg1": st.MakePodGroup().Namespace(namespace).Name("pg1").ParentCompositePodGroup("cpg1").Obj(),
				},
			},
			compositePodGroupLister: &mockCompositePodGroupLister{
				compositePodGroups: map[string]*schedulingv1alpha3.CompositePodGroup{
					"cpg1": {
						ObjectMeta: metav1.ObjectMeta{Namespace: namespace, Name: "cpg1"},
						Spec: schedulingv1alpha3.CompositePodGroupSpec{
							ParentCompositePodGroupName: new("cpg2"),
						},
					},
					"cpg2": {
						ObjectMeta: metav1.ObjectMeta{Namespace: namespace, Name: "cpg2"},
					},
				},
			},
			expectedVisitedKeys: []fwk.EntityKey{
				fwk.PodGroupKey(namespace, "pg1"),
				fwk.CompositePodGroupKey(namespace, "cpg1"),
				fwk.CompositePodGroupKey(namespace, "cpg2"),
			},
		},
		{
			name:           "start directly from CPG",
			startKey:       fwk.CompositePodGroupKey(namespace, "cpg1"),
			podGroupLister: &mockPodGroupLister{podGroups: map[string]*schedulingv1beta1.PodGroup{}},
			compositePodGroupLister: &mockCompositePodGroupLister{
				compositePodGroups: map[string]*schedulingv1alpha3.CompositePodGroup{
					"cpg1": {
						ObjectMeta: metav1.ObjectMeta{Namespace: namespace, Name: "cpg1"},
						Spec: schedulingv1alpha3.CompositePodGroupSpec{
							ParentCompositePodGroupName: new("cpg2"),
						},
					},
					"cpg2": {
						ObjectMeta: metav1.ObjectMeta{Namespace: namespace, Name: "cpg2"},
					},
				},
			},
			expectedVisitedKeys: []fwk.EntityKey{
				fwk.CompositePodGroupKey(namespace, "cpg1"),
				fwk.CompositePodGroupKey(namespace, "cpg2"),
			},
		},
		{
			name:     "early stop via visitFn",
			startKey: fwk.PodGroupKey(namespace, "pg1"),
			podGroupLister: &mockPodGroupLister{
				podGroups: map[string]*schedulingv1beta1.PodGroup{
					"pg1": st.MakePodGroup().Namespace(namespace).Name("pg1").ParentCompositePodGroup("cpg1").Obj(),
				},
			},
			compositePodGroupLister: &mockCompositePodGroupLister{
				compositePodGroups: map[string]*schedulingv1alpha3.CompositePodGroup{
					"cpg1": {
						ObjectMeta: metav1.ObjectMeta{Namespace: namespace, Name: "cpg1"},
						Spec: schedulingv1alpha3.CompositePodGroupSpec{
							ParentCompositePodGroupName: new("cpg2"),
						},
					},
					"cpg2": {
						ObjectMeta: metav1.ObjectMeta{Namespace: namespace, Name: "cpg2"},
					},
				},
			},
			stopAt: "cpg1",
			expectedVisitedKeys: []fwk.EntityKey{
				fwk.PodGroupKey(namespace, "pg1"),
				fwk.CompositePodGroupKey(namespace, "cpg1"),
			},
		},
		{
			name:     "cycle detection between CPGs",
			startKey: fwk.PodGroupKey(namespace, "pg1"),
			podGroupLister: &mockPodGroupLister{
				podGroups: map[string]*schedulingv1beta1.PodGroup{
					"pg1": st.MakePodGroup().Namespace(namespace).Name("pg1").ParentCompositePodGroup("cpg1").Obj(),
				},
			},
			compositePodGroupLister: &mockCompositePodGroupLister{
				compositePodGroups: map[string]*schedulingv1alpha3.CompositePodGroup{
					"cpg1": {
						ObjectMeta: metav1.ObjectMeta{Namespace: namespace, Name: "cpg1"},
						Spec: schedulingv1alpha3.CompositePodGroupSpec{
							ParentCompositePodGroupName: new("cpg2"),
						},
					},
					"cpg2": {
						ObjectMeta: metav1.ObjectMeta{Namespace: namespace, Name: "cpg2"},
						Spec: schedulingv1alpha3.CompositePodGroupSpec{
							ParentCompositePodGroupName: new("cpg1"),
						},
					},
				},
			},
			expectedVisitedKeys: []fwk.EntityKey{
				fwk.PodGroupKey(namespace, "pg1"),
				fwk.CompositePodGroupKey(namespace, "cpg1"),
				fwk.CompositePodGroupKey(namespace, "cpg2"),
			},
		},
		{
			name:                    "missing PG in lister",
			startKey:                fwk.PodGroupKey(namespace, "missing-pg"),
			podGroupLister:          &mockPodGroupLister{podGroups: map[string]*schedulingv1beta1.PodGroup{}},
			compositePodGroupLister: &mockCompositePodGroupLister{compositePodGroups: map[string]*schedulingv1alpha3.CompositePodGroup{}},
			expectedVisitedKeys:     nil,
		},
		{
			name:     "missing parent CPG in lister stops gracefully",
			startKey: fwk.PodGroupKey(namespace, "pg1"),
			podGroupLister: &mockPodGroupLister{
				podGroups: map[string]*schedulingv1beta1.PodGroup{
					"pg1": st.MakePodGroup().Namespace(namespace).Name("pg1").ParentCompositePodGroup("missing-cpg").Obj(),
				},
			},
			compositePodGroupLister: &mockCompositePodGroupLister{compositePodGroups: map[string]*schedulingv1alpha3.CompositePodGroup{}},
			expectedVisitedKeys: []fwk.EntityKey{
				fwk.PodGroupKey(namespace, "pg1"),
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var visitedKeys []fwk.EntityKey
			traverseFn := func(key fwk.EntityKey, pg *schedulingv1beta1.PodGroup, cpg *schedulingv1alpha3.CompositePodGroup) bool {
				visitedKeys = append(visitedKeys, key)
				return key.Name == tt.stopAt
			}
			TraverseHierarchyUp(namespace, tt.startKey, tt.podGroupLister, tt.compositePodGroupLister, traverseFn)
			if diff := cmp.Diff(tt.expectedVisitedKeys, visitedKeys); diff != "" {
				t.Errorf("TraverseHierarchyUp() mismatch (-want, +got):\n%s", diff)
			}
		})
	}
}
