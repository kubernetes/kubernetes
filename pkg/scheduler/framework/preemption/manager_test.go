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
	"errors"
	"fmt"
	"maps"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	v1 "k8s.io/api/core/v1"
	policy "k8s.io/api/policy/v1"
	schedulingv1beta1 "k8s.io/api/scheduling/v1beta1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/sets"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/client-go/informers"
	clientsetfake "k8s.io/client-go/kubernetes/fake"
	policylisters "k8s.io/client-go/listers/policy/v1"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/klog/v2/ktesting"
	fwk "k8s.io/kube-scheduler/framework"
	"k8s.io/kubernetes/pkg/features"
	internalcache "k8s.io/kubernetes/pkg/scheduler/backend/cache"
	"k8s.io/kubernetes/pkg/scheduler/framework"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/defaultbinder"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/feature"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/queuesort"
	frameworkruntime "k8s.io/kubernetes/pkg/scheduler/framework/runtime"
	st "k8s.io/kubernetes/pkg/scheduler/testing"
	tf "k8s.io/kubernetes/pkg/scheduler/testing/framework"
)

func newTestPodGroupInfo(pg *schedulingv1beta1.PodGroup, cpg *schedulingv1beta1.CompositePodGroup, pods []*v1.Pod) fwk.PodGroupInfo {
	gpg := &fwk.GenericPodGroup{
		PodGroup:          pg,
		CompositePodGroup: cpg,
	}
	if pg != nil {
		return &framework.PodGroupInfo{
			GenericPodGroup: gpg,
			UnscheduledPods: pods,
		}
	}
	if cpg != nil && len(pods) > 0 {
		return framework.NewPodGroupInfo(gpg, &framework.PodGroupInfo{
			GenericPodGroup: fwk.NewGenericPodGroup(st.MakePodGroup().Name("child-pg").Obj()),
			UnscheduledPods: pods,
		})
	}
	return framework.NewPodGroupInfo(gpg)
}

type wantVictim struct {
	pods          sets.Set[string]
	pdbViolations int
}

func wantVic(pods ...string) wantVictim {
	return wantVictim{pods: sets.New(pods...)}
}

func wantViolatingVic(pdbViolations int, pods ...string) wantVictim {
	return wantVictim{pods: sets.New(pods...), pdbViolations: pdbViolations}
}

type errSharedLister struct {
	fwk.SharedLister
}

func (e *errSharedLister) NodeInfos() fwk.NodeInfoLister {
	return &errNodeInfoLister{}
}

type errNodeInfoLister struct {
	fwk.NodeInfoLister
}

func (e *errNodeInfoLister) List() ([]fwk.NodeInfo, error) {
	return nil, errors.New("snapshot list error")
}

type errPDBLister struct {
	policylisters.PodDisruptionBudgetLister
}

func (e *errPDBLister) List(labels.Selector) ([]*policy.PodDisruptionBudget, error) {
	return nil, errors.New("pdb list error")
}

func TestDefaultPreemptionManager_GenerateVictims(t *testing.T) {
	nodes := []*v1.Node{
		st.MakeNode().Name("node1").Capacity(veryLargeRes).Obj(),
		st.MakeNode().Name("node2").Capacity(veryLargeRes).Obj(),
	}
	appLabels := map[string]string{"app": "protected"}

	makePod := func(name string, priority int32) *v1.Pod {
		return st.MakePod().Name(name).UID(name).Namespace("default").Node("node1").Priority(priority).Obj()
	}
	makeGroupPod := func(name string, priority int32, pgName string) *v1.Pod {
		return st.MakePod().Name(name).UID(name).Namespace("default").Node("node1").Priority(priority).PodGroupName(pgName).Obj()
	}
	makeGroupPodOnNode := func(name string, priority int32, pgName, nodeName string) *v1.Pod {
		return st.MakePod().Name(name).UID(name).Namespace("default").Node(nodeName).Priority(priority).PodGroupName(pgName).Obj()
	}
	makeLabeledPod := func(name string, priority int32, lbls map[string]string) *v1.Pod {
		return st.MakePod().Name(name).UID(name).Namespace("default").Node("node1").Priority(priority).Labels(lbls).Obj()
	}
	makeLabeledGroupPod := func(name string, priority int32, pgName string, lbls map[string]string) *v1.Pod {
		return st.MakePod().Name(name).UID(name).Namespace("default").Node("node1").Priority(priority).PodGroupName(pgName).Labels(lbls).Obj()
	}
	makePG := func(name string, priority int32, disruptionAll bool, parentCPG string) *schedulingv1beta1.PodGroup {
		pg := st.MakePodGroup().Name(name).Namespace("default").UID(types.UID(name)).Priority(priority)
		if disruptionAll {
			pg = pg.DisruptionModeAll()
		} else {
			pg = pg.DisruptionModeSingle()
		}
		if parentCPG != "" {
			pg = pg.ParentCompositePodGroup(parentCPG)
		}
		return pg.Obj()
	}
	makeCPG := func(name string, priority int32, disruptionAll bool) *schedulingv1beta1.CompositePodGroup {
		cpg := st.MakeCompositePodGroup().Name(name).Namespace("default").UID(name).Priority(priority)
		if disruptionAll {
			cpg = cpg.DisruptionModeAll()
		} else {
			cpg = cpg.DisruptionModeSingle()
		}
		return cpg.Obj()
	}
	makePDB := func(name string, disruptionsAllowed int32, selector map[string]string) *policy.PodDisruptionBudget {
		return &policy.PodDisruptionBudget{
			ObjectMeta: metav1.ObjectMeta{Name: name, Namespace: "default"},
			Spec: policy.PodDisruptionBudgetSpec{
				Selector: &metav1.LabelSelector{MatchLabels: selector},
			},
			Status: policy.PodDisruptionBudgetStatus{
				DisruptionsAllowed: disruptionsAllowed,
			},
		}
	}
	makePreemptor := func(priority int32) fwk.PodGroupInfo {
		return newTestPodGroupInfo(
			st.MakePodGroup().Name("preemptor-pg").Namespace("default").Priority(priority).Obj(),
			nil,
			[]*v1.Pod{makePod("preemptor-pod", priority)},
		)
	}

	tests := []struct {
		name                   string
		featureGates           featuregatetesting.FeatureOverrides
		initPods               []*v1.Pod
		initPodGroups          []*schedulingv1beta1.PodGroup
		initCompositePodGroups []*schedulingv1beta1.CompositePodGroup
		pdbs                   []*policy.PodDisruptionBudget
		preemptor              fwk.PodGroupInfo
		injectSnapshotErr      bool
		injectPDBListerErr     bool
		wantVictims            []wantVictim
		wantStatusCode         fwk.Code
	}{
		{
			name: "grouping: standalone pods and DisruptionModeSingle pod groups form individual victims",
			initPods: []*v1.Pod{
				makePod("standalone", lowPriority),
				makeGroupPod("single-1", lowPriority, "pg-single"),
				makeGroupPod("single-2", lowPriority, "pg-single"),
			},
			initPodGroups: []*schedulingv1beta1.PodGroup{
				makePG("pg-single", lowPriority, false, ""),
			},
			preemptor: makePreemptor(highPriority),
			wantVictims: []wantVictim{
				wantVic("standalone"),
				wantVic("single-2"),
				wantVic("single-1"),
			},
		},
		{
			name: "grouping: PodGroup with DisruptionModeAll groups all pods across nodes into one atomic victim",
			initPods: []*v1.Pod{
				makeGroupPodOnNode("all-1", lowPriority, "pg-all", "node1"),
				makeGroupPodOnNode("all-2", lowPriority, "pg-all", "node2"),
			},
			initPodGroups: []*schedulingv1beta1.PodGroup{
				makePG("pg-all", lowPriority, true, ""),
			},
			preemptor: makePreemptor(highPriority),
			wantVictims: []wantVictim{
				wantVic("all-1", "all-2"),
			},
		},
		{
			name: "grouping: CompositePodGroup with DisruptionModeAll groups all leaf PodGroup pods into one atomic victim",
			initPods: []*v1.Pod{
				makeGroupPod("child1-p1", lowPriority, "pg-child1"),
				makeGroupPod("child2-p1", lowPriority, "pg-child2"),
			},
			initPodGroups: []*schedulingv1beta1.PodGroup{
				makePG("pg-child1", lowPriority, false, "cpg-all"),
				makePG("pg-child2", lowPriority, false, "cpg-all"),
			},
			initCompositePodGroups: []*schedulingv1beta1.CompositePodGroup{
				makeCPG("cpg-all", lowPriority, true),
			},
			preemptor: makePreemptor(highPriority),
			wantVictims: []wantVictim{
				wantVic("child1-p1", "child2-p1"),
			},
		},
		{
			name: "grouping: CompositePodGroup with DisruptionModeSingle delegates grouping to each child PodGroup",
			initPods: []*v1.Pod{
				makeGroupPod("all-1", lowPriority, "pg-all"),
				makeGroupPod("all-2", lowPriority, "pg-all"),
				makeGroupPod("single-1", lowPriority, "pg-single"),
			},
			initPodGroups: []*schedulingv1beta1.PodGroup{
				makePG("pg-all", lowPriority, true, "cpg-single"),
				makePG("pg-single", lowPriority, false, "cpg-single"),
			},
			initCompositePodGroups: []*schedulingv1beta1.CompositePodGroup{
				makeCPG("cpg-single", lowPriority, false),
			},
			preemptor: makePreemptor(highPriority),
			wantVictims: []wantVictim{
				wantVic("single-1"),
				wantVic("all-1", "all-2"),
			},
		},
		{
			name: "ordering by priority: filters equal or higher priority victims and orders remaining lowest to highest",
			initPods: []*v1.Pod{
				makePod("low-pod", lowPriority),
				makePod("mid-pod", midPriority),
				makePod("high-pod", highPriority),
			},
			preemptor: makePreemptor(highPriority),
			wantVictims: []wantVictim{
				wantVic("low-pod"),
				wantVic("mid-pod"),
			},
		},
		{
			name: "ordering by priority: pod group members are ordered by their own priority, which matches their group's",
			initPods: []*v1.Pod{
				makeGroupPod("mid-pod-in-mid-pg", midPriority, "pg-mid"),
				makePod("standalone-low", lowPriority),
			},
			initPodGroups: []*schedulingv1beta1.PodGroup{
				makePG("pg-mid", midPriority, false, ""),
			},
			preemptor: makePreemptor(highPriority),
			wantVictims: []wantVictim{
				wantVic("standalone-low"),
				wantVic("mid-pod-in-mid-pg"),
			},
		},
		{
			name: "ordering by PDB: non-violating victims are ordered before violating victims regardless of priority",
			initPods: []*v1.Pod{
				makePod("mid-unprotected", midPriority),
				makeLabeledPod("low-protected", lowPriority, appLabels),
			},
			pdbs: []*policy.PodDisruptionBudget{
				makePDB("pdb-zero", 0, appLabels),
			},
			preemptor: makePreemptor(highPriority),
			wantVictims: []wantVictim{
				wantVic("mid-unprotected"),
				wantViolatingVic(1, "low-protected"),
			},
		},
		{
			name: "ordering by PDB: budget consumption across victims marks excess victims as violating",
			initPods: []*v1.Pod{
				makeLabeledPod("mid-protected", midPriority, appLabels),
				makeLabeledPod("low-protected", lowPriority, appLabels),
			},
			pdbs: []*policy.PodDisruptionBudget{
				makePDB("pdb-one", 1, appLabels),
			},
			preemptor: makePreemptor(highPriority),
			wantVictims: []wantVictim{
				wantVic("mid-protected"),
				wantViolatingVic(1, "low-protected"),
			},
		},
		{
			name: "ordering by PDB: atomic group victim is violating if any member pod violates PDB",
			initPods: []*v1.Pod{
				makeLabeledGroupPod("all-protected", lowPriority, "pg-all", appLabels),
				makeGroupPod("all-unprotected", lowPriority, "pg-all"),
				makePod("mid-unprotected", midPriority),
			},
			initPodGroups: []*schedulingv1beta1.PodGroup{
				makePG("pg-all", lowPriority, true, ""),
			},
			pdbs: []*policy.PodDisruptionBudget{
				makePDB("pdb-zero", 0, appLabels),
			},
			preemptor: makePreemptor(highPriority),
			wantVictims: []wantVictim{
				wantVic("mid-unprotected"),
				wantViolatingVic(1, "all-protected", "all-unprotected"),
			},
		},
		{
			name: "feature gates: CompositePodGroup disabled ignores parent CompositePodGroup disruption mode",
			featureGates: featuregatetesting.FeatureOverrides{
				features.CompositePodGroup: false,
			},
			initPods: []*v1.Pod{
				makeGroupPod("p1", lowPriority, "pg-child"),
				makeGroupPod("p2", lowPriority, "pg-child"),
			},
			initPodGroups: []*schedulingv1beta1.PodGroup{
				// pg-child has DisruptionModeSingle, so its pods are individual victims.
				makePG("pg-child", lowPriority, false, "cpg-root"),
			},
			initCompositePodGroups: []*schedulingv1beta1.CompositePodGroup{
				// If enabled, cpg-root's DisruptionModeAll would group p1/p2 into one atomic victim.
				makeCPG("cpg-root", lowPriority, true),
			},
			preemptor: makePreemptor(highPriority),
			wantVictims: []wantVictim{
				wantVic("p2"),
				wantVic("p1"),
			},
		},
		{
			name: "non-successful status: preemptor PodGroup with PreemptNever returns Unschedulable",
			preemptor: newTestPodGroupInfo(
				st.MakePodGroup().Name("pg-never").Priority(highPriority).PreemptionPolicy(schedulingv1beta1.PreemptNever).Obj(),
				nil,
				[]*v1.Pod{makePod("preemptor-pod", highPriority)},
			),
			wantStatusCode: fwk.Unschedulable,
		},
		{
			name: "non-successful status: preemptor CompositePodGroup with PreemptNever returns Unschedulable",
			preemptor: newTestPodGroupInfo(
				nil,
				st.MakeCompositePodGroup().Name("cpg-never").Priority(highPriority).PreemptionPolicy(schedulingv1beta1.PreemptNever).Obj(),
				[]*v1.Pod{makePod("preemptor-pod", highPriority)},
			),
			wantStatusCode: fwk.Unschedulable,
		},
		{
			name: "non-successful status: preemptor pod with PreemptNever and unset PodGroup PreemptionPolicy returns Unschedulable",
			preemptor: newTestPodGroupInfo(
				st.MakePodGroup().Name("pg").Priority(highPriority).Obj(),
				nil,
				[]*v1.Pod{st.MakePod().Name("preemptor-pod").Priority(highPriority).PreemptionPolicy(v1.PreemptNever).Obj()},
			),
			wantStatusCode: fwk.Unschedulable,
		},
		{
			name:              "non-successful status: snapshot error returns Error",
			injectSnapshotErr: true,
			preemptor:         makePreemptor(highPriority),
			wantStatusCode:    fwk.Error,
		},
		{
			name:               "non-successful status: PDB lister error returns Error",
			injectPDBListerErr: true,
			preemptor:          makePreemptor(highPriority),
			wantStatusCode:     fwk.Error,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			overrides := featuregatetesting.FeatureOverrides{
				features.GenericWorkload:                 true,
				features.TopologyAwareWorkloadScheduling: true,
				features.CompositePodGroup:               true,
			}
			maps.Copy(overrides, tt.featureGates)
			featuregatetesting.SetFeatureGatesDuringTest(t, utilfeature.DefaultFeatureGate, overrides)

			_, ctx := ktesting.NewTestContext(t)
			var clientObjs []runtime.Object
			for _, pdb := range tt.pdbs {
				clientObjs = append(clientObjs, pdb)
			}
			client := clientsetfake.NewSimpleClientset(clientObjs...)
			informerFactory := informers.NewSharedInformerFactory(client, 0)
			for _, pdb := range tt.pdbs {
				_ = informerFactory.Policy().V1().PodDisruptionBudgets().Informer().GetStore().Add(pdb)
			}

			snapshot := internalcache.NewTestSnapshotWithPodGroups(tt.initPods, nodes, tt.initPodGroups, tt.initCompositePodGroups)
			var sharedLister fwk.SharedLister = snapshot
			if tt.injectSnapshotErr {
				sharedLister = &errSharedLister{SharedLister: snapshot}
			}

			registeredPlugins := []tf.RegisterPluginFunc{
				tf.RegisterQueueSortPlugin(queuesort.Name, queuesort.New),
				tf.RegisterBindPlugin(defaultbinder.Name, defaultbinder.New),
			}

			fh, err := tf.NewFramework(
				ctx,
				registeredPlugins,
				"",
				frameworkruntime.WithInformerFactory(informerFactory),
				frameworkruntime.WithSnapshotSharedLister(sharedLister),
			)
			if err != nil {
				t.Fatalf("failed to create framework handle: %v", err)
			}

			fts := feature.NewSchedulerFeaturesFromGates(utilfeature.DefaultFeatureGate)
			mgr := NewPreemptionManager(fh, fts).(*defaultPreemptionManager)
			if tt.injectPDBListerErr {
				mgr.pdbLister = &errPDBLister{}
			}

			victims, status := mgr.GenerateVictims(ctx, tt.preemptor)
			if status.Code() != tt.wantStatusCode {
				t.Fatalf("GenerateVictims() status code = %v, want %v (msg: %s)", status.Code(), tt.wantStatusCode, status.Message())
			}
			if tt.wantStatusCode != fwk.Success {
				return
			}

			var gotVictims []wantVictim
			for _, v := range victims {
				podNames := sets.New[string]()
				for _, p := range v.Pods() {
					podNames.Insert(p.GetPod().Name)
				}
				gotVictims = append(gotVictims, wantVictim{
					pods:          podNames,
					pdbViolations: v.NumPDBViolations(),
				})
			}

			if diff := cmp.Diff(tt.wantVictims, gotVictims, cmp.AllowUnexported(wantVictim{})); diff != "" {
				t.Errorf("GenerateVictims() mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

func TestGetPreemptionPolicy(t *testing.T) {
	preemptNeverPod := st.MakePod().Name("p-never").PreemptionPolicy(v1.PreemptNever).Obj()
	preemptLowerPriorityPod := st.MakePod().Name("p-lower").PreemptionPolicy(v1.PreemptLowerPriority).Obj()
	noPolicyPod := st.MakePod().Name("p-nil").Obj()

	tests := []struct {
		name       string
		pg         *schedulingv1beta1.PodGroup
		cpg        *schedulingv1beta1.CompositePodGroup
		pods       []*v1.Pod
		wantPolicy v1.PreemptionPolicy
	}{
		{
			name:       "PreemptionPolicy PreemptNever is resolved from PodGroup, ignoring policy in pod",
			pg:         st.MakePodGroup().Name("pg").PreemptionPolicy(schedulingv1beta1.PreemptNever).Obj(),
			pods:       []*v1.Pod{preemptLowerPriorityPod},
			wantPolicy: v1.PreemptNever,
		},
		{
			name:       "PreemptionPolicy PreemptLowerPriority is resolved from PodGroup, ignoring different policies in pods",
			pg:         st.MakePodGroup().Name("pg").PreemptionPolicy(schedulingv1beta1.PreemptLowerPriority).Obj(),
			pods:       []*v1.Pod{preemptLowerPriorityPod, preemptNeverPod},
			wantPolicy: v1.PreemptLowerPriority,
		},
		{
			name:       "PreemptionPolicy is resolved from pods when PodGroup has nil PreemptionPolicy: PreemptLowerPriority when no pod is PreemptNever",
			pg:         st.MakePodGroup().Name("pg").Obj(),
			pods:       []*v1.Pod{preemptLowerPriorityPod, noPolicyPod},
			wantPolicy: v1.PreemptLowerPriority,
		},
		{
			name:       "PreemptionPolicy is resolved from pods when PodGroup has nil PreemptionPolicy: PreemptNever when any pod is PreemptNever",
			pg:         st.MakePodGroup().Name("pg").Obj(),
			pods:       []*v1.Pod{preemptNeverPod, preemptLowerPriorityPod, noPolicyPod},
			wantPolicy: v1.PreemptNever,
		},
		{
			name:       "PreemptionPolicy PreemptNever is resolved from CompositePodGroup, ignoring policy in pod",
			cpg:        st.MakeCompositePodGroup().Name("cpg1").PreemptionPolicy(schedulingv1beta1.PreemptNever).Obj(),
			pods:       []*v1.Pod{preemptLowerPriorityPod},
			wantPolicy: v1.PreemptNever,
		},
		{
			name:       "PreemptionPolicy PreemptLowerPriority is resolved from CompositePodGroup, ignoring different policies in pods",
			cpg:        st.MakeCompositePodGroup().Name("cpg1").PreemptionPolicy(schedulingv1beta1.PreemptLowerPriority).Obj(),
			pods:       []*v1.Pod{preemptLowerPriorityPod, preemptNeverPod},
			wantPolicy: v1.PreemptLowerPriority,
		},
		{
			name:       "PreemptionPolicy is resolved from pods when CompositePodGroup has nil PreemptionPolicy: PreemptLowerPriority when no pod is PreemptNever",
			cpg:        st.MakeCompositePodGroup().Name("cpg1").Obj(),
			pods:       []*v1.Pod{preemptLowerPriorityPod, noPolicyPod},
			wantPolicy: v1.PreemptLowerPriority,
		},
		{
			name:       "PreemptionPolicy is resolved from pods when CompositePodGroup has nil PreemptionPolicy: PreemptNever when any pod is PreemptNever",
			cpg:        st.MakeCompositePodGroup().Name("cpg1").Obj(),
			pods:       []*v1.Pod{preemptNeverPod, preemptLowerPriorityPod},
			wantPolicy: v1.PreemptNever,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			preemptionPolicy := getPreemptionPolicy(newTestPodGroupInfo(tt.pg, tt.cpg, tt.pods))
			if preemptionPolicy != tt.wantPolicy {
				t.Errorf("expected preemption policy %q, got %q", tt.wantPolicy, preemptionPolicy)
			}
		})
	}
}

func TestCompareVictimSets(t *testing.T) {
	now := time.Now()
	makeVictim := func(name string, priority int32, keyType fwk.EntityKeyType, podCount int, startOffset time.Duration, pdbViolations int) fwk.PreemptionVictim {
		var pods []fwk.PodInfo
		for i := range podCount {
			podName := fmt.Sprintf("%s-%d", name, i)
			pod := st.MakePod().Name(podName).UID(podName).Priority(priority).StartTime(metav1.NewTime(now.Add(startOffset))).Node("node1").Obj()
			pi, _ := framework.NewPodInfo(pod)
			pods = append(pods, pi)
		}
		v, err := NewVictim(pods, priority, keyType)
		if err != nil {
			t.Fatalf("NewVictim failed: %v", err)
		}
		return &victimWithPDBViolations{
			Victim:           v,
			numPDBViolations: pdbViolations,
		}
	}

	veryLowPrioPod := makeVictim("very-low-pod", 1, fwk.PodKeyType, 1, -10*time.Minute, 0)
	veryLowPrioPDBViolatingPod := makeVictim("very-low-pod-pdb", 1, fwk.PodKeyType, 1, -10*time.Minute, 1)
	veryLowPrioPDBViolatingPodGroupSize1 := makeVictim("very-low-pg-1-pdb", 1, fwk.PodGroupKeyType, 1, -10*time.Minute, 1)
	veryLowPrioPodGroupSize3 := makeVictim("very-low-pg-3", 1, fwk.PodGroupKeyType, 3, -10*time.Minute, 0)
	lowPrioPod := makeVictim("low-pod", 10, fwk.PodKeyType, 1, -10*time.Minute, 0)
	lowPrioPDBViolatingPod := makeVictim("low-pod-pdb", 10, fwk.PodKeyType, 1, -10*time.Minute, 1)
	midLowPrioPod := makeVictim("mid-low-pod", 9, fwk.PodKeyType, 1, -10*time.Minute, 0)
	midLowPrioPDBViolatingPod := makeVictim("mid-low-pod-pdb", 9, fwk.PodKeyType, 1, -10*time.Minute, 1)
	midHighPrioPod := makeVictim("mid-high-pod", 19, fwk.PodKeyType, 1, -10*time.Minute, 0)
	highPrioPod := makeVictim("high-pod", 20, fwk.PodKeyType, 1, -10*time.Minute, 0)
	highPrioPodGroupSize1 := makeVictim("high-pg-1", 20, fwk.PodGroupKeyType, 1, -10*time.Minute, 0)
	highPrioPodGroupSize2 := makeVictim("high-pg-2", 20, fwk.PodGroupKeyType, 2, -10*time.Minute, 0)
	lowPrioPodGroupSize1 := makeVictim("pg-1", 10, fwk.PodGroupKeyType, 1, -10*time.Minute, 0)
	lowPrioPodGroupSize2 := makeVictim("pg-2", 10, fwk.PodGroupKeyType, 2, -10*time.Minute, 0)
	lowPrioPodGroupSize3 := makeVictim("pg-3", 10, fwk.PodGroupKeyType, 3, -10*time.Minute, 0)
	lowPrioPodGroupSize4 := makeVictim("pg-4", 10, fwk.PodGroupKeyType, 4, -10*time.Minute, 0)
	youngerLowPrioPod := makeVictim("young-low-pod", 10, fwk.PodKeyType, 1, -1*time.Minute, 0)

	tests := []struct {
		name      string
		set1      []fwk.PreemptionVictim
		set2      []fwk.PreemptionVictim
		wantOrder int // -1 if set1 is strictly better to preempt than set2, 1 if worse, 0 if equal
	}{
		{
			name:      "empty set strictly preferred over non-empty set",
			set1:      nil,
			set2:      []fwk.PreemptionVictim{lowPrioPod},
			wantOrder: -1,
		},
		{
			name:      "fewer PDB violations strictly preferred",
			set1:      []fwk.PreemptionVictim{highPrioPod},
			set2:      []fwk.PreemptionVictim{lowPrioPDBViolatingPod},
			wantOrder: -1,
		},
		{
			name:      "when PDB violations tie, highest priority across both violating and non-violating victims is compared",
			set1:      []fwk.PreemptionVictim{midLowPrioPDBViolatingPod, veryLowPrioPod},
			set2:      []fwk.PreemptionVictim{veryLowPrioPDBViolatingPod, highPrioPod},
			wantOrder: -1,
		},
		{
			name:      "when PDB violations tie, equal-priority bands merge violating and non-violating victims",
			set1:      []fwk.PreemptionVictim{lowPrioPDBViolatingPod, midLowPrioPod, midLowPrioPod},
			set2:      []fwk.PreemptionVictim{lowPrioPDBViolatingPod, lowPrioPod},
			wantOrder: -1,
		},
		{
			name:      "when PDB violations and priority bands tie, structural importance is compared in priority order across runs",
			set1:      []fwk.PreemptionVictim{veryLowPrioPDBViolatingPodGroupSize1, highPrioPod},
			set2:      []fwk.PreemptionVictim{veryLowPrioPDBViolatingPod, highPrioPodGroupSize1},
			wantOrder: -1,
		},
		{
			name:      "lower priority victim preferred over higher priority victim",
			set1:      []fwk.PreemptionVictim{lowPrioPod},
			set2:      []fwk.PreemptionVictim{highPrioPod},
			wantOrder: -1,
		},
		{
			name:      "priorities across the set compared before total pod count",
			set1:      []fwk.PreemptionVictim{highPrioPod, veryLowPrioPodGroupSize3},
			set2:      []fwk.PreemptionVictim{highPrioPod, midHighPrioPod},
			wantOrder: -1,
		},
		{
			name:      "fewer evicted pods at highest priority preferred even with higher secondary priority",
			set1:      []fwk.PreemptionVictim{lowPrioPod, midLowPrioPod},
			set2:      []fwk.PreemptionVictim{lowPrioPodGroupSize3, veryLowPrioPod},
			wantOrder: -1,
		},
		{
			name:      "fewer evicted pods at highest priority preferred over multi-pod group at same priority",
			set1:      []fwk.PreemptionVictim{highPrioPod, midHighPrioPod},
			set2:      []fwk.PreemptionVictim{highPrioPodGroupSize2, veryLowPrioPod},
			wantOrder: -1,
		},
		{
			name:      "fewer evicted pods in a priority band preferred across different victim types",
			set1:      []fwk.PreemptionVictim{lowPrioPodGroupSize2},
			set2:      []fwk.PreemptionVictim{lowPrioPod, lowPrioPod, lowPrioPod},
			wantOrder: -1,
		},
		{
			name:      "one 4-pod PodGroup preferred over three 3-pod PodGroups of same priority",
			set1:      []fwk.PreemptionVictim{lowPrioPodGroupSize4},
			set2:      []fwk.PreemptionVictim{lowPrioPodGroupSize3, lowPrioPodGroupSize3, lowPrioPodGroupSize3},
			wantOrder: -1,
		},
		{
			name:      "set without additional lower-priority band preferred when higher priority bands tie",
			set1:      []fwk.PreemptionVictim{highPrioPod},
			set2:      []fwk.PreemptionVictim{highPrioPod, veryLowPrioPod},
			wantOrder: -1,
		},
		{
			name:      "standalone pod preferred over PodGroup of same priority and total pod count",
			set1:      []fwk.PreemptionVictim{lowPrioPod},
			set2:      []fwk.PreemptionVictim{lowPrioPodGroupSize1},
			wantOrder: -1,
		},
		{
			name:      "smaller PodGroups preferred over larger PodGroup when priority band pod counts tie",
			set1:      []fwk.PreemptionVictim{lowPrioPodGroupSize2, lowPrioPodGroupSize2},
			set2:      []fwk.PreemptionVictim{lowPrioPodGroupSize4},
			wantOrder: -1,
		},
		{
			name:      "equal cost when priority bands and structural importance tie",
			set1:      []fwk.PreemptionVictim{youngerLowPrioPod},
			set2:      []fwk.PreemptionVictim{lowPrioPod},
			wantOrder: 0,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			preemptionManager := &defaultPreemptionManager{}

			got := preemptionManager.CompareVictimSets(t.Context(), tt.set1, tt.set2)
			if got != tt.wantOrder {
				t.Errorf("CompareVictimSets(set1, set2) = %d, want %d", got, tt.wantOrder)
			}
			gotRev := preemptionManager.CompareVictimSets(t.Context(), tt.set2, tt.set1)
			if gotRev != -tt.wantOrder {
				t.Errorf("CompareVictimSets(set2, set1) = %d, want %d", gotRev, -tt.wantOrder)
			}
		})
	}
}
