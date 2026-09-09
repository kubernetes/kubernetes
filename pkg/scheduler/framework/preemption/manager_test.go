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
	"maps"
	"testing"

	"github.com/google/go-cmp/cmp"
	v1 "k8s.io/api/core/v1"
	policy "k8s.io/api/policy/v1"
	schedulingv1alpha3 "k8s.io/api/scheduling/v1alpha3"
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

func newTestPodGroupInfo(pg *schedulingv1beta1.PodGroup, cpg *schedulingv1alpha3.CompositePodGroup, pods []*v1.Pod) fwk.PodGroupInfo {
	pgi := &framework.PodGroupInfo{
		GenericPodGroup: &fwk.GenericPodGroup{
			PodGroup:          pg,
			CompositePodGroup: cpg,
		},
	}
	if pg != nil {
		pgi.UnscheduledPods = pods
	} else if cpg != nil && len(pods) > 0 {
		pgi.Children = []*framework.PodGroupInfo{
			{
				GenericPodGroup: fwk.NewGenericPodGroup(st.MakePodGroup().Name("child-pg").Obj()),
				UnscheduledPods: pods,
			},
		}
	}
	return pgi
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
	makeCPG := func(name string, priority int32, disruptionAll bool) *schedulingv1alpha3.CompositePodGroup {
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
		initCompositePodGroups []*schedulingv1alpha3.CompositePodGroup
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
			initCompositePodGroups: []*schedulingv1alpha3.CompositePodGroup{
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
			initCompositePodGroups: []*schedulingv1alpha3.CompositePodGroup{
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
			name: "ordering by priority: group hierarchy root priority overrides individual pod priority",
			initPods: []*v1.Pod{
				makeGroupPod("low-pod-in-mid-pg", lowPriority, "pg-mid"),
				makePod("standalone-low", lowPriority),
			},
			initPodGroups: []*schedulingv1beta1.PodGroup{
				makePG("pg-mid", midPriority, false, ""),
			},
			preemptor: makePreemptor(highPriority),
			wantVictims: []wantVictim{
				wantVic("standalone-low"),
				wantVic("low-pod-in-mid-pg"),
			},
		},
		{
			name: "ordering by priority: CompositePodGroup root priority overrides child PodGroup and pod priority",
			initPods: []*v1.Pod{
				makeGroupPod("pod-in-cpg", lowPriority, "pg-low"),
				makePod("standalone-low", lowPriority),
			},
			initPodGroups: []*schedulingv1beta1.PodGroup{
				makePG("pg-low", lowPriority, false, "cpg-mid"),
			},
			initCompositePodGroups: []*schedulingv1alpha3.CompositePodGroup{
				makeCPG("cpg-mid", midPriority, false),
			},
			preemptor: makePreemptor(highPriority),
			wantVictims: []wantVictim{
				wantVic("standalone-low"),
				wantVic("pod-in-cpg"),
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
			name: "feature gates: CompositePodGroup disabled ignores parent CompositePodGroup disruption mode and priority",
			featureGates: featuregatetesting.FeatureOverrides{
				features.CompositePodGroup: false,
			},
			initPods: []*v1.Pod{
				makeGroupPod("p1", lowPriority, "pg-child"),
				makeGroupPod("p2", lowPriority, "pg-child"),
			},
			initPodGroups: []*schedulingv1beta1.PodGroup{
				// pg-child has lowPriority (eligible for preemption) and DisruptionModeSingle (individual victims).
				makePG("pg-child", lowPriority, false, "cpg-root"),
			},
			initCompositePodGroups: []*schedulingv1alpha3.CompositePodGroup{
				// If enabled, cpg-root's highPriority would protect p1/p2 from highPriority preemptor,
				// and DisruptionModeAll would group them together.
				makeCPG("cpg-root", highPriority, true),
			},
			preemptor: makePreemptor(highPriority),
			wantVictims: []wantVictim{
				wantVic("p2"),
				wantVic("p1"),
			},
		},
		{
			name: "feature gates: PodGroupPreemptionPolicy disabled ignores PodGroup PreemptNever and uses pod PreemptLowerPriority",
			featureGates: featuregatetesting.FeatureOverrides{
				features.PodGroupPreemptionPolicy: false,
			},
			initPods: []*v1.Pod{
				makePod("low-pod", lowPriority),
			},
			preemptor: newTestPodGroupInfo(
				st.MakePodGroup().Name("pg-never").Priority(highPriority).PreemptionPolicy(schedulingv1beta1.PreemptNever).Obj(),
				nil,
				[]*v1.Pod{st.MakePod().Name("preemptor-pod").Priority(highPriority).PreemptionPolicy(v1.PreemptLowerPriority).Obj()},
			),
			wantVictims: []wantVictim{
				wantVic("low-pod"),
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
				st.MakeCompositePodGroup().Name("cpg-never").Priority(highPriority).PreemptionPolicy(schedulingv1alpha3.PreemptNever).Obj(),
				[]*v1.Pod{makePod("preemptor-pod", highPriority)},
			),
			wantStatusCode: fwk.Unschedulable,
		},
		{
			name: "non-successful status: preemptor pod with PreemptNever and PodGroupPreemptionPolicy disabled returns Unschedulable",
			featureGates: featuregatetesting.FeatureOverrides{
				features.PodGroupPreemptionPolicy: false,
			},
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
				features.PodGroupPreemptionPolicy:        true,
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

			snapshot := internalcache.NewTestSnapshotWithCompositePodGroups(tt.initPods, nodes, tt.initPodGroups, tt.initCompositePodGroups)
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
		name                           string
		pg                             *schedulingv1beta1.PodGroup
		cpg                            *schedulingv1alpha3.CompositePodGroup
		pods                           []*v1.Pod
		enablePodGroupPreemptionPolicy bool
		wantPolicy                     v1.PreemptionPolicy
	}{
		{
			name:                           "PreemptionPolicy PreemptNever is resolved from PodGroup, ignoring policy in pod, with PodGroupPreemptionPolicy enabled",
			pg:                             st.MakePodGroup().Name("pg").PreemptionPolicy(schedulingv1beta1.PreemptNever).Obj(),
			pods:                           []*v1.Pod{preemptLowerPriorityPod},
			enablePodGroupPreemptionPolicy: true,
			wantPolicy:                     v1.PreemptNever,
		},
		{
			name:                           "PreemptionPolicy PreemptLowerPriority is resolved from PodGroup, ignoring different policies in pods, with PodGroupPreemptionPolicy enabled",
			pg:                             st.MakePodGroup().Name("pg").PreemptionPolicy(schedulingv1beta1.PreemptLowerPriority).Obj(),
			pods:                           []*v1.Pod{preemptLowerPriorityPod, noPolicyPod},
			enablePodGroupPreemptionPolicy: true,
			wantPolicy:                     v1.PreemptLowerPriority,
		},
		{
			name:                           "PreemptionPolicy is resolved from pods with PodGroupPreemptionPolicy disabled",
			pg:                             st.MakePodGroup().Name("pg").PreemptionPolicy(schedulingv1beta1.PreemptNever).Obj(),
			pods:                           []*v1.Pod{preemptLowerPriorityPod},
			enablePodGroupPreemptionPolicy: false,
			wantPolicy:                     v1.PreemptLowerPriority,
		},
		{
			name:                           "PreemptionPolicy is resolved from pods when multiple pods have different policies, with PodGroupPreemptionPolicy disabled",
			pg:                             st.MakePodGroup().Name("pg").PreemptionPolicy(schedulingv1beta1.PreemptLowerPriority).Obj(),
			pods:                           []*v1.Pod{preemptNeverPod, preemptLowerPriorityPod, noPolicyPod},
			enablePodGroupPreemptionPolicy: false,
			wantPolicy:                     v1.PreemptNever,
		},
		{
			name:       "PreemptionPolicy is resolved from pods when CompositePodGroup is active: PreemptLowerPriority when no pod is PreemptNever",
			cpg:        st.MakeCompositePodGroup().Name("cpg1").Obj(),
			pods:       []*v1.Pod{preemptLowerPriorityPod, noPolicyPod},
			wantPolicy: v1.PreemptLowerPriority,
		},
		{
			name:       "PreemptionPolicy is resolved from pods when CompositePodGroup is active: PreemptNever when any pod is PreemptNever",
			cpg:        st.MakeCompositePodGroup().Name("cpg1").Obj(),
			pods:       []*v1.Pod{preemptNeverPod, preemptLowerPriorityPod},
			wantPolicy: v1.PreemptNever,
		},
		{
			name:                           "PreemptionPolicy PreemptNever is resolved from CompositePodGroup, ignoring policy in pod, with PodGroupPreemptionPolicy enabled",
			cpg:                            st.MakeCompositePodGroup().Name("cpg1").PreemptionPolicy(schedulingv1alpha3.PreemptNever).Obj(),
			pods:                           []*v1.Pod{preemptLowerPriorityPod},
			enablePodGroupPreemptionPolicy: true,
			wantPolicy:                     v1.PreemptNever,
		},
		{
			name:                           "PreemptionPolicy PreemptLowerPriority is resolved from CompositePodGroup, ignoring different policies in pods, with PodGroupPreemptionPolicy enabled",
			cpg:                            st.MakeCompositePodGroup().Name("cpg1").PreemptionPolicy(schedulingv1alpha3.PreemptLowerPriority).Obj(),
			pods:                           []*v1.Pod{preemptLowerPriorityPod, preemptNeverPod},
			enablePodGroupPreemptionPolicy: true,
			wantPolicy:                     v1.PreemptLowerPriority,
		},
		{
			name:                           "PreemptionPolicy is resolved from pods when CompositePodGroup has policy but PodGroupPreemptionPolicy is disabled",
			cpg:                            st.MakeCompositePodGroup().Name("cpg1").PreemptionPolicy(schedulingv1alpha3.PreemptNever).Obj(),
			pods:                           []*v1.Pod{preemptLowerPriorityPod, noPolicyPod},
			enablePodGroupPreemptionPolicy: false,
			wantPolicy:                     v1.PreemptLowerPriority,
		},
		{
			name:                           "PreemptionPolicy defaults to PreemptLowerPriority when CompositePodGroup has no policy set with PodGroupPreemptionPolicy enabled, even if pods are PreemptNever",
			cpg:                            st.MakeCompositePodGroup().Name("cpg1").Obj(),
			pods:                           []*v1.Pod{preemptNeverPod},
			enablePodGroupPreemptionPolicy: true,
			wantPolicy:                     v1.PreemptLowerPriority,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			preemptionPolicy := getPreemptionPolicy(newTestPodGroupInfo(tt.pg, tt.cpg, tt.pods), tt.enablePodGroupPreemptionPolicy)
			if preemptionPolicy != tt.wantPolicy {
				t.Errorf("expected preemption policy %q, got %q", tt.wantPolicy, preemptionPolicy)
			}
		})
	}
}
