/*
Copyright 2019 The Kubernetes Authors.

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

package volumebinding

import (
	"context"
	"reflect"
	"testing"

	"github.com/google/go-cmp/cmp"
	"github.com/google/go-cmp/cmp/cmpopts"
	v1 "k8s.io/api/core/v1"
	storagev1 "k8s.io/api/storage/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/client-go/informers"
	"k8s.io/client-go/kubernetes/fake"
	"k8s.io/component-base/featuregate"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	pvutil "k8s.io/kubernetes/pkg/controller/volume/persistentvolume/util"
	"k8s.io/kubernetes/pkg/controller/volume/scheduling"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/scheduler/apis/config"
	"k8s.io/kubernetes/pkg/scheduler/framework"
	"k8s.io/kubernetes/pkg/scheduler/framework/runtime"
	"k8s.io/utils/pointer"
)

var (
	immediate            = storagev1.VolumeBindingImmediate
	waitForFirstConsumer = storagev1.VolumeBindingWaitForFirstConsumer
	immediateSC          = &storagev1.StorageClass{
		ObjectMeta: metav1.ObjectMeta{
			Name: "immediate-sc",
		},
		VolumeBindingMode: &immediate,
	}
	waitSC = &storagev1.StorageClass{
		ObjectMeta: metav1.ObjectMeta{
			Name: "wait-sc",
		},
		VolumeBindingMode: &waitForFirstConsumer,
	}
	waitHDDSC = &storagev1.StorageClass{
		ObjectMeta: metav1.ObjectMeta{
			Name: "wait-hdd-sc",
		},
		VolumeBindingMode: &waitForFirstConsumer,
	}

	defaultShapePoint = []config.UtilizationShapePoint{
		{
			Utilization: 0,
			Score:       0,
		},
		{
			Utilization: 100,
			Score:       int32(config.MaxCustomPriorityScore),
		},
	}
)

func makeNode(name string) *v1.Node {
	return &v1.Node{
		ObjectMeta: metav1.ObjectMeta{
			Name: name,
			Labels: map[string]string{
				v1.LabelHostname: name,
			},
		},
	}
}

func mergeNodeLabels(node *v1.Node, labels map[string]string) *v1.Node {
	for k, v := range labels {
		node.Labels[k] = v
	}
	return node
}

func makePV(name string, className string) *v1.PersistentVolume {
	return &v1.PersistentVolume{
		ObjectMeta: metav1.ObjectMeta{
			Name: name,
		},
		Spec: v1.PersistentVolumeSpec{
			StorageClassName: className,
		},
		Status: v1.PersistentVolumeStatus{
			Phase: v1.VolumeAvailable,
		},
	}
}

func setPVNodeAffinity(pv *v1.PersistentVolume, keyValues map[string][]string) *v1.PersistentVolume {
	matchExpressions := make([]v1.NodeSelectorRequirement, 0)
	for key, values := range keyValues {
		matchExpressions = append(matchExpressions, v1.NodeSelectorRequirement{
			Key:      key,
			Operator: v1.NodeSelectorOpIn,
			Values:   values,
		})
	}
	pv.Spec.NodeAffinity = &v1.VolumeNodeAffinity{
		Required: &v1.NodeSelector{
			NodeSelectorTerms: []v1.NodeSelectorTerm{
				{
					MatchExpressions: matchExpressions,
				},
			},
		},
	}
	return pv
}

func setPVCapacity(pv *v1.PersistentVolume, capacity resource.Quantity) *v1.PersistentVolume {
	pv.Spec.Capacity = v1.ResourceList{
		v1.ResourceName(v1.ResourceStorage): capacity,
	}
	return pv
}

func makePVC(name string, boundPVName string, storageClassName string) *v1.PersistentVolumeClaim {
	pvc := &v1.PersistentVolumeClaim{
		ObjectMeta: metav1.ObjectMeta{
			Name:      name,
			Namespace: v1.NamespaceDefault,
		},
		Spec: v1.PersistentVolumeClaimSpec{
			StorageClassName: pointer.StringPtr(storageClassName),
		},
	}
	if boundPVName != "" {
		pvc.Spec.VolumeName = boundPVName
		metav1.SetMetaDataAnnotation(&pvc.ObjectMeta, pvutil.AnnBindCompleted, "true")
	}
	return pvc
}

func setPVCRequestStorage(pvc *v1.PersistentVolumeClaim, request resource.Quantity) *v1.PersistentVolumeClaim {
	pvc.Spec.Resources = v1.ResourceRequirements{
		Requests: v1.ResourceList{
			v1.ResourceName(v1.ResourceStorage): request,
		},
	}
	return pvc
}

func makePod(name string, pvcNames []string) *v1.Pod {
	p := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      name,
			Namespace: v1.NamespaceDefault,
		},
	}
	p.Spec.Volumes = make([]v1.Volume, 0)
	for _, pvcName := range pvcNames {
		p.Spec.Volumes = append(p.Spec.Volumes, v1.Volume{
			VolumeSource: v1.VolumeSource{
				PersistentVolumeClaim: &v1.PersistentVolumeClaimVolumeSource{
					ClaimName: pvcName,
				},
			},
		})
	}
	return p
}

func TestVolumeBinding(t *testing.T) {
	table := []struct {
		name                    string
		pod                     *v1.Pod
		nodes                   []*v1.Node
		pvcs                    []*v1.PersistentVolumeClaim
		pvs                     []*v1.PersistentVolume
		feature                 featuregate.Feature
		args                    *config.VolumeBindingArgs
		wantPreFilterStatus     *framework.Status
		wantStateAfterPreFilter *stateData
		wantFilterStatus        []*framework.Status
		wantScores              []int64
	}{
		{
			name: "pod has not pvcs",
			pod:  makePod("pod-a", nil),
			nodes: []*v1.Node{
				makeNode("node-a"),
			},
			wantStateAfterPreFilter: &stateData{
				skip: true,
			},
			wantFilterStatus: []*framework.Status{
				nil,
			},
			wantScores: []int64{
				0,
			},
		},
		{
			name: "all bound",
			pod:  makePod("pod-a", []string{"pvc-a"}),
			nodes: []*v1.Node{
				makeNode("node-a"),
			},
			pvcs: []*v1.PersistentVolumeClaim{
				makePVC("pvc-a", "pv-a", waitSC.Name),
			},
			pvs: []*v1.PersistentVolume{
				makePV("pv-a", waitSC.Name),
			},
			wantStateAfterPreFilter: &stateData{
				boundClaims: []*v1.PersistentVolumeClaim{
					makePVC("pvc-a", "pv-a", waitSC.Name),
				},
				claimsToBind:     []*v1.PersistentVolumeClaim{},
				podVolumesByNode: map[string]*scheduling.PodVolumes{},
			},
			wantFilterStatus: []*framework.Status{
				nil,
			},
			wantScores: []int64{
				0,
			},
		},
		{
			name: "PVC does not exist",
			pod:  makePod("pod-a", []string{"pvc-a"}),
			nodes: []*v1.Node{
				makeNode("node-a"),
			},
			pvcs:                []*v1.PersistentVolumeClaim{},
			wantPreFilterStatus: framework.NewStatus(framework.UnschedulableAndUnresolvable, `persistentvolumeclaim "pvc-a" not found`),
			wantFilterStatus: []*framework.Status{
				nil,
			},
			wantScores: []int64{
				0,
			},
		},
		{
			name: "Part of PVCs do not exist",
			pod:  makePod("pod-a", []string{"pvc-a", "pvc-b"}),
			nodes: []*v1.Node{
				makeNode("node-a"),
			},
			pvcs: []*v1.PersistentVolumeClaim{
				makePVC("pvc-a", "pv-a", waitSC.Name),
			},
			wantPreFilterStatus: framework.NewStatus(framework.UnschedulableAndUnresolvable, `persistentvolumeclaim "pvc-b" not found`),
			wantFilterStatus: []*framework.Status{
				nil,
			},
			wantScores: []int64{
				0,
			},
		},
		{
			name: "immediate claims not bound",
			pod:  makePod("pod-a", []string{"pvc-a"}),
			nodes: []*v1.Node{
				makeNode("node-a"),
			},
			pvcs: []*v1.PersistentVolumeClaim{
				makePVC("pvc-a", "", immediateSC.Name),
			},
			wantPreFilterStatus: framework.NewStatus(framework.UnschedulableAndUnresolvable, "pod has unbound immediate PersistentVolumeClaims"),
			wantFilterStatus: []*framework.Status{
				nil,
			},
			wantScores: []int64{
				0,
			},
		},
		{
			name: "unbound claims no matches",
			pod:  makePod("pod-a", []string{"pvc-a"}),
			nodes: []*v1.Node{
				makeNode("node-a"),
			},
			pvcs: []*v1.PersistentVolumeClaim{
				makePVC("pvc-a", "", waitSC.Name),
			},
			wantStateAfterPreFilter: &stateData{
				boundClaims: []*v1.PersistentVolumeClaim{},
				claimsToBind: []*v1.PersistentVolumeClaim{
					makePVC("pvc-a", "", waitSC.Name),
				},
				podVolumesByNode: map[string]*scheduling.PodVolumes{},
			},
			wantFilterStatus: []*framework.Status{
				framework.NewStatus(framework.UnschedulableAndUnresolvable, string(scheduling.ErrReasonBindConflict)),
			},
			wantScores: []int64{
				0,
			},
		},
		{
			name: "bound and unbound unsatisfied",
			pod:  makePod("pod-a", []string{"pvc-a", "pvc-b"}),
			nodes: []*v1.Node{
				mergeNodeLabels(makeNode("node-a"), map[string]string{
					"foo": "barbar",
				}),
			},
			pvcs: []*v1.PersistentVolumeClaim{
				makePVC("pvc-a", "pv-a", waitSC.Name),
				makePVC("pvc-b", "", waitSC.Name),
			},
			pvs: []*v1.PersistentVolume{
				setPVNodeAffinity(makePV("pv-a", waitSC.Name), map[string][]string{
					"foo": {"bar"},
				}),
			},
			wantStateAfterPreFilter: &stateData{
				boundClaims: []*v1.PersistentVolumeClaim{
					makePVC("pvc-a", "pv-a", waitSC.Name),
				},
				claimsToBind: []*v1.PersistentVolumeClaim{
					makePVC("pvc-b", "", waitSC.Name),
				},
				podVolumesByNode: map[string]*scheduling.PodVolumes{},
			},
			wantFilterStatus: []*framework.Status{
				framework.NewStatus(framework.UnschedulableAndUnresolvable, string(scheduling.ErrReasonNodeConflict), string(scheduling.ErrReasonBindConflict)),
			},
			wantScores: []int64{
				0,
			},
		},
		{
			name: "pvc not found",
			pod:  makePod("pod-a", []string{"pvc-a"}),
			nodes: []*v1.Node{
				makeNode("node-a"),
			},
			wantPreFilterStatus: framework.NewStatus(framework.UnschedulableAndUnresolvable, `persistentvolumeclaim "pvc-a" not found`),
			wantFilterStatus: []*framework.Status{
				nil,
			},
			wantScores: []int64{
				0,
			},
		},
		{
			name: "pv not found",
			pod:  makePod("pod-a", []string{"pvc-a"}),
			nodes: []*v1.Node{
				makeNode("node-a"),
			},
			pvcs: []*v1.PersistentVolumeClaim{
				makePVC("pvc-a", "pv-a", waitSC.Name),
			},
			wantPreFilterStatus: nil,
			wantStateAfterPreFilter: &stateData{
				boundClaims: []*v1.PersistentVolumeClaim{
					makePVC("pvc-a", "pv-a", waitSC.Name),
				},
				claimsToBind:     []*v1.PersistentVolumeClaim{},
				podVolumesByNode: map[string]*scheduling.PodVolumes{},
			},
			wantFilterStatus: []*framework.Status{
				framework.NewStatus(framework.UnschedulableAndUnresolvable, `pvc(s) bound to non-existent pv(s)`),
			},
			wantScores: []int64{
				0,
			},
		},
		{
			name: "local volumes with close capacity are preferred",
			pod:  makePod("pod-a", []string{"pvc-a"}),
			nodes: []*v1.Node{
				makeNode("node-a"),
				makeNode("node-b"),
				makeNode("node-c"),
			},
			pvcs: []*v1.PersistentVolumeClaim{
				setPVCRequestStorage(makePVC("pvc-a", "", waitSC.Name), resource.MustParse("50Gi")),
			},
			pvs: []*v1.PersistentVolume{
				setPVNodeAffinity(setPVCapacity(makePV("pv-a-0", waitSC.Name), resource.MustParse("200Gi")), map[string][]string{v1.LabelHostname: {"node-a"}}),
				setPVNodeAffinity(setPVCapacity(makePV("pv-a-1", waitSC.Name), resource.MustParse("200Gi")), map[string][]string{v1.LabelHostname: {"node-a"}}),
				setPVNodeAffinity(setPVCapacity(makePV("pv-b-0", waitSC.Name), resource.MustParse("100Gi")), map[string][]string{v1.LabelHostname: {"node-b"}}),
				setPVNodeAffinity(setPVCapacity(makePV("pv-b-1", waitSC.Name), resource.MustParse("100Gi")), map[string][]string{v1.LabelHostname: {"node-b"}}),
			},
			feature:             features.VolumeCapacityPriority,
			wantPreFilterStatus: nil,
			wantStateAfterPreFilter: &stateData{
				boundClaims: []*v1.PersistentVolumeClaim{},
				claimsToBind: []*v1.PersistentVolumeClaim{
					setPVCRequestStorage(makePVC("pvc-a", "", waitSC.Name), resource.MustParse("50Gi")),
				},
				podVolumesByNode: map[string]*scheduling.PodVolumes{},
			},
			wantFilterStatus: []*framework.Status{
				nil,
				nil,
				framework.NewStatus(framework.UnschedulableAndUnresolvable, `node(s) didn't find available persistent volumes to bind`),
			},
			wantScores: []int64{
				25,
				50,
				0,
			},
		},
		{
			name: "local volumes with close capacity are preferred (multiple pvcs)",
			pod:  makePod("pod-a", []string{"pvc-0", "pvc-1"}),
			nodes: []*v1.Node{
				makeNode("node-a"),
				makeNode("node-b"),
				makeNode("node-c"),
			},
			pvcs: []*v1.PersistentVolumeClaim{
				setPVCRequestStorage(makePVC("pvc-0", "", waitSC.Name), resource.MustParse("50Gi")),
				setPVCRequestStorage(makePVC("pvc-1", "", waitHDDSC.Name), resource.MustParse("100Gi")),
			},
			pvs: []*v1.PersistentVolume{
				setPVNodeAffinity(setPVCapacity(makePV("pv-a-0", waitSC.Name), resource.MustParse("200Gi")), map[string][]string{v1.LabelHostname: {"node-a"}}),
				setPVNodeAffinity(setPVCapacity(makePV("pv-a-1", waitSC.Name), resource.MustParse("200Gi")), map[string][]string{v1.LabelHostname: {"node-a"}}),
				setPVNodeAffinity(setPVCapacity(makePV("pv-a-2", waitHDDSC.Name), resource.MustParse("200Gi")), map[string][]string{v1.LabelHostname: {"node-a"}}),
				setPVNodeAffinity(setPVCapacity(makePV("pv-a-3", waitHDDSC.Name), resource.MustParse("200Gi")), map[string][]string{v1.LabelHostname: {"node-a"}}),
				setPVNodeAffinity(setPVCapacity(makePV("pv-b-0", waitSC.Name), resource.MustParse("100Gi")), map[string][]string{v1.LabelHostname: {"node-b"}}),
				setPVNodeAffinity(setPVCapacity(makePV("pv-b-1", waitSC.Name), resource.MustParse("100Gi")), map[string][]string{v1.LabelHostname: {"node-b"}}),
				setPVNodeAffinity(setPVCapacity(makePV("pv-b-2", waitHDDSC.Name), resource.MustParse("100Gi")), map[string][]string{v1.LabelHostname: {"node-b"}}),
				setPVNodeAffinity(setPVCapacity(makePV("pv-b-3", waitHDDSC.Name), resource.MustParse("100Gi")), map[string][]string{v1.LabelHostname: {"node-b"}}),
			},
			feature:             features.VolumeCapacityPriority,
			wantPreFilterStatus: nil,
			wantStateAfterPreFilter: &stateData{
				boundClaims: []*v1.PersistentVolumeClaim{},
				claimsToBind: []*v1.PersistentVolumeClaim{
					setPVCRequestStorage(makePVC("pvc-0", "", waitSC.Name), resource.MustParse("50Gi")),
					setPVCRequestStorage(makePVC("pvc-1", "", waitHDDSC.Name), resource.MustParse("100Gi")),
				},
				podVolumesByNode: map[string]*scheduling.PodVolumes{},
			},
			wantFilterStatus: []*framework.Status{
				nil,
				nil,
				framework.NewStatus(framework.UnschedulableAndUnresolvable, `node(s) didn't find available persistent volumes to bind`),
			},
			wantScores: []int64{
				38,
				75,
				0,
			},
		},
		{
			name: "zonal volumes with close capacity are preferred",
			pod:  makePod("pod-a", []string{"pvc-a"}),
			nodes: []*v1.Node{
				mergeNodeLabels(makeNode("zone-a-node-a"), map[string]string{
					"topology.kubernetes.io/region": "region-a",
					"topology.kubernetes.io/zone":   "zone-a",
				}),
				mergeNodeLabels(makeNode("zone-a-node-b"), map[string]string{
					"topology.kubernetes.io/region": "region-a",
					"topology.kubernetes.io/zone":   "zone-a",
				}),
				mergeNodeLabels(makeNode("zone-b-node-a"), map[string]string{
					"topology.kubernetes.io/region": "region-b",
					"topology.kubernetes.io/zone":   "zone-b",
				}),
				mergeNodeLabels(makeNode("zone-b-node-b"), map[string]string{
					"topology.kubernetes.io/region": "region-b",
					"topology.kubernetes.io/zone":   "zone-b",
				}),
				mergeNodeLabels(makeNode("zone-c-node-a"), map[string]string{
					"topology.kubernetes.io/region": "region-c",
					"topology.kubernetes.io/zone":   "zone-c",
				}),
				mergeNodeLabels(makeNode("zone-c-node-b"), map[string]string{
					"topology.kubernetes.io/region": "region-c",
					"topology.kubernetes.io/zone":   "zone-c",
				}),
			},
			pvcs: []*v1.PersistentVolumeClaim{
				setPVCRequestStorage(makePVC("pvc-a", "", waitSC.Name), resource.MustParse("50Gi")),
			},
			pvs: []*v1.PersistentVolume{
				setPVNodeAffinity(setPVCapacity(makePV("pv-a-0", waitSC.Name), resource.MustParse("200Gi")), map[string][]string{
					"topology.kubernetes.io/region": {"region-a"},
					"topology.kubernetes.io/zone":   {"zone-a"},
				}),
				setPVNodeAffinity(setPVCapacity(makePV("pv-a-1", waitSC.Name), resource.MustParse("200Gi")), map[string][]string{
					"topology.kubernetes.io/region": {"region-a"},
					"topology.kubernetes.io/zone":   {"zone-a"},
				}),
				setPVNodeAffinity(setPVCapacity(makePV("pv-b-0", waitSC.Name), resource.MustParse("100Gi")), map[string][]string{
					"topology.kubernetes.io/region": {"region-b"},
					"topology.kubernetes.io/zone":   {"zone-b"},
				}),
				setPVNodeAffinity(setPVCapacity(makePV("pv-b-1", waitSC.Name), resource.MustParse("100Gi")), map[string][]string{
					"topology.kubernetes.io/region": {"region-b"},
					"topology.kubernetes.io/zone":   {"zone-b"},
				}),
			},
			feature:             features.VolumeCapacityPriority,
			wantPreFilterStatus: nil,
			wantStateAfterPreFilter: &stateData{
				boundClaims: []*v1.PersistentVolumeClaim{},
				claimsToBind: []*v1.PersistentVolumeClaim{
					setPVCRequestStorage(makePVC("pvc-a", "", waitSC.Name), resource.MustParse("50Gi")),
				},
				podVolumesByNode: map[string]*scheduling.PodVolumes{},
			},
			wantFilterStatus: []*framework.Status{
				nil,
				nil,
				nil,
				nil,
				framework.NewStatus(framework.UnschedulableAndUnresolvable, `node(s) didn't find available persistent volumes to bind`),
				framework.NewStatus(framework.UnschedulableAndUnresolvable, `node(s) didn't find available persistent volumes to bind`),
			},
			wantScores: []int64{
				25,
				25,
				50,
				50,
				0,
				0,
			},
		},
		{
			name: "zonal volumes with close capacity are preferred (custom shape)",
			pod:  makePod("pod-a", []string{"pvc-a"}),
			nodes: []*v1.Node{
				mergeNodeLabels(makeNode("zone-a-node-a"), map[string]string{
					"topology.kubernetes.io/region": "region-a",
					"topology.kubernetes.io/zone":   "zone-a",
				}),
				mergeNodeLabels(makeNode("zone-a-node-b"), map[string]string{
					"topology.kubernetes.io/region": "region-a",
					"topology.kubernetes.io/zone":   "zone-a",
				}),
				mergeNodeLabels(makeNode("zone-b-node-a"), map[string]string{
					"topology.kubernetes.io/region": "region-b",
					"topology.kubernetes.io/zone":   "zone-b",
				}),
				mergeNodeLabels(makeNode("zone-b-node-b"), map[string]string{
					"topology.kubernetes.io/region": "region-b",
					"topology.kubernetes.io/zone":   "zone-b",
				}),
				mergeNodeLabels(makeNode("zone-c-node-a"), map[string]string{
					"topology.kubernetes.io/region": "region-c",
					"topology.kubernetes.io/zone":   "zone-c",
				}),
				mergeNodeLabels(makeNode("zone-c-node-b"), map[string]string{
					"topology.kubernetes.io/region": "region-c",
					"topology.kubernetes.io/zone":   "zone-c",
				}),
			},
			pvcs: []*v1.PersistentVolumeClaim{
				setPVCRequestStorage(makePVC("pvc-a", "", waitSC.Name), resource.MustParse("50Gi")),
			},
			pvs: []*v1.PersistentVolume{
				setPVNodeAffinity(setPVCapacity(makePV("pv-a-0", waitSC.Name), resource.MustParse("200Gi")), map[string][]string{
					"topology.kubernetes.io/region": {"region-a"},
					"topology.kubernetes.io/zone":   {"zone-a"},
				}),
				setPVNodeAffinity(setPVCapacity(makePV("pv-a-1", waitSC.Name), resource.MustParse("200Gi")), map[string][]string{
					"topology.kubernetes.io/region": {"region-a"},
					"topology.kubernetes.io/zone":   {"zone-a"},
				}),
				setPVNodeAffinity(setPVCapacity(makePV("pv-b-0", waitSC.Name), resource.MustParse("100Gi")), map[string][]string{
					"topology.kubernetes.io/region": {"region-b"},
					"topology.kubernetes.io/zone":   {"zone-b"},
				}),
				setPVNodeAffinity(setPVCapacity(makePV("pv-b-1", waitSC.Name), resource.MustParse("100Gi")), map[string][]string{
					"topology.kubernetes.io/region": {"region-b"},
					"topology.kubernetes.io/zone":   {"zone-b"},
				}),
			},
			feature: features.VolumeCapacityPriority,
			args: &config.VolumeBindingArgs{
				BindTimeoutSeconds: 300,
				Shape: []config.UtilizationShapePoint{
					{
						Utilization: 0,
						Score:       0,
					},
					{
						Utilization: 50,
						Score:       3,
					},
					{
						Utilization: 100,
						Score:       5,
					},
				},
			},
			wantPreFilterStatus: nil,
			wantStateAfterPreFilter: &stateData{
				boundClaims: []*v1.PersistentVolumeClaim{},
				claimsToBind: []*v1.PersistentVolumeClaim{
					setPVCRequestStorage(makePVC("pvc-a", "", waitSC.Name), resource.MustParse("50Gi")),
				},
				podVolumesByNode: map[string]*scheduling.PodVolumes{},
			},
			wantFilterStatus: []*framework.Status{
				nil,
				nil,
				nil,
				nil,
				framework.NewStatus(framework.UnschedulableAndUnresolvable, `node(s) didn't find available persistent volumes to bind`),
				framework.NewStatus(framework.UnschedulableAndUnresolvable, `node(s) didn't find available persistent volumes to bind`),
			},
			wantScores: []int64{
				15,
				15,
				30,
				30,
				0,
				0,
			},
		},
	}

	for _, item := range table {
		t.Run(item.name, func(t *testing.T) {
			if item.feature != "" {
				defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, item.feature, true)()
			}
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			client := fake.NewSimpleClientset()
			informerFactory := informers.NewSharedInformerFactory(client, 0)
			opts := []runtime.Option{
				runtime.WithClientSet(client),
				runtime.WithInformerFactory(informerFactory),
			}
			fh, err := runtime.NewFramework(nil, nil, opts...)
			if err != nil {
				t.Fatal(err)
			}

			args := item.args
			if args == nil {
				// default args if the args is not specified in test cases
				args = &config.VolumeBindingArgs{
					BindTimeoutSeconds: 300,
				}
				if utilfeature.DefaultFeatureGate.Enabled(features.VolumeCapacityPriority) {
					args.Shape = defaultShapePoint
				}
			}

			pl, err := New(args, fh)
			if err != nil {
				t.Fatal(err)
			}

			t.Log("Feed testing data and wait for them to be synced")
			client.StorageV1().StorageClasses().Create(ctx, immediateSC, metav1.CreateOptions{})
			client.StorageV1().StorageClasses().Create(ctx, waitSC, metav1.CreateOptions{})
			client.StorageV1().StorageClasses().Create(ctx, waitHDDSC, metav1.CreateOptions{})
			for _, node := range item.nodes {
				client.CoreV1().Nodes().Create(ctx, node, metav1.CreateOptions{})
			}
			for _, pvc := range item.pvcs {
				client.CoreV1().PersistentVolumeClaims(pvc.Namespace).Create(ctx, pvc, metav1.CreateOptions{})
			}
			for _, pv := range item.pvs {
				client.CoreV1().PersistentVolumes().Create(ctx, pv, metav1.CreateOptions{})
			}

			t.Log("Start informer factory after initialization")
			informerFactory.Start(ctx.Done())

			t.Log("Wait for all started informers' cache were synced")
			informerFactory.WaitForCacheSync(ctx.Done())

			t.Log("Verify")

			p := pl.(*VolumeBinding)
			nodeInfos := make([]*framework.NodeInfo, 0)
			for _, node := range item.nodes {
				nodeInfo := framework.NewNodeInfo()
				nodeInfo.SetNode(node)
				nodeInfos = append(nodeInfos, nodeInfo)
			}
			state := framework.NewCycleState()

			t.Logf("Verify: call PreFilter and check status")
			gotPreFilterStatus := p.PreFilter(ctx, state, item.pod)
			if !reflect.DeepEqual(gotPreFilterStatus, item.wantPreFilterStatus) {
				t.Errorf("filter prefilter status does not match: %v, want: %v", gotPreFilterStatus, item.wantPreFilterStatus)
			}
			if !gotPreFilterStatus.IsSuccess() {
				// scheduler framework will skip Filter if PreFilter fails
				return
			}

			t.Logf("Verify: check state after prefilter phase")
			got, err := getStateData(state)
			if err != nil {
				t.Fatal(err)
			}
			stateCmpOpts := []cmp.Option{
				cmp.AllowUnexported(stateData{}),
				cmpopts.IgnoreFields(stateData{}, "Mutex"),
			}
			if diff := cmp.Diff(item.wantStateAfterPreFilter, got, stateCmpOpts...); diff != "" {
				t.Errorf("state got after prefilter does not match (-want,+got):\n%s", diff)
			}

			t.Logf("Verify: call Filter and check status")
			for i, nodeInfo := range nodeInfos {
				gotStatus := p.Filter(ctx, state, item.pod, nodeInfo)
				if !reflect.DeepEqual(gotStatus, item.wantFilterStatus[i]) {
					t.Errorf("filter status does not match for node %q, got: %v, want: %v", nodeInfo.Node().Name, gotStatus, item.wantFilterStatus)
				}
			}

			t.Logf("Verify: Score")
			for i, node := range item.nodes {
				score, status := p.Score(ctx, state, item.pod, node.Name)
				if !status.IsSuccess() {
					t.Errorf("Score expects success status, got: %v", status)
				}
				if score != item.wantScores[i] {
					t.Errorf("Score expects score %d for node %q, got: %d", item.wantScores[i], node.Name, score)
				}
			}
		})
	}
}
