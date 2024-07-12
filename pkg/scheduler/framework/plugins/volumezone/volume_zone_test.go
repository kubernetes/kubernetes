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

package volumezone

import (
	"context"
	"fmt"
	"testing"

	"github.com/google/go-cmp/cmp"

	v1 "k8s.io/api/core/v1"
	storagev1 "k8s.io/api/storage/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/klog/v2/ktesting"
	"k8s.io/kubernetes/pkg/scheduler/apis/config"
	"k8s.io/kubernetes/pkg/scheduler/framework"
	plugintesting "k8s.io/kubernetes/pkg/scheduler/framework/plugins/testing"
	"k8s.io/kubernetes/pkg/scheduler/internal/cache"
	st "k8s.io/kubernetes/pkg/scheduler/testing"
	tf "k8s.io/kubernetes/pkg/scheduler/testing/framework"
)

func createPodWithVolume(pod, pvc string) *v1.Pod {
	return st.MakePod().Name(pod).Namespace(metav1.NamespaceDefault).PVC(pvc).Obj()
}

func TestSingleZone(t *testing.T) {
	pvLister := tf.PersistentVolumeLister{
		{
			ObjectMeta: metav1.ObjectMeta{Name: "Vol_1", Labels: map[string]string{v1.LabelFailureDomainBetaZone: "us-west1-a"}},
		},
		{
			ObjectMeta: metav1.ObjectMeta{Name: "Vol_2", Labels: map[string]string{v1.LabelFailureDomainBetaRegion: "us-west1", "uselessLabel": "none"}},
		},
		{
			ObjectMeta: metav1.ObjectMeta{Name: "Vol_3", Labels: map[string]string{v1.LabelFailureDomainBetaRegion: "us-west1"}},
		},
		{
			ObjectMeta: metav1.ObjectMeta{Name: "Vol_Stable_1", Labels: map[string]string{v1.LabelTopologyZone: "us-west1-a"}},
		},
		{
			ObjectMeta: metav1.ObjectMeta{Name: "Vol_Stable_2", Labels: map[string]string{v1.LabelTopologyRegion: "us-west1", "uselessLabel": "none"}},
		},
		{
			ObjectMeta: metav1.ObjectMeta{Name: "Vol_Stable_3", Labels: map[string]string{v1.LabelTopologyZone: "us-west1-a", v1.LabelTopologyRegion: "us-west1-a"}},
		},
	}

	pvcLister := tf.PersistentVolumeClaimLister{
		{
			ObjectMeta: metav1.ObjectMeta{Name: "PVC_1", Namespace: "default"},
			Spec:       v1.PersistentVolumeClaimSpec{VolumeName: "Vol_1"},
		},
		{
			ObjectMeta: metav1.ObjectMeta{Name: "PVC_2", Namespace: "default"},
			Spec:       v1.PersistentVolumeClaimSpec{VolumeName: "Vol_2"},
		},
		{
			ObjectMeta: metav1.ObjectMeta{Name: "PVC_3", Namespace: "default"},
			Spec:       v1.PersistentVolumeClaimSpec{VolumeName: "Vol_3"},
		},
		{
			ObjectMeta: metav1.ObjectMeta{Name: "PVC_4", Namespace: "default"},
			Spec:       v1.PersistentVolumeClaimSpec{VolumeName: "Vol_not_exist"},
		},
		{
			ObjectMeta: metav1.ObjectMeta{Name: "PVC_Stable_1", Namespace: "default"},
			Spec:       v1.PersistentVolumeClaimSpec{VolumeName: "Vol_Stable_1"},
		},
		{
			ObjectMeta: metav1.ObjectMeta{Name: "PVC_Stable_2", Namespace: "default"},
			Spec:       v1.PersistentVolumeClaimSpec{VolumeName: "Vol_Stable_2"},
		},
		{
			ObjectMeta: metav1.ObjectMeta{Name: "PVC_Stable_3", Namespace: "default"},
			Spec:       v1.PersistentVolumeClaimSpec{VolumeName: "Vol_Stable_3"},
		},
	}

	tests := []struct {
		name                string
		Pod                 *v1.Pod
		Node                *v1.Node
		wantPreFilterStatus *framework.Status
		wantFilterStatus    *framework.Status
	}{
		{
			name: "pod without volume",
			Pod:  st.MakePod().Name("pod_1").Namespace(metav1.NamespaceDefault).Obj(),
			Node: &v1.Node{
				ObjectMeta: metav1.ObjectMeta{
					Name:   "host1",
					Labels: map[string]string{v1.LabelFailureDomainBetaZone: "us-west1-a"},
				},
			},
			wantPreFilterStatus: framework.NewStatus(framework.Skip),
		},
		{
			name: "node without labels",
			Pod:  createPodWithVolume("pod_1", "PVC_1"),
			Node: &v1.Node{
				ObjectMeta: metav1.ObjectMeta{
					Name: "host1",
				},
			},
		},
		{
			name: "beta zone label matched",
			Pod:  createPodWithVolume("pod_1", "PVC_1"),
			Node: &v1.Node{
				ObjectMeta: metav1.ObjectMeta{
					Name:   "host1",
					Labels: map[string]string{v1.LabelFailureDomainBetaZone: "us-west1-a", "uselessLabel": "none"},
				},
			},
		},
		{
			name: "beta region label matched",
			Pod:  createPodWithVolume("pod_1", "PVC_2"),
			Node: &v1.Node{
				ObjectMeta: metav1.ObjectMeta{
					Name:   "host1",
					Labels: map[string]string{v1.LabelFailureDomainBetaRegion: "us-west1", "uselessLabel": "none"},
				},
			},
		},
		{
			name: "beta region label doesn't match",
			Pod:  createPodWithVolume("pod_1", "PVC_2"),
			Node: &v1.Node{
				ObjectMeta: metav1.ObjectMeta{
					Name:   "host1",
					Labels: map[string]string{v1.LabelFailureDomainBetaRegion: "no_us-west1", "uselessLabel": "none"},
				},
			},
			wantFilterStatus: framework.NewStatus(framework.UnschedulableAndUnresolvable, ErrReasonConflict),
		},
		{
			name: "beta zone label doesn't match",
			Pod:  createPodWithVolume("pod_1", "PVC_1"),
			Node: &v1.Node{
				ObjectMeta: metav1.ObjectMeta{
					Name:   "host1",
					Labels: map[string]string{v1.LabelFailureDomainBetaZone: "no_us-west1-a", "uselessLabel": "none"},
				},
			},
			wantFilterStatus: framework.NewStatus(framework.UnschedulableAndUnresolvable, ErrReasonConflict),
		},
		{
			name: "zone label matched",
			Pod:  createPodWithVolume("pod_1", "PVC_Stable_1"),
			Node: &v1.Node{
				ObjectMeta: metav1.ObjectMeta{
					Name:   "host1",
					Labels: map[string]string{v1.LabelTopologyZone: "us-west1-a", "uselessLabel": "none"},
				},
			},
		},
		{
			name: "region label matched",
			Pod:  createPodWithVolume("pod_1", "PVC_Stable_2"),
			Node: &v1.Node{
				ObjectMeta: metav1.ObjectMeta{
					Name:   "host1",
					Labels: map[string]string{v1.LabelTopologyRegion: "us-west1", "uselessLabel": "none"},
				},
			},
		},
		{
			name: "region label doesn't match",
			Pod:  createPodWithVolume("pod_1", "PVC_Stable_2"),
			Node: &v1.Node{
				ObjectMeta: metav1.ObjectMeta{
					Name:   "host1",
					Labels: map[string]string{v1.LabelTopologyRegion: "no_us-west1", "uselessLabel": "none"},
				},
			},
			wantFilterStatus: framework.NewStatus(framework.UnschedulableAndUnresolvable, ErrReasonConflict),
		},
		{
			name: "zone label doesn't match",
			Pod:  createPodWithVolume("pod_1", "PVC_Stable_1"),
			Node: &v1.Node{
				ObjectMeta: metav1.ObjectMeta{
					Name:   "host1",
					Labels: map[string]string{v1.LabelTopologyZone: "no_us-west1-a", "uselessLabel": "none"},
				},
			},
			wantFilterStatus: framework.NewStatus(framework.UnschedulableAndUnresolvable, ErrReasonConflict),
		},
		{
			name: "pv with zone and region, node with only zone",
			Pod:  createPodWithVolume("pod_1", "PVC_Stable_3"),
			Node: &v1.Node{
				ObjectMeta: metav1.ObjectMeta{
					Name: "host1",
					Labels: map[string]string{
						v1.LabelTopologyZone: "us-west1-a",
					},
				},
			},
			wantFilterStatus: framework.NewStatus(framework.UnschedulableAndUnresolvable, ErrReasonConflict),
		},
		{
			name: "pv with zone,node with beta zone",
			Pod:  createPodWithVolume("pod_1", "PVC_Stable_1"),
			Node: &v1.Node{
				ObjectMeta: metav1.ObjectMeta{
					Name: "host1",
					Labels: map[string]string{
						v1.LabelFailureDomainBetaZone: "us-west1-a",
					},
				},
			},
			wantFilterStatus: framework.NewStatus(framework.UnschedulableAndUnresolvable, ErrReasonConflict),
		},
		{
			name: "pv with beta label,node with ga label, matched",
			Pod:  createPodWithVolume("pod_1", "PVC_1"),
			Node: &v1.Node{
				ObjectMeta: metav1.ObjectMeta{
					Name: "host1",
					Labels: map[string]string{
						v1.LabelTopologyZone: "us-west1-a",
					},
				},
			},
		},
		{
			name: "pv with beta label,node with ga label, don't match",
			Pod:  createPodWithVolume("pod_1", "PVC_1"),
			Node: &v1.Node{
				ObjectMeta: metav1.ObjectMeta{
					Name: "host1",
					Labels: map[string]string{
						v1.LabelTopologyZone: "us-west1-b",
					},
				},
			},
			wantFilterStatus: framework.NewStatus(framework.UnschedulableAndUnresolvable, ErrReasonConflict),
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			_, ctx := ktesting.NewTestContext(t)
			ctx, cancel := context.WithCancel(ctx)
			defer cancel()

			state := framework.NewCycleState()
			node := &framework.NodeInfo{}
			node.SetNode(test.Node)
			p := &VolumeZone{
				pvLister,
				pvcLister,
				nil,
			}
			_, preFilterStatus := p.PreFilter(ctx, state, test.Pod)
			if diff := cmp.Diff(preFilterStatus, test.wantPreFilterStatus); diff != "" {
				t.Errorf("PreFilter: status does not match (-want,+got):\n%s", diff)
			}
			filterStatus := p.Filter(ctx, state, test.Pod, node)
			if diff := cmp.Diff(filterStatus, test.wantFilterStatus); diff != "" {
				t.Errorf("Filter: status does not match (-want,+got):\n%s", diff)
			}
		})
	}
}

func TestMultiZone(t *testing.T) {
	pvLister := tf.PersistentVolumeLister{
		{
			ObjectMeta: metav1.ObjectMeta{Name: "Vol_1", Labels: map[string]string{v1.LabelFailureDomainBetaZone: "us-west1-a"}},
		},
		{
			ObjectMeta: metav1.ObjectMeta{Name: "Vol_2", Labels: map[string]string{v1.LabelFailureDomainBetaZone: "us-west1-b", "uselessLabel": "none"}},
		},
		{
			ObjectMeta: metav1.ObjectMeta{Name: "Vol_3", Labels: map[string]string{v1.LabelFailureDomainBetaZone: "us-west1-c__us-west1-a"}},
		},
		{
			ObjectMeta: metav1.ObjectMeta{Name: "Vol_Stable_1", Labels: map[string]string{v1.LabelTopologyZone: "us-west1-a"}},
		},
		{
			ObjectMeta: metav1.ObjectMeta{Name: "Vol_Stable_2", Labels: map[string]string{v1.LabelTopologyZone: "us-west1-c__us-west1-a"}},
		},
	}

	pvcLister := tf.PersistentVolumeClaimLister{
		{
			ObjectMeta: metav1.ObjectMeta{Name: "PVC_1", Namespace: "default"},
			Spec:       v1.PersistentVolumeClaimSpec{VolumeName: "Vol_1"},
		},
		{
			ObjectMeta: metav1.ObjectMeta{Name: "PVC_2", Namespace: "default"},
			Spec:       v1.PersistentVolumeClaimSpec{VolumeName: "Vol_2"},
		},
		{
			ObjectMeta: metav1.ObjectMeta{Name: "PVC_3", Namespace: "default"},
			Spec:       v1.PersistentVolumeClaimSpec{VolumeName: "Vol_3"},
		},
		{
			ObjectMeta: metav1.ObjectMeta{Name: "PVC_4", Namespace: "default"},
			Spec:       v1.PersistentVolumeClaimSpec{VolumeName: "Vol_not_exist"},
		},
		{
			ObjectMeta: metav1.ObjectMeta{Name: "PVC_Stable_1", Namespace: "default"},
			Spec:       v1.PersistentVolumeClaimSpec{VolumeName: "Vol_Stable_1"},
		},
		{
			ObjectMeta: metav1.ObjectMeta{Name: "PVC_Stable_2", Namespace: "default"},
			Spec:       v1.PersistentVolumeClaimSpec{VolumeName: "Vol_Stable_2"},
		},
	}

	tests := []struct {
		name                string
		Pod                 *v1.Pod
		Node                *v1.Node
		wantPreFilterStatus *framework.Status
		wantFilterStatus    *framework.Status
	}{
		{
			name: "node without labels",
			Pod:  createPodWithVolume("pod_1", "PVC_3"),
			Node: &v1.Node{
				ObjectMeta: metav1.ObjectMeta{
					Name: "host1",
				},
			},
		},
		{
			name: "beta zone label matched",
			Pod:  createPodWithVolume("pod_1", "PVC_3"),
			Node: &v1.Node{
				ObjectMeta: metav1.ObjectMeta{
					Name:   "host1",
					Labels: map[string]string{v1.LabelFailureDomainBetaZone: "us-west1-a", "uselessLabel": "none"},
				},
			},
		},
		{
			name: "beta zone label doesn't match",
			Pod:  createPodWithVolume("pod_1", "PVC_1"),
			Node: &v1.Node{
				ObjectMeta: metav1.ObjectMeta{
					Name:   "host1",
					Labels: map[string]string{v1.LabelFailureDomainBetaZone: "us-west1-b", "uselessLabel": "none"},
				},
			},
			wantFilterStatus: framework.NewStatus(framework.UnschedulableAndUnresolvable, ErrReasonConflict),
		},
		{
			name: "zone label matched",
			Pod:  createPodWithVolume("pod_1", "PVC_Stable_2"),
			Node: &v1.Node{
				ObjectMeta: metav1.ObjectMeta{
					Name:   "host1",
					Labels: map[string]string{v1.LabelTopologyZone: "us-west1-a", "uselessLabel": "none"},
				},
			},
		},
		{
			name: "zone label doesn't match",
			Pod:  createPodWithVolume("pod_1", "PVC_Stable_1"),
			Node: &v1.Node{
				ObjectMeta: metav1.ObjectMeta{
					Name:   "host1",
					Labels: map[string]string{v1.LabelTopologyZone: "us-west1-b", "uselessLabel": "none"},
				},
			},
			wantFilterStatus: framework.NewStatus(framework.UnschedulableAndUnresolvable, ErrReasonConflict),
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			_, ctx := ktesting.NewTestContext(t)
			ctx, cancel := context.WithCancel(ctx)
			defer cancel()

			state := framework.NewCycleState()
			node := &framework.NodeInfo{}
			node.SetNode(test.Node)
			p := &VolumeZone{
				pvLister,
				pvcLister,
				nil,
			}
			_, preFilterStatus := p.PreFilter(ctx, state, test.Pod)
			if diff := cmp.Diff(preFilterStatus, test.wantPreFilterStatus); diff != "" {
				t.Errorf("PreFilter: status does not match (-want,+got):\n%s", diff)
			}
			filterStatus := p.Filter(ctx, state, test.Pod, node)
			if diff := cmp.Diff(filterStatus, test.wantFilterStatus); diff != "" {
				t.Errorf("Filter: status does not match (-want,+got):\n%s", diff)
			}
		})
	}
}

func TestWithBinding(t *testing.T) {
	var (
		modeWait = storagev1.VolumeBindingWaitForFirstConsumer

		class0         = "Class_0"
		classWait      = "Class_Wait"
		classImmediate = "Class_Immediate"
	)

	scLister := tf.StorageClassLister{
		{
			ObjectMeta: metav1.ObjectMeta{Name: classImmediate},
		},
		{
			ObjectMeta:        metav1.ObjectMeta{Name: classWait},
			VolumeBindingMode: &modeWait,
		},
	}

	pvLister := tf.PersistentVolumeLister{
		{
			ObjectMeta: metav1.ObjectMeta{Name: "Vol_1", Labels: map[string]string{v1.LabelFailureDomainBetaZone: "us-west1-a"}},
		},
	}

	pvcLister := tf.PersistentVolumeClaimLister{
		{
			ObjectMeta: metav1.ObjectMeta{Name: "PVC_1", Namespace: "default"},
			Spec:       v1.PersistentVolumeClaimSpec{VolumeName: "Vol_1"},
		},
		{
			ObjectMeta: metav1.ObjectMeta{Name: "PVC_NoSC", Namespace: "default"},
			Spec:       v1.PersistentVolumeClaimSpec{StorageClassName: &class0},
		},
		{
			ObjectMeta: metav1.ObjectMeta{Name: "PVC_EmptySC", Namespace: "default"},
		},
		{
			ObjectMeta: metav1.ObjectMeta{Name: "PVC_WaitSC", Namespace: "default"},
			Spec:       v1.PersistentVolumeClaimSpec{StorageClassName: &classWait},
		},
		{
			ObjectMeta: metav1.ObjectMeta{Name: "PVC_ImmediateSC", Namespace: "default"},
			Spec:       v1.PersistentVolumeClaimSpec{StorageClassName: &classImmediate},
		},
	}

	testNode := &v1.Node{
		ObjectMeta: metav1.ObjectMeta{
			Name:   "host1",
			Labels: map[string]string{v1.LabelFailureDomainBetaZone: "us-west1-a", "uselessLabel": "none"},
		},
	}

	tests := []struct {
		name                string
		Pod                 *v1.Pod
		Node                *v1.Node
		wantPreFilterStatus *framework.Status
		wantFilterStatus    *framework.Status
	}{
		{
			name: "label zone failure domain matched",
			Pod:  createPodWithVolume("pod_1", "PVC_1"),
			Node: testNode,
		},
		{
			name: "unbound volume empty storage class",
			Pod:  createPodWithVolume("pod_1", "PVC_EmptySC"),
			Node: testNode,
			wantPreFilterStatus: framework.NewStatus(framework.UnschedulableAndUnresolvable,
				"PersistentVolumeClaim had no pv name and storageClass name"),
			wantFilterStatus: framework.NewStatus(framework.UnschedulableAndUnresolvable,
				"PersistentVolumeClaim had no pv name and storageClass name"),
		},
		{
			name: "unbound volume no storage class",
			Pod:  createPodWithVolume("pod_1", "PVC_NoSC"),
			Node: testNode,
			wantPreFilterStatus: framework.NewStatus(framework.UnschedulableAndUnresolvable,
				`storageclasses.storage.k8s.io "Class_0" not found`),
			wantFilterStatus: framework.NewStatus(framework.UnschedulableAndUnresolvable,
				`storageclasses.storage.k8s.io "Class_0" not found`),
		},
		{
			name:                "unbound volume immediate binding mode",
			Pod:                 createPodWithVolume("pod_1", "PVC_ImmediateSC"),
			Node:                testNode,
			wantPreFilterStatus: framework.NewStatus(framework.UnschedulableAndUnresolvable, "VolumeBindingMode not set for StorageClass \"Class_Immediate\""),
			wantFilterStatus:    framework.NewStatus(framework.UnschedulableAndUnresolvable, "VolumeBindingMode not set for StorageClass \"Class_Immediate\""),
		},
		{
			name:                "unbound volume wait binding mode",
			Pod:                 createPodWithVolume("pod_1", "PVC_WaitSC"),
			Node:                testNode,
			wantPreFilterStatus: framework.NewStatus(framework.Skip),
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			_, ctx := ktesting.NewTestContext(t)
			ctx, cancel := context.WithCancel(ctx)
			defer cancel()

			state := framework.NewCycleState()
			node := &framework.NodeInfo{}
			node.SetNode(test.Node)
			p := &VolumeZone{
				pvLister,
				pvcLister,
				scLister,
			}
			_, preFilterStatus := p.PreFilter(ctx, state, test.Pod)
			if diff := cmp.Diff(preFilterStatus, test.wantPreFilterStatus); diff != "" {
				t.Errorf("PreFilter: status does not match (-want,+got):\n%s", diff)
			}
			filterStatus := p.Filter(ctx, state, test.Pod, node)
			if diff := cmp.Diff(filterStatus, test.wantFilterStatus); diff != "" {
				t.Errorf("Filter: status does not match (-want,+got):\n%s", diff)
			}
		})
	}
}

func TestIsSchedulableAfterPersistentVolumeClaimAdded(t *testing.T) {
	testcases := map[string]struct {
		pod            *v1.Pod
		oldObj, newObj interface{}
		expectedHint   framework.QueueingHint
		expectedErr    bool
	}{
		"error-wrong-new-object": {
			pod:          createPodWithVolume("pod_1", "PVC_1"),
			newObj:       "not-a-pvc",
			expectedHint: framework.Queue,
			expectedErr:  true,
		},
		"pvc-was-added-but-pod-refers-no-pvc": {
			pod: st.MakePod().Name("pod_1").Namespace("default").Obj(),
			newObj: &v1.PersistentVolumeClaim{
				ObjectMeta: metav1.ObjectMeta{Name: "PVC_1", Namespace: "default"},
				Spec:       v1.PersistentVolumeClaimSpec{VolumeName: "Vol_1"},
			},
			expectedHint: framework.QueueSkip,
		},
		"pvc-was-added-and-pod-was-bound-to-different-pvc": {
			pod: createPodWithVolume("pod_1", "PVC_2"),
			newObj: &v1.PersistentVolumeClaim{
				ObjectMeta: metav1.ObjectMeta{Name: "PVC_1", Namespace: "default"},
				Spec:       v1.PersistentVolumeClaimSpec{VolumeName: "Vol_1"},
			},
			expectedHint: framework.QueueSkip,
		},
		"pvc-was-added-and-pod-was-bound-to-pvc-but-different-ns": {
			pod: createPodWithVolume("pod_1", "PVC_1"),
			newObj: &v1.PersistentVolumeClaim{
				ObjectMeta: metav1.ObjectMeta{Name: "PVC_1", Namespace: "ns1"},
				Spec:       v1.PersistentVolumeClaimSpec{VolumeName: "Vol_1"},
			},
			expectedHint: framework.QueueSkip,
		},
		"pvc-was-added-and-pod-was-bound-to-the-pvc": {
			pod: createPodWithVolume("pod_1", "PVC_1"),
			newObj: &v1.PersistentVolumeClaim{
				ObjectMeta: metav1.ObjectMeta{Name: "PVC_1", Namespace: "default"},
				Spec:       v1.PersistentVolumeClaimSpec{VolumeName: "Vol_1"},
			},
			expectedHint: framework.Queue,
		},
		"pvc-was-updated-and-pod-was-bound-to-the-pvc": {
			pod: createPodWithVolume("pod_1", "PVC_1"),
			oldObj: &v1.PersistentVolumeClaim{
				ObjectMeta: metav1.ObjectMeta{Name: "PVC_1", Namespace: "default"},
				Spec:       v1.PersistentVolumeClaimSpec{VolumeName: ""},
			},
			newObj: &v1.PersistentVolumeClaim{
				ObjectMeta: metav1.ObjectMeta{Name: "PVC_1", Namespace: "default"},
				Spec:       v1.PersistentVolumeClaimSpec{VolumeName: "Vol_1"},
			},
			expectedHint: framework.Queue,
		},
		"pvc-was-updated-but-pod-refers-no-pvc": {
			pod: st.MakePod().Name("pod_1").Namespace(metav1.NamespaceDefault).Obj(),
			oldObj: &v1.PersistentVolumeClaim{
				ObjectMeta: metav1.ObjectMeta{Name: "PVC_1", Namespace: "default"},
				Spec:       v1.PersistentVolumeClaimSpec{VolumeName: ""},
			},
			newObj: &v1.PersistentVolumeClaim{
				ObjectMeta: metav1.ObjectMeta{Name: "PVC_1", Namespace: "default"},
				Spec:       v1.PersistentVolumeClaimSpec{VolumeName: "Vol_1"},
			},
			expectedHint: framework.QueueSkip,
		},
	}

	for name, tc := range testcases {
		t.Run(name, func(t *testing.T) {
			logger, _ := ktesting.NewTestContext(t)
			p := &VolumeZone{}

			got, err := p.isSchedulableAfterPersistentVolumeClaimChange(logger, tc.pod, tc.oldObj, tc.newObj)
			if err != nil && !tc.expectedErr {
				t.Errorf("unexpected error: %v", err)
			}
			if got != tc.expectedHint {
				t.Errorf("isSchedulableAfterPersistentVolumeClaimChange() = %v, want %v", got, tc.expectedHint)
			}
		})
	}
}

func TestIsSchedulableAfterStorageClassAdded(t *testing.T) {
	var modeWait = storagev1.VolumeBindingWaitForFirstConsumer

	testcases := map[string]struct {
		pod            *v1.Pod
		oldObj, newObj interface{}
		expectedHint   framework.QueueingHint
		expectedErr    bool
	}{
		"error-wrong-new-object": {
			pod:          createPodWithVolume("pod_1", "PVC_1"),
			newObj:       "not-a-storageclass",
			expectedHint: framework.Queue,
			expectedErr:  true,
		},
		"sc-doesn't-have-volume-binding-mode": {
			pod: createPodWithVolume("pod_1", "PVC_1"),
			newObj: &storagev1.StorageClass{
				ObjectMeta: metav1.ObjectMeta{Name: "SC_1"},
			},
			expectedHint: framework.QueueSkip,
		},
		"new-sc-is-wait-for-first-consumer-mode": {
			pod: createPodWithVolume("pod_1", "PVC_1"),
			newObj: &storagev1.StorageClass{
				ObjectMeta:        metav1.ObjectMeta{Name: "SC_1"},
				VolumeBindingMode: &modeWait,
			},
			expectedHint: framework.Queue,
		},
	}

	for name, tc := range testcases {
		t.Run(name, func(t *testing.T) {
			logger, _ := ktesting.NewTestContext(t)
			p := &VolumeZone{}

			got, err := p.isSchedulableAfterStorageClassAdded(logger, tc.pod, tc.oldObj, tc.newObj)
			if err != nil && !tc.expectedErr {
				t.Errorf("unexpected error: %v", err)
			}
			if got != tc.expectedHint {
				t.Errorf("isSchedulableAfterStorageClassAdded() = %v, want %v", got, tc.expectedHint)
			}
		})
	}
}

func TestIsSchedulableAfterPersistentVolumeChange(t *testing.T) {
	testcases := map[string]struct {
		pod            *v1.Pod
		oldObj, newObj interface{}
		expectedHint   framework.QueueingHint
		expectedErr    bool
	}{
		"error-wrong-new-object": {
			pod:          createPodWithVolume("pod_1", "PVC_1"),
			newObj:       "not-a-pv",
			expectedHint: framework.Queue,
			expectedErr:  true,
		},
		"error-wrong-old-object": {
			pod:          createPodWithVolume("pod_1", "PVC_1"),
			oldObj:       "not-a-pv",
			newObj:       st.MakePersistentVolume().Name("Vol_1").Obj(),
			expectedHint: framework.Queue,
			expectedErr:  true,
		},
		"new-pv-was-added": {
			pod: createPodWithVolume("pod_1", "PVC_1"),
			newObj: &v1.PersistentVolume{
				ObjectMeta: metav1.ObjectMeta{
					Name:   "Vol_1",
					Labels: map[string]string{v1.LabelFailureDomainBetaZone: "us-west1-b"},
				},
			},
			expectedHint: framework.Queue,
		},
		"pv-was-updated-and-changed-topology": {
			pod: createPodWithVolume("pod_1", "PVC_1"),
			oldObj: &v1.PersistentVolume{
				ObjectMeta: metav1.ObjectMeta{
					Name:   "Vol_1",
					Labels: map[string]string{v1.LabelFailureDomainBetaZone: "us-west1-a"},
				},
			},
			newObj: &v1.PersistentVolume{
				ObjectMeta: metav1.ObjectMeta{
					Name:   "Vol_1",
					Labels: map[string]string{v1.LabelFailureDomainBetaZone: "us-west1-b"},
				},
			},
			expectedHint: framework.Queue,
		},
		"pv-was-updated-and-added-topology-label": {
			pod: createPodWithVolume("pod_1", "PVC_1"),
			oldObj: &v1.PersistentVolume{
				ObjectMeta: metav1.ObjectMeta{
					Name:   "Vol_1",
					Labels: map[string]string{v1.LabelFailureDomainBetaZone: "us-west1-a"},
				},
			},
			newObj: &v1.PersistentVolume{
				ObjectMeta: metav1.ObjectMeta{
					Name: "Vol_1",
					Labels: map[string]string{v1.LabelFailureDomainBetaZone: "us-west1-a",
						v1.LabelTopologyZone: "zone"},
				},
			},
			expectedHint: framework.Queue,
		},
		"pv-was-updated-but-no-topology-is-changed": {
			pod: createPodWithVolume("pod_1", "PVC_1"),
			oldObj: &v1.PersistentVolume{
				ObjectMeta: metav1.ObjectMeta{
					Name: "Vol_1",
					Labels: map[string]string{v1.LabelFailureDomainBetaZone: "us-west1-a",
						v1.LabelTopologyZone: "zone"},
				},
			},
			newObj: &v1.PersistentVolume{
				ObjectMeta: metav1.ObjectMeta{
					Name: "Vol_1",
					Labels: map[string]string{v1.LabelFailureDomainBetaZone: "us-west1-a",
						v1.LabelTopologyZone: "zone"},
				},
			},
			expectedHint: framework.QueueSkip,
		},
	}

	for name, tc := range testcases {
		t.Run(name, func(t *testing.T) {
			logger, _ := ktesting.NewTestContext(t)
			p := &VolumeZone{}

			got, err := p.isSchedulableAfterPersistentVolumeChange(logger, tc.pod, tc.oldObj, tc.newObj)
			if err != nil && !tc.expectedErr {
				t.Errorf("unexpected error: %v", err)
			}
			if got != tc.expectedHint {
				t.Errorf("isSchedulableAfterPersistentVolumeChange() = %v, want %v", got, tc.expectedHint)
			}
		})
	}
}

func BenchmarkVolumeZone(b *testing.B) {
	tests := []struct {
		Name      string
		Pod       *v1.Pod
		NumPV     int
		NumPVC    int
		NumNodes  int
		PreFilter bool
	}{
		{
			Name:      "with prefilter",
			Pod:       createPodWithVolume("pod_0", "PVC_Stable_0"),
			NumPV:     1000,
			NumPVC:    1000,
			NumNodes:  1000,
			PreFilter: true,
		},
		{
			Name:      "without prefilter",
			Pod:       createPodWithVolume("pod_0", "PVC_Stable_0"),
			NumPV:     1000,
			NumPVC:    1000,
			NumNodes:  1000,
			PreFilter: false,
		},
	}

	for _, tt := range tests {
		b.Run(tt.Name, func(b *testing.B) {
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			nodes := makeNodesWithTopologyZone(tt.NumNodes)
			pl := newPluginWithListers(ctx, b, []*v1.Pod{tt.Pod}, nodes, makePVCsWithPV(tt.NumPVC), makePVsWithZoneLabel(tt.NumPV))
			nodeInfos := make([]*framework.NodeInfo, len(nodes))
			for i := 0; i < len(nodes); i++ {
				nodeInfo := &framework.NodeInfo{}
				nodeInfo.SetNode(nodes[i])
				nodeInfos[i] = nodeInfo
			}
			p := pl.(*VolumeZone)
			state := framework.NewCycleState()

			b.ResetTimer()

			for i := 0; i < b.N; i++ {
				if tt.PreFilter {
					_, _ = p.PreFilter(ctx, state, tt.Pod)
				}
				for _, node := range nodeInfos {
					_ = p.Filter(ctx, state, tt.Pod, node)
				}
			}
		})
	}
}

func newPluginWithListers(ctx context.Context, tb testing.TB, pods []*v1.Pod, nodes []*v1.Node, pvcs []*v1.PersistentVolumeClaim, pvs []*v1.PersistentVolume) framework.Plugin {
	snapshot := cache.NewSnapshot(pods, nodes)

	objects := make([]runtime.Object, 0, len(pvcs))
	for _, pvc := range pvcs {
		objects = append(objects, pvc)
	}
	for _, pv := range pvs {
		objects = append(objects, pv)
	}
	return plugintesting.SetupPluginWithInformers(ctx, tb, New, &config.InterPodAffinityArgs{}, snapshot, objects)
}

func makePVsWithZoneLabel(num int) []*v1.PersistentVolume {
	pvList := make([]*v1.PersistentVolume, num)
	for i := 0; i < len(pvList); i++ {
		pvName := fmt.Sprintf("Vol_Stable_%d", i)
		zone := fmt.Sprintf("us-west-%d", i)
		pvList[i] = &v1.PersistentVolume{
			ObjectMeta: metav1.ObjectMeta{Name: pvName, Labels: map[string]string{v1.LabelTopologyZone: zone}},
		}
	}
	return pvList
}

func makePVCsWithPV(num int) []*v1.PersistentVolumeClaim {
	pvcList := make([]*v1.PersistentVolumeClaim, num)
	for i := 0; i < len(pvcList); i++ {
		pvcName := fmt.Sprintf("PVC_Stable_%d", i)
		pvName := fmt.Sprintf("Vol_Stable_%d", i)
		pvcList[i] = &v1.PersistentVolumeClaim{
			ObjectMeta: metav1.ObjectMeta{Name: pvcName, Namespace: "default"},
			Spec:       v1.PersistentVolumeClaimSpec{VolumeName: pvName},
		}
	}
	return pvcList
}

func makeNodesWithTopologyZone(num int) []*v1.Node {
	nodeList := make([]*v1.Node, num)
	for i := 0; i < len(nodeList); i++ {
		nodeName := fmt.Sprintf("host_%d", i)
		zone := "us-west-0"
		nodeList[i] = &v1.Node{
			ObjectMeta: metav1.ObjectMeta{
				Name:   nodeName,
				Labels: map[string]string{v1.LabelTopologyZone: zone, "uselessLabel": "none"},
			},
		}
	}
	return nodeList
}
