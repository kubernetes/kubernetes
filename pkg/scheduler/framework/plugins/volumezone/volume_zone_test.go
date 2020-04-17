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
	"reflect"
	"testing"

	v1 "k8s.io/api/core/v1"
	storagev1 "k8s.io/api/storage/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	framework "k8s.io/kubernetes/pkg/scheduler/framework/v1alpha1"
	fakeframework "k8s.io/kubernetes/pkg/scheduler/framework/v1alpha1/fake"
)

func createPodWithVolume(pod, pv, pvc string) *v1.Pod {
	return &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{Name: pod, Namespace: "default"},
		Spec: v1.PodSpec{
			Volumes: []v1.Volume{
				{
					Name: pv,
					VolumeSource: v1.VolumeSource{
						PersistentVolumeClaim: &v1.PersistentVolumeClaimVolumeSource{
							ClaimName: pvc,
						},
					},
				},
			},
		},
	}
}

func TestSingleZone(t *testing.T) {
	pvLister := fakeframework.PersistentVolumeLister{
		{
			ObjectMeta: metav1.ObjectMeta{Name: "Vol_1", Labels: map[string]string{v1.LabelZoneFailureDomain: "us-west1-a"}},
		},
		{
			ObjectMeta: metav1.ObjectMeta{Name: "Vol_2", Labels: map[string]string{v1.LabelZoneRegion: "us-west1", "uselessLabel": "none"}},
		},
		{
			ObjectMeta: metav1.ObjectMeta{Name: "Vol_3", Labels: map[string]string{v1.LabelZoneRegion: "us-west1"}},
		},
		{
			ObjectMeta: metav1.ObjectMeta{Name: "Vol_Stable_1", Labels: map[string]string{v1.LabelZoneFailureDomainStable: "us-west1-a"}},
		},
		{
			ObjectMeta: metav1.ObjectMeta{Name: "Vol_Stable_2", Labels: map[string]string{v1.LabelZoneRegionStable: "us-west1", "uselessLabel": "none"}},
		},
	}

	pvcLister := fakeframework.PersistentVolumeClaimLister{
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
		name       string
		Pod        *v1.Pod
		Node       *v1.Node
		wantStatus *framework.Status
	}{
		{
			name: "pod without volume",
			Pod: &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{Name: "pod_1", Namespace: "default"},
			},
			Node: &v1.Node{
				ObjectMeta: metav1.ObjectMeta{
					Name:   "host1",
					Labels: map[string]string{v1.LabelZoneFailureDomain: "us-west1-a"},
				},
			},
		},
		{
			name: "node without labels",
			Pod:  createPodWithVolume("pod_1", "vol_1", "PVC_1"),
			Node: &v1.Node{
				ObjectMeta: metav1.ObjectMeta{
					Name: "host1",
				},
			},
		},
		{
			name: "beta zone label matched",
			Pod:  createPodWithVolume("pod_1", "vol_1", "PVC_1"),
			Node: &v1.Node{
				ObjectMeta: metav1.ObjectMeta{
					Name:   "host1",
					Labels: map[string]string{v1.LabelZoneFailureDomain: "us-west1-a", "uselessLabel": "none"},
				},
			},
		},
		{
			name: "beta region label matched",
			Pod:  createPodWithVolume("pod_1", "vol_1", "PVC_2"),
			Node: &v1.Node{
				ObjectMeta: metav1.ObjectMeta{
					Name:   "host1",
					Labels: map[string]string{v1.LabelZoneRegion: "us-west1", "uselessLabel": "none"},
				},
			},
		},
		{
			name: "beta region label doesn't match",
			Pod:  createPodWithVolume("pod_1", "vol_1", "PVC_2"),
			Node: &v1.Node{
				ObjectMeta: metav1.ObjectMeta{
					Name:   "host1",
					Labels: map[string]string{v1.LabelZoneRegion: "no_us-west1", "uselessLabel": "none"},
				},
			},
			wantStatus: framework.NewStatus(framework.UnschedulableAndUnresolvable, ErrReasonConflict),
		},
		{
			name: "beta zone label doesn't match",
			Pod:  createPodWithVolume("pod_1", "vol_1", "PVC_1"),
			Node: &v1.Node{
				ObjectMeta: metav1.ObjectMeta{
					Name:   "host1",
					Labels: map[string]string{v1.LabelZoneFailureDomain: "no_us-west1-a", "uselessLabel": "none"},
				},
			},
			wantStatus: framework.NewStatus(framework.UnschedulableAndUnresolvable, ErrReasonConflict),
		},
		{
			name: "zone label matched",
			Pod:  createPodWithVolume("pod_1", "Vol_Stable_1", "PVC_Stable_1"),
			Node: &v1.Node{
				ObjectMeta: metav1.ObjectMeta{
					Name:   "host1",
					Labels: map[string]string{v1.LabelZoneFailureDomainStable: "us-west1-a", "uselessLabel": "none"},
				},
			},
		},
		{
			name: "region label matched",
			Pod:  createPodWithVolume("pod_1", "Vol_Stable_2", "PVC_Stable_2"),
			Node: &v1.Node{
				ObjectMeta: metav1.ObjectMeta{
					Name:   "host1",
					Labels: map[string]string{v1.LabelZoneRegionStable: "us-west1", "uselessLabel": "none"},
				},
			},
		},
		{
			name: "region label doesn't match",
			Pod:  createPodWithVolume("pod_1", "Vol_Stable_2", "PVC_Stable_2"),
			Node: &v1.Node{
				ObjectMeta: metav1.ObjectMeta{
					Name:   "host1",
					Labels: map[string]string{v1.LabelZoneRegionStable: "no_us-west1", "uselessLabel": "none"},
				},
			},
			wantStatus: framework.NewStatus(framework.UnschedulableAndUnresolvable, ErrReasonConflict),
		},
		{
			name: "zone label doesn't match",
			Pod:  createPodWithVolume("pod_1", "Vol_Stable_1", "PVC_Stable_1"),
			Node: &v1.Node{
				ObjectMeta: metav1.ObjectMeta{
					Name:   "host1",
					Labels: map[string]string{v1.LabelZoneFailureDomainStable: "no_us-west1-a", "uselessLabel": "none"},
				},
			},
			wantStatus: framework.NewStatus(framework.UnschedulableAndUnresolvable, ErrReasonConflict),
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			node := &framework.NodeInfo{}
			node.SetNode(test.Node)
			p := &VolumeZone{
				pvLister,
				pvcLister,
				nil,
			}
			gotStatus := p.Filter(context.Background(), nil, test.Pod, node)
			if !reflect.DeepEqual(gotStatus, test.wantStatus) {
				t.Errorf("status does not match: %v, want: %v", gotStatus, test.wantStatus)
			}
		})
	}
}

func TestMultiZone(t *testing.T) {
	pvLister := fakeframework.PersistentVolumeLister{
		{
			ObjectMeta: metav1.ObjectMeta{Name: "Vol_1", Labels: map[string]string{v1.LabelZoneFailureDomain: "us-west1-a"}},
		},
		{
			ObjectMeta: metav1.ObjectMeta{Name: "Vol_2", Labels: map[string]string{v1.LabelZoneFailureDomain: "us-west1-b", "uselessLabel": "none"}},
		},
		{
			ObjectMeta: metav1.ObjectMeta{Name: "Vol_3", Labels: map[string]string{v1.LabelZoneFailureDomain: "us-west1-c__us-west1-a"}},
		},
		{
			ObjectMeta: metav1.ObjectMeta{Name: "Vol_Stable_1", Labels: map[string]string{v1.LabelZoneFailureDomainStable: "us-west1-a"}},
		},
		{
			ObjectMeta: metav1.ObjectMeta{Name: "Vol_Stable_2", Labels: map[string]string{v1.LabelZoneFailureDomainStable: "us-west1-c__us-west1-a"}},
		},
	}

	pvcLister := fakeframework.PersistentVolumeClaimLister{
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
		name       string
		Pod        *v1.Pod
		Node       *v1.Node
		wantStatus *framework.Status
	}{
		{
			name: "node without labels",
			Pod:  createPodWithVolume("pod_1", "Vol_3", "PVC_3"),
			Node: &v1.Node{
				ObjectMeta: metav1.ObjectMeta{
					Name: "host1",
				},
			},
		},
		{
			name: "beta zone label matched",
			Pod:  createPodWithVolume("pod_1", "Vol_3", "PVC_3"),
			Node: &v1.Node{
				ObjectMeta: metav1.ObjectMeta{
					Name:   "host1",
					Labels: map[string]string{v1.LabelZoneFailureDomain: "us-west1-a", "uselessLabel": "none"},
				},
			},
		},
		{
			name: "beta zone label doesn't match",
			Pod:  createPodWithVolume("pod_1", "vol_1", "PVC_1"),
			Node: &v1.Node{
				ObjectMeta: metav1.ObjectMeta{
					Name:   "host1",
					Labels: map[string]string{v1.LabelZoneFailureDomain: "us-west1-b", "uselessLabel": "none"},
				},
			},
			wantStatus: framework.NewStatus(framework.UnschedulableAndUnresolvable, ErrReasonConflict),
		},
		{
			name: "zone label matched",
			Pod:  createPodWithVolume("pod_1", "Vol_Stable_2", "PVC_Stable_2"),
			Node: &v1.Node{
				ObjectMeta: metav1.ObjectMeta{
					Name:   "host1",
					Labels: map[string]string{v1.LabelZoneFailureDomainStable: "us-west1-a", "uselessLabel": "none"},
				},
			},
		},
		{
			name: "zone label doesn't match",
			Pod:  createPodWithVolume("pod_1", "Vol_Stable_1", "PVC_Stable_1"),
			Node: &v1.Node{
				ObjectMeta: metav1.ObjectMeta{
					Name:   "host1",
					Labels: map[string]string{v1.LabelZoneFailureDomainStable: "us-west1-b", "uselessLabel": "none"},
				},
			},
			wantStatus: framework.NewStatus(framework.UnschedulableAndUnresolvable, ErrReasonConflict),
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			node := &framework.NodeInfo{}
			node.SetNode(test.Node)
			p := &VolumeZone{
				pvLister,
				pvcLister,
				nil,
			}
			gotStatus := p.Filter(context.Background(), nil, test.Pod, node)
			if !reflect.DeepEqual(gotStatus, test.wantStatus) {
				t.Errorf("status does not match: %v, want: %v", gotStatus, test.wantStatus)
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

	scLister := fakeframework.StorageClassLister{
		{
			ObjectMeta: metav1.ObjectMeta{Name: classImmediate},
		},
		{
			ObjectMeta:        metav1.ObjectMeta{Name: classWait},
			VolumeBindingMode: &modeWait,
		},
	}

	pvLister := fakeframework.PersistentVolumeLister{
		{
			ObjectMeta: metav1.ObjectMeta{Name: "Vol_1", Labels: map[string]string{v1.LabelZoneFailureDomain: "us-west1-a"}},
		},
	}

	pvcLister := fakeframework.PersistentVolumeClaimLister{
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
			Labels: map[string]string{v1.LabelZoneFailureDomain: "us-west1-a", "uselessLabel": "none"},
		},
	}

	tests := []struct {
		name       string
		Pod        *v1.Pod
		Node       *v1.Node
		wantStatus *framework.Status
	}{
		{
			name: "label zone failure domain matched",
			Pod:  createPodWithVolume("pod_1", "vol_1", "PVC_1"),
			Node: testNode,
		},
		{
			name: "unbound volume empty storage class",
			Pod:  createPodWithVolume("pod_1", "vol_1", "PVC_EmptySC"),
			Node: testNode,
			wantStatus: framework.NewStatus(framework.Error,
				"PersistentVolumeClaim had no pv name and storageClass name"),
		},
		{
			name: "unbound volume no storage class",
			Pod:  createPodWithVolume("pod_1", "vol_1", "PVC_NoSC"),
			Node: testNode,
			wantStatus: framework.NewStatus(framework.Error,
				"StorageClass \"Class_0\" claimed by PersistentVolumeClaim \"PVC_NoSC\" not found"),
		},
		{
			name:       "unbound volume immediate binding mode",
			Pod:        createPodWithVolume("pod_1", "vol_1", "PVC_ImmediateSC"),
			Node:       testNode,
			wantStatus: framework.NewStatus(framework.Error, "VolumeBindingMode not set for StorageClass \"Class_Immediate\""),
		},
		{
			name: "unbound volume wait binding mode",
			Pod:  createPodWithVolume("pod_1", "vol_1", "PVC_WaitSC"),
			Node: testNode,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			node := &framework.NodeInfo{}
			node.SetNode(test.Node)
			p := &VolumeZone{
				pvLister,
				pvcLister,
				scLister,
			}
			gotStatus := p.Filter(context.Background(), nil, test.Pod, node)
			if !reflect.DeepEqual(gotStatus, test.wantStatus) {
				t.Errorf("status does not match: %v, want: %v", gotStatus, test.wantStatus)
			}
		})
	}
}
