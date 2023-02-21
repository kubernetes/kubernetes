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

package volumerestrictions

import (
	"context"
	"reflect"
	"testing"

	"github.com/google/go-cmp/cmp"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/scheduler/apis/config"
	"k8s.io/kubernetes/pkg/scheduler/framework"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/feature"
	plugintesting "k8s.io/kubernetes/pkg/scheduler/framework/plugins/testing"
	"k8s.io/kubernetes/pkg/scheduler/internal/cache"
	st "k8s.io/kubernetes/pkg/scheduler/testing"
)

func TestGCEDiskConflicts(t *testing.T) {
	volState := v1.Volume{
		VolumeSource: v1.VolumeSource{
			GCEPersistentDisk: &v1.GCEPersistentDiskVolumeSource{
				PDName: "foo",
			},
		},
	}
	volState2 := v1.Volume{
		VolumeSource: v1.VolumeSource{
			GCEPersistentDisk: &v1.GCEPersistentDiskVolumeSource{
				PDName: "bar",
			},
		},
	}
	errStatus := framework.NewStatus(framework.Unschedulable, ErrReasonDiskConflict)
	tests := []struct {
		pod        *v1.Pod
		nodeInfo   *framework.NodeInfo
		isOk       bool
		name       string
		wantStatus *framework.Status
	}{
		{&v1.Pod{}, framework.NewNodeInfo(), true, "nothing", nil},
		{&v1.Pod{}, framework.NewNodeInfo(st.MakePod().Volume(volState).Obj()), true, "one state", nil},
		{st.MakePod().Volume(volState).Obj(), framework.NewNodeInfo(st.MakePod().Volume(volState).Obj()), false, "same state", errStatus},
		{st.MakePod().Volume(volState2).Obj(), framework.NewNodeInfo(st.MakePod().Volume(volState).Obj()), true, "different state", nil},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			p := newPlugin(ctx, t)
			cycleState := framework.NewCycleState()
			p.(framework.PreFilterPlugin).PreFilter(ctx, cycleState, test.pod)
			gotStatus := p.(framework.FilterPlugin).Filter(ctx, cycleState, test.pod, test.nodeInfo)
			if !reflect.DeepEqual(gotStatus, test.wantStatus) {
				t.Errorf("status does not match: %v, want: %v", gotStatus, test.wantStatus)
			}
		})
	}
}

func TestAWSDiskConflicts(t *testing.T) {
	volState := v1.Volume{
		VolumeSource: v1.VolumeSource{
			AWSElasticBlockStore: &v1.AWSElasticBlockStoreVolumeSource{
				VolumeID: "foo",
			},
		},
	}
	volState2 := v1.Volume{
		VolumeSource: v1.VolumeSource{
			AWSElasticBlockStore: &v1.AWSElasticBlockStoreVolumeSource{
				VolumeID: "bar",
			},
		},
	}
	errStatus := framework.NewStatus(framework.Unschedulable, ErrReasonDiskConflict)
	tests := []struct {
		pod        *v1.Pod
		nodeInfo   *framework.NodeInfo
		isOk       bool
		name       string
		wantStatus *framework.Status
	}{
		{&v1.Pod{}, framework.NewNodeInfo(), true, "nothing", nil},
		{&v1.Pod{}, framework.NewNodeInfo(st.MakePod().Volume(volState).Obj()), true, "one state", nil},
		{st.MakePod().Volume(volState).Obj(), framework.NewNodeInfo(st.MakePod().Volume(volState).Obj()), false, "same state", errStatus},
		{st.MakePod().Volume(volState2).Obj(), framework.NewNodeInfo(st.MakePod().Volume(volState).Obj()), true, "different state", nil},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			p := newPlugin(ctx, t)
			cycleState := framework.NewCycleState()
			p.(framework.PreFilterPlugin).PreFilter(ctx, cycleState, test.pod)
			gotStatus := p.(framework.FilterPlugin).Filter(ctx, cycleState, test.pod, test.nodeInfo)
			if !reflect.DeepEqual(gotStatus, test.wantStatus) {
				t.Errorf("status does not match: %v, want: %v", gotStatus, test.wantStatus)
			}
		})
	}
}

func TestRBDDiskConflicts(t *testing.T) {
	volState := v1.Volume{
		VolumeSource: v1.VolumeSource{
			RBD: &v1.RBDVolumeSource{
				CephMonitors: []string{"a", "b"},
				RBDPool:      "foo",
				RBDImage:     "bar",
				FSType:       "ext4",
			},
		},
	}
	volState2 := v1.Volume{
		VolumeSource: v1.VolumeSource{
			RBD: &v1.RBDVolumeSource{
				CephMonitors: []string{"c", "d"},
				RBDPool:      "foo",
				RBDImage:     "bar",
				FSType:       "ext4",
			},
		},
	}
	errStatus := framework.NewStatus(framework.Unschedulable, ErrReasonDiskConflict)
	tests := []struct {
		pod        *v1.Pod
		nodeInfo   *framework.NodeInfo
		isOk       bool
		name       string
		wantStatus *framework.Status
	}{
		{&v1.Pod{}, framework.NewNodeInfo(), true, "nothing", nil},
		{&v1.Pod{}, framework.NewNodeInfo(st.MakePod().Volume(volState).Obj()), true, "one state", nil},
		{st.MakePod().Volume(volState).Obj(), framework.NewNodeInfo(st.MakePod().Volume(volState).Obj()), false, "same state", errStatus},
		{st.MakePod().Volume(volState2).Obj(), framework.NewNodeInfo(st.MakePod().Volume(volState).Obj()), true, "different state", nil},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			p := newPlugin(ctx, t)
			cycleState := framework.NewCycleState()
			p.(framework.PreFilterPlugin).PreFilter(ctx, cycleState, test.pod)
			gotStatus := p.(framework.FilterPlugin).Filter(ctx, cycleState, test.pod, test.nodeInfo)
			if !reflect.DeepEqual(gotStatus, test.wantStatus) {
				t.Errorf("status does not match: %v, want: %v", gotStatus, test.wantStatus)
			}
		})
	}
}

func TestISCSIDiskConflicts(t *testing.T) {
	volState := v1.Volume{
		VolumeSource: v1.VolumeSource{
			ISCSI: &v1.ISCSIVolumeSource{
				TargetPortal: "127.0.0.1:3260",
				IQN:          "iqn.2016-12.server:storage.target01",
				FSType:       "ext4",
				Lun:          0,
			},
		},
	}
	volState2 := v1.Volume{
		VolumeSource: v1.VolumeSource{
			ISCSI: &v1.ISCSIVolumeSource{
				TargetPortal: "127.0.0.1:3260",
				IQN:          "iqn.2017-12.server:storage.target01",
				FSType:       "ext4",
				Lun:          0,
			},
		},
	}
	errStatus := framework.NewStatus(framework.Unschedulable, ErrReasonDiskConflict)
	tests := []struct {
		pod        *v1.Pod
		nodeInfo   *framework.NodeInfo
		isOk       bool
		name       string
		wantStatus *framework.Status
	}{
		{&v1.Pod{}, framework.NewNodeInfo(), true, "nothing", nil},
		{&v1.Pod{}, framework.NewNodeInfo(st.MakePod().Volume(volState).Obj()), true, "one state", nil},
		{st.MakePod().Volume(volState).Obj(), framework.NewNodeInfo(st.MakePod().Volume(volState).Obj()), false, "same state", errStatus},
		{st.MakePod().Volume(volState2).Obj(), framework.NewNodeInfo(st.MakePod().Volume(volState).Obj()), true, "different state", nil},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			p := newPlugin(ctx, t)
			cycleState := framework.NewCycleState()
			p.(framework.PreFilterPlugin).PreFilter(ctx, cycleState, test.pod)
			gotStatus := p.(framework.FilterPlugin).Filter(ctx, cycleState, test.pod, test.nodeInfo)
			if !reflect.DeepEqual(gotStatus, test.wantStatus) {
				t.Errorf("status does not match: %v, want: %v", gotStatus, test.wantStatus)
			}
		})
	}
}

func TestAccessModeConflicts(t *testing.T) {
	defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.ReadWriteOncePod, true)()

	// Required for querying lister for PVCs in the same namespace.
	podWithOnePVC := st.MakePod().Name("pod-with-one-pvc").Namespace(metav1.NamespaceDefault).PVC("claim-with-rwop-1").Node("node-1").Obj()
	podWithTwoPVCs := st.MakePod().Name("pod-with-two-pvcs").Namespace(metav1.NamespaceDefault).PVC("claim-with-rwop-1").PVC("claim-with-rwop-2").Node("node-1").Obj()
	podWithOneConflict := st.MakePod().Name("pod-with-one-conflict").Namespace(metav1.NamespaceDefault).PVC("claim-with-rwop-1").Node("node-1").Obj()
	podWithTwoConflicts := st.MakePod().Name("pod-with-two-conflicts").Namespace(metav1.NamespaceDefault).PVC("claim-with-rwop-1").PVC("claim-with-rwop-2").Node("node-1").Obj()
	// Required for querying lister for PVCs in the same namespace.
	podWithReadWriteManyPVC := st.MakePod().Name("pod-with-rwx").Namespace(metav1.NamespaceDefault).PVC("claim-with-rwx").Node("node-1").Obj()

	node := &v1.Node{
		ObjectMeta: metav1.ObjectMeta{
			Namespace: "default",
			Name:      "node-1",
		},
	}

	readWriteOncePodPVC1 := &v1.PersistentVolumeClaim{
		ObjectMeta: metav1.ObjectMeta{
			Namespace: "default",
			Name:      "claim-with-rwop-1",
		},
		Spec: v1.PersistentVolumeClaimSpec{
			AccessModes: []v1.PersistentVolumeAccessMode{v1.ReadWriteOncePod},
		},
	}
	readWriteOncePodPVC2 := &v1.PersistentVolumeClaim{
		ObjectMeta: metav1.ObjectMeta{
			Namespace: "default",
			Name:      "claim-with-rwop-2",
		},
		Spec: v1.PersistentVolumeClaimSpec{
			AccessModes: []v1.PersistentVolumeAccessMode{v1.ReadWriteOncePod},
		},
	}
	readWriteManyPVC := &v1.PersistentVolumeClaim{
		ObjectMeta: metav1.ObjectMeta{
			Namespace: "default",
			Name:      "claim-with-rwx",
		},
		Spec: v1.PersistentVolumeClaimSpec{
			AccessModes: []v1.PersistentVolumeAccessMode{v1.ReadWriteMany},
		},
	}

	tests := []struct {
		name                   string
		pod                    *v1.Pod
		nodeInfo               *framework.NodeInfo
		existingPods           []*v1.Pod
		existingNodes          []*v1.Node
		existingPVCs           []*v1.PersistentVolumeClaim
		enableReadWriteOncePod bool
		preFilterWantStatus    *framework.Status
		wantStatus             *framework.Status
	}{
		{
			name:                   "nothing",
			pod:                    &v1.Pod{},
			nodeInfo:               framework.NewNodeInfo(),
			existingPods:           []*v1.Pod{},
			existingNodes:          []*v1.Node{},
			existingPVCs:           []*v1.PersistentVolumeClaim{},
			enableReadWriteOncePod: true,
			preFilterWantStatus:    nil,
			wantStatus:             nil,
		},
		{
			name:                   "failed to get PVC",
			pod:                    podWithOnePVC,
			nodeInfo:               framework.NewNodeInfo(),
			existingPods:           []*v1.Pod{},
			existingNodes:          []*v1.Node{},
			existingPVCs:           []*v1.PersistentVolumeClaim{},
			enableReadWriteOncePod: true,
			preFilterWantStatus:    framework.NewStatus(framework.UnschedulableAndUnresolvable, "persistentvolumeclaim \"claim-with-rwop-1\" not found"),
			wantStatus:             nil,
		},
		{
			name:                   "no access mode conflict",
			pod:                    podWithOnePVC,
			nodeInfo:               framework.NewNodeInfo(podWithReadWriteManyPVC),
			existingPods:           []*v1.Pod{podWithReadWriteManyPVC},
			existingNodes:          []*v1.Node{node},
			existingPVCs:           []*v1.PersistentVolumeClaim{readWriteOncePodPVC1, readWriteManyPVC},
			enableReadWriteOncePod: true,
			preFilterWantStatus:    nil,
			wantStatus:             nil,
		},
		{
			name:                   "access mode conflict, unschedulable",
			pod:                    podWithOneConflict,
			nodeInfo:               framework.NewNodeInfo(podWithOnePVC, podWithReadWriteManyPVC),
			existingPods:           []*v1.Pod{podWithOnePVC, podWithReadWriteManyPVC},
			existingNodes:          []*v1.Node{node},
			existingPVCs:           []*v1.PersistentVolumeClaim{readWriteOncePodPVC1, readWriteManyPVC},
			enableReadWriteOncePod: true,
			preFilterWantStatus:    nil,
			wantStatus:             framework.NewStatus(framework.Unschedulable, ErrReasonReadWriteOncePodConflict),
		},
		{
			name:                   "two conflicts, unschedulable",
			pod:                    podWithTwoConflicts,
			nodeInfo:               framework.NewNodeInfo(podWithTwoPVCs, podWithReadWriteManyPVC),
			existingPods:           []*v1.Pod{podWithTwoPVCs, podWithReadWriteManyPVC},
			existingNodes:          []*v1.Node{node},
			existingPVCs:           []*v1.PersistentVolumeClaim{readWriteOncePodPVC1, readWriteOncePodPVC2, readWriteManyPVC},
			enableReadWriteOncePod: true,
			preFilterWantStatus:    nil,
			wantStatus:             framework.NewStatus(framework.Unschedulable, ErrReasonReadWriteOncePodConflict),
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			p := newPluginWithListers(ctx, t, test.existingPods, test.existingNodes, test.existingPVCs, test.enableReadWriteOncePod)
			cycleState := framework.NewCycleState()
			_, preFilterGotStatus := p.(framework.PreFilterPlugin).PreFilter(ctx, cycleState, test.pod)
			if diff := cmp.Diff(test.preFilterWantStatus, preFilterGotStatus); diff != "" {
				t.Errorf("Unexpected PreFilter status (-want, +got): %s", diff)
			}
			// If PreFilter fails, then Filter will not run.
			if test.preFilterWantStatus.IsSuccess() {
				gotStatus := p.(framework.FilterPlugin).Filter(ctx, cycleState, test.pod, test.nodeInfo)
				if diff := cmp.Diff(test.wantStatus, gotStatus); diff != "" {
					t.Errorf("Unexpected Filter status (-want, +got): %s", diff)
				}
			}
		})
	}
}

func newPlugin(ctx context.Context, t *testing.T) framework.Plugin {
	return newPluginWithListers(ctx, t, nil, nil, nil, true)
}

func newPluginWithListers(ctx context.Context, t *testing.T, pods []*v1.Pod, nodes []*v1.Node, pvcs []*v1.PersistentVolumeClaim, enableReadWriteOncePod bool) framework.Plugin {
	pluginFactory := func(plArgs runtime.Object, fh framework.Handle) (framework.Plugin, error) {
		return New(plArgs, fh, feature.Features{
			EnableReadWriteOncePod: enableReadWriteOncePod,
		})
	}
	snapshot := cache.NewSnapshot(pods, nodes)

	objects := make([]runtime.Object, 0, len(pvcs))
	for _, pvc := range pvcs {
		objects = append(objects, pvc)
	}

	return plugintesting.SetupPluginWithInformers(ctx, t, pluginFactory, &config.InterPodAffinityArgs{}, snapshot, objects)
}
