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

	v1 "k8s.io/api/core/v1"
	framework "k8s.io/kubernetes/pkg/scheduler/framework/v1alpha1"
	schedulernodeinfo "k8s.io/kubernetes/pkg/scheduler/nodeinfo"
)

func TestGCEDiskConflicts(t *testing.T) {
	volState := v1.PodSpec{
		Volumes: []v1.Volume{
			{
				VolumeSource: v1.VolumeSource{
					GCEPersistentDisk: &v1.GCEPersistentDiskVolumeSource{
						PDName: "foo",
					},
				},
			},
		},
	}
	volState2 := v1.PodSpec{
		Volumes: []v1.Volume{
			{
				VolumeSource: v1.VolumeSource{
					GCEPersistentDisk: &v1.GCEPersistentDiskVolumeSource{
						PDName: "bar",
					},
				},
			},
		},
	}
	errStatus := framework.NewStatus(framework.Unschedulable, ErrReasonDiskConflict)
	tests := []struct {
		pod        *v1.Pod
		nodeInfo   *schedulernodeinfo.NodeInfo
		isOk       bool
		name       string
		wantStatus *framework.Status
	}{
		{&v1.Pod{}, schedulernodeinfo.NewNodeInfo(), true, "nothing", nil},
		{&v1.Pod{}, schedulernodeinfo.NewNodeInfo(&v1.Pod{Spec: volState}), true, "one state", nil},
		{&v1.Pod{Spec: volState}, schedulernodeinfo.NewNodeInfo(&v1.Pod{Spec: volState}), false, "same state", errStatus},
		{&v1.Pod{Spec: volState2}, schedulernodeinfo.NewNodeInfo(&v1.Pod{Spec: volState}), true, "different state", nil},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			p, _ := New(nil, nil)
			gotStatus := p.(framework.FilterPlugin).Filter(context.Background(), nil, test.pod, test.nodeInfo)
			if !reflect.DeepEqual(gotStatus, test.wantStatus) {
				t.Errorf("status does not match: %v, want: %v", gotStatus, test.wantStatus)
			}
		})
	}
}

func TestAWSDiskConflicts(t *testing.T) {
	volState := v1.PodSpec{
		Volumes: []v1.Volume{
			{
				VolumeSource: v1.VolumeSource{
					AWSElasticBlockStore: &v1.AWSElasticBlockStoreVolumeSource{
						VolumeID: "foo",
					},
				},
			},
		},
	}
	volState2 := v1.PodSpec{
		Volumes: []v1.Volume{
			{
				VolumeSource: v1.VolumeSource{
					AWSElasticBlockStore: &v1.AWSElasticBlockStoreVolumeSource{
						VolumeID: "bar",
					},
				},
			},
		},
	}
	errStatus := framework.NewStatus(framework.Unschedulable, ErrReasonDiskConflict)
	tests := []struct {
		pod        *v1.Pod
		nodeInfo   *schedulernodeinfo.NodeInfo
		isOk       bool
		name       string
		wantStatus *framework.Status
	}{
		{&v1.Pod{}, schedulernodeinfo.NewNodeInfo(), true, "nothing", nil},
		{&v1.Pod{}, schedulernodeinfo.NewNodeInfo(&v1.Pod{Spec: volState}), true, "one state", nil},
		{&v1.Pod{Spec: volState}, schedulernodeinfo.NewNodeInfo(&v1.Pod{Spec: volState}), false, "same state", errStatus},
		{&v1.Pod{Spec: volState2}, schedulernodeinfo.NewNodeInfo(&v1.Pod{Spec: volState}), true, "different state", nil},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			p, _ := New(nil, nil)
			gotStatus := p.(framework.FilterPlugin).Filter(context.Background(), nil, test.pod, test.nodeInfo)
			if !reflect.DeepEqual(gotStatus, test.wantStatus) {
				t.Errorf("status does not match: %v, want: %v", gotStatus, test.wantStatus)
			}
		})
	}
}

func TestRBDDiskConflicts(t *testing.T) {
	volState := v1.PodSpec{
		Volumes: []v1.Volume{
			{
				VolumeSource: v1.VolumeSource{
					RBD: &v1.RBDVolumeSource{
						CephMonitors: []string{"a", "b"},
						RBDPool:      "foo",
						RBDImage:     "bar",
						FSType:       "ext4",
					},
				},
			},
		},
	}
	volState2 := v1.PodSpec{
		Volumes: []v1.Volume{
			{
				VolumeSource: v1.VolumeSource{
					RBD: &v1.RBDVolumeSource{
						CephMonitors: []string{"c", "d"},
						RBDPool:      "foo",
						RBDImage:     "bar",
						FSType:       "ext4",
					},
				},
			},
		},
	}
	errStatus := framework.NewStatus(framework.Unschedulable, ErrReasonDiskConflict)
	tests := []struct {
		pod        *v1.Pod
		nodeInfo   *schedulernodeinfo.NodeInfo
		isOk       bool
		name       string
		wantStatus *framework.Status
	}{
		{&v1.Pod{}, schedulernodeinfo.NewNodeInfo(), true, "nothing", nil},
		{&v1.Pod{}, schedulernodeinfo.NewNodeInfo(&v1.Pod{Spec: volState}), true, "one state", nil},
		{&v1.Pod{Spec: volState}, schedulernodeinfo.NewNodeInfo(&v1.Pod{Spec: volState}), false, "same state", errStatus},
		{&v1.Pod{Spec: volState2}, schedulernodeinfo.NewNodeInfo(&v1.Pod{Spec: volState}), true, "different state", nil},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			p, _ := New(nil, nil)
			gotStatus := p.(framework.FilterPlugin).Filter(context.Background(), nil, test.pod, test.nodeInfo)
			if !reflect.DeepEqual(gotStatus, test.wantStatus) {
				t.Errorf("status does not match: %v, want: %v", gotStatus, test.wantStatus)
			}
		})
	}
}

func TestISCSIDiskConflicts(t *testing.T) {
	volState := v1.PodSpec{
		Volumes: []v1.Volume{
			{
				VolumeSource: v1.VolumeSource{
					ISCSI: &v1.ISCSIVolumeSource{
						TargetPortal: "127.0.0.1:3260",
						IQN:          "iqn.2016-12.server:storage.target01",
						FSType:       "ext4",
						Lun:          0,
					},
				},
			},
		},
	}
	volState2 := v1.PodSpec{
		Volumes: []v1.Volume{
			{
				VolumeSource: v1.VolumeSource{
					ISCSI: &v1.ISCSIVolumeSource{
						TargetPortal: "127.0.0.1:3260",
						IQN:          "iqn.2017-12.server:storage.target01",
						FSType:       "ext4",
						Lun:          0,
					},
				},
			},
		},
	}
	errStatus := framework.NewStatus(framework.Unschedulable, ErrReasonDiskConflict)
	tests := []struct {
		pod        *v1.Pod
		nodeInfo   *schedulernodeinfo.NodeInfo
		isOk       bool
		name       string
		wantStatus *framework.Status
	}{
		{&v1.Pod{}, schedulernodeinfo.NewNodeInfo(), true, "nothing", nil},
		{&v1.Pod{}, schedulernodeinfo.NewNodeInfo(&v1.Pod{Spec: volState}), true, "one state", nil},
		{&v1.Pod{Spec: volState}, schedulernodeinfo.NewNodeInfo(&v1.Pod{Spec: volState}), false, "same state", errStatus},
		{&v1.Pod{Spec: volState2}, schedulernodeinfo.NewNodeInfo(&v1.Pod{Spec: volState}), true, "different state", nil},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			p, _ := New(nil, nil)
			gotStatus := p.(framework.FilterPlugin).Filter(context.Background(), nil, test.pod, test.nodeInfo)
			if !reflect.DeepEqual(gotStatus, test.wantStatus) {
				t.Errorf("status does not match: %v, want: %v", gotStatus, test.wantStatus)
			}
		})
	}
}
