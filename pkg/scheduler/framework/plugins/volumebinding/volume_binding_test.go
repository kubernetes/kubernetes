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
	"fmt"
	"reflect"
	"testing"

	v1 "k8s.io/api/core/v1"
	volumescheduling "k8s.io/kubernetes/pkg/controller/volume/scheduling"
	framework "k8s.io/kubernetes/pkg/scheduler/framework/v1alpha1"
	schedulernodeinfo "k8s.io/kubernetes/pkg/scheduler/nodeinfo"
	"k8s.io/kubernetes/pkg/scheduler/volumebinder"
)

func TestVolumeBinding(t *testing.T) {
	findErr := fmt.Errorf("find err")
	volState := v1.PodSpec{
		Volumes: []v1.Volume{
			{
				VolumeSource: v1.VolumeSource{
					PersistentVolumeClaim: &v1.PersistentVolumeClaimVolumeSource{},
				},
			},
		},
	}
	table := []struct {
		name               string
		pod                *v1.Pod
		node               *v1.Node
		volumeBinderConfig *volumescheduling.FakeVolumeBinderConfig
		wantStatus         *framework.Status
	}{
		{
			name:       "nothing",
			pod:        &v1.Pod{},
			node:       &v1.Node{},
			wantStatus: nil,
		},
		{
			name: "all bound",
			pod:  &v1.Pod{Spec: volState},
			node: &v1.Node{},
			volumeBinderConfig: &volumescheduling.FakeVolumeBinderConfig{
				AllBound:             true,
				FindUnboundSatsified: true,
				FindBoundSatsified:   true,
			},
			wantStatus: nil,
		},
		{
			name: "unbound/no matches",
			pod:  &v1.Pod{Spec: volState},
			node: &v1.Node{},
			volumeBinderConfig: &volumescheduling.FakeVolumeBinderConfig{
				FindUnboundSatsified: false,
				FindBoundSatsified:   true,
			},
			wantStatus: framework.NewStatus(framework.UnschedulableAndUnresolvable, ErrReasonBindConflict),
		},
		{
			name: "bound and unbound unsatisfied",
			pod:  &v1.Pod{Spec: volState},
			node: &v1.Node{},
			volumeBinderConfig: &volumescheduling.FakeVolumeBinderConfig{
				FindUnboundSatsified: false,
				FindBoundSatsified:   false,
			},
			wantStatus: framework.NewStatus(framework.UnschedulableAndUnresolvable, ErrReasonNodeConflict,
				ErrReasonBindConflict),
		},
		{
			name: "unbound/found matches/bind succeeds",
			pod:  &v1.Pod{Spec: volState},
			node: &v1.Node{},
			volumeBinderConfig: &volumescheduling.FakeVolumeBinderConfig{
				FindUnboundSatsified: true,
				FindBoundSatsified:   true,
			},
			wantStatus: nil,
		},
		{
			name: "predicate error",
			pod:  &v1.Pod{Spec: volState},
			node: &v1.Node{},
			volumeBinderConfig: &volumescheduling.FakeVolumeBinderConfig{
				FindErr: findErr,
			},
			wantStatus: framework.NewStatus(framework.Error, findErr.Error()),
		},
	}

	for _, item := range table {
		t.Run(item.name, func(t *testing.T) {
			nodeInfo := schedulernodeinfo.NewNodeInfo()
			nodeInfo.SetNode(item.node)
			fakeVolumeBinder := volumebinder.NewFakeVolumeBinder(item.volumeBinderConfig)
			p := &VolumeBinding{
				binder: fakeVolumeBinder,
			}
			gotStatus := p.Filter(context.Background(), nil, item.pod, nodeInfo)
			if !reflect.DeepEqual(gotStatus, item.wantStatus) {
				t.Errorf("status does not match: %v, want: %v", gotStatus, item.wantStatus)
			}

		})
	}
}
