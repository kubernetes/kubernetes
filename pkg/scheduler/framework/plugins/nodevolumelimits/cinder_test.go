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

package nodevolumelimits

import (
	"context"
	"reflect"
	"testing"

	v1 "k8s.io/api/core/v1"
	"k8s.io/kubernetes/pkg/scheduler/algorithm/predicates"
	framework "k8s.io/kubernetes/pkg/scheduler/framework/v1alpha1"
)

func TestCinderLimits(t *testing.T) {
	twoVolCinderPod := &v1.Pod{
		Spec: v1.PodSpec{
			Volumes: []v1.Volume{
				{
					VolumeSource: v1.VolumeSource{
						Cinder: &v1.CinderVolumeSource{VolumeID: "tvp1"},
					},
				},
				{
					VolumeSource: v1.VolumeSource{
						Cinder: &v1.CinderVolumeSource{VolumeID: "tvp2"},
					},
				},
			},
		},
	}
	oneVolCinderPod := &v1.Pod{
		Spec: v1.PodSpec{
			Volumes: []v1.Volume{
				{
					VolumeSource: v1.VolumeSource{
						Cinder: &v1.CinderVolumeSource{VolumeID: "ovp"},
					},
				},
			},
		},
	}

	tests := []struct {
		newPod       *v1.Pod
		existingPods []*v1.Pod
		filterName   string
		driverName   string
		maxVols      int
		test         string
		wantStatus   *framework.Status
	}{
		{
			newPod:       oneVolCinderPod,
			existingPods: []*v1.Pod{twoVolCinderPod},
			filterName:   predicates.CinderVolumeFilterType,
			maxVols:      4,
			test:         "fits when node capacity >= new pod's Cinder volumes",
		},
		{
			newPod:       oneVolCinderPod,
			existingPods: []*v1.Pod{twoVolCinderPod},
			filterName:   predicates.CinderVolumeFilterType,
			maxVols:      2,
			test:         "not fit when node capacity < new pod's Cinder volumes",
			wantStatus:   framework.NewStatus(framework.Unschedulable, predicates.ErrMaxVolumeCountExceeded.GetReason()),
		},
	}

	for _, test := range tests {
		t.Run(test.test, func(t *testing.T) {
			node, csiNode := getNodeWithPodAndVolumeLimits("node", test.existingPods, int64(test.maxVols), test.filterName)
			p := &CinderLimits{
				predicate: predicates.NewMaxPDVolumeCountPredicate(test.filterName, getFakeCSINodeLister(csiNode), getFakeCSIStorageClassLister(test.filterName, test.driverName), getFakePVLister(test.filterName), getFakePVCLister(test.filterName)),
			}
			gotStatus := p.Filter(context.Background(), nil, test.newPod, node)
			if !reflect.DeepEqual(gotStatus, test.wantStatus) {
				t.Errorf("status does not match: %v, want: %v", gotStatus, test.wantStatus)
			}
		})
	}
}
