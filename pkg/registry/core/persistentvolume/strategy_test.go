/*
Copyright 2015 The Kubernetes Authors.

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

package persistentvolume

import (
	"context"

	"github.com/google/go-cmp/cmp"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	apitesting "k8s.io/kubernetes/pkg/api/testing"
	api "k8s.io/kubernetes/pkg/apis/core"
	"reflect"
	"testing"
	"time"

	// ensure types are installed
	_ "k8s.io/kubernetes/pkg/apis/core/install"
)

func TestSelectableFieldLabelConversions(t *testing.T) {
	apitesting.TestSelectableFieldLabelConversionsOfKind(t,
		"v1",
		"PersistentVolume",
		PersistentVolumeToSelectableFields(&api.PersistentVolume{}),
		map[string]string{"name": "metadata.name"},
	)
}

func TestStatusUpdate(t *testing.T) {
	now := metav1.Now()
	origin := metav1.NewTime(now.Add(time.Hour))
	later := metav1.NewTime(now.Add(time.Hour * 2))
	NowFunc = func() metav1.Time { return now }
	defer func() {
		NowFunc = metav1.Now
	}()
	tests := []struct {
		name        string
		oldObj      *api.PersistentVolume
		newObj      *api.PersistentVolume
		expectedObj *api.PersistentVolume
	}{
		{
			name: "timestamp is updated when phase changes",
			oldObj: &api.PersistentVolume{
				ObjectMeta: metav1.ObjectMeta{
					Name: "foo",
				},
				Status: api.PersistentVolumeStatus{
					Phase: api.VolumePending,
				},
			},
			newObj: &api.PersistentVolume{
				ObjectMeta: metav1.ObjectMeta{
					Name: "foo",
				},
				Status: api.PersistentVolumeStatus{
					Phase: api.VolumeBound,
				},
			},
			expectedObj: &api.PersistentVolume{
				ObjectMeta: metav1.ObjectMeta{
					Name: "foo",
				},
				Status: api.PersistentVolumeStatus{
					Phase:                   api.VolumeBound,
					LastPhaseTransitionTime: &now,
				},
			},
		},
		{
			name: "timestamp is updated when phase changes and old pv has a timestamp",
			oldObj: &api.PersistentVolume{
				ObjectMeta: metav1.ObjectMeta{
					Name: "foo",
				},
				Status: api.PersistentVolumeStatus{
					Phase:                   api.VolumePending,
					LastPhaseTransitionTime: &origin,
				},
			},
			newObj: &api.PersistentVolume{
				ObjectMeta: metav1.ObjectMeta{
					Name: "foo",
				},
				Status: api.PersistentVolumeStatus{
					Phase: api.VolumeBound,
				},
			},
			expectedObj: &api.PersistentVolume{
				ObjectMeta: metav1.ObjectMeta{
					Name: "foo",
				},
				Status: api.PersistentVolumeStatus{
					Phase:                   api.VolumeBound,
					LastPhaseTransitionTime: &now,
				},
			},
		},
		{
			name: "user timestamp change is respected on no phase change",
			oldObj: &api.PersistentVolume{
				ObjectMeta: metav1.ObjectMeta{
					Name: "foo",
				},
				Status: api.PersistentVolumeStatus{
					Phase: api.VolumePending,
				},
			},
			newObj: &api.PersistentVolume{
				ObjectMeta: metav1.ObjectMeta{
					Name: "foo",
				},
				Status: api.PersistentVolumeStatus{
					Phase:                   api.VolumePending,
					LastPhaseTransitionTime: &later,
				},
			},
			expectedObj: &api.PersistentVolume{
				ObjectMeta: metav1.ObjectMeta{
					Name: "foo",
				},
				Status: api.PersistentVolumeStatus{
					Phase:                   api.VolumePending,
					LastPhaseTransitionTime: &later,
				},
			},
		},
		{
			name: "user timestamp is respected on phase change",
			oldObj: &api.PersistentVolume{
				ObjectMeta: metav1.ObjectMeta{
					Name: "foo",
				},
				Status: api.PersistentVolumeStatus{
					Phase:                   api.VolumePending,
					LastPhaseTransitionTime: &origin,
				},
			},
			newObj: &api.PersistentVolume{
				ObjectMeta: metav1.ObjectMeta{
					Name: "foo",
				},
				Status: api.PersistentVolumeStatus{
					Phase:                   api.VolumeBound,
					LastPhaseTransitionTime: &later,
				},
			},
			expectedObj: &api.PersistentVolume{
				ObjectMeta: metav1.ObjectMeta{
					Name: "foo",
				},
				Status: api.PersistentVolumeStatus{
					Phase:                   api.VolumeBound,
					LastPhaseTransitionTime: &later,
				},
			},
		},
		{
			name: "user timestamp change is respected on no phase change when old pv has a timestamp",
			oldObj: &api.PersistentVolume{
				ObjectMeta: metav1.ObjectMeta{
					Name: "foo",
				},
				Status: api.PersistentVolumeStatus{
					Phase:                   api.VolumeBound,
					LastPhaseTransitionTime: &origin,
				},
			},
			newObj: &api.PersistentVolume{
				ObjectMeta: metav1.ObjectMeta{
					Name: "foo",
				},
				Status: api.PersistentVolumeStatus{
					Phase:                   api.VolumeBound,
					LastPhaseTransitionTime: &later,
				},
			},
			expectedObj: &api.PersistentVolume{
				ObjectMeta: metav1.ObjectMeta{
					Name: "foo",
				},
				Status: api.PersistentVolumeStatus{
					Phase:                   api.VolumeBound,
					LastPhaseTransitionTime: &later,
				},
			},
		},
		{
			name: "timestamp is updated when phase changes and both new and old timestamp matches",
			oldObj: &api.PersistentVolume{
				ObjectMeta: metav1.ObjectMeta{
					Name: "foo",
				},
				Status: api.PersistentVolumeStatus{
					Phase:                   api.VolumePending,
					LastPhaseTransitionTime: &origin,
				},
			},
			newObj: &api.PersistentVolume{
				ObjectMeta: metav1.ObjectMeta{
					Name: "foo",
				},
				Status: api.PersistentVolumeStatus{
					Phase:                   api.VolumeBound,
					LastPhaseTransitionTime: &origin,
				},
			},
			expectedObj: &api.PersistentVolume{
				ObjectMeta: metav1.ObjectMeta{
					Name: "foo",
				},
				Status: api.PersistentVolumeStatus{
					Phase:                   api.VolumeBound,
					LastPhaseTransitionTime: &now,
				},
			},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {

			obj := tc.newObj.DeepCopy()
			StatusStrategy.PrepareForUpdate(context.TODO(), obj, tc.oldObj.DeepCopy())
			if !reflect.DeepEqual(obj, tc.expectedObj) {
				t.Errorf("object diff: %s", cmp.Diff(obj, tc.expectedObj))
			}
		})
	}
}

func TestStatusCreate(t *testing.T) {
	now := metav1.Now()
	NowFunc = func() metav1.Time { return now }
	defer func() {
		NowFunc = metav1.Now
	}()
	tests := []struct {
		name        string
		newObj      *api.PersistentVolume
		expectedObj *api.PersistentVolume
	}{
		{
			name: "pv is in pending phase and has a timestamp",
			newObj: &api.PersistentVolume{
				ObjectMeta: metav1.ObjectMeta{
					Name: "foo",
				},
			},
			expectedObj: &api.PersistentVolume{
				ObjectMeta: metav1.ObjectMeta{
					Name: "foo",
				},
				Status: api.PersistentVolumeStatus{
					Phase:                   api.VolumePending,
					LastPhaseTransitionTime: &now,
				},
			},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {

			obj := tc.newObj.DeepCopy()
			StatusStrategy.PrepareForCreate(context.TODO(), obj)
			if !reflect.DeepEqual(obj, tc.expectedObj) {
				t.Errorf("object diff: %s", cmp.Diff(obj, tc.expectedObj))
			}
		})
	}
}
