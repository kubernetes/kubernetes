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

	"reflect"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	api "k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/kubernetes/pkg/features"

	// ensure types are installed
	_ "k8s.io/kubernetes/pkg/apis/core/install"
)

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
		fg          bool
		oldObj      *api.PersistentVolume
		newObj      *api.PersistentVolume
		expectedObj *api.PersistentVolume
	}{
		{
			name: "feature enabled: timestamp is updated when phase changes",
			fg:   true,
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
			name: "feature enabled: timestamp is updated when phase changes and old pv has a timestamp",
			fg:   true,
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
			name: "feature enabled: user timestamp change is respected on no phase change",
			fg:   true,
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
			name: "feature enabled: user timestamp is respected on phase change",
			fg:   true,
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
			name: "feature enabled: user timestamp change is respected on no phase change when old pv has a timestamp",
			fg:   true,
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
			name: "feature enabled: timestamp is updated when phase changes and both new and old timestamp matches",
			fg:   true,
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
		{
			name: "feature disabled: timestamp is not updated",
			fg:   false,
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
					Phase: api.VolumeBound,
				},
			},
		},
		{
			name: "feature disabled: user timestamp is overwritten on phase change to nil",
			fg:   false,
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
					LastPhaseTransitionTime: nil,
				},
			},
		},
		{
			name: "feature disabled: user timestamp change is respected on phase change",
			fg:   false,
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
			name: "feature disabled: user timestamp change is respected on no phase change",
			fg:   false,
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
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.PersistentVolumeLastPhaseTransitionTime, tc.fg)

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
		fg          bool
		newObj      *api.PersistentVolume
		expectedObj *api.PersistentVolume
	}{
		{
			name: "feature enabled: pv is in pending phase and has a timestamp",
			fg:   true,
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
		{
			name: "feature disabled: pv does not have phase and timestamp",
			fg:   false,
			newObj: &api.PersistentVolume{
				ObjectMeta: metav1.ObjectMeta{
					Name: "foo",
				},
			},
			expectedObj: &api.PersistentVolume{
				ObjectMeta: metav1.ObjectMeta{
					Name: "foo",
				},
			},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {

			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.PersistentVolumeLastPhaseTransitionTime, tc.fg)
			obj := tc.newObj.DeepCopy()
			StatusStrategy.PrepareForCreate(context.TODO(), obj)
			if !reflect.DeepEqual(obj, tc.expectedObj) {
				t.Errorf("object diff: %s", cmp.Diff(obj, tc.expectedObj))
			}
		})
	}
}
