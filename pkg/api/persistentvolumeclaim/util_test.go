/*
Copyright 2017 The Kubernetes Authors.

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

package persistentvolumeclaim

import (
	"fmt"
	"reflect"
	"testing"

	"k8s.io/apimachinery/pkg/api/resource"
	"k8s.io/apimachinery/pkg/util/diff"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/kubernetes/pkg/features"
)

func TestDropDisabledSnapshotDataSource(t *testing.T) {
	pvcWithoutDataSource := func() *core.PersistentVolumeClaim {
		return &core.PersistentVolumeClaim{
			Spec: core.PersistentVolumeClaimSpec{
				DataSource: nil,
			},
		}
	}
	apiGroup := "snapshot.storage.k8s.io"
	pvcWithDataSource := func() *core.PersistentVolumeClaim {
		return &core.PersistentVolumeClaim{
			Spec: core.PersistentVolumeClaimSpec{
				DataSource: &core.TypedLocalObjectReference{
					APIGroup: &apiGroup,
					Kind:     "VolumeSnapshot",
					Name:     "test_snapshot",
				},
			},
		}
	}

	pvcInfo := []struct {
		description   string
		hasDataSource bool
		pvc           func() *core.PersistentVolumeClaim
	}{
		{
			description:   "pvc without DataSource",
			hasDataSource: false,
			pvc:           pvcWithoutDataSource,
		},
		{
			description:   "pvc with DataSource",
			hasDataSource: true,
			pvc:           pvcWithDataSource,
		},
		{
			description:   "is nil",
			hasDataSource: false,
			pvc:           func() *core.PersistentVolumeClaim { return nil },
		},
	}

	// Ensure that any data sources aren't enabled for this test
	defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.AnyVolumeDataSource, false)()

	for _, enabled := range []bool{true, false} {
		for _, oldpvcInfo := range pvcInfo {
			for _, newpvcInfo := range pvcInfo {
				oldPvcHasDataSource, oldpvc := oldpvcInfo.hasDataSource, oldpvcInfo.pvc()
				newPvcHasDataSource, newpvc := newpvcInfo.hasDataSource, newpvcInfo.pvc()
				if newpvc == nil {
					continue
				}

				t.Run(fmt.Sprintf("feature enabled=%v, old pvc %v, new pvc %v", enabled, oldpvcInfo.description, newpvcInfo.description), func(t *testing.T) {
					defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.VolumeSnapshotDataSource, enabled)()

					var oldpvcSpec *core.PersistentVolumeClaimSpec
					if oldpvc != nil {
						oldpvcSpec = &oldpvc.Spec
					}
					DropDisabledFields(&newpvc.Spec, oldpvcSpec)

					// old pvc should never be changed
					if !reflect.DeepEqual(oldpvc, oldpvcInfo.pvc()) {
						t.Errorf("old pvc changed: %v", diff.ObjectReflectDiff(oldpvc, oldpvcInfo.pvc()))
					}

					switch {
					case enabled || oldPvcHasDataSource:
						// new pvc should not be changed if the feature is enabled, or if the old pvc had DataSource
						if !reflect.DeepEqual(newpvc, newpvcInfo.pvc()) {
							t.Errorf("new pvc changed: %v", diff.ObjectReflectDiff(newpvc, newpvcInfo.pvc()))
						}
					case newPvcHasDataSource:
						// new pvc should be changed
						if reflect.DeepEqual(newpvc, newpvcInfo.pvc()) {
							t.Errorf("new pvc was not changed")
						}
						// new pvc should not have DataSource
						if !reflect.DeepEqual(newpvc, pvcWithoutDataSource()) {
							t.Errorf("new pvc had DataSource: %v", diff.ObjectReflectDiff(newpvc, pvcWithoutDataSource()))
						}
					default:
						// new pvc should not need to be changed
						if !reflect.DeepEqual(newpvc, newpvcInfo.pvc()) {
							t.Errorf("new pvc changed: %v", diff.ObjectReflectDiff(newpvc, newpvcInfo.pvc()))
						}
					}
				})
			}
		}
	}
}

// TestPVCDataSourceSpecFilter checks to ensure the DropDisabledFields function behaves correctly for PVCDataSource featuregate
func TestPVCDataSourceSpecFilter(t *testing.T) {
	apiGroup := ""
	validSpec := core.PersistentVolumeClaimSpec{
		DataSource: &core.TypedLocalObjectReference{
			APIGroup: &apiGroup,
			Kind:     "PersistentVolumeClaim",
			Name:     "test_clone",
		},
	}
	validSpecNilAPIGroup := core.PersistentVolumeClaimSpec{
		DataSource: &core.TypedLocalObjectReference{
			Kind: "PersistentVolumeClaim",
			Name: "test_clone",
		},
	}

	invalidAPIGroup := "invalid.pvc.api.group"
	invalidSpec := core.PersistentVolumeClaimSpec{
		DataSource: &core.TypedLocalObjectReference{
			APIGroup: &invalidAPIGroup,
			Kind:     "PersistentVolumeClaim",
			Name:     "test_clone_invalid",
		},
	}

	var tests = map[string]struct {
		spec core.PersistentVolumeClaimSpec
		want *core.TypedLocalObjectReference
	}{
		"enabled with empty ds": {
			spec: core.PersistentVolumeClaimSpec{},
			want: nil,
		},
		"enabled with invalid spec": {
			spec: invalidSpec,
			want: nil,
		},
		"enabled with valid spec": {
			spec: validSpec,
			want: validSpec.DataSource,
		},
		"enabled with valid spec but nil APIGroup": {
			spec: validSpecNilAPIGroup,
			want: validSpecNilAPIGroup.DataSource,
		},
	}

	// Ensure that any data sources aren't enabled for this test
	defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.AnyVolumeDataSource, false)()

	for testName, test := range tests {
		t.Run(testName, func(t *testing.T) {
			DropDisabledFields(&test.spec, nil)
			if test.spec.DataSource != test.want {
				t.Errorf("expected drop datasource condition was not met, test: %s, spec: %v, expected: %v", testName, test.spec, test.want)
			}

		})

	}

}

// TestAnyDataSourceFilter checks to ensure the AnyVolumeDataSource feature gate works
func TestAnyDataSourceFilter(t *testing.T) {
	makeDataSource := func(apiGroup, kind, name string) *core.TypedLocalObjectReference {
		return &core.TypedLocalObjectReference{
			APIGroup: &apiGroup,
			Kind:     kind,
			Name:     name,
		}
	}

	volumeDataSource := makeDataSource("", "PersistentVolumeClaim", "my-vol")
	snapshotDataSource := makeDataSource("snapshot.storage.k8s.io", "VolumeSnapshot", "my-snap")
	genericDataSource := makeDataSource("generic.storage.k8s.io", "Generic", "my-foo")

	var tests = map[string]struct {
		spec            core.PersistentVolumeClaimSpec
		snapshotEnabled bool
		anyEnabled      bool
		want            *core.TypedLocalObjectReference
	}{
		"both disabled with empty ds": {
			spec: core.PersistentVolumeClaimSpec{},
			want: nil,
		},
		"both disabled with volume ds": {
			spec: core.PersistentVolumeClaimSpec{DataSource: volumeDataSource},
			want: volumeDataSource,
		},
		"both disabled with snapshot ds": {
			spec: core.PersistentVolumeClaimSpec{DataSource: snapshotDataSource},
			want: nil,
		},
		"both disabled with generic ds": {
			spec: core.PersistentVolumeClaimSpec{DataSource: genericDataSource},
			want: nil,
		},
		"any enabled with empty ds": {
			spec:       core.PersistentVolumeClaimSpec{},
			anyEnabled: true,
			want:       nil,
		},
		"any enabled with volume ds": {
			spec:       core.PersistentVolumeClaimSpec{DataSource: volumeDataSource},
			anyEnabled: true,
			want:       volumeDataSource,
		},
		"any enabled with snapshot ds": {
			spec:       core.PersistentVolumeClaimSpec{DataSource: snapshotDataSource},
			anyEnabled: true,
			want:       snapshotDataSource,
		},
		"any enabled with generic ds": {
			spec:       core.PersistentVolumeClaimSpec{DataSource: genericDataSource},
			anyEnabled: true,
			want:       genericDataSource,
		},
		"snapshot enabled with snapshot ds": {
			spec:            core.PersistentVolumeClaimSpec{DataSource: snapshotDataSource},
			snapshotEnabled: true,
			want:            snapshotDataSource,
		},
		"snapshot enabled with generic ds": {
			spec:            core.PersistentVolumeClaimSpec{DataSource: genericDataSource},
			snapshotEnabled: true,
			want:            nil,
		},
		"both enabled with empty ds": {
			spec:            core.PersistentVolumeClaimSpec{},
			snapshotEnabled: true,
			anyEnabled:      true,
			want:            nil,
		},
		"both enabled with volume ds": {
			spec:            core.PersistentVolumeClaimSpec{DataSource: volumeDataSource},
			snapshotEnabled: true,
			anyEnabled:      true,
			want:            volumeDataSource,
		},
		"both enabled with snapshot ds": {
			spec:            core.PersistentVolumeClaimSpec{DataSource: snapshotDataSource},
			snapshotEnabled: true,
			anyEnabled:      true,
			want:            snapshotDataSource,
		},
		"both enabled with generic ds": {
			spec:            core.PersistentVolumeClaimSpec{DataSource: genericDataSource},
			snapshotEnabled: true,
			anyEnabled:      true,
			want:            genericDataSource,
		},
	}

	for testName, test := range tests {
		t.Run(testName, func(t *testing.T) {
			defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.VolumeSnapshotDataSource, test.snapshotEnabled)()
			defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.AnyVolumeDataSource, test.anyEnabled)()
			DropDisabledFields(&test.spec, nil)
			if test.spec.DataSource != test.want {
				t.Errorf("expected condition was not met, test: %s, snapshotEnabled: %v, anyEnabled: %v, spec: %v, expected: %v",
					testName, test.snapshotEnabled, test.anyEnabled, test.spec, test.want)
			}
		})
	}
}

func TestDropAllocatedResources(t *testing.T) {
	tests := []struct {
		name     string
		feature  bool
		spec     *core.PersistentVolumeClaimSpec
		oldSpec  *core.PersistentVolumeClaimSpec
		expected *core.PersistentVolumeClaimSpec
	}{
		{
			name:     "for:newPVC=hasfield,oldPVC=doesnot,featuregate=false; should drop field",
			feature:  false,
			spec:     withAllocatedResource("5G"),
			oldSpec:  getPVCSpec(),
			expected: getPVCSpec(),
		},
		{
			name:     "for:newPVC=hasfield,oldPVC=doesnot,featuregate=true; should keep field",
			feature:  true,
			spec:     withAllocatedResource("5G"),
			oldSpec:  getPVCSpec(),
			expected: withAllocatedResource("5G"),
		},
		{
			name:     "for:newPVC=hasfield,oldPVC=hasfield,featuregate=false; should keep field",
			feature:  false,
			spec:     withAllocatedResource("10G"),
			oldSpec:  withAllocatedResource("5G"),
			expected: withAllocatedResource("10G"),
		},
		{
			name:     "for:newPVC=hasfield,oldPVC=nil,featuregate=false; should drop field",
			feature:  false,
			spec:     withAllocatedResource("5G"),
			oldSpec:  nil,
			expected: getPVCSpec(),
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.RecoverVolumeExpansionFailure, test.feature)()

			DropDisabledFields(test.spec, test.oldSpec)

			if !reflect.DeepEqual(*test.expected, *test.spec) {
				t.Errorf("Unexpected change: %+v", diff.ObjectDiff(test.expected, test.spec))
			}
		})
	}
}

func TestSetAllocatedResources(t *testing.T) {
	tests := []struct {
		name     string
		feature  bool
		spec     *core.PersistentVolumeClaimSpec
		oldSpec  *core.PersistentVolumeClaimSpec
		expected *core.PersistentVolumeClaimSpec
	}{
		{
			name:     "feature:true; should default to requested size when creating new PVC",
			feature:  true,
			spec:     withResource(getPVCSpec(), "5G"),
			oldSpec:  nil,
			expected: withResource(withAllocatedResource("5G"), "5G"),
		},
		{
			name:     "feature:true; should default to older allocated size if updated PVC is smaller",
			feature:  true,
			spec:     withResource(getPVCSpec(), "5G"),
			oldSpec:  withResource(withAllocatedResource("10G"), "10G"),
			expected: withResource(withAllocatedResource("10G"), "5G"),
		},
		{
			name:     "feature:true; should default to higher allocated size if older allocated size is small",
			feature:  true,
			spec:     withResource(getPVCSpec(), "5G"),
			oldSpec:  withResource(withAllocatedResource("4G"), "4G"),
			expected: withResource(withAllocatedResource("5G"), "5G"),
		},
		{
			name:     "feature:true; should default to older allocated size even if new allocated size is bigger",
			feature:  true,
			spec:     withResource(withAllocatedResource("30G"), "20G"),
			oldSpec:  withResource(withAllocatedResource("20G"), "20G"),
			expected: withResource(withAllocatedResource("20G"), "20G"),
		},
		{
			name:     "feature:true; should default to older resource size if older resource size is bigger and old Allocatedresources is nil",
			feature:  true,
			spec:     withResource(withAllocatedResource("10G"), "20G"),
			oldSpec:  withResource(getPVCSpec(), "40G"),
			expected: withResource(withAllocatedResource("40G"), "20G"),
		},
		{
			name:     "feature:true; should default to newer resource size if older resources size is smaller and old allocatedresources is nil",
			feature:  true,
			spec:     withResource(withAllocatedResource("10G"), "50G"),
			oldSpec:  withResource(getPVCSpec(), "40G"),
			expected: withResource(withAllocatedResource("50G"), "50G"),
		},
		{
			name:     "feature:true, should default to older allocated size if older allocated size is bigger than old resource size",
			feature:  true,
			spec:     withResource(withAllocatedResource("10G"), "5G"),
			oldSpec:  withResource(withAllocatedResource("20G"), "10G"),
			expected: withResource(withAllocatedResource("20G"), "5G"),
		},
		{
			name:     "feature:true; should default to new resource size if older resource size is smaller",
			feature:  true,
			spec:     withResource(withAllocatedResource("30G"), "30G"),
			oldSpec:  withResource(withAllocatedResource("20G"), "20G"),
			expected: withResource(withAllocatedResource("30G"), "30G"),
		},
		{
			name:     "feature:true should default to higher allocated size even if updated pvc already had a allocated size",
			feature:  true,
			spec:     withResource(withAllocatedResource("10G"), "20G"),
			oldSpec:  withResource(withAllocatedResource("10G"), "10G"),
			expected: withResource(withAllocatedResource("20G"), "20G"),
		},
		{
			name:     "feature:false; should not set allocatedSize when creating new PVC",
			feature:  false,
			spec:     withResource(getPVCSpec(), "5G"),
			oldSpec:  nil,
			expected: withResource(getPVCSpec(), "5G"),
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.RecoverVolumeExpansionFailure, test.feature)()

			SetAllocatedResources(test.spec, test.oldSpec)

			if !reflect.DeepEqual(*test.expected, *test.spec) {
				t.Errorf("Unexpected change: %+v", diff.ObjectDiff(test.expected, test.spec))
			}
		})
	}
}

func getPVCSpec() *core.PersistentVolumeClaimSpec {
	return &core.PersistentVolumeClaimSpec{}
}

func withResource(s *core.PersistentVolumeClaimSpec, q string) *core.PersistentVolumeClaimSpec {
	sc := s.DeepCopy()
	sc.Resources = core.ResourceRequirements{
		Requests: core.ResourceList{
			core.ResourceStorage: resource.MustParse(q),
		},
	}
	return sc
}

func withAllocatedResource(q string) *core.PersistentVolumeClaimSpec {
	return &core.PersistentVolumeClaimSpec{
		AllocatedResources: &core.ResourceRequirements{
			Requests: core.ResourceList{
				core.ResourceStorage: resource.MustParse(q),
			},
		},
	}
}
