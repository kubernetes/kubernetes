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

	"github.com/google/go-cmp/cmp"
	"k8s.io/apimachinery/pkg/api/resource"

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
		description string
		pvc         func() *core.PersistentVolumeClaim
	}{
		{
			description: "pvc without DataSource",
			pvc:         pvcWithoutDataSource,
		},
		{
			description: "pvc with DataSource",
			pvc:         pvcWithDataSource,
		},
		{
			description: "is nil",
			pvc:         func() *core.PersistentVolumeClaim { return nil },
		},
	}

	for _, oldpvcInfo := range pvcInfo {
		for _, newpvcInfo := range pvcInfo {
			oldpvc := oldpvcInfo.pvc()
			newpvc := newpvcInfo.pvc()
			if newpvc == nil {
				continue
			}

			t.Run(fmt.Sprintf("old pvc %v, new pvc %v", oldpvcInfo.description, newpvcInfo.description), func(t *testing.T) {
				EnforceDataSourceBackwardsCompatibility(&newpvc.Spec, nil)

				// old pvc should never be changed
				if !reflect.DeepEqual(oldpvc, oldpvcInfo.pvc()) {
					t.Errorf("old pvc changed: %v", cmp.Diff(oldpvc, oldpvcInfo.pvc()))
				}

				// new pvc should not be changed
				if !reflect.DeepEqual(newpvc, newpvcInfo.pvc()) {
					t.Errorf("new pvc changed: %v", cmp.Diff(newpvc, newpvcInfo.pvc()))
				}
			})
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

	for testName, test := range tests {
		t.Run(testName, func(t *testing.T) {
			EnforceDataSourceBackwardsCompatibility(&test.spec, nil)
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

	var tests = map[string]struct {
		spec       core.PersistentVolumeClaimSpec
		anyEnabled bool
		want       *core.TypedLocalObjectReference
		wantRef    *core.TypedLocalObjectReference
	}{
		"any disabled with empty ds": {
			spec: core.PersistentVolumeClaimSpec{},
		},
		"any disabled with volume ds": {
			spec: core.PersistentVolumeClaimSpec{DataSource: volumeDataSource},
			want: volumeDataSource,
		},
		"any disabled with volume ds ref": {
			spec: core.PersistentVolumeClaimSpec{DataSourceRef: volumeDataSource},
		},
		"any disabled with both data sources": {
			spec: core.PersistentVolumeClaimSpec{DataSource: volumeDataSource, DataSourceRef: volumeDataSource},
			want: volumeDataSource,
		},
		"any enabled with empty ds": {
			spec:       core.PersistentVolumeClaimSpec{},
			anyEnabled: true,
		},
		"any enabled with volume ds": {
			spec:       core.PersistentVolumeClaimSpec{DataSource: volumeDataSource},
			anyEnabled: true,
			want:       volumeDataSource,
		},
		"any enabled with volume ds ref": {
			spec:       core.PersistentVolumeClaimSpec{DataSourceRef: volumeDataSource},
			anyEnabled: true,
			wantRef:    volumeDataSource,
		},
		"any enabled with both data sources": {
			spec:       core.PersistentVolumeClaimSpec{DataSource: volumeDataSource, DataSourceRef: volumeDataSource},
			anyEnabled: true,
			want:       volumeDataSource,
			wantRef:    volumeDataSource,
		},
	}

	for testName, test := range tests {
		t.Run(testName, func(t *testing.T) {
			defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.AnyVolumeDataSource, test.anyEnabled)()
			DropDisabledFields(&test.spec)
			if test.spec.DataSource != test.want || test.spec.DataSourceRef != test.wantRef {
				t.Errorf("expected condition was not met, test: %s, anyEnabled: %v, spec: %v, expected: %v %v",
					testName, test.anyEnabled, test.spec, test.want, test.wantRef)
			}
		})
	}
}

// TestDataSourceRef checks to ensure the DataSourceRef field handles backwards
// compatibility with the DataSource field
func TestDataSourceRef(t *testing.T) {
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
	coreDataSource := makeDataSource("", "Pod", "my-pod")

	var tests = map[string]struct {
		spec core.PersistentVolumeClaimSpec
		want *core.TypedLocalObjectReference
	}{
		"empty ds": {
			spec: core.PersistentVolumeClaimSpec{},
		},
		"volume ds": {
			spec: core.PersistentVolumeClaimSpec{DataSource: volumeDataSource},
			want: volumeDataSource,
		},
		"snapshot ds": {
			spec: core.PersistentVolumeClaimSpec{DataSource: snapshotDataSource},
			want: snapshotDataSource,
		},
		"generic ds": {
			spec: core.PersistentVolumeClaimSpec{DataSource: genericDataSource},
			want: genericDataSource,
		},
		"core ds": {
			spec: core.PersistentVolumeClaimSpec{DataSource: coreDataSource},
			want: coreDataSource,
		},
		"volume ds ref": {
			spec: core.PersistentVolumeClaimSpec{DataSourceRef: volumeDataSource},
			want: volumeDataSource,
		},
		"snapshot ds ref": {
			spec: core.PersistentVolumeClaimSpec{DataSourceRef: snapshotDataSource},
			want: snapshotDataSource,
		},
		"generic ds ref": {
			spec: core.PersistentVolumeClaimSpec{DataSourceRef: genericDataSource},
			want: genericDataSource,
		},
		"core ds ref": {
			spec: core.PersistentVolumeClaimSpec{DataSourceRef: coreDataSource},
			want: coreDataSource,
		},
	}

	defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.AnyVolumeDataSource, true)()

	for testName, test := range tests {
		t.Run(testName, func(t *testing.T) {
			NormalizeDataSources(&test.spec)
			if !reflect.DeepEqual(test.spec.DataSource, test.want) ||
				!reflect.DeepEqual(test.spec.DataSourceRef, test.want) {
				t.Errorf("expected condition was not met, test: %s, spec: %v, expected: %v",
					testName, test.spec, test.want)
			}
		})
	}
}

func TestDropDisabledFieldsFromStatus(t *testing.T) {
	tests := []struct {
		name     string
		feature  bool
		pvc      *core.PersistentVolumeClaim
		oldPVC   *core.PersistentVolumeClaim
		expected *core.PersistentVolumeClaim
	}{
		{
			name:     "for:newPVC=hasAllocatedResource,oldPVC=doesnot,featuregate=false; should drop field",
			feature:  false,
			pvc:      withAllocatedResource("5G"),
			oldPVC:   getPVC(),
			expected: getPVC(),
		},
		{
			name:     "for:newPVC=hasAllocatedResource,oldPVC=doesnot,featuregate=true; should keep field",
			feature:  true,
			pvc:      withAllocatedResource("5G"),
			oldPVC:   getPVC(),
			expected: withAllocatedResource("5G"),
		},
		{
			name:     "for:newPVC=hasAllocatedResource,oldPVC=hasAllocatedResource,featuregate=true; should keep field",
			feature:  true,
			pvc:      withAllocatedResource("5G"),
			oldPVC:   withAllocatedResource("5G"),
			expected: withAllocatedResource("5G"),
		},
		{
			name:     "for:newPVC=hasAllocatedResource,oldPVC=hasAllocatedResource,featuregate=false; should keep field",
			feature:  false,
			pvc:      withAllocatedResource("10G"),
			oldPVC:   withAllocatedResource("5G"),
			expected: withAllocatedResource("10G"),
		},
		{
			name:     "for:newPVC=hasAllocatedResource,oldPVC=nil,featuregate=false; should drop field",
			feature:  false,
			pvc:      withAllocatedResource("5G"),
			oldPVC:   nil,
			expected: getPVC(),
		},
		{
			name:     "for:newPVC=hasResizeStatus,oldPVC=nil, featuregate=false should drop field",
			feature:  false,
			pvc:      withResizeStatus(core.PersistentVolumeClaimNodeExpansionFailed),
			oldPVC:   nil,
			expected: getPVC(),
		},
		{
			name:     "for:newPVC=hasResizeStatus,oldPVC=doesnot,featuregate=true; should keep field",
			feature:  true,
			pvc:      withResizeStatus(core.PersistentVolumeClaimNodeExpansionFailed),
			oldPVC:   getPVC(),
			expected: withResizeStatus(core.PersistentVolumeClaimNodeExpansionFailed),
		},
		{
			name:     "for:newPVC=hasResizeStatus,oldPVC=hasResizeStatus,featuregate=true; should keep field",
			feature:  true,
			pvc:      withResizeStatus(core.PersistentVolumeClaimNodeExpansionFailed),
			oldPVC:   withResizeStatus(core.PersistentVolumeClaimNodeExpansionFailed),
			expected: withResizeStatus(core.PersistentVolumeClaimNodeExpansionFailed),
		},
		{
			name:     "for:newPVC=hasResizeStatus,oldPVC=hasResizeStatus,featuregate=false; should keep field",
			feature:  false,
			pvc:      withResizeStatus(core.PersistentVolumeClaimNodeExpansionFailed),
			oldPVC:   withResizeStatus(core.PersistentVolumeClaimNodeExpansionFailed),
			expected: withResizeStatus(core.PersistentVolumeClaimNodeExpansionFailed),
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.RecoverVolumeExpansionFailure, test.feature)()

			DropDisabledFieldsFromStatus(test.pvc, test.oldPVC)

			if !reflect.DeepEqual(*test.expected, *test.pvc) {
				t.Errorf("Unexpected change: %+v", cmp.Diff(test.expected, test.pvc))
			}
		})
	}
}

func getPVC() *core.PersistentVolumeClaim {
	return &core.PersistentVolumeClaim{}
}

func withAllocatedResource(q string) *core.PersistentVolumeClaim {
	return &core.PersistentVolumeClaim{
		Status: core.PersistentVolumeClaimStatus{
			AllocatedResources: core.ResourceList{
				core.ResourceStorage: resource.MustParse(q),
			},
		},
	}
}

func withResizeStatus(status core.PersistentVolumeClaimResizeStatus) *core.PersistentVolumeClaim {
	return &core.PersistentVolumeClaim{
		Status: core.PersistentVolumeClaimStatus{
			ResizeStatus: &status,
		},
	}
}
