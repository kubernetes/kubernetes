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

package persistentvolumeclaim

import (
	"fmt"
	"reflect"
	"testing"

	"k8s.io/apimachinery/pkg/util/diff"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	apitesting "k8s.io/kubernetes/pkg/api/testing"
	api "k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/kubernetes/pkg/features"

	// ensure types are installed
	_ "k8s.io/kubernetes/pkg/apis/core/install"
)

func TestSelectableFieldLabelConversions(t *testing.T) {
	apitesting.TestSelectableFieldLabelConversionsOfKind(t,
		"v1",
		"PersistentVolumeClaim",
		PersistentVolumeClaimToSelectableFields(&api.PersistentVolumeClaim{}),
		map[string]string{"name": "metadata.name"},
	)
}

func TestDropConditions(t *testing.T) {
	ctx := genericapirequest.NewDefaultContext()
	pvcWithConditions := func() *api.PersistentVolumeClaim {
		return &api.PersistentVolumeClaim{
			Status: api.PersistentVolumeClaimStatus{
				Conditions: []api.PersistentVolumeClaimCondition{
					{Type: api.PersistentVolumeClaimResizing, Status: api.ConditionTrue},
				},
			},
		}
	}
	pvcWithoutConditions := func() *api.PersistentVolumeClaim {
		return &api.PersistentVolumeClaim{
			Status: api.PersistentVolumeClaimStatus{},
		}
	}

	pvcInfo := []struct {
		description   string
		hasConditions bool
		pvc           func() *api.PersistentVolumeClaim
	}{
		{
			description:   "has Conditions",
			hasConditions: true,
			pvc:           pvcWithConditions,
		},
		{
			description:   "does not have Conditions",
			hasConditions: false,
			pvc:           pvcWithoutConditions,
		},
	}

	for _, enabled := range []bool{true} {
		for _, oldPvcInfo := range pvcInfo {
			for _, newPvcInfo := range pvcInfo {
				oldPvcHasConditins, oldPvc := oldPvcInfo.hasConditions, oldPvcInfo.pvc()
				newPvcHasConditions, newPvc := newPvcInfo.hasConditions, newPvcInfo.pvc()

				t.Run(fmt.Sprintf("feature enabled=%v, old pvc %v, new pvc %v", enabled, oldPvcInfo.description, newPvcInfo.description), func(t *testing.T) {
					StatusStrategy.PrepareForUpdate(ctx, newPvc, oldPvc)

					// old pvc should never be changed
					if !reflect.DeepEqual(oldPvc, oldPvcInfo.pvc()) {
						t.Errorf("old pvc changed: %v", diff.ObjectReflectDiff(oldPvc, oldPvcInfo.pvc()))
					}

					switch {
					case enabled || oldPvcHasConditins:
						// new pvc should not be changed if the feature is enabled, or if the old pvc had Conditions
						if !reflect.DeepEqual(newPvc, newPvcInfo.pvc()) {
							t.Errorf("new pvc changed: %v", diff.ObjectReflectDiff(newPvc, newPvcInfo.pvc()))
						}
					case newPvcHasConditions:
						// new pvc should be changed
						if reflect.DeepEqual(newPvc, newPvcInfo.pvc()) {
							t.Errorf("new pvc was not changed")
						}
						// new pvc should not have Conditions
						if !reflect.DeepEqual(newPvc, pvcWithoutConditions()) {
							t.Errorf("new pvc had Conditions: %v", diff.ObjectReflectDiff(newPvc, pvcWithoutConditions()))
						}
					default:
						// new pvc should not need to be changed
						if !reflect.DeepEqual(newPvc, newPvcInfo.pvc()) {
							t.Errorf("new pvc changed: %v", diff.ObjectReflectDiff(newPvc, newPvcInfo.pvc()))
						}
					}
				})
			}
		}
	}
}

func TestPrepareForCreate(t *testing.T) {
	ctx := genericapirequest.NewDefaultContext()

	makeDataSource := func(apiGroup, kind, name string) *api.TypedLocalObjectReference {
		return &api.TypedLocalObjectReference{
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
		anyEnabled    bool
		dataSource    *api.TypedLocalObjectReference
		dataSourceRef *api.TypedLocalObjectReference
		want          *api.TypedLocalObjectReference
		wantRef       *api.TypedLocalObjectReference
	}{
		"any disabled with empty ds": {
			want: nil,
		},
		"any disabled with volume ds": {
			dataSource: volumeDataSource,
			want:       volumeDataSource,
		},
		"any disabled with snapshot ds": {
			dataSource: snapshotDataSource,
			want:       snapshotDataSource,
		},
		"any disabled with generic ds": {
			dataSource: genericDataSource,
			want:       nil,
		},
		"any disabled with invalid ds": {
			dataSource: coreDataSource,
			want:       nil,
		},
		"any disabled with volume ds ref": {
			dataSourceRef: volumeDataSource,
		},
		"any disabled with snapshot ds ref": {
			dataSourceRef: snapshotDataSource,
		},
		"any disabled with generic ds ref": {
			dataSourceRef: genericDataSource,
		},
		"any disabled with invalid ds ref": {
			dataSourceRef: coreDataSource,
		},
		"any enabled with empty ds": {
			anyEnabled: true,
			want:       nil,
		},
		"any enabled with volume ds": {
			dataSource: volumeDataSource,
			anyEnabled: true,
			want:       volumeDataSource,
			wantRef:    volumeDataSource,
		},
		"any enabled with snapshot ds": {
			dataSource: snapshotDataSource,
			anyEnabled: true,
			want:       snapshotDataSource,
			wantRef:    snapshotDataSource,
		},
		"any enabled with generic ds": {
			dataSource: genericDataSource,
			anyEnabled: true,
		},
		"any enabled with invalid ds": {
			dataSource: coreDataSource,
			anyEnabled: true,
		},
		"any enabled with volume ds ref": {
			dataSourceRef: volumeDataSource,
			anyEnabled:    true,
			want:          volumeDataSource,
			wantRef:       volumeDataSource,
		},
		"any enabled with snapshot ds ref": {
			dataSourceRef: snapshotDataSource,
			anyEnabled:    true,
			want:          snapshotDataSource,
			wantRef:       snapshotDataSource,
		},
		"any enabled with generic ds ref": {
			dataSourceRef: genericDataSource,
			anyEnabled:    true,
			want:          genericDataSource,
			wantRef:       genericDataSource,
		},
		"any enabled with invalid ds ref": {
			dataSourceRef: coreDataSource,
			anyEnabled:    true,
			want:          coreDataSource,
			wantRef:       coreDataSource,
		},
		"any enabled with mismatched data sources": {
			dataSource:    volumeDataSource,
			dataSourceRef: snapshotDataSource,
			anyEnabled:    true,
			want:          volumeDataSource,
			wantRef:       snapshotDataSource,
		},
	}

	for testName, test := range tests {
		t.Run(testName, func(t *testing.T) {
			defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.AnyVolumeDataSource, test.anyEnabled)()
			pvc := api.PersistentVolumeClaim{
				Spec: api.PersistentVolumeClaimSpec{
					DataSource:    test.dataSource,
					DataSourceRef: test.dataSourceRef,
				},
			}

			// Method under test
			Strategy.PrepareForCreate(ctx, &pvc)

			if !reflect.DeepEqual(pvc.Spec.DataSource, test.want) {
				t.Errorf("data source does not match, test: %s, anyEnabled: %v, dataSource: %v, expected: %v",
					testName, test.anyEnabled, test.dataSource, test.want)
			}
			if !reflect.DeepEqual(pvc.Spec.DataSourceRef, test.wantRef) {
				t.Errorf("data source ref does not match, test: %s, anyEnabled: %v, dataSourceRef: %v, expected: %v",
					testName, test.anyEnabled, test.dataSourceRef, test.wantRef)
			}
		})
	}
}
