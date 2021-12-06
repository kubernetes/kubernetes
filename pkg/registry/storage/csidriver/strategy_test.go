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

package csidriver

import (
	"testing"

	"github.com/stretchr/testify/require"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/validation/field"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/kubernetes/pkg/apis/storage"
	"k8s.io/kubernetes/pkg/features"
)

func getValidCSIDriver(name string) *storage.CSIDriver {
	enabled := true
	return &storage.CSIDriver{
		ObjectMeta: metav1.ObjectMeta{
			Name: name,
		},
		Spec: storage.CSIDriverSpec{
			AttachRequired:    &enabled,
			PodInfoOnMount:    &enabled,
			StorageCapacity:   &enabled,
			RequiresRepublish: &enabled,
		},
	}
}

func TestCSIDriverStrategy(t *testing.T) {
	ctx := genericapirequest.WithRequestInfo(genericapirequest.NewContext(), &genericapirequest.RequestInfo{
		APIGroup:   "storage.k8s.io",
		APIVersion: "v1",
		Resource:   "csidrivers",
	})
	if Strategy.NamespaceScoped() {
		t.Errorf("CSIDriver must not be namespace scoped")
	}
	if Strategy.AllowCreateOnUpdate() {
		t.Errorf("CSIDriver should not allow create on update")
	}

	csiDriver := getValidCSIDriver("valid-csidriver")

	Strategy.PrepareForCreate(ctx, csiDriver)

	errs := Strategy.Validate(ctx, csiDriver)
	if len(errs) != 0 {
		t.Errorf("unexpected error validating %v", errs)
	}

	// Update of spec is disallowed
	newCSIDriver := csiDriver.DeepCopy()
	attachNotRequired := false
	newCSIDriver.Spec.AttachRequired = &attachNotRequired

	Strategy.PrepareForUpdate(ctx, newCSIDriver, csiDriver)

	errs = Strategy.ValidateUpdate(ctx, newCSIDriver, csiDriver)
	if len(errs) == 0 {
		t.Errorf("Expected a validation error")
	}
}

func TestCSIDriverPrepareForCreate(t *testing.T) {
	ctx := genericapirequest.WithRequestInfo(genericapirequest.NewContext(), &genericapirequest.RequestInfo{
		APIGroup:   "storage.k8s.io",
		APIVersion: "v1",
		Resource:   "csidrivers",
	})

	attachRequired := true
	podInfoOnMount := true
	storageCapacity := true
	requiresRepublish := true

	tests := []struct {
		name         string
		withCapacity bool
		withInline   bool
	}{
		{
			name:       "inline enabled",
			withInline: true,
		},
		{
			name:       "inline disabled",
			withInline: false,
		},
		{
			name:         "capacity enabled",
			withCapacity: true,
		},
		{
			name:         "capacity disabled",
			withCapacity: false,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.CSIStorageCapacity, test.withCapacity)()
			defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.CSIInlineVolume, test.withInline)()

			csiDriver := &storage.CSIDriver{
				ObjectMeta: metav1.ObjectMeta{
					Name: "foo",
				},
				Spec: storage.CSIDriverSpec{
					AttachRequired:  &attachRequired,
					PodInfoOnMount:  &podInfoOnMount,
					StorageCapacity: &storageCapacity,
					VolumeLifecycleModes: []storage.VolumeLifecycleMode{
						storage.VolumeLifecyclePersistent,
					},
					TokenRequests:     []storage.TokenRequest{},
					RequiresRepublish: &requiresRepublish,
				},
			}
			Strategy.PrepareForCreate(ctx, csiDriver)
			errs := Strategy.Validate(ctx, csiDriver)
			if len(errs) != 0 {
				t.Errorf("unexpected validating errors: %v", errs)
			}
			if test.withCapacity {
				if csiDriver.Spec.StorageCapacity == nil || *csiDriver.Spec.StorageCapacity != storageCapacity {
					t.Errorf("StorageCapacity modified: %v", csiDriver.Spec.StorageCapacity)
				}
			} else {
				if csiDriver.Spec.StorageCapacity != nil {
					t.Errorf("StorageCapacity not stripped: %v", csiDriver.Spec.StorageCapacity)
				}
			}
			if test.withInline {
				if len(csiDriver.Spec.VolumeLifecycleModes) != 1 {
					t.Errorf("VolumeLifecycleModes modified: %v", csiDriver.Spec)
				}
			} else {
				if len(csiDriver.Spec.VolumeLifecycleModes) != 0 {
					t.Errorf("VolumeLifecycleModes not stripped: %v", csiDriver.Spec)
				}
			}
		})
	}
}

func TestCSIDriverPrepareForUpdate(t *testing.T) {
	ctx := genericapirequest.WithRequestInfo(genericapirequest.NewContext(), &genericapirequest.RequestInfo{
		APIGroup:   "storage.k8s.io",
		APIVersion: "v1",
		Resource:   "csidrivers",
	})

	attachRequired := true
	podInfoOnMount := true
	driverWithNothing := &storage.CSIDriver{
		ObjectMeta: metav1.ObjectMeta{
			Name: "foo",
		},
	}
	driverWithPersistent := &storage.CSIDriver{
		ObjectMeta: metav1.ObjectMeta{
			Name: "foo",
		},
		Spec: storage.CSIDriverSpec{
			AttachRequired: &attachRequired,
			PodInfoOnMount: &podInfoOnMount,
			VolumeLifecycleModes: []storage.VolumeLifecycleMode{
				storage.VolumeLifecyclePersistent,
			},
		},
	}
	driverWithEphemeral := &storage.CSIDriver{
		ObjectMeta: metav1.ObjectMeta{
			Name: "foo",
		},
		Spec: storage.CSIDriverSpec{
			AttachRequired: &attachRequired,
			PodInfoOnMount: &podInfoOnMount,
			VolumeLifecycleModes: []storage.VolumeLifecycleMode{
				storage.VolumeLifecycleEphemeral,
			},
		},
	}
	enabled := true
	disabled := false
	gcp := "gcp"
	driverWithCapacityEnabled := &storage.CSIDriver{
		ObjectMeta: metav1.ObjectMeta{
			Name: "foo",
		},
		Spec: storage.CSIDriverSpec{
			StorageCapacity: &enabled,
		},
	}
	driverWithCapacityDisabled := &storage.CSIDriver{
		ObjectMeta: metav1.ObjectMeta{
			Name: "foo",
		},
		Spec: storage.CSIDriverSpec{
			StorageCapacity: &disabled,
		},
	}
	driverWithServiceAccountTokenGCP := &storage.CSIDriver{
		ObjectMeta: metav1.ObjectMeta{
			Name: "foo",
		},
		Spec: storage.CSIDriverSpec{
			TokenRequests:     []storage.TokenRequest{{Audience: gcp}},
			RequiresRepublish: &enabled,
		},
	}

	resultPersistent := []storage.VolumeLifecycleMode{storage.VolumeLifecyclePersistent}

	tests := []struct {
		name                      string
		old, update               *storage.CSIDriver
		csiStorageCapacityEnabled bool
		csiInlineVolumeEnabled    bool
		wantCapacity              *bool
		wantModes                 []storage.VolumeLifecycleMode
		wantTokenRequests         []storage.TokenRequest
		wantRequiresRepublish     *bool
		wantGeneration            int64
	}{
		{
			name:                      "capacity feature enabled, before: none, update: enabled",
			csiStorageCapacityEnabled: true,
			old:                       driverWithNothing,
			update:                    driverWithCapacityEnabled,
			wantCapacity:              &enabled,
		},
		{
			name:         "capacity feature disabled, before: none, update: disabled",
			old:          driverWithNothing,
			update:       driverWithCapacityDisabled,
			wantCapacity: nil,
		},
		{
			name:         "capacity feature disabled, before: enabled, update: disabled",
			old:          driverWithCapacityEnabled,
			update:       driverWithCapacityDisabled,
			wantCapacity: &disabled,
		},
		{
			name:                   "inline feature enabled, before: none, update: persitent",
			csiInlineVolumeEnabled: true,
			old:                    driverWithNothing,
			update:                 driverWithPersistent,
			wantModes:              resultPersistent,
		},
		{
			name:      "inline feature disabled, before: none, update: persitent",
			old:       driverWithNothing,
			update:    driverWithPersistent,
			wantModes: nil,
		},
		{
			name:      "inline feature disabled, before: ephemeral, update: persitent",
			old:       driverWithEphemeral,
			update:    driverWithPersistent,
			wantModes: resultPersistent,
		},
		{
			name:                  "service account token feature enabled, before: none, update: audience=gcp",
			old:                   driverWithNothing,
			update:                driverWithServiceAccountTokenGCP,
			wantTokenRequests:     []storage.TokenRequest{{Audience: gcp}},
			wantRequiresRepublish: &enabled,
			wantGeneration:        1,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.CSIStorageCapacity, test.csiStorageCapacityEnabled)()
			defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.CSIInlineVolume, test.csiInlineVolumeEnabled)()

			csiDriver := test.update.DeepCopy()
			Strategy.PrepareForUpdate(ctx, csiDriver, test.old)
			require.Equal(t, test.wantGeneration, csiDriver.GetGeneration())
			require.Equal(t, test.wantCapacity, csiDriver.Spec.StorageCapacity)
			require.Equal(t, test.wantModes, csiDriver.Spec.VolumeLifecycleModes)
			require.Equal(t, test.wantTokenRequests, csiDriver.Spec.TokenRequests)
			require.Equal(t, test.wantRequiresRepublish, csiDriver.Spec.RequiresRepublish)
		})
	}
}

func TestCSIDriverValidation(t *testing.T) {
	enabled := true
	disabled := true
	gcp := "gcp"

	tests := []struct {
		name        string
		csiDriver   *storage.CSIDriver
		expectError bool
	}{
		{
			"valid csidriver",
			getValidCSIDriver("foo"),
			false,
		},
		{
			"true for all flags",
			&storage.CSIDriver{
				ObjectMeta: metav1.ObjectMeta{
					Name: "foo",
				},
				Spec: storage.CSIDriverSpec{
					AttachRequired:    &enabled,
					PodInfoOnMount:    &enabled,
					StorageCapacity:   &enabled,
					RequiresRepublish: &enabled,
				},
			},
			false,
		},
		{
			"false for all flags",
			&storage.CSIDriver{
				ObjectMeta: metav1.ObjectMeta{
					Name: "foo",
				},
				Spec: storage.CSIDriverSpec{
					AttachRequired:    &disabled,
					PodInfoOnMount:    &disabled,
					StorageCapacity:   &disabled,
					RequiresRepublish: &disabled,
				},
			},
			false,
		},
		{
			"invalid driver name",
			&storage.CSIDriver{
				ObjectMeta: metav1.ObjectMeta{
					Name: "*foo#",
				},
				Spec: storage.CSIDriverSpec{
					AttachRequired:    &enabled,
					PodInfoOnMount:    &enabled,
					StorageCapacity:   &enabled,
					RequiresRepublish: &enabled,
				},
			},
			true,
		},
		{
			"invalid volume mode",
			&storage.CSIDriver{
				ObjectMeta: metav1.ObjectMeta{
					Name: "foo",
				},
				Spec: storage.CSIDriverSpec{
					AttachRequired:  &enabled,
					PodInfoOnMount:  &enabled,
					StorageCapacity: &enabled,
					VolumeLifecycleModes: []storage.VolumeLifecycleMode{
						storage.VolumeLifecycleMode("no-such-mode"),
					},
					RequiresRepublish: &enabled,
				},
			},
			true,
		},
		{
			"persistent volume mode",
			&storage.CSIDriver{
				ObjectMeta: metav1.ObjectMeta{
					Name: "foo",
				},
				Spec: storage.CSIDriverSpec{
					AttachRequired:  &enabled,
					PodInfoOnMount:  &enabled,
					StorageCapacity: &enabled,
					VolumeLifecycleModes: []storage.VolumeLifecycleMode{
						storage.VolumeLifecyclePersistent,
					},
					RequiresRepublish: &enabled,
				},
			},
			false,
		},
		{
			"ephemeral volume mode",
			&storage.CSIDriver{
				ObjectMeta: metav1.ObjectMeta{
					Name: "foo",
				},
				Spec: storage.CSIDriverSpec{
					AttachRequired:  &enabled,
					PodInfoOnMount:  &enabled,
					StorageCapacity: &enabled,
					VolumeLifecycleModes: []storage.VolumeLifecycleMode{
						storage.VolumeLifecycleEphemeral,
					},
					RequiresRepublish: &enabled,
				},
			},
			false,
		},
		{
			"both volume modes",
			&storage.CSIDriver{
				ObjectMeta: metav1.ObjectMeta{
					Name: "foo",
				},
				Spec: storage.CSIDriverSpec{
					AttachRequired:  &enabled,
					PodInfoOnMount:  &enabled,
					StorageCapacity: &enabled,
					VolumeLifecycleModes: []storage.VolumeLifecycleMode{
						storage.VolumeLifecyclePersistent,
						storage.VolumeLifecycleEphemeral,
					},
					RequiresRepublish: &enabled,
				},
			},
			false,
		},
		{
			"service account token with gcp as audience",
			&storage.CSIDriver{
				ObjectMeta: metav1.ObjectMeta{
					Name: "foo",
				},
				Spec: storage.CSIDriverSpec{
					AttachRequired:    &enabled,
					PodInfoOnMount:    &enabled,
					StorageCapacity:   &enabled,
					TokenRequests:     []storage.TokenRequest{{Audience: gcp}},
					RequiresRepublish: &enabled,
				},
			},
			false,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			testValidation := func(csiDriver *storage.CSIDriver, apiVersion string) field.ErrorList {
				ctx := genericapirequest.WithRequestInfo(genericapirequest.NewContext(), &genericapirequest.RequestInfo{
					APIGroup:   "storage.k8s.io",
					APIVersion: "v1",
					Resource:   "csidrivers",
				})
				return Strategy.Validate(ctx, csiDriver)
			}

			err := testValidation(test.csiDriver, "v1")
			if len(err) > 0 && !test.expectError {
				t.Errorf("Validation of v1 object failed: %+v", err)
			}
			if len(err) == 0 && test.expectError {
				t.Errorf("Validation of v1 object unexpectedly succeeded")
			}
		})
	}
}
