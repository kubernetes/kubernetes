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
			SELinuxMount:      &enabled,
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

func TestCSIDriverPrepareForUpdate(t *testing.T) {
	ctx := genericapirequest.WithRequestInfo(genericapirequest.NewContext(), &genericapirequest.RequestInfo{
		APIGroup:   "storage.k8s.io",
		APIVersion: "v1",
		Resource:   "csidrivers",
	})

	attachRequired := true
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
			VolumeLifecycleModes: []storage.VolumeLifecycleMode{
				storage.VolumeLifecyclePersistent,
			},
		},
	}
	enabled := true
	disabled := false
	gcp := "gcp"
	noneFsGroupPolicy := storage.NoneFSGroupPolicy
	readWriteOnceWithFSTypeFSGroupPolicy := storage.ReadWriteOnceWithFSTypeFSGroupPolicy
	fileFSGroupPolicy := storage.FileFSGroupPolicy
	driverWithPodInfoOnMountEnabled := &storage.CSIDriver{
		ObjectMeta: metav1.ObjectMeta{
			Name: "foo",
		},
		Spec: storage.CSIDriverSpec{
			PodInfoOnMount: &enabled,
		},
	}
	driverWithPodInfoOnMountDisabled := &storage.CSIDriver{
		ObjectMeta: metav1.ObjectMeta{
			Name: "foo",
		},
		Spec: storage.CSIDriverSpec{
			PodInfoOnMount: &disabled,
		},
	}
	driverWithNoneFSGroupPolicy := &storage.CSIDriver{
		ObjectMeta: metav1.ObjectMeta{
			Name: "foo",
		},
		Spec: storage.CSIDriverSpec{
			FSGroupPolicy: &noneFsGroupPolicy,
		},
	}
	driverWithReadWriteOnceWithFSTypeFSGroupPolicy := &storage.CSIDriver{
		ObjectMeta: metav1.ObjectMeta{
			Name: "foo",
		},
		Spec: storage.CSIDriverSpec{
			FSGroupPolicy: &readWriteOnceWithFSTypeFSGroupPolicy,
		},
	}
	driverWithFileFSGroupPolicy := &storage.CSIDriver{
		ObjectMeta: metav1.ObjectMeta{
			Name: "foo",
		},
		Spec: storage.CSIDriverSpec{
			FSGroupPolicy: &fileFSGroupPolicy,
		},
	}
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
	driverWithSELinuxMountEnabled := &storage.CSIDriver{
		ObjectMeta: metav1.ObjectMeta{
			Name: "foo",
		},
		Spec: storage.CSIDriverSpec{
			SELinuxMount: &enabled,
		},
	}
	driverWithSELinuxMountDisabled := &storage.CSIDriver{
		ObjectMeta: metav1.ObjectMeta{
			Name: "foo",
		},
		Spec: storage.CSIDriverSpec{
			SELinuxMount: &disabled,
		},
	}

	resultPersistent := []storage.VolumeLifecycleMode{storage.VolumeLifecyclePersistent}

	tests := []struct {
		name                                string
		old, update                         *storage.CSIDriver
		seLinuxMountReadWriteOncePodEnabled bool
		wantCapacity                        *bool
		wantModes                           []storage.VolumeLifecycleMode
		wantTokenRequests                   []storage.TokenRequest
		wantRequiresRepublish               *bool
		wantGeneration                      int64
		wantSELinuxMount                    *bool
	}{
		{
			name:           "podInfoOnMount feature enabled, before: none, update: enabled",
			old:            driverWithNothing,
			update:         driverWithPodInfoOnMountEnabled,
			wantGeneration: 1,
		},
		{
			name:           "podInfoOnMount feature enabled, before: enabled, update: disabled",
			old:            driverWithPodInfoOnMountEnabled,
			update:         driverWithPodInfoOnMountDisabled,
			wantGeneration: 1,
		},
		{
			name:           "fSGroupPolicy feature enabled, before: nil, update: none",
			old:            driverWithNothing,
			update:         driverWithNoneFSGroupPolicy,
			wantGeneration: 1,
		},
		{
			name:           "fSGroupPolicy feature enabled, before: nil, update: readWriteOnceWithFSType",
			old:            driverWithNothing,
			update:         driverWithReadWriteOnceWithFSTypeFSGroupPolicy,
			wantGeneration: 1,
		},
		{
			name:           "fSGroupPolicy feature enabled, before: nil, update: file",
			old:            driverWithNothing,
			update:         driverWithFileFSGroupPolicy,
			wantGeneration: 1,
		},
		{
			name:           "fSGroupPolicy feature enabled, before: none, update: readWriteOnceWithFSType",
			old:            driverWithNoneFSGroupPolicy,
			update:         driverWithReadWriteOnceWithFSTypeFSGroupPolicy,
			wantGeneration: 1,
		},
		{
			name:           "fSGroupPolicy feature enabled, before: none, update: file",
			old:            driverWithNoneFSGroupPolicy,
			update:         driverWithFileFSGroupPolicy,
			wantGeneration: 1,
		},
		{
			name:           "fSGroupPolicy feature enabled, before: readWriteOnceWithFSType, update: none",
			old:            driverWithReadWriteOnceWithFSTypeFSGroupPolicy,
			update:         driverWithNoneFSGroupPolicy,
			wantGeneration: 1,
		},
		{
			name:           "fSGroupPolicy feature enabled, before: readWriteOnceWithFSType, update: file",
			old:            driverWithReadWriteOnceWithFSTypeFSGroupPolicy,
			update:         driverWithFileFSGroupPolicy,
			wantGeneration: 1,
		},
		{
			name:           "fSGroupPolicy feature enabled, before: file, update: none",
			old:            driverWithFileFSGroupPolicy,
			update:         driverWithNoneFSGroupPolicy,
			wantGeneration: 1,
		},
		{
			name:           "fSGroupPolicy feature enabled, before: file, update: readWriteOnceWithFSType",
			old:            driverWithFileFSGroupPolicy,
			update:         driverWithReadWriteOnceWithFSTypeFSGroupPolicy,
			wantGeneration: 1,
		},
		{
			name:           "capacity feature enabled, before: none, update: enabled",
			old:            driverWithNothing,
			update:         driverWithCapacityEnabled,
			wantCapacity:   &enabled,
			wantGeneration: 1,
		},
		{
			name:           "capacity feature enabled, before: enabled, update: disabled",
			old:            driverWithCapacityEnabled,
			update:         driverWithCapacityDisabled,
			wantCapacity:   &disabled,
			wantGeneration: 1,
		},
		{
			name:           "inline feature enabled, before: none, update: persistent",
			old:            driverWithNothing,
			update:         driverWithPersistent,
			wantModes:      resultPersistent,
			wantGeneration: 1,
		},
		{
			name:                  "service account token feature enabled, before: none, update: audience=gcp",
			old:                   driverWithNothing,
			update:                driverWithServiceAccountTokenGCP,
			wantTokenRequests:     []storage.TokenRequest{{Audience: gcp}},
			wantRequiresRepublish: &enabled,
			wantGeneration:        1,
		},
		{
			name:                                "SELinux mount support feature enabled, before: nil, update: on",
			seLinuxMountReadWriteOncePodEnabled: true,
			old:                                 driverWithNothing,
			update:                              driverWithSELinuxMountEnabled,
			wantSELinuxMount:                    &enabled,
			wantGeneration:                      1,
		},
		{
			name:                                "SELinux mount support feature enabled, before: off, update: on",
			seLinuxMountReadWriteOncePodEnabled: true,
			old:                                 driverWithSELinuxMountDisabled,
			update:                              driverWithSELinuxMountEnabled,
			wantSELinuxMount:                    &enabled,
			wantGeneration:                      1,
		},
		{
			name:                                "SELinux mount support feature enabled, before: on, update: off",
			seLinuxMountReadWriteOncePodEnabled: true,
			old:                                 driverWithSELinuxMountEnabled,
			update:                              driverWithSELinuxMountDisabled,
			wantSELinuxMount:                    &disabled,
			wantGeneration:                      1,
		},
		{
			name:                                "SELinux mount support feature disabled, before: nil, update: on",
			seLinuxMountReadWriteOncePodEnabled: false,
			old:                                 driverWithNothing,
			update:                              driverWithSELinuxMountEnabled,
			wantSELinuxMount:                    nil,
			wantGeneration:                      0,
		},
		{
			name:                                "SELinux mount support feature disabled, before: off, update: on",
			seLinuxMountReadWriteOncePodEnabled: false,
			old:                                 driverWithSELinuxMountDisabled,
			update:                              driverWithSELinuxMountEnabled,
			wantSELinuxMount:                    &enabled,
			wantGeneration:                      1,
		},
		{
			name:                                "SELinux mount support feature enabled, before: on, update: off",
			seLinuxMountReadWriteOncePodEnabled: false,
			old:                                 driverWithSELinuxMountEnabled,
			update:                              driverWithSELinuxMountDisabled,
			wantSELinuxMount:                    &disabled,
			wantGeneration:                      1,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.SELinuxMountReadWriteOncePod, test.seLinuxMountReadWriteOncePodEnabled)

			csiDriver := test.update.DeepCopy()
			Strategy.PrepareForUpdate(ctx, csiDriver, test.old)
			require.Equal(t, test.wantGeneration, csiDriver.GetGeneration())
			require.Equal(t, test.wantCapacity, csiDriver.Spec.StorageCapacity)
			require.Equal(t, test.wantModes, csiDriver.Spec.VolumeLifecycleModes)
			require.Equal(t, test.wantTokenRequests, csiDriver.Spec.TokenRequests)
			require.Equal(t, test.wantRequiresRepublish, csiDriver.Spec.RequiresRepublish)
			require.Equal(t, test.wantSELinuxMount, csiDriver.Spec.SELinuxMount)
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
					SELinuxMount:      &enabled,
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
					SELinuxMount:      &disabled,
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
					SELinuxMount:      &enabled,
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
					SELinuxMount:      &enabled,
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
					SELinuxMount:      &enabled,
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
					SELinuxMount:      &enabled,
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
					SELinuxMount:      &enabled,
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
					SELinuxMount:      &enabled,
				},
			},
			false,
		},
		{
			"invalid SELinuxMount",
			&storage.CSIDriver{
				ObjectMeta: metav1.ObjectMeta{
					Name: "foo",
				},
				Spec: storage.CSIDriverSpec{
					AttachRequired:  &enabled,
					PodInfoOnMount:  &enabled,
					StorageCapacity: &enabled,
					SELinuxMount:    nil,
				},
			},
			true,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			// assume this feature is on for this test, detailed enabled/disabled tests in TestCSIDriverValidationSELinuxMountEnabledDisabled
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.SELinuxMountReadWriteOncePod, true)

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
