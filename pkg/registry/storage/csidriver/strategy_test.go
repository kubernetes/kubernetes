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
	attachRequired := true
	podInfoOnMount := true
	seLinuxMountSupported := false

	return &storage.CSIDriver{
		ObjectMeta: metav1.ObjectMeta{
			Name: name,
		},
		Spec: storage.CSIDriverSpec{
			AttachRequired:        &attachRequired,
			PodInfoOnMount:        &podInfoOnMount,
			SELinuxMountSupported: &seLinuxMountSupported,
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
	seLinuxMountSupported := true
	csiDriver := &storage.CSIDriver{
		ObjectMeta: metav1.ObjectMeta{
			Name: "foo",
		},
		Spec: storage.CSIDriverSpec{
			AttachRequired:        &attachRequired,
			PodInfoOnMount:        &podInfoOnMount,
			SELinuxMountSupported: &seLinuxMountSupported,
			VolumeLifecycleModes: []storage.VolumeLifecycleMode{
				storage.VolumeLifecyclePersistent,
			},
		},
	}

	tests := []struct {
		name              string
		withInline        bool
		withSELinuxPolicy bool
	}{
		{
			name:              "inline enabled",
			withInline:        true,
			withSELinuxPolicy: true,
		},
		{
			name:              "inline disabled",
			withInline:        false,
			withSELinuxPolicy: true,
		},
		{
			name:              "SELinux enabled",
			withInline:        true,
			withSELinuxPolicy: true,
		},
		{
			name:              "SELinux disabled",
			withInline:        true,
			withSELinuxPolicy: false,
		},
		{
			name:              "seLinuxPolicy disabled",
			withSELinuxPolicy: false,
		},
		{
			name:              "seLinuxPolicy enabled",
			withSELinuxPolicy: true,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.CSIInlineVolume, test.withInline)()
			defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.SELinuxRelabelPolicy, test.withSELinuxPolicy)()

			csiDriver := csiDriver.DeepCopy()
			Strategy.PrepareForCreate(ctx, csiDriver)
			errs := Strategy.Validate(ctx, csiDriver)
			if len(errs) != 0 {
				t.Errorf("unexpected validating errors: %v", errs)
			}
			if test.withInline {
				if len(csiDriver.Spec.VolumeLifecycleModes) != 1 {
					t.Errorf("VolumeLifecycleModes modified: %+v", csiDriver.Spec)
				}
			} else {
				if len(csiDriver.Spec.VolumeLifecycleModes) != 0 {
					t.Errorf("VolumeLifecycleModes not stripped: %+v", csiDriver.Spec)
				}
			}
			if test.withSELinuxPolicy {
				if csiDriver.Spec.SELinuxMountSupported == nil {
					t.Errorf("SELinuxMountSupported modified: %+v", csiDriver.Spec)
				}
			} else {
				if csiDriver.Spec.SELinuxMountSupported != nil {
					t.Errorf("SELinuxMountSupported not stripped: %+v", csiDriver.Spec)
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
	driverWithoutModes := &storage.CSIDriver{
		ObjectMeta: metav1.ObjectMeta{
			Name: "foo",
		},
		Spec: storage.CSIDriverSpec{
			AttachRequired: &attachRequired,
			PodInfoOnMount: &podInfoOnMount,
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
	var resultEmpty []storage.VolumeLifecycleMode
	resultPersistent := []storage.VolumeLifecycleMode{storage.VolumeLifecyclePersistent}
	resultEphemeral := []storage.VolumeLifecycleMode{storage.VolumeLifecycleEphemeral}

	inlineTests := []struct {
		name                      string
		old, update               *storage.CSIDriver
		withInline, withoutInline []storage.VolumeLifecycleMode
	}{
		{
			name:          "before: no mode, update: no mode",
			old:           driverWithoutModes,
			update:        driverWithoutModes,
			withInline:    resultEmpty,
			withoutInline: resultEmpty,
		},
		{
			name:          "before: no mode, update: persistent",
			old:           driverWithoutModes,
			update:        driverWithPersistent,
			withInline:    resultPersistent,
			withoutInline: resultEmpty,
		},
		{
			name:          "before: persistent, update: ephemeral",
			old:           driverWithPersistent,
			update:        driverWithEphemeral,
			withInline:    resultEphemeral,
			withoutInline: resultEphemeral,
		},
		{
			name:          "before: persistent, update: no mode",
			old:           driverWithPersistent,
			update:        driverWithoutModes,
			withInline:    resultEmpty,
			withoutInline: resultEmpty,
		},
	}

	runInlineTests := func(t *testing.T, withInline bool) {
		for _, test := range inlineTests {
			t.Run(test.name, func(t *testing.T) {
				defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.CSIInlineVolume, withInline)()

				csiDriver := test.update.DeepCopy()
				Strategy.PrepareForUpdate(ctx, csiDriver, test.old)
				if withInline {
					require.Equal(t, csiDriver.Spec.VolumeLifecycleModes, test.withInline)
				} else {
					require.Equal(t, csiDriver.Spec.VolumeLifecycleModes, test.withoutInline)
				}
			})
		}
	}

	t.Run("with inline volumes", func(t *testing.T) {
		runInlineTests(t, true)
	})
	t.Run("without inline volumes", func(t *testing.T) {
		runInlineTests(t, false)
	})

	bTrue := true
	bFalse := false
	driverWithSELinuxTrue := &storage.CSIDriver{
		ObjectMeta: metav1.ObjectMeta{
			Name: "foo",
		},
		Spec: storage.CSIDriverSpec{
			AttachRequired:        &attachRequired,
			PodInfoOnMount:        &podInfoOnMount,
			SELinuxMountSupported: &bTrue,
		},
	}
	driverWithSELinuxFalse := &storage.CSIDriver{
		ObjectMeta: metav1.ObjectMeta{
			Name: "foo",
		},
		Spec: storage.CSIDriverSpec{
			AttachRequired:        &attachRequired,
			PodInfoOnMount:        &podInfoOnMount,
			SELinuxMountSupported: &bFalse,
		},
	}
	driverWithoutSELinux := &storage.CSIDriver{
		ObjectMeta: metav1.ObjectMeta{
			Name: "foo",
		},
		Spec: storage.CSIDriverSpec{
			AttachRequired:        &attachRequired,
			PodInfoOnMount:        &podInfoOnMount,
			SELinuxMountSupported: nil,
		},
	}
	seLinuxTests := []struct {
		name                        string
		old, update                 *storage.CSIDriver
		withSELinux, withoutSELinux *bool
	}{
		{
			name:           "before: no seLinux, update: no seLinux",
			old:            driverWithoutSELinux,
			update:         driverWithoutSELinux,
			withSELinux:    nil,
			withoutSELinux: nil,
		},
		{
			name:           "before: no seLinux, update: with seLinux",
			old:            driverWithoutSELinux,
			update:         driverWithSELinuxTrue,
			withSELinux:    &bTrue,
			withoutSELinux: nil,
		},
		{
			name:           "before: no seLinux, update: with seLinux false",
			old:            driverWithoutSELinux,
			update:         driverWithSELinuxFalse,
			withSELinux:    &bFalse,
			withoutSELinux: nil,
		},
		{
			name:           "before: SELinux true, update: nil",
			old:            driverWithSELinuxTrue,
			update:         driverWithoutSELinux,
			withSELinux:    nil,
			withoutSELinux: nil,
		},
		{
			name:           "before: SELinux true, update: false",
			old:            driverWithSELinuxTrue,
			update:         driverWithSELinuxFalse,
			withSELinux:    &bFalse,
			withoutSELinux: &bFalse,
		},
	}

	runSELinuxTests := func(t *testing.T, withSELinux bool) {
		for _, test := range seLinuxTests {
			t.Run(test.name, func(t *testing.T) {
				defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.SELinuxRelabelPolicy, withSELinux)()

				csiDriver := test.update.DeepCopy()
				Strategy.PrepareForUpdate(ctx, csiDriver, test.old)
				if withSELinux {
					require.Equal(t, csiDriver.Spec.SELinuxMountSupported, test.withSELinux)
				} else {
					require.Equal(t, csiDriver.Spec.SELinuxMountSupported, test.withoutSELinux)
				}
			})
		}
	}

	t.Run("with SELinuxMountSupported", func(t *testing.T) {
		runSELinuxTests(t, true)
	})
	t.Run("without SELinuxMountSupported", func(t *testing.T) {
		runSELinuxTests(t, false)
	})
}

func TestCSIDriverValidation(t *testing.T) {
	attachRequired := true
	notAttachRequired := false
	podInfoOnMount := true
	notPodInfoOnMount := false
	seLinuxMountSupported := true
	seLinuxMountNotSupported := true

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
			"true PodInfoOnMount, AttachRequired and SELinuxMountSupported",
			&storage.CSIDriver{
				ObjectMeta: metav1.ObjectMeta{
					Name: "foo",
				},
				Spec: storage.CSIDriverSpec{
					AttachRequired:        &attachRequired,
					PodInfoOnMount:        &podInfoOnMount,
					SELinuxMountSupported: &seLinuxMountSupported,
				},
			},
			false,
		},
		{
			"false PodInfoOnMount, AttachRequired and SELinuxMountSupported",
			&storage.CSIDriver{
				ObjectMeta: metav1.ObjectMeta{
					Name: "foo",
				},
				Spec: storage.CSIDriverSpec{
					AttachRequired:        &notAttachRequired,
					PodInfoOnMount:        &notPodInfoOnMount,
					SELinuxMountSupported: &seLinuxMountNotSupported,
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
					AttachRequired:        &attachRequired,
					PodInfoOnMount:        &podInfoOnMount,
					SELinuxMountSupported: &seLinuxMountSupported,
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
					AttachRequired:        &attachRequired,
					PodInfoOnMount:        &podInfoOnMount,
					SELinuxMountSupported: &seLinuxMountSupported,
					VolumeLifecycleModes: []storage.VolumeLifecycleMode{
						storage.VolumeLifecycleMode("no-such-mode"),
					},
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
					AttachRequired:        &attachRequired,
					PodInfoOnMount:        &podInfoOnMount,
					SELinuxMountSupported: &seLinuxMountSupported,
					VolumeLifecycleModes: []storage.VolumeLifecycleMode{
						storage.VolumeLifecyclePersistent,
					},
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
					AttachRequired:        &attachRequired,
					PodInfoOnMount:        &podInfoOnMount,
					SELinuxMountSupported: &seLinuxMountSupported,
					VolumeLifecycleModes: []storage.VolumeLifecycleMode{
						storage.VolumeLifecycleEphemeral,
					},
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
					AttachRequired:        &attachRequired,
					PodInfoOnMount:        &podInfoOnMount,
					SELinuxMountSupported: &seLinuxMountSupported,
					VolumeLifecycleModes: []storage.VolumeLifecycleMode{
						storage.VolumeLifecyclePersistent,
						storage.VolumeLifecycleEphemeral,
					},
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
