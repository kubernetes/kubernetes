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
	return &storage.CSIDriver{
		ObjectMeta: metav1.ObjectMeta{
			Name: name,
		},
		Spec: storage.CSIDriverSpec{
			AttachRequired: &attachRequired,
			PodInfoOnMount: &podInfoOnMount,
		},
	}
}

func TestCSIDriverStrategy(t *testing.T) {
	ctx := genericapirequest.WithRequestInfo(genericapirequest.NewContext(), &genericapirequest.RequestInfo{
		APIGroup:   "storage.k8s.io",
		APIVersion: "v1beta1",
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
		APIVersion: "v1beta1",
		Resource:   "csidrivers",
	})

	attachRequired := true
	podInfoOnMount := true
	csiDriver := &storage.CSIDriver{
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

	tests := []struct {
		name       string
		withInline bool
	}{
		{
			name:       "inline enabled",
			withInline: true,
		},
		{
			name:       "inline disabled",
			withInline: false,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.CSIInlineVolume, test.withInline)()

			Strategy.PrepareForCreate(ctx, csiDriver)
			errs := Strategy.Validate(ctx, csiDriver)
			if len(errs) != 0 {
				t.Errorf("unexpected validating errors: %v", errs)
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
		APIVersion: "v1beta1",
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

	tests := []struct {
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

	runAll := func(t *testing.T, withInline bool) {
		for _, test := range tests {
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
		runAll(t, true)
	})
	t.Run("without inline volumes", func(t *testing.T) {
		runAll(t, false)
	})
}

func TestCSIDriverValidation(t *testing.T) {
	attachRequired := true
	notAttachRequired := false
	podInfoOnMount := true
	notPodInfoOnMount := false

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
			"true PodInfoOnMount and AttachRequired",
			&storage.CSIDriver{
				ObjectMeta: metav1.ObjectMeta{
					Name: "foo",
				},
				Spec: storage.CSIDriverSpec{
					AttachRequired: &attachRequired,
					PodInfoOnMount: &podInfoOnMount,
				},
			},
			false,
		},
		{
			"false PodInfoOnMount and AttachRequired",
			&storage.CSIDriver{
				ObjectMeta: metav1.ObjectMeta{
					Name: "foo",
				},
				Spec: storage.CSIDriverSpec{
					AttachRequired: &notAttachRequired,
					PodInfoOnMount: &notPodInfoOnMount,
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
					AttachRequired: &attachRequired,
					PodInfoOnMount: &podInfoOnMount,
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
					AttachRequired: &attachRequired,
					PodInfoOnMount: &podInfoOnMount,
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
					AttachRequired: &attachRequired,
					PodInfoOnMount: &podInfoOnMount,
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
					AttachRequired: &attachRequired,
					PodInfoOnMount: &podInfoOnMount,
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
					AttachRequired: &attachRequired,
					PodInfoOnMount: &podInfoOnMount,
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
					APIVersion: "v1beta1",
					Resource:   "csidrivers",
				})
				return Strategy.Validate(ctx, csiDriver)
			}

			betaErr := testValidation(test.csiDriver, "v1beta1")
			if len(betaErr) > 0 && !test.expectError {
				t.Errorf("Validation of v1beta1 object failed: %+v", betaErr)
			}
			if len(betaErr) == 0 && test.expectError {
				t.Errorf("Validation of v1beta1 object unexpectedly succeeded")
			}
		})
	}
}
