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

package v1_test

import (
	"reflect"
	"testing"

	"github.com/google/go-cmp/cmp"
	storagev1 "k8s.io/api/storage/v1"
	"k8s.io/apimachinery/pkg/runtime"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/kubernetes/pkg/api/legacyscheme"
	_ "k8s.io/kubernetes/pkg/apis/storage/install"
	"k8s.io/kubernetes/pkg/features"
)

func roundTrip(t *testing.T, obj runtime.Object) runtime.Object {
	codec := legacyscheme.Codecs.LegacyCodec(storagev1.SchemeGroupVersion)
	data, err := runtime.Encode(codec, obj)
	if err != nil {
		t.Errorf("%v\n %#v", err, obj)
		return nil
	}
	obj2, err := runtime.Decode(codec, data)
	if err != nil {
		t.Errorf("%v\nData: %s\nSource: %#v", err, string(data), obj)
		return nil
	}
	obj3 := reflect.New(reflect.TypeOf(obj).Elem()).Interface().(runtime.Object)
	err = legacyscheme.Scheme.Convert(obj2, obj3, nil)
	if err != nil {
		t.Errorf("%v\nSource: %#v", err, obj2)
		return nil
	}
	return obj3
}

func TestSetDefaultStorageCapacityEnabled(t *testing.T) {
	driver := &storagev1.CSIDriver{}

	// field should be defaulted
	defaultStorageCapacity := false
	output := roundTrip(t, runtime.Object(driver)).(*storagev1.CSIDriver)
	outStorageCapacity := output.Spec.StorageCapacity
	if outStorageCapacity == nil {
		t.Errorf("Expected StorageCapacity to be defaulted to: %+v, got: nil", defaultStorageCapacity)
	} else if *outStorageCapacity != defaultStorageCapacity {
		t.Errorf("Expected StorageCapacity to be defaulted to: %+v, got: %+v", defaultStorageCapacity, outStorageCapacity)
	}
}

func TestSetDefaultVolumeBindingMode(t *testing.T) {
	class := &storagev1.StorageClass{}

	// field should be defaulted
	defaultMode := storagev1.VolumeBindingImmediate
	output := roundTrip(t, runtime.Object(class)).(*storagev1.StorageClass)
	outMode := output.VolumeBindingMode
	if outMode == nil {
		t.Errorf("Expected VolumeBindingMode to be defaulted to: %+v, got: nil", defaultMode)
	} else if *outMode != defaultMode {
		t.Errorf("Expected VolumeBindingMode to be defaulted to: %+v, got: %+v", defaultMode, outMode)
	}
}

func TestSetDefaultCSIDriver(t *testing.T) {
	enabled := true
	disabled := false
	tests := []struct {
		desc     string
		field    string
		wantSpec *storagev1.CSIDriverSpec
	}{
		{
			desc:     "AttachRequired default to true",
			field:    "AttachRequired",
			wantSpec: &storagev1.CSIDriverSpec{AttachRequired: &enabled},
		},
		{
			desc:     "PodInfoOnMount default to false",
			field:    "PodInfoOnMount",
			wantSpec: &storagev1.CSIDriverSpec{PodInfoOnMount: &disabled},
		},
		{
			desc:     "VolumeLifecycleModes default to VolumeLifecyclePersistent",
			field:    "VolumeLifecycleModes",
			wantSpec: &storagev1.CSIDriverSpec{VolumeLifecycleModes: []storagev1.VolumeLifecycleMode{storagev1.VolumeLifecyclePersistent}},
		},
		{
			desc:     "RequiresRepublish default to false",
			field:    "RequiresRepublish",
			wantSpec: &storagev1.CSIDriverSpec{RequiresRepublish: &disabled},
		},
	}

	for _, test := range tests {
		t.Run(test.desc, func(t *testing.T) {
			gotSpec := roundTrip(t, runtime.Object(&storagev1.CSIDriver{})).(*storagev1.CSIDriver).Spec
			got := reflect.Indirect(reflect.ValueOf(gotSpec)).FieldByName(test.field).Interface()
			want := reflect.Indirect(reflect.ValueOf(test.wantSpec)).FieldByName(test.field).Interface()
			if diff := cmp.Diff(want, got); diff != "" {
				t.Errorf("CSIDriver defaults diff (-want +got):\n%s", diff)
			}
		})
	}
}

func TestSetDefaultSELinuxMountReadWriteOncePodEnabled(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.SELinuxMountReadWriteOncePod, true)
	driver := &storagev1.CSIDriver{}

	// field should be defaulted
	defaultSELinuxMount := false
	output := roundTrip(t, runtime.Object(driver)).(*storagev1.CSIDriver)
	outSELinuxMount := output.Spec.SELinuxMount
	if outSELinuxMount == nil {
		t.Errorf("Expected SELinuxMount to be defaulted to: %+v, got: nil", defaultSELinuxMount)
	} else if *outSELinuxMount != defaultSELinuxMount {
		t.Errorf("Expected SELinuxMount to be defaulted to: %+v, got: %+v", defaultSELinuxMount, outSELinuxMount)
	}
}

func TestSetDefaultSELinuxMountReadWriteOncePodDisabled(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.SELinuxMountReadWriteOncePod, false)
	driver := &storagev1.CSIDriver{}

	// field should not be defaulted
	output := roundTrip(t, runtime.Object(driver)).(*storagev1.CSIDriver)
	outSELinuxMount := output.Spec.SELinuxMount
	if outSELinuxMount != nil {
		t.Errorf("Expected SELinuxMount to remain nil, got: %+v", outSELinuxMount)
	}
}
