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

package v1beta1_test

import (
	"reflect"
	"testing"

	"github.com/google/go-cmp/cmp"
	storagev1beta1 "k8s.io/api/storage/v1beta1"
	"k8s.io/apimachinery/pkg/runtime"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/kubernetes/pkg/api/legacyscheme"
	_ "k8s.io/kubernetes/pkg/apis/storage/install"
	"k8s.io/kubernetes/pkg/features"
)

func roundTrip(t *testing.T, obj runtime.Object) runtime.Object {
	codec := legacyscheme.Codecs.LegacyCodec(storagev1beta1.SchemeGroupVersion)
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

func TestSetDefaultVolumeBindingMode(t *testing.T) {
	class := &storagev1beta1.StorageClass{}

	// field should be defaulted
	defaultMode := storagev1beta1.VolumeBindingImmediate
	output := roundTrip(t, runtime.Object(class)).(*storagev1beta1.StorageClass)
	outMode := output.VolumeBindingMode
	if outMode == nil {
		t.Errorf("Expected VolumeBindingMode to be defaulted to: %+v, got: nil", defaultMode)
	} else if *outMode != defaultMode {
		t.Errorf("Expected VolumeBindingMode to be defaulted to: %+v, got: %+v", defaultMode, outMode)
	}
}

func TestSetDefaultAttachRequired(t *testing.T) {
	driver := &storagev1beta1.CSIDriver{}

	// field should be defaulted
	defaultAttach := true
	defaultPodInfo := false
	output := roundTrip(t, runtime.Object(driver)).(*storagev1beta1.CSIDriver)
	outAttach := output.Spec.AttachRequired
	if outAttach == nil {
		t.Errorf("Expected AttachRequired to be defaulted to: %+v, got: nil", defaultAttach)
	} else if *outAttach != defaultAttach {
		t.Errorf("Expected AttachRequired to be defaulted to: %+v, got: %+v", defaultAttach, outAttach)
	}
	outPodInfo := output.Spec.PodInfoOnMount
	if outPodInfo == nil {
		t.Errorf("Expected PodInfoOnMount to be defaulted to: %+v, got: nil", defaultPodInfo)
	} else if *outPodInfo != defaultPodInfo {
		t.Errorf("Expected PodInfoOnMount to be defaulted to: %+v, got: %+v", defaultPodInfo, outPodInfo)
	}
}

func TestSetDefaultStorageCapacityEnabled(t *testing.T) {
	driver := &storagev1beta1.CSIDriver{}

	// field should be defaulted
	defaultStorageCapacity := false
	output := roundTrip(t, runtime.Object(driver)).(*storagev1beta1.CSIDriver)
	outStorageCapacity := output.Spec.StorageCapacity
	if outStorageCapacity == nil {
		t.Errorf("Expected StorageCapacity to be defaulted to: %+v, got: nil", defaultStorageCapacity)
	} else if *outStorageCapacity != defaultStorageCapacity {
		t.Errorf("Expected StorageCapacity to be defaulted to: %+v, got: %+v", defaultStorageCapacity, outStorageCapacity)
	}
}

func TestSetDefaultVolumeLifecycleModesEnabled(t *testing.T) {
	driver := &storagev1beta1.CSIDriver{}

	// field should be defaulted
	defaultMode := storagev1beta1.VolumeLifecyclePersistent
	output := roundTrip(t, runtime.Object(driver)).(*storagev1beta1.CSIDriver)
	outModes := output.Spec.VolumeLifecycleModes
	if len(outModes) != 1 {
		t.Errorf("Expected VolumeLifecycleModes to be defaulted to: %+v, got: %+v", defaultMode, outModes)
	} else if outModes[0] != defaultMode {
		t.Errorf("Expected VolumeLifecycleModes to be defaulted to: %+v, got: %+v", defaultMode, outModes)
	}
}

func TestSetDefaultCSIDriver(t *testing.T) {
	enabled := true
	disabled := false
	tests := []struct {
		desc     string
		field    string
		wantSpec *storagev1beta1.CSIDriverSpec
	}{
		{
			desc:     "AttachRequired default to true",
			field:    "AttachRequired",
			wantSpec: &storagev1beta1.CSIDriverSpec{AttachRequired: &enabled},
		},
		{
			desc:     "PodInfoOnMount default to false",
			field:    "PodInfoOnMount",
			wantSpec: &storagev1beta1.CSIDriverSpec{PodInfoOnMount: &disabled},
		},
		{
			desc:     "RequiresRepublish default to false",
			field:    "RequiresRepublish",
			wantSpec: &storagev1beta1.CSIDriverSpec{RequiresRepublish: &disabled},
		},
	}

	for _, test := range tests {
		t.Run(test.desc, func(t *testing.T) {
			gotSpec := roundTrip(t, runtime.Object(&storagev1beta1.CSIDriver{})).(*storagev1beta1.CSIDriver).Spec
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
	driver := &storagev1beta1.CSIDriver{}

	// field should be defaulted
	defaultSELinuxMount := false
	output := roundTrip(t, runtime.Object(driver)).(*storagev1beta1.CSIDriver)
	outSELinuxMount := output.Spec.SELinuxMount
	if outSELinuxMount == nil {
		t.Errorf("Expected SELinuxMount to be defaulted to: %+v, got: nil", defaultSELinuxMount)
	} else if *outSELinuxMount != defaultSELinuxMount {
		t.Errorf("Expected SELinuxMount to be defaulted to: %+v, got: %+v", defaultSELinuxMount, outSELinuxMount)
	}
}

func TestSetDefaultSELinuxMountReadWriteOncePodDisabled(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.SELinuxMountReadWriteOncePod, false)
	driver := &storagev1beta1.CSIDriver{}

	// field should not be defaulted
	output := roundTrip(t, runtime.Object(driver)).(*storagev1beta1.CSIDriver)
	outSELinuxMount := output.Spec.SELinuxMount
	if outSELinuxMount != nil {
		t.Errorf("Expected SELinuxMount remain nil, got: %+v", outSELinuxMount)
	}
}
