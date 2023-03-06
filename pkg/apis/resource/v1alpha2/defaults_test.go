/*
Copyright 2022 The Kubernetes Authors.

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

package v1alpha2_test

import (
	"reflect"
	"testing"

	v1alpha2 "k8s.io/api/resource/v1alpha2"
	"k8s.io/apimachinery/pkg/runtime"

	// ensure types are installed
	"k8s.io/kubernetes/pkg/api/legacyscheme"
	_ "k8s.io/kubernetes/pkg/apis/resource/install"
)

func TestSetDefaultAllocationMode(t *testing.T) {
	claim := &v1alpha2.ResourceClaim{}

	// field should be defaulted
	defaultMode := v1alpha2.AllocationModeWaitForFirstConsumer
	output := roundTrip(t, runtime.Object(claim)).(*v1alpha2.ResourceClaim)
	outMode := output.Spec.AllocationMode
	if outMode != defaultMode {
		t.Errorf("Expected AllocationMode to be defaulted to: %+v, got: %+v", defaultMode, outMode)
	}

	// field should not change
	nonDefaultMode := v1alpha2.AllocationModeImmediate
	claim = &v1alpha2.ResourceClaim{
		Spec: v1alpha2.ResourceClaimSpec{
			AllocationMode: nonDefaultMode,
		},
	}
	output = roundTrip(t, runtime.Object(claim)).(*v1alpha2.ResourceClaim)
	outMode = output.Spec.AllocationMode
	if outMode != v1alpha2.AllocationModeImmediate {
		t.Errorf("Expected AllocationMode to remain %+v, got: %+v", nonDefaultMode, outMode)
	}
}

func roundTrip(t *testing.T, obj runtime.Object) runtime.Object {
	codec := legacyscheme.Codecs.LegacyCodec(v1alpha2.SchemeGroupVersion)
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
