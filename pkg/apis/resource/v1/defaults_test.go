/*
Copyright 2025 The Kubernetes Authors.

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
	"time"

	"github.com/stretchr/testify/assert"

	v1 "k8s.io/api/resource/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/utils/ptr"

	// ensure types are installed
	"k8s.io/kubernetes/pkg/api/legacyscheme"
	_ "k8s.io/kubernetes/pkg/apis/resource/install"
)

func TestSetDefaultAllocationMode(t *testing.T) {
	claim := &v1.ResourceClaim{
		Spec: v1.ResourceClaimSpec{
			Devices: v1.DeviceClaim{
				Requests: []v1.DeviceRequest{
					{
						Exactly: &v1.ExactDeviceRequest{},
					},
				},
			},
		},
	}

	// fields should be defaulted
	defaultMode := v1.DeviceAllocationModeExactCount
	defaultCount := int64(1)
	output := roundTrip(t, runtime.Object(claim)).(*v1.ResourceClaim)
	assert.Equal(t, defaultMode, output.Spec.Devices.Requests[0].Exactly.AllocationMode)
	assert.Equal(t, defaultCount, output.Spec.Devices.Requests[0].Exactly.Count)

	// field should not change
	nonDefaultMode := v1.DeviceAllocationModeExactCount
	nonDefaultCount := int64(10)
	claim = &v1.ResourceClaim{
		Spec: v1.ResourceClaimSpec{
			Devices: v1.DeviceClaim{
				Requests: []v1.DeviceRequest{{
					Exactly: &v1.ExactDeviceRequest{
						AllocationMode: nonDefaultMode,
						Count:          nonDefaultCount,
					},
				}},
			},
		},
	}
	output = roundTrip(t, runtime.Object(claim)).(*v1.ResourceClaim)
	assert.Equal(t, nonDefaultMode, output.Spec.Devices.Requests[0].Exactly.AllocationMode)
	assert.Equal(t, nonDefaultCount, output.Spec.Devices.Requests[0].Exactly.Count)
}

func TestSetDefaultAllocationModeWithSubRequests(t *testing.T) {
	claim := &v1.ResourceClaim{
		Spec: v1.ResourceClaimSpec{
			Devices: v1.DeviceClaim{
				Requests: []v1.DeviceRequest{
					{
						Name: "req-1",
						FirstAvailable: []v1.DeviceSubRequest{
							{
								Name: "subReq-1",
							},
							{
								Name: "subReq-2",
							},
						},
					},
				},
			},
		},
	}

	defaultMode := v1.DeviceAllocationModeExactCount
	defaultCount := int64(1)
	output := roundTrip(t, runtime.Object(claim)).(*v1.ResourceClaim)
	// the exactly field is not set.
	assert.Nil(t, output.Spec.Devices.Requests[0].Exactly)
	// fields on the subRequests should be defaulted.
	assert.Equal(t, defaultMode, output.Spec.Devices.Requests[0].FirstAvailable[0].AllocationMode)
	assert.Equal(t, defaultCount, output.Spec.Devices.Requests[0].FirstAvailable[0].Count)
	assert.Equal(t, defaultMode, output.Spec.Devices.Requests[0].FirstAvailable[1].AllocationMode)
	assert.Equal(t, defaultCount, output.Spec.Devices.Requests[0].FirstAvailable[1].Count)

	// field should not change
	nonDefaultMode := v1.DeviceAllocationModeExactCount
	nonDefaultCount := int64(10)
	claim = &v1.ResourceClaim{
		Spec: v1.ResourceClaimSpec{
			Devices: v1.DeviceClaim{
				Requests: []v1.DeviceRequest{{
					Name: "req-1",
					FirstAvailable: []v1.DeviceSubRequest{
						{
							Name:           "subReq-1",
							AllocationMode: nonDefaultMode,
							Count:          nonDefaultCount,
						},
						{
							Name:           "subReq-2",
							AllocationMode: nonDefaultMode,
							Count:          nonDefaultCount,
						},
					},
				}},
			},
		},
	}
	output = roundTrip(t, runtime.Object(claim)).(*v1.ResourceClaim)
	assert.Equal(t, nonDefaultMode, output.Spec.Devices.Requests[0].FirstAvailable[0].AllocationMode)
	assert.Equal(t, nonDefaultCount, output.Spec.Devices.Requests[0].FirstAvailable[0].Count)
	assert.Equal(t, nonDefaultMode, output.Spec.Devices.Requests[0].FirstAvailable[1].AllocationMode)
	assert.Equal(t, nonDefaultCount, output.Spec.Devices.Requests[0].FirstAvailable[1].Count)
}

func TestSetDefaultDeviceTaint(t *testing.T) {
	slice := &v1.ResourceSlice{
		Spec: v1.ResourceSliceSpec{
			Devices: []v1.Device{{
				Name:   "device-0",
				Taints: []v1.DeviceTaint{{}},
			}},
		},
	}

	// fields should be defaulted
	output := roundTrip(t, slice).(*v1.ResourceSlice)
	assert.WithinDuration(t, time.Now(), ptr.Deref(output.Spec.Devices[0].Taints[0].TimeAdded, metav1.Time{}).Time, time.Minute /* allow for some processing delay */, "time added default")

	// field should not change
	timeAdded, _ := time.ParseInLocation(time.RFC3339, "2006-01-02T15:04:05Z", time.UTC)
	slice.Spec.Devices[0].Taints[0].TimeAdded = &metav1.Time{Time: timeAdded}
	output = roundTrip(t, slice).(*v1.ResourceSlice)
	assert.WithinDuration(t, timeAdded, ptr.Deref(output.Spec.Devices[0].Taints[0].TimeAdded, metav1.Time{}).Time, 0 /* semantically the same, different time zone allowed */, "time added fixed")
}

func roundTrip(t *testing.T, obj runtime.Object) runtime.Object {
	codec := legacyscheme.Codecs.LegacyCodec(v1.SchemeGroupVersion)
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
