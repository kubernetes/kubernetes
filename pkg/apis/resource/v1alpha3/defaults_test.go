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

package v1alpha3_test

import (
	"reflect"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"

	v1alpha3 "k8s.io/api/resource/v1alpha3"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/utils/ptr"

	// ensure types are installed
	"k8s.io/kubernetes/pkg/api/legacyscheme"
	_ "k8s.io/kubernetes/pkg/apis/resource/install"
)

func TestSetDefaultDeviceTaint(t *testing.T) {
	taintRule := &v1alpha3.DeviceTaintRule{}

	// fields should be defaulted
	output := roundTrip(t, taintRule).(*v1alpha3.DeviceTaintRule)
	assert.WithinDuration(t, time.Now(), ptr.Deref(output.Spec.Taint.TimeAdded, metav1.Time{}).Time, time.Minute /* allow for some processing delay */, "time added default")

	// field should not change
	timeAdded, _ := time.ParseInLocation(time.RFC3339, "2006-01-02T15:04:05Z", time.UTC)
	taintRule.Spec.Taint.TimeAdded = &metav1.Time{Time: timeAdded}
	output = roundTrip(t, taintRule).(*v1alpha3.DeviceTaintRule)
	assert.WithinDuration(t, timeAdded, ptr.Deref(output.Spec.Taint.TimeAdded, metav1.Time{}).Time, 0 /* semantically the same, different time zone allowed */, "time added fixed")
}

func roundTrip(t *testing.T, obj runtime.Object) runtime.Object {
	codec := legacyscheme.Codecs.LegacyCodec(v1alpha3.SchemeGroupVersion)
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
