/*
Copyright 2016 The Kubernetes Authors.

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

package v2alpha1_test

import (
	"reflect"
	"testing"

	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/kubernetes/pkg/api"
	_ "k8s.io/kubernetes/pkg/api/install"
	_ "k8s.io/kubernetes/pkg/apis/batch/install"
	. "k8s.io/kubernetes/pkg/apis/batch/v2alpha1"
)

func TestSetDefaultCronJob(t *testing.T) {
	tests := map[string]struct {
		original *CronJob
		expected *CronJob
	}{
		"empty CronJob should default ConcurrencyPolicy and Suspend": {
			original: &CronJob{},
			expected: &CronJob{
				Spec: CronJobSpec{
					ConcurrencyPolicy: AllowConcurrent,
					Suspend:           newBool(false),
				},
			},
		},
		"nothing should be defaulted": {
			original: &CronJob{
				Spec: CronJobSpec{
					ConcurrencyPolicy: ForbidConcurrent,
					Suspend:           newBool(true),
				},
			},
			expected: &CronJob{
				Spec: CronJobSpec{
					ConcurrencyPolicy: ForbidConcurrent,
					Suspend:           newBool(true),
				},
			},
		},
	}

	for name, test := range tests {
		original := test.original
		expected := test.expected
		obj2 := roundTrip(t, runtime.Object(original))
		actual, ok := obj2.(*CronJob)
		if !ok {
			t.Errorf("%s: unexpected object: %v", name, actual)
			t.FailNow()
		}
		if actual.Spec.ConcurrencyPolicy != expected.Spec.ConcurrencyPolicy {
			t.Errorf("%s: got different concurrencyPolicy than expected: %v %v", name, actual.Spec.ConcurrencyPolicy, expected.Spec.ConcurrencyPolicy)
		}
		if *actual.Spec.Suspend != *expected.Spec.Suspend {
			t.Errorf("%s: got different suspend than expected: %v %v", name, *actual.Spec.Suspend, *expected.Spec.Suspend)
		}
	}
}

func roundTrip(t *testing.T, obj runtime.Object) runtime.Object {
	data, err := runtime.Encode(api.Codecs.LegacyCodec(SchemeGroupVersion), obj)
	if err != nil {
		t.Errorf("%v\n %#v", err, obj)
		return nil
	}
	obj2, err := runtime.Decode(api.Codecs.UniversalDecoder(), data)
	if err != nil {
		t.Errorf("%v\nData: %s\nSource: %#v", err, string(data), obj)
		return nil
	}
	obj3 := reflect.New(reflect.TypeOf(obj).Elem()).Interface().(runtime.Object)
	err = api.Scheme.Convert(obj2, obj3, nil)
	if err != nil {
		t.Errorf("%v\nSource: %#v", err, obj2)
		return nil
	}
	return obj3
}

func newBool(val bool) *bool {
	p := new(bool)
	*p = val
	return p
}
