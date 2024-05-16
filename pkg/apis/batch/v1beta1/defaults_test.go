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

	batchv1beta1 "k8s.io/api/batch/v1beta1"

	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/kubernetes/pkg/api/legacyscheme"
	_ "k8s.io/kubernetes/pkg/apis/batch/install"
	. "k8s.io/kubernetes/pkg/apis/batch/v1beta1"
	_ "k8s.io/kubernetes/pkg/apis/core/install"
	utilpointer "k8s.io/utils/pointer"
)

func TestSetDefaultCronJob(t *testing.T) {
	tests := map[string]struct {
		original *batchv1beta1.CronJob
		expected *batchv1beta1.CronJob
	}{
		"empty batchv1beta1.CronJob should default batchv1beta1.ConcurrencyPolicy and Suspend": {
			original: &batchv1beta1.CronJob{},
			expected: &batchv1beta1.CronJob{
				Spec: batchv1beta1.CronJobSpec{
					ConcurrencyPolicy:          batchv1beta1.AllowConcurrent,
					Suspend:                    newBool(false),
					SuccessfulJobsHistoryLimit: utilpointer.Int32(3),
					FailedJobsHistoryLimit:     utilpointer.Int32(1),
				},
			},
		},
		"set fields should not be defaulted": {
			original: &batchv1beta1.CronJob{
				Spec: batchv1beta1.CronJobSpec{
					ConcurrencyPolicy:          batchv1beta1.ForbidConcurrent,
					Suspend:                    newBool(true),
					SuccessfulJobsHistoryLimit: utilpointer.Int32(5),
					FailedJobsHistoryLimit:     utilpointer.Int32(5),
				},
			},
			expected: &batchv1beta1.CronJob{
				Spec: batchv1beta1.CronJobSpec{
					ConcurrencyPolicy:          batchv1beta1.ForbidConcurrent,
					Suspend:                    newBool(true),
					SuccessfulJobsHistoryLimit: utilpointer.Int32(5),
					FailedJobsHistoryLimit:     utilpointer.Int32(5),
				},
			},
		},
	}

	for name, test := range tests {
		original := test.original
		expected := test.expected
		obj2 := roundTrip(t, runtime.Object(original))
		actual, ok := obj2.(*batchv1beta1.CronJob)
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
		if *actual.Spec.SuccessfulJobsHistoryLimit != *expected.Spec.SuccessfulJobsHistoryLimit {
			t.Errorf("%s: got different successfulJobsHistoryLimit than expected: %v %v", name, *actual.Spec.SuccessfulJobsHistoryLimit, *expected.Spec.SuccessfulJobsHistoryLimit)
		}
		if *actual.Spec.FailedJobsHistoryLimit != *expected.Spec.FailedJobsHistoryLimit {
			t.Errorf("%s: got different failedJobsHistoryLimit than expected: %v %v", name, *actual.Spec.FailedJobsHistoryLimit, *expected.Spec.FailedJobsHistoryLimit)
		}
	}
}

func roundTrip(t *testing.T, obj runtime.Object) runtime.Object {
	data, err := runtime.Encode(legacyscheme.Codecs.LegacyCodec(SchemeGroupVersion), obj)
	if err != nil {
		t.Errorf("%v\n %#v", err, obj)
		return nil
	}
	obj2, err := runtime.Decode(legacyscheme.Codecs.UniversalDecoder(), data)
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

func newBool(val bool) *bool {
	p := new(bool)
	*p = val
	return p
}
