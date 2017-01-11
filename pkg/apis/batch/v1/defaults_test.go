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

package v1_test

import (
	"reflect"
	"testing"

	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/kubernetes/pkg/api"
	_ "k8s.io/kubernetes/pkg/api/install"
	"k8s.io/kubernetes/pkg/api/v1"
	_ "k8s.io/kubernetes/pkg/apis/batch/install"
	. "k8s.io/kubernetes/pkg/apis/batch/v1"
)

func TestSetDefaultJob(t *testing.T) {
	defaultLabels := map[string]string{"default": "default"}
	tests := map[string]struct {
		original     *Job
		expected     *Job
		expectLabels bool
	}{
		"both unspecified -> sets both to 1": {
			original: &Job{
				Spec: JobSpec{
					Template: v1.PodTemplateSpec{
						ObjectMeta: v1.ObjectMeta{Labels: defaultLabels},
					},
				},
			},
			expected: &Job{
				Spec: JobSpec{
					Completions: newInt32(1),
					Parallelism: newInt32(1),
				},
			},
			expectLabels: true,
		},
		"both unspecified -> sets both to 1 and no default labels": {
			original: &Job{
				ObjectMeta: v1.ObjectMeta{
					Labels: map[string]string{"mylabel": "myvalue"},
				},
				Spec: JobSpec{
					Template: v1.PodTemplateSpec{
						ObjectMeta: v1.ObjectMeta{Labels: defaultLabels},
					},
				},
			},
			expected: &Job{
				Spec: JobSpec{
					Completions: newInt32(1),
					Parallelism: newInt32(1),
				},
			},
		},
		"WQ: Parallelism explicitly 0 and completions unset -> no change": {
			original: &Job{
				Spec: JobSpec{
					Parallelism: newInt32(0),
					Template: v1.PodTemplateSpec{
						ObjectMeta: v1.ObjectMeta{Labels: defaultLabels},
					},
				},
			},
			expected: &Job{
				Spec: JobSpec{
					Parallelism: newInt32(0),
				},
			},
			expectLabels: true,
		},
		"WQ: Parallelism explicitly 2 and completions unset -> no change": {
			original: &Job{
				Spec: JobSpec{
					Parallelism: newInt32(2),
					Template: v1.PodTemplateSpec{
						ObjectMeta: v1.ObjectMeta{Labels: defaultLabels},
					},
				},
			},
			expected: &Job{
				Spec: JobSpec{
					Parallelism: newInt32(2),
				},
			},
			expectLabels: true,
		},
		"Completions explicitly 2 and parallelism unset -> parallelism is defaulted": {
			original: &Job{
				Spec: JobSpec{
					Completions: newInt32(2),
					Template: v1.PodTemplateSpec{
						ObjectMeta: v1.ObjectMeta{Labels: defaultLabels},
					},
				},
			},
			expected: &Job{
				Spec: JobSpec{
					Completions: newInt32(2),
					Parallelism: newInt32(1),
				},
			},
			expectLabels: true,
		},
		"Both set -> no change": {
			original: &Job{
				Spec: JobSpec{
					Completions: newInt32(10),
					Parallelism: newInt32(11),
					Template: v1.PodTemplateSpec{
						ObjectMeta: v1.ObjectMeta{Labels: defaultLabels},
					},
				},
			},
			expected: &Job{
				Spec: JobSpec{
					Completions: newInt32(10),
					Parallelism: newInt32(11),
					Template: v1.PodTemplateSpec{
						ObjectMeta: v1.ObjectMeta{Labels: defaultLabels},
					},
				},
			},
			expectLabels: true,
		},
		"Both set, flipped -> no change": {
			original: &Job{
				Spec: JobSpec{
					Completions: newInt32(11),
					Parallelism: newInt32(10),
					Template: v1.PodTemplateSpec{
						ObjectMeta: v1.ObjectMeta{Labels: defaultLabels},
					},
				},
			},
			expected: &Job{
				Spec: JobSpec{
					Completions: newInt32(11),
					Parallelism: newInt32(10),
				},
			},
			expectLabels: true,
		},
	}

	for name, test := range tests {
		original := test.original
		expected := test.expected
		obj2 := roundTrip(t, runtime.Object(original))
		actual, ok := obj2.(*Job)
		if !ok {
			t.Errorf("%s: unexpected object: %v", name, actual)
			t.FailNow()
		}
		if (actual.Spec.Completions == nil) != (expected.Spec.Completions == nil) {
			t.Errorf("%s: got different *completions than expected: %v %v", name, actual.Spec.Completions, expected.Spec.Completions)
		}
		if actual.Spec.Completions != nil && expected.Spec.Completions != nil {
			if *actual.Spec.Completions != *expected.Spec.Completions {
				t.Errorf("%s: got different completions than expected: %d %d", name, *actual.Spec.Completions, *expected.Spec.Completions)
			}
		}
		if (actual.Spec.Parallelism == nil) != (expected.Spec.Parallelism == nil) {
			t.Errorf("%s: got different *Parallelism than expected: %v %v", name, actual.Spec.Parallelism, expected.Spec.Parallelism)
		}
		if actual.Spec.Parallelism != nil && expected.Spec.Parallelism != nil {
			if *actual.Spec.Parallelism != *expected.Spec.Parallelism {
				t.Errorf("%s: got different parallelism than expected: %d %d", name, *actual.Spec.Parallelism, *expected.Spec.Parallelism)
			}
		}
		if test.expectLabels != reflect.DeepEqual(actual.Labels, actual.Spec.Template.Labels) {
			if test.expectLabels {
				t.Errorf("%s: expected: %v, got: %v", name, actual.Spec.Template.Labels, actual.Labels)
			} else {
				t.Errorf("%s: unexpected equality: %v", name, actual.Labels)
			}
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

func newInt32(val int32) *int32 {
	p := new(int32)
	*p = val
	return p
}
