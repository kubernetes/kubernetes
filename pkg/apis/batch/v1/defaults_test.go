/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

	_ "k8s.io/kubernetes/pkg/api/install"
	"k8s.io/kubernetes/pkg/api/v1"
	. "k8s.io/kubernetes/pkg/apis/batch/v1"
	_ "k8s.io/kubernetes/pkg/apis/extensions/install"
	"k8s.io/kubernetes/pkg/runtime"
)

func TestSetDefaultJobParallelismAndCompletions(t *testing.T) {
	tests := []struct {
		original *Job
		expected *Job
	}{
		// both unspecified -> sets both to 1
		{
			original: &Job{
				Spec: JobSpec{},
			},
			expected: &Job{
				Spec: JobSpec{
					Completions: newInt32(1),
					Parallelism: newInt32(1),
				},
			},
		},
		// WQ: Parallelism explicitly 0 and completions unset -> no change
		{
			original: &Job{
				Spec: JobSpec{
					Parallelism: newInt32(0),
				},
			},
			expected: &Job{
				Spec: JobSpec{
					Parallelism: newInt32(0),
				},
			},
		},
		// WQ: Parallelism explicitly 2 and completions unset -> no change
		{
			original: &Job{
				Spec: JobSpec{
					Parallelism: newInt32(2),
				},
			},
			expected: &Job{
				Spec: JobSpec{
					Parallelism: newInt32(2),
				},
			},
		},
		// Completions explicitly 2 and parallelism unset -> parallelism is defaulted
		{
			original: &Job{
				Spec: JobSpec{
					Completions: newInt32(2),
				},
			},
			expected: &Job{
				Spec: JobSpec{
					Completions: newInt32(2),
					Parallelism: newInt32(1),
				},
			},
		},
		// Both set -> no change
		{
			original: &Job{
				Spec: JobSpec{
					Completions: newInt32(10),
					Parallelism: newInt32(11),
				},
			},
			expected: &Job{
				Spec: JobSpec{
					Completions: newInt32(10),
					Parallelism: newInt32(11),
				},
			},
		},
		// Both set, flipped -> no change
		{
			original: &Job{
				Spec: JobSpec{
					Completions: newInt32(11),
					Parallelism: newInt32(10),
				},
			},
			expected: &Job{
				Spec: JobSpec{
					Completions: newInt32(11),
					Parallelism: newInt32(10),
				},
			},
		},
	}

	for _, tc := range tests {
		original := tc.original
		expected := tc.expected
		obj2 := roundTrip(t, runtime.Object(original))
		got, ok := obj2.(*Job)
		if !ok {
			t.Errorf("unexpected object: %v", got)
			t.FailNow()
		}
		if (got.Spec.Completions == nil) != (expected.Spec.Completions == nil) {
			t.Errorf("got different *completions than expected: %v %v", got.Spec.Completions, expected.Spec.Completions)
		}
		if got.Spec.Completions != nil && expected.Spec.Completions != nil {
			if *got.Spec.Completions != *expected.Spec.Completions {
				t.Errorf("got different completions than expected: %d %d", *got.Spec.Completions, *expected.Spec.Completions)
			}
		}
		if (got.Spec.Parallelism == nil) != (expected.Spec.Parallelism == nil) {
			t.Errorf("got different *Parallelism than expected: %v %v", got.Spec.Parallelism, expected.Spec.Parallelism)
		}
		if got.Spec.Parallelism != nil && expected.Spec.Parallelism != nil {
			if *got.Spec.Parallelism != *expected.Spec.Parallelism {
				t.Errorf("got different parallelism than expected: %d %d", *got.Spec.Parallelism, *expected.Spec.Parallelism)
			}
		}
	}
}

func TestSetDefaultJobSelector(t *testing.T) {
	expected := &Job{
		Spec: JobSpec{
			Selector: &LabelSelector{
				MatchLabels: map[string]string{"job": "selector"},
			},
			Completions: newInt32(1),
			Parallelism: newInt32(1),
		},
	}
	tests := []*Job{
		// selector set explicitly, completions and parallelism - default
		{
			Spec: JobSpec{
				Selector: &LabelSelector{
					MatchLabels: map[string]string{"job": "selector"},
				},
			},
		},
		// selector from template labels, completions and parallelism - default
		{
			Spec: JobSpec{
				Template: v1.PodTemplateSpec{
					ObjectMeta: v1.ObjectMeta{
						Labels: map[string]string{"job": "selector"},
					},
				},
			},
		},
	}

	for _, original := range tests {
		obj2 := roundTrip(t, runtime.Object(original))
		got, ok := obj2.(*Job)
		if !ok {
			t.Errorf("unexpected object: %v", got)
			t.FailNow()
		}
		if !reflect.DeepEqual(got.Spec.Selector, expected.Spec.Selector) {
			t.Errorf("got different selectors %#v %#v", got.Spec.Selector, expected.Spec.Selector)
		}
	}
}

func newInt32(val int32) *int32 {
	p := new(int32)
	*p = val
	return p
}
