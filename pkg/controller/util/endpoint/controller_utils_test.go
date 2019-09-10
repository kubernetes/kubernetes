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

package endpoint

import (
	"testing"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/util/sets"
)

func TestDetermineNeededServiceUpdates(t *testing.T) {
	testCases := []struct {
		name  string
		a     sets.String
		b     sets.String
		union sets.String
		xor   sets.String
	}{
		{
			name:  "no services changed",
			a:     sets.NewString("a", "b", "c"),
			b:     sets.NewString("a", "b", "c"),
			xor:   sets.NewString(),
			union: sets.NewString("a", "b", "c"),
		},
		{
			name:  "all old services removed, new services added",
			a:     sets.NewString("a", "b", "c"),
			b:     sets.NewString("d", "e", "f"),
			xor:   sets.NewString("a", "b", "c", "d", "e", "f"),
			union: sets.NewString("a", "b", "c", "d", "e", "f"),
		},
		{
			name:  "all old services removed, no new services added",
			a:     sets.NewString("a", "b", "c"),
			b:     sets.NewString(),
			xor:   sets.NewString("a", "b", "c"),
			union: sets.NewString("a", "b", "c"),
		},
		{
			name:  "no old services, but new services added",
			a:     sets.NewString(),
			b:     sets.NewString("a", "b", "c"),
			xor:   sets.NewString("a", "b", "c"),
			union: sets.NewString("a", "b", "c"),
		},
		{
			name:  "one service removed, one service added, two unchanged",
			a:     sets.NewString("a", "b", "c"),
			b:     sets.NewString("b", "c", "d"),
			xor:   sets.NewString("a", "d"),
			union: sets.NewString("a", "b", "c", "d"),
		},
		{
			name:  "no services",
			a:     sets.NewString(),
			b:     sets.NewString(),
			xor:   sets.NewString(),
			union: sets.NewString(),
		},
	}
	for _, testCase := range testCases {
		retval := determineNeededServiceUpdates(testCase.a, testCase.b, false)
		if !retval.Equal(testCase.xor) {
			t.Errorf("%s (with podChanged=false): expected: %v  got: %v", testCase.name, testCase.xor.List(), retval.List())
		}

		retval = determineNeededServiceUpdates(testCase.a, testCase.b, true)
		if !retval.Equal(testCase.union) {
			t.Errorf("%s (with podChanged=true): expected: %v  got: %v", testCase.name, testCase.union.List(), retval.List())
		}
	}
}

// There are 3*5 possibilities(3 types of RestartPolicy by 5 types of PodPhase).
// Not listing them all here. Just listing all of the 3 false cases and 3 of the
// 12 true cases.
func TestShouldPodBeInEndpoints(t *testing.T) {
	testCases := []struct {
		name     string
		pod      *v1.Pod
		expected bool
	}{
		// Pod should not be in endpoints:
		{
			name: "Failed pod with Never RestartPolicy",
			pod: &v1.Pod{
				Spec: v1.PodSpec{
					RestartPolicy: v1.RestartPolicyNever,
				},
				Status: v1.PodStatus{
					Phase: v1.PodFailed,
					PodIP: "1.2.3.4",
				},
			},
			expected: false,
		},
		{
			name: "Succeeded pod with Never RestartPolicy",
			pod: &v1.Pod{
				Spec: v1.PodSpec{
					RestartPolicy: v1.RestartPolicyNever,
				},
				Status: v1.PodStatus{
					Phase: v1.PodSucceeded,
					PodIP: "1.2.3.4",
				},
			},
			expected: false,
		},
		{
			name: "Succeeded pod with OnFailure RestartPolicy",
			pod: &v1.Pod{
				Spec: v1.PodSpec{
					RestartPolicy: v1.RestartPolicyOnFailure,
				},
				Status: v1.PodStatus{
					Phase: v1.PodSucceeded,
					PodIP: "1.2.3.4",
				},
			},
			expected: false,
		},
		{
			name: "Empty Pod IPs, Running pod with OnFailure RestartPolicy",
			pod: &v1.Pod{
				Spec: v1.PodSpec{
					RestartPolicy: v1.RestartPolicyNever,
				},
				Status: v1.PodStatus{
					Phase:  v1.PodRunning,
					PodIP:  "",
					PodIPs: []v1.PodIP{},
				},
			},
			expected: false,
		},
		// Pod should be in endpoints:
		{
			name: "Failed pod with Always RestartPolicy",
			pod: &v1.Pod{
				Spec: v1.PodSpec{
					RestartPolicy: v1.RestartPolicyAlways,
				},
				Status: v1.PodStatus{
					Phase: v1.PodFailed,
					PodIP: "1.2.3.4",
				},
			},
			expected: true,
		},
		{
			name: "Pending pod with Never RestartPolicy",
			pod: &v1.Pod{
				Spec: v1.PodSpec{
					RestartPolicy: v1.RestartPolicyNever,
				},
				Status: v1.PodStatus{
					Phase: v1.PodPending,
					PodIP: "1.2.3.4",
				},
			},
			expected: true,
		},
		{
			name: "Unknown pod with OnFailure RestartPolicy",
			pod: &v1.Pod{
				Spec: v1.PodSpec{
					RestartPolicy: v1.RestartPolicyOnFailure,
				},
				Status: v1.PodStatus{
					Phase: v1.PodUnknown,
					PodIP: "1.2.3.4",
				},
			},
			expected: true,
		},
		{
			name: "Running pod with Never RestartPolicy",
			pod: &v1.Pod{
				Spec: v1.PodSpec{
					RestartPolicy: v1.RestartPolicyNever,
				},
				Status: v1.PodStatus{
					Phase: v1.PodRunning,
					PodIP: "1.2.3.4",
				},
			},
			expected: true,
		},
		{
			name: "Multiple Pod IPs, Running pod with OnFailure RestartPolicy",
			pod: &v1.Pod{
				Spec: v1.PodSpec{
					RestartPolicy: v1.RestartPolicyNever,
				},
				Status: v1.PodStatus{
					Phase:  v1.PodRunning,
					PodIPs: []v1.PodIP{{IP: "1.2.3.4"}, {IP: "1234::5678:0000:0000:9abc:def0"}},
				},
			},
			expected: true,
		},
	}
	for _, test := range testCases {
		result := ShouldPodBeInEndpoints(test.pod)
		if result != test.expected {
			t.Errorf("%s: expected : %t, got: %t", test.name, test.expected, result)
		}
	}
}
