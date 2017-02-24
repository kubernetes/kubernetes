// +build linux

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

package cm

import "testing"

func TestLibcontainerAdapterAdaptToSystemd(t *testing.T) {
	testCases := []struct {
		input    string
		expected string
	}{
		{
			input:    "/",
			expected: "-.slice",
		},
		{
			input:    "/Burstable",
			expected: "Burstable.slice",
		},
		{
			input:    "/Burstable/pod_123",
			expected: "Burstable-pod_123.slice",
		},
		{
			input:    "/BestEffort/pod_6c1a4e95-6bb6-11e6-bc26-28d2444e470d",
			expected: "BestEffort-pod_6c1a4e95_6bb6_11e6_bc26_28d2444e470d.slice",
		},
	}
	for _, testCase := range testCases {
		f := newLibcontainerAdapter(libcontainerSystemd)
		if actual := f.adaptName(CgroupName(testCase.input), false); actual != testCase.expected {
			t.Errorf("Unexpected result, input: %v, expected: %v, actual: %v", testCase.input, testCase.expected, actual)
		}
	}
}

func TestLibcontainerAdapterAdaptToSystemdAsCgroupFs(t *testing.T) {
	testCases := []struct {
		input    string
		expected string
	}{
		{
			input:    "/",
			expected: "/",
		},
		{
			input:    "/Burstable",
			expected: "Burstable.slice/",
		},
		{
			input:    "/Burstable/pod_123",
			expected: "Burstable.slice/Burstable-pod_123.slice/",
		},
		{
			input:    "/BestEffort/pod_6c1a4e95-6bb6-11e6-bc26-28d2444e470d",
			expected: "BestEffort.slice/BestEffort-pod_6c1a4e95_6bb6_11e6_bc26_28d2444e470d.slice/",
		},
	}
	for _, testCase := range testCases {
		f := newLibcontainerAdapter(libcontainerSystemd)
		if actual := f.adaptName(CgroupName(testCase.input), true); actual != testCase.expected {
			t.Errorf("Unexpected result, input: %v, expected: %v, actual: %v", testCase.input, testCase.expected, actual)
		}
	}
}
