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

package util

import (
	"testing"

	storagev1 "k8s.io/api/storage/v1"
)

type bindingTest struct {
	class    *storagev1.StorageClass
	expected bool
}

func TestIsBindingModeWaitForFirstConsumer(t *testing.T) {
	immediateMode := storagev1.VolumeBindingImmediate
	waitingMode := storagev1.VolumeBindingWaitForFirstConsumer
	cases := map[string]bindingTest{
		"nil binding mode": {
			&storagev1.StorageClass{},
			false,
		},
		"immediate binding mode": {
			&storagev1.StorageClass{VolumeBindingMode: &immediateMode},
			false,
		},
		"waiting binding mode": {
			&storagev1.StorageClass{VolumeBindingMode: &waitingMode},
			true,
		},
	}

	for testName, testCase := range cases {
		result := IsBindingModeWaitForFirstConsumer(testCase.class)
		if result != testCase.expected {
			t.Errorf("Test %q failed. Expected %v, got %v", testName, testCase.expected, result)
		}
	}

}
