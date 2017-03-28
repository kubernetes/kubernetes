/*
Copyright 2014 The Kubernetes Authors.

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

import "testing"

func TestGetVersion(t *testing.T) {
	testCases := []struct {
		groupVersion string
		output       string
	}{
		{
			"v1",
			"v1",
		},
		{
			"extensions/v1beta1",
			"v1beta1",
		},
	}
	for _, test := range testCases {
		actual := GetVersion(test.groupVersion)
		if test.output != actual {
			t.Errorf("expect version: %s, got: %s\n", test.output, actual)
		}
	}
}

func TestGetGroup(t *testing.T) {
	testCases := []struct {
		groupVersion string
		output       string
	}{
		{
			"v1",
			"",
		},
		{
			"extensions/v1beta1",
			"extensions",
		},
	}
	for _, test := range testCases {
		actual := GetGroup(test.groupVersion)
		if test.output != actual {
			t.Errorf("expect version: %s, got: %s\n", test.output, actual)
		}
	}
}

func TestGetGroupVersion(t *testing.T) {
	testCases := []struct {
		group   string
		version string
		output  string
	}{
		{
			"",
			"v1",
			"v1",
		},
		{
			"extensions",
			"",
			"extensions/",
		},
		{
			"extensions",
			"v1beta1",
			"extensions/v1beta1",
		},
	}
	for _, test := range testCases {
		actual := GetGroupVersion(test.group, test.version)
		if test.output != actual {
			t.Errorf("expect version: %s, got: %s\n", test.output, actual)
		}
	}
}
