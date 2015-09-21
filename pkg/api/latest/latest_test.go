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

package latest

import "testing"

func TestAllPreferredGroupVersions(t *testing.T) {
	testCases := []struct {
		groupMetaMap GroupMetaMap
		expect       string
	}{
		{
			groupMetaMap: GroupMetaMap{
				"group1": &GroupMeta{
					GroupVersion: "group1/v1",
				},
				"group2": &GroupMeta{
					GroupVersion: "group2/v2",
				},
				"": &GroupMeta{
					GroupVersion: "v1",
				},
			},
			expect: "group1/v1,group2/v2,v1",
		},
		{
			groupMetaMap: GroupMetaMap{
				"": &GroupMeta{
					GroupVersion: "v1",
				},
			},
			expect: "v1",
		},
		{
			groupMetaMap: GroupMetaMap{},
			expect:       "",
		},
	}
	for _, testCase := range testCases {
		output := testCase.groupMetaMap.AllPreferredGroupVersions()
		if testCase.expect != output {
			t.Errorf("Error. expect: %s, got: %s", testCase.expect, output)
		}
	}
}
