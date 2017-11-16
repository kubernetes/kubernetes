/*
Copyright 2015 The Kubernetes Authors.

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

package registered

import (
	"testing"

	"k8s.io/apimachinery/pkg/apimachinery"
	"k8s.io/apimachinery/pkg/runtime/schema"
)

func TestAllPreferredGroupVersions(t *testing.T) {
	testCases := []struct {
		groupMetas []apimachinery.GroupMeta
		expect     string
	}{
		{
			groupMetas: []apimachinery.GroupMeta{
				{
					GroupVersion: schema.GroupVersion{Group: "group1", Version: "v1"},
				},
				{
					GroupVersion: schema.GroupVersion{Group: "group2", Version: "v2"},
				},
				{
					GroupVersion: schema.GroupVersion{Group: "", Version: "v1"},
				},
			},
			expect: "group1/v1,group2/v2,v1",
		},
		{
			groupMetas: []apimachinery.GroupMeta{
				{
					GroupVersion: schema.GroupVersion{Group: "", Version: "v1"},
				},
			},
			expect: "v1",
		},
		{
			groupMetas: []apimachinery.GroupMeta{},
			expect:     "",
		},
	}
	for _, testCase := range testCases {
		m, err := NewAPIRegistrationManager("")
		if err != nil {
			t.Fatalf("Unexpected failure to make a manager: %v", err)
		}
		for _, groupMeta := range testCase.groupMetas {
			m.RegisterGroup(groupMeta)
		}
		output := m.AllPreferredGroupVersions()
		if testCase.expect != output {
			t.Errorf("Error. expect: %s, got: %s", testCase.expect, output)
		}
	}
}
