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

	"k8s.io/kubernetes/pkg/apimachinery"
	"k8s.io/kubernetes/pkg/runtime/schema"
)

func TestAddThirdPartyVersionsBasic(t *testing.T) {
	m, err := NewAPIRegistrationManager("")
	if err != nil {
		t.Fatalf("Unexpected failure to make a manager: %v", err)
	}

	registered := []schema.GroupVersion{
		{
			Group:   "",
			Version: "v1",
		},
	}
	skipped := registered
	thirdParty := []schema.GroupVersion{
		{
			Group:   "company.com",
			Version: "v1",
		},
		{
			Group:   "company.com",
			Version: "v2",
		},
	}
	gvs := append(registered, thirdParty...)

	m.RegisterVersions(registered)
	wasSkipped := m.AddThirdPartyAPIGroupVersions(gvs...)
	if len(wasSkipped) != len(skipped) {
		t.Errorf("Expected %v, found %v", skipped, wasSkipped)
	}
	for ix := range wasSkipped {
		found := false
		for _, gv := range skipped {
			if gv.String() == wasSkipped[ix].String() {
				found = true
				break
			}
		}
		if !found {
			t.Errorf("Couldn't find %v in %v", wasSkipped[ix], skipped)
		}
	}
	for _, gv := range thirdParty {
		if !m.IsThirdPartyAPIGroupVersion(gv) {
			t.Errorf("Expected %v to be third party.", gv)
		}
	}
}

func TestAddThirdPartyVersionsMultiple(t *testing.T) {
	thirdParty := []schema.GroupVersion{
		{
			Group:   "company.com",
			Version: "v1",
		},
		{
			Group:   "company.com",
			Version: "v2",
		},
	}
	m, err := NewAPIRegistrationManager("")
	if err != nil {
		t.Fatalf("Unexpected failure to make a manager: %v", err)
	}
	for _, gv := range thirdParty {
		wasSkipped := m.AddThirdPartyAPIGroupVersions(gv)
		if len(wasSkipped) != 0 {
			t.Errorf("Expected length 0, found %v", wasSkipped)
		}
	}
	for _, gv := range thirdParty {
		if !m.IsThirdPartyAPIGroupVersion(gv) {
			t.Errorf("Expected %v to be third party.", gv)
		}
	}
}

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
