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

package core

import (
	"reflect"
	"testing"

	"github.com/davecgh/go-spew/spew"
	"k8s.io/apimachinery/pkg/api/resource"
)

func TestResourceNameString(t *testing.T) {
	testResourceName := ResourceName("secret")
	expectedResourceNameString := "secret"

	if testResourceName.String() != expectedResourceNameString {
		t.Errorf("expected ResourceName string to be %q, but got %q instead", expectedResourceNameString, testResourceName.String())
	}
}

func TestResourceListCpu(t *testing.T) {
	testCases := []struct {
		description  string
		resourceList ResourceList
		expected     resource.Quantity
	}{
		{
			description:  "ResourceCPU is present in ResourceList",
			resourceList: ResourceList{ResourceCPU: resource.MustParse("100m")},
			expected:     resource.MustParse("100m"),
		},
		{
			description:  "ResourceCPU is not present in ResourceList",
			resourceList: ResourceList{ResourceConfigMaps: resource.MustParse("10")},
			expected:     resource.Quantity{Format: resource.DecimalSI},
		},
	}

	for _, test := range testCases {
		rq := test.resourceList.Cpu()

		if !reflect.DeepEqual(rq, &test.expected) {
			t.Errorf("test - %q - failed\nexpected Quantity to be -\n%q\nbut got -\n%q", test.description, spew.Sdump(test.expected), spew.Sdump(*rq))
		}
	}
}

func TestResourceListMemory(t *testing.T) {
	testCases := []struct {
		description  string
		resourceList ResourceList
		expected     resource.Quantity
	}{
		{
			description:  "ResourceMemory is present in ResourceList",
			resourceList: ResourceList{ResourceMemory: resource.MustParse("4Gi")},
			expected:     resource.MustParse("4Gi"),
		},
		{
			description:  "ResourceMemory is not present in ResourceList",
			resourceList: ResourceList{ResourceConfigMaps: resource.MustParse("10")},
			expected:     resource.Quantity{Format: resource.BinarySI},
		},
	}

	for _, test := range testCases {
		rq := test.resourceList.Memory()

		if !reflect.DeepEqual(rq, &test.expected) {
			t.Errorf("test - %q - failed\nexpected Quantity to be -\n%q\nbut got -\n%q", test.description, spew.Sdump(test.expected), spew.Sdump(*rq))
		}
	}
}

func TestResourceListPods(t *testing.T) {
	testCases := []struct {
		description  string
		resourceList ResourceList
		expected     resource.Quantity
	}{
		{
			description:  "ResourcePods is present in ResourceList",
			resourceList: ResourceList{ResourcePods: resource.MustParse("42")},
			expected:     resource.MustParse("42"),
		},
		{
			description:  "ResourcePods is not present in ResourceList",
			resourceList: ResourceList{ResourceConfigMaps: resource.MustParse("10")},
			expected:     resource.Quantity{},
		},
	}

	for _, test := range testCases {
		rq := test.resourceList.Pods()

		if !reflect.DeepEqual(rq, &test.expected) {
			t.Errorf("test - %q - failed\nexpected Quantity to be -\n%q\nbut got -\n%q", test.description, spew.Sdump(test.expected), spew.Sdump(*rq))
		}
	}
}

func TestResourceListNvidiaGPU(t *testing.T) {
	testCases := []struct {
		description  string
		resourceList ResourceList
		expected     resource.Quantity
	}{
		{
			description:  "ResourceNvidiaGPU is present in ResourceList",
			resourceList: ResourceList{ResourceNvidiaGPU: resource.MustParse("1")},
			expected:     resource.MustParse("1"),
		},
		{
			description:  "ResourceNvidiaGPU is not present in ResourceList",
			resourceList: ResourceList{ResourceConfigMaps: resource.MustParse("10")},
			expected:     resource.Quantity{},
		},
	}

	for _, test := range testCases {
		rq := test.resourceList.NvidiaGPU()

		if !reflect.DeepEqual(rq, &test.expected) {
			t.Errorf("test - %q - failed\nexpected Quantity to be -\n%q\nbut got -\n%q", test.description, spew.Sdump(test.expected), spew.Sdump(*rq))
		}
	}
}

func TestResourceListStorageEphemeral(t *testing.T) {
	testCases := []struct {
		description  string
		resourceList ResourceList
		expected     resource.Quantity
	}{
		{
			description:  "ResourceEphemeralStorage is present in ResourceList",
			resourceList: ResourceList{ResourceEphemeralStorage: resource.MustParse("32Mi")},
			expected:     resource.MustParse("32Mi"),
		},
		{
			description:  "ResourceNvidiaGPU is not present in ResourceList",
			resourceList: ResourceList{ResourceConfigMaps: resource.MustParse("10")},
			expected:     resource.Quantity{},
		},
	}

	for _, test := range testCases {
		rq := test.resourceList.StorageEphemeral()

		if !reflect.DeepEqual(rq, &test.expected) {
			t.Errorf("test - %q - failed\nexpected Quantity to be -\n%q\nbut got -\n%q", test.description, spew.Sdump(test.expected), spew.Sdump(*rq))
		}
	}
}
