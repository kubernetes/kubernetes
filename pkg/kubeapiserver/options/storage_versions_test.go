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

package options

import (
	"reflect"
	"testing"

	"k8s.io/apimachinery/pkg/runtime/schema"
)

func TestGenerateStorageVersionMap(t *testing.T) {
	testCases := []struct {
		storageVersions string
		defaultVersions string
		expectedMap     map[string]schema.GroupVersion
	}{
		{
			storageVersions: "v1,extensions/v1beta1",
			expectedMap: map[string]schema.GroupVersion{
				"":           {Version: "v1"},
				"extensions": {Group: "extensions", Version: "v1beta1"},
			},
		},
		{
			storageVersions: "extensions/v1beta1,v1",
			expectedMap: map[string]schema.GroupVersion{
				"":           {Version: "v1"},
				"extensions": {Group: "extensions", Version: "v1beta1"},
			},
		},
		{
			storageVersions: "autoscaling=extensions/v1beta1,v1",
			defaultVersions: "extensions/v1beta1,v1,autoscaling/v1",
			expectedMap: map[string]schema.GroupVersion{
				"":            {Version: "v1"},
				"autoscaling": {Group: "extensions", Version: "v1beta1"},
				"extensions":  {Group: "extensions", Version: "v1beta1"},
			},
		},
		{
			storageVersions: "",
			expectedMap:     map[string]schema.GroupVersion{},
		},
	}
	for i, test := range testCases {
		s := &StorageSerializationOptions{
			StorageVersions:        test.storageVersions,
			DefaultStorageVersions: test.defaultVersions,
		}
		output, err := s.StorageGroupsToEncodingVersion()
		if err != nil {
			t.Errorf("%v: unexpected error: %v", i, err)
		}
		if !reflect.DeepEqual(test.expectedMap, output) {
			t.Errorf("%v: unexpected error. expect: %v, got: %v", i, test.expectedMap, output)
		}
	}
}
