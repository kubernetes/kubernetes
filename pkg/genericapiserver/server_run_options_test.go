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

package genericapiserver

import (
	"reflect"
	"testing"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/apis/autoscaling"
	"k8s.io/kubernetes/pkg/apis/extensions"
	"k8s.io/kubernetes/pkg/genericapiserver/options"
	"k8s.io/kubernetes/pkg/runtime/schema"
)

func TestGenerateStorageVersionMap(t *testing.T) {
	testCases := []struct {
		legacyVersion   string
		storageVersions string
		defaultVersions string
		expectedMap     map[string]schema.GroupVersion
	}{
		{
			legacyVersion:   "v1",
			storageVersions: "v1,extensions/v1beta1",
			expectedMap: map[string]schema.GroupVersion{
				api.GroupName:        {Version: "v1"},
				extensions.GroupName: {Group: "extensions", Version: "v1beta1"},
			},
		},
		{
			legacyVersion:   "",
			storageVersions: "extensions/v1beta1,v1",
			expectedMap: map[string]schema.GroupVersion{
				api.GroupName:        {Version: "v1"},
				extensions.GroupName: {Group: "extensions", Version: "v1beta1"},
			},
		},
		{
			legacyVersion:   "",
			storageVersions: "autoscaling=extensions/v1beta1,v1",
			defaultVersions: "extensions/v1beta1,v1,autoscaling/v1",
			expectedMap: map[string]schema.GroupVersion{
				api.GroupName:         {Version: "v1"},
				autoscaling.GroupName: {Group: "extensions", Version: "v1beta1"},
				extensions.GroupName:  {Group: "extensions", Version: "v1beta1"},
			},
		},
		{
			legacyVersion:   "",
			storageVersions: "",
			expectedMap:     map[string]schema.GroupVersion{},
		},
	}
	for i, test := range testCases {
		s := options.ServerRunOptions{
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
