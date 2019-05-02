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

package plugins

import (
	"reflect"
	"testing"

	"k8s.io/api/core/v1"
	storage "k8s.io/api/storage/v1"
)

func NewStorageClass(params map[string]string) storage.StorageClass {
	return storage.StorageClass{
		Parameters: params,
	}
}

func TestTranslatePDInTreeVolumeOptionsToCSI(t *testing.T) {
	g := NewGCEPersistentDiskCSITranslator()

	tcs := []struct {
		name       string
		options    storage.StorageClass
		expOptions storage.StorageClass
	}{
		{
			name:       "nothing special",
			options:    NewStorageClass(map[string]string{"foo": "bar"}),
			expOptions: NewStorageClass(map[string]string{"foo": "bar"}),
		},
		{
			name:       "fstype",
			options:    NewStorageClass(map[string]string{"fstype": "myfs"}),
			expOptions: NewStorageClass(map[string]string{"csi.storage.k8s.io/fstype": "myfs"}),
		},
		{
			name:       "empty params",
			options:    NewStorageClass(map[string]string{}),
			expOptions: NewStorageClass(map[string]string{}),
		},
	}

	for _, tc := range tcs {
		t.Logf("Testing %v", tc.name)
		gotOptions, err := g.TranslateInTreeVolumeOptionsToCSI(tc.options)
		if err != nil {
			t.Errorf("Did not expect error but got: %v", err)
		}
		if !reflect.DeepEqual(gotOptions, tc.expOptions) {
			t.Errorf("Got parameters: %v, expected :%v", gotOptions, tc.expOptions)
		}
	}
}

func TestBackwardCompatibleAccessModes(t *testing.T) {
	testCases := []struct {
		name           string
		accessModes    []v1.PersistentVolumeAccessMode
		expAccessModes []v1.PersistentVolumeAccessMode
	}{
		{
			name: "multiple normals",
			accessModes: []v1.PersistentVolumeAccessMode{
				v1.ReadOnlyMany,
				v1.ReadWriteOnce,
			},
			expAccessModes: []v1.PersistentVolumeAccessMode{
				v1.ReadOnlyMany,
				v1.ReadWriteOnce,
			},
		},
		{
			name: "one normal",
			accessModes: []v1.PersistentVolumeAccessMode{
				v1.ReadWriteOnce,
			},
			expAccessModes: []v1.PersistentVolumeAccessMode{
				v1.ReadWriteOnce,
			},
		},
		{
			name: "some readwritemany",
			accessModes: []v1.PersistentVolumeAccessMode{
				v1.ReadWriteOnce,
				v1.ReadWriteMany,
			},
			expAccessModes: []v1.PersistentVolumeAccessMode{
				v1.ReadWriteOnce,
				v1.ReadWriteOnce,
			},
		},
	}

	for _, tc := range testCases {
		t.Logf("running test: %v", tc.name)

		got := backwardCompatibleAccessModes(tc.accessModes)

		if !reflect.DeepEqual(tc.expAccessModes, got) {
			t.Fatalf("Expected access modes: %v, instead got: %v", tc.expAccessModes, got)
		}
	}
}
