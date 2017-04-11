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

package util

import (
	"reflect"
	"testing"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/apis/extensions"
)

// TestVolumeSourceFSTypeDrift ensures that for every known type of volume source (by the fields on
// a VolumeSource object that GetVolumeFSType is returning a good value.  This ensures both that we're
// returning an FSType for the VolumeSource field (protect the GetVolumeFSType method) and that we
// haven't drifted (ensure new fields in VolumeSource are covered).
func TestVolumeSourceFSTypeDrift(t *testing.T) {
	allFSTypes := GetAllFSTypesAsSet()
	val := reflect.ValueOf(api.VolumeSource{})

	for i := 0; i < val.NumField(); i++ {
		fieldVal := val.Type().Field(i)

		volumeSource := api.VolumeSource{}
		volumeSourceVolume := reflect.New(fieldVal.Type.Elem())

		reflect.ValueOf(&volumeSource).Elem().FieldByName(fieldVal.Name).Set(volumeSourceVolume)

		fsType, err := GetVolumeFSType(api.Volume{VolumeSource: volumeSource})
		if err != nil {
			t.Errorf("error getting fstype for field %s.  This likely means that drift has occured between FSType and VolumeSource.  Please update the api and getVolumeFSType", fieldVal.Name)
		}

		if !allFSTypes.Has(string(fsType)) {
			t.Errorf("%s was missing from GetFSTypesAsSet", fsType)
		}
	}
}

func TestPSPAllowsFSType(t *testing.T) {
	tests := map[string]struct {
		psp    *extensions.PodSecurityPolicy
		fsType extensions.FSType
		allows bool
	}{
		"nil psp": {
			psp:    nil,
			fsType: extensions.HostPath,
			allows: false,
		},
		"empty volumes": {
			psp:    &extensions.PodSecurityPolicy{},
			fsType: extensions.HostPath,
			allows: false,
		},
		"non-matching": {
			psp: &extensions.PodSecurityPolicy{
				Spec: extensions.PodSecurityPolicySpec{
					Volumes: []extensions.FSType{extensions.AWSElasticBlockStore},
				},
			},
			fsType: extensions.HostPath,
			allows: false,
		},
		"match on FSTypeAll": {
			psp: &extensions.PodSecurityPolicy{
				Spec: extensions.PodSecurityPolicySpec{
					Volumes: []extensions.FSType{extensions.All},
				},
			},
			fsType: extensions.HostPath,
			allows: true,
		},
		"match on direct match": {
			psp: &extensions.PodSecurityPolicy{
				Spec: extensions.PodSecurityPolicySpec{
					Volumes: []extensions.FSType{extensions.HostPath},
				},
			},
			fsType: extensions.HostPath,
			allows: true,
		},
	}

	for k, v := range tests {
		allows := PSPAllowsFSType(v.psp, v.fsType)
		if v.allows != allows {
			t.Errorf("%s expected PSPAllowsFSType to return %t but got %t", k, v.allows, allows)
		}
	}
}
