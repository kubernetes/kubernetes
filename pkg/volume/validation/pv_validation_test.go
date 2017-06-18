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

package validation

import (
	"testing"

	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/kubernetes/pkg/api"
)

func TestValidatePersistentVolumes(t *testing.T) {
	scenarios := map[string]struct {
		isExpectedFailure bool
		volume            *api.PersistentVolume
	}{
		"volume with valid mount option for nfs": {
			isExpectedFailure: false,
			volume: testVolumeWithMountOption("good-nfs-mount-volume", "", "ro,nfsvers=3", api.PersistentVolumeSpec{
				Capacity: api.ResourceList{
					api.ResourceName(api.ResourceStorage): resource.MustParse("10G"),
				},
				AccessModes: []api.PersistentVolumeAccessMode{api.ReadWriteOnce},
				PersistentVolumeSource: api.PersistentVolumeSource{
					NFS: &api.NFSVolumeSource{Server: "localhost", Path: "/srv", ReadOnly: false},
				},
			}),
		},
		"volume with mount option for host path": {
			isExpectedFailure: true,
			volume: testVolumeWithMountOption("bad-hostpath-mount-volume", "", "ro,nfsvers=3", api.PersistentVolumeSpec{
				Capacity: api.ResourceList{
					api.ResourceName(api.ResourceStorage): resource.MustParse("10G"),
				},
				AccessModes: []api.PersistentVolumeAccessMode{api.ReadWriteOnce},
				PersistentVolumeSource: api.PersistentVolumeSource{
					HostPath: &api.HostPathVolumeSource{Path: "/a/.."},
				},
			}),
		},
	}

	for name, scenario := range scenarios {
		errs := ValidatePersistentVolume(scenario.volume)
		if len(errs) == 0 && scenario.isExpectedFailure {
			t.Errorf("Unexpected success for scenario: %s", name)
		}
		if len(errs) > 0 && !scenario.isExpectedFailure {
			t.Errorf("Unexpected failure for scenario: %s - %+v", name, errs)
		}
	}
}

func testVolumeWithMountOption(name string, namespace string, mountOptions string, spec api.PersistentVolumeSpec) *api.PersistentVolume {
	annotations := map[string]string{
		api.MountOptionAnnotation: mountOptions,
	}
	objMeta := metav1.ObjectMeta{
		Name:        name,
		Annotations: annotations,
	}

	if namespace != "" {
		objMeta.Namespace = namespace
	}

	return &api.PersistentVolume{
		ObjectMeta: objMeta,
		Spec:       spec,
	}
}

func TestValidatePathNoBacksteps(t *testing.T) {
	testCases := map[string]struct {
		path        string
		expectedErr bool
	}{
		"valid path": {
			path: "/foo/bar",
		},
		"invalid path": {
			path:        "/foo/bar/..",
			expectedErr: true,
		},
	}

	for name, tc := range testCases {
		err := ValidatePathNoBacksteps(tc.path)

		if err == nil && tc.expectedErr {
			t.Fatalf("expected test `%s` to return an error but it didnt", name)
		}

		if err != nil && !tc.expectedErr {
			t.Fatalf("expected test `%s` to return no error but got `%v`", name, err)
		}
	}
}
