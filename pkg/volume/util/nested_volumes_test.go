/*
Copyright 2018 The Kubernetes Authors.

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
	"io/ioutil"
	"os"
	"path"
	"testing"

	"k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/sets"
)

type testCases struct {
	name     string
	err      bool
	expected sets.String
	volname  string
	pod      v1.Pod
}

func TestGetNestedMountpoints(t *testing.T) {
	var (
		testNamespace = "test_namespace"
		testPodUID    = types.UID("test_pod_uid")
	)

	tc := []testCases{
		{
			name:     "Simple Pod",
			err:      false,
			expected: sets.NewString(),
			volname:  "vol1",
			pod: v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Namespace: testNamespace,
					UID:       testPodUID,
				},
				Spec: v1.PodSpec{
					Containers: []v1.Container{
						{
							VolumeMounts: []v1.VolumeMount{
								{MountPath: "/dir", Name: "vol1"},
							},
						},
					},
				},
			},
		},
		{
			name:     "Simple Nested Pod",
			err:      false,
			expected: sets.NewString("nested"),
			volname:  "vol1",
			pod: v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Namespace: testNamespace,
					UID:       testPodUID,
				},
				Spec: v1.PodSpec{
					Containers: []v1.Container{
						{
							VolumeMounts: []v1.VolumeMount{
								{MountPath: "/dir", Name: "vol1"},
								{MountPath: "/dir/nested", Name: "vol2"},
							},
						},
					},
				},
			},
		},
		{
			name:     "Unsorted Nested Pod",
			err:      false,
			expected: sets.NewString("nested", "nested2"),
			volname:  "vol1",
			pod: v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Namespace: testNamespace,
					UID:       testPodUID,
				},
				Spec: v1.PodSpec{
					Containers: []v1.Container{
						{
							VolumeMounts: []v1.VolumeMount{
								{MountPath: "/dir/nested/double", Name: "vol3"},
								{MountPath: "/ignore", Name: "vol4"},
								{MountPath: "/dir/nested", Name: "vol2"},
								{MountPath: "/ignore2", Name: "vol5"},
								{MountPath: "/dir", Name: "vol1"},
								{MountPath: "/dir/nested2", Name: "vol3"},
							},
						},
					},
				},
			},
		},
		{
			name:     "Multiple vol1 mounts Pod",
			err:      false,
			expected: sets.NewString("nested", "nested2"),
			volname:  "vol1",
			pod: v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Namespace: testNamespace,
					UID:       testPodUID,
				},
				Spec: v1.PodSpec{
					Containers: []v1.Container{
						{
							VolumeMounts: []v1.VolumeMount{
								{MountPath: "/dir", Name: "vol1"},
								{MountPath: "/dir/nested", Name: "vol2"},
								{MountPath: "/ignore", Name: "vol4"},
								{MountPath: "/other", Name: "vol1"},
								{MountPath: "/other/nested2", Name: "vol3"},
							},
						},
					},
				},
			},
		},
		{
			name:     "Big Pod",
			err:      false,
			volname:  "vol1",
			expected: sets.NewString("sub1/sub2/sub3", "sub1/sub2/sub4", "sub1/sub2/sub6", "sub"),
			pod: v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Namespace: testNamespace,
					UID:       testPodUID,
				},
				Spec: v1.PodSpec{
					Containers: []v1.Container{
						{
							VolumeMounts: []v1.VolumeMount{
								{MountPath: "/mnt", Name: "vol1"},
								{MountPath: "/ignore", Name: "vol2"},
								{MountPath: "/mnt/sub1/sub2/sub3", Name: "vol3"},
								{MountPath: "/mnt/sub1/sub2/sub4", Name: "vol4"},
								{MountPath: "/mnt/sub1/sub2/sub4/skip", Name: "vol5"},
								{MountPath: "/mnt/sub1/sub2/sub4/skip2", Name: "vol5a"},
								{MountPath: "/mnt/sub1/sub2/sub6", Name: "vol6"},
								{MountPath: "/mnt7", Name: "vol7"},
							},
						},
					},
					InitContainers: []v1.Container{
						{
							VolumeMounts: []v1.VolumeMount{
								{MountPath: "/mnt/dir", Name: "vol1"},
								{MountPath: "/mnt/dir_ignore", Name: "vol8"},
								{MountPath: "/ignore", Name: "vol9"},
								{MountPath: "/mnt/dir/sub", Name: "vol11"},
							},
						},
					},
				},
			},
		},
		{
			name:     "Naughty Pod",
			err:      true,
			expected: nil,
			volname:  "vol1",
			pod: v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Namespace: testNamespace,
					UID:       testPodUID,
				},
				Spec: v1.PodSpec{
					Containers: []v1.Container{
						{
							VolumeMounts: []v1.VolumeMount{
								{MountPath: "foo/../../dir", Name: "vol1"},
								{MountPath: "foo/../../dir/skip", Name: "vol10"},
							},
						},
					},
				},
			},
		},
	}
	for _, test := range tc {
		dir, err := ioutil.TempDir("", "TestMakeNestedMountpoints.")
		if err != nil {
			t.Errorf("Unexpected error trying to create temp directory: %v", err)
			return
		}
		defer os.RemoveAll(dir)

		rootdir := path.Join(dir, "vol")
		err = os.Mkdir(rootdir, 0755)
		if err != nil {
			t.Errorf("Unexpected error trying to create temp root directory: %v", err)
			return
		}

		dirs, err := getNestedMountpoints(test.volname, rootdir, test.pod)
		if test.err {
			if err == nil {
				t.Errorf("%v: expected error, got nil", test.name)
			}
			continue
		} else {
			if err != nil {
				t.Errorf("%v: expected no error, got %v", test.name, err)
				continue
			}
		}
		actual := sets.NewString(dirs...)
		if !test.expected.Equal(actual) {
			t.Errorf("%v: unexpected nested directories created:\nexpected: %v\n     got: %v", test.name, test.expected, actual)
		}
	}
}
