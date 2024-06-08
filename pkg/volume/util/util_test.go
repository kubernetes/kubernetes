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

package util

import (
	"os"
	"reflect"
	"runtime"
	"strings"
	"testing"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/sets"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	_ "k8s.io/kubernetes/pkg/apis/core/install"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/util/slice"
	"k8s.io/kubernetes/pkg/volume"
	utilptr "k8s.io/utils/pointer"
)

func TestLoadPodFromFile(t *testing.T) {
	tests := []struct {
		name        string
		content     string
		expectError bool
	}{
		{
			"yaml",
			`
apiVersion: v1
kind: Pod
metadata:
  name: testpod
spec:
  containers:
    - image: registry.k8s.io/busybox
`,
			false,
		},

		{
			"json",
			`
{
  "apiVersion": "v1",
  "kind": "Pod",
  "metadata": {
    "name": "testpod"
  },
  "spec": {
    "containers": [
      {
        "image": "registry.k8s.io/busybox"
      }
    ]
  }
}`,
			false,
		},

		{
			"invalid pod",
			`
apiVersion: v1
kind: Pod
metadata:
  name: testpod
spec:
  - image: registry.k8s.io/busybox
`,
			true,
		},
	}

	for _, test := range tests {
		tempFile, err := os.CreateTemp("", "podfile")
		defer os.Remove(tempFile.Name())
		if err != nil {
			t.Fatalf("cannot create temporary file: %v", err)
		}
		if _, err = tempFile.Write([]byte(test.content)); err != nil {
			t.Fatalf("cannot save temporary file: %v", err)
		}
		if err = tempFile.Close(); err != nil {
			t.Fatalf("cannot close temporary file: %v", err)
		}

		pod, err := LoadPodFromFile(tempFile.Name())
		if test.expectError {
			if err == nil {
				t.Errorf("test %q expected error, got nil", test.name)
			}
		} else {
			// no error expected
			if err != nil {
				t.Errorf("error loading pod %q: %v", test.name, err)
			}
			if pod == nil {
				t.Errorf("test %q expected pod, got nil", test.name)
			}
		}
	}
}

func TestCalculateTimeoutForVolume(t *testing.T) {
	pv := &v1.PersistentVolume{
		Spec: v1.PersistentVolumeSpec{
			Capacity: v1.ResourceList{
				v1.ResourceName(v1.ResourceStorage): resource.MustParse("500M"),
			},
		},
	}

	timeout := CalculateTimeoutForVolume(50, 30, pv)
	if timeout != 50 {
		t.Errorf("Expected 50 for timeout but got %v", timeout)
	}

	pv.Spec.Capacity[v1.ResourceStorage] = resource.MustParse("2Gi")
	timeout = CalculateTimeoutForVolume(50, 30, pv)
	if timeout != 60 {
		t.Errorf("Expected 60 for timeout but got %v", timeout)
	}

	pv.Spec.Capacity[v1.ResourceStorage] = resource.MustParse("150Gi")
	timeout = CalculateTimeoutForVolume(50, 30, pv)
	if timeout != 4500 {
		t.Errorf("Expected 4500 for timeout but got %v", timeout)
	}
}

func TestFsUserFrom(t *testing.T) {
	tests := []struct {
		desc       string
		pod        *v1.Pod
		wantFsUser *int64
	}{
		{
			desc: "no runAsUser specified",
			pod: &v1.Pod{
				Spec: v1.PodSpec{},
			},
			wantFsUser: nil,
		},
		{
			desc: "some have runAsUser specified",
			pod: &v1.Pod{
				Spec: v1.PodSpec{
					SecurityContext: &v1.PodSecurityContext{},
					InitContainers: []v1.Container{
						{
							SecurityContext: &v1.SecurityContext{
								RunAsUser: utilptr.Int64Ptr(1000),
							},
						},
					},
					Containers: []v1.Container{
						{
							SecurityContext: &v1.SecurityContext{
								RunAsUser: utilptr.Int64Ptr(1000),
							},
						},
						{
							SecurityContext: &v1.SecurityContext{},
						},
					},
				},
			},
			wantFsUser: nil,
		},
		{
			desc: "all have runAsUser specified but not the same",
			pod: &v1.Pod{
				Spec: v1.PodSpec{
					SecurityContext: &v1.PodSecurityContext{},
					InitContainers: []v1.Container{
						{
							SecurityContext: &v1.SecurityContext{
								RunAsUser: utilptr.Int64Ptr(999),
							},
						},
					},
					Containers: []v1.Container{
						{
							SecurityContext: &v1.SecurityContext{
								RunAsUser: utilptr.Int64Ptr(1000),
							},
						},
						{
							SecurityContext: &v1.SecurityContext{
								RunAsUser: utilptr.Int64Ptr(1000),
							},
						},
					},
				},
			},
			wantFsUser: nil,
		},
		{
			desc: "all have runAsUser specified and the same",
			pod: &v1.Pod{
				Spec: v1.PodSpec{
					SecurityContext: &v1.PodSecurityContext{},
					InitContainers: []v1.Container{
						{
							SecurityContext: &v1.SecurityContext{
								RunAsUser: utilptr.Int64Ptr(1000),
							},
						},
					},
					Containers: []v1.Container{
						{
							SecurityContext: &v1.SecurityContext{
								RunAsUser: utilptr.Int64Ptr(1000),
							},
						},
						{
							SecurityContext: &v1.SecurityContext{
								RunAsUser: utilptr.Int64Ptr(1000),
							},
						},
					},
				},
			},
			wantFsUser: utilptr.Int64Ptr(1000),
		},
	}

	for _, test := range tests {
		t.Run(test.desc, func(t *testing.T) {
			fsUser := FsUserFrom(test.pod)
			if fsUser == nil && test.wantFsUser != nil {
				t.Errorf("FsUserFrom(%v) = %v, want %d", test.pod, fsUser, *test.wantFsUser)
			}
			if fsUser != nil && test.wantFsUser == nil {
				t.Errorf("FsUserFrom(%v) = %d, want %v", test.pod, *fsUser, test.wantFsUser)
			}
			if fsUser != nil && test.wantFsUser != nil && *fsUser != *test.wantFsUser {
				t.Errorf("FsUserFrom(%v) = %d, want %d", test.pod, *fsUser, *test.wantFsUser)
			}
		})
	}
}

func TestGenerateVolumeName(t *testing.T) {

	// Normal operation, no truncate
	v1 := GenerateVolumeName("kubernetes", "pv-cinder-abcde", 255)
	if v1 != "kubernetes-dynamic-pv-cinder-abcde" {
		t.Errorf("Expected kubernetes-dynamic-pv-cinder-abcde, got %s", v1)
	}

	// Truncate trailing "6789-dynamic"
	prefix := strings.Repeat("0123456789", 9) // 90 characters prefix + 8 chars. of "-dynamic"
	v2 := GenerateVolumeName(prefix, "pv-cinder-abcde", 100)
	expect := prefix[:84] + "-pv-cinder-abcde"
	if v2 != expect {
		t.Errorf("Expected %s, got %s", expect, v2)
	}

	// Truncate really long cluster name
	prefix = strings.Repeat("0123456789", 1000) // 10000 characters prefix
	v3 := GenerateVolumeName(prefix, "pv-cinder-abcde", 100)
	if v3 != expect {
		t.Errorf("Expected %s, got %s", expect, v3)
	}
}

func TestHasMountRefs(t *testing.T) {
	testCases := map[string]struct {
		mountPath string
		mountRefs []string
		expected  bool
	}{
		"plugin mounts only": {
			mountPath: "/var/lib/kubelet/plugins/kubernetes.io/some-plugin/mounts/volume-XXXX",
			mountRefs: []string{
				"/home/somewhere/var/lib/kubelet/plugins/kubernetes.io/some-plugin/mounts/volume-XXXX",
				"/var/lib/kubelet/plugins/kubernetes.io/some-plugin/mounts/volume-XXXX",
				"/mnt/kubelet/plugins/kubernetes.io/some-plugin/mounts/volume-XXXX",
				"/mnt/plugins/kubernetes.io/some-plugin/mounts/volume-XXXX",
			},
			expected: false,
		},
		"extra local mount": {
			mountPath: "/var/lib/kubelet/plugins/kubernetes.io/some-plugin/mounts/volume-XXXX",
			mountRefs: []string{
				"/home/somewhere/var/lib/kubelet/plugins/kubernetes.io/some-plugin/mounts/volume-XXXX",
				"/local/data/kubernetes.io/some-plugin/mounts/volume-XXXX",
				"/mnt/kubelet/plugins/kubernetes.io/some-plugin/mounts/volume-XXXX",
				"/mnt/plugins/kubernetes.io/some-plugin/mounts/volume-XXXX",
			},
			expected: true,
		},
	}
	for name, test := range testCases {
		actual := HasMountRefs(test.mountPath, test.mountRefs)
		if actual != test.expected {
			t.Errorf("for %s expected %v but got %v", name, test.expected, actual)
		}
	}
}

func TestMountOptionFromSpec(t *testing.T) {
	scenarios := map[string]struct {
		volume            *volume.Spec
		expectedMountList []string
		systemOptions     []string
	}{
		"volume-with-mount-options": {
			volume: createVolumeSpecWithMountOption("good-mount-opts", "ro,nfsvers=3", v1.PersistentVolumeSpec{
				PersistentVolumeSource: v1.PersistentVolumeSource{
					NFS: &v1.NFSVolumeSource{Server: "localhost", Path: "/srv", ReadOnly: false},
				},
			}),
			expectedMountList: []string{"ro", "nfsvers=3"},
			systemOptions:     nil,
		},
		"volume-with-bad-mount-options": {
			volume: createVolumeSpecWithMountOption("good-mount-opts", "", v1.PersistentVolumeSpec{
				PersistentVolumeSource: v1.PersistentVolumeSource{
					NFS: &v1.NFSVolumeSource{Server: "localhost", Path: "/srv", ReadOnly: false},
				},
			}),
			expectedMountList: []string{},
			systemOptions:     nil,
		},
		"vol-with-sys-opts": {
			volume: createVolumeSpecWithMountOption("good-mount-opts", "ro,nfsvers=3", v1.PersistentVolumeSpec{
				PersistentVolumeSource: v1.PersistentVolumeSource{
					NFS: &v1.NFSVolumeSource{Server: "localhost", Path: "/srv", ReadOnly: false},
				},
			}),
			expectedMountList: []string{"ro", "nfsvers=3", "fsid=100", "hard"},
			systemOptions:     []string{"fsid=100", "hard"},
		},
		"vol-with-sys-opts-with-dup": {
			volume: createVolumeSpecWithMountOption("good-mount-opts", "ro,nfsvers=3", v1.PersistentVolumeSpec{
				PersistentVolumeSource: v1.PersistentVolumeSource{
					NFS: &v1.NFSVolumeSource{Server: "localhost", Path: "/srv", ReadOnly: false},
				},
			}),
			expectedMountList: []string{"ro", "nfsvers=3", "fsid=100"},
			systemOptions:     []string{"fsid=100", "ro"},
		},
	}

	for name, scenario := range scenarios {
		mountOptions := MountOptionFromSpec(scenario.volume, scenario.systemOptions...)
		if !reflect.DeepEqual(slice.SortStrings(mountOptions), slice.SortStrings(scenario.expectedMountList)) {
			t.Errorf("for %s expected mount options : %v got %v", name, scenario.expectedMountList, mountOptions)
		}
	}
}

func createVolumeSpecWithMountOption(name string, mountOptions string, spec v1.PersistentVolumeSpec) *volume.Spec {
	annotations := map[string]string{
		v1.MountOptionAnnotation: mountOptions,
	}
	objMeta := metav1.ObjectMeta{
		Name:        name,
		Annotations: annotations,
	}

	pv := &v1.PersistentVolume{
		ObjectMeta: objMeta,
		Spec:       spec,
	}
	return &volume.Spec{PersistentVolume: pv}
}

func TestGetWindowsPath(t *testing.T) {
	tests := []struct {
		path         string
		expectedPath string
	}{
		{
			path:         `/var/lib/kubelet/pods/146f8428-83e7-11e7-8dd4-000d3a31dac4/volumes/kubernetes.io~disk`,
			expectedPath: `c:\var\lib\kubelet\pods\146f8428-83e7-11e7-8dd4-000d3a31dac4\volumes\kubernetes.io~disk`,
		},
		{
			path:         `\var/lib/kubelet/pods/146f8428-83e7-11e7-8dd4-000d3a31dac4\volumes\kubernetes.io~disk`,
			expectedPath: `c:\var\lib\kubelet\pods\146f8428-83e7-11e7-8dd4-000d3a31dac4\volumes\kubernetes.io~disk`,
		},
		{
			path:         `/`,
			expectedPath: `c:\`,
		},
		{
			path:         ``,
			expectedPath: ``,
		},
	}

	for _, test := range tests {
		result := GetWindowsPath(test.path)
		if result != test.expectedPath {
			t.Errorf("GetWindowsPath(%v) returned (%v), want (%v)", test.path, result, test.expectedPath)
		}
	}
}

func TestIsWindowsUNCPath(t *testing.T) {
	tests := []struct {
		goos      string
		path      string
		isUNCPath bool
	}{
		{
			goos:      "linux",
			path:      `/usr/bin`,
			isUNCPath: false,
		},
		{
			goos:      "linux",
			path:      `\\.\pipe\foo`,
			isUNCPath: false,
		},
		{
			goos:      "windows",
			path:      `C:\foo`,
			isUNCPath: false,
		},
		{
			goos:      "windows",
			path:      `\\server\share\foo`,
			isUNCPath: true,
		},
		{
			goos:      "windows",
			path:      `\\?\server\share`,
			isUNCPath: true,
		},
		{
			goos:      "windows",
			path:      `\\?\c:\`,
			isUNCPath: true,
		},
		{
			goos:      "windows",
			path:      `\\.\pipe\valid_pipe`,
			isUNCPath: true,
		},
	}

	for _, test := range tests {
		result := IsWindowsUNCPath(test.goos, test.path)
		if result != test.isUNCPath {
			t.Errorf("IsWindowsUNCPath(%v) returned (%v), expected (%v)", test.path, result, test.isUNCPath)
		}
	}
}

func TestIsWindowsLocalPath(t *testing.T) {
	tests := []struct {
		goos               string
		path               string
		isWindowsLocalPath bool
	}{
		{
			goos:               "linux",
			path:               `/usr/bin`,
			isWindowsLocalPath: false,
		},
		{
			goos:               "linux",
			path:               `\\.\pipe\foo`,
			isWindowsLocalPath: false,
		},
		{
			goos:               "windows",
			path:               `C:\foo`,
			isWindowsLocalPath: false,
		},
		{
			goos:               "windows",
			path:               `:\foo`,
			isWindowsLocalPath: false,
		},
		{
			goos:               "windows",
			path:               `X:\foo`,
			isWindowsLocalPath: false,
		},
		{
			goos:               "windows",
			path:               `\\server\share\foo`,
			isWindowsLocalPath: false,
		},
		{
			goos:               "windows",
			path:               `\\?\server\share`,
			isWindowsLocalPath: false,
		},
		{
			goos:               "windows",
			path:               `\\?\c:\`,
			isWindowsLocalPath: false,
		},
		{
			goos:               "windows",
			path:               `\\.\pipe\valid_pipe`,
			isWindowsLocalPath: false,
		},
		{
			goos:               "windows",
			path:               `foo`,
			isWindowsLocalPath: false,
		},
		{
			goos:               "windows",
			path:               `:foo`,
			isWindowsLocalPath: false,
		},
		{
			goos:               "windows",
			path:               `\foo`,
			isWindowsLocalPath: true,
		},
		{
			goos:               "windows",
			path:               `\foo\bar`,
			isWindowsLocalPath: true,
		},
		{
			goos:               "windows",
			path:               `/foo`,
			isWindowsLocalPath: true,
		},
		{
			goos:               "windows",
			path:               `/foo/bar`,
			isWindowsLocalPath: true,
		},
	}

	for _, test := range tests {
		result := IsWindowsLocalPath(test.goos, test.path)
		if result != test.isWindowsLocalPath {
			t.Errorf("isWindowsLocalPath(%v) returned (%v), expected (%v)", test.path, result, test.isWindowsLocalPath)
		}
	}
}

func TestMakeAbsolutePath(t *testing.T) {
	tests := []struct {
		goos         string
		path         string
		expectedPath string
		name         string
	}{
		{
			goos:         "linux",
			path:         "non-absolute/path",
			expectedPath: "/non-absolute/path",
			name:         "linux non-absolute path",
		},
		{
			goos:         "linux",
			path:         "/absolute/path",
			expectedPath: "/absolute/path",
			name:         "linux absolute path",
		},
		{
			goos:         "windows",
			path:         "some\\path",
			expectedPath: "c:\\some\\path",
			name:         "basic windows",
		},
		{
			goos:         "windows",
			path:         "/some/path",
			expectedPath: "c:/some/path",
			name:         "linux path on windows",
		},
		{
			goos:         "windows",
			path:         "\\some\\path",
			expectedPath: "c:\\some\\path",
			name:         "windows path no drive",
		},
		{
			goos:         "windows",
			path:         "\\:\\some\\path",
			expectedPath: "\\:\\some\\path",
			name:         "windows path with colon",
		},
	}
	for _, test := range tests {
		if runtime.GOOS == test.goos {
			path := MakeAbsolutePath(test.goos, test.path)
			if path != test.expectedPath {
				t.Errorf("[%s] Expected %s saw %s", test.name, test.expectedPath, path)
			}
		}
	}
}

func TestGetPodVolumeNames(t *testing.T) {
	defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.ReadWriteOncePod, true)()
	defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.SELinuxMountReadWriteOncePod, true)()
	tests := []struct {
		name                    string
		pod                     *v1.Pod
		expectedMounts          sets.String
		expectedDevices         sets.String
		expectedSELinuxContexts map[string][]*v1.SELinuxOptions
	}{
		{
			name: "empty pod",
			pod: &v1.Pod{
				Spec: v1.PodSpec{},
			},
			expectedMounts:  sets.NewString(),
			expectedDevices: sets.NewString(),
		},
		{
			name: "pod with volumes",
			pod: &v1.Pod{
				Spec: v1.PodSpec{
					Containers: []v1.Container{
						{
							Name: "container",
							VolumeMounts: []v1.VolumeMount{
								{
									Name: "vol1",
								},
								{
									Name: "vol2",
								},
							},
							VolumeDevices: []v1.VolumeDevice{
								{
									Name: "vol3",
								},
								{
									Name: "vol4",
								},
							},
						},
					},
					Volumes: []v1.Volume{
						{
							Name: "vol1",
						},
						{
							Name: "vol2",
						},
						{
							Name: "vol3",
						},
						{
							Name: "vol4",
						},
					},
				},
			},
			expectedMounts:  sets.NewString("vol1", "vol2"),
			expectedDevices: sets.NewString("vol3", "vol4"),
		},
		{
			name: "pod with init containers",
			pod: &v1.Pod{
				Spec: v1.PodSpec{
					InitContainers: []v1.Container{
						{
							Name: "initContainer",
							VolumeMounts: []v1.VolumeMount{
								{
									Name: "vol1",
								},
								{
									Name: "vol2",
								},
							},
							VolumeDevices: []v1.VolumeDevice{
								{
									Name: "vol3",
								},
								{
									Name: "vol4",
								},
							},
						},
					},
					Volumes: []v1.Volume{
						{
							Name: "vol1",
						},
						{
							Name: "vol2",
						},
						{
							Name: "vol3",
						},
						{
							Name: "vol4",
						},
					},
				},
			},
			expectedMounts:  sets.NewString("vol1", "vol2"),
			expectedDevices: sets.NewString("vol3", "vol4"),
		},
		{
			name: "pod with multiple containers",
			pod: &v1.Pod{
				Spec: v1.PodSpec{
					InitContainers: []v1.Container{
						{
							Name: "initContainer1",
							VolumeMounts: []v1.VolumeMount{
								{
									Name: "vol1",
								},
							},
						},
						{
							Name: "initContainer2",
							VolumeDevices: []v1.VolumeDevice{
								{
									Name: "vol2",
								},
							},
						},
					},
					Containers: []v1.Container{
						{
							Name: "container1",
							VolumeMounts: []v1.VolumeMount{
								{
									Name: "vol3",
								},
							},
						},
						{
							Name: "container2",
							VolumeDevices: []v1.VolumeDevice{
								{
									Name: "vol4",
								},
							},
						},
					},
					Volumes: []v1.Volume{
						{
							Name: "vol1",
						},
						{
							Name: "vol2",
						},
						{
							Name: "vol3",
						},
						{
							Name: "vol4",
						},
					},
				},
			},
			expectedMounts:  sets.NewString("vol1", "vol3"),
			expectedDevices: sets.NewString("vol2", "vol4"),
		},
		{
			name: "pod with ephemeral containers",
			pod: &v1.Pod{
				Spec: v1.PodSpec{
					Containers: []v1.Container{
						{
							Name: "container1",
							VolumeMounts: []v1.VolumeMount{
								{
									Name: "vol1",
								},
							},
						},
					},
					EphemeralContainers: []v1.EphemeralContainer{
						{
							EphemeralContainerCommon: v1.EphemeralContainerCommon{
								Name: "debugger",
								VolumeMounts: []v1.VolumeMount{
									{
										Name: "vol1",
									},
									{
										Name: "vol2",
									},
								},
							},
						},
					},
					Volumes: []v1.Volume{
						{
							Name: "vol1",
						},
						{
							Name: "vol2",
						},
					},
				},
			},
			expectedMounts:  sets.NewString("vol1", "vol2"),
			expectedDevices: sets.NewString(),
		},
		{
			name: "pod with SELinuxOptions",
			pod: &v1.Pod{
				Spec: v1.PodSpec{
					SecurityContext: &v1.PodSecurityContext{
						SELinuxOptions: &v1.SELinuxOptions{
							Type:  "global_context_t",
							Level: "s0:c1,c2",
						},
					},
					InitContainers: []v1.Container{
						{
							Name: "initContainer1",
							SecurityContext: &v1.SecurityContext{
								SELinuxOptions: &v1.SELinuxOptions{
									Type:  "initcontainer1_context_t",
									Level: "s0:c3,c4",
								},
							},
							VolumeMounts: []v1.VolumeMount{
								{
									Name: "vol1",
								},
							},
						},
					},
					Containers: []v1.Container{
						{
							Name: "container1",
							SecurityContext: &v1.SecurityContext{
								SELinuxOptions: &v1.SELinuxOptions{
									Type:  "container1_context_t",
									Level: "s0:c5,c6",
								},
							},
							VolumeMounts: []v1.VolumeMount{
								{
									Name: "vol1",
								},
								{
									Name: "vol2",
								},
							},
						},
						{
							Name: "container2",
							// No SELinux context, will be inherited from PodSecurityContext
							VolumeMounts: []v1.VolumeMount{
								{
									Name: "vol2",
								},
								{
									Name: "vol3",
								},
							},
						},
					},
					Volumes: []v1.Volume{
						{
							Name: "vol1",
						},
						{
							Name: "vol2",
						},
						{
							Name: "vol3",
						},
					},
				},
			},
			expectedMounts: sets.NewString("vol1", "vol2", "vol3"),
			expectedSELinuxContexts: map[string][]*v1.SELinuxOptions{
				"vol1": {
					{
						Type:  "initcontainer1_context_t",
						Level: "s0:c3,c4",
					},
					{
						Type:  "container1_context_t",
						Level: "s0:c5,c6",
					},
				},
				"vol2": {
					{
						Type:  "container1_context_t",
						Level: "s0:c5,c6",
					},
					{
						Type:  "global_context_t",
						Level: "s0:c1,c2",
					},
				},
				"vol3": {
					{
						Type:  "global_context_t",
						Level: "s0:c1,c2",
					},
				},
			},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			mounts, devices, contexts := GetPodVolumeNames(test.pod)
			if !mounts.Equal(test.expectedMounts) {
				t.Errorf("Expected mounts: %q, got %q", mounts.List(), test.expectedMounts.List())
			}
			if !devices.Equal(test.expectedDevices) {
				t.Errorf("Expected devices: %q, got %q", devices.List(), test.expectedDevices.List())
			}
			if len(contexts) == 0 {
				contexts = nil
			}
			if !reflect.DeepEqual(test.expectedSELinuxContexts, contexts) {
				t.Errorf("Expected SELinuxContexts: %+v\ngot: %+v", test.expectedSELinuxContexts, contexts)
			}
		})
	}
}
