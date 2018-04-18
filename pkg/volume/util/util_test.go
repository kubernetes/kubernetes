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
	"io/ioutil"
	"os"
	"testing"

	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/util/sets"
	utiltesting "k8s.io/client-go/util/testing"
	// util.go uses api.Codecs.LegacyCodec so import this package to do some
	// resource initialization.
	"hash/fnv"

	_ "k8s.io/kubernetes/pkg/apis/core/install"
	"k8s.io/kubernetes/pkg/apis/core/v1/helper"
	"k8s.io/kubernetes/pkg/util/mount"

	"reflect"
	"strings"

	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

	"k8s.io/kubernetes/pkg/util/slice"
	"k8s.io/kubernetes/pkg/volume"
)

var nodeLabels map[string]string = map[string]string{
	"test-key1": "test-value1",
	"test-key2": "test-value2",
}

func TestCheckAlphaNodeAffinity(t *testing.T) {
	type affinityTest struct {
		name          string
		expectSuccess bool
		pv            *v1.PersistentVolume
	}

	cases := []affinityTest{
		{
			name:          "valid-no-constraints",
			expectSuccess: true,
			pv:            testVolumeWithAlphaNodeAffinity(t, &v1.NodeAffinity{}),
		},
		{
			name:          "valid-constraints",
			expectSuccess: true,
			pv: testVolumeWithAlphaNodeAffinity(t, &v1.NodeAffinity{
				RequiredDuringSchedulingIgnoredDuringExecution: &v1.NodeSelector{
					NodeSelectorTerms: []v1.NodeSelectorTerm{
						{
							MatchExpressions: []v1.NodeSelectorRequirement{
								{
									Key:      "test-key1",
									Operator: v1.NodeSelectorOpIn,
									Values:   []string{"test-value1", "test-value3"},
								},
								{
									Key:      "test-key2",
									Operator: v1.NodeSelectorOpIn,
									Values:   []string{"test-value0", "test-value2"},
								},
							},
						},
					},
				},
			}),
		},
		{
			name:          "invalid-key",
			expectSuccess: false,
			pv: testVolumeWithAlphaNodeAffinity(t, &v1.NodeAffinity{
				RequiredDuringSchedulingIgnoredDuringExecution: &v1.NodeSelector{
					NodeSelectorTerms: []v1.NodeSelectorTerm{
						{
							MatchExpressions: []v1.NodeSelectorRequirement{
								{
									Key:      "test-key1",
									Operator: v1.NodeSelectorOpIn,
									Values:   []string{"test-value1", "test-value3"},
								},
								{
									Key:      "test-key3",
									Operator: v1.NodeSelectorOpIn,
									Values:   []string{"test-value0", "test-value2"},
								},
							},
						},
					},
				},
			}),
		},
		{
			name:          "invalid-values",
			expectSuccess: false,
			pv: testVolumeWithAlphaNodeAffinity(t, &v1.NodeAffinity{
				RequiredDuringSchedulingIgnoredDuringExecution: &v1.NodeSelector{
					NodeSelectorTerms: []v1.NodeSelectorTerm{
						{
							MatchExpressions: []v1.NodeSelectorRequirement{
								{
									Key:      "test-key1",
									Operator: v1.NodeSelectorOpIn,
									Values:   []string{"test-value3", "test-value4"},
								},
								{
									Key:      "test-key2",
									Operator: v1.NodeSelectorOpIn,
									Values:   []string{"test-value0", "test-value2"},
								},
							},
						},
					},
				},
			}),
		},
		{
			name:          "invalid-multiple-terms",
			expectSuccess: false,
			pv: testVolumeWithNodeAffinity(t, &v1.VolumeNodeAffinity{
				Required: &v1.NodeSelector{
					NodeSelectorTerms: []v1.NodeSelectorTerm{
						{
							MatchExpressions: []v1.NodeSelectorRequirement{
								{
									Key:      "test-key3",
									Operator: v1.NodeSelectorOpIn,
									Values:   []string{"test-value1", "test-value3"},
								},
							},
						},
						{
							MatchExpressions: []v1.NodeSelectorRequirement{
								{
									Key:      "test-key2",
									Operator: v1.NodeSelectorOpIn,
									Values:   []string{"test-value0", "test-value1"},
								},
							},
						},
					},
				},
			}),
		},
	}

	for _, c := range cases {
		err := CheckNodeAffinity(c.pv, nodeLabels)

		if err != nil && c.expectSuccess {
			t.Errorf("CheckTopology %v returned error: %v", c.name, err)
		}
		if err == nil && !c.expectSuccess {
			t.Errorf("CheckTopology %v returned success, expected error", c.name)
		}
	}
}

func TestCheckVolumeNodeAffinity(t *testing.T) {
	type affinityTest struct {
		name          string
		expectSuccess bool
		pv            *v1.PersistentVolume
	}

	cases := []affinityTest{
		{
			name:          "valid-nil",
			expectSuccess: true,
			pv:            testVolumeWithNodeAffinity(t, nil),
		},
		{
			name:          "valid-no-constraints",
			expectSuccess: true,
			pv:            testVolumeWithNodeAffinity(t, &v1.VolumeNodeAffinity{}),
		},
		{
			name:          "select-nothing",
			expectSuccess: false,
			pv:            testVolumeWithNodeAffinity(t, &v1.VolumeNodeAffinity{Required: &v1.NodeSelector{}}),
		},
		{
			name:          "select-nothing-empty-terms",
			expectSuccess: false,
			pv: testVolumeWithNodeAffinity(t, &v1.VolumeNodeAffinity{
				Required: &v1.NodeSelector{
					NodeSelectorTerms: []v1.NodeSelectorTerm{
						{
							MatchExpressions: []v1.NodeSelectorRequirement{},
						},
					},
				},
			}),
		},
		{
			name:          "valid-multiple-terms",
			expectSuccess: true,
			pv: testVolumeWithNodeAffinity(t, &v1.VolumeNodeAffinity{
				Required: &v1.NodeSelector{
					NodeSelectorTerms: []v1.NodeSelectorTerm{
						{
							MatchExpressions: []v1.NodeSelectorRequirement{
								{
									Key:      "test-key3",
									Operator: v1.NodeSelectorOpIn,
									Values:   []string{"test-value1", "test-value3"},
								},
							},
						},
						{
							MatchExpressions: []v1.NodeSelectorRequirement{
								{
									Key:      "test-key2",
									Operator: v1.NodeSelectorOpIn,
									Values:   []string{"test-value0", "test-value2"},
								},
							},
						},
					},
				},
			}),
		},
		{
			name:          "valid-multiple-match-expressions",
			expectSuccess: true,
			pv: testVolumeWithNodeAffinity(t, &v1.VolumeNodeAffinity{
				Required: &v1.NodeSelector{
					NodeSelectorTerms: []v1.NodeSelectorTerm{
						{
							MatchExpressions: []v1.NodeSelectorRequirement{
								{
									Key:      "test-key1",
									Operator: v1.NodeSelectorOpIn,
									Values:   []string{"test-value1", "test-value3"},
								},
								{
									Key:      "test-key2",
									Operator: v1.NodeSelectorOpIn,
									Values:   []string{"test-value0", "test-value2"},
								},
							},
						},
					},
				},
			}),
		},
		{
			name:          "invalid-multiple-match-expressions-key",
			expectSuccess: false,
			pv: testVolumeWithNodeAffinity(t, &v1.VolumeNodeAffinity{
				Required: &v1.NodeSelector{
					NodeSelectorTerms: []v1.NodeSelectorTerm{
						{
							MatchExpressions: []v1.NodeSelectorRequirement{
								{
									Key:      "test-key1",
									Operator: v1.NodeSelectorOpIn,
									Values:   []string{"test-value1", "test-value3"},
								},
								{
									Key:      "test-key3",
									Operator: v1.NodeSelectorOpIn,
									Values:   []string{"test-value0", "test-value2"},
								},
							},
						},
					},
				},
			}),
		},
		{
			name:          "invalid-multiple-match-expressions-values",
			expectSuccess: false,
			pv: testVolumeWithNodeAffinity(t, &v1.VolumeNodeAffinity{
				Required: &v1.NodeSelector{
					NodeSelectorTerms: []v1.NodeSelectorTerm{
						{
							MatchExpressions: []v1.NodeSelectorRequirement{
								{
									Key:      "test-key1",
									Operator: v1.NodeSelectorOpIn,
									Values:   []string{"test-value3", "test-value4"},
								},
								{
									Key:      "test-key2",
									Operator: v1.NodeSelectorOpIn,
									Values:   []string{"test-value0", "test-value2"},
								},
							},
						},
					},
				},
			}),
		},
	}

	for _, c := range cases {
		err := CheckNodeAffinity(c.pv, nodeLabels)

		if err != nil && c.expectSuccess {
			t.Errorf("CheckTopology %v returned error: %v", c.name, err)
		}
		if err == nil && !c.expectSuccess {
			t.Errorf("CheckTopology %v returned success, expected error", c.name)
		}
	}
}

func testVolumeWithAlphaNodeAffinity(t *testing.T, affinity *v1.NodeAffinity) *v1.PersistentVolume {
	objMeta := metav1.ObjectMeta{Name: "test-constraints"}
	objMeta.Annotations = map[string]string{}
	err := helper.StorageNodeAffinityToAlphaAnnotation(objMeta.Annotations, affinity)
	if err != nil {
		t.Fatalf("Failed to get node affinity annotation: %v", err)
	}

	return &v1.PersistentVolume{
		ObjectMeta: objMeta,
	}
}

func testVolumeWithNodeAffinity(t *testing.T, affinity *v1.VolumeNodeAffinity) *v1.PersistentVolume {
	objMeta := metav1.ObjectMeta{Name: "test-constraints"}
	return &v1.PersistentVolume{
		ObjectMeta: objMeta,
		Spec: v1.PersistentVolumeSpec{
			NodeAffinity: affinity,
		},
	}
}

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
    - image: k8s.gcr.io/busybox
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
        "image": "k8s.gcr.io/busybox"
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
  - image: k8s.gcr.io/busybox
`,
			true,
		},
	}

	for _, test := range tests {
		tempFile, err := ioutil.TempFile("", "podfile")
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
func TestZonesToSet(t *testing.T) {
	functionUnderTest := "ZonesToSet"
	// First part: want an error
	sliceOfZones := []string{"", ",", "us-east-1a, , us-east-1d", ", us-west-1b", "us-west-2b,"}
	for _, zones := range sliceOfZones {
		if got, err := ZonesToSet(zones); err == nil {
			t.Errorf("%v(%v) returned (%v), want (%v)", functionUnderTest, zones, got, "an error")
		}
	}

	// Second part: want no error
	tests := []struct {
		zones string
		want  sets.String
	}{
		{
			zones: "us-east-1a",
			want:  sets.String{"us-east-1a": sets.Empty{}},
		},
		{
			zones: "us-east-1a, us-west-2a",
			want: sets.String{
				"us-east-1a": sets.Empty{},
				"us-west-2a": sets.Empty{},
			},
		},
	}
	for _, tt := range tests {
		if got, err := ZonesToSet(tt.zones); err != nil || !got.Equal(tt.want) {
			t.Errorf("%v(%v) returned (%v), want (%v)", functionUnderTest, tt.zones, got, tt.want)
		}
	}
}

func TestDoUnmountMountPoint(t *testing.T) {

	tmpDir1, err1 := utiltesting.MkTmpdir("umount_test1")
	if err1 != nil {
		t.Fatalf("error creating temp dir: %v", err1)
	}
	defer os.RemoveAll(tmpDir1)

	tmpDir2, err2 := utiltesting.MkTmpdir("umount_test2")
	if err2 != nil {
		t.Fatalf("error creating temp dir: %v", err2)
	}
	defer os.RemoveAll(tmpDir2)

	// Second part: want no error
	tests := []struct {
		mountPath    string
		corruptedMnt bool
	}{
		{
			mountPath:    tmpDir1,
			corruptedMnt: true,
		},
		{
			mountPath:    tmpDir2,
			corruptedMnt: false,
		},
	}

	fake := &mount.FakeMounter{}

	for _, tt := range tests {
		err := doUnmountMountPoint(tt.mountPath, fake, false, tt.corruptedMnt)
		if err != nil {
			t.Errorf("err Expected nil, but got: %v", err)
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

func checkFnv32(t *testing.T, s string, expected uint32) {
	h := fnv.New32()
	h.Write([]byte(s))
	h.Sum32()

	if h.Sum32() != expected {
		t.Fatalf("hash of %q was %v, expected %v", s, h.Sum32(), expected)
	}
}

func TestChooseZoneForVolume(t *testing.T) {
	checkFnv32(t, "henley", 1180403676)
	// 1180403676 mod 3 == 0, so the offset from "henley" is 0, which makes it easier to verify this by inspection

	// A few others
	checkFnv32(t, "henley-", 2652299129)
	checkFnv32(t, "henley-a", 1459735322)
	checkFnv32(t, "", 2166136261)

	tests := []struct {
		Zones      sets.String
		VolumeName string
		Expected   string
	}{
		// Test for PVC names that don't have a dash
		{
			Zones:      sets.NewString("a", "b", "c"),
			VolumeName: "henley",
			Expected:   "a", // hash("henley") == 0
		},
		// Tests for PVC names that end in - number, but don't look like statefulset PVCs
		{
			Zones:      sets.NewString("a", "b", "c"),
			VolumeName: "henley-0",
			Expected:   "a", // hash("henley") == 0
		},
		{
			Zones:      sets.NewString("a", "b", "c"),
			VolumeName: "henley-1",
			Expected:   "b", // hash("henley") + 1 == 1
		},
		{
			Zones:      sets.NewString("a", "b", "c"),
			VolumeName: "henley-2",
			Expected:   "c", // hash("henley") + 2 == 2
		},
		{
			Zones:      sets.NewString("a", "b", "c"),
			VolumeName: "henley-3",
			Expected:   "a", // hash("henley") + 3 == 3 === 0 mod 3
		},
		{
			Zones:      sets.NewString("a", "b", "c"),
			VolumeName: "henley-4",
			Expected:   "b", // hash("henley") + 4 == 4 === 1 mod 3
		},
		// Tests for PVC names that are edge cases
		{
			Zones:      sets.NewString("a", "b", "c"),
			VolumeName: "henley-",
			Expected:   "c", // hash("henley-") = 2652299129 === 2 mod 3
		},
		{
			Zones:      sets.NewString("a", "b", "c"),
			VolumeName: "henley-a",
			Expected:   "c", // hash("henley-a") = 1459735322 === 2 mod 3
		},
		{
			Zones:      sets.NewString("a", "b", "c"),
			VolumeName: "medium--1",
			Expected:   "c", // hash("") + 1 == 2166136261 + 1 === 2 mod 3
		},
		// Tests for PVC names for simple StatefulSet cases
		{
			Zones:      sets.NewString("a", "b", "c"),
			VolumeName: "medium-henley-1",
			Expected:   "b", // hash("henley") + 1 == 1
		},
		{
			Zones:      sets.NewString("a", "b", "c"),
			VolumeName: "loud-henley-1",
			Expected:   "b", // hash("henley") + 1 == 1
		},
		{
			Zones:      sets.NewString("a", "b", "c"),
			VolumeName: "quiet-henley-2",
			Expected:   "c", // hash("henley") + 2 == 2
		},
		{
			Zones:      sets.NewString("a", "b", "c"),
			VolumeName: "medium-henley-2",
			Expected:   "c", // hash("henley") + 2 == 2
		},
		{
			Zones:      sets.NewString("a", "b", "c"),
			VolumeName: "medium-henley-3",
			Expected:   "a", // hash("henley") + 3 == 3 === 0 mod 3
		},
		{
			Zones:      sets.NewString("a", "b", "c"),
			VolumeName: "medium-henley-4",
			Expected:   "b", // hash("henley") + 4 == 4 === 1 mod 3
		},
		// Tests for statefulsets (or claims) with dashes in the names
		{
			Zones:      sets.NewString("a", "b", "c"),
			VolumeName: "medium-alpha-henley-2",
			Expected:   "c", // hash("henley") + 2 == 2
		},
		{
			Zones:      sets.NewString("a", "b", "c"),
			VolumeName: "medium-beta-henley-3",
			Expected:   "a", // hash("henley") + 3 == 3 === 0 mod 3
		},
		{
			Zones:      sets.NewString("a", "b", "c"),
			VolumeName: "medium-gamma-henley-4",
			Expected:   "b", // hash("henley") + 4 == 4 === 1 mod 3
		},
		// Tests for statefulsets name ending in -
		{
			Zones:      sets.NewString("a", "b", "c"),
			VolumeName: "medium-henley--2",
			Expected:   "a", // hash("") + 2 == 0 mod 3
		},
		{
			Zones:      sets.NewString("a", "b", "c"),
			VolumeName: "medium-henley--3",
			Expected:   "b", // hash("") + 3 == 1 mod 3
		},
		{
			Zones:      sets.NewString("a", "b", "c"),
			VolumeName: "medium-henley--4",
			Expected:   "c", // hash("") + 4 == 2 mod 3
		},
	}

	for _, test := range tests {
		actual := ChooseZoneForVolume(test.Zones, test.VolumeName)

		if actual != test.Expected {
			t.Errorf("Test %v failed, expected zone %q, actual %q", test, test.Expected, actual)
		}
	}
}

func TestChooseZonesForVolume(t *testing.T) {
	checkFnv32(t, "henley", 1180403676)
	// 1180403676 mod 3 == 0, so the offset from "henley" is 0, which makes it easier to verify this by inspection

	// A few others
	checkFnv32(t, "henley-", 2652299129)
	checkFnv32(t, "henley-a", 1459735322)
	checkFnv32(t, "", 2166136261)

	tests := []struct {
		Zones      sets.String
		VolumeName string
		NumZones   uint32
		Expected   sets.String
	}{
		// Test for PVC names that don't have a dash
		{
			Zones:      sets.NewString("a", "b", "c"),
			VolumeName: "henley",
			NumZones:   1,
			Expected:   sets.NewString("a" /* hash("henley") == 0 */),
		},
		{
			Zones:      sets.NewString("a", "b", "c"),
			VolumeName: "henley",
			NumZones:   2,
			Expected:   sets.NewString("a" /* hash("henley") == 0 */, "b"),
		},
		// Tests for PVC names that end in - number, but don't look like statefulset PVCs
		{
			Zones:      sets.NewString("a", "b", "c"),
			VolumeName: "henley-0",
			NumZones:   1,
			Expected:   sets.NewString("a" /* hash("henley") == 0 */),
		},
		{
			Zones:      sets.NewString("a", "b", "c"),
			VolumeName: "henley-0",
			NumZones:   2,
			Expected:   sets.NewString("a" /* hash("henley") == 0 */, "b"),
		},
		{
			Zones:      sets.NewString("a", "b", "c"),
			VolumeName: "henley-1",
			NumZones:   1,
			Expected:   sets.NewString("b" /* hash("henley") + 1 == 1 */),
		},
		{
			Zones:      sets.NewString("a", "b", "c"),
			VolumeName: "henley-1",
			NumZones:   2,
			Expected:   sets.NewString("c" /* hash("henley") + 1 + 1(startingIndex) == 2 */, "a"),
		},
		{
			Zones:      sets.NewString("a", "b", "c"),
			VolumeName: "henley-2",
			NumZones:   1,
			Expected:   sets.NewString("c" /* hash("henley") + 2 == 2 */),
		},
		{
			Zones:      sets.NewString("a", "b", "c"),
			VolumeName: "henley-2",
			NumZones:   2,
			Expected:   sets.NewString("b" /* hash("henley") + 2 + 2(startingIndex) == 4 */, "c"),
		},
		{
			Zones:      sets.NewString("a", "b", "c"),
			VolumeName: "henley-3",
			NumZones:   1,
			Expected:   sets.NewString("a" /* hash("henley") + 3 == 3 === 0 mod 3 */),
		},
		{
			Zones:      sets.NewString("a", "b", "c"),
			VolumeName: "henley-3",
			NumZones:   2,
			Expected:   sets.NewString("a" /* hash("henley") + 3 + 3(startingIndex) == 6 */, "b"),
		},
		{
			Zones:      sets.NewString("a", "b", "c"),
			VolumeName: "henley-4",
			NumZones:   1,
			Expected:   sets.NewString("b" /* hash("henley") + 4 == 4 === 1 mod 3 */),
		},
		{
			Zones:      sets.NewString("a", "b", "c"),
			VolumeName: "henley-4",
			NumZones:   2,
			Expected:   sets.NewString("c" /* hash("henley") + 4 + 4(startingIndex) == 8 */, "a"),
		},
		// Tests for PVC names that are edge cases
		{
			Zones:      sets.NewString("a", "b", "c"),
			VolumeName: "henley-",
			NumZones:   1,
			Expected:   sets.NewString("c" /* hash("henley-") = 2652299129 === 2 mod 3 */),
		},
		{
			Zones:      sets.NewString("a", "b", "c"),
			VolumeName: "henley-",
			NumZones:   2,
			Expected:   sets.NewString("c" /* hash("henley-") = 2652299129 === 2 mod 3 = 2 */, "a"),
		},
		{
			Zones:      sets.NewString("a", "b", "c"),
			VolumeName: "henley-a",
			NumZones:   1,
			Expected:   sets.NewString("c" /* hash("henley-a") = 1459735322 === 2 mod 3 */),
		},
		{
			Zones:      sets.NewString("a", "b", "c"),
			VolumeName: "henley-a",
			NumZones:   2,
			Expected:   sets.NewString("c" /* hash("henley-a") = 1459735322 === 2 mod 3 = 2 */, "a"),
		},
		{
			Zones:      sets.NewString("a", "b", "c"),
			VolumeName: "medium--1",
			NumZones:   1,
			Expected:   sets.NewString("c" /* hash("") + 1 == 2166136261 + 1 === 2 mod 3 */),
		},
		{
			Zones:      sets.NewString("a", "b", "c"),
			VolumeName: "medium--1",
			NumZones:   2,
			Expected:   sets.NewString("a" /* hash("") + 1 + 1(startingIndex) == 2166136261 + 1 + 1 === 3 mod 3 = 0 */, "b"),
		},
		// Tests for PVC names for simple StatefulSet cases
		{
			Zones:      sets.NewString("a", "b", "c"),
			VolumeName: "medium-henley-1",
			NumZones:   1,
			Expected:   sets.NewString("b" /* hash("henley") + 1 == 1 */),
		},
		// Tests for PVC names for simple StatefulSet cases
		{
			Zones:      sets.NewString("a", "b", "c"),
			VolumeName: "medium-henley-1",
			NumZones:   2,
			Expected:   sets.NewString("c" /* hash("henley") + 1 + 1(startingIndex) == 2 */, "a"),
		},
		{
			Zones:      sets.NewString("a", "b", "c"),
			VolumeName: "loud-henley-1",
			NumZones:   1,
			Expected:   sets.NewString("b" /* hash("henley") + 1 == 1 */),
		},
		{
			Zones:      sets.NewString("a", "b", "c"),
			VolumeName: "loud-henley-1",
			NumZones:   2,
			Expected:   sets.NewString("c" /* hash("henley") + 1 + 1(startingIndex) == 2 */, "a"),
		},
		{
			Zones:      sets.NewString("a", "b", "c"),
			VolumeName: "quiet-henley-2",
			NumZones:   1,
			Expected:   sets.NewString("c" /* hash("henley") + 2 == 2 */),
		},
		{
			Zones:      sets.NewString("a", "b", "c"),
			VolumeName: "quiet-henley-2",
			NumZones:   2,
			Expected:   sets.NewString("b" /* hash("henley") + 2 + 2(startingIndex) == 4 */, "c"),
		},
		{
			Zones:      sets.NewString("a", "b", "c"),
			VolumeName: "medium-henley-2",
			NumZones:   1,
			Expected:   sets.NewString("c" /* hash("henley") + 2 == 2 */),
		},
		{
			Zones:      sets.NewString("a", "b", "c"),
			VolumeName: "medium-henley-2",
			NumZones:   2,
			Expected:   sets.NewString("b" /* hash("henley") + 2 + 2(startingIndex) == 4 */, "c"),
		},
		{
			Zones:      sets.NewString("a", "b", "c"),
			VolumeName: "medium-henley-3",
			NumZones:   1,
			Expected:   sets.NewString("a" /* hash("henley") + 3 == 3 === 0 mod 3 */),
		},
		{
			Zones:      sets.NewString("a", "b", "c"),
			VolumeName: "medium-henley-3",
			NumZones:   2,
			Expected:   sets.NewString("a" /* hash("henley") + 3 + 3(startingIndex) == 6 === 6 mod 3 = 0 */, "b"),
		},
		{
			Zones:      sets.NewString("a", "b", "c"),
			VolumeName: "medium-henley-4",
			NumZones:   1,
			Expected:   sets.NewString("b" /* hash("henley") + 4 == 4 === 1 mod 3 */),
		},
		{
			Zones:      sets.NewString("a", "b", "c"),
			VolumeName: "medium-henley-4",
			NumZones:   2,
			Expected:   sets.NewString("c" /* hash("henley") + 4 + 4(startingIndex) == 8 === 2 mod 3 */, "a"),
		},
		// Tests for statefulsets (or claims) with dashes in the names
		{
			Zones:      sets.NewString("a", "b", "c"),
			VolumeName: "medium-alpha-henley-2",
			NumZones:   1,
			Expected:   sets.NewString("c" /* hash("henley") + 2 == 2 */),
		},
		{
			Zones:      sets.NewString("a", "b", "c"),
			VolumeName: "medium-alpha-henley-2",
			NumZones:   2,
			Expected:   sets.NewString("b" /* hash("henley") + 2 + 2(startingIndex) == 4 */, "c"),
		},
		{
			Zones:      sets.NewString("a", "b", "c"),
			VolumeName: "medium-beta-henley-3",
			NumZones:   1,
			Expected:   sets.NewString("a" /* hash("henley") + 3 == 3 === 0 mod 3 */),
		},
		{
			Zones:      sets.NewString("a", "b", "c"),
			VolumeName: "medium-beta-henley-3",
			NumZones:   2,
			Expected:   sets.NewString("a" /* hash("henley") + 3 + 3(startingIndex) == 6 === 0 mod 3 */, "b"),
		},
		{
			Zones:      sets.NewString("a", "b", "c"),
			VolumeName: "medium-gamma-henley-4",
			NumZones:   1,
			Expected:   sets.NewString("b" /* hash("henley") + 4 == 4 === 1 mod 3 */),
		},
		{
			Zones:      sets.NewString("a", "b", "c"),
			VolumeName: "medium-gamma-henley-4",
			NumZones:   2,
			Expected:   sets.NewString("c" /* hash("henley") + 4 + 4(startingIndex) == 8 === 2 mod 3 */, "a"),
		},
		// Tests for statefulsets name ending in -
		{
			Zones:      sets.NewString("a", "b", "c"),
			VolumeName: "medium-henley--2",
			NumZones:   1,
			Expected:   sets.NewString("a" /* hash("") + 2 == 0 mod 3 */),
		},
		{
			Zones:      sets.NewString("a", "b", "c"),
			VolumeName: "medium-henley--2",
			NumZones:   2,
			Expected:   sets.NewString("c" /* hash("") + 2 + 2(startingIndex) == 2 mod 3 */, "a"),
		},
		{
			Zones:      sets.NewString("a", "b", "c"),
			VolumeName: "medium-henley--3",
			NumZones:   1,
			Expected:   sets.NewString("b" /* hash("") + 3 == 1 mod 3 */),
		},
		{
			Zones:      sets.NewString("a", "b", "c"),
			VolumeName: "medium-henley--3",
			NumZones:   2,
			Expected:   sets.NewString("b" /* hash("") + 3 + 3(startingIndex) == 1 mod 3 */, "c"),
		},
		{
			Zones:      sets.NewString("a", "b", "c"),
			VolumeName: "medium-henley--4",
			NumZones:   1,
			Expected:   sets.NewString("c" /* hash("") + 4 == 2 mod 3 */),
		},
		{
			Zones:      sets.NewString("a", "b", "c"),
			VolumeName: "medium-henley--4",
			NumZones:   2,
			Expected:   sets.NewString("a" /* hash("") + 4 + 4(startingIndex) == 0 mod 3 */, "b"),
		},
		{
			Zones:      sets.NewString("a", "b", "c"),
			VolumeName: "medium-henley--4",
			NumZones:   3,
			Expected:   sets.NewString("c" /* hash("") + 4 == 2 mod 3 */, "a", "b"),
		},
		{
			Zones:      sets.NewString("a", "b", "c"),
			VolumeName: "medium-henley--4",
			NumZones:   4,
			Expected:   sets.NewString("c" /* hash("") + 4 + 9(startingIndex) == 2 mod 3 */, "a", "b", "c"),
		},
		{
			Zones:      sets.NewString("a", "b", "c", "d", "e", "f", "g", "h", "i"),
			VolumeName: "henley-0",
			NumZones:   2,
			Expected:   sets.NewString("a" /* hash("henley") == 0 */, "b"),
		},
		{
			Zones:      sets.NewString("a", "b", "c", "d", "e", "f", "g", "h", "i"),
			VolumeName: "henley-1",
			NumZones:   2,
			Expected:   sets.NewString("c" /* hash("henley") == 0 + 2 */, "d"),
		},
		{
			Zones:      sets.NewString("a", "b", "c", "d", "e", "f", "g", "h", "i"),
			VolumeName: "henley-2",
			NumZones:   2,
			Expected:   sets.NewString("e" /* hash("henley") == 0 + 2 + 2(startingIndex) */, "f"),
		},
		{
			Zones:      sets.NewString("a", "b", "c", "d", "e", "f", "g", "h", "i"),
			VolumeName: "henley-3",
			NumZones:   2,
			Expected:   sets.NewString("g" /* hash("henley") == 0 + 2 + 4(startingIndex) */, "h"),
		},
		{
			Zones:      sets.NewString("a", "b", "c", "d", "e", "f", "g", "h", "i"),
			VolumeName: "henley-0",
			NumZones:   3,
			Expected:   sets.NewString("a" /* hash("henley") == 0 */, "b", "c"),
		},
		{
			Zones:      sets.NewString("a", "b", "c", "d", "e", "f", "g", "h", "i"),
			VolumeName: "henley-1",
			NumZones:   3,
			Expected:   sets.NewString("d" /* hash("henley") == 0 + 1 + 2(startingIndex) */, "e", "f"),
		},
		{
			Zones:      sets.NewString("a", "b", "c", "d", "e", "f", "g", "h", "i"),
			VolumeName: "henley-2",
			NumZones:   3,
			Expected:   sets.NewString("g" /* hash("henley") == 0 + 2 + 4(startingIndex) */, "h", "i"),
		},
		{
			Zones:      sets.NewString("a", "b", "c", "d", "e", "f", "g", "h", "i"),
			VolumeName: "henley-3",
			NumZones:   3,
			Expected:   sets.NewString("a" /* hash("henley") == 0 + 3 + 6(startingIndex) */, "b", "c"),
		},
	}

	for _, test := range tests {
		actual := ChooseZonesForVolume(test.Zones, test.VolumeName, test.NumZones)

		if !actual.Equal(test.Expected) {
			t.Errorf("Test %v failed, expected zone %#v, actual %#v", test, test.Expected, actual)
		}
	}
}

func TestValidateZone(t *testing.T) {
	functionUnderTest := "ValidateZone"

	// First part: want an error
	errCases := []string{"", " 	 	 "}
	for _, errCase := range errCases {
		if got := ValidateZone(errCase); got == nil {
			t.Errorf("%v(%v) returned (%v), want (%v)", functionUnderTest, errCase, got, "an error")
		}
	}

	// Second part: want no error
	succCases := []string{" us-east-1a	"}
	for _, succCase := range succCases {
		if got := ValidateZone(succCase); got != nil {
			t.Errorf("%v(%v) returned (%v), want (%v)", functionUnderTest, succCase, got, nil)
		}
	}
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
