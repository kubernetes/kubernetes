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
	"runtime"
	"testing"

	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/util/sets"
	utiltesting "k8s.io/client-go/util/testing"
	// util.go uses api.Codecs.LegacyCodec so import this package to do some
	// resource initialization.
	"hash/fnv"

	_ "k8s.io/kubernetes/pkg/apis/core/install"
	"k8s.io/kubernetes/pkg/util/mount"

	"reflect"
	"strings"

	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	kubeletapis "k8s.io/kubernetes/pkg/kubelet/apis"

	"k8s.io/kubernetes/pkg/util/slice"
	"k8s.io/kubernetes/pkg/volume"
)

var nodeLabels map[string]string = map[string]string{
	"test-key1": "test-value1",
	"test-key2": "test-value2",
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

func TestSelectZoneForVolume(t *testing.T) {

	nodeWithZoneLabels := &v1.Node{}
	nodeWithZoneLabels.Labels = map[string]string{kubeletapis.LabelZoneFailureDomain: "zoneX"}

	nodeWithNoLabels := &v1.Node{}

	tests := []struct {
		// Parameters passed by test to SelectZoneForVolume
		Name                          string
		ZonePresent                   bool
		Zone                          string
		ZonesPresent                  bool
		Zones                         string
		ZonesWithNodes                string
		Node                          *v1.Node
		AllowedTopologies             []v1.TopologySelectorTerm
		DynamicProvisioningScheduling bool
		// Expectations around returned zone from SelectZoneForVolume
		Reject             bool   // expect error due to validation failing
		ExpectSpecificZone bool   // expect returned zone to specifically match a single zone (rather than one from a set)
		ExpectedZone       string // single zone that should perfectly match returned zone (requires ExpectSpecificZone to be true)
		ExpectedZones      string // set of zones one of whose members should match returned zone (requires ExpectSpecificZone to be false)
	}{
		// NEGATIVE TESTS

		// Zone and Zones are both specified [Fail]
		// [1] Node irrelevant
		// [2] Zone and Zones parameters presents
		// [3] AllowedTopologies irrelevant
		// [4] DynamicProvisioningScheduling  irrelevant
		{
			Name:         "Nil_Node_with_Zone_Zones_parameters_present",
			ZonePresent:  true,
			Zone:         "zoneX",
			ZonesPresent: true,
			Zones:        "zoneX,zoneY",
			Reject:       true,
		},

		// Node has no zone labels [Fail]
		// [1] Node with no zone labels
		// [2] Zone/Zones parameter irrelevant
		// [3] AllowedTopologies irrelevant
		// [4] DynamicProvisioningScheduling enabled
		{
			Name: "Node_with_no_Zone_labels",
			Node: nodeWithNoLabels,
			DynamicProvisioningScheduling: true,
			Reject: true,
		},

		// Node with Zone labels as well as Zone parameter specified [Fail]
		// [1] Node with zone labels
		// [2] Zone parameter specified
		// [3] AllowedTopologies irrelevant
		// [4] DynamicProvisioningScheduling enabled
		{
			Name:        "Node_with_Zone_labels_and_Zone_parameter_present",
			Node:        nodeWithZoneLabels,
			ZonePresent: true,
			Zone:        "zoneX",
			DynamicProvisioningScheduling: true,
			Reject: true,
		},

		// Node with Zone labels as well as Zones parameter specified [Fail]
		// [1] Node with zone labels
		// [2] Zones parameter specified
		// [3] AllowedTopologies irrelevant
		// [4] DynamicProvisioningScheduling enabled
		{
			Name:         "Node_with_Zone_labels_and_Zones_parameter_present",
			Node:         nodeWithZoneLabels,
			ZonesPresent: true,
			Zones:        "zoneX,zoneY",
			DynamicProvisioningScheduling: true,
			Reject: true,
		},

		// Zone parameter as well as AllowedTopologies specified [Fail]
		// [1] nil Node
		// [2] Zone parameter specified
		// [3] AllowedTopologies specified
		// [4] DynamicProvisioningScheduling enabled
		{
			Name:        "Nil_Node_and_Zone_parameter_and_Allowed_Topology_term",
			Node:        nil,
			ZonePresent: true,
			Zone:        "zoneX",
			DynamicProvisioningScheduling: true,
			AllowedTopologies: []v1.TopologySelectorTerm{
				{
					MatchLabelExpressions: []v1.TopologySelectorLabelRequirement{
						{
							Key:    kubeletapis.LabelZoneFailureDomain,
							Values: []string{"zoneX"},
						},
					},
				},
			},
			Reject: true,
		},

		// Zones parameter as well as AllowedTopologies specified [Fail]
		// [1] nil Node
		// [2] Zones parameter specified
		// [3] AllowedTopologies specified
		// [4] DynamicProvisioningScheduling enabled
		{
			Name:         "Nil_Node_and_Zones_parameter_and_Allowed_Topology_term",
			Node:         nil,
			ZonesPresent: true,
			Zones:        "zoneX,zoneY",
			DynamicProvisioningScheduling: true,
			AllowedTopologies: []v1.TopologySelectorTerm{
				{
					MatchLabelExpressions: []v1.TopologySelectorLabelRequirement{
						{
							Key:    kubeletapis.LabelZoneFailureDomain,
							Values: []string{"zoneX"},
						},
					},
				},
			},
			Reject: true,
		},

		// Key specified in AllowedTopologies is not LabelZoneFailureDomain [Fail]
		// [1] nil Node
		// [2] no Zone/Zones parameter
		// [3] AllowedTopologies with invalid key specified
		// [4] DynamicProvisioningScheduling enabled
		{
			Name: "Nil_Node_and_Invalid_Allowed_Topology_Key",
			Node: nil,
			DynamicProvisioningScheduling: true,
			AllowedTopologies: []v1.TopologySelectorTerm{
				{
					MatchLabelExpressions: []v1.TopologySelectorLabelRequirement{
						{
							Key:    "invalid_key",
							Values: []string{"zoneX"},
						},
						{
							Key:    kubeletapis.LabelZoneFailureDomain,
							Values: []string{"zoneY"},
						},
					},
				},
			},
			Reject: true,
		},

		// AllowedTopologies without keys specifying LabelZoneFailureDomain [Fail]
		// [1] nil Node
		// [2] no Zone/Zones parameter
		// [3] Invalid AllowedTopologies
		// [4] DynamicProvisioningScheduling enabled
		{
			Name: "Nil_Node_and_Invalid_AllowedTopologies",
			Node: nil,
			DynamicProvisioningScheduling: true,
			AllowedTopologies: []v1.TopologySelectorTerm{
				{
					MatchLabelExpressions: []v1.TopologySelectorLabelRequirement{},
				},
			},
			Reject: true,
		},

		// POSITIVE TESTS WITH DynamicProvisioningScheduling DISABLED

		// Select zone from active zones [Pass]
		// [1] nil Node (Node irrelevant)
		// [2] no Zone parameter
		// [3] no AllowedTopologies
		// [4] DynamicProvisioningScheduling disabled
		{
			Name:                          "No_Zone_Zones_parameter_and_DynamicProvisioningScheduling_disabled",
			ZonesWithNodes:                "zoneX,zoneY",
			DynamicProvisioningScheduling: false,
			Reject:        false,
			ExpectedZones: "zoneX,zoneY",
		},

		// Select zone from single zone parameter [Pass]
		// [1] nil Node (Node irrelevant)
		// [2] Zone parameter specified
		// [3] no AllowedTopologies
		// [4] DynamicProvisioningScheduling disabled
		{
			Name:        "Zone_parameter_present_and_DynamicProvisioningScheduling_disabled",
			ZonePresent: true,
			Zone:        "zoneX",
			DynamicProvisioningScheduling: false,
			Reject:             false,
			ExpectSpecificZone: true,
			ExpectedZone:       "zoneX",
		},

		// Select zone from zones parameter [Pass]
		// [1] nil Node (Node irrelevant)
		// [2] Zones parameter specified
		// [3] no AllowedTopologies
		// [4] DynamicProvisioningScheduling disabled
		{
			Name:         "Zones_parameter_present_and_DynamicProvisioningScheduling_disabled",
			ZonesPresent: true,
			Zones:        "zoneX,zoneY",
			DynamicProvisioningScheduling: false,
			Reject:        false,
			ExpectedZones: "zoneX,zoneY",
		},

		// POSITIVE TESTS WITH DynamicProvisioningScheduling ENABLED

		// Select zone from active zones [Pass]
		// [1] nil Node
		// [2] no Zone parameter specified
		// [3] no AllowedTopologies
		// [4] DynamicProvisioningScheduling enabled
		{
			Name:                          "Nil_Node_and_No_Zone_Zones_parameter_and_no_Allowed_topologies_and_DynamicProvisioningScheduling_enabled",
			Node:                          nil,
			ZonesWithNodes:                "zoneX,zoneY",
			DynamicProvisioningScheduling: true,
			Reject:        false,
			ExpectedZones: "zoneX,zoneY",
		},

		// Select zone from single zone parameter [Pass]
		// [1] nil Node
		// [2] Zone parameter specified
		// [3] no AllowedTopology specified
		// [4] DynamicSchedulingEnabled enabled
		{
			Name:        "Nil_Node_and_Zone_parameter_present_and_DynamicProvisioningScheduling_enabled",
			ZonePresent: true,
			Zone:        "zoneX",
			Node:        nil,
			DynamicProvisioningScheduling: true,
			Reject:             false,
			ExpectSpecificZone: true,
			ExpectedZone:       "zoneX",
		},

		// Select zone from zones parameter [Pass]
		// [1] nil Node
		// [2] Zones parameter specified
		// [3] no AllowedTopology
		// [4] DynamicSchedulingEnabled enabled
		{
			Name:         "Nil_Node_and_Zones_parameter_present_and_DynamicProvisioningScheduling_enabled",
			ZonesPresent: true,
			Zones:        "zoneX,zoneY",
			Node:         nil,
			DynamicProvisioningScheduling: true,
			Reject:        false,
			ExpectedZones: "zoneX,zoneY",
		},

		// Select zone from node label [Pass]
		// [1] Node with zone labels
		// [2] no zone/zones parameters
		// [3] no AllowedTopology
		// [4] DynamicProvisioningScheduling enabled
		{
			Name: "Node_with_Zone_labels_and_DynamicProvisioningScheduling_enabled",
			Node: nodeWithZoneLabels,
			DynamicProvisioningScheduling: true,
			Reject:             false,
			ExpectSpecificZone: true,
			ExpectedZone:       "zoneX",
		},

		// Select zone from node label [Pass]
		// [1] Node with zone labels
		// [2] no Zone/Zones parameters
		// [3] AllowedTopology with single term with multiple values specified (ignored)
		// [4] DynamicProvisioningScheduling enabled
		{
			Name: "Node_with_Zone_labels_and_Multiple_Allowed_Topology_values_and_DynamicProvisioningScheduling_enabled",
			Node: nodeWithZoneLabels,
			DynamicProvisioningScheduling: true,
			AllowedTopologies: []v1.TopologySelectorTerm{
				{
					MatchLabelExpressions: []v1.TopologySelectorLabelRequirement{
						{
							Key:    kubeletapis.LabelZoneFailureDomain,
							Values: []string{"zoneZ", "zoneY"},
						},
					},
				},
			},
			Reject:             false,
			ExpectSpecificZone: true,
			ExpectedZone:       "zoneX",
		},

		// Select Zone from AllowedTopologies [Pass]
		// [1] nil Node
		// [2] no Zone/Zones parametes specified
		// [3] AllowedTopologies with single term with multiple values specified
		// [4] DynamicProvisioningScheduling enabled
		{
			Name: "Nil_Node_with_Multiple_Allowed_Topology_values_and_DynamicProvisioningScheduling_enabled",
			Node: nil,
			DynamicProvisioningScheduling: true,
			AllowedTopologies: []v1.TopologySelectorTerm{
				{
					MatchLabelExpressions: []v1.TopologySelectorLabelRequirement{
						{
							Key:    kubeletapis.LabelZoneFailureDomain,
							Values: []string{"zoneX", "zoneY"},
						},
					},
				},
			},
			Reject:        false,
			ExpectedZones: "zoneX,zoneY",
		},

		// Select zone from AllowedTopologies [Pass]
		// [1] nil Node
		// [2] no Zone/Zones parametes specified
		// [3] AllowedTopologies with multiple terms specified
		// [4] DynamicProvisioningScheduling enabled
		{
			Name: "Nil_Node_and_Multiple_Allowed_Topology_terms_and_DynamicProvisioningScheduling_enabled",
			Node: nil,
			DynamicProvisioningScheduling: true,
			AllowedTopologies: []v1.TopologySelectorTerm{
				{
					MatchLabelExpressions: []v1.TopologySelectorLabelRequirement{
						{
							Key:    kubeletapis.LabelZoneFailureDomain,
							Values: []string{"zoneX"},
						},
					},
				},
				{
					MatchLabelExpressions: []v1.TopologySelectorLabelRequirement{
						{
							Key:    kubeletapis.LabelZoneFailureDomain,
							Values: []string{"zoneY"},
						},
					},
				},
			},
			Reject:        false,
			ExpectedZones: "zoneX,zoneY",
		},

		// Select Zone from AllowedTopologies [Pass]
		// Note: Dual replica with same AllowedTopologies will fail: Nil_Node_and_Single_Allowed_Topology_term_value_and_Dual_replicas
		// [1] nil Node
		// [2] no Zone/Zones parametes specified
		// [3] AllowedTopologies with single term and value specified
		// [4] DynamicProvisioningScheduling enabled
		{
			Name: "Nil_Node_and_Single_Allowed_Topology_term_value_and_DynamicProvisioningScheduling_enabled",
			Node: nil,
			DynamicProvisioningScheduling: true,
			AllowedTopologies: []v1.TopologySelectorTerm{
				{
					MatchLabelExpressions: []v1.TopologySelectorLabelRequirement{
						{
							Key:    kubeletapis.LabelZoneFailureDomain,
							Values: []string{"zoneX"},
						},
					},
				},
			},
			Reject:             false,
			ExpectSpecificZone: true,
			ExpectedZone:       "zoneX",
		},
	}

	for _, test := range tests {
		utilfeature.DefaultFeatureGate.Set("DynamicProvisioningScheduling=false")
		if test.DynamicProvisioningScheduling {
			utilfeature.DefaultFeatureGate.Set("DynamicProvisioningScheduling=true")
		}

		var zonesParameter, zonesWithNodes sets.String
		var err error

		if test.Zones != "" {
			zonesParameter, err = ZonesToSet(test.Zones)
			if err != nil {
				t.Errorf("Could not convert Zones to a set: %s. This is a test error %s", test.Zones, test.Name)
				continue
			}
		}

		if test.ZonesWithNodes != "" {
			zonesWithNodes, err = ZonesToSet(test.ZonesWithNodes)
			if err != nil {
				t.Errorf("Could not convert specified ZonesWithNodes to a set: %s. This is a test error %s", test.ZonesWithNodes, test.Name)
				continue
			}
		}

		zone, err := SelectZoneForVolume(test.ZonePresent, test.ZonesPresent, test.Zone, zonesParameter, zonesWithNodes, test.Node, test.AllowedTopologies, test.Name)

		if test.Reject && err == nil {
			t.Errorf("Unexpected zone from SelectZoneForVolume for %s", zone)
			continue
		}

		if !test.Reject {
			if err != nil {
				t.Errorf("Unexpected error from SelectZoneForVolume for %s; Error: %v", test.Name, err)
				continue
			}

			if test.ExpectSpecificZone == true {
				if zone != test.ExpectedZone {
					t.Errorf("Expected zone %v does not match obtained zone %v for %s", test.ExpectedZone, zone, test.Name)
				}
				continue
			}

			expectedZones, err := ZonesToSet(test.ExpectedZones)
			if err != nil {
				t.Errorf("Could not convert ExpectedZones to a set: %s. This is a test error", test.ExpectedZones)
				continue
			}
			if !expectedZones.Has(zone) {
				t.Errorf("Obtained zone %s not member of expectedZones %s", zone, expectedZones)
			}
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
