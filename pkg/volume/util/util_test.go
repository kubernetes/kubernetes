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
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/sets"
	// util.go uses api.Codecs.LegacyCodec so import this package to do some
	// resource initialization.
	_ "k8s.io/kubernetes/pkg/api/install"
	"k8s.io/kubernetes/pkg/api/v1/helper"
)

var nodeLabels map[string]string = map[string]string{
	"test-key1": "test-value1",
	"test-key2": "test-value2",
}

func TestCheckNodeAffinity(t *testing.T) {
	type affinityTest struct {
		name          string
		expectSuccess bool
		pv            *v1.PersistentVolume
	}

	cases := []affinityTest{
		{
			name:          "valid-no-constraints",
			expectSuccess: true,
			pv:            testVolumeWithNodeAffinity(t, &v1.NodeAffinity{}),
		},
		{
			name:          "valid-constraints",
			expectSuccess: true,
			pv: testVolumeWithNodeAffinity(t, &v1.NodeAffinity{
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
			pv: testVolumeWithNodeAffinity(t, &v1.NodeAffinity{
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
			pv: testVolumeWithNodeAffinity(t, &v1.NodeAffinity{
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

func testVolumeWithNodeAffinity(t *testing.T, affinity *v1.NodeAffinity) *v1.PersistentVolume {
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
    - image: gcr.io/google_containers/busybox
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
        "image": "gcr.io/google_containers/busybox"
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
  - image: gcr.io/google_containers/busybox
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
