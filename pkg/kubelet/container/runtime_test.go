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

package container

import (
	"testing"

	"k8s.io/apimachinery/pkg/types"
)

func TestParseContainerID(t *testing.T) {
	tests := []struct {
		Input   string
		OutType string
		OutID   string
	}{
		{
			Input:   "test://some-container-id",
			OutType: "test",
			OutID:   "some-container-id",
		},
		{
			Input:   "test:/some-container-id",
			OutType: "",
			OutID:   "",
		},
	}

	for _, test := range tests {
		cID := ParseContainerID(test.Input)

		if cID.Type != test.OutType {
			t.Errorf("container type %s != %s", cID.Type, test.OutType)
		}

		if cID.ID != test.OutID {
			t.Errorf("contaier ID %s != %s", cID.ID, test.OutID)
		}
	}
}

func getTestPods() Pods {
	var testPods = Pods{
		&Pod{
			ID:        "uid_1",
			Name:      "pod 1",
			Namespace: "namespace 1",
		},
		&Pod{
			ID:        "uid_2",
			Name:      "pod 2",
			Namespace: "namespace 1",
		},
	}
	return testPods
}

func compareTestPods(p1 *Pod, p2 *Pod) bool {
	return p1.ID == p2.ID && p1.Name == p2.Name && p1.Namespace == p2.Namespace
}

func TestFindPodByID(t *testing.T) {
	testPods := getTestPods()
	tests := []struct {
		Input    types.UID
		FoundPod *Pod
	}{
		{
			Input:    "uid_1",
			FoundPod: testPods[0],
		},
		{
			Input:    "uid_2",
			FoundPod: testPods[1],
		},
		{
			Input:    "uid_NONEXIST",
			FoundPod: nil,
		},
	}

	for _, test := range tests {
		foundPod := testPods.FindPodByID(test.Input)
		if test.FoundPod != nil {
			if compareTestPods(&foundPod, test.FoundPod) == false {
				t.Errorf("Incorrect pod found when searching for ID %s", test.Input)
			}
		} else {
			if !foundPod.IsEmpty() {
				t.Errorf("Incorrect pod found when searching for unfindable pod ID")
			}
		}
	}
}

func TestFindPodByFullName(t *testing.T) {
	testPods := getTestPods()
	tests := []struct {
		Input    string
		FoundPod *Pod
	}{
		{
			Input:    "pod 1_namespace 1",
			FoundPod: testPods[0],
		},
		{
			Input:    "pod 2_namespace 1",
			FoundPod: testPods[1],
		},
		{
			Input:    "unfindable name",
			FoundPod: nil,
		},
	}

	for _, test := range tests {
		foundPod := testPods.FindPodByFullName(test.Input)
		if test.FoundPod != nil {
			if compareTestPods(&foundPod, test.FoundPod) == false {
				t.Errorf("Incorrect pod found when searching for Name %s", test.Input)
			}
		} else {
			if !foundPod.IsEmpty() {
				t.Errorf("Incorrect pod found when searching for unfindable pod name")
			}
		}
	}
}

func TestFindPod(t *testing.T) {
	testPods := getTestPods()
	foundPod := testPods.FindPod("pod 1_namespace 1", "")
	if compareTestPods(&foundPod, testPods[0]) == false {
		t.Errorf("Incorrect pod found when searching for pod pod 1_namespace 1")
	}

	foundPod = testPods.FindPod("", "uid_1")
	if compareTestPods(&foundPod, testPods[0]) == false {
		t.Errorf("Incorrect pod found when searching for pod uid_1")
	}
}

func TestToAPIPod(t *testing.T) {
	testPod := Pod{
		ID:        "test_uid",
		Name:      "test name",
		Namespace: "test namespace",
		Containers: []*Container{
			{
				ID: ContainerID{
					Type: "test_type",
					ID:   "test_id",
				},
				Name:    "container name",
				Image:   "container image",
				ImageID: "container_image_id",
				State:   ContainerStateRunning,
			},
		},
	}

	testAPIPod := testPod.ToAPIPod()
	if testAPIPod.UID != testPod.ID {
		t.Errorf("converted API pod UID %s != %s", testAPIPod.UID, testPod.ID)
	}
	if testAPIPod.Name != testPod.Name {
		t.Errorf("converted API pod Name %s != %s", testAPIPod.Name, testPod.Name)
	}
	if testAPIPod.Namespace != testPod.Namespace {
		t.Errorf("converted API pod Namespace %s != %s", testAPIPod.Namespace, testPod.Namespace)
	}
}
