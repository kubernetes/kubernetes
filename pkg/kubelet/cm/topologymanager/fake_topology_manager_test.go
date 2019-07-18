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

package topologymanager

import (
	"reflect"
	"testing"

	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/kubernetes/pkg/kubelet/lifecycle"
)

func TestNewFakeManager(t *testing.T) {
	fm := NewFakeManager()

	if _, ok := fm.(Manager); !ok {
		t.Errorf("Result is not Manager type")

	}
}

func TestFakeGetAffinity(t *testing.T) {
	tcases := []struct {
		name          string
		containerName string
		podUID        string
		expected      TopologyHint
	}{
		{
			name:          "Case1",
			containerName: "nginx",
			podUID:        "0aafa4c4-38e8-11e9-bcb1-a4bf01040474",
			expected:      TopologyHint{},
		},
	}
	for _, tc := range tcases {
		fm := fakeManager{}
		actual := fm.GetAffinity(tc.podUID, tc.containerName)
		if !reflect.DeepEqual(actual, tc.expected) {
			t.Errorf("Expected Affinity in result to be %v, got %v", tc.expected, actual)
		}
	}
}

func TestFakeAddContainer(t *testing.T) {
	testCases := []struct {
		name        string
		containerID string
		podUID      types.UID
	}{
		{
			name:        "Case1",
			containerID: "nginx",
			podUID:      "0aafa4c4-38e8-11e9-bcb1-a4bf01040474",
		},
		{
			name:        "Case2",
			containerID: "Busy_Box",
			podUID:      "b3ee37fc-39a5-11e9-bcb1-a4bf01040474",
		},
	}
	fm := fakeManager{}
	mngr := manager{}
	mngr.podMap = make(map[string]string)
	for _, tc := range testCases {
		pod := v1.Pod{}
		pod.UID = tc.podUID
		err := fm.AddContainer(&pod, tc.containerID)
		if err != nil {
			t.Errorf("Expected error to be nil but got: %v", err)

		}

	}
}

func TestFakeRemoveContainer(t *testing.T) {
	testCases := []struct {
		name        string
		containerID string
		podUID      string
	}{
		{
			name:        "Case1",
			containerID: "nginx",
			podUID:      "0aafa4c4-38e8-11e9-bcb1-a4bf01040474",
		},
		{
			name:        "Case2",
			containerID: "Busy_Box",
			podUID:      "b3ee37fc-39a5-11e9-bcb1-a4bf01040474",
		},
	}
	fm := fakeManager{}
	mngr := manager{}
	mngr.podMap = make(map[string]string)
	for _, tc := range testCases {
		err := fm.RemoveContainer(tc.containerID)
		if err != nil {
			t.Errorf("Expected error to be nil but got: %v", err)
		}

	}

}

func TestFakeAdmit(t *testing.T) {
	tcases := []struct {
		name     string
		result   lifecycle.PodAdmitResult
		qosClass v1.PodQOSClass
		expected bool
	}{
		{
			name:     "QOSClass set as Guaranteed",
			result:   lifecycle.PodAdmitResult{},
			qosClass: v1.PodQOSGuaranteed,
			expected: true,
		},
		{
			name:     "QOSClass set as Burstable",
			result:   lifecycle.PodAdmitResult{},
			qosClass: v1.PodQOSBurstable,
			expected: true,
		},
		{
			name:     "QOSClass set as BestEffort",
			result:   lifecycle.PodAdmitResult{},
			qosClass: v1.PodQOSBestEffort,
			expected: true,
		},
	}
	fm := fakeManager{}
	for _, tc := range tcases {
		mngr := manager{}
		mngr.podTopologyHints = make(map[string]map[string]TopologyHint)
		podAttr := lifecycle.PodAdmitAttributes{}
		pod := v1.Pod{}
		pod.Status.QOSClass = tc.qosClass
		podAttr.Pod = &pod
		actual := fm.Admit(&podAttr)
		if reflect.DeepEqual(actual, tc.result) {
			t.Errorf("Error occurred, expected Admit in result to be %v got %v", tc.result, actual.Admit)
		}
	}
}
