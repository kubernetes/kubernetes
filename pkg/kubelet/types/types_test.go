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

package types

import (
	"reflect"
	"testing"

	"github.com/stretchr/testify/assert"
	"k8s.io/kubernetes/pkg/api/v1"
)

func TestConvertToTimestamp(t *testing.T) {
	timestamp := "2017-02-17T15:34:49.830882016+08:00"
	convertedTimeStamp := ConvertToTimestamp(timestamp).GetString()
	assert.Equal(t, timestamp, convertedTimeStamp)
}

func TestLen(t *testing.T) {
	var cases = []struct {
		statuses SortedContainerStatuses
		expected int
	}{
		{
			statuses: SortedContainerStatuses{{Name: "first"}},
			expected: 1,
		},
		{
			statuses: SortedContainerStatuses{{Name: "first"}, {Name: "second"}},
			expected: 2,
		},
	}
	for _, data := range cases {
		assert.Equal(t, data.expected, data.statuses.Len())
	}
}

func TestSwap(t *testing.T) {
	var cases = []struct {
		statuses SortedContainerStatuses
		expected SortedContainerStatuses
	}{
		{
			statuses: SortedContainerStatuses{{Name: "first"}, {Name: "second"}},
			expected: SortedContainerStatuses{{Name: "second"}, {Name: "first"}},
		},
	}
	for _, data := range cases {
		data.statuses.Swap(0, 1)
		if !reflect.DeepEqual(data.statuses, data.expected) {
			t.Errorf(
				"failed Swap:\n\texpected: %v\n\t  actual: %v",
				data.expected,
				data.statuses,
			)
		}
	}
}

func TestLess(t *testing.T) {
	var cases = []struct {
		statuses SortedContainerStatuses
		expected bool
	}{
		{
			statuses: SortedContainerStatuses{{Name: "first"}, {Name: "second"}},
			expected: true,
		},
		{
			statuses: SortedContainerStatuses{{Name: "second"}, {Name: "first"}},
			expected: false,
		},
	}
	for _, data := range cases {
		actual := data.statuses.Less(0, 1)
		if actual != data.expected {
			t.Errorf(
				"failed Less:\n\texpected: %t\n\t  actual: %t",
				data.expected,
				actual,
			)
		}
	}
}

func TestSortInitContainerStatuses(t *testing.T) {
	pod := v1.Pod{
		Spec: v1.PodSpec{},
	}
	var cases = []struct {
		containers     []v1.Container
		statuses       []v1.ContainerStatus
		sortedStatuses []v1.ContainerStatus
	}{
		{
			containers:     []v1.Container{{Name: "first"}, {Name: "second"}, {Name: "third"}, {Name: "fourth"}},
			statuses:       []v1.ContainerStatus{{Name: "first"}, {Name: "second"}, {Name: "third"}, {Name: "fourth"}},
			sortedStatuses: []v1.ContainerStatus{{Name: "first"}, {Name: "second"}, {Name: "third"}, {Name: "fourth"}},
		},
		{
			containers:     []v1.Container{{Name: "first"}, {Name: "second"}, {Name: "third"}, {Name: "fourth"}},
			statuses:       []v1.ContainerStatus{{Name: "second"}, {Name: "first"}, {Name: "fourth"}, {Name: "third"}},
			sortedStatuses: []v1.ContainerStatus{{Name: "first"}, {Name: "second"}, {Name: "third"}, {Name: "fourth"}},
		},
		{
			containers:     []v1.Container{{Name: "first"}, {Name: "second"}, {Name: "third"}, {Name: "fourth"}},
			statuses:       []v1.ContainerStatus{{Name: "fourth"}, {Name: "first"}},
			sortedStatuses: []v1.ContainerStatus{{Name: "first"}, {Name: "fourth"}},
		},
		{
			containers:     []v1.Container{{Name: "first"}, {Name: "second"}, {Name: "third"}, {Name: "fourth"}},
			statuses:       []v1.ContainerStatus{{Name: "first"}, {Name: "third"}},
			sortedStatuses: []v1.ContainerStatus{{Name: "first"}, {Name: "third"}},
		},
	}
	for _, data := range cases {
		pod.Spec.InitContainers = data.containers
		SortInitContainerStatuses(&pod, data.statuses)
		if !reflect.DeepEqual(data.statuses, data.sortedStatuses) {
			t.Errorf("SortInitContainerStatuses result wrong:\nContainers order: %v\nExpected order: %v\nReturne order: %v",
				data.containers, data.sortedStatuses, data.statuses)
		}
	}
}
