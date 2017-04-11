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

	"k8s.io/kubernetes/pkg/api/v1"
)

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
