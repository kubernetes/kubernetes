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

	"k8s.io/kubernetes/pkg/api"
)

func TestSortInitContainerStatuses(t *testing.T) {
	pod := api.Pod{
		Spec: api.PodSpec{},
	}
	var cases = []struct {
		containers     []api.Container
		statuses       []api.ContainerStatus
		sortedStatuses []api.ContainerStatus
	}{
		{
			containers:     []api.Container{{Name: "first"}, {Name: "second"}, {Name: "third"}, {Name: "fourth"}},
			statuses:       []api.ContainerStatus{{Name: "first"}, {Name: "second"}, {Name: "third"}, {Name: "fourth"}},
			sortedStatuses: []api.ContainerStatus{{Name: "first"}, {Name: "second"}, {Name: "third"}, {Name: "fourth"}},
		},
		{
			containers:     []api.Container{{Name: "first"}, {Name: "second"}, {Name: "third"}, {Name: "fourth"}},
			statuses:       []api.ContainerStatus{{Name: "second"}, {Name: "first"}, {Name: "fourth"}, {Name: "third"}},
			sortedStatuses: []api.ContainerStatus{{Name: "first"}, {Name: "second"}, {Name: "third"}, {Name: "fourth"}},
		},
		{
			containers:     []api.Container{{Name: "first"}, {Name: "second"}, {Name: "third"}, {Name: "fourth"}},
			statuses:       []api.ContainerStatus{{Name: "fourth"}, {Name: "first"}},
			sortedStatuses: []api.ContainerStatus{{Name: "first"}, {Name: "fourth"}},
		},
		{
			containers:     []api.Container{{Name: "first"}, {Name: "second"}, {Name: "third"}, {Name: "fourth"}},
			statuses:       []api.ContainerStatus{{Name: "first"}, {Name: "third"}},
			sortedStatuses: []api.ContainerStatus{{Name: "first"}, {Name: "third"}},
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
