/*
Copyright 2014 Google Inc. All rights reserved.

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

package limitranger

import (
	"testing"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/resource"
)

func getResourceRequirements(cpu, memory string) api.ResourceRequirements {
	res := api.ResourceRequirements{}
	res.Limits = api.ResourceList{}
	if cpu != "" {
		res.Limits[api.ResourceCPU] = resource.MustParse(cpu)
	}
	if memory != "" {
		res.Limits[api.ResourceMemory] = resource.MustParse(memory)
	}

	return res
}

func TestPodLimitFunc(t *testing.T) {
	limitRange := &api.LimitRange{
		ObjectMeta: api.ObjectMeta{
			Name: "abc",
		},
		Spec: api.LimitRangeSpec{
			Limits: []api.LimitRangeItem{
				{
					Type: api.LimitTypePod,
					Max:  getResourceRequirements("200m", "4Gi").Limits,
					Min:  getResourceRequirements("50m", "2Mi").Limits,
				},
				{
					Type: api.LimitTypeContainer,
					Max:  getResourceRequirements("100m", "2Gi").Limits,
					Min:  getResourceRequirements("25m", "1Mi").Limits,
				},
			},
		},
	}

	successCases := []api.Pod{
		{
			ObjectMeta: api.ObjectMeta{Name: "foo"},
			Spec: api.PodSpec{
				Containers: []api.Container{
					{
						Image:     "foo:V1",
						Resources: getResourceRequirements("100m", "2Gi"),
					},
					{
						Image:     "boo:V1",
						Resources: getResourceRequirements("100m", "2Gi"),
					},
				},
			},
		},
		{
			ObjectMeta: api.ObjectMeta{Name: "bar"},
			Spec: api.PodSpec{
				Containers: []api.Container{
					{
						Image:     "boo:V1",
						Resources: getResourceRequirements("100m", "2Gi"),
					},
				},
			},
		},
	}

	errorCases := map[string]api.Pod{
		"min-container-cpu": {
			ObjectMeta: api.ObjectMeta{Name: "foo"},
			Spec: api.PodSpec{
				Containers: []api.Container{
					{
						Image:     "boo:V1",
						Resources: getResourceRequirements("25m", "2Gi"),
					},
				},
			},
		},
		"max-container-cpu": {
			ObjectMeta: api.ObjectMeta{Name: "foo"},
			Spec: api.PodSpec{
				Containers: []api.Container{
					{
						Image:     "boo:V1",
						Resources: getResourceRequirements("110m", "1Gi"),
					},
				},
			},
		},
		"min-container-mem": {
			ObjectMeta: api.ObjectMeta{Name: "foo"},
			Spec: api.PodSpec{
				Containers: []api.Container{
					{
						Image:     "boo:V1",
						Resources: getResourceRequirements("30m", "0"),
					},
				},
			},
		},
		"max-container-mem": {
			ObjectMeta: api.ObjectMeta{Name: "foo"},
			Spec: api.PodSpec{
				Containers: []api.Container{
					{
						Image:     "boo:V1",
						Resources: getResourceRequirements("30m", "3Gi"),
					},
				},
			},
		},
		"min-pod-cpu": {
			ObjectMeta: api.ObjectMeta{Name: "foo"},
			Spec: api.PodSpec{
				Containers: []api.Container{
					{
						Image:     "boo:V1",
						Resources: getResourceRequirements("40m", "2Gi"),
					},
				},
			},
		},
		"max-pod-cpu": {
			ObjectMeta: api.ObjectMeta{Name: "foo"},
			Spec: api.PodSpec{
				Containers: []api.Container{
					{
						Image:     "boo:V1",
						Resources: getResourceRequirements("60m", "1Mi"),
					},
					{
						Image:     "boo:V2",
						Resources: getResourceRequirements("60m", "1Mi"),
					},
					{
						Image:     "boo:V3",
						Resources: getResourceRequirements("60m", "1Mi"),
					},
					{
						Image:     "boo:V4",
						Resources: getResourceRequirements("60m", "1Mi"),
					},
				},
			},
		},
		"max-pod-memory": {
			ObjectMeta: api.ObjectMeta{Name: "foo"},
			Spec: api.PodSpec{
				Containers: []api.Container{
					{
						Image:     "boo:V1",
						Resources: getResourceRequirements("60m", "2Gi"),
					},
					{
						Image:     "boo:V2",
						Resources: getResourceRequirements("60m", "2Gi"),
					},
					{
						Image:     "boo:V3",
						Resources: getResourceRequirements("60m", "2Gi"),
					},
				},
			},
		},
		"min-pod-memory": {
			ObjectMeta: api.ObjectMeta{Name: "foo"},
			Spec: api.PodSpec{
				Containers: []api.Container{
					{
						Image:     "boo:V1",
						Resources: getResourceRequirements("60m", "0"),
					},
					{
						Image:     "boo:V2",
						Resources: getResourceRequirements("60m", "0"),
					},
					{
						Image:     "boo:V3",
						Resources: getResourceRequirements("60m", "0"),
					},
				},
			},
		},
	}

	for i := range successCases {
		err := PodLimitFunc(limitRange, "pods", &successCases[i])
		if err != nil {
			t.Errorf("Unexpected error for valid pod: %v, %v", successCases[i].Name, err)
		}
	}

	for k, v := range errorCases {
		err := PodLimitFunc(limitRange, "pods", &v)
		if err == nil {
			t.Errorf("Expected error for %s", k)
		}
	}
}
