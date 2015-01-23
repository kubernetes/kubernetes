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

func TestPodLimitFunc(t *testing.T) {
	limitRange := &api.LimitRange{
		ObjectMeta: api.ObjectMeta{
			Name: "abc",
		},
		Spec: api.LimitRangeSpec{
			Limits: []api.LimitRangeItem{
				{
					Kind: "pods",
					Max: api.ResourceList{
						api.ResourceCPU:    resource.MustParse("200m"),
						api.ResourceMemory: resource.MustParse("4Gi"),
					},
					Min: api.ResourceList{
						api.ResourceCPU:    resource.MustParse("50m"),
						api.ResourceMemory: resource.MustParse("2Mi"),
					},
				},
				{
					Kind: "containers",
					Max: api.ResourceList{
						api.ResourceCPU:    resource.MustParse("100m"),
						api.ResourceMemory: resource.MustParse("2Gi"),
					},
					Min: api.ResourceList{
						api.ResourceCPU:    resource.MustParse("25m"),
						api.ResourceMemory: resource.MustParse("1Mi"),
					},
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
						Image:  "foo:V1",
						CPU:    resource.MustParse("100m"),
						Memory: resource.MustParse("2Gi"),
					},
					{
						Image:  "boo:V1",
						CPU:    resource.MustParse("100m"),
						Memory: resource.MustParse("2Gi"),
					},
				},
			},
		},
		{
			ObjectMeta: api.ObjectMeta{Name: "bar"},
			Spec: api.PodSpec{
				Containers: []api.Container{
					{
						Image:  "boo:V1",
						CPU:    resource.MustParse("100m"),
						Memory: resource.MustParse("2Gi"),
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
						Image:  "boo:V1",
						CPU:    resource.MustParse("25m"),
						Memory: resource.MustParse("2Gi"),
					},
				},
			},
		},
		"max-container-cpu": {
			ObjectMeta: api.ObjectMeta{Name: "foo"},
			Spec: api.PodSpec{
				Containers: []api.Container{
					{
						Image:  "boo:V1",
						CPU:    resource.MustParse("110m"),
						Memory: resource.MustParse("1Gi"),
					},
				},
			},
		},
		"min-container-mem": {
			ObjectMeta: api.ObjectMeta{Name: "foo"},
			Spec: api.PodSpec{
				Containers: []api.Container{
					{
						Image:  "boo:V1",
						CPU:    resource.MustParse("30m"),
						Memory: resource.MustParse("0"),
					},
				},
			},
		},
		"max-container-mem": {
			ObjectMeta: api.ObjectMeta{Name: "foo"},
			Spec: api.PodSpec{
				Containers: []api.Container{
					{
						Image:  "boo:V1",
						CPU:    resource.MustParse("30m"),
						Memory: resource.MustParse("3Gi"),
					},
				},
			},
		},
		"min-pod-cpu": {
			ObjectMeta: api.ObjectMeta{Name: "foo"},
			Spec: api.PodSpec{
				Containers: []api.Container{
					{
						Image:  "boo:V1",
						CPU:    resource.MustParse("40m"),
						Memory: resource.MustParse("2Gi"),
					},
				},
			},
		},
		"max-pod-cpu": {
			ObjectMeta: api.ObjectMeta{Name: "foo"},
			Spec: api.PodSpec{
				Containers: []api.Container{
					{
						Image:  "boo:V1",
						CPU:    resource.MustParse("60m"),
						Memory: resource.MustParse("1Mi"),
					},
					{
						Image:  "boo:V2",
						CPU:    resource.MustParse("60m"),
						Memory: resource.MustParse("1Mi"),
					},
					{
						Image:  "boo:V3",
						CPU:    resource.MustParse("60m"),
						Memory: resource.MustParse("1Mi"),
					},
					{
						Image:  "boo:V4",
						CPU:    resource.MustParse("60m"),
						Memory: resource.MustParse("1Mi"),
					},
				},
			},
		},
		"max-pod-memory": {
			ObjectMeta: api.ObjectMeta{Name: "foo"},
			Spec: api.PodSpec{
				Containers: []api.Container{
					{
						Image:  "boo:V1",
						CPU:    resource.MustParse("60m"),
						Memory: resource.MustParse("2Gi"),
					},
					{
						Image:  "boo:V2",
						CPU:    resource.MustParse("60m"),
						Memory: resource.MustParse("2Gi"),
					},
					{
						Image:  "boo:V3",
						CPU:    resource.MustParse("60m"),
						Memory: resource.MustParse("2Gi"),
					},
				},
			},
		},
		"min-pod-memory": {
			ObjectMeta: api.ObjectMeta{Name: "foo"},
			Spec: api.PodSpec{
				Containers: []api.Container{
					{
						Image:  "boo:V1",
						CPU:    resource.MustParse("60m"),
						Memory: resource.MustParse("0"),
					},
					{
						Image:  "boo:V2",
						CPU:    resource.MustParse("60m"),
						Memory: resource.MustParse("0"),
					},
					{
						Image:  "boo:V3",
						CPU:    resource.MustParse("60m"),
						Memory: resource.MustParse("0"),
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
