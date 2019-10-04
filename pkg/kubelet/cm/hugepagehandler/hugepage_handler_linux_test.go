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

package hugepagehandler

import (
	"fmt"
	"reflect"
	"testing"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	api "k8s.io/kubernetes/pkg/apis/core"

	runtimeapi "k8s.io/cri-api/pkg/apis/runtime/v1alpha2"
)

type mockRuntimeService struct {
	err error
}

func (rt mockRuntimeService) UpdateContainerResources(id string, resources *runtimeapi.LinuxContainerResources) error {
	result := &runtimeapi.HugepageLimit{
		PageSize: "1GB",
		Limit:    1073741824,
	}
	if !reflect.DeepEqual(result, resources.HugepageLimits[0]) {
		return fmt.Errorf("Fail to parse hugepage limit")
	}
	return rt.err
}

func makePod(hugepageSize string, hugepageAmount string) *v1.Pod {
	return &v1.Pod{
		Spec: v1.PodSpec{
			Containers: []v1.Container{
				{
					Resources: v1.ResourceRequirements{
						Limits: v1.ResourceList{
							v1.ResourceName(api.ResourceHugePagesPrefix + hugepageSize): resource.MustParse(hugepageAmount),
						},
					},
				},
			},
		},
	}
}

func TestHugepageHandlerAdd(t *testing.T) {
	testCases := []struct {
		description string
		updateErr   error
		wantErr     bool
	}{
		{
			description: "hugepage handler add - no error",
			updateErr:   nil,
			wantErr:     false,
		},
		{
			description: "hugepage handler add - container update error",
			updateErr:   fmt.Errorf("fake update error"),
			wantErr:     true,
		},
	}

	for _, testCase := range testCases {
		hdr := &handler{
			containerRuntime: mockRuntimeService{
				err: testCase.updateErr,
			},
		}

		pod := makePod("1Gi", "1Gi")
		container := &pod.Spec.Containers[0]
		err := hdr.AddContainer(pod, container, "fakeID")
		if err != nil {
			if testCase.wantErr {
				t.Logf("AddContainer() expected error = %v", err)
			} else {
				t.Errorf("Hugepage Handler AddContainer() error (%v). wantErr: %v but got: %v",
					testCase.description, testCase.wantErr, err)
			}
			return
		}
	}
}
