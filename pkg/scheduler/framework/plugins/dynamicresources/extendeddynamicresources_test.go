/*
Copyright 2025 The Kubernetes Authors.

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

package dynamicresources

import (
	"sort"
	"testing"

	v1 "k8s.io/api/core/v1"
	resourceapi "k8s.io/api/resource/v1"
	st "k8s.io/kubernetes/pkg/scheduler/testing"
)

func Test_createDeviceRequests(t *testing.T) {
	pod1 := st.MakePod().Name(podName).Namespace(namespace).
		UID(podUID).
		Res(map[v1.ResourceName]string{
			v1.ResourceName(extendedResourceName):       "1",
			v1.ResourceName(extendedResourceName + "1"): "2",
		}).
		Obj()
	pod2 := st.MakePod().Name(podName).Namespace(namespace).
		UID(podUID).
		Res(map[v1.ResourceName]string{
			v1.ResourceName(extendedResourceName): "1",
		}).
		Res(map[v1.ResourceName]string{
			v1.ResourceName(extendedResourceName + "1"): "2",
		}).
		Obj()

	podInit := st.MakePod().Name(podName).Namespace(namespace).
		UID(podUID).
		Res(map[v1.ResourceName]string{
			v1.ResourceName(extendedResourceName): "1",
		}).
		InitReq(map[v1.ResourceName]string{
			v1.ResourceName(extendedResourceName + "init"): "2",
		}).
		Obj()

	res := map[v1.ResourceName]int64{
		v1.ResourceName(extendedResourceName): 1,
	}
	res2 := map[v1.ResourceName]int64{
		v1.ResourceName(extendedResourceName):       1,
		v1.ResourceName(extendedResourceName + "1"): 2,
	}
	resInit := map[v1.ResourceName]int64{
		v1.ResourceName(extendedResourceName):          1,
		v1.ResourceName(extendedResourceName + "init"): 2,
	}
	devMap := map[v1.ResourceName]string{
		v1.ResourceName(extendedResourceName): "class",
	}
	devMap2 := map[v1.ResourceName]string{
		v1.ResourceName(extendedResourceName):       "class",
		v1.ResourceName(extendedResourceName + "1"): "class1",
	}
	devMapInit := map[v1.ResourceName]string{
		v1.ResourceName(extendedResourceName):          "class",
		v1.ResourceName(extendedResourceName + "init"): "classInit",
	}
	devReq := resourceapi.DeviceRequest{
		Name: "container-0-request-0",
		Exactly: &resourceapi.ExactDeviceRequest{
			DeviceClassName: "class",
			AllocationMode:  resourceapi.DeviceAllocationModeExactCount,
			Count:           1,
		},
	}
	devReq2 := resourceapi.DeviceRequest{
		Name: "container-0-request-1",
		Exactly: &resourceapi.ExactDeviceRequest{
			DeviceClassName: "class1",
			AllocationMode:  resourceapi.DeviceAllocationModeExactCount,
			Count:           2,
		},
	}
	devReq3 := resourceapi.DeviceRequest{
		Name: "container-1-request-0",
		Exactly: &resourceapi.ExactDeviceRequest{
			DeviceClassName: "class1",
			AllocationMode:  resourceapi.DeviceAllocationModeExactCount,
			Count:           2,
		},
	}
	devReqInit := resourceapi.DeviceRequest{
		Name: "container-1-request-0",
		Exactly: &resourceapi.ExactDeviceRequest{
			DeviceClassName: "class",
			AllocationMode:  resourceapi.DeviceAllocationModeExactCount,
			Count:           1,
		},
	}
	devReq2Init := resourceapi.DeviceRequest{
		Name: "container-0-request-0",
		Exactly: &resourceapi.ExactDeviceRequest{
			DeviceClassName: "classInit",
			AllocationMode:  resourceapi.DeviceAllocationModeExactCount,
			Count:           2,
		},
	}

	testcases := map[string]struct {
		pod                *v1.Pod
		extendedResources  map[v1.ResourceName]int64
		deviceClassMapping map[v1.ResourceName]string
		wantDeviceRequests []resourceapi.DeviceRequest
	}{
		"nil": {
			pod:                pod1,
			wantDeviceRequests: nil,
		},
		"one resource match": {
			pod:                pod1,
			extendedResources:  res,
			deviceClassMapping: devMap,
			wantDeviceRequests: []resourceapi.DeviceRequest{devReq},
		},
		"one resource match, one resource not match": {
			pod:                pod1,
			extendedResources:  res2,
			deviceClassMapping: devMap,
			wantDeviceRequests: []resourceapi.DeviceRequest{devReq},
		},
		"two resources match": {
			pod:                pod1,
			extendedResources:  res2,
			deviceClassMapping: devMap2,
			wantDeviceRequests: []resourceapi.DeviceRequest{devReq, devReq2},
		},
		"two containers match": {
			pod:                pod2,
			extendedResources:  res2,
			deviceClassMapping: devMap2,
			wantDeviceRequests: []resourceapi.DeviceRequest{devReq, devReq3},
		},
		"one init container, one regular container": {
			pod:                podInit,
			extendedResources:  resInit,
			deviceClassMapping: devMapInit,
			wantDeviceRequests: []resourceapi.DeviceRequest{devReq2Init, devReqInit},
		},
	}

	for name, tc := range testcases {
		t.Run(name, func(t *testing.T) {
			gotDeviceRequests := createDeviceRequests(tc.pod, tc.extendedResources, tc.deviceClassMapping)
			if len(tc.wantDeviceRequests) != len(gotDeviceRequests) {
				t.Fatalf("different length, want %#v, got %#v", tc.wantDeviceRequests, gotDeviceRequests)
			}
			sort.Slice(gotDeviceRequests, func(i, j int) bool { return gotDeviceRequests[i].Name < gotDeviceRequests[j].Name })
			for i, r := range tc.wantDeviceRequests {
				if r.Name != gotDeviceRequests[i].Name {
					t.Fatalf("different name, want %#v, got %#v", r, gotDeviceRequests[i])
				}
				if r.Exactly.DeviceClassName != gotDeviceRequests[i].Exactly.DeviceClassName {
					t.Fatalf("different deviceClassName, want %#v, got %#v", r, gotDeviceRequests[i])
				}
				if r.Exactly.AllocationMode != gotDeviceRequests[i].Exactly.AllocationMode {
					t.Fatalf("different allocationMode, want %#v, got %#v", r, gotDeviceRequests[i])
				}
				if r.Exactly.Count != gotDeviceRequests[i].Exactly.Count {
					t.Fatalf("different count, want %#v, got %#v", r, gotDeviceRequests[i])
				}
			}
		})
	}
}

func Test_createRequestMappings(t *testing.T) {
	pod1 := st.MakePod().Name(podName).Namespace(namespace).
		UID(podUID).
		Res(map[v1.ResourceName]string{
			v1.ResourceName(extendedResourceName):       "1",
			v1.ResourceName(extendedResourceName + "1"): "2",
		}).
		Obj()
	pod2 := st.MakePod().Name(podName).Namespace(namespace).
		UID(podUID).
		Res(map[v1.ResourceName]string{
			v1.ResourceName(extendedResourceName): "1",
		}).
		Res(map[v1.ResourceName]string{
			v1.ResourceName(extendedResourceName + "1"): "2",
		}).
		Obj()

	podInit := st.MakePod().Name(podName).Namespace(namespace).
		UID(podUID).
		Res(map[v1.ResourceName]string{
			v1.ResourceName(extendedResourceName): "1",
		}).
		InitReq(map[v1.ResourceName]string{
			v1.ResourceName(extendedResourceName + "init"): "2",
		}).
		Obj()

	claim := st.MakeResourceClaim().
		Name(claimName).
		Namespace(namespace).
		RequestWithName("container-0-request-0", className).
		Obj()
	claim2 := st.MakeResourceClaim().
		Name(claimName).
		Namespace(namespace).
		RequestWithName("container-0-request-0", className).
		RequestWithName("container-1-request-0", className).
		Obj()

	cer := v1.ContainerExtendedResourceRequest{
		ContainerName: "con0",
		ResourceName:  extendedResourceName,
		RequestName:   "container-0-request-0",
	}
	cer2 := v1.ContainerExtendedResourceRequest{
		ContainerName: "con1",
		ResourceName:  extendedResourceName + "1",
		RequestName:   "container-1-request-0",
	}
	cer3 := v1.ContainerExtendedResourceRequest{
		ContainerName: "con0",
		ResourceName:  extendedResourceName,
		RequestName:   "container-1-request-0",
	}
	cerInit := v1.ContainerExtendedResourceRequest{
		ContainerName: "init-con0",
		ResourceName:  extendedResourceName + "init",
		RequestName:   "container-0-request-0",
	}

	testcases := map[string]struct {
		claim           *resourceapi.ResourceClaim
		pod             *v1.Pod
		wantReqMappings []v1.ContainerExtendedResourceRequest
	}{
		"one container, one request": {
			claim:           claim,
			pod:             pod1,
			wantReqMappings: []v1.ContainerExtendedResourceRequest{cer},
		},
		"two containers, one request": {
			claim:           claim,
			pod:             pod2,
			wantReqMappings: []v1.ContainerExtendedResourceRequest{cer},
		},
		"one init container, one regular container, one request": {
			claim:           claim,
			pod:             podInit,
			wantReqMappings: []v1.ContainerExtendedResourceRequest{cerInit},
		},
		"two containers, two requests": {
			claim:           claim2,
			pod:             pod2,
			wantReqMappings: []v1.ContainerExtendedResourceRequest{cer, cer2},
		},
		"two containers (one is init container), two requests": {
			claim:           claim2,
			pod:             podInit,
			wantReqMappings: []v1.ContainerExtendedResourceRequest{cerInit, cer3},
		},
	}

	for name, tc := range testcases {
		t.Run(name, func(t *testing.T) {
			gotReqMappings := createRequestMappings(tc.claim, tc.pod)
			if len(tc.wantReqMappings) != len(gotReqMappings) {
				t.Fatalf("different length, want %#v, got %#v", tc.wantReqMappings, gotReqMappings)
			}
			sort.Slice(gotReqMappings, func(i, j int) bool { return gotReqMappings[i].RequestName < gotReqMappings[j].RequestName })
			for i, r := range tc.wantReqMappings {
				if r.RequestName != gotReqMappings[i].RequestName {
					t.Fatalf("different request name, want %#v, got %#v", r, gotReqMappings[i])
				}
				if r.ContainerName != gotReqMappings[i].ContainerName {
					t.Fatalf("different container name, want %#v, got %#v", r, gotReqMappings[i])
				}
				if r.ResourceName != gotReqMappings[i].ResourceName {
					t.Fatalf("different resource name, want %#v, got %#v", r, gotReqMappings[i])
				}
			}
		})
	}
}
