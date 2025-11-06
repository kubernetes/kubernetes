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
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	fwk "k8s.io/kube-scheduler/framework"
	st "k8s.io/kubernetes/pkg/scheduler/testing"
	"k8s.io/kubernetes/test/utils/ktesting"
)

func Test_createRequestsAndMappings_requests(t *testing.T) {
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
	podInit2 := st.MakePod().Name(podName).Namespace(namespace).
		UID(podUID).
		Res(map[v1.ResourceName]string{
			v1.ResourceName(extendedResourceName): "1",
		}).
		InitReq(map[v1.ResourceName]string{
			v1.ResourceName(extendedResourceName + "init"): "1",
		}).
		InitReq(map[v1.ResourceName]string{
			v1.ResourceName(extendedResourceName + "init"): "2",
		}).
		Obj()
	podInit3 := st.MakePod().Name(podName).Namespace(namespace).
		UID(podUID).
		Res(map[v1.ResourceName]string{
			v1.ResourceName(extendedResourceName): "1",
		}).
		InitReq(map[v1.ResourceName]string{
			v1.ResourceName(extendedResourceName + "init"): "1",
		}).
		SidecarReq(map[v1.ResourceName]string{
			v1.ResourceName(extendedResourceName + "init"): "1",
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
	devMap := map[v1.ResourceName]*resourceapi.DeviceClass{
		v1.ResourceName(extendedResourceName): {
			ObjectMeta: metav1.ObjectMeta{
				Name: "class",
			},
		},
	}
	devMap2 := map[v1.ResourceName]*resourceapi.DeviceClass{
		v1.ResourceName(extendedResourceName): {
			ObjectMeta: metav1.ObjectMeta{
				Name: "class",
			},
		},
		v1.ResourceName(extendedResourceName + "1"): {
			ObjectMeta: metav1.ObjectMeta{
				Name: "class1",
			},
		},
	}
	devMapInit := map[v1.ResourceName]*resourceapi.DeviceClass{
		v1.ResourceName(extendedResourceName): {
			ObjectMeta: metav1.ObjectMeta{
				Name: "class",
			},
		},
		v1.ResourceName(extendedResourceName + "init"): {
			ObjectMeta: metav1.ObjectMeta{
				Name: "classInit",
			},
		},
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
	devReqSidecar := resourceapi.DeviceRequest{
		Name: "container-1-request-0",
		Exactly: &resourceapi.ExactDeviceRequest{
			DeviceClassName: "classInit",
			AllocationMode:  resourceapi.DeviceAllocationModeExactCount,
			Count:           1,
		},
	}
	devReq2Init := resourceapi.DeviceRequest{
		Name: "container-1-request-0",
		Exactly: &resourceapi.ExactDeviceRequest{
			DeviceClassName: "classInit",
			AllocationMode:  resourceapi.DeviceAllocationModeExactCount,
			Count:           2,
		},
	}
	devReq6Init := resourceapi.DeviceRequest{
		Name: "container-0-request-0",
		Exactly: &resourceapi.ExactDeviceRequest{
			DeviceClassName: "classInit",
			AllocationMode:  resourceapi.DeviceAllocationModeExactCount,
			Count:           2,
		},
	}
	devReq3Init := resourceapi.DeviceRequest{
		Name: "container-2-request-0",
		Exactly: &resourceapi.ExactDeviceRequest{
			DeviceClassName: "class",
			AllocationMode:  resourceapi.DeviceAllocationModeExactCount,
			Count:           1,
		},
	}
	devReq4Init := resourceapi.DeviceRequest{
		Name: "container-3-request-0",
		Exactly: &resourceapi.ExactDeviceRequest{
			DeviceClassName: "class",
			AllocationMode:  resourceapi.DeviceAllocationModeExactCount,
			Count:           1,
		},
	}
	devReq5Init := resourceapi.DeviceRequest{
		Name: "container-2-request-0",
		Exactly: &resourceapi.ExactDeviceRequest{
			DeviceClassName: "classInit",
			AllocationMode:  resourceapi.DeviceAllocationModeExactCount,
			Count:           2,
		},
	}

	testcases := map[string]struct {
		pod                *v1.Pod
		extendedResources  map[v1.ResourceName]int64
		cache              fwk.DeviceClassResolver
		wantDeviceRequests []resourceapi.DeviceRequest
	}{
		"nil": {
			pod:                pod1,
			wantDeviceRequests: nil,
		},
		"one resource match": {
			pod:                pod1,
			extendedResources:  res,
			cache:              &mockDeviceClassResolver{mapping: devMap},
			wantDeviceRequests: []resourceapi.DeviceRequest{devReq},
		},
		"one resource match, one resource not match": {
			pod:                pod1,
			extendedResources:  res2,
			cache:              &mockDeviceClassResolver{mapping: devMap},
			wantDeviceRequests: []resourceapi.DeviceRequest{devReq},
		},
		"two resources match": {
			pod:                pod1,
			extendedResources:  res2,
			cache:              &mockDeviceClassResolver{mapping: devMap2},
			wantDeviceRequests: []resourceapi.DeviceRequest{devReq, devReq2},
		},
		"two containers match": {
			pod:                pod2,
			extendedResources:  res2,
			cache:              &mockDeviceClassResolver{mapping: devMap2},
			wantDeviceRequests: []resourceapi.DeviceRequest{devReq, devReq3},
		},
		"one init container, one regular container": {
			pod:                podInit,
			extendedResources:  resInit,
			cache:              &mockDeviceClassResolver{mapping: devMapInit},
			wantDeviceRequests: []resourceapi.DeviceRequest{devReq6Init, devReqInit},
		},
		"two init containers, one regular container": {
			pod:                podInit2,
			extendedResources:  resInit,
			cache:              &mockDeviceClassResolver{mapping: devMapInit},
			wantDeviceRequests: []resourceapi.DeviceRequest{devReq2Init, devReq3Init},
		},
		"three init containers, one sidecar, one regular container": {
			pod:                podInit3,
			extendedResources:  resInit,
			cache:              &mockDeviceClassResolver{mapping: devMapInit},
			wantDeviceRequests: []resourceapi.DeviceRequest{devReqSidecar, devReq5Init, devReq4Init},
		},
	}

	for name, tc := range testcases {
		t.Run(name, func(t *testing.T) {
			logger, _ := ktesting.NewTestContext(t)
			gotDeviceRequests, _ := createRequestsAndMappings(tc.pod, tc.extendedResources, logger, tc.cache)
			if len(tc.wantDeviceRequests) != len(gotDeviceRequests) {
				t.Fatalf("different length, want %#v, len=%v, got %#v, len=%v", tc.wantDeviceRequests, len(tc.wantDeviceRequests), gotDeviceRequests, len(gotDeviceRequests))
			}
			// gotDeviceRequests should already be sorted by createRequestsAndMappings
			for i, r := range tc.wantDeviceRequests {
				if r.Name != gotDeviceRequests[i].Name {
					t.Errorf("different name, want %#v, got %#v", r, gotDeviceRequests[i])
				}
				if r.Exactly.DeviceClassName != gotDeviceRequests[i].Exactly.DeviceClassName {
					t.Errorf("different deviceClassName, want %#v, got %#v", r, gotDeviceRequests[i])
				}
				if r.Exactly.AllocationMode != gotDeviceRequests[i].Exactly.AllocationMode {
					t.Errorf("different allocationMode, want %#v, got %#v", r, gotDeviceRequests[i])
				}
				if r.Exactly.Count != gotDeviceRequests[i].Exactly.Count {
					t.Errorf("different count, want %#v, got %#v", r.Exactly.Count, gotDeviceRequests[i].Exactly.Count)
				}
			}
		})
	}
}

func Test_createRequestsAndMappings_mappings(t *testing.T) {
	pod1 := st.MakePod().Name(podName).Namespace(namespace).
		UID(podUID).
		Res(map[v1.ResourceName]string{
			v1.ResourceName(extendedResourceName):       "1",
			v1.ResourceName(extendedResourceName + "1"): "2",
		}).
		Obj()
	pod1InitImplicit := st.MakePod().Name(podName).Namespace(namespace).
		UID(podUID).
		InitReq(map[v1.ResourceName]string{
			v1.ResourceName(extendedResourceName + "init"):                       "1",
			v1.ResourceName(resourceapi.ResourceDeviceClassPrefix + "classInit"): "2",
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
	podInit2 := st.MakePod().Name(podName).Namespace(namespace).
		UID(podUID).
		Res(map[v1.ResourceName]string{
			v1.ResourceName(extendedResourceName): "1",
		}).
		InitReq(map[v1.ResourceName]string{
			v1.ResourceName(extendedResourceName + "init"): "1",
		}).
		InitReq(map[v1.ResourceName]string{
			v1.ResourceName(extendedResourceName + "init"): "2",
		}).
		Obj()
	podInitImplicit := st.MakePod().Name(podName).Namespace(namespace).
		UID(podUID).
		Res(map[v1.ResourceName]string{
			v1.ResourceName(extendedResourceName): "1",
		}).
		InitReq(map[v1.ResourceName]string{
			v1.ResourceName(extendedResourceName + "init"): "1",
		}).
		InitReq(map[v1.ResourceName]string{
			v1.ResourceName(resourceapi.ResourceDeviceClassPrefix + "classInit"): "2",
		}).
		Obj()
	podInit3 := st.MakePod().Name(podName).Namespace(namespace).
		UID(podUID).
		Res(map[v1.ResourceName]string{
			v1.ResourceName(extendedResourceName): "1",
		}).
		InitReq(map[v1.ResourceName]string{
			v1.ResourceName(extendedResourceName + "init"): "1",
		}).
		SidecarReq(map[v1.ResourceName]string{
			v1.ResourceName(extendedResourceName + "init"): "1",
		}).
		InitReq(map[v1.ResourceName]string{
			v1.ResourceName(extendedResourceName + "init"): "2",
		}).
		Obj()
	res := map[v1.ResourceName]int64{
		v1.ResourceName(extendedResourceName):       1,
		v1.ResourceName(extendedResourceName + "1"): 2,
	}
	resInit := map[v1.ResourceName]int64{
		v1.ResourceName(extendedResourceName):          1,
		v1.ResourceName(extendedResourceName + "init"): 2,
	}
	resInitImplicit := map[v1.ResourceName]int64{
		v1.ResourceName(extendedResourceName):                                1,
		v1.ResourceName(extendedResourceName + "init"):                       2,
		v1.ResourceName(resourceapi.ResourceDeviceClassPrefix + "classInit"): 2,
	}
	devMap := map[v1.ResourceName]*resourceapi.DeviceClass{
		v1.ResourceName(extendedResourceName): {
			ObjectMeta: metav1.ObjectMeta{
				Name: "class",
			},
		},
		v1.ResourceName(extendedResourceName + "1"): {
			ObjectMeta: metav1.ObjectMeta{
				Name: "class1",
			},
		},
	}
	devMapInit := map[v1.ResourceName]*resourceapi.DeviceClass{
		v1.ResourceName(extendedResourceName): {
			ObjectMeta: metav1.ObjectMeta{
				Name: "class",
			},
		},
		v1.ResourceName(extendedResourceName + "init"): {
			ObjectMeta: metav1.ObjectMeta{
				Name: "classInit",
			},
		},
		v1.ResourceName(resourceapi.ResourceDeviceClassPrefix + "classInit"): {
			ObjectMeta: metav1.ObjectMeta{
				Name: "classInit",
			},
		},
	}
	cer := v1.ContainerExtendedResourceRequest{
		ContainerName: "con0",
		ResourceName:  extendedResourceName,
		RequestName:   "container-0-request-0",
	}
	cer1 := v1.ContainerExtendedResourceRequest{
		ContainerName: "con0",
		ResourceName:  extendedResourceName + "1",
		RequestName:   "container-0-request-1",
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
	cer4 := v1.ContainerExtendedResourceRequest{
		ContainerName: "con0",
		ResourceName:  extendedResourceName,
		RequestName:   "container-2-request-0",
	}
	cer5 := v1.ContainerExtendedResourceRequest{
		ContainerName: "con0",
		ResourceName:  extendedResourceName,
		RequestName:   "container-3-request-0",
	}
	cerInit := v1.ContainerExtendedResourceRequest{
		ContainerName: "init-con0",
		ResourceName:  extendedResourceName + "init",
		RequestName:   "container-1-request-0",
	}
	cerInit0 := v1.ContainerExtendedResourceRequest{
		ContainerName: "init-con0",
		ResourceName:  extendedResourceName + "init",
		RequestName:   "container-0-request-0",
	}
	cerInit1 := v1.ContainerExtendedResourceRequest{
		ContainerName: "init-con0",
		ResourceName:  extendedResourceName + "init",
		RequestName:   "container-1-request-0",
	}
	cerInit2 := v1.ContainerExtendedResourceRequest{
		ContainerName: "init-con1",
		ResourceName:  extendedResourceName + "init",
		RequestName:   "container-1-request-0",
	}
	cerInit3 := v1.ContainerExtendedResourceRequest{
		ContainerName: "init-con2",
		ResourceName:  extendedResourceName + "init",
		RequestName:   "container-2-request-0",
	}
	cerSidecar := v1.ContainerExtendedResourceRequest{
		ContainerName: "sidecar-con1",
		ResourceName:  extendedResourceName + "init",
		RequestName:   "container-1-request-0",
	}

	cerInitImplicit := v1.ContainerExtendedResourceRequest{
		ContainerName: "init-con0",
		ResourceName:  extendedResourceName + "init",
		RequestName:   "container-0-request-0",
	}
	cerInit4Implicit := v1.ContainerExtendedResourceRequest{
		ContainerName: "init-con0",
		ResourceName:  extendedResourceName + "init",
		RequestName:   "container-0-request-1",
	}
	cerInit2Implicit := v1.ContainerExtendedResourceRequest{
		ContainerName: "init-con1",
		ResourceName:  resourceapi.ResourceDeviceClassPrefix + "classInit",
		RequestName:   "container-1-request-0",
	}
	cerInit3Implicit := v1.ContainerExtendedResourceRequest{
		ContainerName: "init-con0",
		ResourceName:  resourceapi.ResourceDeviceClassPrefix + "classInit",
		RequestName:   "container-0-request-0",
	}

	testcases := map[string]struct {
		pod                *v1.Pod
		extnededResources  map[v1.ResourceName]int64
		deviceClassMapping fwk.DeviceClassResolver
		wantReqMappings    []v1.ContainerExtendedResourceRequest
	}{
		"one container, two requests": {
			pod:                pod1,
			extnededResources:  res,
			deviceClassMapping: &mockDeviceClassResolver{devMap},
			wantReqMappings:    []v1.ContainerExtendedResourceRequest{cer, cer1},
		},
		"one container, one explicit and one implicit request": {
			pod:                pod1InitImplicit,
			extnededResources:  resInitImplicit,
			deviceClassMapping: &mockDeviceClassResolver{devMapInit},
			wantReqMappings:    []v1.ContainerExtendedResourceRequest{cerInit3Implicit, cerInit4Implicit},
		},
		"two containers, two requests": {
			pod:                pod2,
			extnededResources:  res,
			deviceClassMapping: &mockDeviceClassResolver{devMap},
			wantReqMappings:    []v1.ContainerExtendedResourceRequest{cer, cer2},
		},
		"one init container, one regular container, one request": {
			pod:                podInit,
			extnededResources:  resInit,
			deviceClassMapping: &mockDeviceClassResolver{devMapInit},
			wantReqMappings:    []v1.ContainerExtendedResourceRequest{cerInit0, cer3},
		},
		"three containers (two are init container), two requests": {
			pod:                podInit2,
			extnededResources:  resInit,
			deviceClassMapping: &mockDeviceClassResolver{devMapInit},
			wantReqMappings:    []v1.ContainerExtendedResourceRequest{cerInit, cerInit2, cer4},
		},
		"three containers (two are init container), both explicit and implicit resources": {
			pod:                podInitImplicit,
			extnededResources:  resInitImplicit,
			deviceClassMapping: &mockDeviceClassResolver{devMapInit},
			wantReqMappings:    []v1.ContainerExtendedResourceRequest{cerInitImplicit, cerInit2Implicit, cer4},
		},
		"four containers (two are init container, one sidecar), three requests": {
			pod:                podInit3,
			extnededResources:  resInit,
			deviceClassMapping: &mockDeviceClassResolver{devMapInit},
			wantReqMappings:    []v1.ContainerExtendedResourceRequest{cerInit1, cerSidecar, cerInit3, cer5},
		},
	}

	for name, tc := range testcases {
		t.Run(name, func(t *testing.T) {
			logger, _ := ktesting.NewTestContext(t)
			_, gotReqMappings := createRequestsAndMappings(tc.pod, tc.extnededResources, logger, tc.deviceClassMapping)
			if len(tc.wantReqMappings) != len(gotReqMappings) {
				t.Fatalf("different length, want %#v, got %#v", tc.wantReqMappings, gotReqMappings)
			}
			sort.Slice(gotReqMappings, func(i, j int) bool {
				if gotReqMappings[i].RequestName < gotReqMappings[j].RequestName {
					return true
				}
				if gotReqMappings[i].RequestName > gotReqMappings[j].RequestName {
					return false
				}
				return gotReqMappings[i].ContainerName < gotReqMappings[j].ContainerName
			})
			for i, r := range tc.wantReqMappings {
				if r.RequestName != gotReqMappings[i].RequestName {
					t.Errorf("different request name, want %#v, got %#v", r, gotReqMappings[i])
				}
				if r.ContainerName != gotReqMappings[i].ContainerName {
					t.Errorf("different container name, want %#v, got %#v", r, gotReqMappings[i])
				}
				if r.ResourceName != gotReqMappings[i].ResourceName {
					t.Errorf("different resource name, want %#v, got %#v", r, gotReqMappings[i])
				}
			}
		})
	}
}

// Test_createRequestsAndMappings_deterministic verifies that the function produces
// consistent, deterministic output regardless of Go's map iteration order.
// This is critical for reproducible pod status and reliable testing.
func Test_createRequestsAndMappings_deterministic(t *testing.T) {
	logger, _ := ktesting.NewTestContext(t)

	// Create a pod with multiple extended resources
	// Use resource names that would be in different orders if not sorted
	pod := st.MakePod().Name(podName).Namespace(namespace).
		UID(podUID).
		Res(map[v1.ResourceName]string{
			"zzz.example.com/resource": "1", // Would be last alphabetically
			"aaa.example.com/resource": "1", // Would be first alphabetically  
			"mmm.example.com/resource": "1", // Would be middle alphabetically
		}).
		Obj()

	// Map of extended resources (maps have non-deterministic iteration order)
	extendedResources := map[v1.ResourceName]int64{
		"zzz.example.com/resource": 1,
		"aaa.example.com/resource": 1,
		"mmm.example.com/resource": 1,
	}

	// Mock device class resolver with DeviceClass objects
	cache := &mockDeviceClassResolver{
		mapping: map[v1.ResourceName]*resourceapi.DeviceClass{
			"zzz.example.com/resource": {ObjectMeta: metav1.ObjectMeta{Name: "zzz-class"}},
			"aaa.example.com/resource": {ObjectMeta: metav1.ObjectMeta{Name: "aaa-class"}},
			"mmm.example.com/resource": {ObjectMeta: metav1.ObjectMeta{Name: "mmm-class"}},
		},
	}

	// Run the function multiple times
	const iterations = 10
	var firstDeviceRequests []resourceapi.DeviceRequest
	var firstMappings []v1.ContainerExtendedResourceRequest

	for i := 0; i < iterations; i++ {
		deviceRequests, mappings := createRequestsAndMappings(pod, extendedResources, logger, cache)

		if i == 0 {
			// Save first result
			firstDeviceRequests = deviceRequests
			firstMappings = mappings
		} else {
			// Verify all subsequent results match the first
			if len(deviceRequests) != len(firstDeviceRequests) {
				t.Fatalf("iteration %d: different device request count: got %d, want %d",
					i, len(deviceRequests), len(firstDeviceRequests))
			}

			for j, req := range deviceRequests {
				if req.Name != firstDeviceRequests[j].Name {
					t.Errorf("iteration %d: device request %d has different name: got %q, want %q",
						i, j, req.Name, firstDeviceRequests[j].Name)
				}
				if req.Exactly.DeviceClassName != firstDeviceRequests[j].Exactly.DeviceClassName {
					t.Errorf("iteration %d: device request %d has different class: got %q, want %q",
						i, j, req.Exactly.DeviceClassName, firstDeviceRequests[j].Exactly.DeviceClassName)
				}
			}

			if len(mappings) != len(firstMappings) {
				t.Fatalf("iteration %d: different mappings count: got %d, want %d",
					i, len(mappings), len(firstMappings))
			}

			for j, mapping := range mappings {
				if mapping.ResourceName != firstMappings[j].ResourceName {
					t.Errorf("iteration %d: mapping %d has different resource name: got %q, want %q",
						i, j, mapping.ResourceName, firstMappings[j].ResourceName)
				}
			}
		}
	}

	// Verify the output is in sorted order (alphabetically by resource name)
	// The device requests should be sorted by their names which are derived from sorted resource names
	expectedOrder := []string{
		"aaa.example.com/resource", // First alphabetically
		"mmm.example.com/resource", // Middle alphabetically
		"zzz.example.com/resource", // Last alphabetically
	}

	// Verify mappings contain resources in sorted order
	seenResources := make(map[string]bool)
	var resourceOrder []string
	for _, mapping := range firstMappings {
		if !seenResources[mapping.ResourceName] {
			seenResources[mapping.ResourceName] = true
			resourceOrder = append(resourceOrder, mapping.ResourceName)
		}
	}

	for i, resource := range expectedOrder {
		if i >= len(resourceOrder) {
			t.Errorf("missing resource in output: %s", resource)
			continue
		}
		if resourceOrder[i] != resource {
			t.Errorf("resource at position %d: got %q, want %q", i, resourceOrder[i], resource)
		}
	}

	t.Logf("✓ Function produced consistent output across %d iterations", iterations)
	t.Logf("✓ Resources processed in sorted order: %v", resourceOrder)
}
