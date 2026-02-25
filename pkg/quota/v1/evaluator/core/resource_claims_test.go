/*
Copyright 2023 The Kubernetes Authors.

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

package core

import (
	"context"
	"strings"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	corev1 "k8s.io/api/core/v1"
	resourceapi "k8s.io/api/resource/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apiserver/pkg/admission"
	quota "k8s.io/apiserver/pkg/quota/v1"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/client-go/informers"
	"k8s.io/client-go/kubernetes/fake"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/dynamic-resource-allocation/deviceclass/extendedresourcecache"
	"k8s.io/klog/v2/ktesting"
	api "k8s.io/kubernetes/pkg/apis/resource"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/utils/ptr"
)

func testResourceClaim(name string, namespace string, isExtended bool, podName string, spec api.ResourceClaimSpec) *api.ResourceClaim {
	claim := &api.ResourceClaim{
		ObjectMeta: metav1.ObjectMeta{Name: name, Namespace: namespace},
		Spec:       spec,
	}
	if isExtended {
		claim.Annotations = map[string]string{resourceapi.ExtendedResourceClaimAnnotation: "true"}
		claim.OwnerReferences = []metav1.OwnerReference{
			{
				APIVersion: "v1",
				Kind:       "Pod",
				Name:       podName,
				UID:        "uid",
				Controller: ptr.To(true),
			},
		}
	}
	return claim
}

func TestResourceClaimEvaluatorUsage(t *testing.T) {
	classGpu := "gpu"
	classTpu := "tpu"
	validClaim := testResourceClaim("foo", "ns", false, "", api.ResourceClaimSpec{
		Devices: api.DeviceClaim{
			Requests: []api.DeviceRequest{
				{
					Name: "req-0",
					Exactly: &api.ExactDeviceRequest{
						DeviceClassName: classGpu,
						AllocationMode:  api.DeviceAllocationModeExactCount,
						Count:           1,
					},
				},
			},
		},
	})
	validClaimWithPrioritizedList := testResourceClaim("foo", "ns", false, "", api.ResourceClaimSpec{
		Devices: api.DeviceClaim{
			Requests: []api.DeviceRequest{
				{
					Name: "req-0",
					FirstAvailable: []api.DeviceSubRequest{
						{
							Name:            "subreq-0",
							DeviceClassName: classGpu,
							AllocationMode:  api.DeviceAllocationModeExactCount,
							Count:           1,
						},
					},
				},
			},
		},
	})
	explicitExtendedResourceClaim := testResourceClaim("foo", "ns", true, "pod-explicit", api.ResourceClaimSpec{
		Devices: api.DeviceClaim{
			Requests: []api.DeviceRequest{
				{
					Name: "container-0-request-0",
					Exactly: &api.ExactDeviceRequest{
						DeviceClassName: classGpu,
						AllocationMode:  api.DeviceAllocationModeExactCount,
						Count:           1,
					},
				},
			},
		},
	})
	implicitExtendedResourceClaim := testResourceClaim("foo", "ns", true, "pod-implicit", api.ResourceClaimSpec{
		Devices: api.DeviceClaim{
			Requests: []api.DeviceRequest{
				{
					Name: "container-0-request-0",
					Exactly: &api.ExactDeviceRequest{
						DeviceClassName: classGpu,
						AllocationMode:  api.DeviceAllocationModeExactCount,
						Count:           1,
					},
				},
			},
		},
	})
	hybridExtendedResourceClaim := testResourceClaim("foo", "ns", true, "pod-hybrid", api.ResourceClaimSpec{
		Devices: api.DeviceClaim{
			Requests: []api.DeviceRequest{
				{
					Name: "container-1-request-0",
					Exactly: &api.ExactDeviceRequest{
						DeviceClassName: classGpu,
						AllocationMode:  api.DeviceAllocationModeExactCount,
						Count:           1,
					},
				},
				{
					Name: "container-1-request-1",
					Exactly: &api.ExactDeviceRequest{
						DeviceClassName: classGpu,
						AllocationMode:  api.DeviceAllocationModeExactCount,
						Count:           1,
					},
				},
			},
		},
	})
	nilStatusExtendedResourceClaim := testResourceClaim("foo", "ns", true, "pod-nil-status", api.ResourceClaimSpec{
		Devices: api.DeviceClaim{
			Requests: []api.DeviceRequest{
				{
					Name: "request-0",
					Exactly: &api.ExactDeviceRequest{
						DeviceClassName: classGpu,
						AllocationMode:  api.DeviceAllocationModeExactCount,
						Count:           1,
					},
				},
				{
					Name: "container-1-request-0",
					Exactly: &api.ExactDeviceRequest{
						DeviceClassName: classGpu,
						AllocationMode:  api.DeviceAllocationModeExactCount,
						Count:           1,
					},
				},
			},
		},
	})
	nilStatusExtendedResourceClaimExternal, err := toExternalResourceClaimOrError(nilStatusExtendedResourceClaim)
	if err != nil {
		t.Fatal(err)
	}
	nilStatusExtendedResourceClaim.Name = "foo-copy"
	initExtendedResourceClaim := testResourceClaim("foo", "ns", true, "pod-init", api.ResourceClaimSpec{
		Devices: api.DeviceClaim{
			Requests: []api.DeviceRequest{
				{
					Name: "container-2-request-0",
					Exactly: &api.ExactDeviceRequest{
						DeviceClassName: classGpu,
						AllocationMode:  api.DeviceAllocationModeExactCount,
						Count:           1,
					},
				},
				{
					Name: "container-2-request-1",
					Exactly: &api.ExactDeviceRequest{
						DeviceClassName: classGpu,
						AllocationMode:  api.DeviceAllocationModeExactCount,
						Count:           1,
					},
				},
			},
		},
	})

	podImplicit := &corev1.Pod{
		TypeMeta: metav1.TypeMeta{
			APIVersion: "v1",
		},
		ObjectMeta: metav1.ObjectMeta{
			Namespace: "ns",
			Name:      "pod-implicit",
			UID:       "uid",
		},
		Spec: corev1.PodSpec{
			Containers: []corev1.Container{
				{
					Resources: corev1.ResourceRequirements{
						Requests: corev1.ResourceList{
							corev1.ResourceName("deviceclass.resource.kubernetes.io/" + classGpu): resource.MustParse("1"),
						},
					},
				},
			},
		},
		Status: corev1.PodStatus{
			ExtendedResourceClaimStatus: &corev1.PodExtendedResourceClaimStatus{
				ResourceClaimName: "foo",
			},
		},
	}
	podExplicit := &corev1.Pod{
		TypeMeta: metav1.TypeMeta{
			APIVersion: "v1",
		},
		ObjectMeta: metav1.ObjectMeta{
			Namespace: "ns",
			Name:      "pod-explicit",
			UID:       "uid",
		},
		Spec: corev1.PodSpec{
			InitContainers: []corev1.Container{
				{
					Resources: corev1.ResourceRequirements{
						Requests: corev1.ResourceList{
							"example.com/gpu": resource.MustParse("1"),
						},
					},
				},
			},
		},
		Status: corev1.PodStatus{
			ExtendedResourceClaimStatus: &corev1.PodExtendedResourceClaimStatus{
				ResourceClaimName: "foo",
			},
		},
	}
	podHybrid := &corev1.Pod{
		TypeMeta: metav1.TypeMeta{
			APIVersion: "v1",
		},
		ObjectMeta: metav1.ObjectMeta{
			Namespace: "ns",
			Name:      "pod-hybrid",
			UID:       "uid",
		},
		Spec: corev1.PodSpec{
			InitContainers: []corev1.Container{
				{
					Resources: corev1.ResourceRequirements{
						Requests: corev1.ResourceList{
							"example.com/gpu": resource.MustParse("1"),
							corev1.ResourceName("deviceclass.resource.kubernetes.io/" + classGpu): resource.MustParse("1"),
						},
					},
				},
			},
			Containers: []corev1.Container{
				{
					Resources: corev1.ResourceRequirements{
						Requests: corev1.ResourceList{
							"example.com/gpu": resource.MustParse("1"),
							corev1.ResourceName("deviceclass.resource.kubernetes.io/" + classGpu): resource.MustParse("1"),
						},
					},
				},
			},
		},
		Status: corev1.PodStatus{
			ExtendedResourceClaimStatus: &corev1.PodExtendedResourceClaimStatus{
				ResourceClaimName: "foo",
			},
		},
	}
	podNilStatus := &corev1.Pod{
		TypeMeta: metav1.TypeMeta{
			APIVersion: "v1",
		},
		ObjectMeta: metav1.ObjectMeta{
			Namespace: "ns",
			Name:      "pod-nil-status",
			UID:       "uid",
		},
		Spec: corev1.PodSpec{
			InitContainers: []corev1.Container{
				{
					Resources: corev1.ResourceRequirements{
						Requests: corev1.ResourceList{
							"example.com/gpu": resource.MustParse("1"),
						},
					},
				},
			},
			Containers: []corev1.Container{
				{
					Resources: corev1.ResourceRequirements{
						Requests: corev1.ResourceList{
							corev1.ResourceName("deviceclass.resource.kubernetes.io/" + classGpu): resource.MustParse("1"),
						},
					},
				},
			},
		},
		Status: corev1.PodStatus{},
	}
	podInit := &corev1.Pod{
		TypeMeta: metav1.TypeMeta{
			APIVersion: "v1",
		},
		ObjectMeta: metav1.ObjectMeta{
			Namespace: "ns",
			Name:      "pod-init",
			UID:       "uid",
		},
		Spec: corev1.PodSpec{
			InitContainers: []corev1.Container{
				{
					Name: "init-container-1",
					Resources: corev1.ResourceRequirements{
						Requests: corev1.ResourceList{
							"example.com/gpu": resource.MustParse("1"),
							corev1.ResourceName("deviceclass.resource.kubernetes.io/" + classGpu): resource.MustParse("1"),
						},
					},
				},
				{
					Name: "init-container-2",
					Resources: corev1.ResourceRequirements{
						Requests: corev1.ResourceList{
							"example.com/gpu": resource.MustParse("1"),
							corev1.ResourceName("deviceclass.resource.kubernetes.io/" + classGpu): resource.MustParse("1"),
						},
					},
				},
			},
			Containers: []corev1.Container{
				{
					Name: "container-1",
					Resources: corev1.ResourceRequirements{
						Requests: corev1.ResourceList{
							"example.com/gpu": resource.MustParse("1"),
							corev1.ResourceName("deviceclass.resource.kubernetes.io/" + classGpu): resource.MustParse("1"),
						},
					},
				},
			},
		},
		Status: corev1.PodStatus{
			ExtendedResourceClaimStatus: &corev1.PodExtendedResourceClaimStatus{
				ResourceClaimName: "foo",
			},
		},
	}

	deviceClass1 := &resourceapi.DeviceClass{
		ObjectMeta: metav1.ObjectMeta{
			Name: classGpu,
		},
		Spec: resourceapi.DeviceClassSpec{
			ExtendedResourceName: ptr.To("example.com/gpu"),
		},
	}

	logger, ctx := ktesting.NewTestContext(t)
	tCtx, tCancel := context.WithCancel(ctx)
	client := fake.NewClientset(deviceClass1, podImplicit, podExplicit, podHybrid, podNilStatus, podInit)
	informerFactory := informers.NewSharedInformerFactory(client, 0)
	deviceclassmapping := extendedresourcecache.NewExtendedResourceCache(logger)
	if _, err := informerFactory.Resource().V1().DeviceClasses().Informer().AddEventHandler(deviceclassmapping); err != nil {
		t.Fatal(err)
	}
	var otherOwnedClaims []*resourceapi.ResourceClaim
	claimGetter := func(namespace string, podUID types.UID) ([]*resourceapi.ResourceClaim, error) {
		return otherOwnedClaims, nil
	}
	evaluatorWithDeviceMapping := NewResourceClaimEvaluator(nil, deviceclassmapping, informerFactory.Core().V1().Pods().Lister(), claimGetter)

	informerFactory.Start(tCtx.Done())
	t.Cleanup(func() {
		// Need to cancel before waiting for the shutdown.
		tCancel()
		// Now we can wait for all goroutines to stop.
		informerFactory.Shutdown()
	})
	informerFactory.WaitForCacheSync(tCtx.Done())

	// wait for informer sync
	time.Sleep(1 * time.Second)

	evaluator := NewResourceClaimEvaluator(nil, nil, nil, claimGetter)
	testCases := map[string]struct {
		evaluator   quota.Evaluator
		claim       *api.ResourceClaim
		otherClaims []*resourceapi.ResourceClaim
		usage       corev1.ResourceList
		errMsg      string
	}{
		"implicit-extended-resource-claim": {
			evaluator: evaluatorWithDeviceMapping,
			claim:     implicitExtendedResourceClaim,
			usage: corev1.ResourceList{
				"count/resourceclaims.resource.k8s.io":    resource.MustParse("1"),
				"gpu.deviceclass.resource.k8s.io/devices": resource.MustParse("1"),
				"requests.example.com/gpu":                resource.MustParse("1"),
			},
		},
		"explicit-extended-resource-claim": {
			evaluator: evaluatorWithDeviceMapping,
			claim:     explicitExtendedResourceClaim,
			usage: corev1.ResourceList{
				"count/resourceclaims.resource.k8s.io":            resource.MustParse("1"),
				"gpu.deviceclass.resource.k8s.io/devices":         resource.MustParse("1"),
				"requests.deviceclass.resource.kubernetes.io/gpu": resource.MustParse("1"),
			},
		},
		"hybrid-extended-resource-claim": {
			evaluator: evaluatorWithDeviceMapping,
			claim:     hybridExtendedResourceClaim,
			usage: corev1.ResourceList{
				"count/resourceclaims.resource.k8s.io":            resource.MustParse("1"),
				"gpu.deviceclass.resource.k8s.io/devices":         resource.MustParse("2"),
				"requests.deviceclass.resource.kubernetes.io/gpu": resource.MustParse("1"),
				"requests.example.com/gpu":                        resource.MustParse("1"),
			},
		},
		"nil-status-extended-resource-claim": {
			evaluator: evaluatorWithDeviceMapping,
			claim:     nilStatusExtendedResourceClaim,
			usage: corev1.ResourceList{
				"count/resourceclaims.resource.k8s.io":            resource.MustParse("1"),
				"gpu.deviceclass.resource.k8s.io/devices":         resource.MustParse("2"),
				"requests.deviceclass.resource.kubernetes.io/gpu": resource.MustParse("1"),
				"requests.example.com/gpu":                        resource.MustParse("1"),
			},
		},
		// both claims set the same pod as owner, the second claim cannot get the usage
		// subtraction from pod requests.
		"nil-status-two-extended-resource-claims": {
			evaluator: evaluatorWithDeviceMapping,
			claim:     nilStatusExtendedResourceClaim,
			otherClaims: []*resourceapi.ResourceClaim{
				nilStatusExtendedResourceClaimExternal,
			},
			usage: corev1.ResourceList{
				"count/resourceclaims.resource.k8s.io":            resource.MustParse("1"),
				"gpu.deviceclass.resource.k8s.io/devices":         resource.MustParse("2"),
				"requests.deviceclass.resource.kubernetes.io/gpu": resource.MustParse("2"),
				"requests.example.com/gpu":                        resource.MustParse("2"),
			},
		},
		"init-extended-resource-claim": {
			evaluator: evaluatorWithDeviceMapping,
			claim:     initExtendedResourceClaim,
			usage: corev1.ResourceList{
				"count/resourceclaims.resource.k8s.io":            resource.MustParse("1"),
				"gpu.deviceclass.resource.k8s.io/devices":         resource.MustParse("2"),
				"requests.deviceclass.resource.kubernetes.io/gpu": resource.MustParse("1"),
				"requests.example.com/gpu":                        resource.MustParse("1"),
			},
		},
		"simple": {
			claim: validClaim,
			usage: corev1.ResourceList{
				"count/resourceclaims.resource.k8s.io":            resource.MustParse("1"),
				"gpu.deviceclass.resource.k8s.io/devices":         resource.MustParse("1"),
				"requests.deviceclass.resource.kubernetes.io/gpu": resource.MustParse("1"),
			},
		},
		"many-requests": {
			claim: func() *api.ResourceClaim {
				claim := validClaim.DeepCopy()
				for i := 0; i < 4; i++ {
					claim.Spec.Devices.Requests = append(claim.Spec.Devices.Requests, claim.Spec.Devices.Requests[0])
				}
				return claim
			}(),
			usage: corev1.ResourceList{
				"count/resourceclaims.resource.k8s.io":            resource.MustParse("1"),
				"gpu.deviceclass.resource.k8s.io/devices":         resource.MustParse("5"),
				"requests.deviceclass.resource.kubernetes.io/gpu": resource.MustParse("5"),
			},
		},
		"count": {
			claim: func() *api.ResourceClaim {
				claim := validClaim.DeepCopy()
				claim.Spec.Devices.Requests[0].Exactly.Count = 5
				return claim
			}(),
			usage: corev1.ResourceList{
				"count/resourceclaims.resource.k8s.io":            resource.MustParse("1"),
				"gpu.deviceclass.resource.k8s.io/devices":         resource.MustParse("5"),
				"requests.deviceclass.resource.kubernetes.io/gpu": resource.MustParse("5"),
			},
		},
		"all": {
			claim: func() *api.ResourceClaim {
				claim := validClaim.DeepCopy()
				claim.Spec.Devices.Requests[0].Exactly.AllocationMode = api.DeviceAllocationModeAll
				return claim
			}(),
			usage: corev1.ResourceList{
				"count/resourceclaims.resource.k8s.io":            resource.MustParse("1"),
				"gpu.deviceclass.resource.k8s.io/devices":         *resource.NewQuantity(api.AllocationResultsMaxSize, resource.DecimalSI),
				"requests.deviceclass.resource.kubernetes.io/gpu": *resource.NewQuantity(api.AllocationResultsMaxSize, resource.DecimalSI),
			},
		},
		"unknown-count-mode": {
			claim: func() *api.ResourceClaim {
				claim := validClaim.DeepCopy()
				claim.Spec.Devices.Requests[0].Exactly.AllocationMode = "future-mode"
				return claim
			}(),
			usage: corev1.ResourceList{
				"count/resourceclaims.resource.k8s.io":    resource.MustParse("1"),
				"gpu.deviceclass.resource.k8s.io/devices": resource.MustParse("0"),
			},
		},
		"admin": {
			claim: func() *api.ResourceClaim {
				claim := validClaim.DeepCopy()
				// Admins are *not* exempt from quota.
				claim.Spec.Devices.Requests[0].Exactly.AdminAccess = ptr.To(true)
				return claim
			}(),
			usage: corev1.ResourceList{
				"count/resourceclaims.resource.k8s.io":            resource.MustParse("1"),
				"gpu.deviceclass.resource.k8s.io/devices":         resource.MustParse("1"),
				"requests.deviceclass.resource.kubernetes.io/gpu": resource.MustParse("1"),
			},
		},
		"prioritized-list": {
			claim: validClaimWithPrioritizedList,
			usage: corev1.ResourceList{
				"count/resourceclaims.resource.k8s.io":            resource.MustParse("1"),
				"gpu.deviceclass.resource.k8s.io/devices":         resource.MustParse("1"),
				"requests.deviceclass.resource.kubernetes.io/gpu": resource.MustParse("1"),
			},
		},
		"prioritized-list-multiple-subrequests": {
			claim: func() *api.ResourceClaim {
				claim := validClaimWithPrioritizedList.DeepCopy()
				claim.Spec.Devices.Requests[0].FirstAvailable[0].Count = 2
				claim.Spec.Devices.Requests[0].FirstAvailable = append(claim.Spec.Devices.Requests[0].FirstAvailable, api.DeviceSubRequest{
					Name:            "subreq-1",
					DeviceClassName: classGpu,
					AllocationMode:  api.DeviceAllocationModeExactCount,
					Count:           1,
				})
				return claim
			}(),
			usage: corev1.ResourceList{
				"count/resourceclaims.resource.k8s.io":            resource.MustParse("1"),
				"gpu.deviceclass.resource.k8s.io/devices":         resource.MustParse("2"),
				"requests.deviceclass.resource.kubernetes.io/gpu": resource.MustParse("2"),
			},
		},
		"prioritized-list-multiple-subrequests-allocation-mode-all": {
			claim: func() *api.ResourceClaim {
				claim := validClaimWithPrioritizedList.DeepCopy()
				claim.Spec.Devices.Requests[0].FirstAvailable = append(claim.Spec.Devices.Requests[0].FirstAvailable, api.DeviceSubRequest{
					Name:            "subreq-1",
					DeviceClassName: classGpu,
					AllocationMode:  api.DeviceAllocationModeAll,
				})
				return claim
			}(),
			usage: corev1.ResourceList{
				"count/resourceclaims.resource.k8s.io":            resource.MustParse("1"),
				"gpu.deviceclass.resource.k8s.io/devices":         resource.MustParse("32"),
				"requests.deviceclass.resource.kubernetes.io/gpu": resource.MustParse("32"),
			},
		},
		"prioritized-list-multiple-subrequests-different-device-classes": {
			claim: func() *api.ResourceClaim {
				claim := validClaimWithPrioritizedList.DeepCopy()
				claim.Spec.Devices.Requests[0].FirstAvailable = append(claim.Spec.Devices.Requests[0].FirstAvailable, api.DeviceSubRequest{
					Name:            "subreq-1",
					DeviceClassName: classTpu,
					AllocationMode:  api.DeviceAllocationModeAll,
				})
				return claim
			}(),
			usage: corev1.ResourceList{
				"count/resourceclaims.resource.k8s.io":            resource.MustParse("1"),
				"gpu.deviceclass.resource.k8s.io/devices":         resource.MustParse("1"),
				"requests.deviceclass.resource.kubernetes.io/gpu": resource.MustParse("1"),
				"tpu.deviceclass.resource.k8s.io/devices":         resource.MustParse("32"),
				"requests.deviceclass.resource.kubernetes.io/tpu": resource.MustParse("32"),
			},
		},
	}
	for testName, testCase := range testCases {
		t.Run(testName, func(t *testing.T) {
			otherOwnedClaims = testCase.otherClaims
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.DRAExtendedResource, true)
			if testCase.evaluator == nil {
				testCase.evaluator = evaluator
			}
			actual, err := testCase.evaluator.Usage(testCase.claim)
			if err != nil {
				if testCase.errMsg == "" {
					t.Fatalf("Unexpected error: %v", err)
				}
				if !strings.Contains(err.Error(), testCase.errMsg) {
					t.Fatalf("Expected error %q, got error: %v", testCase.errMsg, err.Error())
				}
			}
			if err == nil && testCase.errMsg != "" {
				t.Fatalf("Expected error %q, got none", testCase.errMsg)
			}
			if diff := cmp.Diff(testCase.usage, actual); diff != "" {
				t.Errorf("Unexpected usage (-want, +got):\n%s", diff)
			}
		})

	}
}

func TestResourceClaimEvaluatorMatchingResources(t *testing.T) {
	deviceClass1 := &resourceapi.DeviceClass{
		ObjectMeta: metav1.ObjectMeta{
			Name: "gpu",
		},
		Spec: resourceapi.DeviceClassSpec{
			ExtendedResourceName: ptr.To("example.com/gpu"),
		},
	}

	logger, ctx := ktesting.NewTestContext(t)
	tCtx, tCancel := context.WithCancel(ctx)
	client := fake.NewClientset(deviceClass1)
	informerFactory := informers.NewSharedInformerFactory(client, 0)
	deviceclassmapping := extendedresourcecache.NewExtendedResourceCache(logger)
	if _, err := informerFactory.Resource().V1().DeviceClasses().Informer().AddEventHandler(deviceclassmapping); err != nil {
		logger.Error(err, "failed to add device class informer event handler")
	}
	evaluator := NewResourceClaimEvaluator(nil, deviceclassmapping, informerFactory.Core().V1().Pods().Lister(), nil)

	informerFactory.Start(tCtx.Done())
	t.Cleanup(func() {
		// Need to cancel before waiting for the shutdown.
		tCancel()
		// Now we can wait for all goroutines to stop.
		informerFactory.Shutdown()
	})
	informerFactory.WaitForCacheSync(tCtx.Done())

	// wait for informer sync
	time.Sleep(1 * time.Second)

	testCases := map[string]struct {
		items []corev1.ResourceName
		want  []corev1.ResourceName
	}{
		"supported-resources": {
			items: []corev1.ResourceName{
				"count/resourceclaims.resource.k8s.io",
				"gpu.deviceclass.resource.k8s.io/devices",
				"requests.example.com/gpu",
				"requests.deviceclass.resource.kubernetes.io/gpu",
			},

			want: []corev1.ResourceName{
				"count/resourceclaims.resource.k8s.io",
				"gpu.deviceclass.resource.k8s.io/devices",
				"requests.example.com/gpu",
				"requests.deviceclass.resource.kubernetes.io/gpu",
			},
		},
		"unsupported-resources": {
			items: []corev1.ResourceName{
				"resourceclaims", // no such alias
				"storage",
				"ephemeral-storage",
				"bronze.deviceclass.resource.k8s.io/storage",
				"gpu.storage.k8s.io/requests.storage",
			},
			want: []corev1.ResourceName{},
		},
	}
	for testName, testCase := range testCases {
		t.Run(testName, func(t *testing.T) {
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.DRAExtendedResource, true)
			actual := evaluator.MatchingResources(testCase.items)

			if diff := cmp.Diff(testCase.want, actual); diff != "" {
				t.Errorf("Unexpected response (-want, +got):\n%s", diff)
			}
		})
	}
}

func TestResourceClaimEvaluatorHandles(t *testing.T) {
	evaluator := NewResourceClaimEvaluator(nil, nil, nil, nil)
	testCases := []struct {
		name  string
		attrs admission.Attributes
		want  bool
	}{
		{
			name:  "create",
			attrs: admission.NewAttributesRecord(nil, nil, schema.GroupVersionKind{Group: "core", Version: "v1", Kind: "Pod"}, "", "", schema.GroupVersionResource{Group: "core", Version: "v1", Resource: "pods"}, "", admission.Create, nil, false, nil),
			want:  true,
		},
		{
			name:  "update",
			attrs: admission.NewAttributesRecord(nil, nil, schema.GroupVersionKind{Group: "core", Version: "v1", Kind: "Pod"}, "", "", schema.GroupVersionResource{Group: "core", Version: "v1", Resource: "pods"}, "", admission.Update, nil, false, nil),
			want:  true,
		},
		{
			name:  "delete",
			attrs: admission.NewAttributesRecord(nil, nil, schema.GroupVersionKind{Group: "core", Version: "v1", Kind: "Pod"}, "", "", schema.GroupVersionResource{Group: "core", Version: "v1", Resource: "pods"}, "", admission.Delete, nil, false, nil),
			want:  false,
		},
		{
			name:  "connect",
			attrs: admission.NewAttributesRecord(nil, nil, schema.GroupVersionKind{Group: "core", Version: "v1", Kind: "Pod"}, "", "", schema.GroupVersionResource{Group: "core", Version: "v1", Resource: "pods"}, "", admission.Connect, nil, false, nil),
			want:  false,
		},
		{
			name:  "create-subresource",
			attrs: admission.NewAttributesRecord(nil, nil, schema.GroupVersionKind{Group: "core", Version: "v1", Kind: "Pod"}, "", "", schema.GroupVersionResource{Group: "core", Version: "v1", Resource: "pods"}, "subresource", admission.Create, nil, false, nil),
			want:  false,
		},
		{
			name:  "update-subresource",
			attrs: admission.NewAttributesRecord(nil, nil, schema.GroupVersionKind{Group: "core", Version: "v1", Kind: "Pod"}, "", "", schema.GroupVersionResource{Group: "core", Version: "v1", Resource: "pods"}, "subresource", admission.Update, nil, false, nil),
			want:  false,
		},
	}
	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			actual := evaluator.Handles(tc.attrs)

			if tc.want != actual {
				t.Errorf("%s expected:\n%v\n, actual:\n%v", tc.name, tc.want, actual)
			}
		})
	}
}
