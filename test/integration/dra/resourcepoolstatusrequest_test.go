/*
Copyright The Kubernetes Authors.

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

package dra

import (
	"fmt"
	"sync"
	"testing"
	"time"

	"github.com/onsi/gomega"

	resourceapi "k8s.io/api/resource/v1"
	resourcev1alpha1 "k8s.io/api/resource/v1alpha1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/client-go/informers"
	"k8s.io/component-base/featuregate"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/klog/v2"
	kubeapiservertesting "k8s.io/kubernetes/cmd/kube-apiserver/app/testing"
	"k8s.io/kubernetes/pkg/controller/resourcepoolstatusrequest"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/test/integration/framework"
	"k8s.io/kubernetes/test/utils/ktesting"
	"k8s.io/utils/ptr"
)

func TestResourcePoolStatusRequest(t *testing.T) {
	for name, tc := range map[string]struct {
		features map[featuregate.Feature]bool
		f        func(tCtx ktesting.TContext)
	}{
		"feature-enabled": {
			features: map[featuregate.Feature]bool{
				features.DynamicResourceAllocation: true,
				features.DRAResourcePoolStatus:     true,
			},
			f: func(tCtx ktesting.TContext) {
				tCtx.Run("ProcessRequest", testProcessResourcePoolStatusRequest)
				tCtx.Run("OneTimeProcessing", testOneTimeProcessing)
				tCtx.Run("LimitTruncation", testLimitTruncation)
				tCtx.Run("FilterByPoolName", testFilterByPoolName)
				tCtx.Run("ValidationErrors", testValidationErrors)
				// Note: RBAC enforcement is standard Kubernetes behavior tested in
				// test/integration/auth/. The test API server uses AlwaysAllow authorizer.
			},
		},
	} {
		t.Run(name, func(t *testing.T) {
			tCtx := ktesting.Init(t)

			// Enable required feature gates
			featuregatetesting.SetFeatureGatesDuringTest(t, utilfeature.DefaultFeatureGate, tc.features)

			etcdOptions := framework.SharedEtcd()
			apiServerOptions := kubeapiservertesting.NewDefaultTestServerOptions()
			apiServerFlags := framework.DefaultTestServerFlags()
			// Enable the v1alpha1 API for ResourcePoolStatusRequest
			apiServerFlags = append(apiServerFlags, "--runtime-config=resource.k8s.io/v1alpha1=true")
			server := kubeapiservertesting.StartTestServerOrDie(t, apiServerOptions, apiServerFlags, etcdOptions)
			tCtx.CleanupCtx(func(tCtx ktesting.TContext) {
				tCtx.Log("Stopping the apiserver...")
				server.TearDownFn()
			})
			tCtx = tCtx.WithRESTConfig(server.ClientConfig)

			tCtx = prepareResourcePoolStatusRequestController(tCtx)

			tc.f(tCtx)
		})
	}
}

// prepareResourcePoolStatusRequestController prepares the context for the controller.
func prepareResourcePoolStatusRequestController(tCtx ktesting.TContext) ktesting.TContext {
	controller := &resourcePoolStatusRequestControllerSingleton{
		rootCtx: tCtx,
	}
	return tCtx.WithValue(resourcePoolStatusRequestControllerKey, controller)
}

type resourcePoolStatusRequestControllerKeyType int

var resourcePoolStatusRequestControllerKey resourcePoolStatusRequestControllerKeyType

type resourcePoolStatusRequestControllerSingleton struct {
	rootCtx ktesting.TContext

	mutex           sync.Mutex
	usageCount      int
	wg              sync.WaitGroup
	informerFactory informers.SharedInformerFactory
	cancel          func(cause string)
}

func (c *resourcePoolStatusRequestControllerSingleton) start(tCtx ktesting.TContext) {
	tCtx.Helper()
	c.mutex.Lock()
	defer c.mutex.Unlock()

	c.usageCount++
	tCtx.CleanupCtx(c.stop)
	if c.usageCount > 1 {
		return
	}

	tCtx = c.rootCtx
	tCtx.Logf("Starting the ResourcePoolStatusRequest controller for test %s...", tCtx.Name())
	tCtx = tCtx.WithLogger(klog.LoggerWithName(tCtx.Logger(), "resourcePoolStatusRequestController"))

	controllerCtx := tCtx.WithCancel()
	c.cancel = controllerCtx.Cancel

	client := controllerCtx.Client()
	c.informerFactory = informers.NewSharedInformerFactory(client, 0)
	controller, err := resourcepoolstatusrequest.NewController(
		controllerCtx,
		client,
		c.informerFactory,
	)
	tCtx.ExpectNoError(err, "create ResourcePoolStatusRequest controller")

	c.informerFactory.Start(controllerCtx.Done())
	c.wg.Go(func() {
		controller.Run(controllerCtx, 1)
	})
	tCtx.Logf("Started the ResourcePoolStatusRequest controller for test %s.", tCtx.Name())
}

func (c *resourcePoolStatusRequestControllerSingleton) stop(tCtx ktesting.TContext) {
	c.mutex.Lock()
	defer c.mutex.Unlock()

	c.usageCount--
	if c.usageCount > 0 {
		return
	}

	c.rootCtx.Logf("Stopping the ResourcePoolStatusRequest controller after test %s...", tCtx.Name())
	if c.cancel != nil {
		c.cancel("test is done")
	}
	if c.informerFactory != nil {
		c.informerFactory.Shutdown()
	}
	c.wg.Wait()
}

func startResourcePoolStatusRequestController(tCtx ktesting.TContext) {
	tCtx.Helper()
	value := tCtx.Value(resourcePoolStatusRequestControllerKey)
	if value == nil {
		tCtx.Fatal("internal error: startResourcePoolStatusRequestController without a prior prepareResourcePoolStatusRequestController call")
	}
	controller := value.(*resourcePoolStatusRequestControllerSingleton)
	controller.start(tCtx)
}

func testProcessResourcePoolStatusRequest(tCtx ktesting.TContext) {
	startResourcePoolStatusRequestController(tCtx)

	driverName := "test.example.com"
	poolName := "test-pool"
	nodeName := "test-node"

	// Create a ResourceSlice
	slice := &resourceapi.ResourceSlice{
		ObjectMeta: metav1.ObjectMeta{
			Name: "test-slice",
		},
		Spec: resourceapi.ResourceSliceSpec{
			Driver:   driverName,
			Pool:     resourceapi.ResourcePool{Name: poolName, ResourceSliceCount: 1, Generation: 1},
			NodeName: ptr.To(nodeName),
			Devices: []resourceapi.Device{
				{Name: "device-1"},
				{Name: "device-2"},
				{Name: "device-3"},
				{Name: "device-4"},
			},
		},
	}
	slice = must(tCtx, tCtx.Client().ResourceV1().ResourceSlices().Create, slice, metav1.CreateOptions{})
	tCtx.CleanupCtx(func(tCtx ktesting.TContext) {
		deleteAndWait(tCtx, tCtx.Client().ResourceV1().ResourceSlices().Delete, tCtx.Client().ResourceV1().ResourceSlices().Get, slice.Name)
	})

	// Create a ResourcePoolStatusRequest
	request := &resourcev1alpha1.ResourcePoolStatusRequest{
		ObjectMeta: metav1.ObjectMeta{
			Name: "test-request",
		},
		Spec: resourcev1alpha1.ResourcePoolStatusRequestSpec{
			Driver: driverName,
			Limit:  ptr.To(int32(100)),
		},
	}
	request = must(tCtx, tCtx.Client().ResourceV1alpha1().ResourcePoolStatusRequests().Create, request, metav1.CreateOptions{})
	tCtx.CleanupCtx(func(tCtx ktesting.TContext) {
		deleteAndWait(tCtx, tCtx.Client().ResourceV1alpha1().ResourcePoolStatusRequests().Delete, tCtx.Client().ResourceV1alpha1().ResourcePoolStatusRequests().Get, request.Name)
	})

	// Wait for the controller to process the request
	tCtx.Eventually(func(tCtx ktesting.TContext) (*resourcev1alpha1.ResourcePoolStatusRequest, error) {
		return tCtx.Client().ResourceV1alpha1().ResourcePoolStatusRequests().Get(tCtx, request.Name, metav1.GetOptions{})
	}).WithTimeout(30*time.Second).Should(
		gomega.And(
			gomega.HaveField("Status.ObservationTime", gomega.Not(gomega.BeNil())),
			gomega.HaveField("Status.TotalMatchingPools", gomega.Equal(int32(1))),
			gomega.HaveField("Status.Pools", gomega.HaveLen(1)),
		),
		"ResourcePoolStatusRequest should have been processed",
	)

	// Verify the pool status details
	request, err := tCtx.Client().ResourceV1alpha1().ResourcePoolStatusRequests().Get(tCtx, request.Name, metav1.GetOptions{})
	tCtx.ExpectNoError(err, "get processed request")

	if len(request.Status.Pools) != 1 {
		tCtx.Fatalf("expected 1 pool, got %d", len(request.Status.Pools))
	}

	pool := request.Status.Pools[0]
	if pool.Driver != driverName {
		tCtx.Errorf("expected driver %q, got %q", driverName, pool.Driver)
	}
	if pool.PoolName != poolName {
		tCtx.Errorf("expected pool name %q, got %q", poolName, pool.PoolName)
	}
	if pool.NodeName != nodeName {
		tCtx.Errorf("expected node name %q, got %q", nodeName, pool.NodeName)
	}
	if pool.TotalDevices != 4 {
		tCtx.Errorf("expected 4 total devices, got %d", pool.TotalDevices)
	}
	if pool.AvailableDevices != 4 {
		tCtx.Errorf("expected 4 available devices, got %d", pool.AvailableDevices)
	}
	if pool.AllocatedDevices != 0 {
		tCtx.Errorf("expected 0 allocated devices, got %d", pool.AllocatedDevices)
	}
	if pool.SliceCount != 1 {
		tCtx.Errorf("expected slice count 1, got %d", pool.SliceCount)
	}
}

func testOneTimeProcessing(tCtx ktesting.TContext) {
	startResourcePoolStatusRequestController(tCtx)

	driverName := "test.example.com"

	// Create a ResourceSlice
	slice := &resourceapi.ResourceSlice{
		ObjectMeta: metav1.ObjectMeta{
			Name: "test-slice-one-time",
		},
		Spec: resourceapi.ResourceSliceSpec{
			Driver:   driverName,
			Pool:     resourceapi.ResourcePool{Name: "pool-1", ResourceSliceCount: 1, Generation: 1},
			NodeName: ptr.To("node-1"),
			Devices: []resourceapi.Device{
				{Name: "device-1"},
			},
		},
	}
	slice = must(tCtx, tCtx.Client().ResourceV1().ResourceSlices().Create, slice, metav1.CreateOptions{})
	tCtx.CleanupCtx(func(tCtx ktesting.TContext) {
		deleteAndWait(tCtx, tCtx.Client().ResourceV1().ResourceSlices().Delete, tCtx.Client().ResourceV1().ResourceSlices().Get, slice.Name)
	})

	// Create a ResourcePoolStatusRequest
	request := &resourcev1alpha1.ResourcePoolStatusRequest{
		ObjectMeta: metav1.ObjectMeta{
			Name: "test-request-one-time",
		},
		Spec: resourcev1alpha1.ResourcePoolStatusRequestSpec{
			Driver: driverName,
		},
	}
	request = must(tCtx, tCtx.Client().ResourceV1alpha1().ResourcePoolStatusRequests().Create, request, metav1.CreateOptions{})
	tCtx.CleanupCtx(func(tCtx ktesting.TContext) {
		deleteAndWait(tCtx, tCtx.Client().ResourceV1alpha1().ResourcePoolStatusRequests().Delete, tCtx.Client().ResourceV1alpha1().ResourcePoolStatusRequests().Get, request.Name)
	})

	// Wait for the controller to process the request
	tCtx.Eventually(func(tCtx ktesting.TContext) (*resourcev1alpha1.ResourcePoolStatusRequest, error) {
		return tCtx.Client().ResourceV1alpha1().ResourcePoolStatusRequests().Get(tCtx, request.Name, metav1.GetOptions{})
	}).WithTimeout(30*time.Second).Should(
		gomega.HaveField("Status.ObservationTime", gomega.Not(gomega.BeNil())),
		"ResourcePoolStatusRequest should have been processed",
	)

	// Get the processed request
	processedRequest, err := tCtx.Client().ResourceV1alpha1().ResourcePoolStatusRequests().Get(tCtx, request.Name, metav1.GetOptions{})
	tCtx.ExpectNoError(err, "get processed request")
	originalObservationTime := *processedRequest.Status.ObservationTime

	// Add a new slice - the request should NOT be reprocessed
	slice2 := &resourceapi.ResourceSlice{
		ObjectMeta: metav1.ObjectMeta{
			Name: "test-slice-one-time-2",
		},
		Spec: resourceapi.ResourceSliceSpec{
			Driver:   driverName,
			Pool:     resourceapi.ResourcePool{Name: "pool-2", ResourceSliceCount: 1, Generation: 1},
			NodeName: ptr.To("node-2"),
			Devices: []resourceapi.Device{
				{Name: "device-1"},
			},
		},
	}
	slice2 = must(tCtx, tCtx.Client().ResourceV1().ResourceSlices().Create, slice2, metav1.CreateOptions{})
	tCtx.CleanupCtx(func(tCtx ktesting.TContext) {
		deleteAndWait(tCtx, tCtx.Client().ResourceV1().ResourceSlices().Delete, tCtx.Client().ResourceV1().ResourceSlices().Get, slice2.Name)
	})

	// Wait a bit and verify the request was NOT reprocessed
	time.Sleep(2 * time.Second)

	finalRequest, err := tCtx.Client().ResourceV1alpha1().ResourcePoolStatusRequests().Get(tCtx, request.Name, metav1.GetOptions{})
	tCtx.ExpectNoError(err, "get final request")

	if finalRequest.Status.ObservationTime.Time != originalObservationTime.Time {
		tCtx.Errorf("request was reprocessed: original observation time %v, final %v",
			originalObservationTime, finalRequest.Status.ObservationTime)
	}
	// Should still show only 1 pool (the original one)
	if finalRequest.Status.TotalMatchingPools != 1 {
		tCtx.Errorf("expected 1 total matching pool (from original processing), got %d", finalRequest.Status.TotalMatchingPools)
	}
}

func testLimitTruncation(tCtx ktesting.TContext) {
	startResourcePoolStatusRequestController(tCtx)

	driverName := "test-limit.example.com"

	// Create multiple ResourceSlices with different pools
	for i := range 5 {
		sliceName := fmt.Sprintf("test-slice-limit-%d", i)
		poolName := fmt.Sprintf("pool-%d", i)
		nodeName := fmt.Sprintf("node-%d", i)
		slice := &resourceapi.ResourceSlice{
			ObjectMeta: metav1.ObjectMeta{
				Name: sliceName,
			},
			Spec: resourceapi.ResourceSliceSpec{
				Driver:   driverName,
				Pool:     resourceapi.ResourcePool{Name: poolName, ResourceSliceCount: 1, Generation: 1},
				NodeName: ptr.To(nodeName),
				Devices: []resourceapi.Device{
					{Name: "device-1"},
				},
			},
		}
		slice = must(tCtx, tCtx.Client().ResourceV1().ResourceSlices().Create, slice, metav1.CreateOptions{})
		tCtx.CleanupCtx(func(tCtx ktesting.TContext) {
			deleteAndWait(tCtx, tCtx.Client().ResourceV1().ResourceSlices().Delete, tCtx.Client().ResourceV1().ResourceSlices().Get, slice.Name)
		})
	}

	// Create a ResourcePoolStatusRequest with limit=2
	request := &resourcev1alpha1.ResourcePoolStatusRequest{
		ObjectMeta: metav1.ObjectMeta{
			Name: "test-request-limit",
		},
		Spec: resourcev1alpha1.ResourcePoolStatusRequestSpec{
			Driver: driverName,
			Limit:  ptr.To(int32(2)),
		},
	}
	request = must(tCtx, tCtx.Client().ResourceV1alpha1().ResourcePoolStatusRequests().Create, request, metav1.CreateOptions{})
	tCtx.CleanupCtx(func(tCtx ktesting.TContext) {
		deleteAndWait(tCtx, tCtx.Client().ResourceV1alpha1().ResourcePoolStatusRequests().Delete, tCtx.Client().ResourceV1alpha1().ResourcePoolStatusRequests().Get, request.Name)
	})

	// Wait for the controller to process the request
	tCtx.Eventually(func(tCtx ktesting.TContext) (*resourcev1alpha1.ResourcePoolStatusRequest, error) {
		return tCtx.Client().ResourceV1alpha1().ResourcePoolStatusRequests().Get(tCtx, request.Name, metav1.GetOptions{})
	}).WithTimeout(30*time.Second).Should(
		gomega.And(
			gomega.HaveField("Status.ObservationTime", gomega.Not(gomega.BeNil())),
			// TotalMatchingPools should be 5 (all pools match)
			gomega.HaveField("Status.TotalMatchingPools", gomega.Equal(int32(5))),
			// But Pools should only contain 2 (due to limit)
			gomega.HaveField("Status.Pools", gomega.HaveLen(2)),
		),
		"ResourcePoolStatusRequest should have been processed with limit truncation",
	)
}

func testFilterByPoolName(tCtx ktesting.TContext) {
	startResourcePoolStatusRequestController(tCtx)

	driverName := "test-filter.example.com"
	targetPoolName := "target-pool"

	// Create multiple ResourceSlices - one with the target pool, others with different pools
	sliceTarget := &resourceapi.ResourceSlice{
		ObjectMeta: metav1.ObjectMeta{
			Name: "test-slice-filter-target",
		},
		Spec: resourceapi.ResourceSliceSpec{
			Driver:   driverName,
			Pool:     resourceapi.ResourcePool{Name: targetPoolName, ResourceSliceCount: 1, Generation: 1},
			NodeName: ptr.To("node-target"),
			Devices: []resourceapi.Device{
				{Name: "device-1"},
				{Name: "device-2"},
			},
		},
	}
	sliceTarget = must(tCtx, tCtx.Client().ResourceV1().ResourceSlices().Create, sliceTarget, metav1.CreateOptions{})
	tCtx.CleanupCtx(func(tCtx ktesting.TContext) {
		deleteAndWait(tCtx, tCtx.Client().ResourceV1().ResourceSlices().Delete, tCtx.Client().ResourceV1().ResourceSlices().Get, sliceTarget.Name)
	})

	sliceOther := &resourceapi.ResourceSlice{
		ObjectMeta: metav1.ObjectMeta{
			Name: "test-slice-filter-other",
		},
		Spec: resourceapi.ResourceSliceSpec{
			Driver:   driverName,
			Pool:     resourceapi.ResourcePool{Name: "other-pool", ResourceSliceCount: 1, Generation: 1},
			NodeName: ptr.To("node-other"),
			Devices: []resourceapi.Device{
				{Name: "device-1"},
			},
		},
	}
	sliceOther = must(tCtx, tCtx.Client().ResourceV1().ResourceSlices().Create, sliceOther, metav1.CreateOptions{})
	tCtx.CleanupCtx(func(tCtx ktesting.TContext) {
		deleteAndWait(tCtx, tCtx.Client().ResourceV1().ResourceSlices().Delete, tCtx.Client().ResourceV1().ResourceSlices().Get, sliceOther.Name)
	})

	// Create a ResourcePoolStatusRequest filtered by pool name
	request := &resourcev1alpha1.ResourcePoolStatusRequest{
		ObjectMeta: metav1.ObjectMeta{
			Name: "test-request-filter",
		},
		Spec: resourcev1alpha1.ResourcePoolStatusRequestSpec{
			Driver:   driverName,
			PoolName: targetPoolName,
		},
	}
	request = must(tCtx, tCtx.Client().ResourceV1alpha1().ResourcePoolStatusRequests().Create, request, metav1.CreateOptions{})
	tCtx.CleanupCtx(func(tCtx ktesting.TContext) {
		deleteAndWait(tCtx, tCtx.Client().ResourceV1alpha1().ResourcePoolStatusRequests().Delete, tCtx.Client().ResourceV1alpha1().ResourcePoolStatusRequests().Get, request.Name)
	})

	// Wait for the controller to process the request
	tCtx.Eventually(func(tCtx ktesting.TContext) (*resourcev1alpha1.ResourcePoolStatusRequest, error) {
		return tCtx.Client().ResourceV1alpha1().ResourcePoolStatusRequests().Get(tCtx, request.Name, metav1.GetOptions{})
	}).WithTimeout(30*time.Second).Should(
		gomega.And(
			gomega.HaveField("Status.ObservationTime", gomega.Not(gomega.BeNil())),
			// Should match only 1 pool (the target pool)
			gomega.HaveField("Status.TotalMatchingPools", gomega.Equal(int32(1))),
			gomega.HaveField("Status.Pools", gomega.HaveLen(1)),
		),
		"ResourcePoolStatusRequest should have been processed with pool name filter",
	)

	// Verify the pool status details
	processedRequest, err := tCtx.Client().ResourceV1alpha1().ResourcePoolStatusRequests().Get(tCtx, request.Name, metav1.GetOptions{})
	tCtx.ExpectNoError(err, "get processed request")

	if len(processedRequest.Status.Pools) != 1 {
		tCtx.Fatalf("expected 1 pool, got %d", len(processedRequest.Status.Pools))
	}

	pool := processedRequest.Status.Pools[0]
	if pool.PoolName != targetPoolName {
		tCtx.Errorf("expected pool name %q, got %q", targetPoolName, pool.PoolName)
	}
	if pool.TotalDevices != 2 {
		tCtx.Errorf("expected 2 total devices, got %d", pool.TotalDevices)
	}
}

func testValidationErrors(tCtx ktesting.TContext) {
	// Test that validation errors are properly detected and reported by the API server.
	// This doesn't require the controller to be running since validation happens at admission.

	// Test 1: Missing driver field (required)
	invalidRequest := &resourcev1alpha1.ResourcePoolStatusRequest{
		ObjectMeta: metav1.ObjectMeta{
			Name: "test-invalid-no-driver",
		},
		Spec: resourcev1alpha1.ResourcePoolStatusRequestSpec{
			// Driver is missing - should fail validation
		},
	}
	_, err := tCtx.Client().ResourceV1alpha1().ResourcePoolStatusRequests().Create(tCtx, invalidRequest, metav1.CreateOptions{})
	if err == nil {
		tCtx.Fatal("expected validation error for missing driver, but create succeeded")
	}
	if !apierrors.IsInvalid(err) {
		tCtx.Errorf("expected Invalid error, got: %v", err)
	}

	// Test 2: Invalid limit (0 is not allowed, must be >= 1)
	invalidLimitRequest := &resourcev1alpha1.ResourcePoolStatusRequest{
		ObjectMeta: metav1.ObjectMeta{
			Name: "test-invalid-limit-zero",
		},
		Spec: resourcev1alpha1.ResourcePoolStatusRequestSpec{
			Driver: "test.example.com",
			Limit:  ptr.To(int32(0)),
		},
	}
	_, err = tCtx.Client().ResourceV1alpha1().ResourcePoolStatusRequests().Create(tCtx, invalidLimitRequest, metav1.CreateOptions{})
	if err == nil {
		tCtx.Fatal("expected validation error for limit=0, but create succeeded")
	}
	if !apierrors.IsInvalid(err) {
		tCtx.Errorf("expected Invalid error for limit=0, got: %v", err)
	}

	// Test 3: Limit exceeds maximum (1000)
	invalidLimitMaxRequest := &resourcev1alpha1.ResourcePoolStatusRequest{
		ObjectMeta: metav1.ObjectMeta{
			Name: "test-invalid-limit-max",
		},
		Spec: resourcev1alpha1.ResourcePoolStatusRequestSpec{
			Driver: "test.example.com",
			Limit:  ptr.To(int32(1001)),
		},
	}
	_, err = tCtx.Client().ResourceV1alpha1().ResourcePoolStatusRequests().Create(tCtx, invalidLimitMaxRequest, metav1.CreateOptions{})
	if err == nil {
		tCtx.Fatal("expected validation error for limit>1000, but create succeeded")
	}
	if !apierrors.IsInvalid(err) {
		tCtx.Errorf("expected Invalid error for limit>1000, got: %v", err)
	}

	// Test 4: Invalid driver name format
	invalidDriverRequest := &resourcev1alpha1.ResourcePoolStatusRequest{
		ObjectMeta: metav1.ObjectMeta{
			Name: "test-invalid-driver-name",
		},
		Spec: resourcev1alpha1.ResourcePoolStatusRequestSpec{
			Driver: "invalid driver name with spaces",
		},
	}
	_, err = tCtx.Client().ResourceV1alpha1().ResourcePoolStatusRequests().Create(tCtx, invalidDriverRequest, metav1.CreateOptions{})
	if err == nil {
		tCtx.Fatal("expected validation error for invalid driver name, but create succeeded")
	}
	if !apierrors.IsInvalid(err) {
		tCtx.Errorf("expected Invalid error for invalid driver name, got: %v", err)
	}
}
