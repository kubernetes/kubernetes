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
	"context"
	"errors"
	"fmt"
	"sync"
	"sync/atomic"
	"time"

	"github.com/google/go-cmp/cmp"
	"github.com/onsi/gomega"
	"github.com/onsi/gomega/gstruct"
	gtypes "github.com/onsi/gomega/types"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	v1 "k8s.io/api/core/v1"
	resourceapi "k8s.io/api/resource/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/client-go/informers"
	"k8s.io/client-go/tools/cache"
	"k8s.io/component-base/featuregate"
	"k8s.io/component-base/metrics/testutil"
	draclient "k8s.io/dynamic-resource-allocation/client"
	resourceclaimmetrics "k8s.io/dynamic-resource-allocation/resourceclaim/metrics"
	"k8s.io/dynamic-resource-allocation/resourceslice"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/pkg/controller/resourceclaim"
	"k8s.io/kubernetes/pkg/features"
	st "k8s.io/kubernetes/pkg/scheduler/testing"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	"k8s.io/kubernetes/test/integration/util"
	"k8s.io/kubernetes/test/utils/client-go/ktesting"
	"k8s.io/utils/ptr"
)

// testPod creates a pod with a resource claim reference and then checks
// whether that field is or isn't getting dropped.
func testPod(tCtx ktesting.TContext, draEnabled bool) {
	tCtx.Parallel()
	namespace := createTestNamespace(tCtx, nil)
	podWithClaimName := podWithClaimName.DeepCopy()
	podWithClaimName.Namespace = namespace
	pod, err := tCtx.Client().CoreV1().Pods(namespace).Create(tCtx, podWithClaimName, metav1.CreateOptions{FieldValidation: "Strict"})
	tCtx.ExpectNoError(err, "create pod")
	if draEnabled {
		assert.NotEmpty(tCtx, pod.Spec.ResourceClaims, "should store resource claims in pod spec")
	} else {
		assert.Empty(tCtx, pod.Spec.ResourceClaims, "should drop resource claims from pod spec")
	}
}

// testAPIDisabled checks that the resource.k8s.io API is disabled.
func testAPIDisabled(tCtx ktesting.TContext) {
	tCtx.Parallel()
	_, err := tCtx.Client().ResourceV1().ResourceClaims(claim.Namespace).Create(tCtx, claim, metav1.CreateOptions{FieldValidation: "Strict"})
	if !apierrors.IsNotFound(err) {
		tCtx.Fatalf("expected 'resource not found' error, got %v", err)
	}
}

// testConvert creates a claim using a one API version and reads it with another.
func testConvert(tCtx ktesting.TContext) {
	tCtx.Parallel()
	namespace := createTestNamespace(tCtx, nil)
	claim := claim.DeepCopy()
	claim.Namespace = namespace
	claim, err := tCtx.Client().ResourceV1().ResourceClaims(namespace).Create(tCtx, claim, metav1.CreateOptions{FieldValidation: "Strict"})
	tCtx.ExpectNoError(err, "create claim")
	claimBeta2, err := tCtx.Client().ResourceV1beta2().ResourceClaims(namespace).Get(tCtx, claim.Name, metav1.GetOptions{})
	tCtx.ExpectNoError(err, "get claim")
	// We could check more fields, but there are unit tests which cover this better.
	assert.Equal(tCtx, claim.Name, claimBeta2.Name, "claim name")
}

// testFilterTimeout covers the scheduler plugin's filter timeout configuration and behavior.
//
// It runs the scheduler with non-standard settings and thus cannot run in parallel.
func testFilterTimeout(tCtx ktesting.TContext, requestDeviceCount int) {
	namespace := createTestNamespace(tCtx, nil)
	class, driverName := createTestClass(tCtx, namespace)
	deviceNames := make([]string, requestDeviceCount)
	for i := range requestDeviceCount {
		deviceNames[i] = fmt.Sprintf("dev-%d", i)
	}
	slice := st.MakeResourceSlice("worker-0", driverName).Devices(deviceNames...)
	createSlice(tCtx, slice.Obj())
	otherSlice := st.MakeResourceSlice("worker-1", driverName).Devices(deviceNames[:requestDeviceCount-1]...)
	createdOtherSlice := createSlice(tCtx, otherSlice.Obj())

	// Impossible to allocate on worker-1: not enough devices, but allocation is too
	// dumb to notice that upfront and keeps trying until it times out.
	// On worker-0 we can allocate, but don't schedule because of the timeout on worker-1.
	newClaim := func(suffix string) *resourceapi.ResourceClaim {
		c := claim.DeepCopy()
		c.Spec.Devices.Requests[0].Exactly.Count = int64(requestDeviceCount)
		return createClaim(tCtx, namespace, suffix, class, c)
	}

	runSubTest(tCtx, "disabled", func(tCtx ktesting.TContext) {
		cl := newClaim("-disabled")
		pod := createPod(tCtx, namespace, "-disabled", podWithClaimName, cl)
		startSchedulerWithConfig(tCtx, `
profiles:
- schedulerName: default-scheduler
  pluginConfig:
  - name: DynamicResources
    args:
      filterTimeout: 0s
`)
		// Without a timeout, the allocator runs to completion on both nodes.
		// worker-0 has enough devices and succeeds, so the pod gets scheduled.
		tCtx.ExpectNoError(e2epod.WaitForPodScheduled(tCtx, tCtx.Client(), namespace, pod.Name))
	})

	runSubTest(tCtx, "enabled", func(tCtx ktesting.TContext) {
		cl := newClaim("-enabled")
		pod := createPod(tCtx, namespace, "-enabled", podWithClaimName, cl)
		startSchedulerWithConfig(tCtx, `
profiles:
- schedulerName: default-scheduler
  pluginConfig:
  - name: DynamicResources
    args:
      filterTimeout: 10ms
`)
		expectPodSchedulerError(tCtx, pod, "timed out trying to allocate devices")

		// Update the smaller slice such that allocation also succeeds.
		// The scheduler retries automatically (timeouts go through
		// backoff queue, not unschedulable pool) and should succeed now.
		createdOtherSlice.Spec.Devices = append(createdOtherSlice.Spec.Devices, resourceapi.Device{
			Name: deviceNames[requestDeviceCount-1],
		})
		_, err := tCtx.Client().ResourceV1().ResourceSlices().Update(tCtx, createdOtherSlice, metav1.UpdateOptions{})
		tCtx.ExpectNoError(err, "update worker-1's ResourceSlice")
		tCtx.ExpectNoError(e2epod.WaitForPodScheduled(tCtx, tCtx.Client(), namespace, pod.Name))
	})
}

func testPublishResourceSlices(tCtx ktesting.TContext, haveLatestAPI bool, disabledFeatures ...featuregate.Feature) {
	tCtx.Parallel()

	namespace := createTestNamespace(tCtx, nil)
	driverName := namespace + ".example.com"
	listDriverSlices := metav1.ListOptions{
		FieldSelector: resourceapi.ResourceSliceSelectorDriver + "=" + driverName,
	}
	poolName := "global"
	resources := &resourceslice.DriverResources{
		Pools: map[string]resourceslice.Pool{
			poolName: {
				Slices: []resourceslice.Slice{
					{
						Devices: []resourceapi.Device{
							{
								Name: "device-simple",
							},
						},
					},
					{
						SharedCounters: []resourceapi.CounterSet{
							{
								Name: "gpu-0",
								Counters: map[string]resourceapi.Counter{
									"mem": {Value: resource.MustParse("1")},
								},
							},
						},
					},
					{
						Devices: []resourceapi.Device{
							{
								Name: "device-tainted-default",
								Taints: []resourceapi.DeviceTaint{{
									Key:    "dra.example.com/taint",
									Value:  "taint-value",
									Effect: resourceapi.DeviceTaintEffectNoExecute,
									// TimeAdded is added by apiserver.
								}},
							},
							{
								Name: "device-tainted-time-added",
								Taints: []resourceapi.DeviceTaint{{
									Key:       "dra.example.com/taint",
									Value:     "taint-value",
									Effect:    resourceapi.DeviceTaintEffectNoExecute,
									TimeAdded: ptr.To(metav1.Time{Time: time.Now().Truncate(time.Second)}),
								}},
							},
							{
								Name: "gpu",
								ConsumesCounters: []resourceapi.DeviceCounterConsumption{{
									CounterSet: "gpu-0",
									Counters: map[string]resourceapi.Counter{
										"mem": {Value: resource.MustParse("1")},
									},
								}},
							},
							{
								Name: "device-binding-conditions",
								BindingConditions: []string{
									"condition-1",
									"condition-2",
								},
								BindingFailureConditions: []string{
									"failure-condition-1",
									"failure-condition-2",
								},
								BindsToNode: ptr.To(true),
							},
						},
					},
				},
				AllNodes: true,
			},
		},
	}

	// Manually turn into the expected slices, considering that some fields get dropped.
	expectedResources := resources.DeepCopy()
	var expectedSliceSpecs []resourceapi.ResourceSliceSpec
	for _, sl := range expectedResources.Pools[poolName].Slices {
		expectedSliceSpecs = append(expectedSliceSpecs, resourceapi.ResourceSliceSpec{
			Driver: driverName,
			Pool: resourceapi.ResourcePool{
				Name:               poolName,
				ResourceSliceCount: int64(len(expectedResources.Pools[poolName].Slices)),
			},
			AllNodes:       ptr.To(true),
			SharedCounters: sl.SharedCounters,
			Devices:        sl.Devices,
		})
	}

	// Keep track of the disabled features used in each slice. This allows us to check
	// that we get the right set of features listed in the droppedFields error.
	disabledFeaturesBySlice := make([]sets.Set[featuregate.Feature], len(resources.Pools[poolName].Slices))
	for i := range expectedSliceSpecs {
		disabledFeaturesForSlice := sets.New[featuregate.Feature]()
		for _, disabled := range disabledFeatures {
			switch disabled {
			case features.DRADeviceTaints:
				for e, device := range expectedSliceSpecs[i].Devices {
					if device.Taints != nil {
						expectedSliceSpecs[i].Devices[e].Taints = nil
						disabledFeaturesForSlice.Insert(disabled)
					}
				}
			case features.DRAPartitionableDevices:
				if expectedSliceSpecs[i].SharedCounters != nil {
					expectedSliceSpecs[i].SharedCounters = nil
					disabledFeaturesForSlice.Insert(disabled)
				}
				for e, device := range expectedSliceSpecs[i].Devices {
					if device.ConsumesCounters != nil {
						expectedSliceSpecs[i].Devices[e].ConsumesCounters = nil
						disabledFeaturesForSlice.Insert(disabled)
					}
				}
			case features.DRADeviceBindingConditions:
				for e, device := range expectedSliceSpecs[i].Devices {
					if device.BindingConditions != nil || device.BindingFailureConditions != nil || device.BindsToNode != nil {
						expectedSliceSpecs[i].Devices[e].BindingConditions = nil
						expectedSliceSpecs[i].Devices[e].BindingFailureConditions = nil
						expectedSliceSpecs[i].Devices[e].BindsToNode = nil
						disabledFeaturesForSlice.Insert(disabled)
					}
				}
			default:
				tCtx.Fatalf("faulty test, case for %s missing", disabled)
			}
		}
		disabledFeaturesBySlice[i] = disabledFeaturesForSlice
	}
	var expectedSlices []any
	for _, spec := range expectedSliceSpecs {
		// The matcher is precise and matches all fields, except for those few which are known to
		// be not exactly as sent by the client. New fields have to be added when extending the API.
		expectedSlices = append(expectedSlices, gomega.HaveField("Spec", gstruct.MatchAllFields(gstruct.Fields{
			"Driver": gomega.Equal(driverName),
			"Pool": gstruct.MatchAllFields(gstruct.Fields{
				"Name":               gomega.Equal(poolName),
				"Generation":         gomega.BeNumerically(">=", int64(1)),
				"ResourceSliceCount": gomega.Equal(int64(len(expectedResources.Pools[poolName].Slices))),
			}),
			"NodeName":     matchPointer(spec.NodeName),
			"NodeSelector": matchPointer(spec.NodeSelector),
			"AllNodes":     gstruct.PointTo(gomega.BeTrue()),
			"Devices": gomega.HaveExactElements(func() []any {
				var expected []any
				for _, device := range spec.Devices {
					expected = append(expected, gstruct.MatchAllFields(gstruct.Fields{
						"Name":                     gomega.Equal(device.Name),
						"AllowMultipleAllocations": gomega.Equal(device.AllowMultipleAllocations),
						"Attributes":               gomega.Equal(device.Attributes),
						"Capacity":                 gomega.Equal(device.Capacity),
						"ConsumesCounters":         gomega.Equal(device.ConsumesCounters),
						"NodeName":                 matchPointer(device.NodeName),
						"NodeSelector":             matchPointer(device.NodeSelector),
						"AllNodes":                 matchPointer(device.AllNodes),
						"Taints": gomega.HaveExactElements(func() []any {
							var expected []any
							for _, taint := range device.Taints {
								if taint.TimeAdded != nil {
									// Can do exact match.
									expected = append(expected, gomega.Equal(taint))
								} else {
									// Ignore TimeAdded value.
									expected = append(expected, gstruct.MatchAllFields(gstruct.Fields{
										"Key":       gomega.Equal(taint.Key),
										"Value":     gomega.Equal(taint.Value),
										"Effect":    gomega.Equal(taint.Effect),
										"TimeAdded": gomega.Not(gomega.BeNil()),
									}))
								}
							}
							return expected
						}()...),
						"BindingConditions":               gomega.Equal(device.BindingConditions),
						"BindingFailureConditions":        gomega.Equal(device.BindingFailureConditions),
						"BindsToNode":                     gomega.Equal(device.BindsToNode),
						"NodeAllocatableResourceMappings": gomega.Equal(device.NodeAllocatableResourceMappings),
					}))
				}
				return expected
			}()...),
			"PerDeviceNodeSelection": matchPointer(spec.PerDeviceNodeSelection),
			"SharedCounters":         gomega.Equal(spec.SharedCounters),
		})))
	}

	expectSlices := func(tCtx ktesting.TContext) {
		tCtx.Helper()

		if !haveLatestAPI {
			return
		}
		slices, err := tCtx.Client().ResourceV1().ResourceSlices().List(tCtx, listDriverSlices)
		tCtx.ExpectNoError(err, "list slices")
		gomega.NewGomegaWithT(tCtx).Expect(slices.Items).Should(gomega.ConsistOf(expectedSlices...))
	}

	deleteSlices := func(tCtx ktesting.TContext) {
		tCtx.Helper()

		err := draclient.New(tCtx.Client()).ResourceSlices().DeleteCollection(tCtx, metav1.DeleteOptions{}, listDriverSlices)
		tCtx.ExpectNoError(err, "delete slices")
	}

	// Speed up testing a bit...
	factor := time.Duration(10)
	mutationCacheTTL := resourceslice.DefaultMutationCacheTTL / factor
	syncDelay := resourceslice.DefaultSyncDelay / factor
	quiesencePeriod := max(mutationCacheTTL, syncDelay)
	quiesencePeriod += 10 * time.Second

	var gotDroppedFieldError atomic.Bool
	var gotValidationError atomic.Bool
	var validationErrorsOkay atomic.Bool

	setup := func(tCtx ktesting.TContext) (*resourceslice.Controller, func(tCtx ktesting.TContext) resourceslice.Stats, resourceslice.Stats) {
		tCtx.Helper()

		tCtx.CleanupCtx(deleteSlices)

		gotDroppedFieldError.Store(false)
		gotValidationError.Store(false)
		validationErrorsOkay.Store(false)

		opts := resourceslice.Options{
			DriverName:       driverName,
			KubeClient:       tCtx.Client(),
			Resources:        resources,
			MutationCacheTTL: &mutationCacheTTL,
			SyncDelay:        &syncDelay,
			ErrorHandler: func(ctx context.Context, err error, msg string) {
				klog.FromContext(ctx).Info("ErrorHandler called", "err", err, "msg", msg)
				if !validationErrorsOkay.Load() && len(disabledFeatures) == 0 {
					assert.NoError(tCtx, err, msg)
					return
				}

				var droppedFields *resourceslice.DroppedFieldsError
				if errors.As(err, &droppedFields) {
					var disabled []string
					for _, feature := range disabledFeaturesBySlice[droppedFields.SliceIndex].UnsortedList() {
						disabled = append(disabled, string(feature))
					}
					// Make sure the error is about the right resource pool.
					assert.Equal(tCtx, poolName, droppedFields.PoolName)
					// Make sure the error identifies the correct disabled features.
					assert.ElementsMatch(tCtx, disabled, droppedFields.DisabledFeatures())
					gotDroppedFieldError.Store(true)
				} else if validationErrorsOkay.Load() && apierrors.IsInvalid(err) {
					gotValidationError.Store(true)
				} else {
					tCtx.Errorf("unexpected error: %v", err)
				}
			},
		}

		controller, err := resourceslice.StartController(tCtx, opts)
		tCtx.ExpectNoError(err, "start controller")
		tCtx.Cleanup(controller.Stop)

		expectedStats := resourceslice.Stats{
			NumCreates: int64(len(expectedSlices)),
		}
		getStats := func(tCtx ktesting.TContext) resourceslice.Stats {
			return controller.GetStats()
		}
		tCtx.Eventually(getStats).WithTimeout(syncDelay + 5*time.Second).Should(gomega.Equal(expectedStats))
		expectSlices(tCtx)

		return controller, getStats, expectedStats
	}

	// Each sub-test starts with no slices and must clean up after itself.

	runSubTest(tCtx, "create", func(tCtx ktesting.TContext) {
		controller, getStats, expectedStats := setup(tCtx)

		// No further changes necessary.
		tCtx.Consistently(getStats).WithTimeout(quiesencePeriod).Should(gomega.Equal(expectedStats))

		if len(disabledFeatures) > 0 && !gotDroppedFieldError.Load() {
			tCtx.Error("expected dropped fields error, got none")
		}

		// Now switch to one invalid slice.
		resources := resources.DeepCopy()
		pool := resources.Pools[poolName]
		pool.Slices = pool.Slices[:1]
		pool.Slices[0].Devices[0].Attributes = map[resourceapi.QualifiedName]resourceapi.DeviceAttribute{"empty": {}}
		resources.Pools[poolName] = pool
		validationErrorsOkay.Store(true)
		controller.Update(resources)
		tCtx.Eventually(getStats).WithTimeout(10*time.Second).Should(gomega.HaveField("NumDeletes", gomega.BeNumerically(">=", int64(1))), "Slice should have been removed.")
		tCtx.Eventually(func(tCtx ktesting.TContext) bool {
			return gotValidationError.Load()
		}).WithTimeout(time.Minute).Should(gomega.BeTrueBecause("Should have gotten another error because the slice is invalid."))
	})

	if !haveLatestAPI {
		return
	}

	runSubTest(tCtx, "recreate-after-delete", func(tCtx ktesting.TContext) {
		_, getStats, expectedStats := setup(tCtx)
		tCtx.Consistently(getStats).WithTimeout(quiesencePeriod).Should(gomega.Equal(expectedStats))

		// Stress the controller by repeatedly deleting the slices.
		// One delete occurs after the sync period is over (because of the Consistently),
		// the second before (because it's done as quickly as possible).
		for range 2 {
			tCtx.Log("deleting ResourceSlices")
			tCtx.ExpectNoError(tCtx.Client().ResourceV1().ResourceSlices().DeleteCollection(tCtx, metav1.DeleteOptions{}, listDriverSlices), "delete driver slices")
			expectedStats.NumCreates += int64(len(expectedSlices))
			tCtx.Eventually(getStats).WithTimeout(syncDelay + 5*time.Second).Should(gomega.Equal(expectedStats))
			expectSlices(tCtx)
		}
	})

	runSubTest(tCtx, "fix-after-update", func(tCtx ktesting.TContext) {
		_, getStats, expectedStats := setup(tCtx)

		// Stress the controller by repeatedly updatings the slices.
		for range 2 {
			slices, err := tCtx.Client().ResourceV1().ResourceSlices().List(tCtx, listDriverSlices)
			tCtx.ExpectNoError(err, "list slices")
			for _, slice := range slices.Items {
				if len(slice.Spec.Devices) == 0 {
					continue
				}
				if slice.Spec.Devices[0].Attributes == nil {
					slice.Spec.Devices[0].Attributes = make(map[resourceapi.QualifiedName]resourceapi.DeviceAttribute)
				}
				slice.Spec.Devices[0].Attributes["someUnwantedAttribute"] = resourceapi.DeviceAttribute{BoolValue: ptr.To(true)}
				_, err := tCtx.Client().ResourceV1().ResourceSlices().Update(tCtx, &slice, metav1.UpdateOptions{})
				tCtx.ExpectNoError(err, "update slice")
			}
			expectedStats.NumUpdates += int64(len(expectedSlices))
			tCtx.Eventually(getStats).WithTimeout(syncDelay + 5*time.Second).Should(gomega.Equal(expectedStats))
			expectSlices(tCtx)
		}
	})
}

// testMaxResourceSlice creates ResourceSlices that are as large as possible
// and prints some information about it.
func testMaxResourceSlice(tCtx ktesting.TContext) {
	for name, slice := range NewMaxResourceSlices() {
		runSubTest(tCtx, name, func(tCtx ktesting.TContext) {
			createdSlice := createSlice(tCtx, slice)
			totalSize := createdSlice.Size()
			var managedFieldsSize int
			for _, f := range createdSlice.ManagedFields {
				managedFieldsSize += f.Size()
			}
			specSize := createdSlice.Spec.Size()
			tCtx.Logf("\n\nDevices: %d\nTotal size: %s\nManagedFields size: %s (%.0f%%)\nSpec size: %s (%.0f)%%\n\nManagedFields:\n%s",
				len(createdSlice.Spec.Devices),
				resource.NewQuantity(int64(totalSize), resource.BinarySI),
				resource.NewQuantity(int64(managedFieldsSize), resource.BinarySI), float64(managedFieldsSize)*100/float64(totalSize),
				resource.NewQuantity(int64(specSize), resource.BinarySI), float64(specSize)*100/float64(totalSize),
				klog.Format(createdSlice.ManagedFields),
			)
			if diff := cmp.Diff(slice.Spec, createdSlice.Spec); diff != "" {
				tCtx.Errorf("ResourceSliceSpec got modified during Create (- want, + got):\n%s", diff)
			}
		})
	}
}

// testControllerManagerMetrics tests ResourceClaim metrics.
// It must run sequentially.
func testControllerManagerMetrics(tCtx ktesting.TContext) {
	namespace := createTestNamespace(tCtx, nil)
	class, _ := createTestClass(tCtx, namespace)

	informerFactory := informers.NewSharedInformerFactory(tCtx.Client(), 0)
	features := resourceclaim.Features{
		AdminAccess:     true,
		PrioritizedList: true,
	}
	runResourceClaimController := util.CreateResourceClaimController(tCtx, tCtx, tCtx.Client(), informerFactory, features)
	informerFactory.Start(tCtx.Done())
	cache.WaitForCacheSync(tCtx.Done(),
		informerFactory.Core().V1().Pods().Informer().HasSynced,
		informerFactory.Resource().V1().ResourceClaims().Informer().HasSynced,
		informerFactory.Resource().V1().ResourceClaimTemplates().Informer().HasSynced,
	)

	// Start the controller (this will run in background and stop when tCtx is cancelled)
	var wg sync.WaitGroup
	tCtx.Cleanup(func() {
		tCtx.Cancel("test is done")
		wg.Wait()
	})
	wg.Go(runResourceClaimController)

	tCtx.Log("ResourceClaim controller started successfully")
	tCtx.Log("Testing ResourceClaim controller success metrics with admin access labels")

	// Helper function to get metrics from the metric counter directly
	getMetricValue := func(status, adminAccess string) float64 {
		value, err := testutil.GetCounterMetricValue(resourceclaimmetrics.ResourceClaimCreate.WithLabelValues(status, adminAccess))
		if err != nil {
			// If the metric doesn't exist yet, default to 0
			return 0
		}
		return value
	}

	// Get initial success metrics (only testing success cases since failure cases are not easily testable)
	initialSuccessNoAdmin := getMetricValue("success", "false")
	initialSuccessWithAdmin := getMetricValue("success", "true")

	expectMetricValue := func(status, adminAccess string, want float64, message string) {
		tCtx.Eventually(func(tCtx ktesting.TContext) float64 {
			return getMetricValue(status, adminAccess)
		}).
			WithTimeout(30*time.Second).
			WithPolling(200*time.Millisecond).
			Should(gomega.BeNumerically("~", want, 0.1), message)
	}

	// Test 1: Create Pod with ResourceClaimTemplate without admin access (should succeed and trigger controller)
	template1 := &resourceapi.ResourceClaimTemplate{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test-template-no-admin",
			Namespace: namespace,
		},
		Spec: resourceapi.ResourceClaimTemplateSpec{
			Spec: resourceapi.ResourceClaimSpec{
				Devices: resourceapi.DeviceClaim{
					Requests: []resourceapi.DeviceRequest{
						{
							Name: "req-0",
							Exactly: &resourceapi.ExactDeviceRequest{
								DeviceClassName: class.Name,
								// AdminAccess defaults to false/nil
							},
						},
					},
				},
			},
		},
	}

	_, err := tCtx.Client().ResourceV1().ResourceClaimTemplates(namespace).Create(tCtx, template1, metav1.CreateOptions{FieldValidation: "Strict"})
	tCtx.ExpectNoError(err, "create ResourceClaimTemplate without admin access")

	pod1 := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test-pod-no-admin",
			Namespace: namespace,
		},
		Spec: v1.PodSpec{
			Containers: []v1.Container{{
				Name:  "test-container",
				Image: "busybox",
			}},
			ResourceClaims: []v1.PodResourceClaim{{
				Name:                      "my-claim",
				ResourceClaimTemplateName: &template1.Name,
			}},
		},
	}

	_, err = tCtx.Client().CoreV1().Pods(namespace).Create(tCtx, pod1, metav1.CreateOptions{FieldValidation: "Strict"})
	tCtx.ExpectNoError(err, "create Pod with ResourceClaimTemplate without admin access")

	expectMetricValue("success", "false", initialSuccessNoAdmin+1,
		"success metric with admin_access=false should increment")

	// Test 2: Create admin namespace and Pod with ResourceClaimTemplate with admin access (should succeed)
	adminNS := createTestNamespace(tCtx, map[string]string{"resource.kubernetes.io/admin-access": "true"})
	adminClass, _ := createTestClass(tCtx, adminNS)

	template2 := &resourceapi.ResourceClaimTemplate{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test-template-admin",
			Namespace: adminNS,
		},
		Spec: resourceapi.ResourceClaimTemplateSpec{
			Spec: resourceapi.ResourceClaimSpec{
				Devices: resourceapi.DeviceClaim{
					Requests: []resourceapi.DeviceRequest{{
						Name: "req-0",
						Exactly: &resourceapi.ExactDeviceRequest{
							DeviceClassName: adminClass.Name,
							AdminAccess:     ptr.To(true),
						},
					},
					},
				},
			},
		},
	}

	_, err = tCtx.Client().ResourceV1().ResourceClaimTemplates(adminNS).Create(tCtx, template2, metav1.CreateOptions{FieldValidation: "Strict"})
	tCtx.ExpectNoError(err, "create ResourceClaimTemplate with admin access")

	pod2 := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test-pod-admin",
			Namespace: adminNS,
		},
		Spec: v1.PodSpec{
			Containers: []v1.Container{{
				Name:  "test-container",
				Image: "busybox",
			}},
			ResourceClaims: []v1.PodResourceClaim{{
				Name:                      "my-claim",
				ResourceClaimTemplateName: &template2.Name,
			}},
		},
	}

	_, err = tCtx.Client().CoreV1().Pods(adminNS).Create(tCtx, pod2, metav1.CreateOptions{FieldValidation: "Strict"})
	tCtx.ExpectNoError(err, "create Pod with ResourceClaimTemplate with admin access in admin namespace")

	expectMetricValue("success", "true", initialSuccessWithAdmin+1,
		"success metric with admin_access=true should increment")

	// Test 3: Try to create ResourceClaimTemplate with admin access in non-admin namespace
	// should fail at API level, controller not triggered, no metrics change expected
	invalidTemplate := &resourceapi.ResourceClaimTemplate{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test-template-invalid-admin",
			Namespace: namespace, // regular namespace without admin label
		},
		Spec: resourceapi.ResourceClaimTemplateSpec{
			Spec: resourceapi.ResourceClaimSpec{
				Devices: resourceapi.DeviceClaim{
					Requests: []resourceapi.DeviceRequest{{
						Name: "req-0",
						Exactly: &resourceapi.ExactDeviceRequest{
							DeviceClassName: class.Name,
							AdminAccess:     ptr.To(true),
						},
					}},
				},
			},
		},
	}

	_, err = tCtx.Client().ResourceV1().ResourceClaimTemplates(namespace).Create(tCtx, invalidTemplate, metav1.CreateOptions{FieldValidation: "Strict"})
	require.Error(tCtx, err, "should fail to create ResourceClaimTemplate with AdminAccess in non-admin namespace")
	require.ErrorContains(tCtx, err, "admin access to devices requires the `resource.kubernetes.io/admin-access: true` label on the containing namespace")

	// Test 4: Create another Pod with ResourceClaimTemplate without admin access to further verify metrics
	template4 := &resourceapi.ResourceClaimTemplate{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test-template-no-admin-2",
			Namespace: namespace,
		},
		Spec: resourceapi.ResourceClaimTemplateSpec{
			Spec: resourceapi.ResourceClaimSpec{
				Devices: resourceapi.DeviceClaim{
					Requests: []resourceapi.DeviceRequest{{
						Name: "req-0",
						Exactly: &resourceapi.ExactDeviceRequest{
							DeviceClassName: class.Name,
						},
					}},
				},
			},
		},
	}

	_, err = tCtx.Client().ResourceV1().ResourceClaimTemplates(namespace).Create(tCtx, template4, metav1.CreateOptions{FieldValidation: "Strict"})
	tCtx.ExpectNoError(err, "create second ResourceClaimTemplate without admin access")

	pod4 := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test-pod-no-admin-2",
			Namespace: namespace,
		},
		Spec: v1.PodSpec{
			Containers: []v1.Container{{
				Name:  "test-container",
				Image: "busybox",
			}},
			ResourceClaims: []v1.PodResourceClaim{{
				Name:                      "my-claim",
				ResourceClaimTemplateName: &template4.Name,
			}},
		},
	}

	_, err = tCtx.Client().CoreV1().Pods(namespace).Create(tCtx, pod4, metav1.CreateOptions{FieldValidation: "Strict"})
	tCtx.ExpectNoError(err, "create second Pod with ResourceClaimTemplate without admin access")

	expectMetricValue("success", "false", initialSuccessNoAdmin+2,
		"should have 2 more success metrics with admin_access=false")
	expectMetricValue("success", "true", initialSuccessWithAdmin+1,
		"should have 1 more success metric with admin_access=true")

	tCtx.Log("ResourceClaim controller success metrics correctly track operations with admin_access labels")
}

func matchPointer[T any](p *T) gtypes.GomegaMatcher {
	if p == nil {
		return gomega.BeNil()
	}
	return gstruct.PointTo(gomega.Equal(*p))
}

func testInvalidResourceSlices(tCtx ktesting.TContext) {
	driverNamePlaceholder := "driver"
	startScheduler(tCtx)
	testCases := map[string]struct {
		slices                        []*st.ResourceSliceWrapper
		expectPodToSchedule           bool
		expectedPodScheduledCondition gstruct.Fields
		expectedPool                  string
	}{
		"invalid-one-node-and-valid-other": {
			slices: []*st.ResourceSliceWrapper{
				func() *st.ResourceSliceWrapper {
					invalidPoolSlice := st.MakeResourceSlice("worker-0", driverNamePlaceholder).Devices("device-1")
					invalidPoolSlice.Name += "-1"
					invalidPoolSlice.Spec.Pool.ResourceSliceCount = 2
					return invalidPoolSlice
				}(),
				func() *st.ResourceSliceWrapper {
					invalidPoolSlice := st.MakeResourceSlice("worker-0", driverNamePlaceholder).Devices("device-1")
					invalidPoolSlice.Name += "-2"
					invalidPoolSlice.Spec.Pool.ResourceSliceCount = 2
					return invalidPoolSlice
				}(),
				func() *st.ResourceSliceWrapper {
					validPoolSlice := st.MakeResourceSlice("worker-1", driverNamePlaceholder).Devices("device-1")
					return validPoolSlice
				}(),
			},
			expectPodToSchedule: true,
			expectedPodScheduledCondition: gstruct.Fields{
				"Type":   gomega.Equal(v1.PodScheduled),
				"Status": gomega.Equal(v1.ConditionTrue),
			},
			expectedPool: "worker-1",
		},
		"only-invalid-for-one-node": {
			slices: []*st.ResourceSliceWrapper{
				func() *st.ResourceSliceWrapper {
					invalidPoolSlice := st.MakeResourceSlice("worker-0", driverNamePlaceholder).Devices("device-1")
					invalidPoolSlice.Name += "-1"
					invalidPoolSlice.Spec.Pool.ResourceSliceCount = 2
					return invalidPoolSlice
				}(),
				func() *st.ResourceSliceWrapper {
					invalidPoolSlice := st.MakeResourceSlice("worker-0", driverNamePlaceholder).Devices("device-1")
					invalidPoolSlice.Name += "-2"
					invalidPoolSlice.Spec.Pool.ResourceSliceCount = 2
					return invalidPoolSlice
				}(),
			},
			expectPodToSchedule: false,
			expectedPodScheduledCondition: gstruct.Fields{
				"Type":    gomega.Equal(v1.PodScheduled),
				"Status":  gomega.Equal(v1.ConditionFalse),
				"Message": gomega.Equal("0/8 nodes are available: 1 invalid resource pools were encountered, 7 cannot allocate all claims. still not schedulable, preemption: 0/8 nodes are available: 8 Preemption is not helpful for scheduling."),
			},
		},
		"invalid-for-all-nodes": {
			slices: func() []*st.ResourceSliceWrapper {
				var slices []*st.ResourceSliceWrapper
				for i := range 8 {
					nodeName := fmt.Sprintf("worker-%d", i)
					invalidPoolSlice1 := st.MakeResourceSlice(nodeName, driverNamePlaceholder).Devices("device-1")
					invalidPoolSlice1.Name += "-1"
					invalidPoolSlice1.Spec.Pool.ResourceSliceCount = 2

					invalidPoolSlice2 := st.MakeResourceSlice(nodeName, driverNamePlaceholder).Devices("device-1")
					invalidPoolSlice2.Name += "-2"
					invalidPoolSlice2.Spec.Pool.ResourceSliceCount = 2
					slices = append(slices, invalidPoolSlice1, invalidPoolSlice2)
				}
				return slices
			}(),
			expectPodToSchedule: false,
			expectedPodScheduledCondition: gstruct.Fields{
				"Type":    gomega.Equal(v1.PodScheduled),
				"Status":  gomega.Equal(v1.ConditionFalse),
				"Message": gomega.Equal("0/8 nodes are available: 8 invalid resource pools were encountered. still not schedulable, preemption: 0/8 nodes are available: 8 Preemption is not helpful for scheduling."),
			},
		},
	}

	for tn, tc := range testCases {
		runSubTest(tCtx, tn, func(tCtx ktesting.TContext) {
			namespace := createTestNamespace(tCtx, nil)
			class, driverName := createTestClass(tCtx, namespace)
			for _, slice := range tc.slices {
				// update the driver since we don't know the actual name until the
				// class is created.
				slice.Spec.Driver = driverName
				createSlice(tCtx, slice.Obj())
			}

			claim := createClaim(tCtx, namespace, "", class, claim)
			pod := createPod(tCtx, namespace, "", podWithClaimName, claim)
			schedulingAttempted := gomega.HaveField("Status.Conditions", gomega.ContainElement(
				gstruct.MatchFields(gstruct.IgnoreExtras, tc.expectedPodScheduledCondition),
			))
			tCtx.Eventually(func(tCtx ktesting.TContext) (*v1.Pod, error) {
				return tCtx.Client().CoreV1().Pods(namespace).Get(tCtx, pod.Name, metav1.GetOptions{})
			}).WithTimeout(schedulingTimeout).WithPolling(time.Second).Should(schedulingAttempted)

			// Only check the ResourceClaim if we expected the Pod to schedule.
			if tc.expectPodToSchedule {
				tCtx.Eventually(func(tCtx ktesting.TContext) (*resourceapi.ResourceClaim, error) {
					return tCtx.Client().ResourceV1().ResourceClaims(namespace).Get(tCtx, claim.Name, metav1.GetOptions{})
				}).WithTimeout(schedulingTimeout).WithPolling(time.Second).Should(gomega.HaveField("Status.Allocation", gstruct.PointTo(gstruct.MatchFields(gstruct.IgnoreExtras, gstruct.Fields{
					"Devices": gstruct.MatchFields(gstruct.IgnoreExtras, gstruct.Fields{
						"Results": gomega.HaveExactElements(gstruct.MatchFields(gstruct.IgnoreExtras, gstruct.Fields{
							"Driver": gomega.Equal(driverName),
							"Pool":   gomega.Equal(tc.expectedPool),
						})),
					}),
				}))))
			}
		})
	}
}
