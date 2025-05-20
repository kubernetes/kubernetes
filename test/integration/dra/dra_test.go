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

package dra

import (
	"context"
	"errors"
	"fmt"
	"regexp"
	"sort"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	"github.com/onsi/gomega"
	"github.com/onsi/gomega/gstruct"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	v1 "k8s.io/api/core/v1"
	resourcealphaapi "k8s.io/api/resource/v1alpha3"
	resourceapi "k8s.io/api/resource/v1beta1"
	resourcev1beta2 "k8s.io/api/resource/v1beta2"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/component-base/featuregate"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/dynamic-resource-allocation/resourceslice"
	"k8s.io/klog/v2"
	kubeschedulerconfigv1 "k8s.io/kube-scheduler/config/v1"
	kubeapiservertesting "k8s.io/kubernetes/cmd/kube-apiserver/app/testing"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/scheduler/apis/config"
	kubeschedulerscheme "k8s.io/kubernetes/pkg/scheduler/apis/config/scheme"
	st "k8s.io/kubernetes/pkg/scheduler/testing"
	"k8s.io/kubernetes/test/integration/framework"
	"k8s.io/kubernetes/test/integration/util"
	"k8s.io/kubernetes/test/utils/format"
	"k8s.io/kubernetes/test/utils/ktesting"
	"k8s.io/utils/ptr"
)

var (
	// For more test data see pkg/scheduler/framework/plugin/dynamicresources/dynamicresources_test.go.

	podName          = "my-pod"
	namespace        = "default"
	resourceName     = "my-resource"
	className        = "my-resource-class"
	claimName        = podName + "-" + resourceName
	podWithClaimName = st.MakePod().Name(podName).Namespace(namespace).
				Container("my-container").
				PodResourceClaims(v1.PodResourceClaim{Name: resourceName, ResourceClaimName: &claimName}).
				Obj()
	class = &resourceapi.DeviceClass{
		ObjectMeta: metav1.ObjectMeta{
			Name: className,
		},
	}
	claim = st.MakeResourceClaim().
		Name(claimName).
		Namespace(namespace).
		Request(className).
		Obj()
	claimPrioritizedList = st.MakeResourceClaim().
				Name(claimName).
				Namespace(namespace).
				RequestWithPrioritizedList(className).
				Obj()

	numNodes = 2
)

// createTestNamespace creates a namespace with a name that is derived from the
// current test name:
// - Non-alpha-numeric characters replaced by hyphen.
// - Truncated in the middle to make it short enough for GenerateName.
// - Hyphen plus random suffix added by the apiserver.
func createTestNamespace(tCtx ktesting.TContext, labels map[string]string) string {
	tCtx.Helper()
	name := regexp.MustCompile(`[^[:alnum:]_-]`).ReplaceAllString(tCtx.Name(), "-")
	name = strings.ToLower(name)
	if len(name) > 63 {
		name = name[:30] + "--" + name[len(name)-30:]
	}
	ns := &v1.Namespace{ObjectMeta: metav1.ObjectMeta{GenerateName: name + "-"}}
	ns.Labels = labels
	ns, err := tCtx.Client().CoreV1().Namespaces().Create(tCtx, ns, metav1.CreateOptions{})
	tCtx.ExpectNoError(err, "create test namespace")
	tCtx.CleanupCtx(func(tCtx ktesting.TContext) {
		tCtx.ExpectNoError(tCtx.Client().CoreV1().Namespaces().Delete(tCtx, ns.Name, metav1.DeleteOptions{}), "delete test namespace")
	})
	return ns.Name
}

func TestDRA(t *testing.T) {
	// Each sub-test brings up the API server in a certain
	// configuration. These sub-tests must run sequentially because they
	// change the global DefaultFeatureGate. For each configuration,
	// multiple tests can run in parallel as long as they are careful
	// about what they create.
	//
	// Each configuration starts with two Nodes (ready, sufficient RAM and CPU for multiple pods)
	// and no ResourceSlices. To test scheduling, a sub-test must create ResourceSlices.
	// createTestNamespace can be used to create a unique per-test namespace. The name of that
	// namespace then can be used to create cluster-scoped objects without conflicts between tests.
	for name, tc := range map[string]struct {
		apis     map[schema.GroupVersion]bool
		features map[featuregate.Feature]bool
		f        func(tCtx ktesting.TContext)
	}{
		"default": {
			f: func(tCtx ktesting.TContext) {
				tCtx.Run("Pod", func(tCtx ktesting.TContext) { testPod(tCtx, false) })
				tCtx.Run("APIDisabled", testAPIDisabled)
			},
		},
		"core": {
			apis: map[schema.GroupVersion]bool{
				resourceapi.SchemeGroupVersion:     true,
				resourcev1beta2.SchemeGroupVersion: true,
			},
			features: map[featuregate.Feature]bool{features.DynamicResourceAllocation: true},
			f: func(tCtx ktesting.TContext) {
				tCtx.Run("AdminAccess", func(tCtx ktesting.TContext) { testAdminAccess(tCtx, false) })
				tCtx.Run("PrioritizedList", func(tCtx ktesting.TContext) { testPrioritizedList(tCtx, false) })
				tCtx.Run("Pod", func(tCtx ktesting.TContext) { testPod(tCtx, true) })
				tCtx.Run("PublishResourceSlices", func(tCtx ktesting.TContext) {
					testPublishResourceSlices(tCtx, features.DRADeviceTaints, features.DRAPartitionableDevices)
				})
			},
		},
		"v1beta1": {
			apis: map[schema.GroupVersion]bool{
				resourceapi.SchemeGroupVersion: true,
			},
			features: map[featuregate.Feature]bool{features.DynamicResourceAllocation: true},
			f: func(tCtx ktesting.TContext) {
				tCtx.Run("PublishResourceSlices", func(tCtx ktesting.TContext) {
					testPublishResourceSlices(tCtx, features.DRADeviceTaints, features.DRAPartitionableDevices)
				})
			},
		},
		"v1beta2": {
			apis: map[schema.GroupVersion]bool{
				resourcev1beta2.SchemeGroupVersion: true,
			},
			features: map[featuregate.Feature]bool{features.DynamicResourceAllocation: true},
			f: func(tCtx ktesting.TContext) {
				tCtx.Run("PublishResourceSlices", func(tCtx ktesting.TContext) {
					testPublishResourceSlices(tCtx, features.DRADeviceTaints, features.DRAPartitionableDevices)
				})
			},
		},
		"all": {
			apis: map[schema.GroupVersion]bool{
				resourceapi.SchemeGroupVersion:      true,
				resourcev1beta2.SchemeGroupVersion:  true,
				resourcealphaapi.SchemeGroupVersion: true,
			},
			features: map[featuregate.Feature]bool{
				features.DynamicResourceAllocation: true,
				// Additional DRA feature gates go here,
				// in alphabetical order,
				// as needed by tests for them.
				features.DRAAdminAccess:          true,
				features.DRADeviceTaints:         true,
				features.DRAPartitionableDevices: true,
				features.DRAPrioritizedList:      true,
			},
			f: func(tCtx ktesting.TContext) {
				tCtx.Run("AdminAccess", func(tCtx ktesting.TContext) { testAdminAccess(tCtx, true) })
				tCtx.Run("Convert", testConvert)
				tCtx.Run("PrioritizedList", func(tCtx ktesting.TContext) { testPrioritizedList(tCtx, true) })
				tCtx.Run("PublishResourceSlices", func(tCtx ktesting.TContext) { testPublishResourceSlices(tCtx) })
				tCtx.Run("MaxResourceSlice", testMaxResourceSlice)
			},
		},
	} {
		t.Run(name, func(t *testing.T) {
			tCtx := ktesting.Init(t)
			var entries []string
			for key, value := range tc.features {
				entries = append(entries, fmt.Sprintf("%s=%t", key, value))
			}
			for key, value := range tc.apis {
				entries = append(entries, fmt.Sprintf("%s=%t", key, value))
			}
			sort.Strings(entries)
			t.Logf("Config: %s", strings.Join(entries, ","))

			for key, value := range tc.features {
				featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, key, value)
			}

			etcdOptions := framework.SharedEtcd()
			apiServerOptions := kubeapiservertesting.NewDefaultTestServerOptions()
			apiServerFlags := framework.DefaultTestServerFlags()
			var runtimeConfigs []string
			for key, value := range tc.apis {
				runtimeConfigs = append(runtimeConfigs, fmt.Sprintf("%s=%t", key, value))
			}
			apiServerFlags = append(apiServerFlags, "--runtime-config="+strings.Join(runtimeConfigs, ","))
			server := kubeapiservertesting.StartTestServerOrDie(t, apiServerOptions, apiServerFlags, etcdOptions)
			tCtx.Cleanup(server.TearDownFn)
			tCtx = ktesting.WithRESTConfig(tCtx, server.ClientConfig)

			createNodes(tCtx)

			tc.f(tCtx)
		})
	}
}

func createNodes(tCtx ktesting.TContext) {
	for i := 0; i < numNodes; i++ {
		// Create node.
		node := &v1.Node{
			ObjectMeta: metav1.ObjectMeta{
				Name: fmt.Sprintf("worker-%d", i),
			},
		}
		node, err := tCtx.Client().CoreV1().Nodes().Create(tCtx, node, metav1.CreateOptions{})
		tCtx.ExpectNoError(err, fmt.Sprintf("creating node #%d", i))

		// Make the node ready.
		node.Status = v1.NodeStatus{
			Capacity: v1.ResourceList{
				v1.ResourceCPU:    resource.MustParse("100"),
				v1.ResourceMemory: resource.MustParse("1000"),
				v1.ResourcePods:   resource.MustParse("100"),
			},
			Phase: v1.NodeRunning,
			Conditions: []v1.NodeCondition{
				{
					Type:   v1.NodeReady,
					Status: v1.ConditionTrue,
				},
			},
		}
		node, err = tCtx.Client().CoreV1().Nodes().UpdateStatus(tCtx, node, metav1.UpdateOptions{})
		tCtx.ExpectNoError(err, fmt.Sprintf("setting status of node #%d", i))

		// Remove taint added by TaintNodesByCondition admission check.
		node.Spec.Taints = nil
		_, err = tCtx.Client().CoreV1().Nodes().Update(tCtx, node, metav1.UpdateOptions{})
		tCtx.ExpectNoError(err, fmt.Sprintf("removing node taint from #%d", i))
	}

	tCtx.CleanupCtx(func(tCtx ktesting.TContext) {
		if !tCtx.Failed() {
			return
		}

		// Dump information about the cluster.
		nodes, err := tCtx.Client().CoreV1().Nodes().List(tCtx, metav1.ListOptions{})
		if err != nil {
			tCtx.Logf("Retrieving nodes failed: %v", err)
		} else {
			tCtx.Logf("Nodes:\n%s", format.Object(nodes.Items, 1))
		}
	})
}

func startScheduler(tCtx ktesting.TContext) {
	// Run scheduler with default configuration.
	tCtx.Log("Scheduler starting...")
	schedulerCtx := klog.NewContext(tCtx, klog.LoggerWithName(tCtx.Logger(), "scheduler"))
	schedulerCtx, cancel := context.WithCancelCause(schedulerCtx)
	_, informerFactory := util.StartScheduler(schedulerCtx, tCtx.Client(), tCtx.RESTConfig(), newDefaultSchedulerComponentConfig(tCtx), nil)
	// Stop clients of the apiserver before stopping the apiserver itself,
	// otherwise it delays its shutdown.
	tCtx.Cleanup(informerFactory.Shutdown)
	tCtx.Cleanup(func() {
		tCtx.Log("Stoping scheduler...")
		cancel(errors.New("test is done"))
	})
}

func newDefaultSchedulerComponentConfig(tCtx ktesting.TContext) *config.KubeSchedulerConfiguration {
	gvk := kubeschedulerconfigv1.SchemeGroupVersion.WithKind("KubeSchedulerConfiguration")
	cfg := config.KubeSchedulerConfiguration{}
	_, _, err := kubeschedulerscheme.Codecs.UniversalDecoder().Decode(nil, &gvk, &cfg)
	tCtx.ExpectNoError(err, "decode default scheduler configuration")
	return &cfg
}

// testPod creates a pod with a resource claim reference and then checks
// whether that field is or isn't getting dropped.
func testPod(tCtx ktesting.TContext, draEnabled bool) {
	tCtx.Parallel()
	namespace := createTestNamespace(tCtx, nil)
	podWithClaimName := podWithClaimName.DeepCopy()
	podWithClaimName.Namespace = namespace
	pod, err := tCtx.Client().CoreV1().Pods(namespace).Create(tCtx, podWithClaimName, metav1.CreateOptions{})
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
	_, err := tCtx.Client().ResourceV1beta1().ResourceClaims(claim.Namespace).Create(tCtx, claim, metav1.CreateOptions{})
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
	claim, err := tCtx.Client().ResourceV1beta1().ResourceClaims(namespace).Create(tCtx, claim, metav1.CreateOptions{})
	tCtx.ExpectNoError(err, "create claim")
	claimAlpha, err := tCtx.Client().ResourceV1alpha3().ResourceClaims(namespace).Get(tCtx, claim.Name, metav1.GetOptions{})
	tCtx.ExpectNoError(err, "get claim")
	// We could check more fields, but there are unit tests which cover this better.
	assert.Equal(tCtx, claim.Name, claimAlpha.Name, "claim name")
}

// testAdminAccess creates a claim with AdminAccess and then checks
// whether that field is or isn't getting dropped.
// when the AdminAccess feature is enabled, it also checks that the field
// is only allowed to be used in namespace with the Resource Admin Access label
func testAdminAccess(tCtx ktesting.TContext, adminAccessEnabled bool) {
	namespace := createTestNamespace(tCtx, nil)
	claim1 := claim.DeepCopy()
	claim1.Namespace = namespace
	claim1.Spec.Devices.Requests[0].AdminAccess = ptr.To(true)
	// create claim with AdminAccess in non-admin namespace
	_, err := tCtx.Client().ResourceV1beta1().ResourceClaims(namespace).Create(tCtx, claim1, metav1.CreateOptions{})
	if adminAccessEnabled {
		if err != nil {
			// should result in validation error
			assert.ErrorContains(tCtx, err, "admin access to devices requires the `resource.k8s.io/admin-access: true` label on the containing namespace", "the error message should have contained the expected error message")
			return
		} else {
			tCtx.Fatal("expected validation error(s), got none")
		}

		// create claim with AdminAccess in admin namespace
		adminNS := createTestNamespace(tCtx, map[string]string{"resource.k8s.io/admin-access": "true"})
		claim2 := claim.DeepCopy()
		claim2.Namespace = adminNS
		claim2.Name = "claim2"
		claim2.Spec.Devices.Requests[0].AdminAccess = ptr.To(true)
		claim2, err := tCtx.Client().ResourceV1beta1().ResourceClaims(adminNS).Create(tCtx, claim2, metav1.CreateOptions{})
		tCtx.ExpectNoError(err, "create claim")
		if !ptr.Deref(claim2.Spec.Devices.Requests[0].AdminAccess, true) {
			tCtx.Fatalf("should store AdminAccess in ResourceClaim %v", claim2)
		}
	} else {
		if claim.Spec.Devices.Requests[0].AdminAccess != nil {
			tCtx.Fatal("should drop AdminAccess in ResourceClaim")
		}
	}
}

func testPrioritizedList(tCtx ktesting.TContext, enabled bool) {
	tCtx.Parallel()
	namespace := createTestNamespace(tCtx, nil)
	class := class.DeepCopy()
	class.Name = namespace
	class.Spec.Selectors = []resourceapi.DeviceSelector{{
		CEL: &resourceapi.CELDeviceSelector{
			Expression: fmt.Sprintf("device.driver == %q", namespace),
		},
	}}
	_, err := tCtx.Client().ResourceV1beta1().DeviceClasses().Create(tCtx, class, metav1.CreateOptions{})
	tCtx.ExpectNoError(err, "create class")
	tCtx.CleanupCtx(func(tCtx ktesting.TContext) {
		err := tCtx.Client().ResourceV1beta1().DeviceClasses().Delete(tCtx, class.Name, metav1.DeleteOptions{})
		tCtx.ExpectNoError(err, "delete class")
	})
	claim := claimPrioritizedList.DeepCopy()
	claim.Namespace = namespace
	claim.Spec.Devices.Requests[0].FirstAvailable[0].DeviceClassName = class.Name
	claim, err = tCtx.Client().ResourceV1beta1().ResourceClaims(namespace).Create(tCtx, claim, metav1.CreateOptions{})

	if !enabled {
		require.Error(tCtx, err, "claim should have become invalid after dropping FirstAvailable")
		return
	}

	require.NotEmpty(tCtx, claim.Spec.Devices.Requests[0].FirstAvailable, "should store FirstAvailable")
	tCtx.Run("scheduler", func(tCtx ktesting.TContext) {
		startScheduler(tCtx)

		// We could create ResourceSlices for some node with the right driver.
		// But failing during Filter is sufficient to determine that it did
		// not fail during PreFilter because of FirstAvailable.
		pod := podWithClaimName.DeepCopy()
		pod.Namespace = namespace
		_, err := tCtx.Client().CoreV1().Pods(namespace).Create(tCtx, pod, metav1.CreateOptions{})
		tCtx.ExpectNoError(err, "create pod")
		schedulingAttempted := gomega.HaveField("Status.Conditions", gomega.ContainElement(
			gstruct.MatchFields(gstruct.IgnoreExtras, gstruct.Fields{
				"Type":    gomega.Equal(v1.PodScheduled),
				"Status":  gomega.Equal(v1.ConditionFalse),
				"Reason":  gomega.Equal("Unschedulable"),
				"Message": gomega.Equal("0/2 nodes are available: 2 cannot allocate all claims. still not schedulable, preemption: 0/2 nodes are available: 2 Preemption is not helpful for scheduling."),
			}),
		))
		ktesting.Eventually(tCtx, func(tCtx ktesting.TContext) *v1.Pod {
			pod, err := tCtx.Client().CoreV1().Pods(namespace).Get(tCtx, pod.Name, metav1.GetOptions{})
			tCtx.ExpectNoError(err, "get pod")
			return pod
		}).WithTimeout(10 * time.Second).WithPolling(time.Second).Should(schedulingAttempted)
	})
}

func testPublishResourceSlices(tCtx ktesting.TContext, disabledFeatures ...featuregate.Feature) {
	tCtx.Parallel()

	tCtx = ktesting.WithTimeout(tCtx, 30*time.Second, "test timed out")
	driverName := "dra.example.com"
	poolName := "global"
	resources := &resourceslice.DriverResources{
		Pools: map[string]resourceslice.Pool{
			poolName: {
				Slices: []resourceslice.Slice{
					{
						Devices: []resourcev1beta2.Device{
							{
								Name: "device-simple",
							},
						},
					},
					{
						SharedCounters: []resourcev1beta2.CounterSet{{
							Name: "gpu-0",
							Counters: map[string]resourcev1beta2.Counter{
								"mem": {Value: resource.MustParse("1")},
							},
						}},
						Devices: []resourcev1beta2.Device{
							{
								Name: "device-tainted-default",
								Taints: []resourcev1beta2.DeviceTaint{{
									Key:    "dra.example.com/taint",
									Value:  "taint-value",
									Effect: resourcev1beta2.DeviceTaintEffectNoExecute,
									// TimeAdded is added by apiserver.
								}},
							},
							{
								Name: "device-tainted-time-added",
								Taints: []resourcev1beta2.DeviceTaint{{
									Key:       "dra.example.com/taint",
									Value:     "taint-value",
									Effect:    resourcev1beta2.DeviceTaintEffectNoExecute,
									TimeAdded: ptr.To(metav1.Now()),
								}},
							},
							{
								Name: "gpu",
								ConsumesCounters: []resourcev1beta2.DeviceCounterConsumption{{
									CounterSet: "gpu-0",
									Counters: map[string]resourcev1beta2.Counter{
										"mem": {Value: resource.MustParse("1")},
									},
								}},
							},
						},
					},
				},
			},
		},
	}
	var gotDroppedFieldError atomic.Bool
	var gotValidationError atomic.Bool
	var validationErrorsOkay atomic.Bool
	opts := resourceslice.Options{
		DriverName: driverName,
		KubeClient: tCtx.Client(),
		SyncDelay:  ptr.To(0 * time.Second),
		Resources:  resources,
		ErrorHandler: func(ctx context.Context, err error, msg string) {
			klog.FromContext(ctx).Info("ErrorHandler called", "err", err, "msg", msg)
			if !validationErrorsOkay.Load() && len(disabledFeatures) == 0 {
				assert.NoError(tCtx, err, msg)
				return
			}

			var droppedFields *resourceslice.DroppedFieldsError
			if errors.As(err, &droppedFields) {
				var disabled []string
				for _, feature := range disabledFeatures {
					disabled = append(disabled, string(feature))
				}
				assert.ErrorContains(tCtx, err, fmt.Sprintf("pool %q, slice #1: some fields were dropped by the apiserver, probably because these features are disabled: %s", poolName, strings.Join(disabled, " ")))
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
	defer controller.Stop()

	// Two create calls should be all that are needed.
	expectedStats := resourceslice.Stats{
		NumCreates: 2,
	}
	getStats := func(tCtx ktesting.TContext) resourceslice.Stats {
		return controller.GetStats()
	}
	ktesting.Eventually(tCtx, getStats).WithTimeout(10 * time.Second).Should(gomega.Equal(expectedStats))

	// No further changes necessary.
	ktesting.Consistently(tCtx, getStats).WithTimeout(10 * time.Second).Should(gomega.Equal(expectedStats))

	if len(disabledFeatures) > 0 && !gotDroppedFieldError.Load() {
		tCtx.Error("expected dropped fields error, got none")
	}

	// Now switch to one invalid slice.
	pool := resources.Pools[poolName]
	pool.Slices = pool.Slices[:1]
	pool.Slices[0].Devices[0].Attributes = map[resourcev1beta2.QualifiedName]resourcev1beta2.DeviceAttribute{"empty": {}}
	resources.Pools[poolName] = pool
	validationErrorsOkay.Store(true)
	controller.Update(resources)
	ktesting.Eventually(tCtx, getStats).WithTimeout(10*time.Second).Should(gomega.HaveField("NumDeletes", gomega.BeNumerically(">=", int64(1))), "Slice should have been removed.")
	ktesting.Eventually(tCtx, func(tCtx ktesting.TContext) bool {
		return gotValidationError.Load()
	}).WithTimeout(time.Minute).Should(gomega.BeTrueBecause("Should have gotten another error because the slice is invalid."))

}

// testMaxResourceSlice creates a ResourceSlice that is as large as possible
// and prints some information about it.
func testMaxResourceSlice(tCtx ktesting.TContext) {
	slice := NewMaxResourceSlice()
	createdSlice, err := tCtx.Client().ResourceV1beta2().ResourceSlices().Create(tCtx, slice, metav1.CreateOptions{})
	tCtx.ExpectNoError(err)
	totalSize := createdSlice.Size()
	var managedFieldsSize int
	for _, f := range createdSlice.ManagedFields {
		managedFieldsSize += f.Size()
	}
	specSize := createdSlice.Spec.Size()
	tCtx.Logf("\n\nTotal size: %s\nManagedFields size: %s (%.0f%%)\nSpec size: %s (%.0f)%%\n\nManagedFields:\n%s",
		resource.NewQuantity(int64(totalSize), resource.BinarySI),
		resource.NewQuantity(int64(managedFieldsSize), resource.BinarySI), float64(managedFieldsSize)*100/float64(totalSize),
		resource.NewQuantity(int64(specSize), resource.BinarySI), float64(specSize)*100/float64(totalSize),
		klog.Format(createdSlice.ManagedFields),
	)
	if diff := cmp.Diff(slice.Spec, createdSlice.Spec); diff != "" {
		tCtx.Errorf("ResourceSliceSpec got modified during Create (- want, + got):\n%s", diff)
	}
}
