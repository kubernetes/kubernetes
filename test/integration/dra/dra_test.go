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
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"math/rand/v2"
	"sort"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	"github.com/onsi/gomega"
	"github.com/onsi/gomega/gstruct"
	gtypes "github.com/onsi/gomega/types"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	v1 "k8s.io/api/core/v1"
	resourceapi "k8s.io/api/resource/v1"
	resourcealphaapi "k8s.io/api/resource/v1alpha3"
	resourcev1beta1 "k8s.io/api/resource/v1beta1"
	resourcev1beta2 "k8s.io/api/resource/v1beta2"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/version"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	resourceapiac "k8s.io/client-go/applyconfigurations/resource/v1"
	"k8s.io/client-go/informers"
	"k8s.io/client-go/tools/cache"
	"k8s.io/component-base/featuregate"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/component-base/metrics/testutil"
	draclient "k8s.io/dynamic-resource-allocation/client"
	"k8s.io/dynamic-resource-allocation/resourceslice"
	"k8s.io/klog/v2"
	kubeschedulerconfigv1 "k8s.io/kube-scheduler/config/v1"
	kubeapiservertesting "k8s.io/kubernetes/cmd/kube-apiserver/app/testing"
	resourceclaimmetrics "k8s.io/kubernetes/pkg/controller/resourceclaim/metrics"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/scheduler/apis/config"
	kubeschedulerscheme "k8s.io/kubernetes/pkg/scheduler/apis/config/scheme"
	st "k8s.io/kubernetes/pkg/scheduler/testing"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	"k8s.io/kubernetes/test/integration/framework"
	"k8s.io/kubernetes/test/integration/util"
	"k8s.io/kubernetes/test/utils/format"
	"k8s.io/kubernetes/test/utils/ktesting"
	"k8s.io/utils/ptr"
)

var (
	// For more test data see pkg/scheduler/framework/plugin/dynamicresources/dynamicresources_test.go.

	podName                     = "my-pod"
	podWithExtendedResourceName = "my-pod-with-extended-resource"
	namespace                   = "default"
	resourceName                = "my-resource"
	extendedResourceName        = "my-example.com/my-extended-resource"
	claimName                   = podName + "-" + resourceName
	className                   = "my-resource-class"
	extendedClassName           = "my-extended-resource-class"
	device1                     = "device-1"
	device2                     = "device-2"
	podWithClaimName            = st.MakePod().Name(podName).Namespace(namespace).
					Container("my-container").
					PodResourceClaims(v1.PodResourceClaim{Name: resourceName, ResourceClaimName: &claimName}).
					Obj()
	podWithExtendedResource = st.MakePod().Name(podWithExtendedResourceName).Namespace(namespace).
				Container("my-container").
				Res(map[v1.ResourceName]string{v1.ResourceName(extendedResourceName): "1"}).
				Obj()
	class = &resourceapi.DeviceClass{
		ObjectMeta: metav1.ObjectMeta{
			Name: className,
		},
	}
	classWithExtendedResource = &resourceapi.DeviceClass{
		ObjectMeta: metav1.ObjectMeta{
			Name: extendedClassName,
		},
		Spec: resourceapi.DeviceClassSpec{
			ExtendedResourceName: &extendedResourceName,
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
				RequestWithPrioritizedList(st.SubRequest("subreq-1", className, 1)).
				Obj()

	numNodes = 8
)

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
		"disabled": {
			apis:     map[schema.GroupVersion]bool{resourceapi.SchemeGroupVersion: false},
			features: map[featuregate.Feature]bool{features.DynamicResourceAllocation: false},
			f: func(tCtx ktesting.TContext) {
				tCtx.Run("APIDisabled", testAPIDisabled)
				tCtx.Run("Pod", func(tCtx ktesting.TContext) { testPod(tCtx, false) })
			},
		},
		"default": {
			apis:     map[schema.GroupVersion]bool{},
			features: map[featuregate.Feature]bool{},
			f: func(tCtx ktesting.TContext) {
				tCtx.Run("Pod", func(tCtx ktesting.TContext) { testPod(tCtx, true) })
				tCtx.Run("FilterTimeout", testFilterTimeout)
			},
		},
		"GA": {
			apis: map[schema.GroupVersion]bool{},
			features: map[featuregate.Feature]bool{
				featuregate.Feature("AllBeta"): false,
			},
			f: func(tCtx ktesting.TContext) {
				tCtx.Run("AdminAccess", func(tCtx ktesting.TContext) { testAdminAccess(tCtx, false) })
				tCtx.Run("PrioritizedList", func(tCtx ktesting.TContext) { testPrioritizedList(tCtx, false) })
				tCtx.Run("Pod", func(tCtx ktesting.TContext) { testPod(tCtx, true) })
				tCtx.Run("PublishResourceSlices", func(tCtx ktesting.TContext) {
					testPublishResourceSlices(tCtx, true, features.DRADeviceTaints, features.DRAPartitionableDevices, features.DRADeviceBindingConditions)
				})
				tCtx.Run("ExtendedResource", func(tCtx ktesting.TContext) { testExtendedResource(tCtx, false) })
				tCtx.Run("ResourceClaimDeviceStatus", func(tCtx ktesting.TContext) { testResourceClaimDeviceStatus(tCtx, false) })
				tCtx.Run("DeviceBindingConditions", func(tCtx ktesting.TContext) { testDeviceBindingConditions(tCtx, false) })
				tCtx.Run("ResourceSliceController", func(tCtx ktesting.TContext) {
					namespace := createTestNamespace(tCtx, nil)
					tCtx = tCtx.WithNamespace(namespace)
					TestCreateResourceSlices(tCtx, 100)
				})
			},
		},
		"v1beta1": {
			apis: map[schema.GroupVersion]bool{
				resourceapi.SchemeGroupVersion:     false,
				resourcev1beta1.SchemeGroupVersion: true,
			},
			features: map[featuregate.Feature]bool{features.DynamicResourceAllocation: true},
			f: func(tCtx ktesting.TContext) {
				tCtx.Run("PublishResourceSlices", func(tCtx ktesting.TContext) {
					testPublishResourceSlices(tCtx, false, features.DRADeviceTaints, features.DRAPartitionableDevices, features.DRADeviceBindingConditions)
				})
			},
		},
		"v1beta2": {
			apis: map[schema.GroupVersion]bool{
				resourceapi.SchemeGroupVersion:     false,
				resourcev1beta2.SchemeGroupVersion: true,
			},
			features: map[featuregate.Feature]bool{features.DynamicResourceAllocation: true},
			f: func(tCtx ktesting.TContext) {
				tCtx.Run("PublishResourceSlices", func(tCtx ktesting.TContext) {
					testPublishResourceSlices(tCtx, false, features.DRADeviceTaints, features.DRAPartitionableDevices, features.DRADeviceBindingConditions)
				})
			},
		},
		"slice-taints": {
			features: map[featuregate.Feature]bool{
				features.DRADeviceTaints: true,
			},
			f: func(tCtx ktesting.TContext) {
				tCtx.Run("EvictClusterWithSlices", func(tCtx ktesting.TContext) { testEvictCluster(tCtx, false) })
			},
		},
		"all": {
			apis: map[schema.GroupVersion]bool{
				resourcev1beta1.SchemeGroupVersion:  true,
				resourcev1beta2.SchemeGroupVersion:  true,
				resourcealphaapi.SchemeGroupVersion: true,
			},
			features: map[featuregate.Feature]bool{
				// Additional DRA feature gates go here,
				// in alphabetical order,
				// as needed by tests for them.
				features.DRAAdminAccess:               true,
				features.DRADeviceBindingConditions:   true,
				features.DRAConsumableCapacity:        true,
				features.DRADeviceTaints:              true,
				features.DRADeviceTaintRules:          true,
				features.DRAPartitionableDevices:      true,
				features.DRAPrioritizedList:           true,
				features.DRAResourceClaimDeviceStatus: true,
				features.DRAExtendedResource:          true,
			},
			f: func(tCtx ktesting.TContext) {
				tCtx.Run("AdminAccess", func(tCtx ktesting.TContext) { testAdminAccess(tCtx, true) })
				tCtx.Run("Convert", testConvert)
				tCtx.Run("ControllerManagerMetrics", testControllerManagerMetrics)
				tCtx.Run("DeviceBindingConditions", func(tCtx ktesting.TContext) { testDeviceBindingConditions(tCtx, true) })
				tCtx.Run("PrioritizedList", func(tCtx ktesting.TContext) { testPrioritizedList(tCtx, true) })
				tCtx.Run("PrioritizedListScoring", func(tCtx ktesting.TContext) { testPrioritizedListScoring(tCtx) })
				tCtx.Run("PublishResourceSlices", func(tCtx ktesting.TContext) { testPublishResourceSlices(tCtx, true) })
				// note testExtendedResource depends on testPublishResourceSlices to provide the devices
				tCtx.Run("ExtendedResource", func(tCtx ktesting.TContext) { testExtendedResource(tCtx, true) })
				tCtx.Run("ResourceClaimDeviceStatus", func(tCtx ktesting.TContext) { testResourceClaimDeviceStatus(tCtx, true) })
				tCtx.Run("MaxResourceSlice", testMaxResourceSlice)
				tCtx.Run("EvictClusterWithRule", func(tCtx ktesting.TContext) { testEvictCluster(tCtx, true) })
				tCtx.Run("EvictClusterWithSlices", func(tCtx ktesting.TContext) { testEvictCluster(tCtx, false) })
				tCtx.Run("InvalidResourceSlices", testInvalidResourceSlices)
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

			// We need to set emulation version for DynamicResourceAllocation feature gate, which is locked at 1.35.
			if draEnabled, draExists := tc.features[features.DynamicResourceAllocation]; draExists && !draEnabled {
				featuregatetesting.SetFeatureGateEmulationVersionDuringTest(t, utilfeature.DefaultFeatureGate, version.MustParse("1.34"))
			}
			featuregatetesting.SetFeatureGatesDuringTest(t, utilfeature.DefaultFeatureGate, tc.features)

			etcdOptions := framework.SharedEtcd()
			apiServerOptions := kubeapiservertesting.NewDefaultTestServerOptions()
			apiServerFlags := framework.DefaultTestServerFlags()
			var runtimeConfigs []string
			for key, value := range tc.apis {
				runtimeConfigs = append(runtimeConfigs, fmt.Sprintf("%s=%t", key, value))
			}
			apiServerFlags = append(apiServerFlags, "--runtime-config="+strings.Join(runtimeConfigs, ","))
			server := kubeapiservertesting.StartTestServerOrDie(t, apiServerOptions, apiServerFlags, etcdOptions)
			tCtx.CleanupCtx(func(tCtx ktesting.TContext) {
				tCtx.Log("Stopping the apiserver...")
				server.TearDownFn()
			})
			tCtx = ktesting.WithRESTConfig(tCtx, server.ClientConfig)

			createNodes(tCtx)
			tCtx = prepareScheduler(tCtx)

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
		node, err := tCtx.Client().CoreV1().Nodes().Create(tCtx, node, metav1.CreateOptions{FieldValidation: "Strict"})
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

// prepareScheduler returns a TContext which can be passed to startScheduler
// to actually start the scheduler when there is a demand for it.
//
// Under the hood, schedulerSingleton ensures that at most one scheduler
// instance is created and tears it down when the last test using it is
// done. This could lead to starting and stopping it multiple times, but in
// practice tests start together in parallel and share a single instance.
func prepareScheduler(tCtx ktesting.TContext) ktesting.TContext {
	scheduler := &schedulerSingleton{
		rootCtx: tCtx,
	}

	return ktesting.WithValue(tCtx, schedulerKey, scheduler)
}

// startScheduler starts the scheduler with an empty config, i.e. all settings at their default.
// This may be used in parallel tests.
func startScheduler(tCtx ktesting.TContext) {
	startSchedulerWithConfig(tCtx, "")
}

// startScheduler starts the scheduler with the given config.
// This may be used only in tests which run sequentially if the config is non-empty.
func startSchedulerWithConfig(tCtx ktesting.TContext, config string) {
	tCtx.Helper()
	value := tCtx.Value(schedulerKey)
	if value == nil {
		tCtx.Fatal("internal error: startScheduler without a prior prepareScheduler call")
	}
	scheduler := value.(*schedulerSingleton)
	scheduler.start(tCtx, config)
}

type schedulerSingleton struct {
	rootCtx ktesting.TContext

	mutex           sync.Mutex
	usageCount      int
	informerFactory informers.SharedInformerFactory
	cancel          func(cause string)
}

func (scheduler *schedulerSingleton) start(tCtx ktesting.TContext, config string) {
	tCtx.Helper()
	scheduler.mutex.Lock()
	defer scheduler.mutex.Unlock()

	scheduler.usageCount++
	tCtx.CleanupCtx(scheduler.stop)
	if scheduler.usageCount > 1 {
		// Already started earlier.
		return
	}

	// Run scheduler with default configuration. This must use the root context because
	// the per-test tCtx passed to start will get canceled once the test which triggered
	// starting the scheduler is done.
	tCtx = scheduler.rootCtx
	tCtx.Logf("Starting the scheduler for test %s...", tCtx.Name())
	tCtx = ktesting.WithLogger(tCtx, klog.LoggerWithName(tCtx.Logger(), "scheduler"))
	schedulerCtx := ktesting.WithCancel(tCtx)
	scheduler.cancel = schedulerCtx.Cancel
	cfg := newSchedulerComponentConfig(schedulerCtx, config)
	_, scheduler.informerFactory = util.StartScheduler(schedulerCtx, cfg, nil)
	tCtx.Logf("Started the scheduler for test %s.", tCtx.Name())
}

func (scheduler *schedulerSingleton) stop(tCtx ktesting.TContext) {
	scheduler.mutex.Lock()
	defer scheduler.mutex.Unlock()

	scheduler.usageCount--
	if scheduler.usageCount > 0 {
		// Still in use by some other test.
		return
	}

	scheduler.rootCtx.Logf("Stopping the scheduler after test %s...", tCtx.Name())
	if scheduler.cancel != nil {
		scheduler.cancel("test is done")
	}
	if scheduler.informerFactory != nil {
		scheduler.informerFactory.Shutdown()
	}
}

type schedulerKeyType int

var schedulerKey schedulerKeyType

func newSchedulerComponentConfig(tCtx ktesting.TContext, cfgData string) *config.KubeSchedulerConfiguration {
	tCtx.Helper()
	gvk := kubeschedulerconfigv1.SchemeGroupVersion.WithKind("KubeSchedulerConfiguration")
	cfg := config.KubeSchedulerConfiguration{}
	_, _, err := kubeschedulerscheme.Codecs.UniversalDecoder().Decode([]byte(cfgData), &gvk, &cfg)
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

// testAdminAccess creates a claim with AdminAccess and then checks
// whether that field is or isn't getting dropped.
// when the AdminAccess feature is enabled, it also checks that the field
// is only allowed to be used in namespace with the Resource Admin Access label
func testAdminAccess(tCtx ktesting.TContext, adminAccessEnabled bool) {
	namespace := createTestNamespace(tCtx, nil)
	claim1 := claim.DeepCopy()
	claim1.Namespace = namespace
	claim1.Spec.Devices.Requests[0].Exactly.AdminAccess = ptr.To(true)
	// create claim with AdminAccess in non-admin namespace
	_, err := tCtx.Client().ResourceV1().ResourceClaims(namespace).Create(tCtx, claim1, metav1.CreateOptions{FieldValidation: "Strict"})
	if adminAccessEnabled {
		if err != nil {
			// should result in validation error
			assert.ErrorContains(tCtx, err, "admin access to devices requires the `resource.kubernetes.io/admin-access: true` label on the containing namespace", "the error message should have contained the expected error message")
			return
		} else {
			tCtx.Fatal("expected validation error(s), got none")
		}

		// create claim with AdminAccess in admin namespace
		adminNS := createTestNamespace(tCtx, map[string]string{"resource.kubernetes.io/admin-access": "true"})
		claim2 := claim.DeepCopy()
		claim2.Namespace = adminNS
		claim2.Name = "claim2"
		claim2.Spec.Devices.Requests[0].Exactly.AdminAccess = ptr.To(true)
		claim2, err := tCtx.Client().ResourceV1().ResourceClaims(adminNS).Create(tCtx, claim2, metav1.CreateOptions{FieldValidation: "Strict"})
		tCtx.ExpectNoError(err, "create claim")
		if !ptr.Deref(claim2.Spec.Devices.Requests[0].Exactly.AdminAccess, true) {
			tCtx.Fatalf("should store AdminAccess in ResourceClaim %v", claim2)
		}
	} else {
		if claim.Spec.Devices.Requests[0].Exactly.AdminAccess != nil {
			tCtx.Fatal("should drop AdminAccess in ResourceClaim")
		}
	}
}

// testFilterTimeout covers the scheduler plugin's filter timeout configuration and behavior.
//
// It runs the scheduler with non-standard settings and thus cannot run in parallel.
func testFilterTimeout(tCtx ktesting.TContext) {
	namespace := createTestNamespace(tCtx, nil)
	class, driverName := createTestClass(tCtx, namespace)
	// Chosen so that Filter takes a few seconds:
	// without a timeout, the test doesn't run too long,
	// but long enough that a short timeout triggers.
	devicesPerSlice := 9
	deviceNames := make([]string, devicesPerSlice)
	for i := 0; i < devicesPerSlice; i++ {
		deviceNames[i] = fmt.Sprintf("dev-%d", i)
	}
	slice := st.MakeResourceSlice("worker-0", driverName).Devices(deviceNames...)
	createSlice(tCtx, slice.Obj())
	otherSlice := st.MakeResourceSlice("worker-1", driverName).Devices(deviceNames...)
	createdOtherSlice := createSlice(tCtx, otherSlice.Obj())
	claim := claim.DeepCopy()
	claim.Spec.Devices.Requests[0].Exactly.Count = int64(devicesPerSlice + 1) // Impossible to allocate.
	claim = createClaim(tCtx, namespace, "", class, claim)

	tCtx.Run("disabled", func(tCtx ktesting.TContext) {
		pod := createPod(tCtx, namespace, "", podWithClaimName, claim)
		startSchedulerWithConfig(tCtx, `
profiles:
- schedulerName: default-scheduler
  pluginConfig:
  - name: DynamicResources
    args:
      filterTimeout: 0s
`)
		expectPodUnschedulable(tCtx, pod, "cannot allocate all claims")
	})

	tCtx.Run("enabled", func(tCtx ktesting.TContext) {
		pod := createPod(tCtx, namespace, "", podWithClaimName, claim)
		startSchedulerWithConfig(tCtx, `
profiles:
- schedulerName: default-scheduler
  pluginConfig:
  - name: DynamicResources
    args:
      filterTimeout: 10ms
`)
		expectPodUnschedulable(tCtx, pod, "timed out trying to allocate devices")

		// Update one slice such that allocation succeeds.
		// The scheduler must retry and should succeed now.
		createdOtherSlice.Spec.Devices = append(createdOtherSlice.Spec.Devices, resourceapi.Device{
			Name: fmt.Sprintf("dev-%d", devicesPerSlice),
		})
		_, err := tCtx.Client().ResourceV1().ResourceSlices().Update(tCtx, createdOtherSlice, metav1.UpdateOptions{})
		tCtx.ExpectNoError(err, "update worker-1's ResourceSlice")
		tCtx.ExpectNoError(e2epod.WaitForPodScheduled(tCtx, tCtx.Client(), namespace, pod.Name))
	})
}

func expectPodUnschedulable(tCtx ktesting.TContext, pod *v1.Pod, reason string) {
	tCtx.Helper()
	tCtx.ExpectNoError(e2epod.WaitForPodNameUnschedulableInNamespace(tCtx, tCtx.Client(), pod.Name, pod.Namespace), fmt.Sprintf("expected pod to be unschedulable because %q", reason))
	pod, err := tCtx.Client().CoreV1().Pods(pod.Namespace).Get(tCtx, pod.Name, metav1.GetOptions{})
	tCtx.ExpectNoError(err)
	gomega.NewWithT(tCtx).Expect(pod).To(gomega.HaveField("Status.Conditions", gomega.ContainElement(gstruct.MatchFields(gstruct.IgnoreExtras, gstruct.Fields{
		"Type":    gomega.Equal(v1.PodScheduled),
		"Status":  gomega.Equal(v1.ConditionFalse),
		"Reason":  gomega.Equal(v1.PodReasonUnschedulable),
		"Message": gomega.ContainSubstring(reason),
	}))))
}

func testPrioritizedList(tCtx ktesting.TContext, enabled bool) {
	tCtx.Parallel()
	namespace := createTestNamespace(tCtx, nil)
	class, _ := createTestClass(tCtx, namespace)
	// This is allowed to fail if the feature is disabled.
	// createClaim normally doesn't return errors because this is unusual, but we can get it indirectly.
	claim, err := func() (claim *resourceapi.ResourceClaim, finalError error) {
		tCtx, finalize := ktesting.WithError(tCtx, &finalError)
		defer finalize()
		return createClaim(tCtx, namespace, "", class, claimPrioritizedList), nil
	}()

	if !enabled {
		require.Error(tCtx, err, "claim should have become invalid after dropping FirstAvailable")
		return
	}

	require.NotEmpty(tCtx, claim.Spec.Devices.Requests[0].FirstAvailable, "should store FirstAvailable")
	startScheduler(tCtx)

	// We could create ResourceSlices for some node with the right driver.
	// But failing during Filter is sufficient to determine that it did
	// not fail during PreFilter because of FirstAvailable.
	pod := createPod(tCtx, namespace, "", podWithClaimName, claim)
	schedulingAttempted := gomega.HaveField("Status.Conditions", gomega.ContainElement(
		gstruct.MatchFields(gstruct.IgnoreExtras, gstruct.Fields{
			"Type":    gomega.Equal(v1.PodScheduled),
			"Status":  gomega.Equal(v1.ConditionFalse),
			"Reason":  gomega.Equal("Unschedulable"),
			"Message": gomega.Equal("0/8 nodes are available: 8 cannot allocate all claims. still not schedulable, preemption: 0/8 nodes are available: 8 Preemption is not helpful for scheduling."),
		}),
	))
	ktesting.Eventually(tCtx, func(tCtx ktesting.TContext) *v1.Pod {
		pod, err := tCtx.Client().CoreV1().Pods(namespace).Get(tCtx, pod.Name, metav1.GetOptions{})
		tCtx.ExpectNoError(err, "get pod")
		return pod
	}).WithTimeout(10 * time.Second).WithPolling(time.Second).Should(schedulingAttempted)
}

type nodeInfo struct {
	name       string
	driverName string
	class      *resourceapi.DeviceClass
	pool       string
}

func testPrioritizedListScoring(tCtx ktesting.TContext) {
	tCtx.Parallel()
	namespace := createTestNamespace(tCtx, nil)
	nodes, err := tCtx.Client().CoreV1().Nodes().List(tCtx, metav1.ListOptions{})
	tCtx.ExpectNoError(err, "list nodes")

	// We don't want to use more than 8 nodes since we limit the number
	// of subrequests to max 8.
	var nodesForTest []v1.Node
	if len(nodes.Items) > 8 {
		nodesForTest = nodes.Items[:8]
	} else {
		nodesForTest = nodes.Items
	}

	// Create a separate device class and driver for each node. This makes
	// it easy to create subrequests that can only be satisfied by specific
	// nodes.
	var nodeInfos []nodeInfo
	for _, node := range nodesForTest {
		driverName := fmt.Sprintf("%s-%s", namespace, node.Name)
		class, driverName := createTestClass(tCtx, driverName)
		slice := st.MakeResourceSlice(node.Name, driverName).Devices(device1, device2)
		createSlice(tCtx, slice.Obj())
		nodeInfos = append(nodeInfos, nodeInfo{
			name:       node.Name,
			driverName: driverName,
			class:      class,
			pool:       slice.Spec.Pool.Name,
		})
	}

	// Randomize the list of nodes so the selected node isn't always the first one.
	rand.Shuffle(len(nodeInfos), func(i, j int) {
		nodeInfos[i], nodeInfos[j] = nodeInfos[j], nodeInfos[i]
	})

	startScheduler(tCtx)

	tCtx.Run("single-claim", func(tCtx ktesting.TContext) {
		var firstAvailable []resourceapi.DeviceSubRequest
		for i := range nodeInfos {
			firstAvailable = append(firstAvailable, resourceapi.DeviceSubRequest{
				Name:            fmt.Sprintf("subreq-%d", i),
				DeviceClassName: nodeInfos[i].class.Name,
			})
		}
		claimPrioritizedList := st.MakeResourceClaim().
			Name(claimName + "-pl-single-claim").
			Namespace(namespace).
			RequestWithPrioritizedList(firstAvailable...).
			Obj()
		claim, err := tCtx.Client().ResourceV1().ResourceClaims(namespace).Create(tCtx, claimPrioritizedList, metav1.CreateOptions{})
		tCtx.ExpectNoError(err, "create claim "+claimName)

		_ = createPod(tCtx, namespace, "-pl-single-claim", podWithClaimName, claim)
		expectedSelectedRequest := fmt.Sprintf("%s/%s", claim.Spec.Devices.Requests[0].Name, claim.Spec.Devices.Requests[0].FirstAvailable[0].Name)
		ktesting.Eventually(tCtx, func(tCtx ktesting.TContext) *resourceapi.ResourceClaim {
			c, err := tCtx.Client().ResourceV1().ResourceClaims(namespace).Get(tCtx, claim.Name, metav1.GetOptions{})
			tCtx.ExpectNoError(err)
			return c
		}).WithTimeout(10 * time.Second).WithPolling(time.Second).Should(expectedAllocatedClaim(expectedSelectedRequest, nodeInfos[0]))
	})

	tCtx.Run("multi-claim", func(tCtx ktesting.TContext) {
		// Set up two claims where the node in nodeInfos[2] will be the best
		// option:
		// nodeInfos[1]: rank 1 in claim1 and rank 3 in claim2, so it will get a score of 14
		// nodeInfos[2]: rank 2 in claim1 and rank 1 in claim2, so it will get a score of 15
		// nodeInfos[3]: rank 3 in claim1 and rank 2 in claim2, so it will get a score of 13
		claimPrioritizedList1 := st.MakeResourceClaim().
			Name(claimName + "-pl-multiclaim-1").
			Namespace(namespace).
			RequestWithPrioritizedList([]resourceapi.DeviceSubRequest{
				{
					Name:            "subreq-0",
					DeviceClassName: nodeInfos[1].class.Name,
				},
				{
					Name:            "subreq-1",
					DeviceClassName: nodeInfos[2].class.Name,
				},
				{
					Name:            "subreq-2",
					DeviceClassName: nodeInfos[3].class.Name,
				},
			}...).
			Obj()
		claim1, err := tCtx.Client().ResourceV1().ResourceClaims(namespace).Create(tCtx, claimPrioritizedList1, metav1.CreateOptions{})
		tCtx.ExpectNoError(err, "create claim "+claimName)

		claimPrioritizedList2 := st.MakeResourceClaim().
			Name(claimName + "-pl-multiclaim-2").
			Namespace(namespace).
			RequestWithPrioritizedList([]resourceapi.DeviceSubRequest{
				{
					Name:            "subreq-0",
					DeviceClassName: nodeInfos[2].class.Name,
				},
				{
					Name:            "subreq-1",
					DeviceClassName: nodeInfos[3].class.Name,
				},
				{
					Name:            "subreq-2",
					DeviceClassName: nodeInfos[1].class.Name,
				},
			}...).
			Obj()
		claim2, err := tCtx.Client().ResourceV1().ResourceClaims(namespace).Create(tCtx, claimPrioritizedList2, metav1.CreateOptions{})
		tCtx.ExpectNoError(err, "create claim "+claimName)

		pod := st.MakePod().Name(podName).Namespace(namespace).
			Container("my-container").
			Obj()
		_ = createPod(tCtx, namespace, "-pl-multiclaim", pod, claim1, claim2)

		// The second subrequest in claim1 is for nodeInfos[2], so it should be chosen.
		expectedSelectedRequest := fmt.Sprintf("%s/%s", claim1.Spec.Devices.Requests[0].Name, claim1.Spec.Devices.Requests[0].FirstAvailable[1].Name)
		ktesting.Eventually(tCtx, func(tCtx ktesting.TContext) *resourceapi.ResourceClaim {
			c, err := tCtx.Client().ResourceV1().ResourceClaims(namespace).Get(tCtx, claimPrioritizedList1.Name, metav1.GetOptions{})
			tCtx.ExpectNoError(err)
			return c
		}).WithTimeout(10 * time.Second).WithPolling(time.Second).Should(expectedAllocatedClaim(expectedSelectedRequest, nodeInfos[2]))

		// The first subrequest in claim2 is for nodeInfos[2], so it should be chosen.
		expectedSelectedRequest = fmt.Sprintf("%s/%s", claim2.Spec.Devices.Requests[0].Name, claim2.Spec.Devices.Requests[0].FirstAvailable[0].Name)
		ktesting.Eventually(tCtx, func(tCtx ktesting.TContext) *resourceapi.ResourceClaim {
			c, err := tCtx.Client().ResourceV1().ResourceClaims(namespace).Get(tCtx, claimPrioritizedList2.Name, metav1.GetOptions{})
			tCtx.ExpectNoError(err)
			return c
		}).WithTimeout(10 * time.Second).WithPolling(time.Second).Should(expectedAllocatedClaim(expectedSelectedRequest, nodeInfos[2]))
	})
}

func expectedAllocatedClaim(request string, nodeInfo nodeInfo) gtypes.GomegaMatcher {
	return gomega.HaveField("Status.Allocation", gstruct.PointTo(gstruct.MatchFields(gstruct.IgnoreExtras, gstruct.Fields{
		"Devices": gstruct.MatchFields(gstruct.IgnoreExtras, gstruct.Fields{
			"Results": gomega.HaveExactElements(gstruct.MatchFields(gstruct.IgnoreExtras, gstruct.Fields{
				"Request": gomega.Equal(request),
				"Driver":  gomega.Equal(nodeInfo.driverName),
				"Pool":    gomega.Equal(nodeInfo.pool),
			})),
		}),
		"NodeSelector": gstruct.PointTo(gstruct.MatchFields(gstruct.IgnoreExtras, gstruct.Fields{
			"NodeSelectorTerms": gomega.HaveExactElements(gstruct.MatchFields(gstruct.IgnoreExtras, gstruct.Fields{
				"MatchFields": gomega.HaveExactElements(gstruct.MatchFields(gstruct.IgnoreExtras, gstruct.Fields{
					"Key":      gomega.Equal("metadata.name"),
					"Operator": gomega.Equal(v1.NodeSelectorOpIn),
					"Values":   gomega.HaveExactElements(gomega.Equal(nodeInfo.name)),
				})),
			})),
		})),
	})))
}

func testExtendedResource(tCtx ktesting.TContext, enabled bool) {
	tCtx.Parallel()

	namespace := createTestNamespace(tCtx, nil)
	driverName := namespace + driverNameSuffix
	class := classWithExtendedResource.DeepCopy()
	class.Spec.Selectors = []resourceapi.DeviceSelector{{
		CEL: &resourceapi.CELDeviceSelector{
			Expression: fmt.Sprintf("device.driver == %q", driverName),
		},
	}}
	c, err := tCtx.Client().ResourceV1().DeviceClasses().Create(tCtx, class, metav1.CreateOptions{FieldValidation: "Strict"})
	tCtx.ExpectNoError(err, "create class")

	slice := st.MakeResourceSlice("worker-0", driverName).Devices(device1)
	createSlice(tCtx, slice.Obj())

	if enabled {
		require.NotEmpty(tCtx, c.Spec.ExtendedResourceName, "should store ExtendedResourceName")
	}

	tCtx.Run("scheduler", func(tCtx ktesting.TContext) {
		startScheduler(tCtx)

		pod := podWithExtendedResource.DeepCopy()
		pod.Namespace = namespace
		_, err := tCtx.Client().CoreV1().Pods(namespace).Create(tCtx, pod, metav1.CreateOptions{FieldValidation: "Strict"})
		tCtx.ExpectNoError(err, "create pod")
		schedulingAttempted := gomega.HaveField("Status.Conditions", gomega.ContainElement(
			gstruct.MatchFields(gstruct.IgnoreExtras, gstruct.Fields{
				"Type":    gomega.Equal(v1.PodScheduled),
				"Status":  gomega.Equal(v1.ConditionFalse),
				"Reason":  gomega.Equal("Unschedulable"),
				"Message": gomega.Equal("0/8 nodes are available: 8 Insufficient my-example.com/my-extended-resource. no new claims to deallocate, preemption: 0/8 nodes are available: 8 Preemption is not helpful for scheduling."),
			}),
		))
		if enabled {
			// pod can be scheduled as the drivers in testPublishResourceSlices provide the devices.
			schedulingAttempted = gomega.HaveField("Status.Conditions", gomega.ContainElement(
				gstruct.MatchFields(gstruct.IgnoreExtras, gstruct.Fields{
					"Type":   gomega.Equal(v1.PodScheduled),
					"Status": gomega.Equal(v1.ConditionTrue),
				}),
			))
		}
		ktesting.Eventually(tCtx, func(tCtx ktesting.TContext) *v1.Pod {
			pod, err := tCtx.Client().CoreV1().Pods(namespace).Get(tCtx, pod.Name, metav1.GetOptions{})
			tCtx.ExpectNoError(err, "get pod")
			return pod
		}).WithTimeout(time.Minute).WithPolling(time.Second).Should(schedulingAttempted)
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
						"BindingConditions":        gomega.Equal(device.BindingConditions),
						"BindingFailureConditions": gomega.Equal(device.BindingFailureConditions),
						"BindsToNode":              gomega.Equal(device.BindsToNode),
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
	quiesencePeriod := syncDelay
	if mutationCacheTTL > quiesencePeriod {
		quiesencePeriod = mutationCacheTTL
	}
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
		ktesting.Eventually(tCtx, getStats).WithTimeout(syncDelay + 5*time.Second).Should(gomega.Equal(expectedStats))
		expectSlices(tCtx)

		return controller, getStats, expectedStats
	}

	// Each sub-test starts with no slices and must clean up after itself.

	tCtx.Run("create", func(tCtx ktesting.TContext) {
		controller, getStats, expectedStats := setup(tCtx)

		// No further changes necessary.
		ktesting.Consistently(tCtx, getStats).WithTimeout(quiesencePeriod).Should(gomega.Equal(expectedStats))

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
		ktesting.Eventually(tCtx, getStats).WithTimeout(10*time.Second).Should(gomega.HaveField("NumDeletes", gomega.BeNumerically(">=", int64(1))), "Slice should have been removed.")
		ktesting.Eventually(tCtx, func(tCtx ktesting.TContext) bool {
			return gotValidationError.Load()
		}).WithTimeout(time.Minute).Should(gomega.BeTrueBecause("Should have gotten another error because the slice is invalid."))
	})

	if !haveLatestAPI {
		return
	}

	tCtx.Run("recreate-after-delete", func(tCtx ktesting.TContext) {
		_, getStats, expectedStats := setup(tCtx)
		ktesting.Consistently(tCtx, getStats).WithTimeout(quiesencePeriod).Should(gomega.Equal(expectedStats))

		// Stress the controller by repeatedly deleting the slices.
		// One delete occurs after the sync period is over (because of the Consistently),
		// the second before (because it's done as quickly as possible).
		for i := 0; i < 2; i++ {
			tCtx.Log("deleting ResourceSlices")
			tCtx.ExpectNoError(tCtx.Client().ResourceV1().ResourceSlices().DeleteCollection(tCtx, metav1.DeleteOptions{}, listDriverSlices), "delete driver slices")
			expectedStats.NumCreates += int64(len(expectedSlices))
			ktesting.Eventually(tCtx, getStats).WithTimeout(syncDelay + 5*time.Second).Should(gomega.Equal(expectedStats))
			expectSlices(tCtx)
		}
	})

	tCtx.Run("fix-after-update", func(tCtx ktesting.TContext) {
		_, getStats, expectedStats := setup(tCtx)

		// Stress the controller by repeatedly updatings the slices.
		for i := 0; i < 2; i++ {
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
			ktesting.Eventually(tCtx, getStats).WithTimeout(syncDelay + 5*time.Second).Should(gomega.Equal(expectedStats))
			expectSlices(tCtx)
		}
	})
}

// testResourceClaimDeviceStatus creates a ResourceClaim with an invalid device (not allocated device)
// and checks that the object is not validated (feature enabled) resp. accepted without the field (disabled).
//
// When enabled, it tries server-side-apply (SSA) with different clients. This is what DRA drivers should be using.
func testResourceClaimDeviceStatus(tCtx ktesting.TContext, enabled bool) {
	namespace := createTestNamespace(tCtx, nil)

	claim := &resourceapi.ResourceClaim{
		ObjectMeta: metav1.ObjectMeta{
			Name: claimName,
		},
		Spec: resourceapi.ResourceClaimSpec{
			Devices: resourceapi.DeviceClaim{
				Requests: []resourceapi.DeviceRequest{
					{
						Name: "foo",
						Exactly: &resourceapi.ExactDeviceRequest{
							DeviceClassName: "foo",
						},
					},
				},
			},
		},
	}

	claim, err := tCtx.Client().ResourceV1().ResourceClaims(namespace).Create(tCtx, claim, metav1.CreateOptions{FieldValidation: "Strict"})
	tCtx.ExpectNoError(err, "create ResourceClaim")

	deviceStatus := []resourceapi.AllocatedDeviceStatus{{
		Driver: "one",
		Pool:   "global",
		Device: "my-device",
		Data: &runtime.RawExtension{
			Raw: []byte(`{"kind": "foo", "apiVersion": "dra.example.com/v1"}`),
		},
		NetworkData: &resourceapi.NetworkDeviceData{
			InterfaceName: "net-1",
			IPs: []string{
				"10.9.8.0/24",
				"2001:db8::/64",
			},
			HardwareAddress: "ea:9f:cb:40:b1:7b",
		},
	}}
	claim.Status.Devices = deviceStatus
	updatedClaim, err := tCtx.Client().ResourceV1().ResourceClaims(namespace).UpdateStatus(tCtx, claim, metav1.UpdateOptions{})
	if !enabled {
		tCtx.ExpectNoError(err, "updating the status with an invalid AllocatedDeviceStatus should have worked because the field should have been dropped")
		require.Empty(tCtx, updatedClaim.Status.Devices, "field should have been dropped")
		return
	}

	// Tests for enabled feature follow.

	if err == nil {
		tCtx.Fatal("updating the status with an invalid AllocatedDeviceStatus should have failed and didn't")
	}

	// Add an allocation result.
	claim.Status.Allocation = &resourceapi.AllocationResult{
		Devices: resourceapi.DeviceAllocationResult{
			Results: []resourceapi.DeviceRequestAllocationResult{
				{
					Request: "foo",
					Driver:  "one",
					Pool:    "global",
					Device:  "my-device",
				},
				{
					Request: "foo",
					Driver:  "two",
					Pool:    "global",
					Device:  "another-device",
				},
				{
					Request: "foo",
					Driver:  "three",
					Pool:    "global",
					Device:  "my-device",
				},
			},
		},
	}
	claim, err = tCtx.Client().ResourceV1().ResourceClaims(namespace).UpdateStatus(tCtx, claim, metav1.UpdateOptions{})
	tCtx.ExpectNoError(err, "add allocation result")

	// Now adding the device status should work.
	claim.Status.Devices = deviceStatus
	claim, err = tCtx.Client().ResourceV1().ResourceClaims(namespace).UpdateStatus(tCtx, claim, metav1.UpdateOptions{})
	tCtx.ExpectNoError(err, "add device status")
	require.Equal(tCtx, deviceStatus, claim.Status.Devices, "after adding device status")

	// Strip the RawExtension. SSA re-encodes it, which causes negligble differences that nonetheless break assert.Equal.
	claim.Status.Devices[0].Data = nil
	deviceStatus[0].Data = nil
	claim, err = tCtx.Client().ResourceV1().ResourceClaims(namespace).UpdateStatus(tCtx, claim, metav1.UpdateOptions{})
	tCtx.ExpectNoError(err, "add device status")
	require.Equal(tCtx, deviceStatus, claim.Status.Devices, "after stripping RawExtension")

	// Exercise SSA.
	deviceStatusAC := resourceapiac.AllocatedDeviceStatus().
		WithDriver("two").
		WithPool("global").
		WithDevice("another-device").
		WithNetworkData(resourceapiac.NetworkDeviceData().WithInterfaceName("net-2"))
	deviceStatus = append(deviceStatus, resourceapi.AllocatedDeviceStatus{
		Driver: "two",
		Pool:   "global",
		Device: "another-device",
		NetworkData: &resourceapi.NetworkDeviceData{
			InterfaceName: "net-2",
		},
	})
	claimAC := resourceapiac.ResourceClaim(claim.Name, claim.Namespace).
		WithStatus(resourceapiac.ResourceClaimStatus().WithDevices(deviceStatusAC))
	claim, err = tCtx.Client().ResourceV1().ResourceClaims(namespace).ApplyStatus(tCtx, claimAC, metav1.ApplyOptions{
		Force:        true,
		FieldManager: "manager-1",
	})
	tCtx.ExpectNoError(err, "apply device status two")
	require.Equal(tCtx, deviceStatus, claim.Status.Devices, "after applying device status two")

	deviceStatusAC = resourceapiac.AllocatedDeviceStatus().
		WithDriver("three").
		WithPool("global").
		WithDevice("my-device").
		WithNetworkData(resourceapiac.NetworkDeviceData().WithInterfaceName("net-3"))
	deviceStatus = append(deviceStatus, resourceapi.AllocatedDeviceStatus{
		Driver: "three",
		Pool:   "global",
		Device: "my-device",
		NetworkData: &resourceapi.NetworkDeviceData{
			InterfaceName: "net-3",
		},
	})
	claimAC = resourceapiac.ResourceClaim(claim.Name, claim.Namespace).
		WithStatus(resourceapiac.ResourceClaimStatus().WithDevices(deviceStatusAC))
	claim, err = tCtx.Client().ResourceV1().ResourceClaims(namespace).ApplyStatus(tCtx, claimAC, metav1.ApplyOptions{
		Force:        true,
		FieldManager: "manager-2",
	})
	tCtx.ExpectNoError(err, "apply device status three")
	require.Equal(tCtx, deviceStatus, claim.Status.Devices, "after applying device status three")
	var buffer bytes.Buffer
	encoder := json.NewEncoder(&buffer)
	encoder.SetIndent("   ", "   ")
	tCtx.ExpectNoError(encoder.Encode(claim))
	tCtx.Logf("Final ResourceClaim:\n%s", buffer.String())

	// Update one entry, remove the other.
	deviceStatusAC = resourceapiac.AllocatedDeviceStatus().
		WithDriver("two").
		WithPool("global").
		WithDevice("another-device").
		WithNetworkData(resourceapiac.NetworkDeviceData().WithInterfaceName("yet-another-net"))
	deviceStatus[1].NetworkData.InterfaceName = "yet-another-net"
	claimAC = resourceapiac.ResourceClaim(claim.Name, claim.Namespace).
		WithStatus(resourceapiac.ResourceClaimStatus().WithDevices(deviceStatusAC))
	claim, err = tCtx.Client().ResourceV1().ResourceClaims(namespace).ApplyStatus(tCtx, claimAC, metav1.ApplyOptions{
		Force:        true,
		FieldManager: "manager-1",
	})
	tCtx.ExpectNoError(err, "update device status two")
	require.Equal(tCtx, deviceStatus, claim.Status.Devices, "after updating device status two")
	claimAC = resourceapiac.ResourceClaim(claim.Name, claim.Namespace)
	deviceStatus = deviceStatus[0:2]
	claim, err = tCtx.Client().ResourceV1().ResourceClaims(namespace).ApplyStatus(tCtx, claimAC, metav1.ApplyOptions{
		Force:        true,
		FieldManager: "manager-2",
	})
	tCtx.ExpectNoError(err, "remove device status three")
	require.Equal(tCtx, deviceStatus, claim.Status.Devices, "after removing device status three")
}

// testMaxResourceSlice creates ResourceSlices that are as large as possible
// and prints some information about it.
func testMaxResourceSlice(tCtx ktesting.TContext) {
	for name, slice := range NewMaxResourceSlices() {
		tCtx.Run(name, func(tCtx ktesting.TContext) {
			createdSlice := createSlice(tCtx, slice)
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
		})
	}
}

// testControllerManagerMetrics tests ResourceClaim metrics
func testControllerManagerMetrics(tCtx ktesting.TContext) {
	namespace := createTestNamespace(tCtx, nil)
	class, _ := createTestClass(tCtx, namespace)

	informerFactory := informers.NewSharedInformerFactory(tCtx.Client(), 0)
	runResourceClaimController := util.CreateResourceClaimController(tCtx, tCtx, tCtx.Client(), informerFactory)
	informerFactory.Start(tCtx.Done())
	cache.WaitForCacheSync(tCtx.Done(),
		informerFactory.Core().V1().Pods().Informer().HasSynced,
		informerFactory.Resource().V1().ResourceClaims().Informer().HasSynced,
		informerFactory.Resource().V1().ResourceClaimTemplates().Informer().HasSynced,
	)

	// Start the controller (this will run in background and stop when tCtx is cancelled)
	runResourceClaimController()

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

	time.Sleep(200 * time.Millisecond)

	// Verify metrics: success counter with admin_access=false should increment
	successNoAdmin := getMetricValue("success", "false")
	require.InDelta(tCtx, initialSuccessNoAdmin+1, successNoAdmin, 0.1, "success metric with admin_access=false should increment")

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

	time.Sleep(200 * time.Millisecond)

	// Verify metrics: success counter with admin_access=true should increment
	successWithAdmin := getMetricValue("success", "true")
	require.InDelta(tCtx, initialSuccessWithAdmin+1, successWithAdmin, 0.1, "success metric with admin_access=true should increment")

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

	time.Sleep(200 * time.Millisecond)

	// Verify final metrics
	finalSuccessNoAdmin := getMetricValue("success", "false")
	finalSuccessWithAdmin := getMetricValue("success", "true")

	require.InDelta(tCtx, initialSuccessNoAdmin+2, finalSuccessNoAdmin, 0.1, "should have 2 more success metrics with admin_access=false")
	require.InDelta(tCtx, initialSuccessWithAdmin+1, finalSuccessWithAdmin, 0.1, "should have 1 more success metric with admin_access=true")

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
				for i := 0; i < 8; i++ {
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
		tCtx.Run(tn, func(tCtx ktesting.TContext) {
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
			ktesting.Eventually(tCtx, func(tCtx ktesting.TContext) *v1.Pod {
				pod, err := tCtx.Client().CoreV1().Pods(namespace).Get(tCtx, pod.Name, metav1.GetOptions{})
				tCtx.ExpectNoError(err, "get pod")
				return pod
			}).WithTimeout(10 * time.Second).WithPolling(time.Second).Should(schedulingAttempted)

			// Only check the ResourceClaim if we expected the Pod to schedule.
			if tc.expectPodToSchedule {
				ktesting.Eventually(tCtx, func(tCtx ktesting.TContext) *resourceapi.ResourceClaim {
					c, err := tCtx.Client().ResourceV1().ResourceClaims(namespace).Get(tCtx, claim.Name, metav1.GetOptions{})
					tCtx.ExpectNoError(err)
					return c
				}).WithTimeout(10 * time.Second).WithPolling(time.Second).Should(gomega.HaveField("Status.Allocation", gstruct.PointTo(gstruct.MatchFields(gstruct.IgnoreExtras, gstruct.Fields{
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
