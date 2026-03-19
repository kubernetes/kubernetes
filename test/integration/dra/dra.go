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
	"fmt"
	"regexp"
	"sort"
	"strings"
	"sync"
	"testing"
	"time"

	v1 "k8s.io/api/core/v1"
	resourceapi "k8s.io/api/resource/v1"
	resourcealphaapi "k8s.io/api/resource/v1alpha3"
	resourcev1beta1 "k8s.io/api/resource/v1beta1"
	resourcev1beta2 "k8s.io/api/resource/v1beta2"
	schedulingapi "k8s.io/api/scheduling/v1alpha2"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/version"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/client-go/informers"
	"k8s.io/client-go/tools/cache"
	"k8s.io/component-base/featuregate"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/klog/v2"
	kubeschedulerconfigv1 "k8s.io/kube-scheduler/config/v1"
	kubeapiservertesting "k8s.io/kubernetes/cmd/kube-apiserver/app/testing"
	"k8s.io/kubernetes/pkg/controller/resourceclaim"
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

	podName      = "my-pod"
	namespace    = "default"
	resourceName = "my-resource"
	claimName    = podName + "-" + resourceName
	className    = "my-resource-class"
	device1      = "device-1"
	device2      = "device-2"

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
				RequestWithPrioritizedList(st.SubRequest("subreq-1", className, 1)).
				Obj()
)

const (
	numNodes           = 8
	maxPodsPerNode     = 5000 // This should never be the limiting factor, no matter how many tests run in parallel.
	nodeCPUCapacity    = "100"
	nodeMemoryCapacity = "1k"

	// schedulingTimeout is the time we grant the scheduler for one scheduling attempt,
	// whether it's successful or not.
	schedulingTimeout = time.Minute
)

func Run(t *testing.T, whatRE string) { run(ktesting.Init(t), whatRE) }
func run(tCtx ktesting.TContext, whatRE string) {
	re, err := regexp.Compile(whatRE)
	if err != nil {
		tCtx.Fatalf("%s: %v", whatRE, err)
	}

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
				runSubTest(tCtx, "APIDisabled", testAPIDisabled)
				runSubTest(tCtx, "Pod", func(tCtx ktesting.TContext) { testPod(tCtx, false) })
			},
		},
		"default": {
			apis:     map[schema.GroupVersion]bool{},
			features: map[featuregate.Feature]bool{},
			f: func(tCtx ktesting.TContext) {
				runSubTest(tCtx, "Pod", func(tCtx ktesting.TContext) { testPod(tCtx, true) })
				runSubTest(tCtx, "EvictClusterWithSlices", func(tCtx ktesting.TContext) { testEvictCluster(tCtx, useNoRule) })
				// Number of devices per slice is chosen so that Filter takes a few seconds:
				// without a timeout, the test doesn't run too long, but long enough that a short timeout triggers.
				runSubTest(tCtx, "FilterTimeout", func(tCtx ktesting.TContext) { testFilterTimeout(tCtx, 21) })
				runSubTest(tCtx, "UsesAllResources", testUsesAllResources)
			},
		},
		"GA": {
			apis: map[schema.GroupVersion]bool{},
			features: map[featuregate.Feature]bool{
				featuregate.Feature("AllBeta"): false,
			},
			f: func(tCtx ktesting.TContext) {
				runSubTest(tCtx, "AdminAccess", func(tCtx ktesting.TContext) { testAdminAccess(tCtx, false) })
				runSubTest(tCtx, "PartitionableDevices", func(tCtx ktesting.TContext) { testPartitionableDevices(tCtx, false) })
				runSubTest(tCtx, "PrioritizedList", func(tCtx ktesting.TContext) { testPrioritizedList(tCtx, true) })
				runSubTest(tCtx, "Pod", func(tCtx ktesting.TContext) { testPod(tCtx, true) })
				runSubTest(tCtx, "PublishResourceSlices", func(tCtx ktesting.TContext) {
					testPublishResourceSlices(tCtx, true, features.DRADeviceTaints, features.DRAPartitionableDevices, features.DRADeviceBindingConditions)
				})
				runSubTest(tCtx, "ExplicitExtendedResource", func(tCtx ktesting.TContext) { testExtendedResource(tCtx, false, true) })
				runSubTest(tCtx, "ImplicitExtendedResource", func(tCtx ktesting.TContext) { testExtendedResource(tCtx, false, false) })
				runSubTest(tCtx, "ResourceClaimDeviceStatus", func(tCtx ktesting.TContext) { testResourceClaimDeviceStatus(tCtx, false) })
				runSubTest(tCtx, "DeviceBindingConditions", func(tCtx ktesting.TContext) { testDeviceBindingConditions(tCtx, false) })
				runSubTest(tCtx, "ResourceSliceController", func(tCtx ktesting.TContext) {
					namespace := createTestNamespace(tCtx, nil)
					tCtx = tCtx.WithNamespace(namespace)
					TestCreateResourceSlices(tCtx, 100)
				})
				runSubTest(tCtx, "ShareResourceClaimSequentially", testShareResourceClaimSequentially)
				runSubTest(tCtx, "UsesAllResources", testUsesAllResources)
			},
		},
		// This scenario verifies that features which have graduated to GA can
		// still be explicitly disabled via feature gates.
		"GA-opt-out": {
			apis: map[schema.GroupVersion]bool{},
			features: map[featuregate.Feature]bool{
				featuregate.Feature("AllBeta"): false,
				features.DRAPrioritizedList:    false,
			},
			f: func(tCtx ktesting.TContext) {
				runSubTest(tCtx, "PrioritizedList", func(tCtx ktesting.TContext) { testPrioritizedList(tCtx, false) })
			},
		},
		"v1beta1": {
			apis: map[schema.GroupVersion]bool{
				resourceapi.SchemeGroupVersion:     false,
				resourcev1beta1.SchemeGroupVersion: true,
			},
			features: map[featuregate.Feature]bool{features.DynamicResourceAllocation: true},
			f: func(tCtx ktesting.TContext) {
				runSubTest(tCtx, "PublishResourceSlices", func(tCtx ktesting.TContext) {
					testPublishResourceSlices(tCtx, false)
				})
			},
		},
		"v1beta2": {
			apis: map[schema.GroupVersion]bool{
				resourceapi.SchemeGroupVersion:     false,
				resourcev1beta2.SchemeGroupVersion: true,
			},
			features: map[featuregate.Feature]bool{
				features.DynamicResourceAllocation: true,
				features.DRADeviceTaintRules:       true,
			},
			f: func(tCtx ktesting.TContext) {
				runSubTest(tCtx, "PublishResourceSlices", func(tCtx ktesting.TContext) {
					testPublishResourceSlices(tCtx, false)
				})
			},
		},
		"all": {
			apis: map[schema.GroupVersion]bool{
				resourcev1beta1.SchemeGroupVersion:  true,
				resourcev1beta2.SchemeGroupVersion:  true,
				resourcealphaapi.SchemeGroupVersion: true,
				schedulingapi.SchemeGroupVersion:    true,
			},
			features: map[featuregate.Feature]bool{
				// Additional DRA feature gates go here,
				// in alphabetical order,
				// as needed by tests for them.
				features.DRAAdminAccess:               true,
				features.DRADeviceBindingConditions:   true,
				features.DRAConsumableCapacity:        true,
				features.DRADeviceTaintRules:          true,
				features.DRAPartitionableDevices:      true,
				features.DRAPrioritizedList:           true,
				features.DRAResourceClaimDeviceStatus: true,
				features.DRAExtendedResource:          true,
				features.DRANodeAllocatableResources:  true,
				features.GangScheduling:               true,
				features.GenericWorkload:              true,
			},
			f: func(tCtx ktesting.TContext) {
				// These tests must run in parallel as much as possible to keep overall runtime low!

				runSubTest(tCtx, "AdminAccess", func(tCtx ktesting.TContext) { testAdminAccess(tCtx, true) })
				runSubTest(tCtx, "Convert", testConvert)
				runSubTest(tCtx, "ControllerManagerMetrics", testControllerManagerMetrics)
				runSubTest(tCtx, "DeviceBindingConditions", func(tCtx ktesting.TContext) { testDeviceBindingConditions(tCtx, true) })
				runSubTest(tCtx, "PartitionableDevices", func(tCtx ktesting.TContext) { testPartitionableDevices(tCtx, true) })
				runSubTest(tCtx, "PrioritizedList", func(tCtx ktesting.TContext) { testPrioritizedList(tCtx, true) })
				runSubTest(tCtx, "PrioritizedListScoring", func(tCtx ktesting.TContext) { testPrioritizedListScoring(tCtx) })
				runSubTest(tCtx, "PublishResourceSlices", func(tCtx ktesting.TContext) { testPublishResourceSlices(tCtx, true) })
				runSubTest(tCtx, "ExplicitExtendedResource", func(tCtx ktesting.TContext) { testExtendedResource(tCtx, true, true) })
				runSubTest(tCtx, "ImplicitExtendedResource", func(tCtx ktesting.TContext) { testExtendedResource(tCtx, true, false) })
				runSubTest(tCtx, "ResourceClaimDeviceStatus", func(tCtx ktesting.TContext) { testResourceClaimDeviceStatus(tCtx, true) })
				runSubTest(tCtx, "MaxResourceSlice", testMaxResourceSlice)
				runSubTest(tCtx, "EvictClusterWithV1alpha3Rule", func(tCtx ktesting.TContext) { testEvictCluster(tCtx, useV1alpha3Rule) })
				runSubTest(tCtx, "EvictClusterWithV1beta2Rule", func(tCtx ktesting.TContext) { testEvictCluster(tCtx, useV1beta2Rule) })
				runSubTest(tCtx, "EvictClusterWithSlices", func(tCtx ktesting.TContext) { testEvictCluster(tCtx, useNoRule) })
				runSubTest(tCtx, "InvalidResourceSlices", testInvalidResourceSlices)
				// Number of devices per slice is chosen so that Filter takes a few seconds: The allocator
				// in the experimental channel has an improvement that requires a higher number here than
				// in the incubating and stable channels.
				runSubTest(tCtx, "FilterTimeout", func(tCtx ktesting.TContext) { testFilterTimeout(tCtx, 21) })
				runSubTest(tCtx, "ShareResourceClaimSequentially", testShareResourceClaimSequentially)
				runSubTest(tCtx, "UsesAllResources", testUsesAllResources)
				runSubTest(tCtx, "DRANodeAllocatableResources", func(tCtx ktesting.TContext) { testNodeAllocatableResources(tCtx, true) })
				runSubTest(tCtx, "PodGroup", testPodGroup)
			},
		},
	} {
		if !re.MatchString(name) {
			continue
		}

		tCtx.Run(name, func(tCtx ktesting.TContext) {
			var entries []string
			for key, value := range tc.features {
				entries = append(entries, fmt.Sprintf("%s=%t", key, value))
			}
			for key, value := range tc.apis {
				entries = append(entries, fmt.Sprintf("%s=%t", key, value))
			}
			sort.Strings(entries)
			tCtx.Logf("Config: %s", strings.Join(entries, ","))

			// We need to set emulation version for DynamicResourceAllocation feature gate, which is locked at 1.35.
			if draEnabled, draExists := tc.features[features.DynamicResourceAllocation]; draExists && !draEnabled {
				featuregatetesting.SetFeatureGateEmulationVersionDuringTest(tCtx, utilfeature.DefaultFeatureGate, version.MustParse("1.34"))
			}
			featuregatetesting.SetFeatureGatesDuringTest(tCtx, utilfeature.DefaultFeatureGate, tc.features)

			etcdOptions := framework.SharedEtcd()
			apiServerOptions := kubeapiservertesting.NewDefaultTestServerOptions()
			apiServerFlags := framework.DefaultTestServerFlags()
			var runtimeConfigs []string
			for key, value := range tc.apis {
				runtimeConfigs = append(runtimeConfigs, fmt.Sprintf("%s=%t", key, value))
			}
			apiServerFlags = append(apiServerFlags, "--runtime-config="+strings.Join(runtimeConfigs, ","))

			// Keep the apiserver running despite context cancelation.
			// We may want to collect some information from it below.
			// It then gets stopped explicitly at the end of shutdown.
			server, err := kubeapiservertesting.StartTestServer(tCtx.WithoutCancel(), apiServerOptions, apiServerFlags, etcdOptions)
			tCtx.ExpectNoError(err, "start apiserver")
			tCtx.CleanupCtx(func(tCtx ktesting.TContext) {
				tCtx.Log("Stopping the apiserver...")
				server.TearDownFn()
			})
			tCtx = tCtx.WithRESTConfig(server.ClientConfig)

			createNodes(tCtx)
			tCtx = prepareScheduler(tCtx)
			tCtx = prepareClaimController(tCtx)

			// To isolate different tests properly, each test must ensure that it
			// only allocates its own devices. We can verify that here by making sure that
			// driver names of allocated devices contain the namespace. That is how all tests
			// should construct their driver name(s).
			//
			// We run this in parallel to the tests because post-test checking might miss claims.
			informerFactory := informers.NewSharedInformerFactory(tCtx.Client(), 0)
			claimHandle, err := informerFactory.Resource().V1().ResourceClaims().Informer().AddEventHandlerWithOptions(cache.ResourceEventHandlerFuncs{
				UpdateFunc: func(oldObj, newObj any) {
					oldClaim := oldObj.(*resourceapi.ResourceClaim)
					newClaim := newObj.(*resourceapi.ResourceClaim)

					// Check an allocation once when it appears.
					if oldClaim.Status.Allocation == nil && newClaim.Status.Allocation != nil {
						for _, result := range newClaim.Status.Allocation.Devices.Results {
							if !strings.Contains(result.Driver, newClaim.Namespace) {
								tCtx.Errorf("Claim %s has allocated device %#v from a driver in some other test (%s):\n%s", klog.KObj(newClaim), result, result.Driver, format.Object(newClaim, 1))
								break
							}
						}
					}
				},
			}, cache.HandlerOptions{
				Logger: ptr.To(tCtx.Logger()),
			})
			tCtx.ExpectNoError(err, "add claim event handler")
			informerFactory.StartWithContext(tCtx)
			tCtx.Cleanup(func() {
				tCtx.Cancel("test is done")
				_ = informerFactory.Resource().V1().ResourceClaims().Informer().RemoveEventHandler(claimHandle)
				informerFactory.Shutdown()
			})

			// Dump some information in case of a test failure before continuing with the shutdown.
			// If we timed out, then we should still get this done, even if the shutdown itself
			// then fails.
			tCtx.CleanupCtx(func(tCtx ktesting.TContext) {
				if !tCtx.Failed() {
					return
				}

				namespaces, _ := tCtx.Client().CoreV1().Namespaces().List(tCtx, metav1.ListOptions{})
				tCtx.Logf("Namespaces:\n%s", format.Object(namespaces, 0))
				claims, _ := tCtx.Client().ResourceV1().ResourceClaims("").List(tCtx, metav1.ListOptions{})
				tCtx.Logf("ResourceClaims:\n%s", format.Object(claims, 0))
				classes, _ := tCtx.Client().ResourceV1().DeviceClasses().List(tCtx, metav1.ListOptions{})
				tCtx.Logf("DeviceClasses:\n%s", format.Object(classes, 0))
				slices, _ := tCtx.Client().ResourceV1().ResourceSlices().List(tCtx, metav1.ListOptions{})
				tCtx.Logf("ResourceSlices:\n%s", format.Object(slices, 0))
			})

			tc.f(tCtx)
		})
	}
}

func createNodes(tCtx ktesting.TContext) {
	for i := range numNodes {
		nodeName := fmt.Sprintf("worker-%d", i)
		// Create node.
		node := &v1.Node{
			ObjectMeta: metav1.ObjectMeta{
				Name: nodeName,
				Labels: map[string]string{
					"kubernetes.io/hostname": nodeName,
				},
			},
		}
		node, err := tCtx.Client().CoreV1().Nodes().Create(tCtx, node, metav1.CreateOptions{FieldValidation: "Strict"})
		tCtx.ExpectNoError(err, fmt.Sprintf("creating node #%d", i))

		// Make the node ready.
		node.Status = v1.NodeStatus{
			Capacity: v1.ResourceList{
				v1.ResourceCPU:    resource.MustParse(nodeCPUCapacity),
				v1.ResourceMemory: resource.MustParse(nodeMemoryCapacity),
				v1.ResourcePods:   *resource.NewScaledQuantity(maxPodsPerNode, 0),
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

// runSubTest re-initializes the client inside the sub-test to ensure that the
// user agent is set based on the sub-test's name. This then shows up in
// apiserver output. Unfortunately it does not change the credentials and therefore
// all tests appear as the same manager in ManagedFields.
//
// tCtx.Run() itself doesn't do that because the tests might have set REST
// config and clients differently in the parent context.
func runSubTest(tCtx ktesting.TContext, name string, cb func(tCtx ktesting.TContext)) {
	tCtx.Run(name, func(tCtx ktesting.TContext) {
		tCtx = tCtx.WithRESTConfig(tCtx.RESTConfig())
		cb(tCtx)
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

	return tCtx.WithValue(schedulerKey, scheduler)
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

// prepareClaimController does the same as prepareScheduler for the ResourceClaimController.
func prepareClaimController(tCtx ktesting.TContext) ktesting.TContext {
	claimController := &claimControllerSingleton{
		rootCtx: tCtx,
	}

	return tCtx.WithValue(claimControllerKey, claimController)
}

// startClaimController can be used by tests to ensure that the ResourceClaim controller is running.
// This may be used in parallel tests.
func startClaimController(tCtx ktesting.TContext) {
	tCtx.Helper()
	value := tCtx.Value(claimControllerKey)
	if value == nil {
		tCtx.Fatal("internal error: startClaimController without a prior prepareClaimController call")
	}
	claimController := value.(*claimControllerSingleton)
	claimController.start(tCtx)
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
	tCtx = tCtx.WithLogger(klog.LoggerWithName(tCtx.Logger(), "scheduler"))
	schedulerCtx := tCtx.WithCancel()
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

type claimControllerSingleton struct {
	rootCtx ktesting.TContext

	mutex           sync.Mutex
	usageCount      int
	wg              sync.WaitGroup
	informerFactory informers.SharedInformerFactory
	cancel          func(cause string)
}

func (claimController *claimControllerSingleton) start(tCtx ktesting.TContext) {
	tCtx.Helper()
	claimController.mutex.Lock()
	defer claimController.mutex.Unlock()

	claimController.usageCount++
	tCtx.CleanupCtx(claimController.stop)
	if claimController.usageCount > 1 {
		// Already started earlier.
		return
	}

	// Run claimController with default configuration. This must use the root context because
	// the per-test tCtx passed to start will get canceled once the test which triggered
	// starting the claimController is done.
	tCtx = claimController.rootCtx
	tCtx.Logf("Starting the ResourceClaim controller for test %s...", tCtx.Name())
	tCtx = tCtx.WithLogger(klog.LoggerWithName(tCtx.Logger(), "claimController"))

	claimControllerCtx := tCtx.WithCancel()
	claimController.cancel = claimControllerCtx.Cancel

	client := claimControllerCtx.Client()
	claimController.informerFactory = informers.NewSharedInformerFactory(client, 0 /* resync period */)
	controller, err := resourceclaim.NewController(
		klog.FromContext(claimControllerCtx),
		resourceclaim.Features{
			AdminAccess:     utilfeature.DefaultFeatureGate.Enabled(features.DRAAdminAccess),
			PrioritizedList: utilfeature.DefaultFeatureGate.Enabled(features.DRAPrioritizedList),
		},
		claimControllerCtx.Client(),
		claimController.informerFactory.Core().V1().Pods(),
		claimController.informerFactory.Resource().V1().ResourceClaims(),
		claimController.informerFactory.Resource().V1().ResourceClaimTemplates(),
	)
	tCtx.ExpectNoError(err, "create ResourceClaim controller")

	claimController.informerFactory.Start(claimControllerCtx.Done())
	claimController.wg.Go(func() {
		controller.Run(claimControllerCtx, 1 /* one worker to get more readable log output without interleaving */)
	})
	tCtx.Logf("Started the claimController for test %s.", tCtx.Name())
}

func (claimController *claimControllerSingleton) stop(tCtx ktesting.TContext) {
	claimController.mutex.Lock()
	defer claimController.mutex.Unlock()

	claimController.usageCount--
	if claimController.usageCount > 0 {
		// Still in use by some other test.
		return
	}

	claimController.rootCtx.Logf("Stopping the ResourceClaim controller after test %s...", tCtx.Name())
	if claimController.cancel != nil {
		claimController.cancel("test is done")
	}
	if claimController.informerFactory != nil {
		claimController.informerFactory.Shutdown()
	}
	claimController.wg.Wait()
}

type claimControllerKeyType int

var claimControllerKey claimControllerKeyType
