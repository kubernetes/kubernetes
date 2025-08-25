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
	"regexp"
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
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	resourceapiac "k8s.io/client-go/applyconfigurations/resource/v1"
	"k8s.io/client-go/informers"
	"k8s.io/client-go/kubernetes"
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

// createSlice creates the given ResourceSlice and removes it when the test is done.
func createSlice(tCtx ktesting.TContext, slice *resourceapi.ResourceSlice) *resourceapi.ResourceSlice {
	tCtx.Helper()
	slice, err := tCtx.Client().ResourceV1().ResourceSlices().Create(tCtx, slice, metav1.CreateOptions{})
	tCtx.ExpectNoError(err, "create ResourceSlice")
	tCtx.CleanupCtx(func(tCtx ktesting.TContext) {
		tCtx.Log("Cleaning up ResourceSlice...")
		err := tCtx.Client().ResourceV1().ResourceSlices().Delete(tCtx, slice.Name, metav1.DeleteOptions{})
		tCtx.ExpectNoError(err, "delete ResourceSlice")
	})
	return slice
}

// createTestClass creates a DeviceClass with a driver name derived from the test namespace
func createTestClass(tCtx ktesting.TContext, namespace string) (*resourceapi.DeviceClass, string) {
	tCtx.Helper()
	driverName := namespace + ".driver"
	class := class.DeepCopy()
	class.Name = namespace + ".class"
	class.Spec.Selectors = []resourceapi.DeviceSelector{{
		CEL: &resourceapi.CELDeviceSelector{
			Expression: fmt.Sprintf("device.driver == %q", driverName),
		},
	}}
	_, err := tCtx.Client().ResourceV1().DeviceClasses().Create(tCtx, class, metav1.CreateOptions{})
	tCtx.ExpectNoError(err, "create class")
	tCtx.CleanupCtx(func(tCtx ktesting.TContext) {
		tCtx.Log("Cleaning up DeviceClass...")
		err := tCtx.Client().ResourceV1().DeviceClasses().Delete(tCtx, class.Name, metav1.DeleteOptions{})
		tCtx.ExpectNoError(err, "delete class")
	})

	return class, driverName
}

// createClaim creates a claim and in the namespace.
// The class must already exist and is used for all requests.
func createClaim(tCtx ktesting.TContext, namespace string, suffix string, class *resourceapi.DeviceClass, claim *resourceapi.ResourceClaim) *resourceapi.ResourceClaim {
	tCtx.Helper()
	claim = claim.DeepCopy()
	claim.Namespace = namespace
	claim.Name += suffix
	claimName := claim.Name
	for i := range claim.Spec.Devices.Requests {
		request := &claim.Spec.Devices.Requests[i]
		if request.Exactly != nil && request.Exactly.DeviceClassName != "" {
			request.Exactly.DeviceClassName = class.Name
			continue
		}
		for e := range request.FirstAvailable {
			subRequest := &request.FirstAvailable[e]
			subRequest.DeviceClassName = class.Name
		}
	}
	claim, err := tCtx.Client().ResourceV1().ResourceClaims(namespace).Create(tCtx, claim, metav1.CreateOptions{})
	tCtx.ExpectNoError(err, "create claim "+claimName)
	return claim
}

// createPod create a pod in the namespace, referencing the given claim.
func createPod(tCtx ktesting.TContext, namespace string, suffix string, claim *resourceapi.ResourceClaim, pod *v1.Pod) *v1.Pod {
	tCtx.Helper()
	pod = pod.DeepCopy()
	pod.Name += suffix
	podName := pod.Name
	pod.Namespace = namespace
	pod.Spec.ResourceClaims[0].ResourceClaimName = &claim.Name
	pod, err := tCtx.Client().CoreV1().Pods(namespace).Create(tCtx, pod, metav1.CreateOptions{})
	tCtx.ExpectNoError(err, "create pod "+podName)
	tCtx.CleanupCtx(func(tCtx ktesting.TContext) {
		tCtx.Log("Cleaning up Pod...")
		err := tCtx.Client().CoreV1().Pods(namespace).Delete(tCtx, pod.Name, metav1.DeleteOptions{})
		tCtx.ExpectNoError(err, "delete Pod")
	})
	return pod
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
				tCtx.Run("PublishResourceSlices", func(tCtx ktesting.TContext) { testPublishResourceSlices(tCtx, true) })
				// note testExtendedResource depends on testPublishResourceSlices to provide the devices
				tCtx.Run("ExtendedResource", func(tCtx ktesting.TContext) { testExtendedResource(tCtx, true) })
				tCtx.Run("ResourceClaimDeviceStatus", func(tCtx ktesting.TContext) { testResourceClaimDeviceStatus(tCtx, true) })
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
	_, err := tCtx.Client().ResourceV1().ResourceClaims(claim.Namespace).Create(tCtx, claim, metav1.CreateOptions{})
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
	claim, err := tCtx.Client().ResourceV1().ResourceClaims(namespace).Create(tCtx, claim, metav1.CreateOptions{})
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
	_, err := tCtx.Client().ResourceV1().ResourceClaims(namespace).Create(tCtx, claim1, metav1.CreateOptions{})
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
		claim2, err := tCtx.Client().ResourceV1().ResourceClaims(adminNS).Create(tCtx, claim2, metav1.CreateOptions{})
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
		pod := createPod(tCtx, namespace, "", claim, podWithClaimName)
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
		pod := createPod(tCtx, namespace, "", claim, podWithClaimName)
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
	pod := createPod(tCtx, namespace, "", claim, podWithClaimName)
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
}

func testExtendedResource(tCtx ktesting.TContext, enabled bool) {
	tCtx.Parallel()
	c, err := tCtx.Client().ResourceV1().DeviceClasses().Create(tCtx, classWithExtendedResource, metav1.CreateOptions{})
	tCtx.ExpectNoError(err, "create class")
	if enabled {
		require.NotEmpty(tCtx, c.Spec.ExtendedResourceName, "should store ExtendedResourceName")
	}
	namespace := createTestNamespace(tCtx, nil)
	tCtx.Run("scheduler", func(tCtx ktesting.TContext) {
		startScheduler(tCtx)

		pod := podWithExtendedResource.DeepCopy()
		pod.Namespace = namespace
		_, err := tCtx.Client().CoreV1().Pods(namespace).Create(tCtx, pod, metav1.CreateOptions{})
		tCtx.ExpectNoError(err, "create pod")
		schedulingAttempted := gomega.HaveField("Status.Conditions", gomega.ContainElement(
			gstruct.MatchFields(gstruct.IgnoreExtras, gstruct.Fields{
				"Type":    gomega.Equal(v1.PodScheduled),
				"Status":  gomega.Equal(v1.ConditionFalse),
				"Reason":  gomega.Equal("Unschedulable"),
				"Message": gomega.Equal("0/2 nodes are available: 2 Insufficient my-example.com/my-extended-resource. no new claims to deallocate, preemption: 0/2 nodes are available: 2 Preemption is not helpful for scheduling."),
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
						SharedCounters: []resourceapi.CounterSet{{
							Name: "gpu-0",
							Counters: map[string]resourceapi.Counter{
								"mem": {Value: resource.MustParse("1")},
							},
						}},
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
	for _, disabled := range disabledFeatures {
		switch disabled {
		case features.DRADeviceTaints:
			for i := range expectedSliceSpecs {
				for e := range expectedSliceSpecs[i].Devices {
					expectedSliceSpecs[i].Devices[e].Taints = nil
				}
			}
		case features.DRAPartitionableDevices:
			for i := range expectedSliceSpecs {
				expectedSliceSpecs[i].SharedCounters = nil
				for e := range expectedSliceSpecs[i].Devices {
					expectedSliceSpecs[i].Devices[e].ConsumesCounters = nil
				}
			}
		case features.DRADeviceBindingConditions:
			for i := range expectedSliceSpecs {
				for e := range expectedSliceSpecs[i].Devices {
					expectedSliceSpecs[i].Devices[e].BindingConditions = nil
					expectedSliceSpecs[i].Devices[e].BindingFailureConditions = nil
					expectedSliceSpecs[i].Devices[e].BindsToNode = nil
				}
			}
		default:
			tCtx.Fatalf("faulty test, case for %s missing", disabled)
		}
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

	claim, err := tCtx.Client().ResourceV1().ResourceClaims(namespace).Create(tCtx, claim, metav1.CreateOptions{})
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

// testMaxResourceSlice creates a ResourceSlice that is as large as possible
// and prints some information about it.
func testMaxResourceSlice(tCtx ktesting.TContext) {
	slice := NewMaxResourceSlice()
	createdSlice, err := tCtx.Client().ResourceV1().ResourceSlices().Create(tCtx, slice, metav1.CreateOptions{})
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

	_, err := tCtx.Client().ResourceV1().ResourceClaimTemplates(namespace).Create(tCtx, template1, metav1.CreateOptions{})
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

	_, err = tCtx.Client().CoreV1().Pods(namespace).Create(tCtx, pod1, metav1.CreateOptions{})
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

	_, err = tCtx.Client().ResourceV1().ResourceClaimTemplates(adminNS).Create(tCtx, template2, metav1.CreateOptions{})
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

	_, err = tCtx.Client().CoreV1().Pods(adminNS).Create(tCtx, pod2, metav1.CreateOptions{})
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

	_, err = tCtx.Client().ResourceV1().ResourceClaimTemplates(namespace).Create(tCtx, invalidTemplate, metav1.CreateOptions{})
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

	_, err = tCtx.Client().ResourceV1().ResourceClaimTemplates(namespace).Create(tCtx, template4, metav1.CreateOptions{})
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

	_, err = tCtx.Client().CoreV1().Pods(namespace).Create(tCtx, pod4, metav1.CreateOptions{})
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

// testDeviceBindingConditions tests scheduling with mixed devices: one with BindingConditions, one without.
// It verifies that the scheduler prioritizes the device without BindingConditions for the first pod.
// The second pod then uses the device with BindingConditions. The test checks that the scheduler retries
// after an initial binding failure of the second pod, ensuring successful scheduling after rescheduling.
func testDeviceBindingConditions(tCtx ktesting.TContext, enabled bool) {
	namespace := createTestNamespace(tCtx, nil)
	class, driverName := createTestClass(tCtx, namespace)

	nodeName := "worker-0"
	poolWithBinding := nodeName + "-with-binding"
	poolWithoutBinding := nodeName + "-without-binding"
	bindingCondition := "attached"
	failureCondition := "failed"
	startScheduler(tCtx)

	slice := &resourceapi.ResourceSlice{
		ObjectMeta: metav1.ObjectMeta{
			GenerateName: namespace + "-",
		},
		Spec: resourceapi.ResourceSliceSpec{
			NodeName: &nodeName,
			Pool: resourceapi.ResourcePool{
				Name:               poolWithBinding,
				ResourceSliceCount: 1,
			},
			Driver: driverName,
			Devices: []resourceapi.Device{
				{
					Name:                     "with-binding",
					BindingConditions:        []string{bindingCondition},
					BindingFailureConditions: []string{failureCondition},
				},
			},
		},
	}
	slice, err := tCtx.Client().ResourceV1().ResourceSlices().Create(tCtx, slice, metav1.CreateOptions{})
	tCtx.ExpectNoError(err, "create slice")

	haveBindingConditionFields := len(slice.Spec.Devices[0].BindingConditions) > 0 || len(slice.Spec.Devices[0].BindingFailureConditions) > 0
	if !enabled {
		if haveBindingConditionFields {
			tCtx.Fatalf("Expected device binding condition fields to get dropped, got instead:\n%s", format.Object(slice, 1))
		}
		return
	}
	if !haveBindingConditionFields {
		tCtx.Fatalf("Expected device binding condition fields to be stored, got instead:\n%s", format.Object(slice, 1))
	}

	sliceWithoutBinding := &resourceapi.ResourceSlice{
		ObjectMeta: metav1.ObjectMeta{
			GenerateName: namespace + "-without-binding-",
		},
		Spec: resourceapi.ResourceSliceSpec{
			NodeName: &nodeName,
			Pool: resourceapi.ResourcePool{
				Name:               poolWithoutBinding,
				ResourceSliceCount: 1,
			},
			Driver: driverName,
			Devices: []resourceapi.Device{
				{
					Name: "without-binding",
				},
			},
		},
	}
	_, err = tCtx.Client().ResourceV1().ResourceSlices().Create(tCtx, sliceWithoutBinding, metav1.CreateOptions{})
	tCtx.ExpectNoError(err, "create slice without binding conditions")

	// Schedule first pod and wait for the scheduler to reach the binding phase, which marks the claim as allocated.
	start := time.Now()
	claim1 := createClaim(tCtx, namespace, "-a", class, claim)
	pod := createPod(tCtx, namespace, "-a", claim1, podWithClaimName)
	ktesting.Eventually(tCtx, func(tCtx ktesting.TContext) *resourceapi.ResourceClaim {
		c, err := tCtx.Client().ResourceV1().ResourceClaims(namespace).Get(tCtx, claim1.Name, metav1.GetOptions{})
		tCtx.ExpectNoError(err)
		claim1 = c
		return claim1
	}).WithTimeout(10*time.Second).WithPolling(time.Second).Should(gomega.HaveField("Status.Allocation", gomega.Not(gomega.BeNil())), "Claim should have been allocated.")
	end := time.Now()
	gomega.NewWithT(tCtx).Expect(claim1).To(gomega.HaveField("Status.Allocation", gstruct.PointTo(gstruct.MatchFields(gstruct.IgnoreExtras, gstruct.Fields{
		"Devices": gomega.Equal(resourceapi.DeviceAllocationResult{
			Results: []resourceapi.DeviceRequestAllocationResult{{
				Request: claim1.Spec.Devices.Requests[0].Name,
				Driver:  driverName,
				Pool:    poolWithoutBinding,
				Device:  "without-binding",
			}}}),
		// NodeSelector intentionally not checked - that's covered elsewhere.
		"AllocationTimestamp": gomega.HaveField("Time", gomega.And(
			gomega.BeTemporally(">=", start.Truncate(time.Second) /* may get rounded down during round-tripping */),
			gomega.BeTemporally("<=", end),
		)),
	}))), "first allocated claim")

	err = waitForPodScheduled(tCtx, tCtx.Client(), namespace, pod.Name)
	tCtx.ExpectNoError(err, "first pod scheduled")

	// Second pod should get the device with binding conditions.
	claim2 := createClaim(tCtx, namespace, "-b", class, claim)
	pod = createPod(tCtx, namespace, "-b", claim2, podWithClaimName)
	ktesting.Eventually(tCtx, func(tCtx ktesting.TContext) *resourceapi.ResourceClaim {
		c, err := tCtx.Client().ResourceV1().ResourceClaims(namespace).Get(tCtx, claim2.Name, metav1.GetOptions{})
		tCtx.ExpectNoError(err)
		claim2 = c
		return claim2
	}).WithTimeout(10*time.Second).WithPolling(time.Second).Should(gomega.HaveField("Status.Allocation", gomega.Not(gomega.BeNil())), "Claim should have been allocated.")
	end = time.Now()
	gomega.NewWithT(tCtx).Expect(claim2).To(gomega.HaveField("Status.Allocation", gstruct.PointTo(gstruct.MatchFields(gstruct.IgnoreExtras, gstruct.Fields{
		"Devices": gomega.Equal(resourceapi.DeviceAllocationResult{
			Results: []resourceapi.DeviceRequestAllocationResult{{
				Request:                  claim2.Spec.Devices.Requests[0].Name,
				Driver:                   driverName,
				Pool:                     poolWithBinding,
				Device:                   "with-binding",
				BindingConditions:        []string{bindingCondition},
				BindingFailureConditions: []string{failureCondition},
			}}}),
		// NodeSelector intentionally not checked - that's covered elsewhere.
		"AllocationTimestamp": gomega.HaveField("Time", gomega.And(
			gomega.BeTemporally(">=", start.Truncate(time.Second) /* may get rounded down during round-tripping */),
			gomega.BeTemporally("<=", end),
		)),
	}))), "second allocated claim")

	// fail the binding condition for the second claim, so that it gets scheduled later.
	claim2.Status.Devices = []resourceapi.AllocatedDeviceStatus{{
		Driver: driverName,
		Pool:   poolWithBinding,
		Device: "with-binding",
		Conditions: []metav1.Condition{{
			Type:               failureCondition,
			Status:             metav1.ConditionTrue,
			ObservedGeneration: claim2.Generation,
			LastTransitionTime: metav1.Now(),
			Reason:             "Testing",
			Message:            "The test has seen the allocation and is failing the binding.",
		}},
	}}

	claim2, err = tCtx.Client().ResourceV1().ResourceClaims(namespace).UpdateStatus(tCtx, claim2, metav1.UpdateOptions{})
	tCtx.ExpectNoError(err, "add binding failure condition to second claim")

	// Wait until the claim.status.Devices[0].Conditions become nil again after rescheduling.
	setConditionsFlag := false
	ktesting.Eventually(tCtx, func(tCtx ktesting.TContext) *resourceapi.ResourceClaim {
		c, err := tCtx.Client().ResourceV1().ResourceClaims(namespace).Get(tCtx, claim2.Name, metav1.GetOptions{})
		tCtx.ExpectNoError(err, "get claim")
		claim2 = c
		if claim2.Status.Devices != nil && len(claim2.Status.Devices[0].Conditions) != 0 {
			setConditionsFlag = true
		}
		if setConditionsFlag && len(claim2.Status.Devices) == 0 {
			// The scheduler has retried and removed the conditions.
			// This is the expected state. Finish waiting.
			return nil
		}
		return claim2
	}).WithTimeout(30*time.Second).WithPolling(time.Second).Should(gomega.BeNil(), "claim should not have any condition")

	// Allow the scheduler to proceed.
	claim2.Status.Devices = []resourceapi.AllocatedDeviceStatus{{
		Driver: driverName,
		Pool:   poolWithBinding,
		Device: "with-binding",
		Conditions: []metav1.Condition{{
			Type:               bindingCondition,
			Status:             metav1.ConditionTrue,
			ObservedGeneration: claim2.Generation,
			LastTransitionTime: metav1.Now(),
			Reason:             "Testing",
			Message:            "The test has seen the allocation.",
		}},
	}}

	claim2, err = tCtx.Client().ResourceV1().ResourceClaims(namespace).UpdateStatus(tCtx, claim2, metav1.UpdateOptions{})
	tCtx.ExpectNoError(err, "add binding condition to second claim")
	err = waitForPodScheduled(tCtx, tCtx.Client(), namespace, pod.Name)
	tCtx.ExpectNoError(err, "second pod scheduled")
}

func waitForPodScheduled(ctx context.Context, client kubernetes.Interface, namespace, podName string) error {
	timeout := time.After(60 * time.Second)
	tick := time.Tick(1 * time.Second)
	for {
		select {
		case <-timeout:
			return fmt.Errorf("timed out waiting for pod %s/%s to be scheduled", namespace, podName)
		case <-tick:
			pod, err := client.CoreV1().Pods(namespace).Get(ctx, podName, metav1.GetOptions{})
			if err != nil {
				continue
			}
			for _, cond := range pod.Status.Conditions {
				if cond.Type == v1.PodScheduled && cond.Status == v1.ConditionTrue {
					return nil
				}
			}
		}
	}
}
