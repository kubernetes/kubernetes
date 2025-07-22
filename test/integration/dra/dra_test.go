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
	resourcealphaapi "k8s.io/api/resource/v1alpha3"
	resourceapi "k8s.io/api/resource/v1beta1"
	resourcev1beta2 "k8s.io/api/resource/v1beta2"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	resourceapiac "k8s.io/client-go/applyconfigurations/resource/v1beta1"
	"k8s.io/client-go/informers"
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
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
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

// createSlice creates the given ResourceSlice and removes it when the test is done.
func createSlice(tCtx ktesting.TContext, slice *resourceapi.ResourceSlice) *resourceapi.ResourceSlice {
	tCtx.Helper()
	slice, err := tCtx.Client().ResourceV1beta1().ResourceSlices().Create(tCtx, slice, metav1.CreateOptions{})
	tCtx.ExpectNoError(err, "create ResourceSlice")
	tCtx.CleanupCtx(func(tCtx ktesting.TContext) {
		tCtx.Log("Cleaning up ResourceSlice...")
		err := tCtx.Client().ResourceV1beta1().ResourceSlices().Delete(tCtx, slice.Name, metav1.DeleteOptions{})
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
	_, err := tCtx.Client().ResourceV1beta1().DeviceClasses().Create(tCtx, class, metav1.CreateOptions{})
	tCtx.ExpectNoError(err, "create class")
	tCtx.CleanupCtx(func(tCtx ktesting.TContext) {
		tCtx.Log("Cleaning up DeviceClass...")
		err := tCtx.Client().ResourceV1beta1().DeviceClasses().Delete(tCtx, class.Name, metav1.DeleteOptions{})
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
		if request.DeviceClassName != "" {
			request.DeviceClassName = class.Name
			continue
		}
		for e := range request.FirstAvailable {
			subRequest := &request.FirstAvailable[e]
			subRequest.DeviceClassName = class.Name
		}
	}
	claim, err := tCtx.Client().ResourceV1beta1().ResourceClaims(namespace).Create(tCtx, claim, metav1.CreateOptions{})
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
		"default": {
			f: func(tCtx ktesting.TContext) {
				tCtx.Run("Pod", func(tCtx ktesting.TContext) { testPod(tCtx, false) })
				tCtx.Run("APIDisabled", testAPIDisabled)
			},
		},
		"GA": {
			// TODO (https://github.com/kubernetes/kubernetes/issues/131903): remove enabling the beta when promoting to GA.
			apis: map[schema.GroupVersion]bool{
				resourceapi.SchemeGroupVersion:     true,
				resourcev1beta2.SchemeGroupVersion: true,
			},
			features: map[featuregate.Feature]bool{
				features.DynamicResourceAllocation: true,
				// TODO: replace specific list with AllBeta once DRA is not beta.
				features.DRAResourceClaimDeviceStatus: false,
				// featuregate.Feature("AllBeta"):     false,
			},
			f: func(tCtx ktesting.TContext) {
				tCtx.Run("AdminAccess", func(tCtx ktesting.TContext) { testAdminAccess(tCtx, false) })
				tCtx.Run("FilterTimeout", testFilterTimeout)
				tCtx.Run("PrioritizedList", func(tCtx ktesting.TContext) { testPrioritizedList(tCtx, false) })
				tCtx.Run("Pod", func(tCtx ktesting.TContext) { testPod(tCtx, true) })
				tCtx.Run("PublishResourceSlices", func(tCtx ktesting.TContext) {
					testPublishResourceSlices(tCtx, true, features.DRADeviceTaints, features.DRAPartitionableDevices)
				})
				tCtx.Run("ResourceClaimDeviceStatus", func(tCtx ktesting.TContext) { testResourceClaimDeviceStatus(tCtx, false) })
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
					testPublishResourceSlices(tCtx, true, features.DRADeviceTaints, features.DRAPartitionableDevices)
				})
				tCtx.Run("ResourceClaimDeviceStatus", func(tCtx ktesting.TContext) { testResourceClaimDeviceStatus(tCtx, true) })
			},
		},
		"v1beta1": {
			apis: map[schema.GroupVersion]bool{
				resourceapi.SchemeGroupVersion: true,
			},
			features: map[featuregate.Feature]bool{features.DynamicResourceAllocation: true},
			f: func(tCtx ktesting.TContext) {
				tCtx.Run("PublishResourceSlices", func(tCtx ktesting.TContext) {
					testPublishResourceSlices(tCtx, false, features.DRADeviceTaints, features.DRAPartitionableDevices)
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
					testPublishResourceSlices(tCtx, true, features.DRADeviceTaints, features.DRAPartitionableDevices)
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
				tCtx.Run("PublishResourceSlices", func(tCtx ktesting.TContext) { testPublishResourceSlices(tCtx, true) })
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
	claim1.Spec.Devices.Requests[0].AdminAccess = ptr.To(true)
	// create claim with AdminAccess in non-admin namespace
	_, err := tCtx.Client().ResourceV1beta1().ResourceClaims(namespace).Create(tCtx, claim1, metav1.CreateOptions{})
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
	claim.Spec.Devices.Requests[0].Count = int64(devicesPerSlice + 1) // Impossible to allocate.
	claim := createClaim(tCtx, namespace, "", class, claim)

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
			Name:  fmt.Sprintf("dev-%d", devicesPerSlice),
			Basic: &resourceapi.BasicDevice{},
		})
		_, err := tCtx.Client().ResourceV1beta1().ResourceSlices().Update(tCtx, createdOtherSlice, metav1.UpdateOptions{})
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

func testPublishResourceSlices(tCtx ktesting.TContext, haveLatestAPI bool, disabledFeatures ...featuregate.Feature) {
	tCtx.Parallel()

	namespace := createTestNamespace(tCtx, nil)
	driverName := namespace + ".example.com"
	listDriverSlices := metav1.ListOptions{
		FieldSelector: resourcev1beta2.ResourceSliceSelectorDriver + "=" + driverName,
	}
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
									TimeAdded: ptr.To(metav1.Time{Time: time.Now().Truncate(time.Second)}),
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

	// Manually turn into the expected slices, considering that some fields get dropped.
	expectedResources := resources.DeepCopy()
	var expectedSliceSpecs []resourcev1beta2.ResourceSliceSpec
	for _, sl := range expectedResources.Pools[poolName].Slices {
		expectedSliceSpecs = append(expectedSliceSpecs, resourcev1beta2.ResourceSliceSpec{
			Driver: driverName,
			Pool: resourcev1beta2.ResourcePool{
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
						"Name":             gomega.Equal(device.Name),
						"Attributes":       gomega.Equal(device.Attributes),
						"Capacity":         gomega.Equal(device.Capacity),
						"ConsumesCounters": gomega.Equal(device.ConsumesCounters),
						"NodeName":         matchPointer(device.NodeName),
						"NodeSelector":     matchPointer(device.NodeSelector),
						"AllNodes":         matchPointer(device.AllNodes),
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
		slices, err := tCtx.Client().ResourceV1beta2().ResourceSlices().List(tCtx, listDriverSlices)
		tCtx.ExpectNoError(err, "list slices")
		gomega.NewGomegaWithT(tCtx).Expect(slices.Items).Should(gomega.ConsistOf(expectedSlices...))
	}

	deleteSlices := func(tCtx ktesting.TContext) {
		tCtx.Helper()

		// At least one of the APIs must be enabled...
		var err error
		err = tCtx.Client().ResourceV1beta1().ResourceSlices().DeleteCollection(tCtx, metav1.DeleteOptions{}, listDriverSlices)
		if err == nil {
			return
		}
		err = tCtx.Client().ResourceV1beta2().ResourceSlices().DeleteCollection(tCtx, metav1.DeleteOptions{}, listDriverSlices)
		if err == nil {
			return
		}
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
		pool.Slices[0].Devices[0].Attributes = map[resourcev1beta2.QualifiedName]resourcev1beta2.DeviceAttribute{"empty": {}}
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
			tCtx.ExpectNoError(tCtx.Client().ResourceV1beta2().ResourceSlices().DeleteCollection(tCtx, metav1.DeleteOptions{}, listDriverSlices), "delete driver slices")
			expectedStats.NumCreates += int64(len(expectedSlices))
			ktesting.Eventually(tCtx, getStats).WithTimeout(syncDelay + 5*time.Second).Should(gomega.Equal(expectedStats))
			expectSlices(tCtx)
		}
	})

	tCtx.Run("fix-after-update", func(tCtx ktesting.TContext) {
		_, getStats, expectedStats := setup(tCtx)

		// Stress the controller by repeatedly updatings the slices.
		for i := 0; i < 2; i++ {
			slices, err := tCtx.Client().ResourceV1beta2().ResourceSlices().List(tCtx, listDriverSlices)
			tCtx.ExpectNoError(err, "list slices")
			for _, slice := range slices.Items {
				if slice.Spec.Devices[0].Attributes == nil {
					slice.Spec.Devices[0].Attributes = make(map[resourcev1beta2.QualifiedName]resourcev1beta2.DeviceAttribute)
				}
				slice.Spec.Devices[0].Attributes["someUnwantedAttribute"] = resourcev1beta2.DeviceAttribute{BoolValue: ptr.To(true)}
				_, err := tCtx.Client().ResourceV1beta2().ResourceSlices().Update(tCtx, &slice, metav1.UpdateOptions{})
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
						Name:            "foo",
						DeviceClassName: "foo",
					},
				},
			},
		},
	}

	claim, err := tCtx.Client().ResourceV1beta1().ResourceClaims(namespace).Create(tCtx, claim, metav1.CreateOptions{})
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
	updatedClaim, err := tCtx.Client().ResourceV1beta1().ResourceClaims(namespace).UpdateStatus(tCtx, claim, metav1.UpdateOptions{})
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
	claim, err = tCtx.Client().ResourceV1beta1().ResourceClaims(namespace).UpdateStatus(tCtx, claim, metav1.UpdateOptions{})
	tCtx.ExpectNoError(err, "add allocation result")

	// Now adding the device status should work.
	claim.Status.Devices = deviceStatus
	claim, err = tCtx.Client().ResourceV1beta1().ResourceClaims(namespace).UpdateStatus(tCtx, claim, metav1.UpdateOptions{})
	tCtx.ExpectNoError(err, "add device status")
	require.Equal(tCtx, deviceStatus, claim.Status.Devices, "after adding device status")

	// Strip the RawExtension. SSA re-encodes it, which causes negligble differences that nonetheless break assert.Equal.
	claim.Status.Devices[0].Data = nil
	deviceStatus[0].Data = nil
	claim, err = tCtx.Client().ResourceV1beta1().ResourceClaims(namespace).UpdateStatus(tCtx, claim, metav1.UpdateOptions{})
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
	claim, err = tCtx.Client().ResourceV1beta1().ResourceClaims(namespace).ApplyStatus(tCtx, claimAC, metav1.ApplyOptions{
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
	claim, err = tCtx.Client().ResourceV1beta1().ResourceClaims(namespace).ApplyStatus(tCtx, claimAC, metav1.ApplyOptions{
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
	claim, err = tCtx.Client().ResourceV1beta1().ResourceClaims(namespace).ApplyStatus(tCtx, claimAC, metav1.ApplyOptions{
		Force:        true,
		FieldManager: "manager-1",
	})
	tCtx.ExpectNoError(err, "update device status two")
	require.Equal(tCtx, deviceStatus, claim.Status.Devices, "after updating device status two")
	claimAC = resourceapiac.ResourceClaim(claim.Name, claim.Namespace)
	deviceStatus = deviceStatus[0:2]
	claim, err = tCtx.Client().ResourceV1beta1().ResourceClaims(namespace).ApplyStatus(tCtx, claimAC, metav1.ApplyOptions{
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

func matchPointer[T any](p *T) gtypes.GomegaMatcher {
	if p == nil {
		return gomega.BeNil()
	}
	return gstruct.PointTo(gomega.Equal(*p))
}
