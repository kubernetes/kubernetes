/*
Copyright 2021 The Kubernetes Authors.

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

package scheduler

import (
	"context"
	"fmt"
	"testing"
	"time"

	v1 "k8s.io/api/core/v1"
	apiextensionsv1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1"
	apiextensionsclient "k8s.io/apiextensions-apiserver/pkg/client/clientset/clientset"
	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/uuid"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/client-go/dynamic"
	"k8s.io/client-go/kubernetes"
	configv1 "k8s.io/kube-scheduler/config/v1"
	apiservertesting "k8s.io/kubernetes/cmd/kube-apiserver/app/testing"
	"k8s.io/kubernetes/pkg/scheduler"
	configtesting "k8s.io/kubernetes/pkg/scheduler/apis/config/testing"
	"k8s.io/kubernetes/pkg/scheduler/framework"
	frameworkruntime "k8s.io/kubernetes/pkg/scheduler/framework/runtime"
	st "k8s.io/kubernetes/pkg/scheduler/testing"
	testfwk "k8s.io/kubernetes/test/integration/framework"
	testutils "k8s.io/kubernetes/test/integration/util"
	imageutils "k8s.io/kubernetes/test/utils/image"
	"k8s.io/utils/pointer"
)

// TestCoreResourceEnqueue verify Pods failed by in-tree default plugins can be
// moved properly upon their registered events.
func TestCoreResourceEnqueue(t *testing.T) {
	// Use zero backoff seconds to bypass backoffQ.
	// It's intended to not start the scheduler's queue, and hence to
	// not start any flushing logic. We will pop and schedule the Pods manually later.
	testCtx := testutils.InitTestSchedulerWithOptions(
		t,
		testutils.InitTestAPIServer(t, "core-res-enqueue", nil),
		0,
		scheduler.WithPodInitialBackoffSeconds(0),
		scheduler.WithPodMaxBackoffSeconds(0),
	)
	testutils.SyncInformerFactory(testCtx)

	defer testutils.CleanupTest(t, testCtx)

	cs, ns, ctx := testCtx.ClientSet, testCtx.NS.Name, testCtx.Ctx
	// Create one Node with a taint.
	node := st.MakeNode().Name("fake-node").Capacity(map[v1.ResourceName]string{v1.ResourceCPU: "2"}).Obj()
	node.Spec.Taints = []v1.Taint{{Key: "foo", Effect: v1.TaintEffectNoSchedule}}
	if _, err := cs.CoreV1().Nodes().Create(ctx, node, metav1.CreateOptions{}); err != nil {
		t.Fatalf("Failed to create Node %q: %v", node.Name, err)
	}

	// Create two Pods that are both unschedulable.
	// - Pod1 is a best-effort Pod, but doesn't have the required toleration.
	// - Pod2 requests a large amount of CPU resource that the node cannot fit.
	//   Note: Pod2 will fail the tainttoleration plugin b/c that's ordered prior to noderesources.
	// - Pod3 has the required toleration, but requests a non-existing PVC.
	pod1 := st.MakePod().Namespace(ns).Name("pod1").Container("image").Obj()
	pod2 := st.MakePod().Namespace(ns).Name("pod2").Req(map[v1.ResourceName]string{v1.ResourceCPU: "4"}).Obj()
	pod3 := st.MakePod().Namespace(ns).Name("pod3").Toleration("foo").PVC("pvc").Container("image").Obj()
	for _, pod := range []*v1.Pod{pod1, pod2, pod3} {
		if _, err := cs.CoreV1().Pods(ns).Create(ctx, pod, metav1.CreateOptions{}); err != nil {
			t.Fatalf("Failed to create Pod %q: %v", pod.Name, err)
		}
	}

	// Wait for the three pods to be present in the scheduling queue.
	if err := wait.Poll(time.Millisecond*200, wait.ForeverTestTimeout, func() (bool, error) {
		pendingPods, _ := testCtx.Scheduler.SchedulingQueue.PendingPods()
		return len(pendingPods) == 3, nil
	}); err != nil {
		t.Fatal(err)
	}

	// Pop the three pods out. They should be unschedulable.
	for i := 0; i < 3; i++ {
		podInfo := nextPodOrDie(t, testCtx)
		fwk, ok := testCtx.Scheduler.Profiles[podInfo.Pod.Spec.SchedulerName]
		if !ok {
			t.Fatalf("Cannot find the profile for Pod %v", podInfo.Pod.Name)
		}
		// Schedule the Pod manually.
		_, fitError := testCtx.Scheduler.SchedulePod(ctx, fwk, framework.NewCycleState(), podInfo.Pod)
		if fitError == nil {
			t.Fatalf("Expect Pod %v to fail at scheduling.", podInfo.Pod.Name)
		}
		testCtx.Scheduler.FailureHandler(ctx, fwk, podInfo, fitError, v1.PodReasonUnschedulable, nil)
	}

	// Trigger a NodeTaintChange event.
	// We expect this event to trigger moving the test Pod from unschedulablePods to activeQ.
	node.Spec.Taints = nil
	if _, err := cs.CoreV1().Nodes().Update(ctx, node, metav1.UpdateOptions{}); err != nil {
		t.Fatalf("Failed to remove taints off the node: %v", err)
	}

	// Now we should be able to pop the Pod from activeQ again.
	podInfo := nextPodOrDie(t, testCtx)
	if podInfo.Attempts != 2 {
		t.Fatalf("Expected the Pod to be attempted 2 times, but got %v", podInfo.Attempts)
	}
	if got := podInfo.Pod.Name; got != "pod1" {
		t.Fatalf("Expected pod1 to be popped, but got %v", got)
	}

	// Pod2 and Pod3 are not expected to be popped out.
	// - Although the failure reason has been lifted, Pod2 still won't be moved to active due to
	//   the node event's preCheckForNode().
	// - Regarding Pod3, the NodeTaintChange event is irrelevant with its scheduling failure.
	podInfo = nextPod(t, testCtx)
	if podInfo != nil {
		t.Fatalf("Unexpected pod %v get popped out", podInfo.Pod.Name)
	}
}

var _ framework.FilterPlugin = &fakeCRPlugin{}
var _ framework.EnqueueExtensions = &fakeCRPlugin{}

type fakeCRPlugin struct{}

func (f *fakeCRPlugin) Name() string {
	return "fakeCRPlugin"
}

func (f *fakeCRPlugin) Filter(_ context.Context, _ *framework.CycleState, _ *v1.Pod, _ *framework.NodeInfo) *framework.Status {
	return framework.NewStatus(framework.Unschedulable, "always fail")
}

// EventsToRegister returns the possible events that may make a Pod
// failed by this plugin schedulable.
func (f *fakeCRPlugin) EventsToRegister() []framework.ClusterEvent {
	return []framework.ClusterEvent{
		{Resource: "foos.v1.example.com", ActionType: framework.All},
	}
}

// TestCustomResourceEnqueue constructs a fake plugin that registers custom resources
// to verify Pods failed by this plugin can be moved properly upon CR events.
func TestCustomResourceEnqueue(t *testing.T) {
	// Start API Server with apiextensions supported.
	server := apiservertesting.StartTestServerOrDie(
		t, apiservertesting.NewDefaultTestServerOptions(),
		[]string{"--disable-admission-plugins=ServiceAccount,TaintNodesByCondition", "--runtime-config=api/all=true"},
		testfwk.SharedEtcd(),
	)
	testCtx := &testutils.TestContext{}
	testCtx.Ctx, testCtx.CancelFn = context.WithCancel(context.Background())
	testCtx.CloseFn = func() { server.TearDownFn() }

	apiExtensionClient := apiextensionsclient.NewForConfigOrDie(server.ClientConfig)
	dynamicClient := dynamic.NewForConfigOrDie(server.ClientConfig)

	// Create a Foo CRD.
	fooCRD := &apiextensionsv1.CustomResourceDefinition{
		ObjectMeta: metav1.ObjectMeta{
			Name: "foos.example.com",
		},
		Spec: apiextensionsv1.CustomResourceDefinitionSpec{
			Group: "example.com",
			Scope: apiextensionsv1.NamespaceScoped,
			Names: apiextensionsv1.CustomResourceDefinitionNames{
				Plural: "foos",
				Kind:   "Foo",
			},
			Versions: []apiextensionsv1.CustomResourceDefinitionVersion{
				{
					Name:    "v1",
					Served:  true,
					Storage: true,
					Schema: &apiextensionsv1.CustomResourceValidation{
						OpenAPIV3Schema: &apiextensionsv1.JSONSchemaProps{
							Type: "object",
							Properties: map[string]apiextensionsv1.JSONSchemaProps{
								"field": {Type: "string"},
							},
						},
					},
				},
			},
		},
	}
	var err error
	fooCRD, err = apiExtensionClient.ApiextensionsV1().CustomResourceDefinitions().Create(testCtx.Ctx, fooCRD, metav1.CreateOptions{})
	if err != nil {
		t.Fatal(err)
	}

	registry := frameworkruntime.Registry{
		"fakeCRPlugin": func(_ runtime.Object, fh framework.Handle) (framework.Plugin, error) {
			return &fakeCRPlugin{}, nil
		},
	}
	cfg := configtesting.V1ToInternalWithDefaults(t, configv1.KubeSchedulerConfiguration{
		Profiles: []configv1.KubeSchedulerProfile{{
			SchedulerName: pointer.StringPtr(v1.DefaultSchedulerName),
			Plugins: &configv1.Plugins{
				Filter: configv1.PluginSet{
					Enabled: []configv1.Plugin{
						{Name: "fakeCRPlugin"},
					},
				},
			},
		}}})

	testCtx.KubeConfig = server.ClientConfig
	testCtx.ClientSet = kubernetes.NewForConfigOrDie(server.ClientConfig)
	testCtx.NS, err = testCtx.ClientSet.CoreV1().Namespaces().Create(testCtx.Ctx, &v1.Namespace{
		ObjectMeta: metav1.ObjectMeta{Name: fmt.Sprintf("cr-enqueue-%v", string(uuid.NewUUID()))}}, metav1.CreateOptions{})
	if err != nil && !errors.IsAlreadyExists(err) {
		t.Fatalf("Failed to integration test ns: %v", err)
	}

	// Use zero backoff seconds to bypass backoffQ.
	// It's intended to not start the scheduler's queue, and hence to
	// not start any flushing logic. We will pop and schedule the Pods manually later.
	testCtx = testutils.InitTestSchedulerWithOptions(
		t,
		testCtx,
		0,
		scheduler.WithProfiles(cfg.Profiles...),
		scheduler.WithFrameworkOutOfTreeRegistry(registry),
		scheduler.WithPodInitialBackoffSeconds(0),
		scheduler.WithPodMaxBackoffSeconds(0),
	)
	testutils.SyncInformerFactory(testCtx)

	defer testutils.CleanupTest(t, testCtx)

	cs, ns, ctx := testCtx.ClientSet, testCtx.NS.Name, testCtx.Ctx
	// Create one Node.
	node := st.MakeNode().Name("fake-node").Obj()
	if _, err := cs.CoreV1().Nodes().Create(ctx, node, metav1.CreateOptions{}); err != nil {
		t.Fatalf("Failed to create Node %q: %v", node.Name, err)
	}

	// Create a testing Pod.
	pause := imageutils.GetPauseImageName()
	pod := st.MakePod().Namespace(ns).Name("fake-pod").Container(pause).Obj()
	if _, err := cs.CoreV1().Pods(ns).Create(ctx, pod, metav1.CreateOptions{}); err != nil {
		t.Fatalf("Failed to create Pod %q: %v", pod.Name, err)
	}

	// Wait for the testing Pod to be present in the scheduling queue.
	if err := wait.Poll(time.Millisecond*200, wait.ForeverTestTimeout, func() (bool, error) {
		pendingPods, _ := testCtx.Scheduler.SchedulingQueue.PendingPods()
		return len(pendingPods) == 1, nil
	}); err != nil {
		t.Fatal(err)
	}

	// Pop fake-pod out. It should be unschedulable.
	podInfo := nextPodOrDie(t, testCtx)
	fwk, ok := testCtx.Scheduler.Profiles[podInfo.Pod.Spec.SchedulerName]
	if !ok {
		t.Fatalf("Cannot find the profile for Pod %v", podInfo.Pod.Name)
	}
	// Schedule the Pod manually.
	_, fitError := testCtx.Scheduler.SchedulePod(ctx, fwk, framework.NewCycleState(), podInfo.Pod)
	// The fitError is expected to be non-nil as it failed the fakeCRPlugin plugin.
	if fitError == nil {
		t.Fatalf("Expect Pod %v to fail at scheduling.", podInfo.Pod.Name)
	}
	testCtx.Scheduler.FailureHandler(ctx, fwk, podInfo, fitError, v1.PodReasonUnschedulable, nil)

	// Scheduling cycle is incremented from 0 to 1 after NextPod() is called, so
	// pass a number larger than 1 to move Pod to unschedulablePods.
	testCtx.Scheduler.SchedulingQueue.AddUnschedulableIfNotPresent(podInfo, 10)

	// Trigger a Custom Resource event.
	// We expect this event to trigger moving the test Pod from unschedulablePods to activeQ.
	crdGVR := schema.GroupVersionResource{Group: fooCRD.Spec.Group, Version: fooCRD.Spec.Versions[0].Name, Resource: "foos"}
	crClient := dynamicClient.Resource(crdGVR).Namespace(ns)
	if _, err := crClient.Create(ctx, &unstructured.Unstructured{
		Object: map[string]interface{}{
			"apiVersion": "example.com/v1",
			"kind":       "Foo",
			"metadata":   map[string]interface{}{"name": "foo1"},
		},
	}, metav1.CreateOptions{}); err != nil {
		t.Fatalf("Unable to create cr: %v", err)
	}

	// Now we should be able to pop the Pod from activeQ again.
	podInfo = nextPodOrDie(t, testCtx)
	if podInfo.Attempts != 2 {
		t.Errorf("Expected the Pod to be attempted 2 times, but got %v", podInfo.Attempts)
	}
}
