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
	"k8s.io/kube-scheduler/config/v1beta1"
	"k8s.io/kube-scheduler/config/v1beta2"
	apiservertesting "k8s.io/kubernetes/cmd/kube-apiserver/app/testing"
	"k8s.io/kubernetes/pkg/scheduler"
	configtesting "k8s.io/kubernetes/pkg/scheduler/apis/config/testing"
	"k8s.io/kubernetes/pkg/scheduler/framework"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/serviceaffinity"
	frameworkruntime "k8s.io/kubernetes/pkg/scheduler/framework/runtime"
	st "k8s.io/kubernetes/pkg/scheduler/testing"
	testfwk "k8s.io/kubernetes/test/integration/framework"
	testutils "k8s.io/kubernetes/test/integration/util"
	imageutils "k8s.io/kubernetes/test/utils/image"
	"k8s.io/utils/pointer"
)

func TestServiceAffinityEnqueue(t *testing.T) {
	cfg := configtesting.V1beta1ToInternalWithDefaults(t, v1beta1.KubeSchedulerConfiguration{
		Profiles: []v1beta1.KubeSchedulerProfile{{
			SchedulerName: pointer.StringPtr(v1.DefaultSchedulerName),
			Plugins: &v1beta1.Plugins{
				PreFilter: &v1beta1.PluginSet{
					Enabled: []v1beta1.Plugin{
						{Name: serviceaffinity.Name},
					},
				},
				Filter: &v1beta1.PluginSet{
					Enabled: []v1beta1.Plugin{
						{Name: serviceaffinity.Name},
					},
				},
			},
			PluginConfig: []v1beta1.PluginConfig{
				{
					Name: serviceaffinity.Name,
					Args: runtime.RawExtension{
						Object: &v1beta1.ServiceAffinityArgs{
							AffinityLabels: []string{"hostname"},
						},
					},
				},
			},
		}},
	})

	// Use zero backoff seconds to bypass backoffQ.
	testCtx := testutils.InitTestSchedulerWithOptions(
		t,
		testutils.InitTestAPIServer(t, "serviceaffinity-enqueue", nil),
		nil,
		scheduler.WithProfiles(cfg.Profiles...),
		scheduler.WithPodInitialBackoffSeconds(0),
		scheduler.WithPodMaxBackoffSeconds(0),
	)
	testutils.SyncInformerFactory(testCtx)
	// It's intended to not start the scheduler's queue, and hence to
	// not start any flushing logic. We will pop and schedule the Pods manually later.
	defer testutils.CleanupTest(t, testCtx)

	cs, ns, ctx := testCtx.ClientSet, testCtx.NS.Name, testCtx.Ctx
	// Create two Nodes.
	for i := 1; i <= 2; i++ {
		nodeName := fmt.Sprintf("node%d", i)
		capacity := map[v1.ResourceName]string{v1.ResourcePods: "1"}
		node := st.MakeNode().Name(nodeName).Label("hostname", nodeName).Capacity(capacity).Obj()
		if _, err := cs.CoreV1().Nodes().Create(ctx, node, metav1.CreateOptions{}); err != nil {
			t.Fatalf("Failed to create Node %q: %v", nodeName, err)
		}
	}

	// Create a Service.
	svc := &v1.Service{
		ObjectMeta: metav1.ObjectMeta{
			Name: "svc",
		},
		Spec: v1.ServiceSpec{
			Ports:    []v1.ServicePort{{Port: int32(80)}},
			Selector: map[string]string{"foo": "bar"},
		},
	}
	if _, err := cs.CoreV1().Services(ns).Create(ctx, svc, metav1.CreateOptions{}); err != nil {
		t.Fatalf("Failed to create Service %q: %v", svc.Name, err)
	}

	// Create two Pods.
	pause := imageutils.GetPauseImageName()
	for i := 1; i <= 2; i++ {
		podName := fmt.Sprintf("pod%d", i)
		pod := st.MakePod().Namespace(ns).Name(podName).Label("foo", "bar").Container(pause).Obj()
		// Make Pod1 an assigned Pod.
		if i == 1 {
			pod.Spec.NodeName = fmt.Sprintf("node%d", i)
		}
		if _, err := cs.CoreV1().Pods(ns).Create(ctx, pod, metav1.CreateOptions{}); err != nil {
			t.Fatalf("Failed to create Pod %q: %v", pod.Name, err)
		}
	}

	// Wait for pod2 to be present in the scheduling queue.
	if err := wait.Poll(time.Millisecond*200, wait.ForeverTestTimeout, func() (bool, error) {
		return len(testCtx.Scheduler.SchedulingQueue.PendingPods()) == 1, nil
	}); err != nil {
		t.Fatal(err)
	}

	// Pop Pod2 out. It should be unschedulable.
	podInfo := testCtx.Scheduler.NextPod()
	fwk, ok := testCtx.Scheduler.Profiles[podInfo.Pod.Spec.SchedulerName]
	if !ok {
		t.Fatalf("Cannot find the profile for Pod %v", podInfo.Pod.Name)
	}
	// Schedule the Pod manually.
	_, fitError := testCtx.Scheduler.Algorithm.Schedule(ctx, nil, fwk, framework.NewCycleState(), podInfo.Pod)
	// The fitError is expected to be:
	// 0/2 nodes are available: 1 Too many pods, 1 node(s) didn't match service affinity.
	if fitError == nil {
		t.Fatalf("Expect Pod %v to fail at scheduling.", podInfo.Pod.Name)
	}
	testCtx.Scheduler.Error(podInfo, fitError)

	// Scheduling cycle is incremented from 0 to 1 after NextPod() is called, so
	// pass a number larger than 1 to move Pod to unschedulableQ.
	testCtx.Scheduler.SchedulingQueue.AddUnschedulableIfNotPresent(podInfo, 10)

	// Trigger a Service event.
	// We expect this event to trigger moving the test Pod from unschedulableQ to activeQ.
	if err := cs.CoreV1().Services(ns).Delete(ctx, "svc", metav1.DeleteOptions{}); err != nil {
		t.Fatalf("Failed to delete service 'svc': %v", err)
	}

	// Now we should be able to pop the Pod from activeQ again.
	podInfo = testCtx.Scheduler.NextPod()
	if podInfo.Attempts != 2 {
		t.Errorf("Expected the Pod to be attempted 2 times, but got %v", podInfo.Attempts)
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
	cfg := configtesting.V1beta2ToInternalWithDefaults(t, v1beta2.KubeSchedulerConfiguration{
		Profiles: []v1beta2.KubeSchedulerProfile{{
			SchedulerName: pointer.StringPtr(v1.DefaultSchedulerName),
			Plugins: &v1beta2.Plugins{
				Filter: v1beta2.PluginSet{
					Enabled: []v1beta2.Plugin{
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
	testCtx = testutils.InitTestSchedulerWithOptions(
		t,
		testCtx,
		nil,
		scheduler.WithProfiles(cfg.Profiles...),
		scheduler.WithFrameworkOutOfTreeRegistry(registry),
		scheduler.WithPodInitialBackoffSeconds(0),
		scheduler.WithPodMaxBackoffSeconds(0),
	)
	testutils.SyncInformerFactory(testCtx)
	// It's intended to not start the scheduler's queue, and hence to
	// not start any flushing logic. We will pop and schedule the Pods manually later.
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
		return len(testCtx.Scheduler.SchedulingQueue.PendingPods()) == 1, nil
	}); err != nil {
		t.Fatal(err)
	}

	// Pop fake-pod out. It should be unschedulable.
	podInfo := testCtx.Scheduler.NextPod()
	fwk, ok := testCtx.Scheduler.Profiles[podInfo.Pod.Spec.SchedulerName]
	if !ok {
		t.Fatalf("Cannot find the profile for Pod %v", podInfo.Pod.Name)
	}
	// Schedule the Pod manually.
	_, fitError := testCtx.Scheduler.Algorithm.Schedule(ctx, nil, fwk, framework.NewCycleState(), podInfo.Pod)
	// The fitError is expected to be non-nil as it failed the fakeCRPlugin plugin.
	if fitError == nil {
		t.Fatalf("Expect Pod %v to fail at scheduling.", podInfo.Pod.Name)
	}
	testCtx.Scheduler.Error(podInfo, fitError)

	// Scheduling cycle is incremented from 0 to 1 after NextPod() is called, so
	// pass a number larger than 1 to move Pod to unschedulableQ.
	testCtx.Scheduler.SchedulingQueue.AddUnschedulableIfNotPresent(podInfo, 10)

	// Trigger a Custom Resource event.
	// We expect this event to trigger moving the test Pod from unschedulableQ to activeQ.
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
	podInfo = testCtx.Scheduler.NextPod()
	if podInfo.Attempts != 2 {
		t.Errorf("Expected the Pod to be attempted 2 times, but got %v", podInfo.Attempts)
	}
}
