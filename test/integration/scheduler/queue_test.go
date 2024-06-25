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
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/uuid"
	"k8s.io/apimachinery/pkg/util/wait"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/client-go/dynamic"
	"k8s.io/client-go/kubernetes"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/klog/v2"
	configv1 "k8s.io/kube-scheduler/config/v1"
	apiservertesting "k8s.io/kubernetes/cmd/kube-apiserver/app/testing"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/scheduler"
	configtesting "k8s.io/kubernetes/pkg/scheduler/apis/config/testing"
	"k8s.io/kubernetes/pkg/scheduler/framework"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/defaultbinder"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/names"
	frameworkruntime "k8s.io/kubernetes/pkg/scheduler/framework/runtime"
	st "k8s.io/kubernetes/pkg/scheduler/testing"
	testfwk "k8s.io/kubernetes/test/integration/framework"
	testutils "k8s.io/kubernetes/test/integration/util"
	imageutils "k8s.io/kubernetes/test/utils/image"
	"k8s.io/utils/pointer"
)

func TestSchedulingGates(t *testing.T) {
	tests := []struct {
		name     string
		pods     []*v1.Pod
		schedule []string
		delete   []string
		rmGates  []string
	}{
		{
			name: "regular pods",
			pods: []*v1.Pod{
				st.MakePod().Name("p1").Container("pause").Obj(),
				st.MakePod().Name("p2").Container("pause").Obj(),
			},
			schedule: []string{"p1", "p2"},
		},
		{
			name: "one pod carrying scheduling gates",
			pods: []*v1.Pod{
				st.MakePod().Name("p1").SchedulingGates([]string{"foo"}).Container("pause").Obj(),
				st.MakePod().Name("p2").Container("pause").Obj(),
			},
			schedule: []string{"p2"},
		},
		{
			name: "two pod carrying scheduling gates, and remove gates of one pod",
			pods: []*v1.Pod{
				st.MakePod().Name("p1").SchedulingGates([]string{"foo"}).Container("pause").Obj(),
				st.MakePod().Name("p2").SchedulingGates([]string{"bar"}).Container("pause").Obj(),
				st.MakePod().Name("p3").Container("pause").Obj(),
			},
			schedule: []string{"p3"},
			rmGates:  []string{"p2"},
		},
		{
			name: "gated pod schedulable after deleting the scheduled pod and removing gate",
			pods: []*v1.Pod{
				st.MakePod().Name("p1").SchedulingGates([]string{"foo"}).Container("pause").Obj(),
				st.MakePod().Name("p2").Container("pause").Obj(),
			},
			schedule: []string{"p2"},
			delete:   []string{"p2"},
			rmGates:  []string{"p1"},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Use zero backoff seconds to bypass backoffQ.
			// It's intended to not start the scheduler's queue, and hence to
			// not start any flushing logic. We will pop and schedule the Pods manually later.
			testCtx := testutils.InitTestSchedulerWithOptions(
				t,
				testutils.InitTestAPIServer(t, "pod-scheduling-gates", nil),
				0,
				scheduler.WithPodInitialBackoffSeconds(0),
				scheduler.WithPodMaxBackoffSeconds(0),
			)
			testutils.SyncSchedulerInformerFactory(testCtx)

			cs, ns, ctx := testCtx.ClientSet, testCtx.NS.Name, testCtx.Ctx

			// Create node, so we can schedule pods.
			node := st.MakeNode().Name("node").Obj()
			if _, err := cs.CoreV1().Nodes().Create(ctx, node, metav1.CreateOptions{}); err != nil {
				t.Fatal("Failed to create node")

			}

			// Create pods.
			for _, p := range tt.pods {
				p.Namespace = ns
				if _, err := cs.CoreV1().Pods(ns).Create(ctx, p, metav1.CreateOptions{}); err != nil {
					t.Fatalf("Failed to create Pod %q: %v", p.Name, err)
				}
			}

			// Wait for the pods to be present in the scheduling queue.
			if err := wait.PollUntilContextTimeout(ctx, time.Millisecond*200, wait.ForeverTestTimeout, false, func(ctx context.Context) (bool, error) {
				pendingPods, _ := testCtx.Scheduler.SchedulingQueue.PendingPods()
				return len(pendingPods) == len(tt.pods), nil
			}); err != nil {
				t.Fatal(err)
			}

			// Schedule pods.
			for _, podName := range tt.schedule {
				testCtx.Scheduler.ScheduleOne(testCtx.Ctx)
				if err := wait.PollUntilContextTimeout(ctx, time.Millisecond*200, wait.ForeverTestTimeout, false, testutils.PodScheduled(cs, ns, podName)); err != nil {
					t.Fatalf("Failed to schedule %s", podName)
				}
			}

			// Delete pods, which triggers AssignedPodDelete event in the scheduling queue.
			for _, podName := range tt.delete {
				if err := cs.CoreV1().Pods(ns).Delete(ctx, podName, metav1.DeleteOptions{}); err != nil {
					t.Fatalf("Error calling Delete on %s", podName)
				}
				if err := wait.PollUntilContextTimeout(ctx, time.Millisecond*200, wait.ForeverTestTimeout, false, testutils.PodDeleted(ctx, cs, ns, podName)); err != nil {
					t.Fatalf("Failed to delete %s", podName)
				}
			}

			// Ensure gated pods are not in ActiveQ
			if len(testCtx.Scheduler.SchedulingQueue.PodsInActiveQ()) > 0 {
				t.Fatal("Expected no schedulable pods")
			}

			// Remove scheduling gates from the pod spec.
			for _, podName := range tt.rmGates {
				patch := `{"spec": {"schedulingGates": null}}`
				if _, err := cs.CoreV1().Pods(ns).Patch(ctx, podName, types.StrategicMergePatchType, []byte(patch), metav1.PatchOptions{}); err != nil {
					t.Fatalf("Failed to patch pod %v: %v", podName, err)
				}
			}

			// Schedule pods which no longer have gates.
			for _, podName := range tt.rmGates {
				testCtx.Scheduler.ScheduleOne(testCtx.Ctx)
				if err := wait.PollUntilContextTimeout(ctx, time.Millisecond*200, wait.ForeverTestTimeout, false, testutils.PodScheduled(cs, ns, podName)); err != nil {
					t.Fatalf("Failed to schedule %s", podName)
				}
			}
		})
	}
}

// TestCoreResourceEnqueue verify Pods failed by in-tree default plugins can be
// moved properly upon their registered events.
func TestCoreResourceEnqueue(t *testing.T) {
	tests := []struct {
		name string
		// initialNode is the Node to be created at first.
		initialNode *v1.Node
		// initialPod is the Pod to be created at first if it's not empty.
		initialPod *v1.Pod
		// pods are the list of Pods to be created.
		// All of them are expected to be unschedulable at first.
		pods []*v1.Pod
		// triggerFn is the function that triggers the event to move Pods.
		triggerFn func(testCtx *testutils.TestContext) error
		// wantRequeuedPods is the map of Pods that are expected to be requeued after triggerFn.
		wantRequeuedPods sets.Set[string]
	}{
		{
			name:        "Pod without a required toleration to a node isn't requeued to activeQ",
			initialNode: st.MakeNode().Name("fake-node").Capacity(map[v1.ResourceName]string{v1.ResourceCPU: "2"}).Taints([]v1.Taint{{Key: v1.TaintNodeNotReady, Effect: v1.TaintEffectNoSchedule}}).Obj(),
			pods: []*v1.Pod{
				// - Pod1 doesn't have the required toleration and will be rejected by the TaintToleration plugin.
				//   (TaintToleration plugin is evaluated before NodeResourcesFit plugin.)
				// - Pod2 has the required toleration, but requests a large amount of CPU - will be rejected by the NodeResourcesFit plugin.
				st.MakePod().Name("pod1").Req(map[v1.ResourceName]string{v1.ResourceCPU: "4"}).Container("image").Obj(),
				st.MakePod().Name("pod2").Toleration(v1.TaintNodeNotReady).Req(map[v1.ResourceName]string{v1.ResourceCPU: "4"}).Container("image").Obj(),
			},
			triggerFn: func(testCtx *testutils.TestContext) error {
				// Trigger a NodeChange event by increasing CPU capacity.
				// It makes Pod2 schedulable.
				// Pod1 is not requeued because the Node is still unready and it doesn't have the required toleration.
				if _, err := testCtx.ClientSet.CoreV1().Nodes().UpdateStatus(testCtx.Ctx, st.MakeNode().Name("fake-node").Capacity(map[v1.ResourceName]string{v1.ResourceCPU: "4"}).Taints([]v1.Taint{{Key: v1.TaintNodeNotReady, Effect: v1.TaintEffectNoSchedule}}).Obj(), metav1.UpdateOptions{}); err != nil {
					return fmt.Errorf("failed to update the node: %w", err)
				}
				return nil
			},
			wantRequeuedPods: sets.New("pod2"),
		},
		{
			name:        "Pod rejected by the PodAffinity plugin is requeued when a new Node is created and turned to ready",
			initialNode: st.MakeNode().Name("fake-node").Label("node", "fake-node").Capacity(map[v1.ResourceName]string{v1.ResourceCPU: "2"}).Obj(),
			initialPod:  st.MakePod().Label("anti", "anti").Name("pod1").PodAntiAffinityExists("anti", "node", st.PodAntiAffinityWithRequiredReq).Container("image").Node("fake-node").Obj(),
			pods: []*v1.Pod{
				// - Pod2 will be rejected by the PodAffinity plugin.
				st.MakePod().Label("anti", "anti").Name("pod2").PodAntiAffinityExists("anti", "node", st.PodAntiAffinityWithRequiredReq).Container("image").Obj(),
			},
			triggerFn: func(testCtx *testutils.TestContext) error {
				// Trigger a NodeCreated event.
				// Note that this Node has a un-ready taint and pod2 should be requeued ideally because unschedulable plugins registered for pod2 is PodAffinity.
				// However, due to preCheck, it's not requeueing pod2 to activeQ.
				// It'll be fixed by the removal of preCheck in the future.
				// https://github.com/kubernetes/kubernetes/issues/110175
				node := st.MakeNode().Name("fake-node2").Capacity(map[v1.ResourceName]string{v1.ResourceCPU: "2"}).Taints([]v1.Taint{{Key: v1.TaintNodeNotReady, Effect: v1.TaintEffectNoSchedule}}).Obj()
				if _, err := testCtx.ClientSet.CoreV1().Nodes().Create(testCtx.Ctx, st.MakeNode().Name("fake-node2").Capacity(map[v1.ResourceName]string{v1.ResourceCPU: "2"}).Taints([]v1.Taint{{Key: "foo", Effect: v1.TaintEffectNoSchedule}}).Obj(), metav1.CreateOptions{}); err != nil {
					return fmt.Errorf("failed to create a newnode: %w", err)
				}

				// As a mitigation of an issue described above, all plugins subscribing Node/Add event register UpdateNodeTaint too.
				// So, this removal of taint moves pod2 to activeQ.
				node.Spec.Taints = nil
				if _, err := testCtx.ClientSet.CoreV1().Nodes().Update(testCtx.Ctx, node, metav1.UpdateOptions{}); err != nil {
					return fmt.Errorf("failed to remove taints off the node: %w", err)
				}
				return nil
			},
			wantRequeuedPods: sets.New("pod2"),
		},
	}

	for _, featureEnabled := range []bool{false, true} {
		for _, tt := range tests {
			t.Run(fmt.Sprintf("%s [SchedulerQueueingHints enabled: %v]", tt.name, featureEnabled), func(t *testing.T) {
				defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.SchedulerQueueingHints, featureEnabled)()

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
				testutils.SyncSchedulerInformerFactory(testCtx)

				defer testCtx.Scheduler.SchedulingQueue.Close()

				cs, ns, ctx := testCtx.ClientSet, testCtx.NS.Name, testCtx.Ctx
				// Create one Node with a taint.
				if _, err := cs.CoreV1().Nodes().Create(ctx, tt.initialNode, metav1.CreateOptions{}); err != nil {
					t.Fatalf("Failed to create an initial Node %q: %v", tt.initialNode.Name, err)
				}

				if tt.initialPod != nil {
					if _, err := cs.CoreV1().Pods(ns).Create(ctx, tt.initialPod, metav1.CreateOptions{}); err != nil {
						t.Fatalf("Failed to create an initial Pod %q: %v", tt.initialPod.Name, err)
					}
				}

				for _, pod := range tt.pods {
					if _, err := cs.CoreV1().Pods(ns).Create(ctx, pod, metav1.CreateOptions{}); err != nil {
						t.Fatalf("Failed to create Pod %q: %v", pod.Name, err)
					}
				}

				// Wait for the tt.pods to be present in the scheduling queue.
				if err := wait.PollUntilContextTimeout(ctx, time.Millisecond*200, wait.ForeverTestTimeout, false, func(ctx context.Context) (bool, error) {
					pendingPods, _ := testCtx.Scheduler.SchedulingQueue.PendingPods()
					return len(pendingPods) == len(tt.pods), nil
				}); err != nil {
					t.Fatal(err)
				}

				t.Log("Confirmed Pods in the scheduling queue, starting to schedule them")

				// Pop all pods out. They should be unschedulable.
				for i := 0; i < len(tt.pods); i++ {
					testCtx.Scheduler.ScheduleOne(testCtx.Ctx)
				}
				// Wait for the tt.pods to be still present in the scheduling queue.
				if err := wait.PollUntilContextTimeout(ctx, time.Millisecond*200, wait.ForeverTestTimeout, false, func(ctx context.Context) (bool, error) {
					pendingPods, _ := testCtx.Scheduler.SchedulingQueue.PendingPods()
					return len(pendingPods) == len(tt.pods), nil
				}); err != nil {
					t.Fatal(err)
				}

				t.Log("finished initial schedulings for all Pods, will trigger triggerFn")

				err := tt.triggerFn(testCtx)
				if err != nil {
					t.Fatalf("Failed to trigger the event: %v", err)
				}

				t.Log("triggered tt.triggerFn, will check if tt.requeuedPods are requeued")

				// Wait for the tt.pods to be still present in the scheduling queue.
				var requeuedPods sets.Set[string]
				if err := wait.PollUntilContextTimeout(ctx, time.Millisecond*200, wait.ForeverTestTimeout, false, func(ctx context.Context) (bool, error) {
					requeuedPods = sets.Set[string]{} // reset
					for _, requeuedPod := range testCtx.Scheduler.SchedulingQueue.PodsInActiveQ() {
						requeuedPods.Insert(requeuedPod.Name)
					}

					return requeuedPods.Equal(tt.wantRequeuedPods), nil
				}); err != nil {
					t.Fatalf("Expect Pods %v to be requeued, but %v are requeued actually", tt.wantRequeuedPods, requeuedPods)
				}
			})
		}
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
func (f *fakeCRPlugin) EventsToRegister() []framework.ClusterEventWithHint {
	return []framework.ClusterEventWithHint{
		{Event: framework.ClusterEvent{Resource: "foos.v1.example.com", ActionType: framework.All}},
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
	ctx, cancel := context.WithCancel(context.Background())
	testCtx.Ctx = ctx
	testCtx.CloseFn = func() {
		cancel()
		server.TearDownFn()
	}

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
		"fakeCRPlugin": func(_ context.Context, _ runtime.Object, fh framework.Handle) (framework.Plugin, error) {
			return &fakeCRPlugin{}, nil
		},
	}
	cfg := configtesting.V1ToInternalWithDefaults(t, configv1.KubeSchedulerConfiguration{
		Profiles: []configv1.KubeSchedulerProfile{{
			SchedulerName: pointer.String(v1.DefaultSchedulerName),
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
	testutils.SyncSchedulerInformerFactory(testCtx)

	defer testutils.CleanupTest(t, testCtx)

	cs, ns, ctx := testCtx.ClientSet, testCtx.NS.Name, testCtx.Ctx
	logger := klog.FromContext(ctx)
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
	if err := wait.PollUntilContextTimeout(ctx, time.Millisecond*200, wait.ForeverTestTimeout, false, func(ctx context.Context) (bool, error) {
		pendingPods, _ := testCtx.Scheduler.SchedulingQueue.PendingPods()
		return len(pendingPods) == 1, nil
	}); err != nil {
		t.Fatal(err)
	}

	// Pop fake-pod out. It should be unschedulable.
	podInfo := testutils.NextPodOrDie(t, testCtx)
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
	testCtx.Scheduler.FailureHandler(ctx, fwk, podInfo, framework.NewStatus(framework.Unschedulable).WithError(fitError), nil, time.Now())

	// Scheduling cycle is incremented from 0 to 1 after NextPod() is called, so
	// pass a number larger than 1 to move Pod to unschedulablePods.
	testCtx.Scheduler.SchedulingQueue.AddUnschedulableIfNotPresent(logger, podInfo, 10)

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
	podInfo = testutils.NextPodOrDie(t, testCtx)
	if podInfo.Attempts != 2 {
		t.Errorf("Expected the Pod to be attempted 2 times, but got %v", podInfo.Attempts)
	}
}

// TestRequeueByBindFailure verify Pods failed by bind plugin are
// put back to the queue regardless of whether event happens or not.
func TestRequeueByBindFailure(t *testing.T) {
	fakeBind := &firstFailBindPlugin{}
	registry := frameworkruntime.Registry{
		"firstFailBindPlugin": func(ctx context.Context, o runtime.Object, fh framework.Handle) (framework.Plugin, error) {
			binder, err := defaultbinder.New(ctx, nil, fh)
			if err != nil {
				return nil, err
			}

			fakeBind.defaultBinderPlugin = binder.(framework.BindPlugin)
			return fakeBind, nil
		},
	}

	cfg := configtesting.V1ToInternalWithDefaults(t, configv1.KubeSchedulerConfiguration{
		Profiles: []configv1.KubeSchedulerProfile{{
			SchedulerName: pointer.String(v1.DefaultSchedulerName),
			Plugins: &configv1.Plugins{
				MultiPoint: configv1.PluginSet{
					Enabled: []configv1.Plugin{
						{Name: "firstFailBindPlugin"},
					},
					Disabled: []configv1.Plugin{
						{Name: names.DefaultBinder},
					},
				},
			},
		}}})

	// Use zero backoff seconds to bypass backoffQ.
	testCtx := testutils.InitTestSchedulerWithOptions(
		t,
		testutils.InitTestAPIServer(t, "core-res-enqueue", nil),
		0,
		scheduler.WithPodInitialBackoffSeconds(0),
		scheduler.WithPodMaxBackoffSeconds(0),
		scheduler.WithProfiles(cfg.Profiles...),
		scheduler.WithFrameworkOutOfTreeRegistry(registry),
	)
	testutils.SyncSchedulerInformerFactory(testCtx)

	go testCtx.Scheduler.Run(testCtx.Ctx)

	cs, ns, ctx := testCtx.ClientSet, testCtx.NS.Name, testCtx.Ctx
	node := st.MakeNode().Name("fake-node").Obj()
	if _, err := cs.CoreV1().Nodes().Create(ctx, node, metav1.CreateOptions{}); err != nil {
		t.Fatalf("Failed to create Node %q: %v", node.Name, err)
	}
	// create a pod.
	pod := st.MakePod().Namespace(ns).Name("pod-1").Container(imageutils.GetPauseImageName()).Obj()
	if _, err := cs.CoreV1().Pods(ns).Create(ctx, pod, metav1.CreateOptions{}); err != nil {
		t.Fatalf("Failed to create Pod %q: %v", pod.Name, err)
	}

	// 1. first binding try should fail.
	// 2. The pod should be enqueued to activeQ/backoffQ without any event.
	// 3. The pod should be scheduled in the second binding try.
	// Here, waiting until (3).
	err := wait.PollUntilContextTimeout(ctx, 200*time.Millisecond, wait.ForeverTestTimeout, false, testutils.PodScheduled(cs, ns, pod.Name))
	if err != nil {
		t.Fatalf("Expect pod-1 to be scheduled by the bind plugin: %v", err)
	}

	// Make sure the first binding trial was failed, and this pod is scheduled at the second trial.
	if fakeBind.counter != 1 {
		t.Fatalf("Expect pod-1 to be scheduled by the bind plugin in the second binding try: %v", err)
	}
}

// firstFailBindPlugin rejects the Pod in the first Bind call.
type firstFailBindPlugin struct {
	counter             int
	defaultBinderPlugin framework.BindPlugin
}

func (*firstFailBindPlugin) Name() string {
	return "firstFailBindPlugin"
}

func (p *firstFailBindPlugin) Bind(ctx context.Context, state *framework.CycleState, pod *v1.Pod, nodename string) *framework.Status {
	if p.counter == 0 {
		// fail in the first Bind call.
		p.counter++
		return framework.NewStatus(framework.Error, "firstFailBindPlugin rejects the Pod")
	}

	return p.defaultBinderPlugin.Bind(ctx, state, pod, nodename)
}

// TestRequeueByPermitRejection verify Pods failed by permit plugins in the binding cycle are
// put back to the queue, according to the correct scheduling cycle number.
func TestRequeueByPermitRejection(t *testing.T) {
	defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.SchedulerQueueingHints, true)()
	queueingHintCalledCounter := 0
	fakePermit := &fakePermitPlugin{}
	registry := frameworkruntime.Registry{
		fakePermitPluginName: func(ctx context.Context, o runtime.Object, fh framework.Handle) (framework.Plugin, error) {
			fakePermit = &fakePermitPlugin{
				frameworkHandler: fh,
				schedulingHint: func(logger klog.Logger, pod *v1.Pod, oldObj, newObj interface{}) (framework.QueueingHint, error) {
					queueingHintCalledCounter++
					return framework.Queue, nil
				},
			}
			return fakePermit, nil
		},
	}
	cfg := configtesting.V1ToInternalWithDefaults(t, configv1.KubeSchedulerConfiguration{
		Profiles: []configv1.KubeSchedulerProfile{{
			SchedulerName: pointer.String(v1.DefaultSchedulerName),
			Plugins: &configv1.Plugins{
				MultiPoint: configv1.PluginSet{
					Enabled: []configv1.Plugin{
						{Name: fakePermitPluginName},
					},
				},
			},
		}}})

	// Use zero backoff seconds to bypass backoffQ.
	testCtx := testutils.InitTestSchedulerWithOptions(
		t,
		testutils.InitTestAPIServer(t, "core-res-enqueue", nil),
		0,
		scheduler.WithPodInitialBackoffSeconds(0),
		scheduler.WithPodMaxBackoffSeconds(0),
		scheduler.WithProfiles(cfg.Profiles...),
		scheduler.WithFrameworkOutOfTreeRegistry(registry),
	)
	testutils.SyncSchedulerInformerFactory(testCtx)

	go testCtx.Scheduler.Run(testCtx.Ctx)

	cs, ns, ctx := testCtx.ClientSet, testCtx.NS.Name, testCtx.Ctx
	node := st.MakeNode().Name("fake-node").Obj()
	if _, err := cs.CoreV1().Nodes().Create(ctx, node, metav1.CreateOptions{}); err != nil {
		t.Fatalf("Failed to create Node %q: %v", node.Name, err)
	}
	// create a pod.
	pod := st.MakePod().Namespace(ns).Name("pod-1").Container(imageutils.GetPauseImageName()).Obj()
	if _, err := cs.CoreV1().Pods(ns).Create(ctx, pod, metav1.CreateOptions{}); err != nil {
		t.Fatalf("Failed to create Pod %q: %v", pod.Name, err)
	}

	// update node label. (causes the NodeUpdate event)
	node.Labels = map[string]string{"updated": ""}
	if _, err := cs.CoreV1().Nodes().Update(ctx, node, metav1.UpdateOptions{}); err != nil {
		t.Fatalf("Failed to add labels to the node: %v", err)
	}

	// create a pod to increment the scheduling cycle number in the scheduling queue.
	// We can make sure NodeUpdate event, that has happened in the previous scheduling cycle, makes Pod to be enqueued to activeQ via the scheduling queue.
	pod = st.MakePod().Namespace(ns).Name("pod-2").Container(imageutils.GetPauseImageName()).Obj()
	if _, err := cs.CoreV1().Pods(ns).Create(ctx, pod, metav1.CreateOptions{}); err != nil {
		t.Fatalf("Failed to create Pod %q: %v", pod.Name, err)
	}

	// reject pod-1 to simulate the failure in Permit plugins.
	// This pod-1 should be enqueued to activeQ because the NodeUpdate event has happened.
	fakePermit.frameworkHandler.IterateOverWaitingPods(func(wp framework.WaitingPod) {
		if wp.GetPod().Name == "pod-1" {
			wp.Reject(fakePermitPluginName, "fakePermitPlugin rejects the Pod")
			return
		}
	})

	// Wait for pod-2 to be scheduled.
	err := wait.PollUntilContextTimeout(ctx, 200*time.Millisecond, wait.ForeverTestTimeout, false, func(ctx context.Context) (done bool, err error) {
		fakePermit.frameworkHandler.IterateOverWaitingPods(func(wp framework.WaitingPod) {
			if wp.GetPod().Name == "pod-2" {
				wp.Allow(fakePermitPluginName)
			}
		})

		return testutils.PodScheduled(cs, ns, "pod-2")(ctx)
	})
	if err != nil {
		t.Fatalf("Expect pod-2 to be scheduled")
	}

	err = wait.PollUntilContextTimeout(ctx, 200*time.Millisecond, wait.ForeverTestTimeout, false, func(ctx context.Context) (done bool, err error) {
		pod1Found := false
		fakePermit.frameworkHandler.IterateOverWaitingPods(func(wp framework.WaitingPod) {
			if wp.GetPod().Name == "pod-1" {
				pod1Found = true
				wp.Allow(fakePermitPluginName)
			}
		})
		return pod1Found, nil
	})
	if err != nil {
		t.Fatal("Expect pod-1 to be scheduled again")
	}

	if queueingHintCalledCounter != 1 {
		t.Fatalf("Expected the scheduling hint to be called 1 time, but %v", queueingHintCalledCounter)
	}
}

type fakePermitPlugin struct {
	frameworkHandler framework.Handle
	schedulingHint   framework.QueueingHintFn
}

const fakePermitPluginName = "fakePermitPlugin"

func (p *fakePermitPlugin) Name() string {
	return fakePermitPluginName
}

func (p *fakePermitPlugin) Permit(ctx context.Context, state *framework.CycleState, _ *v1.Pod, _ string) (*framework.Status, time.Duration) {
	return framework.NewStatus(framework.Wait), wait.ForeverTestTimeout
}

func (p *fakePermitPlugin) EventsToRegister() []framework.ClusterEventWithHint {
	return []framework.ClusterEventWithHint{
		{Event: framework.ClusterEvent{Resource: framework.Node, ActionType: framework.UpdateNodeLabel}, QueueingHintFn: p.schedulingHint},
	}
}
