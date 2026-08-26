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

package asyncframework

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"sync"
	"testing"
	"time"

	v1 "k8s.io/api/core/v1"
	schedulingapi "k8s.io/api/scheduling/v1beta1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/wait"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/client-go/kubernetes"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/klog/v2"
	configv1 "k8s.io/kube-scheduler/config/v1"
	fwk "k8s.io/kube-scheduler/framework"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/scheduler"
	"k8s.io/kubernetes/pkg/scheduler/apis/config"
	configtesting "k8s.io/kubernetes/pkg/scheduler/apis/config/testing"
	"k8s.io/kubernetes/pkg/scheduler/backend/queue"
	"k8s.io/kubernetes/pkg/scheduler/framework"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/defaultbinder"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/defaultpreemption"
	plfeature "k8s.io/kubernetes/pkg/scheduler/framework/plugins/feature"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/names"
	"k8s.io/kubernetes/pkg/scheduler/framework/preemption"
	frameworkruntime "k8s.io/kubernetes/pkg/scheduler/framework/runtime"
	st "k8s.io/kubernetes/pkg/scheduler/testing"
	testutils "k8s.io/kubernetes/test/integration/util"
)

const (
	// PodBlockedInBindingName is the name of the pod that is blocked in binding.
	PodBlockedInBindingName = "pod-blocked-in-binding"

	// ReservingPodName is the name of the pod that is reserving resources.
	ReservingPodName = "reserving-pod"

	// BlockingPodName is the name of the pod that is blocking the preemption.
	BlockingPodName = "blocking-pod"
)

var (
	// LowPriority is a priority used in the tests for low priority pods.
	LowPriority = int32(100)

	// MediumPriority is a priority used in the tests for medium priority pods.
	MediumPriority = int32(200)

	// HighPriority is a priority used in the tests for high priority pods.
	HighPriority = int32(300)
)

// CreatePodGroup is a struct used to create a PodGroup.
type CreatePodGroup struct {
	PodGroup *schedulingapi.PodGroup
}

// CreatePod is a struct used by CreatePod step action.
// It contains information about a pod and number of times it should be created.
type CreatePod struct {
	Pod *v1.Pod
	// Count is the number of times the pod should be created by this action.
	// i.e., if you use it, you have to use GenerateName.
	// By default, it's 1.
	Count *int
}

// SchedulePod is a struct used to for step scheduling a pod.
// It contains information of expected status after scheduling.
type SchedulePod struct {
	PodName             string
	ExpectSuccess       bool
	ExpectUnschedulable bool
	ExpectInQueue       bool
}

// SchedulePodGroup is a struct used for step scheduling a pod group.
// It contains information of expected status after scheduling.
type SchedulePodGroup struct {
	PodGroupName        string
	ExpectSuccess       bool
	ExpectUnschedulable bool
	ExpectInQueue       bool
}

// Step represents an action in the async preemption test.
// Test is containing list of steps and is doing them in order.
type Step struct {
	// Name is this step's Name, just for the debugging purpose.
	Name string

	// Only one of the following actions should be set.

	// CreatePodGroup creates a PodGroup.
	CreatePodGroup *CreatePodGroup
	// CreatePod creates a Pod.
	CreatePod *CreatePod
	// CreateNode creates an additional Node.
	CreateNode string
	// SchedulePod schedules one Pod by a given Pod name.
	SchedulePod *SchedulePod
	// SchedulePodGroup schedules one PodGroup by a given PodGroup name.
	SchedulePodGroup *SchedulePodGroup
	// CompletePreemption completes the preemption that is currently on-going.
	// You should give a Pod/PodGroup name.
	CompletePreemption string
	// PodGatedInQueue checks if the given Pod is in the scheduling queue and gated by the preemption.
	// You should give a Pod name.
	PodGatedInQueue string
	// PodRunningPreemption checks if the given Pod is running preemption.
	// You should give a Pod index representing the order of Pod creation.
	// e.g., if you want to check the Pod created first in the test case, you should give 0.
	PodRunningPreemption *int
	// ActivatePod moves the pod from unschedulable to active or backoff.
	// The value is the name of the pod to activate.
	ActivatePod string
	// ResumeBind resumes the binding operation that keeps the pod blocked.
	// Note: The pod will only become blocked in the first place, if pod name matches string defined in podBlockedInBinding.
	ResumeBind bool
	// VerifyPodInUnschedulable waits for some time and confirms that the given pod is in the unschedulable pool.
	// The value is the name of the checked pod.
	VerifyPodInUnschedulable string
	// FlushUnschedulable flushes the unschedulable queue.
	FlushUnschedulable bool
	// WaitForPodsDeleted waits for the specified pods to be deleted from the cluster.
	// The value is the array of Pod indexes representing the order of Pod creation.
	WaitForPodsDeleted []int
}

// AsyncPreemptionStepRunnerConfig is a configuration for running async preemption test steps.
type AsyncPreemptionStepRunnerConfig struct {
	CreatedPods            []*v1.Pod
	ClientSet              kubernetes.Interface
	PreemptionDoneChannels *sync.Map
	Logger                 klog.Logger
	PreemptionPlugin       *defaultpreemption.DefaultPreemption
	BlockBindingChannel    chan struct{}
}

// RunAsyncPreemptionSteps runs the async preemption test steps in order.
func RunAsyncPreemptionSteps(testCtx *testutils.TestContext, t *testing.T, steps []Step, config AsyncPreemptionStepRunnerConfig) {
	for _, step := range steps {
		t.Logf("Running scenario: %s", step.Name)
		switch {
		case step.CreateNode != "":
			nodeCreation(testCtx, t, step.CreateNode, config.ClientSet)
		case step.CreatePodGroup != nil:
			createPodGroup(testCtx, t, config.ClientSet, step.CreatePodGroup)
		case step.CreatePod != nil:
			createPod(testCtx, t, step.CreatePod, config.ClientSet, &config.CreatedPods)
		case step.SchedulePod != nil:
			schedulePod(testCtx, t, step.SchedulePod, config.ClientSet, config.PreemptionDoneChannels)
		case step.SchedulePodGroup != nil:
			schedulePodGroup(testCtx, t, step.SchedulePodGroup, config.ClientSet, config.PreemptionDoneChannels)
		case step.ActivatePod != "":
			activatePod(testCtx, t, step.ActivatePod, config.Logger)
		case step.CompletePreemption != "":
			completePreemption(t, step.CompletePreemption, config.PreemptionDoneChannels)
		case step.PodGatedInQueue != "":
			podGatedInQueue(testCtx, t, step.PodGatedInQueue, config.Logger)
		case step.PodRunningPreemption != nil:
			podRunningPreemption(testCtx, t, config.CreatedPods, step.PodRunningPreemption, config.PreemptionPlugin)
		case step.ResumeBind:
			config.BlockBindingChannel <- struct{}{}
		case step.VerifyPodInUnschedulable != "":
			verifyPodInUnschedulable(testCtx, t, step.VerifyPodInUnschedulable)
		case step.FlushUnschedulable:
			testCtx.Scheduler.SchedulingQueue.MoveAllToActiveOrBackoffQueue(config.Logger, framework.EventUnschedulableTimeout, nil, nil, nil)
		case len(step.WaitForPodsDeleted) != 0:
			waitForPodsDeleted(testCtx, t, step.WaitForPodsDeleted, config.CreatedPods, config.ClientSet)
		}
	}
}

// AsyncPreemptionTestConfig is a config for initialising the environment for async preemption tests.
type AsyncPreemptionTestConfig struct {
	EnableGenericWorkload  bool
	PreemptionDoneChannels *sync.Map
	BlockBindingChannel    chan struct{}
	InitialBackoffSeconds  int64
	MaxBackoffSeconds      int64
}

// InitTestForAsyncPreemption initializes the test environment for async preemption tests.
// It enables required feature gates, creates required plugins and returns test context, preemption plugin and client set.
func InitTestForAsyncPreemption(t *testing.T, config AsyncPreemptionTestConfig) (*testutils.TestContext, *defaultpreemption.DefaultPreemption, kubernetes.Interface) {
	featuresOverrides := featuregatetesting.FeatureOverrides{
		features.SchedulerAsyncAPICalls:   true,
		features.SchedulerAsyncPreemption: true,
		features.GenericWorkload:          config.EnableGenericWorkload,
	}
	featuregatetesting.SetFeatureGatesDuringTest(t, utilfeature.DefaultFeatureGate, featuresOverrides)

	registry := make(frameworkruntime.Registry)
	// We need to use a custom preemption plugin to test async preemption behavior
	delayedPreemptionPluginName, getPreemptionPlugin, err := registerDelayedPreemptionPlugin(&registry, config.PreemptionDoneChannels, config.EnableGenericWorkload)
	if err != nil {
		t.Fatalf("Error registering a preemption plugin: %v", err)
	}

	blockingBindPluginName, err := registerBlockBindingPlugin(registry, config.BlockBindingChannel)
	if err != nil {
		t.Fatalf("Error registering a bind plugin: %v", err)
	}

	reservingPluginName, err := registerReservingPlugin(registry)
	if err != nil {
		t.Fatalf("Error registering a reserving plugin: %v", err)
	}

	queueSkipFilterPluginName, err := registerSkipFilterPlugin(registry)
	if err != nil {
		t.Fatalf("Error registering a queueSkipFilterPlugin plugin: %v", err)
	}

	cfg := configtesting.V1ToInternalWithDefaults(t, configv1.KubeSchedulerConfiguration{
		Profiles: []configv1.KubeSchedulerProfile{{
			SchedulerName: new(v1.DefaultSchedulerName),
			Plugins: &configv1.Plugins{
				MultiPoint: configv1.PluginSet{
					Enabled: []configv1.Plugin{
						{Name: blockingBindPluginName},
						{Name: delayedPreemptionPluginName},
						{Name: reservingPluginName},
						{Name: queueSkipFilterPluginName},
					},
					Disabled: []configv1.Plugin{
						{Name: names.DefaultPreemption},
						{Name: names.DefaultBinder},
					},
				},
			},
		}},
	})

	// It initializes the scheduler, but doesn't start.
	// We manually trigger the scheduling cycle.
	testCtx := testutils.InitTestSchedulerWithOptions(t,
		testutils.InitTestAPIServer(t, "preemption", nil),
		0,
		scheduler.WithProfiles(cfg.Profiles...),
		scheduler.WithFrameworkOutOfTreeRegistry(registry),
		scheduler.WithPodMaxBackoffSeconds(config.MaxBackoffSeconds),
		scheduler.WithPodInitialBackoffSeconds(config.InitialBackoffSeconds),
	)
	testutils.SyncSchedulerInformerFactory(testCtx)
	cs := testCtx.ClientSet
	return testCtx, getPreemptionPlugin(), cs
}

// registerSkipFilterPlugin register fake plugin that will filter nodes with a specific name.
// Importantly, this plugin always returns QueueSkip as the queue hint, simulating faulty queue hint implementation.
func registerSkipFilterPlugin(registry frameworkruntime.Registry) (string, error) {
	queueSkipFilterPluginName := "queueSkipFilterPlugin"
	err := registry.Register(queueSkipFilterPluginName, func(ctx context.Context, o runtime.Object, fh fwk.Handle) (fwk.Plugin, error) {
		return &queueSkipFilterPlugin{
			name:              queueSkipFilterPluginName,
			nameOfBlockingPod: BlockingPodName,
		}, nil
	})
	return queueSkipFilterPluginName, err
}

// registerReservingPlugin register fake plugin that will reserve some fake resources for one pod.
// This could be used to check scheduler's behavior when the victim has to unreserve these resources to let the preemptor schedule.
func registerReservingPlugin(registry frameworkruntime.Registry) (string, error) {
	reservingPluginName := "reservingPlugin"
	err := registry.Register(reservingPluginName, func(ctx context.Context, o runtime.Object, fh fwk.Handle) (fwk.Plugin, error) {
		return &reservingPlugin{
			name:               reservingPluginName,
			nameOfPodToReserve: ReservingPodName,
			fh:                 fh,
		}, nil
	})
	return reservingPluginName, err
}

// registerBlockBindingPlugin register fake bind plugin that will block on binding for the specified pod name, until it receives a resume signal via the blockBindingChannel.
func registerBlockBindingPlugin(registry frameworkruntime.Registry, blockBindingChannel chan struct{}) (string, error) {
	blockingBindPluginName := "blockingBindPlugin"
	err := registry.Register(blockingBindPluginName, func(ctx context.Context, o runtime.Object, fh fwk.Handle) (fwk.Plugin, error) {
		db, err := defaultbinder.New(ctx, o, fh)
		if err != nil {
			return nil, err
		}
		var bindPlugin = blockingBindPlugin{
			name:                blockingBindPluginName,
			nameOfPodToBlock:    PodBlockedInBindingName,
			realPlugin:          db.(fwk.BindPlugin),
			blockBindingChannel: blockBindingChannel,
		}
		return &bindPlugin, nil
	})
	return blockingBindPluginName, err
}

// registerDelayedPreemptionPlugin register a custom preemption plugin to test async preemption behavior.
func registerDelayedPreemptionPlugin(registry *frameworkruntime.Registry, preemptionDoneChannels *sync.Map, enableGenericWorkload bool) (string, func() *defaultpreemption.DefaultPreemption, error) {
	delayedPreemptionPluginName := "delay-preemption"
	var preemptionPlugin *defaultpreemption.DefaultPreemption
	err := registry.Register(delayedPreemptionPluginName, func(c context.Context, r runtime.Object, fh fwk.Handle) (fwk.Plugin, error) {
		p, err := frameworkruntime.FactoryAdapter(plfeature.Features{EnableAsyncPreemption: true, EnableGenericWorkload: enableGenericWorkload}, defaultpreemption.New)(c, &config.DefaultPreemptionArgs{
			// Set default values to pass the validation at the initialization, not related to the test.
			MinCandidateNodesPercentage: 10,
			MinCandidateNodesAbsolute:   100,
		}, fh)
		if err != nil {
			return nil, fmt.Errorf("error creating default preemption plugin: %w", err)
		}

		var ok bool
		preemptionPlugin, ok = p.(*defaultpreemption.DefaultPreemption)
		if !ok {
			return nil, fmt.Errorf("unexpected plugin type %T", p)
		}

		preemptPodFn := preemptionPlugin.Executor.PreemptPod
		preemptionPlugin.Executor.PreemptPod = func(ctx context.Context, c preemption.Candidate, preemptor preemption.ExecutorPreemptor, victim *v1.Pod, pluginName string) (bool, error) {
			// block the preemption goroutine to complete until the test case allows it to proceed.
			ch, ok := preemptionDoneChannels.Load(preemptor.GetName())
			if ok {
				<-ch.(chan struct{})
			}
			return preemptPodFn(ctx, c, preemptor, victim, pluginName)
		}

		return preemptionPlugin, nil
	})
	return delayedPreemptionPluginName, func() *defaultpreemption.DefaultPreemption { return preemptionPlugin }, err
}

func waitForPodsDeleted(testCtx *testutils.TestContext, t *testing.T, podIndexes []int, createdPods []*v1.Pod, cs kubernetes.Interface) {
	for _, podIndex := range podIndexes {
		podName := createdPods[podIndex].Name
		if err := wait.PollUntilContextTimeout(testCtx.Ctx, 50*time.Millisecond, wait.ForeverTestTimeout, false, testutils.PodDeleted(testCtx.Ctx, cs, testCtx.NS.Name, podName)); err != nil {
			t.Fatalf("Failed to wait for pod %s to be deleted: %v", podName, err)
		}
	}
}

func verifyPodInUnschedulable(testCtx *testutils.TestContext, t *testing.T, podName string) {
	if err := wait.PollUntilContextTimeout(testCtx.Ctx, 50*time.Millisecond, 200*time.Millisecond, false, func(ctx context.Context) (bool, error) {
		if !PodInUnschedulablePodPool(t, testCtx.Scheduler.SchedulingQueue, podName) {
			return false, fmt.Errorf("expected the pod %s to remain in the unschedulable queue after the scheduling attempt", podName)
		}
		// Continue polling to confirm that pod remains in unschedulable queue and does not get activated.
		return false, nil
	}); err != nil && !errors.Is(err, context.DeadlineExceeded) {
		// If timeout was reached or context was cancelled without finding that vanished from unschedulable, it means the state is as expected.
		// If a different error occurred, it means that the pod got unexpectedly activated, or something else went wrong.
		t.Fatalf("Error in scenario verifyPodInUnschedulable: %v", err)
	}
}

func podRunningPreemption(testCtx *testutils.TestContext, t *testing.T, createdPods []*v1.Pod, podIndex *int, preemptionPlugin *defaultpreemption.DefaultPreemption) {
	if err := wait.PollUntilContextTimeout(testCtx.Ctx, time.Millisecond*200, wait.ForeverTestTimeout, false, func(ctx context.Context) (bool, error) {
		pod := createdPods[*podIndex]
		if pod.Spec.SchedulingGroup != nil && pod.Spec.SchedulingGroup.PodGroupName != nil {
			pg, err := testCtx.InformerFactory.Scheduling().V1beta1().PodGroups().Lister().PodGroups(pod.Namespace).Get(*pod.Spec.SchedulingGroup.PodGroupName)
			if err == nil {
				return preemptionPlugin.Executor.IsPodGroupRunningPreemption(pg.UID), nil
			}
		}
		return preemptionPlugin.Executor.IsPodRunningPreemption(pod.GetUID()), nil
	}); err != nil {
		t.Fatalf("Expected the pod %s to be running preemption", createdPods[*podIndex].Name)
	}
}

func podGatedInQueue(testCtx *testutils.TestContext, t *testing.T, podName string, logger klog.Logger) {
	pod := unschedulablePod(t, testCtx.Scheduler.SchedulingQueue, podName)
	if pod == nil {
		t.Fatalf("Expected the pod %s to be in the queue", podName)
	}

	// Make sure this Pod is gated by the preemption at PreEnqueue extension point
	// by activating the Pod and see if it's still in the unsched pod pool.
	testCtx.Scheduler.SchedulingQueue.Activate(logger, map[string]*v1.Pod{podName: pod})
	if !PodInUnschedulablePodPool(t, testCtx.Scheduler.SchedulingQueue, podName) {
		t.Fatalf("Expected the pod %s to be in the queue even after the activation", podName)
	}
	if pInfo, _ := testCtx.Scheduler.SchedulingQueue.GetPod(podName, testCtx.NS.Name, pod.Spec.SchedulingGroup); pInfo == nil || !pInfo.Gated() {
		t.Fatalf("Expected the pod %s to be gated", podName)
	}
}

func completePreemption(t *testing.T, preemptorName string, preemptionDoneChannels *sync.Map) {
	ch, ok := preemptionDoneChannels.Load(preemptorName)
	if !ok {
		t.Fatalf("The preemptor Pod %q is not running preemption", preemptorName)
	}
	close(ch.(chan struct{}))
	preemptionDoneChannels.Delete(preemptorName)
}

func activatePod(testCtx *testutils.TestContext, t *testing.T, podName string, logger klog.Logger) {
	pod := unschedulablePod(t, testCtx.Scheduler.SchedulingQueue, podName)
	if pod == nil {
		t.Fatalf("Expected the pod %s to be in unschedulable queue before activation phase", podName)
	}
	m := map[string]*v1.Pod{podName: pod}
	testCtx.Scheduler.SchedulingQueue.Activate(logger, m)
}

func schedulePod(testCtx *testutils.TestContext, t *testing.T, schedulePodStep *SchedulePod, cs kubernetes.Interface, preemptionDoneChannels *sync.Map) {
	lastFailure := ""
	if err := wait.PollUntilContextTimeout(testCtx.Ctx, time.Millisecond*200, wait.ForeverTestTimeout, false, func(ctx context.Context) (bool, error) {
		if len(testCtx.Scheduler.SchedulingQueue.PodsInActiveQ()) == 0 {
			lastFailure = fmt.Sprintf("Expected the pod %s to be scheduled, but no pod arrives at the activeQ", schedulePodStep.PodName)
			return false, nil
		}

		if testCtx.Scheduler.SchedulingQueue.PodsInActiveQ()[0].Name != schedulePodStep.PodName {
			// need to wait more because maybe the queue will get another Pod that higher priority than the current top pod.
			lastFailure = fmt.Sprintf("The pod %s is expected to be scheduled, but the top Pod is %s", schedulePodStep.PodName, testCtx.Scheduler.SchedulingQueue.PodsInActiveQ()[0].Name)
			return false, nil
		}

		return true, nil
	}); err != nil {
		t.Fatal(lastFailure)
	}

	ch := make(chan struct{})
	preemptionDoneChannels.Store(schedulePodStep.PodName, ch)

	testCtx.Scheduler.ScheduleOne(testCtx.Ctx)

	if schedulePodStep.ExpectSuccess {
		if err := wait.PollUntilContextTimeout(testCtx.Ctx, 200*time.Millisecond, wait.ForeverTestTimeout, false, testutils.PodScheduled(cs, testCtx.NS.Name, schedulePodStep.PodName)); err != nil {
			t.Fatalf("Expected the pod %s to be scheduled", schedulePodStep.PodName)
		}
	} else if schedulePodStep.ExpectUnschedulable {
		if err := wait.PollUntilContextTimeout(testCtx.Ctx, 200*time.Millisecond, wait.ForeverTestTimeout, true, func(ctx context.Context) (bool, error) {
			return PodInUnschedulablePodPool(t, testCtx.Scheduler.SchedulingQueue, schedulePodStep.PodName), nil
		}); err != nil {
			t.Fatalf("Expected the pod %s to be in the unschedulable queue after the scheduling attempt", schedulePodStep.PodName)
		}
	} else if schedulePodStep.ExpectInQueue {
		if err := wait.PollUntilContextTimeout(testCtx.Ctx, 200*time.Millisecond, wait.ForeverTestTimeout, true, func(ctx context.Context) (bool, error) {
			return podInQueue(t, testCtx.Scheduler.SchedulingQueue, schedulePodStep.PodName), nil
		}); err != nil {
			t.Fatalf("Expected the pod %s to be in the queue after the scheduling attempt", schedulePodStep.PodName)
		}
	}
}

func schedulePodGroup(testCtx *testutils.TestContext, t *testing.T, schedulePodGroupStep *SchedulePodGroup, cs kubernetes.Interface, preemptionDoneChannels *sync.Map) {
	lastFailure := ""
	if err := wait.PollUntilContextTimeout(testCtx.Ctx, time.Millisecond*200, wait.ForeverTestTimeout, false, func(ctx context.Context) (bool, error) {
		if len(testCtx.Scheduler.SchedulingQueue.PodsInActiveQ()) == 0 {
			lastFailure = fmt.Sprintf("Expected the pod group %s to be scheduled, but no pod arrives at the activeQ", schedulePodGroupStep.PodGroupName)
			return false, nil
		}

		topPod := testCtx.Scheduler.SchedulingQueue.PodsInActiveQ()[0]
		topPodGroupName := ""
		if topPod.Spec.SchedulingGroup != nil && topPod.Spec.SchedulingGroup.PodGroupName != nil {
			topPodGroupName = *topPod.Spec.SchedulingGroup.PodGroupName
		}
		if topPodGroupName != schedulePodGroupStep.PodGroupName {
			// need to wait more because maybe the queue will get another PodGroup that higher priority than the current top entity.
			if topPodGroupName != "" {
				lastFailure = fmt.Sprintf("The pod group %s is expected to be scheduled, but the top PodGroup is %s", schedulePodGroupStep.PodGroupName, topPodGroupName)
			} else {
				lastFailure = fmt.Sprintf("The pod group %s is expected to be scheduled, but the top Pod is %s", schedulePodGroupStep.PodGroupName, topPod.Name)
			}
			return false, nil
		}

		return true, nil
	}); err != nil {
		t.Fatal(lastFailure)
	}

	ch := make(chan struct{})
	if _, ok := preemptionDoneChannels.Load(schedulePodGroupStep.PodGroupName); !ok {
		preemptionDoneChannels.Store(schedulePodGroupStep.PodGroupName, ch)
	}
	testCtx.Scheduler.ScheduleOne(testCtx.Ctx)

	pods, err := cs.CoreV1().Pods(testCtx.NS.Name).List(testCtx.Ctx, metav1.ListOptions{})
	if err != nil {
		t.Fatalf("Failed to list pods in namespace %s: %v", testCtx.NS.Name, err)
	}
	var pgPods []*v1.Pod
	for i := range pods.Items {
		pod := &pods.Items[i]
		if pod.Spec.SchedulingGroup != nil && pod.Spec.SchedulingGroup.PodGroupName != nil && *pod.Spec.SchedulingGroup.PodGroupName == schedulePodGroupStep.PodGroupName {
			pgPods = append(pgPods, pod)
		}
	}
	if len(pgPods) == 0 {
		t.Fatalf("No pods found for pod group %s", schedulePodGroupStep.PodGroupName)
	}

	if schedulePodGroupStep.ExpectSuccess {
		for _, pod := range pgPods {
			if err := wait.PollUntilContextTimeout(testCtx.Ctx, 200*time.Millisecond, wait.ForeverTestTimeout, false, testutils.PodScheduled(cs, testCtx.NS.Name, pod.Name)); err != nil {
				t.Fatalf("Expected the pod %s to be scheduled", pod.Name)
			}
		}
	} else if schedulePodGroupStep.ExpectUnschedulable {
		for _, pod := range pgPods {
			if err := wait.PollUntilContextTimeout(testCtx.Ctx, 200*time.Millisecond, wait.ForeverTestTimeout, true, func(ctx context.Context) (bool, error) {
				return PodInUnschedulablePodPool(t, testCtx.Scheduler.SchedulingQueue, pod.Name), nil
			}); err != nil {
				t.Fatalf("Expected the pod %s to be in the unschedulable queue after the scheduling attempt", pod.Name)
			}
		}
	} else if schedulePodGroupStep.ExpectInQueue {
		for _, pod := range pgPods {
			if err := wait.PollUntilContextTimeout(testCtx.Ctx, 200*time.Millisecond, wait.ForeverTestTimeout, true, func(ctx context.Context) (bool, error) {
				return podInQueue(t, testCtx.Scheduler.SchedulingQueue, pod.Name), nil
			}); err != nil {
				t.Fatalf("Expected the pod %s to be in the queue after the scheduling attempt", pod.Name)
			}
		}
	}
}

func createPod(testCtx *testutils.TestContext, t *testing.T, createPodStep *CreatePod, cs kubernetes.Interface, createdPods *[]*v1.Pod) {
	if createPodStep.Count == nil {
		createPodStep.Count = new(1)
	}

	for i := 0; i < *createPodStep.Count; i++ {
		pod, err := cs.CoreV1().Pods(testCtx.NS.Name).Create(testCtx.Ctx, createPodStep.Pod, metav1.CreateOptions{})
		if err != nil {
			t.Fatalf("Failed to create a Pod %q: %v", pod.Name, err)
		}
		*createdPods = append(*createdPods, pod)
	}
}

func createPodGroup(testCtx *testutils.TestContext, t *testing.T, cs kubernetes.Interface, createPodGroupStep *CreatePodGroup) {
	_, err := cs.SchedulingV1beta1().PodGroups(testCtx.NS.Name).Create(testCtx.Ctx, createPodGroupStep.PodGroup, metav1.CreateOptions{})
	if err != nil && !apierrors.IsAlreadyExists(err) {
		t.Fatalf("Failed to create a PodGroup %q: %v", createPodGroupStep.PodGroup.Name, err)
	}
	if err := wait.PollUntilContextTimeout(testCtx.Ctx, 2*time.Second, wait.ForeverTestTimeout, false, func(ctx context.Context) (bool, error) {
		_, err := testCtx.InformerFactory.Scheduling().V1beta1().PodGroups().Lister().PodGroups(testCtx.NS.Name).Get(createPodGroupStep.PodGroup.Name)
		if err != nil {
			if apierrors.IsNotFound(err) {
				return false, nil
			}
			return false, err
		}
		return true, nil
	}); err != nil {
		t.Fatalf("Failed to wait for PodGroup %q to sync: %v", createPodGroupStep.PodGroup.Name, err)
	}
}

func nodeCreation(testCtx *testutils.TestContext, t *testing.T, nodeName string, cs kubernetes.Interface) {
	newNode := st.MakeNode().Name(nodeName).Capacity(map[v1.ResourceName]string{v1.ResourceCPU: "4"}).Obj()
	if _, err := cs.CoreV1().Nodes().Create(testCtx.Ctx, newNode, metav1.CreateOptions{}); err != nil {
		t.Fatalf("Failed to create an initial Node %q: %v", newNode.Name, err)
	}
	t.Cleanup(func() {
		if err := cs.CoreV1().Nodes().Delete(testCtx.Ctx, newNode.Name, metav1.DeleteOptions{}); err != nil {
			t.Fatalf("Failed to delete the Node %q: %v", newNode.Name, err)
		}
	})
}

// PodInUnschedulablePodPool checks if the given Pod is in the unschedulable pod pool.
func PodInUnschedulablePodPool(t *testing.T, queue queue.SchedulingQueue, podName string) bool {
	t.Helper()
	// First, look for the pod in the activeQ.
	for _, pod := range queue.PodsInActiveQ() {
		if pod.Name == podName {
			return false
		}
	}

	pendingPods, _ := queue.PendingPods()
	for _, pod := range pendingPods {
		if pod.Name == podName {
			return true
		}
	}

	return false
}

func podInQueue(t *testing.T, queue queue.SchedulingQueue, podName string) bool {
	t.Helper()
	pendingPods, _ := queue.PendingPods()
	for _, pod := range pendingPods {
		if pod.Name == podName {
			return true
		}
	}

	return false
}

// unschedulablePod checks if the given Pod is in the unschedulable queue and returns it.
func unschedulablePod(t *testing.T, queue queue.SchedulingQueue, podName string) *v1.Pod {
	t.Helper()
	unschedPods := queue.UnschedulablePods()
	for _, pod := range unschedPods {
		if pod.Name == podName {
			return pod
		}
	}

	return nil
}

type queueSkipFilterPlugin struct {
	name              string
	nameOfBlockingPod string
}

func (fp *queueSkipFilterPlugin) EventsToRegister(context.Context) ([]fwk.ClusterEventWithHint, error) {
	return []fwk.ClusterEventWithHint{
		{
			Event: fwk.ClusterEvent{Resource: fwk.Pod, ActionType: fwk.Delete},
			QueueingHintFn: func(_ klog.Logger, _ *v1.Pod, _, _ interface{}) (fwk.QueueingHint, error) {
				return fwk.QueueSkip, nil
			},
		},
	}, nil
}

func (fp *queueSkipFilterPlugin) Filter(ctx context.Context, state fwk.CycleState, pod *v1.Pod, nodeInfo fwk.NodeInfo) *fwk.Status {
	for _, scheduledPod := range nodeInfo.GetPods() {
		if strings.Contains(scheduledPod.GetPod().Name, fp.nameOfBlockingPod) {
			return fwk.NewStatus(fwk.Unschedulable, fmt.Sprintf("node %s has blocking pod %s", nodeInfo.Node().Name, scheduledPod.GetPod().Name))
		}
	}
	return nil
}

func (fp *queueSkipFilterPlugin) Name() string {
	return fp.name
}

var _ fwk.FilterPlugin = &queueSkipFilterPlugin{}
var _ fwk.EnqueueExtensions = &queueSkipFilterPlugin{}

// blockingBindPlugin is a fake plugin that simulates a long binding operation.
// Underneath it calls realPlugin.Bind(), after receiving a signal that binding can be unblocked.
type blockingBindPlugin struct {
	name                string
	nameOfPodToBlock    string
	realPlugin          fwk.BindPlugin
	blockBindingChannel chan struct{}
}

func (bp *blockingBindPlugin) Name() string {
	return bp.name
}

func (bp *blockingBindPlugin) Bind(ctx context.Context, state fwk.CycleState, p *v1.Pod, nodeName string) *fwk.Status {
	if strings.Contains(p.Name, bp.nameOfPodToBlock) {
		// block the bind goroutine to complete until the test case allows it to proceed.
		select {
		case <-bp.blockBindingChannel:
		case <-ctx.Done():
		}
	}
	return bp.realPlugin.Bind(ctx, state, p, nodeName)
}

var _ fwk.BindPlugin = &blockingBindPlugin{}

// reservingPlugin is a fake plugin that reserves some resource in memory for nameOfPodToReserve pod.
// Other pods won't be scheduled, unless the resources are unreserved.
type reservingPlugin struct {
	lock               sync.Mutex
	name               string
	nameOfPodToReserve string
	reserved           bool
	fh                 fwk.Handle
}

func (rp *reservingPlugin) Name() string {
	return rp.name
}

func (rp *reservingPlugin) EventsToRegister(_ context.Context) ([]fwk.ClusterEventWithHint, error) {
	return []fwk.ClusterEventWithHint{
		// Plugin will wake up the pod on any Pod/Delete event.
		{Event: fwk.ClusterEvent{Resource: fwk.Pod, ActionType: fwk.Delete}},
	}, nil
}

const reservingPluginStateKey = "PreFilterReserving"

type reservingPluginState struct {
	reserved bool
}

func (s reservingPluginState) Clone() fwk.StateData {
	return reservingPluginState{
		reserved: s.reserved,
	}
}

func (rp *reservingPlugin) PreFilter(ctx context.Context, state fwk.CycleState, pod *v1.Pod, nodes []fwk.NodeInfo) (*fwk.PreFilterResult, *fwk.Status) {
	rp.lock.Lock()
	state.Write(reservingPluginStateKey, reservingPluginState{reserved: rp.reserved})
	rp.lock.Unlock()
	return nil, nil
}

func (rp *reservingPlugin) Filter(ctx context.Context, state fwk.CycleState, pod *v1.Pod, nodeInfo fwk.NodeInfo) *fwk.Status {
	if state.IsPodGroupSchedulingCycle() {
		// check if it is a preemption simulation
		if err := rp.fh.MutableSnapshotSharedLister().StartMutations(); err != nil {
			nodes, _ := rp.fh.MutableSnapshotSharedLister().NodeInfos().List()
			for _, n := range nodes {
				for _, p := range n.GetPods() {
					// check if pod with reservation is not a victim, if it is not, return "resource are reserved"
					if strings.Contains(p.GetPod().Name, rp.nameOfPodToReserve) {
						return fwk.NewStatus(fwk.Unschedulable, "resources are reserved")
					}
				}
			}
			// this is simulation and pod with reservation is a victim, so resources are available
			return nil
		} else {
			if err := rp.fh.MutableSnapshotSharedLister().EndMutations(); err != nil {
				return fwk.AsStatus(err)
			}
		}
	}
	s, err := state.Read(reservingPluginStateKey)
	if err != nil {
		return fwk.AsStatus(err)
	}
	if s.(reservingPluginState).reserved {
		return fwk.NewStatus(fwk.Unschedulable, "resources are reserved")
	}
	return nil
}

func (rp *reservingPlugin) Reserve(ctx context.Context, state fwk.CycleState, p *v1.Pod, nodeName string) *fwk.Status {
	if strings.Contains(p.Name, rp.nameOfPodToReserve) {
		rp.lock.Lock()
		rp.reserved = true
		rp.lock.Unlock()
	}
	return nil
}

func (rp *reservingPlugin) Unreserve(ctx context.Context, state fwk.CycleState, p *v1.Pod, nodeName string) {
	if strings.Contains(p.Name, rp.nameOfPodToReserve) {
		rp.lock.Lock()
		rp.reserved = false
		rp.lock.Unlock()
	}
}

func (rp *reservingPlugin) PreFilterExtensions() fwk.PreFilterExtensions {
	return rp
}

func (rp *reservingPlugin) AddPod(ctx context.Context, state fwk.CycleState, podToSchedule *v1.Pod, podInfoToAdd fwk.PodInfo, nodeInfo fwk.NodeInfo) *fwk.Status {
	if strings.Contains(podInfoToAdd.GetPod().Name, rp.nameOfPodToReserve) {
		state.Write(reservingPluginStateKey, reservingPluginState{reserved: true})
	}
	return nil
}

func (rp *reservingPlugin) RemovePod(ctx context.Context, state fwk.CycleState, podToSchedule *v1.Pod, podInfoToRemove fwk.PodInfo, nodeInfo fwk.NodeInfo) *fwk.Status {
	if strings.Contains(podInfoToRemove.GetPod().Name, rp.nameOfPodToReserve) {
		state.Write(reservingPluginStateKey, reservingPluginState{reserved: false})
	}
	return nil
}

var _ fwk.PreFilterPlugin = &reservingPlugin{}
var _ fwk.FilterPlugin = &reservingPlugin{}
var _ fwk.PreFilterExtensions = &reservingPlugin{}
var _ fwk.ReservePlugin = &reservingPlugin{}

type BlockedPod struct {
	Blocked chan struct{}
}

// blockingPermitPlugin is a Permit plugin that blocks until a signal is received.
type blockingPermitPlugin struct {
	podsToBlock map[string]*BlockedPod
}

// BlockingPermitPluginName is the name of the blocking permit plugin.
const BlockingPermitPluginName = "blocking-permit-plugin"

var _ fwk.PermitPlugin = &blockingPermitPlugin{}

// NewBlockingPermitPlugin creates a new blocking permit plugin.
// With map of pods to block on Permit until for preemptor pod.
func NewBlockingPermitPlugin(_ context.Context, _ runtime.Object, h fwk.Handle, podsToBlock map[string]*BlockedPod) fwk.Plugin {
	return &blockingPermitPlugin{
		podsToBlock: podsToBlock,
	}
}

func (pl *blockingPermitPlugin) Name() string {
	return BlockingPermitPluginName
}

func (pl *blockingPermitPlugin) Permit(ctx context.Context, state fwk.CycleState, pod *v1.Pod, nodeName string) (*fwk.Status, time.Duration) {
	if p, ok := pl.podsToBlock[pod.Name]; ok {
		delete(pl.podsToBlock, pod.Name)
		p.Blocked <- struct{}{}
		return fwk.NewStatus(fwk.Wait, "waiting"), time.Minute
	}
	return nil, 0
}
