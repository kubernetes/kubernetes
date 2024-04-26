/*
Copyright 2014 The Kubernetes Authors.

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
	"errors"
	"fmt"
	"math"
	"math/rand"
	"reflect"
	"regexp"
	"sort"
	"strconv"
	"sync"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	v1 "k8s.io/api/core/v1"
	eventsv1 "k8s.io/api/events/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/client-go/informers"
	clientsetfake "k8s.io/client-go/kubernetes/fake"
	"k8s.io/client-go/kubernetes/scheme"
	clienttesting "k8s.io/client-go/testing"
	clientcache "k8s.io/client-go/tools/cache"
	"k8s.io/client-go/tools/events"
	"k8s.io/component-helpers/storage/volume"
	"k8s.io/klog/v2"
	"k8s.io/klog/v2/ktesting"
	extenderv1 "k8s.io/kube-scheduler/extender/v1"
	schedulerapi "k8s.io/kubernetes/pkg/scheduler/apis/config"
	"k8s.io/kubernetes/pkg/scheduler/framework"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/defaultbinder"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/feature"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/nodeports"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/noderesources"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/podtopologyspread"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/queuesort"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/selectorspread"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/volumebinding"
	frameworkruntime "k8s.io/kubernetes/pkg/scheduler/framework/runtime"
	internalcache "k8s.io/kubernetes/pkg/scheduler/internal/cache"
	fakecache "k8s.io/kubernetes/pkg/scheduler/internal/cache/fake"
	internalqueue "k8s.io/kubernetes/pkg/scheduler/internal/queue"
	"k8s.io/kubernetes/pkg/scheduler/profile"
	st "k8s.io/kubernetes/pkg/scheduler/testing"
	schedutil "k8s.io/kubernetes/pkg/scheduler/util"
	"k8s.io/utils/pointer"
)

const (
	testSchedulerName = "test-scheduler"
)

var (
	emptySnapshot         = internalcache.NewEmptySnapshot()
	podTopologySpreadFunc = frameworkruntime.FactoryAdapter(feature.Features{}, podtopologyspread.New)
	errPrioritize         = fmt.Errorf("priority map encounters an error")
)

type mockScheduleResult struct {
	result ScheduleResult
	err    error
}

type fakeExtender struct {
	isBinder          bool
	interestedPodName string
	ignorable         bool
	gotBind           bool
}

func (f *fakeExtender) Name() string {
	return "fakeExtender"
}

func (f *fakeExtender) IsIgnorable() bool {
	return f.ignorable
}

func (f *fakeExtender) ProcessPreemption(
	_ *v1.Pod,
	_ map[string]*extenderv1.Victims,
	_ framework.NodeInfoLister,
) (map[string]*extenderv1.Victims, error) {
	return nil, nil
}

func (f *fakeExtender) SupportsPreemption() bool {
	return false
}

func (f *fakeExtender) Filter(pod *v1.Pod, nodes []*v1.Node) ([]*v1.Node, extenderv1.FailedNodesMap, extenderv1.FailedNodesMap, error) {
	return nil, nil, nil, nil
}

func (f *fakeExtender) Prioritize(
	_ *v1.Pod,
	_ []*v1.Node,
) (hostPriorities *extenderv1.HostPriorityList, weight int64, err error) {
	return nil, 0, nil
}

func (f *fakeExtender) Bind(binding *v1.Binding) error {
	if f.isBinder {
		f.gotBind = true
		return nil
	}
	return errors.New("not a binder")
}

func (f *fakeExtender) IsBinder() bool {
	return f.isBinder
}

func (f *fakeExtender) IsInterested(pod *v1.Pod) bool {
	return pod != nil && pod.Name == f.interestedPodName
}

type falseMapPlugin struct{}

func newFalseMapPlugin() frameworkruntime.PluginFactory {
	return func(_ runtime.Object, _ framework.Handle) (framework.Plugin, error) {
		return &falseMapPlugin{}, nil
	}
}

func (pl *falseMapPlugin) Name() string {
	return "FalseMap"
}

func (pl *falseMapPlugin) Score(_ context.Context, _ *framework.CycleState, _ *v1.Pod, _ string) (int64, *framework.Status) {
	return 0, framework.AsStatus(errPrioritize)
}

func (pl *falseMapPlugin) ScoreExtensions() framework.ScoreExtensions {
	return nil
}

type numericMapPlugin struct{}

func newNumericMapPlugin() frameworkruntime.PluginFactory {
	return func(_ runtime.Object, _ framework.Handle) (framework.Plugin, error) {
		return &numericMapPlugin{}, nil
	}
}

func (pl *numericMapPlugin) Name() string {
	return "NumericMap"
}

func (pl *numericMapPlugin) Score(_ context.Context, _ *framework.CycleState, _ *v1.Pod, nodeName string) (int64, *framework.Status) {
	score, err := strconv.Atoi(nodeName)
	if err != nil {
		return 0, framework.NewStatus(framework.Error, fmt.Sprintf("Error converting nodename to int: %+v", nodeName))
	}
	return int64(score), nil
}

func (pl *numericMapPlugin) ScoreExtensions() framework.ScoreExtensions {
	return nil
}

// NewNoPodsFilterPlugin initializes a noPodsFilterPlugin and returns it.
func NewNoPodsFilterPlugin(_ runtime.Object, _ framework.Handle) (framework.Plugin, error) {
	return &noPodsFilterPlugin{}, nil
}

type reverseNumericMapPlugin struct{}

func (pl *reverseNumericMapPlugin) Name() string {
	return "ReverseNumericMap"
}

func (pl *reverseNumericMapPlugin) Score(_ context.Context, _ *framework.CycleState, _ *v1.Pod, nodeName string) (int64, *framework.Status) {
	score, err := strconv.Atoi(nodeName)
	if err != nil {
		return 0, framework.NewStatus(framework.Error, fmt.Sprintf("Error converting nodename to int: %+v", nodeName))
	}
	return int64(score), nil
}

func (pl *reverseNumericMapPlugin) ScoreExtensions() framework.ScoreExtensions {
	return pl
}

func (pl *reverseNumericMapPlugin) NormalizeScore(_ context.Context, _ *framework.CycleState, _ *v1.Pod, nodeScores framework.NodeScoreList) *framework.Status {
	var maxScore float64
	minScore := math.MaxFloat64

	for _, hostPriority := range nodeScores {
		maxScore = math.Max(maxScore, float64(hostPriority.Score))
		minScore = math.Min(minScore, float64(hostPriority.Score))
	}
	for i, hostPriority := range nodeScores {
		nodeScores[i] = framework.NodeScore{
			Name:  hostPriority.Name,
			Score: int64(maxScore + minScore - float64(hostPriority.Score)),
		}
	}
	return nil
}

func newReverseNumericMapPlugin() frameworkruntime.PluginFactory {
	return func(_ runtime.Object, _ framework.Handle) (framework.Plugin, error) {
		return &reverseNumericMapPlugin{}, nil
	}
}

type trueMapPlugin struct{}

func (pl *trueMapPlugin) Name() string {
	return "TrueMap"
}

func (pl *trueMapPlugin) Score(_ context.Context, _ *framework.CycleState, _ *v1.Pod, _ string) (int64, *framework.Status) {
	return 1, nil
}

func (pl *trueMapPlugin) ScoreExtensions() framework.ScoreExtensions {
	return pl
}

func (pl *trueMapPlugin) NormalizeScore(_ context.Context, _ *framework.CycleState, _ *v1.Pod, nodeScores framework.NodeScoreList) *framework.Status {
	for _, host := range nodeScores {
		if host.Name == "" {
			return framework.NewStatus(framework.Error, "unexpected empty host name")
		}
	}
	return nil
}

func newTrueMapPlugin() frameworkruntime.PluginFactory {
	return func(_ runtime.Object, _ framework.Handle) (framework.Plugin, error) {
		return &trueMapPlugin{}, nil
	}
}

type noPodsFilterPlugin struct{}

// Name returns name of the plugin.
func (pl *noPodsFilterPlugin) Name() string {
	return "NoPodsFilter"
}

// Filter invoked at the filter extension point.
func (pl *noPodsFilterPlugin) Filter(_ context.Context, _ *framework.CycleState, pod *v1.Pod, nodeInfo *framework.NodeInfo) *framework.Status {
	if len(nodeInfo.Pods) == 0 {
		return nil
	}
	return framework.NewStatus(framework.Unschedulable, st.ErrReasonFake)
}

type fakeNodeSelectorArgs struct {
	NodeName string `json:"nodeName"`
}

type fakeNodeSelector struct {
	fakeNodeSelectorArgs
}

func (s *fakeNodeSelector) Name() string {
	return "FakeNodeSelector"
}

func (s *fakeNodeSelector) Filter(_ context.Context, _ *framework.CycleState, _ *v1.Pod, nodeInfo *framework.NodeInfo) *framework.Status {
	if nodeInfo.Node().Name != s.NodeName {
		return framework.NewStatus(framework.UnschedulableAndUnresolvable)
	}
	return nil
}

func newFakeNodeSelector(args runtime.Object, _ framework.Handle) (framework.Plugin, error) {
	pl := &fakeNodeSelector{}
	if err := frameworkruntime.DecodeInto(args, &pl.fakeNodeSelectorArgs); err != nil {
		return nil, err
	}
	return pl, nil
}

const (
	fakeSpecifiedNodeNameAnnotation = "fake-specified-node-name"
)

// fakeNodeSelectorDependOnPodAnnotation schedules pod to the specified one node from pod.Annotations[fakeSpecifiedNodeNameAnnotation].
type fakeNodeSelectorDependOnPodAnnotation struct{}

func (f *fakeNodeSelectorDependOnPodAnnotation) Name() string {
	return "FakeNodeSelectorDependOnPodAnnotation"
}

// Filter selects the specified one node and rejects other non-specified nodes.
func (f *fakeNodeSelectorDependOnPodAnnotation) Filter(_ context.Context, _ *framework.CycleState, pod *v1.Pod, nodeInfo *framework.NodeInfo) *framework.Status {
	resolveNodeNameFromPodAnnotation := func(pod *v1.Pod) (string, error) {
		if pod == nil {
			return "", fmt.Errorf("empty pod")
		}
		nodeName, ok := pod.Annotations[fakeSpecifiedNodeNameAnnotation]
		if !ok {
			return "", fmt.Errorf("no specified node name on pod %s/%s annotation", pod.Namespace, pod.Name)
		}
		return nodeName, nil
	}

	nodeName, err := resolveNodeNameFromPodAnnotation(pod)
	if err != nil {
		return framework.AsStatus(err)
	}
	if nodeInfo.Node().Name != nodeName {
		return framework.NewStatus(framework.UnschedulableAndUnresolvable)
	}
	return nil
}

func newFakeNodeSelectorDependOnPodAnnotation(_ runtime.Object, _ framework.Handle) (framework.Plugin, error) {
	return &fakeNodeSelectorDependOnPodAnnotation{}, nil
}

type TestPlugin struct {
	name string
}

var _ framework.ScorePlugin = &TestPlugin{}
var _ framework.FilterPlugin = &TestPlugin{}

func (t *TestPlugin) Name() string {
	return t.name
}

func (t *TestPlugin) Score(ctx context.Context, state *framework.CycleState, p *v1.Pod, nodeName string) (int64, *framework.Status) {
	return 1, nil
}

func (t *TestPlugin) ScoreExtensions() framework.ScoreExtensions {
	return nil
}

func (t *TestPlugin) Filter(ctx context.Context, state *framework.CycleState, pod *v1.Pod, nodeInfo *framework.NodeInfo) *framework.Status {
	return nil
}

func TestSchedulerMultipleProfilesScheduling(t *testing.T) {
	nodes := []runtime.Object{
		st.MakeNode().Name("node1").UID("node1").Obj(),
		st.MakeNode().Name("node2").UID("node2").Obj(),
		st.MakeNode().Name("node3").UID("node3").Obj(),
	}
	pods := []*v1.Pod{
		st.MakePod().Name("pod1").UID("pod1").SchedulerName("match-node3").Obj(),
		st.MakePod().Name("pod2").UID("pod2").SchedulerName("match-node2").Obj(),
		st.MakePod().Name("pod3").UID("pod3").SchedulerName("match-node2").Obj(),
		st.MakePod().Name("pod4").UID("pod4").SchedulerName("match-node3").Obj(),
	}
	wantBindings := map[string]string{
		"pod1": "node3",
		"pod2": "node2",
		"pod3": "node2",
		"pod4": "node3",
	}
	wantControllers := map[string]string{
		"pod1": "match-node3",
		"pod2": "match-node2",
		"pod3": "match-node2",
		"pod4": "match-node3",
	}

	// Set up scheduler for the 3 nodes.
	// We use a fake filter that only allows one particular node. We create two
	// profiles, each with a different node in the filter configuration.
	objs := append([]runtime.Object{
		&v1.Namespace{ObjectMeta: metav1.ObjectMeta{Name: ""}}}, nodes...)
	client := clientsetfake.NewSimpleClientset(objs...)
	broadcaster := events.NewBroadcaster(&events.EventSinkImpl{Interface: client.EventsV1()})
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	informerFactory := informers.NewSharedInformerFactory(client, 0)
	sched, err := New(
		ctx,
		client,
		informerFactory,
		nil,
		profile.NewRecorderFactory(broadcaster),
		WithProfiles(
			schedulerapi.KubeSchedulerProfile{SchedulerName: "match-node2",
				Plugins: &schedulerapi.Plugins{
					Filter:    schedulerapi.PluginSet{Enabled: []schedulerapi.Plugin{{Name: "FakeNodeSelector"}}},
					QueueSort: schedulerapi.PluginSet{Enabled: []schedulerapi.Plugin{{Name: "PrioritySort"}}},
					Bind:      schedulerapi.PluginSet{Enabled: []schedulerapi.Plugin{{Name: "DefaultBinder"}}},
				},
				PluginConfig: []schedulerapi.PluginConfig{
					{
						Name: "FakeNodeSelector",
						Args: &runtime.Unknown{Raw: []byte(`{"nodeName":"node2"}`)},
					},
				},
			},
			schedulerapi.KubeSchedulerProfile{
				SchedulerName: "match-node3",
				Plugins: &schedulerapi.Plugins{
					Filter:    schedulerapi.PluginSet{Enabled: []schedulerapi.Plugin{{Name: "FakeNodeSelector"}}},
					QueueSort: schedulerapi.PluginSet{Enabled: []schedulerapi.Plugin{{Name: "PrioritySort"}}},
					Bind:      schedulerapi.PluginSet{Enabled: []schedulerapi.Plugin{{Name: "DefaultBinder"}}},
				},
				PluginConfig: []schedulerapi.PluginConfig{
					{
						Name: "FakeNodeSelector",
						Args: &runtime.Unknown{Raw: []byte(`{"nodeName":"node3"}`)},
					},
				},
			},
		),
		WithFrameworkOutOfTreeRegistry(frameworkruntime.Registry{
			"FakeNodeSelector": newFakeNodeSelector,
		}),
	)
	if err != nil {
		t.Fatal(err)
	}

	// Capture the bindings and events' controllers.
	var wg sync.WaitGroup
	wg.Add(2 * len(pods))
	bindings := make(map[string]string)
	client.PrependReactor("create", "pods", func(action clienttesting.Action) (bool, runtime.Object, error) {
		if action.GetSubresource() != "binding" {
			return false, nil, nil
		}
		binding := action.(clienttesting.CreateAction).GetObject().(*v1.Binding)
		bindings[binding.Name] = binding.Target.Name
		wg.Done()
		return true, binding, nil
	})
	controllers := make(map[string]string)
	stopFn, err := broadcaster.StartEventWatcher(func(obj runtime.Object) {
		e, ok := obj.(*eventsv1.Event)
		if !ok || e.Reason != "Scheduled" {
			return
		}
		controllers[e.Regarding.Name] = e.ReportingController
		wg.Done()
	})
	if err != nil {
		t.Fatal(err)
	}
	defer stopFn()

	// Run scheduler.
	informerFactory.Start(ctx.Done())
	informerFactory.WaitForCacheSync(ctx.Done())
	if err = sched.WaitForHandlersSync(ctx); err != nil {
		t.Fatalf("Handlers failed to sync: %v: ", err)
	}
	go sched.Run(ctx)

	// Send pods to be scheduled.
	for _, p := range pods {
		_, err := client.CoreV1().Pods("").Create(ctx, p, metav1.CreateOptions{})
		if err != nil {
			t.Fatal(err)
		}
	}
	wg.Wait()

	// Verify correct bindings and reporting controllers.
	if diff := cmp.Diff(wantBindings, bindings); diff != "" {
		t.Errorf("pods were scheduled incorrectly (-want, +got):\n%s", diff)
	}
	if diff := cmp.Diff(wantControllers, controllers); diff != "" {
		t.Errorf("events were reported with wrong controllers (-want, +got):\n%s", diff)
	}
}

// TestSchedulerGuaranteeNonNilNodeInSchedulingCycle is for detecting potential panic on nil Node when iterating Nodes.
func TestSchedulerGuaranteeNonNilNodeInSchedulingCycle(t *testing.T) {
	random := rand.New(rand.NewSource(time.Now().UnixNano()))
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	var (
		initialNodeNumber        = 1000
		initialPodNumber         = 500
		waitSchedulingPodNumber  = 200
		deleteNodeNumberPerRound = 20
		createPodNumberPerRound  = 50

		fakeSchedulerName = "fake-scheduler"
		fakeNamespace     = "fake-namespace"

		initialNodes []runtime.Object
		initialPods  []runtime.Object
	)

	for i := 0; i < initialNodeNumber; i++ {
		nodeName := fmt.Sprintf("node%d", i)
		initialNodes = append(initialNodes, st.MakeNode().Name(nodeName).UID(nodeName).Obj())
	}
	// Randomly scatter initial pods onto nodes.
	for i := 0; i < initialPodNumber; i++ {
		podName := fmt.Sprintf("scheduled-pod%d", i)
		assignedNodeName := fmt.Sprintf("node%d", random.Intn(initialNodeNumber))
		initialPods = append(initialPods, st.MakePod().Name(podName).UID(podName).Node(assignedNodeName).Obj())
	}

	objs := []runtime.Object{&v1.Namespace{ObjectMeta: metav1.ObjectMeta{Name: fakeNamespace}}}
	objs = append(objs, initialNodes...)
	objs = append(objs, initialPods...)
	client := clientsetfake.NewSimpleClientset(objs...)
	broadcaster := events.NewBroadcaster(&events.EventSinkImpl{Interface: client.EventsV1()})

	informerFactory := informers.NewSharedInformerFactory(client, 0)
	sched, err := New(
		ctx,
		client,
		informerFactory,
		nil,
		profile.NewRecorderFactory(broadcaster),
		WithProfiles(
			schedulerapi.KubeSchedulerProfile{SchedulerName: fakeSchedulerName,
				Plugins: &schedulerapi.Plugins{
					Filter:    schedulerapi.PluginSet{Enabled: []schedulerapi.Plugin{{Name: "FakeNodeSelectorDependOnPodAnnotation"}}},
					QueueSort: schedulerapi.PluginSet{Enabled: []schedulerapi.Plugin{{Name: "PrioritySort"}}},
					Bind:      schedulerapi.PluginSet{Enabled: []schedulerapi.Plugin{{Name: "DefaultBinder"}}},
				},
			},
		),
		WithFrameworkOutOfTreeRegistry(frameworkruntime.Registry{
			"FakeNodeSelectorDependOnPodAnnotation": newFakeNodeSelectorDependOnPodAnnotation,
		}),
	)
	if err != nil {
		t.Fatal(err)
	}

	// Run scheduler.
	informerFactory.Start(ctx.Done())
	informerFactory.WaitForCacheSync(ctx.Done())
	go sched.Run(ctx)

	var deleteNodeIndex int
	deleteNodesOneRound := func() {
		for i := 0; i < deleteNodeNumberPerRound; i++ {
			if deleteNodeIndex >= initialNodeNumber {
				// all initial nodes are already deleted
				return
			}
			deleteNodeName := fmt.Sprintf("node%d", deleteNodeIndex)
			if err := client.CoreV1().Nodes().Delete(ctx, deleteNodeName, metav1.DeleteOptions{}); err != nil {
				t.Fatal(err)
			}
			deleteNodeIndex++
		}
	}
	var createPodIndex int
	createPodsOneRound := func() {
		if createPodIndex > waitSchedulingPodNumber {
			return
		}
		for i := 0; i < createPodNumberPerRound; i++ {
			podName := fmt.Sprintf("pod%d", createPodIndex)
			// Note: the node(specifiedNodeName) may already be deleted, which leads pod scheduled failed.
			specifiedNodeName := fmt.Sprintf("node%d", random.Intn(initialNodeNumber))

			waitSchedulingPod := st.MakePod().Namespace(fakeNamespace).Name(podName).UID(podName).Annotation(fakeSpecifiedNodeNameAnnotation, specifiedNodeName).SchedulerName(fakeSchedulerName).Obj()
			if _, err := client.CoreV1().Pods(fakeNamespace).Create(ctx, waitSchedulingPod, metav1.CreateOptions{}); err != nil {
				t.Fatal(err)
			}
			createPodIndex++
		}
	}

	// Following we start 2 goroutines asynchronously to detect potential racing issues:
	// 1) One is responsible for deleting several nodes in each round;
	// 2) Another is creating several pods in each round to trigger scheduling;
	// Those two goroutines will stop until ctx.Done() is called, which means all waiting pods are scheduled at least once.
	go wait.Until(deleteNodesOneRound, 10*time.Millisecond, ctx.Done())
	go wait.Until(createPodsOneRound, 9*time.Millisecond, ctx.Done())

	// Capture the events to wait all pods to be scheduled at least once.
	allWaitSchedulingPods := sets.NewString()
	for i := 0; i < waitSchedulingPodNumber; i++ {
		allWaitSchedulingPods.Insert(fmt.Sprintf("pod%d", i))
	}
	var wg sync.WaitGroup
	wg.Add(waitSchedulingPodNumber)
	stopFn, err := broadcaster.StartEventWatcher(func(obj runtime.Object) {
		e, ok := obj.(*eventsv1.Event)
		if !ok || (e.Reason != "Scheduled" && e.Reason != "FailedScheduling") {
			return
		}
		if allWaitSchedulingPods.Has(e.Regarding.Name) {
			wg.Done()
			allWaitSchedulingPods.Delete(e.Regarding.Name)
		}
	})
	if err != nil {
		t.Fatal(err)
	}
	defer stopFn()

	wg.Wait()
}

func TestSchedulerScheduleOne(t *testing.T) {
	testNode := v1.Node{ObjectMeta: metav1.ObjectMeta{Name: "node1", UID: types.UID("node1")}}
	client := clientsetfake.NewSimpleClientset(&testNode)
	eventBroadcaster := events.NewBroadcaster(&events.EventSinkImpl{Interface: client.EventsV1()})
	errS := errors.New("scheduler")
	errB := errors.New("binder")
	preBindErr := errors.New("on PreBind")

	table := []struct {
		name                string
		injectBindError     error
		sendPod             *v1.Pod
		registerPluginFuncs []st.RegisterPluginFunc
		expectErrorPod      *v1.Pod
		expectForgetPod     *v1.Pod
		expectAssumedPod    *v1.Pod
		expectError         error
		expectBind          *v1.Binding
		eventReason         string
		mockResult          mockScheduleResult
	}{
		{
			name:       "error reserve pod",
			sendPod:    podWithID("foo", ""),
			mockResult: mockScheduleResult{ScheduleResult{SuggestedHost: testNode.Name, EvaluatedNodes: 1, FeasibleNodes: 1}, nil},
			registerPluginFuncs: []st.RegisterPluginFunc{
				st.RegisterReservePlugin("FakeReserve", st.NewFakeReservePlugin(framework.NewStatus(framework.Error, "reserve error"))),
			},
			expectErrorPod:   podWithID("foo", testNode.Name),
			expectForgetPod:  podWithID("foo", testNode.Name),
			expectAssumedPod: podWithID("foo", testNode.Name),
			expectError:      fmt.Errorf(`running Reserve plugin "FakeReserve": %w`, errors.New("reserve error")),
			eventReason:      "FailedScheduling",
		},
		{
			name:       "error permit pod",
			sendPod:    podWithID("foo", ""),
			mockResult: mockScheduleResult{ScheduleResult{SuggestedHost: testNode.Name, EvaluatedNodes: 1, FeasibleNodes: 1}, nil},
			registerPluginFuncs: []st.RegisterPluginFunc{
				st.RegisterPermitPlugin("FakePermit", st.NewFakePermitPlugin(framework.NewStatus(framework.Error, "permit error"), time.Minute)),
			},
			expectErrorPod:   podWithID("foo", testNode.Name),
			expectForgetPod:  podWithID("foo", testNode.Name),
			expectAssumedPod: podWithID("foo", testNode.Name),
			expectError:      fmt.Errorf(`running Permit plugin "FakePermit": %w`, errors.New("permit error")),
			eventReason:      "FailedScheduling",
		},
		{
			name:       "error prebind pod",
			sendPod:    podWithID("foo", ""),
			mockResult: mockScheduleResult{ScheduleResult{SuggestedHost: testNode.Name, EvaluatedNodes: 1, FeasibleNodes: 1}, nil},
			registerPluginFuncs: []st.RegisterPluginFunc{
				st.RegisterPreBindPlugin("FakePreBind", st.NewFakePreBindPlugin(framework.AsStatus(preBindErr))),
			},
			expectErrorPod:   podWithID("foo", testNode.Name),
			expectForgetPod:  podWithID("foo", testNode.Name),
			expectAssumedPod: podWithID("foo", testNode.Name),
			expectError:      fmt.Errorf(`running PreBind plugin "FakePreBind": %w`, preBindErr),
			eventReason:      "FailedScheduling",
		},
		{
			name:             "bind assumed pod scheduled",
			sendPod:          podWithID("foo", ""),
			mockResult:       mockScheduleResult{ScheduleResult{SuggestedHost: testNode.Name, EvaluatedNodes: 1, FeasibleNodes: 1}, nil},
			expectBind:       &v1.Binding{ObjectMeta: metav1.ObjectMeta{Name: "foo", UID: types.UID("foo")}, Target: v1.ObjectReference{Kind: "Node", Name: testNode.Name}},
			expectAssumedPod: podWithID("foo", testNode.Name),
			eventReason:      "Scheduled",
		},
		{
			name:           "error pod failed scheduling",
			sendPod:        podWithID("foo", ""),
			mockResult:     mockScheduleResult{ScheduleResult{SuggestedHost: testNode.Name, EvaluatedNodes: 1, FeasibleNodes: 1}, errS},
			expectError:    errS,
			expectErrorPod: podWithID("foo", ""),
			eventReason:    "FailedScheduling",
		},
		{
			name:             "error bind forget pod failed scheduling",
			sendPod:          podWithID("foo", ""),
			mockResult:       mockScheduleResult{ScheduleResult{SuggestedHost: testNode.Name, EvaluatedNodes: 1, FeasibleNodes: 1}, nil},
			expectBind:       &v1.Binding{ObjectMeta: metav1.ObjectMeta{Name: "foo", UID: types.UID("foo")}, Target: v1.ObjectReference{Kind: "Node", Name: testNode.Name}},
			expectAssumedPod: podWithID("foo", testNode.Name),
			injectBindError:  errB,
			expectError:      fmt.Errorf("running Bind plugin %q: %w", "DefaultBinder", errors.New("binder")),
			expectErrorPod:   podWithID("foo", testNode.Name),
			expectForgetPod:  podWithID("foo", testNode.Name),
			eventReason:      "FailedScheduling",
		},
		{
			name:        "deleting pod",
			sendPod:     deletingPod("foo"),
			mockResult:  mockScheduleResult{ScheduleResult{}, nil},
			eventReason: "FailedScheduling",
		},
	}

	for _, item := range table {
		t.Run(item.name, func(t *testing.T) {
			var gotError error
			var gotPod *v1.Pod
			var gotForgetPod *v1.Pod
			var gotAssumedPod *v1.Pod
			var gotBinding *v1.Binding
			cache := &fakecache.Cache{
				ForgetFunc: func(pod *v1.Pod) {
					gotForgetPod = pod
				},
				AssumeFunc: func(pod *v1.Pod) {
					gotAssumedPod = pod
				},
				IsAssumedPodFunc: func(pod *v1.Pod) bool {
					if pod == nil || gotAssumedPod == nil {
						return false
					}
					return pod.UID == gotAssumedPod.UID
				},
			}
			client := clientsetfake.NewSimpleClientset(item.sendPod)
			client.PrependReactor("create", "pods", func(action clienttesting.Action) (bool, runtime.Object, error) {
				if action.GetSubresource() != "binding" {
					return false, nil, nil
				}
				gotBinding = action.(clienttesting.CreateAction).GetObject().(*v1.Binding)
				return true, gotBinding, item.injectBindError
			})
			registerPluginFuncs := append(item.registerPluginFuncs,
				st.RegisterQueueSortPlugin(queuesort.Name, queuesort.New),
				st.RegisterBindPlugin(defaultbinder.Name, defaultbinder.New),
			)
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			fwk, err := st.NewFramework(ctx,
				registerPluginFuncs,
				testSchedulerName,
				frameworkruntime.WithClientSet(client),
				frameworkruntime.WithEventRecorder(eventBroadcaster.NewRecorder(scheme.Scheme, testSchedulerName)))
			if err != nil {
				t.Fatal(err)
			}

			sched := &Scheduler{
				Cache:  cache,
				client: client,
				NextPod: func() (*framework.QueuedPodInfo, error) {
					return &framework.QueuedPodInfo{PodInfo: mustNewPodInfo(t, item.sendPod)}, nil
				},
				SchedulingQueue: internalqueue.NewTestQueue(ctx, nil),
				Profiles:        profile.Map{testSchedulerName: fwk},
			}

			sched.SchedulePod = func(ctx context.Context, fwk framework.Framework, state *framework.CycleState, pod *v1.Pod) (ScheduleResult, error) {
				return item.mockResult.result, item.mockResult.err
			}
			sched.FailureHandler = func(_ context.Context, fwk framework.Framework, p *framework.QueuedPodInfo, status *framework.Status, _ *framework.NominatingInfo, _ time.Time) {
				gotPod = p.Pod
				gotError = status.AsError()

				msg := truncateMessage(gotError.Error())
				fwk.EventRecorder().Eventf(p.Pod, nil, v1.EventTypeWarning, "FailedScheduling", "Scheduling", msg)
			}
			called := make(chan struct{})
			stopFunc, err := eventBroadcaster.StartEventWatcher(func(obj runtime.Object) {
				e, _ := obj.(*eventsv1.Event)
				if e.Reason != item.eventReason {
					t.Errorf("got event %v, want %v", e.Reason, item.eventReason)
				}
				close(called)
			})
			if err != nil {
				t.Fatal(err)
			}
			sched.scheduleOne(ctx)
			<-called
			if e, a := item.expectAssumedPod, gotAssumedPod; !reflect.DeepEqual(e, a) {
				t.Errorf("assumed pod: wanted %v, got %v", e, a)
			}
			if e, a := item.expectErrorPod, gotPod; !reflect.DeepEqual(e, a) {
				t.Errorf("error pod: wanted %v, got %v", e, a)
			}
			if e, a := item.expectForgetPod, gotForgetPod; !reflect.DeepEqual(e, a) {
				t.Errorf("forget pod: wanted %v, got %v", e, a)
			}
			if e, a := item.expectError, gotError; !reflect.DeepEqual(e, a) {
				t.Errorf("error: wanted %v, got %v", e, a)
			}
			if diff := cmp.Diff(item.expectBind, gotBinding); diff != "" {
				t.Errorf("got binding diff (-want, +got): %s", diff)
			}
			stopFunc()
		})
	}
}

func TestSchedulerNoPhantomPodAfterExpire(t *testing.T) {
	logger, ctx := ktesting.NewTestContext(t)
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()
	queuedPodStore := clientcache.NewFIFO(clientcache.MetaNamespaceKeyFunc)
	scache := internalcache.New(ctx, 100*time.Millisecond)
	pod := podWithPort("pod.Name", "", 8080)
	node := v1.Node{ObjectMeta: metav1.ObjectMeta{Name: "node1", UID: types.UID("node1")}}
	scache.AddNode(logger, &node)

	fns := []st.RegisterPluginFunc{
		st.RegisterQueueSortPlugin(queuesort.Name, queuesort.New),
		st.RegisterBindPlugin(defaultbinder.Name, defaultbinder.New),
		st.RegisterPluginAsExtensions(nodeports.Name, nodeports.New, "Filter", "PreFilter"),
	}
	scheduler, bindingChan, errChan := setupTestSchedulerWithOnePodOnNode(ctx, t, queuedPodStore, scache, pod, &node, fns...)

	waitPodExpireChan := make(chan struct{})
	timeout := make(chan struct{})
	go func() {
		for {
			select {
			case <-timeout:
				return
			default:
			}
			pods, err := scache.PodCount()
			if err != nil {
				errChan <- fmt.Errorf("cache.List failed: %v", err)
				return
			}
			if pods == 0 {
				close(waitPodExpireChan)
				return
			}
			time.Sleep(100 * time.Millisecond)
		}
	}()
	// waiting for the assumed pod to expire
	select {
	case err := <-errChan:
		t.Fatal(err)
	case <-waitPodExpireChan:
	case <-time.After(wait.ForeverTestTimeout):
		close(timeout)
		t.Fatalf("timeout timeout in waiting pod expire after %v", wait.ForeverTestTimeout)
	}

	// We use conflicted pod ports to incur fit predicate failure if first pod not removed.
	secondPod := podWithPort("bar", "", 8080)
	queuedPodStore.Add(secondPod)
	scheduler.scheduleOne(ctx)
	select {
	case b := <-bindingChan:
		expectBinding := &v1.Binding{
			ObjectMeta: metav1.ObjectMeta{Name: "bar", UID: types.UID("bar")},
			Target:     v1.ObjectReference{Kind: "Node", Name: node.Name},
		}
		if !reflect.DeepEqual(expectBinding, b) {
			t.Errorf("binding want=%v, get=%v", expectBinding, b)
		}
	case <-time.After(wait.ForeverTestTimeout):
		t.Fatalf("timeout in binding after %v", wait.ForeverTestTimeout)
	}
}

func TestSchedulerNoPhantomPodAfterDelete(t *testing.T) {
	logger, ctx := ktesting.NewTestContext(t)
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()
	queuedPodStore := clientcache.NewFIFO(clientcache.MetaNamespaceKeyFunc)
	scache := internalcache.New(ctx, 10*time.Minute)
	firstPod := podWithPort("pod.Name", "", 8080)
	node := v1.Node{ObjectMeta: metav1.ObjectMeta{Name: "node1", UID: types.UID("node1")}}
	scache.AddNode(logger, &node)
	fns := []st.RegisterPluginFunc{
		st.RegisterQueueSortPlugin(queuesort.Name, queuesort.New),
		st.RegisterBindPlugin(defaultbinder.Name, defaultbinder.New),
		st.RegisterPluginAsExtensions(nodeports.Name, nodeports.New, "Filter", "PreFilter"),
	}
	scheduler, bindingChan, errChan := setupTestSchedulerWithOnePodOnNode(ctx, t, queuedPodStore, scache, firstPod, &node, fns...)

	// We use conflicted pod ports to incur fit predicate failure.
	secondPod := podWithPort("bar", "", 8080)
	queuedPodStore.Add(secondPod)
	// queuedPodStore: [bar:8080]
	// cache: [(assumed)foo:8080]

	scheduler.scheduleOne(ctx)
	select {
	case err := <-errChan:
		expectErr := &framework.FitError{
			Pod:         secondPod,
			NumAllNodes: 1,
			Diagnosis: framework.Diagnosis{
				NodeToStatusMap: framework.NodeToStatusMap{
					node.Name: framework.NewStatus(framework.Unschedulable, nodeports.ErrReason).WithFailedPlugin(nodeports.Name),
				},
				UnschedulablePlugins: sets.New(nodeports.Name),
			},
		}
		if !reflect.DeepEqual(expectErr, err) {
			t.Errorf("err want=%v, get=%v", expectErr, err)
		}
	case <-time.After(wait.ForeverTestTimeout):
		t.Fatalf("timeout in fitting after %v", wait.ForeverTestTimeout)
	}

	// We mimic the workflow of cache behavior when a pod is removed by user.
	// Note: if the schedulernodeinfo timeout would be super short, the first pod would expire
	// and would be removed itself (without any explicit actions on schedulernodeinfo). Even in that case,
	// explicitly AddPod will as well correct the behavior.
	firstPod.Spec.NodeName = node.Name
	if err := scache.AddPod(logger, firstPod); err != nil {
		t.Fatalf("err: %v", err)
	}
	if err := scache.RemovePod(logger, firstPod); err != nil {
		t.Fatalf("err: %v", err)
	}

	queuedPodStore.Add(secondPod)
	scheduler.scheduleOne(ctx)
	select {
	case b := <-bindingChan:
		expectBinding := &v1.Binding{
			ObjectMeta: metav1.ObjectMeta{Name: "bar", UID: types.UID("bar")},
			Target:     v1.ObjectReference{Kind: "Node", Name: node.Name},
		}
		if !reflect.DeepEqual(expectBinding, b) {
			t.Errorf("binding want=%v, get=%v", expectBinding, b)
		}
	case <-time.After(wait.ForeverTestTimeout):
		t.Fatalf("timeout in binding after %v", wait.ForeverTestTimeout)
	}
}

func TestSchedulerFailedSchedulingReasons(t *testing.T) {
	logger, ctx := ktesting.NewTestContext(t)
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()
	queuedPodStore := clientcache.NewFIFO(clientcache.MetaNamespaceKeyFunc)
	scache := internalcache.New(ctx, 10*time.Minute)

	// Design the baseline for the pods, and we will make nodes that don't fit it later.
	var cpu = int64(4)
	var mem = int64(500)
	podWithTooBigResourceRequests := podWithResources("bar", "", v1.ResourceList{
		v1.ResourceCPU:    *(resource.NewQuantity(cpu, resource.DecimalSI)),
		v1.ResourceMemory: *(resource.NewQuantity(mem, resource.DecimalSI)),
	}, v1.ResourceList{
		v1.ResourceCPU:    *(resource.NewQuantity(cpu, resource.DecimalSI)),
		v1.ResourceMemory: *(resource.NewQuantity(mem, resource.DecimalSI)),
	})

	// create several nodes which cannot schedule the above pod
	var nodes []*v1.Node
	var objects []runtime.Object
	for i := 0; i < 100; i++ {
		uid := fmt.Sprintf("node%v", i)
		node := v1.Node{
			ObjectMeta: metav1.ObjectMeta{Name: uid, UID: types.UID(uid)},
			Status: v1.NodeStatus{
				Capacity: v1.ResourceList{
					v1.ResourceCPU:    *(resource.NewQuantity(cpu/2, resource.DecimalSI)),
					v1.ResourceMemory: *(resource.NewQuantity(mem/5, resource.DecimalSI)),
					v1.ResourcePods:   *(resource.NewQuantity(10, resource.DecimalSI)),
				},
				Allocatable: v1.ResourceList{
					v1.ResourceCPU:    *(resource.NewQuantity(cpu/2, resource.DecimalSI)),
					v1.ResourceMemory: *(resource.NewQuantity(mem/5, resource.DecimalSI)),
					v1.ResourcePods:   *(resource.NewQuantity(10, resource.DecimalSI)),
				}},
		}
		scache.AddNode(logger, &node)
		nodes = append(nodes, &node)
		objects = append(objects, &node)
	}

	// Create expected failure reasons for all the nodes. Hopefully they will get rolled up into a non-spammy summary.
	failedNodeStatues := framework.NodeToStatusMap{}
	for _, node := range nodes {
		failedNodeStatues[node.Name] = framework.NewStatus(
			framework.Unschedulable,
			fmt.Sprintf("Insufficient %v", v1.ResourceCPU),
			fmt.Sprintf("Insufficient %v", v1.ResourceMemory),
		).WithFailedPlugin(noderesources.Name)
	}
	fns := []st.RegisterPluginFunc{
		st.RegisterQueueSortPlugin(queuesort.Name, queuesort.New),
		st.RegisterBindPlugin(defaultbinder.Name, defaultbinder.New),
		st.RegisterPluginAsExtensions(noderesources.Name, frameworkruntime.FactoryAdapter(feature.Features{}, noderesources.NewFit), "Filter", "PreFilter"),
	}

	informerFactory := informers.NewSharedInformerFactory(clientsetfake.NewSimpleClientset(objects...), 0)
	scheduler, _, errChan := setupTestScheduler(ctx, t, queuedPodStore, scache, informerFactory, nil, fns...)

	queuedPodStore.Add(podWithTooBigResourceRequests)
	scheduler.scheduleOne(ctx)
	select {
	case err := <-errChan:
		expectErr := &framework.FitError{
			Pod:         podWithTooBigResourceRequests,
			NumAllNodes: len(nodes),
			Diagnosis: framework.Diagnosis{
				NodeToStatusMap:      failedNodeStatues,
				UnschedulablePlugins: sets.New(noderesources.Name),
			},
		}
		if len(fmt.Sprint(expectErr)) > 150 {
			t.Errorf("message is too spammy ! %v ", len(fmt.Sprint(expectErr)))
		}
		if !reflect.DeepEqual(expectErr, err) {
			t.Errorf("\n err \nWANT=%+v,\nGOT=%+v", expectErr, err)
		}
	case <-time.After(wait.ForeverTestTimeout):
		t.Fatalf("timeout after %v", wait.ForeverTestTimeout)
	}
}

func TestSchedulerWithVolumeBinding(t *testing.T) {
	findErr := fmt.Errorf("find err")
	assumeErr := fmt.Errorf("assume err")
	bindErr := fmt.Errorf("bind err")
	client := clientsetfake.NewSimpleClientset()

	eventBroadcaster := events.NewBroadcaster(&events.EventSinkImpl{Interface: client.EventsV1()})

	// This can be small because we wait for pod to finish scheduling first
	chanTimeout := 2 * time.Second

	table := []struct {
		name               string
		expectError        error
		expectPodBind      *v1.Binding
		expectAssumeCalled bool
		expectBindCalled   bool
		eventReason        string
		volumeBinderConfig *volumebinding.FakeVolumeBinderConfig
	}{
		{
			name: "all bound",
			volumeBinderConfig: &volumebinding.FakeVolumeBinderConfig{
				AllBound: true,
			},
			expectAssumeCalled: true,
			expectPodBind:      &v1.Binding{ObjectMeta: metav1.ObjectMeta{Name: "foo", Namespace: "foo-ns", UID: types.UID("foo")}, Target: v1.ObjectReference{Kind: "Node", Name: "node1"}},
			eventReason:        "Scheduled",
		},
		{
			name: "bound/invalid pv affinity",
			volumeBinderConfig: &volumebinding.FakeVolumeBinderConfig{
				AllBound:    true,
				FindReasons: volumebinding.ConflictReasons{volumebinding.ErrReasonNodeConflict},
			},
			eventReason: "FailedScheduling",
			expectError: makePredicateError("1 node(s) had volume node affinity conflict"),
		},
		{
			name: "unbound/no matches",
			volumeBinderConfig: &volumebinding.FakeVolumeBinderConfig{
				FindReasons: volumebinding.ConflictReasons{volumebinding.ErrReasonBindConflict},
			},
			eventReason: "FailedScheduling",
			expectError: makePredicateError("1 node(s) didn't find available persistent volumes to bind"),
		},
		{
			name: "bound and unbound unsatisfied",
			volumeBinderConfig: &volumebinding.FakeVolumeBinderConfig{
				FindReasons: volumebinding.ConflictReasons{volumebinding.ErrReasonBindConflict, volumebinding.ErrReasonNodeConflict},
			},
			eventReason: "FailedScheduling",
			expectError: makePredicateError("1 node(s) didn't find available persistent volumes to bind, 1 node(s) had volume node affinity conflict"),
		},
		{
			name:               "unbound/found matches/bind succeeds",
			volumeBinderConfig: &volumebinding.FakeVolumeBinderConfig{},
			expectAssumeCalled: true,
			expectBindCalled:   true,
			expectPodBind:      &v1.Binding{ObjectMeta: metav1.ObjectMeta{Name: "foo", Namespace: "foo-ns", UID: types.UID("foo")}, Target: v1.ObjectReference{Kind: "Node", Name: "node1"}},
			eventReason:        "Scheduled",
		},
		{
			name: "predicate error",
			volumeBinderConfig: &volumebinding.FakeVolumeBinderConfig{
				FindErr: findErr,
			},
			eventReason: "FailedScheduling",
			expectError: fmt.Errorf("running %q filter plugin: %v", volumebinding.Name, findErr),
		},
		{
			name: "assume error",
			volumeBinderConfig: &volumebinding.FakeVolumeBinderConfig{
				AssumeErr: assumeErr,
			},
			expectAssumeCalled: true,
			eventReason:        "FailedScheduling",
			expectError:        fmt.Errorf("running Reserve plugin %q: %w", volumebinding.Name, assumeErr),
		},
		{
			name: "bind error",
			volumeBinderConfig: &volumebinding.FakeVolumeBinderConfig{
				BindErr: bindErr,
			},
			expectAssumeCalled: true,
			expectBindCalled:   true,
			eventReason:        "FailedScheduling",
			expectError:        fmt.Errorf("running PreBind plugin %q: %w", volumebinding.Name, bindErr),
		},
	}

	for _, item := range table {
		t.Run(item.name, func(t *testing.T) {
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			fakeVolumeBinder := volumebinding.NewFakeVolumeBinder(item.volumeBinderConfig)
			s, bindingChan, errChan := setupTestSchedulerWithVolumeBinding(ctx, t, fakeVolumeBinder, eventBroadcaster)
			eventChan := make(chan struct{})
			stopFunc, err := eventBroadcaster.StartEventWatcher(func(obj runtime.Object) {
				e, _ := obj.(*eventsv1.Event)
				if e, a := item.eventReason, e.Reason; e != a {
					t.Errorf("expected %v, got %v", e, a)
				}
				close(eventChan)
			})
			if err != nil {
				t.Fatal(err)
			}
			s.scheduleOne(ctx)
			// Wait for pod to succeed or fail scheduling
			select {
			case <-eventChan:
			case <-time.After(wait.ForeverTestTimeout):
				t.Fatalf("scheduling timeout after %v", wait.ForeverTestTimeout)
			}
			stopFunc()
			// Wait for scheduling to return an error or succeed binding.
			var (
				gotErr  error
				gotBind *v1.Binding
			)
			select {
			case gotErr = <-errChan:
			case gotBind = <-bindingChan:
			case <-time.After(chanTimeout):
				t.Fatalf("did not receive pod binding or error after %v", chanTimeout)
			}
			if item.expectError != nil {
				if gotErr == nil || item.expectError.Error() != gotErr.Error() {
					t.Errorf("err \nWANT=%+v,\nGOT=%+v", item.expectError, gotErr)
				}
			} else if gotErr != nil {
				t.Errorf("err \nWANT=%+v,\nGOT=%+v", item.expectError, gotErr)
			}
			if !cmp.Equal(item.expectPodBind, gotBind) {
				t.Errorf("err \nWANT=%+v,\nGOT=%+v", item.expectPodBind, gotBind)
			}

			if item.expectAssumeCalled != fakeVolumeBinder.AssumeCalled {
				t.Errorf("expectedAssumeCall %v", item.expectAssumeCalled)
			}

			if item.expectBindCalled != fakeVolumeBinder.BindCalled {
				t.Errorf("expectedBindCall %v", item.expectBindCalled)
			}
		})
	}
}

func TestSchedulerBinding(t *testing.T) {
	table := []struct {
		podName      string
		extenders    []framework.Extender
		wantBinderID int
		name         string
	}{
		{
			name:    "the extender is not a binder",
			podName: "pod0",
			extenders: []framework.Extender{
				&fakeExtender{isBinder: false, interestedPodName: "pod0"},
			},
			wantBinderID: -1, // default binding.
		},
		{
			name:    "one of the extenders is a binder and interested in pod",
			podName: "pod0",
			extenders: []framework.Extender{
				&fakeExtender{isBinder: false, interestedPodName: "pod0"},
				&fakeExtender{isBinder: true, interestedPodName: "pod0"},
			},
			wantBinderID: 1,
		},
		{
			name:    "one of the extenders is a binder, but not interested in pod",
			podName: "pod1",
			extenders: []framework.Extender{
				&fakeExtender{isBinder: false, interestedPodName: "pod1"},
				&fakeExtender{isBinder: true, interestedPodName: "pod0"},
			},
			wantBinderID: -1, // default binding.
		},
	}

	for _, test := range table {
		t.Run(test.name, func(t *testing.T) {
			pod := st.MakePod().Name(test.podName).Obj()
			defaultBound := false
			client := clientsetfake.NewSimpleClientset(pod)
			client.PrependReactor("create", "pods", func(action clienttesting.Action) (bool, runtime.Object, error) {
				if action.GetSubresource() == "binding" {
					defaultBound = true
				}
				return false, nil, nil
			})
			_, ctx := ktesting.NewTestContext(t)
			ctx, cancel := context.WithCancel(ctx)
			defer cancel()
			fwk, err := st.NewFramework(ctx,
				[]st.RegisterPluginFunc{
					st.RegisterQueueSortPlugin(queuesort.Name, queuesort.New),
					st.RegisterBindPlugin(defaultbinder.Name, defaultbinder.New),
				}, "", frameworkruntime.WithClientSet(client), frameworkruntime.WithEventRecorder(&events.FakeRecorder{}))
			if err != nil {
				t.Fatal(err)
			}
			sched := &Scheduler{
				Extenders:                test.extenders,
				Cache:                    internalcache.New(ctx, 100*time.Millisecond),
				nodeInfoSnapshot:         nil,
				percentageOfNodesToScore: 0,
			}
			status := sched.bind(ctx, fwk, pod, "node", nil)
			if !status.IsSuccess() {
				t.Error(status.AsError())
			}

			// Checking default binding.
			if wantBound := test.wantBinderID == -1; defaultBound != wantBound {
				t.Errorf("got bound with default binding: %v, want %v", defaultBound, wantBound)
			}

			// Checking extenders binding.
			for i, ext := range test.extenders {
				wantBound := i == test.wantBinderID
				if gotBound := ext.(*fakeExtender).gotBind; gotBound != wantBound {
					t.Errorf("got bound with extender #%d: %v, want %v", i, gotBound, wantBound)
				}
			}

		})
	}
}

func TestUpdatePod(t *testing.T) {
	tests := []struct {
		name                     string
		currentPodConditions     []v1.PodCondition
		newPodCondition          *v1.PodCondition
		currentNominatedNodeName string
		newNominatingInfo        *framework.NominatingInfo
		expectedPatchRequests    int
		expectedPatchDataPattern string
	}{
		{
			name:                 "Should make patch request to add pod condition when there are none currently",
			currentPodConditions: []v1.PodCondition{},
			newPodCondition: &v1.PodCondition{
				Type:               "newType",
				Status:             "newStatus",
				LastProbeTime:      metav1.NewTime(time.Date(2020, 5, 13, 1, 1, 1, 1, time.UTC)),
				LastTransitionTime: metav1.NewTime(time.Date(2020, 5, 12, 1, 1, 1, 1, time.UTC)),
				Reason:             "newReason",
				Message:            "newMessage",
			},
			expectedPatchRequests:    1,
			expectedPatchDataPattern: `{"status":{"conditions":\[{"lastProbeTime":"2020-05-13T01:01:01Z","lastTransitionTime":".*","message":"newMessage","reason":"newReason","status":"newStatus","type":"newType"}]}}`,
		},
		{
			name: "Should make patch request to add a new pod condition when there is already one with another type",
			currentPodConditions: []v1.PodCondition{
				{
					Type:               "someOtherType",
					Status:             "someOtherTypeStatus",
					LastProbeTime:      metav1.NewTime(time.Date(2020, 5, 11, 0, 0, 0, 0, time.UTC)),
					LastTransitionTime: metav1.NewTime(time.Date(2020, 5, 10, 0, 0, 0, 0, time.UTC)),
					Reason:             "someOtherTypeReason",
					Message:            "someOtherTypeMessage",
				},
			},
			newPodCondition: &v1.PodCondition{
				Type:               "newType",
				Status:             "newStatus",
				LastProbeTime:      metav1.NewTime(time.Date(2020, 5, 13, 1, 1, 1, 1, time.UTC)),
				LastTransitionTime: metav1.NewTime(time.Date(2020, 5, 12, 1, 1, 1, 1, time.UTC)),
				Reason:             "newReason",
				Message:            "newMessage",
			},
			expectedPatchRequests:    1,
			expectedPatchDataPattern: `{"status":{"\$setElementOrder/conditions":\[{"type":"someOtherType"},{"type":"newType"}],"conditions":\[{"lastProbeTime":"2020-05-13T01:01:01Z","lastTransitionTime":".*","message":"newMessage","reason":"newReason","status":"newStatus","type":"newType"}]}}`,
		},
		{
			name: "Should make patch request to update an existing pod condition",
			currentPodConditions: []v1.PodCondition{
				{
					Type:               "currentType",
					Status:             "currentStatus",
					LastProbeTime:      metav1.NewTime(time.Date(2020, 5, 13, 0, 0, 0, 0, time.UTC)),
					LastTransitionTime: metav1.NewTime(time.Date(2020, 5, 12, 0, 0, 0, 0, time.UTC)),
					Reason:             "currentReason",
					Message:            "currentMessage",
				},
			},
			newPodCondition: &v1.PodCondition{
				Type:               "currentType",
				Status:             "newStatus",
				LastProbeTime:      metav1.NewTime(time.Date(2020, 5, 13, 1, 1, 1, 1, time.UTC)),
				LastTransitionTime: metav1.NewTime(time.Date(2020, 5, 12, 1, 1, 1, 1, time.UTC)),
				Reason:             "newReason",
				Message:            "newMessage",
			},
			expectedPatchRequests:    1,
			expectedPatchDataPattern: `{"status":{"\$setElementOrder/conditions":\[{"type":"currentType"}],"conditions":\[{"lastProbeTime":"2020-05-13T01:01:01Z","lastTransitionTime":".*","message":"newMessage","reason":"newReason","status":"newStatus","type":"currentType"}]}}`,
		},
		{
			name: "Should make patch request to update an existing pod condition, but the transition time should remain unchanged because the status is the same",
			currentPodConditions: []v1.PodCondition{
				{
					Type:               "currentType",
					Status:             "currentStatus",
					LastProbeTime:      metav1.NewTime(time.Date(2020, 5, 13, 0, 0, 0, 0, time.UTC)),
					LastTransitionTime: metav1.NewTime(time.Date(2020, 5, 12, 0, 0, 0, 0, time.UTC)),
					Reason:             "currentReason",
					Message:            "currentMessage",
				},
			},
			newPodCondition: &v1.PodCondition{
				Type:               "currentType",
				Status:             "currentStatus",
				LastProbeTime:      metav1.NewTime(time.Date(2020, 5, 13, 1, 1, 1, 1, time.UTC)),
				LastTransitionTime: metav1.NewTime(time.Date(2020, 5, 12, 0, 0, 0, 0, time.UTC)),
				Reason:             "newReason",
				Message:            "newMessage",
			},
			expectedPatchRequests:    1,
			expectedPatchDataPattern: `{"status":{"\$setElementOrder/conditions":\[{"type":"currentType"}],"conditions":\[{"lastProbeTime":"2020-05-13T01:01:01Z","message":"newMessage","reason":"newReason","type":"currentType"}]}}`,
		},
		{
			name: "Should not make patch request if pod condition already exists and is identical and nominated node name is not set",
			currentPodConditions: []v1.PodCondition{
				{
					Type:               "currentType",
					Status:             "currentStatus",
					LastProbeTime:      metav1.NewTime(time.Date(2020, 5, 13, 0, 0, 0, 0, time.UTC)),
					LastTransitionTime: metav1.NewTime(time.Date(2020, 5, 12, 0, 0, 0, 0, time.UTC)),
					Reason:             "currentReason",
					Message:            "currentMessage",
				},
			},
			newPodCondition: &v1.PodCondition{
				Type:               "currentType",
				Status:             "currentStatus",
				LastProbeTime:      metav1.NewTime(time.Date(2020, 5, 13, 0, 0, 0, 0, time.UTC)),
				LastTransitionTime: metav1.NewTime(time.Date(2020, 5, 12, 0, 0, 0, 0, time.UTC)),
				Reason:             "currentReason",
				Message:            "currentMessage",
			},
			currentNominatedNodeName: "node1",
			expectedPatchRequests:    0,
		},
		{
			name: "Should make patch request if pod condition already exists and is identical but nominated node name is set and different",
			currentPodConditions: []v1.PodCondition{
				{
					Type:               "currentType",
					Status:             "currentStatus",
					LastProbeTime:      metav1.NewTime(time.Date(2020, 5, 13, 0, 0, 0, 0, time.UTC)),
					LastTransitionTime: metav1.NewTime(time.Date(2020, 5, 12, 0, 0, 0, 0, time.UTC)),
					Reason:             "currentReason",
					Message:            "currentMessage",
				},
			},
			newPodCondition: &v1.PodCondition{
				Type:               "currentType",
				Status:             "currentStatus",
				LastProbeTime:      metav1.NewTime(time.Date(2020, 5, 13, 0, 0, 0, 0, time.UTC)),
				LastTransitionTime: metav1.NewTime(time.Date(2020, 5, 12, 0, 0, 0, 0, time.UTC)),
				Reason:             "currentReason",
				Message:            "currentMessage",
			},
			newNominatingInfo:        &framework.NominatingInfo{NominatingMode: framework.ModeOverride, NominatedNodeName: "node1"},
			expectedPatchRequests:    1,
			expectedPatchDataPattern: `{"status":{"nominatedNodeName":"node1"}}`,
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			actualPatchRequests := 0
			var actualPatchData string
			cs := &clientsetfake.Clientset{}
			cs.AddReactor("patch", "pods", func(action clienttesting.Action) (bool, runtime.Object, error) {
				actualPatchRequests++
				patch := action.(clienttesting.PatchAction)
				actualPatchData = string(patch.GetPatch())
				// For this test, we don't care about the result of the patched pod, just that we got the expected
				// patch request, so just returning &v1.Pod{} here is OK because scheduler doesn't use the response.
				return true, &v1.Pod{}, nil
			})

			pod := st.MakePod().Name("foo").NominatedNodeName(test.currentNominatedNodeName).Conditions(test.currentPodConditions).Obj()

			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			if err := updatePod(ctx, cs, pod, test.newPodCondition, test.newNominatingInfo); err != nil {
				t.Fatalf("Error calling update: %v", err)
			}

			if actualPatchRequests != test.expectedPatchRequests {
				t.Fatalf("Actual patch requests (%d) does not equal expected patch requests (%d), actual patch data: %v", actualPatchRequests, test.expectedPatchRequests, actualPatchData)
			}

			regex, err := regexp.Compile(test.expectedPatchDataPattern)
			if err != nil {
				t.Fatalf("Error compiling regexp for %v: %v", test.expectedPatchDataPattern, err)
			}

			if test.expectedPatchRequests > 0 && !regex.MatchString(actualPatchData) {
				t.Fatalf("Patch data mismatch: Actual was %v, but expected to match regexp %v", actualPatchData, test.expectedPatchDataPattern)
			}
		})
	}
}

func Test_SelectHost(t *testing.T) {
	tests := []struct {
		name              string
		list              []framework.NodePluginScores
		topNodesCnt       int
		possibleNodes     sets.Set[string]
		possibleNodeLists [][]framework.NodePluginScores
		wantError         error
	}{
		{
			name: "unique properly ordered scores",
			list: []framework.NodePluginScores{
				{Name: "node1", TotalScore: 1},
				{Name: "node2", TotalScore: 2},
			},
			topNodesCnt:   2,
			possibleNodes: sets.New("node2"),
			possibleNodeLists: [][]framework.NodePluginScores{
				{
					{Name: "node2", TotalScore: 2},
					{Name: "node1", TotalScore: 1},
				},
			},
		},
		{
			name: "numberOfNodeScoresToReturn > len(list)",
			list: []framework.NodePluginScores{
				{Name: "node1", TotalScore: 1},
				{Name: "node2", TotalScore: 2},
			},
			topNodesCnt:   100,
			possibleNodes: sets.New("node2"),
			possibleNodeLists: [][]framework.NodePluginScores{
				{
					{Name: "node2", TotalScore: 2},
					{Name: "node1", TotalScore: 1},
				},
			},
		},
		{
			name: "equal scores",
			list: []framework.NodePluginScores{
				{Name: "node2.1", TotalScore: 2},
				{Name: "node2.2", TotalScore: 2},
				{Name: "node2.3", TotalScore: 2},
			},
			topNodesCnt:   2,
			possibleNodes: sets.New("node2.1", "node2.2", "node2.3"),
			possibleNodeLists: [][]framework.NodePluginScores{
				{
					{Name: "node2.1", TotalScore: 2},
					{Name: "node2.2", TotalScore: 2},
				},
				{
					{Name: "node2.1", TotalScore: 2},
					{Name: "node2.3", TotalScore: 2},
				},
				{
					{Name: "node2.2", TotalScore: 2},
					{Name: "node2.1", TotalScore: 2},
				},
				{
					{Name: "node2.2", TotalScore: 2},
					{Name: "node2.3", TotalScore: 2},
				},
				{
					{Name: "node2.3", TotalScore: 2},
					{Name: "node2.1", TotalScore: 2},
				},
				{
					{Name: "node2.3", TotalScore: 2},
					{Name: "node2.2", TotalScore: 2},
				},
			},
		},
		{
			name: "out of order scores",
			list: []framework.NodePluginScores{
				{Name: "node3.1", TotalScore: 3},
				{Name: "node2.1", TotalScore: 2},
				{Name: "node1.1", TotalScore: 1},
				{Name: "node3.2", TotalScore: 3},
			},
			topNodesCnt:   3,
			possibleNodes: sets.New("node3.1", "node3.2"),
			possibleNodeLists: [][]framework.NodePluginScores{
				{
					{Name: "node3.1", TotalScore: 3},
					{Name: "node3.2", TotalScore: 3},
					{Name: "node2.1", TotalScore: 2},
				},
				{
					{Name: "node3.2", TotalScore: 3},
					{Name: "node3.1", TotalScore: 3},
					{Name: "node2.1", TotalScore: 2},
				},
			},
		},
		{
			name:          "empty priority list",
			list:          []framework.NodePluginScores{},
			possibleNodes: sets.Set[string]{},
			wantError:     errEmptyPriorityList,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			// increase the randomness
			for i := 0; i < 10; i++ {
				got, scoreList, err := selectHost(test.list, test.topNodesCnt)
				if err != test.wantError {
					t.Fatalf("unexpected error is returned from selectHost: got: %v want: %v", err, test.wantError)
				}
				if test.possibleNodes.Len() == 0 {
					if got != "" {
						t.Fatalf("expected nothing returned as selected Node, but actually %s is returned from selectHost", got)
					}
					return
				}
				if !test.possibleNodes.Has(got) {
					t.Errorf("got %s is not in the possible map %v", got, test.possibleNodes)
				}
				if got != scoreList[0].Name {
					t.Errorf("The head of list should be the selected Node's score: got: %v, expected: %v", scoreList[0], got)
				}
				for _, list := range test.possibleNodeLists {
					if cmp.Equal(list, scoreList) {
						return
					}
				}
				t.Errorf("Unexpected scoreList: %v", scoreList)
			}
		})
	}
}

func TestFindNodesThatPassExtenders(t *testing.T) {
	tests := []struct {
		name                  string
		extenders             []st.FakeExtender
		nodes                 []*v1.Node
		filteredNodesStatuses framework.NodeToStatusMap
		expectsErr            bool
		expectedNodes         []*v1.Node
		expectedStatuses      framework.NodeToStatusMap
	}{
		{
			name: "error",
			extenders: []st.FakeExtender{
				{
					ExtenderName: "FakeExtender1",
					Predicates:   []st.FitPredicate{st.ErrorPredicateExtender},
				},
			},
			nodes:                 makeNodeList([]string{"a"}),
			filteredNodesStatuses: make(framework.NodeToStatusMap),
			expectsErr:            true,
		},
		{
			name: "success",
			extenders: []st.FakeExtender{
				{
					ExtenderName: "FakeExtender1",
					Predicates:   []st.FitPredicate{st.TruePredicateExtender},
				},
			},
			nodes:                 makeNodeList([]string{"a"}),
			filteredNodesStatuses: make(framework.NodeToStatusMap),
			expectsErr:            false,
			expectedNodes:         makeNodeList([]string{"a"}),
			expectedStatuses:      make(framework.NodeToStatusMap),
		},
		{
			name: "unschedulable",
			extenders: []st.FakeExtender{
				{
					ExtenderName: "FakeExtender1",
					Predicates: []st.FitPredicate{func(pod *v1.Pod, node *v1.Node) *framework.Status {
						if node.Name == "a" {
							return framework.NewStatus(framework.Success)
						}
						return framework.NewStatus(framework.Unschedulable, fmt.Sprintf("node %q is not allowed", node.Name))
					}},
				},
			},
			nodes:                 makeNodeList([]string{"a", "b"}),
			filteredNodesStatuses: make(framework.NodeToStatusMap),
			expectsErr:            false,
			expectedNodes:         makeNodeList([]string{"a"}),
			expectedStatuses: framework.NodeToStatusMap{
				"b": framework.NewStatus(framework.Unschedulable, fmt.Sprintf("FakeExtender: node %q failed", "b")),
			},
		},
		{
			name: "unschedulable and unresolvable",
			extenders: []st.FakeExtender{
				{
					ExtenderName: "FakeExtender1",
					Predicates: []st.FitPredicate{func(pod *v1.Pod, node *v1.Node) *framework.Status {
						if node.Name == "a" {
							return framework.NewStatus(framework.Success)
						}
						if node.Name == "b" {
							return framework.NewStatus(framework.Unschedulable, fmt.Sprintf("node %q is not allowed", node.Name))
						}
						return framework.NewStatus(framework.UnschedulableAndUnresolvable, fmt.Sprintf("node %q is not allowed", node.Name))
					}},
				},
			},
			nodes:                 makeNodeList([]string{"a", "b", "c"}),
			filteredNodesStatuses: make(framework.NodeToStatusMap),
			expectsErr:            false,
			expectedNodes:         makeNodeList([]string{"a"}),
			expectedStatuses: framework.NodeToStatusMap{
				"b": framework.NewStatus(framework.Unschedulable, fmt.Sprintf("FakeExtender: node %q failed", "b")),
				"c": framework.NewStatus(framework.UnschedulableAndUnresolvable, fmt.Sprintf("FakeExtender: node %q failed and unresolvable", "c")),
			},
		},
		{
			name: "extender may overwrite the statuses",
			extenders: []st.FakeExtender{
				{
					ExtenderName: "FakeExtender1",
					Predicates: []st.FitPredicate{func(pod *v1.Pod, node *v1.Node) *framework.Status {
						if node.Name == "a" {
							return framework.NewStatus(framework.Success)
						}
						if node.Name == "b" {
							return framework.NewStatus(framework.Unschedulable, fmt.Sprintf("node %q is not allowed", node.Name))
						}
						return framework.NewStatus(framework.UnschedulableAndUnresolvable, fmt.Sprintf("node %q is not allowed", node.Name))
					}},
				},
			},
			nodes: makeNodeList([]string{"a", "b", "c"}),
			filteredNodesStatuses: framework.NodeToStatusMap{
				"c": framework.NewStatus(framework.Unschedulable, fmt.Sprintf("FakeFilterPlugin: node %q failed", "c")),
			},
			expectsErr:    false,
			expectedNodes: makeNodeList([]string{"a"}),
			expectedStatuses: framework.NodeToStatusMap{
				"b": framework.NewStatus(framework.Unschedulable, fmt.Sprintf("FakeExtender: node %q failed", "b")),
				"c": framework.NewStatus(framework.UnschedulableAndUnresolvable, fmt.Sprintf("FakeFilterPlugin: node %q failed", "c"), fmt.Sprintf("FakeExtender: node %q failed and unresolvable", "c")),
			},
		},
		{
			name: "multiple extenders",
			extenders: []st.FakeExtender{
				{
					ExtenderName: "FakeExtender1",
					Predicates: []st.FitPredicate{func(pod *v1.Pod, node *v1.Node) *framework.Status {
						if node.Name == "a" {
							return framework.NewStatus(framework.Success)
						}
						if node.Name == "b" {
							return framework.NewStatus(framework.Unschedulable, fmt.Sprintf("node %q is not allowed", node.Name))
						}
						return framework.NewStatus(framework.UnschedulableAndUnresolvable, fmt.Sprintf("node %q is not allowed", node.Name))
					}},
				},
				{
					ExtenderName: "FakeExtender1",
					Predicates: []st.FitPredicate{func(pod *v1.Pod, node *v1.Node) *framework.Status {
						if node.Name == "a" {
							return framework.NewStatus(framework.Success)
						}
						return framework.NewStatus(framework.Unschedulable, fmt.Sprintf("node %q is not allowed", node.Name))
					}},
				},
			},
			nodes:                 makeNodeList([]string{"a", "b", "c"}),
			filteredNodesStatuses: make(framework.NodeToStatusMap),
			expectsErr:            false,
			expectedNodes:         makeNodeList([]string{"a"}),
			expectedStatuses: framework.NodeToStatusMap{
				"b": framework.NewStatus(framework.Unschedulable, fmt.Sprintf("FakeExtender: node %q failed", "b")),
				"c": framework.NewStatus(framework.UnschedulableAndUnresolvable, fmt.Sprintf("FakeExtender: node %q failed and unresolvable", "c")),
			},
		},
	}

	cmpOpts := []cmp.Option{
		cmp.Comparer(func(s1 framework.Status, s2 framework.Status) bool {
			return s1.Code() == s2.Code() && reflect.DeepEqual(s1.Reasons(), s2.Reasons())
		}),
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, ctx := ktesting.NewTestContext(t)
			var extenders []framework.Extender
			for ii := range tt.extenders {
				extenders = append(extenders, &tt.extenders[ii])
			}

			pod := st.MakePod().Name("1").UID("1").Obj()
			got, err := findNodesThatPassExtenders(ctx, extenders, pod, tt.nodes, tt.filteredNodesStatuses)
			if tt.expectsErr {
				if err == nil {
					t.Error("Unexpected non-error")
				}
			} else {
				if err != nil {
					t.Errorf("Unexpected error: %v", err)
				}
				if diff := cmp.Diff(tt.expectedNodes, got); diff != "" {
					t.Errorf("filtered nodes (-want,+got):\n%s", diff)
				}
				if diff := cmp.Diff(tt.expectedStatuses, tt.filteredNodesStatuses, cmpOpts...); diff != "" {
					t.Errorf("filtered statuses (-want,+got):\n%s", diff)
				}
			}
		})
	}
}

func TestSchedulerSchedulePod(t *testing.T) {
	fts := feature.Features{}
	tests := []struct {
		name               string
		registerPlugins    []st.RegisterPluginFunc
		extenders          []st.FakeExtender
		nodes              []string
		pvcs               []v1.PersistentVolumeClaim
		pod                *v1.Pod
		pods               []*v1.Pod
		wantNodes          sets.Set[string]
		wantEvaluatedNodes *int32
		wErr               error
	}{
		{
			registerPlugins: []st.RegisterPluginFunc{
				st.RegisterQueueSortPlugin(queuesort.Name, queuesort.New),
				st.RegisterFilterPlugin("FalseFilter", st.NewFalseFilterPlugin),
				st.RegisterBindPlugin(defaultbinder.Name, defaultbinder.New),
			},
			nodes: []string{"node1", "node2"},
			pod:   st.MakePod().Name("2").UID("2").Obj(),
			name:  "test 1",
			wErr: &framework.FitError{
				Pod:         st.MakePod().Name("2").UID("2").Obj(),
				NumAllNodes: 2,
				Diagnosis: framework.Diagnosis{
					NodeToStatusMap: framework.NodeToStatusMap{
						"node1": framework.NewStatus(framework.Unschedulable, st.ErrReasonFake).WithFailedPlugin("FalseFilter"),
						"node2": framework.NewStatus(framework.Unschedulable, st.ErrReasonFake).WithFailedPlugin("FalseFilter"),
					},
					UnschedulablePlugins: sets.New("FalseFilter"),
				},
			},
		},
		{
			registerPlugins: []st.RegisterPluginFunc{
				st.RegisterQueueSortPlugin(queuesort.Name, queuesort.New),
				st.RegisterFilterPlugin("TrueFilter", st.NewTrueFilterPlugin),
				st.RegisterBindPlugin(defaultbinder.Name, defaultbinder.New),
			},
			nodes:     []string{"node1", "node2"},
			pod:       st.MakePod().Name("ignore").UID("ignore").Obj(),
			wantNodes: sets.New("node1", "node2"),
			name:      "test 2",
			wErr:      nil,
		},
		{
			// Fits on a node where the pod ID matches the node name
			registerPlugins: []st.RegisterPluginFunc{
				st.RegisterQueueSortPlugin(queuesort.Name, queuesort.New),
				st.RegisterFilterPlugin("MatchFilter", st.NewMatchFilterPlugin),
				st.RegisterBindPlugin(defaultbinder.Name, defaultbinder.New),
			},
			nodes:     []string{"node1", "node2"},
			pod:       st.MakePod().Name("node2").UID("node2").Obj(),
			wantNodes: sets.New("node2"),
			name:      "test 3",
			wErr:      nil,
		},
		{
			registerPlugins: []st.RegisterPluginFunc{
				st.RegisterQueueSortPlugin(queuesort.Name, queuesort.New),
				st.RegisterFilterPlugin("TrueFilter", st.NewTrueFilterPlugin),
				st.RegisterScorePlugin("NumericMap", newNumericMapPlugin(), 1),
				st.RegisterBindPlugin(defaultbinder.Name, defaultbinder.New),
			},
			nodes:     []string{"3", "2", "1"},
			pod:       st.MakePod().Name("ignore").UID("ignore").Obj(),
			wantNodes: sets.New("3"),
			name:      "test 4",
			wErr:      nil,
		},
		{
			registerPlugins: []st.RegisterPluginFunc{
				st.RegisterQueueSortPlugin(queuesort.Name, queuesort.New),
				st.RegisterFilterPlugin("MatchFilter", st.NewMatchFilterPlugin),
				st.RegisterScorePlugin("NumericMap", newNumericMapPlugin(), 1),
				st.RegisterBindPlugin(defaultbinder.Name, defaultbinder.New),
			},
			nodes:     []string{"3", "2", "1"},
			pod:       st.MakePod().Name("2").UID("2").Obj(),
			wantNodes: sets.New("2"),
			name:      "test 5",
			wErr:      nil,
		},
		{
			registerPlugins: []st.RegisterPluginFunc{
				st.RegisterQueueSortPlugin(queuesort.Name, queuesort.New),
				st.RegisterFilterPlugin("TrueFilter", st.NewTrueFilterPlugin),
				st.RegisterScorePlugin("NumericMap", newNumericMapPlugin(), 1),
				st.RegisterScorePlugin("ReverseNumericMap", newReverseNumericMapPlugin(), 2),
				st.RegisterBindPlugin(defaultbinder.Name, defaultbinder.New),
			},
			nodes:     []string{"3", "2", "1"},
			pod:       st.MakePod().Name("2").UID("2").Obj(),
			wantNodes: sets.New("1"),
			name:      "test 6",
			wErr:      nil,
		},
		{
			registerPlugins: []st.RegisterPluginFunc{
				st.RegisterQueueSortPlugin(queuesort.Name, queuesort.New),
				st.RegisterFilterPlugin("TrueFilter", st.NewTrueFilterPlugin),
				st.RegisterFilterPlugin("FalseFilter", st.NewFalseFilterPlugin),
				st.RegisterScorePlugin("NumericMap", newNumericMapPlugin(), 1),
				st.RegisterBindPlugin(defaultbinder.Name, defaultbinder.New),
			},
			nodes: []string{"3", "2", "1"},
			pod:   st.MakePod().Name("2").UID("2").Obj(),
			name:  "test 7",
			wErr: &framework.FitError{
				Pod:         st.MakePod().Name("2").UID("2").Obj(),
				NumAllNodes: 3,
				Diagnosis: framework.Diagnosis{
					NodeToStatusMap: framework.NodeToStatusMap{
						"3": framework.NewStatus(framework.Unschedulable, st.ErrReasonFake).WithFailedPlugin("FalseFilter"),
						"2": framework.NewStatus(framework.Unschedulable, st.ErrReasonFake).WithFailedPlugin("FalseFilter"),
						"1": framework.NewStatus(framework.Unschedulable, st.ErrReasonFake).WithFailedPlugin("FalseFilter"),
					},
					UnschedulablePlugins: sets.New("FalseFilter"),
				},
			},
		},
		{
			registerPlugins: []st.RegisterPluginFunc{
				st.RegisterQueueSortPlugin(queuesort.Name, queuesort.New),
				st.RegisterFilterPlugin("NoPodsFilter", NewNoPodsFilterPlugin),
				st.RegisterFilterPlugin("MatchFilter", st.NewMatchFilterPlugin),
				st.RegisterScorePlugin("NumericMap", newNumericMapPlugin(), 1),
				st.RegisterBindPlugin(defaultbinder.Name, defaultbinder.New),
			},
			pods: []*v1.Pod{
				st.MakePod().Name("2").UID("2").Node("2").Phase(v1.PodRunning).Obj(),
			},
			pod:   st.MakePod().Name("2").UID("2").Obj(),
			nodes: []string{"1", "2"},
			name:  "test 8",
			wErr: &framework.FitError{
				Pod:         st.MakePod().Name("2").UID("2").Obj(),
				NumAllNodes: 2,
				Diagnosis: framework.Diagnosis{
					NodeToStatusMap: framework.NodeToStatusMap{
						"1": framework.NewStatus(framework.Unschedulable, st.ErrReasonFake).WithFailedPlugin("MatchFilter"),
						"2": framework.NewStatus(framework.Unschedulable, st.ErrReasonFake).WithFailedPlugin("NoPodsFilter"),
					},
					UnschedulablePlugins: sets.New("MatchFilter", "NoPodsFilter"),
				},
			},
		},
		{
			// Pod with existing PVC
			registerPlugins: []st.RegisterPluginFunc{
				st.RegisterQueueSortPlugin(queuesort.Name, queuesort.New),
				st.RegisterPreFilterPlugin(volumebinding.Name, frameworkruntime.FactoryAdapter(fts, volumebinding.New)),
				st.RegisterFilterPlugin("TrueFilter", st.NewTrueFilterPlugin),
				st.RegisterBindPlugin(defaultbinder.Name, defaultbinder.New),
			},
			nodes: []string{"node1", "node2"},
			pvcs: []v1.PersistentVolumeClaim{
				{
					ObjectMeta: metav1.ObjectMeta{Name: "existingPVC", UID: types.UID("existingPVC"), Namespace: v1.NamespaceDefault},
					Spec:       v1.PersistentVolumeClaimSpec{VolumeName: "existingPV"},
				},
			},
			pod:       st.MakePod().Name("ignore").UID("ignore").Namespace(v1.NamespaceDefault).PVC("existingPVC").Obj(),
			wantNodes: sets.New("node1", "node2"),
			name:      "existing PVC",
			wErr:      nil,
		},
		{
			// Pod with non existing PVC
			registerPlugins: []st.RegisterPluginFunc{
				st.RegisterQueueSortPlugin(queuesort.Name, queuesort.New),
				st.RegisterPreFilterPlugin(volumebinding.Name, frameworkruntime.FactoryAdapter(fts, volumebinding.New)),
				st.RegisterFilterPlugin("TrueFilter", st.NewTrueFilterPlugin),
				st.RegisterBindPlugin(defaultbinder.Name, defaultbinder.New),
			},
			nodes: []string{"node1", "node2"},
			pod:   st.MakePod().Name("ignore").UID("ignore").PVC("unknownPVC").Obj(),
			name:  "unknown PVC",
			wErr: &framework.FitError{
				Pod:         st.MakePod().Name("ignore").UID("ignore").PVC("unknownPVC").Obj(),
				NumAllNodes: 2,
				Diagnosis: framework.Diagnosis{
					NodeToStatusMap: framework.NodeToStatusMap{
						"node1": framework.NewStatus(framework.UnschedulableAndUnresolvable, `persistentvolumeclaim "unknownPVC" not found`).WithFailedPlugin("VolumeBinding"),
						"node2": framework.NewStatus(framework.UnschedulableAndUnresolvable, `persistentvolumeclaim "unknownPVC" not found`).WithFailedPlugin("VolumeBinding"),
					},
					PreFilterMsg:         `persistentvolumeclaim "unknownPVC" not found`,
					UnschedulablePlugins: sets.New(volumebinding.Name),
				},
			},
		},
		{
			// Pod with deleting PVC
			registerPlugins: []st.RegisterPluginFunc{
				st.RegisterQueueSortPlugin(queuesort.Name, queuesort.New),
				st.RegisterPreFilterPlugin(volumebinding.Name, frameworkruntime.FactoryAdapter(fts, volumebinding.New)),
				st.RegisterFilterPlugin("TrueFilter", st.NewTrueFilterPlugin),
				st.RegisterBindPlugin(defaultbinder.Name, defaultbinder.New),
			},
			nodes: []string{"node1", "node2"},
			pvcs:  []v1.PersistentVolumeClaim{{ObjectMeta: metav1.ObjectMeta{Name: "existingPVC", UID: types.UID("existingPVC"), Namespace: v1.NamespaceDefault, DeletionTimestamp: &metav1.Time{}}}},
			pod:   st.MakePod().Name("ignore").UID("ignore").Namespace(v1.NamespaceDefault).PVC("existingPVC").Obj(),
			name:  "deleted PVC",
			wErr: &framework.FitError{
				Pod:         st.MakePod().Name("ignore").UID("ignore").Namespace(v1.NamespaceDefault).PVC("existingPVC").Obj(),
				NumAllNodes: 2,
				Diagnosis: framework.Diagnosis{
					NodeToStatusMap: framework.NodeToStatusMap{
						"node1": framework.NewStatus(framework.UnschedulableAndUnresolvable, `persistentvolumeclaim "existingPVC" is being deleted`).WithFailedPlugin("VolumeBinding"),
						"node2": framework.NewStatus(framework.UnschedulableAndUnresolvable, `persistentvolumeclaim "existingPVC" is being deleted`).WithFailedPlugin("VolumeBinding"),
					},
					PreFilterMsg:         `persistentvolumeclaim "existingPVC" is being deleted`,
					UnschedulablePlugins: sets.New(volumebinding.Name),
				},
			},
		},
		{
			registerPlugins: []st.RegisterPluginFunc{
				st.RegisterQueueSortPlugin(queuesort.Name, queuesort.New),
				st.RegisterFilterPlugin("TrueFilter", st.NewTrueFilterPlugin),
				st.RegisterScorePlugin("FalseMap", newFalseMapPlugin(), 1),
				st.RegisterScorePlugin("TrueMap", newTrueMapPlugin(), 2),
				st.RegisterBindPlugin(defaultbinder.Name, defaultbinder.New),
			},
			nodes: []string{"2", "1"},
			pod:   st.MakePod().Name("2").Obj(),
			name:  "test error with priority map",
			wErr:  fmt.Errorf("running Score plugins: %w", fmt.Errorf(`plugin "FalseMap" failed with: %w`, errPrioritize)),
		},
		{
			name: "test podtopologyspread plugin - 2 nodes with maxskew=1",
			registerPlugins: []st.RegisterPluginFunc{
				st.RegisterQueueSortPlugin(queuesort.Name, queuesort.New),
				st.RegisterPluginAsExtensions(
					podtopologyspread.Name,
					podTopologySpreadFunc,
					"PreFilter",
					"Filter",
				),
				st.RegisterBindPlugin(defaultbinder.Name, defaultbinder.New),
			},
			nodes: []string{"node1", "node2"},
			pod: st.MakePod().Name("p").UID("p").Label("foo", "").SpreadConstraint(1, "hostname", v1.DoNotSchedule, &metav1.LabelSelector{
				MatchExpressions: []metav1.LabelSelectorRequirement{
					{
						Key:      "foo",
						Operator: metav1.LabelSelectorOpExists,
					},
				},
			}, nil, nil, nil, nil).Obj(),
			pods: []*v1.Pod{
				st.MakePod().Name("pod1").UID("pod1").Label("foo", "").Node("node1").Phase(v1.PodRunning).Obj(),
			},
			wantNodes: sets.New("node2"),
			wErr:      nil,
		},
		{
			name: "test podtopologyspread plugin - 3 nodes with maxskew=2",
			registerPlugins: []st.RegisterPluginFunc{
				st.RegisterQueueSortPlugin(queuesort.Name, queuesort.New),
				st.RegisterPluginAsExtensions(
					podtopologyspread.Name,
					podTopologySpreadFunc,
					"PreFilter",
					"Filter",
				),
				st.RegisterBindPlugin(defaultbinder.Name, defaultbinder.New),
			},
			nodes: []string{"node1", "node2", "node3"},
			pod: st.MakePod().Name("p").UID("p").Label("foo", "").SpreadConstraint(2, "hostname", v1.DoNotSchedule, &metav1.LabelSelector{
				MatchExpressions: []metav1.LabelSelectorRequirement{
					{
						Key:      "foo",
						Operator: metav1.LabelSelectorOpExists,
					},
				},
			}, nil, nil, nil, nil).Obj(),
			pods: []*v1.Pod{
				st.MakePod().Name("pod1a").UID("pod1a").Label("foo", "").Node("node1").Phase(v1.PodRunning).Obj(),
				st.MakePod().Name("pod1b").UID("pod1b").Label("foo", "").Node("node1").Phase(v1.PodRunning).Obj(),
				st.MakePod().Name("pod2").UID("pod2").Label("foo", "").Node("node2").Phase(v1.PodRunning).Obj(),
			},
			wantNodes: sets.New("node2", "node3"),
			wErr:      nil,
		},
		{
			name: "test with filter plugin returning Unschedulable status",
			registerPlugins: []st.RegisterPluginFunc{
				st.RegisterQueueSortPlugin(queuesort.Name, queuesort.New),
				st.RegisterFilterPlugin(
					"FakeFilter",
					st.NewFakeFilterPlugin(map[string]framework.Code{"3": framework.Unschedulable}),
				),
				st.RegisterScorePlugin("NumericMap", newNumericMapPlugin(), 1),
				st.RegisterBindPlugin(defaultbinder.Name, defaultbinder.New),
			},
			nodes:     []string{"3"},
			pod:       st.MakePod().Name("test-filter").UID("test-filter").Obj(),
			wantNodes: nil,
			wErr: &framework.FitError{
				Pod:         st.MakePod().Name("test-filter").UID("test-filter").Obj(),
				NumAllNodes: 1,
				Diagnosis: framework.Diagnosis{
					NodeToStatusMap: framework.NodeToStatusMap{
						"3": framework.NewStatus(framework.Unschedulable, "injecting failure for pod test-filter").WithFailedPlugin("FakeFilter"),
					},
					UnschedulablePlugins: sets.New("FakeFilter"),
				},
			},
		},
		{
			name: "test with extender which filters out some Nodes",
			registerPlugins: []st.RegisterPluginFunc{
				st.RegisterQueueSortPlugin(queuesort.Name, queuesort.New),
				st.RegisterFilterPlugin(
					"FakeFilter",
					st.NewFakeFilterPlugin(map[string]framework.Code{"3": framework.Unschedulable}),
				),
				st.RegisterScorePlugin("NumericMap", newNumericMapPlugin(), 1),
				st.RegisterBindPlugin(defaultbinder.Name, defaultbinder.New),
			},
			extenders: []st.FakeExtender{
				{
					ExtenderName: "FakeExtender1",
					Predicates:   []st.FitPredicate{st.FalsePredicateExtender},
				},
			},
			nodes:     []string{"1", "2", "3"},
			pod:       st.MakePod().Name("test-filter").UID("test-filter").Obj(),
			wantNodes: nil,
			wErr: &framework.FitError{
				Pod:         st.MakePod().Name("test-filter").UID("test-filter").Obj(),
				NumAllNodes: 3,
				Diagnosis: framework.Diagnosis{
					NodeToStatusMap: framework.NodeToStatusMap{
						"1": framework.NewStatus(framework.Unschedulable, `FakeExtender: node "1" failed`),
						"2": framework.NewStatus(framework.Unschedulable, `FakeExtender: node "2" failed`),
						"3": framework.NewStatus(framework.Unschedulable, "injecting failure for pod test-filter").WithFailedPlugin("FakeFilter"),
					},
					UnschedulablePlugins: sets.New("FakeFilter", framework.ExtenderName),
				},
			},
		},
		{
			name: "test with filter plugin returning UnschedulableAndUnresolvable status",
			registerPlugins: []st.RegisterPluginFunc{
				st.RegisterQueueSortPlugin(queuesort.Name, queuesort.New),
				st.RegisterFilterPlugin(
					"FakeFilter",
					st.NewFakeFilterPlugin(map[string]framework.Code{"3": framework.UnschedulableAndUnresolvable}),
				),
				st.RegisterScorePlugin("NumericMap", newNumericMapPlugin(), 1),
				st.RegisterBindPlugin(defaultbinder.Name, defaultbinder.New),
			},
			nodes:     []string{"3"},
			pod:       st.MakePod().Name("test-filter").UID("test-filter").Obj(),
			wantNodes: nil,
			wErr: &framework.FitError{
				Pod:         st.MakePod().Name("test-filter").UID("test-filter").Obj(),
				NumAllNodes: 1,
				Diagnosis: framework.Diagnosis{
					NodeToStatusMap: framework.NodeToStatusMap{
						"3": framework.NewStatus(framework.UnschedulableAndUnresolvable, "injecting failure for pod test-filter").WithFailedPlugin("FakeFilter"),
					},
					UnschedulablePlugins: sets.New("FakeFilter"),
				},
			},
		},
		{
			name: "test with partial failed filter plugin",
			registerPlugins: []st.RegisterPluginFunc{
				st.RegisterQueueSortPlugin(queuesort.Name, queuesort.New),
				st.RegisterFilterPlugin(
					"FakeFilter",
					st.NewFakeFilterPlugin(map[string]framework.Code{"1": framework.Unschedulable}),
				),
				st.RegisterScorePlugin("NumericMap", newNumericMapPlugin(), 1),
				st.RegisterBindPlugin(defaultbinder.Name, defaultbinder.New),
			},
			nodes:     []string{"1", "2"},
			pod:       st.MakePod().Name("test-filter").UID("test-filter").Obj(),
			wantNodes: nil,
			wErr:      nil,
		},
		{
			name: "test prefilter plugin returning Unschedulable status",
			registerPlugins: []st.RegisterPluginFunc{
				st.RegisterQueueSortPlugin(queuesort.Name, queuesort.New),
				st.RegisterPreFilterPlugin(
					"FakePreFilter",
					st.NewFakePreFilterPlugin("FakePreFilter", nil, framework.NewStatus(framework.UnschedulableAndUnresolvable, "injected unschedulable status")),
				),
				st.RegisterBindPlugin(defaultbinder.Name, defaultbinder.New),
			},
			nodes:     []string{"1", "2"},
			pod:       st.MakePod().Name("test-prefilter").UID("test-prefilter").Obj(),
			wantNodes: nil,
			wErr: &framework.FitError{
				Pod:         st.MakePod().Name("test-prefilter").UID("test-prefilter").Obj(),
				NumAllNodes: 2,
				Diagnosis: framework.Diagnosis{
					NodeToStatusMap: framework.NodeToStatusMap{
						"1": framework.NewStatus(framework.UnschedulableAndUnresolvable, "injected unschedulable status").WithFailedPlugin("FakePreFilter"),
						"2": framework.NewStatus(framework.UnschedulableAndUnresolvable, "injected unschedulable status").WithFailedPlugin("FakePreFilter"),
					},
					PreFilterMsg:         "injected unschedulable status",
					UnschedulablePlugins: sets.New("FakePreFilter"),
				},
			},
		},
		{
			name: "test prefilter plugin returning error status",
			registerPlugins: []st.RegisterPluginFunc{
				st.RegisterQueueSortPlugin(queuesort.Name, queuesort.New),
				st.RegisterPreFilterPlugin(
					"FakePreFilter",
					st.NewFakePreFilterPlugin("FakePreFilter", nil, framework.NewStatus(framework.Error, "injected error status")),
				),
				st.RegisterBindPlugin(defaultbinder.Name, defaultbinder.New),
			},
			nodes:     []string{"1", "2"},
			pod:       st.MakePod().Name("test-prefilter").UID("test-prefilter").Obj(),
			wantNodes: nil,
			wErr:      fmt.Errorf(`running PreFilter plugin "FakePreFilter": %w`, errors.New("injected error status")),
		},
		{
			name: "test prefilter plugin returning node",
			registerPlugins: []st.RegisterPluginFunc{
				st.RegisterQueueSortPlugin(queuesort.Name, queuesort.New),
				st.RegisterPreFilterPlugin(
					"FakePreFilter1",
					st.NewFakePreFilterPlugin("FakePreFilter1", nil, nil),
				),
				st.RegisterPreFilterPlugin(
					"FakePreFilter2",
					st.NewFakePreFilterPlugin("FakePreFilter2", &framework.PreFilterResult{NodeNames: sets.New("node2")}, nil),
				),
				st.RegisterPreFilterPlugin(
					"FakePreFilter3",
					st.NewFakePreFilterPlugin("FakePreFilter3", &framework.PreFilterResult{NodeNames: sets.New("node1", "node2")}, nil),
				),
				st.RegisterBindPlugin(defaultbinder.Name, defaultbinder.New),
			},
			nodes:              []string{"node1", "node2", "node3"},
			pod:                st.MakePod().Name("test-prefilter").UID("test-prefilter").Obj(),
			wantNodes:          sets.New("node2"),
			wantEvaluatedNodes: pointer.Int32(3),
		},
		{
			name: "test prefilter plugin returning non-intersecting nodes",
			registerPlugins: []st.RegisterPluginFunc{
				st.RegisterQueueSortPlugin(queuesort.Name, queuesort.New),
				st.RegisterPreFilterPlugin(
					"FakePreFilter1",
					st.NewFakePreFilterPlugin("FakePreFilter1", nil, nil),
				),
				st.RegisterPreFilterPlugin(
					"FakePreFilter2",
					st.NewFakePreFilterPlugin("FakePreFilter2", &framework.PreFilterResult{NodeNames: sets.New("node2")}, nil),
				),
				st.RegisterPreFilterPlugin(
					"FakePreFilter3",
					st.NewFakePreFilterPlugin("FakePreFilter3", &framework.PreFilterResult{NodeNames: sets.New("node1")}, nil),
				),
				st.RegisterBindPlugin(defaultbinder.Name, defaultbinder.New),
			},
			nodes: []string{"node1", "node2", "node3"},
			pod:   st.MakePod().Name("test-prefilter").UID("test-prefilter").Obj(),
			wErr: &framework.FitError{
				Pod:         st.MakePod().Name("test-prefilter").UID("test-prefilter").Obj(),
				NumAllNodes: 3,
				Diagnosis: framework.Diagnosis{
					NodeToStatusMap: framework.NodeToStatusMap{
						"node1": framework.NewStatus(framework.Unschedulable, "node(s) didn't satisfy plugin(s) [FakePreFilter2 FakePreFilter3] simultaneously"),
						"node2": framework.NewStatus(framework.Unschedulable, "node(s) didn't satisfy plugin(s) [FakePreFilter2 FakePreFilter3] simultaneously"),
						"node3": framework.NewStatus(framework.Unschedulable, "node(s) didn't satisfy plugin(s) [FakePreFilter2 FakePreFilter3] simultaneously"),
					},
					UnschedulablePlugins: sets.Set[string]{},
					PreFilterMsg:         "node(s) didn't satisfy plugin(s) [FakePreFilter2 FakePreFilter3] simultaneously",
				},
			},
		},
		{
			name: "test prefilter plugin returning empty node set",
			registerPlugins: []st.RegisterPluginFunc{
				st.RegisterQueueSortPlugin(queuesort.Name, queuesort.New),
				st.RegisterPreFilterPlugin(
					"FakePreFilter1",
					st.NewFakePreFilterPlugin("FakePreFilter1", nil, nil),
				),
				st.RegisterPreFilterPlugin(
					"FakePreFilter2",
					st.NewFakePreFilterPlugin("FakePreFilter2", &framework.PreFilterResult{NodeNames: sets.New[string]()}, nil),
				),
				st.RegisterBindPlugin(defaultbinder.Name, defaultbinder.New),
			},
			nodes: []string{"node1"},
			pod:   st.MakePod().Name("test-prefilter").UID("test-prefilter").Obj(),
			wErr: &framework.FitError{
				Pod:         st.MakePod().Name("test-prefilter").UID("test-prefilter").Obj(),
				NumAllNodes: 1,
				Diagnosis: framework.Diagnosis{
					NodeToStatusMap: framework.NodeToStatusMap{
						"node1": framework.NewStatus(framework.Unschedulable, "node(s) didn't satisfy plugin FakePreFilter2"),
					},
					UnschedulablePlugins: sets.Set[string]{},
					PreFilterMsg:         "node(s) didn't satisfy plugin FakePreFilter2",
				},
			},
		},
		{
			name: "test prefilter plugin returning skip",
			registerPlugins: []st.RegisterPluginFunc{
				st.RegisterQueueSortPlugin(queuesort.Name, queuesort.New),
				st.RegisterPreFilterPlugin(
					"FakePreFilter1",
					st.NewFakePreFilterPlugin("FakeFilter1", nil, nil),
				),
				st.RegisterFilterPlugin(
					"FakeFilter1",
					st.NewFakeFilterPlugin(map[string]framework.Code{
						"node1": framework.Unschedulable,
					}),
				),
				st.RegisterPluginAsExtensions("FakeFilter2", func(configuration runtime.Object, f framework.Handle) (framework.Plugin, error) {
					return st.FakePreFilterAndFilterPlugin{
						FakePreFilterPlugin: &st.FakePreFilterPlugin{
							Result: nil,
							Status: framework.NewStatus(framework.Skip),
						},
						FakeFilterPlugin: &st.FakeFilterPlugin{
							// This Filter plugin shouldn't be executed in the Filter extension point due to skip.
							// To confirm that, return the status code Error to all Nodes.
							FailedNodeReturnCodeMap: map[string]framework.Code{
								"node1": framework.Error, "node2": framework.Error, "node3": framework.Error,
							},
						},
					}, nil
				}, "PreFilter", "Filter"),
				st.RegisterBindPlugin(defaultbinder.Name, defaultbinder.New),
			},
			nodes:              []string{"node1", "node2", "node3"},
			pod:                st.MakePod().Name("test-prefilter").UID("test-prefilter").Obj(),
			wantNodes:          sets.New("node2", "node3"),
			wantEvaluatedNodes: pointer.Int32(3),
		},
		{
			name: "test all prescore plugins return skip",
			registerPlugins: []st.RegisterPluginFunc{
				st.RegisterQueueSortPlugin(queuesort.Name, queuesort.New),
				st.RegisterFilterPlugin("TrueFilter", st.NewTrueFilterPlugin),
				st.RegisterBindPlugin(defaultbinder.Name, defaultbinder.New),
				st.RegisterPluginAsExtensions("FakePreScoreAndScorePlugin", st.NewFakePreScoreAndScorePlugin("FakePreScoreAndScorePlugin", 0,
					framework.NewStatus(framework.Skip, "fake skip"),
					framework.NewStatus(framework.Error, "this score function shouldn't be executed because this plugin returned Skip in the PreScore"),
				), "PreScore", "Score"),
			},
			nodes:     []string{"node1", "node2"},
			pod:       st.MakePod().Name("ignore").UID("ignore").Obj(),
			wantNodes: sets.New("node1", "node2"),
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			logger, ctx := ktesting.NewTestContext(t)
			ctx, cancel := context.WithCancel(ctx)
			defer cancel()

			cache := internalcache.New(ctx, time.Duration(0))
			for _, pod := range test.pods {
				cache.AddPod(logger, pod)
			}
			var nodes []*v1.Node
			for _, name := range test.nodes {
				node := &v1.Node{ObjectMeta: metav1.ObjectMeta{Name: name, Labels: map[string]string{"hostname": name}}}
				nodes = append(nodes, node)
				cache.AddNode(logger, node)
			}

			cs := clientsetfake.NewSimpleClientset()
			informerFactory := informers.NewSharedInformerFactory(cs, 0)
			for _, pvc := range test.pvcs {
				metav1.SetMetaDataAnnotation(&pvc.ObjectMeta, volume.AnnBindCompleted, "true")
				cs.CoreV1().PersistentVolumeClaims(pvc.Namespace).Create(ctx, &pvc, metav1.CreateOptions{})
				if pvName := pvc.Spec.VolumeName; pvName != "" {
					pv := v1.PersistentVolume{ObjectMeta: metav1.ObjectMeta{Name: pvName}}
					cs.CoreV1().PersistentVolumes().Create(ctx, &pv, metav1.CreateOptions{})
				}
			}
			snapshot := internalcache.NewSnapshot(test.pods, nodes)
			fwk, err := st.NewFramework(
				ctx,
				test.registerPlugins, "",
				frameworkruntime.WithSnapshotSharedLister(snapshot),
				frameworkruntime.WithInformerFactory(informerFactory),
				frameworkruntime.WithPodNominator(internalqueue.NewPodNominator(informerFactory.Core().V1().Pods().Lister())),
			)
			if err != nil {
				t.Fatal(err)
			}

			var extenders []framework.Extender
			for ii := range test.extenders {
				extenders = append(extenders, &test.extenders[ii])
			}
			sched := &Scheduler{
				Cache:                    cache,
				nodeInfoSnapshot:         snapshot,
				percentageOfNodesToScore: schedulerapi.DefaultPercentageOfNodesToScore,
				Extenders:                extenders,
			}
			sched.applyDefaultHandlers()

			informerFactory.Start(ctx.Done())
			informerFactory.WaitForCacheSync(ctx.Done())

			result, err := sched.SchedulePod(ctx, fwk, framework.NewCycleState(), test.pod)
			if err != test.wErr {
				gotFitErr, gotOK := err.(*framework.FitError)
				wantFitErr, wantOK := test.wErr.(*framework.FitError)
				if gotOK != wantOK {
					t.Errorf("Expected err to be FitError: %v, but got %v (error: %v)", wantOK, gotOK, err)
				} else if gotOK {
					if diff := cmp.Diff(gotFitErr, wantFitErr); diff != "" {
						t.Errorf("Unexpected fitErr: (-want, +got): %s", diff)
					}
				}
			}
			if test.wantNodes != nil && !test.wantNodes.Has(result.SuggestedHost) {
				t.Errorf("Expected: %s, got: %s", test.wantNodes, result.SuggestedHost)
			}
			wantEvaluatedNodes := len(test.nodes)
			if test.wantEvaluatedNodes != nil {
				wantEvaluatedNodes = int(*test.wantEvaluatedNodes)
			}
			if test.wErr == nil && wantEvaluatedNodes != result.EvaluatedNodes {
				t.Errorf("Expected EvaluatedNodes: %d, got: %d", wantEvaluatedNodes, result.EvaluatedNodes)
			}
		})
	}
}

func TestFindFitAllError(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	nodes := makeNodeList([]string{"3", "2", "1"})
	scheduler := makeScheduler(ctx, nodes)

	fwk, err := st.NewFramework(
		ctx,
		[]st.RegisterPluginFunc{
			st.RegisterQueueSortPlugin(queuesort.Name, queuesort.New),
			st.RegisterFilterPlugin("TrueFilter", st.NewTrueFilterPlugin),
			st.RegisterFilterPlugin("MatchFilter", st.NewMatchFilterPlugin),
			st.RegisterBindPlugin(defaultbinder.Name, defaultbinder.New),
		},
		"",
		frameworkruntime.WithPodNominator(internalqueue.NewPodNominator(nil)),
	)
	if err != nil {
		t.Fatal(err)
	}

	_, diagnosis, err := scheduler.findNodesThatFitPod(ctx, fwk, framework.NewCycleState(), &v1.Pod{})
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	expected := framework.Diagnosis{
		NodeToStatusMap: framework.NodeToStatusMap{
			"1": framework.NewStatus(framework.Unschedulable, st.ErrReasonFake).WithFailedPlugin("MatchFilter"),
			"2": framework.NewStatus(framework.Unschedulable, st.ErrReasonFake).WithFailedPlugin("MatchFilter"),
			"3": framework.NewStatus(framework.Unschedulable, st.ErrReasonFake).WithFailedPlugin("MatchFilter"),
		},
		UnschedulablePlugins: sets.New("MatchFilter"),
	}
	if diff := cmp.Diff(diagnosis, expected); diff != "" {
		t.Errorf("Unexpected diagnosis: (-want, +got): %s", diff)
	}
}

func TestFindFitSomeError(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	nodes := makeNodeList([]string{"3", "2", "1"})
	scheduler := makeScheduler(ctx, nodes)

	fwk, err := st.NewFramework(
		ctx,
		[]st.RegisterPluginFunc{
			st.RegisterQueueSortPlugin(queuesort.Name, queuesort.New),
			st.RegisterFilterPlugin("TrueFilter", st.NewTrueFilterPlugin),
			st.RegisterFilterPlugin("MatchFilter", st.NewMatchFilterPlugin),
			st.RegisterBindPlugin(defaultbinder.Name, defaultbinder.New),
		},
		"",
		frameworkruntime.WithPodNominator(internalqueue.NewPodNominator(nil)),
	)
	if err != nil {
		t.Fatal(err)
	}

	pod := st.MakePod().Name("1").UID("1").Obj()
	_, diagnosis, err := scheduler.findNodesThatFitPod(ctx, fwk, framework.NewCycleState(), pod)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	if len(diagnosis.NodeToStatusMap) != len(nodes)-1 {
		t.Errorf("unexpected failed status map: %v", diagnosis.NodeToStatusMap)
	}

	if diff := cmp.Diff(sets.New("MatchFilter"), diagnosis.UnschedulablePlugins); diff != "" {
		t.Errorf("Unexpected unschedulablePlugins: (-want, +got): %s", diagnosis.UnschedulablePlugins)
	}

	for _, node := range nodes {
		if node.Name == pod.Name {
			continue
		}
		t.Run(node.Name, func(t *testing.T) {
			status, found := diagnosis.NodeToStatusMap[node.Name]
			if !found {
				t.Errorf("failed to find node %v in %v", node.Name, diagnosis.NodeToStatusMap)
			}
			reasons := status.Reasons()
			if len(reasons) != 1 || reasons[0] != st.ErrReasonFake {
				t.Errorf("unexpected failures: %v", reasons)
			}
		})
	}
}

func TestFindFitPredicateCallCounts(t *testing.T) {
	tests := []struct {
		name          string
		pod           *v1.Pod
		expectedCount int32
	}{
		{
			name:          "nominated pods have lower priority, predicate is called once",
			pod:           st.MakePod().Name("1").UID("1").Priority(highPriority).Obj(),
			expectedCount: 1,
		},
		{
			name:          "nominated pods have higher priority, predicate is called twice",
			pod:           st.MakePod().Name("1").UID("1").Priority(lowPriority).Obj(),
			expectedCount: 2,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			nodes := makeNodeList([]string{"1"})

			plugin := st.FakeFilterPlugin{}
			registerFakeFilterFunc := st.RegisterFilterPlugin(
				"FakeFilter",
				func(_ runtime.Object, fh framework.Handle) (framework.Plugin, error) {
					return &plugin, nil
				},
			)
			registerPlugins := []st.RegisterPluginFunc{
				st.RegisterQueueSortPlugin(queuesort.Name, queuesort.New),
				registerFakeFilterFunc,
				st.RegisterBindPlugin(defaultbinder.Name, defaultbinder.New),
			}
			logger, ctx := ktesting.NewTestContext(t)
			ctx, cancel := context.WithCancel(ctx)
			defer cancel()
			fwk, err := st.NewFramework(
				ctx,
				registerPlugins, "",
				frameworkruntime.WithPodNominator(internalqueue.NewPodNominator(nil)),
			)
			if err != nil {
				t.Fatal(err)
			}

			scheduler := makeScheduler(ctx, nodes)
			if err := scheduler.Cache.UpdateSnapshot(logger, scheduler.nodeInfoSnapshot); err != nil {
				t.Fatal(err)
			}
			podinfo, err := framework.NewPodInfo(st.MakePod().UID("nominated").Priority(midPriority).Obj())
			if err != nil {
				t.Fatal(err)
			}
			fwk.AddNominatedPod(logger, podinfo, &framework.NominatingInfo{NominatingMode: framework.ModeOverride, NominatedNodeName: "1"})

			_, _, err = scheduler.findNodesThatFitPod(ctx, fwk, framework.NewCycleState(), test.pod)
			if err != nil {
				t.Errorf("unexpected error: %v", err)
			}
			if test.expectedCount != plugin.NumFilterCalled {
				t.Errorf("predicate was called %d times, expected is %d", plugin.NumFilterCalled, test.expectedCount)
			}
		})
	}
}

// The point of this test is to show that you:
//   - get the same priority for a zero-request pod as for a pod with the defaults requests,
//     both when the zero-request pod is already on the node and when the zero-request pod
//     is the one being scheduled.
//   - don't get the same score no matter what we schedule.
func TestZeroRequest(t *testing.T) {
	// A pod with no resources. We expect spreading to count it as having the default resources.
	noResources := v1.PodSpec{
		Containers: []v1.Container{
			{},
		},
	}
	noResources1 := noResources
	noResources1.NodeName = "node1"
	// A pod with the same resources as a 0-request pod gets by default as its resources (for spreading).
	small := v1.PodSpec{
		Containers: []v1.Container{
			{
				Resources: v1.ResourceRequirements{
					Requests: v1.ResourceList{
						v1.ResourceCPU: resource.MustParse(
							strconv.FormatInt(schedutil.DefaultMilliCPURequest, 10) + "m"),
						v1.ResourceMemory: resource.MustParse(
							strconv.FormatInt(schedutil.DefaultMemoryRequest, 10)),
					},
				},
			},
		},
	}
	small2 := small
	small2.NodeName = "node2"
	// A larger pod.
	large := v1.PodSpec{
		Containers: []v1.Container{
			{
				Resources: v1.ResourceRequirements{
					Requests: v1.ResourceList{
						v1.ResourceCPU: resource.MustParse(
							strconv.FormatInt(schedutil.DefaultMilliCPURequest*3, 10) + "m"),
						v1.ResourceMemory: resource.MustParse(
							strconv.FormatInt(schedutil.DefaultMemoryRequest*3, 10)),
					},
				},
			},
		},
	}
	large1 := large
	large1.NodeName = "node1"
	large2 := large
	large2.NodeName = "node2"
	tests := []struct {
		pod           *v1.Pod
		pods          []*v1.Pod
		nodes         []*v1.Node
		name          string
		expectedScore int64
	}{
		// The point of these next two tests is to show you get the same priority for a zero-request pod
		// as for a pod with the defaults requests, both when the zero-request pod is already on the node
		// and when the zero-request pod is the one being scheduled.
		{
			pod:   &v1.Pod{Spec: noResources},
			nodes: []*v1.Node{makeNode("node1", 1000, schedutil.DefaultMemoryRequest*10), makeNode("node2", 1000, schedutil.DefaultMemoryRequest*10)},
			name:  "test priority of zero-request pod with node with zero-request pod",
			pods: []*v1.Pod{
				{Spec: large1}, {Spec: noResources1},
				{Spec: large2}, {Spec: small2},
			},
			expectedScore: 250,
		},
		{
			pod:   &v1.Pod{Spec: small},
			nodes: []*v1.Node{makeNode("node1", 1000, schedutil.DefaultMemoryRequest*10), makeNode("node2", 1000, schedutil.DefaultMemoryRequest*10)},
			name:  "test priority of nonzero-request pod with node with zero-request pod",
			pods: []*v1.Pod{
				{Spec: large1}, {Spec: noResources1},
				{Spec: large2}, {Spec: small2},
			},
			expectedScore: 250,
		},
		// The point of this test is to verify that we're not just getting the same score no matter what we schedule.
		{
			pod:   &v1.Pod{Spec: large},
			nodes: []*v1.Node{makeNode("node1", 1000, schedutil.DefaultMemoryRequest*10), makeNode("node2", 1000, schedutil.DefaultMemoryRequest*10)},
			name:  "test priority of larger pod with node with zero-request pod",
			pods: []*v1.Pod{
				{Spec: large1}, {Spec: noResources1},
				{Spec: large2}, {Spec: small2},
			},
			expectedScore: 230,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			client := clientsetfake.NewSimpleClientset()
			informerFactory := informers.NewSharedInformerFactory(client, 0)

			snapshot := internalcache.NewSnapshot(test.pods, test.nodes)
			fts := feature.Features{}
			pluginRegistrations := []st.RegisterPluginFunc{
				st.RegisterQueueSortPlugin(queuesort.Name, queuesort.New),
				st.RegisterScorePlugin(noderesources.Name, frameworkruntime.FactoryAdapter(fts, noderesources.NewFit), 1),
				st.RegisterScorePlugin(noderesources.BalancedAllocationName, frameworkruntime.FactoryAdapter(fts, noderesources.NewBalancedAllocation), 1),
				st.RegisterScorePlugin(selectorspread.Name, selectorspread.New, 1),
				st.RegisterPreScorePlugin(selectorspread.Name, selectorspread.New),
				st.RegisterBindPlugin(defaultbinder.Name, defaultbinder.New),
			}
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			fwk, err := st.NewFramework(
				ctx,
				pluginRegistrations, "",
				frameworkruntime.WithInformerFactory(informerFactory),
				frameworkruntime.WithSnapshotSharedLister(snapshot),
				frameworkruntime.WithClientSet(client),
				frameworkruntime.WithPodNominator(internalqueue.NewPodNominator(informerFactory.Core().V1().Pods().Lister())),
			)
			if err != nil {
				t.Fatalf("error creating framework: %+v", err)
			}

			sched := &Scheduler{
				nodeInfoSnapshot:         snapshot,
				percentageOfNodesToScore: schedulerapi.DefaultPercentageOfNodesToScore,
			}
			sched.applyDefaultHandlers()

			state := framework.NewCycleState()
			_, _, err = sched.findNodesThatFitPod(ctx, fwk, state, test.pod)
			if err != nil {
				t.Fatalf("error filtering nodes: %+v", err)
			}
			fwk.RunPreScorePlugins(ctx, state, test.pod, test.nodes)
			list, err := prioritizeNodes(ctx, nil, fwk, state, test.pod, test.nodes)
			if err != nil {
				t.Errorf("unexpected error: %v", err)
			}
			for _, hp := range list {
				if hp.TotalScore != test.expectedScore {
					t.Errorf("expected %d for all priorities, got list %#v", test.expectedScore, list)
				}
			}
		})
	}
}

func Test_prioritizeNodes(t *testing.T) {
	tests := []struct {
		name                string
		pod                 *v1.Pod
		pods                []*v1.Pod
		nodes               []*v1.Node
		pluginRegistrations []st.RegisterPluginFunc
		extenders           []st.FakeExtender
		want                []framework.NodePluginScores
	}{
		{
			name:  "the score from all plugins should be recorded in PluginToNodeScores",
			pod:   &v1.Pod{},
			nodes: []*v1.Node{makeNode("node1", 1000, schedutil.DefaultMemoryRequest*10), makeNode("node2", 1000, schedutil.DefaultMemoryRequest*10)},
			pluginRegistrations: []st.RegisterPluginFunc{
				st.RegisterQueueSortPlugin(queuesort.Name, queuesort.New),
				st.RegisterScorePlugin(noderesources.BalancedAllocationName, frameworkruntime.FactoryAdapter(feature.Features{}, noderesources.NewBalancedAllocation), 1),
				st.RegisterScorePlugin("Node2Prioritizer", st.NewNode2PrioritizerPlugin(), 1),
				st.RegisterBindPlugin(defaultbinder.Name, defaultbinder.New),
			},
			extenders: nil,
			want: []framework.NodePluginScores{
				{
					Name: "node1",
					Scores: []framework.PluginScore{
						{
							Name:  "Node2Prioritizer",
							Score: 10,
						},
						{
							Name:  "NodeResourcesBalancedAllocation",
							Score: 100,
						},
					},
					TotalScore: 110,
				},
				{
					Name: "node2",
					Scores: []framework.PluginScore{
						{
							Name:  "Node2Prioritizer",
							Score: 100,
						},
						{
							Name:  "NodeResourcesBalancedAllocation",
							Score: 100,
						},
					},
					TotalScore: 200,
				},
			},
		},
		{
			name:  "the score from extender should also be recorded in PluginToNodeScores with plugin scores",
			pod:   &v1.Pod{},
			nodes: []*v1.Node{makeNode("node1", 1000, schedutil.DefaultMemoryRequest*10), makeNode("node2", 1000, schedutil.DefaultMemoryRequest*10)},
			pluginRegistrations: []st.RegisterPluginFunc{
				st.RegisterQueueSortPlugin(queuesort.Name, queuesort.New),
				st.RegisterScorePlugin(noderesources.BalancedAllocationName, frameworkruntime.FactoryAdapter(feature.Features{}, noderesources.NewBalancedAllocation), 1),
				st.RegisterBindPlugin(defaultbinder.Name, defaultbinder.New),
			},
			extenders: []st.FakeExtender{
				{
					ExtenderName: "FakeExtender1",
					Weight:       1,
					Prioritizers: []st.PriorityConfig{
						{
							Weight:   3,
							Function: st.Node1PrioritizerExtender,
						},
					},
				},
				{
					ExtenderName: "FakeExtender2",
					Weight:       1,
					Prioritizers: []st.PriorityConfig{
						{
							Weight:   2,
							Function: st.Node2PrioritizerExtender,
						},
					},
				},
			},
			want: []framework.NodePluginScores{
				{
					Name: "node1",
					Scores: []framework.PluginScore{

						{
							Name:  "FakeExtender1",
							Score: 300,
						},
						{
							Name:  "FakeExtender2",
							Score: 20,
						},
						{
							Name:  "NodeResourcesBalancedAllocation",
							Score: 100,
						},
					},
					TotalScore: 420,
				},
				{
					Name: "node2",
					Scores: []framework.PluginScore{
						{
							Name:  "FakeExtender1",
							Score: 30,
						},
						{
							Name:  "FakeExtender2",
							Score: 200,
						},
						{
							Name:  "NodeResourcesBalancedAllocation",
							Score: 100,
						},
					},
					TotalScore: 330,
				},
			},
		},
		{
			name:  "plugin which returned skip in preScore shouldn't be executed in the score phase",
			pod:   &v1.Pod{},
			nodes: []*v1.Node{makeNode("node1", 1000, schedutil.DefaultMemoryRequest*10), makeNode("node2", 1000, schedutil.DefaultMemoryRequest*10)},
			pluginRegistrations: []st.RegisterPluginFunc{
				st.RegisterQueueSortPlugin(queuesort.Name, queuesort.New),
				st.RegisterScorePlugin(noderesources.BalancedAllocationName, frameworkruntime.FactoryAdapter(feature.Features{}, noderesources.NewBalancedAllocation), 1),
				st.RegisterScorePlugin("Node2Prioritizer", st.NewNode2PrioritizerPlugin(), 1),
				st.RegisterBindPlugin(defaultbinder.Name, defaultbinder.New),
				st.RegisterPluginAsExtensions("FakePreScoreAndScorePlugin", st.NewFakePreScoreAndScorePlugin("FakePreScoreAndScorePlugin", 0,
					framework.NewStatus(framework.Skip, "fake skip"),
					framework.NewStatus(framework.Error, "this score function shouldn't be executed because this plugin returned Skip in the PreScore"),
				), "PreScore", "Score"),
			},
			extenders: nil,
			want: []framework.NodePluginScores{
				{
					Name: "node1",
					Scores: []framework.PluginScore{
						{
							Name:  "Node2Prioritizer",
							Score: 10,
						},
						{
							Name:  "NodeResourcesBalancedAllocation",
							Score: 100,
						},
					},
					TotalScore: 110,
				},
				{
					Name: "node2",
					Scores: []framework.PluginScore{
						{
							Name:  "Node2Prioritizer",
							Score: 100,
						},
						{
							Name:  "NodeResourcesBalancedAllocation",
							Score: 100,
						},
					},
					TotalScore: 200,
				},
			},
		},
		{
			name:  "all score plugins are skipped",
			pod:   &v1.Pod{},
			nodes: []*v1.Node{makeNode("node1", 1000, schedutil.DefaultMemoryRequest*10), makeNode("node2", 1000, schedutil.DefaultMemoryRequest*10)},
			pluginRegistrations: []st.RegisterPluginFunc{
				st.RegisterQueueSortPlugin(queuesort.Name, queuesort.New),
				st.RegisterBindPlugin(defaultbinder.Name, defaultbinder.New),
				st.RegisterPluginAsExtensions("FakePreScoreAndScorePlugin", st.NewFakePreScoreAndScorePlugin("FakePreScoreAndScorePlugin", 0,
					framework.NewStatus(framework.Skip, "fake skip"),
					framework.NewStatus(framework.Error, "this score function shouldn't be executed because this plugin returned Skip in the PreScore"),
				), "PreScore", "Score"),
			},
			extenders: nil,
			want: []framework.NodePluginScores{
				{Name: "node1", Scores: []framework.PluginScore{}},
				{Name: "node2", Scores: []framework.PluginScore{}},
			},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			client := clientsetfake.NewSimpleClientset()
			informerFactory := informers.NewSharedInformerFactory(client, 0)

			snapshot := internalcache.NewSnapshot(test.pods, test.nodes)
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			fwk, err := st.NewFramework(
				ctx,
				test.pluginRegistrations, "",
				frameworkruntime.WithInformerFactory(informerFactory),
				frameworkruntime.WithSnapshotSharedLister(snapshot),
				frameworkruntime.WithClientSet(client),
				frameworkruntime.WithPodNominator(internalqueue.NewPodNominator(informerFactory.Core().V1().Pods().Lister())),
			)
			if err != nil {
				t.Fatalf("error creating framework: %+v", err)
			}

			state := framework.NewCycleState()
			var extenders []framework.Extender
			for ii := range test.extenders {
				extenders = append(extenders, &test.extenders[ii])
			}
			nodesscores, err := prioritizeNodes(ctx, extenders, fwk, state, test.pod, test.nodes)
			if err != nil {
				t.Errorf("unexpected error: %v", err)
			}
			for i := range nodesscores {
				sort.Slice(nodesscores[i].Scores, func(j, k int) bool {
					return nodesscores[i].Scores[j].Name < nodesscores[i].Scores[k].Name
				})
			}

			if diff := cmp.Diff(test.want, nodesscores); diff != "" {
				t.Errorf("returned nodes scores (-want,+got):\n%s", diff)
			}
		})
	}
}

var lowPriority, midPriority, highPriority = int32(0), int32(100), int32(1000)

func TestNumFeasibleNodesToFind(t *testing.T) {
	tests := []struct {
		name              string
		globalPercentage  int32
		profilePercentage *int32
		numAllNodes       int32
		wantNumNodes      int32
	}{
		{
			name:         "not set percentageOfNodesToScore and nodes number not more than 50",
			numAllNodes:  10,
			wantNumNodes: 10,
		},
		{
			name:              "set profile percentageOfNodesToScore and nodes number not more than 50",
			profilePercentage: pointer.Int32(40),
			numAllNodes:       10,
			wantNumNodes:      10,
		},
		{
			name:         "not set percentageOfNodesToScore and nodes number more than 50",
			numAllNodes:  1000,
			wantNumNodes: 420,
		},
		{
			name:              "set profile percentageOfNodesToScore and nodes number more than 50",
			profilePercentage: pointer.Int32(40),
			numAllNodes:       1000,
			wantNumNodes:      400,
		},
		{
			name:              "set global and profile percentageOfNodesToScore and nodes number more than 50",
			globalPercentage:  100,
			profilePercentage: pointer.Int32(40),
			numAllNodes:       1000,
			wantNumNodes:      400,
		},
		{
			name:             "set global percentageOfNodesToScore and nodes number more than 50",
			globalPercentage: 40,
			numAllNodes:      1000,
			wantNumNodes:     400,
		},
		{
			name:         "not set profile percentageOfNodesToScore and nodes number more than 50*125",
			numAllNodes:  6000,
			wantNumNodes: 300,
		},
		{
			name:              "set profile percentageOfNodesToScore and nodes number more than 50*125",
			profilePercentage: pointer.Int32(40),
			numAllNodes:       6000,
			wantNumNodes:      2400,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			sched := &Scheduler{
				percentageOfNodesToScore: tt.globalPercentage,
			}
			if gotNumNodes := sched.numFeasibleNodesToFind(tt.profilePercentage, tt.numAllNodes); gotNumNodes != tt.wantNumNodes {
				t.Errorf("Scheduler.numFeasibleNodesToFind() = %v, want %v", gotNumNodes, tt.wantNumNodes)
			}
		})
	}
}

func TestFairEvaluationForNodes(t *testing.T) {
	numAllNodes := 500
	nodeNames := make([]string, 0, numAllNodes)
	for i := 0; i < numAllNodes; i++ {
		nodeNames = append(nodeNames, strconv.Itoa(i))
	}
	nodes := makeNodeList(nodeNames)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	sched := makeScheduler(ctx, nodes)

	fwk, err := st.NewFramework(
		ctx,
		[]st.RegisterPluginFunc{
			st.RegisterQueueSortPlugin(queuesort.Name, queuesort.New),
			st.RegisterFilterPlugin("TrueFilter", st.NewTrueFilterPlugin),
			st.RegisterBindPlugin(defaultbinder.Name, defaultbinder.New),
		},
		"",
		frameworkruntime.WithPodNominator(internalqueue.NewPodNominator(nil)),
	)
	if err != nil {
		t.Fatal(err)
	}

	// To make numAllNodes % nodesToFind != 0
	sched.percentageOfNodesToScore = 30
	nodesToFind := int(sched.numFeasibleNodesToFind(fwk.PercentageOfNodesToScore(), int32(numAllNodes)))

	// Iterating over all nodes more than twice
	for i := 0; i < 2*(numAllNodes/nodesToFind+1); i++ {
		nodesThatFit, _, err := sched.findNodesThatFitPod(ctx, fwk, framework.NewCycleState(), &v1.Pod{})
		if err != nil {
			t.Errorf("unexpected error: %v", err)
		}
		if len(nodesThatFit) != nodesToFind {
			t.Errorf("got %d nodes filtered, want %d", len(nodesThatFit), nodesToFind)
		}
		if sched.nextStartNodeIndex != (i+1)*nodesToFind%numAllNodes {
			t.Errorf("got %d lastProcessedNodeIndex, want %d", sched.nextStartNodeIndex, (i+1)*nodesToFind%numAllNodes)
		}
	}
}

func TestPreferNominatedNodeFilterCallCounts(t *testing.T) {
	tests := []struct {
		name                  string
		pod                   *v1.Pod
		nodeReturnCodeMap     map[string]framework.Code
		expectedCount         int32
		expectedPatchRequests int
	}{
		{
			name:          "pod has the nominated node set, filter is called only once",
			pod:           st.MakePod().Name("p_with_nominated_node").UID("p").Priority(highPriority).NominatedNodeName("node1").Obj(),
			expectedCount: 1,
		},
		{
			name:          "pod without the nominated pod, filter is called for each node",
			pod:           st.MakePod().Name("p_without_nominated_node").UID("p").Priority(highPriority).Obj(),
			expectedCount: 3,
		},
		{
			name:              "nominated pod cannot pass the filter, filter is called for each node",
			pod:               st.MakePod().Name("p_with_nominated_node").UID("p").Priority(highPriority).NominatedNodeName("node1").Obj(),
			nodeReturnCodeMap: map[string]framework.Code{"node1": framework.Unschedulable},
			expectedCount:     4,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			logger, ctx := ktesting.NewTestContext(t)
			ctx, cancel := context.WithCancel(ctx)
			defer cancel()

			// create three nodes in the cluster.
			nodes := makeNodeList([]string{"node1", "node2", "node3"})
			client := clientsetfake.NewSimpleClientset(test.pod)
			informerFactory := informers.NewSharedInformerFactory(client, 0)
			cache := internalcache.New(ctx, time.Duration(0))
			for _, n := range nodes {
				cache.AddNode(logger, n)
			}
			plugin := st.FakeFilterPlugin{FailedNodeReturnCodeMap: test.nodeReturnCodeMap}
			registerFakeFilterFunc := st.RegisterFilterPlugin(
				"FakeFilter",
				func(_ runtime.Object, fh framework.Handle) (framework.Plugin, error) {
					return &plugin, nil
				},
			)
			registerPlugins := []st.RegisterPluginFunc{
				st.RegisterQueueSortPlugin(queuesort.Name, queuesort.New),
				registerFakeFilterFunc,
				st.RegisterBindPlugin(defaultbinder.Name, defaultbinder.New),
			}
			fwk, err := st.NewFramework(
				ctx,
				registerPlugins, "",
				frameworkruntime.WithClientSet(client),
				frameworkruntime.WithPodNominator(internalqueue.NewPodNominator(informerFactory.Core().V1().Pods().Lister())),
			)
			if err != nil {
				t.Fatal(err)
			}
			snapshot := internalcache.NewSnapshot(nil, nodes)

			sched := &Scheduler{
				Cache:                    cache,
				nodeInfoSnapshot:         snapshot,
				percentageOfNodesToScore: schedulerapi.DefaultPercentageOfNodesToScore,
			}
			sched.applyDefaultHandlers()

			_, _, err = sched.findNodesThatFitPod(ctx, fwk, framework.NewCycleState(), test.pod)
			if err != nil {
				t.Errorf("unexpected error: %v", err)
			}
			if test.expectedCount != plugin.NumFilterCalled {
				t.Errorf("predicate was called %d times, expected is %d", plugin.NumFilterCalled, test.expectedCount)
			}
		})
	}
}

func podWithID(id, desiredHost string) *v1.Pod {
	return st.MakePod().Name(id).UID(id).Node(desiredHost).SchedulerName(testSchedulerName).Obj()
}

func deletingPod(id string) *v1.Pod {
	return st.MakePod().Name(id).UID(id).Terminating().Node("").SchedulerName(testSchedulerName).Obj()
}

func podWithPort(id, desiredHost string, port int) *v1.Pod {
	pod := podWithID(id, desiredHost)
	pod.Spec.Containers = []v1.Container{
		{Name: "ctr", Ports: []v1.ContainerPort{{HostPort: int32(port)}}},
	}
	return pod
}

func podWithResources(id, desiredHost string, limits v1.ResourceList, requests v1.ResourceList) *v1.Pod {
	pod := podWithID(id, desiredHost)
	pod.Spec.Containers = []v1.Container{
		{Name: "ctr", Resources: v1.ResourceRequirements{Limits: limits, Requests: requests}},
	}
	return pod
}

func makeNodeList(nodeNames []string) []*v1.Node {
	result := make([]*v1.Node, 0, len(nodeNames))
	for _, nodeName := range nodeNames {
		result = append(result, &v1.Node{ObjectMeta: metav1.ObjectMeta{Name: nodeName}})
	}
	return result
}

// makeScheduler makes a simple Scheduler for testing.
func makeScheduler(ctx context.Context, nodes []*v1.Node) *Scheduler {
	logger := klog.FromContext(ctx)
	cache := internalcache.New(ctx, time.Duration(0))
	for _, n := range nodes {
		cache.AddNode(logger, n)
	}

	sched := &Scheduler{
		Cache:                    cache,
		nodeInfoSnapshot:         emptySnapshot,
		percentageOfNodesToScore: schedulerapi.DefaultPercentageOfNodesToScore,
	}
	sched.applyDefaultHandlers()
	cache.UpdateSnapshot(logger, sched.nodeInfoSnapshot)
	return sched
}

func makeNode(node string, milliCPU, memory int64) *v1.Node {
	return &v1.Node{
		ObjectMeta: metav1.ObjectMeta{Name: node},
		Status: v1.NodeStatus{
			Capacity: v1.ResourceList{
				v1.ResourceCPU:    *resource.NewMilliQuantity(milliCPU, resource.DecimalSI),
				v1.ResourceMemory: *resource.NewQuantity(memory, resource.BinarySI),
				"pods":            *resource.NewQuantity(100, resource.DecimalSI),
			},
			Allocatable: v1.ResourceList{

				v1.ResourceCPU:    *resource.NewMilliQuantity(milliCPU, resource.DecimalSI),
				v1.ResourceMemory: *resource.NewQuantity(memory, resource.BinarySI),
				"pods":            *resource.NewQuantity(100, resource.DecimalSI),
			},
		},
	}
}

// queuedPodStore: pods queued before processing.
// cache: scheduler cache that might contain assumed pods.
func setupTestSchedulerWithOnePodOnNode(ctx context.Context, t *testing.T, queuedPodStore *clientcache.FIFO, scache internalcache.Cache,
	pod *v1.Pod, node *v1.Node, fns ...st.RegisterPluginFunc) (*Scheduler, chan *v1.Binding, chan error) {
	scheduler, bindingChan, errChan := setupTestScheduler(ctx, t, queuedPodStore, scache, nil, nil, fns...)

	queuedPodStore.Add(pod)
	// queuedPodStore: [foo:8080]
	// cache: []

	scheduler.scheduleOne(ctx)
	// queuedPodStore: []
	// cache: [(assumed)foo:8080]

	select {
	case b := <-bindingChan:
		expectBinding := &v1.Binding{
			ObjectMeta: metav1.ObjectMeta{Name: pod.Name, UID: types.UID(pod.Name)},
			Target:     v1.ObjectReference{Kind: "Node", Name: node.Name},
		}
		if !reflect.DeepEqual(expectBinding, b) {
			t.Errorf("binding want=%v, get=%v", expectBinding, b)
		}
	case <-time.After(wait.ForeverTestTimeout):
		t.Fatalf("timeout after %v", wait.ForeverTestTimeout)
	}
	return scheduler, bindingChan, errChan
}

// queuedPodStore: pods queued before processing.
// scache: scheduler cache that might contain assumed pods.
func setupTestScheduler(ctx context.Context, t *testing.T, queuedPodStore *clientcache.FIFO, cache internalcache.Cache, informerFactory informers.SharedInformerFactory, broadcaster events.EventBroadcaster, fns ...st.RegisterPluginFunc) (*Scheduler, chan *v1.Binding, chan error) {
	bindingChan := make(chan *v1.Binding, 1)
	client := clientsetfake.NewSimpleClientset()
	client.PrependReactor("create", "pods", func(action clienttesting.Action) (bool, runtime.Object, error) {
		var b *v1.Binding
		if action.GetSubresource() == "binding" {
			b := action.(clienttesting.CreateAction).GetObject().(*v1.Binding)
			bindingChan <- b
		}
		return true, b, nil
	})

	var recorder events.EventRecorder
	if broadcaster != nil {
		recorder = broadcaster.NewRecorder(scheme.Scheme, testSchedulerName)
	} else {
		recorder = &events.FakeRecorder{}
	}

	if informerFactory == nil {
		informerFactory = informers.NewSharedInformerFactory(clientsetfake.NewSimpleClientset(), 0)
	}
	schedulingQueue := internalqueue.NewTestQueueWithInformerFactory(ctx, nil, informerFactory)

	fwk, _ := st.NewFramework(
		ctx,
		fns,
		testSchedulerName,
		frameworkruntime.WithClientSet(client),
		frameworkruntime.WithEventRecorder(recorder),
		frameworkruntime.WithInformerFactory(informerFactory),
		frameworkruntime.WithPodNominator(internalqueue.NewPodNominator(informerFactory.Core().V1().Pods().Lister())),
	)

	errChan := make(chan error, 1)
	sched := &Scheduler{
		Cache:                    cache,
		client:                   client,
		nodeInfoSnapshot:         internalcache.NewEmptySnapshot(),
		percentageOfNodesToScore: schedulerapi.DefaultPercentageOfNodesToScore,
		NextPod: func() (*framework.QueuedPodInfo, error) {
			return &framework.QueuedPodInfo{PodInfo: mustNewPodInfo(t, clientcache.Pop(queuedPodStore).(*v1.Pod))}, nil
		},
		SchedulingQueue: schedulingQueue,
		Profiles:        profile.Map{testSchedulerName: fwk},
	}

	sched.SchedulePod = sched.schedulePod
	sched.FailureHandler = func(_ context.Context, _ framework.Framework, p *framework.QueuedPodInfo, status *framework.Status, _ *framework.NominatingInfo, _ time.Time) {
		err := status.AsError()
		errChan <- err

		msg := truncateMessage(err.Error())
		fwk.EventRecorder().Eventf(p.Pod, nil, v1.EventTypeWarning, "FailedScheduling", "Scheduling", msg)
	}
	return sched, bindingChan, errChan
}

func setupTestSchedulerWithVolumeBinding(ctx context.Context, t *testing.T, volumeBinder volumebinding.SchedulerVolumeBinder, broadcaster events.EventBroadcaster) (*Scheduler, chan *v1.Binding, chan error) {
	logger := klog.FromContext(ctx)
	testNode := v1.Node{ObjectMeta: metav1.ObjectMeta{Name: "node1", UID: types.UID("node1")}}
	queuedPodStore := clientcache.NewFIFO(clientcache.MetaNamespaceKeyFunc)
	pod := podWithID("foo", "")
	pod.Namespace = "foo-ns"
	pod.Spec.Volumes = append(pod.Spec.Volumes, v1.Volume{Name: "testVol",
		VolumeSource: v1.VolumeSource{PersistentVolumeClaim: &v1.PersistentVolumeClaimVolumeSource{ClaimName: "testPVC"}}})
	queuedPodStore.Add(pod)
	scache := internalcache.New(ctx, 10*time.Minute)
	scache.AddNode(logger, &testNode)
	testPVC := v1.PersistentVolumeClaim{ObjectMeta: metav1.ObjectMeta{Name: "testPVC", Namespace: pod.Namespace, UID: types.UID("testPVC")}}
	client := clientsetfake.NewSimpleClientset(&testNode, &testPVC)
	informerFactory := informers.NewSharedInformerFactory(client, 0)
	pvcInformer := informerFactory.Core().V1().PersistentVolumeClaims()
	pvcInformer.Informer().GetStore().Add(&testPVC)

	fns := []st.RegisterPluginFunc{
		st.RegisterQueueSortPlugin(queuesort.Name, queuesort.New),
		st.RegisterBindPlugin(defaultbinder.Name, defaultbinder.New),
		st.RegisterPluginAsExtensions(volumebinding.Name, func(plArgs runtime.Object, handle framework.Handle) (framework.Plugin, error) {
			return &volumebinding.VolumeBinding{Binder: volumeBinder, PVCLister: pvcInformer.Lister()}, nil
		}, "PreFilter", "Filter", "Reserve", "PreBind"),
	}
	s, bindingChan, errChan := setupTestScheduler(ctx, t, queuedPodStore, scache, informerFactory, broadcaster, fns...)
	return s, bindingChan, errChan
}

// This is a workaround because golint complains that errors cannot
// end with punctuation.  However, the real predicate error message does
// end with a period.
func makePredicateError(failReason string) error {
	s := fmt.Sprintf("0/1 nodes are available: %v.", failReason)
	return fmt.Errorf(s)
}

func mustNewPodInfo(t *testing.T, pod *v1.Pod) *framework.PodInfo {
	podInfo, err := framework.NewPodInfo(pod)
	if err != nil {
		t.Fatal(err)
	}
	return podInfo
}
