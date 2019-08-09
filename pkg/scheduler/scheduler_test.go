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
	"errors"
	"fmt"
	"io/ioutil"
	"os"
	"path"
	"reflect"
	"testing"
	"time"

	v1 "k8s.io/api/core/v1"
	"k8s.io/api/events/v1beta1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/diff"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/client-go/informers"
	clientsetfake "k8s.io/client-go/kubernetes/fake"
	"k8s.io/client-go/kubernetes/scheme"
	clientcache "k8s.io/client-go/tools/cache"
	"k8s.io/client-go/tools/events"
	volumescheduling "k8s.io/kubernetes/pkg/controller/volume/scheduling"
	"k8s.io/kubernetes/pkg/scheduler/algorithm"
	"k8s.io/kubernetes/pkg/scheduler/algorithm/predicates"
	"k8s.io/kubernetes/pkg/scheduler/algorithm/priorities"
	schedulerapi "k8s.io/kubernetes/pkg/scheduler/api"
	kubeschedulerconfig "k8s.io/kubernetes/pkg/scheduler/apis/config"
	"k8s.io/kubernetes/pkg/scheduler/core"
	"k8s.io/kubernetes/pkg/scheduler/factory"
	framework "k8s.io/kubernetes/pkg/scheduler/framework/v1alpha1"
	internalcache "k8s.io/kubernetes/pkg/scheduler/internal/cache"
	fakecache "k8s.io/kubernetes/pkg/scheduler/internal/cache/fake"
	internalqueue "k8s.io/kubernetes/pkg/scheduler/internal/queue"
	schedulernodeinfo "k8s.io/kubernetes/pkg/scheduler/nodeinfo"
	"k8s.io/kubernetes/pkg/scheduler/volumebinder"
)

var (
	emptyPluginRegistry = framework.Registry{}
	// emptyPluginConfig is an empty plugin config used in tests.
	emptyPluginConfig []kubeschedulerconfig.PluginConfig
	// emptyFramework is an empty framework used in tests.
	// Note: If the test runs in goroutine, please don't use this variable to avoid a race condition.
	emptyFramework, _ = framework.NewFramework(emptyPluginRegistry, nil, emptyPluginConfig)
)

type fakeBinder struct {
	b func(binding *v1.Binding) error
}

func (fb fakeBinder) Bind(binding *v1.Binding) error { return fb.b(binding) }

type fakePodConditionUpdater struct{}

func (fc fakePodConditionUpdater) Update(pod *v1.Pod, podCondition *v1.PodCondition) error {
	return nil
}

type fakePodPreemptor struct{}

func (fp fakePodPreemptor) GetUpdatedPod(pod *v1.Pod) (*v1.Pod, error) {
	return pod, nil
}

func (fp fakePodPreemptor) DeletePod(pod *v1.Pod) error {
	return nil
}

func (fp fakePodPreemptor) SetNominatedNodeName(pod *v1.Pod, nomNodeName string) error {
	return nil
}

func (fp fakePodPreemptor) RemoveNominatedNodeName(pod *v1.Pod) error {
	return nil
}

func podWithID(id, desiredHost string) *v1.Pod {
	return &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:     id,
			UID:      types.UID(id),
			SelfLink: fmt.Sprintf("/api/v1/%s/%s", string(v1.ResourcePods), id),
		},
		Spec: v1.PodSpec{
			NodeName: desiredHost,
		},
	}
}

func deletingPod(id string) *v1.Pod {
	deletionTimestamp := metav1.Now()
	return &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:              id,
			UID:               types.UID(id),
			DeletionTimestamp: &deletionTimestamp,
			SelfLink:          fmt.Sprintf("/api/v1/%s/%s", string(v1.ResourcePods), id),
		},
		Spec: v1.PodSpec{
			NodeName: "",
		},
	}
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

func PredicateOne(pod *v1.Pod, meta predicates.PredicateMetadata, nodeInfo *schedulernodeinfo.NodeInfo) (bool, []predicates.PredicateFailureReason, error) {
	return true, nil, nil
}

func PriorityOne(pod *v1.Pod, nodeNameToInfo map[string]*schedulernodeinfo.NodeInfo, nodes []*v1.Node) (schedulerapi.HostPriorityList, error) {
	return []schedulerapi.HostPriority{}, nil
}

type mockScheduler struct {
	result core.ScheduleResult
	err    error
}

func (es mockScheduler) Schedule(pod *v1.Pod, pc *framework.PluginContext) (core.ScheduleResult, error) {
	return es.result, es.err
}

func (es mockScheduler) Predicates() map[string]predicates.FitPredicate {
	return nil
}
func (es mockScheduler) Prioritizers() []priorities.PriorityConfig {
	return nil
}

func (es mockScheduler) Preempt(pod *v1.Pod, scheduleErr error) (*v1.Node, []*v1.Pod, []*v1.Pod, error) {
	return nil, nil, nil, nil
}

func TestSchedulerCreation(t *testing.T) {
	client := clientsetfake.NewSimpleClientset()
	informerFactory := informers.NewSharedInformerFactory(client, 0)

	testSource := "testProvider"
	eventBroadcaster := events.NewBroadcaster(&events.EventSinkImpl{Interface: client.EventsV1beta1().Events("")})

	defaultBindTimeout := int64(30)
	factory.RegisterFitPredicate("PredicateOne", PredicateOne)
	factory.RegisterPriorityFunction("PriorityOne", PriorityOne, 1)
	factory.RegisterAlgorithmProvider(testSource, sets.NewString("PredicateOne"), sets.NewString("PriorityOne"))

	stopCh := make(chan struct{})
	defer close(stopCh)
	_, err := New(client,
		informerFactory.Core().V1().Nodes(),
		factory.NewPodInformer(client, 0),
		informerFactory.Core().V1().PersistentVolumes(),
		informerFactory.Core().V1().PersistentVolumeClaims(),
		informerFactory.Core().V1().ReplicationControllers(),
		informerFactory.Apps().V1().ReplicaSets(),
		informerFactory.Apps().V1().StatefulSets(),
		informerFactory.Core().V1().Services(),
		informerFactory.Policy().V1beta1().PodDisruptionBudgets(),
		informerFactory.Storage().V1().StorageClasses(),
		informerFactory.Storage().V1beta1().CSINodes(),
		eventBroadcaster.NewRecorder(scheme.Scheme, "scheduler"),
		kubeschedulerconfig.SchedulerAlgorithmSource{Provider: &testSource},
		stopCh,
		emptyPluginRegistry,
		nil,
		emptyPluginConfig,
		WithBindTimeoutSeconds(defaultBindTimeout))

	if err != nil {
		t.Fatalf("Failed to create scheduler: %v", err)
	}
}

func TestScheduler(t *testing.T) {
	testNode := v1.Node{ObjectMeta: metav1.ObjectMeta{Name: "machine1", UID: types.UID("machine1")}}
	client := clientsetfake.NewSimpleClientset(&testNode)
	eventBroadcaster := events.NewBroadcaster(&events.EventSinkImpl{Interface: client.EventsV1beta1().Events("")})
	errS := errors.New("scheduler")
	errB := errors.New("binder")

	table := []struct {
		name             string
		injectBindError  error
		sendPod          *v1.Pod
		algo             core.ScheduleAlgorithm
		expectErrorPod   *v1.Pod
		expectForgetPod  *v1.Pod
		expectAssumedPod *v1.Pod
		expectError      error
		expectBind       *v1.Binding
		eventReason      string
	}{
		{
			name:             "bind assumed pod scheduled",
			sendPod:          podWithID("foo", ""),
			algo:             mockScheduler{core.ScheduleResult{SuggestedHost: testNode.Name, EvaluatedNodes: 1, FeasibleNodes: 1}, nil},
			expectBind:       &v1.Binding{ObjectMeta: metav1.ObjectMeta{Name: "foo", UID: types.UID("foo")}, Target: v1.ObjectReference{Kind: "Node", Name: testNode.Name}},
			expectAssumedPod: podWithID("foo", testNode.Name),
			eventReason:      "Scheduled",
		},
		{
			name:           "error pod failed scheduling",
			sendPod:        podWithID("foo", ""),
			algo:           mockScheduler{core.ScheduleResult{SuggestedHost: testNode.Name, EvaluatedNodes: 1, FeasibleNodes: 1}, errS},
			expectError:    errS,
			expectErrorPod: podWithID("foo", ""),
			eventReason:    "FailedScheduling",
		},
		{
			name:             "error bind forget pod failed scheduling",
			sendPod:          podWithID("foo", ""),
			algo:             mockScheduler{core.ScheduleResult{SuggestedHost: testNode.Name, EvaluatedNodes: 1, FeasibleNodes: 1}, nil},
			expectBind:       &v1.Binding{ObjectMeta: metav1.ObjectMeta{Name: "foo", UID: types.UID("foo")}, Target: v1.ObjectReference{Kind: "Node", Name: testNode.Name}},
			expectAssumedPod: podWithID("foo", testNode.Name),
			injectBindError:  errB,
			expectError:      errB,
			expectErrorPod:   podWithID("foo", testNode.Name),
			expectForgetPod:  podWithID("foo", testNode.Name),
			eventReason:      "FailedScheduling",
		}, {
			sendPod:     deletingPod("foo"),
			algo:        mockScheduler{core.ScheduleResult{}, nil},
			eventReason: "FailedScheduling",
		},
	}

	stop := make(chan struct{})
	defer close(stop)
	informerFactory := informers.NewSharedInformerFactory(client, 0)

	informerFactory.Start(stop)
	informerFactory.WaitForCacheSync(stop)

	for _, item := range table {
		t.Run(item.name, func(t *testing.T) {
			var gotError error
			var gotPod *v1.Pod
			var gotForgetPod *v1.Pod
			var gotAssumedPod *v1.Pod
			var gotBinding *v1.Binding
			sCache := &fakecache.Cache{
				ForgetFunc: func(pod *v1.Pod) {
					gotForgetPod = pod
				},
				AssumeFunc: func(pod *v1.Pod) {
					gotAssumedPod = pod
				},
			}

			s := NewFromConfig(&factory.Config{
				SchedulerCache: sCache,
				Algorithm:      item.algo,
				GetBinder: func(pod *v1.Pod) factory.Binder {
					return fakeBinder{func(b *v1.Binding) error {
						gotBinding = b
						return item.injectBindError
					}}
				},
				PodConditionUpdater: fakePodConditionUpdater{},
				Error: func(p *v1.Pod, err error) {
					gotPod = p
					gotError = err
				},
				NextPod: func() *v1.Pod {
					return item.sendPod
				},
				Framework:    emptyFramework,
				Recorder:     eventBroadcaster.NewRecorder(scheme.Scheme, "scheduler"),
				VolumeBinder: volumebinder.NewFakeVolumeBinder(&volumescheduling.FakeVolumeBinderConfig{AllBound: true}),
			})
			called := make(chan struct{})
			stopFunc := eventBroadcaster.StartEventWatcher(func(obj runtime.Object) {
				e, _ := obj.(*v1beta1.Event)
				if e, a := item.eventReason, e.Reason; e != a {
					t.Errorf("expected %v, got %v", e, a)
				}
				close(called)
			})
			s.scheduleOne()
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
			if e, a := item.expectBind, gotBinding; !reflect.DeepEqual(e, a) {
				t.Errorf("error: %s", diff.ObjectDiff(e, a))
			}
			stopFunc()
		})
	}
}

func TestSchedulerNoPhantomPodAfterExpire(t *testing.T) {
	stop := make(chan struct{})
	defer close(stop)
	queuedPodStore := clientcache.NewFIFO(clientcache.MetaNamespaceKeyFunc)
	scache := internalcache.New(100*time.Millisecond, stop)
	pod := podWithPort("pod.Name", "", 8080)
	node := v1.Node{ObjectMeta: metav1.ObjectMeta{Name: "machine1", UID: types.UID("machine1")}}
	scache.AddNode(&node)
	client := clientsetfake.NewSimpleClientset(&node)
	informerFactory := informers.NewSharedInformerFactory(client, 0)
	predicateMap := map[string]predicates.FitPredicate{"PodFitsHostPorts": predicates.PodFitsHostPorts}
	scheduler, bindingChan, _ := setupTestSchedulerWithOnePodOnNode(t, queuedPodStore, scache, informerFactory, stop, predicateMap, pod, &node)

	waitPodExpireChan := make(chan struct{})
	timeout := make(chan struct{})
	go func() {
		for {
			select {
			case <-timeout:
				return
			default:
			}
			pods, err := scache.List(labels.Everything())
			if err != nil {
				t.Fatalf("cache.List failed: %v", err)
			}
			if len(pods) == 0 {
				close(waitPodExpireChan)
				return
			}
			time.Sleep(100 * time.Millisecond)
		}
	}()
	// waiting for the assumed pod to expire
	select {
	case <-waitPodExpireChan:
	case <-time.After(wait.ForeverTestTimeout):
		close(timeout)
		t.Fatalf("timeout timeout in waiting pod expire after %v", wait.ForeverTestTimeout)
	}

	// We use conflicted pod ports to incur fit predicate failure if first pod not removed.
	secondPod := podWithPort("bar", "", 8080)
	queuedPodStore.Add(secondPod)
	scheduler.scheduleOne()
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
	stop := make(chan struct{})
	defer close(stop)
	queuedPodStore := clientcache.NewFIFO(clientcache.MetaNamespaceKeyFunc)
	scache := internalcache.New(10*time.Minute, stop)
	firstPod := podWithPort("pod.Name", "", 8080)
	node := v1.Node{ObjectMeta: metav1.ObjectMeta{Name: "machine1", UID: types.UID("machine1")}}
	scache.AddNode(&node)
	client := clientsetfake.NewSimpleClientset(&node)
	informerFactory := informers.NewSharedInformerFactory(client, 0)
	predicateMap := map[string]predicates.FitPredicate{"PodFitsHostPorts": predicates.PodFitsHostPorts}
	scheduler, bindingChan, errChan := setupTestSchedulerWithOnePodOnNode(t, queuedPodStore, scache, informerFactory, stop, predicateMap, firstPod, &node)

	// We use conflicted pod ports to incur fit predicate failure.
	secondPod := podWithPort("bar", "", 8080)
	queuedPodStore.Add(secondPod)
	// queuedPodStore: [bar:8080]
	// cache: [(assumed)foo:8080]

	scheduler.scheduleOne()
	select {
	case err := <-errChan:
		expectErr := &core.FitError{
			Pod:         secondPod,
			NumAllNodes: 1,
			FailedPredicates: core.FailedPredicateMap{
				node.Name: []predicates.PredicateFailureReason{predicates.ErrPodNotFitsHostPorts},
			},
			FilteredNodesStatuses: framework.NodeToStatusMap{},
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
	if err := scache.AddPod(firstPod); err != nil {
		t.Fatalf("err: %v", err)
	}
	if err := scache.RemovePod(firstPod); err != nil {
		t.Fatalf("err: %v", err)
	}

	queuedPodStore.Add(secondPod)
	scheduler.scheduleOne()
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

// Scheduler should preserve predicate constraint even if binding was longer
// than cache ttl
func TestSchedulerErrorWithLongBinding(t *testing.T) {
	stop := make(chan struct{})
	defer close(stop)

	firstPod := podWithPort("foo", "", 8080)
	conflictPod := podWithPort("bar", "", 8080)
	pods := map[string]*v1.Pod{firstPod.Name: firstPod, conflictPod.Name: conflictPod}
	for _, test := range []struct {
		name            string
		Expected        map[string]bool
		CacheTTL        time.Duration
		BindingDuration time.Duration
	}{
		{
			name:            "long cache ttl",
			Expected:        map[string]bool{firstPod.Name: true},
			CacheTTL:        100 * time.Millisecond,
			BindingDuration: 300 * time.Millisecond,
		},
		{
			name:            "short cache ttl",
			Expected:        map[string]bool{firstPod.Name: true},
			CacheTTL:        10 * time.Second,
			BindingDuration: 300 * time.Millisecond,
		},
	} {
		t.Run(test.name, func(t *testing.T) {
			queuedPodStore := clientcache.NewFIFO(clientcache.MetaNamespaceKeyFunc)
			scache := internalcache.New(test.CacheTTL, stop)

			node := v1.Node{ObjectMeta: metav1.ObjectMeta{Name: "machine1", UID: types.UID("machine1")}}
			scache.AddNode(&node)

			client := clientsetfake.NewSimpleClientset(&node)
			informerFactory := informers.NewSharedInformerFactory(client, 0)
			predicateMap := map[string]predicates.FitPredicate{"PodFitsHostPorts": predicates.PodFitsHostPorts}

			scheduler, bindingChan := setupTestSchedulerLongBindingWithRetry(
				queuedPodStore, scache, informerFactory, predicateMap, stop, test.BindingDuration)

			informerFactory.Start(stop)
			informerFactory.WaitForCacheSync(stop)

			scheduler.Run()
			queuedPodStore.Add(firstPod)
			queuedPodStore.Add(conflictPod)

			resultBindings := map[string]bool{}
			waitChan := time.After(5 * time.Second)
			for finished := false; !finished; {
				select {
				case b := <-bindingChan:
					resultBindings[b.Name] = true
					p := pods[b.Name]
					p.Spec.NodeName = b.Target.Name
					scache.AddPod(p)
				case <-waitChan:
					finished = true
				}
			}
			if !reflect.DeepEqual(resultBindings, test.Expected) {
				t.Errorf("Result binding are not equal to expected. %v != %v", resultBindings, test.Expected)
			}
		})
	}
}

// queuedPodStore: pods queued before processing.
// cache: scheduler cache that might contain assumed pods.
func setupTestSchedulerWithOnePodOnNode(t *testing.T, queuedPodStore *clientcache.FIFO, scache internalcache.Cache,
	informerFactory informers.SharedInformerFactory, stop chan struct{}, predicateMap map[string]predicates.FitPredicate, pod *v1.Pod, node *v1.Node) (*Scheduler, chan *v1.Binding, chan error) {

	scheduler, bindingChan, errChan := setupTestScheduler(queuedPodStore, scache, informerFactory, predicateMap, nil)

	informerFactory.Start(stop)
	informerFactory.WaitForCacheSync(stop)

	queuedPodStore.Add(pod)
	// queuedPodStore: [foo:8080]
	// cache: []

	scheduler.scheduleOne()
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

func TestSchedulerFailedSchedulingReasons(t *testing.T) {
	stop := make(chan struct{})
	defer close(stop)
	queuedPodStore := clientcache.NewFIFO(clientcache.MetaNamespaceKeyFunc)
	scache := internalcache.New(10*time.Minute, stop)

	// Design the baseline for the pods, and we will make nodes that dont fit it later.
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
		uid := fmt.Sprintf("machine%v", i)
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
		scache.AddNode(&node)
		nodes = append(nodes, &node)
		objects = append(objects, &node)
	}
	client := clientsetfake.NewSimpleClientset(objects...)
	informerFactory := informers.NewSharedInformerFactory(client, 0)
	predicateMap := map[string]predicates.FitPredicate{
		"PodFitsResources": predicates.PodFitsResources,
	}

	// Create expected failure reasons for all the nodes.  Hopefully they will get rolled up into a non-spammy summary.
	failedPredicatesMap := core.FailedPredicateMap{}
	for _, node := range nodes {
		failedPredicatesMap[node.Name] = []predicates.PredicateFailureReason{
			predicates.NewInsufficientResourceError(v1.ResourceCPU, 4000, 0, 2000),
			predicates.NewInsufficientResourceError(v1.ResourceMemory, 500, 0, 100),
		}
	}
	scheduler, _, errChan := setupTestScheduler(queuedPodStore, scache, informerFactory, predicateMap, nil)

	informerFactory.Start(stop)
	informerFactory.WaitForCacheSync(stop)

	queuedPodStore.Add(podWithTooBigResourceRequests)
	scheduler.scheduleOne()
	select {
	case err := <-errChan:
		expectErr := &core.FitError{
			Pod:                   podWithTooBigResourceRequests,
			NumAllNodes:           len(nodes),
			FailedPredicates:      failedPredicatesMap,
			FilteredNodesStatuses: framework.NodeToStatusMap{},
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

// queuedPodStore: pods queued before processing.
// scache: scheduler cache that might contain assumed pods.
func setupTestScheduler(queuedPodStore *clientcache.FIFO, scache internalcache.Cache, informerFactory informers.SharedInformerFactory, predicateMap map[string]predicates.FitPredicate, recorder events.EventRecorder) (*Scheduler, chan *v1.Binding, chan error) {
	algo := core.NewGenericScheduler(
		scache,
		internalqueue.NewSchedulingQueue(nil, nil),
		predicateMap,
		predicates.EmptyPredicateMetadataProducer,
		[]priorities.PriorityConfig{},
		priorities.EmptyPriorityMetadataProducer,
		emptyFramework,
		[]algorithm.SchedulerExtender{},
		nil,
		informerFactory.Core().V1().PersistentVolumeClaims().Lister(),
		informerFactory.Policy().V1beta1().PodDisruptionBudgets().Lister(),
		false,
		false,
		schedulerapi.DefaultPercentageOfNodesToScore,
		false,
	)
	bindingChan := make(chan *v1.Binding, 1)
	errChan := make(chan error, 1)

	config := &factory.Config{
		SchedulerCache: scache,
		Algorithm:      algo,
		GetBinder: func(pod *v1.Pod) factory.Binder {
			return fakeBinder{func(b *v1.Binding) error {
				bindingChan <- b
				return nil
			}}
		},
		NextPod: func() *v1.Pod {
			return clientcache.Pop(queuedPodStore).(*v1.Pod)
		},
		Error: func(p *v1.Pod, err error) {
			errChan <- err
		},
		Recorder:            &events.FakeRecorder{},
		PodConditionUpdater: fakePodConditionUpdater{},
		PodPreemptor:        fakePodPreemptor{},
		Framework:           emptyFramework,
		VolumeBinder:        volumebinder.NewFakeVolumeBinder(&volumescheduling.FakeVolumeBinderConfig{AllBound: true}),
	}

	if recorder != nil {
		config.Recorder = recorder
	}

	sched := NewFromConfig(config)

	return sched, bindingChan, errChan
}

func setupTestSchedulerLongBindingWithRetry(queuedPodStore *clientcache.FIFO, scache internalcache.Cache, informerFactory informers.SharedInformerFactory, predicateMap map[string]predicates.FitPredicate, stop chan struct{}, bindingTime time.Duration) (*Scheduler, chan *v1.Binding) {
	framework, _ := framework.NewFramework(emptyPluginRegistry, nil, []kubeschedulerconfig.PluginConfig{})
	algo := core.NewGenericScheduler(
		scache,
		internalqueue.NewSchedulingQueue(nil, nil),
		predicateMap,
		predicates.EmptyPredicateMetadataProducer,
		[]priorities.PriorityConfig{},
		priorities.EmptyPriorityMetadataProducer,
		framework,
		[]algorithm.SchedulerExtender{},
		nil,
		informerFactory.Core().V1().PersistentVolumeClaims().Lister(),
		informerFactory.Policy().V1beta1().PodDisruptionBudgets().Lister(),
		false,
		false,
		schedulerapi.DefaultPercentageOfNodesToScore,
		false,
	)
	bindingChan := make(chan *v1.Binding, 2)

	sched := NewFromConfig(&factory.Config{
		SchedulerCache: scache,
		Algorithm:      algo,
		GetBinder: func(pod *v1.Pod) factory.Binder {
			return fakeBinder{func(b *v1.Binding) error {
				time.Sleep(bindingTime)
				bindingChan <- b
				return nil
			}}
		},
		WaitForCacheSync: func() bool {
			return true
		},
		NextPod: func() *v1.Pod {
			return clientcache.Pop(queuedPodStore).(*v1.Pod)
		},
		Error: func(p *v1.Pod, err error) {
			queuedPodStore.AddIfNotPresent(p)
		},
		Recorder:            &events.FakeRecorder{},
		PodConditionUpdater: fakePodConditionUpdater{},
		PodPreemptor:        fakePodPreemptor{},
		StopEverything:      stop,
		Framework:           framework,
		VolumeBinder:        volumebinder.NewFakeVolumeBinder(&volumescheduling.FakeVolumeBinderConfig{AllBound: true}),
	})

	return sched, bindingChan
}

func setupTestSchedulerWithVolumeBinding(fakeVolumeBinder *volumebinder.VolumeBinder, stop <-chan struct{}, broadcaster events.EventBroadcaster) (*Scheduler, chan *v1.Binding, chan error) {
	testNode := v1.Node{ObjectMeta: metav1.ObjectMeta{Name: "machine1", UID: types.UID("machine1")}}
	queuedPodStore := clientcache.NewFIFO(clientcache.MetaNamespaceKeyFunc)
	pod := podWithID("foo", "")
	pod.Namespace = "foo-ns"
	pod.Spec.Volumes = append(pod.Spec.Volumes, v1.Volume{Name: "testVol",
		VolumeSource: v1.VolumeSource{PersistentVolumeClaim: &v1.PersistentVolumeClaimVolumeSource{ClaimName: "testPVC"}}})
	queuedPodStore.Add(pod)
	scache := internalcache.New(10*time.Minute, stop)
	scache.AddNode(&testNode)
	testPVC := v1.PersistentVolumeClaim{ObjectMeta: metav1.ObjectMeta{Name: "testPVC", Namespace: pod.Namespace, UID: types.UID("testPVC")}}
	client := clientsetfake.NewSimpleClientset(&testNode, &testPVC)
	informerFactory := informers.NewSharedInformerFactory(client, 0)

	predicateMap := map[string]predicates.FitPredicate{
		predicates.CheckVolumeBindingPred: predicates.NewVolumeBindingPredicate(fakeVolumeBinder),
	}

	recorder := broadcaster.NewRecorder(scheme.Scheme, "scheduler")
	s, bindingChan, errChan := setupTestScheduler(queuedPodStore, scache, informerFactory, predicateMap, recorder)
	informerFactory.Start(stop)
	informerFactory.WaitForCacheSync(stop)
	s.VolumeBinder = fakeVolumeBinder
	return s, bindingChan, errChan
}

// This is a workaround because golint complains that errors cannot
// end with punctuation.  However, the real predicate error message does
// end with a period.
func makePredicateError(failReason string) error {
	s := fmt.Sprintf("0/1 nodes are available: %v.", failReason)
	return fmt.Errorf(s)
}

func TestSchedulerWithVolumeBinding(t *testing.T) {
	findErr := fmt.Errorf("find err")
	assumeErr := fmt.Errorf("assume err")
	bindErr := fmt.Errorf("bind err")
	client := clientsetfake.NewSimpleClientset()

	eventBroadcaster := events.NewBroadcaster(&events.EventSinkImpl{Interface: client.EventsV1beta1().Events("")})

	// This can be small because we wait for pod to finish scheduling first
	chanTimeout := 2 * time.Second

	table := []struct {
		name               string
		expectError        error
		expectPodBind      *v1.Binding
		expectAssumeCalled bool
		expectBindCalled   bool
		eventReason        string
		volumeBinderConfig *volumescheduling.FakeVolumeBinderConfig
	}{
		{
			name: "all bound",
			volumeBinderConfig: &volumescheduling.FakeVolumeBinderConfig{
				AllBound:             true,
				FindUnboundSatsified: true,
				FindBoundSatsified:   true,
			},
			expectAssumeCalled: true,
			expectPodBind:      &v1.Binding{ObjectMeta: metav1.ObjectMeta{Name: "foo", Namespace: "foo-ns", UID: types.UID("foo")}, Target: v1.ObjectReference{Kind: "Node", Name: "machine1"}},
			eventReason:        "Scheduled",
		},
		{
			name: "bound/invalid pv affinity",
			volumeBinderConfig: &volumescheduling.FakeVolumeBinderConfig{
				AllBound:             true,
				FindUnboundSatsified: true,
				FindBoundSatsified:   false,
			},
			eventReason: "FailedScheduling",
			expectError: makePredicateError("1 node(s) had volume node affinity conflict"),
		},
		{
			name: "unbound/no matches",
			volumeBinderConfig: &volumescheduling.FakeVolumeBinderConfig{
				FindUnboundSatsified: false,
				FindBoundSatsified:   true,
			},
			eventReason: "FailedScheduling",
			expectError: makePredicateError("1 node(s) didn't find available persistent volumes to bind"),
		},
		{
			name: "bound and unbound unsatisfied",
			volumeBinderConfig: &volumescheduling.FakeVolumeBinderConfig{
				FindUnboundSatsified: false,
				FindBoundSatsified:   false,
			},
			eventReason: "FailedScheduling",
			expectError: makePredicateError("1 node(s) didn't find available persistent volumes to bind, 1 node(s) had volume node affinity conflict"),
		},
		{
			name: "unbound/found matches/bind succeeds",
			volumeBinderConfig: &volumescheduling.FakeVolumeBinderConfig{
				FindUnboundSatsified: true,
				FindBoundSatsified:   true,
			},
			expectAssumeCalled: true,
			expectBindCalled:   true,
			expectPodBind:      &v1.Binding{ObjectMeta: metav1.ObjectMeta{Name: "foo", Namespace: "foo-ns", UID: types.UID("foo")}, Target: v1.ObjectReference{Kind: "Node", Name: "machine1"}},
			eventReason:        "Scheduled",
		},
		{
			name: "predicate error",
			volumeBinderConfig: &volumescheduling.FakeVolumeBinderConfig{
				FindErr: findErr,
			},
			eventReason: "FailedScheduling",
			expectError: findErr,
		},
		{
			name: "assume error",
			volumeBinderConfig: &volumescheduling.FakeVolumeBinderConfig{
				FindUnboundSatsified: true,
				FindBoundSatsified:   true,
				AssumeErr:            assumeErr,
			},
			expectAssumeCalled: true,
			eventReason:        "FailedScheduling",
			expectError:        assumeErr,
		},
		{
			name: "bind error",
			volumeBinderConfig: &volumescheduling.FakeVolumeBinderConfig{
				FindUnboundSatsified: true,
				FindBoundSatsified:   true,
				BindErr:              bindErr,
			},
			expectAssumeCalled: true,
			expectBindCalled:   true,
			eventReason:        "FailedScheduling",
			expectError:        bindErr,
		},
	}

	for _, item := range table {
		t.Run(item.name, func(t *testing.T) {
			stop := make(chan struct{})
			fakeVolumeBinder := volumebinder.NewFakeVolumeBinder(item.volumeBinderConfig)
			internalBinder, ok := fakeVolumeBinder.Binder.(*volumescheduling.FakeVolumeBinder)
			if !ok {
				t.Fatalf("Failed to get fake volume binder")
			}
			s, bindingChan, errChan := setupTestSchedulerWithVolumeBinding(fakeVolumeBinder, stop, eventBroadcaster)
			eventChan := make(chan struct{})
			stopFunc := eventBroadcaster.StartEventWatcher(func(obj runtime.Object) {
				e, _ := obj.(*v1beta1.Event)
				if e, a := item.eventReason, e.Reason; e != a {
					t.Errorf("expected %v, got %v", e, a)
				}
				close(eventChan)
			})
			s.scheduleOne()
			// Wait for pod to succeed or fail scheduling
			select {
			case <-eventChan:
			case <-time.After(wait.ForeverTestTimeout):
				t.Fatalf("scheduling timeout after %v", wait.ForeverTestTimeout)
			}
			stopFunc()
			// Wait for scheduling to return an error
			select {
			case err := <-errChan:
				if item.expectError == nil || !reflect.DeepEqual(item.expectError.Error(), err.Error()) {
					t.Errorf("err \nWANT=%+v,\nGOT=%+v", item.expectError, err)
				}
			case <-time.After(chanTimeout):
				if item.expectError != nil {
					t.Errorf("did not receive error after %v", chanTimeout)
				}
			}

			// Wait for pod to succeed binding
			select {
			case b := <-bindingChan:
				if !reflect.DeepEqual(item.expectPodBind, b) {
					t.Errorf("err \nWANT=%+v,\nGOT=%+v", item.expectPodBind, b)
				}
			case <-time.After(chanTimeout):
				if item.expectPodBind != nil {
					t.Errorf("did not receive pod binding after %v", chanTimeout)
				}
			}

			if item.expectAssumeCalled != internalBinder.AssumeCalled {
				t.Errorf("expectedAssumeCall %v", item.expectAssumeCalled)
			}

			if item.expectBindCalled != internalBinder.BindCalled {
				t.Errorf("expectedBindCall %v", item.expectBindCalled)
			}

			close(stop)
		})
	}
}

func TestInitPolicyFromFile(t *testing.T) {
	dir, err := ioutil.TempDir(os.TempDir(), "policy")
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	defer os.RemoveAll(dir)

	for i, test := range []struct {
		policy               string
		expectedPredicates   sets.String
		expectedPrioritizers sets.String
	}{
		// Test json format policy file
		{
			policy: `{
				"kind" : "Policy",
				"apiVersion" : "v1",
				"predicates" : [
					{"name" : "PredicateOne"},
					{"name" : "PredicateTwo"}
				],
				"priorities" : [
					{"name" : "PriorityOne", "weight" : 1},
					{"name" : "PriorityTwo", "weight" : 5}
				]
			}`,
			expectedPredicates: sets.NewString(
				"PredicateOne",
				"PredicateTwo",
			),
			expectedPrioritizers: sets.NewString(
				"PriorityOne",
				"PriorityTwo",
			),
		},
		// Test yaml format policy file
		{
			policy: `apiVersion: v1
kind: Policy
predicates:
- name: PredicateOne
- name: PredicateTwo
priorities:
- name: PriorityOne
  weight: 1
- name: PriorityTwo
  weight: 5
`,
			expectedPredicates: sets.NewString(
				"PredicateOne",
				"PredicateTwo",
			),
			expectedPrioritizers: sets.NewString(
				"PriorityOne",
				"PriorityTwo",
			),
		},
	} {
		file := fmt.Sprintf("scheduler-policy-config-file-%d", i)
		fullPath := path.Join(dir, file)

		if err := ioutil.WriteFile(fullPath, []byte(test.policy), 0644); err != nil {
			t.Fatalf("Failed writing a policy config file: %v", err)
		}

		policy := &schedulerapi.Policy{}

		if err := initPolicyFromFile(fullPath, policy); err != nil {
			t.Fatalf("Failed writing a policy config file: %v", err)
		}

		// Verify that the policy is initialized correctly.
		schedPredicates := sets.NewString()
		for _, p := range policy.Predicates {
			schedPredicates.Insert(p.Name)
		}
		schedPrioritizers := sets.NewString()
		for _, p := range policy.Priorities {
			schedPrioritizers.Insert(p.Name)
		}
		if !schedPredicates.Equal(test.expectedPredicates) {
			t.Errorf("Expected predicates %v, got %v", test.expectedPredicates, schedPredicates)
		}
		if !schedPrioritizers.Equal(test.expectedPrioritizers) {
			t.Errorf("Expected priority functions %v, got %v", test.expectedPrioritizers, schedPrioritizers)
		}
	}
}
