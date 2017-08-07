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
	"reflect"
	"testing"
	"time"

	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/util/diff"
	"k8s.io/apimachinery/pkg/util/wait"
	clientv1 "k8s.io/client-go/pkg/api/v1"
	clientcache "k8s.io/client-go/tools/cache"
	"k8s.io/client-go/tools/record"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/testapi"
	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/plugin/pkg/scheduler/algorithm"
	"k8s.io/kubernetes/plugin/pkg/scheduler/algorithm/predicates"
	"k8s.io/kubernetes/plugin/pkg/scheduler/core"
	"k8s.io/kubernetes/plugin/pkg/scheduler/schedulercache"
	schedulertesting "k8s.io/kubernetes/plugin/pkg/scheduler/testing"
)

type fakeBinder struct {
	b func(binding *v1.Binding) error
}

func (fb fakeBinder) Bind(binding *v1.Binding) error { return fb.b(binding) }

type fakePodConditionUpdater struct{}

func (fc fakePodConditionUpdater) Update(pod *v1.Pod, podCondition *v1.PodCondition) error {
	return nil
}

func podWithID(id, desiredHost string) *v1.Pod {
	return &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{Name: id, SelfLink: testapi.Default.SelfLink("pods", id)},
		Spec: v1.PodSpec{
			NodeName: desiredHost,
		},
	}
}

func deletingPod(id string) *v1.Pod {
	deletionTimestamp := metav1.Now()
	return &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{Name: id, SelfLink: testapi.Default.SelfLink("pods", id), DeletionTimestamp: &deletionTimestamp},
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

type mockScheduler struct {
	machine string
	err     error
}

func (es mockScheduler) Schedule(pod *v1.Pod, ml algorithm.NodeLister) (string, error) {
	return es.machine, es.err
}

func (es mockScheduler) Predicates() map[string]algorithm.FitPredicate {
	return nil
}
func (es mockScheduler) Prioritizers() []algorithm.PriorityConfig {
	return nil
}

func TestScheduler(t *testing.T) {
	eventBroadcaster := record.NewBroadcaster()
	eventBroadcaster.StartLogging(t.Logf).Stop()
	errS := errors.New("scheduler")
	errB := errors.New("binder")
	testNode := v1.Node{ObjectMeta: metav1.ObjectMeta{Name: "machine1"}}

	table := []struct {
		injectBindError  error
		sendPod          *v1.Pod
		algo             algorithm.ScheduleAlgorithm
		expectErrorPod   *v1.Pod
		expectForgetPod  *v1.Pod
		expectAssumedPod *v1.Pod
		expectError      error
		expectBind       *v1.Binding
		eventReason      string
	}{
		{
			sendPod:          podWithID("foo", ""),
			algo:             mockScheduler{testNode.Name, nil},
			expectBind:       &v1.Binding{ObjectMeta: metav1.ObjectMeta{Name: "foo"}, Target: v1.ObjectReference{Kind: "Node", Name: testNode.Name}},
			expectAssumedPod: podWithID("foo", testNode.Name),
			eventReason:      "Scheduled",
		}, {
			sendPod:        podWithID("foo", ""),
			algo:           mockScheduler{testNode.Name, errS},
			expectError:    errS,
			expectErrorPod: podWithID("foo", ""),
			eventReason:    "FailedScheduling",
		}, {
			sendPod:          podWithID("foo", ""),
			algo:             mockScheduler{testNode.Name, nil},
			expectBind:       &v1.Binding{ObjectMeta: metav1.ObjectMeta{Name: "foo"}, Target: v1.ObjectReference{Kind: "Node", Name: testNode.Name}},
			expectAssumedPod: podWithID("foo", testNode.Name),
			injectBindError:  errB,
			expectError:      errB,
			expectErrorPod:   podWithID("foo", testNode.Name),
			expectForgetPod:  podWithID("foo", testNode.Name),
			eventReason:      "FailedScheduling",
		}, {
			sendPod:     deletingPod("foo"),
			algo:        mockScheduler{"", nil},
			eventReason: "FailedScheduling",
		},
	}

	for i, item := range table {
		var gotError error
		var gotPod *v1.Pod
		var gotForgetPod *v1.Pod
		var gotAssumedPod *v1.Pod
		var gotBinding *v1.Binding
		configurator := &FakeConfigurator{
			Config: &Config{
				SchedulerCache: &schedulertesting.FakeCache{
					ForgetFunc: func(pod *v1.Pod) {
						gotForgetPod = pod
					},
					AssumeFunc: func(pod *v1.Pod) {
						gotAssumedPod = pod
					},
				},
				NodeLister: schedulertesting.FakeNodeLister(
					[]*v1.Node{&testNode},
				),
				Algorithm: item.algo,
				Binder: fakeBinder{func(b *v1.Binding) error {
					gotBinding = b
					return item.injectBindError
				}},
				PodConditionUpdater: fakePodConditionUpdater{},
				Error: func(p *v1.Pod, err error) {
					gotPod = p
					gotError = err
				},
				NextPod: func() *v1.Pod {
					return item.sendPod
				},
				Recorder: eventBroadcaster.NewRecorder(api.Scheme, clientv1.EventSource{Component: "scheduler"}),
			},
		}

		s, _ := NewFromConfigurator(configurator, nil...)
		called := make(chan struct{})
		events := eventBroadcaster.StartEventWatcher(func(e *clientv1.Event) {
			if e, a := item.eventReason, e.Reason; e != a {
				t.Errorf("%v: expected %v, got %v", i, e, a)
			}
			close(called)
		})
		s.scheduleOne()
		<-called
		if e, a := item.expectAssumedPod, gotAssumedPod; !reflect.DeepEqual(e, a) {
			t.Errorf("%v: assumed pod: wanted %v, got %v", i, e, a)
		}
		if e, a := item.expectErrorPod, gotPod; !reflect.DeepEqual(e, a) {
			t.Errorf("%v: error pod: wanted %v, got %v", i, e, a)
		}
		if e, a := item.expectForgetPod, gotForgetPod; !reflect.DeepEqual(e, a) {
			t.Errorf("%v: forget pod: wanted %v, got %v", i, e, a)
		}
		if e, a := item.expectError, gotError; !reflect.DeepEqual(e, a) {
			t.Errorf("%v: error: wanted %v, got %v", i, e, a)
		}
		if e, a := item.expectBind, gotBinding; !reflect.DeepEqual(e, a) {
			t.Errorf("%v: error: %s", i, diff.ObjectDiff(e, a))
		}
		events.Stop()
	}
}

func TestSchedulerNoPhantomPodAfterExpire(t *testing.T) {
	stop := make(chan struct{})
	defer close(stop)
	queuedPodStore := clientcache.NewFIFO(clientcache.MetaNamespaceKeyFunc)
	scache := schedulercache.New(100*time.Millisecond, stop)
	pod := podWithPort("pod.Name", "", 8080)
	node := v1.Node{ObjectMeta: metav1.ObjectMeta{Name: "machine1"}}
	scache.AddNode(&node)
	nodeLister := schedulertesting.FakeNodeLister([]*v1.Node{&node})
	predicateMap := map[string]algorithm.FitPredicate{"PodFitsHostPorts": predicates.PodFitsHostPorts}
	scheduler, bindingChan, _ := setupTestSchedulerWithOnePodOnNode(t, queuedPodStore, scache, nodeLister, predicateMap, pod, &node)

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
		t.Fatalf("timeout after %v", wait.ForeverTestTimeout)
	}

	// We use conflicted pod ports to incur fit predicate failure if first pod not removed.
	secondPod := podWithPort("bar", "", 8080)
	queuedPodStore.Add(secondPod)
	scheduler.scheduleOne()
	select {
	case b := <-bindingChan:
		expectBinding := &v1.Binding{
			ObjectMeta: metav1.ObjectMeta{Name: "bar"},
			Target:     v1.ObjectReference{Kind: "Node", Name: node.Name},
		}
		if !reflect.DeepEqual(expectBinding, b) {
			t.Errorf("binding want=%v, get=%v", expectBinding, b)
		}
	case <-time.After(wait.ForeverTestTimeout):
		t.Fatalf("timeout after %v", wait.ForeverTestTimeout)
	}
}

func TestSchedulerNoPhantomPodAfterDelete(t *testing.T) {
	stop := make(chan struct{})
	defer close(stop)
	queuedPodStore := clientcache.NewFIFO(clientcache.MetaNamespaceKeyFunc)
	scache := schedulercache.New(10*time.Minute, stop)
	firstPod := podWithPort("pod.Name", "", 8080)
	node := v1.Node{ObjectMeta: metav1.ObjectMeta{Name: "machine1"}}
	scache.AddNode(&node)
	nodeLister := schedulertesting.FakeNodeLister([]*v1.Node{&node})
	predicateMap := map[string]algorithm.FitPredicate{"PodFitsHostPorts": predicates.PodFitsHostPorts}
	scheduler, bindingChan, errChan := setupTestSchedulerWithOnePodOnNode(t, queuedPodStore, scache, nodeLister, predicateMap, firstPod, &node)

	// We use conflicted pod ports to incur fit predicate failure.
	secondPod := podWithPort("bar", "", 8080)
	queuedPodStore.Add(secondPod)
	// queuedPodStore: [bar:8080]
	// cache: [(assumed)foo:8080]

	scheduler.scheduleOne()
	select {
	case err := <-errChan:
		expectErr := &core.FitError{
			Pod:              secondPod,
			FailedPredicates: core.FailedPredicateMap{node.Name: []algorithm.PredicateFailureReason{predicates.ErrPodNotFitsHostPorts}},
		}
		if !reflect.DeepEqual(expectErr, err) {
			t.Errorf("err want=%v, get=%v", expectErr, err)
		}
	case <-time.After(wait.ForeverTestTimeout):
		t.Fatalf("timeout after %v", wait.ForeverTestTimeout)
	}

	// We mimic the workflow of cache behavior when a pod is removed by user.
	// Note: if the schedulercache timeout would be super short, the first pod would expire
	// and would be removed itself (without any explicit actions on schedulercache). Even in that case,
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
			ObjectMeta: metav1.ObjectMeta{Name: "bar"},
			Target:     v1.ObjectReference{Kind: "Node", Name: node.Name},
		}
		if !reflect.DeepEqual(expectBinding, b) {
			t.Errorf("binding want=%v, get=%v", expectBinding, b)
		}
	case <-time.After(wait.ForeverTestTimeout):
		t.Fatalf("timeout after %v", wait.ForeverTestTimeout)
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
		Expected        map[string]bool
		CacheTTL        time.Duration
		BindingDuration time.Duration
	}{
		{
			Expected:        map[string]bool{firstPod.Name: true},
			CacheTTL:        100 * time.Millisecond,
			BindingDuration: 300 * time.Millisecond,
		},
		{
			Expected:        map[string]bool{firstPod.Name: true},
			CacheTTL:        10 * time.Second,
			BindingDuration: 300 * time.Millisecond,
		},
	} {
		queuedPodStore := clientcache.NewFIFO(clientcache.MetaNamespaceKeyFunc)
		scache := schedulercache.New(test.CacheTTL, stop)

		node := v1.Node{ObjectMeta: metav1.ObjectMeta{Name: "machine1"}}
		scache.AddNode(&node)

		nodeLister := schedulertesting.FakeNodeLister([]*v1.Node{&node})
		predicateMap := map[string]algorithm.FitPredicate{"PodFitsHostPorts": predicates.PodFitsHostPorts}

		scheduler, bindingChan := setupTestSchedulerLongBindingWithRetry(
			queuedPodStore, scache, nodeLister, predicateMap, stop, test.BindingDuration)
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
	}
}

// queuedPodStore: pods queued before processing.
// cache: scheduler cache that might contain assumed pods.
func setupTestSchedulerWithOnePodOnNode(t *testing.T, queuedPodStore *clientcache.FIFO, scache schedulercache.Cache,
	nodeLister schedulertesting.FakeNodeLister, predicateMap map[string]algorithm.FitPredicate, pod *v1.Pod, node *v1.Node) (*Scheduler, chan *v1.Binding, chan error) {

	scheduler, bindingChan, errChan := setupTestScheduler(queuedPodStore, scache, nodeLister, predicateMap)

	queuedPodStore.Add(pod)
	// queuedPodStore: [foo:8080]
	// cache: []

	scheduler.scheduleOne()
	// queuedPodStore: []
	// cache: [(assumed)foo:8080]

	select {
	case b := <-bindingChan:
		expectBinding := &v1.Binding{
			ObjectMeta: metav1.ObjectMeta{Name: pod.Name},
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
	scache := schedulercache.New(10*time.Minute, stop)

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
	nodes := []*v1.Node{}
	for i := 0; i < 100; i++ {
		node := v1.Node{
			ObjectMeta: metav1.ObjectMeta{Name: fmt.Sprintf("machine%v", i)},
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
	}
	nodeLister := schedulertesting.FakeNodeLister(nodes)
	predicateMap := map[string]algorithm.FitPredicate{
		"PodFitsResources": predicates.PodFitsResources,
	}

	// Create expected failure reasons for all the nodes.  Hopefully they will get rolled up into a non-spammy summary.
	failedPredicatesMap := core.FailedPredicateMap{}
	for _, node := range nodes {
		failedPredicatesMap[node.Name] = []algorithm.PredicateFailureReason{
			predicates.NewInsufficientResourceError(v1.ResourceCPU, 4000, 0, 2000),
			predicates.NewInsufficientResourceError(v1.ResourceMemory, 500, 0, 100),
		}
	}
	scheduler, _, errChan := setupTestScheduler(queuedPodStore, scache, nodeLister, predicateMap)

	queuedPodStore.Add(podWithTooBigResourceRequests)
	scheduler.scheduleOne()
	select {
	case err := <-errChan:
		expectErr := &core.FitError{
			Pod:              podWithTooBigResourceRequests,
			FailedPredicates: failedPredicatesMap,
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
func setupTestScheduler(queuedPodStore *clientcache.FIFO, scache schedulercache.Cache, nodeLister schedulertesting.FakeNodeLister, predicateMap map[string]algorithm.FitPredicate) (*Scheduler, chan *v1.Binding, chan error) {
	algo := core.NewGenericScheduler(
		scache,
		nil,
		predicateMap,
		algorithm.EmptyMetadataProducer,
		[]algorithm.PriorityConfig{},
		algorithm.EmptyMetadataProducer,
		[]algorithm.SchedulerExtender{})
	bindingChan := make(chan *v1.Binding, 1)
	errChan := make(chan error, 1)
	configurator := &FakeConfigurator{
		Config: &Config{
			SchedulerCache: scache,
			NodeLister:     nodeLister,
			Algorithm:      algo,
			Binder: fakeBinder{func(b *v1.Binding) error {
				bindingChan <- b
				return nil
			}},
			NextPod: func() *v1.Pod {
				return clientcache.Pop(queuedPodStore).(*v1.Pod)
			},
			Error: func(p *v1.Pod, err error) {
				errChan <- err
			},
			Recorder:            &record.FakeRecorder{},
			PodConditionUpdater: fakePodConditionUpdater{},
		},
	}

	sched, _ := NewFromConfigurator(configurator, nil...)

	return sched, bindingChan, errChan
}

func setupTestSchedulerLongBindingWithRetry(queuedPodStore *clientcache.FIFO, scache schedulercache.Cache, nodeLister schedulertesting.FakeNodeLister, predicateMap map[string]algorithm.FitPredicate, stop chan struct{}, bindingTime time.Duration) (*Scheduler, chan *v1.Binding) {
	algo := core.NewGenericScheduler(
		scache,
		nil,
		predicateMap,
		algorithm.EmptyMetadataProducer,
		[]algorithm.PriorityConfig{},
		algorithm.EmptyMetadataProducer,
		[]algorithm.SchedulerExtender{})
	bindingChan := make(chan *v1.Binding, 2)
	configurator := &FakeConfigurator{
		Config: &Config{
			SchedulerCache: scache,
			NodeLister:     nodeLister,
			Algorithm:      algo,
			Binder: fakeBinder{func(b *v1.Binding) error {
				time.Sleep(bindingTime)
				bindingChan <- b
				return nil
			}},
			WaitForCacheSync: func() bool {
				return true
			},
			NextPod: func() *v1.Pod {
				return clientcache.Pop(queuedPodStore).(*v1.Pod)
			},
			Error: func(p *v1.Pod, err error) {
				queuedPodStore.AddIfNotPresent(p)
			},
			Recorder:            &record.FakeRecorder{},
			PodConditionUpdater: fakePodConditionUpdater{},
			StopEverything:      stop,
		},
	}

	sched, _ := NewFromConfigurator(configurator, nil...)

	return sched, bindingChan
}
