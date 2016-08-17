/*
Copyright 2015 The Kubernetes Authors.

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

package executor

import (
	"fmt"
	"io/ioutil"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"reflect"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	assertext "k8s.io/kubernetes/contrib/mesos/pkg/assert"
	"k8s.io/kubernetes/contrib/mesos/pkg/executor/messages"
	"k8s.io/kubernetes/contrib/mesos/pkg/podutil"
	kmruntime "k8s.io/kubernetes/contrib/mesos/pkg/runtime"
	"k8s.io/kubernetes/contrib/mesos/pkg/scheduler/podtask"
	"k8s.io/kubernetes/contrib/mesos/pkg/scheduler/podtask/hostport"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/testapi"
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/client/cache"
	"k8s.io/kubernetes/pkg/kubelet/dockertools"
	"k8s.io/kubernetes/pkg/runtime"
	utiltesting "k8s.io/kubernetes/pkg/util/testing"
	"k8s.io/kubernetes/pkg/util/wait"
	"k8s.io/kubernetes/pkg/watch"

	"github.com/mesos/mesos-go/mesosproto"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/mock"
)

// TestExecutorRegister ensures that the executor thinks it is connected
// after Register is called.
func TestExecutorRegister(t *testing.T) {
	mockDriver := &MockExecutorDriver{}
	executor := NewTestKubernetesExecutor()

	executor.Init(mockDriver)
	executor.Registered(mockDriver, nil, nil, nil)

	assert.Equal(t, true, executor.isConnected(), "executor should be connected")
	mockDriver.AssertExpectations(t)
}

// TestExecutorDisconnect ensures that the executor thinks that it is not
// connected after a call to Disconnected has occurred.
func TestExecutorDisconnect(t *testing.T) {
	mockDriver := &MockExecutorDriver{}
	executor := NewTestKubernetesExecutor()

	executor.Init(mockDriver)
	executor.Registered(mockDriver, nil, nil, nil)
	executor.Disconnected(mockDriver)

	assert.Equal(t, false, executor.isConnected(),
		"executor should not be connected after Disconnected")
	mockDriver.AssertExpectations(t)
}

// TestExecutorReregister ensures that the executor thinks it is connected
// after a connection problem happens, followed by a call to Reregistered.
func TestExecutorReregister(t *testing.T) {
	mockDriver := &MockExecutorDriver{}
	executor := NewTestKubernetesExecutor()

	executor.Init(mockDriver)
	executor.Registered(mockDriver, nil, nil, nil)
	executor.Disconnected(mockDriver)
	executor.Reregistered(mockDriver, nil)

	assert.Equal(t, true, executor.isConnected(), "executor should be connected")
	mockDriver.AssertExpectations(t)
}

type fakeRegistry struct {
	sync.Mutex
	boundTasks map[string]*api.Pod
	updates    chan *PodEvent
}

func newFakeRegistry() *fakeRegistry {
	return &fakeRegistry{boundTasks: map[string]*api.Pod{}, updates: make(chan *PodEvent, 100)}
}

func (r *fakeRegistry) empty() bool {
	r.Lock()
	defer r.Unlock()
	return len(r.boundTasks) == 0
}

func (r *fakeRegistry) pod(taskID string) *api.Pod {
	r.Lock()
	defer r.Unlock()
	return r.boundTasks[taskID]
}

func (r *fakeRegistry) watch() <-chan *PodEvent { return r.updates }

func (r *fakeRegistry) shutdown() {
	r.Lock()
	defer r.Unlock()
	r.boundTasks = map[string]*api.Pod{}
}

func (r *fakeRegistry) bind(taskID string, pod *api.Pod) error {
	r.Lock()
	defer r.Unlock()
	pod.Annotations = map[string]string{
		"k8s.mesosphere.io/taskId": taskID,
	}
	r.boundTasks[taskID] = pod

	// the normal registry sends a bind..
	r.updates <- &PodEvent{pod: pod, taskID: taskID, eventType: PodEventBound}
	return nil
}

func (r *fakeRegistry) Update(pod *api.Pod) (*PodEvent, error) {
	r.Lock()
	defer r.Unlock()
	taskID, err := taskIDFor(pod)
	if err != nil {
		return nil, err
	}
	if _, ok := r.boundTasks[taskID]; !ok {
		return nil, errUnknownTask
	}
	rp := &PodEvent{pod: pod, taskID: taskID, eventType: PodEventUpdated}
	r.updates <- rp
	return rp, nil
}

func (r *fakeRegistry) Remove(taskID string) error {
	r.Lock()
	defer r.Unlock()
	pod, ok := r.boundTasks[taskID]
	if !ok {
		return errUnknownTask
	}
	delete(r.boundTasks, taskID)
	r.updates <- &PodEvent{pod: pod, taskID: taskID, eventType: PodEventDeleted}
	return nil
}

// phaseChange simulates a pod source update; normally this update is generated from a watch
func (r *fakeRegistry) phaseChange(pod *api.Pod, phase api.PodPhase) error {
	clone, err := api.Scheme.DeepCopy(pod)
	if err != nil {
		return err
	}

	phasedPod := clone.(*api.Pod)
	phasedPod.Status.Phase = phase
	_, err = r.Update(phasedPod)
	return err
}

// TestExecutorLaunchAndKillTask ensures that the executor is able to launch tasks and generates
// appropriate status messages for mesos. It then kills the task and validates that appropriate
// actions are taken by the executor.
func TestExecutorLaunchAndKillTask(t *testing.T) {
	var (
		mockDriver = &MockExecutorDriver{}
		registry   = newFakeRegistry()
		executor   = New(Config{
			Docker:    dockertools.ConnectToDockerOrDie("fake://", 0),
			NodeInfos: make(chan NodeInfo, 1),
			Registry:  registry,
		})
		mockKubeAPI  = &mockKubeAPI{}
		pod          = NewTestPod(1)
		executorinfo = &mesosproto.ExecutorInfo{}
	)
	executor.kubeAPI = mockKubeAPI
	executor.Init(mockDriver)
	executor.Registered(mockDriver, nil, nil, nil)

	podTask, err := podtask.New(
		api.NewDefaultContext(),
		podtask.Config{
			Prototype:        executorinfo,
			HostPortStrategy: hostport.StrategyWildcard,
		},
		pod,
	)
	assert.Equal(t, nil, err, "must be able to create a task from a pod")

	pod.Annotations = map[string]string{
		"k8s.mesosphere.io/taskId": podTask.ID,
	}

	podTask.Spec = &podtask.Spec{Executor: executorinfo}
	taskInfo, err := podTask.BuildTaskInfo()
	assert.Equal(t, nil, err, "must be able to build task info")

	data, err := runtime.Encode(testapi.Default.Codec(), pod)
	assert.Equal(t, nil, err, "must be able to encode a pod's spec data")

	taskInfo.Data = data
	var statusUpdateCalls sync.WaitGroup
	statusUpdateCalls.Add(1)
	statusUpdateDone := func(_ mock.Arguments) { statusUpdateCalls.Done() }

	mockDriver.On(
		"SendStatusUpdate",
		mesosproto.TaskState_TASK_STARTING,
	).Return(mesosproto.Status_DRIVER_RUNNING, nil).Run(statusUpdateDone).Once()

	statusUpdateCalls.Add(1)
	mockDriver.On(
		"SendStatusUpdate",
		mesosproto.TaskState_TASK_RUNNING,
	).Return(mesosproto.Status_DRIVER_RUNNING, nil).Run(statusUpdateDone).Once()

	executor.LaunchTask(mockDriver, taskInfo)

	assertext.EventuallyTrue(t, wait.ForeverTestTimeout, func() bool {
		executor.lock.Lock()
		defer executor.lock.Unlock()
		return !registry.empty()
	}, "executor must be able to create a task and a pod")

	// simulate a pod source update; normally this update is generated when binding a pod
	err = registry.phaseChange(pod, api.PodPending)
	assert.NoError(t, err)

	// simulate a pod source update; normally this update is generated by the kubelet once the pod is healthy
	err = registry.phaseChange(pod, api.PodRunning)
	assert.NoError(t, err)

	// Allow some time for asynchronous requests to the driver.
	finished := kmruntime.After(statusUpdateCalls.Wait)
	select {
	case <-finished:
	case <-time.After(wait.ForeverTestTimeout):
		t.Fatalf("timed out waiting for status update calls to finish")
	}

	statusUpdateCalls.Add(1)
	mockDriver.On(
		"SendStatusUpdate",
		mesosproto.TaskState_TASK_KILLED,
	).Return(mesosproto.Status_DRIVER_RUNNING, nil).Run(statusUpdateDone).Once()

	// simulate what happens when the apiserver is told to delete a pod
	mockKubeAPI.On("killPod", pod.Namespace, pod.Name).Return(nil).Run(func(_ mock.Arguments) {
		registry.Remove(podTask.ID)
	})

	executor.KillTask(mockDriver, taskInfo.TaskId)
	assertext.EventuallyTrue(t, wait.ForeverTestTimeout, func() bool {
		executor.lock.Lock()
		defer executor.lock.Unlock()
		return registry.empty()
	}, "executor must be able to kill a created task and pod")

	// Allow some time for asynchronous requests to the driver.
	finished = kmruntime.After(statusUpdateCalls.Wait)
	select {
	case <-finished:
	case <-time.After(wait.ForeverTestTimeout):
		t.Fatalf("timed out waiting for status update calls to finish")
	}

	mockDriver.AssertExpectations(t)
	mockKubeAPI.AssertExpectations(t)
}

// TestExecutorStaticPods test that the ExecutorInfo.data is parsed
// as a zip archive with pod definitions.
func TestExecutorInitializeStaticPodsSource(t *testing.T) {
	// create some zip with static pod definition
	givenPodsDir, err := utiltesting.MkTmpdir("executor-givenpods")
	assert.NoError(t, err)
	defer os.RemoveAll(givenPodsDir)

	var wg sync.WaitGroup
	reportErrors := func(errCh <-chan error) {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for err := range errCh {
				t.Error(err)
			}
		}()
	}

	createStaticPodFile := func(fileName, name string) {
		spod := `{
			"apiVersion": "v1",
			"kind": "Pod",
			"metadata": {
				"name": "%v",
				"namespace": "staticpods",
				"labels": { "name": "foo", "cluster": "bar" }
			},
			"spec": {
				"containers": [{
					"name": "%v",
					"image": "library/nginx",
					"ports": [{ "containerPort": 80, "name": "http" }]
				}]
			}
		}`
		destfile := filepath.Join(givenPodsDir, fileName)
		err = os.MkdirAll(filepath.Dir(destfile), 0770)
		assert.NoError(t, err)
		err = ioutil.WriteFile(destfile, []byte(fmt.Sprintf(spod, name, name)), 0660)
		assert.NoError(t, err)
	}

	createStaticPodFile("spod.json", "spod-01")
	createStaticPodFile("spod2.json", "spod-02")
	createStaticPodFile("dir/spod.json", "spod-03") // same file name as first one to check for overwriting
	staticpods, errs := podutil.ReadFromDir(givenPodsDir)
	reportErrors(errs)

	gzipped, err := podutil.Gzip(staticpods)
	assert.NoError(t, err)

	expectedStaticPodsNum := 2 // subdirectories are ignored by FileSource, hence only 2

	// temporary directory which is normally located in the executor sandbox
	staticPodsConfigPath, err := utiltesting.MkTmpdir("executor-k8sm-archive")
	assert.NoError(t, err)
	defer os.RemoveAll(staticPodsConfigPath)

	executor := &Executor{
		staticPodsConfigPath: staticPodsConfigPath,
	}

	// extract the pods into staticPodsConfigPath
	executor.initializeStaticPodsSource(&mesosproto.ExecutorInfo{Data: gzipped})

	actualpods, errs := podutil.ReadFromDir(staticPodsConfigPath)
	reportErrors(errs)

	list := podutil.List(actualpods)
	assert.NotNil(t, list)
	assert.Equal(t, expectedStaticPodsNum, len(list.Items))

	var (
		expectedNames = map[string]struct{}{
			"spod-01": {},
			"spod-02": {},
		}
		actualNames = map[string]struct{}{}
	)
	for _, pod := range list.Items {
		actualNames[pod.Name] = struct{}{}
	}
	assert.True(t, reflect.DeepEqual(expectedNames, actualNames), "expected %v instead of %v", expectedNames, actualNames)

	wg.Wait()
}

// TestExecutorFrameworkMessage ensures that the executor is able to
// handle messages from the framework, specifically about lost tasks
// and Kamikaze.  When a task is lost, the executor needs to clean up
// its state.  When a Kamikaze message is received, the executor should
// attempt suicide.
func TestExecutorFrameworkMessage(t *testing.T) {
	// TODO(jdef): Fix the unexpected call in the mocking system.
	t.Skip("This test started failing when panic catching was disabled.")
	var (
		mockDriver      = &MockExecutorDriver{}
		kubeletFinished = make(chan struct{})
		registry        = newFakeRegistry()
		executor        = New(Config{
			Docker:    dockertools.ConnectToDockerOrDie("fake://", 0),
			NodeInfos: make(chan NodeInfo, 1),
			ShutdownAlert: func() {
				close(kubeletFinished)
			},
			KubeletFinished: kubeletFinished,
			Registry:        registry,
		})
		pod         = NewTestPod(1)
		mockKubeAPI = &mockKubeAPI{}
	)

	executor.kubeAPI = mockKubeAPI
	executor.Init(mockDriver)
	executor.Registered(mockDriver, nil, nil, nil)
	executor.FrameworkMessage(mockDriver, "test framework message")

	// set up a pod to then lose
	executorinfo := &mesosproto.ExecutorInfo{}
	podTask, _ := podtask.New(
		api.NewDefaultContext(),
		podtask.Config{
			ID:               "foo",
			Prototype:        executorinfo,
			HostPortStrategy: hostport.StrategyWildcard,
		},
		pod,
	)
	pod.Annotations = map[string]string{
		"k8s.mesosphere.io/taskId": podTask.ID,
	}
	podTask.Spec = &podtask.Spec{
		Executor: executorinfo,
	}

	taskInfo, err := podTask.BuildTaskInfo()
	assert.Equal(t, nil, err, "must be able to build task info")

	data, _ := runtime.Encode(testapi.Default.Codec(), pod)
	taskInfo.Data = data

	mockDriver.On(
		"SendStatusUpdate",
		mesosproto.TaskState_TASK_STARTING,
	).Return(mesosproto.Status_DRIVER_RUNNING, nil).Once()

	called := make(chan struct{})
	mockDriver.On(
		"SendStatusUpdate",
		mesosproto.TaskState_TASK_RUNNING,
	).Return(mesosproto.Status_DRIVER_RUNNING, nil).Run(func(_ mock.Arguments) { close(called) }).Once()

	executor.LaunchTask(mockDriver, taskInfo)

	// must wait for this otherwise phase changes may not apply
	assertext.EventuallyTrue(t, wait.ForeverTestTimeout, func() bool {
		executor.lock.Lock()
		defer executor.lock.Unlock()
		return !registry.empty()
	}, "executor must be able to create a task and a pod")

	err = registry.phaseChange(pod, api.PodPending)
	assert.NoError(t, err)
	err = registry.phaseChange(pod, api.PodRunning)
	assert.NoError(t, err)

	// waiting until the pod is really running b/c otherwise a TASK_FAILED could be
	// triggered by the asynchronously running executor methods when removing the task
	// from k.tasks through the "task-lost:foo" message below.
	select {
	case <-called:
	case <-time.After(wait.ForeverTestTimeout):
		t.Fatalf("timed out waiting for SendStatusUpdate for the running task")
	}

	// send task-lost message for it
	called = make(chan struct{})
	mockDriver.On(
		"SendStatusUpdate",
		mesosproto.TaskState_TASK_LOST,
	).Return(mesosproto.Status_DRIVER_RUNNING, nil).Run(func(_ mock.Arguments) { close(called) }).Once()

	// simulate what happens when the apiserver is told to delete a pod
	mockKubeAPI.On("killPod", pod.Namespace, pod.Name).Return(nil).Run(func(_ mock.Arguments) {
		registry.Remove(podTask.ID)
	})

	executor.FrameworkMessage(mockDriver, "task-lost:foo")

	assertext.EventuallyTrue(t, wait.ForeverTestTimeout, func() bool {
		executor.lock.Lock()
		defer executor.lock.Unlock()
		return registry.empty()
	}, "executor must be able to kill a created task and pod")

	select {
	case <-called:
	case <-time.After(wait.ForeverTestTimeout):
		t.Fatalf("timed out waiting for SendStatusUpdate")
	}

	mockDriver.On("Stop").Return(mesosproto.Status_DRIVER_STOPPED, nil).Once()

	executor.FrameworkMessage(mockDriver, messages.Kamikaze)
	assert.Equal(t, true, executor.isDone(),
		"executor should have shut down after receiving a Kamikaze message")

	mockDriver.AssertExpectations(t)
	mockKubeAPI.AssertExpectations(t)
}

// Create a pod with a given index, requiring one port
func NewTestPod(i int) *api.Pod {
	name := fmt.Sprintf("pod%d", i)
	return &api.Pod{
		TypeMeta: unversioned.TypeMeta{APIVersion: testapi.Default.GroupVersion().String()},
		ObjectMeta: api.ObjectMeta{
			Name:      name,
			Namespace: api.NamespaceDefault,
			SelfLink:  testapi.Default.SelfLink("pods", string(i)),
		},
		Spec: api.PodSpec{
			Containers: []api.Container{
				{
					Name: "foo",
					Ports: []api.ContainerPort{
						{
							ContainerPort: int32(8000 + i),
							Protocol:      api.ProtocolTCP,
						},
					},
				},
			},
		},
		Status: api.PodStatus{
			Conditions: []api.PodCondition{
				{
					Type:   api.PodReady,
					Status: api.ConditionTrue,
				},
			},
		},
	}
}

// Create mock of pods ListWatch, usually listening on the apiserver pods watch endpoint
type MockPodsListWatch struct {
	ListWatch   cache.ListWatch
	fakeWatcher *watch.FakeWatcher
	list        api.PodList
}

// A apiserver mock which partially mocks the pods API
type TestServer struct {
	server *httptest.Server
	Stats  map[string]uint
	lock   sync.Mutex
}

func NewTestServer(t *testing.T, namespace string) *TestServer {
	ts := TestServer{
		Stats: map[string]uint{},
	}
	mux := http.NewServeMux()

	mux.HandleFunc(testapi.Default.ResourcePath("bindings", namespace, ""), func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
	})

	ts.server = httptest.NewServer(mux)
	return &ts
}

func NewMockPodsListWatch(initialPodList api.PodList) *MockPodsListWatch {
	lw := MockPodsListWatch{
		fakeWatcher: watch.NewFake(),
		list:        initialPodList,
	}
	lw.ListWatch = cache.ListWatch{
		WatchFunc: func(options api.ListOptions) (watch.Interface, error) {
			return lw.fakeWatcher, nil
		},
		ListFunc: func(options api.ListOptions) (runtime.Object, error) {
			return &lw.list, nil
		},
	}
	return &lw
}

// TestExecutorShutdown ensures that the executor properly shuts down
// when Shutdown is called.
func TestExecutorShutdown(t *testing.T) {
	var (
		mockDriver      = &MockExecutorDriver{}
		kubeletFinished = make(chan struct{})
		exitCalled      = int32(0)
		executor        = New(Config{
			Docker:    dockertools.ConnectToDockerOrDie("fake://", 0),
			NodeInfos: make(chan NodeInfo, 1),
			ShutdownAlert: func() {
				close(kubeletFinished)
			},
			KubeletFinished: kubeletFinished,
			ExitFunc: func(_ int) {
				atomic.AddInt32(&exitCalled, 1)
			},
			Registry: newFakeRegistry(),
		})
	)

	executor.Init(mockDriver)
	executor.Registered(mockDriver, nil, nil, nil)
	mockDriver.On("Stop").Return(mesosproto.Status_DRIVER_STOPPED, nil).Once()
	executor.Shutdown(mockDriver)

	assert.Equal(t, false, executor.isConnected(),
		"executor should not be connected after Shutdown")
	assert.Equal(t, true, executor.isDone(),
		"executor should be in Done state after Shutdown")
	assert.Equal(t, true, atomic.LoadInt32(&exitCalled) > 0,
		"the executor should call its ExitFunc when it is ready to close down")
	mockDriver.AssertExpectations(t)
}

func TestExecutorsendFrameworkMessage(t *testing.T) {
	mockDriver := &MockExecutorDriver{}
	executor := NewTestKubernetesExecutor()

	executor.Init(mockDriver)
	executor.Registered(mockDriver, nil, nil, nil)

	called := make(chan struct{})
	mockDriver.On(
		"SendFrameworkMessage",
		"foo bar baz",
	).Return(mesosproto.Status_DRIVER_RUNNING, nil).Run(func(_ mock.Arguments) { close(called) }).Once()
	executor.sendFrameworkMessage(mockDriver, "foo bar baz")

	// guard against data race in mock driver between AssertExpectations and Called
	select {
	case <-called: // expected
	case <-time.After(wait.ForeverTestTimeout):
		t.Fatalf("expected call to SendFrameworkMessage")
	}
	mockDriver.AssertExpectations(t)
}
