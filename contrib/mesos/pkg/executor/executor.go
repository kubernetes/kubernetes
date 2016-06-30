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
	"bytes"
	"encoding/json"
	"fmt"
	"math"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	clientset "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset"

	dockertypes "github.com/docker/engine-api/types"
	"github.com/gogo/protobuf/proto"
	log "github.com/golang/glog"
	bindings "github.com/mesos/mesos-go/executor"
	mesos "github.com/mesos/mesos-go/mesosproto"
	mutil "github.com/mesos/mesos-go/mesosutil"
	"k8s.io/kubernetes/contrib/mesos/pkg/executor/messages"
	"k8s.io/kubernetes/contrib/mesos/pkg/node"
	"k8s.io/kubernetes/contrib/mesos/pkg/podutil"
	"k8s.io/kubernetes/contrib/mesos/pkg/runtime"
	"k8s.io/kubernetes/contrib/mesos/pkg/scheduler/executorinfo"
	"k8s.io/kubernetes/contrib/mesos/pkg/scheduler/meta"
	"k8s.io/kubernetes/pkg/api"
	apierrors "k8s.io/kubernetes/pkg/api/errors"
	"k8s.io/kubernetes/pkg/kubelet/container"
	"k8s.io/kubernetes/pkg/kubelet/dockertools"
	kruntime "k8s.io/kubernetes/pkg/runtime"
	utilruntime "k8s.io/kubernetes/pkg/util/runtime"
)

type stateType int32

const (
	disconnectedState stateType = iota
	connectedState
	suicidalState
	terminalState
)

func (s *stateType) get() stateType {
	return stateType(atomic.LoadInt32((*int32)(s)))
}

func (s *stateType) transition(from, to stateType) bool {
	return atomic.CompareAndSwapInt32((*int32)(s), int32(from), int32(to))
}

func (s *stateType) transitionTo(to stateType, unless ...stateType) bool {
	if len(unless) == 0 {
		atomic.StoreInt32((*int32)(s), int32(to))
		return true
	}
	for {
		state := s.get()
		for _, x := range unless {
			if state == x {
				return false
			}
		}
		if s.transition(state, to) {
			return true
		}
	}
}

// KubernetesExecutor is an mesos executor that runs pods
// in a minion machine.
type Executor struct {
	state                stateType
	lock                 sync.Mutex
	terminate            chan struct{}                     // signals that the executor is shutting down
	outgoing             chan func() (mesos.Status, error) // outgoing queue to the mesos driver
	dockerClient         dockertools.DockerInterface
	suicideWatch         suicideWatcher
	suicideTimeout       time.Duration
	shutdownAlert        func()          // invoked just prior to executor shutdown
	kubeletFinished      <-chan struct{} // signals that kubelet Run() died
	exitFunc             func(int)
	staticPodsConfigPath string
	staticPodsFilters    podutil.Filters
	launchGracePeriod    time.Duration
	nodeInfos            chan<- NodeInfo
	initCompleted        chan struct{} // closes upon completion of Init()
	registry             Registry
	watcher              *watcher
	kubeAPI              kubeAPI
	nodeAPI              nodeAPI
}

type Config struct {
	APIClient         *clientset.Clientset
	Docker            dockertools.DockerInterface
	ShutdownAlert     func()
	SuicideTimeout    time.Duration
	KubeletFinished   <-chan struct{} // signals that kubelet Run() died
	ExitFunc          func(int)
	LaunchGracePeriod time.Duration
	NodeInfos         chan<- NodeInfo
	Registry          Registry
	Options           []Option // functional options
}

// Option is a functional option type for Executor
type Option func(*Executor)

func (k *Executor) isConnected() bool {
	return connectedState == (&k.state).get()
}

// New creates a new kubernetes executor.
func New(config Config) *Executor {
	launchGracePeriod := config.LaunchGracePeriod
	if launchGracePeriod == 0 {
		// this is the equivalent of saying "the timer never expires" and simplies nil
		// timer checks elsewhere in the code. it's a little hacky but less code to
		// maintain that alternative approaches.
		launchGracePeriod = time.Duration(math.MaxInt64)
	}
	k := &Executor{
		state:             disconnectedState,
		terminate:         make(chan struct{}),
		outgoing:          make(chan func() (mesos.Status, error), 1024),
		dockerClient:      config.Docker,
		suicideTimeout:    config.SuicideTimeout,
		kubeletFinished:   config.KubeletFinished,
		suicideWatch:      &suicideTimer{},
		shutdownAlert:     config.ShutdownAlert,
		exitFunc:          config.ExitFunc,
		launchGracePeriod: launchGracePeriod,
		nodeInfos:         config.NodeInfos,
		initCompleted:     make(chan struct{}),
		registry:          config.Registry,
	}
	if config.APIClient != nil {
		k.kubeAPI = &clientAPIWrapper{config.APIClient.Core()}
		k.nodeAPI = &clientAPIWrapper{config.APIClient.Core()}
	}

	// apply functional options
	for _, opt := range config.Options {
		opt(k)
	}

	runtime.On(k.initCompleted, k.runSendLoop)

	k.watcher = newWatcher(k.registry.watch())
	runtime.On(k.initCompleted, k.watcher.run)

	return k
}

// StaticPods creates a static pods Option for an Executor
func StaticPods(configPath string, f podutil.Filters) Option {
	return func(k *Executor) {
		k.staticPodsFilters = f
		k.staticPodsConfigPath = configPath
	}
}

// Done returns a chan that closes when the executor is shutting down
func (k *Executor) Done() <-chan struct{} {
	return k.terminate
}

func (k *Executor) Init(driver bindings.ExecutorDriver) {
	defer close(k.initCompleted)

	k.killKubeletContainers()
	k.resetSuicideWatch(driver)

	k.watcher.addFilter(func(podEvent *PodEvent) bool {
		switch podEvent.eventType {
		case PodEventIncompatibleUpdate:
			log.Warningf("killing %s because of an incompatible update", podEvent.FormatShort())
			k.killPodTask(driver, podEvent.taskID)
			// halt processing of this event; when the pod is deleted we'll receive another
			// event for that.
			return false

		case PodEventDeleted:
			// an active pod-task was deleted, alert mesos:
			// send back a TASK_KILLED status, we completed the pod-task lifecycle normally.
			k.resetSuicideWatch(driver)
			k.sendStatus(driver, newStatus(mutil.NewTaskID(podEvent.taskID), mesos.TaskState_TASK_KILLED, "pod-deleted"))
		}
		return true
	})

	//TODO(jdef) monitor kubeletFinished and shutdown if it happens
}

func (k *Executor) isDone() bool {
	select {
	case <-k.terminate:
		return true
	default:
		return false
	}
}

// Registered is called when the executor is successfully registered with the slave.
func (k *Executor) Registered(
	driver bindings.ExecutorDriver,
	executorInfo *mesos.ExecutorInfo,
	frameworkInfo *mesos.FrameworkInfo,
	slaveInfo *mesos.SlaveInfo,
) {
	if k.isDone() {
		return
	}

	log.Infof(
		"Executor %v of framework %v registered with slave %v\n",
		executorInfo, frameworkInfo, slaveInfo,
	)

	if !(&k.state).transition(disconnectedState, connectedState) {
		log.Errorf("failed to register/transition to a connected state")
	}

	k.initializeStaticPodsSource(executorInfo)

	annotations, err := annotationsFor(executorInfo)
	if err != nil {
		log.Errorf(
			"cannot get node annotations from executor info %v error %v",
			executorInfo, err,
		)
	}

	if slaveInfo != nil {
		_, err := k.nodeAPI.createOrUpdate(
			slaveInfo.GetHostname(),
			node.SlaveAttributesToLabels(slaveInfo.Attributes),
			annotations,
		)

		if err != nil {
			log.Errorf("cannot update node labels: %v", err)
		}
	}

	k.lock.Lock()
	defer k.lock.Unlock()

	if slaveInfo != nil && k.nodeInfos != nil {
		k.nodeInfos <- nodeInfo(slaveInfo, executorInfo) // leave it behind the upper lock to avoid panics
	}
}

// Reregistered is called when the executor is successfully re-registered with the slave.
// This can happen when the slave fails over.
func (k *Executor) Reregistered(driver bindings.ExecutorDriver, slaveInfo *mesos.SlaveInfo) {
	if k.isDone() {
		return
	}
	log.Infof("Reregistered with slave %v\n", slaveInfo)
	if !(&k.state).transition(disconnectedState, connectedState) {
		log.Errorf("failed to reregister/transition to a connected state")
	}

	if slaveInfo != nil {
		_, err := k.nodeAPI.createOrUpdate(
			slaveInfo.GetHostname(),
			node.SlaveAttributesToLabels(slaveInfo.Attributes),
			nil, // don't change annotations
		)

		if err != nil {
			log.Errorf("cannot update node labels: %v", err)
		}
	}

	if slaveInfo != nil && k.nodeInfos != nil {
		// make sure nodeInfos is not nil and send new NodeInfo
		k.lock.Lock()
		defer k.lock.Unlock()
		if k.isDone() {
			return
		}
		k.nodeInfos <- nodeInfo(slaveInfo, nil)
	}
}

// initializeStaticPodsSource unzips the data slice into the static-pods directory
func (k *Executor) initializeStaticPodsSource(executorInfo *mesos.ExecutorInfo) {
	if data := executorInfo.GetData(); len(data) > 0 && k.staticPodsConfigPath != "" {
		log.V(2).Infof("extracting static pods config to %s", k.staticPodsConfigPath)
		err := podutil.WriteToDir(
			k.staticPodsFilters.Do(podutil.Gunzip(executorInfo.Data)),
			k.staticPodsConfigPath,
		)
		if err != nil {
			log.Errorf("failed to initialize static pod configuration: %v", err)
		}
	}
}

// Disconnected is called when the executor is disconnected from the slave.
func (k *Executor) Disconnected(driver bindings.ExecutorDriver) {
	if k.isDone() {
		return
	}
	log.Infof("Slave is disconnected\n")
	if !(&k.state).transition(connectedState, disconnectedState) {
		log.Errorf("failed to disconnect/transition to a disconnected state")
	}
}

// LaunchTask is called when the executor receives a request to launch a task.
// The happens when the k8sm scheduler has decided to schedule the pod
// (which corresponds to a Mesos Task) onto the node where this executor
// is running, but the binding is not recorded in the Kubernetes store yet.
// This function is invoked to tell the executor to record the binding in the
// Kubernetes store and start the pod via the Kubelet.
func (k *Executor) LaunchTask(driver bindings.ExecutorDriver, taskInfo *mesos.TaskInfo) {
	if k.isDone() {
		return
	}

	log.Infof("Launch task %v\n", taskInfo)

	taskID := taskInfo.GetTaskId().GetValue()
	if p := k.registry.pod(taskID); p != nil {
		log.Warningf("task %v already launched", taskID)
		// Not to send back TASK_RUNNING or TASK_FAILED here, because
		// may be duplicated messages
		return
	}

	if !k.isConnected() {
		log.Errorf("Ignore launch task because the executor is disconnected\n")
		k.sendStatus(driver, newStatus(taskInfo.GetTaskId(), mesos.TaskState_TASK_FAILED,
			messages.ExecutorUnregistered))
		return
	}

	obj, err := kruntime.Decode(api.Codecs.UniversalDecoder(), taskInfo.GetData())
	if err != nil {
		log.Errorf("failed to extract yaml data from the taskInfo.data %v", err)
		k.sendStatus(driver, newStatus(taskInfo.GetTaskId(), mesos.TaskState_TASK_FAILED,
			messages.UnmarshalTaskDataFailure))
		return
	}
	pod, ok := obj.(*api.Pod)
	if !ok {
		log.Errorf("expected *api.Pod instead of %T: %+v", pod, pod)
		k.sendStatus(driver, newStatus(taskInfo.GetTaskId(), mesos.TaskState_TASK_FAILED,
			messages.UnmarshalTaskDataFailure))
		return
	}

	k.resetSuicideWatch(driver)

	// run the next step aync because it calls out to apiserver and we don't want to block here
	go k.bindAndWatchTask(driver, taskInfo, time.NewTimer(k.launchGracePeriod), pod)
}

// determine whether we need to start a suicide countdown. if so, then start
// a timer that, upon expiration, causes this executor to commit suicide.
// this implementation runs asynchronously. callers that wish to wait for the
// reset to complete may wait for the returned signal chan to close.
func (k *Executor) resetSuicideWatch(driver bindings.ExecutorDriver) <-chan struct{} {
	ch := make(chan struct{})
	go func() {
		defer close(ch)
		k.lock.Lock()
		defer k.lock.Unlock()

		if k.suicideTimeout < 1 {
			return
		}

		if k.suicideWatch != nil {
			if !k.registry.empty() {
				k.suicideWatch.Stop()
				return
			}
			if k.suicideWatch.Reset(k.suicideTimeout) {
				// valid timer, reset was successful
				return
			}
		}

		//TODO(jdef) reduce verbosity here once we're convinced that suicide watch is working properly
		log.Infof("resetting suicide watch timer for %v", k.suicideTimeout)

		k.suicideWatch = k.suicideWatch.Next(k.suicideTimeout, driver, jumper(k.attemptSuicide))
	}()
	return ch
}

func (k *Executor) attemptSuicide(driver bindings.ExecutorDriver, abort <-chan struct{}) {
	k.lock.Lock()
	defer k.lock.Unlock()

	// this attempt may have been queued and since been aborted
	select {
	case <-abort:
		//TODO(jdef) reduce verbosity once suicide watch is working properly
		log.Infof("aborting suicide attempt since watch was cancelled")
		return
	default: // continue
	}

	// fail-safe, will abort kamikaze attempts if there are tasks
	if !k.registry.empty() {
		log.Errorf("suicide attempt failed, there are still running tasks")
		return
	}

	log.Infoln("Attempting suicide")
	if (&k.state).transitionTo(suicidalState, suicidalState, terminalState) {
		//TODO(jdef) let the scheduler know?
		//TODO(jdef) is suicide more graceful than slave-demanded shutdown?
		k.doShutdown(driver)
	}
}

func podStatusData(pod *api.Pod, status api.PodStatus) ([]byte, string, error) {
	podFullName := container.GetPodFullName(pod)
	data, err := json.Marshal(api.PodStatusResult{
		ObjectMeta: api.ObjectMeta{
			Name:     podFullName,
			SelfLink: "/podstatusresult",
		},
		Status: status,
	})
	return data, podFullName, err
}

// async continuation of LaunchTask
func (k *Executor) bindAndWatchTask(driver bindings.ExecutorDriver, task *mesos.TaskInfo, launchTimer *time.Timer, pod *api.Pod) {
	success := false
	defer func() {
		if !success {
			k.killPodTask(driver, task.TaskId.GetValue())
			k.resetSuicideWatch(driver)
		}
	}()

	// allow a recently failed-over scheduler the chance to recover the task/pod binding:
	// it may have failed and recovered before the apiserver is able to report the updated
	// binding information. replays of this status event will signal to the scheduler that
	// the apiserver should be up-to-date.
	startingData, _, err := podStatusData(pod, api.PodStatus{})
	if err != nil {
		log.Errorf("failed to generate pod-task starting data for task %v pod %v/%v: %v",
			task.TaskId.GetValue(), pod.Namespace, pod.Name, err)
		k.sendStatus(driver, newStatus(task.TaskId, mesos.TaskState_TASK_FAILED, err.Error()))
		return
	}

	err = k.registry.bind(task.TaskId.GetValue(), pod)
	if err != nil {
		log.Errorf("failed to bind task %v pod %v/%v: %v",
			task.TaskId.GetValue(), pod.Namespace, pod.Name, err)
		k.sendStatus(driver, newStatus(task.TaskId, mesos.TaskState_TASK_FAILED, err.Error()))
		return
	}

	// send TASK_STARTING
	k.sendStatus(driver, &mesos.TaskStatus{
		TaskId:  task.TaskId,
		State:   mesos.TaskState_TASK_STARTING.Enum(),
		Message: proto.String(messages.CreateBindingSuccess),
		Data:    startingData,
	})

	// within the launch timeout window we should see a pod-task update via the registry.
	// if we see a Running update then we need to generate a TASK_RUNNING status update for mesos.
	handlerFinished := false
	handler := &watchHandler{
		expiration: watchExpiration{
			timeout: launchTimer.C,
			onEvent: func(taskID string) {
				if !handlerFinished {
					// launch timeout expired
					k.killPodTask(driver, task.TaskId.GetValue())
				}
			},
		},
		onEvent: func(podEvent *PodEvent) (bool, error) {
			switch podEvent.eventType {
			case PodEventUpdated:
				log.V(2).Infof("Found status: '%v' for %s", podEvent.pod.Status, podEvent.FormatShort())

				if podEvent.pod.Status.Phase != api.PodRunning {
					// still waiting for pod to transition to a running state, so
					// we're not done monitoring yet; check back later..
					break
				}

				data, podFullName, err := podStatusData(podEvent.pod, podEvent.pod.Status)
				if err != nil {
					return false, fmt.Errorf("failed to marshal pod status result: %v", err)
				}

				defer k.sendStatus(driver, &mesos.TaskStatus{
					TaskId:  task.TaskId,
					State:   mesos.TaskState_TASK_RUNNING.Enum(),
					Message: proto.String("pod-running:" + podFullName),
					Data:    data,
				})
				fallthrough

			case PodEventDeleted:
				// we're done monitoring because pod has been deleted
				handlerFinished = true
				launchTimer.Stop()
			}
			return handlerFinished, nil
		},
	}
	k.watcher.forTask(task.TaskId.GetValue(), handler)
	success = true
}

// KillTask is called when the executor receives a request to kill a task.
func (k *Executor) KillTask(driver bindings.ExecutorDriver, taskId *mesos.TaskID) {
	k.killPodTask(driver, taskId.GetValue())
}

// deletes the pod and task associated with the task identified by taskID and sends a task
// status update to mesos. also attempts to reset the suicide watch.
func (k *Executor) killPodTask(driver bindings.ExecutorDriver, taskID string) {
	pod := k.registry.pod(taskID)
	if pod == nil {
		log.V(1).Infof("Failed to remove task, unknown task %v\n", taskID)
		k.sendStatus(driver, newStatus(&mesos.TaskID{Value: &taskID}, mesos.TaskState_TASK_LOST, "kill-pod-task"))
		return
	}

	// force-delete the pod from the API server
	// TODO(jdef) possibly re-use eviction code from stock k8s once it lands?
	err := k.kubeAPI.killPod(pod.Namespace, pod.Name)
	if err != nil {
		log.V(1).Infof("failed to delete task %v pod %v/%v from apiserver: %+v", taskID, pod.Namespace, pod.Name, err)
		if apierrors.IsNotFound(err) {
			k.sendStatus(driver, newStatus(&mesos.TaskID{Value: &taskID}, mesos.TaskState_TASK_LOST, "kill-pod-task"))
		}
	}
}

// FrameworkMessage is called when the framework sends some message to the executor
func (k *Executor) FrameworkMessage(driver bindings.ExecutorDriver, message string) {
	if k.isDone() {
		return
	}
	if !k.isConnected() {
		log.Warningf("Ignore framework message because the executor is disconnected\n")
		return
	}

	log.Infof("Receives message from framework %v\n", message)
	//TODO(jdef) master reported a lost task, reconcile this! @see framework.go:handleTaskLost
	if strings.HasPrefix(message, messages.TaskLost+":") {
		taskId := message[len(messages.TaskLost)+1:]
		if taskId != "" {
			// TODO(jdef) would it make more sense to check the status of the task and
			// just replay the last non-terminal message that we sent if the task is
			// still active?

			// clean up pod state
			k.sendStatus(driver, newStatus(&mesos.TaskID{Value: &taskId}, mesos.TaskState_TASK_LOST, messages.TaskLostAck))
			k.killPodTask(driver, taskId)
		}
		return
	}

	switch message {
	case messages.Kamikaze:
		k.attemptSuicide(driver, nil)
	}
}

// Shutdown is called when the executor receives a shutdown request.
func (k *Executor) Shutdown(driver bindings.ExecutorDriver) {
	k.lock.Lock()
	defer k.lock.Unlock()
	k.doShutdown(driver)
}

// assumes that caller has obtained state lock
func (k *Executor) doShutdown(driver bindings.ExecutorDriver) {
	defer func() {
		log.Errorf("exiting with unclean shutdown: %v", recover())
		if k.exitFunc != nil {
			k.exitFunc(1)
		}
	}()

	(&k.state).transitionTo(terminalState)

	// signal to all listeners that this KubeletExecutor is done!
	close(k.terminate)
	close(k.nodeInfos)

	if k.shutdownAlert != nil {
		func() {
			utilruntime.HandleCrash()
			k.shutdownAlert()
		}()
	}

	log.Infoln("Stopping executor driver")
	_, err := driver.Stop()
	if err != nil {
		log.Warningf("failed to stop executor driver: %v", err)
	}

	log.Infoln("Shutdown the executor")

	// according to docs, mesos will generate TASK_LOST updates for us
	// if needed, so don't take extra time to do that here.
	k.registry.shutdown()

	select {
	// the main Run() func may still be running... wait for it to finish: it will
	// clear the pod configuration cleanly, telling k8s "there are no pods" and
	// clean up resources (pods, volumes, etc).
	case <-k.kubeletFinished:

	//TODO(jdef) attempt to wait for events to propagate to API server?

	// TODO(jdef) extract constant, should be smaller than whatever the
	// slave graceful shutdown timeout period is.
	case <-time.After(15 * time.Second):
		log.Errorf("timed out waiting for kubelet Run() to die")
	}
	log.Infoln("exiting")
	if k.exitFunc != nil {
		k.exitFunc(0)
	}
}

// Destroy existing k8s containers
func (k *Executor) killKubeletContainers() {
	if containers, err := dockertools.GetKubeletDockerContainers(k.dockerClient, true); err == nil {
		opts := dockertypes.ContainerRemoveOptions{
			RemoveVolumes: true,
			Force:         true,
		}
		for _, container := range containers {
			log.V(2).Infof("Removing container: %v", container.ID)
			if err := k.dockerClient.RemoveContainer(container.ID, opts); err != nil {
				log.Warning(err)
			}
		}
	} else {
		log.Warningf("Failed to list kubelet docker containers: %v", err)
	}
}

// Error is called when some error happens.
func (k *Executor) Error(driver bindings.ExecutorDriver, message string) {
	log.Errorln(message)
}

func newStatus(taskId *mesos.TaskID, state mesos.TaskState, message string) *mesos.TaskStatus {
	return &mesos.TaskStatus{
		TaskId:  taskId,
		State:   &state,
		Message: proto.String(message),
	}
}

func (k *Executor) sendStatus(driver bindings.ExecutorDriver, status *mesos.TaskStatus) {
	select {
	case <-k.terminate:
	default:
		k.outgoing <- func() (mesos.Status, error) { return driver.SendStatusUpdate(status) }
	}
}

func (k *Executor) sendFrameworkMessage(driver bindings.ExecutorDriver, msg string) {
	select {
	case <-k.terminate:
	default:
		k.outgoing <- func() (mesos.Status, error) { return driver.SendFrameworkMessage(msg) }
	}
}

func (k *Executor) runSendLoop() {
	defer log.V(1).Info("sender loop exiting")
	for {
		select {
		case <-k.terminate:
			return
		default:
			if !k.isConnected() {
				select {
				case <-k.terminate:
				case <-time.After(1 * time.Second):
				}
				continue
			}
			sender, ok := <-k.outgoing
			if !ok {
				// programming error
				panic("someone closed the outgoing channel")
			}
			if status, err := sender(); err == nil {
				continue
			} else {
				log.Error(err)
				if status == mesos.Status_DRIVER_ABORTED {
					return
				}
			}
			// attempt to re-queue the sender
			select {
			case <-k.terminate:
			case k.outgoing <- sender:
			}
		}
	}
}

func annotationsFor(ei *mesos.ExecutorInfo) (annotations map[string]string, err error) {
	annotations = map[string]string{}
	if ei == nil {
		return
	}

	var buf bytes.Buffer
	if err = executorinfo.EncodeResources(&buf, ei.GetResources()); err != nil {
		return
	}

	annotations[meta.ExecutorIdKey] = ei.GetExecutorId().GetValue()
	annotations[meta.ExecutorResourcesKey] = buf.String()

	return
}
