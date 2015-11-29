/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

	"github.com/fsouza/go-dockerclient"
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
	"k8s.io/kubernetes/pkg/client/cache"
	client "k8s.io/kubernetes/pkg/client/unversioned"
	"k8s.io/kubernetes/pkg/kubelet/container"
	"k8s.io/kubernetes/pkg/kubelet/dockertools"
	kubetypes "k8s.io/kubernetes/pkg/kubelet/types"
	"k8s.io/kubernetes/pkg/util"
)

const (
	containerPollTime = 1 * time.Second
	podRelistPeriod   = 5 * time.Minute
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

type kuberTask struct {
	mesosTaskInfo *mesos.TaskInfo
	launchTimer   *time.Timer // launchTimer expires when the launch-task process duration exceeds launchGracePeriod
	podName       string      // empty until pod is sent to kubelet and registed in KubernetesExecutor.pods
}

type podStatusFunc func() (*api.PodStatus, error)

// KubernetesExecutor is an mesos executor that runs pods
// in a minion machine.
type Executor struct {
	updateChan           chan<- kubetypes.PodUpdate // sent to the kubelet, closed on shutdown
	state                stateType
	tasks                map[string]*kuberTask
	pods                 map[string]*api.Pod
	lock                 sync.Mutex
	client               *client.Client
	terminate            chan struct{}                     // signals that the executor should shutdown
	outgoing             chan func() (mesos.Status, error) // outgoing queue to the mesos driver
	dockerClient         dockertools.DockerInterface
	suicideWatch         suicideWatcher
	suicideTimeout       time.Duration
	shutdownAlert        func()          // invoked just prior to executor shutdown
	kubeletFinished      <-chan struct{} // signals that kubelet Run() died
	exitFunc             func(int)
	podStatusFunc        func(*api.Pod) (*api.PodStatus, error)
	staticPodsConfigPath string
	launchGracePeriod    time.Duration
	nodeInfos            chan<- NodeInfo
	initCompleted        chan struct{} // closes upon completion of Init()
}

type Config struct {
	Updates              chan<- kubetypes.PodUpdate // to send pod config updates to the kubelet
	APIClient            *client.Client
	Docker               dockertools.DockerInterface
	ShutdownAlert        func()
	SuicideTimeout       time.Duration
	KubeletFinished      <-chan struct{} // signals that kubelet Run() died
	ExitFunc             func(int)
	PodStatusFunc        func(*api.Pod) (*api.PodStatus, error)
	StaticPodsConfigPath string
	PodLW                cache.ListerWatcher // mandatory, otherwise initialiation will panic
	LaunchGracePeriod    time.Duration
	NodeInfos            chan<- NodeInfo
}

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
		updateChan:           config.Updates,
		state:                disconnectedState,
		tasks:                make(map[string]*kuberTask),
		pods:                 make(map[string]*api.Pod),
		client:               config.APIClient,
		terminate:            make(chan struct{}),
		outgoing:             make(chan func() (mesos.Status, error), 1024),
		dockerClient:         config.Docker,
		suicideTimeout:       config.SuicideTimeout,
		kubeletFinished:      config.KubeletFinished,
		suicideWatch:         &suicideTimer{},
		shutdownAlert:        config.ShutdownAlert,
		exitFunc:             config.ExitFunc,
		podStatusFunc:        config.PodStatusFunc,
		staticPodsConfigPath: config.StaticPodsConfigPath,
		launchGracePeriod:    launchGracePeriod,
		nodeInfos:            config.NodeInfos,
		initCompleted:        make(chan struct{}),
	}
	runtime.On(k.initCompleted, k.runSendLoop)

	po := newPodObserver(config.PodLW, k.updateTask, k.terminate)
	runtime.On(k.initCompleted, po.run)

	return k
}

func (k *Executor) Init(driver bindings.ExecutorDriver) {
	defer close(k.initCompleted)
	k.killKubeletContainers()
	k.resetSuicideWatch(driver)
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

// sendPodsSnapshot assumes that caller is holding state lock; returns true when update is sent otherwise false
func (k *Executor) sendPodsSnapshot() bool {
	if k.isDone() {
		return false
	}
	snapshot := make([]*api.Pod, 0, len(k.pods))
	for _, v := range k.pods {
		snapshot = append(snapshot, v)
	}
	u := &kubetypes.PodUpdate{
		Op:   kubetypes.SET,
		Pods: snapshot,
	}
	k.updateChan <- *u
	return true
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

	if executorInfo != nil && executorInfo.Data != nil {
		err := k.initializeStaticPodsSource(slaveInfo.GetHostname(), executorInfo.Data)
		if err != nil {
			log.Errorf("failed to initialize static pod configuration: %v", err)
		}
	}

	annotations, err := annotationsFor(executorInfo)
	if err != nil {
		log.Errorf(
			"cannot get node annotations from executor info %v error %v",
			executorInfo, err,
		)
	}

	if slaveInfo != nil {
		_, err := node.CreateOrUpdate(
			k.client,
			slaveInfo.GetHostname(),
			node.SlaveAttributesToLabels(slaveInfo.Attributes),
			annotations,
		)

		if err != nil {
			log.Errorf("cannot update node labels: %v", err)
		}
	}

	// emit an empty update to allow the mesos "source" to be marked as seen
	k.lock.Lock()
	defer k.lock.Unlock()
	k.sendPodsSnapshot()

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
		_, err := node.CreateOrUpdate(
			k.client,
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
func (k *Executor) initializeStaticPodsSource(hostname string, data []byte) error {
	log.V(2).Infof("extracting static pods config to %s", k.staticPodsConfigPath)
	// annotate the pod with BindingHostKey so that the scheduler will ignore the pod
	// once it appears in the pod registry. the stock kubelet sets the pod host in order
	// to accomplish the same; we do this because the k8sm scheduler works differently.
	annotator := podutil.Annotator(map[string]string{
		meta.BindingHostKey: hostname,
	})
	return podutil.WriteToDir(annotator.Do(podutil.Gunzip(data)), k.staticPodsConfigPath)
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

	if !k.isConnected() {
		log.Errorf("Ignore launch task because the executor is disconnected\n")
		k.sendStatus(driver, newStatus(taskInfo.GetTaskId(), mesos.TaskState_TASK_FAILED,
			messages.ExecutorUnregistered))
		return
	}

	obj, err := api.Codec.Decode(taskInfo.GetData())
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

	taskId := taskInfo.GetTaskId().GetValue()
	k.lock.Lock()
	defer k.lock.Unlock()

	if _, found := k.tasks[taskId]; found {
		log.Errorf("task already launched\n")
		// Not to send back TASK_RUNNING here, because
		// may be duplicated messages or duplicated task id.
		return
	}
	// remember this task so that:
	// (a) we ignore future launches for it
	// (b) we have a record of it so that we can kill it if needed
	// (c) we're leaving podName == "" for now, indicates we don't need to delete containers
	k.tasks[taskId] = &kuberTask{
		mesosTaskInfo: taskInfo,
		launchTimer:   time.NewTimer(k.launchGracePeriod),
	}
	k.resetSuicideWatch(driver)

	go k.launchTask(driver, taskId, pod)
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
			if len(k.tasks) > 0 {
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
	if len(k.tasks) > 0 {
		ids := []string{}
		for taskid := range k.tasks {
			ids = append(ids, taskid)
		}
		log.Errorf("suicide attempt failed, there are still running tasks: %v", ids)
		return
	}

	log.Infoln("Attempting suicide")
	if (&k.state).transitionTo(suicidalState, suicidalState, terminalState) {
		//TODO(jdef) let the scheduler know?
		//TODO(jdef) is suicide more graceful than slave-demanded shutdown?
		k.doShutdown(driver)
	}
}

// async continuation of LaunchTask
func (k *Executor) launchTask(driver bindings.ExecutorDriver, taskId string, pod *api.Pod) {
	deleteTask := func() {
		k.lock.Lock()
		defer k.lock.Unlock()
		delete(k.tasks, taskId)
		k.resetSuicideWatch(driver)
	}

	// TODO(k8s): use Pods interface for binding once clusters are upgraded
	// return b.Pods(binding.Namespace).Bind(binding)
	if pod.Spec.NodeName == "" {
		//HACK(jdef): cloned binding construction from k8s plugin/pkg/scheduler/framework.go
		binding := &api.Binding{
			ObjectMeta: api.ObjectMeta{
				Namespace:   pod.Namespace,
				Name:        pod.Name,
				Annotations: make(map[string]string),
			},
			Target: api.ObjectReference{
				Kind: "Node",
				Name: pod.Annotations[meta.BindingHostKey],
			},
		}

		// forward the annotations that the scheduler wants to apply
		for k, v := range pod.Annotations {
			binding.Annotations[k] = v
		}

		// create binding on apiserver
		log.Infof("Binding '%v/%v' to '%v' with annotations %+v...", pod.Namespace, pod.Name, binding.Target.Name, binding.Annotations)
		ctx := api.WithNamespace(api.NewContext(), binding.Namespace)
		err := k.client.Post().Namespace(api.NamespaceValue(ctx)).Resource("bindings").Body(binding).Do().Error()
		if err != nil {
			deleteTask()
			k.sendStatus(driver, newStatus(mutil.NewTaskID(taskId), mesos.TaskState_TASK_FAILED,
				messages.CreateBindingFailure))
			return
		}
	} else {
		// post annotations update to apiserver
		patch := struct {
			Metadata struct {
				Annotations map[string]string `json:"annotations"`
			} `json:"metadata"`
		}{}
		patch.Metadata.Annotations = pod.Annotations
		patchJson, _ := json.Marshal(patch)
		log.V(4).Infof("Patching annotations %v of pod %v/%v: %v", pod.Annotations, pod.Namespace, pod.Name, string(patchJson))
		err := k.client.Patch(api.MergePatchType).RequestURI(pod.SelfLink).Body(patchJson).Do().Error()
		if err != nil {
			log.Errorf("Error updating annotations of ready-to-launch pod %v/%v: %v", pod.Namespace, pod.Name, err)
			deleteTask()
			k.sendStatus(driver, newStatus(mutil.NewTaskID(taskId), mesos.TaskState_TASK_FAILED,
				messages.AnnotationUpdateFailure))
			return
		}
	}

	podFullName := container.GetPodFullName(pod)

	// allow a recently failed-over scheduler the chance to recover the task/pod binding:
	// it may have failed and recovered before the apiserver is able to report the updated
	// binding information. replays of this status event will signal to the scheduler that
	// the apiserver should be up-to-date.
	data, err := json.Marshal(api.PodStatusResult{
		ObjectMeta: api.ObjectMeta{
			Name:     podFullName,
			SelfLink: "/podstatusresult",
		},
	})
	if err != nil {
		deleteTask()
		log.Errorf("failed to marshal pod status result: %v", err)
		k.sendStatus(driver, newStatus(mutil.NewTaskID(taskId), mesos.TaskState_TASK_FAILED,
			err.Error()))
		return
	}

	k.lock.Lock()
	defer k.lock.Unlock()

	// find task
	task, found := k.tasks[taskId]
	if !found {
		log.V(1).Infof("task %v not found, probably killed: aborting launch, reporting lost", taskId)
		k.reportLostTask(driver, taskId, messages.LaunchTaskFailed)
		return
	}

	//TODO(jdef) check for duplicate pod name, if found send TASK_ERROR

	// send the new pod to the kubelet which will spin it up
	task.podName = podFullName
	k.pods[podFullName] = pod
	ok := k.sendPodsSnapshot()
	if !ok {
		task.podName = ""
		delete(k.pods, podFullName)
		return // executor is terminating, cancel launch
	}

	// From here on, we need to delete containers associated with the task upon
	// it going into a terminal state.

	// report task is starting to scheduler
	statusUpdate := &mesos.TaskStatus{
		TaskId:  mutil.NewTaskID(taskId),
		State:   mesos.TaskState_TASK_STARTING.Enum(),
		Message: proto.String(messages.CreateBindingSuccess),
		Data:    data,
	}
	k.sendStatus(driver, statusUpdate)

	// Delay reporting 'task running' until container is up.
	psf := podStatusFunc(func() (*api.PodStatus, error) {
		return k.podStatusFunc(pod)
	})
	go k._launchTask(driver, taskId, podFullName, psf, task.launchTimer.C)
}

func (k *Executor) _launchTask(driver bindings.ExecutorDriver, taskId, podFullName string, psf podStatusFunc, expired <-chan time.Time) {
	getMarshalledInfo := func() (data []byte, cancel bool) {
		// potentially long call..
		if podStatus, err := psf(); err == nil && podStatus != nil {
			select {
			case <-expired:
				cancel = true
			default:
				k.lock.Lock()
				defer k.lock.Unlock()
				if _, found := k.tasks[taskId]; !found {
					// don't bother with the pod status if the task is already gone
					cancel = true
					break
				} else if podStatus.Phase != api.PodRunning {
					// avoid sending back a running status before it's really running
					break
				}
				log.V(2).Infof("Found pod status: '%v'", podStatus)
				result := api.PodStatusResult{
					ObjectMeta: api.ObjectMeta{
						Name:     podFullName,
						SelfLink: "/podstatusresult",
					},
					Status: *podStatus,
				}
				if data, err = json.Marshal(result); err != nil {
					log.Errorf("failed to marshal pod status result: %v", err)
				}
			}
		}
		return
	}

waitForRunningPod:
	for {
		select {
		case <-expired:
			log.Warningf("Launch expired grace period of '%v'", k.launchGracePeriod)
			break waitForRunningPod
		case <-time.After(containerPollTime):
			if data, cancel := getMarshalledInfo(); cancel {
				break waitForRunningPod
			} else if data == nil {
				continue waitForRunningPod
			} else {
				k.lock.Lock()
				defer k.lock.Unlock()
				task, found := k.tasks[taskId]
				if !found {
					goto reportLost
				}

				statusUpdate := &mesos.TaskStatus{
					TaskId:  mutil.NewTaskID(taskId),
					State:   mesos.TaskState_TASK_RUNNING.Enum(),
					Message: proto.String(fmt.Sprintf("pod-running:%s", podFullName)),
					Data:    data,
				}

				k.sendStatus(driver, statusUpdate)
				task.launchTimer.Stop()

				// continue to monitor the health of the pod
				go k.__launchTask(driver, taskId, podFullName, psf)
				return
			}
		}
	}

	k.lock.Lock()
	defer k.lock.Unlock()
reportLost:
	k.reportLostTask(driver, taskId, messages.KubeletPodLaunchFailed)
}

func (k *Executor) __launchTask(driver bindings.ExecutorDriver, taskId, podFullName string, psf podStatusFunc) {
	// TODO(nnielsen): Monitor health of pod and report if lost.
	// Should we also allow this to fail a couple of times before reporting lost?
	// What if the docker daemon is restarting and we can't connect, but it's
	// going to bring the pods back online as soon as it restarts?
	knownPod := func() bool {
		_, err := psf()
		return err == nil
	}
	// Wait for the pod to go away and stop monitoring once it does
	// TODO (jdefelice) replace with an /events watch?
	for {
		time.Sleep(containerPollTime)
		if k.checkForLostPodTask(driver, taskId, knownPod) {
			return
		}
	}
}

// Intended to be executed as part of the pod monitoring loop, this fn (ultimately) checks with Docker
// whether the pod is running. It will only return false if the task is still registered and the pod is
// registered in Docker. Otherwise it returns true. If there's still a task record on file, but no pod
// in Docker, then we'll also send a TASK_LOST event.
func (k *Executor) checkForLostPodTask(driver bindings.ExecutorDriver, taskId string, isKnownPod func() bool) bool {
	// TODO (jdefelice) don't send false alarms for deleted pods (KILLED tasks)

	// isKnownPod() can block so we take special precaution to avoid locking this mutex while calling it
	knownTask := func() (ok bool) {
		k.lock.Lock()
		defer k.lock.Unlock()
		_, ok = k.tasks[taskId]
		return
	}()

	// TODO(jdef) we should really consider k.pods here, along with what docker is reporting, since the
	// kubelet may constantly attempt to instantiate a pod as long as it's in the pod state that we're
	// handing to it. otherwise, we're probably reporting a TASK_LOST prematurely. Should probably
	// consult RestartPolicy to determine appropriate behavior. Should probably also gracefully handle
	// docker daemon restarts.
	if knownTask {
		if isKnownPod() {
			return false
		} else {
			log.Warningf("Detected lost pod, reporting lost task %v", taskId)

			k.lock.Lock()
			defer k.lock.Unlock()
			k.reportLostTask(driver, taskId, messages.ContainersDisappeared)
		}
	} else {
		log.V(2).Infof("Task %v no longer registered, stop monitoring for lost pods", taskId)
	}
	return true
}

// KillTask is called when the executor receives a request to kill a task.
func (k *Executor) KillTask(driver bindings.ExecutorDriver, taskId *mesos.TaskID) {
	if k.isDone() {
		return
	}
	log.Infof("Kill task %v\n", taskId)

	if !k.isConnected() {
		//TODO(jdefelice) sent TASK_LOST here?
		log.Warningf("Ignore kill task because the executor is disconnected\n")
		return
	}

	k.lock.Lock()
	defer k.lock.Unlock()
	k.removePodTask(driver, taskId.GetValue(), messages.TaskKilled, mesos.TaskState_TASK_KILLED)
}

// Reports a lost task to the slave and updates internal task and pod tracking state.
// Assumes that the caller is locking around pod and task state.
func (k *Executor) reportLostTask(driver bindings.ExecutorDriver, tid, reason string) {
	k.removePodTask(driver, tid, reason, mesos.TaskState_TASK_LOST)
}

// deletes the pod and task associated with the task identified by tid and sends a task
// status update to mesos. also attempts to reset the suicide watch.
// Assumes that the caller is locking around pod and task state.
func (k *Executor) removePodTask(driver bindings.ExecutorDriver, tid, reason string, state mesos.TaskState) {
	task, ok := k.tasks[tid]
	if !ok {
		log.V(1).Infof("Failed to remove task, unknown task %v\n", tid)
		return
	}
	delete(k.tasks, tid)
	k.resetSuicideWatch(driver)

	pid := task.podName
	_, found := k.pods[pid]
	if !found {
		log.Warningf("Cannot remove unknown pod %v for task %v", pid, tid)
	} else {
		log.V(2).Infof("deleting pod %v for task %v", pid, tid)
		delete(k.pods, pid)

		// tell the kubelet to remove the pod
		k.sendPodsSnapshot()
	}
	// TODO(jdef): ensure that the update propagates, perhaps return a signal chan?
	k.sendStatus(driver, newStatus(mutil.NewTaskID(tid), state, reason))
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
			// clean up pod state
			k.lock.Lock()
			defer k.lock.Unlock()
			k.reportLostTask(driver, taskId, messages.TaskLostAck)
		}
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
	close(k.updateChan)
	close(k.nodeInfos)

	if k.shutdownAlert != nil {
		func() {
			util.HandleCrash()
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
	k.tasks = map[string]*kuberTask{}

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
		opts := docker.RemoveContainerOptions{
			RemoveVolumes: true,
			Force:         true,
		}
		for _, container := range containers {
			opts.ID = container.ID
			log.V(2).Infof("Removing container: %v", opts.ID)
			if err := k.dockerClient.RemoveContainer(opts); err != nil {
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

// updateTask executes some mutating operation for the given task/pod, blocking until the update is either
// attempted or discarded. uses the executor state lock to synchronize concurrent invocation. returns true
// only if the specified update operation was attempted and also returns true. a result of true also indicates
// changes have been sent upstream to the kubelet.
func (k *Executor) updateTask(taskId string, f func(*kuberTask, *api.Pod) bool) (changed bool, err error) {
	k.lock.Lock()
	defer k.lock.Unlock()

	// exclude tasks which are already deleted from our task registry
	task := k.tasks[taskId]
	if task == nil {
		// the pod has completed the launch-task-binding phase because it's been annotated with
		// the task-id, but we don't have a record of it; it's best to let the scheduler reconcile.
		// it's also possible that our update queue is backed up and hasn't caught up with the state
		// of the world yet.

		// TODO(jdef) should we hint to the scheduler (via TASK_FAILED, reason=PodRefersToUnknownTask)?

		err = fmt.Errorf("task %s not found", taskId)
		return
	}

	oldPod := k.pods[task.podName]
	changed = f(task, oldPod)

	// TODO(jdef) this abstraction isn't perfect since changes that only impact the task struct,
	// and not the pod, don't require a new pod snapshot sent back to the kubelet.
	if changed {
		k.sendPodsSnapshot()
	}
	return
}
