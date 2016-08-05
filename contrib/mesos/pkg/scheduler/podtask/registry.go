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

package podtask

import (
	"container/ring"
	"encoding/json"
	"fmt"
	"sync"
	"time"

	log "github.com/golang/glog"
	mesos "github.com/mesos/mesos-go/mesosproto"
	"k8s.io/kubernetes/contrib/mesos/pkg/scheduler/metrics"
	"k8s.io/kubernetes/pkg/api"
)

const (
	//TODO(jdef) move this somewhere else
	PodPath = "/pods"

	// length of historical record of finished tasks
	defaultFinishedTasksSize = 1024
)

// state store for pod tasks
type Registry interface {
	// register the specified task with this registry, as long as the current error
	// condition is nil. if no errors occur then return a copy of the registered task.
	Register(*T) (*T, error)

	// unregister the specified task from this registry
	Unregister(*T)

	// update state for the registered task identified by task.ID, returning a copy of
	// the updated task, if any.
	Update(task *T) error

	// return the task registered for the specified task ID and its current state.
	// if there is no such task then StateUnknown is returned.
	Get(taskId string) (task *T, currentState StateType)

	// return the non-terminal task corresponding to the specified pod ID
	ForPod(podID string) (task *T, currentState StateType)

	// update the task status given the specified mesos task status update, returning a
	// copy of the updated task (if any) and its state.
	UpdateStatus(status *mesos.TaskStatus) (*T, StateType)

	// return a list of task ID's that match the given filter, or all task ID's if filter == nil.
	List(filter func(*T) bool) []*T
}

type inMemoryRegistry struct {
	rw            sync.RWMutex
	taskRegistry  map[string]*T
	tasksFinished *ring.Ring
	podToTask     map[string]string
}

func NewInMemoryRegistry() Registry {
	return &inMemoryRegistry{
		taskRegistry:  make(map[string]*T),
		tasksFinished: ring.New(defaultFinishedTasksSize),
		podToTask:     make(map[string]string),
	}
}

func (k *inMemoryRegistry) List(accepts func(t *T) bool) (tasks []*T) {
	k.rw.RLock()
	defer k.rw.RUnlock()
	for _, task := range k.taskRegistry {
		if accepts == nil || accepts(task) {
			tasks = append(tasks, task.Clone())
		}
	}
	return
}

func (k *inMemoryRegistry) ForPod(podID string) (task *T, currentState StateType) {
	k.rw.RLock()
	defer k.rw.RUnlock()
	tid, ok := k.podToTask[podID]
	if !ok {
		return nil, StateUnknown
	}
	t, state := k._get(tid)
	return t.Clone(), state
}

// registers a pod task unless the spec'd error is not nil
func (k *inMemoryRegistry) Register(task *T) (*T, error) {
	k.rw.Lock()
	defer k.rw.Unlock()
	if _, found := k.podToTask[task.podKey]; found {
		return nil, fmt.Errorf("task already registered for pod key %q", task.podKey)
	}
	if _, found := k.taskRegistry[task.ID]; found {
		return nil, fmt.Errorf("task already registered for id %q", task.ID)
	}
	k.podToTask[task.podKey] = task.ID
	k.taskRegistry[task.ID] = task

	return task.Clone(), nil
}

// updates internal task state. updates are limited to Spec, Flags, and Offer for
// StatePending tasks, and are limited to Flag updates (additive only) for StateRunning tasks.
func (k *inMemoryRegistry) Update(task *T) error {
	if task == nil {
		return nil
	}
	k.rw.Lock()
	defer k.rw.Unlock()
	switch internal, state := k._get(task.ID); state {
	case StateUnknown:
		return fmt.Errorf("no such task: %v", task.ID)
	case StatePending:
		internal.Offer = task.Offer
		internal.Spec = task.Spec
		internal.Flags = map[FlagType]struct{}{}
		fallthrough
	case StateRunning:
		for k, v := range task.Flags {
			internal.Flags[k] = v
		}
		return nil
	default:
		return fmt.Errorf("may not update task %v in state %v", task.ID, state)
	}
}

func (k *inMemoryRegistry) Unregister(task *T) {
	k.rw.Lock()
	defer k.rw.Unlock()
	delete(k.podToTask, task.podKey)
	delete(k.taskRegistry, task.ID)
}

func (k *inMemoryRegistry) Get(taskId string) (*T, StateType) {
	k.rw.RLock()
	defer k.rw.RUnlock()
	t, state := k._get(taskId)
	return t.Clone(), state
}

// assume that the caller has already locked around access to task state.
// the caller is also responsible for cloning the task object before it leaves
// the context of this registry.
func (k *inMemoryRegistry) _get(taskId string) (*T, StateType) {
	if task, found := k.taskRegistry[taskId]; found {
		return task, task.State
	}
	return nil, StateUnknown
}

func (k *inMemoryRegistry) UpdateStatus(status *mesos.TaskStatus) (*T, StateType) {
	taskId := status.GetTaskId().GetValue()

	k.rw.Lock()
	defer k.rw.Unlock()
	task, state := k._get(taskId)

	switch status.GetState() {
	case mesos.TaskState_TASK_STAGING:
		k.handleTaskStaging(task, state, status)
	case mesos.TaskState_TASK_STARTING:
		k.handleTaskStarting(task, state, status)
	case mesos.TaskState_TASK_RUNNING:
		k.handleTaskRunning(task, state, status)
	case mesos.TaskState_TASK_FINISHED:
		k.handleTaskFinished(task, state, status)
	case mesos.TaskState_TASK_FAILED:
		k.handleTaskFailed(task, state, status)
	case mesos.TaskState_TASK_ERROR:
		k.handleTaskError(task, state, status)
	case mesos.TaskState_TASK_KILLED:
		k.handleTaskKilled(task, state, status)
	case mesos.TaskState_TASK_LOST:
		k.handleTaskLost(task, state, status)
	default:
		log.Warningf("unhandled status update for task: %v", taskId)
	}
	return task.Clone(), state
}

func (k *inMemoryRegistry) handleTaskStaging(task *T, state StateType, status *mesos.TaskStatus) {
	if status.GetSource() != mesos.TaskStatus_SOURCE_MASTER {
		log.Errorf("received STAGING for task %v with unexpected source: %v",
			status.GetTaskId().GetValue(), status.GetSource())
	}
}

func (k *inMemoryRegistry) handleTaskStarting(task *T, state StateType, status *mesos.TaskStatus) {
	// we expect to receive this when a launched task is finally "bound"
	// via the API server. however, there's nothing specific for us to do here.
	switch state {
	case StatePending:
		task.UpdatedTime = time.Now()
		if !task.Has(Bound) {
			task.Set(Bound)
			task.bindTime = task.UpdatedTime
			timeToBind := task.bindTime.Sub(task.launchTime)
			metrics.BindLatency.Observe(metrics.InMicroseconds(timeToBind))
		}
	default:
		taskId := status.GetTaskId().GetValue()
		log.Warningf("Ignore status TASK_STARTING because the task %v is not pending", taskId)
	}
}

func (k *inMemoryRegistry) handleTaskRunning(task *T, state StateType, status *mesos.TaskStatus) {
	taskId := status.GetTaskId().GetValue()
	switch state {
	case StatePending:
		task.UpdatedTime = time.Now()
		log.Infof("Received running status for pending task: %v", taskId)
		fillRunningPodInfo(task, status)
		task.State = StateRunning
	case StateRunning:
		task.UpdatedTime = time.Now()
		log.V(2).Infof("Ignore status TASK_RUNNING because the task %v is already running", taskId)
	case StateFinished:
		log.Warningf("Ignore status TASK_RUNNING because the task %v is already finished", taskId)
	default:
		log.Warningf("Ignore status TASK_RUNNING because the task %v is discarded", taskId)
	}
}

func ParsePodStatusResult(taskStatus *mesos.TaskStatus) (result api.PodStatusResult, err error) {
	if taskStatus.Data != nil {
		err = json.Unmarshal(taskStatus.Data, &result)
	} else {
		err = fmt.Errorf("missing TaskStatus.Data")
	}
	return
}

func fillRunningPodInfo(task *T, taskStatus *mesos.TaskStatus) {
	if taskStatus.GetReason() == mesos.TaskStatus_REASON_RECONCILIATION && taskStatus.GetSource() == mesos.TaskStatus_SOURCE_MASTER {
		// there is no data..
		return
	}
	//TODO(jdef) determine the usefullness of this information (if any)
	if result, err := ParsePodStatusResult(taskStatus); err != nil {
		log.Errorf("invalid TaskStatus.Data for task '%v': %v", task.ID, err)
	} else {
		task.podStatus = result.Status
		log.Infof("received pod status for task %v: %+v", task.ID, result.Status)
	}
}

func (k *inMemoryRegistry) handleTaskFinished(task *T, state StateType, status *mesos.TaskStatus) {
	taskId := status.GetTaskId().GetValue()
	switch state {
	case StatePending:
		panic(fmt.Sprintf("Pending task %v finished, this couldn't happen", taskId))
	case StateRunning:
		log.V(2).Infof("received finished status for running task: %v", taskId)
		delete(k.podToTask, task.podKey)
		task.State = StateFinished
		task.UpdatedTime = time.Now()
		k.tasksFinished = k.recordFinishedTask(task.ID)
	case StateFinished:
		log.Warningf("Ignore status TASK_FINISHED because the task %v is already finished", taskId)
	default:
		log.Warningf("Ignore status TASK_FINISHED because the task %v is not running", taskId)
	}
}

// record that a task has finished.
// older record are expunged one at a time once the historical ring buffer is saturated.
// assumes caller is holding state lock.
func (k *inMemoryRegistry) recordFinishedTask(taskId string) *ring.Ring {
	slot := k.tasksFinished.Next()
	if slot.Value != nil {
		// garbage collect older finished task from the registry
		gctaskId := slot.Value.(string)
		if gctask, found := k.taskRegistry[gctaskId]; found && gctask.State == StateFinished {
			delete(k.taskRegistry, gctaskId)
		}
	}
	slot.Value = taskId
	return slot
}

func (k *inMemoryRegistry) handleTaskFailed(task *T, state StateType, status *mesos.TaskStatus) {
	switch state {
	case StatePending, StateRunning:
		delete(k.taskRegistry, task.ID)
		delete(k.podToTask, task.podKey)
	}
}

func (k *inMemoryRegistry) handleTaskError(task *T, state StateType, status *mesos.TaskStatus) {
	switch state {
	case StatePending, StateRunning:
		delete(k.taskRegistry, task.ID)
		delete(k.podToTask, task.podKey)
	}
}

func (k *inMemoryRegistry) handleTaskKilled(task *T, state StateType, status *mesos.TaskStatus) {
	defer func() {
		msg := fmt.Sprintf("task killed: %+v, task %+v", status, task)
		if task != nil && task.Has(Deleted) {
			// we were expecting this, nothing out of the ordinary
			log.V(2).Infoln(msg)
		} else {
			log.Errorln(msg)
		}
	}()
	switch state {
	case StatePending, StateRunning:
		delete(k.taskRegistry, task.ID)
		delete(k.podToTask, task.podKey)
	}
}

func (k *inMemoryRegistry) handleTaskLost(task *T, state StateType, status *mesos.TaskStatus) {
	switch state {
	case StateRunning, StatePending:
		delete(k.taskRegistry, task.ID)
		delete(k.podToTask, task.podKey)
	}
}
