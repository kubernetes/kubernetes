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
	"fmt"
	"testing"
	"time"

	mesos "github.com/mesos/mesos-go/mesosproto"
	"github.com/mesos/mesos-go/mesosutil"
	"github.com/stretchr/testify/assert"
	"k8s.io/kubernetes/contrib/mesos/pkg/offers"
	"k8s.io/kubernetes/contrib/mesos/pkg/proc"
)

func TestInMemoryRegistry_RegisterGetUnregister(t *testing.T) {
	assert := assert.New(t)

	registry := NewInMemoryRegistry()

	// it's empty at the beginning
	tasks := registry.List(func(t *T) bool { return true })
	assert.Empty(tasks)

	// add a task
	a := fakePodTask("a", nil, nil)
	a_clone, err := registry.Register(a)
	assert.NoError(err)
	assert.Equal(a_clone.ID, a.ID)
	assert.Equal(a_clone.podKey, a.podKey)

	// add another task
	b := fakePodTask("b", nil, nil)
	b_clone, err := registry.Register(b)
	assert.NoError(err)
	assert.Equal(b_clone.ID, b.ID)
	assert.Equal(b_clone.podKey, b.podKey)

	// find tasks in the registry
	tasks = registry.List(func(t *T) bool { return true })
	assert.Len(tasks, 2)
	assertContains(t, a_clone, tasks...)
	assertContains(t, b_clone, tasks...)

	tasks = registry.List(func(t *T) bool { return t.ID == a.ID })
	assert.Len(tasks, 1)
	assertContains(t, a_clone, tasks...)

	task, _ := registry.ForPod(a.podKey)
	assert.NotNil(task)
	assert.Equal(task.ID, a.ID)

	task, _ = registry.ForPod(b.podKey)
	assert.NotNil(task)
	assert.Equal(task.ID, b.ID)

	task, _ = registry.ForPod("no-pod-key")
	assert.Nil(task)

	task, _ = registry.Get(a.ID)
	assert.NotNil(task)
	assert.Equal(task.ID, a.ID)

	task, _ = registry.Get("unknown-task-id")
	assert.Nil(task)

	// re-add a task
	a_clone, err = registry.Register(a)
	assert.Error(err)
	assert.Nil(a_clone)

	// re-add a task with another podKey, but same task id
	another_a := a.Clone()
	another_a.podKey = "another-pod"
	another_a_clone, err := registry.Register(another_a)
	assert.Error(err)
	assert.Nil(another_a_clone)

	// re-add a task with another task ID, but same podKey
	another_b := b.Clone()
	another_b.ID = "another-task-id"
	another_b_clone, err := registry.Register(another_b)
	assert.Error(err)
	assert.Nil(another_b_clone)

	// unregister a task
	registry.Unregister(b)

	tasks = registry.List(func(t *T) bool { return true })
	assert.Len(tasks, 1)
	assertContains(t, a, tasks...)

	// unregister a task not registered
	unregistered_task := fakePodTask("unregistered-task", nil, nil)
	registry.Unregister(unregistered_task)
}

func fakeStatusUpdate(taskId string, state mesos.TaskState) *mesos.TaskStatus {
	status := mesosutil.NewTaskStatus(mesosutil.NewTaskID(taskId), state)
	status.Data = []byte("{}") // empty json
	masterSource := mesos.TaskStatus_SOURCE_MASTER
	status.Source = &masterSource
	return status
}

func TestInMemoryRegistry_State(t *testing.T) {
	assert := assert.New(t)

	registry := NewInMemoryRegistry()

	// add a task
	a := fakePodTask("a", nil, nil)
	a_clone, err := registry.Register(a)
	assert.NoError(err)
	assert.Equal(a.State, a_clone.State)

	// update the status
	assert.Equal(a_clone.State, StatePending)
	a_clone, state := registry.UpdateStatus(fakeStatusUpdate(a.ID, mesos.TaskState_TASK_RUNNING))
	assert.Equal(state, StatePending)         // old state
	assert.Equal(a_clone.State, StateRunning) // new state

	// update unknown task
	unknown_clone, state := registry.UpdateStatus(fakeStatusUpdate("unknown-task-id", mesos.TaskState_TASK_RUNNING))
	assert.Nil(unknown_clone)
	assert.Equal(state, StateUnknown)
}

func TestInMemoryRegistry_Update(t *testing.T) {
	assert := assert.New(t)

	// create offers registry
	ttl := time.Second / 4
	config := offers.RegistryConfig{
		DeclineOffer: func(offerId string) <-chan error {
			return proc.ErrorChan(nil)
		},
		Compat: func(o *mesos.Offer) bool {
			return true
		},
		TTL:       ttl,
		LingerTTL: 2 * ttl,
	}
	storage := offers.CreateRegistry(config)

	// Add offer
	offerId := mesosutil.NewOfferID("foo")
	mesosOffer := &mesos.Offer{Id: offerId}
	storage.Add([]*mesos.Offer{mesosOffer})
	offer, ok := storage.Get(offerId.GetValue())
	assert.True(ok)

	// create registry
	registry := NewInMemoryRegistry()
	a := fakePodTask("a", nil, nil)
	registry.Register(a.Clone()) // here clone a because we change it below

	// state changes are ignored
	a.State = StateRunning
	err := registry.Update(a)
	assert.NoError(err)
	a_clone, _ := registry.Get(a.ID)
	assert.Equal(StatePending, a_clone.State)

	// offer is updated while pending
	a.Offer = offer
	err = registry.Update(a)
	assert.NoError(err)
	a_clone, _ = registry.Get(a.ID)
	assert.Equal(offer.Id(), a_clone.Offer.Id())

	// spec is updated while pending
	a.Spec = &Spec{SlaveID: "slave-1"}
	err = registry.Update(a)
	assert.NoError(err)
	a_clone, _ = registry.Get(a.ID)
	assert.Equal("slave-1", a_clone.Spec.SlaveID)

	// flags are updated while pending
	a.Flags[Launched] = struct{}{}
	err = registry.Update(a)
	assert.NoError(err)
	a_clone, _ = registry.Get(a.ID)

	_, found_launched := a_clone.Flags[Launched]
	assert.True(found_launched)

	// flags are updated while running
	registry.UpdateStatus(fakeStatusUpdate(a.ID, mesos.TaskState_TASK_RUNNING))
	a.Flags[Bound] = struct{}{}
	err = registry.Update(a)
	assert.NoError(err)
	a_clone, _ = registry.Get(a.ID)

	_, found_launched = a_clone.Flags[Launched]
	assert.True(found_launched)
	_, found_bound := a_clone.Flags[Bound]
	assert.True(found_bound)

	// spec is ignored while running
	a.Spec = &Spec{SlaveID: "slave-2"}
	err = registry.Update(a)
	assert.NoError(err)
	a_clone, _ = registry.Get(a.ID)
	assert.Equal("slave-1", a_clone.Spec.SlaveID)

	// error when finished
	registry.UpdateStatus(fakeStatusUpdate(a.ID, mesos.TaskState_TASK_FINISHED))
	err = registry.Update(a)
	assert.Error(err)

	// update unknown task
	unknown_task := fakePodTask("unknown-task", nil, nil)
	err = registry.Update(unknown_task)
	assert.Error(err)

	// update nil task
	err = registry.Update(nil)
	assert.Nil(err)
}

type transition struct {
	statusUpdate  mesos.TaskState
	expectedState *StateType
	expectPanic   bool
}

func NewTransition(statusUpdate mesos.TaskState, expectedState StateType) transition {
	return transition{statusUpdate: statusUpdate, expectedState: &expectedState, expectPanic: false}
}

func NewTransitionToDeletedTask(statusUpdate mesos.TaskState) transition {
	return transition{statusUpdate: statusUpdate, expectedState: nil, expectPanic: false}
}

func NewTransitionWhichPanics(statusUpdate mesos.TaskState) transition {
	return transition{statusUpdate: statusUpdate, expectPanic: true}
}

func testStateTrace(t *testing.T, transitions []transition) *Registry {
	assert := assert.New(t)

	registry := NewInMemoryRegistry()
	a := fakePodTask("a", nil, nil)
	a, _ = registry.Register(a)

	// initial pending state
	assert.Equal(a.State, StatePending)

	for _, transition := range transitions {
		if transition.expectPanic {
			assert.Panics(func() {
				registry.UpdateStatus(fakeStatusUpdate(a.ID, transition.statusUpdate))
			})
		} else {
			a, _ = registry.UpdateStatus(fakeStatusUpdate(a.ID, transition.statusUpdate))
			if transition.expectedState == nil {
				a, _ = registry.Get(a.ID)
				assert.Nil(a, "expected task to be deleted from registry after status update to %v", transition.statusUpdate)
			} else {
				assert.Equal(a.State, *transition.expectedState)
			}
		}
	}

	return &registry
}

func TestInMemoryRegistry_TaskLifeCycle(t *testing.T) {
	testStateTrace(t, []transition{
		NewTransition(mesos.TaskState_TASK_STAGING, StatePending),
		NewTransition(mesos.TaskState_TASK_STARTING, StatePending),
		NewTransitionWhichPanics(mesos.TaskState_TASK_FINISHED),
		NewTransition(mesos.TaskState_TASK_RUNNING, StateRunning),
		NewTransition(mesos.TaskState_TASK_RUNNING, StateRunning),
		NewTransition(mesos.TaskState_TASK_STARTING, StateRunning),
		NewTransition(mesos.TaskState_TASK_FINISHED, StateFinished),
		NewTransition(mesos.TaskState_TASK_FINISHED, StateFinished),
		NewTransition(mesos.TaskState_TASK_RUNNING, StateFinished),
	})
}

func TestInMemoryRegistry_NotFinished(t *testing.T) {
	// all these behave the same
	notFinishedStates := []mesos.TaskState{
		mesos.TaskState_TASK_ERROR,
		mesos.TaskState_TASK_FAILED,
		mesos.TaskState_TASK_KILLED,
		mesos.TaskState_TASK_LOST,
	}
	for _, notFinishedState := range notFinishedStates {
		testStateTrace(t, []transition{
			NewTransitionToDeletedTask(notFinishedState),
		})

		testStateTrace(t, []transition{
			NewTransition(mesos.TaskState_TASK_RUNNING, StateRunning),
			NewTransitionToDeletedTask(notFinishedState),
		})

		testStateTrace(t, []transition{
			NewTransition(mesos.TaskState_TASK_RUNNING, StateRunning),
			NewTransition(mesos.TaskState_TASK_FINISHED, StateFinished),
			NewTransition(notFinishedState, StateFinished),
		})
	}
}

func assertContains(t *testing.T, want *T, ts ...*T) bool {
	for _, got := range ts {
		if taskEquals(want, got) {
			return true
		}
	}

	return assert.Fail(t, fmt.Sprintf("%v does not contain %v", ts, want))
}

func taskEquals(t1, t2 *T) bool {
	return t1.ID == t2.ID && t1.podKey == t2.podKey
}
