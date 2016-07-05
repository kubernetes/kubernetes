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

package deleter

import (
	"testing"

	"github.com/mesos/mesos-go/mesosproto"
	"github.com/stretchr/testify/assert"
	"k8s.io/kubernetes/contrib/mesos/pkg/queue"
	types "k8s.io/kubernetes/contrib/mesos/pkg/scheduler"
	"k8s.io/kubernetes/contrib/mesos/pkg/scheduler/errors"
	"k8s.io/kubernetes/contrib/mesos/pkg/scheduler/podtask"
	"k8s.io/kubernetes/contrib/mesos/pkg/scheduler/podtask/hostport"
	"k8s.io/kubernetes/contrib/mesos/pkg/scheduler/queuer"
	"k8s.io/kubernetes/pkg/api"
)

func TestDeleteOne_NonexistentPod(t *testing.T) {
	assert := assert.New(t)
	obj := &types.MockScheduler{}
	reg := podtask.NewInMemoryRegistry()
	obj.On("Tasks").Return(reg)

	q := queue.NewDelayFIFO()
	qr := queuer.New(q, nil)
	assert.Equal(0, len(q.List()))
	d := New(obj, qr)
	pod := &queuer.Pod{Pod: &api.Pod{
		ObjectMeta: api.ObjectMeta{
			Name:      "foo",
			Namespace: api.NamespaceDefault,
		}}}
	err := d.DeleteOne(pod)
	assert.Equal(err, errors.NoSuchPodErr)
	obj.AssertExpectations(t)
}

func TestDeleteOne_PendingPod(t *testing.T) {
	assert := assert.New(t)
	obj := &types.MockScheduler{}
	reg := podtask.NewInMemoryRegistry()
	obj.On("Tasks").Return(reg)

	pod := &queuer.Pod{Pod: &api.Pod{
		ObjectMeta: api.ObjectMeta{
			Name:      "foo",
			UID:       "foo0",
			Namespace: api.NamespaceDefault,
		}}}
	task, err := podtask.New(
		api.NewDefaultContext(),
		podtask.Config{
			ID:               "bar",
			Prototype:        &mesosproto.ExecutorInfo{},
			HostPortStrategy: hostport.StrategyWildcard,
		},
		pod.Pod,
	)
	if err != nil {
		t.Fatalf("failed to create task: %v", err)
	}

	_, err = reg.Register(task)
	if err != nil {
		t.Fatalf("failed to register task: %v", err)
	}

	// preconditions
	q := queue.NewDelayFIFO()
	qr := queuer.New(q, nil)
	q.Add(pod, queue.ReplaceExisting)
	assert.Equal(1, len(q.List()))
	_, found := q.Get("default/foo")
	assert.True(found)

	// exec & post conditions
	d := New(obj, qr)
	err = d.DeleteOne(pod)
	assert.Nil(err)
	_, found = q.Get("foo0")
	assert.False(found)
	assert.Equal(0, len(q.List()))
	obj.AssertExpectations(t)
}

func TestDeleteOne_Running(t *testing.T) {
	assert := assert.New(t)
	obj := &types.MockScheduler{}
	reg := podtask.NewInMemoryRegistry()
	obj.On("Tasks").Return(reg)

	pod := &queuer.Pod{Pod: &api.Pod{
		ObjectMeta: api.ObjectMeta{
			Name:      "foo",
			UID:       "foo0",
			Namespace: api.NamespaceDefault,
		}}}
	task, err := podtask.New(
		api.NewDefaultContext(),
		podtask.Config{
			ID:               "bar",
			Prototype:        &mesosproto.ExecutorInfo{},
			HostPortStrategy: hostport.StrategyWildcard,
		},
		pod.Pod,
	)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	task, err = reg.Register(task)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	task.Set(podtask.Launched)
	err = reg.Update(task)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// preconditions
	q := queue.NewDelayFIFO()
	qr := queuer.New(q, nil)
	q.Add(pod, queue.ReplaceExisting)
	assert.Equal(1, len(q.List()))
	_, found := q.Get("default/foo")
	assert.True(found)

	obj.On("KillTask", task.ID).Return(nil)

	// exec & post conditions
	d := New(obj, qr)
	err = d.DeleteOne(pod)
	assert.Nil(err)
	_, found = q.Get("foo0")
	assert.False(found)
	assert.Equal(0, len(q.List()))
	obj.AssertExpectations(t)
}

func TestDeleteOne_badPodNaming(t *testing.T) {
	assert := assert.New(t)
	obj := &types.MockScheduler{}
	pod := &queuer.Pod{Pod: &api.Pod{}}
	q := queue.NewDelayFIFO()
	qr := queuer.New(q, nil)
	d := New(obj, qr)

	err := d.DeleteOne(pod)
	assert.NotNil(err)

	pod.Pod.ObjectMeta.Name = "foo"
	err = d.DeleteOne(pod)
	assert.NotNil(err)

	pod.Pod.ObjectMeta.Name = ""
	pod.Pod.ObjectMeta.Namespace = "bar"
	err = d.DeleteOne(pod)
	assert.NotNil(err)

	obj.AssertExpectations(t)
}
