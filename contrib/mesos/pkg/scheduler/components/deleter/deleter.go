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
	"time"

	log "github.com/golang/glog"
	"k8s.io/kubernetes/contrib/mesos/pkg/queue"
	"k8s.io/kubernetes/contrib/mesos/pkg/runtime"
	"k8s.io/kubernetes/contrib/mesos/pkg/scheduler"
	"k8s.io/kubernetes/contrib/mesos/pkg/scheduler/errors"
	"k8s.io/kubernetes/contrib/mesos/pkg/scheduler/podtask"
	"k8s.io/kubernetes/contrib/mesos/pkg/scheduler/queuer"
	"k8s.io/kubernetes/pkg/api"
)

type Deleter interface {
	Run(updates <-chan queue.Entry, done <-chan struct{})
	DeleteOne(pod *queuer.Pod) error
}

type deleter struct {
	sched scheduler.Scheduler
	qr    queuer.Queuer
}

func New(sched scheduler.Scheduler, qr queuer.Queuer) Deleter {
	return &deleter{
		sched: sched,
		qr:    qr,
	}
}

// currently monitors for "pod deleted" events, upon which handle()
// is invoked.
func (k *deleter) Run(updates <-chan queue.Entry, done <-chan struct{}) {
	go runtime.Until(func() {
		for {
			entry := <-updates
			pod := entry.Value().(*queuer.Pod)
			if entry.Is(queue.DELETE_EVENT) {
				if err := k.DeleteOne(pod); err != nil {
					log.Error(err)
				}
			} else if !entry.Is(queue.POP_EVENT) {
				k.qr.UpdatesAvailable()
			}
		}
	}, 1*time.Second, done)
}

func (k *deleter) DeleteOne(pod *queuer.Pod) error {
	ctx := api.WithNamespace(api.NewDefaultContext(), pod.Namespace)
	podKey, err := podtask.MakePodKey(ctx, pod.Name)
	if err != nil {
		return err
	}

	log.V(2).Infof("pod deleted: %v", podKey)

	// order is important here: we want to make sure we have the lock before
	// removing the pod from the scheduling queue. this makes the concurrent
	// execution of scheduler-error-handling and delete-handling easier to
	// reason about.
	k.sched.Lock()
	defer k.sched.Unlock()

	// prevent the scheduler from attempting to pop this; it's also possible that
	// it's concurrently being scheduled (somewhere between pod scheduling and
	// binding) - if so, then we'll end up removing it from taskRegistry which
	// will abort Bind()ing
	k.qr.Dequeue(pod.GetUID())

	switch task, state := k.sched.Tasks().ForPod(podKey); state {
	case podtask.StateUnknown:
		log.V(2).Infof("Could not resolve pod '%s' to task id", podKey)
		return errors.NoSuchPodErr

	// determine if the task has already been launched to mesos, if not then
	// cleanup is easier (unregister) since there's no state to sync
	case podtask.StatePending:
		if !task.Has(podtask.Launched) {
			// we've been invoked in between Schedule() and Bind()
			if task.HasAcceptedOffer() {
				task.Offer.Release()
				task.Reset()
				task.Set(podtask.Deleted)
				//TODO(jdef) probably want better handling here
				if err := k.sched.Tasks().Update(task); err != nil {
					return err
				}
			}
			k.sched.Tasks().Unregister(task)
			return nil
		}
		fallthrough

	case podtask.StateRunning:
		// signal to watchers that the related pod is going down
		task.Set(podtask.Deleted)
		if err := k.sched.Tasks().Update(task); err != nil {
			log.Errorf("failed to update task w/ Deleted status: %v", err)
		}
		return k.sched.KillTask(task.ID)

	default:
		log.Infof("cannot kill pod '%s': non-terminal task not found %v", podKey, task.ID)
		return errors.NoSuchTaskErr
	}
}
