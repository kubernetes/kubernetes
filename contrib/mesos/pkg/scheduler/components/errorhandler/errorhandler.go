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

package errorhandler

import (
	log "github.com/golang/glog"
	"k8s.io/kubernetes/contrib/mesos/pkg/backoff"
	"k8s.io/kubernetes/contrib/mesos/pkg/queue"
	"k8s.io/kubernetes/contrib/mesos/pkg/scheduler"
	"k8s.io/kubernetes/contrib/mesos/pkg/scheduler/errors"
	"k8s.io/kubernetes/contrib/mesos/pkg/scheduler/podtask"
	"k8s.io/kubernetes/contrib/mesos/pkg/scheduler/queuer"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/util/runtime"
)

type ErrorHandler interface {
	Error(pod *api.Pod, schedulingErr error)
}

type errorHandler struct {
	sched        scheduler.Scheduler
	backoff      *backoff.Backoff
	qr           queuer.Queuer
	newBreakChan func(podKey string) queue.BreakChan
}

func New(sched scheduler.Scheduler, backoff *backoff.Backoff, qr queuer.Queuer, newBC func(podKey string) queue.BreakChan) ErrorHandler {
	return &errorHandler{
		sched:        sched,
		backoff:      backoff,
		qr:           qr,
		newBreakChan: newBC,
	}
}

// implementation of scheduling plugin's Error func; see plugin/pkg/scheduler
func (k *errorHandler) Error(pod *api.Pod, schedulingErr error) {

	if schedulingErr == errors.NoSuchPodErr {
		log.V(2).Infof("Not rescheduling non-existent pod %v", pod.Name)
		return
	}

	log.Infof("Error scheduling %v: %v; retrying", pod.Name, schedulingErr)
	defer runtime.HandleCrash()

	// default upstream scheduler passes pod.Name as binding.PodID
	ctx := api.WithNamespace(api.NewDefaultContext(), pod.Namespace)
	podKey, err := podtask.MakePodKey(ctx, pod.Name)
	if err != nil {
		log.Errorf("Failed to construct pod key, aborting scheduling for pod %v: %v", pod.Name, err)
		return
	}

	k.backoff.GC()
	k.sched.Lock()
	defer k.sched.Unlock()

	switch task, state := k.sched.Tasks().ForPod(podKey); state {
	case podtask.StateUnknown:
		// if we don't have a mapping here any more then someone deleted the pod
		log.V(2).Infof("Could not resolve pod to task, aborting pod reschdule: %s", podKey)
		return

	case podtask.StatePending:
		if task.Has(podtask.Launched) {
			log.V(2).Infof("Skipping re-scheduling for already-launched pod %v", podKey)
			return
		}
		breakoutEarly := queue.BreakChan(nil)
		if schedulingErr == errors.NoSuitableOffersErr {
			log.V(3).Infof("adding backoff breakout handler for pod %v", podKey)
			breakoutEarly = k.newBreakChan(podKey)
		}
		delay := k.backoff.Get(podKey)
		log.V(3).Infof("requeuing pod %v with delay %v", podKey, delay)
		k.qr.Requeue(queuer.NewPod(pod, queuer.Delay(delay), queuer.Notify(breakoutEarly)))

	default:
		log.V(2).Infof("Task is no longer pending, aborting reschedule for pod %v", podKey)
	}
}
