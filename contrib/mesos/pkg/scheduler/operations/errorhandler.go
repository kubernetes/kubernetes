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

package operations

import (
	log "github.com/golang/glog"
	mesos "github.com/mesos/mesos-go/mesosproto"
	"k8s.io/kubernetes/contrib/mesos/pkg/backoff"
	"k8s.io/kubernetes/contrib/mesos/pkg/queue"
	merrors "k8s.io/kubernetes/contrib/mesos/pkg/scheduler/errors"
	"k8s.io/kubernetes/contrib/mesos/pkg/scheduler/podschedulers"
	"k8s.io/kubernetes/contrib/mesos/pkg/scheduler/podtask"
	"k8s.io/kubernetes/contrib/mesos/pkg/scheduler/queuer"
	types "k8s.io/kubernetes/contrib/mesos/pkg/scheduler/types"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/util"
)

type ErrorHandler struct {
	fw      types.Framework
	backoff *backoff.Backoff
	qr      *queuer.Queuer
}

func NewErrorHandler(fw types.Framework, backoff *backoff.Backoff, qr *queuer.Queuer) *ErrorHandler {
	return &ErrorHandler{
		fw:      fw,
		backoff: backoff,
		qr:      qr,
	}
}

// implementation of scheduling plugin's Error func; see plugin/pkg/scheduler
func (k *ErrorHandler) Error(pod *api.Pod, schedulingErr error) {

	if schedulingErr == merrors.NoSuchPodErr {
		log.V(2).Infof("Not rescheduling non-existent pod %v", pod.Name)
		return
	}

	log.Infof("Error scheduling %v: %v; retrying", pod.Name, schedulingErr)
	defer util.HandleCrash()

	// default upstream scheduler passes pod.Name as binding.PodID
	ctx := api.WithNamespace(api.NewDefaultContext(), pod.Namespace)
	podKey, err := podtask.MakePodKey(ctx, pod.Name)
	if err != nil {
		log.Errorf("Failed to construct pod key, aborting scheduling for pod %v: %v", pod.Name, err)
		return
	}

	k.backoff.GC()
	k.fw.Lock()
	defer k.fw.Unlock()

	switch task, state := k.fw.Tasks().ForPod(podKey); state {
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
		if schedulingErr == podschedulers.NoSuitableOffersErr {
			log.V(3).Infof("adding backoff breakout handler for pod %v", podKey)
			breakoutEarly = queue.BreakChan(k.fw.Offers().Listen(podKey, func(offer *mesos.Offer) bool {
				k.fw.Lock()
				defer k.fw.Unlock()
				switch task, state := k.fw.Tasks().Get(task.ID); state {
				case podtask.StatePending:
					// Assess fitness of pod with the current offer. The scheduler normally
					// "backs off" when it can't find an offer that matches up with a pod.
					// The backoff period for a pod can terminate sooner if an offer becomes
					// available that matches up.
					return !task.Has(podtask.Launched) && k.fw.PodScheduler().FitPredicate()(task, offer, nil)
				default:
					// no point in continuing to check for matching offers
					return true
				}
			}))
		}
		delay := k.backoff.Get(podKey)
		log.V(3).Infof("requeuing pod %v with delay %v", podKey, delay)
		k.qr.Requeue(&queuer.Pod{Pod: pod, Delay: &delay, Notify: breakoutEarly})

	default:
		log.V(2).Infof("Task is no longer pending, aborting reschedule for pod %v", podKey)
	}
}
