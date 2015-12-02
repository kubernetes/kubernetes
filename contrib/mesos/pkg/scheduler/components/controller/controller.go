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

package controller

import (
	"time"

	log "github.com/golang/glog"
	"k8s.io/kubernetes/contrib/mesos/pkg/runtime"
	"k8s.io/kubernetes/contrib/mesos/pkg/scheduler"
	"k8s.io/kubernetes/contrib/mesos/pkg/scheduler/components/algorithm"
	"k8s.io/kubernetes/contrib/mesos/pkg/scheduler/components/binder"
	"k8s.io/kubernetes/contrib/mesos/pkg/scheduler/podtask"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/client/record"
	client "k8s.io/kubernetes/pkg/client/unversioned"
)

const (
	recoveryDelay = 100 * time.Millisecond // delay after scheduler plugin crashes, before we resume scheduling

	FailedScheduling = "FailedScheduling"
	Scheduled        = "Scheduled"
)

type Controller interface {
	Run(<-chan struct{})
}

type controller struct {
	algorithm algorithm.SchedulerAlgorithm
	binder    binder.Binder
	nextPod   func() *api.Pod
	error     func(*api.Pod, error)
	recorder  record.EventRecorder
	client    *client.Client
	started   chan<- struct{} // startup latch
	sched     scheduler.Scheduler
}

func New(sched scheduler.Scheduler, client *client.Client, algorithm algorithm.SchedulerAlgorithm,
	recorder record.EventRecorder, nextPod func() *api.Pod, error func(pod *api.Pod, schedulingErr error),
	binder binder.Binder, started chan<- struct{}) Controller {
	return &controller{
		algorithm: algorithm,
		binder:    binder,
		nextPod:   nextPod,
		error:     error,
		recorder:  recorder,
		client:    client,
		started:   started,
		sched:     sched,
	}
}

func (s *controller) Run(done <-chan struct{}) {
	defer close(s.started)
	go runtime.Until(s.scheduleOne, recoveryDelay, done)
}

// hacked from GoogleCloudPlatform/kubernetes/plugin/pkg/scheduler/scheduler.go,
// with the Modeler stuff removed since we don't use it because we have mesos.
func (s *controller) scheduleOne() {
	pod := s.nextPod()

	// pods which are pre-scheduled (i.e. NodeName is set) are deleted by the kubelet
	// in upstream. Not so in Mesos because the kubelet hasn't see that pod yet. Hence,
	// the scheduler has to take care of this:
	if pod.Spec.NodeName != "" && pod.DeletionTimestamp != nil {
		log.V(3).Infof("deleting pre-scheduled, not yet running pod: %s/%s", pod.Namespace, pod.Name)
		s.client.Pods(pod.Namespace).Delete(pod.Name, api.NewDeleteOptions(0))
		return
	}

	log.V(3).Infof("Attempting to schedule: %+v", pod)
	offer, spec, err := s.algorithm.Schedule(pod)
	if err != nil {
		log.V(1).Infof("Failed to schedule: %+v", pod)
		s.recorder.Eventf(pod, api.EventTypeWarning, FailedScheduling, "Error scheduling: %v", err)
		s.error(pod, err)
		return
	}

	// default upstream scheduler passes pod.Name as binding.Name
	ctx := api.WithNamespace(api.NewContext(), pod.Namespace)
	podKey, err := podtask.MakePodKey(ctx, pod.Name)
	if err != nil {
		offer.Release()
		return
	}

	// meanwhile the task might be deleted (compare the deleter). Using the
	// lock and the registry query we make sure we notice this here.
	s.sched.Lock()
	defer s.sched.Unlock()

	task, state := s.sched.Tasks().ForPod(podKey)
	if state != podtask.StatePending {
		// looks like the pod was deleted between scheduling and launch
		offer.Release()
		return
	}

	if err := s.binder.Bind(task, spec); err != nil {
		log.V(1).Infof("Failed to bind pod: %+v", err)
		offer.Release()
		s.recorder.Eventf(pod, api.EventTypeWarning, FailedScheduling, "Binding rejected: %v", err)
		s.error(pod, err)
		return
	}
	s.recorder.Eventf(pod, api.EventTypeNormal, Scheduled, "Successfully assigned %v to %v", pod.Name, spec.AssignedSlave)
}
