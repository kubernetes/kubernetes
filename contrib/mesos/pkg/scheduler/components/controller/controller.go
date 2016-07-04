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

package controller

import (
	"time"

	clientset "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset"

	log "github.com/golang/glog"
	"k8s.io/kubernetes/contrib/mesos/pkg/runtime"
	"k8s.io/kubernetes/contrib/mesos/pkg/scheduler/components/algorithm"
	"k8s.io/kubernetes/contrib/mesos/pkg/scheduler/components/binder"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/client/record"
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
	client    *clientset.Clientset
	started   chan<- struct{} // startup latch
}

func New(client *clientset.Clientset, algorithm algorithm.SchedulerAlgorithm,
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
		s.client.Core().Pods(pod.Namespace).Delete(pod.Name, api.NewDeleteOptions(0))
		return
	}

	log.V(3).Infof("Attempting to schedule: %+v", pod)
	dest, err := s.algorithm.Schedule(pod)
	if err != nil {
		log.V(1).Infof("Failed to schedule: %+v", pod)
		s.recorder.Eventf(pod, api.EventTypeWarning, FailedScheduling, "Error scheduling: %v", err)
		s.error(pod, err)
		return
	}
	b := &api.Binding{
		ObjectMeta: api.ObjectMeta{Namespace: pod.Namespace, Name: pod.Name},
		Target: api.ObjectReference{
			Kind: "Node",
			Name: dest,
		},
	}
	if err := s.binder.Bind(b); err != nil {
		log.V(1).Infof("Failed to bind pod: %+v", err)
		s.recorder.Eventf(pod, api.EventTypeWarning, FailedScheduling, "Binding rejected: %v", err)
		s.error(pod, err)
		return
	}
	s.recorder.Eventf(pod, api.EventTypeNormal, Scheduled, "Successfully assigned %v to %v", pod.Name, dest)
}
