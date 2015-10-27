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
	"time"

	log "github.com/golang/glog"
	"k8s.io/kubernetes/contrib/mesos/pkg/runtime"
	"k8s.io/kubernetes/contrib/mesos/pkg/scheduler/podtask"
	"k8s.io/kubernetes/contrib/mesos/pkg/scheduler/queuer"
	types "k8s.io/kubernetes/contrib/mesos/pkg/scheduler/types"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/client/record"
	client "k8s.io/kubernetes/pkg/client/unversioned"
)

const (
	recoveryDelay = 100 * time.Millisecond // delay after scheduler plugin crashes, before we resume scheduling

	FailedScheduling = "FailedScheduling"
	Scheduled        = "Scheduled"
)

type SchedulerLoopInterface interface {
	ReconcilePodTask(t *podtask.T)

	// execute the Scheduling plugin, should start a go routine and return immediately
	Run(<-chan struct{})
}

type SchedulerLoopConfig struct {
	Algorithm *SchedulerAlgorithm
	Binder    *Binder
	NextPod   func() *api.Pod
	Error     func(*api.Pod, error)
	Recorder  record.EventRecorder
	Fw        types.Framework
	Client    *client.Client
	Qr        *queuer.Queuer
	Pr        *PodReconciler
	Starting  chan struct{} // startup latch
}

func NewSchedulerLoop(c *SchedulerLoopConfig) SchedulerLoopInterface {
	return &SchedulerLoop{
		algorithm: c.Algorithm,
		binder:    c.Binder,
		nextPod:   c.NextPod,
		error:     c.Error,
		recorder:  c.Recorder,
		fw:        c.Fw,
		client:    c.Client,
		qr:        c.Qr,
		pr:        c.Pr,
		starting:  c.Starting,
	}
}

type SchedulerLoop struct {
	algorithm *SchedulerAlgorithm
	binder    *Binder
	nextPod   func() *api.Pod
	error     func(*api.Pod, error)
	recorder  record.EventRecorder

	fw       types.Framework
	client   *client.Client
	qr       *queuer.Queuer
	pr       *PodReconciler
	starting chan struct{}
}

func (s *SchedulerLoop) Run(done <-chan struct{}) {
	defer close(s.starting)
	go runtime.Until(s.scheduleOne, recoveryDelay, done)
}

// hacked from GoogleCloudPlatform/kubernetes/plugin/pkg/scheduler/scheduler.go,
// with the Modeler stuff removed since we don't use it because we have mesos.
func (s *SchedulerLoop) scheduleOne() {
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
	dest, err := s.algorithm.Schedule(pod)
	if err != nil {
		log.V(1).Infof("Failed to schedule: %+v", pod)
		s.recorder.Eventf(pod, FailedScheduling, "Error scheduling: %v", err)
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
		s.recorder.Eventf(pod, FailedScheduling, "Binding rejected: %v", err)
		s.error(pod, err)
		return
	}
	s.recorder.Eventf(pod, Scheduled, "Successfully assigned %v to %v", pod.Name, dest)
}

func (s *SchedulerLoop) ReconcilePodTask(t *podtask.T) {
	s.pr.Reconcile(t)
}
