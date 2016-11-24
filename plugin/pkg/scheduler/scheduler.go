/*
Copyright 2014 The Kubernetes Authors.

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

package scheduler

import (
	"time"

	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/client/record"
	"k8s.io/kubernetes/pkg/util/wait"
	"k8s.io/kubernetes/plugin/pkg/scheduler/algorithm"
	"k8s.io/kubernetes/plugin/pkg/scheduler/metrics"
	"k8s.io/kubernetes/plugin/pkg/scheduler/schedulercache"

	"github.com/golang/glog"
)

// Binder knows how to write a binding.
type Binder interface {
	Bind(binding *v1.Binding) error
}

type PodConditionUpdater interface {
	Update(pod *v1.Pod, podCondition *v1.PodCondition) error
}

// Scheduler watches for new unscheduled pods. It attempts to find
// nodes that they fit on and writes bindings back to the api server.
type Scheduler struct {
	config *Config
}

type Config struct {
	// It is expected that changes made via SchedulerCache will be observed
	// by NodeLister and Algorithm.
	SchedulerCache schedulercache.Cache
	NodeLister     algorithm.NodeLister
	Algorithm      algorithm.ScheduleAlgorithm
	Binder         Binder
	// PodConditionUpdater is used only in case of scheduling errors. If we succeed
	// with scheduling, PodScheduled condition will be updated in apiserver in /bind
	// handler so that binding and setting PodCondition it is atomic.
	PodConditionUpdater PodConditionUpdater

	// NextPod should be a function that blocks until the next pod
	// is available. We don't use a channel for this, because scheduling
	// a pod may take some amount of time and we don't want pods to get
	// stale while they sit in a channel.
	NextPod func() *v1.Pod

	// Error is called if there is an error. It is passed the pod in
	// question, and the error
	Error func(*v1.Pod, error)

	// Recorder is the EventRecorder to use
	Recorder record.EventRecorder

	// Close this to shut down the scheduler.
	StopEverything chan struct{}
}

// New returns a new scheduler.
func New(c *Config) *Scheduler {
	s := &Scheduler{
		config: c,
	}
	metrics.Register()
	return s
}

// Run begins watching and scheduling. It starts a goroutine and returns immediately.
func (s *Scheduler) Run() {
	go wait.Until(s.scheduleOne, 0, s.config.StopEverything)
}

func (s *Scheduler) scheduleOne() {
	pod := s.config.NextPod()

	glog.V(3).Infof("Attempting to schedule pod: %v/%v", pod.Namespace, pod.Name)
	start := time.Now()
	dest, err := s.config.Algorithm.Schedule(pod, s.config.NodeLister)
	if err != nil {
		glog.V(1).Infof("Failed to schedule pod: %v/%v", pod.Namespace, pod.Name)
		s.config.Error(pod, err)
		s.config.Recorder.Eventf(pod, v1.EventTypeWarning, "FailedScheduling", "%v", err)
		s.config.PodConditionUpdater.Update(pod, &v1.PodCondition{
			Type:   v1.PodScheduled,
			Status: v1.ConditionFalse,
			Reason: v1.PodReasonUnschedulable,
		})
		return
	}
	metrics.SchedulingAlgorithmLatency.Observe(metrics.SinceInMicroseconds(start))

	// Optimistically assume that the binding will succeed and send it to apiserver
	// in the background.
	// If the binding fails, scheduler will release resources allocated to assumed pod
	// immediately.
	assumed := *pod
	assumed.Spec.NodeName = dest
	if err := s.config.SchedulerCache.AssumePod(&assumed); err != nil {
		glog.Errorf("scheduler cache AssumePod failed: %v", err)
		// TODO: This means that a given pod is already in cache (which means it
		// is either assumed or already added). This is most probably result of a
		// BUG in retrying logic. As a temporary workaround (which doesn't fully
		// fix the problem, but should reduce its impact), we simply return here,
		// as binding doesn't make sense anyway.
		// This should be fixed properly though.
		return
	}

	go func() {
		defer metrics.E2eSchedulingLatency.Observe(metrics.SinceInMicroseconds(start))

		b := &v1.Binding{
			ObjectMeta: v1.ObjectMeta{Namespace: pod.Namespace, Name: pod.Name},
			Target: v1.ObjectReference{
				Kind: "Node",
				Name: dest,
			},
		}

		bindingStart := time.Now()
		// If binding succeeded then PodScheduled condition will be updated in apiserver so that
		// it's atomic with setting host.
		err := s.config.Binder.Bind(b)
		if err != nil {
			glog.V(1).Infof("Failed to bind pod: %v/%v", pod.Namespace, pod.Name)
			if err := s.config.SchedulerCache.ForgetPod(&assumed); err != nil {
				glog.Errorf("scheduler cache ForgetPod failed: %v", err)
			}
			s.config.Error(pod, err)
			s.config.Recorder.Eventf(pod, v1.EventTypeNormal, "FailedScheduling", "Binding rejected: %v", err)
			s.config.PodConditionUpdater.Update(pod, &v1.PodCondition{
				Type:   v1.PodScheduled,
				Status: v1.ConditionFalse,
				Reason: "BindingRejected",
			})
			return
		}
		metrics.BindingLatency.Observe(metrics.SinceInMicroseconds(bindingStart))
		s.config.Recorder.Eventf(pod, v1.EventTypeNormal, "Scheduled", "Successfully assigned %v to %v", pod.Name, dest)
	}()
}
