/*
Copyright 2014 Google Inc. All rights reserved.

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
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client/record"
	// TODO: move everything from pkg/scheduler into this package. Remove references from registry.
	"github.com/GoogleCloudPlatform/kubernetes/pkg/scheduler"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"

	"github.com/golang/glog"
)

// Binder knows how to write a binding.
type Binder interface {
	Bind(binding *api.Binding) error
}

// Scheduler watches for new unscheduled pods. It attempts to find
// minions that they fit on and writes bindings back to the api server.
type Scheduler struct {
	config *Config
}

type Config struct {
	MinionLister scheduler.MinionLister
	Algorithm    scheduler.Scheduler
	Binder       Binder

	// NextPod should be a function that blocks until the next pod
	// is available. We don't use a channel for this, because scheduling
	// a pod may take some amount of time and we don't want pods to get
	// stale while they sit in a channel.
	NextPod func() *api.Pod

	// Error is called if there is an error. It is passed the pod in
	// question, and the error
	Error func(*api.Pod, error)
}

// New returns a new scheduler.
func New(c *Config) *Scheduler {
	s := &Scheduler{
		config: c,
	}
	return s
}

// Run begins watching and scheduling. It starts a goroutine and returns immediately.
func (s *Scheduler) Run() {
	go util.Forever(s.scheduleOne, 0)
}

func (s *Scheduler) scheduleOne() {
	pod := s.config.NextPod()
	glog.V(3).Infof("Attempting to schedule: %v", pod)
	dest, err := s.config.Algorithm.Schedule(*pod, s.config.MinionLister)
	if err != nil {
		glog.V(1).Infof("Failed to schedule: %v", pod)
		record.Eventf(pod, string(api.PodPending), "failedScheduling", "Error scheduling: %v", err)
		s.config.Error(pod, err)
		return
	}
	b := &api.Binding{
		ObjectMeta: api.ObjectMeta{Namespace: pod.Namespace},
		PodID:      pod.Name,
		Host:       dest,
	}
	if err := s.config.Binder.Bind(b); err != nil {
		glog.V(1).Infof("Failed to bind pod: %v", err)
		record.Eventf(pod, string(api.PodPending), "failedScheduling", "Binding rejected: %v", err)
		s.config.Error(pod, err)
		return
	}
	record.Eventf(pod, string(api.PodPending), "scheduled", "Successfully assigned %v to %v", pod.Name, dest)
}
