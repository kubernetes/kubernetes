/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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

// Note: if you change code in this file, you might need to change code in
// contrib/mesos/pkg/scheduler/.

import (
	"time"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/client/record"
	"k8s.io/kubernetes/pkg/conversion"
	"k8s.io/kubernetes/pkg/util/wait"
	"k8s.io/kubernetes/plugin/pkg/ube-scheduler/algorithm"
	"k8s.io/kubernetes/plugin/pkg/ube-scheduler/metrics"
	"k8s.io/kubernetes/plugin/pkg/ube-scheduler/schedulercache"

	"github.com/golang/glog"
)

// Binder knows how to write a binding.
type Binder interface {
	Bind(binding *api.Binding) error
}

// Scheduler watches for new unscheduled pods. It attempts to find
// nodes that they fit on and writes bindings back to the api server.
type Scheduler struct {
	config *Config
}

type Config struct {
	// It is expected that changes made via SchedulerCache will be observed
	// by NodeLister and Algorithm.
	SchedulerCache   schedulercache.Cache
	ClusterLister    algorithm.ClusterLister
	Algorithm        algorithm.ScheduleAlgorithm
	Binder           Binder

	// NextFederationRC should be a function that blocks until the next pod
	// is available. We don't use a channel for this, because scheduling
	// a pod may take some amount of time and we don't want pods to get
	// stale while they sit in a channel.
	NextFederationRC func() *api.ReplicationController

	// Error is called if there is an error. It is passed the pod in
	// question, and the error
	Error            func(*api.ReplicationController, error)

	// Recorder is the EventRecorder to use
	Recorder         record.EventRecorder

	// Close this to shut down the scheduler.
	StopEverything   chan struct{}
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
	federationRC := s.config.NextFederationRC()

	glog.V(3).Infof("Attempting to schedule: %+v", federationRC)
	start := time.Now()
	dest, err := s.config.Algorithm.Schedule(federationRC, s.config.ClusterLister)
	if err != nil {
		glog.V(1).Infof("Failed to schedule: %+v", federationRC)
		s.config.Recorder.Eventf(federationRC, api.EventTypeWarning, "FailedScheduling", "%v", err)
		s.config.Error(federationRC, err)
		return
	}
	metrics.SchedulingAlgorithmLatency.Observe(metrics.SinceInMicroseconds(start))

	//rc := *federationRC
	//TODO: do we need to define Sub-rc or we only set dest to federationRC on Phase 1
	clone, _ := conversion.NewCloner().DeepCopy(federationRC)
	rc := clone.(api.ReplicationController)
	rc.Spec.Template.Spec.ClusterName = dest

	metrics.E2eSchedulingLatency.Observe(metrics.SinceInMicroseconds(start))
}
