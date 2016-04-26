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

import (
	"time"

	"k8s.io/kubernetes/pkg/api/v1"

	extensions "k8s.io/kubernetes/pkg/apis/extensions/v1beta1"
	"k8s.io/kubernetes/pkg/client/record"
	"k8s.io/kubernetes/pkg/util/wait"
	"k8s.io/kubernetes/federation/apis/federation/unversioned"
	"k8s.io/kubernetes/federation/plugin/pkg/federated-scheduler/algorithm"
	"k8s.io/kubernetes/federation/plugin/pkg/federated-scheduler/metrics"
	"k8s.io/kubernetes/federation/plugin/pkg/federated-scheduler/schedulercache"

	"github.com/golang/glog"
)

// Binder knows how to write a binding.
type Binder interface {
	Bind(replicaSet *extensions.ReplicaSet) error
}

// Scheduler watches for new unscheduled replicaSets. It attempts to find
// nodes that they fit on and writes bindings back to the api server.
type Scheduler struct {
	config *Config
}

type Config struct {
	// It is expected that changes made via SchedulerCache will be observed
	// by NodeLister and Algorithm.
	SchedulerCache schedulercache.Cache
	ClusterLister  algorithm.ClusterLister
	Algorithm      algorithm.ScheduleAlgorithm
	Binder         Binder

	// NextReplicaSet should be a function that blocks until the next replicaSet
	// is available. We don't use a channel for this, because scheduling
	// a replicaSet may take some amount of time and we don't want replicaset to get
	// stale while they sit in a channel.
	NextReplicaSet func() *extensions.ReplicaSet

	// Error is called if there is an error. It is passed the replicaSet in
	// question, and the error
	Error          func(*extensions.ReplicaSet, error)

	// Recorder is the EventRecorder to use
	Recorder       record.EventRecorder

	// Close this to shut down the federated-scheduler.
	StopEverything chan struct{}
}

// New returns a new federated-scheduler.
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
	rs := s.config.NextReplicaSet()
	glog.V(3).Infof("Attempting to schedule: %+v", rs.Name)
	if IsScheduled(rs) {
		glog.V(4).Infof("%v has been scheduled", rs.Name)
		//we do not have a way to filer scheduled rs at this moment, so we need to check if the rs is scheduled
		//and we do not want to send event for such cases.
		//We may have more reasonable solution in later phase
		//s.config.Recorder.Eventf(rs, v1.EventTypeNormal, "Scheduled", "%v has been scheduled", rs.Name)
		return
	}
	start := time.Now()
	dest, err := s.config.Algorithm.Schedule(rs, s.config.ClusterLister)
	if err != nil {
		glog.V(1).Infof("Failed to schedule: %+v", rs)
		s.config.Recorder.Eventf(rs, v1.EventTypeWarning, "FailedScheduling", "%v", err)
		s.config.Error(rs, err)
		return
	}
	metrics.SchedulingAlgorithmLatency.Observe(metrics.SinceInMicroseconds(start))
	rs.Annotations = map[string]string{}
	rs.Annotations[unversioned.TargetClusterKey] = dest
	assumed := *rs
	s.config.SchedulerCache.AssumeReplicaSet(&assumed)

	go func() {
		defer metrics.E2eSchedulingLatency.Observe(metrics.SinceInMicroseconds(start))
		bindingStart := time.Now()
		err := s.config.Binder.Bind(rs)
		if err != nil {
			glog.V(1).Infof("Failed to bind replicaSet: %+v", err)
			s.config.Recorder.Eventf(rs, v1.EventTypeWarning, "FailedBinding", "Binding rejected: %v", err)
			s.config.Error(rs, err)
			return
		}
		metrics.BindingLatency.Observe(metrics.SinceInMicroseconds(bindingStart))
		s.config.Recorder.Eventf(rs, v1.EventTypeNormal, "Scheduled", "Successfully assigned %v to %v", rs.Name, dest)
	}()
}

//check if the rs has been scheduled, as we do not change the rs after schedule, so we need to iterate rs in cache
func IsScheduled(rs *extensions.ReplicaSet) (bool){
	clusterSelection, ok := rs.Annotations[unversioned.TargetClusterKey]
	if ok && clusterSelection != "" {
		return true
	}
	return false
}
