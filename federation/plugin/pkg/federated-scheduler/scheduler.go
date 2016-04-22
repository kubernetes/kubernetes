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

	"k8s.io/kubernetes/pkg/api"
	pkgunversioned "k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/federation/apis/federation/unversioned"
	extensions "k8s.io/kubernetes/pkg/apis/extensions/v1beta1"
	federation "k8s.io/kubernetes/federation/apis/federation/v1alpha1"
	"k8s.io/kubernetes/pkg/client/record"
	"k8s.io/kubernetes/pkg/conversion"
	"k8s.io/kubernetes/pkg/util/wait"
	"k8s.io/kubernetes/federation/plugin/pkg/federated-scheduler/algorithm"
	"k8s.io/kubernetes/federation/plugin/pkg/federated-scheduler/metrics"
	"k8s.io/kubernetes/federation/plugin/pkg/federated-scheduler/schedulercache"


	"github.com/golang/glog"
	"fmt"

)

// Binder knows how to write a binding.
type Binder interface {
	Bind(subRS *federation.SubReplicaSet) error
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
	glog.V(3).Infof("Attempting to schedule: %+v", rs)
	scheduled, err := isScheduled(rs, s.config.SchedulerCache)
	if err != nil {
		glog.V(1).Infof("Failed to schedule: %+v", rs)
		s.config.Recorder.Eventf(rs, v1.EventTypeWarning, "FailedScheduling", "%v", err)
		s.config.Error(rs, err)
		return
	}
	if  scheduled {
		glog.V(3).Infof("%v has been scheduled, skip", rs.Name)
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

	//1. split subReplicaSet
	//2. bind dest cluster on subReplicaSet
	//3. set annotation of parent replicaSet on subReplicaSet

	subRS, err := splitSubReplicaSet(rs, dest)
	if err != nil {
		glog.V(1).Infof("Failed to split sub replicaset: %+v", rs)
		s.config.Recorder.Eventf(rs, v1.EventTypeWarning, "FailedScheduling", "%v", err)
		s.config.Error(rs, err)
		return
	}
	//bind the destination cluster to sub rc

	bindAction := func() bool {
		bindingStart := time.Now()
		err := s.config.Binder.Bind(subRS)
		if err != nil {
			glog.V(1).Infof("Failed to bind replicaSet: %+v", err)
			s.config.Recorder.Eventf(rs, v1.EventTypeNormal, "FailedBinding", "Binding rejected: %v", err)
			s.config.Error(rs, err)
			return false
		}
		metrics.BindingLatency.Observe(metrics.SinceInMicroseconds(bindingStart))
		s.config.Recorder.Eventf(rs, v1.EventTypeNormal, "Scheduled", "Successfully assigned %v to %v", rs.Name, dest)
		return true
	}
	s.config.SchedulerCache.AssumeReplicaSetIfBindSucceed(rs, bindAction)
	metrics.E2eSchedulingLatency.Observe(metrics.SinceInMicroseconds(start))
}

func splitSubReplicaSet(replicaSet *extensions.ReplicaSet, targetCluster string) (*federation.SubReplicaSet, error) {
	subRS, err := generateSubRS(replicaSet)
	if err != nil {
		return nil, err
	}
	// Get the current annotations from the object.
	annotations := subRS.Annotations
	if annotations == nil {
		annotations = map[string]string{}
	}
	// Set federation ReplicaSet name
	annotations[unversioned.FederationReplicaSetKey] = replicaSet.Name
	// Set target cluster name, which is binding cluster
	annotations[unversioned.TargetClusterKey] = targetCluster
	// Update annotation
	subRS.Annotations = annotations
	return subRS, nil
}

func generateSubRS(replicaSet *extensions.ReplicaSet) (*federation.SubReplicaSet, error) {
	clone, err := conversion.NewCloner().DeepCopy(replicaSet)
	if err != nil {
		return nil, err
	}
	rsTemp, ok := clone.(*extensions.ReplicaSet)
	if !ok {
		return nil, fmt.Errorf("Unexpected replicaset cast error : %v\n", rsTemp)
	}
	result := &federation.SubReplicaSet{
		TypeMeta: pkgunversioned.TypeMeta{
			Kind: "subreplicasets",
			APIVersion: "federation/v1alpha1",
		},
	}
	result.ObjectMeta = rsTemp.ObjectMeta
	result.Spec = rsTemp.Spec
	result.Status = rsTemp.Status

	//to generate subrs name, we need a api.ObjectMeta instead of v1
	meta := &api.ObjectMeta{}
	meta.GenerateName = result.ObjectMeta.Name + "-"

	api.GenerateName(api.SimpleNameGenerator, meta)
	result.Name = meta.Name
	result.GenerateName = rsTemp.Name
	//unset resourceVersion before create the actual resource
	result.ResourceVersion = ""
	result.TypeMeta.Kind = "subreplicasets"
	result.TypeMeta.APIVersion = "federation/v1alpha1"
	fmt.Println("generating subreplicaset %v", result)
	return result, nil
}

func CoverSubRSToRS(subRS *federation.SubReplicaSet) ( *extensions.ReplicaSet, error) {
	clone, err := conversion.NewCloner().DeepCopy(subRS)
	if err != nil {
		return nil, err
	}
	subrs, ok := clone.(*federation.SubReplicaSet)
	if !ok {
		return nil, fmt.Errorf("Unexpected subreplicaset cast error : %v\n", subrs)
	}
	result := &extensions.ReplicaSet{
		TypeMeta: pkgunversioned.TypeMeta{
			Kind: "replicaset",
			APIVersion: "extensions/v1beta1",
		},
	}
	result.ObjectMeta = subrs.ObjectMeta
	result.Spec = subrs.Spec
	result.Status = subrs.Status
	result.Name = subrs.GenerateName
	return result, nil
}
func isScheduled(rs *extensions.ReplicaSet,cache schedulercache.Cache) (bool, error){
	replicaSets, err := cache.List()
	if err != nil {
		return false, err
	}
	for _, scheduledRS := range replicaSets {
		if rs.Name == scheduledRS.Name {
			return true, nil
		}
	}
	return false, nil
}