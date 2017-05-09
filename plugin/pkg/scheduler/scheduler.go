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

// Note: if you change code in this file, you might need to change code in
// contrib/mesos/pkg/scheduler/.

import (
	"errors"
	"fmt"
	"time"

	"k8s.io/kubernetes/pkg/api"
	utilpod "k8s.io/kubernetes/pkg/api/pod"
	"k8s.io/kubernetes/pkg/client/record"
	"k8s.io/kubernetes/pkg/util/wait"
	"k8s.io/kubernetes/plugin/pkg/scheduler/algorithm"
	"k8s.io/kubernetes/plugin/pkg/scheduler/metrics"
	"k8s.io/kubernetes/plugin/pkg/scheduler/schedulercache"

	"github.com/golang/glog"
	utilnet "k8s.io/kubernetes/pkg/util/net"
)

// Binder knows how to write a binding.
type Binder interface {
	Bind(binding *api.Binding) error
}

type PodConditionUpdater interface {
	Update(pod *api.Pod, podCondition *api.PodCondition) error
}

type PodUpdater interface {
	Update(pod *api.Pod) error
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

	PodUpdater PodUpdater
	// NextPod should be a function that blocks until the next pod
	// is available. We don't use a channel for this, because scheduling
	// a pod may take some amount of time and we don't want pods to get
	// stale while they sit in a channel.
	NextPod func() *api.Pod

	// Error is called if there is an error. It is passed the pod in
	// question, and the error
	Error func(*api.Pod, error)

	// Recorder is the EventRecorder to use
	Recorder record.EventRecorder

	HostPortRange utilnet.PortRange
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

func getUsedPorts(pods ...*api.Pod) map[int]bool {
	// TODO: Aggregate it at the NodeInfo level.
	ports := make(map[int]bool)
	for _, pod := range pods {
		for _, container := range pod.Spec.Containers {
			for _, podPort := range container.Ports {
				// "0" is explicitly ignored in PodFitsHostPorts,
				// which is the only function that uses this value.
				if podPort.HostPort != 0 {
					ports[int(podPort.HostPort)] = true
				}
			}
		}
	}
	return ports
}

func getAvaPort(allocated map[int]bool, base, max, count int) (int, bool) {
	if count >= max {
		return 0, false
	}
	for i := 0; i < max; i++ {
		if allocated[base+i] != true {
			return base + i, true
		}
	}
	return 0, false
}

func (s *Scheduler) AssignPort(pod *api.Pod, dest string) error {
	_, autoport := pod.ObjectMeta.Annotations[utilpod.PodAutoPortAnnotation]
	if !autoport {
		return nil
	}

	nodeNameToInfo := make(map[string]*schedulercache.NodeInfo)
	err := s.config.SchedulerCache.UpdateNodeNameToInfoMap(nodeNameToInfo)
	if err != nil {
		return err
	}

	nodeInfo := nodeNameToInfo[dest]
	usedPorts := getUsedPorts(nodeInfo.Pods()...)
	for i := range pod.Spec.Containers {
		for j := range pod.Spec.Containers[i].Ports {
			if pod.Spec.Containers[i].Ports[j].HostPort == 0 {
				port, isAssigned := getAvaPort(usedPorts, s.config.HostPortRange.Base, s.config.HostPortRange.Size,
					len(usedPorts))
				if isAssigned {
					pod.Spec.Containers[i].Ports[j].HostPort = int32(port)
					if pod.Spec.SecurityContext.HostNetwork {
						pod.Spec.Containers[i].Ports[j].ContainerPort = int32(port)
					}
					usedPorts[port] = true
				} else {
					glog.V(1).Infof("no hostport can be assigned")
					return errors.New("no hostport can be assigned")
				}
			}
			envVariable := api.EnvVar{
				Name:  fmt.Sprintf("PORT%d", j),
				Value: fmt.Sprintf("%d", pod.Spec.Containers[i].Ports[j].HostPort),
			}
			pod.Spec.Containers[i].Env = append(pod.Spec.Containers[i].Env, envVariable)
			if j == 0 {
				pod.Spec.Containers[i].Env = append(pod.Spec.Containers[i].Env, api.EnvVar{
					Name:  "PORT",
					Value: fmt.Sprintf("%d", pod.Spec.Containers[i].Ports[j].HostPort),
				})
			}
		}
	}

	return nil
}

func (s *Scheduler) scheduleOne() {
	pod := s.config.NextPod()

	glog.V(3).Infof("Attempting to schedule pod: %v/%v", pod.Namespace, pod.Name)
	start := time.Now()
	dest, err := s.config.Algorithm.Schedule(pod, s.config.NodeLister)
	if err != nil {
		glog.V(1).Infof("Failed to schedule pod: %v/%v", pod.Namespace, pod.Name)
		s.config.Error(pod, err)
		s.config.Recorder.Eventf(pod, api.EventTypeWarning, "FailedScheduling", "%v", err)
		s.config.PodConditionUpdater.Update(pod, &api.PodCondition{
			Type:   api.PodScheduled,
			Status: api.ConditionFalse,
			Reason: "Unschedulable",
		})
		return
	}
	portErr := s.AssignPort(pod, dest)
	if portErr != nil {
		glog.V(1).Infof("Failed to schedule: %+v, failed to assign host port.", pod)
		s.config.Error(pod, portErr)
		s.config.Recorder.Eventf(pod, api.EventTypeWarning, "FailedScheduling", "%v", portErr)
		s.config.PodConditionUpdater.Update(pod, &api.PodCondition{
			Type:   api.PodScheduled,
			Status: api.ConditionFalse,
			Reason: "Unschedulable",
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

		errUpdate := s.config.PodUpdater.Update(pod)
		if errUpdate != nil {
			glog.V(1).Infof("Failed to Update pod: %v/%v %v", pod.Namespace, pod.Name, errUpdate)
			s.config.Error(pod, errUpdate)
			s.config.Recorder.Eventf(pod, api.EventTypeNormal, "FailedScheduling", "Update failed: %v", errUpdate)
			s.config.PodConditionUpdater.Update(pod, &api.PodCondition{
				Type:   api.PodScheduled,
				Status: api.ConditionFalse,
				Reason: "BindingRejected",
			})
			return
		}
		b := &api.Binding{
			ObjectMeta: api.ObjectMeta{Namespace: pod.Namespace, Name: pod.Name},
			Target: api.ObjectReference{
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
			s.config.Recorder.Eventf(pod, api.EventTypeNormal, "FailedScheduling", "Binding rejected: %v", err)
			s.config.PodConditionUpdater.Update(pod, &api.PodCondition{
				Type:   api.PodScheduled,
				Status: api.ConditionFalse,
				Reason: "BindingRejected",
			})
			return
		}
		metrics.BindingLatency.Observe(metrics.SinceInMicroseconds(bindingStart))
		s.config.Recorder.Eventf(pod, api.EventTypeNormal, "Scheduled", "Successfully assigned %v to %v", pod.Name, dest)
	}()
}
