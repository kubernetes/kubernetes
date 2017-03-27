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

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/client-go/tools/record"
	"k8s.io/kubernetes/pkg/api/v1"
	clientset "k8s.io/kubernetes/pkg/client/clientset_generated/clientset"
	corelisters "k8s.io/kubernetes/pkg/client/listers/core/v1"
	"k8s.io/kubernetes/plugin/pkg/scheduler/algorithm"
	schedulerapi "k8s.io/kubernetes/plugin/pkg/scheduler/api"
	"k8s.io/kubernetes/plugin/pkg/scheduler/metrics"
	"k8s.io/kubernetes/plugin/pkg/scheduler/schedulercache"
	"k8s.io/kubernetes/plugin/pkg/scheduler/util"

	"github.com/golang/glog"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/client-go/tools/cache"
)

// Binder knows how to write a binding.
type Binder interface {
	Bind(binding *v1.Binding) error
}

// PodConditionUpdater updates the condition of a pod based on the passed
// PodCondition
type PodConditionUpdater interface {
	Update(pod *v1.Pod, podCondition *v1.PodCondition) error
}

// Scheduler watches for new unscheduled pods. It attempts to find
// nodes that they fit on and writes bindings back to the api server.
type Scheduler struct {
	config *Config
}

// StopEverything closes the scheduler config's StopEverything channel, to shut
// down the Scheduler.
func (sched *Scheduler) StopEverything() {
	close(sched.config.StopEverything)
}

// Configurator defines I/O, caching, and other functionality needed to
// construct a new scheduler. An implementation of this can be seen in
// factory.go.
type Configurator interface {
	GetPriorityFunctionConfigs(priorityKeys sets.String) ([]algorithm.PriorityConfig, error)
	GetPriorityMetadataProducer() (algorithm.MetadataProducer, error)
	GetPredicateMetadataProducer() (algorithm.MetadataProducer, error)
	GetPredicates(predicateKeys sets.String) (map[string]algorithm.FitPredicate, error)
	GetHardPodAffinitySymmetricWeight() int
	GetSchedulerName() string
	MakeDefaultErrorFunc(backoff *util.PodBackoff, podQueue *cache.FIFO) func(pod *v1.Pod, err error)

	// Probably doesn't need to be public.  But exposed for now in case.
	ResponsibleForPod(pod *v1.Pod) bool

	// Needs to be exposed for things like integration tests where we want to make fake nodes.
	GetNodeLister() corelisters.NodeLister
	GetClient() clientset.Interface
	GetScheduledPodLister() corelisters.PodLister
	Run()

	Create() (*Config, error)
	CreateFromProvider(providerName string) (*Config, error)
	CreateFromConfig(policy schedulerapi.Policy) (*Config, error)
	CreateFromKeys(predicateKeys, priorityKeys sets.String, extenders []algorithm.SchedulerExtender) (*Config, error)
}

// Config is an implementation of the Scheduler's configured input data.
// TODO over time we should make this struct a hidden implementation detail of the scheduler.
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
// TODO replace this with NewFromConfigurator.
func New(c *Config) *Scheduler {
	s := &Scheduler{
		config: c,
	}
	metrics.Register()
	return s
}

// NewFromConfigurator returns a new scheduler that is created entirely by the Configurator.  Assumes Create() is implemented.
// Supports intermediate Config mutation for now if you provide modifier functions which will run after Config is created.
func NewFromConfigurator(c Configurator, modifiers ...func(c *Config)) (*Scheduler, error) {
	cfg, err := c.Create()
	if err != nil {
		return nil, err
	}
	// Mutate it if any functions were provided, changes might be required for certain types of tests (i.e. change the recorder).
	for _, modifier := range modifiers {
		modifier(cfg)
	}
	// From this point on the config is immutable to the outside.
	s := &Scheduler{
		config: cfg,
	}
	metrics.Register()
	return s, nil
}

// Run begins watching and scheduling. It starts a goroutine and returns immediately.
func (sched *Scheduler) Run() {
	go wait.Until(sched.scheduleOne, 0, sched.config.StopEverything)
}

func (sched *Scheduler) scheduleOne() {
	pod := sched.config.NextPod()
	if pod.DeletionTimestamp != nil {
		sched.config.Recorder.Eventf(pod, v1.EventTypeWarning, "FailedScheduling", "skip schedule deleting pod: %v/%v", pod.Namespace, pod.Name)
		glog.V(3).Infof("Skip schedule deleting pod: %v/%v", pod.Namespace, pod.Name)
		return
	}

	glog.V(3).Infof("Attempting to schedule pod: %v/%v", pod.Namespace, pod.Name)
	start := time.Now()
	dest, err := sched.config.Algorithm.Schedule(pod, sched.config.NodeLister)
	if err != nil {
		glog.V(1).Infof("Failed to schedule pod: %v/%v", pod.Namespace, pod.Name)
		sched.config.Error(pod, err)
		sched.config.Recorder.Eventf(pod, v1.EventTypeWarning, "FailedScheduling", "%v", err)
		sched.config.PodConditionUpdater.Update(pod, &v1.PodCondition{
			Type:    v1.PodScheduled,
			Status:  v1.ConditionFalse,
			Reason:  v1.PodReasonUnschedulable,
			Message: err.Error(),
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
	if err := sched.config.SchedulerCache.AssumePod(&assumed); err != nil {
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
			ObjectMeta: metav1.ObjectMeta{Namespace: pod.Namespace, Name: pod.Name},
			Target: v1.ObjectReference{
				Kind: "Node",
				Name: dest,
			},
		}

		bindingStart := time.Now()
		// If binding succeeded then PodScheduled condition will be updated in apiserver so that
		// it's atomic with setting host.
		err := sched.config.Binder.Bind(b)
		if err := sched.config.SchedulerCache.FinishBinding(&assumed); err != nil {
			glog.Errorf("scheduler cache FinishBinding failed: %v", err)
		}
		if err != nil {
			glog.V(1).Infof("Failed to bind pod: %v/%v", pod.Namespace, pod.Name)
			if err := sched.config.SchedulerCache.ForgetPod(&assumed); err != nil {
				glog.Errorf("scheduler cache ForgetPod failed: %v", err)
			}
			sched.config.Error(pod, err)
			sched.config.Recorder.Eventf(pod, v1.EventTypeNormal, "FailedScheduling", "Binding rejected: %v", err)
			sched.config.PodConditionUpdater.Update(pod, &v1.PodCondition{
				Type:   v1.PodScheduled,
				Status: v1.ConditionFalse,
				Reason: "BindingRejected",
			})
			return
		}
		metrics.BindingLatency.Observe(metrics.SinceInMicroseconds(bindingStart))
		sched.config.Recorder.Eventf(pod, v1.EventTypeNormal, "Scheduled", "Successfully assigned %v to %v", pod.Name, dest)
	}()
}
