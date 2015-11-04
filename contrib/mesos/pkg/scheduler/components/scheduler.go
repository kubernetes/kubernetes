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

package components

import (
	"net/http"
	"sync"

	"k8s.io/kubernetes/contrib/mesos/pkg/backoff"
	"k8s.io/kubernetes/contrib/mesos/pkg/offers"
	"k8s.io/kubernetes/contrib/mesos/pkg/queue"
	"k8s.io/kubernetes/contrib/mesos/pkg/runtime"
	"k8s.io/kubernetes/contrib/mesos/pkg/scheduler"
	"k8s.io/kubernetes/contrib/mesos/pkg/scheduler/components/algorithm"
	"k8s.io/kubernetes/contrib/mesos/pkg/scheduler/components/binder"
	"k8s.io/kubernetes/contrib/mesos/pkg/scheduler/components/deleter"
	"k8s.io/kubernetes/contrib/mesos/pkg/scheduler/components/errorhandler"
	"k8s.io/kubernetes/contrib/mesos/pkg/scheduler/components/framework"
	"k8s.io/kubernetes/contrib/mesos/pkg/scheduler/components/podreconciler"
	"k8s.io/kubernetes/contrib/mesos/pkg/scheduler/components/schedulerloop"
	"k8s.io/kubernetes/contrib/mesos/pkg/scheduler/config"
	"k8s.io/kubernetes/contrib/mesos/pkg/scheduler/podschedulers"
	"k8s.io/kubernetes/contrib/mesos/pkg/scheduler/podtask"
	"k8s.io/kubernetes/contrib/mesos/pkg/scheduler/queuer"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/client/cache"
	"k8s.io/kubernetes/pkg/client/record"
	client "k8s.io/kubernetes/pkg/client/unversioned"
)

// Scheduler implements types.Scheduler
type sched struct {
	podReconciler podreconciler.PodReconciler
	framework     framework.Framework
	loop          schedulerloop.SchedulerLoop

	// unsafe state, needs to be guarded, especially changes to podtask.T objects
	sync.RWMutex
	taskRegistry podtask.Registry
}

func NewScheduler(c *config.Config, fw framework.Framework, ps podschedulers.PodScheduler,
	client *client.Client, recorder record.EventRecorder, terminate <-chan struct{}, mux *http.ServeMux, lw *cache.ListWatch) scheduler.Scheduler {

	core := &sched{
		framework:    fw,
		taskRegistry: podtask.NewInMemoryRegistry(),
	}

	// Watch and queue pods that need scheduling.
	updates := make(chan queue.Entry, c.UpdatesBacklog)
	podUpdates := &podStoreAdapter{queue.NewHistorical(updates)}
	reflector := cache.NewReflector(lw, &api.Pod{}, podUpdates, 0)

	q := queuer.New(podUpdates)

	algorithm := algorithm.New(core, podUpdates, ps)

	podDeleter := deleter.New(core, q)

	core.podReconciler = podreconciler.New(core, client, q, podDeleter)

	bo := backoff.New(c.InitialPodBackoff.Duration, c.MaxPodBackoff.Duration)
	errorHandler := errorhandler.New(core, bo, q, ps)

	binder := binder.New(core)

	startLatch := make(chan struct{})

	runtime.On(startLatch, func() {
		reflector.Run() // TODO(jdef) should listen for termination
		podDeleter.Run(updates, terminate)
		q.Run(terminate)

		q.InstallDebugHandlers(mux)
		podtask.InstallDebugHandlers(core.Tasks(), mux)
	})

	core.loop = schedulerloop.New(client, algorithm, recorder, q.Yield, errorHandler.Error, binder, startLatch)
	return core
}

func (c *sched) Run(done <-chan struct{}) {
	c.loop.Run(done)
}

func (c *sched) Reconcile(t *podtask.T) {
	c.podReconciler.Reconcile(t)
}

func (c *sched) Tasks() podtask.Registry {
	return c.taskRegistry
}

func (c *sched) Offers() offers.Registry {
	return c.framework.Offers()
}

func (c *sched) KillTask(id string) error {
	return c.framework.KillTask(id)
}

func (c *sched) LaunchTask(t *podtask.T) error {
	return c.framework.LaunchTask(t)
}
