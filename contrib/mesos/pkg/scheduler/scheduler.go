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

package scheduler

import (
	"sync"

	"k8s.io/kubernetes/contrib/mesos/pkg/offers"
	"k8s.io/kubernetes/contrib/mesos/pkg/scheduler/podtask"
)

// Scheduler abstracts everything other components of the scheduler need
// to access from eachother
type Scheduler interface {
	Tasks() podtask.Registry
	sync.Locker // synchronize changes to tasks, i.e. lock, get task, change task, store task, unlock

	Offers() offers.Registry
	Reconcile(t *podtask.T)
	KillTask(id string) error
	LaunchTask(t *podtask.T) error
	Run(done <-chan struct{})
}
