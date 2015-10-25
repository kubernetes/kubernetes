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

package api

import (
	"sync"

	"k8s.io/kubernetes/contrib/mesos/pkg/offers"
	malgorithm "k8s.io/kubernetes/contrib/mesos/pkg/scheduler/algorithm"
	"k8s.io/kubernetes/contrib/mesos/pkg/scheduler/podtask"
	"k8s.io/kubernetes/pkg/api"
)

// scheduler abstraction to allow for easier unit testing
type SchedulerApi interface {
	sync.Locker // synchronize scheduler plugin operations

	malgorithm.SlaveIndex
	Algorithm() malgorithm.PodScheduler
	Offers() offers.Registry
	Tasks() podtask.Registry

	// driver calls

	KillTask(taskId string) error
	LaunchTask(*podtask.T) error

	// convenience

	CreatePodTask(api.Context, *api.Pod) (*podtask.T, error)
}
