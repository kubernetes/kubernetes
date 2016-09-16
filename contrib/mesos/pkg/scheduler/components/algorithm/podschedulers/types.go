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

package podschedulers

import (
	"github.com/mesos/mesos-go/mesosproto"
	"k8s.io/kubernetes/contrib/mesos/pkg/offers"
	"k8s.io/kubernetes/contrib/mesos/pkg/scheduler/podtask"
	"k8s.io/kubernetes/pkg/api"
)

// SchedulePod is the interface which schedules pods.
// There can be different implementation for different scheduling policies.
//
// SchedulePod accepts a set of offers and a single pod task, which aligns well
// with the k8s scheduling algorithm. It returns an offer that is acceptable
// for the pod, else nil. The caller is responsible for filling in task
// state w/ relevant offer details.
//
// See the FCFSPodScheduler for example.
//
// Fit checks whether a given podtask can be scheduled for the given offer on the given node.
type PodScheduler interface {
	SchedulePod(r offers.Registry, task *podtask.T) (offers.Perishable, *podtask.Spec, error)

	Fit(*podtask.T, *mesosproto.Offer, *api.Node) bool
}
