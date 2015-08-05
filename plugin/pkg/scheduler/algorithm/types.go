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

package algorithm

import (
	"k8s.io/kubernetes/pkg/api"
)

// FitPredicate is a function that indicates if a pod fits into an existing node.
type FitPredicate func(pod *api.Pod, existingPods []*api.Pod, node string) (bool, error)

// HostPriority represents the priority of scheduling to a particular host, lower priority is better.
type HostPriority struct {
	Host  string
	Score int
}

type HostPriorityList []HostPriority

func (h HostPriorityList) Len() int {
	return len(h)
}

func (h HostPriorityList) Less(i, j int) bool {
	if h[i].Score == h[j].Score {
		return h[i].Host < h[j].Host
	}
	return h[i].Score < h[j].Score
}

func (h HostPriorityList) Swap(i, j int) {
	h[i], h[j] = h[j], h[i]
}

type PriorityFunction func(pod *api.Pod, podLister PodLister, minionLister MinionLister) (HostPriorityList, error)

type PriorityConfig struct {
	Function PriorityFunction
	Weight   int
}
