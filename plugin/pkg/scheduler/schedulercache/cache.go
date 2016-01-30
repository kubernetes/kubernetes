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

package schedulercache

import (
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/labels"
)

type cache struct {
	podLister PodLister
}

// New returns a Cache implementation.
func New(podLister PodLister) Cache {
	return &cache{
		podLister: podLister,
	}
}

func mustListPods(podLister PodLister) []*api.Pod {
	pods, err := podLister.List(labels.Everything())
	if err != nil {
		panic("local cache shouldn't have any List error")
	}
	return pods
}

func (c *cache) GetNodeNameToInfoMap() map[string]*NodeInfo {
	return CreateNodeNameToInfoMap(mustListPods(c.podLister))
}

func (c *cache) List(selector labels.Selector) []*api.Pod {
	return mustListPods(c.podLister)
}
