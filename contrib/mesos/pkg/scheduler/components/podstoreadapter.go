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

package components

import (
	"k8s.io/kubernetes/contrib/mesos/pkg/queue"
	"k8s.io/kubernetes/contrib/mesos/pkg/scheduler/queuer"
	"k8s.io/kubernetes/pkg/api"
)

// Consumes *api.Pod, produces *Pod; the k8s reflector wants to push *api.Pod
// objects at us, but we want to store more flexible (Pod) type defined in
// this package. The adapter implementation facilitates this. It's a little
// hackish since the object type going in is different than the object type
// coming out -- you've been warned.
type podStoreAdapter struct {
	queue.FIFO
}

func (psa *podStoreAdapter) Add(obj interface{}) error {
	pod := obj.(*api.Pod)
	return psa.FIFO.Add(&queuer.Pod{Pod: pod})
}

func (psa *podStoreAdapter) Update(obj interface{}) error {
	pod := obj.(*api.Pod)
	return psa.FIFO.Update(&queuer.Pod{Pod: pod})
}

func (psa *podStoreAdapter) Delete(obj interface{}) error {
	pod := obj.(*api.Pod)
	return psa.FIFO.Delete(&queuer.Pod{Pod: pod})
}

func (psa *podStoreAdapter) Get(obj interface{}) (interface{}, bool, error) {
	pod := obj.(*api.Pod)
	return psa.FIFO.Get(&queuer.Pod{Pod: pod})
}

// Replace will delete the contents of the store, using instead the
// given map. This store implementation does NOT take ownership of the map.
func (psa *podStoreAdapter) Replace(objs []interface{}, resourceVersion string) error {
	newobjs := make([]interface{}, len(objs))
	for i, v := range objs {
		pod := v.(*api.Pod)
		newobjs[i] = &queuer.Pod{Pod: pod}
	}
	return psa.FIFO.Replace(newobjs, resourceVersion)
}
