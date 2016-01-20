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

package podutil

import (
	log "github.com/golang/glog"
	"k8s.io/kubernetes/pkg/api"
)

type defaultFunc func(pod *api.Pod) error

type FilterFunc func(pod *api.Pod) (bool, error)

type Filters []FilterFunc

// Annotate safely copies annotation metadata from kv to meta.Annotations.
func Annotate(meta *api.ObjectMeta, kv map[string]string) {
	//TODO(jdef) this func probably belong in an "apiutil" package, but we don't
	//have much to put there right now so it can just live here.
	if meta.Annotations == nil {
		meta.Annotations = make(map[string]string)
	}
	for k, v := range kv {
		meta.Annotations[k] = v
	}
}

// Annotator returns a filter that copies annotations from map m into a pod
func Annotator(m map[string]string) FilterFunc {
	return FilterFunc(func(pod *api.Pod) (bool, error) {
		Annotate(&pod.ObjectMeta, m)
		return true, nil
	})
}

// Stream returns a chan of pods that yields each pod from the given list.
// No pods are yielded if err is non-nil.
func Stream(list *api.PodList, err error) <-chan *api.Pod {
	out := make(chan *api.Pod)
	go func() {
		defer close(out)
		if err != nil {
			log.Errorf("failed to obtain pod list: %v", err)
			return
		}
		for _, pod := range list.Items {
			pod := pod
			out <- &pod
		}
	}()
	return out
}

func (filter FilterFunc) Do(in <-chan *api.Pod) <-chan *api.Pod {
	out := make(chan *api.Pod)
	go func() {
		defer close(out)
		for pod := range in {
			if ok, err := filter(pod); err != nil {
				log.Errorf("pod failed selection: %v", err)
			} else if ok {
				out <- pod
			}
		}
	}()
	return out
}

// List reads every pod from the pods chan and returns them all in an api.PodList
func List(pods <-chan *api.Pod) *api.PodList {
	list := &api.PodList{}
	for p := range pods {
		list.Items = append(list.Items, *p)
	}
	return list
}
