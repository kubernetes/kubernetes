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

package podutil

import (
	log "github.com/golang/glog"
	"k8s.io/kubernetes/pkg/api"
)

type defaultFunc func(pod *api.Pod) error

// return true if the pod passes the filter
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

// Environment returns a filter that writes environment variables into pod containers
func Environment(env []api.EnvVar) FilterFunc {
	// index the envvar names
	var (
		envcount = len(env)
		m        = make(map[string]int, envcount)
	)
	for j := range env {
		m[env[j].Name] = j
	}
	return func(pod *api.Pod) (bool, error) {
		for i := range pod.Spec.Containers {
			ct := &pod.Spec.Containers[i]
			dup := make(map[string]struct{}, envcount)
			// overwrite dups (and remember them for later)
			for j := range ct.Env {
				name := ct.Env[j].Name
				if k, ok := m[name]; ok {
					ct.Env[j] = env[k]
					dup[name] = struct{}{}
				}
			}
			// append non-dups into ct.Env
			for name, k := range m {
				if _, ok := dup[name]; !ok {
					ct.Env = append(ct.Env, env[k])
				}
			}
		}
		return true, nil
	}
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

func (filters Filters) Do(in <-chan *api.Pod) (out <-chan *api.Pod) {
	out = in
	for _, f := range filters {
		out = f.Do(out)
	}
	return
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
