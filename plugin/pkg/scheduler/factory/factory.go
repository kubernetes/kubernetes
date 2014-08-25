/*
Copyright 2014 Google Inc. All rights reserved.

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

// Package factory can set up a scheduler. This code is here instead of
// plugin/cmd/scheduler for both testability and reuse.
package factory

import (
	"math/rand"
	"time"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client/cache"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/labels"
	algorithm "github.com/GoogleCloudPlatform/kubernetes/pkg/scheduler"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/watch"
	"github.com/GoogleCloudPlatform/kubernetes/plugin/pkg/scheduler"

	"github.com/golang/glog"
)

// ConfigFactory knows how to fill out a scheduler config with its support functions.
type ConfigFactory struct {
	Client *client.Client
}

// Create creates a scheduler and all support functions.
func (factory *ConfigFactory) Create() *scheduler.Config {
	// Watch and queue pods that need scheduling.
	podQueue := cache.NewFIFO()
	cache.NewReflector(factory.createUnassignedPodWatch, &api.Pod{}, podQueue).Run()

	// Watch and cache all running pods. Scheduler needs to find all pods
	// so it knows where it's safe to place a pod. Cache this locally.
	podCache := cache.NewStore()
	cache.NewReflector(factory.createAssignedPodWatch, &api.Pod{}, podCache).Run()

	// Watch minions.
	// Minions may be listed frequently, so provide a local up-to-date cache.
	minionCache := cache.NewStore()
	if false {
		// Disable this code until minions support watches.
		cache.NewReflector(factory.createMinionWatch, &api.Minion{}, minionCache).Run()
	} else {
		cache.NewPoller(factory.pollMinions, 10*time.Second, minionCache).Run()
	}

	r := rand.New(rand.NewSource(time.Now().UnixNano()))
	algo := algorithm.NewRandomFitScheduler(
		&storeToPodLister{podCache}, r)

	return &scheduler.Config{
		MinionLister: &storeToMinionLister{minionCache},
		Algorithm:    algo,
		Binder:       &binder{factory.Client},
		NextPod: func() *api.Pod {
			pod := podQueue.Pop().(*api.Pod)
			// TODO: Remove or reduce verbosity by sep 6th, 2014. Leave until then to
			// make it easy to find scheduling problems.
			glog.Infof("About to try and schedule pod %v\n"+
				"\tknown minions: %v\n"+
				"\tknown scheduled pods: %v\n",
				pod.ID, minionCache.Contains(), podCache.Contains())
			return pod
		},
		Error: factory.makeDefaultErrorFunc(podQueue),
	}
}

// createUnassignedPodWatch starts a watch that finds all pods that need to be
// scheduled.
func (factory *ConfigFactory) createUnassignedPodWatch(resourceVersion uint64) (watch.Interface, error) {
	return factory.Client.
		Get().
		Path("watch").
		Path("pods").
		SelectorParam("fields", labels.Set{"DesiredState.Host": ""}.AsSelector()).
		UintParam("resourceVersion", resourceVersion).
		Watch()
}

// createUnassignedPodWatch starts a watch that finds all pods that are
// already scheduled.
func (factory *ConfigFactory) createAssignedPodWatch(resourceVersion uint64) (watch.Interface, error) {
	return factory.Client.
		Get().
		Path("watch").
		Path("pods").
		ParseSelectorParam("fields", "DesiredState.Host!=").
		UintParam("resourceVersion", resourceVersion).
		Watch()
}

// createMinionWatch starts a watch that gets all changes to minions.
func (factory *ConfigFactory) createMinionWatch(resourceVersion uint64) (watch.Interface, error) {
	return factory.Client.
		Get().
		Path("watch").
		Path("minions").
		UintParam("resourceVersion", resourceVersion).
		Watch()
}

// pollMinions lists all minions and returns an enumerator for cache.Poller.
func (factory *ConfigFactory) pollMinions() (cache.Enumerator, error) {
	list := &api.MinionList{}
	err := factory.Client.Get().Path("minions").Do().Into(list)
	if err != nil {
		return nil, err
	}
	return &minionEnumerator{list}, nil
}

func (factory *ConfigFactory) makeDefaultErrorFunc(podQueue *cache.FIFO) func(pod *api.Pod, err error) {
	return func(pod *api.Pod, err error) {
		glog.Errorf("Error scheduling %v: %v; retrying", pod.ID, err)

		// Retry asynchronously.
		// Note that this is extremely rudimentary and we need a more real error handling path.
		go func() {
			defer util.HandleCrash()
			podID := pod.ID
			// Get the pod again; it may have changed/been scheduled already.
			pod = &api.Pod{}
			err := factory.Client.Get().Path("pods").Path(podID).Do().Into(pod)
			if err != nil {
				glog.Errorf("Error getting pod %v for retry: %v; abandoning", podID, err)
				return
			}
			if pod.DesiredState.Host == "" {
				podQueue.Add(pod.ID, pod)
			}
		}()
	}
}

// storeToMinionLister turns a store into a minion lister. The store must contain (only) minions.
type storeToMinionLister struct {
	cache.Store
}

func (s *storeToMinionLister) List() (machines []string, err error) {
	for _, m := range s.Store.List() {
		machines = append(machines, m.(*api.Minion).ID)
	}
	return machines, nil
}

// storeToPodLister turns a store into a pod lister. The store must contain (only) pods.
type storeToPodLister struct {
	cache.Store
}

func (s *storeToPodLister) ListPods(selector labels.Selector) (pods []api.Pod, err error) {
	for _, m := range s.List() {
		pod := m.(*api.Pod)
		if selector.Matches(labels.Set(pod.Labels)) {
			pods = append(pods, *pod)
		}
	}
	return pods, nil
}

// minionEnumerator allows a cache.Poller to enumerate items in an api.PodList
type minionEnumerator struct {
	*api.MinionList
}

// Returns the number of items in the pod list.
func (me *minionEnumerator) Len() int {
	if me.MinionList == nil {
		return 0
	}
	return len(me.Items)
}

// Returns the item (and ID) with the particular index.
func (me *minionEnumerator) Get(index int) (string, interface{}) {
	return me.Items[index].ID, &me.Items[index]
}

type binder struct {
	*client.Client
}

// Bind just does a POST binding RPC.
func (b *binder) Bind(binding *api.Binding) error {
	// TODO: Remove or reduce verbosity by sep 6th, 2014. Leave until then to
	// make it easy to find scheduling problems.
	glog.Infof("Attempting to bind %v to %v", binding.PodID, binding.Host)
	return b.Post().Path("bindings").Body(binding).Do().Error()
}
