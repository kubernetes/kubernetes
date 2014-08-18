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

package main

import (
	"flag"
	"math/rand"
	"time"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client/cache"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/labels"
	algorithm "github.com/GoogleCloudPlatform/kubernetes/pkg/scheduler"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
	verflag "github.com/GoogleCloudPlatform/kubernetes/pkg/version/flag"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/watch"
	"github.com/GoogleCloudPlatform/kubernetes/plugin/pkg/scheduler"

	"github.com/golang/glog"
)

var (
	master = flag.String("master", "", "The address of the Kubernetes API server")
)

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
	return b.Post().Path("bindings").Body(binding).Do().Error()
}

func main() {
	flag.Parse()
	util.InitLogs()
	defer util.FlushLogs()

	verflag.PrintAndExitIfRequested()

	// This function is long because we inject all the dependencies into scheduler here.

	// TODO: security story for plugins!
	kubeClient := client.New("http://"+*master, nil)

	// Watch and queue pods that need scheduling.
	podQueue := cache.NewFIFO()
	cache.NewReflector(func(resourceVersion uint64) (watch.Interface, error) {
		// This query will only find pods with no assigned host.
		return kubeClient.
			Get().
			Path("watch").
			Path("pods").
			SelectorParam("fields", labels.Set{"DesiredState.Host": ""}.AsSelector()).
			UintParam("resourceVersion", resourceVersion).
			Watch()
	}, &api.Pod{}, podQueue).Run()

	// Watch and cache all running pods. Scheduler needs to find all pods
	// so it knows where it's safe to place a pod. Cache this locally.
	podCache := cache.NewStore()
	cache.NewReflector(func(resourceVersion uint64) (watch.Interface, error) {
		// This query will only find pods that do have an assigned host.
		return kubeClient.
			Get().
			Path("watch").
			Path("pods").
			ParseSelectorParam("fields", "DesiredState.Host!=").
			UintParam("resourceVersion", resourceVersion).
			Watch()
	}, &api.Pod{}, podCache).Run()

	// Watch minions.
	// Minions may be listed frequently, so provide a local up-to-date cache.
	minionCache := cache.NewStore()
	if false {
		// Disable this code until minions support watches.
		cache.NewReflector(func(resourceVersion uint64) (watch.Interface, error) {
			// This query will only find pods that do have an assigned host.
			return kubeClient.
				Get().
				Path("watch").
				Path("minions").
				UintParam("resourceVersion", resourceVersion).
				Watch()
		}, &api.Minion{}, minionCache).Run()
	} else {
		cache.NewPoller(func() (cache.Enumerator, error) {
			// This query will only find pods that do have an assigned host.
			list := &api.MinionList{}
			err := kubeClient.Get().Path("minions").Do().Into(list)
			if err != nil {
				return nil, err
			}
			return &minionEnumerator{list}, nil
		}, 10*time.Second, minionCache).Run()
	}

	r := rand.New(rand.NewSource(time.Now().UnixNano()))
	algo := algorithm.NewRandomFitScheduler(
		&storeToPodLister{podCache}, r)

	s := scheduler.New(&scheduler.Config{
		MinionLister: &storeToMinionLister{minionCache},
		Algorithm:    algo,
		Binder:       &binder{kubeClient},
		NextPod: func() *api.Pod {
			return podQueue.Pop().(*api.Pod)
		},
		Error: func(pod *api.Pod, err error) {
			glog.Errorf("Error scheduling %v: %v; retrying", pod.ID, err)
			podQueue.Add(pod.ID, pod)
		},
	})

	s.Run()

	select {}
}
