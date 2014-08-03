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
)

var (
	master = flag.String("master", "", "The address of the Kubernetes API server")
)

// storeToMinionLister turns a store into a minion lister. The store must contain (only) minions.
type storeToMinionLister struct {
	s cache.Store
}

func (s storeToMinionLister) List() (machines []string, err error) {
	for _, m := range s.s.List() {
		machines = append(machines, m.(*api.Minion).ID)
	}
	return machines, nil
}

// storeToPodLister turns a store into a pod lister. The store must contain (only) pods.
type storeToPodLister struct {
	s cache.Store
}

func (s storeToPodLister) ListPods(selector labels.Selector) (pods []api.Pod, err error) {
	for _, m := range s.s.List() {
		pod := m.(*api.Pod)
		if selector.Matches(labels.Set(pod.Labels)) {
			pods = append(pods, *pod)
		}
	}
	return pods, nil
}

type binder struct {
	kubeClient *client.Client
}

// Bind just does a POST binding RPC.
func (b binder) Bind(binding *api.Binding) error {
	return b.kubeClient.Post().Path("bindings").Body(binding).Do().Error()
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
			Path("pods").
			Path("watch").
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
			Path("pods").
			Path("watch").
			ParseSelectorParam("fields", "DesiredState.Host!=").
			UintParam("resourceVersion", resourceVersion).
			Watch()
	}, &api.Pod{}, podCache).Run()

	// Watch minions.
	// Minions may be listed frequently, so provide a local up-to-date cache.
	minionCache := cache.NewStore()
	cache.NewReflector(func(resourceVersion uint64) (watch.Interface, error) {
		// This query will only find pods that do have an assigned host.
		return kubeClient.
			Get().
			Path("minions").
			Path("watch").
			UintParam("resourceVersion", resourceVersion).
			Watch()
	}, &api.Minion{}, minionCache).Run()

	r := rand.New(rand.NewSource(time.Now().UnixNano()))
	algo := algorithm.NewRandomFitScheduler(
		storeToPodLister{podCache}, r)

	s := scheduler.New(&scheduler.Config{
		MinionLister: storeToMinionLister{minionCache},
		Algorithm:    algo,
		NextPod:      func() *api.Pod { return podQueue.Pop().(*api.Pod) },
		Binder:       binder{kubeClient},
	})

	s.Run()

	select {}
}
