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
	"fmt"
	"math/rand"
	"sync"
	"time"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client/cache"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/labels"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/runtime"
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
	cache.NewReflector(factory.createUnassignedPodLW(), &api.Pod{}, podQueue).Run()

	// Watch and cache all running pods. Scheduler needs to find all pods
	// so it knows where it's safe to place a pod. Cache this locally.
	podCache := cache.NewStore()
	cache.NewReflector(factory.createAssignedPodLW(), &api.Pod{}, podCache).Run()

	// Watch minions.
	// Minions may be listed frequently, so provide a local up-to-date cache.
	minionCache := cache.NewStore()
	if false {
		// Disable this code until minions support watches.
		cache.NewReflector(factory.createMinionLW(), &api.Minion{}, minionCache).Run()
	} else {
		cache.NewPoller(factory.pollMinions, 10*time.Second, minionCache).Run()
	}

	r := rand.New(rand.NewSource(time.Now().UnixNano()))
	minionLister := &storeToMinionLister{minionCache}

	algo := algorithm.NewGenericScheduler(
		[]algorithm.FitPredicate{
			// Fit is defined based on the absence of port conflicts.
			algorithm.PodFitsPorts,
			// Fit is determined by resource availability
			algorithm.NewResourceFitPredicate(minionLister),
			// Fit is determined by non-conflicting disk volumes
			algorithm.NoDiskConflict,
			// Fit is determined by node selector query
			algorithm.NewSelectorMatchPredicate(minionLister),
		},
		// Prioritize nodes by least requested utilization.
		algorithm.LeastRequestedPriority,
		&storeToPodLister{podCache}, r)

	podBackoff := podBackoff{
		perPodBackoff: map[string]*backoffEntry{},
		clock:         realClock{},
	}

	return &scheduler.Config{
		MinionLister: minionLister,
		Algorithm:    algo,
		Binder:       &binder{factory.Client},
		NextPod: func() *api.Pod {
			pod := podQueue.Pop().(*api.Pod)
			glog.V(2).Infof("About to try and schedule pod %v\n"+
				"\tknown minions: %v\n"+
				"\tknown scheduled pods: %v\n",
				pod.Name, minionCache.ContainedIDs(), podCache.ContainedIDs())
			return pod
		},
		Error: factory.makeDefaultErrorFunc(&podBackoff, podQueue),
	}
}

type listWatch struct {
	client        *client.Client
	fieldSelector labels.Selector
	resource      string
}

func (lw *listWatch) List() (runtime.Object, error) {
	return lw.client.
		Get().
		Path(lw.resource).
		SelectorParam("fields", lw.fieldSelector).
		Do().
		Get()
}

func (lw *listWatch) Watch(resourceVersion string) (watch.Interface, error) {
	return lw.client.
		Get().
		Path("watch").
		Path(lw.resource).
		SelectorParam("fields", lw.fieldSelector).
		Param("resourceVersion", resourceVersion).
		Watch()
}

// createUnassignedPodLW returns a listWatch that finds all pods that need to be
// scheduled.
func (factory *ConfigFactory) createUnassignedPodLW() *listWatch {
	return &listWatch{
		client:        factory.Client,
		fieldSelector: labels.Set{"DesiredState.Host": ""}.AsSelector(),
		resource:      "pods",
	}
}

func parseSelectorOrDie(s string) labels.Selector {
	selector, err := labels.ParseSelector(s)
	if err != nil {
		panic(err)
	}
	return selector
}

// createUnassignedPodLW returns a listWatch that finds all pods that are
// already scheduled.
func (factory *ConfigFactory) createAssignedPodLW() *listWatch {
	return &listWatch{
		client:        factory.Client,
		fieldSelector: parseSelectorOrDie("DesiredState.Host!="),
		resource:      "pods",
	}
}

// createMinionLW returns a listWatch that gets all changes to minions.
func (factory *ConfigFactory) createMinionLW() *listWatch {
	return &listWatch{
		client:        factory.Client,
		fieldSelector: parseSelectorOrDie(""),
		resource:      "minions",
	}
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

func (factory *ConfigFactory) makeDefaultErrorFunc(backoff *podBackoff, podQueue *cache.FIFO) func(pod *api.Pod, err error) {
	return func(pod *api.Pod, err error) {
		glog.Errorf("Error scheduling %v: %v; retrying", pod.Name, err)
		backoff.gc()
		// Retry asynchronously.
		// Note that this is extremely rudimentary and we need a more real error handling path.
		go func() {
			defer util.HandleCrash()
			podID := pod.Name
			podNamespace := pod.Namespace
			backoff.wait(podID)
			// Get the pod again; it may have changed/been scheduled already.
			pod = &api.Pod{}
			err := factory.Client.Get().Namespace(podNamespace).Path("pods").Path(podID).Do().Into(pod)
			if err != nil {
				glog.Errorf("Error getting pod %v for retry: %v; abandoning", podID, err)
				return
			}
			if pod.DesiredState.Host == "" {
				podQueue.Add(pod.Name, pod)
			}
		}()
	}
}

// storeToMinionLister turns a store into a minion lister. The store must contain (only) minions.
type storeToMinionLister struct {
	cache.Store
}

func (s *storeToMinionLister) List() (machines api.MinionList, err error) {
	for _, m := range s.Store.List() {
		machines.Items = append(machines.Items, *(m.(*api.Minion)))
	}
	return machines, nil
}

// GetNodeInfo returns cached data for the minion 'id'.
func (s *storeToMinionLister) GetNodeInfo(id string) (*api.Minion, error) {
	if minion, ok := s.Get(id); ok {
		return minion.(*api.Minion), nil
	}
	return nil, fmt.Errorf("minion '%v' is not in cache", id)
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

// Len returns the number of items in the pod list.
func (me *minionEnumerator) Len() int {
	if me.MinionList == nil {
		return 0
	}
	return len(me.Items)
}

// Get returns the item (and ID) with the particular index.
func (me *minionEnumerator) Get(index int) (string, interface{}) {
	return me.Items[index].Name, &me.Items[index]
}

type binder struct {
	*client.Client
}

// Bind just does a POST binding RPC.
func (b *binder) Bind(binding *api.Binding) error {
	glog.V(2).Infof("Attempting to bind %v to %v", binding.PodID, binding.Host)
	ctx := api.WithNamespace(api.NewContext(), binding.Namespace)
	return b.Post().Namespace(api.Namespace(ctx)).Path("bindings").Body(binding).Do().Error()
}

type clock interface {
	Now() time.Time
}

type realClock struct{}

func (realClock) Now() time.Time {
	return time.Now()
}

type backoffEntry struct {
	backoff    time.Duration
	lastUpdate time.Time
}

type podBackoff struct {
	perPodBackoff map[string]*backoffEntry
	lock          sync.Mutex
	clock         clock
}

func (p *podBackoff) getEntry(podID string) *backoffEntry {
	p.lock.Lock()
	defer p.lock.Unlock()
	entry, ok := p.perPodBackoff[podID]
	if !ok {
		entry = &backoffEntry{backoff: 1 * time.Second}
		p.perPodBackoff[podID] = entry
	}
	entry.lastUpdate = p.clock.Now()
	return entry
}

func (p *podBackoff) getBackoff(podID string) time.Duration {
	entry := p.getEntry(podID)
	duration := entry.backoff
	entry.backoff *= 2
	if entry.backoff > 60*time.Second {
		entry.backoff = 60 * time.Second
	}
	glog.V(4).Infof("Backing off %s for pod %s", duration.String(), podID)
	return duration
}

func (p *podBackoff) wait(podID string) {
	time.Sleep(p.getBackoff(podID))
}

func (p *podBackoff) gc() {
	p.lock.Lock()
	defer p.lock.Unlock()
	now := p.clock.Now()
	for podID, entry := range p.perPodBackoff {
		if now.Sub(entry.lastUpdate) > 60*time.Second {
			delete(p.perPodBackoff, podID)
		}
	}
}
