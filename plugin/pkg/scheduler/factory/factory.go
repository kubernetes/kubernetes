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
	"sync"
	"time"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client/cache"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/labels"
	algorithm "github.com/GoogleCloudPlatform/kubernetes/pkg/scheduler"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
	"github.com/GoogleCloudPlatform/kubernetes/plugin/pkg/scheduler"

	"github.com/golang/glog"
)

var (
	PodLister     = &cache.StoreToPodLister{cache.NewStore(cache.MetaNamespaceKeyFunc, cache.DefaultIsOlder)}
	MinionLister  = &cache.StoreToNodeLister{cache.NewStore(cache.MetaNamespaceKeyFunc, cache.DefaultIsOlder)}
	ServiceLister = &cache.StoreToServiceLister{cache.NewStore(cache.MetaNamespaceKeyFunc, cache.DefaultIsOlder)}
)

// ConfigFactory knows how to fill out a scheduler config with its support functions.
type ConfigFactory struct {
	Client *client.Client
	// queue for pods that need scheduling
	PodQueue *cache.FIFO
	// a means to list all scheduled pods
	PodLister *cache.StoreToPodLister
	// a means to list all minions
	MinionLister *cache.StoreToNodeLister
	// a means to list all services
	ServiceLister *cache.StoreToServiceLister
}

// NewConfigFactory initializes the factory.
func NewConfigFactory(client *client.Client) *ConfigFactory {
	return &ConfigFactory{
		Client:        client,
		PodQueue:      cache.NewFIFO(cache.MetaNamespaceKeyFunc, cache.DefaultIsOlder),
		PodLister:     PodLister,
		MinionLister:  MinionLister,
		ServiceLister: ServiceLister,
	}
}

// Create creates a scheduler with the default algorithm provider.
func (f *ConfigFactory) Create() (*scheduler.Config, error) {
	return f.CreateFromProvider(DefaultProvider)
}

// CreateFromProvider creates a scheduler from the name of a registered algorithm provider.
func (f *ConfigFactory) CreateFromProvider(providerName string) (*scheduler.Config, error) {
	glog.V(2).Infof("creating scheduler from algorithm provider '%v'", providerName)
	provider, err := GetAlgorithmProvider(providerName)
	if err != nil {
		return nil, err
	}

	return f.CreateFromKeys(provider.FitPredicateKeys, provider.PriorityFunctionKeys)
}

// CreateFromKeys creates a scheduler from a set of registered fit predicate keys and priority keys.
func (f *ConfigFactory) CreateFromKeys(predicateKeys, priorityKeys util.StringSet) (*scheduler.Config, error) {
	glog.V(2).Infof("creating scheduler with fit predicates '%v' and priority functions '%v", predicateKeys, priorityKeys)
	predicateFuncs, err := getFitPredicateFunctions(predicateKeys)
	if err != nil {
		return nil, err
	}

	priorityConfigs, err := getPriorityFunctionConfigs(priorityKeys)
	if err != nil {
		return nil, err
	}

	// Watch and queue pods that need scheduling.
	cache.NewReflector(f.createUnassignedPodLW(), &api.Pod{}, f.PodQueue).Run()

	// Watch and cache all running pods. Scheduler needs to find all pods
	// so it knows where it's safe to place a pod. Cache this locally.
	cache.NewReflector(f.createAssignedPodLW(), &api.Pod{}, f.PodLister.Store).Run()

	// Watch minions.
	// Minions may be listed frequently, so provide a local up-to-date cache.
	if false {
		// Disable this code until minions support watches. Note when this code is enabled,
		// we need to make sure minion ListWatcher has proper FieldSelector.
		cache.NewReflector(f.createMinionLW(), &api.Node{}, f.MinionLister.Store).Run()
	} else {
		cache.NewPoller(f.pollMinions, 10*time.Second, f.MinionLister.Store).Run()
	}

	// Watch and cache all service objects. Scheduler needs to find all pods
	// created by the same service, so that it can spread them correctly.
	// Cache this locally.
	cache.NewReflector(f.createServiceLW(), &api.Service{}, f.ServiceLister.Store).Run()

	r := rand.New(rand.NewSource(time.Now().UnixNano()))

	algo := algorithm.NewGenericScheduler(predicateFuncs, priorityConfigs, f.PodLister, r)

	podBackoff := podBackoff{
		perPodBackoff: map[string]*backoffEntry{},
		clock:         realClock{},

		defaultDuration: 1 * time.Second,
		maxDuration:     60 * time.Second,
	}

	return &scheduler.Config{
		MinionLister: f.MinionLister,
		Algorithm:    algo,
		Binder:       &binder{f.Client},
		NextPod: func() *api.Pod {
			pod := f.PodQueue.Pop().(*api.Pod)
			glog.V(2).Infof("glog.v2 --> About to try and schedule pod %v", pod.Name)
			return pod
		},
		Error: f.makeDefaultErrorFunc(&podBackoff, f.PodQueue),
	}, nil
}

// createUnassignedPodLW returns a cache.ListWatch that finds all pods that need to be
// scheduled.
func (factory *ConfigFactory) createUnassignedPodLW() *cache.ListWatch {
	return &cache.ListWatch{
		Client:        factory.Client,
		FieldSelector: labels.Set{"DesiredState.Host": ""}.AsSelector(),
		Resource:      "pods",
	}
}

func parseSelectorOrDie(s string) labels.Selector {
	selector, err := labels.ParseSelector(s)
	if err != nil {
		panic(err)
	}
	return selector
}

// createAssignedPodLW returns a cache.ListWatch that finds all pods that are
// already scheduled.
// TODO: return a ListerWatcher interface instead?
func (factory *ConfigFactory) createAssignedPodLW() *cache.ListWatch {
	return &cache.ListWatch{
		Client:        factory.Client,
		FieldSelector: parseSelectorOrDie("DesiredState.Host!="),
		Resource:      "pods",
	}
}

// createMinionLW returns a cache.ListWatch that gets all changes to minions.
func (factory *ConfigFactory) createMinionLW() *cache.ListWatch {
	return &cache.ListWatch{
		Client:        factory.Client,
		FieldSelector: parseSelectorOrDie(""),
		Resource:      "minions",
	}
}

// pollMinions lists all minions and filter out unhealthy ones, then returns
// an enumerator for cache.Poller.
func (factory *ConfigFactory) pollMinions() (cache.Enumerator, error) {
	allNodes := &api.NodeList{}
	err := factory.Client.Get().Resource("minions").Do().Into(allNodes)
	if err != nil {
		return nil, err
	}
	nodes := &api.NodeList{
		TypeMeta: allNodes.TypeMeta,
		ListMeta: allNodes.ListMeta,
	}
	for _, node := range allNodes.Items {
		conditionMap := make(map[api.NodeConditionKind]*api.NodeCondition)
		for i := range node.Status.Conditions {
			cond := node.Status.Conditions[i]
			conditionMap[cond.Kind] = &cond
		}
		if condition, ok := conditionMap[api.NodeReady]; ok {
			if condition.Status == api.ConditionFull {
				nodes.Items = append(nodes.Items, node)
			}
		} else if condition, ok := conditionMap[api.NodeReachable]; ok {
			if condition.Status == api.ConditionFull {
				nodes.Items = append(nodes.Items, node)
			}
		} else {
			// If no condition is set, we get unknown node condition. In such cases,
			// we add nodes unconditionally.
			nodes.Items = append(nodes.Items, node)
		}
	}
	return &nodeEnumerator{nodes}, nil
}

// createServiceLW returns a cache.ListWatch that gets all changes to services.
func (factory *ConfigFactory) createServiceLW() *cache.ListWatch {
	return &cache.ListWatch{
		Client:        factory.Client,
		FieldSelector: parseSelectorOrDie(""),
		Resource:      "services",
	}
}

func (factory *ConfigFactory) makeDefaultErrorFunc(backoff *podBackoff, podQueue *cache.FIFO) func(pod *api.Pod, err error) {
	return func(pod *api.Pod, err error) {
		glog.Errorf("Error scheduling %v %v: %v; retrying", pod.Namespace, pod.Name, err)
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
			err := factory.Client.Get().Namespace(podNamespace).Resource("pods").Name(podID).Do().Into(pod)
			if err != nil {
				glog.Errorf("Error getting pod %v for retry: %v; abandoning", podID, err)
				return
			}
			if pod.Status.Host == "" {
				podQueue.Add(pod)
			}
		}()
	}
}

// nodeEnumerator allows a cache.Poller to enumerate items in an api.NodeList
type nodeEnumerator struct {
	*api.NodeList
}

// Len returns the number of items in the node list.
func (ne *nodeEnumerator) Len() int {
	if ne.NodeList == nil {
		return 0
	}
	return len(ne.Items)
}

// Get returns the item (and ID) with the particular index.
func (ne *nodeEnumerator) Get(index int) interface{} {
	return &ne.Items[index]
}

type binder struct {
	*client.Client
}

// Bind just does a POST binding RPC.
func (b *binder) Bind(binding *api.Binding) error {
	glog.V(2).Infof("Attempting to bind %v to %v", binding.PodID, binding.Host)
	ctx := api.WithNamespace(api.NewContext(), binding.Namespace)
	return b.Post().Namespace(api.NamespaceValue(ctx)).Resource("bindings").Body(binding).Do().Error()
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
	perPodBackoff   map[string]*backoffEntry
	lock            sync.Mutex
	clock           clock
	defaultDuration time.Duration
	maxDuration     time.Duration
}

func (p *podBackoff) getEntry(podID string) *backoffEntry {
	p.lock.Lock()
	defer p.lock.Unlock()
	entry, ok := p.perPodBackoff[podID]
	if !ok {
		entry = &backoffEntry{backoff: p.defaultDuration}
		p.perPodBackoff[podID] = entry
	}
	entry.lastUpdate = p.clock.Now()
	return entry
}

func (p *podBackoff) getBackoff(podID string) time.Duration {
	entry := p.getEntry(podID)
	duration := entry.backoff
	entry.backoff *= 2
	if entry.backoff > p.maxDuration {
		entry.backoff = p.maxDuration
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
		if now.Sub(entry.lastUpdate) > p.maxDuration {
			delete(p.perPodBackoff, podID)
		}
	}
}
