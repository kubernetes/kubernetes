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

// Package factory can set up a scheduler. This code is here instead of
// plugin/cmd/scheduler for both testability and reuse.
package factory

import (
	"math/rand"
	"sync"
	"time"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/errors"
	"k8s.io/kubernetes/pkg/client"
	"k8s.io/kubernetes/pkg/client/cache"
	"k8s.io/kubernetes/pkg/controller/framework"
	"k8s.io/kubernetes/pkg/fields"
	"k8s.io/kubernetes/pkg/util"
	"k8s.io/kubernetes/plugin/pkg/scheduler"
	"k8s.io/kubernetes/plugin/pkg/scheduler/algorithm"
	schedulerapi "k8s.io/kubernetes/plugin/pkg/scheduler/api"
	"k8s.io/kubernetes/plugin/pkg/scheduler/api/validation"

	"github.com/golang/glog"
)

// ConfigFactory knows how to fill out a scheduler config with its support functions.
type ConfigFactory struct {
	Client *client.Client
	// queue for pods that need scheduling
	PodQueue *cache.FIFO
	// a means to list all known scheduled pods.
	ScheduledPodLister *cache.StoreToPodLister
	// a means to list all known scheduled pods and pods assumed to have been scheduled.
	PodLister algorithm.PodLister
	// a means to list all minions
	NodeLister *cache.StoreToNodeLister
	// a means to list all services
	ServiceLister *cache.StoreToServiceLister
	// a means to list all controllers
	ControllerLister *cache.StoreToReplicationControllerLister

	// Close this to stop all reflectors
	StopEverything chan struct{}
	// Rate limiter for binding pods
	BindPodsRateLimiter util.RateLimiter

	scheduledPodPopulator *framework.Controller
	modeler               scheduler.SystemModeler
}

// Initializes the factory.
func NewConfigFactory(client *client.Client, rateLimiter util.RateLimiter) *ConfigFactory {
	c := &ConfigFactory{
		Client:             client,
		PodQueue:           cache.NewFIFO(cache.MetaNamespaceKeyFunc),
		ScheduledPodLister: &cache.StoreToPodLister{},
		// Only nodes in the "Ready" condition with status == "True" are schedulable
		NodeLister:       &cache.StoreToNodeLister{cache.NewStore(cache.MetaNamespaceKeyFunc)},
		ServiceLister:    &cache.StoreToServiceLister{cache.NewStore(cache.MetaNamespaceKeyFunc)},
		ControllerLister: &cache.StoreToReplicationControllerLister{cache.NewStore(cache.MetaNamespaceKeyFunc)},
		StopEverything:   make(chan struct{}),
	}
	modeler := scheduler.NewSimpleModeler(&cache.StoreToPodLister{c.PodQueue}, c.ScheduledPodLister)
	c.modeler = modeler
	c.PodLister = modeler.PodLister()
	c.BindPodsRateLimiter = rateLimiter

	// On add/delete to the scheduled pods, remove from the assumed pods.
	// We construct this here instead of in CreateFromKeys because
	// ScheduledPodLister is something we provide to plug in functions that
	// they may need to call.
	c.ScheduledPodLister.Store, c.scheduledPodPopulator = framework.NewInformer(
		c.createAssignedPodLW(),
		&api.Pod{},
		0,
		framework.ResourceEventHandlerFuncs{
			AddFunc: func(obj interface{}) {
				if pod, ok := obj.(*api.Pod); ok {
					c.modeler.LockedAction(func() {
						c.modeler.ForgetPod(pod)
					})
				}
			},
			DeleteFunc: func(obj interface{}) {
				c.modeler.LockedAction(func() {
					switch t := obj.(type) {
					case *api.Pod:
						c.modeler.ForgetPod(t)
					case cache.DeletedFinalStateUnknown:
						c.modeler.ForgetPodByKey(t.Key)
					}
				})
			},
		},
	)

	return c
}

// Create creates a scheduler with the default algorithm provider.
func (f *ConfigFactory) Create() (*scheduler.Config, error) {
	return f.CreateFromProvider(DefaultProvider)
}

// Creates a scheduler from the name of a registered algorithm provider.
func (f *ConfigFactory) CreateFromProvider(providerName string) (*scheduler.Config, error) {
	glog.V(2).Infof("creating scheduler from algorithm provider '%v'", providerName)
	provider, err := GetAlgorithmProvider(providerName)
	if err != nil {
		return nil, err
	}

	return f.CreateFromKeys(provider.FitPredicateKeys, provider.PriorityFunctionKeys)
}

// Creates a scheduler from the configuration file
func (f *ConfigFactory) CreateFromConfig(policy schedulerapi.Policy) (*scheduler.Config, error) {
	glog.V(2).Infof("creating scheduler from configuration: %v", policy)

	// validate the policy configuration
	if err := validation.ValidatePolicy(policy); err != nil {
		return nil, err
	}

	predicateKeys := util.NewStringSet()
	for _, predicate := range policy.Predicates {
		glog.V(2).Infof("Registering predicate: %s", predicate.Name)
		predicateKeys.Insert(RegisterCustomFitPredicate(predicate))
	}

	priorityKeys := util.NewStringSet()
	for _, priority := range policy.Priorities {
		glog.V(2).Infof("Registering priority: %s", priority.Name)
		priorityKeys.Insert(RegisterCustomPriorityFunction(priority))
	}

	return f.CreateFromKeys(predicateKeys, priorityKeys)
}

// Creates a scheduler from a set of registered fit predicate keys and priority keys.
func (f *ConfigFactory) CreateFromKeys(predicateKeys, priorityKeys util.StringSet) (*scheduler.Config, error) {
	glog.V(2).Infof("creating scheduler with fit predicates '%v' and priority functions '%v", predicateKeys, priorityKeys)
	pluginArgs := PluginFactoryArgs{
		PodLister:        f.PodLister,
		ServiceLister:    f.ServiceLister,
		ControllerLister: f.ControllerLister,
		// All fit predicates only need to consider schedulable nodes.
		NodeLister: f.NodeLister.NodeCondition(api.NodeReady, api.ConditionTrue),
		NodeInfo:   f.NodeLister,
	}
	predicateFuncs, err := getFitPredicateFunctions(predicateKeys, pluginArgs)
	if err != nil {
		return nil, err
	}

	priorityConfigs, err := getPriorityFunctionConfigs(priorityKeys, pluginArgs)
	if err != nil {
		return nil, err
	}

	// Watch and queue pods that need scheduling.
	cache.NewReflector(f.createUnassignedPodLW(), &api.Pod{}, f.PodQueue, 0).RunUntil(f.StopEverything)

	// Begin populating scheduled pods.
	go f.scheduledPodPopulator.Run(f.StopEverything)

	// Watch minions.
	// Minions may be listed frequently, so provide a local up-to-date cache.
	cache.NewReflector(f.createMinionLW(), &api.Node{}, f.NodeLister.Store, 0).RunUntil(f.StopEverything)

	// Watch and cache all service objects. Scheduler needs to find all pods
	// created by the same services or ReplicationControllers, so that it can spread them correctly.
	// Cache this locally.
	cache.NewReflector(f.createServiceLW(), &api.Service{}, f.ServiceLister.Store, 0).RunUntil(f.StopEverything)

	// Watch and cache all ReplicationController objects. Scheduler needs to find all pods
	// created by the same services or ReplicationControllers, so that it can spread them correctly.
	// Cache this locally.
	cache.NewReflector(f.createControllerLW(), &api.ReplicationController{}, f.ControllerLister.Store, 0).RunUntil(f.StopEverything)

	r := rand.New(rand.NewSource(time.Now().UnixNano()))

	algo := scheduler.NewGenericScheduler(predicateFuncs, priorityConfigs, f.PodLister, r)

	podBackoff := podBackoff{
		perPodBackoff: map[string]*backoffEntry{},
		clock:         realClock{},

		defaultDuration: 1 * time.Second,
		maxDuration:     60 * time.Second,
	}

	return &scheduler.Config{
		Modeler: f.modeler,
		// The scheduler only needs to consider schedulable nodes.
		MinionLister: f.NodeLister.NodeCondition(api.NodeReady, api.ConditionTrue),
		Algorithm:    algo,
		Binder:       &binder{f.Client},
		NextPod: func() *api.Pod {
			pod := f.PodQueue.Pop().(*api.Pod)
			glog.V(2).Infof("About to try and schedule pod %v", pod.Name)
			return pod
		},
		Error:               f.makeDefaultErrorFunc(&podBackoff, f.PodQueue),
		BindPodsRateLimiter: f.BindPodsRateLimiter,
		StopEverything:      f.StopEverything,
	}, nil
}

// Returns a cache.ListWatch that finds all pods that need to be
// scheduled.
func (factory *ConfigFactory) createUnassignedPodLW() *cache.ListWatch {
	return cache.NewListWatchFromClient(factory.Client, "pods", api.NamespaceAll, fields.Set{client.PodHost: ""}.AsSelector())
}

func parseSelectorOrDie(s string) fields.Selector {
	selector, err := fields.ParseSelector(s)
	if err != nil {
		panic(err)
	}
	return selector
}

// Returns a cache.ListWatch that finds all pods that are
// already scheduled.
// TODO: return a ListerWatcher interface instead?
func (factory *ConfigFactory) createAssignedPodLW() *cache.ListWatch {
	return cache.NewListWatchFromClient(factory.Client, "pods", api.NamespaceAll,
		parseSelectorOrDie(client.PodHost+"!="))
}

// createMinionLW returns a cache.ListWatch that gets all changes to minions.
func (factory *ConfigFactory) createMinionLW() *cache.ListWatch {
	// TODO: Filter out nodes that doesn't have NodeReady condition.
	fields := fields.Set{client.NodeUnschedulable: "false"}.AsSelector()
	return cache.NewListWatchFromClient(factory.Client, "nodes", api.NamespaceAll, fields)
}

// Returns a cache.ListWatch that gets all changes to services.
func (factory *ConfigFactory) createServiceLW() *cache.ListWatch {
	return cache.NewListWatchFromClient(factory.Client, "services", api.NamespaceAll, parseSelectorOrDie(""))
}

// Returns a cache.ListWatch that gets all changes to controllers.
func (factory *ConfigFactory) createControllerLW() *cache.ListWatch {
	return cache.NewListWatchFromClient(factory.Client, "replicationControllers", api.NamespaceAll, parseSelectorOrDie(""))
}

func (factory *ConfigFactory) makeDefaultErrorFunc(backoff *podBackoff, podQueue *cache.FIFO) func(pod *api.Pod, err error) {
	return func(pod *api.Pod, err error) {
		if err == scheduler.ErrNoNodesAvailable {
			glog.V(4).Infof("Unable to schedule %v %v: no nodes are registered to the cluster; waiting", pod.Namespace, pod.Name)
		} else {
			glog.Errorf("Error scheduling %v %v: %v; retrying", pod.Namespace, pod.Name, err)
		}
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
				if !errors.IsNotFound(err) {
					glog.Errorf("Error getting pod %v for retry: %v; abandoning", podID, err)
				}
				return
			}
			if pod.Spec.NodeName == "" {
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
	glog.V(2).Infof("Attempting to bind %v to %v", binding.Name, binding.Target.Name)
	ctx := api.WithNamespace(api.NewContext(), binding.Namespace)
	return b.Post().Namespace(api.NamespaceValue(ctx)).Resource("bindings").Body(binding).Do().Error()
	// TODO: use Pods interface for binding once clusters are upgraded
	// return b.Pods(binding.Namespace).Bind(binding)
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
