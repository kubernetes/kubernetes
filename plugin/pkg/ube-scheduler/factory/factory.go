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
	"fmt"
	"math/rand"
	"sync"
	"sync/atomic"
	"time"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/errors"
	"k8s.io/kubernetes/pkg/apis/controlplane"
	"k8s.io/kubernetes/pkg/client/cache"
	client "k8s.io/kubernetes/pkg/client/unversioned"
	"k8s.io/kubernetes/pkg/fields"
	"k8s.io/kubernetes/pkg/types"
	"k8s.io/kubernetes/pkg/util/runtime"
	"k8s.io/kubernetes/pkg/util/sets"
	"k8s.io/kubernetes/plugin/pkg/ube-scheduler"
	"k8s.io/kubernetes/plugin/pkg/ube-scheduler/algorithm"
	"k8s.io/kubernetes/plugin/pkg/ube-scheduler/algorithm/predicates"
	schedulerapi "k8s.io/kubernetes/plugin/pkg/ube-scheduler/api"
	"k8s.io/kubernetes/plugin/pkg/ube-scheduler/api/validation"

	"github.com/golang/glog"
	"k8s.io/kubernetes/plugin/pkg/ube-scheduler/schedulercache"
)

const (
	SchedulerAnnotationKey = "scheduler.alpha.kubernetes.io/name"
)

// ConfigFactory knows how to fill out a scheduler config with its support functions.
type ConfigFactory struct {
	Client               *client.Client
	// queue for subRC that need scheduling
	FederationRCQueue    *cache.FIFO
	// a means to list all clusters
	ClusterLister        *cache.StoreToClusterLister
	// a means to list all services
	ServiceLister        *cache.StoreToServiceLister
	// a means to list all controllers
	//TODO: remove this as it is moved to queue
	ControllerLister     *cache.StoreToReplicationControllerLister

	// Close this to stop all reflectors
	StopEverything chan struct{}

	//TODO: no use, define here for api compatibility
	schedulerCache        schedulercache.Cache
	// SchedulerName of a scheduler is used to select which pods will be
	// processed by this scheduler, based on pods's annotation key:
	// 'scheduler.alpha.kubernetes.io/name'
	SchedulerName        string
}

// Initializes the factory.
func NewConfigFactory(client *client.Client, schedulerName string) *ConfigFactory {
	stopEverything := make(chan struct{})
	schedulerCache := schedulercache.New(30*time.Second, stopEverything)

	c := &ConfigFactory{
		Client:             client,
		FederationRCQueue:  cache.NewFIFO(cache.MetaNamespaceKeyFunc),
		// Only cluster in the "Ready" condition with status == "Running" are schedulable
		ClusterLister:      &cache.StoreToClusterLister{Store: cache.NewStore(cache.MetaNamespaceKeyFunc)},
		ServiceLister:    &cache.StoreToServiceLister{Store: cache.NewStore(cache.MetaNamespaceKeyFunc)},
		ControllerLister: &cache.StoreToReplicationControllerLister{Store: cache.NewStore(cache.MetaNamespaceKeyFunc)},
		schedulerCache:   schedulerCache,
		StopEverything:   stopEverything,
		SchedulerName:    schedulerName,
	}
	return c
}

// Create creates a scheduler with the default algorithm provider.
func (f *ConfigFactory) Create() (*scheduler.Config, error) {
	return f.CreateFromProvider(DefaultProvider)
}

// Creates a scheduler from the name of a registered algorithm provider.
func (f *ConfigFactory) CreateFromProvider(providerName string) (*scheduler.Config, error) {
	glog.V(2).Infof("Creating scheduler from algorithm provider '%v'", providerName)
	provider, err := GetAlgorithmProvider(providerName)
	if err != nil {
		return nil, err
	}

	return f.CreateFromKeys(provider.FitPredicateKeys, provider.PriorityFunctionKeys, []algorithm.SchedulerExtender{})
}

// Creates a scheduler from the configuration file
func (f *ConfigFactory) CreateFromConfig(policy schedulerapi.Policy) (*scheduler.Config, error) {
	glog.V(2).Infof("Creating scheduler from configuration: %v", policy)

	// validate the policy configuration
	if err := validation.ValidatePolicy(policy); err != nil {
		return nil, err
	}

	predicateKeys := sets.NewString()
	for _, predicate := range policy.Predicates {
		glog.V(2).Infof("Registering predicate: %s", predicate.Name)
		predicateKeys.Insert(RegisterCustomFitPredicate(predicate))
	}

	priorityKeys := sets.NewString()
	for _, priority := range policy.Priorities {
		glog.V(2).Infof("Registering priority: %s", priority.Name)
		priorityKeys.Insert(RegisterCustomPriorityFunction(priority))
	}

	extenders := make([]algorithm.SchedulerExtender, 0)
	if len(policy.ExtenderConfigs) != 0 {
		for ii := range policy.ExtenderConfigs {
			glog.V(2).Infof("Creating extender with config %+v", policy.ExtenderConfigs[ii])
			if extender, err := scheduler.NewHTTPExtender(&policy.ExtenderConfigs[ii], policy.APIVersion); err != nil {
				return nil, err
			} else {
				extenders = append(extenders, extender)
			}
		}
	}
	return f.CreateFromKeys(predicateKeys, priorityKeys, extenders)
}

// Creates a scheduler from a set of registered fit predicate keys and priority keys.
func (f *ConfigFactory) CreateFromKeys(predicateKeys, priorityKeys sets.String, extenders []algorithm.SchedulerExtender) (*scheduler.Config, error) {
	glog.V(2).Infof("creating scheduler with fit predicates '%v' and priority functions '%v", predicateKeys, priorityKeys)
	//type PluginFactoryArgs struct {
	//	ServiceLister    algorithm.ServiceLister
	//	ControllerLister algorithm.ControllerLister
	//	ClusterLister    algorithm.ClusterLister
	//	ClusterInfo      predicates.ClusterInfo
	//}
	pluginArgs := PluginFactoryArgs{
		ServiceLister:    f.ServiceLister,
		ControllerLister: f.ControllerLister,

		// All fit predicates only need to consider schedulable nodes.
		ClusterLister: f.ClusterLister.ClusterCondition(getClusterConditionPredicate()),
		ClusterInfo:   &predicates.CachedClusterInfo{f.ClusterLister},
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
	cache.NewReflector(f.createUnassignedRCLW(), &api.ReplicationController{}, f.FederationRCQueue, 0).RunUntil(f.StopEverything)

	// Watch nodes.
	// Nodes may be listed frequently, so provide a local up-to-date cache.
	cache.NewReflector(f.createClusterLW(), &controlplane.Cluster{}, f.ClusterLister.Store, 0).RunUntil(f.StopEverything)

	// Watch and cache all service objects. Scheduler needs to find all pods
	// created by the same services or ReplicationControllers/ReplicaSets, so that it can spread them correctly.
	// Cache this locally.
	cache.NewReflector(f.createServiceLW(), &api.Service{}, f.ServiceLister.Store, 0).RunUntil(f.StopEverything)


	r := rand.New(rand.NewSource(time.Now().UnixNano()))

	algo := scheduler.NewGenericScheduler(f.schedulerCache, predicateFuncs, priorityConfigs, extenders, r)

	podBackoff := rcBackoff{
		perRcBackoff: map[types.NamespacedName]*backoffEntry{},
		clock:         realClock{},

		defaultDuration: 1 * time.Second,
		maxDuration:     60 * time.Second,
	}

	return &scheduler.Config{
		SchedulerCache: f.schedulerCache,
		// The scheduler only needs to consider schedulable nodes.
		ClusterLister: f.ClusterLister,
		Algorithm:  algo,
		Binder:     &binder{f.Client},
		NextFederationRC: func() *api.ReplicationController {
			return f.getNextFederationRC()
		},
		Error:          f.makeDefaultErrorFunc(&podBackoff, f.FederationRCQueue),
		StopEverything: f.StopEverything,
	}, nil
}

func (f *ConfigFactory) getNextFederationRC() *api.ReplicationController {
	for {
		federationRC := f.FederationRCQueue.Pop().(*api.ReplicationController)
		if f.responsibleForFederationRC(federationRC) {
			glog.V(4).Infof("About to try and schedule federationRC %v", federationRC.Name)
			return federationRC
		}
	}
}

func (f *ConfigFactory) responsibleForFederationRC(rc *api.ReplicationController) bool {
	if f.SchedulerName == api.DefaultSchedulerName {
		return rc.Annotations[SchedulerAnnotationKey] == f.SchedulerName || rc.Annotations[SchedulerAnnotationKey] == ""
	} else {
		return rc.Annotations[SchedulerAnnotationKey] == f.SchedulerName
	}
}

func getClusterConditionPredicate() cache.ClusterConditionPredicate {
	return func(cluster controlplane.Cluster) bool {
		if cluster.Status.Phase == controlplane.ClusterRunning {
			return true
		}
		return false
	}
}

// Returns a cache.ListWatch that finds all FederationRC that are unscheduled.
// TODO: return a ListerWatcher interface instead? mfanjie to refine selector, if nonTerminated is needed?
func (factory *ConfigFactory) createUnassignedRCLW() *cache.ListWatch {
	selector := fields.ParseSelectorOrDie("spec.clusterName!=" + "")
	return cache.NewListWatchFromClient(factory.Client, "replicationControllers", api.NamespaceAll, selector)
}

// createNodeLW returns a cache.ListWatch that gets all changes to nodes.
func (factory *ConfigFactory) createClusterLW() *cache.ListWatch {
	// TODO: Filter out nodes that doesn't have NodeReady condition.
	fields := fields.Set{api.NodeUnschedulableField: "false"}.AsSelector()
	return cache.NewListWatchFromClient(factory.Client, "nodes", api.NamespaceAll, fields)
}

// Returns a cache.ListWatch that gets all changes to services.
func (factory *ConfigFactory) createServiceLW() *cache.ListWatch {
	return cache.NewListWatchFromClient(factory.Client, "services", api.NamespaceAll, fields.ParseSelectorOrDie(""))
}

// Returns a cache.ListWatch that gets all changes to controllers.
func (factory *ConfigFactory) createControllerLW() *cache.ListWatch {
	return cache.NewListWatchFromClient(factory.Client, "replicationControllers", api.NamespaceAll, fields.ParseSelectorOrDie(""))
}

func (factory *ConfigFactory) makeDefaultErrorFunc(backoff *rcBackoff, federationRCQueue *cache.FIFO) func(rc *api.ReplicationController, err error) {
	return func(rc *api.ReplicationController, err error) {
		if err == scheduler.ErrNoClustersAvailable {
			glog.V(4).Infof("Unable to schedule %v %v: no nodes are registered to the cluster; waiting", rc.Namespace, rc.Name)
		} else {
			glog.Errorf("Error scheduling %v %v: %v; retrying", rc.Namespace, rc.Name, err)
		}
		backoff.gc()
		// Retry asynchronously.
		// Note that this is extremely rudimentary and we need a more real error handling path.
		go func() {
			defer runtime.HandleCrash()
			rcID := types.NamespacedName{
				Namespace: rc.Namespace,
				Name:      rc.Name,
			}

			entry := backoff.getEntry(rcID)
			if !entry.TryWait(backoff.maxDuration) {
				glog.Warningf("Request for replicationcontroller %v already in flight, abandoning", rcID)
				return
			}
			// Get the pod again; it may have changed/been scheduled already.
			rc = &api.ReplicationController{}
			err := factory.Client.Get().Namespace(rcID.Namespace).Resource("replicationcontrollers").Name(rcID.Name).Do().Into(rc)
			if err != nil {
				if !errors.IsNotFound(err) {
					glog.Errorf("Error getting pod %v for retry: %v; abandoning", rcID, err)
				}
				return
			}
			if rc.Spec.Template.Spec.ClusterSelector == "" {
				federationRCQueue.AddIfNotPresent(rc)
			}
		}()
	}
}

// nodeEnumerator allows a cache.Poller to enumerate items in an api.NodeList
type nodeEnumerator struct {
	*api.ClusterList
}

// Len returns the number of items in the node list.
func (ne *nodeEnumerator) Len() int {
	if ne.ClusterList == nil {
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

// backoffEntry is single threaded.  in particular, it only allows a single action to be waiting on backoff at a time.
// It is expected that all users will only use the public TryWait(...) method
// It is also not safe to copy this object.
type backoffEntry struct {
	backoff     time.Duration
	lastUpdate  time.Time
	reqInFlight int32
}

// tryLock attempts to acquire a lock via atomic compare and swap.
// returns true if the lock was acquired, false otherwise
func (b *backoffEntry) tryLock() bool {
	return atomic.CompareAndSwapInt32(&b.reqInFlight, 0, 1)
}

// unlock returns the lock.  panics if the lock isn't held
func (b *backoffEntry) unlock() {
	if !atomic.CompareAndSwapInt32(&b.reqInFlight, 1, 0) {
		panic(fmt.Sprintf("unexpected state on unlocking: %+v", b))
	}
}

// TryWait tries to acquire the backoff lock, maxDuration is the maximum allowed period to wait for.
func (b *backoffEntry) TryWait(maxDuration time.Duration) bool {
	if !b.tryLock() {
		return false
	}
	defer b.unlock()
	b.wait(maxDuration)
	return true
}

func (entry *backoffEntry) getBackoff(maxDuration time.Duration) time.Duration {
	duration := entry.backoff
	newDuration := time.Duration(duration) * 2
	if newDuration > maxDuration {
		newDuration = maxDuration
	}
	entry.backoff = newDuration
	glog.V(4).Infof("Backing off %s for pod %+v", duration.String(), entry)
	return duration
}

func (entry *backoffEntry) wait(maxDuration time.Duration) {
	time.Sleep(entry.getBackoff(maxDuration))
}

type rcBackoff struct {
	perRcBackoff    map[types.NamespacedName]*backoffEntry
	lock            sync.Mutex
	clock           clock
	defaultDuration time.Duration
	maxDuration     time.Duration
}

func (p *rcBackoff) getEntry(RcID types.NamespacedName) *backoffEntry {
	p.lock.Lock()
	defer p.lock.Unlock()
	entry, ok := p.perRcBackoff[RcID]
	if !ok {
		entry = &backoffEntry{backoff: p.defaultDuration}
		p.perRcBackoff[RcID] = entry
	}
	entry.lastUpdate = p.clock.Now()
	return entry
}

func (p *rcBackoff) gc() {
	p.lock.Lock()
	defer p.lock.Unlock()
	now := p.clock.Now()
	for podID, entry := range p.perRcBackoff {
		if now.Sub(entry.lastUpdate) > p.maxDuration {
			delete(p.perRcBackoff, podID)
		}
	}
}
