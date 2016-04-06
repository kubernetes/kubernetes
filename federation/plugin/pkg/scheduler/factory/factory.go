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
	"k8s.io/kubernetes/pkg/apis/extensions"
	"k8s.io/kubernetes/pkg/api/errors"
	"k8s.io/kubernetes/federation/apis/federation"
	"k8s.io/kubernetes/pkg/client/cache"
	client "k8s.io/kubernetes/pkg/client/unversioned"
	"k8s.io/kubernetes/pkg/controller/framework"
	"k8s.io/kubernetes/pkg/fields"
	"k8s.io/kubernetes/pkg/types"
	"k8s.io/kubernetes/pkg/util/runtime"
	"k8s.io/kubernetes/pkg/util/sets"
	"k8s.io/kubernetes/federation/plugin/pkg/scheduler"
	"k8s.io/kubernetes/federation/plugin/pkg/scheduler/algorithm"
	"k8s.io/kubernetes/federation/plugin/pkg/scheduler/algorithm/predicates"
	schedulerapi "k8s.io/kubernetes/federation/plugin/pkg/scheduler/api"
	"k8s.io/kubernetes/federation/plugin/pkg/scheduler/api/validation"
	"k8s.io/kubernetes/federation/plugin/pkg/scheduler/schedulercache"

	"github.com/golang/glog"
)

const (
	SchedulerAnnotationKey = "scheduler.alpha.kubernetes.io/name"
)

// ConfigFactory knows how to fill out a scheduler config with its support functions.
type ConfigFactory struct {
	Client           *client.Client
	// queue for subReplicaSet that need scheduling
	ReplicaSetQueue  *cache.FIFO
	// a means to list all known scheduled replicaSets.
	ScheduledReplicaSetLister *cache.StoreToReplicaSetLister
	// a means to list all known scheduled replicaSets and replicaSets assumed to have been scheduled.
	ReplicaSetLister algorithm.ReplicaSetLister
	// a means to list all clusters
	ClusterLister    *cache.StoreToClusterLister

	// Close this to stop all reflectors
	StopEverything   chan struct{}

	scheduledReplicaSetPopulator *framework.Controller
	schedulerCache   schedulercache.Cache
	// SchedulerName of a scheduler is used to select which replicaSets will be
	// processed by this scheduler, based on replicaSets's annotation key:
	// 'scheduler.alpha.kubernetes.io/name'
	SchedulerName    string
}

// Initializes the factory.
func NewConfigFactory(client *client.Client, schedulerName string) *ConfigFactory {
	stopEverything := make(chan struct{})
	schedulerCache := schedulercache.New(30*time.Second, stopEverything)

	c := &ConfigFactory{
		Client:             client,
		ReplicaSetQueue:  cache.NewFIFO(cache.MetaNamespaceKeyFunc),
		// Only cluster in the "Ready" condition with status == "Running" are schedulable
		ClusterLister:      &cache.StoreToClusterLister{Store: cache.NewStore(cache.MetaNamespaceKeyFunc)},
		schedulerCache:   schedulerCache,
		StopEverything:   stopEverything,
		SchedulerName:    schedulerName,
	}
	// ReplicaSetLister is not needed in phase I
	//c.ReplicaSetLister = schedulerCache

	// On add/delete to the scheduled replicaSets, remove from the assumed replicaSets.
	// We construct this here instead of in CreateFromKeys because
	// ScheduledReplicaSetLister is something we provide to plug in functions that
	// they may need to call.
	c.ScheduledReplicaSetLister.Store, c.scheduledReplicaSetPopulator = framework.NewInformer(
		c.createAssignedNonTerminatedReplicaSetLW(),
		&extensions.ReplicaSet{},
		0,
		framework.ResourceEventHandlerFuncs{
			AddFunc: func(obj interface{}) {
				replicaSet, ok := obj.(*extensions.ReplicaSet)
				if !ok {
					glog.Errorf("cannot convert to *extensions.ReplicaSet")
					return
				}
				if err := schedulerCache.AddReplicaSet(replicaSet); err != nil {
					glog.Errorf("scheduler cache AddReplicaSet failed: %v", err)
				}
			},
			UpdateFunc: func(oldObj, newObj interface{}) {
				oldReplicaSet, ok := oldObj.(*extensions.ReplicaSet)
				if !ok {
					glog.Errorf("cannot convert to *extensions.ReplicaSet")
					return
				}
				newReplicaSet, ok := newObj.(*extensions.ReplicaSet)
				if !ok {
					glog.Errorf("cannot convert to *extensions.ReplicaSet")
					return
				}
				if err := schedulerCache.UpdateReplicaSet(oldReplicaSet, newReplicaSet); err != nil {
					glog.Errorf("scheduler cache UpdateReplicaSet failed: %v", err)
				}
			},
			DeleteFunc: func(obj interface{}) {
				var replicaSet *extensions.ReplicaSet
				switch t := obj.(type) {
				case *extensions.ReplicaSet:
					replicaSet = t
				case cache.DeletedFinalStateUnknown:
					var ok bool
					replicaSet, ok = t.Obj.(*extensions.ReplicaSet)
					if !ok {
						glog.Errorf("cannot convert to *extensions.ReplicaSet")
						return
					}
				default:
					glog.Errorf("cannot convert to *extensions.ReplicaSet")
					return
				}
				if err := schedulerCache.RemoveReplicaSet(replicaSet); err != nil {
					glog.Errorf("scheduler cache RemoveReplicaSet failed: %v", err)
				}
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
	pluginArgs := PluginFactoryArgs{
		// All fit predicates only need to consider schedulable clusters.
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

	// Watch and queue replicaSets that need scheduling.
	cache.NewReflector(f.createUnassignedReplicaSetLW(), &extensions.ReplicaSet{}, f.ReplicaSetQueue, 0).RunUntil(f.StopEverything)

	// Watch clusters.
	// Clusters may be listed frequently, so provide a local up-to-date cache.
	cache.NewReflector(f.createClusterLW(), &federation.Cluster{}, f.ClusterLister.Store, 0).RunUntil(f.StopEverything)

	r := rand.New(rand.NewSource(time.Now().UnixNano()))

	algo := scheduler.NewGenericScheduler(f.schedulerCache, predicateFuncs, priorityConfigs, extenders, r)

	replicaSetBackoff := rsBackoff{
		perRcBackoff: map[types.NamespacedName]*backoffEntry{},
		clock:         realClock{},

		defaultDuration: 1 * time.Second,
		maxDuration:     60 * time.Second,
	}

	return &scheduler.Config{
		SchedulerCache: f.schedulerCache,
		// The scheduler only needs to consider schedulable clusters.
		ClusterLister: f.ClusterLister,
		Algorithm:  algo,
		Binder:     &binder{f.Client},
		NextReplicaSet: func() *extensions.ReplicaSet {
			return f.getNextReplicaSet()
		},
		Error:          f.makeDefaultErrorFunc(&replicaSetBackoff, f.ReplicaSetQueue),
		StopEverything: f.StopEverything,
	}, nil
}

func (f *ConfigFactory) getNextReplicaSet() *extensions.ReplicaSet {
	for {
		replicaSet := f.ReplicaSetQueue.Pop().(*extensions.ReplicaSet)
		if f.responsibleForReplicaSet(replicaSet) {
			glog.V(4).Infof("About to try and schedule federationRC %v", replicaSet.Name)
			return replicaSet
		}
	}
}

func (f *ConfigFactory) responsibleForReplicaSet(rc *extensions.ReplicaSet) bool {
	if f.SchedulerName == api.DefaultSchedulerName {
		return rc.Annotations[SchedulerAnnotationKey] == f.SchedulerName || rc.Annotations[SchedulerAnnotationKey] == ""
	} else {
		return rc.Annotations[SchedulerAnnotationKey] == f.SchedulerName
	}
}

func getClusterConditionPredicate() cache.ClusterConditionPredicate {
	return func(cluster federation.Cluster) bool {
		if cluster.Status.Phase == federation.ClusterRunning {
			return true
		}
		return false
	}
}

// Returns a cache.ListWatch that finds all FederationRC that are unscheduled.
// TODO: discuss - how to get unscheduled replicaset
func (factory *ConfigFactory) createUnassignedReplicaSetLW() *cache.ListWatch {
	selector := fields.ParseSelectorOrDie("")
	return cache.NewListWatchFromClient(factory.Client, "replicaSet", api.NamespaceAll, selector)
}

// Returns a cache.ListWatch that finds all replicaset that are
// already scheduled.
//TODO: discuss - how to get scheduled replicaset, should we set an annotation on replicaset after scheduled
func (factory *ConfigFactory) createAssignedNonTerminatedReplicaSetLW() *cache.ListWatch {
	return cache.NewListWatchFromClient(factory.Client, "replicasets", api.NamespaceAll, fields.ParseSelectorOrDie(""))
}
// createClusterLW returns a cache.ListWatch that gets all changes to clusters.
func (factory *ConfigFactory) createClusterLW() *cache.ListWatch {
	fields := fields.Set{"spec.status.phase": string(federation.ClusterRunning)}.AsSelector()
	return cache.NewListWatchFromClient(factory.Client, "clusters", api.NamespaceAll, fields)
}

func (factory *ConfigFactory) makeDefaultErrorFunc(backoff *rsBackoff, replicaSetQueue *cache.FIFO) func(rs *extensions.ReplicaSet, err error) {
	return func(rs *extensions.ReplicaSet, err error) {
		if err == scheduler.ErrNoClustersAvailable {
			glog.V(4).Infof("Unable to schedule %v %v: no clusters are registered to the federation; waiting", rs.Namespace, rs.Name)
		} else {
			glog.Errorf("Error scheduling %v %v: %v; retrying", rs.Namespace, rs.Name, err)
		}
		backoff.gc()
		// Retry asynchronously.
		// Note that this is extremely rudimentary and we need a more real error handling path.
		go func() {
			defer runtime.HandleCrash()
			rsID := types.NamespacedName{
				Namespace: rs.Namespace,
				Name:      rs.Name,
			}

			entry := backoff.getEntry(rsID)
			if !entry.TryWait(backoff.maxDuration) {
				glog.Warningf("Request for replicationcontroller %v already in flight, abandoning", rsID)
				return
			}
			// Get the replicaSet again; it may have changed/been scheduled already.
			rs = &extensions.ReplicaSet{}
			err := factory.Client.Get().Namespace(rsID.Namespace).Resource("replicasets").Name(rsID.Name).Do().Into(rs)
			if err != nil {
				if !errors.IsNotFound(err) {
					glog.Errorf("Error getting replicaSet %v for retry: %v; abandoning", rsID, err)
				}
				return
			}
			//if rs.Spec.Template.Spec.ClusterSelector == "" {
			//	replicaSetQueue.AddIfNotPresent(rs)
			//}
		}()
	}
}

// clusterEnumerator allows a cache.Poller to enumerate items in an api.ClusterList
type clusterEnumerator struct {
	*federation.ClusterList
}

// Len returns the number of items in the cluster list.
func (ce *clusterEnumerator) Len() int {
	if ce.ClusterList == nil {
		return 0
	}
	return len(ce.Items)
}

// Get returns the item (and ID) with the particular index.
func (ce *clusterEnumerator) Get(index int) interface{} {
	return &ce.Items[index]
}

type binder struct {
	*client.Client
}

// Bind just does a POST binding RPC.
func (b *binder) Bind(binding *api.Binding) error {
	glog.V(2).Infof("Attempting to bind %v to %v", binding.Name, binding.Target.Name)
	ctx := api.WithNamespace(api.NewContext(), binding.Namespace)
	return b.Post().Namespace(api.NamespaceValue(ctx)).Resource("bindings").Body(binding).Do().Error()
	// TODO: use ReplicaSets interface for binding once clusters are upgraded
	// return b.ReplicaSets(binding.Namespace).Bind(binding)
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
	glog.V(4).Infof("Backing off %s for replicaSet %+v", duration.String(), entry)
	return duration
}

func (entry *backoffEntry) wait(maxDuration time.Duration) {
	time.Sleep(entry.getBackoff(maxDuration))
}

type rsBackoff struct {
	perRcBackoff    map[types.NamespacedName]*backoffEntry
	lock            sync.Mutex
	clock           clock
	defaultDuration time.Duration
	maxDuration     time.Duration
}

func (p *rsBackoff) getEntry(RcID types.NamespacedName) *backoffEntry {
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

func (p *rsBackoff) gc() {
	p.lock.Lock()
	defer p.lock.Unlock()
	now := p.clock.Now()
	for replicaSetID, entry := range p.perRcBackoff {
		if now.Sub(entry.lastUpdate) > p.maxDuration {
			delete(p.perRcBackoff, replicaSetID)
		}
	}
}
