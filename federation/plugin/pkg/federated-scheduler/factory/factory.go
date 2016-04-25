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

// Package factory can set up a federated-scheduler. This code is here instead of
// plugin/cmd/federated-scheduler for both testability and reuse.
package factory

import (
	"fmt"
	"math/rand"
	"sync"
	"sync/atomic"
	"time"

	fedclient "k8s.io/kubernetes/federation/client/clientset_generated/release_1_3"
	kubeclient "k8s.io/kubernetes/pkg/client/clientset_generated/release_1_3"
	"k8s.io/kubernetes/pkg/api"
	extensions "k8s.io/kubernetes/pkg/apis/extensions/v1beta1"
	"k8s.io/kubernetes/pkg/api/errors"
	federation "k8s.io/kubernetes/federation/apis/federation/v1alpha1"
	"k8s.io/kubernetes/pkg/client/cache"
	"k8s.io/kubernetes/pkg/controller/framework"
	"k8s.io/kubernetes/pkg/types"
	utilruntime "k8s.io/kubernetes/pkg/util/runtime"
	"k8s.io/kubernetes/pkg/util/sets"
	"k8s.io/kubernetes/federation/plugin/pkg/federated-scheduler"
	"k8s.io/kubernetes/federation/plugin/pkg/federated-scheduler/algorithm/predicates"
	schedulerapi "k8s.io/kubernetes/federation/plugin/pkg/federated-scheduler/api"
	"k8s.io/kubernetes/federation/plugin/pkg/federated-scheduler/api/validation"
	"k8s.io/kubernetes/federation/plugin/pkg/federated-scheduler/schedulercache"

	"github.com/golang/glog"
	"k8s.io/kubernetes/federation/apis/federation/unversioned"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/watch"
	"k8s.io/kubernetes/pkg/api/v1"
)

const (
	SchedulerAnnotationKey = "federated-scheduler.alpha.kubernetes.io/name"
)

// ConfigFactory knows how to fill out a federated-scheduler config with its support functions.
type ConfigFactory struct {
	FederatedClientSet   *fedclient.Clientset
	KubeClientSet        *kubeclient.Clientset
	// queue for subReplicaSet that need scheduling
	ReplicaSetQueue      *cache.FIFO
	// a means to list all known replicaSets.
	ScheduledSubRSLister *cache.StoreToSubRSLister
	// a means to list all clusters
	ClusterLister        *cache.StoreToClusterLister

	// Close this to stop all reflectors
	StopEverything          chan struct{}

	scheduledSubRSPopulator *framework.Controller
	schedulerCache          schedulercache.Cache
	// SchedulerName of a federated-scheduler is used to select which replicaSets will be
	// processed by this federated-scheduler, based on replicaSets's annotation key:
	// 'federated-scheduler.alpha.kubernetes.io/name'
	SchedulerName        string
}

// Initializes the factory.
func NewConfigFactory(federatedClientSet *fedclient.Clientset, kubeClientSet *kubeclient.Clientset, schedulerName string) *ConfigFactory {
	stopEverything := make(chan struct{})
	schedulerCache := schedulercache.New(30*time.Second, stopEverything)

	c := &ConfigFactory{
		FederatedClientSet:             federatedClientSet,
		KubeClientSet:             kubeClientSet,
		ReplicaSetQueue:  cache.NewFIFO(cache.MetaNamespaceKeyFunc),
		ScheduledSubRSLister: &cache.StoreToSubRSLister{},
		// Only cluster in the "Ready" condition with status == "Running" are schedulable
		ClusterLister:      &cache.StoreToClusterLister{Store: cache.NewStore(cache.MetaNamespaceKeyFunc)},
		schedulerCache:   schedulerCache,
		StopEverything:   stopEverything,
		SchedulerName:    schedulerName,
	}

	// On add/delete to the scheduled SubRS, remove from the assumed SubRS.
	// We construct this here instead of in CreateFromKeys because
	// ScheduledReplicaSetLister is something we provide to plug in functions that
	// they may need to call.

	c.ScheduledSubRSLister.Store, c.scheduledSubRSPopulator = framework.NewInformer(
		c.createSubReplicaSetLW(),
		&federation.SubReplicaSet{},
		0,
		framework.ResourceEventHandlerFuncs{
			AddFunc: func(obj interface{}) {
				subRS, ok := obj.(*federation.SubReplicaSet)
				if !ok {
					glog.Errorf("cannot convert to *federation.SubReplicaSet")
					return
				}
				rs, err := scheduler.CoverSubRSToRS(subRS)
				if err != nil {
					glog.Errorf("cannot convert to *v1beta.ReplicaSet")
					return
				}
				if err := schedulerCache.AddReplicaSet(rs); err != nil {
					glog.Errorf("federated-scheduler cache AddReplicaSet failed: %v", err)
				}
			},
			UpdateFunc: func(oldObj, newObj interface{}) {
				oldSub, ok := oldObj.(*federation.SubReplicaSet)
				if !ok {
					glog.Errorf("cannot convert to *federation.SubReplicaSet")
					return
				}
				oldRS, err := scheduler.CoverSubRSToRS(oldSub)
				if err != nil {
					glog.Errorf("cannot convert to *v1beta.ReplicaSet")
					return
				}
				newSub, ok := newObj.(*federation.SubReplicaSet)
				if !ok {
					glog.Errorf("cannot convert to *federation.SubReplicaSet")
					return
				}
				newRS, err := scheduler.CoverSubRSToRS(newSub)
				if err != nil {
					glog.Errorf("cannot convert to *v1beta.ReplicaSet")
					return
				}
				if err := schedulerCache.UpdateReplicaSet(oldRS, newRS); err != nil {
					glog.Errorf("federated-scheduler cache UpdateReplicaSet failed: %v", err)
				}
			},
			DeleteFunc: func(obj interface{}) {
				var subRS *federation.SubReplicaSet
				switch t := obj.(type) {
				case *federation.SubReplicaSet:
					subRS = t
				case cache.DeletedFinalStateUnknown:
					var ok bool
					subRS, ok = t.Obj.(*federation.SubReplicaSet)
					if !ok {
						glog.Errorf("cannot convert to *federation.SubReplicaSet")
						return
					}
				default:
					glog.Errorf("cannot convert to *federation.SubReplicaSet")
					return
				}
				rs, err := scheduler.CoverSubRSToRS(subRS)
				if err != nil {
					glog.Errorf("cannot convert to *v1beta.ReplicaSet")
					return
				}
				if err := schedulerCache.RemoveReplicaSet(rs); err != nil {
					glog.Errorf("federated-scheduler cache RemoveReplicaSet failed: %v", err)
				}
			},
		},
	)
	return c
}

// Create creates a federated-scheduler with the default algorithm provider.
func (f *ConfigFactory) Create() (*scheduler.Config, error) {
	return f.CreateFromProvider(DefaultProvider)
}

// Creates a federated-scheduler from the name of a registered algorithm provider.
func (f *ConfigFactory) CreateFromProvider(providerName string) (*scheduler.Config, error) {
	glog.V(2).Infof("Creating federated-scheduler from algorithm provider '%v'", providerName)
	provider, err := GetAlgorithmProvider(providerName)
	if err != nil {
		return nil, err
	}

	return f.CreateFromKeys(provider.FitPredicateKeys, provider.PriorityFunctionKeys)
}

// Creates a federated-scheduler from the configuration file
func (f *ConfigFactory) CreateFromConfig(policy schedulerapi.Policy) (*scheduler.Config, error) {
	glog.V(2).Infof("Creating federated-scheduler from configuration: %v", policy)

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
	return f.CreateFromKeys(predicateKeys, priorityKeys)
}

// Creates a federated-scheduler from a set of registered fit predicate keys and priority keys.
func (f *ConfigFactory) CreateFromKeys(predicateKeys, priorityKeys sets.String) (*scheduler.Config, error) {
	glog.V(2).Infof("creating federated-scheduler with fit predicates '%v' and priority functions '%v", predicateKeys, priorityKeys)
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

	// Begin populating scheduled subRS.
	go f.scheduledSubRSPopulator.Run(f.StopEverything)

	// Watch and queue replicaSets that need scheduling.
	cache.NewReflector(f.createReplicaSetLW(), &extensions.ReplicaSet{}, f.ReplicaSetQueue, 0).RunUntil(f.StopEverything)

	// Watch clusters.
	// Clusters may be listed frequently, so provide a local up-to-date cache.
	cache.NewReflector(f.createRunningClusterLW(), &federation.Cluster{}, f.ClusterLister.Store, 0).RunUntil(f.StopEverything)

	r := rand.New(rand.NewSource(time.Now().UnixNano()))

	algo := scheduler.NewGenericScheduler(f.schedulerCache, predicateFuncs, priorityConfigs, r)

	replicaSetBackoff := rsBackoff{
		perRsBackoff: map[types.NamespacedName]*backoffEntry{},
		clock:         realClock{},

		defaultDuration: 1 * time.Second,
		maxDuration:     60 * time.Second,
	}

	return &scheduler.Config{
		SchedulerCache: f.schedulerCache,
		// The federated-scheduler only needs to consider schedulable clusters.
		ClusterLister: f.ClusterLister,
		Algorithm:  algo,
		Binder:     &binder{f.FederatedClientSet},
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
			glog.V(4).Infof("About to try and schedule replicaset %v", replicaSet.Name)
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
		// If we have no info, don't accept
		if len(cluster.Status.Conditions) == 0 {
			return false
		}
		for _, cond := range cluster.Status.Conditions {
			//We consider the cluster for load balancing only when its ClusterReady condition status
			//is ConditionTrue
			if cond.Type == federation.ClusterReady && cond.Status != v1.ConditionTrue {
				glog.V(4).Infof("Ignoring cluser %v with %v condition status %v", cluster.Name, cond.Type, cond.Status)
				return false
			}
		}
		return true
	}
}

// Returns a cache.ListWatch that finds all replicasets
func (factory *ConfigFactory) createReplicaSetLW() *cache.ListWatch {
	return &cache.ListWatch{
		ListFunc: func(options api.ListOptions) (runtime.Object, error) {
			return factory.KubeClientSet.ExtensionsClient.ReplicaSets(api.NamespaceAll).List(options)
		},
		WatchFunc: func(options api.ListOptions) (watch.Interface, error) {
			return factory.KubeClientSet.ExtensionsClient.ReplicaSets(api.NamespaceAll).Watch(options)
		},
	}
}

// Returns a cache.ListWatch that finds all subreplicasets
func (factory *ConfigFactory) createSubReplicaSetLW() *cache.ListWatch {
	return &cache.ListWatch{
		ListFunc: func(options api.ListOptions) (runtime.Object, error) {
			return factory.FederatedClientSet.FederationClient.SubReplicaSets(api.NamespaceAll).List(options)
		},
		WatchFunc: func(options api.ListOptions) (watch.Interface, error) {
			return factory.FederatedClientSet.FederationClient.SubReplicaSets(api.NamespaceAll).Watch(options)
		},
	}
}

// createClusterLW returns a cache.ListWatch that gets all changes to clusters.
func (factory *ConfigFactory) createRunningClusterLW() *cache.ListWatch {
	return &cache.ListWatch{
		ListFunc: func(options api.ListOptions) (runtime.Object, error) {
			return factory.FederatedClientSet.FederationClient.Clusters().List(options)
		},
		WatchFunc: func(options api.ListOptions) (watch.Interface, error) {
			return factory.FederatedClientSet.FederationClient.Clusters().Watch(options)
		},
	}
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
			defer utilruntime.HandleCrash()
			rsID := types.NamespacedName{
				Namespace: rs.Namespace,
				Name:      rs.Name,
			}

			entry := backoff.getEntry(rsID)
			if !entry.TryWait(backoff.maxDuration) {
				glog.Warningf("Request for replicaset %v already in flight, abandoning", rsID)
				return
			}
			// Get the replicaSet again; it may have changed/been scheduled already.
			rs = &extensions.ReplicaSet{}
			err := factory.KubeClientSet.ExtensionsClient.Get().Namespace(rsID.Namespace).Resource("replicasets").Name(rsID.Name).Do().Into(rs)
			if err != nil {
				if !errors.IsNotFound(err) {
					glog.Errorf("Error getting replicaSet %v for retry: %v; abandoning", rsID, err)
				}
				return
			}
			target, ok := rs.Annotations[unversioned.TargetClusterKey]
			if  !ok || target == "" {
				replicaSetQueue.AddIfNotPresent(rs)
			}

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
	*fedclient.Clientset
}

// Bind just does a POST binding RPC.
// we simply create the sub replicaset with target cluster annotation
func (b *binder) Bind(subRS *federation.SubReplicaSet) error {
	glog.V(2).Infof("Attempting to create SubReplicaSet %v", subRS)
	_, error := b.Federation().SubReplicaSets(subRS.Namespace).Create(subRS)
	return error
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
	perRsBackoff    map[types.NamespacedName]*backoffEntry
	lock            sync.Mutex
	clock           clock
	defaultDuration time.Duration
	maxDuration     time.Duration
}

func (p *rsBackoff) getEntry(RcID types.NamespacedName) *backoffEntry {
	p.lock.Lock()
	defer p.lock.Unlock()
	entry, ok := p.perRsBackoff[RcID]
	if !ok {
		entry = &backoffEntry{backoff: p.defaultDuration}
		p.perRsBackoff[RcID] = entry
	}
	entry.lastUpdate = p.clock.Now()
	return entry
}

func (p *rsBackoff) gc() {
	p.lock.Lock()
	defer p.lock.Unlock()
	now := p.clock.Now()
	for replicaSetID, entry := range p.perRsBackoff {
		if now.Sub(entry.lastUpdate) > p.maxDuration {
			delete(p.perRsBackoff, replicaSetID)
		}
	}
}
