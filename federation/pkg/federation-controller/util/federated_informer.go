/*
Copyright 2016 The Kubernetes Authors.

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

package util

import (
	"fmt"
	"reflect"
	"sync"
	"time"

	apiv1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	pkgruntime "k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/watch"
	kubeclientset "k8s.io/client-go/kubernetes"
	restclient "k8s.io/client-go/rest"
	"k8s.io/client-go/tools/cache"
	federationapi "k8s.io/kubernetes/federation/apis/federation/v1beta1"
	federationclientset "k8s.io/kubernetes/federation/client/clientset_generated/federation_clientset"

	"github.com/golang/glog"
)

const (
	clusterSyncPeriod = 10 * time.Minute
	userAgentName     = "federation-controller"
)

// An object with an origin information.
type FederatedObject struct {
	Object      interface{}
	ClusterName string
}

// FederatedReadOnlyStore is an overlay over multiple stores created in federated clusters.
type FederatedReadOnlyStore interface {
	// Returns all items in the store.
	List() ([]FederatedObject, error)

	// Returns all items from a cluster.
	ListFromCluster(clusterName string) ([]interface{}, error)

	// GetKeyFor returns the key under which the item would be put in the store.
	GetKeyFor(item interface{}) string

	// GetByKey returns the item stored under the given key in the specified cluster (if exist).
	GetByKey(clusterName string, key string) (interface{}, bool, error)

	// Returns the items stored under the given key in all clusters.
	GetFromAllClusters(key string) ([]FederatedObject, error)

	// Checks whether stores for all clusters form the lists (and only these) are there and
	// are synced. This is only a basic check whether the data inside of the store is usable.
	// It is not a full synchronization/locking mechanism it only tries to ensure that out-of-sync
	// issues occur less often.	All users of the interface should assume
	// that there may be significant delays in content updates of all kinds and write their
	// code that it doesn't break if something is slightly out-of-sync.
	ClustersSynced(clusters []*federationapi.Cluster) bool
}

// An interface to access federation members and clients.
type FederationView interface {
	// GetClientsetForCluster returns a clientset for the cluster, if present.
	GetClientsetForCluster(clusterName string) (kubeclientset.Interface, error)

	// GetUnreadyClusters returns a list of all clusters that are not ready yet.
	GetUnreadyClusters() ([]*federationapi.Cluster, error)

	// GetReadyClusters returns all clusters for which the sub-informers are run.
	GetReadyClusters() ([]*federationapi.Cluster, error)

	// GetReadyCluster returns the cluster with the given name, if found.
	GetReadyCluster(name string) (*federationapi.Cluster, bool, error)

	// ClustersSynced returns true if the view is synced (for the first time).
	ClustersSynced() bool
}

// A structure that combines an informer running against federated api server and listening for cluster updates
// with multiple Kubernetes API informers (called target informers) running against federation members. Whenever a new
// cluster is added to the federation an informer is created for it using TargetInformerFactory. Informers are stopped
// when a cluster is either put offline of deleted. It is assumed that some controller keeps an eye on the cluster list
// and thus the clusters in ETCD are up to date.
type FederatedInformer interface {
	FederationView

	// Returns a store created over all stores from target informers.
	GetTargetStore() FederatedReadOnlyStore

	// Starts all the processes.
	Start()

	// Stops all the processes inside the informer.
	Stop()
}

// FederatedInformer with extra method for setting fake clients.
type FederatedInformerForTestOnly interface {
	FederatedInformer

	SetClientFactory(func(*federationapi.Cluster) (kubeclientset.Interface, error))
}

// A function that should be used to create an informer on the target object. Store should use
// cache.DeletionHandlingMetaNamespaceKeyFunc as a keying function.
type TargetInformerFactory func(*federationapi.Cluster, kubeclientset.Interface) (cache.Store, cache.Controller)

// A structure with cluster lifecycle handler functions. Cluster is available (and ClusterAvailable is fired)
// when it is created in federated etcd and ready. Cluster becomes unavailable (and ClusterUnavailable is fired)
// when it is either deleted or becomes not ready. When cluster spec (IP)is modified both ClusterAvailable
// and ClusterUnavailable are fired.
type ClusterLifecycleHandlerFuncs struct {
	// Fired when the cluster becomes available.
	ClusterAvailable func(*federationapi.Cluster)
	// Fired when the cluster becomes unavailable. The second arg contains data that was present
	// in the cluster before deletion.
	ClusterUnavailable func(*federationapi.Cluster, []interface{})
}

// Builds a FederatedInformer for the given federation client and factory.
func NewFederatedInformer(
	federationClient federationclientset.Interface,
	targetInformerFactory TargetInformerFactory,
	clusterLifecycle *ClusterLifecycleHandlerFuncs) FederatedInformer {

	federatedInformer := &federatedInformerImpl{
		targetInformerFactory: targetInformerFactory,
		clientFactory: func(cluster *federationapi.Cluster) (kubeclientset.Interface, error) {
			clusterConfig, err := BuildClusterConfig(cluster)
			if err == nil && clusterConfig != nil {
				clientset := kubeclientset.NewForConfigOrDie(restclient.AddUserAgent(clusterConfig, userAgentName))
				return clientset, nil
			}
			return nil, err
		},
		targetInformers: make(map[string]informer),
	}

	getClusterData := func(name string) []interface{} {
		data, err := federatedInformer.GetTargetStore().ListFromCluster(name)
		if err != nil {
			glog.Errorf("Failed to list %s content: %v", name, err)
			return make([]interface{}, 0)
		}
		return data
	}

	federatedInformer.clusterInformer.store, federatedInformer.clusterInformer.controller = cache.NewInformer(
		&cache.ListWatch{
			ListFunc: func(options metav1.ListOptions) (pkgruntime.Object, error) {
				return federationClient.Federation().Clusters().List(options)
			},
			WatchFunc: func(options metav1.ListOptions) (watch.Interface, error) {
				return federationClient.Federation().Clusters().Watch(options)
			},
		},
		&federationapi.Cluster{},
		clusterSyncPeriod,
		cache.ResourceEventHandlerFuncs{
			DeleteFunc: func(old interface{}) {
				oldCluster, ok := old.(*federationapi.Cluster)
				if ok {
					var data []interface{}
					if clusterLifecycle.ClusterUnavailable != nil {
						data = getClusterData(oldCluster.Name)
					}
					federatedInformer.deleteCluster(oldCluster)
					if clusterLifecycle.ClusterUnavailable != nil {
						clusterLifecycle.ClusterUnavailable(oldCluster, data)
					}
				}
			},
			AddFunc: func(cur interface{}) {
				curCluster, ok := cur.(*federationapi.Cluster)
				if ok && isClusterReady(curCluster) {
					federatedInformer.addCluster(curCluster)
					if clusterLifecycle.ClusterAvailable != nil {
						clusterLifecycle.ClusterAvailable(curCluster)
					}
				} else {
					glog.Errorf("Cluster %v not added.  Not of correct type, or cluster not ready.", cur)
				}
			},
			UpdateFunc: func(old, cur interface{}) {
				oldCluster, ok := old.(*federationapi.Cluster)
				if !ok {
					glog.Errorf("Internal error: Cluster %v not updated.  Old cluster not of correct type.", old)
					return
				}
				curCluster, ok := cur.(*federationapi.Cluster)
				if !ok {
					glog.Errorf("Internal error: Cluster %v not updated.  New cluster not of correct type.", cur)
					return
				}
				if isClusterReady(oldCluster) != isClusterReady(curCluster) || !reflect.DeepEqual(oldCluster.Spec, curCluster.Spec) || !reflect.DeepEqual(oldCluster.ObjectMeta.Annotations, curCluster.ObjectMeta.Annotations) {
					var data []interface{}
					if clusterLifecycle.ClusterUnavailable != nil {
						data = getClusterData(oldCluster.Name)
					}
					federatedInformer.deleteCluster(oldCluster)
					if clusterLifecycle.ClusterUnavailable != nil {
						clusterLifecycle.ClusterUnavailable(oldCluster, data)
					}

					if isClusterReady(curCluster) {
						federatedInformer.addCluster(curCluster)
						if clusterLifecycle.ClusterAvailable != nil {
							clusterLifecycle.ClusterAvailable(curCluster)
						}
					}
				} else {
					glog.V(4).Infof("Cluster %v not updated to %v as ready status and specs are identical", oldCluster, curCluster)
				}
			},
		},
	)
	return federatedInformer
}

func isClusterReady(cluster *federationapi.Cluster) bool {
	for _, condition := range cluster.Status.Conditions {
		if condition.Type == federationapi.ClusterReady {
			if condition.Status == apiv1.ConditionTrue {
				return true
			}
		}
	}
	return false
}

type informer struct {
	controller cache.Controller
	store      cache.Store
	stopChan   chan struct{}
}

type federatedInformerImpl struct {
	sync.Mutex

	// Informer on federated clusters.
	clusterInformer informer

	// Target informers factory
	targetInformerFactory TargetInformerFactory

	// Structures returned by targetInformerFactory
	targetInformers map[string]informer

	// A function to build clients.
	clientFactory func(*federationapi.Cluster) (kubeclientset.Interface, error)
}

// *federatedInformerImpl implements FederatedInformer interface.
var _ FederatedInformer = &federatedInformerImpl{}

type federatedStoreImpl struct {
	federatedInformer *federatedInformerImpl
}

func (f *federatedInformerImpl) Stop() {
	glog.V(4).Infof("Stopping federated informer.")
	f.Lock()
	defer f.Unlock()

	glog.V(4).Infof("... Closing cluster informer channel.")
	close(f.clusterInformer.stopChan)
	for key, informer := range f.targetInformers {
		glog.V(4).Infof("... Closing informer channel for %q.", key)
		close(informer.stopChan)
		// Remove each informer after it has been stopped to prevent
		// subsequent cluster deletion from attempting to double close
		// an informer's stop channel.
		delete(f.targetInformers, key)
	}
}

func (f *federatedInformerImpl) Start() {
	f.Lock()
	defer f.Unlock()

	f.clusterInformer.stopChan = make(chan struct{})
	go f.clusterInformer.controller.Run(f.clusterInformer.stopChan)
}

func (f *federatedInformerImpl) SetClientFactory(clientFactory func(*federationapi.Cluster) (kubeclientset.Interface, error)) {
	f.Lock()
	defer f.Unlock()

	f.clientFactory = clientFactory
}

// GetClientsetForCluster returns a clientset for the cluster, if present.
func (f *federatedInformerImpl) GetClientsetForCluster(clusterName string) (kubeclientset.Interface, error) {
	f.Lock()
	defer f.Unlock()
	return f.getClientsetForClusterUnlocked(clusterName)
}

func (f *federatedInformerImpl) getClientsetForClusterUnlocked(clusterName string) (kubeclientset.Interface, error) {
	// No locking needed. Will happen in f.GetCluster.
	glog.V(4).Infof("Getting clientset for cluster %q", clusterName)
	if cluster, found, err := f.getReadyClusterUnlocked(clusterName); found && err == nil {
		glog.V(4).Infof("Got clientset for cluster %q", clusterName)
		return f.clientFactory(cluster)
	} else {
		if err != nil {
			return nil, err
		}
	}
	return nil, fmt.Errorf("cluster %q not found", clusterName)
}

func (f *federatedInformerImpl) GetUnreadyClusters() ([]*federationapi.Cluster, error) {
	f.Lock()
	defer f.Unlock()

	items := f.clusterInformer.store.List()
	result := make([]*federationapi.Cluster, 0, len(items))
	for _, item := range items {
		if cluster, ok := item.(*federationapi.Cluster); ok {
			if !isClusterReady(cluster) {
				result = append(result, cluster)
			}
		} else {
			return nil, fmt.Errorf("wrong data in FederatedInformerImpl cluster store: %v", item)
		}
	}
	return result, nil
}

// GetReadyClusters returns all clusters for which the sub-informers are run.
func (f *federatedInformerImpl) GetReadyClusters() ([]*federationapi.Cluster, error) {
	f.Lock()
	defer f.Unlock()

	items := f.clusterInformer.store.List()
	result := make([]*federationapi.Cluster, 0, len(items))
	for _, item := range items {
		if cluster, ok := item.(*federationapi.Cluster); ok {
			if isClusterReady(cluster) {
				result = append(result, cluster)
			}
		} else {
			return nil, fmt.Errorf("wrong data in FederatedInformerImpl cluster store: %v", item)
		}
	}
	return result, nil
}

// GetCluster returns the cluster with the given name, if found.
func (f *federatedInformerImpl) GetReadyCluster(name string) (*federationapi.Cluster, bool, error) {
	f.Lock()
	defer f.Unlock()
	return f.getReadyClusterUnlocked(name)
}

func (f *federatedInformerImpl) getReadyClusterUnlocked(name string) (*federationapi.Cluster, bool, error) {
	if obj, exist, err := f.clusterInformer.store.GetByKey(name); exist && err == nil {
		if cluster, ok := obj.(*federationapi.Cluster); ok {
			if isClusterReady(cluster) {
				return cluster, true, nil
			}
			return nil, false, nil

		}
		return nil, false, fmt.Errorf("wrong data in FederatedInformerImpl cluster store: %v", obj)

	} else {
		return nil, false, err
	}
}

// Synced returns true if the view is synced (for the first time)
func (f *federatedInformerImpl) ClustersSynced() bool {
	return f.clusterInformer.controller.HasSynced()
}

// Adds the given cluster to federated informer.
func (f *federatedInformerImpl) addCluster(cluster *federationapi.Cluster) {
	f.Lock()
	defer f.Unlock()
	name := cluster.Name
	if client, err := f.getClientsetForClusterUnlocked(name); err == nil {
		store, controller := f.targetInformerFactory(cluster, client)
		targetInformer := informer{
			controller: controller,
			store:      store,
			stopChan:   make(chan struct{}),
		}
		f.targetInformers[name] = targetInformer
		go targetInformer.controller.Run(targetInformer.stopChan)
	} else {
		// TODO: create also an event for cluster.
		glog.Errorf("Failed to create a client for cluster: %v", err)
	}
}

// Removes the cluster from federated informer.
func (f *federatedInformerImpl) deleteCluster(cluster *federationapi.Cluster) {
	f.Lock()
	defer f.Unlock()
	name := cluster.Name
	if targetInformer, found := f.targetInformers[name]; found {
		close(targetInformer.stopChan)
	}
	delete(f.targetInformers, name)
}

// Returns a store created over all stores from target informers.
func (f *federatedInformerImpl) GetTargetStore() FederatedReadOnlyStore {
	return &federatedStoreImpl{
		federatedInformer: f,
	}
}

// Returns all items in the store.
func (fs *federatedStoreImpl) List() ([]FederatedObject, error) {
	fs.federatedInformer.Lock()
	defer fs.federatedInformer.Unlock()

	result := make([]FederatedObject, 0)
	for clusterName, targetInformer := range fs.federatedInformer.targetInformers {
		for _, value := range targetInformer.store.List() {
			result = append(result, FederatedObject{ClusterName: clusterName, Object: value})
		}
	}
	return result, nil
}

// Returns all items in the given cluster.
func (fs *federatedStoreImpl) ListFromCluster(clusterName string) ([]interface{}, error) {
	fs.federatedInformer.Lock()
	defer fs.federatedInformer.Unlock()

	result := make([]interface{}, 0)
	if targetInformer, found := fs.federatedInformer.targetInformers[clusterName]; found {
		values := targetInformer.store.List()
		result = append(result, values...)
	}
	return result, nil
}

// GetByKey returns the item stored under the given key in the specified cluster (if exist).
func (fs *federatedStoreImpl) GetByKey(clusterName string, key string) (interface{}, bool, error) {
	fs.federatedInformer.Lock()
	defer fs.federatedInformer.Unlock()
	if targetInformer, found := fs.federatedInformer.targetInformers[clusterName]; found {
		return targetInformer.store.GetByKey(key)
	}
	return nil, false, nil
}

// Returns the items stored under the given key in all clusters.
func (fs *federatedStoreImpl) GetFromAllClusters(key string) ([]FederatedObject, error) {
	fs.federatedInformer.Lock()
	defer fs.federatedInformer.Unlock()

	result := make([]FederatedObject, 0)
	for clusterName, targetInformer := range fs.federatedInformer.targetInformers {
		value, exist, err := targetInformer.store.GetByKey(key)
		if err != nil {
			return nil, err
		}
		if exist {
			result = append(result, FederatedObject{ClusterName: clusterName, Object: value})
		}
	}
	return result, nil
}

// GetKeyFor returns the key under which the item would be put in the store.
func (fs *federatedStoreImpl) GetKeyFor(item interface{}) string {
	// TODO: support other keying functions.
	key, _ := cache.DeletionHandlingMetaNamespaceKeyFunc(item)
	return key
}

// Checks whether stores for all clusters form the lists (and only these) are there and
// are synced.
func (fs *federatedStoreImpl) ClustersSynced(clusters []*federationapi.Cluster) bool {

	// Get the list of informers to check under a lock and check it outside.
	okSoFar, informersToCheck := func() (bool, []informer) {
		fs.federatedInformer.Lock()
		defer fs.federatedInformer.Unlock()

		if len(fs.federatedInformer.targetInformers) != len(clusters) {
			return false, []informer{}
		}
		informersToCheck := make([]informer, 0, len(clusters))
		for _, cluster := range clusters {
			if targetInformer, found := fs.federatedInformer.targetInformers[cluster.Name]; found {
				informersToCheck = append(informersToCheck, targetInformer)
			} else {
				return false, []informer{}
			}
		}
		return true, informersToCheck
	}()

	if !okSoFar {
		return false
	}
	for _, informerToCheck := range informersToCheck {
		if !informerToCheck.controller.HasSynced() {
			return false
		}
	}
	return true
}
