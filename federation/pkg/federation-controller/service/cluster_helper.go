/*
Copyright 2016 The Kubernetes Authors All rights reserved.

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

package service

import (
	"sync"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/v1"
	cache "k8s.io/kubernetes/pkg/client/cache"
	clientset "k8s.io/kubernetes/pkg/client/clientset_generated/release_1_3"
	"k8s.io/kubernetes/pkg/controller/framework"
	pkg_runtime "k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/federation/apis/federation/v1alpha1"
	"k8s.io/kubernetes/pkg/client/unversioned/clientcmd"
	"k8s.io/kubernetes/pkg/client/restclient"
	"k8s.io/kubernetes/pkg/util/workqueue"
	"k8s.io/kubernetes/pkg/util/wait"
	"k8s.io/kubernetes/pkg/watch"

	"github.com/golang/glog"
)

type clusterCache struct {
	clientset          *clientset.Clientset
	// A store of services, populated by the serviceController
	serviceStore       cache.StoreToServiceLister
	// Watches changes to all services
	serviceController  *framework.Controller
	// A store of endpoint, populated by the serviceController
	endpointStore      cache.StoreToEndpointsLister
	// Watches changes to all endpoints
	endpointController *framework.Controller
	// services that need to be synced
	serviceQueue       *workqueue.Type
	// endpoints that need to be synced
	endpointQueue      *workqueue.Type
}

type clusterClientCache struct {
	rwlock    sync.Mutex // protects serviceMap
	clientMap map[string]*clusterCache
}

func (cc *clusterClientCache) startClusterLW(clientset *clientset.Clientset, clusterName string) {
	cachedClusterClient := clusterCache {
		clientset: clientset,
		serviceQueue: workqueue.New(),
		endpointQueue: workqueue.New(),
	}
	cachedClusterClient.endpointStore.Store, cachedClusterClient.endpointController = framework.NewInformer(
		&cache.ListWatch{
			ListFunc: func(options api.ListOptions) (pkg_runtime.Object, error) {
				return clientset.Core().Endpoints(api.NamespaceAll).List(options)
			},
			WatchFunc: func(options api.ListOptions) (watch.Interface, error) {
				return clientset.Core().Endpoints(api.NamespaceAll).Watch(options)
			},
		},
		&v1.Endpoints{},
		serviceSyncPeriod,
		framework.ResourceEventHandlerFuncs{
			AddFunc: func(obj interface{}) {
				cc.enqueueEndpoint(obj, clusterName)
			},
			UpdateFunc: func(old, cur interface{}) {
				cc.enqueueEndpoint(cur, clusterName)
			},
			DeleteFunc: func(obj interface{}) {
				cc.enqueueEndpoint(obj, clusterName)
			},
		},
	)

	cachedClusterClient.serviceStore.Store, cachedClusterClient.serviceController = framework.NewInformer(
		&cache.ListWatch{
			ListFunc: func(options api.ListOptions) (pkg_runtime.Object, error) {
				return clientset.Core().Services(api.NamespaceAll).List(options)
			},
			WatchFunc: func(options api.ListOptions) (watch.Interface, error) {
				return clientset.Core().Services(api.NamespaceAll).Watch(options)
			},
		},
		&v1.Service{},
		serviceSyncPeriod,
		framework.ResourceEventHandlerFuncs{
			AddFunc: func(obj interface{}) {
				cc.enqueueService(obj, clusterName)
			},
			UpdateFunc: func(old, cur interface{}) {
				cc.enqueueService(cur, clusterName)
			},
			DeleteFunc: func(obj interface{}) {
				cc.enqueueService(obj, clusterName)
			},
		},
	)

	cc.clientMap[clusterName] = &cachedClusterClient
	glog.V(2).Infof("Start watching services on cluster %s", clusterName)
	go cachedClusterClient.serviceController.Run(wait.NeverStop)
	glog.V(2).Infof("Start watching endpoints on cluster %s", clusterName)
	go cachedClusterClient.endpointController.Run(wait.NeverStop)
}

//TODO: copied from cluster controller, to make this as common function in pass 2
// delFromClusterSet delete a cluster from clusterSet and
// delete the corresponding restclient from the map clusterKubeClientMap
func (cc *clusterClientCache) delFromClusterSet(obj interface{}) {
	cluster, ok := obj.(*v1alpha1.Cluster)
	cc.rwlock.Lock()
	defer cc.rwlock.Unlock()
	if ok {
		delete(cc.clientMap, cluster.Name)
	} else {
		tombstone, ok := obj.(cache.DeletedFinalStateUnknown)
		if !ok {
			glog.Infof("object contained wasn't a cluster or a deleted key: %+v", obj)
			return
		}
		glog.Infof("Found tombstone for %v", obj)
		delete(cc.clientMap, tombstone.Key)
	}
}

// addToClusterSet insert the new cluster to clusterSet and create a corresponding
// restclient to map clusterKubeClientMap
func (cc *clusterClientCache) addToClientMap(obj interface{}) {
	cluster := obj.(*v1alpha1.Cluster)
	pred := getClusterConditionPredicate()
	if pred(*cluster) {
	//if validateCluster(*cluster) {
		cc.rwlock.Lock()
		defer cc.rwlock.Unlock()
		//create the restclient of cluster
		clientset, err := newClusterClientset(cluster)
		if err != nil || clientset == nil {
			glog.Errorf("Failed to create corresponding restclient of kubernetes cluster: %v", err)
		}
		cc.startClusterLW(clientset, cluster.Name)
	}
	// keep thing going if the cluster status is updated to not ready
}

func newClusterClientset(c *v1alpha1.Cluster) (*clientset.Clientset, error) {
	clusterConfig, err := clientcmd.BuildConfigFromFlags(c.Spec.ServerAddressByClientCIDRs[0].ServerAddress, "")
	if err != nil {
		return nil, err
	}
	clusterConfig.QPS = KubeAPIQPS
	clusterConfig.Burst = KubeAPIBurst
	clientset := clientset.NewForConfigOrDie(restclient.AddUserAgent(clusterConfig, UserAgentName))
	return clientset, nil
}
