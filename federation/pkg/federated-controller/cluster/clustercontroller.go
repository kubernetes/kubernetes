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

package cluster

import (
	"fmt"
	"github.com/golang/glog"
	"k8s.io/kubernetes/federation/apis/federation"
	federationcache "k8s.io/kubernetes/federation/client/cache"
	federationclientset "k8s.io/kubernetes/federation/client/clientset_generated/internalclientset"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/meta"
	"k8s.io/kubernetes/pkg/apis/extensions"
	"k8s.io/kubernetes/pkg/client/cache"
	clientset "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset"
	"k8s.io/kubernetes/pkg/controller"
	"k8s.io/kubernetes/pkg/controller/framework"
	"k8s.io/kubernetes/pkg/runtime"
	utilruntime "k8s.io/kubernetes/pkg/util/runtime"
	"k8s.io/kubernetes/pkg/util/sets"
	"k8s.io/kubernetes/pkg/util/wait"
	"k8s.io/kubernetes/pkg/util/workqueue"
	"k8s.io/kubernetes/pkg/watch"
	"strings"
	"time"
)

const (
	AnnotationKeyOfTargetCluster        = "kubernetes.io/target-cluster"
	AnnotationKeyOfFederationReplicaSet = "kubernetes.io/created-by"
)

type ClusterController struct {
	knownClusterSet sets.String

	//federationClient used to operate cluster and subrs
	federationClient federationclientset.Interface

	//client used to operate rs
	client clientset.Interface

	// To allow injection of syncSubRC for testing.
	syncHandler func(subRcKey string) error

	//clusterMonitorPeriod is the period for updating status of cluster
	clusterMonitorPeriod time.Duration
	//clusterKubeClientMap is a mapping of clusterName and restclient
	clusterKubeClientMap map[string]ClusterClient

	// subRc framework and store
	subReplicaSetController *framework.Controller
	subReplicaSetStore      federationcache.StoreToSubReplicaSetLister

	// cluster framework and store
	clusterController *framework.Controller
	clusterStore      federationcache.StoreToClusterLister

	// UberRc framework and store
	replicaSetStore      cache.StoreToReplicationControllerLister
	replicaSetController *framework.Controller

	// subRC that have been queued up for processing by workers
	queue *workqueue.Type
}

// NewclusterController returns a new cluster controller
func NewclusterController(client clientset.Interface, federationClient federationclientset.Interface, clusterMonitorPeriod time.Duration) *ClusterController {
	cc := &ClusterController{
		knownClusterSet:      make(sets.String),
		federationClient:     federationClient,
		client:               client,
		clusterMonitorPeriod: clusterMonitorPeriod,
		clusterKubeClientMap: make(map[string]ClusterClient),
		queue:                workqueue.New(),
	}

	cc.subReplicaSetStore.Store, cc.subReplicaSetController = framework.NewInformer(
		&cache.ListWatch{
			ListFunc: func(options api.ListOptions) (runtime.Object, error) {
				return cc.federationClient.Federation().SubReplicaSets(api.NamespaceAll).List(options)
			},
			WatchFunc: func(options api.ListOptions) (watch.Interface, error) {
				return cc.federationClient.Federation().SubReplicaSets(api.NamespaceAll).Watch(options)
			},
		},
		&federation.SubReplicaSet{},
		controller.NoResyncPeriodFunc(),
		framework.ResourceEventHandlerFuncs{
			AddFunc: func(obj interface{}) {
				subRc := obj.(*federation.SubReplicaSet)
				cc.enqueueSubRc(subRc)
			},
			UpdateFunc: func(oldObj, newObj interface{}) {
				subRc := newObj.(*federation.SubReplicaSet)
				cc.enqueueSubRc(subRc)
			},
		},
	)

	cc.clusterStore.Store, cc.clusterController = framework.NewInformer(
		&cache.ListWatch{
			ListFunc: func(options api.ListOptions) (runtime.Object, error) {
				return cc.federationClient.Federation().Clusters().List(options)
			},
			WatchFunc: func(options api.ListOptions) (watch.Interface, error) {
				return cc.federationClient.Federation().Clusters().Watch(options)
			},
		},
		&federation.Cluster{},
		controller.NoResyncPeriodFunc(),
		framework.ResourceEventHandlerFuncs{
			DeleteFunc: cc.delFromClusterSet,
			AddFunc:    cc.addToClusterSet,
		},
	)

	cc.replicaSetStore.Store, cc.replicaSetController = framework.NewInformer(
		&cache.ListWatch{
			ListFunc: func(options api.ListOptions) (runtime.Object, error) {
				return cc.client.Extensions().ReplicaSets(api.NamespaceAll).List(options)
			},
			WatchFunc: func(options api.ListOptions) (watch.Interface, error) {
				return cc.client.Extensions().ReplicaSets(api.NamespaceAll).Watch(options)
			},
		},
		&api.ReplicationController{},
		controller.NoResyncPeriodFunc(),
		framework.ResourceEventHandlerFuncs{
			DeleteFunc: cc.deleteSubRs,
		},
	)
	cc.syncHandler = cc.syncSubReplicaSet
	return cc
}

//delFromClusterSet delete a cluster from clusterSet and
//delete the corresponding restclient from the map clusterKubeClientMap
func (cc *ClusterController) delFromClusterSet(obj interface{}) {
	cluster := obj.(*federation.Cluster)
	cc.knownClusterSet.Delete(cluster.Name)
	delete(cc.clusterKubeClientMap, cluster.Name)
}

//addToClusterSet insert the new cluster to clusterSet and creat a corresponding
//restclient to map clusterKubeClientMap
func (cc *ClusterController) addToClusterSet(obj interface{}) {
	cluster := obj.(*federation.Cluster)
	cc.knownClusterSet.Insert(cluster.Name)
	//create the restclient of cluster
	restClient, err := NewClusterClientSet(cluster)
	if err != nil {
		glog.Errorf("Failed to create corresponding restclient of kubernetes cluster: %v", err)
	}
	cc.clusterKubeClientMap[cluster.Name] = *restClient
}

// Run begins watching and syncing.
func (cc *ClusterController) Run(workers int, stopCh <-chan struct{}) {
	defer utilruntime.HandleCrash()
	go cc.clusterController.Run(wait.NeverStop)
	go cc.replicaSetController.Run(wait.NeverStop)
	go cc.subReplicaSetController.Run(wait.NeverStop)
	// monitor cluster status periodically, in phase 1 we just get the health state from "/healthz"
	go wait.Until(func() {
		if err := cc.UpdateClusterStatus(); err != nil {
			glog.Errorf("Error monitoring cluster status: %v", err)
		}
	}, cc.clusterMonitorPeriod, wait.NeverStop)
	for i := 0; i < workers; i++ {
		go wait.Until(cc.worker, time.Second, stopCh)
	}
	<-stopCh
	glog.Infof("Shutting down ClusterController")
	cc.queue.ShutDown()

}

// enqueueSubRc adds an object to the controller work queue
// obj could be an *federation.SubReplicaSet, or a DeletionFinalStateUnknown item.
func (cc *ClusterController) enqueueSubRc(obj interface{}) {
	key, err := controller.KeyFunc(obj)
	if err != nil {
		glog.Errorf("Couldn't get key for object %+v: %v", obj, err)
		return
	}
	cc.queue.Add(key)
}

func (cc *ClusterController) worker() {
	for {
		func() {
			key, quit := cc.queue.Get()
			if quit {
				return
			}
			defer cc.queue.Done(key)
			err := cc.syncHandler(key.(string))
			if err != nil {
				glog.Errorf("Error syncing cluster controller: %v", err)
			}
		}()
	}
}

// syncSubReplicaSet will sync the subrc with the given key,
func (cc *ClusterController) syncSubReplicaSet(key string) error {
	startTime := time.Now()
	defer func() {
		glog.V(4).Infof("Finished syncing controller %q (%v)", key, time.Now().Sub(startTime))
	}()
	obj, exists, err := cc.subReplicaSetStore.Store.GetByKey(key)
	if !exists {
		glog.Infof("sub replicaset: %v has been deleted", key)
		return nil
	}
	if err != nil {
		glog.Infof("Unable to retrieve sub replicaset %v from store: %v", key, err)
		cc.queue.Add(key)
		return err
	}
	subRs := obj.(*federation.SubReplicaSet)
	err = cc.manageSubReplicaSet(subRs)
	if err != nil {
		glog.Infof("Unable to manage subRs in kubernetes cluster: %v", key, err)
		cc.queue.Add(key)
		return err
	}
	return nil
}

//getBindingClusterOfSubRS get the target cluster(scheduled by federation scheduler) of subRS
//return the targetCluster name
func (cc *ClusterController) getBindingClusterOfSubRS(subRs *federation.SubReplicaSet) (string, error) {
	accessor, err := meta.Accessor(subRs)
	if err != nil {
		return "", err
	}
	annotations := accessor.GetAnnotations()
	if annotations == nil {
		return "", fmt.Errorf("Failed to get target cluster from the annotation of subreplicaset")
	}
	targetCluster, found := annotations[AnnotationKeyOfTargetCluster]
	if !found {
		return "", fmt.Errorf("Failed to get target cluster from the annotation of subreplicaset")
	}
	return targetCluster, nil
}

//getFederateRsCreateBy get the federation ReplicaSet created by of subRS
//return the replica set name
func (cc *ClusterController) getFederateRsCreateBy(subRs *federation.SubReplicaSet) (string, error) {
	accessor, err := meta.Accessor(subRs)
	if err != nil {
		return "", err
	}
	annotations := accessor.GetAnnotations()
	if annotations == nil {
		return "", fmt.Errorf("Failed to get Federate Rs Create By from the annotation of subreplicaset")
	}
	rsCreateBy, found := annotations[AnnotationKeyOfFederationReplicaSet]
	if !found {
		return "", fmt.Errorf("Failed to get Federate Rs Create By from the annotation of subreplicaset")
	}
	return rsCreateBy, nil
}

// manageSubReplicaSet will sync the sub replicaset with the given key,and then create
// or update replicaset to kubernetes cluster
func (cc *ClusterController) manageSubReplicaSet(subRs *federation.SubReplicaSet) error {

	targetClusterName, err := cc.getBindingClusterOfSubRS(subRs)
	if targetClusterName == "" || err != nil {
		glog.Infof("Failed to get target cluster of SubRS: %v", err)
		return err
	}

	clusterClient, found := cc.clusterKubeClientMap[targetClusterName]
	if !found {
		glog.Infof("Failed to get restclient of target cluster")
		return fmt.Errorf("Failed to get restclient of target cluster")
	}
	// check the sub replicaset already exists in kubernetes cluster or not
	replicaSet, err := clusterClient.GetReplicaSetFromCluster(subRs.Name, subRs.Namespace)
	if err != nil {
		glog.Infof("Failed to get RC in kubernetes cluster: %v", err)
		return err
	}

	rs := extensions.ReplicaSet(*subRs)
	//if not exist, means that this sub replicaset need to be created
	if replicaSet == nil {
		// create the sub replicaset to kubernetes cluster
		replicaSet, err := clusterClient.CreateReplicaSetToCluster(&rs)
		if err != nil || replicaSet == nil {
			glog.Infof("Failed to create sub replicaset in kubernetes cluster: %v", err)
			return err
		}
	}
	// if exists, then update it
	replicaSet, err = clusterClient.UpdateReplicaSetToCluster(&rs)
	if err != nil || replicaSet == nil {
		glog.Infof("Failed to update sub replicaset in kubernetes cluster: %v", err)
		return err
	}
	return nil
}

func (cc *ClusterController) GetClusterStatus(cluster *federation.Cluster) (*federation.ClusterStatus, error) {
	// just get the status of cluster, by getting the version of kubernetes cluster api-server
	var clusterStatus federation.ClusterStatus
	clusterClient, found := cc.clusterKubeClientMap[cluster.Name]
	if !found {
		glog.Infof("Failed to get restclient of target cluster")
		return nil, fmt.Errorf("Failed to get restclient of target cluster")
	}
	clusterStatus.Phase = clusterClient.GetClusterHealthStatus()
	return &clusterStatus, nil
}

// monitorClusterStatus checks cluster status and get the metrics from cluster's restapi and RC state
func (cc *ClusterController) UpdateClusterStatus() error {
	clusters, err := cc.federationClient.Federation().Clusters().List(api.ListOptions{})
	if err != nil {
		return err
	}
	for _, cluster := range clusters.Items {
		if !cc.knownClusterSet.Has(cluster.Name) {
			glog.V(1).Infof("ClusterController observed a new cluster: %#v", cluster)
			cc.knownClusterSet.Insert(cluster.Name)
		}
	}

	// If there's a difference between lengths of known clusters and observed clusters
	// we must have removed some clusters, and evict the subRs belong to the cluster
	if len(cc.knownClusterSet) != len(clusters.Items) {
		observedSet := make(sets.String)
		for _, cluster := range clusters.Items {
			observedSet.Insert(cluster.Name)
		}
		deleted := cc.knownClusterSet.Difference(observedSet)
		for clusterName := range deleted {
			glog.V(1).Infof("ClusterController observed a Cluster deletion: %v", clusterName)
			//TODO: evict the subRS
			cc.knownClusterSet.Delete(clusterName)
		}
	}
	for _, cluster := range clusters.Items {
		ClusterStatus, err := cc.GetClusterStatus(&cluster)
		if err == nil {
			continue
		}
		cluster.Status.Phase = ClusterStatus.Phase
		_, err = cc.federationClient.Federation().Clusters().Update(&cluster)
	}
	return nil
}

func (cc *ClusterController) deleteSubRs(cur interface{}) {
	rs := cur.(*extensions.ReplicaSet)
	// get the corresponing subrs from the cache
	subRSList, err := cc.federationClient.Federation().SubReplicaSets(api.NamespaceAll).List(api.ListOptions{})
	if err != nil || len(subRSList.Items) == 0 {
		glog.Infof("Couldn't get subRS to delete : %+v", cur)
		return
	}

	// get the related subRS created by the replicaset
	for _, subRs := range subRSList.Items {
		name, err := cc.getFederateRsCreateBy(&subRs)
		if err != nil || !strings.EqualFold(rs.Name, name){
			continue
		}
		targetClusterName, err := cc.getBindingClusterOfSubRS(&subRs)
		if targetClusterName == "" || err != nil {
			continue
		}

		clusterClient, found := cc.clusterKubeClientMap[targetClusterName]
		if !found {
			continue
		}
		rs := extensions.ReplicaSet(subRs)
		err = clusterClient.DeleteReplicasetFromCluster(&rs)
		if err != nil {
			return
		}
	}
	return
}
