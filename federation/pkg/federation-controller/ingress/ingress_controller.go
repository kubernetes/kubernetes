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

package ingress

import (
	"fmt"
	"reflect"
	"sync"
	"time"

	federation_api "k8s.io/kubernetes/federation/apis/federation/v1beta1"
	federation_release_1_4 "k8s.io/kubernetes/federation/client/clientset_generated/federation_release_1_4"
	"k8s.io/kubernetes/federation/pkg/federation-controller/util"
	"k8s.io/kubernetes/federation/pkg/federation-controller/util/eventsink"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/v1"
	extensions_v1beta1 "k8s.io/kubernetes/pkg/apis/extensions/v1beta1"
	"k8s.io/kubernetes/pkg/client/cache"
	kube_release_1_4 "k8s.io/kubernetes/pkg/client/clientset_generated/release_1_4"
	"k8s.io/kubernetes/pkg/client/record"
	"k8s.io/kubernetes/pkg/controller"
	"k8s.io/kubernetes/pkg/controller/framework"
	"k8s.io/kubernetes/pkg/conversion"
	pkg_runtime "k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/types"
	"k8s.io/kubernetes/pkg/util/flowcontrol"
	"k8s.io/kubernetes/pkg/watch"

	"github.com/golang/glog"
)

const (
	allClustersKey        = "ALL_CLUSTERS"
	staticIPAnnotationKey = "ingress.kubernetes.io/static-ip" // TODO: Get this directly from the Kubernetes Ingress Controller constant
	uidAnnotationKey      = "ingress.kubernetes.io/uid"       // The annotation on federation clusters, where we store the ingress UID
	// Name of the config-map and key the ingress controller stores its uid in.
	uidConfigMapName      = "ingress-uid"
	uidConfigMapNamespace = "kube-system"
	uidKey                = "uid" // TODO: Get this directly from the Kubernetes Ingress Controller constant

	DEBUG_LOG = 1 // Hack to get debug messages out of go test
)

type IngressController struct {
	sync.Mutex // Lock used for leader election
	// For triggering single ingress reconcilation. This is used when there is an
	// add/update/delete operation on an ingress in either federated API server or
	// in some member of the federation.
	ingressDeliverer *util.DelayingDeliverer

	// For triggering reconcilation of cluster ingress controller configmap and
	// all ingresses. This is used when a new cluster becomes available.
	clusterDeliverer *util.DelayingDeliverer

	// For triggering reconcilation of cluster ingress controller configmap.
	// This is used when a configmap is updated in the cluster.
	configMapDeliverer *util.DelayingDeliverer

	// Contains ingresses present in members of federation.
	ingressFederatedInformer util.FederatedInformer
	// Contains ingress controller configmaps present in members of federation.
	configMapFederatedInformer util.FederatedInformer
	// For updating ingresses in members of federation.
	federatedIngressUpdater util.FederatedUpdater
	// For updating configmaps in members of federation.
	federatedConfigMapUpdater util.FederatedUpdater
	// Definitions of ingresses that should be federated.
	ingressInformerStore cache.Store
	// Informer controller for ingresses that should be federated.
	ingressInformerController framework.ControllerInterface

	// Client to federated api server.
	federatedApiClient federation_release_1_4.Interface

	// Backoff manager for ingresses
	ingressBackoff *flowcontrol.Backoff
	// Backoff manager for configmaps
	configMapBackoff *flowcontrol.Backoff

	// For events
	eventRecorder record.EventRecorder

	ingressReviewDelay    time.Duration
	configMapReviewDelay  time.Duration
	clusterAvailableDelay time.Duration
	smallDelay            time.Duration
	updateTimeout         time.Duration
}

// NewIngressController returns a new ingress controller
func NewIngressController(client federation_release_1_4.Interface) *IngressController {
	glog.V(4).Infof("->NewIngressController V(4)")
	broadcaster := record.NewBroadcaster()
	broadcaster.StartRecordingToSink(eventsink.NewFederatedEventSink(client))
	recorder := broadcaster.NewRecorder(api.EventSource{Component: "federated-ingress-controller"})
	ic := &IngressController{
		federatedApiClient:    client,
		ingressReviewDelay:    time.Second * 10,
		configMapReviewDelay:  time.Second * 10,
		clusterAvailableDelay: time.Second * 20,
		smallDelay:            time.Second * 3,
		updateTimeout:         time.Second * 30,
		ingressBackoff:        flowcontrol.NewBackOff(5*time.Second, time.Minute),
		eventRecorder:         recorder,
		configMapBackoff:      flowcontrol.NewBackOff(5*time.Second, time.Minute),
	}

	// Build deliverers for triggering reconcilations.
	ic.ingressDeliverer = util.NewDelayingDeliverer()
	ic.clusterDeliverer = util.NewDelayingDeliverer()
	ic.configMapDeliverer = util.NewDelayingDeliverer()

	// Start informer in federated API servers on ingresses that should be federated.
	ic.ingressInformerStore, ic.ingressInformerController = framework.NewInformer(
		&cache.ListWatch{
			ListFunc: func(options api.ListOptions) (pkg_runtime.Object, error) {
				return client.Extensions().Ingresses(api.NamespaceAll).List(options)
			},
			WatchFunc: func(options api.ListOptions) (watch.Interface, error) {
				return client.Extensions().Ingresses(api.NamespaceAll).Watch(options)
			},
		},
		&extensions_v1beta1.Ingress{},
		controller.NoResyncPeriodFunc(),
		util.NewTriggerOnAllChanges(
			func(obj pkg_runtime.Object) {
				ic.deliverIngressObj(obj, 0, false)
			},
		))

	// Federated informer on ingresses in members of federation.
	ic.ingressFederatedInformer = util.NewFederatedInformer(
		client,
		func(cluster *federation_api.Cluster, targetClient kube_release_1_4.Interface) (cache.Store, framework.ControllerInterface) {
			return framework.NewInformer(
				&cache.ListWatch{
					ListFunc: func(options api.ListOptions) (pkg_runtime.Object, error) {
						return targetClient.Extensions().Ingresses(api.NamespaceAll).List(options)
					},
					WatchFunc: func(options api.ListOptions) (watch.Interface, error) {
						return targetClient.Extensions().Ingresses(api.NamespaceAll).Watch(options)
					},
				},
				&extensions_v1beta1.Ingress{},
				controller.NoResyncPeriodFunc(),
				// Trigger reconcilation whenever something in federated cluster is changed. In most cases it
				// would be just confirmation that some ingress operation suceeded.
				util.NewTriggerOnAllChanges(
					func(obj pkg_runtime.Object) {
						ic.deliverIngressObj(obj, ic.ingressReviewDelay, false)
					},
				))
		},

		&util.ClusterLifecycleHandlerFuncs{
			ClusterAvailable: func(cluster *federation_api.Cluster) {
				// When new cluster becomes available process all the ingresses again, and configure it's ingress controller's configmap with the correct UID
				ic.clusterDeliverer.DeliverAfter(cluster.Name, cluster, ic.clusterAvailableDelay)
			},
		},
	)

	// Federated informer on configmaps for ingress controllers in members of the federation.
	ic.configMapFederatedInformer = util.NewFederatedInformer(
		client,
		func(cluster *federation_api.Cluster, targetClient kube_release_1_4.Interface) (cache.Store, framework.ControllerInterface) {
			glog.V(4).Infof("Returning new informer for cluster %q", cluster.Name)
			return framework.NewInformer(
				&cache.ListWatch{
					ListFunc: func(options api.ListOptions) (pkg_runtime.Object, error) {
						if targetClient == nil {
							glog.Errorf("targetClient is nil")
						}
						return targetClient.Core().ConfigMaps(api.NamespaceAll).List(options) // we only want to list one by name - unfortunately Kubernetes don't have a selector for that.
					},
					WatchFunc: func(options api.ListOptions) (watch.Interface, error) {
						return targetClient.Core().ConfigMaps(api.NamespaceAll).Watch(options) // as above
					},
				},
				&v1.ConfigMap{},
				controller.NoResyncPeriodFunc(),
				// Trigger reconcilation whenever the ingress controller's configmap in a federated cluster is changed. In most cases it
				// would be just confirmation that the configmap for the ingress controller is correct.
				util.NewTriggerOnAllChanges(
					func(obj pkg_runtime.Object) {
						ic.deliverConfigMapObj(cluster.Name, obj, ic.configMapReviewDelay, false)
						// TODO: Remove ic.reconcileConfigMapForCluster(cluster.Name)
					},
				))
		},

		&util.ClusterLifecycleHandlerFuncs{
			ClusterAvailable: func(cluster *federation_api.Cluster) {
				ic.clusterDeliverer.DeliverAfter(cluster.Name, cluster, ic.clusterAvailableDelay)
			},
		},
	)

	// Federated ingress updater along with Create/Update/Delete operations.
	ic.federatedIngressUpdater = util.NewFederatedUpdater(ic.ingressFederatedInformer,
		func(client kube_release_1_4.Interface, obj pkg_runtime.Object) error {
			ingress := obj.(*extensions_v1beta1.Ingress)
			glog.V(DEBUG_LOG).Infof("Attempting to create Ingress: %v", ingress)
			_, err := client.Extensions().Ingresses(ingress.Namespace).Create(ingress)
			return err
		},
		func(client kube_release_1_4.Interface, obj pkg_runtime.Object) error {
			ingress := obj.(*extensions_v1beta1.Ingress)
			glog.V(4).Infof("Attempting to update Ingress: %v", ingress)
			_, err := client.Extensions().Ingresses(ingress.Namespace).Update(ingress)
			if err != nil {
				glog.V(4).Infof("Failed to update Ingress: %v", err)
			} else {
				glog.V(4).Infof("Successfully updated Ingress: %q", types.NamespacedName{Name: ingress.Name, Namespace: ingress.Namespace})
			}
			return err
		},
		func(client kube_release_1_4.Interface, obj pkg_runtime.Object) error {
			ingress := obj.(*extensions_v1beta1.Ingress)
			glog.V(4).Infof("Attempting to delete Ingress: %v", ingress)
			err := client.Extensions().Ingresses(ingress.Namespace).Delete(ingress.Name, &api.DeleteOptions{})
			return err
		})

	// Federated configmap updater along with Create/Update/Delete operations.  Only Update should ever be called.
	ic.federatedConfigMapUpdater = util.NewFederatedUpdater(ic.configMapFederatedInformer,
		func(client kube_release_1_4.Interface, obj pkg_runtime.Object) error {
			configMap := obj.(*v1.ConfigMap)
			configMapName := types.NamespacedName{Name: configMap.Name, Namespace: configMap.Namespace}
			glog.Errorf("Internal error: Incorrectly attempting to create ConfigMap: %q", configMapName)
			_, err := client.Core().ConfigMaps(configMap.Namespace).Create(configMap)
			return err // TODO fmt.Errorf("Internal error: Incorrectly attempting to create ConfigMap: %q", configMapName)
		},
		func(client kube_release_1_4.Interface, obj pkg_runtime.Object) error {
			configMap := obj.(*v1.ConfigMap)
			configMapName := types.NamespacedName{Name: configMap.Name, Namespace: configMap.Namespace}
			glog.V(4).Infof("Attempting to update ConfigMap: %v", configMap)
			_, err := client.Core().ConfigMaps(configMap.Namespace).Update(configMap)
			if err == nil {
				glog.V(4).Infof("Successfully updated ConfigMap %q", configMapName)
			} else {
				glog.V(4).Infof("Failed to updated ConfigMap %q: %v", configMapName, err)
			}
			return err
		},
		func(client kube_release_1_4.Interface, obj pkg_runtime.Object) error {
			configMap := obj.(*v1.ConfigMap)
			configMapName := types.NamespacedName{Name: configMap.Name, Namespace: configMap.Namespace}
			glog.Errorf("Internal error: Incorrectly attempting to delete ConfigMap: %q", configMapName)
			err := client.Core().ConfigMaps(configMap.Namespace).Delete(configMap.Name, &api.DeleteOptions{})
			return err // TODO fmt.Errorf("Internal error: Incorrectly attempting to delete ConfigMap: %q", configMapName)
		})
	return ic
}

func (ic *IngressController) Run(stopChan <-chan struct{}) {
	glog.Infof("Starting Ingress Controller")
	go ic.ingressInformerController.Run(stopChan)
	glog.Infof("... Starting Ingress Federated Informer")
	ic.ingressFederatedInformer.Start()
	glog.Infof("... Starting ConfigMap Federated Informer")
	ic.configMapFederatedInformer.Start()
	go func() {
		<-stopChan
		glog.Infof("Stopping Ingress Federated Informer")
		ic.ingressFederatedInformer.Stop()
		glog.Infof("Stopping ConfigMap Federated Informer")
		ic.configMapFederatedInformer.Stop()
	}()
	ic.ingressDeliverer.StartWithHandler(func(item *util.DelayingDelivererItem) {
		ingress := item.Value.(types.NamespacedName)
		glog.V(4).Infof("Ingress change delivered, reconciling: %v", ingress)
		ic.reconcileIngress(ingress)
	})
	ic.clusterDeliverer.StartWithHandler(func(item *util.DelayingDelivererItem) {
		clusterName := item.Key
		if clusterName != allClustersKey {
			glog.V(4).Infof("Cluster change delivered for cluster %q, reconciling configmap and ingress for that cluster", clusterName)
		} else {
			glog.V(4).Infof("Cluster change delivered for all clusters, reconciling configmaps and ingresses for all clusters")
		}
		ic.reconcileIngressesOnClusterChange(clusterName)
		ic.reconcileConfigMapForCluster(clusterName)
	})
	ic.configMapDeliverer.StartWithHandler(func(item *util.DelayingDelivererItem) {
		clusterName := item.Key
		glog.V(4).Infof("ConfigMap change delivered for cluster %q, reconciling configmap for that cluster", clusterName)
		ic.reconcileConfigMapForCluster(clusterName)
	})
	go func() {
		select {
		case <-time.After(time.Minute):
			glog.V(4).Infof("Ingress controller is garbage collecting")
			ic.ingressBackoff.GC()
			ic.configMapBackoff.GC()
			glog.V(4).Infof("Ingress controller garbage collection complete")
		case <-stopChan:
			return
		}
	}()
}

func (ic *IngressController) deliverIngressObj(obj interface{}, delay time.Duration, failed bool) {
	ingress := obj.(*extensions_v1beta1.Ingress)
	ic.deliverIngress(types.NamespacedName{Namespace: ingress.Namespace, Name: ingress.Name}, delay, failed)
}

func (ic *IngressController) deliverIngress(ingress types.NamespacedName, delay time.Duration, failed bool) {
	glog.V(4).Infof("Delivering ingress: %s", ingress)
	key := ingress.String()
	if failed {
		ic.ingressBackoff.Next(key, time.Now())
		delay = delay + ic.ingressBackoff.Get(key)
	} else {
		ic.ingressBackoff.Reset(key)
	}
	ic.ingressDeliverer.DeliverAfter(key, ingress, delay)
}

func (ic *IngressController) deliverConfigMapObj(clusterName string, obj interface{}, delay time.Duration, failed bool) {
	configMap := obj.(*v1.ConfigMap)
	ic.deliverConfigMap(clusterName, types.NamespacedName{Namespace: configMap.Namespace, Name: configMap.Name}, delay, failed)
}

func (ic *IngressController) deliverConfigMap(cluster string, configMap types.NamespacedName, delay time.Duration, failed bool) {
	key := cluster
	if failed {
		ic.configMapBackoff.Next(key, time.Now())
		delay = delay + ic.configMapBackoff.Get(key)
	} else {
		ic.configMapBackoff.Reset(key)
	}
	glog.V(4).Infof("Delivering ConfigMap for cluster %q (delay %q): %s", cluster, delay, configMap)
	ic.configMapDeliverer.DeliverAfter(key, configMap, delay)
}

// Check whether all data stores are in sync. False is returned if any of the informer/stores is not yet
// synced with the coresponding api server.
func (ic *IngressController) isSynced() bool {
	if !ic.ingressFederatedInformer.ClustersSynced() {
		/* TODO Back to V(2) */ glog.Errorf("Cluster list not synced for ingress federated informer")
		return false
	}
	clusters, err := ic.ingressFederatedInformer.GetReadyClusters()
	if err != nil {
		glog.Errorf("Failed to get ready clusters for ingress federated informer: %v", err)
		return false
	}
	if !ic.ingressFederatedInformer.GetTargetStore().ClustersSynced(clusters) {
		/* TODO Back to V(2) */ glog.Errorf("Target store not synced for ingress federated informer")
		return false
	}
	if !ic.configMapFederatedInformer.ClustersSynced() {
		/* TODO Back to V(2) */ glog.Errorf("Cluster list not synced for config map federated informer")
		return false
	}
	clusters, err = ic.configMapFederatedInformer.GetReadyClusters()
	if err != nil {
		glog.Errorf("Failed to get ready clusters for configmap federated informer: %v", err)
		return false
	}
	if !ic.configMapFederatedInformer.GetTargetStore().ClustersSynced(clusters) {
		/* TODO Back to V(2) */ glog.Errorf("Target store not synced for configmap federated informer")
		return false
	}
	glog.V(4).Infof("Cluster list is synced")
	return true
}

// The function triggers reconcilation of all federated ingresses.  clusterName is the name of the cluster that changed
// but all ingresses in all clusters are reconciled
func (ic *IngressController) reconcileIngressesOnClusterChange(clusterName string) {
	glog.V(4).Infof("Reconciling ingresses on cluster change for cluster %q", clusterName)
	if !ic.isSynced() {
		glog.V(4).Infof("Not synced, will try again later to reconcile ingresses.")
		ic.clusterDeliverer.DeliverAfter(clusterName, nil, ic.clusterAvailableDelay)
	}
	ingressList := ic.ingressInformerStore.List()
	if len(ingressList) <= 0 {
		glog.V(4).Infof("No federated ingresses to reconcile.")
	}

	for _, obj := range ingressList {
		ingress := obj.(*extensions_v1beta1.Ingress)
		nsName := types.NamespacedName{Name: ingress.Name, Namespace: ingress.Namespace}
		glog.V(4).Infof("Delivering federated ingress %q for cluster %q", nsName, clusterName)
		ic.deliverIngress(nsName, ic.smallDelay, false)
	}
}

// HACK Because util.FederatedInformer().GetReadyCluster(name) doesn't work, while GetReadyClusters() does.
// Also easy to switch between using configMapFederatedInformer and ingressFederatedInformer for debugging.
func (ic *IngressController) getReadyCluster(clusterName string) (*federation_api.Cluster, error) {
	cluster, found, err := ic.configMapFederatedInformer.GetReadyCluster(clusterName)

	if !found || err != nil {
		// TODO Hack!  Check whether GetReadyClusters can find it! - Looks like a bug in GetReadyCluster
		clusters, err := ic.configMapFederatedInformer.GetReadyClusters()
		if err != nil {
			glog.Errorf("Whoooaah! Internal error.  Failed to get ready clusters: %v", err)
			return nil, err
		} else {
			for _, c := range clusters {
				if c.Name == clusterName {
					glog.V(4).Infof("Whooah! Internal error: Found cluster %q via GetReadyClusters() but not via GetReadyCluster()!!", clusterName)
					found = true
					cluster = c
				}
			}
			if !found {
				glog.Errorf("Internal error: Did not find cluster %q via GetReadyClusters() nor via GetReadyCluster()!!  Clusters: %v", clusterName, clusters)
				return nil, fmt.Errorf("Ready cluster %q not found", clusterName)
			} else {
				return cluster, nil
			}
		}
	}
	return cluster, nil
}

/*
  reconcileConfigMapForCluster ensures that the configmap for the ingress controller in the cluster has objectmeta.data.UID
  consistent with all the other clusters in the federation. If clusterName == allClustersKey, then all avaliable clusters
  configmaps are reconciled.
*/
func (ic *IngressController) reconcileConfigMapForCluster(clusterName string) {
	glog.V(4).Infof("Reconciling ConfigMap for cluster(s) %q", clusterName)

	if !ic.isSynced() {
		ic.configMapDeliverer.DeliverAfter(clusterName, nil, ic.clusterAvailableDelay)
		return
	}

	if clusterName == allClustersKey {
		clusters, err := ic.configMapFederatedInformer.GetReadyClusters()
		if err != nil {
			glog.Errorf("Failed to get ready clusters.  redelivering %q: %v", clusterName, err)
			ic.configMapDeliverer.DeliverAfter(clusterName, nil, ic.clusterAvailableDelay)
			return
		}
		for _, cluster := range clusters {
			glog.V(4).Infof("Delivering ConfigMap for cluster(s) %q", clusterName)
			ic.configMapDeliverer.DeliverAt(cluster.Name, nil, time.Now())
		}
		return
	} else {
		cluster, err := ic.getReadyCluster(clusterName)
		if err != nil {
			glog.Errorf("Internal error: Cluster %q queued for configmap reconciliation, but not found.  Will try again later: error = %v", clusterName, err)
			ic.configMapDeliverer.DeliverAfter(clusterName, nil, ic.clusterAvailableDelay)
			return
		}
		uidConfigMapNamespacedName := types.NamespacedName{Name: uidConfigMapName, Namespace: uidConfigMapNamespace}
		configMapObj, found, err := ic.configMapFederatedInformer.GetTargetStore().GetByKey(cluster.Name, uidConfigMapNamespacedName.String())
		if !found || err != nil {
			glog.Errorf("Failed to get ConfigMap %q for cluster %q.  Will try again later: %v", uidConfigMapNamespacedName, cluster.Name, err)
			ic.configMapDeliverer.DeliverAfter(clusterName, nil, ic.configMapReviewDelay)
			return
		}
		glog.V(4).Infof("Successfully got ConfigMap %q for cluster %q.", uidConfigMapNamespacedName, clusterName)
		configMap, ok := configMapObj.(*v1.ConfigMap)
		if !ok {
			glog.Errorf("Internal error: The object in the ConfigMap cache for cluster %q configmap %q is not a *ConfigMap", cluster.Name, uidConfigMapNamespacedName)
			return
		}
		ic.reconcileConfigMap(cluster, configMap)
		return
	}
}

/*
  reconcileConfigMap ensures that the configmap in the cluster has a UID
  consistent with the federation cluster's associated annotation.

  1. If the UID in the configmap differs from the UID stored in the cluster's annotation, the configmap is updated.
  2. If the UID annotation is missing from the cluster, the cluster's UID annotation is updated to be consistent
  with the master cluster.
  3. If there is no elected master cluster, this cluster attempts to elect itself as the master cluster.

  TODO: There is a potential race condition here.  Add a lock.  In the mean time periodic reconciliation sorts it out eventually anyway.

  In cases 2 and 3, the configmaps will be updated in the next cycle, triggered by the federation cluster update(s)

*/
func (ic *IngressController) reconcileConfigMap(cluster *federation_api.Cluster, configMap *v1.ConfigMap) {
	ic.Lock() // TODO: Reduce the scope of this master election lock.
	defer ic.Unlock()

	configMapNsName := types.NamespacedName{Name: configMap.Name, Namespace: configMap.Namespace}
	glog.V(4).Infof("Reconciling ConfigMap %q in cluster %q", configMapNsName, cluster.Name)

	clusterIngressUID, clusterIngressUIDExists := cluster.ObjectMeta.Annotations[uidAnnotationKey]
	configMapUID, ok := configMap.Data[uidKey]

	if !ok {
		glog.Errorf("Warning: ConfigMap %q in cluster %q does not contain data key %q.  Therefore it cannot become the master.", configMapNsName, cluster.Name, uidKey)
	}

	if !clusterIngressUIDExists || clusterIngressUID == "" {
		ic.updateClusterIngressUIDToMasters(cluster, configMapUID) // Second argument is the fallback, in case this is the only cluster, in which case it becomes the master
		return
	}
	if configMapUID != clusterIngressUID { // An update is required
		glog.V(4).Infof("Ingress UID update is required: configMapUID %q not equal to clusterIngressUID %q", configMapUID, clusterIngressUID)
		configMap.Data[uidKey] = clusterIngressUID
		operations := []util.FederatedOperation{{
			Type:        util.OperationTypeUpdate,
			Obj:         configMap,
			ClusterName: cluster.Name,
		}}
		glog.V(4).Infof("Calling federatedConfigMapUpdater.Update() - operations: %v", operations)
		err := ic.federatedConfigMapUpdater.Update(operations, ic.updateTimeout)
		if err != nil {
			configMapName := types.NamespacedName{Name: configMap.Name, Namespace: configMap.Namespace}
			glog.Errorf("Failed to execute update of ConfigMap %q on cluster %q: %v", configMapName, cluster.Name, err)
			ic.configMapDeliverer.DeliverAfter(cluster.Name, nil, ic.configMapReviewDelay)
		}
	} else {
		glog.V(4).Infof("Ingress UID update is not required: configMapUID %q is equal to clusterIngressUID %q", configMapUID, clusterIngressUID)
	}
}

/*
  getMasterCluster returns the cluster which is the elected master w.r.t. ingress UID, and it's ingress UID.
  If there is no elected master cluster, an error is returned.
  All other clusters must use the ingress UID of the elected master.
*/
func (ic *IngressController) getMasterCluster() (master *federation_api.Cluster, ingressUID string, err error) {
	clusters, err := ic.configMapFederatedInformer.GetReadyClusters() // TODO, get all clusters, not just ready ones.
	if err != nil {
		glog.Errorf("Failed to get cluster list: %v", err)
		return nil, "", err
	}

	for _, c := range clusters {
		UID, exists := c.ObjectMeta.Annotations[uidAnnotationKey]
		if exists && UID != "" { // Found the master cluster
			glog.V(4).Infof("Found master cluster %q with annotation %q=%q", c.Name, uidAnnotationKey, UID)
			return c, UID, nil
		}
	}
	return nil, "", fmt.Errorf("Failed to find master cluster with annotation %q", uidAnnotationKey)
}

/*
  updateClusterIngressUIDToMasters takes the ingress UID annotation on the master cluster and applies it to cluster.
  If there is no master cluster, then fallbackUID is used (and hence this cluster becomes the master).
*/
func (ic *IngressController) updateClusterIngressUIDToMasters(cluster *federation_api.Cluster, fallbackUID string) {
	masterCluster, masterUID, err := ic.getMasterCluster()
	/* TODO REMOVE - FOR DEBUG ONLY */ glog.Errorf("-> updateClusterIngressUIDToMasters: masterCluster: %q, masterUID: %q, cluster: %q, err: %v", masterCluster, masterUID, cluster, err)
	clusterObj, clusterErr := conversion.NewCloner().DeepCopy(cluster) // Make a clone so that we don't clobber our input param
	cluster, ok := clusterObj.(*federation_api.Cluster)
	if clusterErr != nil || !ok {
		glog.Errorf("Internal error: Failed clone cluster resource while attempting to add master ingress UID annotation (%q = %q) from master cluster %q to cluster %q, will try again later: %v", uidAnnotationKey, masterUID, masterCluster.Name, cluster.Name, err)
		return
	}
	if err == nil {
		if masterCluster.Name != cluster.Name { // We're not the master, need to get in sync
			cluster.ObjectMeta.Annotations[uidAnnotationKey] = masterUID
			if _, err = ic.federatedApiClient.Federation().Clusters().Update(cluster); err != nil {
				glog.Errorf("Failed to add master ingress UID annotation (%q = %q) from master cluster %q to cluster %q, will try again later: %v", uidAnnotationKey, masterUID, masterCluster.Name, cluster.Name, err)
				return
			} else {
				glog.V(4).Infof("Successfully added master ingress UID annotation (%q = %q) from master cluster %q to cluster %q.", uidAnnotationKey, masterUID, masterCluster.Name, cluster.Name)
			}
		} else {
			glog.V(4).Infof("Cluster %q with ingress UID is already the master with annotation (%q = %q), no need to update.", cluster.Name, uidAnnotationKey, cluster.ObjectMeta.Annotations[uidAnnotationKey])
		}
	} else {
		/* TODO Back to V(2) */ glog.Errorf("No master cluster found to source an ingress UID from for cluster %q.  Attempting to elect new master cluster %q with ingress UID %q = %q", cluster.Name, cluster.Name, uidAnnotationKey, fallbackUID)
		if fallbackUID != "" {
			cluster.ObjectMeta.Annotations[uidAnnotationKey] = fallbackUID
			if _, err = ic.federatedApiClient.Federation().Clusters().Update(cluster); err != nil {
				glog.Errorf("Failed to add ingress UID annotation (%q = %q) to cluster %q. No master elected. Will try again later: %v", uidAnnotationKey, fallbackUID, cluster.Name, err)
			} else {
				glog.V(4).Infof("Successfully added ingress UID annotation (%q = %q) to cluster %q.", uidAnnotationKey, fallbackUID, cluster.Name)
			}
		} else {
			glog.Errorf("No master cluster exists, and fallbackUID for cluster %q is invalid (%q).  This probably means that no clusters have an ingress controller configmap with key %q.  Federated Ingress currently supports clusters running Google Loadbalancer Controller (\"GLBC\")")
		}
	}
}

func (ic *IngressController) reconcileIngress(ingress types.NamespacedName) {
	glog.V(4).Infof("Reconciling ingress %q for all clusters", ingress)
	if !ic.isSynced() {
		ic.deliverIngress(ingress, ic.clusterAvailableDelay, false)
		return
	}

	key := ingress.String()
	baseIngressObj, exist, err := ic.ingressInformerStore.GetByKey(key)
	if err != nil {
		glog.Errorf("Failed to query main ingress store for %v: %v", ingress, err)
		ic.deliverIngress(ingress, 0, true)
		return
	}
	if !exist {
		// Not federated ingress, ignoring.
		glog.V(4).Infof("Ingress %q is not federated.  Ignoring.", ingress)
		return
	}
	baseIngress := baseIngressObj.(*extensions_v1beta1.Ingress)

	clusters, err := ic.ingressFederatedInformer.GetReadyClusters()
	if err != nil {
		glog.Errorf("Failed to get cluster list: %v", err)
		ic.deliverIngress(ingress, ic.clusterAvailableDelay, false)
		return
	} else {
		glog.V(4).Infof("Found %d ready clusters across which to reconcile ingress %q", len(clusters), ingress)
	}

	operations := make([]util.FederatedOperation, 0)

	for clusterIndex, cluster := range clusters {
		_, baseIPExists := baseIngress.ObjectMeta.Annotations[staticIPAnnotationKey]
		clusterIngressObj, found, err := ic.ingressFederatedInformer.GetTargetStore().GetByKey(cluster.Name, key)
		if err != nil {
			glog.Errorf("Failed to get cached ingress %s for cluster %s, will retry: %v", ingress, cluster.Name, err)
			ic.deliverIngress(ingress, 0, true)
			return
		}
		desiredIngress := &extensions_v1beta1.Ingress{
			ObjectMeta: baseIngress.ObjectMeta,
			Spec:       baseIngress.Spec,
		}

		if !found {
			glog.V(4).Infof("No existing Ingress %s in cluster %s - checking if appropriate to queue a create operation", ingress, cluster.Name)
			// We can't supply server-created fields when creating a new object.
			desiredIngress.ObjectMeta.ResourceVersion = ""
			desiredIngress.ObjectMeta.UID = ""
			ic.eventRecorder.Eventf(baseIngress, api.EventTypeNormal, "CreateInCluster",
				"Creating ingress in cluster %s", cluster.Name)

			// We always first create an ingress in the first available cluster.  Once that ingress
			// has been created and allocated a global IP (visible via an annotation),
			// we record that annotation on the federated ingress, and create all other cluster
			// ingresses with that same global IP.
			// Note: If the first cluster becomes (e.g. temporarily) unavailable, the second cluster will be allocated
			// index 0, but eventually all ingresses will share the single global IP recorded in the annotation
			// of the federated ingress.
			if true /* TODO! */ || baseIPExists || (clusterIndex == 0) {
				glog.V(4).Infof("No existing Ingress %s in cluster %s - queuing a create operation", ingress, cluster.Name)
				operations = append(operations, util.FederatedOperation{
					Type:        util.OperationTypeAdd,
					Obj:         desiredIngress,
					ClusterName: cluster.Name,
				})
			} else {
				glog.V(4).Infof("No annotation %q exists on cluster %q, and cluster index %d is not zero.  Not queueing create operation for ingress %q", staticIPAnnotationKey, cluster.Name, ingress)
			}
		} else {
			clusterIngress := clusterIngressObj.(*extensions_v1beta1.Ingress)
			glog.V(4).Infof("Found existing Ingress %s in cluster %s - checking if update is required", ingress, cluster.Name)
			clusterIPName, clusterIPExists := clusterIngress.ObjectMeta.Annotations[staticIPAnnotationKey]
			if !baseIPExists && clusterIPExists {
				// Add annotation to federated ingress via API.
				original, err := ic.federatedApiClient.Extensions().Ingresses(baseIngress.Namespace).Get(baseIngress.Name)
				if err == nil {
					original.ObjectMeta.Annotations[staticIPAnnotationKey] = clusterIPName
					if _, err = ic.federatedApiClient.Extensions().Ingresses(baseIngress.Namespace).Update(original); err != nil {
						glog.Errorf("Failed to add static IP annotation to federated ingress %q: %v", ingress, err)
					}
				} else {
					glog.Errorf("Failed to get federated ingress %q: %v", ingress, err)
				}
			}
			// Update existing ingress, if needed.
			if !util.ObjectMetaIsEquivalent(desiredIngress.ObjectMeta, clusterIngress.ObjectMeta) ||
				!reflect.DeepEqual(desiredIngress.Spec, clusterIngress.Spec) {
				glog.V(4).Infof("Ingress %s in cluster %s needs an update: cluster ingress %v is not equivalent to federated ingress %v", ingress, cluster.Name, clusterIngress, desiredIngress)
				// Merge any annotations on the federated ingress onto the underlying cluster ingress,
				// overwriting duplicates.
				// TODO: We should probably use a PATCH operation for this instead.
				// TODO: Add the UID annotation to all federated ingresses, so that it gets propagated to underlying ingresses to make them consistent.
				for key, val := range baseIngress.ObjectMeta.Annotations {
					desiredIngress.ObjectMeta.Annotations[key] = val
				}
				ic.eventRecorder.Eventf(baseIngress, api.EventTypeNormal, "UpdateInCluster",
					"Updating ingress in cluster %s", cluster.Name)

				operations = append(operations, util.FederatedOperation{
					Type:        util.OperationTypeUpdate,
					Obj:         desiredIngress,
					ClusterName: cluster.Name,
				})
			} else {
				glog.V(4).Infof("Ingress %q in cluster %q does not need an update: cluster ingress is equivalent to federated ingress", ingress, cluster.Name)
			}
		}
	}

	if len(operations) == 0 {
		// Everything is in order
		glog.V(4).Infof("Ingress %q is up-to-date in all clusters - no action required.", ingress)
		return
	}
	glog.V(4).Infof("Calling federatedUpdater.Update() - operations: %v", operations)
	err = ic.federatedIngressUpdater.UpdateWithOnError(operations, ic.updateTimeout, func(op util.FederatedOperation, operror error) {
		ic.eventRecorder.Eventf(baseIngress, api.EventTypeNormal, "FailedClusterUpdate",
			"Ingress update in cluster %s failed: %v", op.ClusterName, operror)
	})
	if err != nil {
		glog.Errorf("Failed to execute updates for %s: %v", ingress, err)
		ic.deliverIngress(ingress, ic.ingressReviewDelay, true)
		return
	}

	// Evertyhing is in order but lets be double sure - TODO: quinton: Why? This seems like a hack.
	ic.deliverIngress(ingress, ic.ingressReviewDelay, false)
}
