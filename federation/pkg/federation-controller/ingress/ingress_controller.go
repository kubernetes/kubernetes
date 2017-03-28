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
	"crypto/md5"
	"fmt"
	"sync"
	"time"

	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	pkgruntime "k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/watch"
	clientv1 "k8s.io/client-go/pkg/api/v1"
	"k8s.io/client-go/tools/cache"
	"k8s.io/client-go/tools/record"
	"k8s.io/client-go/util/flowcontrol"
	federationapi "k8s.io/kubernetes/federation/apis/federation/v1beta1"
	federationclientset "k8s.io/kubernetes/federation/client/clientset_generated/federation_clientset"
	"k8s.io/kubernetes/federation/pkg/federation-controller/util"
	"k8s.io/kubernetes/federation/pkg/federation-controller/util/deletionhelper"
	"k8s.io/kubernetes/federation/pkg/federation-controller/util/eventsink"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/v1"
	extensionsv1beta1 "k8s.io/kubernetes/pkg/apis/extensions/v1beta1"
	kubeclientset "k8s.io/kubernetes/pkg/client/clientset_generated/clientset"
	"k8s.io/kubernetes/pkg/controller"

	"github.com/golang/glog"
)

const (
	// Special cluster name which denotes all clusters - only used internally.  It's not a valid cluster name, so effectively reserved.
	allClustersKey = ".ALL_CLUSTERS"
	// TODO: Get the constants below directly from the Kubernetes Ingress Controller constants - but thats in a separate repo
	staticIPNameKeyWritable = "kubernetes.io/ingress.global-static-ip-name" // The writable annotation on Ingress to tell the controller to use a specific, named, static IP
	staticIPNameKeyReadonly = "ingress.kubernetes.io/static-ip"             // The readonly key via which the cluster's Ingress Controller communicates which static IP it used.  If staticIPNameKeyWritable above is specified, it is used.
	uidAnnotationKey        = "kubernetes.io/ingress.uid"                   // The annotation on federation clusters, where we store the ingress UID
	uidConfigMapName        = "ingress-uid"                                 // Name of the config-map and key the ingress controller stores its uid in.
	uidConfigMapNamespace   = "kube-system"
	uidKey                  = "uid"
	providerUidKey          = "provider-uid"
	// Annotation on the ingress in federation control plane that is used to keep
	// track of the first cluster in which we create ingress.
	// We wait for ingress to be created in this cluster before creating it any
	// other cluster.
	firstClusterAnnotation = "ingress.federation.kubernetes.io/first-cluster"
	ControllerName         = "ingresses"
)

var (
	RequiredResources = []schema.GroupVersionResource{extensionsv1beta1.SchemeGroupVersion.WithResource("ingresses")}
)

type IngressController struct {
	sync.Mutex // Lock used for leader election
	// For triggering single ingress reconciliation. This is used when there is an
	// add/update/delete operation on an ingress in either federated API server or
	// in some member of the federation.
	ingressDeliverer *util.DelayingDeliverer

	// For triggering reconciliation of cluster ingress controller configmap and
	// all ingresses. This is used when a new cluster becomes available.
	clusterDeliverer *util.DelayingDeliverer

	// For triggering reconciliation of cluster ingress controller configmap.
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
	ingressInformerController cache.Controller

	// Client to federated api server.
	federatedApiClient federationclientset.Interface

	// Backoff manager for ingresses
	ingressBackoff *flowcontrol.Backoff
	// Backoff manager for configmaps
	configMapBackoff *flowcontrol.Backoff

	// For events
	eventRecorder record.EventRecorder

	deletionHelper *deletionhelper.DeletionHelper

	ingressReviewDelay    time.Duration
	configMapReviewDelay  time.Duration
	clusterAvailableDelay time.Duration
	smallDelay            time.Duration
	updateTimeout         time.Duration
}

// NewIngressController returns a new ingress controller
func NewIngressController(client federationclientset.Interface) *IngressController {
	glog.V(4).Infof("->NewIngressController V(4)")
	broadcaster := record.NewBroadcaster()
	broadcaster.StartRecordingToSink(eventsink.NewFederatedEventSink(client))
	recorder := broadcaster.NewRecorder(api.Scheme, clientv1.EventSource{Component: "federated-ingress-controller"})
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

	// Build deliverers for triggering reconciliations.
	ic.ingressDeliverer = util.NewDelayingDeliverer()
	ic.clusterDeliverer = util.NewDelayingDeliverer()
	ic.configMapDeliverer = util.NewDelayingDeliverer()

	// Start informer in federated API servers on ingresses that should be federated.
	ic.ingressInformerStore, ic.ingressInformerController = cache.NewInformer(
		&cache.ListWatch{
			ListFunc: func(options metav1.ListOptions) (pkgruntime.Object, error) {
				return client.Extensions().Ingresses(metav1.NamespaceAll).List(options)
			},
			WatchFunc: func(options metav1.ListOptions) (watch.Interface, error) {
				return client.Extensions().Ingresses(metav1.NamespaceAll).Watch(options)
			},
		},
		&extensionsv1beta1.Ingress{},
		controller.NoResyncPeriodFunc(),
		util.NewTriggerOnAllChanges(
			func(obj pkgruntime.Object) {
				ic.deliverIngressObj(obj, 0, false)
			},
		))

	// Federated informer on ingresses in members of federation.
	ic.ingressFederatedInformer = util.NewFederatedInformer(
		client,
		func(cluster *federationapi.Cluster, targetClient kubeclientset.Interface) (cache.Store, cache.Controller) {
			return cache.NewInformer(
				&cache.ListWatch{
					ListFunc: func(options metav1.ListOptions) (pkgruntime.Object, error) {
						return targetClient.Extensions().Ingresses(metav1.NamespaceAll).List(options)
					},
					WatchFunc: func(options metav1.ListOptions) (watch.Interface, error) {
						return targetClient.Extensions().Ingresses(metav1.NamespaceAll).Watch(options)
					},
				},
				&extensionsv1beta1.Ingress{},
				controller.NoResyncPeriodFunc(),
				// Trigger reconciliation whenever something in federated cluster is changed. In most cases it
				// would be just confirmation that some ingress operation succeeded.
				util.NewTriggerOnAllChanges(
					func(obj pkgruntime.Object) {
						ic.deliverIngressObj(obj, ic.ingressReviewDelay, false)
					},
				))
		},

		&util.ClusterLifecycleHandlerFuncs{
			ClusterAvailable: func(cluster *federationapi.Cluster) {
				// When new cluster becomes available process all the ingresses again, and configure it's ingress controller's configmap with the correct UID
				ic.clusterDeliverer.DeliverAfter(cluster.Name, cluster, ic.clusterAvailableDelay)
			},
		},
	)

	// Federated informer on configmaps for ingress controllers in members of the federation.
	ic.configMapFederatedInformer = util.NewFederatedInformer(
		client,
		func(cluster *federationapi.Cluster, targetClient kubeclientset.Interface) (cache.Store, cache.Controller) {
			glog.V(4).Infof("Returning new informer for cluster %q", cluster.Name)
			return cache.NewInformer(
				&cache.ListWatch{
					ListFunc: func(options metav1.ListOptions) (pkgruntime.Object, error) {
						if targetClient == nil {
							glog.Errorf("Internal error: targetClient is nil")
						}
						return targetClient.Core().ConfigMaps(uidConfigMapNamespace).List(options) // we only want to list one by name - unfortunately Kubernetes don't have a selector for that.
					},
					WatchFunc: func(options metav1.ListOptions) (watch.Interface, error) {
						if targetClient == nil {
							glog.Errorf("Internal error: targetClient is nil")
						}
						return targetClient.Core().ConfigMaps(uidConfigMapNamespace).Watch(options) // as above
					},
				},
				&v1.ConfigMap{},
				controller.NoResyncPeriodFunc(),
				// Trigger reconciliation whenever the ingress controller's configmap in a federated cluster is changed. In most cases it
				// would be just confirmation that the configmap for the ingress controller is correct.
				util.NewTriggerOnAllChanges(
					func(obj pkgruntime.Object) {
						ic.deliverConfigMapObj(cluster.Name, obj, ic.configMapReviewDelay, false)
					},
				))
		},

		&util.ClusterLifecycleHandlerFuncs{
			ClusterAvailable: func(cluster *federationapi.Cluster) {
				ic.clusterDeliverer.DeliverAfter(cluster.Name, cluster, ic.clusterAvailableDelay)
			},
		},
	)

	// Federated ingress updater along with Create/Update/Delete operations.
	ic.federatedIngressUpdater = util.NewFederatedUpdater(ic.ingressFederatedInformer,
		func(client kubeclientset.Interface, obj pkgruntime.Object) error {
			ingress := obj.(*extensionsv1beta1.Ingress)
			glog.V(4).Infof("Attempting to create Ingress: %v", ingress)
			_, err := client.Extensions().Ingresses(ingress.Namespace).Create(ingress)
			if err != nil {
				glog.Errorf("Error creating ingress %q: %v", types.NamespacedName{Name: ingress.Name, Namespace: ingress.Namespace}, err)
			} else {
				glog.V(4).Infof("Successfully created ingress %q", types.NamespacedName{Name: ingress.Name, Namespace: ingress.Namespace})
			}
			return err
		},
		func(client kubeclientset.Interface, obj pkgruntime.Object) error {
			ingress := obj.(*extensionsv1beta1.Ingress)
			glog.V(4).Infof("Attempting to update Ingress: %v", ingress)
			_, err := client.Extensions().Ingresses(ingress.Namespace).Update(ingress)
			if err != nil {
				glog.V(4).Infof("Failed to update Ingress: %v", err)
			} else {
				glog.V(4).Infof("Successfully updated Ingress: %q", types.NamespacedName{Name: ingress.Name, Namespace: ingress.Namespace})
			}
			return err
		},
		func(client kubeclientset.Interface, obj pkgruntime.Object) error {
			ingress := obj.(*extensionsv1beta1.Ingress)
			glog.V(4).Infof("Attempting to delete Ingress: %v", ingress)
			err := client.Extensions().Ingresses(ingress.Namespace).Delete(ingress.Name, &metav1.DeleteOptions{})
			return err
		})

	// Federated configmap updater along with Create/Update/Delete operations.  Only Update should ever be called.
	ic.federatedConfigMapUpdater = util.NewFederatedUpdater(ic.configMapFederatedInformer,
		func(client kubeclientset.Interface, obj pkgruntime.Object) error {
			configMap := obj.(*v1.ConfigMap)
			configMapName := types.NamespacedName{Name: configMap.Name, Namespace: configMap.Namespace}
			glog.Errorf("Internal error: Incorrectly attempting to create ConfigMap: %q", configMapName)
			_, err := client.Core().ConfigMaps(configMap.Namespace).Create(configMap)
			return err
		},
		func(client kubeclientset.Interface, obj pkgruntime.Object) error {
			configMap := obj.(*v1.ConfigMap)
			configMapName := types.NamespacedName{Name: configMap.Name, Namespace: configMap.Namespace}
			glog.V(4).Infof("Attempting to update ConfigMap: %v", configMap)
			_, err := client.Core().ConfigMaps(configMap.Namespace).Update(configMap)
			if err == nil {
				glog.V(4).Infof("Successfully updated ConfigMap %q %v", configMapName, configMap)
			} else {
				glog.V(4).Infof("Failed to update ConfigMap %q: %v", configMapName, err)
			}
			return err
		},
		func(client kubeclientset.Interface, obj pkgruntime.Object) error {
			configMap := obj.(*v1.ConfigMap)
			configMapName := types.NamespacedName{Name: configMap.Name, Namespace: configMap.Namespace}
			glog.Errorf("Internal error: Incorrectly attempting to delete ConfigMap: %q", configMapName)
			err := client.Core().ConfigMaps(configMap.Namespace).Delete(configMap.Name, &metav1.DeleteOptions{})
			return err
		})

	ic.deletionHelper = deletionhelper.NewDeletionHelper(
		ic.hasFinalizerFunc,
		ic.removeFinalizerFunc,
		ic.addFinalizerFunc,
		// objNameFunc
		func(obj pkgruntime.Object) string {
			ingress := obj.(*extensionsv1beta1.Ingress)
			return ingress.Name
		},
		ic.updateTimeout,
		ic.eventRecorder,
		ic.ingressFederatedInformer,
		ic.federatedIngressUpdater,
	)
	return ic
}

// Returns true if the given object has the given finalizer in its ObjectMeta.
func (ic *IngressController) hasFinalizerFunc(obj pkgruntime.Object, finalizer string) bool {
	ingress := obj.(*extensionsv1beta1.Ingress)
	for i := range ingress.ObjectMeta.Finalizers {
		if string(ingress.ObjectMeta.Finalizers[i]) == finalizer {
			return true
		}
	}
	return false
}

// Removes the finalizer from the given objects ObjectMeta.
// Assumes that the given object is a ingress.
func (ic *IngressController) removeFinalizerFunc(obj pkgruntime.Object, finalizer string) (pkgruntime.Object, error) {
	ingress := obj.(*extensionsv1beta1.Ingress)
	newFinalizers := []string{}
	hasFinalizer := false
	for i := range ingress.ObjectMeta.Finalizers {
		if string(ingress.ObjectMeta.Finalizers[i]) != finalizer {
			newFinalizers = append(newFinalizers, ingress.ObjectMeta.Finalizers[i])
		} else {
			hasFinalizer = true
		}
	}
	if !hasFinalizer {
		// Nothing to do.
		return obj, nil
	}
	ingress.ObjectMeta.Finalizers = newFinalizers
	ingress, err := ic.federatedApiClient.Extensions().Ingresses(ingress.Namespace).Update(ingress)
	if err != nil {
		return nil, fmt.Errorf("failed to remove finalizer %s from ingress %s: %v", finalizer, ingress.Name, err)
	}
	return ingress, nil
}

// Adds the given finalizers to the given objects ObjectMeta.
// Assumes that the given object is a ingress.
func (ic *IngressController) addFinalizerFunc(obj pkgruntime.Object, finalizers []string) (pkgruntime.Object, error) {
	ingress := obj.(*extensionsv1beta1.Ingress)
	ingress.ObjectMeta.Finalizers = append(ingress.ObjectMeta.Finalizers, finalizers...)
	ingress, err := ic.federatedApiClient.Extensions().Ingresses(ingress.Namespace).Update(ingress)
	if err != nil {
		return nil, fmt.Errorf("failed to add finalizers %v to ingress %s: %v", finalizers, ingress.Name, err)
	}
	return ingress, nil
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
		glog.Infof("Stopping ingress deliverer")
		ic.ingressDeliverer.Stop()
		glog.Infof("Stopping configmap deliverer")
		ic.configMapDeliverer.Stop()
		glog.Infof("Stopping cluster deliverer")
		ic.clusterDeliverer.Stop()
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
		if clusterName != allClustersKey {
			glog.V(4).Infof("ConfigMap change delivered for cluster %q, reconciling configmap for that cluster", clusterName)
		} else {
			glog.V(4).Infof("ConfigMap change delivered for all clusters, reconciling configmaps for all clusters")
		}
		ic.reconcileConfigMapForCluster(clusterName)
	})

	util.StartBackoffGC(ic.ingressBackoff, stopChan)
	util.StartBackoffGC(ic.configMapBackoff, stopChan)
}

func (ic *IngressController) deliverIngressObj(obj interface{}, delay time.Duration, failed bool) {
	ingress := obj.(*extensionsv1beta1.Ingress)
	ic.deliverIngress(types.NamespacedName{Namespace: ingress.Namespace, Name: ingress.Name}, delay, failed)
}

func (ic *IngressController) deliverIngress(ingress types.NamespacedName, delay time.Duration, failed bool) {
	glog.V(4).Infof("Delivering ingress: %s with delay: %v error: %v", ingress, delay, failed)
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
// synced with the corresponding api server.
func (ic *IngressController) isSynced() bool {
	if !ic.ingressFederatedInformer.ClustersSynced() {
		glog.V(2).Infof("Cluster list not synced for ingress federated informer")
		return false
	}
	clusters, err := ic.ingressFederatedInformer.GetReadyClusters()
	if err != nil {
		glog.Errorf("Failed to get ready clusters for ingress federated informer: %v", err)
		return false
	}
	if !ic.ingressFederatedInformer.GetTargetStore().ClustersSynced(clusters) {
		glog.V(2).Infof("Target store not synced for ingress federated informer")
		return false
	}
	if !ic.configMapFederatedInformer.ClustersSynced() {
		glog.V(2).Infof("Cluster list not synced for config map federated informer")
		return false
	}
	clusters, err = ic.configMapFederatedInformer.GetReadyClusters()
	if err != nil {
		glog.Errorf("Failed to get ready clusters for configmap federated informer: %v", err)
		return false
	}
	if !ic.configMapFederatedInformer.GetTargetStore().ClustersSynced(clusters) {
		glog.V(2).Infof("Target store not synced for configmap federated informer")
		return false
	}
	glog.V(4).Infof("Cluster list is synced")
	return true
}

// The function triggers reconciliation of all federated ingresses.  clusterName is the name of the cluster that changed
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
		ingress := obj.(*extensionsv1beta1.Ingress)
		nsName := types.NamespacedName{Name: ingress.Name, Namespace: ingress.Namespace}
		glog.V(4).Infof("Delivering federated ingress %q for cluster %q", nsName, clusterName)
		ic.deliverIngress(nsName, ic.smallDelay, false)
	}
}

/*
  reconcileConfigMapForCluster ensures that the configmap for the ingress controller in the cluster has objectmeta.data.UID
  consistent with all the other clusters in the federation. If clusterName == allClustersKey, then all available clusters
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
		cluster, found, err := ic.configMapFederatedInformer.GetReadyCluster(clusterName)
		if err != nil || !found {
			glog.Errorf("Internal error: Cluster %q queued for configmap reconciliation, but not found.  Will try again later: error = %v", clusterName, err)
			ic.configMapDeliverer.DeliverAfter(clusterName, nil, ic.clusterAvailableDelay)
			return
		}
		uidConfigMapNamespacedName := types.NamespacedName{Name: uidConfigMapName, Namespace: uidConfigMapNamespace}
		configMapObj, found, err := ic.configMapFederatedInformer.GetTargetStore().GetByKey(cluster.Name, uidConfigMapNamespacedName.String())
		if !found || err != nil {
			logmsg := fmt.Sprintf("Failed to get ConfigMap %q for cluster %q.  Will try again later", uidConfigMapNamespacedName, cluster.Name)
			if err != nil {
				logmsg = fmt.Sprintf("%v: %v", logmsg, err)
			}
			if len(ic.ingressInformerStore.List()) > 0 { // Error-level if ingresses are active, Info-level otherwise.
				glog.Errorf(logmsg)
			} else {
				glog.V(4).Infof(logmsg)
			}
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

// getProviderUid returns a provider ID based on the provided clusterName.
func getProviderUid(clusterName string) string {
	hashedName := md5.Sum([]byte(clusterName))
	return fmt.Sprintf("%x", hashedName[:8])
}

/*
  reconcileConfigMap ensures that the configmap in the cluster has a UID
  consistent with the federation cluster's associated annotation.

  1. If the UID in the configmap differs from the UID stored in the cluster's annotation, the configmap is updated.
  2. If the UID annotation is missing from the cluster, the cluster's UID annotation is updated to be consistent
  with the master cluster.
  3. If there is no elected master cluster, this cluster attempts to elect itself as the master cluster.

  In cases 2 and 3, the configmaps will be updated in the next cycle, triggered by the federation cluster update(s)

*/
func (ic *IngressController) reconcileConfigMap(cluster *federationapi.Cluster, configMap *v1.ConfigMap) {
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
		glog.V(4).Infof("Cluster %q is the only master", cluster.Name)
		// Second argument is the fallback, in case this is the only cluster, in which case it becomes the master
		var err error
		if clusterIngressUID, err = ic.updateClusterIngressUIDToMasters(cluster, configMapUID); err != nil {
			return
		}
		// If we successfully update the Cluster Object, fallthrough and update the configMap.
	}

	// Figure out providerUid.
	providerUid := getProviderUid(cluster.Name)
	configMapProviderUid := configMap.Data[providerUidKey]

	if configMapUID == clusterIngressUID && configMapProviderUid == providerUid {
		glog.V(4).Infof("Ingress configMap update is not required: UID %q and ProviderUid %q are equal", configMapUID, providerUid)
	} else {
		if configMapUID != clusterIngressUID {
			glog.V(4).Infof("Ingress configMap update is required for UID: configMapUID %q not equal to clusterIngressUID %q", configMapUID, clusterIngressUID)
		} else if configMapProviderUid != providerUid {
			glog.V(4).Infof("Ingress configMap update is required: configMapProviderUid %q not equal to providerUid %q", configMapProviderUid, providerUid)
		}
		configMap.Data[uidKey] = clusterIngressUID
		configMap.Data[providerUidKey] = providerUid
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
	}
}

/*
  getMasterCluster returns the cluster which is the elected master w.r.t. ingress UID, and it's ingress UID.
  If there is no elected master cluster, an error is returned.
  All other clusters must use the ingress UID of the elected master.
*/
func (ic *IngressController) getMasterCluster() (master *federationapi.Cluster, ingressUID string, err error) {
	clusters, err := ic.configMapFederatedInformer.GetReadyClusters()
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
func (ic *IngressController) updateClusterIngressUIDToMasters(cluster *federationapi.Cluster, fallbackUID string) (string, error) {
	masterCluster, masterUID, err := ic.getMasterCluster()
	clusterObj, clusterErr := api.Scheme.DeepCopy(cluster) // Make a clone so that we don't clobber our input param
	cluster, ok := clusterObj.(*federationapi.Cluster)
	if clusterErr != nil || !ok {
		glog.Errorf("Internal error: Failed clone cluster resource while attempting to add master ingress UID annotation (%q = %q) from master cluster %q to cluster %q, will try again later: %v", uidAnnotationKey, masterUID, masterCluster.Name, cluster.Name, err)
		return "", err
	}
	if err == nil {
		if masterCluster.Name != cluster.Name { // We're not the master, need to get in sync
			if cluster.ObjectMeta.Annotations == nil {
				cluster.ObjectMeta.Annotations = map[string]string{}
			}
			cluster.ObjectMeta.Annotations[uidAnnotationKey] = masterUID
			if _, err = ic.federatedApiClient.Federation().Clusters().Update(cluster); err != nil {
				glog.Errorf("Failed to add master ingress UID annotation (%q = %q) from master cluster %q to cluster %q, will try again later: %v", uidAnnotationKey, masterUID, masterCluster.Name, cluster.Name, err)
				return "", err
			} else {
				glog.V(4).Infof("Successfully added master ingress UID annotation (%q = %q) from master cluster %q to cluster %q.", uidAnnotationKey, masterUID, masterCluster.Name, cluster.Name)
				return masterUID, nil
			}
		} else {
			glog.V(4).Infof("Cluster %q with ingress UID is already the master with annotation (%q = %q), no need to update.", cluster.Name, uidAnnotationKey, cluster.ObjectMeta.Annotations[uidAnnotationKey])
			return cluster.ObjectMeta.Annotations[uidAnnotationKey], nil
		}
	} else {
		glog.V(2).Infof("No master cluster found to source an ingress UID from for cluster %q.  Attempting to elect new master cluster %q with ingress UID %q = %q", cluster.Name, cluster.Name, uidAnnotationKey, fallbackUID)
		if fallbackUID != "" {
			if cluster.ObjectMeta.Annotations == nil {
				cluster.ObjectMeta.Annotations = map[string]string{}
			}
			cluster.ObjectMeta.Annotations[uidAnnotationKey] = fallbackUID
			if _, err = ic.federatedApiClient.Federation().Clusters().Update(cluster); err != nil {
				glog.Errorf("Failed to add ingress UID annotation (%q = %q) to cluster %q. No master elected. Will try again later: %v", uidAnnotationKey, fallbackUID, cluster.Name, err)
				return "", err
			} else {
				glog.V(4).Infof("Successfully added ingress UID annotation (%q = %q) to cluster %q.", uidAnnotationKey, fallbackUID, cluster.Name)
				return fallbackUID, nil
			}
		} else {
			glog.Errorf("No master cluster exists, and fallbackUID for cluster %q is invalid (%q).  This probably means that no clusters have an ingress controller configmap with key %q.  Federated Ingress currently supports clusters running Google Loadbalancer Controller (\"GLBC\")", cluster.Name, fallbackUID, uidKey)
			return "", err
		}
	}
}

func (ic *IngressController) isClusterReady(clusterName string) bool {
	cluster, isReady, err := ic.ingressFederatedInformer.GetReadyCluster(clusterName)
	return isReady && err == nil && cluster != nil
}

// updateAnnotationOnIngress updates the annotation with the given key on the given federated ingress.
// Queues the ingress for resync when done.
func (ic *IngressController) updateAnnotationOnIngress(ingress *extensionsv1beta1.Ingress, key, value string) {
	if ingress.ObjectMeta.Annotations == nil {
		ingress.ObjectMeta.Annotations = make(map[string]string)
	}
	ingress.ObjectMeta.Annotations[key] = value
	ingressName := types.NamespacedName{Name: ingress.Name, Namespace: ingress.Namespace}
	glog.V(4).Infof("Attempting to update annotation %s:%s on base federated ingress: %v", key, value, ingressName)
	if updatedFedIngress, err := ic.federatedApiClient.Extensions().Ingresses(ingress.Namespace).Update(ingress); err != nil {
		glog.Errorf("Failed to update annotation %s:%s on federated ingress %q, will try again later: %v", key, value, ingressName, err)
		ic.deliverIngress(ingressName, ic.ingressReviewDelay, true)
		return
	} else {
		glog.V(4).Infof("Successfully updated annotation %s:%s on federated ingress %q, after update: %q", key, value, ingress, updatedFedIngress)
		ic.deliverIngress(ingressName, ic.smallDelay, false)
		return
	}
}

func (ic *IngressController) reconcileIngress(ingress types.NamespacedName) {
	glog.V(4).Infof("Reconciling ingress %q for all clusters", ingress)
	if !ic.isSynced() {
		ic.deliverIngress(ingress, ic.clusterAvailableDelay, false)
		return
	}

	key := ingress.String()
	baseIngressObjFromStore, exist, err := ic.ingressInformerStore.GetByKey(key)
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
	baseIngressObj, err := api.Scheme.DeepCopy(baseIngressObjFromStore)
	baseIngress, ok := baseIngressObj.(*extensionsv1beta1.Ingress)
	if err != nil || !ok {
		glog.Errorf("Internal Error %v : Object retrieved from ingressInformerStore with key %q is not of correct type *extensionsv1beta1.Ingress: %v", err, key, baseIngressObj)
	} else {
		glog.V(4).Infof("Base (federated) ingress: %v", baseIngress)
	}

	if baseIngress.DeletionTimestamp != nil {
		if err := ic.delete(baseIngress); err != nil {
			glog.Errorf("Failed to delete %s: %v", ingress, err)
			ic.eventRecorder.Eventf(baseIngress, api.EventTypeNormal, "DeleteFailed",
				"Ingress delete failed: %v", err)
			ic.deliverIngress(ingress, 0, true)
		}
		return
	}

	glog.V(3).Infof("Ensuring delete object from underlying clusters finalizer for ingress: %s",
		baseIngress.Name)
	// Add the required finalizers before creating a ingress in underlying clusters.
	updatedIngressObj, err := ic.deletionHelper.EnsureFinalizers(baseIngress)
	if err != nil {
		glog.Errorf("Failed to ensure delete object from underlying clusters finalizer in ingress %s: %v",
			baseIngress.Name, err)
		ic.deliverIngress(ingress, 0, true)
		return
	}
	baseIngress = updatedIngressObj.(*extensionsv1beta1.Ingress)

	glog.V(3).Infof("Syncing ingress %s in underlying clusters", baseIngress.Name)

	clusters, err := ic.ingressFederatedInformer.GetReadyClusters()
	if err != nil {
		glog.Errorf("Failed to get cluster list: %v", err)
		ic.deliverIngress(ingress, ic.clusterAvailableDelay, false)
		return
	} else {
		glog.V(4).Infof("Found %d ready clusters across which to reconcile ingress %q", len(clusters), ingress)
	}

	operations := make([]util.FederatedOperation, 0)

	for _, cluster := range clusters {
		baseIPName, baseIPAnnotationExists := baseIngress.ObjectMeta.Annotations[staticIPNameKeyWritable]
		firstClusterName, firstClusterExists := baseIngress.ObjectMeta.Annotations[firstClusterAnnotation]
		clusterIngressObj, clusterIngressFound, err := ic.ingressFederatedInformer.GetTargetStore().GetByKey(cluster.Name, key)
		if err != nil {
			glog.Errorf("Failed to get cached ingress %s for cluster %s, will retry: %v", ingress, cluster.Name, err)
			ic.deliverIngress(ingress, 0, true)
			return
		}
		desiredIngress := &extensionsv1beta1.Ingress{}
		objMeta, err := api.Scheme.DeepCopy(&baseIngress.ObjectMeta)
		if err != nil {
			glog.Errorf("Error deep copying ObjectMeta: %v", err)
		}
		objSpec, err := api.Scheme.DeepCopy(&baseIngress.Spec)
		if err != nil {
			glog.Errorf("Error deep copying Spec: %v", err)
		}
		objMetaCopy, ok := objMeta.(*metav1.ObjectMeta)
		if !ok {
			glog.Errorf("Internal error: Failed to cast to *metav1.ObjectMeta: %v", objMeta)
		}
		desiredIngress.ObjectMeta = *objMetaCopy
		objSpecCopy, ok := objSpec.(*extensionsv1beta1.IngressSpec)
		if !ok {
			glog.Errorf("Internal error: Failed to cast to extensionsv1beta1.Ingressespec: %v", objSpec)
		}
		desiredIngress.Spec = *objSpecCopy
		glog.V(4).Infof("Desired Ingress: %v", desiredIngress)

		if !clusterIngressFound {
			glog.V(4).Infof("No existing Ingress %s in cluster %s - checking if appropriate to queue a create operation", ingress, cluster.Name)
			// We can't supply server-created fields when creating a new object.
			desiredIngress.ObjectMeta = util.DeepCopyRelevantObjectMeta(baseIngress.ObjectMeta)
			ic.eventRecorder.Eventf(baseIngress, api.EventTypeNormal, "CreateInCluster",
				"Creating ingress in cluster %s", cluster.Name)

			// We always first create an ingress in the first available cluster. Once that ingress
			// has been created and allocated a global IP (visible via an annotation),
			// we record that annotation on the federated ingress, and create all other cluster
			// ingresses with that same global IP.
			// Note: If the first cluster becomes (e.g. temporarily) unavailable, the
			// second cluster will become the first cluster, but eventually all ingresses
			// will share the single global IP recorded in the annotation of the
			// federated ingress.
			haveFirstCluster := firstClusterExists && firstClusterName != "" && ic.isClusterReady(firstClusterName)
			if !haveFirstCluster {
				glog.V(4).Infof("No cluster has been chosen as the first cluster. Electing cluster %s as the first cluster to create ingress in", cluster.Name)
				ic.updateAnnotationOnIngress(baseIngress, firstClusterAnnotation, cluster.Name)
				return
			}
			if baseIPAnnotationExists || firstClusterName == cluster.Name {
				if baseIPAnnotationExists {
					glog.V(4).Infof("No existing Ingress %s in cluster %s and static IP annotation (%q) exists on base ingress - queuing a create operation", ingress, cluster.Name, staticIPNameKeyWritable)
				} else {
					glog.V(4).Infof("No existing Ingress %s in cluster %s and no static IP annotation (%q) on base ingress - queuing a create operation in first cluster", ingress, cluster.Name, staticIPNameKeyWritable)
				}
				operations = append(operations, util.FederatedOperation{
					Type:        util.OperationTypeAdd,
					Obj:         desiredIngress,
					ClusterName: cluster.Name,
				})
			} else {
				glog.V(4).Infof("No annotation %q exists on ingress %q in federation and waiting for ingress in cluster %s. Not queueing create operation for ingress until annotation exists", staticIPNameKeyWritable, ingress, firstClusterName)
			}
		} else {
			clusterIngress := clusterIngressObj.(*extensionsv1beta1.Ingress)
			glog.V(4).Infof("Found existing Ingress %s in cluster %s - checking if update is required (in either direction)", ingress, cluster.Name)
			clusterIPName, clusterIPNameExists := clusterIngress.ObjectMeta.Annotations[staticIPNameKeyReadonly]
			baseLBStatusExists := len(baseIngress.Status.LoadBalancer.Ingress) > 0
			clusterLBStatusExists := len(clusterIngress.Status.LoadBalancer.Ingress) > 0
			logStr := fmt.Sprintf("Cluster ingress %q has annotation %q=%q, loadbalancer status exists? [%v], federated ingress has annotation %q=%q, loadbalancer status exists? [%v].  %%s annotation and/or loadbalancer status from cluster ingress to federated ingress.", ingress, staticIPNameKeyReadonly, clusterIPName, clusterLBStatusExists, staticIPNameKeyWritable, baseIPName, baseLBStatusExists)
			if (!baseIPAnnotationExists && clusterIPNameExists) || (!baseLBStatusExists && clusterLBStatusExists) { // copy the IP name from the readonly annotation on the cluster ingress, to the writable annotation on the federated ingress
				glog.V(4).Infof(logStr, "Transferring")
				if !baseIPAnnotationExists && clusterIPNameExists {
					ic.updateAnnotationOnIngress(baseIngress, staticIPNameKeyWritable, clusterIPName)
					return
				}
				if !baseLBStatusExists && clusterLBStatusExists {
					lbstatusObj, lbErr := api.Scheme.DeepCopy(&clusterIngress.Status.LoadBalancer)
					lbstatus, ok := lbstatusObj.(*v1.LoadBalancerStatus)
					if lbErr != nil || !ok {
						glog.Errorf("Internal error: Failed to clone LoadBalancerStatus of %q in cluster %q while attempting to update master loadbalancer ingress status, will try again later. error: %v, Object to be cloned: %v", ingress, cluster.Name, lbErr, lbstatusObj)
						ic.deliverIngress(ingress, ic.ingressReviewDelay, true)
						return
					}
					baseIngress.Status.LoadBalancer = *lbstatus
					glog.V(4).Infof("Attempting to update base federated ingress status: %v", baseIngress)
					if updatedFedIngress, err := ic.federatedApiClient.Extensions().Ingresses(baseIngress.Namespace).UpdateStatus(baseIngress); err != nil {
						glog.Errorf("Failed to update federated ingress status of %q (loadbalancer status), will try again later: %v", ingress, err)
						ic.deliverIngress(ingress, ic.ingressReviewDelay, true)
						return
					} else {
						glog.V(4).Infof("Successfully updated federated ingress status of %q (added loadbalancer status), after update: %q", ingress, updatedFedIngress)
						ic.deliverIngress(ingress, ic.smallDelay, false)
						return
					}
				}
			} else {
				glog.V(4).Infof(logStr, "Not transferring")
			}
			// Update existing cluster ingress, if needed.
			if util.ObjectMetaAndSpecEquivalent(baseIngress, clusterIngress) {
				glog.V(4).Infof("Ingress %q in cluster %q does not need an update: cluster ingress is equivalent to federated ingress", ingress, cluster.Name)
			} else {
				glog.V(4).Infof("Ingress %s in cluster %s needs an update: cluster ingress %v is not equivalent to federated ingress %v", ingress, cluster.Name, clusterIngress, desiredIngress)
				objMeta, err := api.Scheme.DeepCopy(&clusterIngress.ObjectMeta)
				if err != nil {
					glog.Errorf("Error deep copying ObjectMeta: %v", err)
					ic.deliverIngress(ingress, ic.ingressReviewDelay, true)
				}
				objMetaCopy, ok := objMeta.(*metav1.ObjectMeta)
				if !ok {
					glog.Errorf("Internal error: Failed to cast to metav1.ObjectMeta: %v", objMeta)
					ic.deliverIngress(ingress, ic.ingressReviewDelay, true)
				}
				desiredIngress.ObjectMeta = *objMetaCopy
				// Merge any annotations and labels on the federated ingress onto the underlying cluster ingress,
				// overwriting duplicates.
				if desiredIngress.ObjectMeta.Annotations == nil {
					desiredIngress.ObjectMeta.Annotations = make(map[string]string)
				}
				for key, val := range baseIngress.ObjectMeta.Annotations {
					desiredIngress.ObjectMeta.Annotations[key] = val
				}
				if desiredIngress.ObjectMeta.Labels == nil {
					desiredIngress.ObjectMeta.Labels = make(map[string]string)
				}
				for key, val := range baseIngress.ObjectMeta.Labels {
					desiredIngress.ObjectMeta.Labels[key] = val
				}
				ic.eventRecorder.Eventf(baseIngress, api.EventTypeNormal, "UpdateInCluster",
					"Updating ingress in cluster %s", cluster.Name)

				operations = append(operations, util.FederatedOperation{
					Type:        util.OperationTypeUpdate,
					Obj:         desiredIngress,
					ClusterName: cluster.Name,
				})
				// TODO: Transfer any readonly (target-proxy, url-map etc) annotations from the master cluster to the federation, if this is the master cluster.
				// This is only for consistency, so that the federation ingress metadata matches the underlying clusters.  It's not actually required				}
			}
		}
	}

	if len(operations) == 0 {
		// Everything is in order
		glog.V(4).Infof("Ingress %q is up-to-date in all clusters - no propagation to clusters required.", ingress)
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
	// Schedule another periodic reconciliation, only to account for possible bugs in watch processing.
	ic.deliverIngress(ingress, ic.ingressReviewDelay, false)
}

// delete deletes the given ingress or returns error if the deletion was not complete.
func (ic *IngressController) delete(ingress *extensionsv1beta1.Ingress) error {
	glog.V(3).Infof("Handling deletion of ingress: %v", *ingress)
	_, err := ic.deletionHelper.HandleObjectInUnderlyingClusters(ingress)
	if err != nil {
		return err
	}

	err = ic.federatedApiClient.Extensions().Ingresses(ingress.Namespace).Delete(ingress.Name, nil)
	if err != nil {
		// Its all good if the error is not found error. That means it is deleted already and we do not have to do anything.
		// This is expected when we are processing an update as a result of ingress finalizer deletion.
		// The process that deleted the last finalizer is also going to delete the ingress and we do not have to do anything.
		if !errors.IsNotFound(err) {
			return fmt.Errorf("failed to delete ingress: %v", err)
		}
	}
	return nil
}
