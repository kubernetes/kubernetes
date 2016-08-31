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
	"reflect"
	"time"

	federation_api "k8s.io/kubernetes/federation/apis/federation/v1beta1"
	federation_release_1_4 "k8s.io/kubernetes/federation/client/clientset_generated/federation_release_1_4"
	"k8s.io/kubernetes/federation/pkg/federation-controller/util"
	"k8s.io/kubernetes/federation/pkg/federation-controller/util/eventsink"
	"k8s.io/kubernetes/pkg/api"
	extensions_v1beta1 "k8s.io/kubernetes/pkg/apis/extensions/v1beta1"
	"k8s.io/kubernetes/pkg/client/cache"
	kube_release_1_4 "k8s.io/kubernetes/pkg/client/clientset_generated/release_1_4"
	"k8s.io/kubernetes/pkg/client/record"
	"k8s.io/kubernetes/pkg/controller"
	"k8s.io/kubernetes/pkg/controller/framework"
	pkg_runtime "k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/types"
	"k8s.io/kubernetes/pkg/util/flowcontrol"
	"k8s.io/kubernetes/pkg/watch"

	"github.com/golang/glog"
)

const (
	allClustersKey        = "ALL_CLUSTERS"
	staticIPAnnotationKey = "ingress.kubernetes.io/static-ip" // TODO: Get this directly from the Kubernetes Ingress Controller constant
)

type IngressController struct {
	// For triggering single ingress reconcilation. This is used when there is an
	// add/update/delete operation on an ingress in either federated API server or
	// in some member of the federation.
	ingressDeliverer *util.DelayingDeliverer

	// For triggering reconcilation of all ingresses. This is used when
	// a new cluster becomes available.
	clusterDeliverer *util.DelayingDeliverer

	// Contains ingresses present in members of federation.
	ingressFederatedInformer util.FederatedInformer
	// For updating members of federation.
	federatedUpdater util.FederatedUpdater
	// Definitions of ingresses that should be federated.
	ingressInformerStore cache.Store
	// Informer controller for ingresses that should be federated.
	ingressInformerController framework.ControllerInterface

	// Client to federated api server.
	federatedApiClient federation_release_1_4.Interface

	// Backoff manager for ingresses
	ingressBackoff *flowcontrol.Backoff

	// For events
	eventRecorder record.EventRecorder

	ingressReviewDelay    time.Duration
	clusterAvailableDelay time.Duration
	smallDelay            time.Duration
	updateTimeout         time.Duration
}

// NewIngressController returns a new ingress controller
func NewIngressController(client federation_release_1_4.Interface) *IngressController {
	broadcaster := record.NewBroadcaster()
	broadcaster.StartRecordingToSink(eventsink.NewFederatedEventSink(client))
	recorder := broadcaster.NewRecorder(api.EventSource{Component: "federated-ingress-controller"})

	ic := &IngressController{
		federatedApiClient:    client,
		ingressReviewDelay:    time.Second * 10,
		clusterAvailableDelay: time.Second * 20,
		smallDelay:            time.Second * 3,
		updateTimeout:         time.Second * 30,
		ingressBackoff:        flowcontrol.NewBackOff(5*time.Second, time.Minute),
		eventRecorder:         recorder,
	}

	// Build deliverers for triggering reconcilations.
	ic.ingressDeliverer = util.NewDelayingDeliverer()
	ic.clusterDeliverer = util.NewDelayingDeliverer()

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
				// When new cluster becomes available process all the ingresses again.
				ic.clusterDeliverer.DeliverAt(allClustersKey, nil, time.Now().Add(ic.clusterAvailableDelay))
			},
		},
	)

	// Federated updater along with Create/Update/Delete operations.
	ic.federatedUpdater = util.NewFederatedUpdater(ic.ingressFederatedInformer,
		func(client kube_release_1_4.Interface, obj pkg_runtime.Object) error {
			ingress := obj.(*extensions_v1beta1.Ingress)
			glog.V(4).Infof("Attempting to create Ingress: %v", ingress)
			_, err := client.Extensions().Ingresses(ingress.Namespace).Create(ingress)
			return err
		},
		func(client kube_release_1_4.Interface, obj pkg_runtime.Object) error {
			ingress := obj.(*extensions_v1beta1.Ingress)
			glog.V(4).Infof("Attempting to update Ingress: %v", ingress)
			_, err := client.Extensions().Ingresses(ingress.Namespace).Update(ingress)
			return err
		},
		func(client kube_release_1_4.Interface, obj pkg_runtime.Object) error {
			ingress := obj.(*extensions_v1beta1.Ingress)
			glog.V(4).Infof("Attempting to delete Ingress: %v", ingress)
			err := client.Extensions().Ingresses(ingress.Namespace).Delete(ingress.Name, &api.DeleteOptions{})
			return err
		})
	return ic
}

func (ic *IngressController) Run(stopChan <-chan struct{}) {
	glog.Infof("Starting Ingress Controller")
	go ic.ingressInformerController.Run(stopChan)
	glog.Infof("... Starting Ingress Federated Informer")
	ic.ingressFederatedInformer.Start()
	go func() {
		<-stopChan
		glog.Infof("Stopping Ingress Controller")
		ic.ingressFederatedInformer.Stop()
	}()
	ic.ingressDeliverer.StartWithHandler(func(item *util.DelayingDelivererItem) {
		ingress := item.Value.(types.NamespacedName)
		glog.V(4).Infof("Ingress change delivered, reconciling: %v", ingress)
		ic.reconcileIngress(ingress)
	})
	ic.clusterDeliverer.StartWithHandler(func(_ *util.DelayingDelivererItem) {
		glog.V(4).Infof("Cluster change delivered, reconciling ingresses")
		ic.reconcileIngressesOnClusterChange()
	})
	go func() {
		select {
		case <-time.After(time.Minute):
			glog.V(4).Infof("Ingress controller is garbage collecting")
			ic.ingressBackoff.GC()
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

// Check whether all data stores are in sync. False is returned if any of the informer/stores is not yet
// synced with the coresponding api server.
func (ic *IngressController) isSynced() bool {
	if !ic.ingressFederatedInformer.ClustersSynced() {
		glog.V(2).Infof("Cluster list not synced")
		return false
	}
	clusters, err := ic.ingressFederatedInformer.GetReadyClusters()
	if err != nil {
		glog.Errorf("Failed to get ready clusters: %v", err)
		return false
	}
	if !ic.ingressFederatedInformer.GetTargetStore().ClustersSynced(clusters) {
		glog.V(2).Infof("Target store not synced")
		return false
	}
	glog.V(4).Infof("Cluster list is synced")
	return true
}

// The function triggers reconcilation of all federated ingresses.
func (ic *IngressController) reconcileIngressesOnClusterChange() {
	glog.V(4).Infof("Reconciling ingresses on cluster change")
	if !ic.isSynced() {
		ic.clusterDeliverer.DeliverAt(allClustersKey, nil, time.Now().Add(ic.clusterAvailableDelay))
	}
	for _, obj := range ic.ingressInformerStore.List() {
		ingress := obj.(*extensions_v1beta1.Ingress)
		ic.deliverIngress(types.NamespacedName{Namespace: ingress.Namespace, Name: ingress.Name}, ic.smallDelay, false)
	}
}

func (ic *IngressController) reconcileIngress(ingress types.NamespacedName) {
	glog.V(4).Infof("Reconciling ingress %q", ingress)
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
	}

	operations := make([]util.FederatedOperation, 0)

	for clusterIndex, cluster := range clusters {
		_, baseIPExists := baseIngress.ObjectMeta.Annotations[staticIPAnnotationKey]
		clusterIngressObj, found, err := ic.ingressFederatedInformer.GetTargetStore().GetByKey(cluster.Name, key)
		if err != nil {
			glog.Errorf("Failed to get %s from %s: %v", ingress, cluster.Name, err)
			ic.deliverIngress(ingress, 0, true)
			return
		}
		desiredIngress := &extensions_v1beta1.Ingress{
			ObjectMeta: baseIngress.ObjectMeta,
			Spec:       baseIngress.Spec,
		}

		if !found {
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
			if baseIPExists || (clusterIndex == 0) {
				operations = append(operations, util.FederatedOperation{
					Type:        util.OperationTypeAdd,
					Obj:         desiredIngress,
					ClusterName: cluster.Name,
				})
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
				// TODO: In some cases Ingress controllers in the clusters add annotations, so we ideally need to exclude those from
				// the equivalence comparison to cut down on unnecessary updates.
				glog.V(4).Infof("Ingress %s in cluster %s needs an update: cluster ingress %v is not equivalent to federated ingress %v", ingress, cluster.Name, clusterIngress, desiredIngress)
				// We need to use server-created fields from the cluster, not the desired object when updating.
				desiredIngress.ObjectMeta.ResourceVersion = clusterIngress.ObjectMeta.ResourceVersion
				desiredIngress.ObjectMeta.UID = clusterIngress.ObjectMeta.UID
				// Merge any annotations on the federated ingress onto the underlying cluster ingress,
				// overwriting duplicates.
				// TODO: We should probably use a PATCH operation for this instead.
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
			}
		}
	}

	if len(operations) == 0 {
		// Everything is in order
		return
	}
	glog.V(4).Infof("Calling federatedUpdater.Update() - operations: %v", operations)
	err = ic.federatedUpdater.UpdateWithOnError(operations, ic.updateTimeout, func(op util.FederatedOperation, operror error) {
		ic.eventRecorder.Eventf(baseIngress, api.EventTypeNormal, "FailedUpdateInCluster",
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
