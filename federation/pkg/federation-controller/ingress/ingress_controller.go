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
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/types"
	// api_v1 "k8s.io/kubernetes/pkg/api/v1"
	// "k8s.io/kubernetes/pkg/apis/extensions"
	extensions_v1beta1 "k8s.io/kubernetes/pkg/apis/extensions/v1beta1"
	"k8s.io/kubernetes/pkg/client/cache"
	"k8s.io/kubernetes/pkg/controller"
	"k8s.io/kubernetes/pkg/controller/framework"
	pkg_runtime "k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/util/flowcontrol"
	"k8s.io/kubernetes/pkg/watch"

	"github.com/golang/glog"
)

const (
	allClustersKey = "ALL_CLUSTERS"
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

	ingressReviewDelay    time.Duration
	clusterAvailableDelay time.Duration
	smallDelay            time.Duration
	updateTimeout         time.Duration
}

// A structure passed by delaying deliverer. It contains an ingress that should be reconciled and
// the number of previous attempts that ended up in some kind of ingress-related
// error (like failure to create).
/*
type ingressItem struct {
	ingress      string
	prevAttempts int64
}
*/

// NewIngressController returns a new ingress controller
func NewIngressController(client federation_release_1_4.Interface) *IngressController {
	ic := &IngressController{
		federatedApiClient:    client,
		ingressReviewDelay:    time.Second * 10,
		clusterAvailableDelay: time.Second * 20,
		smallDelay:            time.Second * 3,
		updateTimeout:         time.Second * 30,
		ingressBackoff:        flowcontrol.NewBackOff(5*time.Second, time.Minute),
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
		func(cluster *federation_api.Cluster, targetClient federation_release_1_4.Interface) (cache.Store, framework.ControllerInterface) {
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
		func(client federation_release_1_4.Interface, obj pkg_runtime.Object) error {
			ingress := obj.(*extensions_v1beta1.Ingress)
			glog.V(4).Infof("Attempting to create Ingress: %v", ingress)
			_, err := client.Extensions().Ingresses(ingress.Namespace).Create(ingress)
			return err
		},
		func(client federation_release_1_4.Interface, obj pkg_runtime.Object) error {
			ingress := obj.(*extensions_v1beta1.Ingress)
			glog.V(4).Infof("Attempting to update Ingress: %v", ingress)
			_, err := client.Extensions().Ingresses(ingress.Namespace).Update(ingress)
			return err
		},
		func(client federation_release_1_4.Interface, obj pkg_runtime.Object) error {
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
	ic.deliverIngress(types.NamespacedName{ingress.Namespace, ingress.Name}, delay, failed)
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

	for _, cluster := range clusters {
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
			operations = append(operations, util.FederatedOperation{
				Type:        util.OperationTypeAdd,
				Obj:         desiredIngress,
				ClusterName: cluster.Name,
			})
		} else {
			clusterIngress := clusterIngressObj.(*extensions_v1beta1.Ingress)
			glog.V(4).Infof("Found existing Ingress %s in cluster %s - checking if update is required", ingress, cluster.Name)
			// Update existing ingress, if needed.
			if !util.ObjectMetaIsEquivalent(desiredIngress.ObjectMeta, clusterIngress.ObjectMeta) ||
				!reflect.DeepEqual(desiredIngress.Spec, clusterIngress.Spec) {
				glog.V(4).Infof("Ingress %s in cluster %s needs an update: cluster ingress %v is not equivalent to federated ingress %v", ingress, cluster.Name, clusterIngress, desiredIngress)
				// We need to use server-created fields from the cluster, not the desired object when updating.
				desiredIngress.ObjectMeta.ResourceVersion = clusterIngress.ObjectMeta.ResourceVersion
				desiredIngress.ObjectMeta.UID = clusterIngress.ObjectMeta.UID

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
	err = ic.federatedUpdater.Update(operations, ic.updateTimeout)
	if err != nil {
		glog.Errorf("Failed to execute updates for %s: %v", ingress, err)
		ic.deliverIngress(ingress, ic.ingressReviewDelay, true)
		return
	}

	// Evertyhing is in order but lets be double sure
	ic.deliverIngress(ingress, ic.ingressReviewDelay, false)
}
