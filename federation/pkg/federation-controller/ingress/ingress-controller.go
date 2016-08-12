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
	// api_v1 "k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/apis/extensions"
	extensions_v1beta1 "k8s.io/kubernetes/pkg/apis/extensions/v1beta1"
	"k8s.io/kubernetes/pkg/client/cache"
	"k8s.io/kubernetes/pkg/controller"
	"k8s.io/kubernetes/pkg/controller/framework"
	pkg_runtime "k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/watch"

	"github.com/golang/glog"
)

const (
	IngressReviewDelay    = time.Second * 10
	ClusterAvailableDelay = time.Second * 20
	SmallDelay            = time.Second * 3
	UpdateTimeout         = time.Second * 30

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

	stopChan chan struct{}
}

// A structure passed by delaying deliverer. It contains an ingress that should be reconciled and
// the number of previous attempts that ended up in some kind of ingress-related
// error (like failure to create).
type ingressItem struct {
	ingress      string
	prevAttempts int64
}

// NewIngressController returns a new ingress controller
func NewIngressController(client federation_release_1_4.Interface) *IngressController {
	ic := &IngressController{
		federatedApiClient: client,
		stopChan:           make(chan struct{}),
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
		&extensions.Ingress{},
		controller.NoResyncPeriodFunc(),
		util.NewTriggerOnAllChanges(func(obj pkg_runtime.Object) { ic.deliverIngressObj(obj, 0, 0) }))

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
				&extensions.Ingress{},
				controller.NoResyncPeriodFunc(),
				// Trigger reconcilation whenever something in federated cluster is changed. In most cases it
				// would be just confirmation that some ingress operation suceeded.
				util.NewTriggerOnMetaAndSpecChangesPreproc(
					func(obj pkg_runtime.Object) { ic.deliverIngressObj(obj, IngressReviewDelay, 0) },
					func(obj pkg_runtime.Object) { util.SetClusterName(obj, cluster.Name) },
				))
		},

		&util.ClusterLifecycleHandlerFuncs{
			ClusterAvailable: func(cluster *federation_api.Cluster) {
				// When new cluster becomes available process all the ingresses again.
				ic.clusterDeliverer.DeliverAt(allClustersKey, nil, time.Now().Add(ClusterAvailableDelay))
			},
		},
	)

	// Federated updater along with Create/Update/Delete operations.
	ic.federatedUpdater = util.NewFederatedUpdater(ic.ingressFederatedInformer,
		func(client federation_release_1_4.Interface, obj pkg_runtime.Object) error {
			ingress := obj.(*extensions_v1beta1.Ingress)
			_, err := client.Extensions().Ingresses(api.NamespaceAll).Create(ingress)
			return err
		},
		func(client federation_release_1_4.Interface, obj pkg_runtime.Object) error {
			ingress := obj.(*extensions_v1beta1.Ingress)
			_, err := client.Extensions().Ingresses(api.NamespaceAll).Update(ingress)
			return err
		},
		func(client federation_release_1_4.Interface, obj pkg_runtime.Object) error {
			ingress := obj.(*extensions_v1beta1.Ingress)
			err := client.Extensions().Ingresses(api.NamespaceAll).Delete(ingress.Name, &api.DeleteOptions{})
			return err
		})
	return ic
}

func (ic *IngressController) Start() {
	ic.ingressInformerController.Run(ic.stopChan)
	ic.ingressFederatedInformer.Start()
	ic.ingressDeliverer.StartWithHandler(func(item *util.DelayingDelivererItem) {
		i := item.Value.(*ingressItem)
		ic.reconcileIngress(i.ingress, i.prevAttempts)
	})
	ic.clusterDeliverer.StartWithHandler(func(_ *util.DelayingDelivererItem) {
		ic.reconcileIngressesOnClusterChange()
	})
}

func (ic *IngressController) Stop() {
	ic.ingressFederatedInformer.Stop()
	close(ic.stopChan)
}

func (ic *IngressController) deliverIngressObj(obj interface{}, delay time.Duration, trial int64) {
	ingress := obj.(*extensions.Ingress)
	ic.deliverIngress(ingress.Name, delay, trial)
}

func (ic *IngressController) deliverIngress(ingress string, delay time.Duration, prevAttempts int64) {
	ic.ingressDeliverer.DeliverAfter(ingress, &ingressItem{ingress: ingress, prevAttempts: prevAttempts}, delay)
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
		return false
	}
	return true
}

// The function triggers reconcilation of all federated ingresses.
func (ic *IngressController) reconcileIngressesOnClusterChange() {
	if !ic.isSynced() {
		ic.clusterDeliverer.DeliverAt(allClustersKey, nil, time.Now().Add(ClusterAvailableDelay))
	}
	for _, obj := range ic.ingressInformerStore.List() {
		ingress := obj.(*extensions.Ingress)
		ic.deliverIngress(ingress.Name, SmallDelay, 0)
	}
}

func backoff(trial int64) time.Duration {
	if trial > 12 {
		return 12 * 5 * time.Second
	}
	return time.Duration(trial) * 5 * time.Second
}

func (ic *IngressController) reconcileIngress(ingress string, trial int64) {
	if !ic.isSynced() {
		ic.deliverIngress(ingress, ClusterAvailableDelay, trial)
	}

	baseIngressObj, exist, err := ic.ingressInformerStore.GetByKey(ingress)
	if err != nil {
		glog.Errorf("Failed to query main ingress store for %v: %v", ingress, err)
		ic.deliverIngress(ingress, backoff(trial+1), trial+1)
		return
	}
	if !exist {
		// Not federated ingress, ignoring.
		return
	}
	baseIngress := baseIngressObj.(*extensions.Ingress)

	/* TODO:  What to do for Ingresses? - this applied to namespaces.
	if baseIngress.Status.Phase == extensions.IngressTerminating {
		// TODO: What about ingresses in subclusters ???
		err = ic.federatedApiClient.Extensions().Ingresses(api.NamespaceAll).Delete(baseIngress.Name, &api.DeleteOptions{})
		if err != nil {
			glog.Errorf("Failed to delete ingress %s: %v", baseIngress.Name, err)
			ic.deliverIngress(ingress, backoff(trial+1), trial+1)
		}
		return
	}
	*/

	clusters, err := ic.ingressFederatedInformer.GetReadyClusters()
	if err != nil {
		glog.Errorf("Failed to get cluster list: %v", err)
		ic.deliverIngress(ingress, ClusterAvailableDelay, trial)
		return
	}

	operations := make([]util.FederatedOperation, 0)

	for _, cluster := range clusters {
		clusterIngressObj, found, err := ic.ingressFederatedInformer.GetTargetStore().GetByKey(cluster.Name, ingress)
		if err != nil {
			glog.Errorf("Failed to get %s from %s: %v", ingress, cluster.Name, err)
			ic.deliverIngress(ingress, backoff(trial+1), trial+1)
			return
		}
		desiredIngress := &extensions.Ingress{
			ObjectMeta: baseIngress.ObjectMeta,
			Spec:       baseIngress.Spec,
		}
		util.SetClusterName(desiredIngress, cluster.Name)

		if !found {
			operations = append(operations, util.FederatedOperation{
				Type: util.OperationTypeAdd,
				Obj:  desiredIngress,
			})
		} else {
			clusterIngress := clusterIngressObj.(*extensions.Ingress)
			// Update existing ingress, if needed.
			if !reflect.DeepEqual(desiredIngress.ObjectMeta, clusterIngress.ObjectMeta) ||
				!reflect.DeepEqual(desiredIngress.Spec, clusterIngress.Spec) {
				operations = append(operations, util.FederatedOperation{
					Type: util.OperationTypeUpdate,
					Obj:  desiredIngress,
				})
			}
		}
	}

	if len(operations) == 0 {
		// Everything is in order
		return
	}
	err = ic.federatedUpdater.Update(operations, UpdateTimeout)
	if err != nil {
		glog.Errorf("Failed to execute updates for %s: %v", ingress, err)
		ic.deliverIngress(ingress, backoff(trial+1), trial+1)
		return
	}

	// Evertyhing is in order but lets be double sure
	ic.deliverIngress(ingress, IngressReviewDelay, 0)
}
