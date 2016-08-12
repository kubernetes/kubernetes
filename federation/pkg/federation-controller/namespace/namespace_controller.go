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

package namespace

import (
	"reflect"
	"time"

	federation_api "k8s.io/kubernetes/federation/apis/federation/v1beta1"
	federation_release_1_4 "k8s.io/kubernetes/federation/client/clientset_generated/federation_release_1_4"
	"k8s.io/kubernetes/federation/pkg/federation-controller/util"
	"k8s.io/kubernetes/pkg/api"
	api_v1 "k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/client/cache"
	"k8s.io/kubernetes/pkg/controller"
	"k8s.io/kubernetes/pkg/controller/framework"
	pkg_runtime "k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/watch"

	"github.com/golang/glog"
)

const (
	allClustersKey = "ALL_CLUSTERS"
)

type NamespaceController struct {
	// For triggering single namespace reconcilation. This is used when there is an
	// add/update/delete operation on a namespace in either federated API server or
	// in some member of the federation.
	namespaceDeliverer *util.DelayingDeliverer

	// For triggering all namespaces reconcilation. This is used when
	// a new cluster becomes available.
	clusterDeliverer *util.DelayingDeliverer

	// Contains namespaces present in members of federation.
	namespaceFederatedInformer util.FederatedInformer
	// For updating members of federation.
	federatedUpdater util.FederatedUpdater
	// Definitions of namespaces that should be federated.
	namespaceInformerStore cache.Store
	// Informer controller for namespaces that should be federated.
	namespaceInformerController framework.ControllerInterface

	// Client to federated api server.
	federatedApiClient federation_release_1_4.Interface

	stopChan chan struct{}

	namespaceReviewDelay  time.Duration
	clusterAvailableDelay time.Duration
	smallDelay            time.Duration
	updateTimeout         time.Duration
}

// A structure passed by delying deliver. It contains a namespace that should be reconciled and
// the number of trials that were made previously and ended up in some kind of namespace-related
// error (like failure to create).
type namespaceItem struct {
	namespace string
	trial     int64
}

// NewNamespaceController returns a new namespace controller
func NewNamespaceController(client federation_release_1_4.Interface) *NamespaceController {
	nc := &NamespaceController{
		federatedApiClient:    client,
		stopChan:              make(chan struct{}),
		namespaceReviewDelay:  time.Second * 10,
		clusterAvailableDelay: time.Second * 20,
		smallDelay:            time.Second * 3,
		updateTimeout:         time.Second * 30,
	}

	// Build delivereres for triggering reconcilations.
	nc.namespaceDeliverer = util.NewDelayingDeliverer()
	nc.clusterDeliverer = util.NewDelayingDeliverer()

	// Start informer in federated API servers on namespaces that should be federated.
	nc.namespaceInformerStore, nc.namespaceInformerController = framework.NewInformer(
		&cache.ListWatch{
			ListFunc: func(options api.ListOptions) (pkg_runtime.Object, error) {
				return client.Core().Namespaces().List(options)
			},
			WatchFunc: func(options api.ListOptions) (watch.Interface, error) {
				return client.Core().Namespaces().Watch(options)
			},
		},
		&api_v1.Namespace{},
		controller.NoResyncPeriodFunc(),
		util.NewTriggerOnAllChanges(func(obj pkg_runtime.Object) { nc.deliverNamespaceObj(obj, 0, 0) }))

	// Federated informer on namespaces in members of federation.
	nc.namespaceFederatedInformer = util.NewFederatedInformer(
		client,
		func(cluster *federation_api.Cluster, targetClient federation_release_1_4.Interface) (cache.Store, framework.ControllerInterface) {
			return framework.NewInformer(
				&cache.ListWatch{
					ListFunc: func(options api.ListOptions) (pkg_runtime.Object, error) {
						return targetClient.Core().Namespaces().List(options)
					},
					WatchFunc: func(options api.ListOptions) (watch.Interface, error) {
						return targetClient.Core().Namespaces().Watch(options)
					},
				},
				&api_v1.Namespace{},
				controller.NoResyncPeriodFunc(),
				// Trigger reconcilation whenever something in federated cluster is changed. In most cases it
				// would be just confirmation that some namespace opration suceeded.
				util.NewTriggerOnMetaAndSpecChangesPreproc(
					func(obj pkg_runtime.Object) { nc.deliverNamespaceObj(obj, nc.namespaceReviewDelay, 0) },
					func(obj pkg_runtime.Object) { util.SetClusterName(obj, cluster.Name) },
				))
		},

		&util.ClusterLifecycleHandlerFuncs{
			ClusterAvailable: func(cluster *federation_api.Cluster) {
				// When new cluster becomes available process all the namespaces again.
				nc.clusterDeliverer.DeliverAt(allClustersKey, nil, time.Now().Add(nc.clusterAvailableDelay))
			},
		},
	)

	// Federated updeater along with Create/Update/Delete operations.
	nc.federatedUpdater = util.NewFederatedUpdater(nc.namespaceFederatedInformer,
		func(client federation_release_1_4.Interface, obj pkg_runtime.Object) error {
			namespace := obj.(*api_v1.Namespace)
			_, err := client.Core().Namespaces().Create(namespace)
			return err
		},
		func(client federation_release_1_4.Interface, obj pkg_runtime.Object) error {
			namespace := obj.(*api_v1.Namespace)
			_, err := client.Core().Namespaces().Update(namespace)
			return err
		},
		func(client federation_release_1_4.Interface, obj pkg_runtime.Object) error {
			namespace := obj.(*api_v1.Namespace)
			err := client.Core().Namespaces().Delete(namespace.Name, &api.DeleteOptions{})
			return err
		})
	return nc
}

func (nc *NamespaceController) Start() {
	go nc.namespaceInformerController.Run(nc.stopChan)
	nc.namespaceFederatedInformer.Start()
	nc.namespaceDeliverer.StartWithHandler(func(item *util.DelayingDelivererItem) {
		ni := item.Value.(*namespaceItem)
		nc.reconcileNamespace(ni.namespace, ni.trial)
	})
	nc.clusterDeliverer.StartWithHandler(func(_ *util.DelayingDelivererItem) {
		nc.reconcileNamespacesOnClusterChange()
	})
}

func (nc *NamespaceController) Stop() {
	nc.namespaceFederatedInformer.Stop()
	close(nc.stopChan)
}

func (nc *NamespaceController) deliverNamespaceObj(obj interface{}, delay time.Duration, trial int64) {
	namespace := obj.(*api_v1.Namespace)
	nc.deliverNamespace(namespace.Name, delay, trial)
}

func (nc *NamespaceController) deliverNamespace(namespace string, delay time.Duration, trial int64) {
	nc.namespaceDeliverer.DeliverAfter(namespace, &namespaceItem{namespace: namespace, trial: trial}, delay)
}

// Check whether all data stores are in sync. False is returned if any of the informer/stores is not yet
// synced with the coresponding api server.
func (nc *NamespaceController) isSynced() bool {
	if !nc.namespaceFederatedInformer.ClustersSynced() {
		glog.V(2).Infof("Cluster list not synced")
		return false
	}
	clusters, err := nc.namespaceFederatedInformer.GetReadyClusters()
	if err != nil {
		glog.Errorf("Failed to get ready clusters: %v", err)
		return false
	}
	if !nc.namespaceFederatedInformer.GetTargetStore().ClustersSynced(clusters) {
		return false
	}
	return true
}

// The function triggers reconcilation of all federated namespaces.
func (nc *NamespaceController) reconcileNamespacesOnClusterChange() {
	if !nc.isSynced() {
		nc.clusterDeliverer.DeliverAt(allClustersKey, nil, time.Now().Add(nc.clusterAvailableDelay))
	}
	for _, obj := range nc.namespaceInformerStore.List() {
		namespace := obj.(*api_v1.Namespace)
		nc.deliverNamespace(namespace.Name, nc.smallDelay, 0)
	}
}

func backoff(trial int64) time.Duration {
	if trial > 12 {
		return 12 * 5 * time.Second
	}
	return time.Duration(trial) * 5 * time.Second
}

func (nc *NamespaceController) reconcileNamespace(namespace string, trial int64) {
	if !nc.isSynced() {
		nc.deliverNamespace(namespace, nc.clusterAvailableDelay, trial)
	}

	baseNamespaceObj, exist, err := nc.namespaceInformerStore.GetByKey(namespace)
	if err != nil {
		glog.Errorf("Failed to query main namespace store for %v: %v", namespace, err)
		nc.deliverNamespace(namespace, backoff(trial+1), trial+1)
		return
	}

	if !exist {
		// Not federated namespace, ignoring.
		return
	}
	baseNamespace := baseNamespaceObj.(*api_v1.Namespace)
	if baseNamespace.Status.Phase == api_v1.NamespaceTerminating {
		// TODO: What about namespaces in subclusters ???
		err = nc.federatedApiClient.Core().Namespaces().Delete(baseNamespace.Name, &api.DeleteOptions{})
		if err != nil {
			glog.Errorf("Failed to delete namespace %s: %v", baseNamespace.Name, err)
			nc.deliverNamespace(namespace, backoff(trial+1), trial+1)
		}
		return
	}

	clusters, err := nc.namespaceFederatedInformer.GetReadyClusters()
	if err != nil {
		glog.Errorf("Failed to get cluster list: %v", err)
		nc.deliverNamespace(namespace, nc.clusterAvailableDelay, trial)
		return
	}

	operations := make([]util.FederatedOperation, 0)
	for _, cluster := range clusters {
		clusterNamespaceObj, found, err := nc.namespaceFederatedInformer.GetTargetStore().GetByKey(cluster.Name, namespace)
		if err != nil {
			glog.Errorf("Failed to get %s from %s: %v", namespace, cluster.Name, err)
			nc.deliverNamespace(namespace, backoff(trial+1), trial+1)
			return
		}
		desiredNamespace := &api_v1.Namespace{
			ObjectMeta: baseNamespace.ObjectMeta,
			Spec:       baseNamespace.Spec,
		}
		util.SetClusterName(desiredNamespace, cluster.Name)

		if !found {
			operations = append(operations, util.FederatedOperation{
				Type: util.OperationTypeAdd,
				Obj:  desiredNamespace,
			})
		} else {
			clusterNamespace := clusterNamespaceObj.(*api_v1.Namespace)

			// Update existing namespace, if needed.
			if !reflect.DeepEqual(desiredNamespace.ObjectMeta, clusterNamespace.ObjectMeta) ||
				!reflect.DeepEqual(desiredNamespace.Spec, clusterNamespace.Spec) {
				operations = append(operations, util.FederatedOperation{
					Type: util.OperationTypeUpdate,
					Obj:  desiredNamespace,
				})
			}
		}
	}

	if len(operations) == 0 {
		// Everything is in order
		return
	}
	err = nc.federatedUpdater.Update(operations, nc.updateTimeout)
	if err != nil {
		glog.Errorf("Failed to execute updates for %s: %v", namespace, err)
		nc.deliverNamespace(namespace, backoff(trial+1), trial+1)
		return
	}

	// Evertyhing is in order but lets be double sure
	nc.deliverNamespace(namespace, nc.namespaceReviewDelay, 0)
}
