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

package configmap

import (
	"time"

	pkgruntime "k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/watch"
	federationapi "k8s.io/kubernetes/federation/apis/federation/v1beta1"
	federationclientset "k8s.io/kubernetes/federation/client/clientset_generated/federation_clientset"
	"k8s.io/kubernetes/federation/pkg/federation-controller/util"
	"k8s.io/kubernetes/federation/pkg/federation-controller/util/eventsink"
	"k8s.io/kubernetes/pkg/api"
	apiv1 "k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/client/cache"
	kubeclientset "k8s.io/kubernetes/pkg/client/clientset_generated/clientset"
	"k8s.io/kubernetes/pkg/client/record"
	"k8s.io/kubernetes/pkg/controller"
	"k8s.io/kubernetes/pkg/util/flowcontrol"

	"github.com/golang/glog"
)

const (
	allClustersKey = "ALL_CLUSTERS"
)

type ConfigMapController struct {
	// For triggering single configmap reconciliation. This is used when there is an
	// add/update/delete operation on a configmap in either federated API server or
	// in some member of the federation.
	configmapDeliverer *util.DelayingDeliverer

	// For triggering all configmaps reconciliation. This is used when
	// a new cluster becomes available.
	clusterDeliverer *util.DelayingDeliverer

	// Contains configmaps present in members of federation.
	configmapFederatedInformer util.FederatedInformer
	// For updating members of federation.
	federatedUpdater util.FederatedUpdater
	// Definitions of configmaps that should be federated.
	configmapInformerStore cache.Store
	// Informer controller for configmaps that should be federated.
	configmapInformerController cache.Controller

	// Client to federated api server.
	federatedApiClient federationclientset.Interface

	// Backoff manager for configmaps
	configmapBackoff *flowcontrol.Backoff

	// For events
	eventRecorder record.EventRecorder

	configmapReviewDelay  time.Duration
	clusterAvailableDelay time.Duration
	smallDelay            time.Duration
	updateTimeout         time.Duration
}

// NewConfigMapController returns a new configmap controller
func NewConfigMapController(client federationclientset.Interface) *ConfigMapController {
	broadcaster := record.NewBroadcaster()
	broadcaster.StartRecordingToSink(eventsink.NewFederatedEventSink(client))
	recorder := broadcaster.NewRecorder(apiv1.EventSource{Component: "federated-configmaps-controller"})

	configmapcontroller := &ConfigMapController{
		federatedApiClient:    client,
		configmapReviewDelay:  time.Second * 10,
		clusterAvailableDelay: time.Second * 20,
		smallDelay:            time.Second * 3,
		updateTimeout:         time.Second * 30,
		configmapBackoff:      flowcontrol.NewBackOff(5*time.Second, time.Minute),
		eventRecorder:         recorder,
	}

	// Build delivereres for triggering reconciliations.
	configmapcontroller.configmapDeliverer = util.NewDelayingDeliverer()
	configmapcontroller.clusterDeliverer = util.NewDelayingDeliverer()

	// Start informer on federated API servers on configmaps that should be federated.
	configmapcontroller.configmapInformerStore, configmapcontroller.configmapInformerController = cache.NewInformer(
		&cache.ListWatch{
			ListFunc: func(options apiv1.ListOptions) (pkgruntime.Object, error) {
				return client.Core().ConfigMaps(apiv1.NamespaceAll).List(options)
			},
			WatchFunc: func(options apiv1.ListOptions) (watch.Interface, error) {
				return client.Core().ConfigMaps(apiv1.NamespaceAll).Watch(options)
			},
		},
		&apiv1.ConfigMap{},
		controller.NoResyncPeriodFunc(),
		util.NewTriggerOnAllChanges(func(obj pkgruntime.Object) { configmapcontroller.deliverConfigMapObj(obj, 0, false) }))

	// Federated informer on configmaps in members of federation.
	configmapcontroller.configmapFederatedInformer = util.NewFederatedInformer(
		client,
		func(cluster *federationapi.Cluster, targetClient kubeclientset.Interface) (cache.Store, cache.Controller) {
			return cache.NewInformer(
				&cache.ListWatch{
					ListFunc: func(options apiv1.ListOptions) (pkgruntime.Object, error) {
						return targetClient.Core().ConfigMaps(apiv1.NamespaceAll).List(options)
					},
					WatchFunc: func(options apiv1.ListOptions) (watch.Interface, error) {
						return targetClient.Core().ConfigMaps(apiv1.NamespaceAll).Watch(options)
					},
				},
				&apiv1.ConfigMap{},
				controller.NoResyncPeriodFunc(),
				// Trigger reconciliation whenever something in federated cluster is changed. In most cases it
				// would be just confirmation that some configmap operation succeeded.
				util.NewTriggerOnAllChanges(
					func(obj pkgruntime.Object) {
						configmapcontroller.deliverConfigMapObj(obj, configmapcontroller.configmapReviewDelay, false)
					},
				))
		},

		&util.ClusterLifecycleHandlerFuncs{
			ClusterAvailable: func(cluster *federationapi.Cluster) {
				// When new cluster becomes available process all the configmaps again.
				configmapcontroller.clusterDeliverer.DeliverAt(allClustersKey, nil, time.Now().Add(configmapcontroller.clusterAvailableDelay))
			},
		},
	)

	// Federated updater along with Create/Update/Delete operations.
	configmapcontroller.federatedUpdater = util.NewFederatedUpdater(configmapcontroller.configmapFederatedInformer,
		func(client kubeclientset.Interface, obj pkgruntime.Object) error {
			configmap := obj.(*apiv1.ConfigMap)
			_, err := client.Core().ConfigMaps(configmap.Namespace).Create(configmap)
			return err
		},
		func(client kubeclientset.Interface, obj pkgruntime.Object) error {
			configmap := obj.(*apiv1.ConfigMap)
			_, err := client.Core().ConfigMaps(configmap.Namespace).Update(configmap)
			return err
		},
		func(client kubeclientset.Interface, obj pkgruntime.Object) error {
			configmap := obj.(*apiv1.ConfigMap)
			err := client.Core().ConfigMaps(configmap.Namespace).Delete(configmap.Name, &apiv1.DeleteOptions{})
			return err
		})
	return configmapcontroller
}

func (configmapcontroller *ConfigMapController) Run(stopChan <-chan struct{}) {
	go configmapcontroller.configmapInformerController.Run(stopChan)
	configmapcontroller.configmapFederatedInformer.Start()
	go func() {
		<-stopChan
		configmapcontroller.configmapFederatedInformer.Stop()
	}()
	configmapcontroller.configmapDeliverer.StartWithHandler(func(item *util.DelayingDelivererItem) {
		configmap := item.Value.(*types.NamespacedName)
		configmapcontroller.reconcileConfigMap(*configmap)
	})
	configmapcontroller.clusterDeliverer.StartWithHandler(func(_ *util.DelayingDelivererItem) {
		configmapcontroller.reconcileConfigMapsOnClusterChange()
	})
	util.StartBackoffGC(configmapcontroller.configmapBackoff, stopChan)
}

func (configmapcontroller *ConfigMapController) deliverConfigMapObj(obj interface{}, delay time.Duration, failed bool) {
	configmap := obj.(*apiv1.ConfigMap)
	configmapcontroller.deliverConfigMap(types.NamespacedName{Namespace: configmap.Namespace, Name: configmap.Name}, delay, failed)
}

// Adds backoff to delay if this delivery is related to some failure. Resets backoff if there was no failure.
func (configmapcontroller *ConfigMapController) deliverConfigMap(configmap types.NamespacedName, delay time.Duration, failed bool) {
	key := configmap.String()
	if failed {
		configmapcontroller.configmapBackoff.Next(key, time.Now())
		delay = delay + configmapcontroller.configmapBackoff.Get(key)
	} else {
		configmapcontroller.configmapBackoff.Reset(key)
	}
	configmapcontroller.configmapDeliverer.DeliverAfter(key, &configmap, delay)
}

// Check whether all data stores are in sync. False is returned if any of the informer/stores is not yet
// synced with the corresponding api server.
func (configmapcontroller *ConfigMapController) isSynced() bool {
	if !configmapcontroller.configmapFederatedInformer.ClustersSynced() {
		glog.V(2).Infof("Cluster list not synced")
		return false
	}
	clusters, err := configmapcontroller.configmapFederatedInformer.GetReadyClusters()
	if err != nil {
		glog.Errorf("Failed to get ready clusters: %v", err)
		return false
	}
	if !configmapcontroller.configmapFederatedInformer.GetTargetStore().ClustersSynced(clusters) {
		return false
	}
	return true
}

// The function triggers reconciliation of all federated configmaps.
func (configmapcontroller *ConfigMapController) reconcileConfigMapsOnClusterChange() {
	if !configmapcontroller.isSynced() {
		glog.V(4).Infof("Configmap controller not synced")
		configmapcontroller.clusterDeliverer.DeliverAt(allClustersKey, nil, time.Now().Add(configmapcontroller.clusterAvailableDelay))
	}
	for _, obj := range configmapcontroller.configmapInformerStore.List() {
		configmap := obj.(*apiv1.ConfigMap)
		configmapcontroller.deliverConfigMap(types.NamespacedName{Namespace: configmap.Namespace, Name: configmap.Name},
			configmapcontroller.smallDelay, false)
	}
}

func (configmapcontroller *ConfigMapController) reconcileConfigMap(configmap types.NamespacedName) {

	if !configmapcontroller.isSynced() {
		glog.V(4).Infof("Configmap controller not synced")
		configmapcontroller.deliverConfigMap(configmap, configmapcontroller.clusterAvailableDelay, false)
		return
	}

	key := configmap.String()
	baseConfigMapObj, exist, err := configmapcontroller.configmapInformerStore.GetByKey(key)
	if err != nil {
		glog.Errorf("Failed to query main configmap store for %v: %v", key, err)
		configmapcontroller.deliverConfigMap(configmap, 0, true)
		return
	}

	if !exist {
		// Not federated configmap, ignoring.
		glog.V(8).Infof("Skipping not federated config map: %s", key)
		return
	}
	baseConfigMap := baseConfigMapObj.(*apiv1.ConfigMap)

	clusters, err := configmapcontroller.configmapFederatedInformer.GetReadyClusters()
	if err != nil {
		glog.Errorf("Failed to get cluster list: %v, retrying shortly", err)
		configmapcontroller.deliverConfigMap(configmap, configmapcontroller.clusterAvailableDelay, false)
		return
	}

	operations := make([]util.FederatedOperation, 0)
	for _, cluster := range clusters {
		clusterConfigMapObj, found, err := configmapcontroller.configmapFederatedInformer.GetTargetStore().GetByKey(cluster.Name, key)
		if err != nil {
			glog.Errorf("Failed to get %s from %s: %v, retrying shortly", key, cluster.Name, err)
			configmapcontroller.deliverConfigMap(configmap, 0, true)
			return
		}

		// Do not modify data.
		desiredConfigMap := &apiv1.ConfigMap{
			ObjectMeta: util.DeepCopyRelevantObjectMeta(baseConfigMap.ObjectMeta),
			Data:       baseConfigMap.Data,
		}

		if !found {
			configmapcontroller.eventRecorder.Eventf(baseConfigMap, api.EventTypeNormal, "CreateInCluster",
				"Creating configmap in cluster %s", cluster.Name)

			operations = append(operations, util.FederatedOperation{
				Type:        util.OperationTypeAdd,
				Obj:         desiredConfigMap,
				ClusterName: cluster.Name,
			})
		} else {
			clusterConfigMap := clusterConfigMapObj.(*apiv1.ConfigMap)

			// Update existing configmap, if needed.
			if !util.ConfigMapEquivalent(desiredConfigMap, clusterConfigMap) {
				configmapcontroller.eventRecorder.Eventf(baseConfigMap, api.EventTypeNormal, "UpdateInCluster",
					"Updating configmap in cluster %s", cluster.Name)
				operations = append(operations, util.FederatedOperation{
					Type:        util.OperationTypeUpdate,
					Obj:         desiredConfigMap,
					ClusterName: cluster.Name,
				})
			}
		}
	}

	if len(operations) == 0 {
		// Everything is in order
		glog.V(8).Infof("No operations needed for %s", key)
		return
	}
	err = configmapcontroller.federatedUpdater.UpdateWithOnError(operations, configmapcontroller.updateTimeout,
		func(op util.FederatedOperation, operror error) {
			configmapcontroller.eventRecorder.Eventf(baseConfigMap, api.EventTypeNormal, "UpdateInClusterFailed",
				"ConfigMap update in cluster %s failed: %v", op.ClusterName, operror)
		})

	if err != nil {
		glog.Errorf("Failed to execute updates for %s: %v, retrying shortly", key, err)
		configmapcontroller.deliverConfigMap(configmap, 0, true)
		return
	}
}
