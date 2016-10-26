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
	"fmt"
	"time"

	federation_api "k8s.io/kubernetes/federation/apis/federation/v1beta1"
	federationclientset "k8s.io/kubernetes/federation/client/clientset_generated/federation_release_1_5"
	"k8s.io/kubernetes/federation/pkg/federation-controller/util"
	"k8s.io/kubernetes/federation/pkg/federation-controller/util/eventsink"
	"k8s.io/kubernetes/pkg/api"
	api_v1 "k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/client/cache"
	kubeclientset "k8s.io/kubernetes/pkg/client/clientset_generated/release_1_5"
	"k8s.io/kubernetes/pkg/client/record"
	"k8s.io/kubernetes/pkg/controller"
	pkg_runtime "k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/util/flowcontrol"
	"k8s.io/kubernetes/pkg/watch"

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
	configmapInformerController cache.ControllerInterface

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
	recorder := broadcaster.NewRecorder(api.EventSource{Component: "federated-configmaps-controller"})

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

	// Start informer in federated API servers on configmaps that should be federated.
	configmapcontroller.configmapInformerStore, configmapcontroller.configmapInformerController = cache.NewInformer(
		&cache.ListWatch{
			ListFunc: func(options api.ListOptions) (pkg_runtime.Object, error) {
				versionedOptions := util.VersionizeV1ListOptions(options)
				return client.Core().ConfigMaps(api_v1.NamespaceAll).List(versionedOptions)
			},
			WatchFunc: func(options api.ListOptions) (watch.Interface, error) {
				versionedOptions := util.VersionizeV1ListOptions(options)
				return client.Core().ConfigMaps(api_v1.NamespaceAll).Watch(versionedOptions)
			},
		},
		&api_v1.ConfigMap{},
		controller.NoResyncPeriodFunc(),
		util.NewTriggerOnAllChanges(func(obj pkg_runtime.Object) { configmapcontroller.deliverConfigMapObj(obj, 0, false) }))

	// Federated informer on configmaps in members of federation.
	configmapcontroller.configmapFederatedInformer = util.NewFederatedInformer(
		client,
		func(cluster *federation_api.Cluster, targetClient kubeclientset.Interface) (cache.Store, cache.ControllerInterface) {
			return cache.NewInformer(
				&cache.ListWatch{
					ListFunc: func(options api.ListOptions) (pkg_runtime.Object, error) {
						versionedOptions := util.VersionizeV1ListOptions(options)
						return targetClient.Core().ConfigMaps(api_v1.NamespaceAll).List(versionedOptions)
					},
					WatchFunc: func(options api.ListOptions) (watch.Interface, error) {
						versionedOptions := util.VersionizeV1ListOptions(options)
						return targetClient.Core().ConfigMaps(api_v1.NamespaceAll).Watch(versionedOptions)
					},
				},
				&api_v1.ConfigMap{},
				controller.NoResyncPeriodFunc(),
				// Trigger reconciliation whenever something in federated cluster is changed. In most cases it
				// would be just confirmation that some configmap opration succeeded.
				util.NewTriggerOnAllChanges(
					func(obj pkg_runtime.Object) {
						configmapcontroller.deliverConfigMapObj(obj, configmapcontroller.configmapReviewDelay, false)
					},
				))
		},

		&util.ClusterLifecycleHandlerFuncs{
			ClusterAvailable: func(cluster *federation_api.Cluster) {
				// When new cluster becomes available process all the configmaps again.
				configmapcontroller.clusterDeliverer.DeliverAt(allClustersKey, nil, time.Now().Add(configmapcontroller.clusterAvailableDelay))
			},
		},
	)

	// Federated updeater along with Create/Update/Delete operations.
	configmapcontroller.federatedUpdater = util.NewFederatedUpdater(configmapcontroller.configmapFederatedInformer,
		func(client kubeclientset.Interface, obj pkg_runtime.Object) error {
			configmap := obj.(*api_v1.ConfigMap)
			_, err := client.Core().ConfigMaps(configmap.Namespace).Create(configmap)
			return err
		},
		func(client kubeclientset.Interface, obj pkg_runtime.Object) error {
			configmap := obj.(*api_v1.ConfigMap)
			_, err := client.Core().ConfigMaps(configmap.Namespace).Update(configmap)
			return err
		},
		func(client kubeclientset.Interface, obj pkg_runtime.Object) error {
			configmap := obj.(*api_v1.ConfigMap)
			err := client.Core().ConfigMaps(configmap.Namespace).Delete(configmap.Name, &api_v1.DeleteOptions{})
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
		configmap := item.Value.(*configmapItem)
		configmapcontroller.reconcileConfigMap(configmap.namespace, configmap.name)
	})
	configmapcontroller.clusterDeliverer.StartWithHandler(func(_ *util.DelayingDelivererItem) {
		configmapcontroller.reconcileConfigMapsOnClusterChange()
	})
	util.StartBackoffGC(configmapcontroller.configmapBackoff, stopChan)
}

func getConfigMapKey(namespace, name string) string {
	return fmt.Sprintf("%s/%s", namespace, name)
}

// Internal structure for data in delaying deliverer.
type configmapItem struct {
	namespace string
	name      string
}

func (configmapcontroller *ConfigMapController) deliverConfigMapObj(obj interface{}, delay time.Duration, failed bool) {
	configmap := obj.(*api_v1.ConfigMap)
	configmapcontroller.deliverConfigMap(configmap.Namespace, configmap.Name, delay, failed)
}

// Adds backoff to delay if this delivery is related to some failure. Resets backoff if there was no failure.
func (configmapcontroller *ConfigMapController) deliverConfigMap(namespace string, name string, delay time.Duration, failed bool) {
	key := getConfigMapKey(namespace, name)
	if failed {
		configmapcontroller.configmapBackoff.Next(key, time.Now())
		delay = delay + configmapcontroller.configmapBackoff.Get(key)
	} else {
		configmapcontroller.configmapBackoff.Reset(key)
	}
	configmapcontroller.configmapDeliverer.DeliverAfter(key,
		&configmapItem{namespace: namespace, name: name}, delay)
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
		configmapcontroller.clusterDeliverer.DeliverAt(allClustersKey, nil, time.Now().Add(configmapcontroller.clusterAvailableDelay))
	}
	for _, obj := range configmapcontroller.configmapInformerStore.List() {
		configmap := obj.(*api_v1.ConfigMap)
		configmapcontroller.deliverConfigMap(configmap.Namespace, configmap.Name, configmapcontroller.smallDelay, false)
	}
}

func (configmapcontroller *ConfigMapController) reconcileConfigMap(namespace string, configmapName string) {

	if !configmapcontroller.isSynced() {
		configmapcontroller.deliverConfigMap(namespace, configmapName, configmapcontroller.clusterAvailableDelay, false)
		return
	}

	key := getConfigMapKey(namespace, configmapName)
	baseConfigMapObj, exist, err := configmapcontroller.configmapInformerStore.GetByKey(key)
	if err != nil {
		glog.Errorf("Failed to query main configmap store for %v: %v", key, err)
		configmapcontroller.deliverConfigMap(namespace, configmapName, 0, true)
		return
	}

	if !exist {
		// Not federated configmap, ignoring.
		return
	}
	baseConfigMap := baseConfigMapObj.(*api_v1.ConfigMap)

	clusters, err := configmapcontroller.configmapFederatedInformer.GetReadyClusters()
	if err != nil {
		glog.Errorf("Failed to get cluster list: %v", err)
		configmapcontroller.deliverConfigMap(namespace, configmapName, configmapcontroller.clusterAvailableDelay, false)
		return
	}

	operations := make([]util.FederatedOperation, 0)
	for _, cluster := range clusters {
		clusterConfigMapObj, found, err := configmapcontroller.configmapFederatedInformer.GetTargetStore().GetByKey(cluster.Name, key)
		if err != nil {
			glog.Errorf("Failed to get %s from %s: %v", key, cluster.Name, err)
			configmapcontroller.deliverConfigMap(namespace, configmapName, 0, true)
			return
		}

		desiredConfigMap := &api_v1.ConfigMap{
			ObjectMeta: util.CopyObjectMeta(baseConfigMap.ObjectMeta),
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
			clusterConfigMap := clusterConfigMapObj.(*api_v1.ConfigMap)

			// Update existing configmap, if needed.
			if !util.ConfigMapEquivalent(*desiredConfigMap, *clusterConfigMap) {

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
		return
	}
	err = configmapcontroller.federatedUpdater.UpdateWithOnError(operations, configmapcontroller.updateTimeout,
		func(op util.FederatedOperation, operror error) {
			configmapcontroller.eventRecorder.Eventf(baseConfigMap, api.EventTypeNormal, "UpdateInClusterFailed",
				"ConfigMap update in cluster %s failed: %v", op.ClusterName, operror)
		})

	if err != nil {
		glog.Errorf("Failed to execute updates for %s: %v", key, err)
		configmapcontroller.deliverConfigMap(namespace, configmapName, 0, true)
		return
	}

	// Evertyhing is in order but lets be double sure
	configmapcontroller.deliverConfigMap(namespace, configmapName, configmapcontroller.configmapReviewDelay, false)
}
