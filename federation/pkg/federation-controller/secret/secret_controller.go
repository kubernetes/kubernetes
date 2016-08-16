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

package secret

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

type SecretController struct {
	// For triggering single secret reconcilation. This is used when there is an
	// add/update/delete operation on a secret in either federated API server or
	// in some member of the federation.
	secretDeliverer *util.DelayingDeliverer

	// For triggering all secrets reconcilation. This is used when
	// a new cluster becomes available.
	clusterDeliverer *util.DelayingDeliverer

	// Contains secrets present in members of federation.
	secretFederatedInformer util.FederatedInformer
	// For updating members of federation.
	federatedUpdater util.FederatedUpdater
	// Definitions of secrets that should be federated.
	secretInformerStore cache.Store
	// Informer controller for secrets that should be federated.
	secretInformerController framework.ControllerInterface

	// Client to federated api server.
	federatedApiClient federation_release_1_4.Interface

	stopChan chan struct{}

	secretReviewDelay     time.Duration
	clusterAvailableDelay time.Duration
	smallDelay            time.Duration
	updateTimeout         time.Duration
}

// A structure passed by delaying deliverer. It contains a secret that should be reconciled and
// the number of trials that were made previously and ended up in some kind of secret-related
// error (like failure to create).
type secretItem struct {
	secret string
	trial  int64
}

// NewSecretController returns a new secret controller
func NewSecretController(client federation_release_1_4.Interface) *SecretController {
	secretcontroller := &SecretController{
		federatedApiClient:    client,
		stopChan:              make(chan struct{}),
		secretReviewDelay:     time.Second * 10,
		clusterAvailableDelay: time.Second * 20,
		smallDelay:            time.Second * 3,
		updateTimeout:         time.Second * 30,
	}

	// Build delivereres for triggering reconcilations.
	secretcontroller.secretDeliverer = util.NewDelayingDeliverer()
	secretcontroller.clusterDeliverer = util.NewDelayingDeliverer()

	// Start informer in federated API servers on secrets that should be federated.
	secretcontroller.secretInformerStore, secretcontroller.secretInformerController = framework.NewInformer(
		&cache.ListWatch{
			ListFunc: func(options api.ListOptions) (pkg_runtime.Object, error) {
				return client.Core().Secrets(api_v1.NamespaceAll).List(options)
			},
			WatchFunc: func(options api.ListOptions) (watch.Interface, error) {
				return client.Core().Secrets(api_v1.NamespaceAll).Watch(options)
			},
		},
		&api_v1.Secret{},
		controller.NoResyncPeriodFunc(),
		util.NewTriggerOnAllChanges(func(obj pkg_runtime.Object) { secretcontroller.deliverSecretObj(obj, 0, 0) }))

	// Federated informer on secrets in members of federation.
	secretcontroller.secretFederatedInformer = util.NewFederatedInformer(
		client,
		func(cluster *federation_api.Cluster, targetClient federation_release_1_4.Interface) (cache.Store, framework.ControllerInterface) {
			return framework.NewInformer(
				&cache.ListWatch{
					ListFunc: func(options api.ListOptions) (pkg_runtime.Object, error) {
						return targetClient.Core().Secrets(api_v1.NamespaceAll).List(options)
					},
					WatchFunc: func(options api.ListOptions) (watch.Interface, error) {
						return targetClient.Core().Secrets(api_v1.NamespaceAll).Watch(options)
					},
				},
				&api_v1.Secret{},
				controller.NoResyncPeriodFunc(),
				// Trigger reconcilation whenever something in federated cluster is changed. In most cases it
				// would be just confirmation that some secret opration suceeded.
				util.NewTriggerOnChangesPreproc(
					func(obj pkg_runtime.Object) {
						secretcontroller.deliverSecretObj(obj, secretcontroller.secretReviewDelay, 0)
					},
					func(obj pkg_runtime.Object) { util.SetClusterName(obj, cluster.Name) },
				))
		},

		&util.ClusterLifecycleHandlerFuncs{
			ClusterAvailable: func(cluster *federation_api.Cluster) {
				// When new cluster becomes available process all the secrets again.
				secretcontroller.clusterDeliverer.DeliverAt(allClustersKey, nil, time.Now().Add(secretcontroller.clusterAvailableDelay))
			},
		},
	)

	// Federated updeater along with Create/Update/Delete operations.
	secretcontroller.federatedUpdater = util.NewFederatedUpdater(secretcontroller.secretFederatedInformer,
		func(client federation_release_1_4.Interface, obj pkg_runtime.Object) error {
			secret := obj.(*api_v1.Secret)
			_, err := client.Core().Secrets(secret.Namespace).Create(secret)
			return err
		},
		func(client federation_release_1_4.Interface, obj pkg_runtime.Object) error {
			secret := obj.(*api_v1.Secret)
			_, err := client.Core().Secrets(secret.Namespace).Update(secret)
			return err
		},
		func(client federation_release_1_4.Interface, obj pkg_runtime.Object) error {
			secret := obj.(*api_v1.Secret)
			err := client.Core().Secrets(secret.Namespace).Delete(secret.Name, &api.DeleteOptions{})
			return err
		})
	return secretcontroller
}

func (secretcontroller *SecretController) Start() {
	go secretcontroller.secretInformerController.Run(secretcontroller.stopChan)
	secretcontroller.secretFederatedInformer.Start()
	secretcontroller.secretDeliverer.StartWithHandler(func(item *util.DelayingDelivererItem) {
		si := item.Value.(*secretItem)
		secretcontroller.reconcileSecret(si.secret, si.trial)
	})
	secretcontroller.clusterDeliverer.StartWithHandler(func(_ *util.DelayingDelivererItem) {
		secretcontroller.reconcileSecretsOnClusterChange()
	})
}

func (secretcontroller *SecretController) Stop() {
	secretcontroller.secretFederatedInformer.Stop()
	close(secretcontroller.stopChan)
}

func (secretcontroller *SecretController) deliverSecretObj(obj interface{}, delay time.Duration, trial int64) {
	secret := obj.(*api_v1.Secret)
	secretcontroller.deliverSecret(secret.Name, delay, trial)
}

func (secretcontroller *SecretController) deliverSecret(secret string, delay time.Duration, trial int64) {
	secretcontroller.secretDeliverer.DeliverAfter(secret, &secretItem{secret: secret, trial: trial}, delay)
}

// Check whether all data stores are in sync. False is returned if any of the informer/stores is not yet
// synced with the coresponding api server.
func (secretcontroller *SecretController) isSynced() bool {
	if !secretcontroller.secretFederatedInformer.ClustersSynced() {
		glog.V(2).Infof("Cluster list not synced")
		return false
	}
	clusters, err := secretcontroller.secretFederatedInformer.GetReadyClusters()
	if err != nil {
		glog.Errorf("Failed to get ready clusters: %v", err)
		return false
	}
	if !secretcontroller.secretFederatedInformer.GetTargetStore().ClustersSynced(clusters) {
		return false
	}
	return true
}

// The function triggers reconcilation of all federated secrets.
func (secretcontroller *SecretController) reconcileSecretsOnClusterChange() {
	if !secretcontroller.isSynced() {
		secretcontroller.clusterDeliverer.DeliverAt(allClustersKey, nil, time.Now().Add(secretcontroller.clusterAvailableDelay))
	}
	for _, obj := range secretcontroller.secretInformerStore.List() {
		secret := obj.(*api_v1.Secret)
		secretcontroller.deliverSecret(secret.Name, secretcontroller.smallDelay, 0)
	}
}

func backoff(trial int64) time.Duration {
	if trial > 12 {
		return 12 * 5 * time.Second
	}
	return time.Duration(trial) * 5 * time.Second
}

func (secretcontroller *SecretController) reconcileSecret(secret string, trial int64) {
	if !secretcontroller.isSynced() {
		secretcontroller.deliverSecret(secret, secretcontroller.clusterAvailableDelay, trial)
	}

	baseSecretObj, exist, err := secretcontroller.secretInformerStore.GetByKey(secret)
	if err != nil {
		glog.Errorf("Failed to query main secret store for %v: %v", secret, err)
		secretcontroller.deliverSecret(secret, backoff(trial+1), trial+1)
		return
	}

	if !exist {
		// Not federated secret, ignoring.
		return
	}
	baseSecret := baseSecretObj.(*api_v1.Secret)

	clusters, err := secretcontroller.secretFederatedInformer.GetReadyClusters()
	if err != nil {
		glog.Errorf("Failed to get cluster list: %v", err)
		secretcontroller.deliverSecret(secret, secretcontroller.clusterAvailableDelay, trial)
		return
	}

	operations := make([]util.FederatedOperation, 0)
	for _, cluster := range clusters {
		clusterSecretObj, found, err := secretcontroller.secretFederatedInformer.GetTargetStore().GetByKey(cluster.Name, secret)
		if err != nil {
			glog.Errorf("Failed to get %s from %s: %v", secret, cluster.Name, err)
			secretcontroller.deliverSecret(secret, backoff(trial+1), trial+1)
			return
		}

		desiredSecret := &api_v1.Secret{
			ObjectMeta: baseSecret.ObjectMeta,
		}
		util.SetClusterName(desiredSecret, cluster.Name)

		if !found {
			operations = append(operations, util.FederatedOperation{
				Type: util.OperationTypeAdd,
				Obj:  desiredSecret,
			})
		} else {
			clusterSecret := clusterSecretObj.(*api_v1.Secret)

			// Update existing secret, if needed.
			if !reflect.DeepEqual(desiredSecret.ObjectMeta, clusterSecret.ObjectMeta) {
				operations = append(operations, util.FederatedOperation{
					Type: util.OperationTypeUpdate,
					Obj:  desiredSecret,
				})
			}
		}
	}

	if len(operations) == 0 {
		// Everything is in order
		return
	}
	err = secretcontroller.federatedUpdater.Update(operations, secretcontroller.updateTimeout)
	if err != nil {
		glog.Errorf("Failed to execute updates for %s: %v", secret, err)
		secretcontroller.deliverSecret(secret, backoff(trial+1), trial+1)
		return
	}

	// Evertyhing is in order but lets be double sure
	secretcontroller.deliverSecret(secret, secretcontroller.secretReviewDelay, 0)
}
