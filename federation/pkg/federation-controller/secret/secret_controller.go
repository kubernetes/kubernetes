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
	"fmt"
	"time"

	"k8s.io/apimachinery/pkg/api/errors"
	pkgruntime "k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/watch"
	federationapi "k8s.io/kubernetes/federation/apis/federation/v1beta1"
	federationclientset "k8s.io/kubernetes/federation/client/clientset_generated/federation_clientset"
	"k8s.io/kubernetes/federation/pkg/federation-controller/util"
	"k8s.io/kubernetes/federation/pkg/federation-controller/util/deletionhelper"
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

type SecretController struct {
	// For triggering single secret reconciliation. This is used when there is an
	// add/update/delete operation on a secret in either federated API server or
	// in some member of the federation.
	secretDeliverer *util.DelayingDeliverer

	// For triggering all secrets reconciliation. This is used when
	// a new cluster becomes available.
	clusterDeliverer *util.DelayingDeliverer

	// Contains secrets present in members of federation.
	secretFederatedInformer util.FederatedInformer
	// For updating members of federation.
	federatedUpdater util.FederatedUpdater
	// Definitions of secrets that should be federated.
	secretInformerStore cache.Store
	// Informer controller for secrets that should be federated.
	secretInformerController cache.Controller

	// Client to federated api server.
	federatedApiClient federationclientset.Interface

	// Backoff manager for secrets
	secretBackoff *flowcontrol.Backoff

	// For events
	eventRecorder record.EventRecorder

	deletionHelper *deletionhelper.DeletionHelper

	secretReviewDelay     time.Duration
	clusterAvailableDelay time.Duration
	smallDelay            time.Duration
	updateTimeout         time.Duration
}

// NewSecretController returns a new secret controller
func NewSecretController(client federationclientset.Interface) *SecretController {
	broadcaster := record.NewBroadcaster()
	broadcaster.StartRecordingToSink(eventsink.NewFederatedEventSink(client))
	recorder := broadcaster.NewRecorder(apiv1.EventSource{Component: "federated-secrets-controller"})

	secretcontroller := &SecretController{
		federatedApiClient:    client,
		secretReviewDelay:     time.Second * 10,
		clusterAvailableDelay: time.Second * 20,
		smallDelay:            time.Second * 3,
		updateTimeout:         time.Second * 30,
		secretBackoff:         flowcontrol.NewBackOff(5*time.Second, time.Minute),
		eventRecorder:         recorder,
	}

	// Build delivereres for triggering reconciliations.
	secretcontroller.secretDeliverer = util.NewDelayingDeliverer()
	secretcontroller.clusterDeliverer = util.NewDelayingDeliverer()

	// Start informer in federated API servers on secrets that should be federated.
	secretcontroller.secretInformerStore, secretcontroller.secretInformerController = cache.NewInformer(
		&cache.ListWatch{
			ListFunc: func(options apiv1.ListOptions) (pkgruntime.Object, error) {
				return client.Core().Secrets(apiv1.NamespaceAll).List(options)
			},
			WatchFunc: func(options apiv1.ListOptions) (watch.Interface, error) {
				return client.Core().Secrets(apiv1.NamespaceAll).Watch(options)
			},
		},
		&apiv1.Secret{},
		controller.NoResyncPeriodFunc(),
		util.NewTriggerOnAllChanges(func(obj pkgruntime.Object) { secretcontroller.deliverSecretObj(obj, 0, false) }))

	// Federated informer on secrets in members of federation.
	secretcontroller.secretFederatedInformer = util.NewFederatedInformer(
		client,
		func(cluster *federationapi.Cluster, targetClient kubeclientset.Interface) (cache.Store, cache.Controller) {
			return cache.NewInformer(
				&cache.ListWatch{
					ListFunc: func(options apiv1.ListOptions) (pkgruntime.Object, error) {
						return targetClient.Core().Secrets(apiv1.NamespaceAll).List(options)
					},
					WatchFunc: func(options apiv1.ListOptions) (watch.Interface, error) {
						return targetClient.Core().Secrets(apiv1.NamespaceAll).Watch(options)
					},
				},
				&apiv1.Secret{},
				controller.NoResyncPeriodFunc(),
				// Trigger reconciliation whenever something in federated cluster is changed. In most cases it
				// would be just confirmation that some secret operation succeeded.
				util.NewTriggerOnAllChanges(
					func(obj pkgruntime.Object) {
						secretcontroller.deliverSecretObj(obj, secretcontroller.secretReviewDelay, false)
					},
				))
		},

		&util.ClusterLifecycleHandlerFuncs{
			ClusterAvailable: func(cluster *federationapi.Cluster) {
				// When new cluster becomes available process all the secrets again.
				secretcontroller.clusterDeliverer.DeliverAt(allClustersKey, nil, time.Now().Add(secretcontroller.clusterAvailableDelay))
			},
		},
	)

	// Federated updeater along with Create/Update/Delete operations.
	secretcontroller.federatedUpdater = util.NewFederatedUpdater(secretcontroller.secretFederatedInformer,
		func(client kubeclientset.Interface, obj pkgruntime.Object) error {
			secret := obj.(*apiv1.Secret)
			_, err := client.Core().Secrets(secret.Namespace).Create(secret)
			return err
		},
		func(client kubeclientset.Interface, obj pkgruntime.Object) error {
			secret := obj.(*apiv1.Secret)
			_, err := client.Core().Secrets(secret.Namespace).Update(secret)
			return err
		},
		func(client kubeclientset.Interface, obj pkgruntime.Object) error {
			secret := obj.(*apiv1.Secret)
			err := client.Core().Secrets(secret.Namespace).Delete(secret.Name, &apiv1.DeleteOptions{})
			return err
		})

	secretcontroller.deletionHelper = deletionhelper.NewDeletionHelper(
		secretcontroller.hasFinalizerFunc,
		secretcontroller.removeFinalizerFunc,
		secretcontroller.addFinalizerFunc,
		// objNameFunc
		func(obj pkgruntime.Object) string {
			secret := obj.(*apiv1.Secret)
			return secret.Name
		},
		secretcontroller.updateTimeout,
		secretcontroller.eventRecorder,
		secretcontroller.secretFederatedInformer,
		secretcontroller.federatedUpdater,
	)

	return secretcontroller
}

// Returns true if the given object has the given finalizer in its ObjectMeta.
func (secretcontroller *SecretController) hasFinalizerFunc(obj pkgruntime.Object, finalizer string) bool {
	secret := obj.(*apiv1.Secret)
	for i := range secret.ObjectMeta.Finalizers {
		if string(secret.ObjectMeta.Finalizers[i]) == finalizer {
			return true
		}
	}
	return false
}

// Removes the finalizer from the given objects ObjectMeta.
// Assumes that the given object is a secret.
func (secretcontroller *SecretController) removeFinalizerFunc(obj pkgruntime.Object, finalizer string) (pkgruntime.Object, error) {
	secret := obj.(*apiv1.Secret)
	newFinalizers := []string{}
	hasFinalizer := false
	for i := range secret.ObjectMeta.Finalizers {
		if string(secret.ObjectMeta.Finalizers[i]) != finalizer {
			newFinalizers = append(newFinalizers, secret.ObjectMeta.Finalizers[i])
		} else {
			hasFinalizer = true
		}
	}
	if !hasFinalizer {
		// Nothing to do.
		return obj, nil
	}
	secret.ObjectMeta.Finalizers = newFinalizers
	secret, err := secretcontroller.federatedApiClient.Core().Secrets(secret.Namespace).Update(secret)
	if err != nil {
		return nil, fmt.Errorf("failed to remove finalizer %s from secret %s: %v", finalizer, secret.Name, err)
	}
	return secret, nil
}

// Adds the given finalizer to the given objects ObjectMeta.
// Assumes that the given object is a secret.
func (secretcontroller *SecretController) addFinalizerFunc(obj pkgruntime.Object, finalizer string) (pkgruntime.Object, error) {
	secret := obj.(*apiv1.Secret)
	secret.ObjectMeta.Finalizers = append(secret.ObjectMeta.Finalizers, finalizer)
	secret, err := secretcontroller.federatedApiClient.Core().Secrets(secret.Namespace).Update(secret)
	if err != nil {
		return nil, fmt.Errorf("failed to add finalizer %s to secret %s: %v", finalizer, secret.Name, err)
	}
	return secret, nil
}

func (secretcontroller *SecretController) Run(stopChan <-chan struct{}) {
	go secretcontroller.secretInformerController.Run(stopChan)
	secretcontroller.secretFederatedInformer.Start()
	go func() {
		<-stopChan
		secretcontroller.secretFederatedInformer.Stop()
	}()
	secretcontroller.secretDeliverer.StartWithHandler(func(item *util.DelayingDelivererItem) {
		secret := item.Value.(*types.NamespacedName)
		secretcontroller.reconcileSecret(*secret)
	})
	secretcontroller.clusterDeliverer.StartWithHandler(func(_ *util.DelayingDelivererItem) {
		secretcontroller.reconcileSecretsOnClusterChange()
	})
	util.StartBackoffGC(secretcontroller.secretBackoff, stopChan)
}

func (secretcontroller *SecretController) deliverSecretObj(obj interface{}, delay time.Duration, failed bool) {
	secret := obj.(*apiv1.Secret)
	secretcontroller.deliverSecret(types.NamespacedName{Namespace: secret.Namespace, Name: secret.Name}, delay, failed)
}

// Adds backoff to delay if this delivery is related to some failure. Resets backoff if there was no failure.
func (secretcontroller *SecretController) deliverSecret(secret types.NamespacedName, delay time.Duration, failed bool) {
	key := secret.String()
	if failed {
		secretcontroller.secretBackoff.Next(key, time.Now())
		delay = delay + secretcontroller.secretBackoff.Get(key)
	} else {
		secretcontroller.secretBackoff.Reset(key)
	}
	secretcontroller.secretDeliverer.DeliverAfter(key, &secret, delay)
}

// Check whether all data stores are in sync. False is returned if any of the informer/stores is not yet
// synced with the corresponding api server.
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

// The function triggers reconciliation of all federated secrets.
func (secretcontroller *SecretController) reconcileSecretsOnClusterChange() {
	if !secretcontroller.isSynced() {
		secretcontroller.clusterDeliverer.DeliverAt(allClustersKey, nil, time.Now().Add(secretcontroller.clusterAvailableDelay))
	}
	for _, obj := range secretcontroller.secretInformerStore.List() {
		secret := obj.(*apiv1.Secret)
		secretcontroller.deliverSecret(types.NamespacedName{Namespace: secret.Namespace, Name: secret.Name}, secretcontroller.smallDelay, false)
	}
}

func (secretcontroller *SecretController) reconcileSecret(secret types.NamespacedName) {
	if !secretcontroller.isSynced() {
		secretcontroller.deliverSecret(secret, secretcontroller.clusterAvailableDelay, false)
		return
	}

	key := secret.String()
	baseSecretObjFromStore, exist, err := secretcontroller.secretInformerStore.GetByKey(key)
	if err != nil {
		glog.Errorf("Failed to query main secret store for %v: %v", key, err)
		secretcontroller.deliverSecret(secret, 0, true)
		return
	}

	if !exist {
		// Not federated secret, ignoring.
		return
	}

	// Create a copy before modifying the obj to prevent race condition with
	// other readers of obj from store.
	baseSecretObj, err := api.Scheme.DeepCopy(baseSecretObjFromStore)
	baseSecret, ok := baseSecretObj.(*apiv1.Secret)
	if err != nil || !ok {
		glog.Errorf("Error in retrieving obj from store: %v, %v", ok, err)
		secretcontroller.deliverSecret(secret, 0, true)
		return
	}
	if baseSecret.DeletionTimestamp != nil {
		if err := secretcontroller.delete(baseSecret); err != nil {
			glog.Errorf("Failed to delete %s: %v", secret, err)
			secretcontroller.eventRecorder.Eventf(baseSecret, api.EventTypeNormal, "DeleteFailed",
				"Secret delete failed: %v", err)
			secretcontroller.deliverSecret(secret, 0, true)
		}
		return
	}

	glog.V(3).Infof("Ensuring delete object from underlying clusters finalizer for secret: %s",
		baseSecret.Name)
	// Add the required finalizers before creating a secret in underlying clusters.
	updatedSecretObj, err := secretcontroller.deletionHelper.EnsureFinalizers(baseSecret)
	if err != nil {
		glog.Errorf("Failed to ensure delete object from underlying clusters finalizer in secret %s: %v",
			baseSecret.Name, err)
		secretcontroller.deliverSecret(secret, 0, false)
		return
	}
	baseSecret = updatedSecretObj.(*apiv1.Secret)

	glog.V(3).Infof("Syncing secret %s in underlying clusters", baseSecret.Name)

	clusters, err := secretcontroller.secretFederatedInformer.GetReadyClusters()
	if err != nil {
		glog.Errorf("Failed to get cluster list: %v", err)
		secretcontroller.deliverSecret(secret, secretcontroller.clusterAvailableDelay, false)
		return
	}

	operations := make([]util.FederatedOperation, 0)
	for _, cluster := range clusters {
		clusterSecretObj, found, err := secretcontroller.secretFederatedInformer.GetTargetStore().GetByKey(cluster.Name, key)
		if err != nil {
			glog.Errorf("Failed to get %s from %s: %v", key, cluster.Name, err)
			secretcontroller.deliverSecret(secret, 0, true)
			return
		}

		// The data should not be modified.
		desiredSecret := &apiv1.Secret{
			ObjectMeta: util.DeepCopyRelevantObjectMeta(baseSecret.ObjectMeta),
			Data:       baseSecret.Data,
			Type:       baseSecret.Type,
		}

		if !found {
			secretcontroller.eventRecorder.Eventf(baseSecret, api.EventTypeNormal, "CreateInCluster",
				"Creating secret in cluster %s", cluster.Name)

			operations = append(operations, util.FederatedOperation{
				Type:        util.OperationTypeAdd,
				Obj:         desiredSecret,
				ClusterName: cluster.Name,
			})
		} else {
			clusterSecret := clusterSecretObj.(*apiv1.Secret)

			// Update existing secret, if needed.
			if !util.SecretEquivalent(*desiredSecret, *clusterSecret) {

				secretcontroller.eventRecorder.Eventf(baseSecret, api.EventTypeNormal, "UpdateInCluster",
					"Updating secret in cluster %s", cluster.Name)
				operations = append(operations, util.FederatedOperation{
					Type:        util.OperationTypeUpdate,
					Obj:         desiredSecret,
					ClusterName: cluster.Name,
				})
			}
		}
	}

	if len(operations) == 0 {
		// Everything is in order
		return
	}
	err = secretcontroller.federatedUpdater.UpdateWithOnError(operations, secretcontroller.updateTimeout,
		func(op util.FederatedOperation, operror error) {
			secretcontroller.eventRecorder.Eventf(baseSecret, api.EventTypeNormal, "UpdateInClusterFailed",
				"Secret update in cluster %s failed: %v", op.ClusterName, operror)
		})

	if err != nil {
		glog.Errorf("Failed to execute updates for %s: %v", key, err)
		secretcontroller.deliverSecret(secret, 0, true)
		return
	}

	// Evertyhing is in order but lets be double sure
	secretcontroller.deliverSecret(secret, secretcontroller.secretReviewDelay, false)
}

// delete deletes the given secret or returns error if the deletion was not complete.
func (secretcontroller *SecretController) delete(secret *apiv1.Secret) error {
	glog.V(3).Infof("Handling deletion of secret: %v", *secret)
	_, err := secretcontroller.deletionHelper.HandleObjectInUnderlyingClusters(secret)
	if err != nil {
		return err
	}

	err = secretcontroller.federatedApiClient.Core().Secrets(secret.Namespace).Delete(secret.Name, nil)
	if err != nil {
		// Its all good if the error is not found error. That means it is deleted already and we do not have to do anything.
		// This is expected when we are processing an update as a result of secret finalizer deletion.
		// The process that deleted the last finalizer is also going to delete the secret and we do not have to do anything.
		if !errors.IsNotFound(err) {
			return fmt.Errorf("failed to delete secret: %v", err)
		}
	}
	return nil
}
