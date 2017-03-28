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
	"fmt"
	"time"

	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/client-go/dynamic"
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
	apiv1 "k8s.io/kubernetes/pkg/api/v1"
	kubeclientset "k8s.io/kubernetes/pkg/client/clientset_generated/clientset"
	"k8s.io/kubernetes/pkg/controller"
	"k8s.io/kubernetes/pkg/controller/namespace/deletion"

	"github.com/golang/glog"
)

const (
	allClustersKey = "ALL_CLUSTERS"
	ControllerName = "namespaces"
)

var (
	RequiredResources = []schema.GroupVersionResource{apiv1.SchemeGroupVersion.WithResource("namespaces")}
)

type NamespaceController struct {
	// For triggering single namespace reconciliation. This is used when there is an
	// add/update/delete operation on a namespace in either federated API server or
	// in some member of the federation.
	namespaceDeliverer *util.DelayingDeliverer

	// For triggering all namespaces reconciliation. This is used when
	// a new cluster becomes available.
	clusterDeliverer *util.DelayingDeliverer

	// Contains namespaces present in members of federation.
	namespaceFederatedInformer util.FederatedInformer
	// For updating members of federation.
	federatedUpdater util.FederatedUpdater
	// Definitions of namespaces that should be federated.
	namespaceInformerStore cache.Store
	// Informer controller for namespaces that should be federated.
	namespaceInformerController cache.Controller

	// Client to federated api server.
	federatedApiClient federationclientset.Interface

	// Backoff manager for namespaces
	namespaceBackoff *flowcontrol.Backoff

	// For events
	eventRecorder record.EventRecorder

	deletionHelper *deletionhelper.DeletionHelper

	// Helper to delete all resources in a namespace.
	namespacedResourcesDeleter deletion.NamespacedResourcesDeleterInterface

	namespaceReviewDelay  time.Duration
	clusterAvailableDelay time.Duration
	smallDelay            time.Duration
	updateTimeout         time.Duration
}

// NewNamespaceController returns a new namespace controller
func NewNamespaceController(client federationclientset.Interface, dynamicClientPool dynamic.ClientPool) *NamespaceController {
	broadcaster := record.NewBroadcaster()
	broadcaster.StartRecordingToSink(eventsink.NewFederatedEventSink(client))
	recorder := broadcaster.NewRecorder(api.Scheme, clientv1.EventSource{Component: "federated-namespace-controller"})

	nc := &NamespaceController{
		federatedApiClient:    client,
		namespaceReviewDelay:  time.Second * 10,
		clusterAvailableDelay: time.Second * 20,
		smallDelay:            time.Second * 3,
		updateTimeout:         time.Second * 30,
		namespaceBackoff:      flowcontrol.NewBackOff(5*time.Second, time.Minute),
		eventRecorder:         recorder,
	}

	// Build deliverers for triggering reconciliations.
	nc.namespaceDeliverer = util.NewDelayingDeliverer()
	nc.clusterDeliverer = util.NewDelayingDeliverer()

	// Start informer in federated API servers on namespaces that should be federated.
	nc.namespaceInformerStore, nc.namespaceInformerController = cache.NewInformer(
		&cache.ListWatch{
			ListFunc: func(options metav1.ListOptions) (runtime.Object, error) {
				return client.Core().Namespaces().List(options)
			},
			WatchFunc: func(options metav1.ListOptions) (watch.Interface, error) {
				return client.Core().Namespaces().Watch(options)
			},
		},
		&apiv1.Namespace{},
		controller.NoResyncPeriodFunc(),
		util.NewTriggerOnAllChanges(func(obj runtime.Object) { nc.deliverNamespaceObj(obj, 0, false) }))

	// Federated informer on namespaces in members of federation.
	nc.namespaceFederatedInformer = util.NewFederatedInformer(
		client,
		func(cluster *federationapi.Cluster, targetClient kubeclientset.Interface) (cache.Store, cache.Controller) {
			return cache.NewInformer(
				&cache.ListWatch{
					ListFunc: func(options metav1.ListOptions) (runtime.Object, error) {
						return targetClient.Core().Namespaces().List(options)
					},
					WatchFunc: func(options metav1.ListOptions) (watch.Interface, error) {
						return targetClient.Core().Namespaces().Watch(options)
					},
				},
				&apiv1.Namespace{},
				controller.NoResyncPeriodFunc(),
				// Trigger reconciliation whenever something in federated cluster is changed. In most cases it
				// would be just confirmation that some namespace operation succeeded.
				util.NewTriggerOnMetaAndSpecChanges(
					func(obj runtime.Object) { nc.deliverNamespaceObj(obj, nc.namespaceReviewDelay, false) },
				))
		},
		&util.ClusterLifecycleHandlerFuncs{
			ClusterAvailable: func(cluster *federationapi.Cluster) {
				// When new cluster becomes available process all the namespaces again.
				nc.clusterDeliverer.DeliverAfter(allClustersKey, nil, nc.clusterAvailableDelay)
			},
		},
	)

	// Federated updater along with Create/Update/Delete operations.
	nc.federatedUpdater = util.NewFederatedUpdater(nc.namespaceFederatedInformer,
		func(client kubeclientset.Interface, obj runtime.Object) error {
			namespace := obj.(*apiv1.Namespace)
			_, err := client.Core().Namespaces().Create(namespace)
			return err
		},
		func(client kubeclientset.Interface, obj runtime.Object) error {
			namespace := obj.(*apiv1.Namespace)
			_, err := client.Core().Namespaces().Update(namespace)
			return err
		},
		func(client kubeclientset.Interface, obj runtime.Object) error {
			namespace := obj.(*apiv1.Namespace)
			err := client.Core().Namespaces().Delete(namespace.Name, &metav1.DeleteOptions{})
			// IsNotFound error is fine since that means the object is deleted already.
			if errors.IsNotFound(err) {
				return nil
			}
			return err
		})

	nc.deletionHelper = deletionhelper.NewDeletionHelper(
		nc.hasFinalizerFunc,
		nc.removeFinalizerFunc,
		nc.addFinalizerFunc,
		// objNameFunc
		func(obj runtime.Object) string {
			namespace := obj.(*apiv1.Namespace)
			return namespace.Name
		},
		nc.updateTimeout,
		nc.eventRecorder,
		nc.namespaceFederatedInformer,
		nc.federatedUpdater,
	)

	discoverResourcesFn := nc.federatedApiClient.Discovery().ServerPreferredNamespacedResources
	nc.namespacedResourcesDeleter = deletion.NewNamespacedResourcesDeleter(
		client.Core().Namespaces(), dynamicClientPool, nil,
		discoverResourcesFn, apiv1.FinalizerKubernetes, false)
	return nc
}

// Returns true if the given object has the given finalizer in its ObjectMeta.
func (nc *NamespaceController) hasFinalizerFunc(obj runtime.Object, finalizer string) bool {
	namespace := obj.(*apiv1.Namespace)
	for i := range namespace.ObjectMeta.Finalizers {
		if string(namespace.ObjectMeta.Finalizers[i]) == finalizer {
			return true
		}
	}
	return false
}

// Removes the finalizer from the given objects ObjectMeta.
// Assumes that the given object is a namespace.
func (nc *NamespaceController) removeFinalizerFunc(obj runtime.Object, finalizer string) (runtime.Object, error) {
	namespace := obj.(*apiv1.Namespace)
	newFinalizers := []string{}
	hasFinalizer := false
	for i := range namespace.ObjectMeta.Finalizers {
		if string(namespace.ObjectMeta.Finalizers[i]) != finalizer {
			newFinalizers = append(newFinalizers, namespace.ObjectMeta.Finalizers[i])
		} else {
			hasFinalizer = true
		}
	}
	if !hasFinalizer {
		// Nothing to do.
		return obj, nil
	}
	namespace.ObjectMeta.Finalizers = newFinalizers
	namespace, err := nc.federatedApiClient.Core().Namespaces().Update(namespace)
	if err != nil {
		return nil, fmt.Errorf("failed to remove finalizer %s from namespace %s: %v", finalizer, namespace.Name, err)
	}
	return namespace, nil
}

// Adds the given finalizers to the given objects ObjectMeta.
// Assumes that the given object is a namespace.
func (nc *NamespaceController) addFinalizerFunc(obj runtime.Object, finalizers []string) (runtime.Object, error) {
	namespace := obj.(*apiv1.Namespace)
	namespace.ObjectMeta.Finalizers = append(namespace.ObjectMeta.Finalizers, finalizers...)
	namespace, err := nc.federatedApiClient.Core().Namespaces().Finalize(namespace)
	if err != nil {
		return nil, fmt.Errorf("failed to add finalizers %v to namespace %s: %v", finalizers, namespace.Name, err)
	}
	return namespace, nil
}

// Returns true if the given object has the given finalizer in its NamespaceSpec.
func (nc *NamespaceController) hasFinalizerFuncInSpec(obj runtime.Object, finalizer apiv1.FinalizerName) bool {
	namespace := obj.(*apiv1.Namespace)
	for i := range namespace.Spec.Finalizers {
		if namespace.Spec.Finalizers[i] == finalizer {
			return true
		}
	}
	return false
}

// Removes the finalizer from the given objects NamespaceSpec.
func (nc *NamespaceController) removeFinalizerFromSpec(namespace *apiv1.Namespace, finalizer apiv1.FinalizerName) (*apiv1.Namespace, error) {
	updatedFinalizers := []apiv1.FinalizerName{}
	for i := range namespace.Spec.Finalizers {
		if namespace.Spec.Finalizers[i] != finalizer {
			updatedFinalizers = append(updatedFinalizers, namespace.Spec.Finalizers[i])
		}
	}
	namespace.Spec.Finalizers = updatedFinalizers
	updatedNamespace, err := nc.federatedApiClient.Core().Namespaces().Finalize(namespace)
	if err != nil {
		return nil, fmt.Errorf("failed to remove finalizer %s from namespace %s: %v", string(finalizer), namespace.Name, err)
	}
	return updatedNamespace, nil
}

func (nc *NamespaceController) Run(stopChan <-chan struct{}) {
	go nc.namespaceInformerController.Run(stopChan)
	nc.namespaceFederatedInformer.Start()
	go func() {
		<-stopChan
		nc.namespaceFederatedInformer.Stop()
	}()
	nc.namespaceDeliverer.StartWithHandler(func(item *util.DelayingDelivererItem) {
		namespace := item.Value.(string)
		nc.reconcileNamespace(namespace)
	})
	nc.clusterDeliverer.StartWithHandler(func(_ *util.DelayingDelivererItem) {
		nc.reconcileNamespacesOnClusterChange()
	})
	util.StartBackoffGC(nc.namespaceBackoff, stopChan)
}

func (nc *NamespaceController) deliverNamespaceObj(obj interface{}, delay time.Duration, failed bool) {
	namespace := obj.(*apiv1.Namespace)
	nc.deliverNamespace(namespace.Name, delay, failed)
}

// Adds backoff to delay if this delivery is related to some failure. Resets backoff if there was no failure.
func (nc *NamespaceController) deliverNamespace(namespace string, delay time.Duration, failed bool) {
	if failed {
		nc.namespaceBackoff.Next(namespace, time.Now())
		delay = delay + nc.namespaceBackoff.Get(namespace)
	} else {
		nc.namespaceBackoff.Reset(namespace)
	}
	nc.namespaceDeliverer.DeliverAfter(namespace, namespace, delay)
}

// Check whether all data stores are in sync. False is returned if any of the informer/stores is not yet
// synced with the corresponding api server.
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

// The function triggers reconciliation of all federated namespaces.
func (nc *NamespaceController) reconcileNamespacesOnClusterChange() {
	if !nc.isSynced() {
		nc.clusterDeliverer.DeliverAfter(allClustersKey, nil, nc.clusterAvailableDelay)
	}
	for _, obj := range nc.namespaceInformerStore.List() {
		namespace := obj.(*apiv1.Namespace)
		nc.deliverNamespace(namespace.Name, nc.smallDelay, false)
	}
}

func (nc *NamespaceController) reconcileNamespace(namespace string) {
	if !nc.isSynced() {
		nc.deliverNamespace(namespace, nc.clusterAvailableDelay, false)
		return
	}

	namespaceObjFromStore, exist, err := nc.namespaceInformerStore.GetByKey(namespace)
	if err != nil {
		glog.Errorf("Failed to query main namespace store for %v: %v", namespace, err)
		nc.deliverNamespace(namespace, 0, true)
		return
	}

	if !exist {
		// Not federated namespace, ignoring.
		return
	}
	// Create a copy before modifying the namespace to prevent race condition with
	// other readers of namespace from store.
	namespaceObj, err := api.Scheme.DeepCopy(namespaceObjFromStore)
	baseNamespace, ok := namespaceObj.(*apiv1.Namespace)
	if err != nil || !ok {
		glog.Errorf("Error in retrieving obj from store: %v, %v", ok, err)
		nc.deliverNamespace(namespace, 0, true)
		return
	}
	if baseNamespace.DeletionTimestamp != nil {
		if err := nc.delete(baseNamespace); err != nil {
			glog.Errorf("Failed to delete %s: %v", namespace, err)
			nc.eventRecorder.Eventf(baseNamespace, api.EventTypeNormal, "DeleteFailed",
				"Namespace delete failed: %v", err)
			nc.deliverNamespace(namespace, 0, true)
		}
		return
	}

	glog.V(3).Infof("Ensuring delete object from underlying clusters finalizer for namespace: %s",
		baseNamespace.Name)
	// Add the required finalizers before creating a namespace in
	// underlying clusters.
	// This ensures that the dependent namespaces are deleted in underlying
	// clusters when the federated namespace is deleted.
	updatedNamespaceObj, err := nc.deletionHelper.EnsureFinalizers(baseNamespace)
	if err != nil {
		glog.Errorf("Failed to ensure delete object from underlying clusters finalizer in namespace %s: %v",
			baseNamespace.Name, err)
		nc.deliverNamespace(namespace, 0, false)
		return
	}
	baseNamespace = updatedNamespaceObj.(*apiv1.Namespace)

	glog.V(3).Infof("Syncing namespace %s in underlying clusters", baseNamespace.Name)
	// Sync the namespace in all underlying clusters.
	clusters, err := nc.namespaceFederatedInformer.GetReadyClusters()
	if err != nil {
		glog.Errorf("Failed to get cluster list: %v", err)
		nc.deliverNamespace(namespace, nc.clusterAvailableDelay, false)
		return
	}

	operations := make([]util.FederatedOperation, 0)
	for _, cluster := range clusters {
		clusterNamespaceObj, found, err := nc.namespaceFederatedInformer.GetTargetStore().GetByKey(cluster.Name, namespace)
		if err != nil {
			glog.Errorf("Failed to get %s from %s: %v", namespace, cluster.Name, err)
			nc.deliverNamespace(namespace, 0, true)
			return
		}
		// The object should not be modified.
		desiredNamespace := &apiv1.Namespace{
			ObjectMeta: util.DeepCopyRelevantObjectMeta(baseNamespace.ObjectMeta),
			Spec:       *(util.DeepCopyApiTypeOrPanic(&baseNamespace.Spec).(*apiv1.NamespaceSpec)),
		}
		glog.V(5).Infof("Desired namespace in underlying clusters: %+v", desiredNamespace)

		if !found {
			nc.eventRecorder.Eventf(baseNamespace, api.EventTypeNormal, "CreateInCluster",
				"Creating namespace in cluster %s", cluster.Name)

			operations = append(operations, util.FederatedOperation{
				Type:        util.OperationTypeAdd,
				Obj:         desiredNamespace,
				ClusterName: cluster.Name,
			})
		} else {
			clusterNamespace := clusterNamespaceObj.(*apiv1.Namespace)

			// Update existing namespace, if needed.
			if !util.ObjectMetaAndSpecEquivalent(desiredNamespace, clusterNamespace) {
				nc.eventRecorder.Eventf(baseNamespace, api.EventTypeNormal, "UpdateInCluster",
					"Updating namespace in cluster %s. Desired: %+v\n Actual: %+v\n", cluster.Name, desiredNamespace, clusterNamespace)

				operations = append(operations, util.FederatedOperation{
					Type:        util.OperationTypeUpdate,
					Obj:         desiredNamespace,
					ClusterName: cluster.Name,
				})
			}
		}
	}

	if len(operations) == 0 {
		// Everything is in order
		return
	}
	glog.V(2).Infof("Updating namespace %s in underlying clusters. Operations: %d", baseNamespace.Name, len(operations))

	err = nc.federatedUpdater.UpdateWithOnError(operations, nc.updateTimeout, func(op util.FederatedOperation, operror error) {
		nc.eventRecorder.Eventf(baseNamespace, api.EventTypeNormal, "UpdateInClusterFailed",
			"Namespace update in cluster %s failed: %v", op.ClusterName, operror)
	})
	if err != nil {
		glog.Errorf("Failed to execute updates for %s: %v", namespace, err)
		nc.deliverNamespace(namespace, 0, true)
		return
	}

	// Everything is in order but lets be double sure
	nc.deliverNamespace(namespace, nc.namespaceReviewDelay, false)
}

// delete  deletes the given namespace or returns error if the deletion was not complete.
func (nc *NamespaceController) delete(namespace *apiv1.Namespace) error {
	// Set Terminating status.
	updatedNamespace := &apiv1.Namespace{
		ObjectMeta: namespace.ObjectMeta,
		Spec:       namespace.Spec,
		Status: apiv1.NamespaceStatus{
			Phase: apiv1.NamespaceTerminating,
		},
	}
	var err error
	if namespace.Status.Phase != apiv1.NamespaceTerminating {
		glog.V(2).Infof("Marking ns %s as terminating", namespace.Name)
		nc.eventRecorder.Event(namespace, api.EventTypeNormal, "DeleteNamespace", fmt.Sprintf("Marking for deletion"))
		_, err = nc.federatedApiClient.Core().Namespaces().Update(updatedNamespace)
		if err != nil {
			return fmt.Errorf("failed to update namespace: %v", err)
		}
	}

	if nc.hasFinalizerFuncInSpec(updatedNamespace, apiv1.FinalizerKubernetes) {
		// Delete resources in this namespace.
		err = nc.namespacedResourcesDeleter.Delete(updatedNamespace.Name)
		if err != nil {
			return fmt.Errorf("error in deleting resources in namespace %s: %v", namespace.Name, err)
		}
		glog.V(2).Infof("Removed kubernetes finalizer from ns %s", namespace.Name)
		// Fetch the updated Namespace.
		updatedNamespace, err = nc.federatedApiClient.Core().Namespaces().Get(updatedNamespace.Name, metav1.GetOptions{})
		if err != nil {
			return fmt.Errorf("error in fetching updated namespace %s: %s", updatedNamespace.Name, err)
		}
	}

	// Delete the namespace from all underlying clusters.
	_, err = nc.deletionHelper.HandleObjectInUnderlyingClusters(updatedNamespace)
	if err != nil {
		return err
	}

	err = nc.federatedApiClient.Core().Namespaces().Delete(namespace.Name, nil)
	if err != nil {
		// Its all good if the error is not found error. That means it is deleted already and we do not have to do anything.
		// This is expected when we are processing an update as a result of namespace finalizer deletion.
		// The process that deleted the last finalizer is also going to delete the namespace and we do not have to do anything.
		if !errors.IsNotFound(err) {
			return fmt.Errorf("failed to delete namespace: %v", err)
		}
	}
	return nil
}
