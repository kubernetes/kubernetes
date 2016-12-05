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

package autoscaler

import (
	"fmt"
	"time"

	"github.com/golang/glog"

	fedv1 "k8s.io/kubernetes/federation/apis/federation/v1beta1"
	fedclientset "k8s.io/kubernetes/federation/client/clientset_generated/federation_clientset"
	fedutil "k8s.io/kubernetes/federation/pkg/federation-controller/util"
	"k8s.io/kubernetes/federation/pkg/federation-controller/util/deletionhelper"
	"k8s.io/kubernetes/federation/pkg/federation-controller/util/eventsink"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/errors"
	apiv1 "k8s.io/kubernetes/pkg/api/v1"
	autoscalingv1 "k8s.io/kubernetes/pkg/apis/autoscaling/v1"
	"k8s.io/kubernetes/pkg/client/cache"
	kubeclientset "k8s.io/kubernetes/pkg/client/clientset_generated/clientset"
	"k8s.io/kubernetes/pkg/client/record"
	"k8s.io/kubernetes/pkg/controller"
	"k8s.io/kubernetes/pkg/conversion"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/util/flowcontrol"
	"k8s.io/kubernetes/pkg/watch"
)

const (
	allClustersKey = "THE_ALL_CLUSTER_KEY"
)

var (
	autoscalerReviewDelay   = 10 * time.Second
	clusterAvailableDelay   = 20 * time.Second
	clusterUnavailableDelay = 60 * time.Second
	updateTimeout           = 30 * time.Second
	backoffInitial          = 5 * time.Second
	backoffMax              = 1 * time.Minute
)

type AutoscalerController struct {
	fedClient fedclientset.Interface

	autoscalerInformerController *cache.Controller
	autoscalerInformerStore      cache.Store

	fedAutoscalerInformer fedutil.FederatedInformer

	autoscalerDeliverer *fedutil.DelayingDeliverer
	clusterDeliverer    *fedutil.DelayingDeliverer

	// For updating members of federation.
	fedUpdater fedutil.FederatedUpdater

	autoscalerBackoff *flowcontrol.Backoff
	// For events
	eventRecorder record.EventRecorder

	deletionHelper *deletionhelper.DeletionHelper
}

// NewclusterController returns a new cluster controller
func NewAutoscalerController(fedClient fedclientset.Interface) *AutoscalerController {
	broadcaster := record.NewBroadcaster()
	broadcaster.StartRecordingToSink(eventsink.NewFederatedEventSink(fedClient))
	recorder := broadcaster.NewRecorder(apiv1.EventSource{Component: "federated-autoscaler-controller"})

	fac := &AutoscalerController{
		fedClient:           fedClient,
		autoscalerDeliverer: fedutil.NewDelayingDeliverer(),
		clusterDeliverer:    fedutil.NewDelayingDeliverer(),
		autoscalerBackoff:   flowcontrol.NewBackOff(backoffInitial, backoffMax),
		eventRecorder:       recorder,
	}

	fac.autoscalerInformerStore, fac.autoscalerInformerController = cache.NewInformer(
		&cache.ListWatch{
			ListFunc: func(options apiv1.ListOptions) (runtime.Object, error) {
				return fac.fedClient.AutoscalingV1().HorizontalPodAutoscalers(apiv1.NamespaceAll).List(options)
			},
			WatchFunc: func(options apiv1.ListOptions) (watch.Interface, error) {
				return fac.fedClient.AutoscalingV1().HorizontalPodAutoscalers(apiv1.NamespaceAll).Watch(options)
			},
		},
		&autoscalingv1.HorizontalPodAutoscaler{},
		controller.NoResyncPeriodFunc(),
		fedutil.NewTriggerOnAllChanges(
			func(obj runtime.Object) { fac.deliverAutoscalerObj(obj, 0, false) },
		),
	)

	// Federated informer on autoscalers in members of federation.
	fac.fedAutoscalerInformer = fedutil.NewFederatedInformer(
		fedClient,
		func(cluster *fedv1.Cluster, targetClient kubeclientset.Interface) (cache.Store, cache.ControllerInterface) {
			return cache.NewInformer(
				&cache.ListWatch{
					ListFunc: func(options apiv1.ListOptions) (runtime.Object, error) {
						return targetClient.AutoscalingV1().HorizontalPodAutoscalers(apiv1.NamespaceAll).List(options)
					},
					WatchFunc: func(options apiv1.ListOptions) (watch.Interface, error) {
						return targetClient.AutoscalingV1().HorizontalPodAutoscalers(apiv1.NamespaceAll).Watch(options)
					},
				},
				&autoscalingv1.HorizontalPodAutoscaler{},
				controller.NoResyncPeriodFunc(),
				// Trigger reconciliation whenever something in federated cluster is changed. In most cases it
				// would be just confirmation that some autoscaler update succeeded.
				fedutil.NewTriggerOnMetaAndSpecChanges(
					func(obj runtime.Object) { fac.deliverAutoscalerObj(obj, autoscalerReviewDelay, false) },
				))
		},
		&fedutil.ClusterLifecycleHandlerFuncs{
			ClusterAvailable: func(cluster *fedv1.Cluster) {
				// When new cluster becomes available process all the autoscalers again.
				fac.clusterDeliverer.DeliverAfter(allClustersKey, nil, clusterAvailableDelay)
			},
		},
	)

	fac.fedUpdater = fedutil.NewFederatedUpdater(fac.fedAutoscalerInformer,
		func(client kubeclientset.Interface, obj runtime.Object) error {
			hpa := obj.(*autoscalingv1.HorizontalPodAutoscaler)
			_, err := client.AutoscalingV1().HorizontalPodAutoscalers(hpa.Namespace).Create(hpa)
			return err
		},
		func(client kubeclientset.Interface, obj runtime.Object) error {
			hpa := obj.(*autoscalingv1.HorizontalPodAutoscaler)
			_, err := client.AutoscalingV1().HorizontalPodAutoscalers(hpa.Namespace).Update(hpa)
			return err
		},
		func(client kubeclientset.Interface, obj runtime.Object) error {
			hpa := obj.(*autoscalingv1.HorizontalPodAutoscaler)
			err := client.AutoscalingV1().HorizontalPodAutoscalers(hpa.Namespace).Delete(hpa.Name, &apiv1.DeleteOptions{})
			return err
		})

	fac.deletionHelper = deletionhelper.NewDeletionHelper(
		fac.hasFinalizerFunc,
		fac.removeFinalizerFunc,
		fac.addFinalizerFunc,
		// objNameFunc
		func(obj runtime.Object) string {
			hpa := obj.(*autoscalingv1.HorizontalPodAutoscaler)
			return hpa.Name
		},
		updateTimeout,
		fac.eventRecorder,
		fac.fedAutoscalerInformer,
		fac.fedUpdater,
	)
	return fac
}

// Returns true if the given object has the given finalizer in its ObjectMeta.
func (fac *AutoscalerController) hasFinalizerFunc(obj runtime.Object, finalizer string) bool {
	hpa := obj.(*autoscalingv1.HorizontalPodAutoscaler)
	for i := range hpa.ObjectMeta.Finalizers {
		if string(hpa.ObjectMeta.Finalizers[i]) == finalizer {
			return true
		}
	}
	return false
}

// Removes the finalizer from the given objects ObjectMeta.
// Assumes that the given object is a hpa.
func (fac *AutoscalerController) removeFinalizerFunc(obj runtime.Object, finalizer string) (runtime.Object, error) {
	hpa := obj.(*autoscalingv1.HorizontalPodAutoscaler)
	newFinalizers := []string{}
	hasFinalizer := false
	for i := range hpa.ObjectMeta.Finalizers {
		if string(hpa.ObjectMeta.Finalizers[i]) != finalizer {
			newFinalizers = append(newFinalizers, hpa.ObjectMeta.Finalizers[i])
		} else {
			hasFinalizer = true
		}
	}
	if !hasFinalizer {
		// Nothing to do.
		return obj, nil
	}
	hpa.ObjectMeta.Finalizers = newFinalizers
	hpa, err := fac.fedClient.AutoscalingV1().HorizontalPodAutoscalers(hpa.Namespace).Update(hpa)
	if err != nil {
		return nil, fmt.Errorf("failed to remove finalizer %v from hpa %s: %v", finalizer, hpa.Name, err)
	}
	return hpa, nil
}

// Adds the given finalizer to the given objects ObjectMeta.
// Assumes that the given object is an hpa.
func (fac *AutoscalerController) addFinalizerFunc(obj runtime.Object, finalizer string) (runtime.Object, error) {
	hpa := obj.(*autoscalingv1.HorizontalPodAutoscaler)
	hpa.ObjectMeta.Finalizers = append(hpa.ObjectMeta.Finalizers, finalizer)
	hpa, err := fac.fedClient.AutoscalingV1().HorizontalPodAutoscalers(hpa.Namespace).Update(hpa)
	if err != nil {
		return nil, fmt.Errorf("failed to add finalizer %v to hpa %s: %v", finalizer, hpa.Name, err)
	}
	return hpa, nil
}

func (fac *AutoscalerController) Run(stopChan <-chan struct{}) {
	go fac.autoscalerInformerController.Run(stopChan)
	fac.fedAutoscalerInformer.Start()
	go func() {
		<-stopChan
		fac.fedAutoscalerInformer.Stop()
	}()
	fac.autoscalerDeliverer.StartWithHandler(func(item *fedutil.DelayingDelivererItem) {
		hpa := item.Key
		fac.reconcileAutoscaler(hpa)
	})
	fac.clusterDeliverer.StartWithHandler(func(_ *fedutil.DelayingDelivererItem) {
		fac.reconcileAutoscalerOnClusterChange()
	})
	fedutil.StartBackoffGC(fac.autoscalerBackoff, stopChan)

	//TODO IRF: do we need to explicitly stop the deliverers at exit of this function
}

func (fac *AutoscalerController) isSynced() bool {
	if !fac.fedAutoscalerInformer.ClustersSynced() {
		glog.V(3).Infof("Cluster list not synced")
		return false
	}
	clusters, err := fac.fedAutoscalerInformer.GetReadyClusters()
	if err != nil {
		glog.Errorf("Failed to get ready clusters: %v", err)
		return false
	}
	if !fac.fedAutoscalerInformer.GetTargetStore().ClustersSynced(clusters) {
		glog.V(2).Infof("cluster hpa list not synced")
		return false
	}

	if !fac.autoscalerInformerController.HasSynced() {
		glog.V(2).Infof("federation hpa list not synced")
		return false
	}
	return true
}

func (fac *AutoscalerController) deliverAutoscalerObj(obj interface{}, delay time.Duration, failed bool) {
	key, err := controller.KeyFunc(obj)
	if err != nil {
		glog.Errorf("Couldn't get key for object %+v: %v", obj, err)
		return
	}

	fac.deliverAutoscaler(key, delay, failed)
}

// Adds backoff to delay if this delivery is related to some failure. Resets backoff if there was no failure.
func (fac *AutoscalerController) deliverAutoscaler(hpa string, delay time.Duration, failed bool) {
	if failed {
		fac.autoscalerBackoff.Next(hpa, time.Now())
		delay = delay + fac.autoscalerBackoff.Get(hpa)
	} else {
		fac.autoscalerBackoff.Reset(hpa)
	}
	fac.autoscalerDeliverer.DeliverAfter(hpa, nil, delay)
}

//func (fac *AutoscalerController) reconcileAutoscaler(key string) (reconciliationStatus, error)
func (fac *AutoscalerController) reconcileAutoscaler(hpa string) {
	if !fac.isSynced() {
		fac.deliverAutoscaler(hpa, clusterAvailableDelay, false)
		return
	}

	autoscalerObjFromStore, exist, err := fac.autoscalerInformerStore.GetByKey(hpa)
	if err != nil {
		glog.Errorf("Failed to query main hpa store for %v: %v", hpa, err)
		fac.deliverAutoscaler(hpa, 0, true)
		return
	}

	if !exist {
		// Not federated hpa, ignoring.
		return
	}
	// Create a copy before modifying the hpa to prevent race condition with
	// other readers of hpa from store.
	autoscalerObj, err := conversion.NewCloner().DeepCopy(autoscalerObjFromStore)
	baseAutoscaler, ok := autoscalerObj.(*autoscalingv1.HorizontalPodAutoscaler)
	if err != nil || !ok {
		glog.Errorf("Error in retrieving obj from store: %v, %v", ok, err)
		fac.deliverAutoscaler(hpa, 0, true)
		return
	}

	if baseAutoscaler.DeletionTimestamp != nil {
		if err := fac.delete(baseAutoscaler); err != nil {
			glog.Errorf("Failed to delete %s: %v", hpa, err)
			fac.eventRecorder.Eventf(baseAutoscaler, api.EventTypeNormal, "DeleteFailed",
				"HPA delete failed: %v", err)
			fac.deliverAutoscaler(hpa, 0, true)
		}
		return
	}

	glog.V(3).Infof("Ensuring delete object from underlying clusters finalizer for hpa: %s",
		baseAutoscaler.Name)
	// Add the required finalizers before creating an hpa in
	// underlying clusters.
	// This ensures that the dependent hpas are deleted in underlying
	// clusters when the federated hpa is deleted.
	updatedAutoscalerObj, err := fac.deletionHelper.EnsureFinalizers(baseAutoscaler)
	if err != nil {
		glog.Errorf("Failed to ensure delete object from underlying clusters finalizer for hpa %s: %v",
			baseAutoscaler.Name, err)
		fac.deliverAutoscaler(hpa, 0, false)
		return
	}
	baseAutoscaler = updatedAutoscalerObj.(*autoscalingv1.HorizontalPodAutoscaler)

	glog.V(3).Infof("Syncing hpa %s in underlying clusters", baseAutoscaler.Name)
	// Sync the hpa in all underlying clusters.
	clusters, err := fac.fedAutoscalerInformer.GetReadyClusters()
	if err != nil {
		glog.Errorf("Failed to get cluster list: %v", err)
		fac.deliverAutoscaler(hpa, clusterAvailableDelay, false)
		return
	}

	operations := make([]fedutil.FederatedOperation, 0)
	for _, cluster := range clusters {
		clusterAutoscalerObj, found, err := fac.fedAutoscalerInformer.GetTargetStore().GetByKey(cluster.Name, hpa)
		if err != nil {
			glog.Errorf("Failed to get %s from %s: %v", hpa, cluster.Name, err)
			fac.deliverAutoscaler(hpa, 0, true)
			return
		}
		// The object should not be modified.
		desiredAutoscaler := &autoscalingv1.HorizontalPodAutoscaler{
			ObjectMeta: fedutil.DeepCopyRelevantObjectMeta(baseAutoscaler.ObjectMeta),
			Spec:       fedutil.DeepCopyApiTypeOrPanic(baseAutoscaler.Spec).(autoscalingv1.HorizontalPodAutoscalerSpec),
		}
		glog.V(5).Infof("Desired hpa in underlying clusters: %+v", desiredAutoscaler)

		if !found {
			fac.eventRecorder.Eventf(baseAutoscaler, api.EventTypeNormal, "CreateInCluster",
				"Creating hpa in cluster %s", cluster.Name)

			operations = append(operations, fedutil.FederatedOperation{
				Type:        fedutil.OperationTypeAdd,
				Obj:         desiredAutoscaler,
				ClusterName: cluster.Name,
			})
		} else {
			clusterAutoscaler := clusterAutoscalerObj.(*autoscalingv1.HorizontalPodAutoscaler)

			// Update existing hpa, if needed.
			if !fedutil.ObjectMetaAndSpecEquivalent(desiredAutoscaler, clusterAutoscaler) {
				fac.eventRecorder.Eventf(baseAutoscaler, api.EventTypeNormal, "UpdateInCluster",
					"Updating hpa in cluster %s. Desired: %+v\n Actual: %+v\n", cluster.Name, desiredAutoscaler, clusterAutoscaler)

				operations = append(operations, fedutil.FederatedOperation{
					Type:        fedutil.OperationTypeUpdate,
					Obj:         desiredAutoscaler,
					ClusterName: cluster.Name,
				})
			}
		}
	}

	if len(operations) == 0 {
		// Everything is in order
		return
	}
	glog.V(2).Infof("Updating hpa %s in underlying clusters. Operations: %d", baseAutoscaler.Name, len(operations))

	err = fac.fedUpdater.UpdateWithOnError(operations, time.Second*30, func(op fedutil.FederatedOperation, operror error) {
		fac.eventRecorder.Eventf(baseAutoscaler, api.EventTypeNormal, "UpdateInClusterFailed",
			"HPA update in cluster %s failed: %v", op.ClusterName, operror)
	})
	if err != nil {
		glog.Errorf("Failed to execute updates for %s: %v", hpa, err)
		fac.deliverAutoscaler(hpa, 0, true)
		return
	}

	// Evertyhing is in order but lets be double sure
	fac.deliverAutoscaler(hpa, autoscalerReviewDelay, false)
}

func (fac *AutoscalerController) reconcileAutoscalerOnClusterChange() {
	if !fac.isSynced() {
		fac.clusterDeliverer.DeliverAfter(allClustersKey, nil, clusterAvailableDelay)
	}

	for _, obj := range fac.autoscalerInformerStore.List() {
		hpa := obj.(*autoscalingv1.HorizontalPodAutoscaler)
		fac.deliverAutoscaler(hpa.Name, time.Second*3, false)
	}
}

// delete deletes the given hpa or returns error if the deletion was not complete.
func (fac *AutoscalerController) delete(hpa *autoscalingv1.HorizontalPodAutoscaler) error {
	glog.V(3).Infof("Handling deletion of hpa: %s/%s\n", hpa.Namespace, hpa.Name)
	_, err := fac.deletionHelper.HandleObjectInUnderlyingClusters(hpa)
	if err != nil {
		return err
	}

	err = fac.fedClient.AutoscalingV1().HorizontalPodAutoscalers(hpa.Namespace).Delete(hpa.Name, nil)
	if err != nil {
		// Its all good if the error type is not found. That means it is deleted already and we do not have to do anything.
		// This is expected when we are processing an update as a result of hpa finalizer deletion.
		// The process that deleted the last finalizer is also going to delete the hpa and we do not have to do anything.
		if !errors.IsNotFound(err) {
			return fmt.Errorf("failed to delete hpa: %s/%s, %v\n", hpa.Namespace, hpa.Name, err)
		}
	}
	return nil
}
