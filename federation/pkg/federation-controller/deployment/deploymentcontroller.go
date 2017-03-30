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

package deployment

import (
	"bytes"
	"encoding/json"
	"fmt"
	"sort"
	"time"

	"github.com/golang/glog"

	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apimachinery/pkg/watch"
	clientv1 "k8s.io/client-go/pkg/api/v1"
	"k8s.io/client-go/tools/cache"
	"k8s.io/client-go/tools/record"
	"k8s.io/client-go/util/flowcontrol"
	"k8s.io/client-go/util/workqueue"
	fed "k8s.io/kubernetes/federation/apis/federation"
	fedv1 "k8s.io/kubernetes/federation/apis/federation/v1beta1"
	fedclientset "k8s.io/kubernetes/federation/client/clientset_generated/federation_clientset"
	fedutil "k8s.io/kubernetes/federation/pkg/federation-controller/util"
	"k8s.io/kubernetes/federation/pkg/federation-controller/util/deletionhelper"
	"k8s.io/kubernetes/federation/pkg/federation-controller/util/eventsink"
	"k8s.io/kubernetes/federation/pkg/federation-controller/util/planner"
	"k8s.io/kubernetes/federation/pkg/federation-controller/util/podanalyzer"
	"k8s.io/kubernetes/pkg/api"
	apiv1 "k8s.io/kubernetes/pkg/api/v1"
	extensionsv1 "k8s.io/kubernetes/pkg/apis/extensions/v1beta1"
	kubeclientset "k8s.io/kubernetes/pkg/client/clientset_generated/clientset"
	"k8s.io/kubernetes/pkg/controller"
)

const (
	FedDeploymentPreferencesAnnotation = "federation.kubernetes.io/deployment-preferences"
	allClustersKey                     = "THE_ALL_CLUSTER_KEY"
	UserAgentName                      = "Federation-Deployment-Controller"
	ControllerName                     = "deployments"
)

var (
	RequiredResources        = []schema.GroupVersionResource{extensionsv1.SchemeGroupVersion.WithResource("deployments")}
	deploymentReviewDelay    = 10 * time.Second
	clusterAvailableDelay    = 20 * time.Second
	clusterUnavailableDelay  = 60 * time.Second
	allDeploymentReviewDelay = 2 * time.Minute
	updateTimeout            = 30 * time.Second
)

func parseFederationDeploymentPreference(fd *extensionsv1.Deployment) (*fed.FederatedReplicaSetPreferences, error) {
	if fd.Annotations == nil {
		return nil, nil
	}
	fdPrefString, found := fd.Annotations[FedDeploymentPreferencesAnnotation]
	if !found {
		return nil, nil
	}
	var fdPref fed.FederatedReplicaSetPreferences
	if err := json.Unmarshal([]byte(fdPrefString), &fdPref); err != nil {
		return nil, err
	}
	return &fdPref, nil
}

type DeploymentController struct {
	fedClient fedclientset.Interface

	deploymentController cache.Controller
	deploymentStore      cache.Store

	fedDeploymentInformer fedutil.FederatedInformer
	fedPodInformer        fedutil.FederatedInformer

	deploymentDeliverer *fedutil.DelayingDeliverer
	clusterDeliverer    *fedutil.DelayingDeliverer
	deploymentWorkQueue workqueue.Interface
	// For updating members of federation.
	fedUpdater        fedutil.FederatedUpdater
	deploymentBackoff *flowcontrol.Backoff
	eventRecorder     record.EventRecorder

	deletionHelper *deletionhelper.DeletionHelper

	defaultPlanner *planner.Planner
}

// NewDeploymentController returns a new deployment controller
func NewDeploymentController(federationClient fedclientset.Interface) *DeploymentController {
	broadcaster := record.NewBroadcaster()
	broadcaster.StartRecordingToSink(eventsink.NewFederatedEventSink(federationClient))
	recorder := broadcaster.NewRecorder(api.Scheme, clientv1.EventSource{Component: "federated-deployment-controller"})

	fdc := &DeploymentController{
		fedClient:           federationClient,
		deploymentDeliverer: fedutil.NewDelayingDeliverer(),
		clusterDeliverer:    fedutil.NewDelayingDeliverer(),
		deploymentWorkQueue: workqueue.New(),
		deploymentBackoff:   flowcontrol.NewBackOff(5*time.Second, time.Minute),
		defaultPlanner: planner.NewPlanner(&fed.FederatedReplicaSetPreferences{
			Clusters: map[string]fed.ClusterReplicaSetPreferences{
				"*": {Weight: 1},
			},
		}),
		eventRecorder: recorder,
	}

	deploymentFedInformerFactory := func(cluster *fedv1.Cluster, clientset kubeclientset.Interface) (cache.Store, cache.Controller) {
		return cache.NewInformer(
			&cache.ListWatch{
				ListFunc: func(options metav1.ListOptions) (runtime.Object, error) {
					return clientset.Extensions().Deployments(metav1.NamespaceAll).List(options)
				},
				WatchFunc: func(options metav1.ListOptions) (watch.Interface, error) {
					return clientset.Extensions().Deployments(metav1.NamespaceAll).Watch(options)
				},
			},
			&extensionsv1.Deployment{},
			controller.NoResyncPeriodFunc(),
			fedutil.NewTriggerOnAllChanges(
				func(obj runtime.Object) { fdc.deliverLocalDeployment(obj, deploymentReviewDelay) },
			),
		)
	}
	clusterLifecycle := fedutil.ClusterLifecycleHandlerFuncs{
		ClusterAvailable: func(cluster *fedv1.Cluster) {
			fdc.clusterDeliverer.DeliverAfter(allClustersKey, nil, clusterAvailableDelay)
		},
		ClusterUnavailable: func(cluster *fedv1.Cluster, _ []interface{}) {
			fdc.clusterDeliverer.DeliverAfter(allClustersKey, nil, clusterUnavailableDelay)
		},
	}
	fdc.fedDeploymentInformer = fedutil.NewFederatedInformer(federationClient, deploymentFedInformerFactory, &clusterLifecycle)

	podFedInformerFactory := func(cluster *fedv1.Cluster, clientset kubeclientset.Interface) (cache.Store, cache.Controller) {
		return cache.NewInformer(
			&cache.ListWatch{
				ListFunc: func(options metav1.ListOptions) (runtime.Object, error) {
					return clientset.Core().Pods(metav1.NamespaceAll).List(options)
				},
				WatchFunc: func(options metav1.ListOptions) (watch.Interface, error) {
					return clientset.Core().Pods(metav1.NamespaceAll).Watch(options)
				},
			},
			&apiv1.Pod{},
			controller.NoResyncPeriodFunc(),
			fedutil.NewTriggerOnAllChanges(
				func(obj runtime.Object) {
					fdc.clusterDeliverer.DeliverAfter(allClustersKey, nil, allDeploymentReviewDelay)
				},
			),
		)
	}
	fdc.fedPodInformer = fedutil.NewFederatedInformer(federationClient, podFedInformerFactory, &fedutil.ClusterLifecycleHandlerFuncs{})

	fdc.deploymentStore, fdc.deploymentController = cache.NewInformer(
		&cache.ListWatch{
			ListFunc: func(options metav1.ListOptions) (runtime.Object, error) {
				return fdc.fedClient.Extensions().Deployments(metav1.NamespaceAll).List(options)
			},
			WatchFunc: func(options metav1.ListOptions) (watch.Interface, error) {
				return fdc.fedClient.Extensions().Deployments(metav1.NamespaceAll).Watch(options)
			},
		},
		&extensionsv1.Deployment{},
		controller.NoResyncPeriodFunc(),
		fedutil.NewTriggerOnMetaAndSpecChanges(
			func(obj runtime.Object) { fdc.deliverFedDeploymentObj(obj, deploymentReviewDelay) },
		),
	)

	fdc.fedUpdater = fedutil.NewFederatedUpdater(fdc.fedDeploymentInformer,
		func(client kubeclientset.Interface, obj runtime.Object) error {
			rs := obj.(*extensionsv1.Deployment)
			_, err := client.Extensions().Deployments(rs.Namespace).Create(rs)
			return err
		},
		func(client kubeclientset.Interface, obj runtime.Object) error {
			rs := obj.(*extensionsv1.Deployment)
			_, err := client.Extensions().Deployments(rs.Namespace).Update(rs)
			return err
		},
		func(client kubeclientset.Interface, obj runtime.Object) error {
			rs := obj.(*extensionsv1.Deployment)
			err := client.Extensions().Deployments(rs.Namespace).Delete(rs.Name, &metav1.DeleteOptions{})
			return err
		})

	fdc.deletionHelper = deletionhelper.NewDeletionHelper(
		fdc.hasFinalizerFunc,
		fdc.removeFinalizerFunc,
		fdc.addFinalizerFunc,
		// objNameFunc
		func(obj runtime.Object) string {
			deployment := obj.(*extensionsv1.Deployment)
			return deployment.Name
		},
		updateTimeout,
		fdc.eventRecorder,
		fdc.fedDeploymentInformer,
		fdc.fedUpdater,
	)

	return fdc
}

// Returns true if the given object has the given finalizer in its ObjectMeta.
func (fdc *DeploymentController) hasFinalizerFunc(obj runtime.Object, finalizer string) bool {
	deployment := obj.(*extensionsv1.Deployment)
	for i := range deployment.ObjectMeta.Finalizers {
		if string(deployment.ObjectMeta.Finalizers[i]) == finalizer {
			return true
		}
	}
	return false
}

// Removes the finalizer from the given objects ObjectMeta.
// Assumes that the given object is a deployment.
func (fdc *DeploymentController) removeFinalizerFunc(obj runtime.Object, finalizer string) (runtime.Object, error) {
	deployment := obj.(*extensionsv1.Deployment)
	newFinalizers := []string{}
	hasFinalizer := false
	for i := range deployment.ObjectMeta.Finalizers {
		if string(deployment.ObjectMeta.Finalizers[i]) != finalizer {
			newFinalizers = append(newFinalizers, deployment.ObjectMeta.Finalizers[i])
		} else {
			hasFinalizer = true
		}
	}
	if !hasFinalizer {
		// Nothing to do.
		return obj, nil
	}
	deployment.ObjectMeta.Finalizers = newFinalizers
	deployment, err := fdc.fedClient.Extensions().Deployments(deployment.Namespace).Update(deployment)
	if err != nil {
		return nil, fmt.Errorf("failed to remove finalizer %s from deployment %s: %v", finalizer, deployment.Name, err)
	}
	return deployment, nil
}

// Adds the given finalizers to the given objects ObjectMeta.
// Assumes that the given object is a deployment.
func (fdc *DeploymentController) addFinalizerFunc(obj runtime.Object, finalizers []string) (runtime.Object, error) {
	deployment := obj.(*extensionsv1.Deployment)
	deployment.ObjectMeta.Finalizers = append(deployment.ObjectMeta.Finalizers, finalizers...)
	deployment, err := fdc.fedClient.Extensions().Deployments(deployment.Namespace).Update(deployment)
	if err != nil {
		return nil, fmt.Errorf("failed to add finalizers %v to deployment %s: %v", finalizers, deployment.Name, err)
	}
	return deployment, nil
}

func (fdc *DeploymentController) Run(workers int, stopCh <-chan struct{}) {
	go fdc.deploymentController.Run(stopCh)
	fdc.fedDeploymentInformer.Start()
	fdc.fedPodInformer.Start()

	fdc.deploymentDeliverer.StartWithHandler(func(item *fedutil.DelayingDelivererItem) {
		fdc.deploymentWorkQueue.Add(item.Key)
	})
	fdc.clusterDeliverer.StartWithHandler(func(_ *fedutil.DelayingDelivererItem) {
		fdc.reconcileDeploymentsOnClusterChange()
	})

	// Wait until the cluster is synced to prevent the update storm at the very beginning.
	for !fdc.isSynced() {
		time.Sleep(5 * time.Millisecond)
		glog.V(3).Infof("Waiting for controller to sync up")
	}

	for i := 0; i < workers; i++ {
		go wait.Until(fdc.worker, time.Second, stopCh)
	}

	fedutil.StartBackoffGC(fdc.deploymentBackoff, stopCh)

	<-stopCh
	glog.Infof("Shutting down DeploymentController")
	fdc.deploymentDeliverer.Stop()
	fdc.clusterDeliverer.Stop()
	fdc.deploymentWorkQueue.ShutDown()
	fdc.fedDeploymentInformer.Stop()
	fdc.fedPodInformer.Stop()
}

func (fdc *DeploymentController) isSynced() bool {
	if !fdc.fedDeploymentInformer.ClustersSynced() {
		glog.V(2).Infof("Cluster list not synced")
		return false
	}
	clustersFromDeps, err := fdc.fedDeploymentInformer.GetReadyClusters()
	if err != nil {
		glog.Errorf("Failed to get ready clusters: %v", err)
		return false
	}
	if !fdc.fedDeploymentInformer.GetTargetStore().ClustersSynced(clustersFromDeps) {
		return false
	}

	if !fdc.fedPodInformer.ClustersSynced() {
		glog.V(2).Infof("Cluster list not synced")
		return false
	}
	clustersFromPods, err := fdc.fedPodInformer.GetReadyClusters()
	if err != nil {
		glog.Errorf("Failed to get ready clusters: %v", err)
		return false
	}

	// This also checks whether podInformer and deploymentInformer have the
	// same cluster lists.
	if !fdc.fedPodInformer.GetTargetStore().ClustersSynced(clustersFromDeps) {
		glog.V(2).Infof("Pod informer not synced")
		return false
	}
	if !fdc.fedPodInformer.GetTargetStore().ClustersSynced(clustersFromPods) {
		glog.V(2).Infof("Pod informer not synced")
		return false
	}

	if !fdc.deploymentController.HasSynced() {
		glog.V(2).Infof("federation deployment list not synced")
		return false
	}
	return true
}

func (fdc *DeploymentController) deliverLocalDeployment(obj interface{}, duration time.Duration) {
	key, err := controller.KeyFunc(obj)
	if err != nil {
		glog.Errorf("Couldn't get key for object %v: %v", obj, err)
		return
	}
	_, exists, err := fdc.deploymentStore.GetByKey(key)
	if err != nil {
		glog.Errorf("Couldn't get federation deployment %v: %v", key, err)
		return
	}
	if exists { // ignore deployments exists only in local k8s
		fdc.deliverDeploymentByKey(key, duration, false)
	}
}

func (fdc *DeploymentController) deliverFedDeploymentObj(obj interface{}, delay time.Duration) {
	key, err := controller.KeyFunc(obj)
	if err != nil {
		glog.Errorf("Couldn't get key for object %+v: %v", obj, err)
		return
	}
	fdc.deliverDeploymentByKey(key, delay, false)
}

func (fdc *DeploymentController) deliverDeploymentByKey(key string, delay time.Duration, failed bool) {
	if failed {
		fdc.deploymentBackoff.Next(key, time.Now())
		delay = delay + fdc.deploymentBackoff.Get(key)
	} else {
		fdc.deploymentBackoff.Reset(key)
	}
	fdc.deploymentDeliverer.DeliverAfter(key, nil, delay)
}

func (fdc *DeploymentController) worker() {
	for {
		item, quit := fdc.deploymentWorkQueue.Get()
		if quit {
			return
		}
		key := item.(string)
		status, err := fdc.reconcileDeployment(key)
		fdc.deploymentWorkQueue.Done(item)
		if err != nil {
			glog.Errorf("Error syncing cluster controller: %v", err)
			fdc.deliverDeploymentByKey(key, 0, true)
		} else {
			switch status {
			case statusAllOk:
				break
			case statusError:
				fdc.deliverDeploymentByKey(key, 0, true)
			case statusNeedRecheck:
				fdc.deliverDeploymentByKey(key, deploymentReviewDelay, false)
			case statusNotSynced:
				fdc.deliverDeploymentByKey(key, clusterAvailableDelay, false)
			default:
				glog.Errorf("Unhandled reconciliation status: %s", status)
				fdc.deliverDeploymentByKey(key, deploymentReviewDelay, false)
			}
		}
	}
}

func (fdc *DeploymentController) schedule(fd *extensionsv1.Deployment, clusters []*fedv1.Cluster,
	current map[string]int64, estimatedCapacity map[string]int64) map[string]int64 {
	// TODO: integrate real scheduler

	plannerToBeUsed := fdc.defaultPlanner
	fdPref, err := parseFederationDeploymentPreference(fd)
	if err != nil {
		glog.Info("Invalid Deployment specific preference, use default. deployment: %v, err: %v", fd.Name, err)
	}
	if fdPref != nil { // create a new planner if user specified a preference
		plannerToBeUsed = planner.NewPlanner(fdPref)
	}
	replicas := int64(0)
	if fd.Spec.Replicas != nil {
		replicas = int64(*fd.Spec.Replicas)
	}
	var clusterNames []string
	for _, cluster := range clusters {
		clusterNames = append(clusterNames, cluster.Name)
	}
	scheduleResult, overflow := plannerToBeUsed.Plan(replicas, clusterNames, current, estimatedCapacity,
		fd.Namespace+"/"+fd.Name)
	// make sure the result contains all clusters that currently have some replicas.
	result := make(map[string]int64)
	for clusterName := range current {
		result[clusterName] = 0
	}
	for clusterName, replicas := range scheduleResult {
		result[clusterName] = replicas
	}
	for clusterName, replicas := range overflow {
		result[clusterName] += replicas
	}
	if glog.V(4) {
		buf := bytes.NewBufferString(fmt.Sprintf("Schedule - Deployment: %s/%s\n", fd.Namespace, fd.Name))
		sort.Strings(clusterNames)
		for _, clusterName := range clusterNames {
			cur := current[clusterName]
			target := scheduleResult[clusterName]
			fmt.Fprintf(buf, "%s: current: %d target: %d", clusterName, cur, target)
			if over, found := overflow[clusterName]; found {
				fmt.Fprintf(buf, " overflow: %d", over)
			}
			if capacity, found := estimatedCapacity[clusterName]; found {
				fmt.Fprintf(buf, " capacity: %d", capacity)
			}
			fmt.Fprintf(buf, "\n")
		}
		glog.V(4).Infof(buf.String())
	}
	return result
}

type reconciliationStatus string

const (
	statusAllOk       = reconciliationStatus("ALL_OK")
	statusNeedRecheck = reconciliationStatus("RECHECK")
	statusError       = reconciliationStatus("ERROR")
	statusNotSynced   = reconciliationStatus("NOSYNC")
)

func (fdc *DeploymentController) reconcileDeployment(key string) (reconciliationStatus, error) {
	if !fdc.isSynced() {
		return statusNotSynced, nil
	}

	glog.V(4).Infof("Start reconcile deployment %q", key)
	startTime := time.Now()
	defer glog.V(4).Infof("Finished reconcile deployment %q (%v)", key, time.Now().Sub(startTime))

	objFromStore, exists, err := fdc.deploymentStore.GetByKey(key)
	if err != nil {
		return statusError, err
	}
	if !exists {
		// don't delete local deployments for now. Do not reconcile it anymore.
		return statusAllOk, nil
	}
	obj, err := api.Scheme.DeepCopy(objFromStore)
	fd, ok := obj.(*extensionsv1.Deployment)
	if err != nil || !ok {
		glog.Errorf("Error in retrieving obj from store: %v, %v", ok, err)
		return statusError, err
	}

	if fd.DeletionTimestamp != nil {
		if err := fdc.delete(fd); err != nil {
			glog.Errorf("Failed to delete %s: %v", fd.Name, err)
			fdc.eventRecorder.Eventf(fd, api.EventTypeNormal, "DeleteFailed",
				"Deployment delete failed: %v", err)
			return statusError, err
		}
		return statusAllOk, nil
	}

	glog.V(3).Infof("Ensuring delete object from underlying clusters finalizer for deployment: %s",
		fd.Name)
	// Add the required finalizers before creating a deployment in underlying clusters.
	updatedDeploymentObj, err := fdc.deletionHelper.EnsureFinalizers(fd)
	if err != nil {
		glog.Errorf("Failed to ensure delete object from underlying clusters finalizer in deployment %s: %v",
			fd.Name, err)
		return statusError, err
	}
	fd = updatedDeploymentObj.(*extensionsv1.Deployment)

	glog.V(3).Infof("Syncing deployment %s in underlying clusters", fd.Name)

	clusters, err := fdc.fedDeploymentInformer.GetReadyClusters()
	if err != nil {
		return statusError, err
	}

	// collect current status and do schedule
	allPods, err := fdc.fedPodInformer.GetTargetStore().List()
	if err != nil {
		return statusError, err
	}
	podStatus, err := podanalyzer.AnalysePods(fd.Spec.Selector, allPods, time.Now())
	current := make(map[string]int64)
	estimatedCapacity := make(map[string]int64)
	for _, cluster := range clusters {
		ldObj, exists, err := fdc.fedDeploymentInformer.GetTargetStore().GetByKey(cluster.Name, key)
		if err != nil {
			return statusError, err
		}
		if exists {
			ld := ldObj.(*extensionsv1.Deployment)
			current[cluster.Name] = int64(podStatus[cluster.Name].RunningAndReady) // include pending as well?
			unschedulable := int64(podStatus[cluster.Name].Unschedulable)
			if unschedulable > 0 {
				estimatedCapacity[cluster.Name] = int64(*ld.Spec.Replicas) - unschedulable
			}
		}
	}

	scheduleResult := fdc.schedule(fd, clusters, current, estimatedCapacity)

	glog.V(4).Infof("Start syncing local deployment %s: %v", key, scheduleResult)

	fedStatus := extensionsv1.DeploymentStatus{ObservedGeneration: fd.Generation}
	operations := make([]fedutil.FederatedOperation, 0)
	for clusterName, replicas := range scheduleResult {

		ldObj, exists, err := fdc.fedDeploymentInformer.GetTargetStore().GetByKey(clusterName, key)
		if err != nil {
			return statusError, err
		}

		// The object can be modified.
		ld := fedutil.DeepCopyDeployment(fd)
		specReplicas := int32(replicas)
		ld.Spec.Replicas = &specReplicas

		if !exists {
			if replicas > 0 {
				fdc.eventRecorder.Eventf(fd, api.EventTypeNormal, "CreateInCluster",
					"Creating deployment in cluster %s", clusterName)

				operations = append(operations, fedutil.FederatedOperation{
					Type:        fedutil.OperationTypeAdd,
					Obj:         ld,
					ClusterName: clusterName,
				})
			}
		} else {
			// TODO: Update only one deployment at a time if update strategy is rolling update.

			currentLd := ldObj.(*extensionsv1.Deployment)
			// Update existing replica set, if needed.
			if !fedutil.DeploymentEquivalent(ld, currentLd) {
				fdc.eventRecorder.Eventf(fd, api.EventTypeNormal, "UpdateInCluster",
					"Updating deployment in cluster %s", clusterName)

				operations = append(operations, fedutil.FederatedOperation{
					Type:        fedutil.OperationTypeUpdate,
					Obj:         ld,
					ClusterName: clusterName,
				})
				glog.Infof("Updating %s in %s", currentLd.Name, clusterName)
			}
			fedStatus.Replicas += currentLd.Status.Replicas
			fedStatus.AvailableReplicas += currentLd.Status.AvailableReplicas
			fedStatus.UnavailableReplicas += currentLd.Status.UnavailableReplicas
			fedStatus.ReadyReplicas += currentLd.Status.ReadyReplicas
		}
	}
	if fedStatus.Replicas != fd.Status.Replicas ||
		fedStatus.AvailableReplicas != fd.Status.AvailableReplicas ||
		fedStatus.UnavailableReplicas != fd.Status.UnavailableReplicas ||
		fedStatus.ReadyReplicas != fd.Status.ReadyReplicas {
		fd.Status = fedStatus
		_, err = fdc.fedClient.Extensions().Deployments(fd.Namespace).UpdateStatus(fd)
		if err != nil {
			return statusError, err
		}
	}

	if len(operations) == 0 {
		// Everything is in order
		return statusAllOk, nil
	}
	err = fdc.fedUpdater.UpdateWithOnError(operations, updateTimeout, func(op fedutil.FederatedOperation, operror error) {
		fdc.eventRecorder.Eventf(fd, api.EventTypeNormal, "FailedUpdateInCluster",
			"Deployment update in cluster %s failed: %v", op.ClusterName, operror)
	})
	if err != nil {
		glog.Errorf("Failed to execute updates for %s: %v", key, err)
		return statusError, err
	}

	// Some operations were made, reconcile after a while.
	return statusNeedRecheck, nil
}

func (fdc *DeploymentController) reconcileDeploymentsOnClusterChange() {
	if !fdc.isSynced() {
		fdc.clusterDeliverer.DeliverAfter(allClustersKey, nil, clusterAvailableDelay)
	}
	deps := fdc.deploymentStore.List()
	for _, dep := range deps {
		key, _ := controller.KeyFunc(dep)
		fdc.deliverDeploymentByKey(key, 0, false)
	}
}

// delete deletes the given deployment or returns error if the deletion was not complete.
func (fdc *DeploymentController) delete(deployment *extensionsv1.Deployment) error {
	glog.V(3).Infof("Handling deletion of deployment: %v", *deployment)
	_, err := fdc.deletionHelper.HandleObjectInUnderlyingClusters(deployment)
	if err != nil {
		return err
	}

	err = fdc.fedClient.Extensions().Deployments(deployment.Namespace).Delete(deployment.Name, nil)
	if err != nil {
		// Its all good if the error is not found error. That means it is deleted already and we do not have to do anything.
		// This is expected when we are processing an update as a result of deployment finalizer deletion.
		// The process that deleted the last finalizer is also going to delete the deployment and we do not have to do anything.
		if !errors.IsNotFound(err) {
			return fmt.Errorf("failed to delete deployment: %v", err)
		}
	}
	return nil
}
