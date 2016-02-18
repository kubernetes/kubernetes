/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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

package persistentvolume

//
//import (
//	"fmt"
//	"time"
//
//	"k8s.io/kubernetes/pkg/api"
//	"k8s.io/kubernetes/pkg/api/unversioned"
//	"k8s.io/kubernetes/pkg/client/cache"
//	clientset "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset"
//	"k8s.io/kubernetes/pkg/cloudprovider"
//	"k8s.io/kubernetes/pkg/controller"
//	"k8s.io/kubernetes/pkg/controller/framework"
//	"k8s.io/kubernetes/pkg/conversion"
//	"k8s.io/kubernetes/pkg/runtime"
//	"k8s.io/kubernetes/pkg/types"
//	"k8s.io/kubernetes/pkg/util/io"
//	"k8s.io/kubernetes/pkg/util/mount"
//	"k8s.io/kubernetes/pkg/util/wait"
//	"k8s.io/kubernetes/pkg/util/workqueue"
//	"k8s.io/kubernetes/pkg/volume"
//	"k8s.io/kubernetes/pkg/watch"
//
//	"github.com/golang/glog"
//)
//
//// PersistentVolumeClaimController is a controller that synchronizes PersistentVolumeClaims.
//type PersistentVolumeClaimController struct {
//	volumeIndex      *persistentVolumeOrderedIndex
//	volumeController *framework.Controller
//	claimController  *framework.Controller
//	volumeStore      cache.Store
//	claimStore       cache.Store
//	client           taskClient
//	stopChannels     map[string]chan struct{}
//	// PersistentVolumeClaims that need to be synced
//	claimQueue           *workqueue.Type
//	conditionQueues      map[api.PersistentVolumeClaimConditionType]*workqueue.Type
//	conditionHandlers    map[api.PersistentVolumeClaimConditionType]func(obj interface{}) error
//	conditionForgiveness map[api.PersistentVolumeClaimConditionType]time.Duration
//	cloud                cloudprovider.Interface
//	provisioner          volume.ProvisionableVolumePlugin
//	pluginMgr            volume.VolumePluginMgr
//}
//
//// NewPersistentVolumeClaimController creates a new PersistentVolumeClaimController
//func NewPersistentVolumeClaimController(taskClient taskClient, syncPeriod time.Duration, plugins []volume.VolumePlugin, provisioner volume.ProvisionableVolumePlugin, cloud cloudprovider.Interface) (*PersistentVolumeClaimController, error) {
//	volumeIndex := NewPersistentVolumeOrderedIndex()
//	ctrl := &PersistentVolumeClaimController{
//		volumeIndex: volumeIndex,
//		client:      taskClient,
//		claimQueue:  workqueue.New(),
//		cloud:       cloud,
//		provisioner: provisioner,
//	}
//
//	if err := ctrl.pluginMgr.InitPlugins(plugins, ctrl); err != nil {
//		return nil, fmt.Errorf("Could not initialize volume plugins for PersistentVolumeProvisionerController: %+v", err)
//	}
//
//	glog.V(5).Infof("Initializing provisioner: %s", ctrl.provisioner.Name())
//	ctrl.provisioner.Init(ctrl)
//
//	ctrl.conditionQueues = map[api.PersistentVolumeClaimConditionType]*workqueue.Type{api.PersistentVolumeClaimBound: workqueue.New()}
//	ctrl.conditionHandlers = map[api.PersistentVolumeClaimConditionType]func(obj interface{}) error{api.PersistentVolumeClaimBound: ctrl.syncBoundCondition}
//	ctrl.conditionForgiveness = map[api.PersistentVolumeClaimConditionType]time.Duration{api.PersistentVolumeClaimBound: syncPeriod}
//
//	ctrl.volumeStore, ctrl.volumeController = framework.NewInformer(
//		&cache.ListWatch{
//			ListFunc: func(options api.ListOptions) (runtime.Object, error) {
//				return taskClient.ListPersistentVolumes(options)
//			},
//			WatchFunc: func(options api.ListOptions) (watch.Interface, error) {
//				return taskClient.WatchPersistentVolumes(options)
//			},
//		},
//		&api.PersistentVolume{},
//		syncPeriod,
//		framework.ResourceEventHandlerFuncs{
//			AddFunc:    ctrl.addVolume,
//			UpdateFunc: ctrl.updateVolume,
//			DeleteFunc: ctrl.deleteVolume,
//		},
//	)
//	ctrl.claimStore, ctrl.claimController = framework.NewInformer(
//		&cache.ListWatch{
//			ListFunc: func(options api.ListOptions) (runtime.Object, error) {
//				return taskClient.ListPersistentVolumeClaims(api.NamespaceAll, options)
//			},
//			WatchFunc: func(options api.ListOptions) (watch.Interface, error) {
//				return taskClient.WatchPersistentVolumeClaims(api.NamespaceAll, options)
//			},
//		},
//		&api.PersistentVolumeClaim{},
//		syncPeriod,
//		framework.ResourceEventHandlerFuncs{
//			AddFunc: func(obj interface{}) {
//				if pvc, ok := obj.(*api.PersistentVolumeClaim); ok {
//					ctrl.enqueueClaim(pvc)
//				}
//			},
//			UpdateFunc: func(obj, current interface{}) {
//				if pvc, ok := current.(*api.PersistentVolumeClaim); ok {
//					ctrl.enqueueClaim(pvc)
//				}
//			},
//		},
//	)
//
//	return ctrl, nil
//}
//
//func (ctrl *PersistentVolumeClaimController) addVolume(obj interface{}) {
//	if pv, ok := obj.(*api.PersistentVolume); ok {
//		ctrl.volumeIndex.Update(pv)
//	}
//}
//
//func (ctrl *PersistentVolumeClaimController) updateVolume(obj, current interface{}) {
//	if pv, ok := current.(*api.PersistentVolume); ok {
//		ctrl.volumeIndex.Update(pv)
//	}
//}
//
//func (ctrl *PersistentVolumeClaimController) deleteVolume(obj interface{}) {
//	if pv, ok := obj.(*api.PersistentVolume); ok {
//		ctrl.volumeIndex.Delete(pv)
//	}
//}
//
//func (ctrl *PersistentVolumeClaimController) enqueueClaim(claim *api.PersistentVolumeClaim) {
//	// might be *api.PVC or DeletionFinalStateUnknown
//	key, err := controller.KeyFunc(claim)
//	if err != nil {
//		glog.Errorf("Couldn't get key for object %+v: %v", claim, err)
//		return
//	}
//
//	ctrl.claimQueue.Add(key)
//}
//
//// worker runs a worker thread that just dequeues items, processes them, and marks them done.
//// It enforces that syncClaim is never invoked concurrently with the same key.
//func (ctrl *PersistentVolumeClaimController) worker(queueName string, queue *workqueue.Type, handleFunc func(obj interface{}) error) {
//	for {
//		func() {
//			glog.V(5).Infof("Waiting on queue %s with %d depth", queueName, queue.Len())
//			key, quit := queue.Get()
//			if quit {
//				return
//			}
//			glog.V(5).Infof("Processing %s from queue %s", key, queueName)
//			defer queue.Done(key)
//			glog.V(5).Infof("Invoking handler %#v", handleFunc)
//			err := handleFunc(key.(string))
//			if err != nil {
//				glog.Errorf("PersistentVolumeClaim[%s] error syncing: %v", err)
//			}
//		}()
//	}
//}
//
//func (ctrl *PersistentVolumeClaimController) syncClaim(key interface{}) (err error) {
//	glog.V(5).Infof("PersistentVolumeClaim[%v] syncing\n", key)
//
//	obj, exists, err := ctrl.claimStore.GetByKey(key.(string))
//	if err != nil {
//		glog.Infof("PersistentVolumeClaim[%v] not found in local store: %v", key, err)
//		ctrl.claimQueue.Add(key)
//		return err
//	}
//
//	if !exists {
//		glog.Infof("PersistentVolumeClaim[%v] has been deleted %v", key)
//		return nil
//	}
//
//	pvc := *obj.(*api.PersistentVolumeClaim)
//
//	glog.V(5).Infof("PersistentVolumeClaim[%s] %d conditions:  %#v", key, len(pvc.Status.Conditions), pvc.Status.Conditions)
//
//	for c, condition := range pvc.Status.Conditions {
//		ctrl.conditionQueues[condition.Type].Add(key)
//		glog.V(5).Infof("PersistentVolumeClaim[%s] equeue for condition %v", key, c)
//	}
//	return nil
//}
//
//func (ctrl *PersistentVolumeClaimController) syncBoundCondition(key interface{}) (err error) {
//	glog.V(5).Infof("PersistentVolumeClaim[%s] syncing bound condition\n", key)
//
//	obj, exists, err := ctrl.claimStore.GetByKey(key.(string))
//	if err != nil {
//		glog.Infof("PersistentVolumeClaim[%s] not found in local store: %v", key, err)
//		return err
//	}
//
//	if !exists {
//		glog.Infof("PersistentVolumeClaim[%s] has been deleted %v", key)
//		return nil
//	}
//
//	pvc := obj.(*api.PersistentVolumeClaim)
//
//	for _, condition := range pvc.Status.Conditions {
//		glog.V(5).Infof("PersistentVolumeClaim[%s] syncing %v condition", key, condition)
//		if condition.Type == api.PersistentVolumeClaimBound && ctrl.hasExceededForgiveness(condition) {
//			glog.V(5).Infof("PersistentVolumeClaim[%s] %v Condition has exceeded its forgiveness, attempting sync", key, api.PersistentVolumeClaimBound)
//			err := ctrl.syncBoundConditionForClaim(pvc)
//			if err != nil {
//				ctrl.conditionQueues[api.PersistentVolumeClaimBound].Add(key)
//				glog.V(5).Info("PersistentVolumeClaim[%s] is re-queued for %v: %v", key, api.PersistentVolumeClaimBound, err)
//				return err
//			}
//			glog.V(5).Infof("PersistentVolumeClaim[%s] is in sync with %v Condition", key, api.PersistentVolumeClaimBound)
//		}
//	}
//	return nil
//}
//
//func (ctrl *PersistentVolumeClaimController) syncBoundConditionForClaim(pvc *api.PersistentVolumeClaim) error {
//	if len(pvc.Status.Conditions) == 0 {
//		return fmt.Errorf("PersistentVolumeClaim[%s/%s] expected more than 0 Conditions, found %d\n", pvc.Namespace, pvc.Name, len(pvc.Status.Conditions))
//	}
//
//	// a cloned claim is used to avoid mutating the object stored in local cache.
//	// this is a locally scoped object clone (i.g, two instances of the same resource)
//	clone, err := conversion.NewCloner().DeepCopy(pvc)
//	if err != nil {
//		return fmt.Errorf("Error cloning pvc: %v", err)
//	}
//	claim, ok := clone.(*api.PersistentVolumeClaim)
//	if !ok {
//		return fmt.Errorf("Unexpected pv cast error : %v\n", clone)
//	}
//
//	// set the latest heartbeat time on the object to take ownership of the claim.
//	// a resource version error from the API means another process may have taken this already.
//	// any competing processes that attempts to persist the heartbeat will receive version error and return.
//	boundIndex := 0
//	for i, condition := range claim.Status.Conditions {
//		if condition.Type == api.PersistentVolumeClaimBound {
//			boundIndex = i
//			claim.Status.Conditions[i].LastProbeTime = unversioned.Now()
//			updatedClaim, err := ctrl.client.UpdatePersistentVolumeClaimStatus(claim)
//			if err != nil {
//				return fmt.Errorf("Error saving PersistentVolumeClaim.Status: %+v", err)
//			}
//			claim = updatedClaim
//			break
//		}
//		return fmt.Errorf("PersistentVolumeClaim[%s/%s] has no bound condition", claim.Namespace, claim.Name, err)
//	}
//
//	// claims will always match their bound volume
//	volume, err := ctrl.volumeIndex.findBestMatchForClaim(claim)
//	if err != nil {
//		return err
//	}
//
//	if volume == nil {
//		claim.Status.Conditions[0].Status = api.ConditionFalse
//		claim.Status.Conditions[0].Reason = "NoMatchFound"
//		_, err := ctrl.client.UpdatePersistentVolumeClaimStatus(claim)
//		if err != nil {
//			return fmt.Errorf("Error saving PersistentVolumeClaim.Status: %+v", err)
//		}
//		return nil
//	}
//
//	// volume is unbound. the claim can take ownership of this storage resource
//	// by saving its own ClaimRef on the volume.  Any competing process will fail with a resource version
//	// error if they attempt to persist the volume.  Likewise, if another process has already altered this volume,
//	// return the error and allow this condition sync task to requeue and try again for another volume
//	if volume.Spec.ClaimRef == nil {
//		claimRef, err := api.GetReference(claim)
//		if err != nil {
//			return fmt.Errorf("Unexpected error getting claim reference: %v\n", err)
//		}
//		clone, err := conversion.NewCloner().DeepCopy(volume)
//		if err != nil {
//			return fmt.Errorf("Error cloning pv: %v", err)
//		}
//		volumeClone, ok := clone.(*api.PersistentVolume)
//		if !ok {
//			return fmt.Errorf("Unexpected pv cast error : %v\n", volumeClone)
//		}
//		volumeClone.Spec.ClaimRef = claimRef
//		if updatedVolume, err := ctrl.client.UpdatePersistentVolume(volumeClone); err != nil {
//			return fmt.Errorf("Unexpected error saving PersistentVolume.Status: %+v", err)
//		} else {
//			volume = updatedVolume
//		}
//	}
//
//	// wait until the volume is also fully bound.
//	// if volume isn't ready, return an error so this task can requeue
//	if len(volume.Status.Conditions) == 0 {
//		return fmt.Errorf("PersistentVolume[%s] missing bound condition", volume.Name)
//	}
//	for _, condition := range volume.Status.Conditions {
//		if condition.Type == api.PersistentVolumeBound && condition.Status != api.ConditionTrue {
//			return fmt.Errorf("PersistentVolumeClaim[%s/%s] is bound but its backing volume is not yet ready. Requeueing", claim.Namespace, claim.Name)
//		}
//	}
//
//	// the bind is persisted on the volume above and will always match the claim in a search.
//	// claim would remain Pending if the update fails, so processing this state is idempotent.
//	// this only needs to be processed once.
//	if claim.Spec.VolumeName != volume.Name {
//		claim.Spec.VolumeName = volume.Name
//		claim, err = ctrl.client.UpdatePersistentVolumeClaim(claim)
//		if err != nil {
//			return fmt.Errorf("Error updating claim with VolumeName %s: %+v\n", volume.Name, err)
//		}
//	}
//
//	// the condition is satisfied. the claim is bound and ready.
//	if claim.Status.Phase != api.ClaimBound || claim.Status.Conditions[boundIndex].Status != api.ConditionTrue {
//		claim.Status.Phase = api.ClaimBound
//		claim.Status.AccessModes = volume.Spec.AccessModes
//		claim.Status.Capacity = volume.Spec.Capacity
//		claim.Status.Conditions[boundIndex].Status = api.ConditionTrue
//		claim.Status.Conditions[boundIndex].Reason = "Bound"
//		_, err = ctrl.client.UpdatePersistentVolumeClaimStatus(claim)
//		if err != nil {
//			return fmt.Errorf("Unexpected error saving claim status: %+v", err)
//		}
//	}
//
//	return nil
//}
//
//func (ctrl *PersistentVolumeClaimController) hasExceededForgiveness(condition api.PersistentVolumeClaimCondition) bool {
//	return condition.LastProbeTime.Add(ctrl.conditionForgiveness[condition.Type]).Before(time.Now())
//}
//
//// Run starts all of this binder's control loops
//func (ctrl *PersistentVolumeClaimController) Run(workers int, stopCh <-chan struct{}) {
//	glog.V(5).Infof("Starting PersistentVolumeClaimController\n")
//	go ctrl.volumeController.Run(stopCh)
//	go ctrl.claimController.Run(stopCh)
//	for i := 0; i < workers; i++ {
//		go wait.Until(func() { ctrl.worker("heartbeat", ctrl.claimQueue, ctrl.syncClaim) }, time.Second, stopCh)
//		for condition, queue := range ctrl.conditionQueues {
//			if handler, exists := ctrl.conditionHandlers[condition]; exists {
//				go wait.Until(func() { ctrl.worker("condition", queue, handler) }, time.Second, stopCh)
//			}
//		}
//	}
//}
//
//// VolumeHost implementation
//// PersistentVolumeRecycler is host to the volume plugins, but does not actually mount any volumes.
//// Because no mounting is performed, most of the VolumeHost methods are not implemented.
//func (c *PersistentVolumeClaimController) GetPluginDir(podUID string) string {
//	return ""
//}
//
//func (c *PersistentVolumeClaimController) GetPodVolumeDir(podUID types.UID, pluginName, volumeName string) string {
//	return ""
//}
//
//func (c *PersistentVolumeClaimController) GetPodPluginDir(podUID types.UID, pluginName string) string {
//	return ""
//}
//
////func (c *PersistentVolumeClaimController) GetKubeClient() clientset.Interface {
////	return c.client.GetKubeClient()
////}
//
//func (c *PersistentVolumeClaimController) NewWrapperBuilder(volName string, spec volume.Spec, pod *api.Pod, opts volume.VolumeOptions) (volume.Builder, error) {
//	return nil, fmt.Errorf("NewWrapperBuilder not supported by PVClaimBinder's VolumeHost implementation")
//}
//
//func (c *PersistentVolumeClaimController) NewWrapperCleaner(volName string, spec volume.Spec, podUID types.UID) (volume.Cleaner, error) {
//	return nil, fmt.Errorf("NewWrapperCleaner not supported by PVClaimBinder's VolumeHost implementation")
//}
//
//func (c *PersistentVolumeClaimController) GetCloudProvider() cloudprovider.Interface {
//	return c.cloud
//}
//
//func (c *PersistentVolumeClaimController) GetMounter() mount.Interface {
//	return nil
//}
//
//func (c *PersistentVolumeClaimController) GetWriter() io.Writer {
//	return nil
//}
//
//func (c *PersistentVolumeClaimController) GetHostName() string {
//	return ""
//}
