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

// Package volume implements a controller to manage snapshot operations.
package snapshot

import (
	"fmt"
	"net"
	"time"

	"github.com/golang/glog"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset"
	"k8s.io/kubernetes/pkg/client/record"
	"k8s.io/kubernetes/pkg/cloudprovider"
	"k8s.io/kubernetes/pkg/controller/framework"
	"k8s.io/kubernetes/pkg/controller/volume/snapshot/cache"
	"k8s.io/kubernetes/pkg/controller/volume/snapshot/reconciler"
	"k8s.io/kubernetes/pkg/types"
	"k8s.io/kubernetes/pkg/util/io"
	"k8s.io/kubernetes/pkg/util/mount"
	"k8s.io/kubernetes/pkg/util/runtime"
	"k8s.io/kubernetes/pkg/volume"
	"k8s.io/kubernetes/pkg/volume/util/operationexecutor"
	"k8s.io/kubernetes/pkg/volume/util/volumehelper"
)

const (
	// loopPeriod is the amount of time the reconciler loop waits between
	// successive executions
	reconcilerLoopPeriod time.Duration = 100 * time.Millisecond
)

// SnapshotController defines the operations supported by this controller.
type SnapshotController interface {
	Run(stopCh <-chan struct{})
}

// NewSnapshotController returns a new instance of SnapshotController.
func NewSnapshotController(
	kubeClient internalclientset.Interface,
	recorder record.EventRecorder,
	pvcInformer framework.SharedInformer,
	pvInformer framework.SharedInformer,
	cloud cloudprovider.Interface,
	plugins []volume.VolumePlugin) (SnapshotController, error) {
	sc := &snapshotController{
		kubeClient:  kubeClient,
		pvcInformer: pvcInformer,
		pvInformer:  pvInformer,
		cloud:       cloud,
	}

	sc.pvcInformer.AddEventHandler(framework.ResourceEventHandlerFuncs{
		AddFunc:    sc.pvcAdd,
		UpdateFunc: sc.pvcUpdate,
		DeleteFunc: sc.pvcDelete,
	})

	if err := sc.volumePluginMgr.InitPlugins(plugins, sc); err != nil {
		return nil, fmt.Errorf("Could not initialize volume plugins for Snapshot Controller: %+v", err)
	}

	sc.actualStateOfWorld = cache.NewActualStateOfWorld(&sc.volumePluginMgr)
	sc.snapshotOperationExecutor =
		operationexecutor.NewOperationExecutor(
			kubeClient,
			&sc.volumePluginMgr,
			recorder)
	sc.reconciler = reconciler.NewReconciler(
		reconcilerLoopPeriod,
		sc.actualStateOfWorld,
		sc.snapshotOperationExecutor)

	return sc, nil
}

type snapshotController struct {
	// kubeClient is the kube API client used by volumehost to communicate with
	// the API server.
	kubeClient internalclientset.Interface

	// pvcInformer is the shared PVC informer used to fetch and store PVC
	// objects from the API server. It is shared with other controllers and
	// therefore the PVC objects in its store should be treated as immutable.
	pvcInformer framework.SharedInformer

	// pvInformer is the shared PV informer used to fetch and store PV objects
	// from the API server. It is shared with other controllers and therefore
	// the PV objects in its store should be treated as immutable.
	pvInformer framework.SharedInformer

	// cloud provider used by volume host
	cloud cloudprovider.Interface

	// volumePluginMgr used to initialize and fetch volume plugins
	volumePluginMgr volume.VolumePluginMgr

	// actualStateOfWorld is a data structure containing which volumes,
	// according to this controller, need to be snapshotted.
	// It is updated when API pvc objects fetched by the informers and
	// upon successful completion of snapshot operations triggered by
	// the controller.
	actualStateOfWorld cache.ActualStateOfWorld

	// snapshotOperationExecutor is used to start asynchronous snapshot operations
	snapshotOperationExecutor operationexecutor.OperationExecutor

	// reconciler is used to run an asynchronous periodic loop to complete
	// requested snapshots.
	reconciler reconciler.Reconciler
}

func (sc *snapshotController) Run(stopCh <-chan struct{}) {
	defer runtime.HandleCrash()
	glog.Infof("Starting Snapshot Controller")

	go sc.reconciler.Run(stopCh)

	<-stopCh
	glog.Infof("Shutting down Snapshot Controller")
}

func (sc *snapshotController) pvcAdd(obj interface{}) {
	pvc, ok := obj.(*api.PersistentVolumeClaim)
	if pvc == nil || !ok {
		return
	}

	snapshotName, found := pvc.Annotations[api.AnnSnapshotCreate]
	if !found {
		// Ignore PVCs without the snapshot creation annotation
		return
	}

	volumeSpec, err := sc.createVolumeSpec(pvc, pvc.Namespace)
	if err != nil {
		glog.V(5).Infof(
			"Error processing pvc %q/%q: %v",
			pvc.Namespace,
			pvc.Name,
			err)
		eventErr := volumehelper.PostEventToPersistentVolumeClaim(
			sc.kubeClient,
			pvc,
			"SnapshotControllerError",
			fmt.Sprintf("Snapshot %q failed due to a controller error.", snapshotName),
			api.EventTypeWarning)
		if eventErr != nil {
			glog.V(5).Infof("Failed posting event to API server err=%v", eventErr)
		}
		removeErr := sc.removeCreateSnapshotAnnotation(pvc)
		if removeErr != nil {
			glog.V(5).Infof("Failed removing snapshot annotation err=%v", removeErr)
		}
		return
	}

	snapshottableVolumePlugin, err :=
		sc.volumePluginMgr.FindSnapshottablePluginBySpec(volumeSpec)
	if err != nil || snapshottableVolumePlugin == nil {
		glog.V(5).Infof(
			"Skipping snapshot on pvc %q/%q: it does not implement Snapshottable interface. err=%v",
			pvc.Namespace,
			pvc.Name,
			err)
		eventErr := volumehelper.PostEventToPersistentVolumeClaim(
			sc.kubeClient,
			pvc,
			"SnapshotSkipped",
			fmt.Sprintf("Snapshot %q skipped: underlying persistent volume does not support snapshotting.", snapshotName),
			api.EventTypeNormal)
		if eventErr != nil {
			glog.V(5).Infof("Failed posting event to API server err=%v", eventErr)
		}
		removeErr := sc.removeCreateSnapshotAnnotation(pvc)
		if removeErr != nil {
			glog.V(5).Infof("Failed removing snapshot annotation err=%v", removeErr)
		}
		return
	}

	_, addVolumeErr := sc.actualStateOfWorld.AddVolume(volumeSpec, pvc, snapshotName)
	if addVolumeErr != nil {
		glog.V(5).Infof(
			"Failed to add volume %q/%q (spec name %q) to actualStateOfTheWorld. err=%v",
			pvc.Namespace,
			pvc.Name,
			volumeSpec.Name(),
			err)
		eventErr := volumehelper.PostEventToPersistentVolumeClaim(
			sc.kubeClient,
			pvc,
			"SnapshotControllerError",
			fmt.Sprintf("Snapshot %q failed due to a controller error.", snapshotName),
			api.EventTypeWarning)
		if eventErr != nil {
			glog.V(5).Infof("Failed posting event to API server err=%v", eventErr)
		}
		removeErr := sc.removeCreateSnapshotAnnotation(pvc)
		if removeErr != nil {
			glog.V(5).Infof("Failed removing snapshot annotation err=%v", removeErr)
		}
		return
	}

	return
}

func (sc *snapshotController) pvcUpdate(oldObj, newObj interface{}) {
	// The flow for update is the same as add.
	sc.pvcAdd(newObj)
}

func (sc *snapshotController) pvcDelete(obj interface{}) {
	// This a no-op
	return
}

// createVolumeSpec creates and returns a mutable volume.Spec object for the
// specified PVC, dereferencing the PVC to get the underlying PV.
func (sc *snapshotController) createVolumeSpec(
	pvc *api.PersistentVolumeClaim, pvcNamespace string) (*volume.Spec, error) {
	//Fetch the real PV behind the claim
	pvName, pvcUID, err := sc.getPVCFromCacheExtractPV(
		pvcNamespace, pvc.Name)
	if err != nil {
		return nil, fmt.Errorf(
			"error processing PVC %q/%q: %v",
			pvcNamespace,
			pvc.Name,
			err)
	}

	glog.V(10).Infof(
		"Found bound PV for PVC (ClaimName %q/%q pvcUID %v): pvName=%q",
		pvcNamespace,
		pvc.Name,
		pvcUID,
		pvName)

	// Fetch actual PV object
	volumeSpec, err := sc.getPVSpecFromCache(
		pvName, true /*read only*/, pvcUID)
	if err != nil {
		return nil, fmt.Errorf(
			"error processing PVC %q/%q: %v",
			pvcNamespace,
			pvc.Name,
			err)
	}

	glog.V(10).Infof(
		"Extracted volumeSpec (%v) from bound PV (pvName %q) and PVC (ClaimName %q/%q pvcUID %v)",
		volumeSpec.Name,
		pvName,
		pvcNamespace,
		pvc.Name,
		pvcUID)

	return volumeSpec, nil
}

// getPVCFromCacheExtractPV fetches the PVC object with the given namespace and
// name from the shared internal PVC store extracts the name of the PV it is
// pointing to and returns it.
// This method returns an error if a PVC object does not exist in the cache
// with the given namespace/name.
// This method returns an error if the PVC object's phase is not "Bound".
func (sc *snapshotController) getPVCFromCacheExtractPV(
	namespace string, name string) (string, types.UID, error) {
	key := name
	if len(namespace) > 0 {
		key = namespace + "/" + name
	}

	pvcObj, exists, err := sc.pvcInformer.GetStore().GetByKey(key)
	if pvcObj == nil || !exists || err != nil {
		return "", "", fmt.Errorf(
			"failed to find PVC %q in PVCInformer cache. %v",
			key,
			err)
	}

	pvc, ok := pvcObj.(*api.PersistentVolumeClaim)
	if !ok || pvc == nil {
		return "", "", fmt.Errorf(
			"failed to cast %q object %#v to PersistentVolumeClaim",
			key,
			pvcObj)
	}

	if pvc.Status.Phase != api.ClaimBound || pvc.Spec.VolumeName == "" {
		return "", "", fmt.Errorf(
			"PVC %q has non-bound phase (%q) or empty pvc.Spec.VolumeName (%q)",
			key,
			pvc.Status.Phase,
			pvc.Spec.VolumeName)
	}

	return pvc.Spec.VolumeName, pvc.UID, nil
}

// getPVSpecFromCache fetches the PV object with the given name from the shared
// internal PV store and returns a volume.Spec representing it.
// This method returns an error if a PV object does not exist in the cache with
// the given name.
// This method deep copies the PV object so the caller may use the returned
// volume.Spec object without worrying about it mutating unexpectedly.
func (sc *snapshotController) getPVSpecFromCache(
	name string,
	pvcReadOnly bool,
	expectedClaimUID types.UID) (*volume.Spec, error) {
	pvObj, exists, err := sc.pvInformer.GetStore().GetByKey(name)
	if pvObj == nil || !exists || err != nil {
		return nil, fmt.Errorf(
			"failed to find PV %q in PVInformer cache. %v", name, err)
	}

	pv, ok := pvObj.(*api.PersistentVolume)
	if !ok || pv == nil {
		return nil, fmt.Errorf(
			"failed to cast %q object %#v to PersistentVolume", name, pvObj)
	}

	if pv.Spec.ClaimRef == nil {
		return nil, fmt.Errorf(
			"found PV object %q but it has a nil pv.Spec.ClaimRef indicating it is not yet bound to the claim",
			name)
	}

	if pv.Spec.ClaimRef.UID != expectedClaimUID {
		return nil, fmt.Errorf(
			"found PV object %q but its pv.Spec.ClaimRef.UID (%q) does not point to claim.UID (%q)",
			name,
			pv.Spec.ClaimRef.UID,
			expectedClaimUID)
	}

	// Do not return the object from the informer, since the store is shared it
	// may be mutated by another consumer.
	clonedPVObj, err := api.Scheme.DeepCopy(*pv)
	if err != nil || clonedPVObj == nil {
		return nil, fmt.Errorf(
			"failed to deep copy %q PV object. err=%v", name, err)
	}

	clonedPV, ok := clonedPVObj.(api.PersistentVolume)
	if !ok {
		return nil, fmt.Errorf(
			"failed to cast %q clonedPV %#v to PersistentVolume", name, pvObj)
	}

	return volume.NewSpecFromPersistentVolume(&clonedPV, pvcReadOnly), nil
}

func (sc *snapshotController) removeCreateSnapshotAnnotation(
	pvc *api.PersistentVolumeClaim) error {
	delete(pvc.Annotations, api.AnnSnapshotCreate)
	_, err := sc.kubeClient.Core().PersistentVolumeClaims(pvc.Namespace).Update(pvc)
	if err != nil {
		return fmt.Errorf(
			"Snapshot controller failed removing create-snapshot annotation from PVC %q with: %v",
			pvc.Name,
			err)
	}
	return nil
}

// VolumeHost implementation
// This is an unfortunate requirement of the current factoring of volume plugin
// initializing code. It requires kubelet specific methods used by the mounting
// code to be implemented by all initializers even if the initializer does not
// do mounting (like this Snapshot controller).
// Issue kubernetes/kubernetes/issues/14217 to fix this.
func (sc *snapshotController) GetPluginDir(podUID string) string {
	return ""
}

func (sc *snapshotController) GetPodVolumeDir(podUID types.UID, pluginName, volumeName string) string {
	return ""
}

func (sc *snapshotController) GetPodPluginDir(podUID types.UID, pluginName string) string {
	return ""
}

func (sc *snapshotController) GetKubeClient() internalclientset.Interface {
	return sc.kubeClient
}

func (sc *snapshotController) NewWrapperMounter(volName string, spec volume.Spec, pod *api.Pod, opts volume.VolumeOptions) (volume.Mounter, error) {
	return nil, fmt.Errorf("NewWrapperMounter not supported by Snapshot controller's VolumeHost implementation")
}

func (sc *snapshotController) NewWrapperUnmounter(volName string, spec volume.Spec, podUID types.UID) (volume.Unmounter, error) {
	return nil, fmt.Errorf("NewWrapperUnmounter not supported by Snapshot controller's VolumeHost implementation")
}

func (sc *snapshotController) GetCloudProvider() cloudprovider.Interface {
	return sc.cloud
}

func (sc *snapshotController) GetMounter() mount.Interface {
	return nil
}

func (sc *snapshotController) GetWriter() io.Writer {
	return nil
}

func (sc *snapshotController) GetHostName() string {
	return ""
}

func (sc *snapshotController) GetHostIP() (net.IP, error) {
	return nil, fmt.Errorf("GetHostIP() not supported by Snapshot controller's VolumeHost implementation")
}

func (sc *snapshotController) GetRootContext() string {
	return ""
}

func (sc *snapshotController) GetNodeAllocatable() (api.ResourceList, error) {
	return api.ResourceList{}, nil
}
