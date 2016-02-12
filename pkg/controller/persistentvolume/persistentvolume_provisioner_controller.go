/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

import (
	"fmt"
	"sync"
	"time"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/client/cache"
	clientset "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset"
	"k8s.io/kubernetes/pkg/cloudprovider"
	"k8s.io/kubernetes/pkg/controller/framework"
	"k8s.io/kubernetes/pkg/conversion"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/types"
	"k8s.io/kubernetes/pkg/util/io"
	"k8s.io/kubernetes/pkg/util/mount"
	"k8s.io/kubernetes/pkg/volume"
	"k8s.io/kubernetes/pkg/watch"

	"github.com/golang/glog"
)

// PersistentVolumeProvisionerController reconciles the state of all PersistentVolumes and PersistentVolumeClaims.
type PersistentVolumeProvisionerController struct {
	volumeController *framework.Controller
	volumeStore      cache.Store
	claimController  *framework.Controller
	claimStore       cache.Store
	client           controllerClient
	cloud            cloudprovider.Interface
	provisioner      volume.ProvisionableVolumePlugin
	pluginMgr        volume.VolumePluginMgr
	stopChannels     map[string]chan struct{}
	mutex            sync.RWMutex
}

// constant name values for the controllers stopChannels map.
// the controller uses these for graceful shutdown
const volumesStopChannel = "volumes"
const claimsStopChannel = "claims"

// NewPersistentVolumeProvisionerController creates a new PersistentVolumeProvisionerController
func NewPersistentVolumeProvisionerController(client controllerClient, syncPeriod time.Duration, plugins []volume.VolumePlugin, provisioner volume.ProvisionableVolumePlugin, cloud cloudprovider.Interface) (*PersistentVolumeProvisionerController, error) {
	controller := &PersistentVolumeProvisionerController{
		client:      client,
		cloud:       cloud,
		provisioner: provisioner,
	}

	if err := controller.pluginMgr.InitPlugins(plugins, controller); err != nil {
		return nil, fmt.Errorf("Could not initialize volume plugins for PersistentVolumeProvisionerController: %+v", err)
	}

	glog.V(5).Infof("Initializing provisioner: %s", controller.provisioner.Name())
	controller.provisioner.Init(controller)

	controller.volumeStore, controller.volumeController = framework.NewInformer(
		&cache.ListWatch{
			ListFunc: func(options api.ListOptions) (runtime.Object, error) {
				return client.ListPersistentVolumes(options)
			},
			WatchFunc: func(options api.ListOptions) (watch.Interface, error) {
				return client.WatchPersistentVolumes(options)
			},
		},
		&api.PersistentVolume{},
		syncPeriod,
		framework.ResourceEventHandlerFuncs{
			AddFunc:    controller.handleAddVolume,
			UpdateFunc: controller.handleUpdateVolume,
			// delete handler not needed in this controller.
			// volume deletion is handled by the recycler controller
		},
	)
	controller.claimStore, controller.claimController = framework.NewInformer(
		&cache.ListWatch{
			ListFunc: func(options api.ListOptions) (runtime.Object, error) {
				return client.ListPersistentVolumeClaims(api.NamespaceAll, options)
			},
			WatchFunc: func(options api.ListOptions) (watch.Interface, error) {
				return client.WatchPersistentVolumeClaims(api.NamespaceAll, options)
			},
		},
		&api.PersistentVolumeClaim{},
		syncPeriod,
		framework.ResourceEventHandlerFuncs{
			AddFunc:    controller.handleAddClaim,
			UpdateFunc: controller.handleUpdateClaim,
			// delete handler not needed.
			// normal recycling applies when a claim is deleted.
			// recycling is handled by the binding controller.
		},
	)

	return controller, nil
}

func (controller *PersistentVolumeProvisionerController) handleAddVolume(obj interface{}) {
	controller.mutex.Lock()
	defer controller.mutex.Unlock()
	cachedPv, _, _ := controller.volumeStore.Get(obj)
	if pv, ok := cachedPv.(*api.PersistentVolume); ok {
		err := controller.reconcileVolume(pv)
		if err != nil {
			glog.Errorf("Error reconciling volume %s: %+v", pv.Name, err)
		}
	}
}

func (controller *PersistentVolumeProvisionerController) handleUpdateVolume(oldObj, newObj interface{}) {
	// The flow for Update is the same as Add.
	// A volume is only provisioned if not done so already.
	controller.handleAddVolume(newObj)
}

func (controller *PersistentVolumeProvisionerController) handleAddClaim(obj interface{}) {
	controller.mutex.Lock()
	defer controller.mutex.Unlock()
	cachedPvc, exists, _ := controller.claimStore.Get(obj)
	if !exists {
		glog.Errorf("PersistentVolumeClaim does not exist in the local cache: %+v", obj)
		return
	}
	if pvc, ok := cachedPvc.(*api.PersistentVolumeClaim); ok {
		err := controller.reconcileClaim(pvc)
		if err != nil {
			glog.Errorf("Error encoutered reconciling claim %s: %+v", pvc.Name, err)
		}
	}
}

func (controller *PersistentVolumeProvisionerController) handleUpdateClaim(oldObj, newObj interface{}) {
	// The flow for Update is the same as Add.
	// A volume is only provisioned for a claim if not done so already.
	controller.handleAddClaim(newObj)
}

func (controller *PersistentVolumeProvisionerController) reconcileClaim(claim *api.PersistentVolumeClaim) error {
	glog.V(5).Infof("Synchronizing PersistentVolumeClaim[%s] for dynamic provisioning", claim.Name)

	// The claim may have been modified by parallel call to reconcileClaim, load
	// the current version.
	newClaim, err := controller.client.GetPersistentVolumeClaim(claim.Namespace, claim.Name)
	if err != nil {
		return fmt.Errorf("Cannot reload claim %s/%s: %v", claim.Namespace, claim.Name, err)
	}
	claim = newClaim
	err = controller.claimStore.Update(claim)
	if err != nil {
		return fmt.Errorf("Cannot update claim %s/%s: %v", claim.Namespace, claim.Name, err)
	}

	if controller.provisioner == nil {
		return fmt.Errorf("No provisioner configured for controller")
	}

	// no provisioning requested, return Pending. Claim may be pending indefinitely without a match.
	if !keyExists(qosProvisioningKey, claim.Annotations) {
		glog.V(5).Infof("PersistentVolumeClaim[%s] no provisioning required", claim.Name)
		return nil
	}
	if len(claim.Spec.VolumeName) != 0 {
		glog.V(5).Infof("PersistentVolumeClaim[%s] already bound. No provisioning required", claim.Name)
		return nil
	}
	if isAnnotationMatch(pvProvisioningRequiredAnnotationKey, pvProvisioningCompletedAnnotationValue, claim.Annotations) {
		glog.V(5).Infof("PersistentVolumeClaim[%s] is already provisioned.", claim.Name)
		return nil
	}

	glog.V(5).Infof("PersistentVolumeClaim[%s] provisioning", claim.Name)
	provisioner, err := newProvisioner(controller.provisioner, claim, nil)
	if err != nil {
		return fmt.Errorf("Unexpected error getting new provisioner for claim %s: %v\n", claim.Name, err)
	}
	newVolume, err := provisioner.NewPersistentVolumeTemplate()
	if err != nil {
		return fmt.Errorf("Unexpected error getting new volume template for claim %s: %v\n", claim.Name, err)
	}

	claimRef, err := api.GetReference(claim)
	if err != nil {
		return fmt.Errorf("Unexpected error getting claim reference for %s: %v\n", claim.Name, err)
	}

	storageClass, _ := claim.Annotations[qosProvisioningKey]

	// the creation of this volume is the bind to the claim.
	// The claim will match the volume during the next sync period when the volume is in the local cache
	newVolume.Spec.ClaimRef = claimRef
	newVolume.Annotations[pvProvisioningRequiredAnnotationKey] = "true"
	newVolume.Annotations[qosProvisioningKey] = storageClass
	newVolume, err = controller.client.CreatePersistentVolume(newVolume)
	glog.V(5).Infof("Unprovisioned PersistentVolume[%s] created for PVC[%s], which will be fulfilled in the storage provider", newVolume.Name, claim.Name)
	if err != nil {
		return fmt.Errorf("PersistentVolumeClaim[%s] failed provisioning: %+v", claim.Name, err)
	}

	claim.Annotations[pvProvisioningRequiredAnnotationKey] = pvProvisioningCompletedAnnotationValue
	_, err = controller.client.UpdatePersistentVolumeClaim(claim)
	if err != nil {
		glog.Errorf("error updating persistent volume claim: %v", err)
	}

	return nil
}

func (controller *PersistentVolumeProvisionerController) reconcileVolume(pv *api.PersistentVolume) error {
	glog.V(5).Infof("PersistentVolume[%s] reconciling", pv.Name)

	// The PV may have been modified by parallel call to reconcileVolume, load
	// the current version.
	newPv, err := controller.client.GetPersistentVolume(pv.Name)
	if err != nil {
		return fmt.Errorf("Cannot reload volume %s: %v", pv.Name, err)
	}
	pv = newPv

	if pv.Spec.ClaimRef == nil {
		glog.V(5).Infof("PersistentVolume[%s] is not bound to a claim.  No provisioning required", pv.Name)
		return nil
	}

	// TODO:  fix this leaky abstraction.  Had to make our own store key because ClaimRef fails the default keyfunc (no Meta on object).
	obj, exists, _ := controller.claimStore.GetByKey(fmt.Sprintf("%s/%s", pv.Spec.ClaimRef.Namespace, pv.Spec.ClaimRef.Name))
	if !exists {
		return fmt.Errorf("PersistentVolumeClaim[%s/%s] not found in local cache", pv.Spec.ClaimRef.Namespace, pv.Spec.ClaimRef.Name)
	}

	claim, ok := obj.(*api.PersistentVolumeClaim)
	if !ok {
		return fmt.Errorf("PersistentVolumeClaim expected, but got %v", obj)
	}

	// no provisioning required, volume is ready and Bound
	if !keyExists(pvProvisioningRequiredAnnotationKey, pv.Annotations) {
		glog.V(5).Infof("PersistentVolume[%s] does not require provisioning", pv.Name)
		return nil
	}

	// provisioning is completed, volume is ready.
	if isProvisioningComplete(pv) {
		glog.V(5).Infof("PersistentVolume[%s] is bound and provisioning is complete", pv.Name)
		if pv.Spec.ClaimRef.Namespace != claim.Namespace || pv.Spec.ClaimRef.Name != claim.Name {
			return fmt.Errorf("pre-bind mismatch - expected %s but found %s/%s", claimToClaimKey(claim), pv.Spec.ClaimRef.Namespace, pv.Spec.ClaimRef.Name)
		}
		return nil
	}

	// provisioning is incomplete.  Attempt to provision the volume.
	glog.V(5).Infof("PersistentVolume[%s] provisioning in progress", pv.Name)
	err = provisionVolume(pv, controller)
	if err != nil {
		return fmt.Errorf("Error provisioning PersistentVolume[%s]: %v", pv.Name, err)
	}

	return nil
}

// provisionVolume provisions a volume that has been created in the cluster but not yet fulfilled by
// the storage provider.
func provisionVolume(pv *api.PersistentVolume, controller *PersistentVolumeProvisionerController) error {
	if isProvisioningComplete(pv) {
		return fmt.Errorf("PersistentVolume[%s] is already provisioned", pv.Name)
	}

	if _, exists := pv.Annotations[qosProvisioningKey]; !exists {
		return fmt.Errorf("PersistentVolume[%s] does not contain a provisioning request.  Provisioning not required.", pv.Name)
	}

	if controller.provisioner == nil {
		return fmt.Errorf("No provisioner found for volume: %s", pv.Name)
	}

	// Find the claim in local cache
	obj, exists, _ := controller.claimStore.GetByKey(fmt.Sprintf("%s/%s", pv.Spec.ClaimRef.Namespace, pv.Spec.ClaimRef.Name))
	if !exists {
		return fmt.Errorf("Could not find PersistentVolumeClaim[%s/%s] in local cache", pv.Spec.ClaimRef.Name, pv.Name)
	}
	claim := obj.(*api.PersistentVolumeClaim)

	provisioner, _ := newProvisioner(controller.provisioner, claim, pv)
	err := provisioner.Provision(pv)
	if err != nil {
		glog.Errorf("Could not provision %s", pv.Name)
		pv.Status.Phase = api.VolumeFailed
		pv.Status.Message = err.Error()
		if pv, apiErr := controller.client.UpdatePersistentVolumeStatus(pv); apiErr != nil {
			return fmt.Errorf("PersistentVolume[%s] failed provisioning and also failed status update: %v  -  %v", pv.Name, err, apiErr)
		}
		return fmt.Errorf("PersistentVolume[%s] failed provisioning: %v", pv.Name, err)
	}

	clone, err := conversion.NewCloner().DeepCopy(pv)
	volumeClone, ok := clone.(*api.PersistentVolume)
	if !ok {
		return fmt.Errorf("Unexpected pv cast error : %v\n", volumeClone)
	}
	volumeClone.Annotations[pvProvisioningRequiredAnnotationKey] = pvProvisioningCompletedAnnotationValue

	pv, err = controller.client.UpdatePersistentVolume(volumeClone)
	if err != nil {
		// TODO:  https://github.com/kubernetes/kubernetes/issues/14443
		// the volume was created in the infrastructure and likely has a PV name on it,
		// but we failed to save the annotation that marks the volume as provisioned.
		return fmt.Errorf("Error updating PersistentVolume[%s] with provisioning completed annotation. There is a potential for dupes and orphans.", volumeClone.Name)
	}
	return nil
}

// Run starts all of this controller's control loops
func (controller *PersistentVolumeProvisionerController) Run() {
	glog.V(5).Infof("Starting PersistentVolumeProvisionerController\n")
	if controller.stopChannels == nil {
		controller.stopChannels = make(map[string]chan struct{})
	}

	if _, exists := controller.stopChannels[volumesStopChannel]; !exists {
		controller.stopChannels[volumesStopChannel] = make(chan struct{})
		go controller.volumeController.Run(controller.stopChannels[volumesStopChannel])
	}

	if _, exists := controller.stopChannels[claimsStopChannel]; !exists {
		controller.stopChannels[claimsStopChannel] = make(chan struct{})
		go controller.claimController.Run(controller.stopChannels[claimsStopChannel])
	}
}

// Stop gracefully shuts down this controller
func (controller *PersistentVolumeProvisionerController) Stop() {
	glog.V(5).Infof("Stopping PersistentVolumeProvisionerController\n")
	for name, stopChan := range controller.stopChannels {
		close(stopChan)
		delete(controller.stopChannels, name)
	}
}

func newProvisioner(plugin volume.ProvisionableVolumePlugin, claim *api.PersistentVolumeClaim, pv *api.PersistentVolume) (volume.Provisioner, error) {
	tags := make(map[string]string)
	tags[cloudVolumeCreatedForClaimNamespaceTag] = claim.Namespace
	tags[cloudVolumeCreatedForClaimNameTag] = claim.Name

	// pv can be nil when the provisioner has not created the PV yet
	if pv != nil {
		tags[cloudVolumeCreatedForVolumeNameTag] = pv.Name
	}

	volumeOptions := volume.VolumeOptions{
		Capacity:                      claim.Spec.Resources.Requests[api.ResourceName(api.ResourceStorage)],
		AccessModes:                   claim.Spec.AccessModes,
		PersistentVolumeReclaimPolicy: api.PersistentVolumeReclaimDelete,
		CloudTags:                     &tags,
	}

	provisioner, err := plugin.NewProvisioner(volumeOptions)
	return provisioner, err
}

// controllerClient abstracts access to PVs and PVCs.  Easy to mock for testing and wrap for real client.
type controllerClient interface {
	CreatePersistentVolume(pv *api.PersistentVolume) (*api.PersistentVolume, error)
	ListPersistentVolumes(options api.ListOptions) (*api.PersistentVolumeList, error)
	WatchPersistentVolumes(options api.ListOptions) (watch.Interface, error)
	GetPersistentVolume(name string) (*api.PersistentVolume, error)
	UpdatePersistentVolume(volume *api.PersistentVolume) (*api.PersistentVolume, error)
	DeletePersistentVolume(volume *api.PersistentVolume) error
	UpdatePersistentVolumeStatus(volume *api.PersistentVolume) (*api.PersistentVolume, error)

	GetPersistentVolumeClaim(namespace, name string) (*api.PersistentVolumeClaim, error)
	ListPersistentVolumeClaims(namespace string, options api.ListOptions) (*api.PersistentVolumeClaimList, error)
	WatchPersistentVolumeClaims(namespace string, options api.ListOptions) (watch.Interface, error)
	UpdatePersistentVolumeClaim(claim *api.PersistentVolumeClaim) (*api.PersistentVolumeClaim, error)
	UpdatePersistentVolumeClaimStatus(claim *api.PersistentVolumeClaim) (*api.PersistentVolumeClaim, error)

	// provided to give VolumeHost and plugins access to the kube client
	GetKubeClient() clientset.Interface
}

func NewControllerClient(c clientset.Interface) controllerClient {
	return &realControllerClient{c}
}

var _ controllerClient = &realControllerClient{}

type realControllerClient struct {
	client clientset.Interface
}

func (c *realControllerClient) GetPersistentVolume(name string) (*api.PersistentVolume, error) {
	return c.client.Core().PersistentVolumes().Get(name)
}

func (c *realControllerClient) ListPersistentVolumes(options api.ListOptions) (*api.PersistentVolumeList, error) {
	return c.client.Core().PersistentVolumes().List(options)
}

func (c *realControllerClient) WatchPersistentVolumes(options api.ListOptions) (watch.Interface, error) {
	return c.client.Core().PersistentVolumes().Watch(options)
}

func (c *realControllerClient) CreatePersistentVolume(pv *api.PersistentVolume) (*api.PersistentVolume, error) {
	return c.client.Core().PersistentVolumes().Create(pv)
}

func (c *realControllerClient) UpdatePersistentVolume(volume *api.PersistentVolume) (*api.PersistentVolume, error) {
	return c.client.Core().PersistentVolumes().Update(volume)
}

func (c *realControllerClient) DeletePersistentVolume(volume *api.PersistentVolume) error {
	return c.client.Core().PersistentVolumes().Delete(volume.Name, nil)
}

func (c *realControllerClient) UpdatePersistentVolumeStatus(volume *api.PersistentVolume) (*api.PersistentVolume, error) {
	return c.client.Core().PersistentVolumes().UpdateStatus(volume)
}

func (c *realControllerClient) GetPersistentVolumeClaim(namespace, name string) (*api.PersistentVolumeClaim, error) {
	return c.client.Core().PersistentVolumeClaims(namespace).Get(name)
}

func (c *realControllerClient) ListPersistentVolumeClaims(namespace string, options api.ListOptions) (*api.PersistentVolumeClaimList, error) {
	return c.client.Core().PersistentVolumeClaims(namespace).List(options)
}

func (c *realControllerClient) WatchPersistentVolumeClaims(namespace string, options api.ListOptions) (watch.Interface, error) {
	return c.client.Core().PersistentVolumeClaims(namespace).Watch(options)
}

func (c *realControllerClient) UpdatePersistentVolumeClaim(claim *api.PersistentVolumeClaim) (*api.PersistentVolumeClaim, error) {
	return c.client.Core().PersistentVolumeClaims(claim.Namespace).Update(claim)
}

func (c *realControllerClient) UpdatePersistentVolumeClaimStatus(claim *api.PersistentVolumeClaim) (*api.PersistentVolumeClaim, error) {
	return c.client.Core().PersistentVolumeClaims(claim.Namespace).UpdateStatus(claim)
}

func (c *realControllerClient) GetKubeClient() clientset.Interface {
	return c.client
}

func keyExists(key string, haystack map[string]string) bool {
	_, exists := haystack[key]
	return exists
}

func isProvisioningComplete(pv *api.PersistentVolume) bool {
	return isAnnotationMatch(pvProvisioningRequiredAnnotationKey, pvProvisioningCompletedAnnotationValue, pv.Annotations)
}

func isAnnotationMatch(key, needle string, haystack map[string]string) bool {
	value, exists := haystack[key]
	if !exists {
		return false
	}
	return value == needle
}

func isRecyclable(policy api.PersistentVolumeReclaimPolicy) bool {
	return policy == api.PersistentVolumeReclaimDelete || policy == api.PersistentVolumeReclaimRecycle
}

// VolumeHost implementation
// PersistentVolumeRecycler is host to the volume plugins, but does not actually mount any volumes.
// Because no mounting is performed, most of the VolumeHost methods are not implemented.
func (c *PersistentVolumeProvisionerController) GetPluginDir(podUID string) string {
	return ""
}

func (c *PersistentVolumeProvisionerController) GetPodVolumeDir(podUID types.UID, pluginName, volumeName string) string {
	return ""
}

func (c *PersistentVolumeProvisionerController) GetPodPluginDir(podUID types.UID, pluginName string) string {
	return ""
}

func (c *PersistentVolumeProvisionerController) GetKubeClient() clientset.Interface {
	return c.client.GetKubeClient()
}

func (c *PersistentVolumeProvisionerController) NewWrapperBuilder(volName string, spec volume.Spec, pod *api.Pod, opts volume.VolumeOptions) (volume.Builder, error) {
	return nil, fmt.Errorf("NewWrapperBuilder not supported by PVClaimBinder's VolumeHost implementation")
}

func (c *PersistentVolumeProvisionerController) NewWrapperCleaner(volName string, spec volume.Spec, podUID types.UID) (volume.Cleaner, error) {
	return nil, fmt.Errorf("NewWrapperCleaner not supported by PVClaimBinder's VolumeHost implementation")
}

func (c *PersistentVolumeProvisionerController) GetCloudProvider() cloudprovider.Interface {
	return c.cloud
}

func (c *PersistentVolumeProvisionerController) GetMounter() mount.Interface {
	return nil
}

func (c *PersistentVolumeProvisionerController) GetWriter() io.Writer {
	return nil
}

func (c *PersistentVolumeProvisionerController) GetHostName() string {
	return ""
}

const (
	// these pair of constants are used by the provisioner.
	// The key is a kube namespaced key that denotes a volume requires provisioning.
	// The value is set only when provisioning is completed.  Any other value will tell the provisioner
	// that provisioning has not yet occurred.
	pvProvisioningRequiredAnnotationKey    = "volume.experimental.kubernetes.io/provisioning-required"
	pvProvisioningCompletedAnnotationValue = "volume.experimental.kubernetes.io/provisioning-completed"
)
