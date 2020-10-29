/*
Copyright 2017 The Kubernetes Authors.

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

package expand

import (
	"context"
	"fmt"
	"net"
	"time"

	"k8s.io/klog/v2"
	utilexec "k8s.io/utils/exec"
	"k8s.io/utils/mount"

	authenticationv1 "k8s.io/api/authentication/v1"
	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/util/wait"
	coreinformers "k8s.io/client-go/informers/core/v1"
	storageclassinformer "k8s.io/client-go/informers/storage/v1"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/client-go/kubernetes/scheme"
	v1core "k8s.io/client-go/kubernetes/typed/core/v1"
	corelisters "k8s.io/client-go/listers/core/v1"
	storagelisters "k8s.io/client-go/listers/storage/v1"
	"k8s.io/client-go/tools/cache"
	kcache "k8s.io/client-go/tools/cache"
	"k8s.io/client-go/tools/record"
	"k8s.io/client-go/util/workqueue"
	cloudprovider "k8s.io/cloud-provider"
	v1helper "k8s.io/kubernetes/pkg/apis/core/v1/helper"
	"k8s.io/kubernetes/pkg/controller/volume/events"
	"k8s.io/kubernetes/pkg/volume"
	"k8s.io/kubernetes/pkg/volume/csimigration"
	"k8s.io/kubernetes/pkg/volume/util"
	"k8s.io/kubernetes/pkg/volume/util/operationexecutor"
	"k8s.io/kubernetes/pkg/volume/util/subpath"
	"k8s.io/kubernetes/pkg/volume/util/volumepathhandler"
)

const (
	// number of default volume expansion workers
	defaultWorkerCount = 10
)

// ExpandController expands the pvs
type ExpandController interface {
	Run(stopCh <-chan struct{})
}

// CSINameTranslator can get the CSI Driver name based on the in-tree plugin name
type CSINameTranslator interface {
	GetCSINameFromInTreeName(pluginName string) (string, error)
}

type expandController struct {
	// kubeClient is the kube API client used by volumehost to communicate with
	// the API server.
	kubeClient clientset.Interface

	// pvcLister is the shared PVC lister used to fetch and store PVC
	// objects from the API server. It is shared with other controllers and
	// therefore the PVC objects in its store should be treated as immutable.
	pvcLister  corelisters.PersistentVolumeClaimLister
	pvcsSynced kcache.InformerSynced

	pvLister corelisters.PersistentVolumeLister
	pvSynced kcache.InformerSynced

	// storageClass lister for fetching provisioner name
	classLister       storagelisters.StorageClassLister
	classListerSynced cache.InformerSynced

	// cloud provider used by volume host
	cloud cloudprovider.Interface

	// volumePluginMgr used to initialize and fetch volume plugins
	volumePluginMgr volume.VolumePluginMgr

	// recorder is used to record events in the API server
	recorder record.EventRecorder

	operationGenerator operationexecutor.OperationGenerator

	queue workqueue.RateLimitingInterface

	translator CSINameTranslator

	csiMigratedPluginManager csimigration.PluginManager
}

// NewExpandController expands the pvs
func NewExpandController(
	kubeClient clientset.Interface,
	pvcInformer coreinformers.PersistentVolumeClaimInformer,
	pvInformer coreinformers.PersistentVolumeInformer,
	scInformer storageclassinformer.StorageClassInformer,
	cloud cloudprovider.Interface,
	plugins []volume.VolumePlugin,
	translator CSINameTranslator,
	csiMigratedPluginManager csimigration.PluginManager) (ExpandController, error) {

	expc := &expandController{
		kubeClient:               kubeClient,
		cloud:                    cloud,
		pvcLister:                pvcInformer.Lister(),
		pvcsSynced:               pvcInformer.Informer().HasSynced,
		pvLister:                 pvInformer.Lister(),
		pvSynced:                 pvInformer.Informer().HasSynced,
		classLister:              scInformer.Lister(),
		classListerSynced:        scInformer.Informer().HasSynced,
		queue:                    workqueue.NewNamedRateLimitingQueue(workqueue.DefaultControllerRateLimiter(), "volume_expand"),
		translator:               translator,
		csiMigratedPluginManager: csiMigratedPluginManager,
	}

	if err := expc.volumePluginMgr.InitPlugins(plugins, nil, expc); err != nil {
		return nil, fmt.Errorf("could not initialize volume plugins for Expand Controller : %+v", err)
	}

	eventBroadcaster := record.NewBroadcaster()
	eventBroadcaster.StartStructuredLogging(0)
	eventBroadcaster.StartRecordingToSink(&v1core.EventSinkImpl{Interface: kubeClient.CoreV1().Events("")})
	expc.recorder = eventBroadcaster.NewRecorder(scheme.Scheme, v1.EventSource{Component: "volume_expand"})
	blkutil := volumepathhandler.NewBlockVolumePathHandler()

	expc.operationGenerator = operationexecutor.NewOperationGenerator(
		kubeClient,
		&expc.volumePluginMgr,
		expc.recorder,
		false,
		blkutil)

	pvcInformer.Informer().AddEventHandler(kcache.ResourceEventHandlerFuncs{
		AddFunc: expc.enqueuePVC,
		UpdateFunc: func(old, new interface{}) {
			oldPVC, ok := old.(*v1.PersistentVolumeClaim)
			if !ok {
				return
			}

			oldSize := oldPVC.Spec.Resources.Requests[v1.ResourceStorage]
			newPVC, ok := new.(*v1.PersistentVolumeClaim)
			if !ok {
				return
			}
			newSize := newPVC.Spec.Resources.Requests[v1.ResourceStorage]
			if newSize.Cmp(oldSize) > 0 {
				expc.enqueuePVC(new)
			}
		},
		DeleteFunc: expc.enqueuePVC,
	})

	return expc, nil
}

func (expc *expandController) enqueuePVC(obj interface{}) {
	pvc, ok := obj.(*v1.PersistentVolumeClaim)
	if !ok {
		return
	}

	size := pvc.Spec.Resources.Requests[v1.ResourceStorage]
	statusSize := pvc.Status.Capacity[v1.ResourceStorage]

	if pvc.Status.Phase == v1.ClaimBound && size.Cmp(statusSize) > 0 {
		key, err := kcache.DeletionHandlingMetaNamespaceKeyFunc(pvc)
		if err != nil {
			runtime.HandleError(fmt.Errorf("couldn't get key for object %#v: %v", pvc, err))
			return
		}
		expc.queue.Add(key)
	}
}

func (expc *expandController) processNextWorkItem() bool {
	key, shutdown := expc.queue.Get()
	if shutdown {
		return false
	}
	defer expc.queue.Done(key)

	err := expc.syncHandler(key.(string))
	if err == nil {
		expc.queue.Forget(key)
		return true
	}

	runtime.HandleError(fmt.Errorf("%v failed with : %v", key, err))
	expc.queue.AddRateLimited(key)

	return true
}

// syncHandler performs actual expansion of volume. If an error is returned
// from this function - PVC will be requeued for resizing.
func (expc *expandController) syncHandler(key string) error {
	namespace, name, err := kcache.SplitMetaNamespaceKey(key)
	if err != nil {
		return err
	}
	pvc, err := expc.pvcLister.PersistentVolumeClaims(namespace).Get(name)
	if errors.IsNotFound(err) {
		return nil
	}
	if err != nil {
		klog.V(5).Infof("Error getting PVC %q (uid: %q) from informer : %v", util.GetPersistentVolumeClaimQualifiedName(pvc), pvc.UID, err)
		return err
	}

	pv, err := expc.getPersistentVolume(pvc)
	if err != nil {
		klog.V(5).Infof("Error getting Persistent Volume for PVC %q (uid: %q) from informer : %v", util.GetPersistentVolumeClaimQualifiedName(pvc), pvc.UID, err)
		return err
	}

	if pv.Spec.ClaimRef == nil || pvc.Namespace != pv.Spec.ClaimRef.Namespace || pvc.UID != pv.Spec.ClaimRef.UID {
		err := fmt.Errorf("persistent Volume is not bound to PVC being updated : %s", util.ClaimToClaimKey(pvc))
		klog.V(4).Infof("%v", err)
		return err
	}

	claimClass := v1helper.GetPersistentVolumeClaimClass(pvc)
	if claimClass == "" {
		klog.V(4).Infof("volume expansion is disabled for PVC without StorageClasses: %s", util.ClaimToClaimKey(pvc))
		return nil
	}

	class, err := expc.classLister.Get(claimClass)
	if err != nil {
		klog.V(4).Infof("failed to expand PVC: %s with error: %v", util.ClaimToClaimKey(pvc), err)
		return nil
	}

	volumeResizerName := class.Provisioner
	volumeSpec := volume.NewSpecFromPersistentVolume(pv, false)
	migratable, err := expc.csiMigratedPluginManager.IsMigratable(volumeSpec)
	if err != nil {
		klog.V(4).Infof("failed to check CSI migration status for PVC: %s with error: %v", util.ClaimToClaimKey(pvc), err)
		return nil
	}
	// handle CSI migration scenarios before invoking FindExpandablePluginBySpec for in-tree
	if migratable {
		msg := fmt.Sprintf("CSI migration enabled for %s; waiting for external resizer to expand the pvc", volumeResizerName)
		expc.recorder.Event(pvc, v1.EventTypeNormal, events.ExternalExpanding, msg)
		csiResizerName, err := expc.translator.GetCSINameFromInTreeName(class.Provisioner)
		if err != nil {
			errorMsg := fmt.Sprintf("error getting CSI driver name for pvc %s, with error %v", util.ClaimToClaimKey(pvc), err)
			expc.recorder.Event(pvc, v1.EventTypeWarning, events.ExternalExpanding, errorMsg)
			return fmt.Errorf(errorMsg)
		}

		pvc, err := util.SetClaimResizer(pvc, csiResizerName, expc.kubeClient)
		if err != nil {
			errorMsg := fmt.Sprintf("error setting resizer annotation to pvc %s, with error %v", util.ClaimToClaimKey(pvc), err)
			expc.recorder.Event(pvc, v1.EventTypeWarning, events.ExternalExpanding, errorMsg)
			return fmt.Errorf(errorMsg)
		}
		return nil
	}

	volumePlugin, err := expc.volumePluginMgr.FindExpandablePluginBySpec(volumeSpec)
	if err != nil || volumePlugin == nil {
		msg := fmt.Errorf("didn't find a plugin capable of expanding the volume; " +
			"waiting for an external controller to process this PVC")
		eventType := v1.EventTypeNormal
		if err != nil {
			eventType = v1.EventTypeWarning
		}
		expc.recorder.Event(pvc, eventType, events.ExternalExpanding, fmt.Sprintf("Ignoring the PVC: %v.", msg))
		klog.Infof("Ignoring the PVC %q (uid: %q) : %v.", util.GetPersistentVolumeClaimQualifiedName(pvc), pvc.UID, msg)
		// If we are expecting that an external plugin will handle resizing this volume then
		// is no point in requeuing this PVC.
		return nil
	}

	return expc.expand(pvc, pv, volumeResizerName)
}

func (expc *expandController) expand(pvc *v1.PersistentVolumeClaim, pv *v1.PersistentVolume, resizerName string) error {
	pvc, err := util.MarkResizeInProgressWithResizer(pvc, resizerName, expc.kubeClient)
	if err != nil {
		klog.V(5).Infof("Error setting PVC %s in progress with error : %v", util.GetPersistentVolumeClaimQualifiedName(pvc), err)
		return err
	}

	generatedOperations, err := expc.operationGenerator.GenerateExpandVolumeFunc(pvc, pv)
	if err != nil {
		klog.Errorf("Error starting ExpandVolume for pvc %s with %v", util.GetPersistentVolumeClaimQualifiedName(pvc), err)
		return err
	}
	klog.V(5).Infof("Starting ExpandVolume for volume %s", util.GetPersistentVolumeClaimQualifiedName(pvc))
	_, detailedErr := generatedOperations.Run()

	return detailedErr
}

// TODO make concurrency configurable (workers/threadiness argument). previously, nestedpendingoperations spawned unlimited goroutines
func (expc *expandController) Run(stopCh <-chan struct{}) {
	defer runtime.HandleCrash()
	defer expc.queue.ShutDown()

	klog.Infof("Starting expand controller")
	defer klog.Infof("Shutting down expand controller")

	if !cache.WaitForNamedCacheSync("expand", stopCh, expc.pvcsSynced, expc.pvSynced, expc.classListerSynced) {
		return
	}

	for i := 0; i < defaultWorkerCount; i++ {
		go wait.Until(expc.runWorker, time.Second, stopCh)
	}

	<-stopCh
}

func (expc *expandController) runWorker() {
	for expc.processNextWorkItem() {
	}
}

func (expc *expandController) getPersistentVolume(pvc *v1.PersistentVolumeClaim) (*v1.PersistentVolume, error) {
	volumeName := pvc.Spec.VolumeName
	pv, err := expc.kubeClient.CoreV1().PersistentVolumes().Get(context.TODO(), volumeName, metav1.GetOptions{})

	if err != nil {
		return nil, fmt.Errorf("failed to get PV %q: %v", volumeName, err)
	}

	return pv.DeepCopy(), nil
}

// Implementing VolumeHost interface
func (expc *expandController) GetPluginDir(pluginName string) string {
	return ""
}

func (expc *expandController) GetVolumeDevicePluginDir(pluginName string) string {
	return ""
}

func (expc *expandController) GetPodsDir() string {
	return ""
}

func (expc *expandController) GetPodVolumeDir(podUID types.UID, pluginName string, volumeName string) string {
	return ""
}

func (expc *expandController) GetPodVolumeDeviceDir(podUID types.UID, pluginName string) string {
	return ""
}

func (expc *expandController) GetPodPluginDir(podUID types.UID, pluginName string) string {
	return ""
}

func (expc *expandController) GetKubeClient() clientset.Interface {
	return expc.kubeClient
}

func (expc *expandController) NewWrapperMounter(volName string, spec volume.Spec, pod *v1.Pod, opts volume.VolumeOptions) (volume.Mounter, error) {
	return nil, fmt.Errorf("NewWrapperMounter not supported by expand controller's VolumeHost implementation")
}

func (expc *expandController) NewWrapperUnmounter(volName string, spec volume.Spec, podUID types.UID) (volume.Unmounter, error) {
	return nil, fmt.Errorf("NewWrapperUnmounter not supported by expand controller's VolumeHost implementation")
}

func (expc *expandController) GetCloudProvider() cloudprovider.Interface {
	return expc.cloud
}

func (expc *expandController) GetMounter(pluginName string) mount.Interface {
	return nil
}

func (expc *expandController) GetExec(pluginName string) utilexec.Interface {
	return utilexec.New()
}

func (expc *expandController) GetHostName() string {
	return ""
}

func (expc *expandController) GetHostIP() (net.IP, error) {
	return nil, fmt.Errorf("GetHostIP not supported by expand controller's VolumeHost implementation")
}

func (expc *expandController) GetNodeAllocatable() (v1.ResourceList, error) {
	return v1.ResourceList{}, nil
}

func (expc *expandController) GetSecretFunc() func(namespace, name string) (*v1.Secret, error) {
	return func(_, _ string) (*v1.Secret, error) {
		return nil, fmt.Errorf("GetSecret unsupported in expandController")
	}
}

func (expc *expandController) GetConfigMapFunc() func(namespace, name string) (*v1.ConfigMap, error) {
	return func(_, _ string) (*v1.ConfigMap, error) {
		return nil, fmt.Errorf("GetConfigMap unsupported in expandController")
	}
}

func (expc *expandController) GetServiceAccountTokenFunc() func(_, _ string, _ *authenticationv1.TokenRequest) (*authenticationv1.TokenRequest, error) {
	return func(_, _ string, _ *authenticationv1.TokenRequest) (*authenticationv1.TokenRequest, error) {
		return nil, fmt.Errorf("GetServiceAccountToken unsupported in expandController")
	}
}

func (expc *expandController) DeleteServiceAccountTokenFunc() func(types.UID) {
	return func(types.UID) {
		klog.Errorf("DeleteServiceAccountToken unsupported in expandController")
	}
}

func (expc *expandController) GetNodeLabels() (map[string]string, error) {
	return nil, fmt.Errorf("GetNodeLabels unsupported in expandController")
}

func (expc *expandController) GetNodeName() types.NodeName {
	return ""
}

func (expc *expandController) GetEventRecorder() record.EventRecorder {
	return expc.recorder
}

func (expc *expandController) GetSubpather() subpath.Interface {
	// not needed for expand controller
	return nil
}
