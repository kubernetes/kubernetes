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

// Package expand implements interfaces that attempt to resize a pvc
// by adding pvc to a volume resize map from which PVCs are picked and
// resized
package expand

import (
	"fmt"
	"net"
	"time"

	"github.com/golang/glog"

	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/runtime"
	coreinformers "k8s.io/client-go/informers/core/v1"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/client-go/kubernetes/scheme"
	v1core "k8s.io/client-go/kubernetes/typed/core/v1"
	corelisters "k8s.io/client-go/listers/core/v1"
	kcache "k8s.io/client-go/tools/cache"
	"k8s.io/client-go/tools/record"
	"k8s.io/kubernetes/pkg/cloudprovider"
	"k8s.io/kubernetes/pkg/controller"
	"k8s.io/kubernetes/pkg/controller/volume/expand/cache"
	"k8s.io/kubernetes/pkg/util/io"
	"k8s.io/kubernetes/pkg/util/mount"
	"k8s.io/kubernetes/pkg/volume"
	"k8s.io/kubernetes/pkg/volume/util/operationexecutor"
)

const (
	// How often resizing loop runs
	syncLoopPeriod time.Duration = 30 * time.Second
	// How often pvc populator runs
	populatorLoopPeriod time.Duration = 2 * time.Minute
)

// ExpandController expands the pvs
type ExpandController interface {
	Run(stopCh <-chan struct{})
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

	// cloud provider used by volume host
	cloud cloudprovider.Interface

	// volumePluginMgr used to initialize and fetch volume plugins
	volumePluginMgr volume.VolumePluginMgr

	// recorder is used to record events in the API server
	recorder record.EventRecorder

	// Volume resize map of volumes that needs resizing
	resizeMap cache.VolumeResizeMap

	// Worker goroutine to process resize requests from resizeMap
	syncResize SyncVolumeResize

	// Operation executor
	opExecutor operationexecutor.OperationExecutor

	// populator for periodically polling all PVCs
	pvcPopulator PVCPopulator
}

func NewExpandController(
	kubeClient clientset.Interface,
	pvcInformer coreinformers.PersistentVolumeClaimInformer,
	pvInformer coreinformers.PersistentVolumeInformer,
	cloud cloudprovider.Interface,
	plugins []volume.VolumePlugin) (ExpandController, error) {

	expc := &expandController{
		kubeClient: kubeClient,
		cloud:      cloud,
		pvcLister:  pvcInformer.Lister(),
		pvcsSynced: pvcInformer.Informer().HasSynced,
		pvLister:   pvInformer.Lister(),
		pvSynced:   pvInformer.Informer().HasSynced,
	}

	if err := expc.volumePluginMgr.InitPlugins(plugins, nil, expc); err != nil {
		return nil, fmt.Errorf("Could not initialize volume plugins for Expand Controller : %+v", err)
	}

	eventBroadcaster := record.NewBroadcaster()
	eventBroadcaster.StartLogging(glog.Infof)
	eventBroadcaster.StartRecordingToSink(&v1core.EventSinkImpl{Interface: v1core.New(kubeClient.CoreV1().RESTClient()).Events("")})
	recorder := eventBroadcaster.NewRecorder(scheme.Scheme, v1.EventSource{Component: "volume_expand"})

	expc.opExecutor = operationexecutor.NewOperationExecutor(operationexecutor.NewOperationGenerator(
		kubeClient,
		&expc.volumePluginMgr,
		recorder,
		false))

	expc.resizeMap = cache.NewVolumeResizeMap(expc.kubeClient)

	pvcInformer.Informer().AddEventHandler(kcache.ResourceEventHandlerFuncs{
		UpdateFunc: expc.pvcUpdate,
		DeleteFunc: expc.deletePVC,
	})

	expc.syncResize = NewSyncVolumeResize(syncLoopPeriod, expc.opExecutor, expc.resizeMap, kubeClient)
	expc.pvcPopulator = NewPVCPopulator(
		populatorLoopPeriod,
		expc.resizeMap,
		expc.pvcLister,
		expc.pvLister,
		kubeClient)
	return expc, nil
}

func (expc *expandController) Run(stopCh <-chan struct{}) {
	defer runtime.HandleCrash()
	glog.Infof("Starting expand controller")
	defer glog.Infof("Shutting down expand controller")

	if !controller.WaitForCacheSync("expand", stopCh, expc.pvcsSynced, expc.pvSynced) {
		return
	}

	// Run volume sync work goroutine
	go expc.syncResize.Run(stopCh)
	// Start the pvc populator loop
	go expc.pvcPopulator.Run(stopCh)
	<-stopCh
}

func (expc *expandController) deletePVC(obj interface{}) {
	pvc, ok := obj.(*v1.PersistentVolumeClaim)

	if pvc == nil || !ok {
		return
	}

	expc.resizeMap.DeletePVC(pvc)
}

func (expc *expandController) pvcUpdate(oldObj, newObj interface{}) {
	oldPvc, ok := oldObj.(*v1.PersistentVolumeClaim)

	if oldPvc == nil || !ok {
		return
	}

	newPVC, ok := newObj.(*v1.PersistentVolumeClaim)

	if newPVC == nil || !ok {
		return
	}
	pv, err := getPersistentVolume(newPVC, expc.pvLister)
	if err != nil {
		glog.V(5).Infof("Error getting Persistent Volume for pvc %q : %v", newPVC.UID, err)
		return
	}
	expc.resizeMap.AddPVCUpdate(newPVC, pv)
}

func getPersistentVolume(pvc *v1.PersistentVolumeClaim, pvLister corelisters.PersistentVolumeLister) (*v1.PersistentVolume, error) {
	volumeName := pvc.Spec.VolumeName
	pv, err := pvLister.Get(volumeName)

	if err != nil {
		return nil, fmt.Errorf("failed to find PV %q in PV informer cache with error : %v", volumeName, err)
	}

	return pv.DeepCopy(), nil
}

// Implementing VolumeHost interface
func (expc *expandController) GetPluginDir(pluginName string) string {
	return ""
}

func (expc *expandController) GetPodVolumeDir(podUID types.UID, pluginName string, volumeName string) string {
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

func (expc *expandController) GetExec(pluginName string) mount.Exec {
	return mount.NewOsExec()
}

func (expc *expandController) GetWriter() io.Writer {
	return nil
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

func (expc *expandController) GetNodeLabels() (map[string]string, error) {
	return nil, fmt.Errorf("GetNodeLabels unsupported in expandController")
}
