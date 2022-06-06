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

// Package attachdetach implements a controller to manage volume attach and detach
// operations.
package attachdetach

import (
	"fmt"
	"net"
	"time"

	"k8s.io/klog/v2"
	"k8s.io/mount-utils"
	utilexec "k8s.io/utils/exec"

	authenticationv1 "k8s.io/api/authentication/v1"
	v1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/util/wait"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	coreinformers "k8s.io/client-go/informers/core/v1"
	storageinformersv1 "k8s.io/client-go/informers/storage/v1"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/client-go/kubernetes/scheme"
	v1core "k8s.io/client-go/kubernetes/typed/core/v1"
	corelisters "k8s.io/client-go/listers/core/v1"
	storagelistersv1 "k8s.io/client-go/listers/storage/v1"
	kcache "k8s.io/client-go/tools/cache"
	"k8s.io/client-go/tools/record"
	"k8s.io/client-go/util/workqueue"
	cloudprovider "k8s.io/cloud-provider"
	csitrans "k8s.io/csi-translation-lib"
	"k8s.io/kubernetes/pkg/controller/volume/attachdetach/cache"
	"k8s.io/kubernetes/pkg/controller/volume/attachdetach/metrics"
	"k8s.io/kubernetes/pkg/controller/volume/attachdetach/populator"
	"k8s.io/kubernetes/pkg/controller/volume/attachdetach/reconciler"
	"k8s.io/kubernetes/pkg/controller/volume/attachdetach/statusupdater"
	"k8s.io/kubernetes/pkg/controller/volume/attachdetach/util"
	"k8s.io/kubernetes/pkg/controller/volume/common"
	proxyutil "k8s.io/kubernetes/pkg/proxy/util"
	"k8s.io/kubernetes/pkg/volume"
	"k8s.io/kubernetes/pkg/volume/csi"
	"k8s.io/kubernetes/pkg/volume/csimigration"
	volumeutil "k8s.io/kubernetes/pkg/volume/util"
	"k8s.io/kubernetes/pkg/volume/util/operationexecutor"
	"k8s.io/kubernetes/pkg/volume/util/subpath"
	"k8s.io/kubernetes/pkg/volume/util/volumepathhandler"
)

// TimerConfig contains configuration of internal attach/detach timers and
// should be used only to speed up tests. DefaultTimerConfig is the suggested
// timer configuration for production.
type TimerConfig struct {
	// ReconcilerLoopPeriod is the amount of time the reconciler loop waits
	// between successive executions
	ReconcilerLoopPeriod time.Duration

	// ReconcilerMaxWaitForUnmountDuration is the maximum amount of time the
	// attach detach controller will wait for a volume to be safely unmounted
	// from its node. Once this time has expired, the controller will assume the
	// node or kubelet are unresponsive and will detach the volume anyway.
	ReconcilerMaxWaitForUnmountDuration time.Duration

	// DesiredStateOfWorldPopulatorLoopSleepPeriod is the amount of time the
	// DesiredStateOfWorldPopulator loop waits between successive executions
	DesiredStateOfWorldPopulatorLoopSleepPeriod time.Duration

	// DesiredStateOfWorldPopulatorListPodsRetryDuration is the amount of
	// time the DesiredStateOfWorldPopulator loop waits between list pods
	// calls.
	DesiredStateOfWorldPopulatorListPodsRetryDuration time.Duration
}

// DefaultTimerConfig is the default configuration of Attach/Detach controller
// timers.
var DefaultTimerConfig TimerConfig = TimerConfig{
	ReconcilerLoopPeriod:                              100 * time.Millisecond,
	ReconcilerMaxWaitForUnmountDuration:               6 * time.Minute,
	DesiredStateOfWorldPopulatorLoopSleepPeriod:       1 * time.Minute,
	DesiredStateOfWorldPopulatorListPodsRetryDuration: 3 * time.Minute,
}

// AttachDetachController defines the operations supported by this controller.
type AttachDetachController interface {
	Run(stopCh <-chan struct{})
	GetDesiredStateOfWorld() cache.DesiredStateOfWorld
}

// NewAttachDetachController returns a new instance of AttachDetachController.
func NewAttachDetachController(
	kubeClient clientset.Interface,
	podInformer coreinformers.PodInformer,
	nodeInformer coreinformers.NodeInformer,
	pvcInformer coreinformers.PersistentVolumeClaimInformer,
	pvInformer coreinformers.PersistentVolumeInformer,
	csiNodeInformer storageinformersv1.CSINodeInformer,
	csiDriverInformer storageinformersv1.CSIDriverInformer,
	volumeAttachmentInformer storageinformersv1.VolumeAttachmentInformer,
	cloud cloudprovider.Interface,
	plugins []volume.VolumePlugin,
	prober volume.DynamicPluginProber,
	disableReconciliationSync bool,
	reconcilerSyncDuration time.Duration,
	timerConfig TimerConfig,
	filteredDialOptions *proxyutil.FilteredDialOptions) (AttachDetachController, error) {

	adc := &attachDetachController{
		kubeClient:          kubeClient,
		pvcLister:           pvcInformer.Lister(),
		pvcsSynced:          pvcInformer.Informer().HasSynced,
		pvLister:            pvInformer.Lister(),
		pvsSynced:           pvInformer.Informer().HasSynced,
		podLister:           podInformer.Lister(),
		podsSynced:          podInformer.Informer().HasSynced,
		podIndexer:          podInformer.Informer().GetIndexer(),
		nodeLister:          nodeInformer.Lister(),
		nodesSynced:         nodeInformer.Informer().HasSynced,
		cloud:               cloud,
		pvcQueue:            workqueue.NewNamedRateLimitingQueue(workqueue.DefaultControllerRateLimiter(), "pvcs"),
		filteredDialOptions: filteredDialOptions,
	}

	adc.csiNodeLister = csiNodeInformer.Lister()
	adc.csiNodeSynced = csiNodeInformer.Informer().HasSynced

	adc.csiDriverLister = csiDriverInformer.Lister()
	adc.csiDriversSynced = csiDriverInformer.Informer().HasSynced

	adc.volumeAttachmentLister = volumeAttachmentInformer.Lister()
	adc.volumeAttachmentSynced = volumeAttachmentInformer.Informer().HasSynced

	if err := adc.volumePluginMgr.InitPlugins(plugins, prober, adc); err != nil {
		return nil, fmt.Errorf("could not initialize volume plugins for Attach/Detach Controller: %w", err)
	}

	eventBroadcaster := record.NewBroadcaster()
	eventBroadcaster.StartStructuredLogging(0)
	eventBroadcaster.StartRecordingToSink(&v1core.EventSinkImpl{Interface: kubeClient.CoreV1().Events("")})
	recorder := eventBroadcaster.NewRecorder(scheme.Scheme, v1.EventSource{Component: "attachdetach-controller"})
	blkutil := volumepathhandler.NewBlockVolumePathHandler()

	adc.desiredStateOfWorld = cache.NewDesiredStateOfWorld(&adc.volumePluginMgr)
	adc.actualStateOfWorld = cache.NewActualStateOfWorld(&adc.volumePluginMgr)
	adc.attacherDetacher =
		operationexecutor.NewOperationExecutor(operationexecutor.NewOperationGenerator(
			kubeClient,
			&adc.volumePluginMgr,
			recorder,
			blkutil))
	adc.nodeStatusUpdater = statusupdater.NewNodeStatusUpdater(
		kubeClient, nodeInformer.Lister(), adc.actualStateOfWorld)

	// Default these to values in options
	adc.reconciler = reconciler.NewReconciler(
		timerConfig.ReconcilerLoopPeriod,
		timerConfig.ReconcilerMaxWaitForUnmountDuration,
		reconcilerSyncDuration,
		disableReconciliationSync,
		adc.desiredStateOfWorld,
		adc.actualStateOfWorld,
		adc.attacherDetacher,
		adc.nodeStatusUpdater,
		adc.nodeLister,
		recorder)

	csiTranslator := csitrans.New()
	adc.intreeToCSITranslator = csiTranslator
	adc.csiMigratedPluginManager = csimigration.NewPluginManager(csiTranslator, utilfeature.DefaultFeatureGate)

	adc.desiredStateOfWorldPopulator = populator.NewDesiredStateOfWorldPopulator(
		timerConfig.DesiredStateOfWorldPopulatorLoopSleepPeriod,
		timerConfig.DesiredStateOfWorldPopulatorListPodsRetryDuration,
		podInformer.Lister(),
		adc.desiredStateOfWorld,
		&adc.volumePluginMgr,
		pvcInformer.Lister(),
		pvInformer.Lister(),
		adc.csiMigratedPluginManager,
		adc.intreeToCSITranslator)

	podInformer.Informer().AddEventHandler(kcache.ResourceEventHandlerFuncs{
		AddFunc:    adc.podAdd,
		UpdateFunc: adc.podUpdate,
		DeleteFunc: adc.podDelete,
	})

	// This custom indexer will index pods by its PVC keys. Then we don't need
	// to iterate all pods every time to find pods which reference given PVC.
	if err := common.AddPodPVCIndexerIfNotPresent(adc.podIndexer); err != nil {
		return nil, fmt.Errorf("could not initialize attach detach controller: %w", err)
	}

	nodeInformer.Informer().AddEventHandler(kcache.ResourceEventHandlerFuncs{
		AddFunc:    adc.nodeAdd,
		UpdateFunc: adc.nodeUpdate,
		DeleteFunc: adc.nodeDelete,
	})

	pvcInformer.Informer().AddEventHandler(kcache.ResourceEventHandlerFuncs{
		AddFunc: func(obj interface{}) {
			adc.enqueuePVC(obj)
		},
		UpdateFunc: func(old, new interface{}) {
			adc.enqueuePVC(new)
		},
	})

	return adc, nil
}

type attachDetachController struct {
	// kubeClient is the kube API client used by volumehost to communicate with
	// the API server.
	kubeClient clientset.Interface

	// pvcLister is the shared PVC lister used to fetch and store PVC
	// objects from the API server. It is shared with other controllers and
	// therefore the PVC objects in its store should be treated as immutable.
	pvcLister  corelisters.PersistentVolumeClaimLister
	pvcsSynced kcache.InformerSynced

	// pvLister is the shared PV lister used to fetch and store PV objects
	// from the API server. It is shared with other controllers and therefore
	// the PV objects in its store should be treated as immutable.
	pvLister  corelisters.PersistentVolumeLister
	pvsSynced kcache.InformerSynced

	podLister  corelisters.PodLister
	podsSynced kcache.InformerSynced
	podIndexer kcache.Indexer

	nodeLister  corelisters.NodeLister
	nodesSynced kcache.InformerSynced

	csiNodeLister storagelistersv1.CSINodeLister
	csiNodeSynced kcache.InformerSynced

	// csiDriverLister is the shared CSIDriver lister used to fetch and store
	// CSIDriver objects from the API server. It is shared with other controllers
	// and therefore the CSIDriver objects in its store should be treated as immutable.
	csiDriverLister  storagelistersv1.CSIDriverLister
	csiDriversSynced kcache.InformerSynced

	// volumeAttachmentLister is the shared volumeAttachment lister used to fetch and store
	// VolumeAttachment objects from the API server. It is shared with other controllers
	// and therefore the VolumeAttachment objects in its store should be treated as immutable.
	volumeAttachmentLister storagelistersv1.VolumeAttachmentLister
	volumeAttachmentSynced kcache.InformerSynced

	// cloud provider used by volume host
	cloud cloudprovider.Interface

	// volumePluginMgr used to initialize and fetch volume plugins
	volumePluginMgr volume.VolumePluginMgr

	// desiredStateOfWorld is a data structure containing the desired state of
	// the world according to this controller: i.e. what nodes the controller
	// is managing, what volumes it wants be attached to these nodes, and which
	// pods are scheduled to those nodes referencing the volumes.
	// The data structure is populated by the controller using a stream of node
	// and pod API server objects fetched by the informers.
	desiredStateOfWorld cache.DesiredStateOfWorld

	// actualStateOfWorld is a data structure containing the actual state of
	// the world according to this controller: i.e. which volumes are attached
	// to which nodes.
	// The data structure is populated upon successful completion of attach and
	// detach actions triggered by the controller and a periodic sync with
	// storage providers for the "true" state of the world.
	actualStateOfWorld cache.ActualStateOfWorld

	// attacherDetacher is used to start asynchronous attach and operations
	attacherDetacher operationexecutor.OperationExecutor

	// reconciler is used to run an asynchronous periodic loop to reconcile the
	// desiredStateOfWorld with the actualStateOfWorld by triggering attach
	// detach operations using the attacherDetacher.
	reconciler reconciler.Reconciler

	// nodeStatusUpdater is used to update node status with the list of attached
	// volumes
	nodeStatusUpdater statusupdater.NodeStatusUpdater

	// desiredStateOfWorldPopulator runs an asynchronous periodic loop to
	// populate the current pods using podInformer.
	desiredStateOfWorldPopulator populator.DesiredStateOfWorldPopulator

	// recorder is used to record events in the API server
	recorder record.EventRecorder

	// pvcQueue is used to queue pvc objects
	pvcQueue workqueue.RateLimitingInterface

	// csiMigratedPluginManager detects in-tree plugins that have been migrated to CSI
	csiMigratedPluginManager csimigration.PluginManager

	// intreeToCSITranslator translates from in-tree volume specs to CSI
	intreeToCSITranslator csimigration.InTreeToCSITranslator

	// filteredDialOptions configures any dialing done by the controller.
	filteredDialOptions *proxyutil.FilteredDialOptions
}

func (adc *attachDetachController) Run(stopCh <-chan struct{}) {
	defer runtime.HandleCrash()
	defer adc.pvcQueue.ShutDown()

	klog.Infof("Starting attach detach controller")
	defer klog.Infof("Shutting down attach detach controller")

	synced := []kcache.InformerSynced{adc.podsSynced, adc.nodesSynced, adc.pvcsSynced, adc.pvsSynced}
	if adc.csiNodeSynced != nil {
		synced = append(synced, adc.csiNodeSynced)
	}
	if adc.csiDriversSynced != nil {
		synced = append(synced, adc.csiDriversSynced)
	}
	if adc.volumeAttachmentSynced != nil {
		synced = append(synced, adc.volumeAttachmentSynced)
	}

	if !kcache.WaitForNamedCacheSync("attach detach", stopCh, synced...) {
		return
	}

	err := adc.populateActualStateOfWorld()
	if err != nil {
		klog.Errorf("Error populating the actual state of world: %v", err)
	}
	err = adc.populateDesiredStateOfWorld()
	if err != nil {
		klog.Errorf("Error populating the desired state of world: %v", err)
	}
	go adc.reconciler.Run(stopCh)
	go adc.desiredStateOfWorldPopulator.Run(stopCh)
	go wait.Until(adc.pvcWorker, time.Second, stopCh)
	metrics.Register(adc.pvcLister,
		adc.pvLister,
		adc.podLister,
		adc.actualStateOfWorld,
		adc.desiredStateOfWorld,
		&adc.volumePluginMgr,
		adc.csiMigratedPluginManager,
		adc.intreeToCSITranslator)

	<-stopCh
}

func (adc *attachDetachController) populateActualStateOfWorld() error {
	klog.V(5).Infof("Populating ActualStateOfworld")
	nodes, err := adc.nodeLister.List(labels.Everything())
	if err != nil {
		return err
	}

	for _, node := range nodes {
		nodeName := types.NodeName(node.Name)
		for _, attachedVolume := range node.Status.VolumesAttached {
			uniqueName := attachedVolume.Name
			// The nil VolumeSpec is safe only in the case the volume is not in use by any pod.
			// In such a case it should be detached in the first reconciliation cycle and the
			// volume spec is not needed to detach a volume. If the volume is used by a pod, it
			// its spec can be: this would happen during in the populateDesiredStateOfWorld which
			// scans the pods and updates their volumes in the ActualStateOfWorld too.
			err = adc.actualStateOfWorld.MarkVolumeAsAttached(uniqueName, nil /* VolumeSpec */, nodeName, attachedVolume.DevicePath)
			if err != nil {
				klog.Errorf("Failed to mark the volume as attached: %v", err)
				continue
			}
			adc.processVolumesInUse(nodeName, node.Status.VolumesInUse)
			adc.addNodeToDswp(node, types.NodeName(node.Name))
		}
	}
	err = adc.processVolumeAttachments()
	if err != nil {
		klog.Errorf("Failed to process volume attachments: %v", err)
	}
	return err
}

func (adc *attachDetachController) getNodeVolumeDevicePath(
	volumeName v1.UniqueVolumeName, nodeName types.NodeName) (string, error) {
	var devicePath string
	var found bool
	node, err := adc.nodeLister.Get(string(nodeName))
	if err != nil {
		return devicePath, err
	}
	for _, attachedVolume := range node.Status.VolumesAttached {
		if volumeName == attachedVolume.Name {
			devicePath = attachedVolume.DevicePath
			found = true
			break
		}
	}
	if !found {
		err = fmt.Errorf("Volume %s not found on node %s", volumeName, nodeName)
	}

	return devicePath, err
}

func (adc *attachDetachController) populateDesiredStateOfWorld() error {
	klog.V(5).Infof("Populating DesiredStateOfworld")

	pods, err := adc.podLister.List(labels.Everything())
	if err != nil {
		return err
	}
	for _, pod := range pods {
		podToAdd := pod
		adc.podAdd(podToAdd)
		for _, podVolume := range podToAdd.Spec.Volumes {
			nodeName := types.NodeName(podToAdd.Spec.NodeName)
			// The volume specs present in the ActualStateOfWorld are nil, let's replace those
			// with the correct ones found on pods. The present in the ASW with no corresponding
			// pod will be detached and the spec is irrelevant.
			volumeSpec, err := util.CreateVolumeSpec(podVolume, podToAdd, nodeName, &adc.volumePluginMgr, adc.pvcLister, adc.pvLister, adc.csiMigratedPluginManager, adc.intreeToCSITranslator)
			if err != nil {
				klog.Errorf(
					"Error creating spec for volume %q, pod %q/%q: %v",
					podVolume.Name,
					podToAdd.Namespace,
					podToAdd.Name,
					err)
				continue
			}
			plugin, err := adc.volumePluginMgr.FindAttachablePluginBySpec(volumeSpec)
			if err != nil || plugin == nil {
				klog.V(10).Infof(
					"Skipping volume %q for pod %q/%q: it does not implement attacher interface. err=%v",
					podVolume.Name,
					podToAdd.Namespace,
					podToAdd.Name,
					err)
				continue
			}
			volumeName, err := volumeutil.GetUniqueVolumeNameFromSpec(plugin, volumeSpec)
			if err != nil {
				klog.Errorf(
					"Failed to find unique name for volume %q, pod %q/%q: %v",
					podVolume.Name,
					podToAdd.Namespace,
					podToAdd.Name,
					err)
				continue
			}
			attachState := adc.actualStateOfWorld.GetAttachState(volumeName, nodeName)
			if attachState == cache.AttachStateAttached {
				klog.V(10).Infof("Volume %q is attached to node %q. Marking as attached in ActualStateOfWorld",
					volumeName,
					nodeName,
				)
				devicePath, err := adc.getNodeVolumeDevicePath(volumeName, nodeName)
				if err != nil {
					klog.Errorf("Failed to find device path: %v", err)
					continue
				}
				err = adc.actualStateOfWorld.MarkVolumeAsAttached(volumeName, volumeSpec, nodeName, devicePath)
				if err != nil {
					klog.Errorf("Failed to update volume spec for node %s: %v", nodeName, err)
				}
			}
		}
	}

	return nil
}

func (adc *attachDetachController) podAdd(obj interface{}) {
	pod, ok := obj.(*v1.Pod)
	if pod == nil || !ok {
		return
	}
	if pod.Spec.NodeName == "" {
		// Ignore pods without NodeName, indicating they are not scheduled.
		return
	}

	volumeActionFlag := util.DetermineVolumeAction(
		pod,
		adc.desiredStateOfWorld,
		true /* default volume action */)

	util.ProcessPodVolumes(pod, volumeActionFlag, /* addVolumes */
		adc.desiredStateOfWorld, &adc.volumePluginMgr, adc.pvcLister, adc.pvLister, adc.csiMigratedPluginManager, adc.intreeToCSITranslator)
}

// GetDesiredStateOfWorld returns desired state of world associated with controller
func (adc *attachDetachController) GetDesiredStateOfWorld() cache.DesiredStateOfWorld {
	return adc.desiredStateOfWorld
}

func (adc *attachDetachController) podUpdate(oldObj, newObj interface{}) {
	pod, ok := newObj.(*v1.Pod)
	if pod == nil || !ok {
		return
	}
	if pod.Spec.NodeName == "" {
		// Ignore pods without NodeName, indicating they are not scheduled.
		return
	}

	volumeActionFlag := util.DetermineVolumeAction(
		pod,
		adc.desiredStateOfWorld,
		true /* default volume action */)

	util.ProcessPodVolumes(pod, volumeActionFlag, /* addVolumes */
		adc.desiredStateOfWorld, &adc.volumePluginMgr, adc.pvcLister, adc.pvLister, adc.csiMigratedPluginManager, adc.intreeToCSITranslator)
}

func (adc *attachDetachController) podDelete(obj interface{}) {
	pod, ok := obj.(*v1.Pod)
	if pod == nil || !ok {
		return
	}

	util.ProcessPodVolumes(pod, false, /* addVolumes */
		adc.desiredStateOfWorld, &adc.volumePluginMgr, adc.pvcLister, adc.pvLister, adc.csiMigratedPluginManager, adc.intreeToCSITranslator)
}

func (adc *attachDetachController) nodeAdd(obj interface{}) {
	node, ok := obj.(*v1.Node)
	// TODO: investigate if nodeName is empty then if we can return
	// kubernetes/kubernetes/issues/37777
	if node == nil || !ok {
		return
	}
	nodeName := types.NodeName(node.Name)
	adc.nodeUpdate(nil, obj)
	// kubernetes/kubernetes/issues/37586
	// This is to workaround the case when a node add causes to wipe out
	// the attached volumes field. This function ensures that we sync with
	// the actual status.
	adc.actualStateOfWorld.SetNodeStatusUpdateNeeded(nodeName)
}

func (adc *attachDetachController) nodeUpdate(oldObj, newObj interface{}) {
	node, ok := newObj.(*v1.Node)
	// TODO: investigate if nodeName is empty then if we can return
	if node == nil || !ok {
		return
	}

	nodeName := types.NodeName(node.Name)
	adc.addNodeToDswp(node, nodeName)
	adc.processVolumesInUse(nodeName, node.Status.VolumesInUse)
}

func (adc *attachDetachController) nodeDelete(obj interface{}) {
	node, ok := obj.(*v1.Node)
	if node == nil || !ok {
		return
	}

	nodeName := types.NodeName(node.Name)
	if err := adc.desiredStateOfWorld.DeleteNode(nodeName); err != nil {
		// This might happen during drain, but we still want it to appear in our logs
		klog.Infof("error removing node %q from desired-state-of-world: %v", nodeName, err)
	}

	adc.processVolumesInUse(nodeName, node.Status.VolumesInUse)
}

func (adc *attachDetachController) enqueuePVC(obj interface{}) {
	key, err := kcache.DeletionHandlingMetaNamespaceKeyFunc(obj)
	if err != nil {
		runtime.HandleError(fmt.Errorf("Couldn't get key for object %+v: %v", obj, err))
		return
	}
	adc.pvcQueue.Add(key)
}

// pvcWorker processes items from pvcQueue
func (adc *attachDetachController) pvcWorker() {
	for adc.processNextItem() {
	}
}

func (adc *attachDetachController) processNextItem() bool {
	keyObj, shutdown := adc.pvcQueue.Get()
	if shutdown {
		return false
	}
	defer adc.pvcQueue.Done(keyObj)

	if err := adc.syncPVCByKey(keyObj.(string)); err != nil {
		// Rather than wait for a full resync, re-add the key to the
		// queue to be processed.
		adc.pvcQueue.AddRateLimited(keyObj)
		runtime.HandleError(fmt.Errorf("Failed to sync pvc %q, will retry again: %v", keyObj.(string), err))
		return true
	}

	// Finally, if no error occurs we Forget this item so it does not
	// get queued again until another change happens.
	adc.pvcQueue.Forget(keyObj)
	return true
}

func (adc *attachDetachController) syncPVCByKey(key string) error {
	klog.V(5).Infof("syncPVCByKey[%s]", key)
	namespace, name, err := kcache.SplitMetaNamespaceKey(key)
	if err != nil {
		klog.V(4).Infof("error getting namespace & name of pvc %q to get pvc from informer: %v", key, err)
		return nil
	}
	pvc, err := adc.pvcLister.PersistentVolumeClaims(namespace).Get(name)
	if apierrors.IsNotFound(err) {
		klog.V(4).Infof("error getting pvc %q from informer: %v", key, err)
		return nil
	}
	if err != nil {
		return err
	}

	if pvc.Status.Phase != v1.ClaimBound || pvc.Spec.VolumeName == "" {
		// Skip unbound PVCs.
		return nil
	}

	objs, err := adc.podIndexer.ByIndex(common.PodPVCIndex, key)
	if err != nil {
		return err
	}
	for _, obj := range objs {
		pod, ok := obj.(*v1.Pod)
		if !ok {
			continue
		}
		// we are only interested in active pods with nodeName set
		if len(pod.Spec.NodeName) == 0 || volumeutil.IsPodTerminated(pod, pod.Status) {
			continue
		}
		volumeActionFlag := util.DetermineVolumeAction(
			pod,
			adc.desiredStateOfWorld,
			true /* default volume action */)

		util.ProcessPodVolumes(pod, volumeActionFlag, /* addVolumes */
			adc.desiredStateOfWorld, &adc.volumePluginMgr, adc.pvcLister, adc.pvLister, adc.csiMigratedPluginManager, adc.intreeToCSITranslator)
	}
	return nil
}

// processVolumesInUse processes the list of volumes marked as "in-use"
// according to the specified Node's Status.VolumesInUse and updates the
// corresponding volume in the actual state of the world to indicate that it is
// mounted.
func (adc *attachDetachController) processVolumesInUse(
	nodeName types.NodeName, volumesInUse []v1.UniqueVolumeName) {
	klog.V(4).Infof("processVolumesInUse for node %q", nodeName)
	for _, attachedVolume := range adc.actualStateOfWorld.GetAttachedVolumesForNode(nodeName) {
		mounted := false
		for _, volumeInUse := range volumesInUse {
			if attachedVolume.VolumeName == volumeInUse {
				mounted = true
				break
			}
		}
		err := adc.actualStateOfWorld.SetVolumeMountedByNode(attachedVolume.VolumeName, nodeName, mounted)
		if err != nil {
			klog.Warningf(
				"SetVolumeMountedByNode(%q, %q, %v) returned an error: %v",
				attachedVolume.VolumeName, nodeName, mounted, err)
		}
	}
}

// Process Volume-Attachment objects.
// Should be called only after populating attached volumes in the ASW.
// For each VA object, this function checks if its present in the ASW.
// If not, adds the volume to ASW as an "uncertain" attachment.
// In the reconciler, the logic checks if the volume is present in the DSW;
//   if yes, the reconciler will attempt attach on the volume;
//   if not (could be a dangling attachment), the reconciler will detach this volume.
func (adc *attachDetachController) processVolumeAttachments() error {
	vas, err := adc.volumeAttachmentLister.List(labels.Everything())
	if err != nil {
		klog.Errorf("failed to list VolumeAttachment objects: %v", err)
		return err
	}
	for _, va := range vas {
		nodeName := types.NodeName(va.Spec.NodeName)
		pvName := va.Spec.Source.PersistentVolumeName
		if pvName == nil {
			// Currently VA objects are created for CSI volumes only. nil pvName is unexpected, generate a warning
			klog.Warningf("Skipping the va as its pvName is nil, va.Name: %q, nodeName: %q",
				va.Name, nodeName)
			continue
		}
		pv, err := adc.pvLister.Get(*pvName)
		if err != nil {
			klog.Errorf("Unable to lookup pv object for: %q, err: %v", *pvName, err)
			continue
		}

		var plugin volume.AttachableVolumePlugin
		volumeSpec := volume.NewSpecFromPersistentVolume(pv, false)

		// Consult csiMigratedPluginManager first before querying the plugins registered during runtime in volumePluginMgr.
		// In-tree plugins that provisioned PVs will not be registered anymore after migration to CSI, once the respective
		// feature gate is enabled.
		if inTreePluginName, err := adc.csiMigratedPluginManager.GetInTreePluginNameFromSpec(pv, nil); err == nil {
			if adc.csiMigratedPluginManager.IsMigrationEnabledForPlugin(inTreePluginName) {
				// PV is migrated and should be handled by the CSI plugin instead of the in-tree one
				plugin, _ = adc.volumePluginMgr.FindAttachablePluginByName(csi.CSIPluginName)
				// podNamespace is not needed here for Azurefile as the volumeName generated will be the same with or without podNamespace
				volumeSpec, err = csimigration.TranslateInTreeSpecToCSI(volumeSpec, "" /* podNamespace */, adc.intreeToCSITranslator)
				if err != nil {
					klog.Errorf(
						"Failed to translate intree volumeSpec to CSI volumeSpec for volume:%q, va.Name:%q, nodeName:%q: %s. Error: %v",
						*pvName,
						va.Name,
						nodeName,
						inTreePluginName,
						err)
					continue
				}
			}
		}

		if plugin == nil {
			plugin, err = adc.volumePluginMgr.FindAttachablePluginBySpec(volumeSpec)
			if err != nil || plugin == nil {
				// Currently VA objects are created for CSI volumes only. nil plugin is unexpected, generate a warning
				klog.Warningf(
					"Skipping processing the volume %q on nodeName: %q, no attacher interface found. err=%v",
					*pvName,
					nodeName,
					err)
				continue
			}
		}

		volumeName, err := volumeutil.GetUniqueVolumeNameFromSpec(plugin, volumeSpec)
		if err != nil {
			klog.Errorf(
				"Failed to find unique name for volume:%q, va.Name:%q, nodeName:%q: %v",
				*pvName,
				va.Name,
				nodeName,
				err)
			continue
		}
		attachState := adc.actualStateOfWorld.GetAttachState(volumeName, nodeName)
		if attachState == cache.AttachStateDetached {
			klog.V(1).Infof("Marking volume attachment as uncertain as volume:%q (%q) is not attached (%v)",
				volumeName, nodeName, attachState)
			err = adc.actualStateOfWorld.MarkVolumeAsUncertain(volumeName, volumeSpec, nodeName)
			if err != nil {
				klog.Errorf("MarkVolumeAsUncertain fail to add the volume %q (%q) to ASW. err: %s", volumeName, nodeName, err)
			}
		}
	}
	return nil
}

var _ volume.VolumeHost = &attachDetachController{}
var _ volume.AttachDetachVolumeHost = &attachDetachController{}

func (adc *attachDetachController) CSINodeLister() storagelistersv1.CSINodeLister {
	return adc.csiNodeLister
}

func (adc *attachDetachController) CSIDriverLister() storagelistersv1.CSIDriverLister {
	return adc.csiDriverLister
}

func (adc *attachDetachController) IsAttachDetachController() bool {
	return true
}

func (adc *attachDetachController) VolumeAttachmentLister() storagelistersv1.VolumeAttachmentLister {
	return adc.volumeAttachmentLister
}

// VolumeHost implementation
// This is an unfortunate requirement of the current factoring of volume plugin
// initializing code. It requires kubelet specific methods used by the mounting
// code to be implemented by all initializers even if the initializer does not
// do mounting (like this attach/detach controller).
// Issue kubernetes/kubernetes/issues/14217 to fix this.
func (adc *attachDetachController) GetPluginDir(podUID string) string {
	return ""
}

func (adc *attachDetachController) GetVolumeDevicePluginDir(podUID string) string {
	return ""
}

func (adc *attachDetachController) GetPodsDir() string {
	return ""
}

func (adc *attachDetachController) GetPodVolumeDir(podUID types.UID, pluginName, volumeName string) string {
	return ""
}

func (adc *attachDetachController) GetPodPluginDir(podUID types.UID, pluginName string) string {
	return ""
}

func (adc *attachDetachController) GetPodVolumeDeviceDir(podUID types.UID, pluginName string) string {
	return ""
}

func (adc *attachDetachController) GetKubeClient() clientset.Interface {
	return adc.kubeClient
}

func (adc *attachDetachController) NewWrapperMounter(volName string, spec volume.Spec, pod *v1.Pod, opts volume.VolumeOptions) (volume.Mounter, error) {
	return nil, fmt.Errorf("NewWrapperMounter not supported by Attach/Detach controller's VolumeHost implementation")
}

func (adc *attachDetachController) NewWrapperUnmounter(volName string, spec volume.Spec, podUID types.UID) (volume.Unmounter, error) {
	return nil, fmt.Errorf("NewWrapperUnmounter not supported by Attach/Detach controller's VolumeHost implementation")
}

func (adc *attachDetachController) GetCloudProvider() cloudprovider.Interface {
	return adc.cloud
}

func (adc *attachDetachController) GetMounter(pluginName string) mount.Interface {
	return nil
}

func (adc *attachDetachController) GetHostName() string {
	return ""
}

func (adc *attachDetachController) GetHostIP() (net.IP, error) {
	return nil, fmt.Errorf("GetHostIP() not supported by Attach/Detach controller's VolumeHost implementation")
}

func (adc *attachDetachController) GetNodeAllocatable() (v1.ResourceList, error) {
	return v1.ResourceList{}, nil
}

func (adc *attachDetachController) GetAttachedVolumesFromNodeStatus() (map[v1.UniqueVolumeName]string, error) {
	return map[v1.UniqueVolumeName]string{}, nil
}

func (adc *attachDetachController) GetSecretFunc() func(namespace, name string) (*v1.Secret, error) {
	return func(_, _ string) (*v1.Secret, error) {
		return nil, fmt.Errorf("GetSecret unsupported in attachDetachController")
	}
}

func (adc *attachDetachController) GetConfigMapFunc() func(namespace, name string) (*v1.ConfigMap, error) {
	return func(_, _ string) (*v1.ConfigMap, error) {
		return nil, fmt.Errorf("GetConfigMap unsupported in attachDetachController")
	}
}

func (adc *attachDetachController) GetServiceAccountTokenFunc() func(_, _ string, _ *authenticationv1.TokenRequest) (*authenticationv1.TokenRequest, error) {
	return func(_, _ string, _ *authenticationv1.TokenRequest) (*authenticationv1.TokenRequest, error) {
		return nil, fmt.Errorf("GetServiceAccountToken unsupported in attachDetachController")
	}
}

func (adc *attachDetachController) DeleteServiceAccountTokenFunc() func(types.UID) {
	return func(types.UID) {
		klog.Errorf("DeleteServiceAccountToken unsupported in attachDetachController")
	}
}

func (adc *attachDetachController) GetExec(pluginName string) utilexec.Interface {
	return utilexec.New()
}

func (adc *attachDetachController) addNodeToDswp(node *v1.Node, nodeName types.NodeName) {
	if _, exists := node.Annotations[volumeutil.ControllerManagedAttachAnnotation]; exists {
		keepTerminatedPodVolumes := false

		if t, ok := node.Annotations[volumeutil.KeepTerminatedPodVolumesAnnotation]; ok {
			keepTerminatedPodVolumes = (t == "true")
		}

		// Node specifies annotation indicating it should be managed by attach
		// detach controller. Add it to desired state of world.
		adc.desiredStateOfWorld.AddNode(nodeName, keepTerminatedPodVolumes)
	}
}

func (adc *attachDetachController) GetNodeLabels() (map[string]string, error) {
	return nil, fmt.Errorf("GetNodeLabels() unsupported in Attach/Detach controller")
}

func (adc *attachDetachController) GetNodeName() types.NodeName {
	return ""
}

func (adc *attachDetachController) GetEventRecorder() record.EventRecorder {
	return adc.recorder
}

func (adc *attachDetachController) GetSubpather() subpath.Interface {
	// Subpaths not needed in attachdetach controller
	return nil
}

func (adc *attachDetachController) GetFilteredDialOptions() *proxyutil.FilteredDialOptions {
	return adc.filteredDialOptions
}

func (adc *attachDetachController) GetCSIDriverLister() storagelistersv1.CSIDriverLister {
	return adc.csiDriverLister
}
