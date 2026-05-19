/*
Copyright 2024 The Kubernetes Authors.

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

package selinuxwarning

import (
	"context"
	"errors"
	"fmt"
	"sync"
	"time"

	v1 "k8s.io/api/core/v1"
	storagev1 "k8s.io/api/storage/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	errorutils "k8s.io/apimachinery/pkg/util/errors"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/util/wait"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	coreinformers "k8s.io/client-go/informers/core/v1"
	storageinformersv1 "k8s.io/client-go/informers/storage/v1"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/client-go/kubernetes/scheme"
	v1core "k8s.io/client-go/kubernetes/typed/core/v1"
	corelisters "k8s.io/client-go/listers/core/v1"
	storagelisters "k8s.io/client-go/listers/storage/v1"
	"k8s.io/client-go/tools/cache"
	"k8s.io/client-go/tools/record"
	"k8s.io/client-go/util/workqueue"
	csitrans "k8s.io/csi-translation-lib"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/pkg/controller/volume/attachdetach/util"
	"k8s.io/kubernetes/pkg/controller/volume/common"
	volumecache "k8s.io/kubernetes/pkg/controller/volume/selinuxwarning/cache"
	"k8s.io/kubernetes/pkg/controller/volume/selinuxwarning/translator"
	"k8s.io/kubernetes/pkg/volume"
	"k8s.io/kubernetes/pkg/volume/csi"
	"k8s.io/kubernetes/pkg/volume/csimigration"
	volumeutil "k8s.io/kubernetes/pkg/volume/util"
	"k8s.io/utils/ptr"
)

// SELinuxWarning controller is a controller that emits a warning event and metrics when
// two pods *might* use the same volume with different SELinux labels.
// It is optional. It does nothing on a cluster that has SELinux disabled.
// It does not modify any API objects except for the warning events.
//
// The controller watches *all* Pods and reports issues on all their volumes,
// regardless if the Pod actually run on are Pending forever or if they have
// correct node anti-affinity and never run on the same node.
type Controller struct {
	kubeClient       clientset.Interface
	podLister        corelisters.PodLister
	podIndexer       cache.Indexer
	podsSynced       cache.InformerSynced
	pvLister         corelisters.PersistentVolumeLister
	pvsSynced        cache.InformerSynced
	pvcLister        corelisters.PersistentVolumeClaimLister
	pvcsSynced       cache.InformerSynced
	csiDriverLister  storagelisters.CSIDriverLister
	csiDriversSynced cache.InformerSynced

	vpm               *volume.VolumePluginMgr
	cmpm              csimigration.PluginManager
	csiTranslator     csimigration.InTreeToCSITranslator
	seLinuxTranslator *translator.ControllerSELinuxTranslator
	eventBroadcaster  record.EventBroadcaster
	eventRecorder     record.EventRecorder
	queue             workqueue.TypedRateLimitingInterface[cache.ObjectName]

	labelCache volumecache.VolumeCache
}

func NewController(
	ctx context.Context,
	kubeClient clientset.Interface,
	podInformer coreinformers.PodInformer,
	pvcInformer coreinformers.PersistentVolumeClaimInformer,
	pvInformer coreinformers.PersistentVolumeInformer,
	csiDriverInformer storageinformersv1.CSIDriverInformer,
	plugins []volume.VolumePlugin,
	prober volume.DynamicPluginProber,
) (*Controller, error) {

	eventBroadcaster := record.NewBroadcaster(record.WithContext(ctx))
	recorder := eventBroadcaster.NewRecorder(scheme.Scheme, v1.EventSource{Component: "selinux_warning"})
	seLinuxTranslator := &translator.ControllerSELinuxTranslator{}

	c := &Controller{
		kubeClient:        kubeClient,
		podLister:         podInformer.Lister(),
		podIndexer:        podInformer.Informer().GetIndexer(),
		podsSynced:        podInformer.Informer().HasSynced,
		pvLister:          pvInformer.Lister(),
		pvsSynced:         pvInformer.Informer().HasSynced,
		pvcLister:         pvcInformer.Lister(),
		pvcsSynced:        pvcInformer.Informer().HasSynced,
		csiDriverLister:   csiDriverInformer.Lister(),
		csiDriversSynced:  csiDriverInformer.Informer().HasSynced,
		vpm:               &volume.VolumePluginMgr{},
		seLinuxTranslator: seLinuxTranslator,

		eventBroadcaster: eventBroadcaster,
		eventRecorder:    recorder,
		queue: workqueue.NewTypedRateLimitingQueueWithConfig[cache.ObjectName](
			workqueue.DefaultTypedControllerRateLimiter[cache.ObjectName](),
			workqueue.TypedRateLimitingQueueConfig[cache.ObjectName]{
				Name: "selinux_warning",
			},
		),
		labelCache: volumecache.NewVolumeLabelCache(seLinuxTranslator),
	}

	err := c.vpm.InitPlugins(plugins, prober, c)
	if err != nil {
		return nil, fmt.Errorf("could not initialize volume plugins for SELinux warning controller: %w", err)
	}
	csiTranslator := csitrans.New()
	c.csiTranslator = csiTranslator
	c.cmpm = csimigration.NewPluginManager(csiTranslator, utilfeature.DefaultFeatureGate)

	// Index pods by its PVC keys. Then we don't need to iterate all pods every time to find
	// pods which reference given PVC.
	err = common.AddPodPVCIndexerIfNotPresent(c.podIndexer)
	if err != nil {
		return nil, fmt.Errorf("could not initialize SELinux warning controller: %w", err)
	}

	logger := klog.FromContext(ctx)
	_, err = podInformer.Informer().AddEventHandler(cache.ResourceEventHandlerFuncs{
		AddFunc:    func(obj interface{}) { c.enqueuePod(logger, obj) },
		UpdateFunc: func(oldObj, newObj interface{}) { c.updatePod(logger, oldObj, newObj) },
		DeleteFunc: func(obj interface{}) { c.enqueuePod(logger, obj) },
	})
	if err != nil {
		return nil, err
	}

	_, err = pvcInformer.Informer().AddEventHandler(cache.ResourceEventHandlerFuncs{
		AddFunc:    func(obj interface{}) { c.addPVC(logger, obj) },
		UpdateFunc: func(oldObj, newObj interface{}) { c.updatePVC(logger, oldObj, newObj) },
	})
	if err != nil {
		return nil, err
	}

	_, err = pvInformer.Informer().AddEventHandler(cache.ResourceEventHandlerFuncs{
		AddFunc:    func(obj interface{}) { c.addPV(logger, obj) },
		UpdateFunc: func(oldObj, newObj interface{}) { c.updatePV(logger, oldObj, newObj) },
	})
	if err != nil {
		return nil, err
	}

	_, err = csiDriverInformer.Informer().AddEventHandler(cache.ResourceEventHandlerFuncs{
		AddFunc:    func(obj interface{}) { c.addCSIDriver(logger, obj) },
		UpdateFunc: func(oldObj, newObj interface{}) { c.updateCSIDriver(logger, oldObj, newObj) },
		DeleteFunc: func(obj interface{}) { c.deleteCSIDriver(logger, obj) },
	})
	if err != nil {
		return nil, err
	}

	RegisterMetrics(logger, c.labelCache)

	return c, nil
}

func (c *Controller) enqueuePod(_ klog.Logger, obj interface{}) {
	podRef, err := cache.DeletionHandlingObjectToName(obj)
	if err != nil {
		utilruntime.HandleError(fmt.Errorf("couldn't get key for pod %#v: %w", obj, err))
	}
	c.queue.Add(podRef)
}

func (c *Controller) updatePod(logger klog.Logger, oldObj, newObj interface{}) {
	// Pod.Spec fields that are relevant to this controller are immutable after creation (i.e.
	// pod volumes, SELinux labels, privileged flag). React to update only when the Pod
	// reaches its final state - kubelet will unmount the Pod volumes and the controller should
	// therefore remove them from the cache.
	oldPod, ok := oldObj.(*v1.Pod)
	if !ok {
		return
	}
	newPod, ok := newObj.(*v1.Pod)
	if !ok {
		return
	}

	// This is an optimization. In theory, passing most pod updates to the controller queue should lead to noop.
	// To save some CPU, pass only pod updates that can cause any action in the controller
	if oldPod.Status.Phase == newPod.Status.Phase {
		return
	}
	if newPod.Status.Phase != v1.PodFailed && newPod.Status.Phase != v1.PodSucceeded {
		return
	}
	c.enqueuePod(logger, newObj)
}

func (c *Controller) addPVC(logger klog.Logger, obj interface{}) {
	pvc, ok := obj.(*v1.PersistentVolumeClaim)
	if !ok {
		return
	}
	if pvc.Spec.VolumeName == "" {
		// Ignore unbound PVCs
		return
	}
	c.enqueueAllPodsForPVC(logger, pvc.Namespace, pvc.Name)
}

func (c *Controller) updatePVC(logger klog.Logger, oldObj, newObj interface{}) {
	oldPVC, ok := oldObj.(*v1.PersistentVolumeClaim)
	if !ok {
		return
	}
	newPVC, ok := newObj.(*v1.PersistentVolumeClaim)
	if !ok {
		return
	}
	if oldPVC.Spec.VolumeName != newPVC.Spec.VolumeName {
		// The PVC was just bound
		c.enqueueAllPodsForPVC(logger, newPVC.Namespace, newPVC.Name)
	}
}

func (c *Controller) addPV(logger klog.Logger, obj interface{}) {
	pv, ok := obj.(*v1.PersistentVolume)
	if !ok {
		return
	}
	claimRef := pv.Spec.ClaimRef
	if claimRef == nil {
		// Ignore unbound PVs
		return
	}
	if claimRef.UID == "" {
		// Ignore PVs with incomplete binding.
		return
	}
	c.enqueueAllPodsForPVC(logger, claimRef.Namespace, claimRef.Name)
}

func (c *Controller) updatePV(logger klog.Logger, oldObj, newObj interface{}) {
	oldPV, ok := oldObj.(*v1.PersistentVolume)
	if !ok {
		return
	}
	newPV, ok := newObj.(*v1.PersistentVolume)
	if !ok {
		return
	}
	newClaimRef := newPV.Spec.ClaimRef
	if newClaimRef == nil {
		// Ignore unbound PVs
		return
	}
	if newClaimRef.UID == "" {
		// Ignore PVs with incomplete binding.
		return
	}
	oldClaimRef := oldPV.Spec.ClaimRef
	if oldClaimRef == nil || oldClaimRef.UID != newClaimRef.UID {
		// The PV was just bound (or un-bound)
		// Re-queue all Pods in the same namespace as the PVC bound to this PV
		c.enqueueAllPodsForPVC(logger, newClaimRef.Namespace, newClaimRef.Name)
	}
}

func (c *Controller) enqueueAllPodsForPVC(logger klog.Logger, namespace, name string) {
	// Re-queue all Pods in the same namespace as the PVC.
	// As consequence, all events for Pods in the namespace will be re-sent.
	// Possible optimizations:
	// - Resolve which Pods use the PVC here, in informer hook, using Pod informer. That could block the hook for longer than necessary.
	// - Resolve which Pods use the PVC here, in informer hook, using a new cache (map?) of Pods that wait for a PVC with given name.
	// - Enqueue the PVC name and find Pods that use the PVC in a worker thread. That would mean that the queue can have either PVCs or Pods.
	objs, err := c.podIndexer.ByIndex(common.PodPVCIndex, fmt.Sprintf("%s/%s", namespace, name))
	if err != nil {
		logger.Error(err, "listing pods from cache")
		return
	}
	for _, obj := range objs {
		c.enqueuePod(logger, obj)
	}
}

func (c *Controller) addCSIDriver(_ klog.Logger, obj interface{}) {
	csiDriver, ok := obj.(*storagev1.CSIDriver)
	if !ok {
		return
	}
	// SELinuxMount may have changed. Pods that use volumes of this driver may have
	// a different effective SELinuxChangePolicy.
	// With SELinuxMount: false / nil, Pod SELinuxChangePolicy is ignored and implied to `Recursive`.
	// With SELinuxMount: true, the actual SELinuxChangePolicy is used.
	// Re-queue all pods that use this CSIDriver to re-evaluate their conflicts.
	c.enqueueAllPodsForCSIDriver(csiDriver.Name)
}

func (c *Controller) updateCSIDriver(_ klog.Logger, oldObj, newObj interface{}) {
	oldCSIDriver, ok := oldObj.(*storagev1.CSIDriver)
	if !ok {
		return
	}
	newCSIDriver, ok := newObj.(*storagev1.CSIDriver)
	if !ok {
		return
	}
	oldSELinuxMount := ptr.Deref(oldCSIDriver.Spec.SELinuxMount, false)
	newSELinuxMount := ptr.Deref(newCSIDriver.Spec.SELinuxMount, false)
	if oldSELinuxMount != newSELinuxMount {
		// SELinuxMount changed. Pods that use volumes of this driver may have
		// a different effective SELinuxChangePolicy.
		// With SELinuxMount: false / nil, Pod SELinuxChangePolicy is ignored and implied to `Recursive`.
		// With SELinuxMount: true, the actual SELinuxChangePolicy is used.
		// Re-queue all pods that use this CSIDriver to re-evaluate their conflicts.
		c.enqueueAllPodsForCSIDriver(newCSIDriver.Name)
	}
}

func (c *Controller) deleteCSIDriver(_ klog.Logger, obj interface{}) {
	csiDriver, ok := obj.(*storagev1.CSIDriver)
	if !ok {
		return
	}
	if ptr.Deref(csiDriver.Spec.SELinuxMount, false) {
		// The deleted CSIDriver announced SELinuxMount support. Drivers without CSIDriver instance default to `SELinuxMount: false`.
		// Pods that use volumes of this driver may have a different effective SELinuxChangePolicy now.
		// With SELinuxMount: true, the actual SELinuxChangePolicy was used.
		// With missing CSIDriver (= SELinuxMount: false), Pod SELinuxChangePolicy is ignored and implied to `Recursive`.
		// Re-queue all pods that use this CSIDriver to re-evaluate their conflicts.
		c.enqueueAllPodsForCSIDriver(csiDriver.Name)
	}
}

func (c *Controller) enqueueAllPodsForCSIDriver(csiDriverName string) {
	podKeys := c.labelCache.GetPodsForCSIDriver(csiDriverName)
	for _, podKey := range podKeys {
		c.queue.Add(podKey)
	}
}

func (c *Controller) Run(ctx context.Context, workers int) {
	defer utilruntime.HandleCrash()

	logger := klog.FromContext(ctx)
	logger.Info("Starting SELinux warning controller")

	var wg sync.WaitGroup
	defer func() {
		logger.Info("Shutting down SELinux warning controller")
		c.queue.ShutDown()
		wg.Wait()
	}()

	c.eventBroadcaster.StartStructuredLogging(3) // verbosity level 3 is used by the other KCM controllers
	c.eventBroadcaster.StartRecordingToSink(&v1core.EventSinkImpl{Interface: c.kubeClient.CoreV1().Events("")})
	defer c.eventBroadcaster.Shutdown()

	if !cache.WaitForNamedCacheSyncWithContext(ctx, c.podsSynced, c.pvcsSynced, c.pvsSynced, c.csiDriversSynced) {
		return
	}

	for i := 0; i < workers; i++ {
		wg.Go(func() {
			wait.UntilWithContext(ctx, c.runWorker, time.Second)
		})
	}
	<-ctx.Done()
}

func (c *Controller) runWorker(ctx context.Context) {
	for c.processNextWorkItem(ctx) {
	}
}

func (c *Controller) processNextWorkItem(ctx context.Context) bool {
	item, shutdown := c.queue.Get()
	if shutdown {
		return false
	}
	defer c.queue.Done(item)

	err := c.sync(ctx, item)
	if err == nil {
		c.queue.Forget(item)
		return true
	}

	utilruntime.HandleError(fmt.Errorf("%v failed with: %w", item, err))
	c.queue.AddRateLimited(item)

	return true
}

// syncHandler is invoked for each pod which might need to be processed.
// If an error is returned from this function, the pod will be requeued.
func (c *Controller) sync(ctx context.Context, podRef cache.ObjectName) error {
	logger := klog.FromContext(ctx)

	pod, err := c.podLister.Pods(podRef.Namespace).Get(podRef.Name)
	if err != nil {
		if apierrors.IsNotFound(err) {
			// The pod must have been deleted
			return c.syncPodDelete(ctx, podRef)
		}
		logger.V(5).Info("Error getting pod from informer", "pod", klog.KObj(pod), "podUID", pod.UID, "err", err)
		return err
	}
	if pod.Status.Phase == v1.PodFailed || pod.Status.Phase == v1.PodSucceeded {
		// The pod has reached its final state and kubelet is unmounting is volumes.
		// Remove them from the cache.
		return c.syncPodDelete(ctx, podRef)
	}

	return c.syncPod(ctx, pod)
}

func (c *Controller) syncPodDelete(ctx context.Context, podKey cache.ObjectName) error {
	logger := klog.FromContext(ctx)
	logger.V(4).Info("Deleting pod", "key", podKey)

	c.labelCache.DeletePod(logger, podKey)
	return nil
}

func (c *Controller) syncPod(ctx context.Context, pod *v1.Pod) error {
	logger := klog.FromContext(ctx)
	logger.V(4).Info("Syncing pod", "pod", klog.KObj(pod))

	errs := []error{}
	volumeSpecs := make(map[string]*volume.Spec)

	// Pre-compute volumes
	for i := range pod.Spec.Volumes {
		spec, err := util.CreateVolumeSpec(logger, pod.Spec.Volumes[i], pod, c.vpm, c.pvcLister, c.pvLister, c.cmpm, c.csiTranslator)
		if err != nil {
			// This can happen frequently when PVC or PV do not exist yet.
			// Report it, but continue further.
			errs = append(errs, err)
			continue
		}
		volumeSpecs[pod.Spec.Volumes[i].Name] = spec
	}

	mounts, _, seLinuxLabels := volumeutil.GetPodVolumeNames(pod, true /* collectSELinuxOptions */)
	for _, mount := range mounts.UnsortedList() {
		opts := seLinuxLabels[mount]
		spec, found := volumeSpecs[mount]
		if !found {
			// This must be a volume that failed CreateVolumeSpec above. Error will be reported.
			logger.V(4).Info("skipping not found volume", "pod", klog.KObj(pod), "volume", mount)
			continue
		}
		mountInfo, err := volumeutil.GetMountSELinuxLabel(spec, opts, pod.Spec.SecurityContext, c.vpm, c.seLinuxTranslator)
		if err != nil {
			errors.Is(err, &volumeutil.MultipleSELinuxLabelsError{})
			if volumeutil.IsMultipleSELinuxLabelsError(err) {
				c.eventRecorder.Eventf(pod, v1.EventTypeWarning, "MultipleSELinuxLabels", "Volume %q is mounted twice with different SELinux labels inside this pod", mount)
			}
			logger.V(4).Error(err, "failed to get SELinux label", "pod", klog.KObj(pod), "volume", mount)
			errs = append(errs, err)
			continue
		}

		// Use the same label as kubelet will use for mount -o context.
		// If the Pod has opted in to Recursive policy, it will be empty string here and no conflicts will be reported for it.
		seLinuxLabel := mountInfo.SELinuxMountLabel

		err = c.syncVolume(logger, pod, spec, seLinuxLabel, mountInfo.PluginSupportsSELinuxContextMount)
		if err != nil {
			errs = append(errs, err)
		}
	}

	return errorutils.NewAggregate(errs)
}

func (c *Controller) syncVolume(logger klog.Logger, pod *v1.Pod, spec *volume.Spec, seLinuxLabel string, pluginSupportsSELinuxContextMount bool) error {
	plugin, err := c.vpm.FindPluginBySpec(spec)
	if err != nil {
		// The controller does not have all volume plugins, only those that affect SELinux.
		logger.V(4).Info("Skipping volume of unknown plugin", "volume", spec.Name(), "err", err)
		return nil
	}

	uniqueVolumeName, err := volumeutil.GetUniqueVolumeNameFromSpec(plugin, spec)
	if err != nil {
		return fmt.Errorf("failed to get unique volume name for volume %q: %w", spec.Name(), err)
	}

	changePolicy := v1.SELinuxChangePolicyMountOption
	if pod.Spec.SecurityContext != nil && pod.Spec.SecurityContext.SELinuxChangePolicy != nil {
		changePolicy = *pod.Spec.SecurityContext.SELinuxChangePolicy
		logger.V(5).Info("Using Pod SELinux change policy", "pod", klog.KObj(pod), "changePolicy", changePolicy)
	}
	if !pluginSupportsSELinuxContextMount && changePolicy != v1.SELinuxChangePolicyRecursive {
		logger.V(5).Info("Volume does not support SELinux context mount, setting changePolicy to Recursive", "pod", klog.KObj(pod), "volume", spec.Name())
		changePolicy = v1.SELinuxChangePolicyRecursive
	}

	if seLinuxLabel == "" && changePolicy != v1.SELinuxChangePolicyRecursive {
		logger.V(5).Info("Pod has empty SELinux label, setting changePolicy to Recursive", "pod", klog.KObj(pod))
		changePolicy = v1.SELinuxChangePolicyRecursive
	}

	csiDriver, err := csi.GetCSIDriverName(spec)
	if err != nil {
		// This is likely not a CSI volume
		csiDriver = ""
	}
	logger.V(4).Info("Syncing pod volume", "pod", klog.KObj(pod), "volume", spec.Name(), "label", seLinuxLabel, "uniqueVolumeName", uniqueVolumeName, "changePolicy", changePolicy, "csiDriver", csiDriver)

	conflicts := c.labelCache.AddVolume(logger, uniqueVolumeName, cache.MetaObjectToName(pod), seLinuxLabel, changePolicy, csiDriver)
	c.reportConflictEvents(logger, conflicts)

	return nil
}

func (c *Controller) reportConflictEvents(logger klog.Logger, conflicts []volumecache.Conflict) {
	for _, conflict := range conflicts {
		pod, err := c.podLister.Pods(conflict.Pod.Namespace).Get(conflict.Pod.Name)
		if err != nil {
			logger.V(2).Error(err, "failed to get first pod for event", "pod", conflict.Pod)
			// It does not make sense to report a conflict that has been resolved by deleting one of the pods.
			return
		}
		c.eventRecorder.Event(pod, v1.EventTypeNormal, conflict.EventReason, conflict.EventMessage())
	}
}
