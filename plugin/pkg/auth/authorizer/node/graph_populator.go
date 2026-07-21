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

package node

import (
	"context"
	"fmt"
	"time"

	certsv1 "k8s.io/api/certificates/v1"
	corev1 "k8s.io/api/core/v1"
	resourceapi "k8s.io/api/resource/v1"
	storagev1 "k8s.io/api/storage/v1"
	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/util/wait"
	certsv1informers "k8s.io/client-go/informers/certificates/v1"
	corev1informers "k8s.io/client-go/informers/core/v1"
	resourceinformers "k8s.io/client-go/informers/resource/v1"
	storageinformers "k8s.io/client-go/informers/storage/v1"
	certsv1listers "k8s.io/client-go/listers/certificates/v1"
	corev1listers "k8s.io/client-go/listers/core/v1"
	resourcev1listers "k8s.io/client-go/listers/resource/v1"
	storagev1listers "k8s.io/client-go/listers/storage/v1"
	"k8s.io/client-go/tools/cache"
	"k8s.io/client-go/util/workqueue"
	"k8s.io/dynamic-resource-allocation/resourceclaim"
	"k8s.io/klog/v2"
	"k8s.io/utils/ptr"
)

type graphPopulator struct {
	graph *Graph

	podQueue        workqueue.TypedRateLimitingInterface[types.NamespacedName]
	pvQueue         workqueue.TypedRateLimitingInterface[types.NamespacedName]
	attachmentQueue workqueue.TypedRateLimitingInterface[types.NamespacedName]
	sliceQueue      workqueue.TypedRateLimitingInterface[types.NamespacedName]
	pcrQueue        workqueue.TypedRateLimitingInterface[types.NamespacedName]

	podLister        corev1listers.PodLister
	pvLister         corev1listers.PersistentVolumeLister
	attachmentLister storagev1listers.VolumeAttachmentLister
	sliceLister      resourcev1listers.ResourceSliceLister
	pcrLister        certsv1listers.PodCertificateRequestLister
}

func newRateLimitingQueue(name string) workqueue.TypedRateLimitingInterface[types.NamespacedName] {
	return workqueue.NewTypedRateLimitingQueueWithConfig(
		workqueue.DefaultTypedControllerRateLimiter[types.NamespacedName](),
		workqueue.TypedRateLimitingQueueConfig[types.NamespacedName]{
			Name: name,
		},
	)
}

func AddGraphEventHandlers(
	ctx context.Context,
	graph *Graph,
	nodes corev1informers.NodeInformer,
	pods corev1informers.PodInformer,
	pvs corev1informers.PersistentVolumeInformer,
	attachments storageinformers.VolumeAttachmentInformer,
	slices resourceinformers.ResourceSliceInformer,
	pcrs certsv1informers.PodCertificateRequestInformer,
) {
	g := &graphPopulator{
		graph:            graph,
		podQueue:         newRateLimitingQueue("node_authorizer_pods"),
		pvQueue:          newRateLimitingQueue("node_authorizer_persistentvolumes"),
		attachmentQueue:  newRateLimitingQueue("node_authorizer_volumeattachments"),
		podLister:        pods.Lister(),
		pvLister:         pvs.Lister(),
		attachmentLister: attachments.Lister(),
	}

	queues := []workqueue.TypedRateLimitingInterface[types.NamespacedName]{
		g.podQueue, g.pvQueue, g.attachmentQueue,
	}
	workers := []func(){
		g.runPodWorker, g.runPVWorker, g.runAttachmentWorker,
	}

	podHandler, _ := pods.Informer().AddEventHandler(cache.ResourceEventHandlerFuncs{
		AddFunc:    g.addPod,
		UpdateFunc: g.updatePod,
		DeleteFunc: g.deletePod,
	})

	pvsHandler, _ := pvs.Informer().AddEventHandler(cache.ResourceEventHandlerFuncs{
		AddFunc:    g.addPV,
		UpdateFunc: g.updatePV,
		DeleteFunc: g.deletePV,
	})

	attachHandler, _ := attachments.Informer().AddEventHandler(cache.ResourceEventHandlerFuncs{
		AddFunc:    g.addVolumeAttachment,
		UpdateFunc: g.updateVolumeAttachment,
		DeleteFunc: g.deleteVolumeAttachment,
	})

	synced := []cache.InformerSynced{
		podHandler.HasSynced, pvsHandler.HasSynced, attachHandler.HasSynced,
	}

	if slices != nil {
		g.sliceQueue = newRateLimitingQueue("node_authorizer_resourceslices")
		g.sliceLister = slices.Lister()
		queues = append(queues, g.sliceQueue)
		workers = append(workers, g.runSliceWorker)
		sliceHandler, _ := slices.Informer().AddEventHandler(cache.ResourceEventHandlerFuncs{
			AddFunc:    g.addResourceSlice,
			UpdateFunc: nil, // Not needed, NodeName is immutable.
			DeleteFunc: g.deleteResourceSlice,
		})
		synced = append(synced, sliceHandler.HasSynced)
	}

	if pcrs != nil {
		g.pcrQueue = newRateLimitingQueue("node_authorizer_podcertificaterequests")
		g.pcrLister = pcrs.Lister()
		queues = append(queues, g.pcrQueue)
		workers = append(workers, g.runPCRWorker)
		pcrHandler, _ := pcrs.Informer().AddEventHandler(cache.ResourceEventHandlerFuncs{
			AddFunc:    g.addPCR,
			UpdateFunc: nil, // Not needed, spec fields are immutable.
			DeleteFunc: g.deletePCR,
		})
		synced = append(synced, pcrHandler.HasSynced)
	}

	go func() {
		<-ctx.Done()
		for _, q := range queues {
			q.ShutDown()
		}
	}()

	for _, run := range workers {
		go wait.Until(run, time.Second, ctx.Done())
	}

	go cache.WaitForNamedCacheSync("node_authorizer", ctx.Done(), synced...)
}

func (g *graphPopulator) addPod(obj interface{}) {
	g.updatePod(nil, obj)
}

func (g *graphPopulator) updatePod(oldObj, obj interface{}) {
	pod, ok := obj.(*corev1.Pod)
	if !ok {
		klog.Infof("unexpected type %T", obj)
		return
	}
	if len(pod.Spec.NodeName) == 0 {
		// No node assigned
		klog.V(5).Infof("updatePod %s/%s, no node", pod.Namespace, pod.Name)
		return
	}
	if oldPod, ok := oldObj.(*corev1.Pod); ok && oldPod != nil {
		// Ephemeral containers can add new secret or config map references to the pod.
		hasNewEphemeralContainers := len(pod.Spec.EphemeralContainers) > len(oldPod.Spec.EphemeralContainers)
		if (pod.Spec.NodeName == oldPod.Spec.NodeName) && (pod.UID == oldPod.UID) &&
			!hasNewEphemeralContainers &&
			resourceclaim.PodStatusEqual(oldPod.Status.ResourceClaimStatuses, pod.Status.ResourceClaimStatuses) &&
			resourceclaim.PodExtendedStatusEqual(oldPod.Status.ExtendedResourceClaimStatus, pod.Status.ExtendedResourceClaimStatus) {
			// Node and uid are unchanged, all object references in the pod spec are immutable respectively unmodified (claim statuses).
			klog.V(5).Infof("updatePod %s/%s, node unchanged", pod.Namespace, pod.Name)
			return
		}
	}

	g.podQueue.Add(types.NamespacedName{Namespace: pod.Namespace, Name: pod.Name})
}

func (g *graphPopulator) deletePod(obj interface{}) {
	if tombstone, ok := obj.(cache.DeletedFinalStateUnknown); ok {
		obj = tombstone.Obj
	}
	pod, ok := obj.(*corev1.Pod)
	if !ok {
		klog.Infof("unexpected type %T", obj)
		return
	}
	if len(pod.Spec.NodeName) == 0 {
		klog.V(5).Infof("deletePod %s/%s, no node", pod.Namespace, pod.Name)
		return
	}

	g.podQueue.Add(types.NamespacedName{Namespace: pod.Namespace, Name: pod.Name})
}

func (g *graphPopulator) addPV(obj interface{}) {
	g.updatePV(nil, obj)
}

func (g *graphPopulator) updatePV(oldObj, obj interface{}) {
	pv, ok := obj.(*corev1.PersistentVolume)
	if !ok {
		klog.Infof("unexpected type %T", obj)
		return
	}
	g.pvQueue.Add(types.NamespacedName{Namespace: pv.Namespace, Name: pv.Name})
}

func (g *graphPopulator) deletePV(obj interface{}) {
	if tombstone, ok := obj.(cache.DeletedFinalStateUnknown); ok {
		obj = tombstone.Obj
	}
	pv, ok := obj.(*corev1.PersistentVolume)
	if !ok {
		klog.Infof("unexpected type %T", obj)
		return
	}
	g.pvQueue.Add(types.NamespacedName{Namespace: pv.Namespace, Name: pv.Name})
}

func (g *graphPopulator) addVolumeAttachment(obj interface{}) {
	g.updateVolumeAttachment(nil, obj)
}

func (g *graphPopulator) updateVolumeAttachment(oldObj, obj interface{}) {
	attachment, ok := obj.(*storagev1.VolumeAttachment)
	if !ok {
		klog.Infof("unexpected type %T", obj)
		return
	}
	if oldAttachment, ok := oldObj.(*storagev1.VolumeAttachment); ok && oldAttachment != nil {
		// skip add if node name is identical
		if oldAttachment.Spec.NodeName == attachment.Spec.NodeName {
			return
		}
	}
	g.attachmentQueue.Add(types.NamespacedName{Namespace: attachment.Namespace, Name: attachment.Name})
}

func (g *graphPopulator) deleteVolumeAttachment(obj interface{}) {
	if tombstone, ok := obj.(cache.DeletedFinalStateUnknown); ok {
		obj = tombstone.Obj
	}
	attachment, ok := obj.(*storagev1.VolumeAttachment)
	if !ok {
		klog.Infof("unexpected type %T", obj)
		return
	}
	g.attachmentQueue.Add(types.NamespacedName{Namespace: attachment.Namespace, Name: attachment.Name})
}

func (g *graphPopulator) addResourceSlice(obj interface{}) {
	slice, ok := obj.(*resourceapi.ResourceSlice)
	if !ok {
		klog.Infof("unexpected type %T", obj)
		return
	}
	g.sliceQueue.Add(types.NamespacedName{Namespace: slice.Namespace, Name: slice.Name})
}

func (g *graphPopulator) deleteResourceSlice(obj interface{}) {
	if tombstone, ok := obj.(cache.DeletedFinalStateUnknown); ok {
		obj = tombstone.Obj
	}
	slice, ok := obj.(*resourceapi.ResourceSlice)
	if !ok {
		klog.Infof("unexpected type %T", obj)
		return
	}
	g.sliceQueue.Add(types.NamespacedName{Namespace: slice.Namespace, Name: slice.Name})
}

func (g *graphPopulator) addPCR(obj any) {
	pcr, ok := obj.(*certsv1.PodCertificateRequest)
	if !ok {
		klog.Infof("unexpected type %T", obj)
		return
	}
	g.pcrQueue.Add(types.NamespacedName{Namespace: pcr.Namespace, Name: pcr.Name})
}

func (g *graphPopulator) deletePCR(obj any) {
	if tombstone, ok := obj.(cache.DeletedFinalStateUnknown); ok {
		obj = tombstone.Obj
	}
	pcr, ok := obj.(*certsv1.PodCertificateRequest)
	if !ok {
		klog.Infof("unexpected type %T", obj)
		return
	}
	g.pcrQueue.Add(types.NamespacedName{Namespace: pcr.Namespace, Name: pcr.Name})
}

func runWorker(queue workqueue.TypedRateLimitingInterface[types.NamespacedName], processKey func(types.NamespacedName) error) {
	for processNextWorkItem(queue, processKey) {
	}
}

func processNextWorkItem(queue workqueue.TypedRateLimitingInterface[types.NamespacedName], processKey func(types.NamespacedName) error) bool {
	key, shutdown := queue.Get()
	if shutdown {
		return false
	}
	defer queue.Done(key)

	err := processKey(key)
	if err == nil {
		queue.Forget(key)
		return true
	}

	utilruntime.HandleError(fmt.Errorf("node authorizer graph populator: failed to process %v: %w", key, err))
	queue.AddRateLimited(key)

	return true
}

func (g *graphPopulator) runPodWorker() {
	runWorker(g.podQueue, g.processPodKey)
}

func (g *graphPopulator) processPodKey(key types.NamespacedName) error {
	pod, err := g.podLister.Pods(key.Namespace).Get(key.Name)
	if err != nil {
		if errors.IsNotFound(err) {
			g.processDeletePod(key.Name, key.Namespace)
			return nil
		}
		return err
	}
	g.processAddOrUpdatePod(pod)
	return nil
}

func (g *graphPopulator) runPVWorker() {
	runWorker(g.pvQueue, g.processPVKey)
}

func (g *graphPopulator) processPVKey(key types.NamespacedName) error {
	pv, err := g.pvLister.Get(key.Name)
	if err != nil {
		if errors.IsNotFound(err) {
			g.processDeletePV(key.Name)
			return nil
		}
		return err
	}
	g.processAddOrUpdatePV(pv)
	return nil
}

func (g *graphPopulator) runAttachmentWorker() {
	runWorker(g.attachmentQueue, g.processAttachmentKey)
}

func (g *graphPopulator) processAttachmentKey(key types.NamespacedName) error {
	attachment, err := g.attachmentLister.Get(key.Name)
	if err != nil {
		if errors.IsNotFound(err) {
			g.processDeleteVolumeAttachment(key.Name)
			return nil
		}
		return err
	}
	g.processAddOrUpdateVolumeAttachment(attachment)
	return nil
}

func (g *graphPopulator) runSliceWorker() {
	runWorker(g.sliceQueue, g.processSliceKey)
}

func (g *graphPopulator) processSliceKey(key types.NamespacedName) error {
	if g.sliceLister == nil {
		return nil
	}
	slice, err := g.sliceLister.Get(key.Name)
	if err != nil {
		if errors.IsNotFound(err) {
			g.processDeleteResourceSlice(key.Name)
			return nil
		}
		return err
	}
	g.processAddResourceSlice(slice)
	return nil
}

func (g *graphPopulator) runPCRWorker() {
	runWorker(g.pcrQueue, g.processPCRKey)
}

func (g *graphPopulator) processPCRKey(key types.NamespacedName) error {
	if g.pcrLister == nil {
		return nil
	}
	pcr, err := g.pcrLister.PodCertificateRequests(key.Namespace).Get(key.Name)
	if err != nil {
		if errors.IsNotFound(err) {
			g.processDeletePCR(key.Name, key.Namespace)
			return nil
		}
		return err
	}
	g.processAddPCR(pcr)
	return nil
}

func (g *graphPopulator) processDeletePod(name, namespace string) {
	klog.V(4).Infof("deletePod %s/%s", namespace, name)
	startTime := time.Now()
	g.graph.DeletePod(name, namespace)
	klog.V(5).Infof("deletePod %s/%s completed in %v", namespace, name, time.Since(startTime))
}

func (g *graphPopulator) processAddOrUpdatePod(pod *corev1.Pod) {
	klog.V(4).Infof("updatePod %s/%s for node %s", pod.Namespace, pod.Name, pod.Spec.NodeName)
	startTime := time.Now()
	g.graph.AddPod(pod)
	klog.V(5).Infof("updatePod %s/%s for node %s completed in %v", pod.Namespace, pod.Name, pod.Spec.NodeName, time.Since(startTime))
}

func (g *graphPopulator) processDeletePV(name string) {
	g.graph.DeletePV(name)
}

func (g *graphPopulator) processAddOrUpdatePV(pv *corev1.PersistentVolume) {
	g.graph.AddPV(pv)
}

func (g *graphPopulator) processDeleteVolumeAttachment(name string) {
	g.graph.DeleteVolumeAttachment(name)
}

func (g *graphPopulator) processAddOrUpdateVolumeAttachment(attachment *storagev1.VolumeAttachment) {
	g.graph.AddVolumeAttachment(attachment.Name, attachment.Spec.NodeName)
}

func (g *graphPopulator) processDeleteResourceSlice(name string) {
	g.graph.DeleteResourceSlice(name)
}

func (g *graphPopulator) processAddResourceSlice(slice *resourceapi.ResourceSlice) {
	g.graph.AddResourceSlice(slice.Name, ptr.Deref(slice.Spec.NodeName, ""))
}

func (g *graphPopulator) processDeletePCR(name, namespace string) {
	g.graph.DeletePodCertificateRequest(&certsv1.PodCertificateRequest{
		ObjectMeta: metav1.ObjectMeta{
			Name:      name,
			Namespace: namespace,
		},
	})
}

func (g *graphPopulator) processAddPCR(pcr *certsv1.PodCertificateRequest) {
	g.graph.AddPodCertificateRequest(pcr)
}
