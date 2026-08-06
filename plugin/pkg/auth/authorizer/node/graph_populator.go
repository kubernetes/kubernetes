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

type eventType string

const (
	podEventType           eventType = "pod"
	pvEventType            eventType = "pv"
	attachmentEventType    eventType = "attachment"
	resourceSliceEventType eventType = "slice"
	pcrEventType           eventType = "pcr"
)

type graphEventKey struct {
	eventType eventType
	namespace string
	name      string
}

type graphPopulator struct {
	graph            *Graph
	queue            workqueue.TypedRateLimitingInterface[graphEventKey]
	podLister        corev1listers.PodLister
	pvLister         corev1listers.PersistentVolumeLister
	attachmentLister storagev1listers.VolumeAttachmentLister
	sliceLister      resourcev1listers.ResourceSliceLister
	pcrLister        certsv1listers.PodCertificateRequestLister
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
		graph: graph,
		queue: workqueue.NewTypedRateLimitingQueueWithConfig(
			workqueue.DefaultTypedControllerRateLimiter[graphEventKey](),
			workqueue.TypedRateLimitingQueueConfig[graphEventKey]{
				Name: "node_authorizer_graph_populator",
			},
		),
		podLister:        pods.Lister(),
		pvLister:         pvs.Lister(),
		attachmentLister: attachments.Lister(),
	}

	if slices != nil {
		g.sliceLister = slices.Lister()
	}
	if pcrs != nil {
		g.pcrLister = pcrs.Lister()
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
		sliceHandler, _ := slices.Informer().AddEventHandler(cache.ResourceEventHandlerFuncs{
			AddFunc:    g.addResourceSlice,
			UpdateFunc: nil, // Not needed, NodeName is immutable.
			DeleteFunc: g.deleteResourceSlice,
		})
		synced = append(synced, sliceHandler.HasSynced)
	}

	if pcrs != nil {
		pcrHandler, _ := pcrs.Informer().AddEventHandler(cache.ResourceEventHandlerFuncs{
			AddFunc:    g.addPCR,
			UpdateFunc: nil, // Not needed, spec fields are immutable.
			DeleteFunc: g.deletePCR,
		})
		synced = append(synced, pcrHandler.HasSynced)
	}

	go func() {
		<-ctx.Done()
		g.queue.ShutDown()
	}()

	go wait.Until(g.runWorker, time.Second, ctx.Done())

	go cache.WaitForNamedCacheSync("node_authorizer", ctx.Done(), synced...)
}

func (g *graphPopulator) addPod(obj interface{}) {
	pod, ok := obj.(*corev1.Pod)
	if !ok {
		return
	}
	if len(pod.Spec.NodeName) == 0 {
		return
	}
	g.queue.Add(graphEventKey{eventType: podEventType, namespace: pod.Namespace, name: pod.Name})
}

func (g *graphPopulator) updatePod(oldObj, obj interface{}) {
	oldPod, ok := oldObj.(*corev1.Pod)
	if !ok {
		return
	}
	pod, ok := obj.(*corev1.Pod)
	if !ok {
		return
	}

	if len(pod.Spec.NodeName) == 0 {
		// No node assigned
		klog.V(5).Infof("updatePod %s/%s, no node", pod.Namespace, pod.Name)
		return
	}

	hasNewEphemeralContainers := len(pod.Spec.EphemeralContainers) > len(oldPod.Spec.EphemeralContainers)
	if (pod.Spec.NodeName == oldPod.Spec.NodeName) && (pod.UID == oldPod.UID) &&
		!hasNewEphemeralContainers &&
		resourceclaim.PodStatusEqual(oldPod.Status.ResourceClaimStatuses, pod.Status.ResourceClaimStatuses) &&
		resourceclaim.PodExtendedStatusEqual(oldPod.Status.ExtendedResourceClaimStatus, pod.Status.ExtendedResourceClaimStatus) {
		return
	}

	g.queue.Add(graphEventKey{eventType: podEventType, namespace: pod.Namespace, name: pod.Name})
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
	g.queue.Add(graphEventKey{eventType: podEventType, namespace: pod.Namespace, name: pod.Name})
}

func (g *graphPopulator) addPV(obj interface{}) {
	pv, ok := obj.(*corev1.PersistentVolume)
	if !ok {
		return
	}
	g.queue.Add(graphEventKey{eventType: pvEventType, name: pv.Name})
}

func (g *graphPopulator) updatePV(oldObj, obj interface{}) {
	pv, ok := obj.(*corev1.PersistentVolume)
	if !ok {
		klog.Infof("unexpected type %T", obj)
		return
	}
	// TODO: skip add if uid, pvc, and secrets are all identical between old and new
	g.queue.Add(graphEventKey{eventType: pvEventType, name: pv.Name})
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
	g.queue.Add(graphEventKey{eventType: pvEventType, name: pv.Name})
}

func (g *graphPopulator) addVolumeAttachment(obj interface{}) {
	attachment, ok := obj.(*storagev1.VolumeAttachment)
	if !ok {
		return
	}
	g.queue.Add(graphEventKey{eventType: attachmentEventType, name: attachment.Name})
}

func (g *graphPopulator) updateVolumeAttachment(oldObj, obj interface{}) {
	oldAttachment, ok := oldObj.(*storagev1.VolumeAttachment)
	if !ok {
		return
	}
	attachment, ok := obj.(*storagev1.VolumeAttachment)
	if !ok {
		klog.Infof("unexpected type %T", obj)
		return
	}
	// skip add if node name is identical
	if oldAttachment.Spec.NodeName == attachment.Spec.NodeName {
		return
	}
	g.queue.Add(graphEventKey{eventType: attachmentEventType, name: attachment.Name})
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
	g.queue.Add(graphEventKey{eventType: attachmentEventType, name: attachment.Name})
}

func (g *graphPopulator) addResourceSlice(obj interface{}) {
	slice, ok := obj.(*resourceapi.ResourceSlice)
	if !ok {
		klog.Infof("unexpected type %T", obj)
		return
	}
	g.queue.Add(graphEventKey{eventType: resourceSliceEventType, name: slice.Name})
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
	g.queue.Add(graphEventKey{eventType: resourceSliceEventType, name: slice.Name})
}

func (g *graphPopulator) addPCR(obj any) {
	pcr, ok := obj.(*certsv1.PodCertificateRequest)
	if !ok {
		klog.Infof("unexpected type %T", obj)
		return
	}
	g.queue.Add(graphEventKey{eventType: pcrEventType, namespace: pcr.Namespace, name: pcr.Name})
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
	g.queue.Add(graphEventKey{eventType: pcrEventType, namespace: pcr.Namespace, name: pcr.Name})
}

func (g *graphPopulator) runWorker() {
	for g.processNextWorkItem() {
	}
}

func (g *graphPopulator) processNextWorkItem() bool {
	key, shutdown := g.queue.Get()
	if shutdown {
		return false
	}
	defer g.queue.Done(key)

	err := g.processGraphEventKey(key)
	if err == nil {
		g.queue.Forget(key)
		return true
	}

	utilruntime.HandleError(fmt.Errorf("node authorizer graph populator: failed to process %v: %w", key, err))
	g.queue.AddRateLimited(key)

	return true
}

func (g *graphPopulator) processGraphEventKey(key graphEventKey) error {
	switch key.eventType {
	case podEventType:
		pod, err := g.podLister.Pods(key.namespace).Get(key.name)
		if err != nil {
			if errors.IsNotFound(err) {
				g.processDeletePod(key.name, key.namespace)
				return nil
			}
			return err
		}
		g.processAddOrUpdatePod(pod)
		return nil
	case pvEventType:
		pv, err := g.pvLister.Get(key.name)
		if err != nil {
			if errors.IsNotFound(err) {
				g.processDeletePV(key.name)
				return nil
			}
			return err
		}
		g.processAddOrUpdatePV(pv)
		return nil
	case attachmentEventType:
		attachment, err := g.attachmentLister.Get(key.name)
		if err != nil {
			if errors.IsNotFound(err) {
				g.processDeleteVolumeAttachment(key.name)
				return nil
			}
			return err
		}
		g.processAddOrUpdateVolumeAttachment(attachment)
		return nil
	case resourceSliceEventType:
		if g.sliceLister == nil {
			return nil
		}
		slice, err := g.sliceLister.Get(key.name)
		if err != nil {
			if errors.IsNotFound(err) {
				g.processDeleteResourceSlice(key.name)
				return nil
			}
			return err
		}
		g.processAddResourceSlice(slice)
		return nil
	case pcrEventType:
		if g.pcrLister == nil {
			return nil
		}
		pcr, err := g.pcrLister.PodCertificateRequests(key.namespace).Get(key.name)
		if err != nil {
			if errors.IsNotFound(err) {
				g.processDeletePCR(key.name, key.namespace)
				return nil
			}
			return err
		}
		g.processAddPCR(pcr)
		return nil
	}
	return nil
}

func (g *graphPopulator) processDeletePod(name, namespace string) {
	g.graph.DeletePod(name, namespace)
}

func (g *graphPopulator) processAddOrUpdatePod(pod *corev1.Pod) {
	g.graph.AddPod(pod)
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
