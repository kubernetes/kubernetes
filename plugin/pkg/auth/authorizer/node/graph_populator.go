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
	"time"

	"k8s.io/klog/v2"

	certsv1beta1 "k8s.io/api/certificates/v1beta1"
	corev1 "k8s.io/api/core/v1"
	resourceapi "k8s.io/api/resource/v1"
	storagev1 "k8s.io/api/storage/v1"
	"k8s.io/apimachinery/pkg/util/wait"
	certsv1beta1informers "k8s.io/client-go/informers/certificates/v1beta1"
	corev1informers "k8s.io/client-go/informers/core/v1"
	resourceinformers "k8s.io/client-go/informers/resource/v1"
	storageinformers "k8s.io/client-go/informers/storage/v1"
	"k8s.io/client-go/tools/cache"
	"k8s.io/dynamic-resource-allocation/resourceclaim"
	"k8s.io/utils/ptr"
)

type graphPopulator struct {
	graph *Graph
}

func AddGraphEventHandlers(
	graph *Graph,
	nodes corev1informers.NodeInformer,
	pods corev1informers.PodInformer,
	pvs corev1informers.PersistentVolumeInformer,
	attachments storageinformers.VolumeAttachmentInformer,
	slices resourceinformers.ResourceSliceInformer,
	pcrs certsv1beta1informers.PodCertificateRequestInformer,
) {
	g := &graphPopulator{
		graph: graph,
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

	go cache.WaitForNamedCacheSync("node_authorizer", wait.NeverStop, synced...)
}

func (g *graphPopulator) addPod(obj interface{}) {
	g.updatePod(nil, obj)
}

func (g *graphPopulator) updatePod(oldObj, obj interface{}) {
	pod := obj.(*corev1.Pod)
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
			resourceclaim.PodStatusEqual(oldPod.Status.ResourceClaimStatuses, pod.Status.ResourceClaimStatuses) {
			// Node and uid are unchanged, all object references in the pod spec are immutable respectively unmodified (claim statuses).
			klog.V(5).Infof("updatePod %s/%s, node unchanged", pod.Namespace, pod.Name)
			return
		}
	}

	klog.V(4).Infof("updatePod %s/%s for node %s", pod.Namespace, pod.Name, pod.Spec.NodeName)
	startTime := time.Now()
	g.graph.AddPod(pod)
	klog.V(5).Infof("updatePod %s/%s for node %s completed in %v", pod.Namespace, pod.Name, pod.Spec.NodeName, time.Since(startTime))
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

	klog.V(4).Infof("deletePod %s/%s for node %s", pod.Namespace, pod.Name, pod.Spec.NodeName)
	startTime := time.Now()
	g.graph.DeletePod(pod.Name, pod.Namespace)
	klog.V(5).Infof("deletePod %s/%s for node %s completed in %v", pod.Namespace, pod.Name, pod.Spec.NodeName, time.Since(startTime))
}

func (g *graphPopulator) addPV(obj interface{}) {
	g.updatePV(nil, obj)
}

func (g *graphPopulator) updatePV(oldObj, obj interface{}) {
	pv := obj.(*corev1.PersistentVolume)
	// TODO: skip add if uid, pvc, and secrets are all identical between old and new
	g.graph.AddPV(pv)
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
	g.graph.DeletePV(pv.Name)
}

func (g *graphPopulator) addVolumeAttachment(obj interface{}) {
	g.updateVolumeAttachment(nil, obj)
}

func (g *graphPopulator) updateVolumeAttachment(oldObj, obj interface{}) {
	attachment := obj.(*storagev1.VolumeAttachment)
	if oldObj != nil {
		// skip add if node name is identical
		oldAttachment := oldObj.(*storagev1.VolumeAttachment)
		if oldAttachment.Spec.NodeName == attachment.Spec.NodeName {
			return
		}
	}
	g.graph.AddVolumeAttachment(attachment.Name, attachment.Spec.NodeName)
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
	g.graph.DeleteVolumeAttachment(attachment.Name)
}

func (g *graphPopulator) addResourceSlice(obj interface{}) {
	slice, ok := obj.(*resourceapi.ResourceSlice)
	if !ok {
		klog.Infof("unexpected type %T", obj)
		return
	}
	g.graph.AddResourceSlice(slice.Name, ptr.Deref(slice.Spec.NodeName, ""))
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
	g.graph.DeleteResourceSlice(slice.Name)
}

func (g *graphPopulator) addPCR(obj any) {
	pcr, ok := obj.(*certsv1beta1.PodCertificateRequest)
	if !ok {
		klog.Infof("unexpected type %T", obj)
		return
	}
	g.graph.AddPodCertificateRequest(pcr)
}

func (g *graphPopulator) deletePCR(obj any) {
	if tombstone, ok := obj.(cache.DeletedFinalStateUnknown); ok {
		obj = tombstone.Obj
	}
	pcr, ok := obj.(*certsv1beta1.PodCertificateRequest)
	if !ok {
		klog.Infof("unexpected type %T", obj)
		return
	}
	g.graph.DeletePodCertificateRequest(pcr)
}
