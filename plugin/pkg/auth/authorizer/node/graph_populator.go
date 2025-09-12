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

	certsv1alpha1 "k8s.io/api/certificates/v1alpha1"
	corev1 "k8s.io/api/core/v1"
	discoveryv1 "k8s.io/api/discovery/v1"
	resourceapi "k8s.io/api/resource/v1"
	storagev1 "k8s.io/api/storage/v1"
	"k8s.io/apimachinery/pkg/util/wait"
	certsv1alpha1informers "k8s.io/client-go/informers/certificates/v1alpha1"
	corev1informers "k8s.io/client-go/informers/core/v1"
	discoveryv1informers "k8s.io/client-go/informers/discovery/v1"
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
	pcrs certsv1alpha1informers.PodCertificateRequestInformer,
	endpoints corev1informers.EndpointsInformer,
	endpointslices discoveryv1informers.EndpointSliceInformer,
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

	if endpoints != nil {
		endpointsHandler, _ := endpoints.Informer().AddEventHandler(cache.ResourceEventHandlerFuncs{
			AddFunc:    g.addEndpoint,
			UpdateFunc: g.updateEndpoint,
			DeleteFunc: g.deleteEndpoint,
		})
		synced = append(synced, endpointsHandler.HasSynced)
	}

	if endpointslices != nil {
		endpointslicesHandler, _ := endpointslices.Informer().AddEventHandler(cache.ResourceEventHandlerFuncs{
			AddFunc:    g.addEndpointslice,
			UpdateFunc: g.updateEndpointslice,
			DeleteFunc: g.deleteEndpointslice,
		})
		synced = append(synced, endpointslicesHandler.HasSynced)
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
	pcr, ok := obj.(*certsv1alpha1.PodCertificateRequest)
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
	pcr, ok := obj.(*certsv1alpha1.PodCertificateRequest)
	if !ok {
		klog.Infof("unexpected type %T", obj)
		return
	}
	g.graph.DeletePodCertificateRequest(pcr)
}

func (g *graphPopulator) addEndpoint(obj any) {
	g.updateEndpoint(nil, obj)
}

func (g *graphPopulator) updateEndpoint(oldObj, obj any) {
	ep := obj.(*corev1.Endpoints)

	if oldEp, ok := oldObj.(*corev1.Endpoints); ok && oldEp != nil {
		hasNewAddresses := false
		epAddrsMap := make(map[string]struct{}, len(ep.Subsets))
		oldEpAddrsMap := make(map[string]struct{}, len(oldEp.Subsets))
		for _, subset := range ep.Subsets {
			for _, addr := range subset.Addresses {
				epAddrsMap[addr.IP] = struct{}{}
			}
		}
		for _, subset := range oldEp.Subsets {
			for _, addr := range subset.Addresses {
				oldEpAddrsMap[addr.IP] = struct{}{}
			}
		}
		if len(epAddrsMap) != len(oldEpAddrsMap) {
			hasNewAddresses = true
		} else {
			for addr := range epAddrsMap {
				if _, exists := oldEpAddrsMap[addr]; !exists {
					hasNewAddresses = true
					break
				}
			}
		}
		if !hasNewAddresses {
			klog.V(5).Infof("updateEndpoint %s/%s, endpoints addresses unchanged", ep.Namespace, ep.Name)
			return
		}
	}

	klog.V(4).Infof("updateEndpoint %s/%s", ep.Namespace, ep.Name)
	startTime := time.Now()
	g.graph.AddEndpoint(ep)
	klog.V(5).Infof("updateEndpoint %s/%s completed in %v", ep.Namespace, ep.Name, time.Since(startTime))
}

func (g *graphPopulator) deleteEndpoint(obj any) {
	if tombstone, ok := obj.(cache.DeletedFinalStateUnknown); ok {
		obj = tombstone.Obj
	}
	ep, ok := obj.(*corev1.Endpoints)
	if !ok {
		klog.Infof("unexpected type %T", obj)
		return
	}
	if len(ep.Subsets) == 0 {
		klog.V(5).Infof("deleteEndpoint %s/%s, no subsets", ep.Namespace, ep.Name)
		return
	}

	klog.V(4).Infof("deleteEndpoint %s/%s", ep.Namespace, ep.Name)
	startTime := time.Now()
	g.graph.DeleteEndpoint(ep.Name, ep.Namespace)
	klog.V(5).Infof("deleteEndpoint %s/%s completed in %v", ep.Namespace, ep.Name, time.Since(startTime))
}

func (g *graphPopulator) addEndpointslice(obj any) {
	g.updateEndpointslice(nil, obj)
}

func (g *graphPopulator) updateEndpointslice(oldObj, obj any) {
	epSlice := obj.(*discoveryv1.EndpointSlice)
	if len(epSlice.Endpoints) == 0 {
		klog.V(5).Infof("updateEndpointslice %s/%s, no endpoints", epSlice.Namespace, epSlice.Name)
		return
	}
	if oldEpSlice, ok := oldObj.(*discoveryv1.EndpointSlice); ok && oldEpSlice != nil {
		hasNewAddresses := false
		epSliceAddrsMap := make(map[string]struct{}, len(epSlice.Endpoints))
		oldEpSliceAddrsMap := make(map[string]struct{}, len(oldEpSlice.Endpoints))
		for _, ep := range epSlice.Endpoints {
			for _, addr := range ep.Addresses {
				epSliceAddrsMap[addr] = struct{}{}
			}
		}
		for _, ep := range oldEpSlice.Endpoints {
			for _, addr := range ep.Addresses {
				oldEpSliceAddrsMap[addr] = struct{}{}
			}
		}
		if len(epSliceAddrsMap) != len(oldEpSliceAddrsMap) {
			hasNewAddresses = true
		} else {
			for addr := range epSliceAddrsMap {
				if _, exists := oldEpSliceAddrsMap[addr]; !exists {
					hasNewAddresses = true
					break
				}
			}
		}
		if !hasNewAddresses {
			klog.V(5).Infof("updateEndpointslice %s/%s, endpoints addresses unchanged", epSlice.Namespace, epSlice.Name)
			return
		}
	}

	klog.V(4).Infof("updateEndpointslice %s/%s", epSlice.Namespace, epSlice.Name)
	startTime := time.Now()
	g.graph.AddEndpointslice(epSlice)
	klog.V(5).Infof("updateEndpointslice %s/%s completed in %v", epSlice.Namespace, epSlice.Name, time.Since(startTime))
}

func (g *graphPopulator) deleteEndpointslice(obj any) {
	if tombstone, ok := obj.(cache.DeletedFinalStateUnknown); ok {
		obj = tombstone.Obj
	}
	epSlice, ok := obj.(*discoveryv1.EndpointSlice)
	if !ok {
		klog.Infof("unexpected type %T", obj)
		return
	}
	if len(epSlice.Endpoints) == 0 {
		klog.V(5).Infof("deleteEndpointslice %s/%s, no endpoints", epSlice.Namespace, epSlice.Name)
		return
	}

	klog.V(4).Infof("deleteEndpointslice %s/%s", epSlice.Namespace, epSlice.Name)
	startTime := time.Now()
	g.graph.DeleteEndpointslice(epSlice.Name, epSlice.Namespace)
	klog.V(5).Infof("deleteEndpointslice %s/%s completed in %v", epSlice.Namespace, epSlice.Name, time.Since(startTime))
}
