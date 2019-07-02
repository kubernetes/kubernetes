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
	"fmt"
	"k8s.io/klog"

	corev1 "k8s.io/api/core/v1"
	storagev1 "k8s.io/api/storage/v1"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	corev1informers "k8s.io/client-go/informers/core/v1"
	storageinformers "k8s.io/client-go/informers/storage/v1"
	"k8s.io/client-go/tools/cache"
	"k8s.io/kubernetes/pkg/features"
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
) {
	g := &graphPopulator{
		graph: graph,
	}

	if utilfeature.DefaultFeatureGate.Enabled(features.DynamicKubeletConfig) {
		nodes.Informer().AddEventHandler(cache.ResourceEventHandlerFuncs{
			AddFunc:    g.addNode,
			UpdateFunc: g.updateNode,
			DeleteFunc: g.deleteNode,
		})
	}

	pods.Informer().AddEventHandler(cache.ResourceEventHandlerFuncs{
		AddFunc:    g.addPod,
		UpdateFunc: g.updatePod,
		DeleteFunc: g.deletePod,
	})

	pvs.Informer().AddEventHandler(cache.ResourceEventHandlerFuncs{
		AddFunc:    g.addPV,
		UpdateFunc: g.updatePV,
		DeleteFunc: g.deletePV,
	})

	attachments.Informer().AddEventHandler(cache.ResourceEventHandlerFuncs{
		AddFunc:    g.addVolumeAttachment,
		UpdateFunc: g.updateVolumeAttachment,
		DeleteFunc: g.deleteVolumeAttachment,
	})
}

func (g *graphPopulator) addNode(obj interface{}) {
	g.updateNode(nil, obj)
}

func (g *graphPopulator) updateNode(oldObj, obj interface{}) {
	node := obj.(*corev1.Node)
	var oldNode *corev1.Node
	if oldObj != nil {
		oldNode = oldObj.(*corev1.Node)
	}

	// we only set up rules for ConfigMap today, because that is the only reference type

	var name, namespace string
	if source := node.Spec.ConfigSource; source != nil && source.ConfigMap != nil {
		name = source.ConfigMap.Name
		namespace = source.ConfigMap.Namespace
	}

	var oldName, oldNamespace string
	if oldNode != nil {
		if oldSource := oldNode.Spec.ConfigSource; oldSource != nil && oldSource.ConfigMap != nil {
			oldName = oldSource.ConfigMap.Name
			oldNamespace = oldSource.ConfigMap.Namespace
		}
	}

	// if Node.Spec.ConfigSource wasn't updated, nothing for us to do
	if name == oldName && namespace == oldNamespace {
		return
	}

	path := "nil"
	if node.Spec.ConfigSource != nil {
		path = fmt.Sprintf("%s/%s", namespace, name)
	}
	klog.V(4).Infof("updateNode configSource reference to %s for node %s", path, node.Name)
	g.graph.SetNodeConfigMap(node.Name, name, namespace)
}

func (g *graphPopulator) deleteNode(obj interface{}) {
	if tombstone, ok := obj.(cache.DeletedFinalStateUnknown); ok {
		obj = tombstone.Obj
	}
	node, ok := obj.(*corev1.Node)
	if !ok {
		klog.Infof("unexpected type %T", obj)
		return
	}

	// NOTE: We don't remove the node, because if the node is re-created not all pod -> node
	// links are re-established (we don't get relevant events because the no mutations need
	// to happen in the API; the state is already there).
	g.graph.SetNodeConfigMap(node.Name, "", "")
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
		if (pod.Spec.NodeName == oldPod.Spec.NodeName) && (pod.UID == oldPod.UID) {
			// Node and uid are unchanged, all object references in the pod spec are immutable
			klog.V(5).Infof("updatePod %s/%s, node unchanged", pod.Namespace, pod.Name)
			return
		}
	}
	klog.V(4).Infof("updatePod %s/%s for node %s", pod.Namespace, pod.Name, pod.Spec.NodeName)
	g.graph.AddPod(pod)
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
	g.graph.DeletePod(pod.Name, pod.Namespace)
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
