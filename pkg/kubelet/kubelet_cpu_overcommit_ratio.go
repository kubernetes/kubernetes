/*
Copyright 2018 The Kubernetes Authors.

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

package kubelet

import (
	"k8s.io/api/core/v1"
	"k8s.io/kubernetes/pkg/kubelet/cm"
	"k8s.io/kubernetes/pkg/kubelet/lifecycle"
	"k8s.io/kubernetes/pkg/kubelet/util"
)

func (kl *Kubelet) cpuOvercommitRatioGetter() func() float64 {
	return func() float64 {
		node, err := kl.nodeInfo.GetNodeInfo(string(kl.nodeName))
		if err != nil {
			return 1.0
		}
		return util.GetCPUOvercommitRatio(node)
	}
}

// Used to update pod cgroup cpu.shares when node cpu-overcommit-ratio is
// changed.
// func (kl *Kubelet) syncPodOnNodeUpdate() {
// if kl.kubeClient == nil {
// glog.Infof("skipped if kube client it not configured")
// return
// }
// fieldSelector := fields.Set{api.ObjectNameField: string(kl.nodeName)}.AsSelector()
// nodeLW := cache.NewListWatchFromClient(kl.kubeClient.CoreV1().RESTClient(), "nodes", metav1.NamespaceAll, fieldSelector)
// sync := func() {
// pods := kl.getPodsToSync()
// pcm := kl.containerManager.NewPodContainerManager()
// glog.V(4).Infof("[k8s.qiniu.com/cpu-overcommit-ratio]: %d pods to sync cgroups", len(pods))
// for _, pod := range pods {
// err := pcm.Update(pod)
// if err != nil {
// glog.Errorf("failed to update cgroups for pod %s/%s", pod.Namespace, pod.Name)
// }
// }
// }
// _, ctrl := cache.NewInformer(nodeLW, &v1.Node{}, 1*time.Minute, &cache.ResourceEventHandlerFuncs{
// AddFunc: func(obj interface{}) {
// sync()
// },
// UpdateFunc: func(old interface{}, obj interface{}) {
// oldnode, ok := old.(*v1.Node)
// if !ok {
// return
// }
// node, ok := obj.(*v1.Node)
// if !ok {
// return
// }
// oldratio := util.GetCPUOvercommitRatio(oldnode)
// newratio := util.GetCPUOvercommitRatio(node)
// if newratio != oldratio {
// glog.V(4).Infof("[k8s.qiniu.com/cpu-overcommit-ratio]: ratio of node %s changed from %v to %v", node.Name, oldratio, newratio)
// }
// // sync anyway for safety
// // TODO sync when needed
// sync()
// },
// })
// go ctrl.Run(wait.NeverStop)
// }

type PodSyncLoopHandler interface {
	// ShouldSync returns true if the pod needs to be synced.
	// This operation must return immediately as its called for each pod.
	// The provided pod should never be modified.
	ShouldSync(pod *v1.Pod) bool
}

type syncPodCgroupConfigsHandler struct {
	pcm cm.PodContainerManager
}

func (*syncPodCgroupConfigsHandler) ShouldSync(pod *v1.Pod) bool {
	// TODO sync pod on node update
	return false
}

func (kl *Kubelet) syncPodCgroupConfigsHandler() lifecycle.PodSyncLoopHandler {
	return &syncPodCgroupConfigsHandler{pcm: kl.containerManager.NewPodContainerManager()}
}
