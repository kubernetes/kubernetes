// Copyright 2014 Google Inc. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package sources

import (
	"fmt"
	"strings"
	"time"

	"github.com/golang/glog"
	"k8s.io/heapster/sources/api"
	"k8s.io/heapster/sources/nodes"
	kapi "k8s.io/kubernetes/pkg/api"
	kcache "k8s.io/kubernetes/pkg/client/cache"
	kclient "k8s.io/kubernetes/pkg/client/unversioned"
	kframework "k8s.io/kubernetes/pkg/controller/framework"
	kSelector "k8s.io/kubernetes/pkg/fields"
	"k8s.io/kubernetes/pkg/labels"
	"k8s.io/kubernetes/pkg/util"
)

// podsApi provides an interface to access all the pods that an instance of heapster
// needs to process.
// TODO(vishh): Add an interface to select specific nodes as part of the Watch.
type podsApi interface {
	// Returns a list of pods that exist on the nodes in 'nodeList'
	List(nodeList *nodes.NodeList) ([]api.Pod, error)

	// Returns debug information.
	DebugInfo() string
}

type realPodsApi struct {
	client *kclient.Client
	// a means to list all scheduled pods
	podLister      *kcache.StoreToPodLister
	namespaceStore kcache.Store
	reflector      *kcache.Reflector
	stopChan       chan struct{}
}

type podNodePair struct {
	pod       *kapi.Pod
	nodeInfo  *nodes.Info
	namespace *kapi.Namespace
}

func (self *realPodsApi) parsePod(podNodePair *podNodePair) *api.Pod {
	pod := podNodePair.pod
	node := podNodePair.nodeInfo
	localPod := api.Pod{
		PodMetadata: api.PodMetadata{
			Name:           pod.Name,
			Namespace:      pod.Namespace,
			NamespaceUID:   string(podNodePair.namespace.UID),
			ID:             string(pod.UID),
			Hostname:       pod.Spec.NodeName,
			HostPublicIP:   pod.Status.HostIP,
			HostInternalIP: node.InternalIP,
			Status:         string(pod.Status.Phase),
			PodIP:          pod.Status.PodIP,
			Labels:         make(map[string]string, 0),
			ExternalID:     node.ExternalID,
		},
		Containers: make([]api.Container, 0),
	}
	for key, value := range pod.Labels {
		localPod.Labels[key] = value
	}
	for _, container := range pod.Spec.Containers {
		localContainer := api.Container{}
		localContainer.Name = container.Name
		localContainer.Image = container.Image
		if cpu, ok := container.Resources.Requests[kapi.ResourceCPU]; ok {
			localContainer.Spec.CpuRequest = cpu.MilliValue()
		}
		if mem, ok := container.Resources.Requests[kapi.ResourceMemory]; ok {
			localContainer.Spec.MemoryRequest = mem.Value()
		}
		localPod.Containers = append(localPod.Containers, localContainer)
	}
	glog.V(5).Infof("parsed kube pod: %+v", localPod)

	return &localPod
}

func (self *realPodsApi) parseAllPods(podNodePairs []podNodePair) []api.Pod {
	out := make([]api.Pod, 0)
	for i := range podNodePairs {
		glog.V(5).Infof("Found kube Pod: %+v", podNodePairs[i].pod)
		out = append(out, *self.parsePod(&podNodePairs[i]))
	}

	return out
}

func (self *realPodsApi) getNodeSelector(nodeList *nodes.NodeList) (labels.Selector, error) {
	nodeLabels := []string{}
	for host := range nodeList.Items {
		nodeLabels = append(nodeLabels, fmt.Sprintf("spec.nodeName==%s", host))
	}
	glog.V(2).Infof("using labels %v to find pods", nodeLabels)
	return labels.Parse(strings.Join(nodeLabels, ","))
}

// Returns a map of minion hostnames to the Pods running in them.
func (self *realPodsApi) List(nodeList *nodes.NodeList) ([]api.Pod, error) {
	pods, err := self.podLister.List(labels.Everything())
	if err != nil {
		return []api.Pod{}, err
	}
	glog.V(5).Infof("got pods from api server %+v", pods)
	selectedPods := []podNodePair{}
	// TODO(vishh): Avoid this loop by setting a node selector on the watcher.
	for i, pod := range pods {
		switch pod.Status.Phase {
		case kapi.PodSucceeded:
		case kapi.PodFailed:
			continue
		}
		if nodeInfo, ok := nodeList.Items[nodes.Host(pod.Spec.NodeName)]; ok {
			nsObj, exists, err := self.namespaceStore.GetByKey(pod.Namespace)
			if err != nil {
				return []api.Pod{}, err
			}
			if !exists {
				glog.V(2).Infof("Ignoring pod %s with namespace %s since namespace object was not found", pod.Name, pod.Namespace)
				continue
			}
			ns, ok := nsObj.(*kapi.Namespace)
			if !ok {
				glog.V(2).Infof("Ignoring pod %s with namespace %s since casting to namespace object failed - %T, %v", pod.Name, pod.Namespace, ns, ns)
				continue
			}
			selectedPods = append(selectedPods, podNodePair{pods[i], &nodeInfo, ns})
		} else {
			glog.V(2).Infof("pod %q with host %q and hostip %q not found in nodeList", pod.Name, pod.Spec.NodeName, pod.Status.HostIP)
		}
	}
	glog.V(4).Infof("selected pods from api server %+v", selectedPods)

	return self.parseAllPods(selectedPods), nil
}

func (self *realPodsApi) DebugInfo() string {
	return ""
}

func createNamespaceLW(kubeClient *kclient.Client) *kcache.ListWatch {
	return kcache.NewListWatchFromClient(kubeClient, "namespaces", kapi.NamespaceAll, kSelector.Everything())
}

const resyncPeriod = time.Minute

func newPodsApi(client *kclient.Client) podsApi {
	// Extend the selector to include specific nodes to monitor
	// or provide an API to update the nodes to monitor.
	selector, err := kSelector.ParseSelector("spec.nodeName!=")
	if err != nil {
		panic(err)
	}

	lw := kcache.NewListWatchFromClient(client, "pods", kapi.NamespaceAll, selector)
	podLister := &kcache.StoreToPodLister{Store: kcache.NewStore(kcache.MetaNamespaceKeyFunc)}
	// Watch and cache all running pods.
	reflector := kcache.NewReflector(lw, &kapi.Pod{}, podLister.Store, time.Hour)
	stopChan := make(chan struct{})
	reflector.RunUntil(stopChan)
	nStore, nController := kframework.NewInformer(
		createNamespaceLW(client),
		&kapi.Namespace{},
		resyncPeriod,
		kframework.ResourceEventHandlerFuncs{})
	go nController.Run(util.NeverStop)

	podsApi := &realPodsApi{
		client:         client,
		podLister:      podLister,
		stopChan:       stopChan,
		reflector:      reflector,
		namespaceStore: nStore,
	}

	return podsApi
}
