/*
Copyright 2014 Google Inc. All rights reserved.

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

package onramp

import (
	"sync"
	"time"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/labels"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/tools"
	"github.com/golang/glog"
)

type podNameIP struct {
	PodName string
	PodIP	string
	NewPod	int
}

// The onramp router structure
type Onramp struct {
	etcdClient tools.EtcdClient
	kubeClient *client.Client
	podNameList []podNameIP
	podNameLock *sync.Mutex
	podMonitor	int
}

// The key/value encoding to claim an ExternalName for a container by an onramp
// router.  The key stored on etcd is the ExternalName from the container, and
// the contents of the structure below is the value
type OnrampRouterClaim struct {
	ExternalIP   string   `json:"externalIP" yaml:"externalIP"`
	InternalIP   string   `json:"internalIP" yaml:"internalIP"`
	ContainerIPS []string `json:"containerIPS" yaml:"containerIPS"`
}

// Helper structs to allow use of foreign types
type Container api.Container

func NewOnramp(ec tools.EtcdClient, ac *client.Client) *Onramp {
	return &Onramp{
		etcdClient: ec,
		kubeClient: ac,
		podNameLock: &sync.Mutex{},
		podMonitor: 0,
	}
}

func (onrmp *Onramp) claimExternalName(container Container) {
	// Need to put code here to mark this router as being the owner of this external path
}

func (onrmp *Onramp) scanContainer(podName string, container Container) {
	if container.ExternalName != "" {
		glog.Infof("Container needs external IP access for %s!\n", container.ExternalName)
		ns := api.NamespaceAll
		pdi := onrmp.kubeClient.Pods(ns)

		pod, err := pdi.Get(podName)

		if (err != nil) {
			glog.Infof("Error getting pod: %s\n", err)
			return
		}

		newPod := podNameIP{ PodName: podName, PodIP: pod.CurrentState.PodIP, NewPod: 1}
		onrmp.podNameLock.Lock()
		onrmp.podNameList = append(onrmp.podNameList, newPod)
		onrmp.podNameLock.Unlock()
		onrmp.claimExternalName(container)
	}
}

func (onrmp *Onramp) scanPods() {
	ns := api.NamespaceAll
	pdi := onrmp.kubeClient.Pods(ns)
	if pdi == nil {
		return
	}
	pods, err := pdi.List(labels.Everything())
	if err != nil {
		return
	}
	for m := range pods.Items {
		var containers = pods.Items[m].DesiredState.Manifest.Containers

		for n := range containers {
			onrmp.scanContainer(pods.Items[m].Name, Container(containers[n]))
		}
	}
}

func (onrmp *Onramp) monitorPods() {

	ns := api.NamespaceAll
	pdi := onrmp.kubeClient.Pods(ns)
	glog.Infof("Starting to monitor pods\n")
	for {
		glog.Infof("Sleeping\n")
		time.Sleep(10 * time.Second)
		glog.Infof("Checknig podMonitor\n")
		// Note: This is racy, fix it
		if (onrmp.podMonitor == 0) {
			return
		}	
		glog.Infof("Continuing monitor\n")
		onrmp.podNameLock.Lock()

		for m := range onrmp.podNameList {
			pod, err := pdi.Get(onrmp.podNameList[m].PodName)
			if (err != nil) {
				glog.Infof("Could not get Pod for %s\n", onrmp.podNameList[m].PodName)
				continue;
			}

			glog.Infof("Checking pod %s\n", onrmp.podNameList[m].PodName)
			if (onrmp.podNameList[m].NewPod == 1) {
				glog.Infof("Pod %s has been added\n", onrmp.podNameList[m].PodName)
				onrmp.podNameList[m].PodIP = pod.CurrentState.PodIP
				onrmp.podNameList[m].NewPod = 0
				// Insert callout for add here
			}

			if (pod.CurrentState.PodIP != onrmp.podNameList[m].PodIP) {
				glog.Infof("Pod %s has changed ip %s => %s, updating\n", onrmp.podNameList[m].PodName, onrmp.podNameList[m].PodIP, pod.CurrentState.PodIP)
				onrmp.podNameList[m].PodIP = pod.CurrentState.PodIP
				// Insert callout for update here
			}
		}
		onrmp.podNameLock.Unlock()
	}
}

// Run starts the kubelet reacting to config updates
func (onrmp *Onramp) Run() {

	response, err := onrmp.etcdClient.Watch("/registry/pods/default/", 0, true, nil, nil)
	if err != nil {
		glog.Infof("Error on etcd watch: %s\n", err)
		return
	}

	if response.Action == "create" {
		onrmp.scanPods()
	}

	onrmp.podNameLock.Lock()
	if ((len(onrmp.podNameList) != 0) && (onrmp.podMonitor == 0)){
		onrmp.podMonitor = 1
		go onrmp.monitorPods()
	} else {
		onrmp.podMonitor = 0
	}
	onrmp.podNameLock.Unlock()
}
