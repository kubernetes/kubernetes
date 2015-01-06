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
	"os"
	"sync"
	"time"
	"encoding/json"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/labels"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/tools"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util/exec"
	"github.com/golang/glog"
)

type podNameIP struct {
	PodName string
	PodIP	string
	ExtIP	string
	NewPod	int
}

// The onramp router structure
type Onramp struct {
	etcdClient tools.EtcdClient
	kubeClient *client.Client
	extInterface string
	intInterface string
	extAddrs util.StringList
	usedExtAddrs util.StringList
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
}

// Helper structs to allow use of foreign types
type Container api.Container

func NewOnramp(ec tools.EtcdClient, ac *client.Client, eintf string, iintf string, addrs util.StringList) *Onramp {
	return &Onramp{
		etcdClient: ec,
		kubeClient: ac,
		extInterface: eintf,
		intInterface: iintf,
		extAddrs: addrs,
		podNameLock: &sync.Mutex{},
		podMonitor: 0,
	}
}

func (onrmp *Onramp) claimExternalName(podRef podNameIP) (bool) {
	// Need to put code here to mark this router as being the owner of this external path
	newClaim := &OnrampRouterClaim{ ExternalIP: podRef.ExtIP, InternalIP: podRef.PodIP}
	var key string = "/onramp/" + podRef.PodName
	claimjson, cerr := json.Marshal(newClaim)

	if (cerr != nil) {
		glog.Infof("Could not marshall new Claim: %s\n", cerr);
		return false
	}
	_, err := onrmp.etcdClient.Create(key, string(claimjson), 500)
	if (err != nil) {
		glog.Infof("Error claiming pod: %s\n", err)
		return false
	}
	return true
}

func (onrmp *Onramp) freeExternalName(podRef podNameIP) (bool) {
	var key string = "/onramp/" + podRef.PodName
	_, err := onrmp.etcdClient.Delete(key, false)
	if (err != nil) {
		glog.Infof("Error removing external claim: %s\n", err)
		return false
	}
	return true
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

		if (onrmp.claimExternalName(newPod) == false) {

		}

		onrmp.podNameLock.Lock()
		onrmp.podNameList = append(onrmp.podNameList, newPod)
		onrmp.podNameLock.Unlock()
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

func (onrmp *Onramp) allocateExtAddr() (extAddr *string) {
	if (len(onrmp.extAddrs) == 0) {
		glog.Infof("No external Addresses to use, skipping\n")
		return nil
	}
	glog.Infof("Getting New External Address\n")
	var newIP *string = &onrmp.extAddrs[0]
	onrmp.extAddrs = onrmp.extAddrs[1:len(onrmp.extAddrs)]
	onrmp.usedExtAddrs = append(onrmp.usedExtAddrs, *newIP)
	glog.Infof("Using IP address %s\n", *newIP)
	return newIP 
}

func (onrmp *Onramp) freeExtAddr(extAddr string) {
	//remove the extAddr from the usedExtAddrs list and add it to the extAddrs list
	for m := range onrmp.usedExtAddrs {
		if (onrmp.usedExtAddrs[m] == extAddr) {
			onrmp.usedExtAddrs = append(onrmp.usedExtAddrs[0:m], onrmp.usedExtAddrs[m+1:]...)
			onrmp.extAddrs = append(onrmp.extAddrs, extAddr)
			return
		}
	}
}

func (onrmp *Onramp) mapAction(action string, eintf string, iintf string, name string, extip string, podip string) (error) {

	ex := exec.New()
	cmd := ex.Command("onramp_iptables_setup.sh", action, eintf, iintf, name, extip, podip)
	_, err := cmd.CombinedOutput()

	return err
}


func (onrmp *Onramp) monitorPods() {

	ns := api.NamespaceAll
	pdi := onrmp.kubeClient.Pods(ns)
	glog.Infof("Starting to monitor pods\n")
	for {
		glog.Infof("Sleeping\n")
		time.Sleep(10 * time.Second)
		glog.Infof("Continuing monitor\n")
		onrmp.podNameLock.Lock()
		if (len(onrmp.podNameList) == 0) {
			onrmp.podMonitor = 0
			onrmp.podNameLock.Unlock()
			return
		}
		for m := range onrmp.podNameList {
			pod, err := pdi.Get(onrmp.podNameList[m].PodName)
			if (err != nil) {
				glog.Infof("Could not get Pod for %s, deleting\n", onrmp.podNameList[m].PodName)
				err := onrmp.mapAction("DELETE", onrmp.extInterface, onrmp.intInterface, onrmp.podNameList[m].PodName, "NONE", "NONE")
				if (err != nil) {
					glog.Infof("Error Executing Delete for pod %s: %s\n", onrmp.podNameList[m].PodName, err)
					continue
				}
				onrmp.freeExtAddr(onrmp.podNameList[m].ExtIP)
				onrmp.freeExternalName(onrmp.podNameList[m])
				onrmp.podNameList = append(onrmp.podNameList[0:m], onrmp.podNameList[m+1:]...)
				break; //Need to break here to restart the for loop with a new range computation		
			}

			glog.Infof("Checking pod %s\n", onrmp.podNameList[m].PodName)
			if (onrmp.podNameList[m].NewPod == 1) {
				var extIP *string = onrmp.allocateExtAddr()
				if (extIP == nil) {
					continue;
				}
				onrmp.podNameList[m].ExtIP = *extIP

				glog.Infof("Pod %s has been added\n", onrmp.podNameList[m].PodName)
				onrmp.podNameList[m].PodIP = pod.CurrentState.PodIP
				onrmp.podNameList[m].NewPod = 0
				err := onrmp.mapAction("ADD", onrmp.extInterface, onrmp.intInterface, pod.Name, *extIP, pod.CurrentState.PodIP)
				if (err != nil) {
					glog.Infof("Error Executing Creation of New IP Tables rules for pod %s: %s\n", onrmp.podNameList[m].PodName, err)
					continue
				}

			}

			if (pod.CurrentState.PodIP != onrmp.podNameList[m].PodIP) {
				glog.Infof("Pod %s has changed ip %s => %s, updating\n", onrmp.podNameList[m].PodName, onrmp.podNameList[m].PodIP, pod.CurrentState.PodIP)
				onrmp.podNameList[m].PodIP = pod.CurrentState.PodIP
				err := onrmp.mapAction("MODIFY", onrmp.extInterface, onrmp.intInterface, pod.Name, onrmp.podNameList[m].ExtIP, pod.CurrentState.PodIP)
				if (err != nil) {
					glog.Infof("Error Executing Modification of New IP Tables rules for pod %s: %s\n", onrmp.podNameList[m].PodName, err)
					continue
				}
			}
		}
		onrmp.podNameLock.Unlock()
	}
}

// Run starts the kubelet reacting to config updates
func (onrmp *Onramp) Run() {

	// Before we do anything else, the internal and external interfaces need to be set to
	// forward
	file, err := os.OpenFile("/proc/sys/net/ipv4/conf/" + onrmp.extInterface + "/forwarding", os.O_RDWR, 0666)
	if (err != nil) {
		glog.Infof("Could not find interface %s: \n", onrmp.extInterface, err)
		return
	}
	file.WriteString("1")
	file.Close()

	file, err = os.OpenFile("/proc/sys/net/ipv4/conf/" + onrmp.intInterface + "/forwarding", os.O_RDWR, 0666)

	if (err != nil) {
		glog.Infof("Could not find interface %s: \n", onrmp.intInterface, err)
		return
	}

	file.WriteString("1")
	file.Close()

	// Put our iptables nat table in a known, empty state
	onrmp.mapAction("RESET", "X", "X", "X", "X", "X")

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
