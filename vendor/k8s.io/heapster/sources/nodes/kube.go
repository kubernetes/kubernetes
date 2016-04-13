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

package nodes

import (
	"fmt"
	"net"
	"sync"
	"time"

	"github.com/golang/glog"
	api "k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/client/cache"
	client "k8s.io/kubernetes/pkg/client/unversioned"
	"k8s.io/kubernetes/pkg/fields"
)

type kubeNodes struct {
	client *client.Client
	// a means to list all minions
	nodeLister *cache.StoreToNodeLister
	reflector  *cache.Reflector
	// Used to stop the existing reflector.
	stopChan   chan struct{}
	goodNodes  []string       // guarded by stateLock
	nodeErrors map[string]int // guarded by stateLock
	stateLock  sync.RWMutex
}

func (self *kubeNodes) recordNodeError(name string) {
	self.stateLock.Lock()
	defer self.stateLock.Unlock()

	self.nodeErrors[name]++
}

func (self *kubeNodes) recordGoodNodes(nodes []string) {
	self.stateLock.Lock()
	defer self.stateLock.Unlock()

	self.goodNodes = nodes
}

func (self *kubeNodes) getNodeInfoAndHostname(node api.Node) (Info, string, error) {
	nodeInfo := Info{}
	hostname := ""
	var nodeErr error
	for _, addr := range node.Status.Addresses {
		switch addr.Type {
		case api.NodeExternalIP:
			nodeInfo.PublicIP = addr.Address
		case api.NodeLegacyHostIP:
			nodeInfo.PublicIP = addr.Address
		case api.NodeInternalIP:
			nodeInfo.InternalIP = addr.Address
		case api.NodeHostName:
			hostname = addr.Address
		}
	}
	if hostname == "" {
		hostname = node.Name
	}
	if nodeInfo.InternalIP == "" {
		if hostname == nodeInfo.PublicIP {
			// If the only identifier we have for the node is a public IP, then use it;
			// don't force a DNS lookup
			glog.V(4).Infof("Only have PublicIP %s for node %s, so using it for InternalIP",
				nodeInfo.PublicIP, node.Name)
			nodeInfo.InternalIP = nodeInfo.PublicIP
		} else {
			addrs, err := net.LookupIP(hostname)
			if err == nil {
				nodeInfo.InternalIP = addrs[0].String()
			} else {
				glog.Errorf("Skipping host %s since looking up its IP failed - %s", node.Name, err)
				self.recordNodeError(node.Name)
				nodeErr = err
			}
		}
	}
	if node.Spec.ExternalID != "" {
		nodeInfo.ExternalID = node.Spec.ExternalID
	}
	cpu := node.Status.Capacity[api.ResourceCPU]
	mem := node.Status.Capacity[api.ResourceMemory]
	nodeInfo.CpuCapacity = uint64(cpu.MilliValue())
	nodeInfo.MemCapacity = uint64(mem.Value())

	for _, condition := range node.Status.Conditions {
		if condition.Type == api.NodeReady {
			if condition.Status != api.ConditionTrue {
				nodeErr = fmt.Errorf("The state of node is not Ready!")
			}
		}
	}

	return nodeInfo, hostname, nodeErr
}

func (self *kubeNodes) List() (*NodeList, error) {
	nodeList := newNodeList()
	allNodes, err := self.nodeLister.List()
	if err != nil {
		glog.Errorf("failed to list minions via watch interface - %v", err)
		return nil, fmt.Errorf("failed to list minions via watch interface - %v", err)
	}
	glog.V(5).Infof("all kube nodes: %+v", allNodes)

	goodNodes := []string{}
	for _, node := range allNodes.Items {
		nodeInfo, hostname, err := self.getNodeInfoAndHostname(node)
		if err == nil {
			nodeList.Items[Host(hostname)] = nodeInfo
			goodNodes = append(goodNodes, node.Name)
		}
	}
	self.recordGoodNodes(goodNodes)
	glog.V(5).Infof("kube nodes found: %+v", nodeList)
	return nodeList, nil
}

func (self *kubeNodes) getState() string {
	self.stateLock.RLock()
	defer self.stateLock.RUnlock()

	state := "\tHealthy Nodes:\n"
	for _, node := range self.goodNodes {
		state += fmt.Sprintf("\t\t%s\n", node)
	}
	if len(self.nodeErrors) > 0 {
		state += fmt.Sprintf("\tNode Errors: %+v\n", self.nodeErrors)
	} else {
		state += "\tNo node errors\n"
	}
	return state
}

func (self *kubeNodes) DebugInfo() string {
	desc := "Kubernetes Nodes plugin: \n"
	desc += self.getState()
	desc += "\n"

	return desc
}

func NewKubeNodes(client *client.Client) (NodesApi, error) {
	if client == nil {
		return nil, fmt.Errorf("client is nil")
	}

	lw := cache.NewListWatchFromClient(client, "nodes", api.NamespaceAll, fields.Everything())
	nodeLister := &cache.StoreToNodeLister{Store: cache.NewStore(cache.MetaNamespaceKeyFunc)}
	reflector := cache.NewReflector(lw, &api.Node{}, nodeLister.Store, time.Hour)
	stopChan := make(chan struct{})
	reflector.RunUntil(stopChan)

	return &kubeNodes{
		client:     client,
		nodeLister: nodeLister,
		reflector:  reflector,
		stopChan:   stopChan,
		nodeErrors: make(map[string]int),
	}, nil
}
