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
	"net/http"
	"time"

	etcd "github.com/coreos/etcd/client"
	fleetClient "github.com/coreos/fleet/client"
	fleetPkg "github.com/coreos/fleet/pkg"
	"github.com/coreos/fleet/registry"
	"github.com/golang/glog"
)

const etcdRegistry = "/_coreos.com/fleet/"

type fleetNodes struct {
	client         fleetClient.API
	nodes          *NodeList
	apiErrors      int
	recentApiError error
}

func (self *fleetNodes) List() (*NodeList, error) {
	nodes, err := self.client.Machines()
	if err != nil {
		self.apiErrors++
		self.recentApiError = err
		glog.Errorf("failed to get list of machines from fleet - %q", err)
		return nil, err
	}
	nodeList := newNodeList()
	for _, node := range nodes {
		nodeList.Items[Host(node.ID)] = Info{PublicIP: node.PublicIP, InternalIP: node.PublicIP}
	}
	self.nodes = nodeList
	return nodeList, nil
}

func (self *fleetNodes) DebugInfo() string {
	output := fmt.Sprintf("Fleet Nodes plugin: Aggregate error count: %d; recent error: %v", self.apiErrors, self.recentApiError)
	if self.nodes != nil {
		output = fmt.Sprintf("%s\nCadvisor Nodes: %v", output, self.nodes.Items)
	}
	return output
}

func getFleetRegistryClient(fleetEndpoints []string) (fleetClient.API, error) {
	var dial func(string, string) (net.Conn, error)

	tlsConfig, err := fleetPkg.ReadTLSConfigFiles("", "", "")
	if err != nil {
		return nil, err
	}

	trans := &http.Transport{
		Dial:            dial,
		TLSClientConfig: tlsConfig,
	}

	timeout := 3 * time.Second
	eCfg := etcd.Config{
		Endpoints: fleetEndpoints,
		Transport: trans,
	}
	eClient, err := etcd.New(eCfg)
	if err != nil {
		return nil, err
	}
	kAPI := etcd.NewKeysAPI(eClient)
	reg := registry.NewEtcdRegistry(kAPI, registry.DefaultKeyPrefix, timeout)

	return &fleetClient.RegistryClient{Registry: reg}, nil
}

func NewCoreOSNodes(fleetEndpoints []string) (NodesApi, error) {
	client, err := getFleetRegistryClient(fleetEndpoints)
	if err != nil {
		return nil, fmt.Errorf("failed to get fleet client - %q", err)
	}
	return &fleetNodes{
		client:         client,
		nodes:          nil,
		apiErrors:      0,
		recentApiError: nil,
	}, nil
}
