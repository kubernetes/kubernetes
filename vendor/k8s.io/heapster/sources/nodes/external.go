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
	"encoding/json"
	"fmt"
	"io/ioutil"
	"os"

	"github.com/golang/glog"
	"k8s.io/heapster/sources/api"
)

// Provides a list of external cadvisor nodes to monitor.
type externalCadvisorNodes struct {
	// File containing list of hosts to monitor.
	hostsFile string

	// Whether to run in "standalone" mode and only consider localhost.
	standalone bool

	nodes *NodeList
}

func (self *externalCadvisorNodes) List() (*NodeList, error) {
	// Standalone means only localhost.
	if self.standalone {
		self.nodes = newNodeList()
		const localhostIP = "127.0.0.1"
		self.nodes.Items["localhost"] = Info{
			PublicIP:   localhostIP,
			InternalIP: localhostIP,
		}
		return self.nodes, nil
	}

	fi, err := os.Stat(self.hostsFile)
	if err != nil {
		return nil, fmt.Errorf("cannot stat file %q: %s", self.hostsFile, err)
	}
	if fi.Size() == 0 {
		return &NodeList{}, nil
	}
	contents, err := ioutil.ReadFile(self.hostsFile)
	if err != nil {
		return nil, err
	}
	var externalNodes api.ExternalNodeList
	err = json.Unmarshal(contents, &externalNodes)
	if err != nil {
		return nil, fmt.Errorf("failed to unmarshal contents of file %s. Error: %s", self.hostsFile, err)
	}
	nodes := newNodeList()
	for _, node := range externalNodes.Items {
		nodes.Items[Host(node.Name)] = Info{PublicIP: node.IP, InternalIP: node.IP}
	}
	glog.V(5).Infof("Using cAdvisor hosts %+v", nodes)
	self.nodes = nodes
	return nodes, nil
}

func (self *externalCadvisorNodes) DebugInfo() string {
	output := "External Nodes plugin:"
	if self.standalone {
		output = fmt.Sprintf(" Running in standalone mode.\n")
	}
	if self.nodes != nil {
		output = fmt.Sprintf("%s hosts are\n %v", output, self.nodes.Items)
	}
	return output
}

func NewExternalNodes(standaloneMode bool, hostsFile string) (NodesApi, error) {
	if standaloneMode {
		glog.Infof("Running in standalone mode, external nodes source will only use localhost")
	} else {
		_, err := os.Stat(hostsFile)
		if err != nil {
			return nil, fmt.Errorf("cannot stat file %q: %s", hostsFile, err)
		}
	}

	return &externalCadvisorNodes{
		hostsFile:  hostsFile,
		standalone: standaloneMode,
		nodes:      nil,
	}, nil
}
