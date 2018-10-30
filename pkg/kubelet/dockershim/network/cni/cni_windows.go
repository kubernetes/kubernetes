// +build windows

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

package cni

import (
	"fmt"
	"github.com/Microsoft/hcsshim"
	"github.com/golang/glog"
	"strings"

	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
	"k8s.io/kubernetes/pkg/kubelet/dockershim/network"
)

func getLoNetwork(binDirs []string) *cniNetwork {
	return nil
}

func (plugin *cniNetworkPlugin) platformInit() error {
	return nil
}

// GetPodNetworkStatus : Assuming addToNetwork is idempotent, we can call this API as many times as required to get the IPAddress
func (plugin *cniNetworkPlugin) GetPodNetworkStatus(namespace string, name string, id kubecontainer.ContainerID) (*network.PodNetworkStatus, error) {
	eps, err := hcsshim.HNSListEndpointRequest()
	if err != nil {
		return nil, err
	}

	for _, ep := range eps {
		glog.V(5).Infof("GetPodNetworkStatus examining endpoint %v, %v", ep.Name, ep.IPAddress)
		if ep.IsRemoteEndpoint {
			continue
		}
		if strings.Contains(ep.Name, id.ID) {
			glog.V(5).Infof("GetPodNetworkStatus Found matching endpoint %v, %v", ep.Name, ep.IPAddress)
			return &network.PodNetworkStatus{IP: ep.IPAddress}, nil
		}
	}

	return nil, fmt.Errorf("GetPodNetworkStatus: Failed to look up endpoint for %s / %s / %s", namespace, name, id)
}
