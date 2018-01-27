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

package util

import (
	"net"

	"k8s.io/apimachinery/pkg/types"
	api "k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/kubernetes/pkg/apis/core/helper"

	"github.com/golang/glog"
)

func IsLocalIP(ip string) (bool, error) {
	addrs, err := net.InterfaceAddrs()
	if err != nil {
		return false, err
	}
	for i := range addrs {
		intf, _, err := net.ParseCIDR(addrs[i].String())
		if err != nil {
			return false, err
		}
		if net.ParseIP(ip).Equal(intf) {
			return true, nil
		}
	}
	return false, nil
}

func ShouldSkipService(svcName types.NamespacedName, service *api.Service) bool {
	// if ClusterIP is "None" or empty, skip proxying
	if !helper.IsServiceIPSet(service) {
		glog.V(3).Infof("Skipping service %s due to clusterIP = %q", svcName, service.Spec.ClusterIP)
		return true
	}
	// Even if ClusterIP is set, ServiceTypeExternalName services don't get proxied
	if service.Spec.Type == api.ServiceTypeExternalName {
		glog.V(3).Infof("Skipping service %s due to Type=ExternalName", svcName)
		return true
	}
	return false
}

// SameTopologyLevel tells if two given labels have the same topology value.
func SameTopologyDomain(labels1, labels2 map[string]string, topologyKey, topologyMode string) bool {
	// return false directly if topology mode is "ignored".
	if topologyMode == string(api.TopologyModeIgnored) {
		return false
	}
	if topologyKey == "" {
		return false
	}
	if labels1 == nil || labels2 == nil {
		return false
	}
	value1, ok1 := labels1[topologyKey]
	value2, ok2 := labels2[topologyKey]
	if !ok1 || !ok2 {
		return false
	}
	if value1 != value2 {
		return false
	}
	return true
}
