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
	"fmt"
	"net"

	"k8s.io/apimachinery/pkg/types"
	utilnet "k8s.io/apimachinery/pkg/util/net"
	"k8s.io/apimachinery/pkg/util/sets"
	api "k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/kubernetes/pkg/apis/core/helper"
	"k8s.io/kubernetes/pkg/proxy/apis/kubeproxyconfig"
	"k8s.io/kubernetes/pkg/util/conntrack"

	"github.com/golang/glog"
)

const (
	IPv4ZeroCIDR = "0.0.0.0/0"
	IPv6ZeroCIDR = "::/0"
)

func IsZeroCIDR(cidr string) bool {
	if cidr == IPv4ZeroCIDR || cidr == IPv6ZeroCIDR {
		return true
	}
	return false
}

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

// GetNodeAddresses list all match node IP addresses based on given cidr list.  Inject NetworkInterface for test purpose.
// We expect the cidr list passed in is already validated.
// Given `default-route`, it will the IP of host interface having default route.
// Given an empty input `[]`, it will return `0.0.0.0/0` and `::/0` directly.
func GetNodeAddresses(cidrs []string, nw NetworkInterface) (sets.String, error) {
	uniqueAddressList := sets.NewString()
	if len(cidrs) == 0 {
		uniqueAddressList.Insert(IPv4ZeroCIDR)
		uniqueAddressList.Insert(IPv6ZeroCIDR)
		return uniqueAddressList, nil
	}
	// First round of iteration to pick out `0.0.0.0/0` or `::/0` for the sake of excluding non-zero IPs.
	for _, cidr := range cidrs {
		if IsZeroCIDR(cidr) {
			uniqueAddressList.Insert(cidr)
		}
	}
	// Second round of iteration to parse IPs based on cidr.
	for _, cidr := range cidrs {
		if IsZeroCIDR(cidr) {
			continue
		}
		if cidr == string(kubeproxyconfig.DefaultRoute) {
			hostIP, err := utilnet.ChooseHostInterface()
			if err != nil {
				return nil, fmt.Errorf("error selecting a host interface having default route, error: %v", err)
			}
			if conntrack.IsIPv6(hostIP) && !uniqueAddressList.Has(IPv6ZeroCIDR) {
				uniqueAddressList.Insert(hostIP.String())
			}
			if !conntrack.IsIPv6(hostIP) && !uniqueAddressList.Has(IPv4ZeroCIDR) {
				uniqueAddressList.Insert(hostIP.String())
			}
			continue
		}
		_, ipNet, _ := net.ParseCIDR(cidr)
		itfs, err := nw.Interfaces()
		if err != nil {
			return nil, fmt.Errorf("error listing all interfaces from host, error: %v", err)
		}
		for _, itf := range itfs {
			addrs, err := nw.Addrs(&itf)
			if err != nil {
				return nil, fmt.Errorf("error getting address from interface %s, error: %v", itf.Name, err)
			}
			for _, addr := range addrs {
				if addr == nil {
					continue
				}
				ip, _, err := net.ParseCIDR(addr.String())
				if err != nil {
					return nil, fmt.Errorf("error parsing CIDR for interface %s, error: %v", itf.Name, err)
				}
				if ipNet.Contains(ip) {
					if conntrack.IsIPv6(ip) && !uniqueAddressList.Has(IPv6ZeroCIDR) {
						uniqueAddressList.Insert(ip.String())
					}
					if !conntrack.IsIPv6(ip) && !uniqueAddressList.Has(IPv4ZeroCIDR) {
						uniqueAddressList.Insert(ip.String())
					}
				}
			}
		}
	}
	return uniqueAddressList, nil
}
