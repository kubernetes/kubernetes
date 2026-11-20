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
	"context"

	"k8s.io/api/core/v1"
	"k8s.io/klog/v2"
	kubeproxyconfig "k8s.io/kubernetes/pkg/proxy/apis/config"
	"k8s.io/kubernetes/pkg/proxy/nodemanager"

	netutils "k8s.io/utils/net"
)

// LocalTrafficDetector generates iptables or nftables rules to detect traffic from local pods.
type LocalTrafficDetector interface {
	// IsImplemented returns true if the implementation does something, false
	// otherwise. You should not call the other methods if IsImplemented() returns
	// false.
	IsImplemented() bool

	// IfLocal returns iptables arguments that will match traffic from a local pod.
	IfLocal() []string

	// IfNotLocal returns iptables arguments that will match traffic that is not from
	// a local pod.
	IfNotLocal() []string

	// IfLocalNFT returns nftables arguments that will match traffic from a local pod.
	IfLocalNFT() []string

	// IfNotLocalNFT returns nftables arguments that will match traffic that is not
	// from a local pod.
	IfNotLocalNFT() []string
}

func GetLocalTrafficDetectors(ctx context.Context, config *kubeproxyconfig.KubeProxyConfiguration, nodeManager nodemanager.NodeManager) map[v1.IPFamily]LocalTrafficDetector {
	logger := klog.FromContext(ctx)

	primaryIPFamily := nodeManager.PrimaryIPFamily()
	nodePodCIDRs := nodeManager.PodCIDRs()

	localDetectors := map[v1.IPFamily]LocalTrafficDetector{
		v1.IPv4Protocol: NewNoOpLocalDetector(),
		v1.IPv6Protocol: NewNoOpLocalDetector(),
	}

	switch config.DetectLocalMode {
	case kubeproxyconfig.LocalModeClusterCIDR:
		for family, cidrs := range MapCIDRsByIPFamily(config.DetectLocal.ClusterCIDRs) {
			localDetectors[family] = NewDetectLocalByCIDR(cidrs[0].String())
		}
		if !localDetectors[primaryIPFamily].IsImplemented() {
			logger.Info("Detect-local-mode set to ClusterCIDR, but no cluster CIDR specified for primary IP family", "ipFamily", primaryIPFamily, "clusterCIDRs", config.DetectLocal.ClusterCIDRs)
		}

	case kubeproxyconfig.LocalModeNodeCIDR:
		for family, cidrs := range MapCIDRsByIPFamily(nodePodCIDRs) {
			localDetectors[family] = NewDetectLocalByCIDR(cidrs[0].String())
		}
		if !localDetectors[primaryIPFamily].IsImplemented() {
			logger.Info("Detect-local-mode set to NodeCIDR, but no PodCIDR defined at node for primary IP family", "ipFamily", primaryIPFamily, "podCIDRs", nodePodCIDRs)
		}

	case kubeproxyconfig.LocalModeBridgeInterface:
		localDetector := NewDetectLocalByBridgeInterface(config.DetectLocal.BridgeInterface)
		localDetectors[v1.IPv4Protocol] = localDetector
		localDetectors[v1.IPv6Protocol] = localDetector

	case kubeproxyconfig.LocalModeInterfaceNamePrefix:
		localDetector := NewDetectLocalByInterfaceNamePrefix(config.DetectLocal.InterfaceNamePrefix)
		localDetectors[v1.IPv4Protocol] = localDetector
		localDetectors[v1.IPv6Protocol] = localDetector

	default:
		logger.Info("Defaulting to no-op detect-local")
	}

	return localDetectors
}

type detectLocal struct {
	ifLocal       []string
	ifNotLocal    []string
	ifLocalNFT    []string
	ifNotLocalNFT []string
}

func (d *detectLocal) IsImplemented() bool {
	return len(d.ifLocal) > 0
}

func (d *detectLocal) IfLocal() []string {
	return d.ifLocal
}

func (d *detectLocal) IfNotLocal() []string {
	return d.ifNotLocal
}

func (d *detectLocal) IfLocalNFT() []string {
	return d.ifLocalNFT
}

func (d *detectLocal) IfNotLocalNFT() []string {
	return d.ifNotLocalNFT
}

// NewNoOpLocalDetector returns a no-op implementation of LocalTrafficDetector.
func NewNoOpLocalDetector() LocalTrafficDetector {
	return &detectLocal{}
}

// NewDetectLocalByCIDR returns a LocalTrafficDetector that considers traffic from the
// provided cidr to be from a local pod, and other traffic to be non-local. cidr is
// assumed to be valid.
func NewDetectLocalByCIDR(cidr string) LocalTrafficDetector {
	nftFamily := "ip"
	if netutils.IsIPv6CIDRString(cidr) {
		nftFamily = "ip6"
	}

	return &detectLocal{
		ifLocal:       []string{"-s", cidr},
		ifNotLocal:    []string{"!", "-s", cidr},
		ifLocalNFT:    []string{nftFamily, "saddr", cidr},
		ifNotLocalNFT: []string{nftFamily, "saddr", "!=", cidr},
	}
}

// NewDetectLocalByBridgeInterface returns a LocalTrafficDetector that considers traffic
// from interfaceName to be from a local pod, and traffic from other interfaces to be
// non-local.
func NewDetectLocalByBridgeInterface(interfaceName string) LocalTrafficDetector {
	return &detectLocal{
		ifLocal:       []string{"-i", interfaceName},
		ifNotLocal:    []string{"!", "-i", interfaceName},
		ifLocalNFT:    []string{"iifname", interfaceName},
		ifNotLocalNFT: []string{"iifname", "!=", interfaceName},
	}
}

// NewDetectLocalByInterfaceNamePrefix returns a LocalTrafficDetector that considers
// traffic from interfaces starting with interfacePrefix to be from a local pod, and
// traffic from other interfaces to be non-local.
func NewDetectLocalByInterfaceNamePrefix(interfacePrefix string) LocalTrafficDetector {
	return &detectLocal{
		ifLocal:       []string{"-i", interfacePrefix + "+"},
		ifNotLocal:    []string{"!", "-i", interfacePrefix + "+"},
		ifLocalNFT:    []string{"iifname", interfacePrefix + "*"},
		ifNotLocalNFT: []string{"iifname", "!=", interfacePrefix + "*"},
	}
}
