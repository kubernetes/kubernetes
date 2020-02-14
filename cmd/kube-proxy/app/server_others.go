// +build !windows

/*
Copyright 2014 The Kubernetes Authors.

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

// Package app does all of the work necessary to configure and run a
// Kubernetes app process.
package app

import (
	"errors"
	"fmt"
	"net"
	"strings"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/types"
	utilnet "k8s.io/apimachinery/pkg/util/net"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/client-go/tools/record"
	"k8s.io/component-base/metrics"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/proxy"
	proxyconfigapi "k8s.io/kubernetes/pkg/proxy/apis/config"
	proxyconfigscheme "k8s.io/kubernetes/pkg/proxy/apis/config/scheme"
	"k8s.io/kubernetes/pkg/proxy/healthcheck"
	"k8s.io/kubernetes/pkg/proxy/iptables"
	"k8s.io/kubernetes/pkg/proxy/ipvs"
	proxymetrics "k8s.io/kubernetes/pkg/proxy/metrics"
	"k8s.io/kubernetes/pkg/proxy/userspace"
	"k8s.io/kubernetes/pkg/util/configz"
	utilipset "k8s.io/kubernetes/pkg/util/ipset"
	utiliptables "k8s.io/kubernetes/pkg/util/iptables"
	utilipvs "k8s.io/kubernetes/pkg/util/ipvs"
	utilnode "k8s.io/kubernetes/pkg/util/node"
	utilsysctl "k8s.io/kubernetes/pkg/util/sysctl"
	"k8s.io/utils/exec"
	utilsnet "k8s.io/utils/net"

	"k8s.io/klog"
)

// NewProxyServer returns a new ProxyServer.
func NewProxyServer(o *Options) (*ProxyServer, error) {
	return newProxyServer(o.config, o.CleanupAndExit, o.master)
}

func newProxyServer(
	config *proxyconfigapi.KubeProxyConfiguration,
	cleanupAndExit bool,
	master string) (*ProxyServer, error) {

	if config == nil {
		return nil, errors.New("config is required")
	}

	if c, err := configz.New(proxyconfigapi.GroupName); err == nil {
		c.Set(config)
	} else {
		return nil, fmt.Errorf("unable to register configz: %s", err)
	}

	protocol := utiliptables.ProtocolIpv4
	if net.ParseIP(config.BindAddress).To4() == nil {
		klog.V(0).Infof("IPv6 bind address (%s), assume IPv6 operation", config.BindAddress)
		protocol = utiliptables.ProtocolIpv6
	}

	var iptInterface utiliptables.Interface
	var ipvsInterface utilipvs.Interface
	var kernelHandler ipvs.KernelHandler
	var ipsetInterface utilipset.Interface

	// Create a iptables utils.
	execer := exec.New()

	iptInterface = utiliptables.New(execer, protocol)
	kernelHandler = ipvs.NewLinuxKernelHandler()
	ipsetInterface = utilipset.New(execer)
	canUseIPVS, _ := ipvs.CanUseIPVSProxier(kernelHandler, ipsetInterface)
	if canUseIPVS {
		ipvsInterface = utilipvs.New(execer)
	}

	// We omit creation of pretty much everything if we run in cleanup mode
	if cleanupAndExit {
		return &ProxyServer{
			execer:         execer,
			IptInterface:   iptInterface,
			IpvsInterface:  ipvsInterface,
			IpsetInterface: ipsetInterface,
		}, nil
	}

	if len(config.ShowHiddenMetricsForVersion) > 0 {
		metrics.SetShowHidden()
	}

	client, eventClient, err := createClients(config.ClientConnection, master)
	if err != nil {
		return nil, err
	}

	// Create event recorder
	hostname, err := utilnode.GetHostname(config.HostnameOverride)
	if err != nil {
		return nil, err
	}
	eventBroadcaster := record.NewBroadcaster()
	recorder := eventBroadcaster.NewRecorder(proxyconfigscheme.Scheme, v1.EventSource{Component: "kube-proxy", Host: hostname})

	nodeRef := &v1.ObjectReference{
		Kind:      "Node",
		Name:      hostname,
		UID:       types.UID(hostname),
		Namespace: "",
	}

	var healthzServer *healthcheck.ProxierHealthServer
	if len(config.HealthzBindAddress) > 0 {
		healthzServer = healthcheck.NewProxierHealthServer(config.HealthzBindAddress, 2*config.IPTables.SyncPeriod.Duration, recorder, nodeRef)
	}

	var proxier proxy.Provider

	proxyMode := getProxyMode(string(config.Mode), kernelHandler, ipsetInterface, iptables.LinuxKernelCompatTester{})
	nodeIP := net.ParseIP(config.BindAddress)
	if nodeIP.IsUnspecified() {
		nodeIP = utilnode.GetNodeIP(client, hostname)
		if nodeIP == nil {
			klog.V(0).Infof("can't determine this node's IP, assuming 127.0.0.1; if this is incorrect, please set the --bind-address flag")
			nodeIP = net.ParseIP("127.0.0.1")
		}
	}
	if proxyMode == proxyModeIPTables {
		klog.V(0).Info("Using iptables Proxier.")
		if config.IPTables.MasqueradeBit == nil {
			// MasqueradeBit must be specified or defaulted.
			return nil, fmt.Errorf("unable to read IPTables MasqueradeBit from config")
		}

		if utilfeature.DefaultFeatureGate.Enabled(features.IPv6DualStack) {
			klog.V(0).Info("creating dualStackProxier for iptables.")

			// Create iptables handlers for both families, one is already created
			// Always ordered as IPv4, IPv6
			var ipt [2]utiliptables.Interface
			if iptInterface.IsIpv6() {
				ipt[1] = iptInterface
				ipt[0] = utiliptables.New(execer, utiliptables.ProtocolIpv4)
			} else {
				ipt[0] = iptInterface
				ipt[1] = utiliptables.New(execer, utiliptables.ProtocolIpv6)
			}

			// TODO this has side effects that should only happen when Run() is invoked.
			proxier, err = iptables.NewDualStackProxier(
				ipt,
				utilsysctl.New(),
				execer,
				config.IPTables.SyncPeriod.Duration,
				config.IPTables.MinSyncPeriod.Duration,
				config.IPTables.MasqueradeAll,
				int(*config.IPTables.MasqueradeBit),
				cidrTuple(config.ClusterCIDR),
				hostname,
				nodeIPTuple(config.BindAddress),
				recorder,
				healthzServer,
				config.NodePortAddresses,
			)
		} else { // Create a single-stack proxier.
			// TODO this has side effects that should only happen when Run() is invoked.
			proxier, err = iptables.NewProxier(
				iptInterface,
				utilsysctl.New(),
				execer,
				config.IPTables.SyncPeriod.Duration,
				config.IPTables.MinSyncPeriod.Duration,
				config.IPTables.MasqueradeAll,
				int(*config.IPTables.MasqueradeBit),
				config.ClusterCIDR,
				hostname,
				nodeIP,
				recorder,
				healthzServer,
				config.NodePortAddresses,
			)
		}

		if err != nil {
			return nil, fmt.Errorf("unable to create proxier: %v", err)
		}
		proxymetrics.RegisterMetrics()
	} else if proxyMode == proxyModeIPVS {
		klog.V(0).Info("Using ipvs Proxier.")
		if utilfeature.DefaultFeatureGate.Enabled(features.IPv6DualStack) {
			klog.V(0).Info("creating dualStackProxier for ipvs.")

			// Create iptables handlers for both families, one is already created
			// Always ordered as IPv4, IPv6
			var ipt [2]utiliptables.Interface
			if iptInterface.IsIpv6() {
				ipt[1] = iptInterface
				ipt[0] = utiliptables.New(execer, utiliptables.ProtocolIpv4)
			} else {
				ipt[0] = iptInterface
				ipt[1] = utiliptables.New(execer, utiliptables.ProtocolIpv6)
			}

			proxier, err = ipvs.NewDualStackProxier(
				ipt,
				ipvsInterface,
				ipsetInterface,
				utilsysctl.New(),
				execer,
				config.IPVS.SyncPeriod.Duration,
				config.IPVS.MinSyncPeriod.Duration,
				config.IPVS.ExcludeCIDRs,
				config.IPVS.StrictARP,
				config.IPVS.TCPTimeout.Duration,
				config.IPVS.TCPFinTimeout.Duration,
				config.IPVS.UDPTimeout.Duration,
				config.IPTables.MasqueradeAll,
				int(*config.IPTables.MasqueradeBit),
				cidrTuple(config.ClusterCIDR),
				hostname,
				nodeIPTuple(config.BindAddress),
				recorder,
				healthzServer,
				config.IPVS.Scheduler,
				config.NodePortAddresses,
			)
		} else {
			proxier, err = ipvs.NewProxier(
				iptInterface,
				ipvsInterface,
				ipsetInterface,
				utilsysctl.New(),
				execer,
				config.IPVS.SyncPeriod.Duration,
				config.IPVS.MinSyncPeriod.Duration,
				config.IPVS.ExcludeCIDRs,
				config.IPVS.StrictARP,
				config.IPVS.TCPTimeout.Duration,
				config.IPVS.TCPFinTimeout.Duration,
				config.IPVS.UDPTimeout.Duration,
				config.IPTables.MasqueradeAll,
				int(*config.IPTables.MasqueradeBit),
				config.ClusterCIDR,
				hostname,
				nodeIP,
				recorder,
				healthzServer,
				config.IPVS.Scheduler,
				config.NodePortAddresses,
			)
		}
		if err != nil {
			return nil, fmt.Errorf("unable to create proxier: %v", err)
		}
		proxymetrics.RegisterMetrics()
	} else {
		klog.V(0).Info("Using userspace Proxier.")

		// TODO this has side effects that should only happen when Run() is invoked.
		proxier, err = userspace.NewProxier(
			userspace.NewLoadBalancerRR(),
			net.ParseIP(config.BindAddress),
			iptInterface,
			execer,
			*utilnet.ParsePortRangeOrDie(config.PortRange),
			config.IPTables.SyncPeriod.Duration,
			config.IPTables.MinSyncPeriod.Duration,
			config.UDPIdleTimeout.Duration,
			config.NodePortAddresses,
		)
		if err != nil {
			return nil, fmt.Errorf("unable to create proxier: %v", err)
		}
	}

	return &ProxyServer{
		Client:                 client,
		EventClient:            eventClient,
		IptInterface:           iptInterface,
		IpvsInterface:          ipvsInterface,
		IpsetInterface:         ipsetInterface,
		execer:                 execer,
		Proxier:                proxier,
		Broadcaster:            eventBroadcaster,
		Recorder:               recorder,
		ConntrackConfiguration: config.Conntrack,
		Conntracker:            &realConntracker{},
		ProxyMode:              proxyMode,
		NodeRef:                nodeRef,
		MetricsBindAddress:     config.MetricsBindAddress,
		EnableProfiling:        config.EnableProfiling,
		OOMScoreAdj:            config.OOMScoreAdj,
		ConfigSyncPeriod:       config.ConfigSyncPeriod.Duration,
		HealthzServer:          healthzServer,
		UseEndpointSlices:      utilfeature.DefaultFeatureGate.Enabled(features.EndpointSliceProxying),
	}, nil
}

// cidrTuple takes a comma separated list of CIDRs and return a tuple (ipv4cidr,ipv6cidr)
// The returned tuple is guaranteed to have the order (ipv4,ipv6) and if no cidr from a family is found an
// empty string "" is inserted.
func cidrTuple(cidrList string) [2]string {
	cidrs := [2]string{"", ""}
	foundIPv4 := false
	foundIPv6 := false

	for _, cidr := range strings.Split(cidrList, ",") {
		if utilsnet.IsIPv6CIDRString(cidr) && !foundIPv6 {
			cidrs[1] = cidr
			foundIPv6 = true
		} else if !foundIPv4 {
			cidrs[0] = cidr
			foundIPv4 = true
		}
		if foundIPv6 && foundIPv4 {
			break
		}
	}

	return cidrs
}

// nodeIPTuple takes an addresses and return a tuple (ipv4,ipv6)
// The returned tuple is guaranteed to have the order (ipv4,ipv6). The address NOT of the passed address
// will have "any" address (0.0.0.0 or ::) inserted.
func nodeIPTuple(bindAddress string) [2]net.IP {
	nodes := [2]net.IP{net.IPv4zero, net.IPv6zero}

	adr := net.ParseIP(bindAddress)
	if utilsnet.IsIPv6(adr) {
		nodes[1] = adr
	} else {
		nodes[0] = adr
	}

	return nodes
}

func getProxyMode(proxyMode string, khandle ipvs.KernelHandler, ipsetver ipvs.IPSetVersioner, kcompat iptables.KernelCompatTester) string {
	switch proxyMode {
	case proxyModeUserspace:
		return proxyModeUserspace
	case proxyModeIPTables:
		return tryIPTablesProxy(kcompat)
	case proxyModeIPVS:
		return tryIPVSProxy(khandle, ipsetver, kcompat)
	}
	klog.Warningf("Unknown proxy mode %q, assuming iptables proxy", proxyMode)
	return tryIPTablesProxy(kcompat)
}

func tryIPVSProxy(khandle ipvs.KernelHandler, ipsetver ipvs.IPSetVersioner, kcompat iptables.KernelCompatTester) string {
	// guaranteed false on error, error only necessary for debugging
	// IPVS Proxier relies on ip_vs_* kernel modules and ipset
	useIPVSProxy, err := ipvs.CanUseIPVSProxier(khandle, ipsetver)
	if err != nil {
		// Try to fallback to iptables before falling back to userspace
		utilruntime.HandleError(fmt.Errorf("can't determine whether to use ipvs proxy, error: %v", err))
	}
	if useIPVSProxy {
		return proxyModeIPVS
	}

	// Try to fallback to iptables before falling back to userspace
	klog.V(1).Infof("Can't use ipvs proxier, trying iptables proxier")
	return tryIPTablesProxy(kcompat)
}

func tryIPTablesProxy(kcompat iptables.KernelCompatTester) string {
	// guaranteed false on error, error only necessary for debugging
	useIPTablesProxy, err := iptables.CanUseIPTablesProxier(kcompat)
	if err != nil {
		utilruntime.HandleError(fmt.Errorf("can't determine whether to use iptables proxy, using userspace proxier: %v", err))
		return proxyModeUserspace
	}
	if useIPTablesProxy {
		return proxyModeIPTables
	}
	// Fallback.
	klog.V(1).Infof("Can't use iptables proxy, using userspace proxier")
	return proxyModeUserspace
}
