/*
Copyright 2015 The Kubernetes Authors.

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

package config

import (
	"fmt"
	"sort"
	"strings"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	componentbaseconfig "k8s.io/component-base/config"
)

// KubeProxyIPTablesConfiguration contains iptables-related configuration
// details for the Kubernetes proxy server.
type KubeProxyIPTablesConfiguration struct {
	// masqueradeBit is the bit of the iptables fwmark space to use for SNAT if using
	// the pure iptables proxy mode. Values must be within the range [0, 31].
	MasqueradeBit *int32
	// masqueradeAll tells kube-proxy to SNAT everything if using the pure iptables proxy mode.
	MasqueradeAll bool
	// LocalhostNodePorts tells kube-proxy to allow service NodePorts to be accessed via
	// localhost (iptables mode only)
	LocalhostNodePorts *bool
	// syncPeriod is the period that iptables rules are refreshed (e.g. '5s', '1m',
	// '2h22m').  Must be greater than 0.
	SyncPeriod metav1.Duration
	// minSyncPeriod is the minimum period that iptables rules are refreshed (e.g. '5s', '1m',
	// '2h22m').
	MinSyncPeriod metav1.Duration
}

// KubeProxyIPVSConfiguration contains ipvs-related configuration
// details for the Kubernetes proxy server.
type KubeProxyIPVSConfiguration struct {
	// syncPeriod is the period that ipvs rules are refreshed (e.g. '5s', '1m',
	// '2h22m').  Must be greater than 0.
	SyncPeriod metav1.Duration
	// minSyncPeriod is the minimum period that ipvs rules are refreshed (e.g. '5s', '1m',
	// '2h22m').
	MinSyncPeriod metav1.Duration
	// ipvs scheduler
	Scheduler string
	// excludeCIDRs is a list of CIDR's which the ipvs proxier should not touch
	// when cleaning up ipvs services.
	ExcludeCIDRs []string
	// strict ARP configure arp_ignore and arp_announce to avoid answering ARP queries
	// from kube-ipvs0 interface
	StrictARP bool
	// tcpTimeout is the timeout value used for idle IPVS TCP sessions.
	// The default value is 0, which preserves the current timeout value on the system.
	TCPTimeout metav1.Duration
	// tcpFinTimeout is the timeout value used for IPVS TCP sessions after receiving a FIN.
	// The default value is 0, which preserves the current timeout value on the system.
	TCPFinTimeout metav1.Duration
	// udpTimeout is the timeout value used for IPVS UDP packets.
	// The default value is 0, which preserves the current timeout value on the system.
	UDPTimeout metav1.Duration
}

// KubeProxyConntrackConfiguration contains conntrack settings for
// the Kubernetes proxy server.
type KubeProxyConntrackConfiguration struct {
	// maxPerCore is the maximum number of NAT connections to track
	// per CPU core (0 to leave the limit as-is and ignore min).
	MaxPerCore *int32
	// min is the minimum value of connect-tracking records to allocate,
	// regardless of maxPerCore (set maxPerCore=0 to leave the limit as-is).
	Min *int32
	// tcpEstablishedTimeout is how long an idle TCP connection will be kept open
	// (e.g. '2s').  Must be greater than 0 to set.
	TCPEstablishedTimeout *metav1.Duration
	// tcpCloseWaitTimeout is how long an idle conntrack entry
	// in CLOSE_WAIT state will remain in the conntrack
	// table. (e.g. '60s'). Must be greater than 0 to set.
	TCPCloseWaitTimeout *metav1.Duration
}

// KubeProxyWinkernelConfiguration contains Windows/HNS settings for
// the Kubernetes proxy server.
type KubeProxyWinkernelConfiguration struct {
	// networkName is the name of the network kube-proxy will use
	// to create endpoints and policies
	NetworkName string
	// sourceVip is the IP address of the source VIP endpoint used for
	// NAT when loadbalancing
	SourceVip string
	// enableDSR tells kube-proxy whether HNS policies should be created
	// with DSR
	EnableDSR bool
	// RootHnsEndpointName is the name of hnsendpoint that is attached to
	// l2bridge for root network namespace
	RootHnsEndpointName string
	// ForwardHealthCheckVip forwards service VIP for health check port on
	// Windows
	ForwardHealthCheckVip bool
}

// DetectLocalConfiguration contains optional settings related to DetectLocalMode option
type DetectLocalConfiguration struct {
	// BridgeInterface is a string argument which represents a single bridge interface name.
	// Kube-proxy considers traffic as local if originating from this given bridge.
	// This argument should be set if DetectLocalMode is set to BridgeInterface.
	BridgeInterface string
	// InterfaceNamePrefix is a string argument which represents a single interface prefix name.
	// Kube-proxy considers traffic as local if originating from one or more interfaces which match
	// the given prefix. This argument should be set if DetectLocalMode is set to InterfaceNamePrefix.
	InterfaceNamePrefix string
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// KubeProxyConfiguration contains everything necessary to configure the
// Kubernetes proxy server.
type KubeProxyConfiguration struct {
	metav1.TypeMeta

	// featureGates is a map of feature names to bools that enable or disable alpha/experimental features.
	FeatureGates map[string]bool

	// bindAddress is the IP address for the proxy server to serve on (set to 0.0.0.0
	// for all interfaces)
	BindAddress string
	// healthzBindAddress is the IP address and port for the health check server to serve on,
	// defaulting to 0.0.0.0:10256
	HealthzBindAddress string
	// metricsBindAddress is the IP address and port for the metrics server to serve on,
	// defaulting to 127.0.0.1:10249 (set to 0.0.0.0 for all interfaces)
	MetricsBindAddress string
	// BindAddressHardFail, if true, kube-proxy will treat failure to bind to a port as fatal and exit
	BindAddressHardFail bool
	// enableProfiling enables profiling via web interface on /debug/pprof handler.
	// Profiling handlers will be handled by metrics server.
	EnableProfiling bool
	// clusterCIDR is the CIDR range of the pods in the cluster. It is used to
	// bridge traffic coming from outside of the cluster. If not provided,
	// no off-cluster bridging will be performed.
	ClusterCIDR string
	// hostnameOverride, if non-empty, will be used as the identity instead of the actual hostname.
	HostnameOverride string
	// clientConnection specifies the kubeconfig file and client connection settings for the proxy
	// server to use when communicating with the apiserver.
	ClientConnection componentbaseconfig.ClientConnectionConfiguration
	// iptables contains iptables-related configuration options.
	IPTables KubeProxyIPTablesConfiguration
	// ipvs contains ipvs-related configuration options.
	IPVS KubeProxyIPVSConfiguration
	// oomScoreAdj is the oom-score-adj value for kube-proxy process. Values must be within
	// the range [-1000, 1000]
	OOMScoreAdj *int32
	// mode specifies which proxy mode to use.
	Mode ProxyMode
	// portRange is the range of host ports (beginPort-endPort, inclusive) that may be consumed
	// in order to proxy service traffic. If unspecified (0-0) then ports will be randomly chosen.
	PortRange string
	// conntrack contains conntrack-related configuration options.
	Conntrack KubeProxyConntrackConfiguration
	// configSyncPeriod is how often configuration from the apiserver is refreshed. Must be greater
	// than 0.
	ConfigSyncPeriod metav1.Duration
	// nodePortAddresses is the --nodeport-addresses value for kube-proxy process. Values must be valid
	// IP blocks. These values are as a parameter to select the interfaces where nodeport works.
	// In case someone would like to expose a service on localhost for local visit and some other interfaces for
	// particular purpose, a list of IP blocks would do that.
	// If set it to "127.0.0.0/8", kube-proxy will only select the loopback interface for NodePort.
	// If set it to a non-zero IP block, kube-proxy will filter that down to just the IPs that applied to the node.
	// An empty string slice is meant to select all network interfaces.
	NodePortAddresses []string
	// winkernel contains winkernel-related configuration options.
	Winkernel KubeProxyWinkernelConfiguration
	// ShowHiddenMetricsForVersion is the version for which you want to show hidden metrics.
	ShowHiddenMetricsForVersion string
	// DetectLocalMode determines mode to use for detecting local traffic, defaults to LocalModeClusterCIDR
	DetectLocalMode LocalMode
	// DetectLocal contains optional configuration settings related to DetectLocalMode.
	DetectLocal DetectLocalConfiguration
}

// ProxyMode represents modes used by the Kubernetes proxy server.
//
// Currently, two modes of proxy are available on Linux platforms: 'iptables' and 'ipvs'.
// One mode of proxy is available on Windows platforms: 'kernelspace'.
//
// If the proxy mode is unspecified, the best-available proxy mode will be used (currently this
// is `iptables` on Linux and `kernelspace` on Windows). If the selected proxy mode cannot be
// used (due to lack of kernel support, missing userspace components, etc) then kube-proxy
// will exit with an error.
type ProxyMode string

const (
	ProxyModeIPTables    ProxyMode = "iptables"
	ProxyModeIPVS        ProxyMode = "ipvs"
	ProxyModeKernelspace ProxyMode = "kernelspace"
)

// LocalMode represents modes to detect local traffic from the node
type LocalMode string

// Currently supported modes for LocalMode
const (
	LocalModeClusterCIDR         LocalMode = "ClusterCIDR"
	LocalModeNodeCIDR            LocalMode = "NodeCIDR"
	LocalModeBridgeInterface     LocalMode = "BridgeInterface"
	LocalModeInterfaceNamePrefix LocalMode = "InterfaceNamePrefix"
)

func (m *ProxyMode) Set(s string) error {
	*m = ProxyMode(s)
	return nil
}

func (m *ProxyMode) String() string {
	if m != nil {
		return string(*m)
	}
	return ""
}

func (m *ProxyMode) Type() string {
	return "ProxyMode"
}

func (m *LocalMode) Set(s string) error {
	*m = LocalMode(s)
	return nil
}

func (m *LocalMode) String() string {
	if m != nil {
		return string(*m)
	}
	return ""
}

func (m *LocalMode) Type() string {
	return "LocalMode"
}

type ConfigurationMap map[string]string

func (m *ConfigurationMap) String() string {
	pairs := []string{}
	for k, v := range *m {
		pairs = append(pairs, fmt.Sprintf("%s=%s", k, v))
	}
	sort.Strings(pairs)
	return strings.Join(pairs, ",")
}

func (m *ConfigurationMap) Set(value string) error {
	for _, s := range strings.Split(value, ",") {
		if len(s) == 0 {
			continue
		}
		arr := strings.SplitN(s, "=", 2)
		if len(arr) == 2 {
			(*m)[strings.TrimSpace(arr[0])] = strings.TrimSpace(arr[1])
		} else {
			(*m)[strings.TrimSpace(arr[0])] = ""
		}
	}
	return nil
}

func (*ConfigurationMap) Type() string {
	return "mapStringString"
}
