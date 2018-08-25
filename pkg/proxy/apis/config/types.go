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

	apimachineryconfig "k8s.io/apimachinery/pkg/apis/config"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// KubeProxyIPTablesConfiguration contains iptables-related configuration
// details for the Kubernetes proxy server.
type KubeProxyIPTablesConfiguration struct {
	// masqueradeBit is the bit of the iptables fwmark space to use for SNAT if using
	// the pure iptables proxy mode. Values must be within the range [0, 31].
	MasqueradeBit *int32
	// masqueradeAll tells kube-proxy to SNAT everything if using the pure iptables proxy mode.
	MasqueradeAll bool
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
}

// KubeProxyConntrackConfiguration contains conntrack settings for
// the Kubernetes proxy server.
type KubeProxyConntrackConfiguration struct {
	// max is the maximum number of NAT connections to track (0 to
	// leave as-is).  This takes precedence over maxPerCore and min.
	Max *int32
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
	ClientConnection apimachineryconfig.ClientConnectionConfiguration
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
	// resourceContainer is the absolute name of the resource-only container to create and run
	// the Kube-proxy in (Default: /kube-proxy).
	ResourceContainer string
	// udpIdleTimeout is how long an idle UDP connection will be kept open (e.g. '250ms', '2s').
	// Must be greater than 0. Only applicable for proxyMode=userspace.
	UDPIdleTimeout metav1.Duration
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
}

// Currently, three modes of proxy are available in Linux platform: 'userspace' (older, going to be EOL), 'iptables'
// (newer, faster), 'ipvs'(newest, better in performance and scalability).
//
// Two modes of proxy are available in Windows platform: 'userspace'(older, stable) and 'kernelspace' (newer, faster).
//
// In Linux platform, if proxy mode is blank, use the best-available proxy (currently iptables, but may change in the
// future). If the iptables proxy is selected, regardless of how, but the system's kernel or iptables versions are
// insufficient, this always falls back to the userspace proxy. IPVS mode will be enabled when proxy mode is set to 'ipvs',
// and the fall back path is firstly iptables and then userspace.

// In Windows platform, if proxy mode is blank, use the best-available proxy (currently userspace, but may change in the
// future). If winkernel proxy is selected, regardless of how, but the Windows kernel can't support this mode of proxy,
// this always falls back to the userspace proxy.
type ProxyMode string

const (
	ProxyModeUserspace   ProxyMode = "userspace"
	ProxyModeIPTables    ProxyMode = "iptables"
	ProxyModeIPVS        ProxyMode = "ipvs"
	ProxyModeKernelspace ProxyMode = "kernelspace"
)

// IPVSSchedulerMethod is the algorithm for allocating TCP connections and
// UDP datagrams to real servers.  Scheduling algorithms are imple-
//wanted as kernel modules. Ten are shipped with the Linux Virtual Server.
type IPVSSchedulerMethod string

const (
	// RoundRobin distributes jobs equally amongst the available real servers.
	RoundRobin IPVSSchedulerMethod = "rr"
	// WeightedRoundRobin assigns jobs to real servers proportionally to there real servers' weight.
	// Servers with higher weights receive new jobs first and get more jobs than servers with lower weights.
	// Servers with equal weights get an equal distribution of new jobs.
	WeightedRoundRobin IPVSSchedulerMethod = "wrr"
	// LeastConnection assigns more jobs to real servers with fewer active jobs.
	LeastConnection IPVSSchedulerMethod = "lc"
	// WeightedLeastConnection assigns more jobs to servers with fewer jobs and
	// relative to the real servers' weight(Ci/Wi).
	WeightedLeastConnection IPVSSchedulerMethod = "wlc"
	// LocalityBasedLeastConnection assigns jobs destined for the same IP address to the same server if
	// the server is not overloaded and available; otherwise assigns jobs to servers with fewer jobs,
	// and keep it for future assignment.
	LocalityBasedLeastConnection IPVSSchedulerMethod = "lblc"
	// LocalityBasedLeastConnectionWithReplication with Replication assigns jobs destined for the same IP address to the
	// least-connection node in the server set for the IP address. If all the node in the server set are overloaded,
	// it picks up a node with fewer jobs in the cluster and adds it to the sever set for the target.
	// If the server set has not been modified for the specified time, the most loaded node is removed from the server set,
	// in order to avoid high degree of replication.
	LocalityBasedLeastConnectionWithReplication IPVSSchedulerMethod = "lblcr"
	// SourceHashing assigns jobs to servers through looking up a statically assigned hash table
	// by their source IP addresses.
	SourceHashing IPVSSchedulerMethod = "sh"
	// DestinationHashing assigns jobs to servers through looking up a statically assigned hash table
	// by their destination IP addresses.
	DestinationHashing IPVSSchedulerMethod = "dh"
	// ShortestExpectedDelay assigns an incoming job to the server with the shortest expected delay.
	// The expected delay that the job will experience is (Ci + 1) / Ui if sent to the ith server, in which
	// Ci is the number of jobs on the the ith server and Ui is the fixed service rate (weight) of the ith server.
	ShortestExpectedDelay IPVSSchedulerMethod = "sed"
	// NeverQueue assigns an incoming job to an idle server if there is, instead of waiting for a fast one;
	// if all the servers are busy, it adopts the ShortestExpectedDelay policy to assign the job.
	NeverQueue IPVSSchedulerMethod = "nq"
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
