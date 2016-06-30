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

package v1alpha1

import "k8s.io/kubernetes/pkg/api/unversioned"

type KubeProxyConfiguration struct {
	unversioned.TypeMeta

	// bindAddress is the IP address for the proxy server to serve on (set to 0.0.0.0
	// for all interfaces)
	BindAddress string `json:"bindAddress"`
	// clusterCIDR is the CIDR range of the pods in the cluster. It is used to
	// bridge traffic coming from outside of the cluster. If not provided,
	// no off-cluster bridging will be performed.
	ClusterCIDR string `json:"clusterCIDR"`
	// healthzBindAddress is the IP address for the health check server to serve on,
	// defaulting to 127.0.0.1 (set to 0.0.0.0 for all interfaces)
	HealthzBindAddress string `json:"healthzBindAddress"`
	// healthzPort is the port to bind the health check server. Use 0 to disable.
	HealthzPort int32 `json:"healthzPort"`
	// hostnameOverride, if non-empty, will be used as the identity instead of the actual hostname.
	HostnameOverride string `json:"hostnameOverride"`
	// iptablesMasqueradeBit is the bit of the iptables fwmark space to use for SNAT if using
	// the pure iptables proxy mode. Values must be within the range [0, 31].
	IPTablesMasqueradeBit *int32 `json:"iptablesMasqueradeBit"`
	// iptablesSyncPeriod is the period that iptables rules are refreshed (e.g. '5s', '1m',
	// '2h22m').  Must be greater than 0.
	IPTablesSyncPeriod unversioned.Duration `json:"iptablesSyncPeriodSeconds"`
	// kubeconfigPath is the path to the kubeconfig file with authorization information (the
	// master location is set by the master flag).
	KubeconfigPath string `json:"kubeconfigPath"`
	// masqueradeAll tells kube-proxy to SNAT everything if using the pure iptables proxy mode.
	MasqueradeAll bool `json:"masqueradeAll"`
	// master is the address of the Kubernetes API server (overrides any value in kubeconfig)
	Master string `json:"master"`
	// oomScoreAdj is the oom-score-adj value for kube-proxy process. Values must be within
	// the range [-1000, 1000]
	OOMScoreAdj *int32 `json:"oomScoreAdj"`
	// mode specifies which proxy mode to use.
	Mode ProxyMode `json:"mode"`
	// portRange is the range of host ports (beginPort-endPort, inclusive) that may be consumed
	// in order to proxy service traffic. If unspecified (0-0) then ports will be randomly chosen.
	PortRange string `json:"portRange"`
	// resourceContainer is the bsolute name of the resource-only container to create and run
	// the Kube-proxy in (Default: /kube-proxy).
	ResourceContainer string `json:"resourceContainer"`
	// udpIdleTimeout is how long an idle UDP connection will be kept open (e.g. '250ms', '2s').
	// Must be greater than 0. Only applicable for proxyMode=userspace.
	UDPIdleTimeout unversioned.Duration `json:"udpTimeoutMilliseconds"`
	// conntrackMax is the maximum number of NAT connections to track (0 to leave as-is)")
	ConntrackMax int32 `json:"conntrackMax"`
	// conntrackTCPEstablishedTimeout is how long an idle UDP connection will be kept open
	// (e.g. '250ms', '2s').  Must be greater than 0. Only applicable for proxyMode is Userspace
	ConntrackTCPEstablishedTimeout unversioned.Duration `json:"conntrackTCPEstablishedTimeout"`
}

// Currently two modes of proxying are available: 'userspace' (older, stable) or 'iptables'
// (experimental). If blank, look at the Node object on the Kubernetes API and respect the
// 'net.experimental.kubernetes.io/proxy-mode' annotation if provided.  Otherwise use the
// best-available proxy (currently userspace, but may change in future versions).  If the
// iptables proxy is selected, regardless of how, but the system's kernel or iptables
// versions are insufficient, this always falls back to the userspace proxy.
type ProxyMode string

const (
	ProxyModeUserspace ProxyMode = "userspace"
	ProxyModeIPTables  ProxyMode = "iptables"
)

type KubeSchedulerConfiguration struct {
	unversioned.TypeMeta

	// port is the port that the scheduler's http service runs on.
	Port int `json:"port"`
	// address is the IP address to serve on.
	Address string `json:"address"`
	// algorithmProvider is the scheduling algorithm provider to use.
	AlgorithmProvider string `json:"algorithmProvider"`
	// policyConfigFile is the filepath to the scheduler policy configuration.
	PolicyConfigFile string `json:"policyConfigFile"`
	// enableProfiling enables profiling via web interface.
	EnableProfiling *bool `json:"enableProfiling"`
	// contentType is contentType of requests sent to apiserver.
	ContentType string `json:"contentType"`
	// kubeAPIQPS is the QPS to use while talking with kubernetes apiserver.
	KubeAPIQPS float32 `json:"kubeAPIQPS"`
	// kubeAPIBurst is the QPS burst to use while talking with kubernetes apiserver.
	KubeAPIBurst int `json:"kubeAPIBurst"`
	// schedulerName is name of the scheduler, used to select which pods
	// will be processed by this scheduler, based on pod's annotation with
	// key 'scheduler.alpha.kubernetes.io/name'.
	SchedulerName string `json:"schedulerName"`
	// RequiredDuringScheduling affinity is not symmetric, but there is an implicit PreferredDuringScheduling affinity rule
	// corresponding to every RequiredDuringScheduling affinity rule.
	// HardPodAffinitySymmetricWeight represents the weight of implicit PreferredDuringScheduling affinity rule, in the range 0-100.
	HardPodAffinitySymmetricWeight int `json:"hardPodAffinitySymmetricWeight"`
	// Indicate the "all topologies" set for empty topologyKey when it's used for PreferredDuringScheduling pod anti-affinity.
	FailureDomains string `json:"failureDomains"`
	// leaderElection defines the configuration of leader election client.
	LeaderElection LeaderElectionConfiguration `json:"leaderElection"`
}

// LeaderElectionConfiguration defines the configuration of leader election
// clients for components that can run with leader election enabled.
type LeaderElectionConfiguration struct {
	// leaderElect enables a leader election client to gain leadership
	// before executing the main loop. Enable this when running replicated
	// components for high availability.
	LeaderElect *bool `json:"leaderElect"`
	// leaseDuration is the duration that non-leader candidates will wait
	// after observing a leadership renewal until attempting to acquire
	// leadership of a led but unrenewed leader slot. This is effectively the
	// maximum duration that a leader can be stopped before it is replaced
	// by another candidate. This is only applicable if leader election is
	// enabled.
	LeaseDuration unversioned.Duration `json:"leaseDuration"`
	// renewDeadline is the interval between attempts by the acting master to
	// renew a leadership slot before it stops leading. This must be less
	// than or equal to the lease duration. This is only applicable if leader
	// election is enabled.
	RenewDeadline unversioned.Duration `json:"renewDeadline"`
	// retryPeriod is the duration the clients should wait between attempting
	// acquisition and renewal of a leadership. This is only applicable if
	// leader election is enabled.
	RetryPeriod unversioned.Duration `json:"retryPeriod"`
}
