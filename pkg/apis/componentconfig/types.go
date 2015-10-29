/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

package componentconfig

import "k8s.io/kubernetes/pkg/api/unversioned"

type KubeProxyConfiguration struct {
	unversioned.TypeMeta

	// bindAddress is the IP address for the proxy server to serve on (set to 0.0.0.0 for all interfaces)
	BindAddress string `json:"bindAddress"`
	// cleanupIPTables
	CleanupIPTables bool `json:"cleanupIPTables"`
	// healthzBindAddress is the IP address for the health check server to serve on, defaulting to 127.0.0.1 (set to 0.0.0.0 for all interfaces)
	HealthzBindAddress string `json:"healthzBindAddress"`
	// healthzPort is the port to bind the health check server. Use 0 to disable.
	HealthzPort int `json:"healthzPort"`
	// hostnameOverride, if non-empty, will be used as the identity instead of the actual hostname.
	HostnameOverride string `json:"hostnameOverride"`
	// iptablesSyncPeriodSeconds is the period that iptables rules are refreshed (e.g. '5s', '1m', '2h22m').  Must be greater than 0.
	IPTablesSyncePeriodSeconds int `json:"iptablesSyncPeriodSeconds"`
	// kubeAPIBurst is the burst to use while talking with kubernetes apiserver
	KubeAPIBurst int `json:"kubeAPIBurst"`
	// kubeAPIQPS is the max QPS to use while talking with kubernetes apiserver
	KubeAPIQPS int `json:"kubeAPIQPS"`
	// kubeconfigPath is the path to the kubeconfig file with authorization information (the master location is set by the master flag).
	KubeconfigPath string `json:"kubeconfigPath"`
	// masqueradeAll tells kube-proxy to SNAT everything if using the pure iptables proxy mode.
	MasqueradeAll bool `json:"masqueradeAll"`
	// master is the address of the Kubernetes API server (overrides any value in kubeconfig)
	Master string `json:"master"`
	// oomScoreAdj is the oom-score-adj value for kube-proxy process. Values must be within the range [-1000, 1000]
	OOMScoreAdj *int `json:"oomScoreAdj"`
	// mode specifies which proxy mode to use.
	Mode ProxyMode `json:"mode"`
	// portRange is the range of host ports (beginPort-endPort, inclusive) that may be consumed in order to proxy service traffic. If unspecified (0-0) then ports will be randomly chosen.
	PortRange string `json:"portRange"`
	// resourceContainer is the bsolute name of the resource-only container to create and run the Kube-proxy in (Default: /kube-proxy).
	ResourceContainer string `json:"resourceContainer"`
	// udpTimeoutMilliseconds is how long an idle UDP connection will be kept open (e.g. '250ms', '2s').  Must be greater than 0. Only applicable for proxyMode=userspace.
	UDPTimeoutMilliseconds int `json:"udpTimeoutMilliseconds"`
}

// Currently two modes of proxying are available: 'userspace' (older, stable) or 'iptables' (experimental). If blank, look at the Node object on the Kubernetes API and respect the 'net.experimental.kubernetes.io/proxy-mode' annotation if provided.  Otherwise use the best-available proxy (currently userspace, but may change in future versions).  If the iptables proxy is selected, regardless of how, but the system's kernel or iptables versions are insufficient, this always falls back to the userspace proxy.
type ProxyMode string

const (
	ProxyModeUserspace ProxyMode = "userspace"
	ProxyModeIPTables  ProxyMode = "iptables"
)
