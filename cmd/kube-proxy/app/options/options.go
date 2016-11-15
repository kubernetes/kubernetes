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

// Package options contains flags for initializing a proxy.
package options

import (
	_ "net/http/pprof"
	"time"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/apis/componentconfig"
	"k8s.io/kubernetes/pkg/apis/componentconfig/v1alpha1"
	"k8s.io/kubernetes/pkg/kubelet/qos"
	"k8s.io/kubernetes/pkg/util"
	"k8s.io/kubernetes/pkg/util/config"

	"github.com/spf13/pflag"
)

const (
	ExperimentalProxyModeAnnotation = "net.experimental.kubernetes.io/proxy-mode"
)

// ProxyServerConfig configures and runs a Kubernetes proxy server
type ProxyServerConfig struct {
	componentconfig.KubeProxyConfiguration
	ResourceContainer string
	ContentType       string
	KubeAPIQPS        float32
	KubeAPIBurst      int32
	ConfigSyncPeriod  time.Duration
	CleanupAndExit    bool
	NodeRef           *api.ObjectReference
	Master            string
	Kubeconfig        string
}

func NewProxyConfig() *ProxyServerConfig {
	versioned := &v1alpha1.KubeProxyConfiguration{}
	api.Scheme.Default(versioned)
	cfg := componentconfig.KubeProxyConfiguration{}
	api.Scheme.Convert(versioned, &cfg, nil)
	return &ProxyServerConfig{
		KubeProxyConfiguration: cfg,
		ContentType:            "application/vnd.kubernetes.protobuf",
		KubeAPIQPS:             5.0,
		KubeAPIBurst:           10,
		ConfigSyncPeriod:       15 * time.Minute,
	}
}

// AddFlags adds flags for a specific ProxyServer to the specified FlagSet
func (s *ProxyServerConfig) AddFlags(fs *pflag.FlagSet) {
	fs.Var(componentconfig.IPVar{Val: &s.BindAddress}, "bind-address", "The IP address for the proxy server to serve on (set to 0.0.0.0 for all interfaces)")
	fs.StringVar(&s.Master, "master", s.Master, "The address of the Kubernetes API server (overrides any value in kubeconfig)")
	fs.Int32Var(&s.HealthzPort, "healthz-port", s.HealthzPort, "The port to bind the health check server. Use 0 to disable.")
	fs.Var(componentconfig.IPVar{Val: &s.HealthzBindAddress}, "healthz-bind-address", "The IP address for the health check server to serve on, defaulting to 127.0.0.1 (set to 0.0.0.0 for all interfaces)")
	fs.Int32Var(s.OOMScoreAdj, "oom-score-adj", util.Int32PtrDerefOr(s.OOMScoreAdj, int32(qos.KubeProxyOOMScoreAdj)), "The oom-score-adj value for kube-proxy process. Values must be within the range [-1000, 1000]")
	fs.StringVar(&s.ResourceContainer, "resource-container", s.ResourceContainer, "Absolute name of the resource-only container to create and run the Kube-proxy in (Default: /kube-proxy).")
	fs.MarkDeprecated("resource-container", "This feature will be removed in a later release.")
	fs.StringVar(&s.Kubeconfig, "kubeconfig", s.Kubeconfig, "Path to kubeconfig file with authorization information (the master location is set by the master flag).")
	fs.Var(componentconfig.PortRangeVar{Val: &s.PortRange}, "proxy-port-range", "Range of host ports (beginPort-endPort, inclusive) that may be consumed in order to proxy service traffic. If unspecified (0-0) then ports will be randomly chosen.")
	fs.StringVar(&s.HostnameOverride, "hostname-override", s.HostnameOverride, "If non-empty, will use this string as identification instead of the actual hostname.")
	fs.Var(&s.Mode, "proxy-mode", "Which proxy mode to use: 'userspace' (older) or 'iptables' (faster). If blank, look at the Node object on the Kubernetes API and respect the '"+ExperimentalProxyModeAnnotation+"' annotation if provided.  Otherwise use the best-available proxy (currently iptables).  If the iptables proxy is selected, regardless of how, but the system's kernel or iptables versions are insufficient, this always falls back to the userspace proxy.")
	fs.Int32Var(s.IPTablesMasqueradeBit, "iptables-masquerade-bit", util.Int32PtrDerefOr(s.IPTablesMasqueradeBit, 14), "If using the pure iptables proxy, the bit of the fwmark space to mark packets requiring SNAT with.  Must be within the range [0, 31].")
	fs.DurationVar(&s.IPTablesSyncPeriod.Duration, "iptables-sync-period", s.IPTablesSyncPeriod.Duration, "The maximum interval of how often iptables rules are refreshed (e.g. '5s', '1m', '2h22m').  Must be greater than 0.")
	fs.DurationVar(&s.IPTablesMinSyncPeriod.Duration, "iptables-min-sync-period", s.IPTablesMinSyncPeriod.Duration, "The minimum interval of how often the iptables rules can be refreshed as endpoints and services change (e.g. '5s', '1m', '2h22m').")
	fs.DurationVar(&s.ConfigSyncPeriod, "config-sync-period", s.ConfigSyncPeriod, "How often configuration from the apiserver is refreshed.  Must be greater than 0.")
	fs.BoolVar(&s.MasqueradeAll, "masquerade-all", s.MasqueradeAll, "If using the pure iptables proxy, SNAT everything")
	fs.StringVar(&s.ClusterCIDR, "cluster-cidr", s.ClusterCIDR, "The CIDR range of pods in the cluster. It is used to bridge traffic coming from outside of the cluster. If not provided, no off-cluster bridging will be performed.")
	fs.BoolVar(&s.CleanupAndExit, "cleanup-iptables", s.CleanupAndExit, "If true cleanup iptables rules and exit.")
	fs.StringVar(&s.ContentType, "kube-api-content-type", s.ContentType, "Content type of requests sent to apiserver.")
	fs.Float32Var(&s.KubeAPIQPS, "kube-api-qps", s.KubeAPIQPS, "QPS to use while talking with kubernetes apiserver")
	fs.Int32Var(&s.KubeAPIBurst, "kube-api-burst", s.KubeAPIBurst, "Burst to use while talking with kubernetes apiserver")
	fs.DurationVar(&s.UDPIdleTimeout.Duration, "udp-timeout", s.UDPIdleTimeout.Duration, "How long an idle UDP connection will be kept open (e.g. '250ms', '2s').  Must be greater than 0. Only applicable for proxy-mode=userspace")
	fs.Int32Var(&s.ConntrackMax, "conntrack-max", s.ConntrackMax,
		"Maximum number of NAT connections to track (0 to leave as-is). This overrides conntrack-max-per-core and conntrack-min.")
	fs.MarkDeprecated("conntrack-max", "This feature will be removed in a later release.")
	fs.Int32Var(&s.ConntrackMaxPerCore, "conntrack-max-per-core", s.ConntrackMaxPerCore,
		"Maximum number of NAT connections to track per CPU core (0 to leave the limit as-is and ignore conntrack-min).")
	fs.Int32Var(&s.ConntrackMin, "conntrack-min", s.ConntrackMin,
		"Minimum number of conntrack entries to allocate, regardless of conntrack-max-per-core (set conntrack-max-per-core=0 to leave the limit as-is).")
	fs.DurationVar(&s.ConntrackTCPEstablishedTimeout.Duration, "conntrack-tcp-timeout-established", s.ConntrackTCPEstablishedTimeout.Duration, "Idle timeout for established TCP connections (0 to leave as-is)")
	fs.DurationVar(
		&s.ConntrackTCPCloseWaitTimeout.Duration, "conntrack-tcp-timeout-close-wait",
		s.ConntrackTCPCloseWaitTimeout.Duration,
		"NAT timeout for TCP connections in the CLOSE_WAIT state")

	config.DefaultFeatureGate.AddFlag(fs)
}
