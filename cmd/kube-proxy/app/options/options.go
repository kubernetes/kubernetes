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

// Package options covers the flags for the kube-proxy.
package options

import (
	"errors"
	"fmt"
	"io/ioutil"
	"os"
	"strings"

	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/serializer/json"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/apiserver/pkg/util/flag"
	"k8s.io/kubernetes/pkg/apis/componentconfig"
	"k8s.io/kubernetes/pkg/master/ports"
	"k8s.io/kubernetes/pkg/proxy/apis/kubeproxyconfig"
	"k8s.io/kubernetes/pkg/proxy/apis/kubeproxyconfig/scheme"
	"k8s.io/kubernetes/pkg/proxy/apis/kubeproxyconfig/v1alpha1"
	utilpointer "k8s.io/kubernetes/pkg/util/pointer"

	"github.com/golang/glog"
	"github.com/spf13/pflag"
)

// Options contains everything necessary to create and run a proxy server.
type Options struct {
	// ConfigFile is the location of the proxy server's configuration file.
	ConfigFile string
	// WriteConfigTo is the path where the default configuration will be written.
	WriteConfigTo string
	// CleanupAndExit, when true, makes the proxy server clean up iptables rules, then exit.
	CleanupAndExit bool
	// CleanupIPVS, when true, makes the proxy server clean up ipvs rules before running.
	CleanupIPVS bool
	// WindowsService should be set to true if kube-proxy is running as a service on Windows.
	// Its corresponding flag only gets registered in Windows builds
	WindowsService bool
	// config is the proxy server's configuration object.
	Config *kubeproxyconfig.KubeProxyConfiguration

	// The fields below here are placeholders for flags that can't be directly mapped into
	// kubeproxyconfig.KubeProxyConfiguration.
	//
	// TODO remove these fields once the deprecated flags are removed.

	// master is used to override the kubeconfig's URL to the apiserver.
	Master string
	// healthzPort is the port to be used by the healthz server.
	HealthzPort int32
}

// AddFlags adds flags to fs and binds them to options.
func (o *Options) AddFlags(fs *pflag.FlagSet) {
	addOSFlags(o, fs)
	fs.StringVar(&o.ConfigFile, "config", o.ConfigFile, "The path to the configuration file.")
	fs.StringVar(&o.WriteConfigTo, "write-config-to", o.WriteConfigTo, "If set, write the default configuration values to this file and exit.")
	fs.BoolVar(&o.CleanupAndExit, "cleanup-iptables", o.CleanupAndExit, "If true cleanup iptables and ipvs rules and exit.")
	fs.MarkDeprecated("cleanup-iptables", "This flag is replaced by --cleanup.")
	fs.BoolVar(&o.CleanupAndExit, "cleanup", o.CleanupAndExit, "If true cleanup iptables and ipvs rules and exit.")
	fs.BoolVar(&o.CleanupIPVS, "cleanup-ipvs", o.CleanupIPVS, "If true make kube-proxy cleanup ipvs rules before running.  Default is true")

	// All flags below here are deprecated and will eventually be removed.

	fs.Var(componentconfig.IPVar{Val: &o.Config.BindAddress}, "bind-address", "The IP address for the proxy server to serve on (set to `0.0.0.0` for all IPv4 interfaces and `::` for all IPv6 interfaces)")
	fs.StringVar(&o.Master, "master", o.Master, "The address of the Kubernetes API server (overrides any value in kubeconfig)")
	fs.Int32Var(&o.HealthzPort, "healthz-port", o.HealthzPort, "The port to bind the health check server. Use 0 to disable.")
	fs.Var(componentconfig.IPVar{Val: &o.Config.HealthzBindAddress}, "healthz-bind-address", "The IP address and port for the health check server to serve on (set to `0.0.0.0` for all IPv4 interfaces and `::` for all IPv6 interfaces)")
	fs.Var(componentconfig.IPVar{Val: &o.Config.MetricsBindAddress}, "metrics-bind-address", "The IP address and port for the metrics server to serve on (set to `0.0.0.0` for all IPv4 interfaces and `::` for all IPv6 interfaces)")
	fs.Int32Var(o.Config.OOMScoreAdj, "oom-score-adj", *o.Config.OOMScoreAdj, "The oom-score-adj value for kube-proxy process. Values must be within the range [-1000, 1000]")
	fs.StringVar(&o.Config.ResourceContainer, "resource-container", o.Config.ResourceContainer, "Absolute name of the resource-only container to create and run the Kube-proxy in (Default: /kube-proxy).")
	fs.MarkDeprecated("resource-container", "This feature will be removed in a later release.")
	fs.StringVar(&o.Config.ClientConnection.KubeConfigFile, "kubeconfig", o.Config.ClientConnection.KubeConfigFile, "Path to kubeconfig file with authorization information (the master location is set by the master flag).")
	fs.Var(componentconfig.PortRangeVar{Val: &o.Config.PortRange}, "proxy-port-range", "Range of host ports (beginPort-endPort, single port or beginPort+offset, inclusive) that may be consumed in order to proxy service traffic. If (unspecified, 0, or 0-0) then ports will be randomly chosen.")
	fs.StringVar(&o.Config.HostnameOverride, "hostname-override", o.Config.HostnameOverride, "If non-empty, will use this string as identification instead of the actual hostname.")
	fs.Var(&o.Config.Mode, "proxy-mode", "Which proxy mode to use: 'userspace' (older) or 'iptables' (faster) or 'ipvs' (experimental). If blank, use the best-available proxy (currently iptables).  If the iptables proxy is selected, regardless of how, but the system's kernel or iptables versions are insufficient, this always falls back to the userspace proxy.")
	fs.Int32Var(o.Config.IPTables.MasqueradeBit, "iptables-masquerade-bit", utilpointer.Int32PtrDerefOr(o.Config.IPTables.MasqueradeBit, 14), "If using the pure iptables proxy, the bit of the fwmark space to mark packets requiring SNAT with.  Must be within the range [0, 31].")
	fs.DurationVar(&o.Config.IPTables.SyncPeriod.Duration, "iptables-sync-period", o.Config.IPTables.SyncPeriod.Duration, "The maximum interval of how often iptables rules are refreshed (e.g. '5s', '1m', '2h22m').  Must be greater than 0.")
	fs.DurationVar(&o.Config.IPTables.MinSyncPeriod.Duration, "iptables-min-sync-period", o.Config.IPTables.MinSyncPeriod.Duration, "The minimum interval of how often the iptables rules can be refreshed as endpoints and services change (e.g. '5s', '1m', '2h22m').")
	fs.DurationVar(&o.Config.IPVS.SyncPeriod.Duration, "ipvs-sync-period", o.Config.IPVS.SyncPeriod.Duration, "The maximum interval of how often ipvs rules are refreshed (e.g. '5s', '1m', '2h22m').  Must be greater than 0.")
	fs.DurationVar(&o.Config.IPVS.MinSyncPeriod.Duration, "ipvs-min-sync-period", o.Config.IPVS.MinSyncPeriod.Duration, "The minimum interval of how often the ipvs rules can be refreshed as endpoints and services change (e.g. '5s', '1m', '2h22m').")
	fs.StringSliceVar(&o.Config.IPVS.ExcludeCIDRs, "ipvs-exclude-cidrs", o.Config.IPVS.ExcludeCIDRs, "A comma-separated list of CIDR's which the ipvs proxier should not touch when cleaning up IPVS rules.")
	fs.DurationVar(&o.Config.ConfigSyncPeriod.Duration, "config-sync-period", o.Config.ConfigSyncPeriod.Duration, "How often configuration from the apiserver is refreshed.  Must be greater than 0.")
	fs.BoolVar(&o.Config.IPTables.MasqueradeAll, "masquerade-all", o.Config.IPTables.MasqueradeAll, "If using the pure iptables proxy, SNAT all traffic sent via Service cluster IPs (this not commonly needed)")
	fs.StringVar(&o.Config.ClusterCIDR, "cluster-cidr", o.Config.ClusterCIDR, "The CIDR range of pods in the cluster. When configured, traffic sent to a Service cluster IP from outside this range will be masqueraded and traffic sent from pods to an external LoadBalancer IP will be directed to the respective cluster IP instead")
	fs.StringVar(&o.Config.ClientConnection.ContentType, "kube-api-content-type", o.Config.ClientConnection.ContentType, "Content type of requests sent to apiserver.")
	fs.Float32Var(&o.Config.ClientConnection.QPS, "kube-api-qps", o.Config.ClientConnection.QPS, "QPS to use while talking with kubernetes apiserver")
	fs.Int32Var(&o.Config.ClientConnection.Burst, "kube-api-burst", o.Config.ClientConnection.Burst, "Burst to use while talking with kubernetes apiserver")
	fs.DurationVar(&o.Config.UDPIdleTimeout.Duration, "udp-timeout", o.Config.UDPIdleTimeout.Duration, "How long an idle UDP connection will be kept open (e.g. '250ms', '2s').  Must be greater than 0. Only applicable for proxy-mode=userspace")
	if o.Config.Conntrack.Max == nil {
		o.Config.Conntrack.Max = utilpointer.Int32Ptr(0)
	}
	fs.Int32Var(o.Config.Conntrack.Max, "conntrack-max", *o.Config.Conntrack.Max,
		"Maximum number of NAT connections to track (0 to leave as-is). This overrides conntrack-max-per-core and conntrack-min.")
	fs.MarkDeprecated("conntrack-max", "This feature will be removed in a later release.")
	fs.Int32Var(o.Config.Conntrack.MaxPerCore, "conntrack-max-per-core", *o.Config.Conntrack.MaxPerCore,
		"Maximum number of NAT connections to track per CPU core (0 to leave the limit as-is and ignore conntrack-min).")
	fs.Int32Var(o.Config.Conntrack.Min, "conntrack-min", *o.Config.Conntrack.Min,
		"Minimum number of conntrack entries to allocate, regardless of conntrack-max-per-core (set conntrack-max-per-core=0 to leave the limit as-is).")
	fs.DurationVar(&o.Config.Conntrack.TCPEstablishedTimeout.Duration, "conntrack-tcp-timeout-established", o.Config.Conntrack.TCPEstablishedTimeout.Duration, "Idle timeout for established TCP connections (0 to leave as-is)")
	fs.DurationVar(
		&o.Config.Conntrack.TCPCloseWaitTimeout.Duration, "conntrack-tcp-timeout-close-wait",
		o.Config.Conntrack.TCPCloseWaitTimeout.Duration,
		"NAT timeout for TCP connections in the CLOSE_WAIT state")
	fs.BoolVar(&o.Config.EnableProfiling, "profiling", o.Config.EnableProfiling, "If true enables profiling via web interface on /debug/pprof handler.")
	fs.StringVar(&o.Config.IPVS.Scheduler, "ipvs-scheduler", o.Config.IPVS.Scheduler, "The ipvs scheduler type when proxy mode is ipvs")
	fs.StringSliceVar(&o.Config.NodePortAddresses, "nodeport-addresses", o.Config.NodePortAddresses,
		"A string slice of values which specify the addresses to use for NodePorts. Values may be valid IP blocks (e.g. 1.2.3.0/24, 1.2.3.4/32). The default empty string slice ([]) means to use all local addresses.")
	fs.Var(flag.NewMapStringBool(&o.Config.FeatureGates), "feature-gates", "A set of key=value pairs that describe feature gates for alpha/experimental features. "+
		"Options are:\n"+strings.Join(utilfeature.DefaultFeatureGate.KnownFeatures(), "\n"))
}

// NewOptions initializes a new Options struct for kube-proxy with
// defaults configured.
func NewOptions() *Options {
	opt := &Options{
		Config:      defaults(),
		HealthzPort: ports.ProxyHealthzPort,
		CleanupIPVS: true,
	}
	return opt
}

func defaults() *kubeproxyconfig.KubeProxyConfiguration {
	// load the scheme defaults
	defaults := &v1alpha1.KubeProxyConfiguration{}
	scheme.Scheme.Default(defaults)
	proxyConfig := &kubeproxyconfig.KubeProxyConfiguration{}
	if err := scheme.Scheme.Convert(defaults, proxyConfig, kubeproxyconfig.SchemeGroupVersion); err != nil {
		panic(err)
	}
	return proxyConfig
}

// WriteConfigFile writes the specified options to WriteConfigTo or
// exits with an error.
func (o *Options) WriteConfigFile() error {
	var encoder runtime.Encoder
	mediaTypes := scheme.Codecs.SupportedMediaTypes()
	for _, info := range mediaTypes {
		if info.MediaType == "application/yaml" {
			encoder = info.Serializer
			break
		}
	}
	if encoder == nil {
		return errors.New("unable to locate yaml encoder")
	}
	encoder = json.NewYAMLSerializer(json.DefaultMetaFactory, scheme.Scheme, scheme.Scheme)
	encoder = scheme.Codecs.EncoderForVersion(encoder, v1alpha1.SchemeGroupVersion)

	configFile, err := os.Create(o.WriteConfigTo)
	if err != nil {
		return err
	}
	defer configFile.Close()

	if err := encoder.Encode(o.Config, configFile); err != nil {
		return err
	}

	glog.Infof("Wrote configuration to: %s\n", o.WriteConfigTo)

	return nil
}

// ApplyDeprecatedHealthzPortToConfig sets o.Config.HealthzBindAddress from
// flags passed on the command line based on the following rules:
//
// 1. If --healthz-port is 0, disable the healthz server.
// 2. Otherwise, use the value of --healthz-port for the port portion of
//    o.Config.HealthzBindAddress
func (o *Options) ApplyDeprecatedHealthzPortToConfig() {
	if o.HealthzPort == 0 {
		o.Config.HealthzBindAddress = ""
		return
	}

	index := strings.Index(o.Config.HealthzBindAddress, ":")
	if index != -1 {
		o.Config.HealthzBindAddress = o.Config.HealthzBindAddress[0:index]
	}

	o.Config.HealthzBindAddress = fmt.Sprintf("%s:%d", o.Config.HealthzBindAddress, o.HealthzPort)
}

// LoadConfigFromFile loads the contents of file and decodes it as a
// KubeProxyConfiguration object.
func (o *Options) LoadConfigFromFile(file string) error {
	data, err := ioutil.ReadFile(file)
	if err != nil {
		return err
	}

	config, err := o.LoadConfig(data)
	if err != nil {
		return err
	}
	o.Config = config
	return nil
}

// LoadConfig decodes data as a KubeProxyConfiguration object.
func (o *Options) LoadConfig(data []byte) (*kubeproxyconfig.KubeProxyConfiguration, error) {
	configObj, gvk, err := scheme.Codecs.UniversalDecoder().Decode(data, nil, nil)
	if err != nil {
		return nil, err
	}
	config, ok := configObj.(*kubeproxyconfig.KubeProxyConfiguration)
	if !ok {
		return nil, fmt.Errorf("got unexpected config type: %v", gvk)
	}
	return config, nil
}
