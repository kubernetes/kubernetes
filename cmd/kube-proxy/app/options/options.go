/*
Copyright 2021 The Kubernetes Authors.

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

package options

import (
	"fmt"
	"github.com/fsnotify/fsnotify"
	"github.com/spf13/pflag"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/serializer"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	cliflag "k8s.io/component-base/cli/flag"
	"k8s.io/klog/v2"
	"k8s.io/kube-proxy/config/v1alpha1"
	"k8s.io/kubernetes/pkg/cluster/ports"
	"k8s.io/kubernetes/pkg/kubelet/qos"
	kubeproxyconfig "k8s.io/kubernetes/pkg/proxy/apis/config"
	proxyconfigscheme "k8s.io/kubernetes/pkg/proxy/apis/config/scheme"
	kubeproxyconfigv1alpha1 "k8s.io/kubernetes/pkg/proxy/apis/config/v1alpha1"
	"k8s.io/kubernetes/pkg/proxy/apis/config/validation"
	proxyutil "k8s.io/kubernetes/pkg/proxy/util"
	"k8s.io/kubernetes/pkg/util/filesystem"
	utilflag "k8s.io/kubernetes/pkg/util/flag"
	utilpointer "k8s.io/utils/pointer"
	"strings"
)

// proxyRun defines the interface to run a specified ProxyServer
type proxyRun interface {
	Run() error
	CleanupAndExit() error
}

// Options contains everything necessary to create and run a proxy server.
type Options struct {
	// ConfigFile is the location of the proxy server's configuration file.
	ConfigFile string
	// WriteConfigTo is the path where the default configuration will be written.
	WriteConfigTo string
	// CleanupAndExit, when true, makes the proxy server clean up iptables and ipvs rules, then exit.
	CleanupAndExit bool
	// WindowsService should be set to true if kube-proxy is running as a service on Windows.
	// Its corresponding flag only gets registered in Windows builds
	WindowsService bool
	// config is the proxy server's configuration object.
	ComponentConfig *kubeproxyconfig.KubeProxyConfiguration
	// watcher is used to watch on the update change of ConfigFile
	watcher filesystem.FSWatcher
	// proxyServer is the interface to run the proxy server
	ProxyServer proxyRun
	// errCh is the channel that errors will be sent
	ErrCh chan error

	// The fields below here are placeholders for flags that can't be directly mapped into
	// config.KubeProxyConfiguration.
	//
	// TODO remove these fields once the deprecated flags are removed.

	// master is used to override the kubeconfig's URL to the apiserver.
	Master string
	// healthzPort is the port to be used by the healthz server.
	healthzPort int32
	// metricsPort is the port to be used by the metrics server.
	metricsPort int32

	// hostnameOverride, if set from the command line flag, takes precedence over the `HostnameOverride` value from the config file
	hostnameOverride string
}

// NewOptions returns initialized Options
func NewOptions() *Options {
	return &Options{
		ComponentConfig: new(kubeproxyconfig.KubeProxyConfiguration), // new defaulted configuration
		healthzPort:     ports.ProxyHealthzPort,
		metricsPort:     ports.ProxyStatusPort,
		ErrCh:           make(chan error),
	}
}

// Validate validates all the required options.
func (o *Options) Validate() error {
	if errs := validation.Validate(o.ComponentConfig); len(errs) != 0 {
		return errs.ToAggregate()
	}

	return nil
}

// ApplyDefaults applies the default values to Options.
func (o *Options) ApplyDefaults(in *kubeproxyconfig.KubeProxyConfiguration) (*kubeproxyconfig.KubeProxyConfiguration, error) {
	external, err := proxyconfigscheme.Scheme.ConvertToVersion(in, v1alpha1.SchemeGroupVersion)
	if err != nil {
		return nil, err
	}

	proxyconfigscheme.Scheme.Default(external)

	internal, err := proxyconfigscheme.Scheme.ConvertToVersion(external, kubeproxyconfig.SchemeGroupVersion)
	if err != nil {
		return nil, err
	}

	out := internal.(*kubeproxyconfig.KubeProxyConfiguration)

	return out, nil
}

// AddFlags adds flags to fs and binds them to options.
func (o *Options) AddFlags(mainfs *pflag.FlagSet) {
	fs := pflag.NewFlagSet("", pflag.ExitOnError)
	defer func() {
		// Unhide deprecated flags. We want deprecated flags to show in Kube-proxy help.
		// We have some hidden flags, but we might as well unhide these when they are deprecated,
		// as silently deprecating and removing (even hidden) things is unkind to people who use them.
		fs.VisitAll(func(f *pflag.Flag) {
			if len(f.Deprecated) > 0 {
				f.Hidden = false
			}
		})
		mainfs.AddFlagSet(fs)
	}()

	o.addOSFlags(fs)

	fs.StringVar(&o.ConfigFile, "config", o.ConfigFile, "The path to the configuration file.")
	fs.StringVar(&o.WriteConfigTo, "write-config-to", o.WriteConfigTo, "If set, write the default configuration values to this file and exit.")
	fs.StringVar(&o.ComponentConfig.ClientConnection.Kubeconfig, "kubeconfig", o.ComponentConfig.ClientConnection.Kubeconfig, "Path to kubeconfig file with authorization information (the master location can be overridden by the master flag).")
	fs.StringVar(&o.ComponentConfig.ClusterCIDR, "cluster-cidr", o.ComponentConfig.ClusterCIDR, "The CIDR range of pods in the cluster. When configured, traffic sent to a Service cluster IP from outside this range will be masqueraded and traffic sent from pods to an external LoadBalancer IP will be directed to the respective cluster IP instead")
	fs.StringVar(&o.ComponentConfig.ClientConnection.ContentType, "kube-api-content-type", o.ComponentConfig.ClientConnection.ContentType, "Content type of requests sent to apiserver.")
	fs.StringVar(&o.Master, "master", o.Master, "The address of the Kubernetes API server (overrides any value in kubeconfig)")
	fs.StringVar(&o.hostnameOverride, "hostname-override", o.hostnameOverride, "If non-empty, will use this string as identification instead of the actual hostname.")
	fs.StringVar(&o.ComponentConfig.IPVS.Scheduler, "ipvs-scheduler", o.ComponentConfig.IPVS.Scheduler, "The ipvs scheduler type when proxy mode is ipvs")
	fs.StringVar(&o.ComponentConfig.ShowHiddenMetricsForVersion, "show-hidden-metrics-for-version", o.ComponentConfig.ShowHiddenMetricsForVersion,
		"The previous version for which you want to show hidden metrics. "+
			"Only the previous minor version is meaningful, other values will not be allowed. "+
			"The format is <major>.<minor>, e.g.: '1.16'. "+
			"The purpose of this format is make sure you have the opportunity to notice if the next release hides additional metrics, "+
			"rather than being surprised when they are permanently removed in the release after that.")

	fs.StringSliceVar(&o.ComponentConfig.IPVS.ExcludeCIDRs, "ipvs-exclude-cidrs", o.ComponentConfig.IPVS.ExcludeCIDRs, "A comma-separated list of CIDR's which the ipvs proxier should not touch when cleaning up IPVS rules.")
	fs.StringSliceVar(&o.ComponentConfig.NodePortAddresses, "nodeport-addresses", o.ComponentConfig.NodePortAddresses,
		"A string slice of values which specify the addresses to use for NodePorts. Values may be valid IP blocks (e.g. 1.2.3.0/24, 1.2.3.4/32). The default empty string slice ([]) means to use all local addresses.")

	fs.BoolVar(&o.CleanupAndExit, "cleanup", o.CleanupAndExit, "If true cleanup iptables and ipvs rules and exit.")

	fs.Var(utilflag.IPVar{Val: &o.ComponentConfig.BindAddress}, "bind-address", "The IP address for the proxy server to serve on (set to '0.0.0.0' for all IPv4 interfaces and '::' for all IPv6 interfaces)")
	fs.Var(utilflag.IPPortVar{Val: &o.ComponentConfig.HealthzBindAddress}, "healthz-bind-address", "The IP address with port for the health check server to serve on (set to '0.0.0.0:10256' for all IPv4 interfaces and '[::]:10256' for all IPv6 interfaces). Set empty to disable.")
	fs.Var(utilflag.IPPortVar{Val: &o.ComponentConfig.MetricsBindAddress}, "metrics-bind-address", "The IP address with port for the metrics server to serve on (set to '0.0.0.0:10249' for all IPv4 interfaces and '[::]:10249' for all IPv6 interfaces). Set empty to disable.")
	fs.BoolVar(&o.ComponentConfig.BindAddressHardFail, "bind-address-hard-fail", o.ComponentConfig.BindAddressHardFail, "If true kube-proxy will treat failure to bind to a port as fatal and exit")
	fs.Var(utilflag.PortRangeVar{Val: &o.ComponentConfig.PortRange}, "proxy-port-range", "Range of host ports (beginPort-endPort, single port or beginPort+offset, inclusive) that may be consumed in order to proxy service traffic. If (unspecified, 0, or 0-0) then ports will be randomly chosen.")
	fs.Var(&o.ComponentConfig.Mode, "proxy-mode", "Which proxy mode to use: 'userspace' (older) or 'iptables' (faster) or 'ipvs' or 'kernelspace' (windows). If blank, use the best-available proxy (currently iptables). If the iptables proxy is selected, regardless of how, but the system's kernel or iptables versions are insufficient, this always falls back to the userspace proxy.")
	fs.Var(cliflag.NewMapStringBool(&o.ComponentConfig.FeatureGates), "feature-gates", "A set of key=value pairs that describe feature gates for alpha/experimental features. "+
		"Options are:\n"+strings.Join(utilfeature.DefaultFeatureGate.KnownFeatures(), "\n"))

	fs.Int32Var(&o.healthzPort, "healthz-port", o.healthzPort, "The port to bind the health check server. Use 0 to disable.")
	fs.MarkDeprecated("healthz-port", "This flag is deprecated and will be removed in a future release. Please use --healthz-bind-address instead.")
	fs.Int32Var(&o.metricsPort, "metrics-port", o.metricsPort, "The port to bind the metrics server. Use 0 to disable.")
	fs.MarkDeprecated("metrics-port", "This flag is deprecated and will be removed in a future release. Please use --metrics-bind-address instead.")
	fs.Int32Var(o.ComponentConfig.OOMScoreAdj, "oom-score-adj", utilpointer.Int32PtrDerefOr(o.ComponentConfig.OOMScoreAdj, int32(qos.KubeProxyOOMScoreAdj)), "The oom-score-adj value for kube-proxy process. Values must be within the range [-1000, 1000]")
	fs.Int32Var(o.ComponentConfig.IPTables.MasqueradeBit, "iptables-masquerade-bit", utilpointer.Int32PtrDerefOr(o.ComponentConfig.IPTables.MasqueradeBit, 14), "If using the pure iptables proxy, the bit of the fwmark space to mark packets requiring SNAT with.  Must be within the range [0, 31].")
	fs.Int32Var(o.ComponentConfig.Conntrack.MaxPerCore, "conntrack-max-per-core", *o.ComponentConfig.Conntrack.MaxPerCore,
		"Maximum number of NAT connections to track per CPU core (0 to leave the limit as-is and ignore conntrack-min).")
	fs.Int32Var(o.ComponentConfig.Conntrack.Min, "conntrack-min", *o.ComponentConfig.Conntrack.Min,
		"Minimum number of conntrack entries to allocate, regardless of conntrack-max-per-core (set conntrack-max-per-core=0 to leave the limit as-is).")
	fs.Int32Var(&o.ComponentConfig.ClientConnection.Burst, "kube-api-burst", o.ComponentConfig.ClientConnection.Burst, "Burst to use while talking with kubernetes apiserver")

	fs.DurationVar(&o.ComponentConfig.IPTables.SyncPeriod.Duration, "iptables-sync-period", o.ComponentConfig.IPTables.SyncPeriod.Duration, "The maximum interval of how often iptables rules are refreshed (e.g. '5s', '1m', '2h22m').  Must be greater than 0.")
	fs.DurationVar(&o.ComponentConfig.IPTables.MinSyncPeriod.Duration, "iptables-min-sync-period", o.ComponentConfig.IPTables.MinSyncPeriod.Duration, "The minimum interval of how often the iptables rules can be refreshed as endpoints and services change (e.g. '5s', '1m', '2h22m').")
	fs.DurationVar(&o.ComponentConfig.IPVS.SyncPeriod.Duration, "ipvs-sync-period", o.ComponentConfig.IPVS.SyncPeriod.Duration, "The maximum interval of how often ipvs rules are refreshed (e.g. '5s', '1m', '2h22m').  Must be greater than 0.")
	fs.DurationVar(&o.ComponentConfig.IPVS.MinSyncPeriod.Duration, "ipvs-min-sync-period", o.ComponentConfig.IPVS.MinSyncPeriod.Duration, "The minimum interval of how often the ipvs rules can be refreshed as endpoints and services change (e.g. '5s', '1m', '2h22m').")
	fs.DurationVar(&o.ComponentConfig.IPVS.TCPTimeout.Duration, "ipvs-tcp-timeout", o.ComponentConfig.IPVS.TCPTimeout.Duration, "The timeout for idle IPVS TCP connections, 0 to leave as-is. (e.g. '5s', '1m', '2h22m').")
	fs.DurationVar(&o.ComponentConfig.IPVS.TCPFinTimeout.Duration, "ipvs-tcpfin-timeout", o.ComponentConfig.IPVS.TCPFinTimeout.Duration, "The timeout for IPVS TCP connections after receiving a FIN packet, 0 to leave as-is. (e.g. '5s', '1m', '2h22m').")
	fs.DurationVar(&o.ComponentConfig.IPVS.UDPTimeout.Duration, "ipvs-udp-timeout", o.ComponentConfig.IPVS.UDPTimeout.Duration, "The timeout for IPVS UDP packets, 0 to leave as-is. (e.g. '5s', '1m', '2h22m').")
	fs.DurationVar(&o.ComponentConfig.Conntrack.TCPEstablishedTimeout.Duration, "conntrack-tcp-timeout-established", o.ComponentConfig.Conntrack.TCPEstablishedTimeout.Duration, "Idle timeout for established TCP connections (0 to leave as-is)")
	fs.DurationVar(
		&o.ComponentConfig.Conntrack.TCPCloseWaitTimeout.Duration, "conntrack-tcp-timeout-close-wait",
		o.ComponentConfig.Conntrack.TCPCloseWaitTimeout.Duration,
		"NAT timeout for TCP connections in the CLOSE_WAIT state")
	fs.DurationVar(&o.ComponentConfig.ConfigSyncPeriod.Duration, "config-sync-period", o.ComponentConfig.ConfigSyncPeriod.Duration, "How often configuration from the apiserver is refreshed.  Must be greater than 0.")
	fs.DurationVar(&o.ComponentConfig.UDPIdleTimeout.Duration, "udp-timeout", o.ComponentConfig.UDPIdleTimeout.Duration, "How long an idle UDP connection will be kept open (e.g. '250ms', '2s').  Must be greater than 0. Only applicable for proxy-mode=userspace")

	fs.BoolVar(&o.ComponentConfig.IPVS.StrictARP, "ipvs-strict-arp", o.ComponentConfig.IPVS.StrictARP, "Enable strict ARP by setting arp_ignore to 1 and arp_announce to 2")
	fs.BoolVar(&o.ComponentConfig.IPTables.MasqueradeAll, "masquerade-all", o.ComponentConfig.IPTables.MasqueradeAll, "If using the pure iptables proxy, SNAT all traffic sent via Service cluster IPs (this not commonly needed)")
	fs.BoolVar(&o.ComponentConfig.EnableProfiling, "profiling", o.ComponentConfig.EnableProfiling, "If true enables profiling via web interface on /debug/pprof handler.")

	fs.Float32Var(&o.ComponentConfig.ClientConnection.QPS, "kube-api-qps", o.ComponentConfig.ClientConnection.QPS, "QPS to use while talking with kubernetes apiserver")
	fs.Var(&o.ComponentConfig.DetectLocalMode, "detect-local-mode", "Mode to use to detect local traffic")
}

// Complete completes all the required options.
func (o *Options) Complete() error {
	if len(o.ConfigFile) == 0 && len(o.WriteConfigTo) == 0 {
		klog.Warning("WARNING: all flags other than --config, --write-config-to, and --cleanup are deprecated. Please begin using a config file ASAP.")
		o.ComponentConfig.HealthzBindAddress = addressFromDeprecatedFlags(o.ComponentConfig.HealthzBindAddress, o.healthzPort)
		o.ComponentConfig.MetricsBindAddress = addressFromDeprecatedFlags(o.ComponentConfig.MetricsBindAddress, o.metricsPort)
	}

	// Load the config file here in Complete, so that Validate validates the fully-resolved config.
	if len(o.ConfigFile) > 0 {
		c, err := o.loadConfigFromFile(o.ConfigFile)
		if err != nil {
			return err
		}
		o.ComponentConfig = c

		if err := o.initWatcher(); err != nil {
			return err
		}
	}

	if err := o.processHostnameOverrideFlag(); err != nil {
		return err
	}

	return utilfeature.DefaultMutableFeatureGate.SetFromMap(o.ComponentConfig.FeatureGates)
}

func (o *Options) eventHandler(ent fsnotify.Event) {
	eventOpIs := func(Op fsnotify.Op) bool {
		return ent.Op&Op == Op
	}
	if eventOpIs(fsnotify.Write) || eventOpIs(fsnotify.Rename) {
		// error out when ConfigFile is updated
		o.ErrCh <- fmt.Errorf("content of the proxy server's configuration file was updated")
		return
	}
	o.ErrCh <- nil
}

func (o *Options) errorHandler(err error) {
	o.ErrCh <- err
}

// Creates a new filesystem watcher and adds watches for the config file.
func (o *Options) initWatcher() error {
	fswatcher := filesystem.NewFsnotifyWatcher()
	err := fswatcher.Init(o.eventHandler, o.errorHandler)
	if err != nil {
		return err
	}
	err = fswatcher.AddWatch(o.ConfigFile)
	if err != nil {
		return err
	}
	o.watcher = fswatcher
	return nil
}

// processHostnameOverrideFlag processes hostname-override flag
func (o *Options) processHostnameOverrideFlag() error {
	// Check if hostname-override flag is set and use value since configFile always overrides
	if len(o.hostnameOverride) > 0 {
		hostName := strings.TrimSpace(o.hostnameOverride)
		if len(hostName) == 0 {
			return fmt.Errorf("empty hostname-override is invalid")
		}
		o.ComponentConfig.HostnameOverride = strings.ToLower(hostName)
	}

	return nil
}

// runLoop will watch on the update change of the proxy server's configuration file.
// Return an error when updated
func (o *Options) RunLoop() error {
	if o.watcher != nil {
		o.watcher.Run()
	}

	// run the proxy in goroutine
	go func() {
		err := o.ProxyServer.Run()
		o.ErrCh <- err
	}()

	for {
		err := <-o.ErrCh
		if err != nil {
			return err
		}
	}
}

// addressFromDeprecatedFlags returns server address from flags
// passed on the command line based on the following rules:
// 1. If port is 0, disable the server (e.g. set address to empty).
// 2. Otherwise, set the port portion of the config accordingly.
func addressFromDeprecatedFlags(addr string, port int32) string {
	if port == 0 {
		return ""
	}
	return proxyutil.AppendPortIfNeeded(addr, port)
}

// newLenientSchemeAndCodecs returns a scheme that has only v1alpha1 registered into
// it and a CodecFactory with strict decoding disabled.
func newLenientSchemeAndCodecs() (*runtime.Scheme, *serializer.CodecFactory, error) {
	lenientScheme := runtime.NewScheme()
	if err := kubeproxyconfig.AddToScheme(lenientScheme); err != nil {
		return nil, nil, fmt.Errorf("failed to add kube-proxy config API to lenient scheme: %v", err)
	}
	if err := kubeproxyconfigv1alpha1.AddToScheme(lenientScheme); err != nil {
		return nil, nil, fmt.Errorf("failed to add kube-proxy config v1alpha1 API to lenient scheme: %v", err)
	}
	lenientCodecs := serializer.NewCodecFactory(lenientScheme, serializer.DisableStrict)
	return lenientScheme, &lenientCodecs, nil
}
