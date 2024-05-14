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
	"context"
	goflag "flag"
	"fmt"
	"net"
	"net/http"
	"os"
	"strings"
	"time"

	"github.com/fsnotify/fsnotify"
	"github.com/spf13/cobra"
	"github.com/spf13/pflag"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/serializer"
	"k8s.io/apimachinery/pkg/selection"
	"k8s.io/apimachinery/pkg/types"
	utilerrors "k8s.io/apimachinery/pkg/util/errors"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apiserver/pkg/server/healthz"
	"k8s.io/apiserver/pkg/server/mux"
	"k8s.io/apiserver/pkg/server/routes"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/client-go/informers"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/client-go/rest"
	"k8s.io/client-go/tools/clientcmd"
	clientcmdapi "k8s.io/client-go/tools/clientcmd/api"
	"k8s.io/client-go/tools/events"
	cliflag "k8s.io/component-base/cli/flag"
	componentbaseconfig "k8s.io/component-base/config"
	"k8s.io/component-base/configz"
	"k8s.io/component-base/logs"
	logsapi "k8s.io/component-base/logs/api/v1"
	"k8s.io/component-base/metrics"
	metricsfeatures "k8s.io/component-base/metrics/features"
	"k8s.io/component-base/metrics/legacyregistry"
	"k8s.io/component-base/metrics/prometheus/slis"
	"k8s.io/component-base/version"
	"k8s.io/component-base/version/verflag"
	nodeutil "k8s.io/component-helpers/node/util"
	"k8s.io/klog/v2"
	"k8s.io/kube-proxy/config/v1alpha1"
	api "k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/kubernetes/pkg/cluster/ports"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/kubelet/qos"
	"k8s.io/kubernetes/pkg/proxy"
	"k8s.io/kubernetes/pkg/proxy/apis"
	kubeproxyconfig "k8s.io/kubernetes/pkg/proxy/apis/config"
	proxyconfigscheme "k8s.io/kubernetes/pkg/proxy/apis/config/scheme"
	kubeproxyconfigv1alpha1 "k8s.io/kubernetes/pkg/proxy/apis/config/v1alpha1"
	"k8s.io/kubernetes/pkg/proxy/apis/config/validation"
	"k8s.io/kubernetes/pkg/proxy/config"
	"k8s.io/kubernetes/pkg/proxy/healthcheck"
	proxymetrics "k8s.io/kubernetes/pkg/proxy/metrics"
	proxyutil "k8s.io/kubernetes/pkg/proxy/util"
	"k8s.io/kubernetes/pkg/util/filesystem"
	utilflag "k8s.io/kubernetes/pkg/util/flag"
	utilnode "k8s.io/kubernetes/pkg/util/node"
	"k8s.io/kubernetes/pkg/util/oom"
	netutils "k8s.io/utils/net"
	"k8s.io/utils/ptr"
)

func init() {
	utilruntime.Must(metricsfeatures.AddFeatureGates(utilfeature.DefaultMutableFeatureGate))
	utilruntime.Must(logsapi.AddFeatureGates(utilfeature.DefaultMutableFeatureGate))
}

// proxyRun defines the interface to run a specified ProxyServer
type proxyRun interface {
	Run(ctx context.Context) error
}

// Options contains everything necessary to create and run a proxy server.
type Options struct {
	// ConfigFile is the location of the proxy server's configuration file.
	ConfigFile string
	// WriteConfigTo is the path where the default configuration will be written.
	WriteConfigTo string
	// CleanupAndExit, when true, makes the proxy server clean up iptables and ipvs rules, then exit.
	CleanupAndExit bool
	// InitAndExit, when true, makes the proxy server makes configurations that need privileged access, then exit.
	InitAndExit bool
	// WindowsService should be set to true if kube-proxy is running as a service on Windows.
	// Its corresponding flag only gets registered in Windows builds
	WindowsService bool
	// config is the proxy server's configuration object.
	config *kubeproxyconfig.KubeProxyConfiguration
	// watcher is used to watch on the update change of ConfigFile
	watcher filesystem.FSWatcher
	// proxyServer is the interface to run the proxy server
	proxyServer proxyRun
	// errCh is the channel that errors will be sent
	errCh chan error

	// The fields below here are placeholders for flags that can't be directly mapped into
	// config.KubeProxyConfiguration.
	//
	// TODO remove these fields once the deprecated flags are removed.

	// master is used to override the kubeconfig's URL to the apiserver.
	master string
	// healthzPort is the port to be used by the healthz server.
	healthzPort int32
	// metricsPort is the port to be used by the metrics server.
	metricsPort int32

	// hostnameOverride, if set from the command line flag, takes precedence over the `HostnameOverride` value from the config file
	hostnameOverride string

	logger klog.Logger
}

// AddFlags adds flags to fs and binds them to options.
func (o *Options) AddFlags(fs *pflag.FlagSet) {
	o.addOSFlags(fs)

	fs.StringVar(&o.ConfigFile, "config", o.ConfigFile, "The path to the configuration file.")
	fs.StringVar(&o.WriteConfigTo, "write-config-to", o.WriteConfigTo, "If set, write the default configuration values to this file and exit.")

	fs.BoolVar(&o.CleanupAndExit, "cleanup", o.CleanupAndExit, "If true cleanup iptables and ipvs rules and exit.")

	fs.Var(cliflag.NewMapStringBool(&o.config.FeatureGates), "feature-gates", "A set of key=value pairs that describe feature gates for alpha/experimental features. "+
		"Options are:\n"+strings.Join(utilfeature.DefaultFeatureGate.KnownFeatures(), "\n")+"\n"+
		"This parameter is ignored if a config file is specified by --config.")

	fs.StringVar(&o.config.ClientConnection.Kubeconfig, "kubeconfig", o.config.ClientConnection.Kubeconfig, "Path to kubeconfig file with authorization information (the master location can be overridden by the master flag).")
	fs.StringVar(&o.master, "master", o.master, "The address of the Kubernetes API server (overrides any value in kubeconfig)")
	fs.StringVar(&o.config.ClientConnection.ContentType, "kube-api-content-type", o.config.ClientConnection.ContentType, "Content type of requests sent to apiserver.")
	fs.Int32Var(&o.config.ClientConnection.Burst, "kube-api-burst", o.config.ClientConnection.Burst, "Burst to use while talking with kubernetes apiserver")
	fs.Float32Var(&o.config.ClientConnection.QPS, "kube-api-qps", o.config.ClientConnection.QPS, "QPS to use while talking with kubernetes apiserver")

	fs.StringVar(&o.hostnameOverride, "hostname-override", o.hostnameOverride, "If non-empty, will be used as the name of the Node that kube-proxy is running on. If unset, the node name is assumed to be the same as the node's hostname.")
	fs.Var(&utilflag.IPVar{Val: &o.config.BindAddress}, "bind-address", "Overrides kube-proxy's idea of what its node's primary IP is. Note that the name is a historical artifact, and kube-proxy does not actually bind any sockets to this IP. This parameter is ignored if a config file is specified by --config.")
	fs.Var(&utilflag.IPPortVar{Val: &o.config.HealthzBindAddress}, "healthz-bind-address", "The IP address and port for the health check server to serve on, defaulting to \"0.0.0.0:10256\". This parameter is ignored if a config file is specified by --config.")
	fs.Var(&utilflag.IPPortVar{Val: &o.config.MetricsBindAddress}, "metrics-bind-address", "The IP address and port for the metrics server to serve on, defaulting to \"127.0.0.1:10249\". (Set to \"0.0.0.0:10249\" / \"[::]:10249\" to bind on all interfaces.) Set empty to disable. This parameter is ignored if a config file is specified by --config.")
	fs.BoolVar(&o.config.BindAddressHardFail, "bind-address-hard-fail", o.config.BindAddressHardFail, "If true kube-proxy will treat failure to bind to a port as fatal and exit")
	fs.BoolVar(&o.config.EnableProfiling, "profiling", o.config.EnableProfiling, "If true enables profiling via web interface on /debug/pprof handler. This parameter is ignored if a config file is specified by --config.")
	fs.StringVar(&o.config.ShowHiddenMetricsForVersion, "show-hidden-metrics-for-version", o.config.ShowHiddenMetricsForVersion,
		"The previous version for which you want to show hidden metrics. "+
			"Only the previous minor version is meaningful, other values will not be allowed. "+
			"The format is <major>.<minor>, e.g.: '1.16'. "+
			"The purpose of this format is make sure you have the opportunity to notice if the next release hides additional metrics, "+
			"rather than being surprised when they are permanently removed in the release after that. "+
			"This parameter is ignored if a config file is specified by --config.")
	fs.BoolVar(&o.InitAndExit, "init-only", o.InitAndExit, "If true, perform any initialization steps that must be done with full root privileges, and then exit. After doing this, you can run kube-proxy again with only the CAP_NET_ADMIN capability.")
	fs.Var(&o.config.Mode, "proxy-mode", "Which proxy mode to use: on Linux this can be 'iptables' (default) or 'ipvs'. On Windows the only supported value is 'kernelspace'."+
		"This parameter is ignored if a config file is specified by --config.")

	fs.Int32Var(o.config.IPTables.MasqueradeBit, "iptables-masquerade-bit", ptr.Deref(o.config.IPTables.MasqueradeBit, 14), "If using the iptables or ipvs proxy mode, the bit of the fwmark space to mark packets requiring SNAT with.  Must be within the range [0, 31].")
	fs.BoolVar(&o.config.IPTables.MasqueradeAll, "masquerade-all", o.config.IPTables.MasqueradeAll, "If using the iptables or ipvs proxy mode, SNAT all traffic sent via Service cluster IPs. This may be required with some CNI plugins.")
	fs.BoolVar(o.config.IPTables.LocalhostNodePorts, "iptables-localhost-nodeports", ptr.Deref(o.config.IPTables.LocalhostNodePorts, true), "If false, kube-proxy will disable the legacy behavior of allowing NodePort services to be accessed via localhost. (Applies only to iptables mode and IPv4; localhost NodePorts are never allowed with other proxy modes or with IPv6.)")
	fs.DurationVar(&o.config.IPTables.SyncPeriod.Duration, "iptables-sync-period", o.config.IPTables.SyncPeriod.Duration, "An interval (e.g. '5s', '1m', '2h22m') indicating how frequently various re-synchronizing and cleanup operations are performed. Must be greater than 0.")
	fs.DurationVar(&o.config.IPTables.MinSyncPeriod.Duration, "iptables-min-sync-period", o.config.IPTables.MinSyncPeriod.Duration, "The minimum period between iptables rule resyncs (e.g. '5s', '1m', '2h22m'). A value of 0 means every Service or EndpointSlice change will result in an immediate iptables resync.")

	fs.DurationVar(&o.config.IPVS.SyncPeriod.Duration, "ipvs-sync-period", o.config.IPVS.SyncPeriod.Duration, "An interval (e.g. '5s', '1m', '2h22m') indicating how frequently various re-synchronizing and cleanup operations are performed. Must be greater than 0.")
	fs.DurationVar(&o.config.IPVS.MinSyncPeriod.Duration, "ipvs-min-sync-period", o.config.IPVS.MinSyncPeriod.Duration, "The minimum period between IPVS rule resyncs (e.g. '5s', '1m', '2h22m'). A value of 0 means every Service or EndpointSlice change will result in an immediate IPVS resync.")
	fs.StringVar(&o.config.IPVS.Scheduler, "ipvs-scheduler", o.config.IPVS.Scheduler, "The ipvs scheduler type when proxy mode is ipvs")
	fs.StringSliceVar(&o.config.IPVS.ExcludeCIDRs, "ipvs-exclude-cidrs", o.config.IPVS.ExcludeCIDRs, "A comma-separated list of CIDRs which the ipvs proxier should not touch when cleaning up IPVS rules.")
	fs.BoolVar(&o.config.IPVS.StrictARP, "ipvs-strict-arp", o.config.IPVS.StrictARP, "Enable strict ARP by setting arp_ignore to 1 and arp_announce to 2")
	fs.DurationVar(&o.config.IPVS.TCPTimeout.Duration, "ipvs-tcp-timeout", o.config.IPVS.TCPTimeout.Duration, "The timeout for idle IPVS TCP connections, 0 to leave as-is. (e.g. '5s', '1m', '2h22m').")
	fs.DurationVar(&o.config.IPVS.TCPFinTimeout.Duration, "ipvs-tcpfin-timeout", o.config.IPVS.TCPFinTimeout.Duration, "The timeout for IPVS TCP connections after receiving a FIN packet, 0 to leave as-is. (e.g. '5s', '1m', '2h22m').")
	fs.DurationVar(&o.config.IPVS.UDPTimeout.Duration, "ipvs-udp-timeout", o.config.IPVS.UDPTimeout.Duration, "The timeout for IPVS UDP packets, 0 to leave as-is. (e.g. '5s', '1m', '2h22m').")

	fs.Var(&o.config.DetectLocalMode, "detect-local-mode", "Mode to use to detect local traffic. This parameter is ignored if a config file is specified by --config.")
	fs.StringVar(&o.config.DetectLocal.BridgeInterface, "pod-bridge-interface", o.config.DetectLocal.BridgeInterface, "A bridge interface name. When --detect-local-mode is set to BridgeInterface, kube-proxy will consider traffic to be local if it originates from this bridge.")
	fs.StringVar(&o.config.DetectLocal.InterfaceNamePrefix, "pod-interface-name-prefix", o.config.DetectLocal.InterfaceNamePrefix, "An interface name prefix. When --detect-local-mode is set to InterfaceNamePrefix, kube-proxy will consider traffic to be local if it originates from any interface whose name begins with this prefix.")
	fs.StringVar(&o.config.ClusterCIDR, "cluster-cidr", o.config.ClusterCIDR, "The CIDR range of the pods in the cluster. (For dual-stack clusters, this can be a comma-separated dual-stack pair of CIDR ranges.). When --detect-local-mode is set to ClusterCIDR, kube-proxy will consider traffic to be local if its source IP is in this range. (Otherwise it is not used.) "+
		"This parameter is ignored if a config file is specified by --config.")

	fs.StringSliceVar(&o.config.NodePortAddresses, "nodeport-addresses", o.config.NodePortAddresses,
		"A list of CIDR ranges that contain valid node IPs, or alternatively, the single string 'primary'. If set to a list of CIDRs, connections to NodePort services will only be accepted on node IPs in one of the indicated ranges. If set to 'primary', NodePort services will only be accepted on the node's primary IP(s) according to the Node object. If unset, NodePort connections will be accepted on all local IPs. This parameter is ignored if a config file is specified by --config.")

	fs.Int32Var(o.config.OOMScoreAdj, "oom-score-adj", ptr.Deref(o.config.OOMScoreAdj, int32(qos.KubeProxyOOMScoreAdj)), "The oom-score-adj value for kube-proxy process. Values must be within the range [-1000, 1000]. This parameter is ignored if a config file is specified by --config.")
	fs.Int32Var(o.config.Conntrack.MaxPerCore, "conntrack-max-per-core", *o.config.Conntrack.MaxPerCore,
		"Maximum number of NAT connections to track per CPU core (0 to leave the limit as-is and ignore conntrack-min).")
	fs.Int32Var(o.config.Conntrack.Min, "conntrack-min", *o.config.Conntrack.Min,
		"Minimum number of conntrack entries to allocate, regardless of conntrack-max-per-core (set conntrack-max-per-core=0 to leave the limit as-is).")

	fs.DurationVar(&o.config.Conntrack.TCPEstablishedTimeout.Duration, "conntrack-tcp-timeout-established", o.config.Conntrack.TCPEstablishedTimeout.Duration, "Idle timeout for established TCP connections (0 to leave as-is)")
	fs.DurationVar(
		&o.config.Conntrack.TCPCloseWaitTimeout.Duration, "conntrack-tcp-timeout-close-wait",
		o.config.Conntrack.TCPCloseWaitTimeout.Duration,
		"NAT timeout for TCP connections in the CLOSE_WAIT state")
	fs.BoolVar(&o.config.Conntrack.TCPBeLiberal, "conntrack-tcp-be-liberal", o.config.Conntrack.TCPBeLiberal, "Enable liberal mode for tracking TCP packets by setting nf_conntrack_tcp_be_liberal to 1")
	fs.DurationVar(&o.config.Conntrack.UDPTimeout.Duration, "conntrack-udp-timeout", o.config.Conntrack.UDPTimeout.Duration, "Idle timeout for UNREPLIED UDP connections (0 to leave as-is)")
	fs.DurationVar(&o.config.Conntrack.UDPStreamTimeout.Duration, "conntrack-udp-timeout-stream", o.config.Conntrack.UDPStreamTimeout.Duration, "Idle timeout for ASSURED UDP connections (0 to leave as-is)")

	fs.DurationVar(&o.config.ConfigSyncPeriod.Duration, "config-sync-period", o.config.ConfigSyncPeriod.Duration, "How often configuration from the apiserver is refreshed.  Must be greater than 0.")

	fs.Int32Var(&o.healthzPort, "healthz-port", o.healthzPort, "The port to bind the health check server. Use 0 to disable.")
	_ = fs.MarkDeprecated("healthz-port", "This flag is deprecated and will be removed in a future release. Please use --healthz-bind-address instead.")
	fs.Int32Var(&o.metricsPort, "metrics-port", o.metricsPort, "The port to bind the metrics server. Use 0 to disable.")
	_ = fs.MarkDeprecated("metrics-port", "This flag is deprecated and will be removed in a future release. Please use --metrics-bind-address instead.")
	fs.Var(utilflag.PortRangeVar{Val: &o.config.PortRange}, "proxy-port-range", "This was previously used to configure the userspace proxy, but is now unused.")
	_ = fs.MarkDeprecated("proxy-port-range", "This flag has no effect and will be removed in a future release.")

	logsapi.AddFlags(&o.config.Logging, fs)
}

// newKubeProxyConfiguration returns a KubeProxyConfiguration with default values
func newKubeProxyConfiguration() *kubeproxyconfig.KubeProxyConfiguration {
	versionedConfig := &v1alpha1.KubeProxyConfiguration{}
	proxyconfigscheme.Scheme.Default(versionedConfig)
	internalConfig, err := proxyconfigscheme.Scheme.ConvertToVersion(versionedConfig, kubeproxyconfig.SchemeGroupVersion)
	if err != nil {
		panic(fmt.Sprintf("Unable to create default config: %v", err))
	}

	return internalConfig.(*kubeproxyconfig.KubeProxyConfiguration)
}

// NewOptions returns initialized Options
func NewOptions() *Options {
	return &Options{
		config:      newKubeProxyConfiguration(),
		healthzPort: ports.ProxyHealthzPort,
		metricsPort: ports.ProxyStatusPort,
		errCh:       make(chan error),
		logger:      klog.FromContext(context.Background()),
	}
}

// Complete completes all the required options.
func (o *Options) Complete(fs *pflag.FlagSet) error {
	if len(o.ConfigFile) == 0 && len(o.WriteConfigTo) == 0 {
		o.config.HealthzBindAddress = addressFromDeprecatedFlags(o.config.HealthzBindAddress, o.healthzPort)
		o.config.MetricsBindAddress = addressFromDeprecatedFlags(o.config.MetricsBindAddress, o.metricsPort)
	}

	// Load the config file here in Complete, so that Validate validates the fully-resolved config.
	if len(o.ConfigFile) > 0 {
		c, err := o.loadConfigFromFile(o.ConfigFile)
		if err != nil {
			return err
		}

		// Before we overwrite the config which holds the parsed
		// command line parameters, we need to copy all modified
		// logging settings over to the loaded config (i.e.  logging
		// command line flags have priority). Otherwise `--config
		// ... -v=5` doesn't work (config resets verbosity even
		// when it contains no logging settings).
		copyLogsFromFlags(fs, &c.Logging)
		o.config = c

		if err := o.initWatcher(); err != nil {
			return err
		}
	}

	o.platformApplyDefaults(o.config)

	if err := o.processHostnameOverrideFlag(); err != nil {
		return err
	}

	return utilfeature.DefaultMutableFeatureGate.SetFromMap(o.config.FeatureGates)
}

// copyLogsFromFlags applies the logging flags from the given flag set to the given
// configuration. Fields for which the corresponding flag was not used are left
// unmodified. For fields that have multiple values (like vmodule), the values from
// the flags get joined so that the command line flags have priority.
//
// TODO (pohly): move this to logsapi
func copyLogsFromFlags(from *pflag.FlagSet, to *logsapi.LoggingConfiguration) error {
	var cloneFS pflag.FlagSet
	logsapi.AddFlags(to, &cloneFS)
	vmodule := to.VModule
	to.VModule = nil
	var err error
	cloneFS.VisitAll(func(f *pflag.Flag) {
		if err != nil {
			return
		}
		fsFlag := from.Lookup(f.Name)
		if fsFlag == nil {
			err = fmt.Errorf("logging flag %s not found in flag set", f.Name)
			return
		}
		if !fsFlag.Changed {
			return
		}
		if setErr := f.Value.Set(fsFlag.Value.String()); setErr != nil {
			err = fmt.Errorf("copying flag %s value: %v", f.Name, setErr)
			return
		}
	})
	to.VModule = append(to.VModule, vmodule...)
	return err
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

func (o *Options) eventHandler(ent fsnotify.Event) {
	if ent.Has(fsnotify.Write) || ent.Has(fsnotify.Rename) {
		// error out when ConfigFile is updated
		o.errCh <- fmt.Errorf("content of the proxy server's configuration file was updated")
		return
	}
	o.errCh <- nil
}

func (o *Options) errorHandler(err error) {
	o.errCh <- err
}

// processHostnameOverrideFlag processes hostname-override flag
func (o *Options) processHostnameOverrideFlag() error {
	// Check if hostname-override flag is set and use value since configFile always overrides
	if len(o.hostnameOverride) > 0 {
		hostName := strings.TrimSpace(o.hostnameOverride)
		if len(hostName) == 0 {
			return fmt.Errorf("empty hostname-override is invalid")
		}
		o.config.HostnameOverride = strings.ToLower(hostName)
	}

	return nil
}

// Validate validates all the required options.
func (o *Options) Validate() error {
	if errs := validation.Validate(o.config); len(errs) != 0 {
		return errs.ToAggregate()
	}

	return nil
}

// Run runs the specified ProxyServer.
func (o *Options) Run(ctx context.Context) error {
	defer close(o.errCh)
	if len(o.WriteConfigTo) > 0 {
		return o.writeConfigFile()
	}

	err := platformCleanup(ctx, o.config.Mode, o.CleanupAndExit)
	if o.CleanupAndExit {
		return err
	}
	// We ignore err otherwise; the cleanup is best-effort, and the backends will have
	// logged messages if they failed in interesting ways.

	proxyServer, err := newProxyServer(ctx, o.config, o.master, o.InitAndExit)
	if err != nil {
		return err
	}
	if o.InitAndExit {
		return nil
	}

	o.proxyServer = proxyServer
	return o.runLoop(ctx)
}

// runLoop will watch on the update change of the proxy server's configuration file.
// Return an error when updated
func (o *Options) runLoop(ctx context.Context) error {
	if o.watcher != nil {
		o.watcher.Run()
	}

	// run the proxy in goroutine
	go func() {
		err := o.proxyServer.Run(ctx)
		o.errCh <- err
	}()

	for {
		err := <-o.errCh
		if err != nil {
			return err
		}
	}
}

func (o *Options) writeConfigFile() (err error) {
	const mediaType = runtime.ContentTypeYAML
	info, ok := runtime.SerializerInfoForMediaType(proxyconfigscheme.Codecs.SupportedMediaTypes(), mediaType)
	if !ok {
		return fmt.Errorf("unable to locate encoder -- %q is not a supported media type", mediaType)
	}

	encoder := proxyconfigscheme.Codecs.EncoderForVersion(info.Serializer, v1alpha1.SchemeGroupVersion)

	configFile, err := os.Create(o.WriteConfigTo)
	if err != nil {
		return err
	}

	defer func() {
		ferr := configFile.Close()
		if ferr != nil && err == nil {
			err = ferr
		}
	}()

	if err = encoder.Encode(o.config, configFile); err != nil {
		return err
	}

	o.logger.Info("Wrote configuration", "file", o.WriteConfigTo)

	return nil
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

// loadConfigFromFile loads the contents of file and decodes it as a
// KubeProxyConfiguration object.
func (o *Options) loadConfigFromFile(file string) (*kubeproxyconfig.KubeProxyConfiguration, error) {
	data, err := os.ReadFile(file)
	if err != nil {
		return nil, err
	}

	return o.loadConfig(data)
}

// loadConfig decodes a serialized KubeProxyConfiguration to the internal type.
func (o *Options) loadConfig(data []byte) (*kubeproxyconfig.KubeProxyConfiguration, error) {

	configObj, gvk, err := proxyconfigscheme.Codecs.UniversalDecoder().Decode(data, nil, nil)
	if err != nil {
		// Try strict decoding first. If that fails decode with a lenient
		// decoder, which has only v1alpha1 registered, and log a warning.
		// The lenient path is to be dropped when support for v1alpha1 is dropped.
		if !runtime.IsStrictDecodingError(err) {
			return nil, fmt.Errorf("failed to decode: %w", err)
		}

		_, lenientCodecs, lenientErr := newLenientSchemeAndCodecs()
		if lenientErr != nil {
			return nil, lenientErr
		}

		configObj, gvk, lenientErr = lenientCodecs.UniversalDecoder().Decode(data, nil, nil)
		if lenientErr != nil {
			// Lenient decoding failed with the current version, return the
			// original strict error.
			return nil, fmt.Errorf("failed lenient decoding: %v", err)
		}

		// Continue with the v1alpha1 object that was decoded leniently, but emit a warning.
		o.logger.Info("Using lenient decoding as strict decoding failed", "err", err)
	}

	proxyConfig, ok := configObj.(*kubeproxyconfig.KubeProxyConfiguration)
	if !ok {
		return nil, fmt.Errorf("got unexpected config type: %v", gvk)
	}
	return proxyConfig, nil
}

// NewProxyCommand creates a *cobra.Command object with default parameters
func NewProxyCommand() *cobra.Command {
	opts := NewOptions()

	cmd := &cobra.Command{
		Use: "kube-proxy",
		Long: `The Kubernetes network proxy runs on each node. This
reflects services as defined in the Kubernetes API on each node and can do simple
TCP, UDP, and SCTP stream forwarding or round robin TCP, UDP, and SCTP forwarding across a set of backends.
Service cluster IPs and ports are currently found through Docker-links-compatible
environment variables specifying ports opened by the service proxy. There is an optional
addon that provides cluster DNS for these cluster IPs. The user must create a service
with the apiserver API to configure the proxy.`,
		RunE: func(cmd *cobra.Command, args []string) error {
			verflag.PrintAndExitIfRequested()

			if err := initForOS(opts.WindowsService); err != nil {
				return fmt.Errorf("failed os init: %w", err)
			}

			if err := opts.Complete(cmd.Flags()); err != nil {
				return fmt.Errorf("failed complete: %w", err)
			}

			logs.InitLogs()
			if err := logsapi.ValidateAndApplyAsField(&opts.config.Logging, utilfeature.DefaultFeatureGate, field.NewPath("logging")); err != nil {
				return fmt.Errorf("initialize logging: %v", err)
			}

			cliflag.PrintFlags(cmd.Flags())

			if err := opts.Validate(); err != nil {
				return fmt.Errorf("failed validate: %w", err)
			}
			// add feature enablement metrics
			utilfeature.DefaultMutableFeatureGate.AddMetrics()
			if err := opts.Run(context.Background()); err != nil {
				opts.logger.Error(err, "Error running ProxyServer")
				return err
			}

			return nil
		},
		Args: func(cmd *cobra.Command, args []string) error {
			for _, arg := range args {
				if len(arg) > 0 {
					return fmt.Errorf("%q does not take any arguments, got %q", cmd.CommandPath(), args)
				}
			}
			return nil
		},
	}

	fs := cmd.Flags()
	opts.AddFlags(fs)
	fs.AddGoFlagSet(goflag.CommandLine) // for --boot-id-file and --machine-id-file

	_ = cmd.MarkFlagFilename("config", "yaml", "yml", "json")

	return cmd
}

// ProxyServer represents all the parameters required to start the Kubernetes proxy server. All
// fields are required.
type ProxyServer struct {
	Config *kubeproxyconfig.KubeProxyConfiguration

	Client          clientset.Interface
	Broadcaster     events.EventBroadcaster
	Recorder        events.EventRecorder
	NodeRef         *v1.ObjectReference
	HealthzServer   *healthcheck.ProxierHealthServer
	Hostname        string
	PrimaryIPFamily v1.IPFamily
	NodeIPs         map[v1.IPFamily]net.IP

	podCIDRs []string // only used for LocalModeNodeCIDR

	Proxier proxy.Provider
}

// newProxyServer creates a ProxyServer based on the given config
func newProxyServer(ctx context.Context, config *kubeproxyconfig.KubeProxyConfiguration, master string, initOnly bool) (*ProxyServer, error) {
	logger := klog.FromContext(ctx)

	s := &ProxyServer{
		Config: config,
	}

	cz, err := configz.New(kubeproxyconfig.GroupName)
	if err != nil {
		return nil, fmt.Errorf("unable to register configz: %s", err)
	}
	cz.Set(config)

	if len(config.ShowHiddenMetricsForVersion) > 0 {
		metrics.SetShowHidden()
	}

	s.Hostname, err = nodeutil.GetHostname(config.HostnameOverride)
	if err != nil {
		return nil, err
	}

	s.Client, err = createClient(ctx, config.ClientConnection, master)
	if err != nil {
		return nil, err
	}

	rawNodeIPs := getNodeIPs(ctx, s.Client, s.Hostname)
	s.PrimaryIPFamily, s.NodeIPs = detectNodeIPs(ctx, rawNodeIPs, config.BindAddress)

	if len(config.NodePortAddresses) == 1 && config.NodePortAddresses[0] == kubeproxyconfig.NodePortAddressesPrimary {
		var nodePortAddresses []string
		if nodeIP := s.NodeIPs[v1.IPv4Protocol]; nodeIP != nil && !nodeIP.IsLoopback() {
			nodePortAddresses = append(nodePortAddresses, fmt.Sprintf("%s/32", nodeIP.String()))
		}
		if nodeIP := s.NodeIPs[v1.IPv6Protocol]; nodeIP != nil && !nodeIP.IsLoopback() {
			nodePortAddresses = append(nodePortAddresses, fmt.Sprintf("%s/128", nodeIP.String()))
		}
		config.NodePortAddresses = nodePortAddresses
	}

	s.Broadcaster = events.NewBroadcaster(&events.EventSinkImpl{Interface: s.Client.EventsV1()})
	s.Recorder = s.Broadcaster.NewRecorder(proxyconfigscheme.Scheme, "kube-proxy")

	s.NodeRef = &v1.ObjectReference{
		Kind:      "Node",
		Name:      s.Hostname,
		UID:       types.UID(s.Hostname),
		Namespace: "",
	}

	if len(config.HealthzBindAddress) > 0 {
		s.HealthzServer = healthcheck.NewProxierHealthServer(config.HealthzBindAddress, 2*config.IPTables.SyncPeriod.Duration)
	}

	err = s.platformSetup(ctx)
	if err != nil {
		return nil, err
	}

	err = checkBadConfig(s)
	if err != nil {
		logger.Error(err, "Kube-proxy configuration may be incomplete or incorrect")
	}

	ipv4Supported, ipv6Supported, dualStackSupported, err := s.platformCheckSupported(ctx)
	if err != nil {
		return nil, err
	} else if (s.PrimaryIPFamily == v1.IPv4Protocol && !ipv4Supported) || (s.PrimaryIPFamily == v1.IPv6Protocol && !ipv6Supported) {
		return nil, fmt.Errorf("no support for primary IP family %q", s.PrimaryIPFamily)
	} else if dualStackSupported {
		logger.Info("kube-proxy running in dual-stack mode", "primary ipFamily", s.PrimaryIPFamily)
	} else {
		logger.Info("kube-proxy running in single-stack mode", "ipFamily", s.PrimaryIPFamily)
	}

	err, fatal := checkBadIPConfig(s, dualStackSupported)
	if err != nil {
		if fatal {
			return nil, fmt.Errorf("kube-proxy configuration is incorrect: %v", err)
		}
		logger.Error(err, "Kube-proxy configuration may be incomplete or incorrect")
	}

	s.Proxier, err = s.createProxier(ctx, config, dualStackSupported, initOnly)
	if err != nil {
		return nil, err
	}

	return s, nil
}

// checkBadConfig checks for bad/deprecated configuation
func checkBadConfig(s *ProxyServer) error {
	var errors []error

	// At this point we haven't seen any actual Services or EndpointSlices, so we
	// don't really know if the cluster is expected to be single- or dual-stack. But
	// we can at least take note of whether there is any explicitly-dual-stack
	// configuration.
	anyDualStackConfig := false
	clusterCIDRs := strings.Split(s.Config.ClusterCIDR, ",")
	for _, config := range [][]string{clusterCIDRs, s.Config.NodePortAddresses, s.Config.IPVS.ExcludeCIDRs, s.podCIDRs} {
		if dual, _ := netutils.IsDualStackCIDRStrings(config); dual {
			anyDualStackConfig = true
			break
		}
	}

	// Warn if NodePortAddresses does not limit connections on all IP families that
	// seem to be in use.
	cidrsByFamily := proxyutil.MapCIDRsByIPFamily(s.Config.NodePortAddresses)
	if len(s.Config.NodePortAddresses) == 0 {
		errors = append(errors, fmt.Errorf("nodePortAddresses is unset; NodePort connections will be accepted on all local IPs. Consider using `--nodeport-addresses primary`"))
	} else if anyDualStackConfig && len(cidrsByFamily[s.PrimaryIPFamily]) == len(s.Config.NodePortAddresses) {
		errors = append(errors, fmt.Errorf("cluster appears to be dual-stack but nodePortAddresses contains only %s addresses; NodePort connections will be accepted on all local %s IPs", s.PrimaryIPFamily, proxyutil.OtherIPFamily(s.PrimaryIPFamily)))
	} else if len(cidrsByFamily[s.PrimaryIPFamily]) == 0 {
		errors = append(errors, fmt.Errorf("cluster appears to be %s-primary but nodePortAddresses contains only %s addresses; NodePort connections will be accepted on all local %s IPs", s.PrimaryIPFamily, proxyutil.OtherIPFamily(s.PrimaryIPFamily), s.PrimaryIPFamily))
	}

	return utilerrors.NewAggregate(errors)
}

// checkBadIPConfig checks for bad configuration relative to s.PrimaryIPFamily.
// Historically, we did not check most of the config options, so we cannot retroactively
// make IP family mismatches in those options be fatal. When we add new options to check
// here, we should make problems with those options be fatal.
func checkBadIPConfig(s *ProxyServer, dualStackSupported bool) (err error, fatal bool) {
	var errors []error
	var badFamily netutils.IPFamily

	if s.PrimaryIPFamily == v1.IPv4Protocol {
		badFamily = netutils.IPv6
	} else {
		badFamily = netutils.IPv4
	}

	var clusterType string
	if dualStackSupported {
		clusterType = fmt.Sprintf("%s-primary", s.PrimaryIPFamily)
	} else {
		clusterType = fmt.Sprintf("%s-only", s.PrimaryIPFamily)
	}

	if s.Config.ClusterCIDR != "" {
		clusterCIDRs := strings.Split(s.Config.ClusterCIDR, ",")
		if badCIDRs(clusterCIDRs, badFamily) {
			errors = append(errors, fmt.Errorf("cluster is %s but clusterCIDRs contains only IPv%s addresses", clusterType, badFamily))
			if s.Config.DetectLocalMode == kubeproxyconfig.LocalModeClusterCIDR && !dualStackSupported {
				// This has always been a fatal error
				fatal = true
			}
		}
	}

	if badCIDRs(s.podCIDRs, badFamily) {
		errors = append(errors, fmt.Errorf("cluster is %s but node.spec.podCIDRs contains only IPv%s addresses", clusterType, badFamily))
		if s.Config.DetectLocalMode == kubeproxyconfig.LocalModeNodeCIDR {
			// This has always been a fatal error
			fatal = true
		}
	}

	if netutils.IPFamilyOfString(s.Config.Winkernel.SourceVip) == badFamily {
		errors = append(errors, fmt.Errorf("cluster is %s but winkernel.sourceVip is IPv%s", clusterType, badFamily))
	}

	// In some cases, wrong-IP-family is only a problem when the secondary IP family
	// isn't present at all.
	if !dualStackSupported {
		if badCIDRs(s.Config.IPVS.ExcludeCIDRs, badFamily) {
			errors = append(errors, fmt.Errorf("cluster is %s but ipvs.excludeCIDRs contains only IPv%s addresses", clusterType, badFamily))
		}

		if badBindAddress(s.Config.HealthzBindAddress, badFamily) {
			errors = append(errors, fmt.Errorf("cluster is %s but healthzBindAddress is IPv%s", clusterType, badFamily))
		}
		if badBindAddress(s.Config.MetricsBindAddress, badFamily) {
			errors = append(errors, fmt.Errorf("cluster is %s but metricsBindAddress is IPv%s", clusterType, badFamily))
		}
	}

	// Note that s.Config.NodePortAddresses gets checked as part of checkBadConfig()
	// so it doesn't need to be checked here.

	return utilerrors.NewAggregate(errors), fatal
}

// badCIDRs returns true if cidrs is a non-empty list of CIDRs, all of wrongFamily.
func badCIDRs(cidrs []string, wrongFamily netutils.IPFamily) bool {
	if len(cidrs) == 0 {
		return false
	}
	for _, cidr := range cidrs {
		if netutils.IPFamilyOfCIDRString(cidr) != wrongFamily {
			return false
		}
	}
	return true
}

// badBindAddress returns true if bindAddress is an "IP:port" string where IP is a
// non-zero IP of wrongFamily.
func badBindAddress(bindAddress string, wrongFamily netutils.IPFamily) bool {
	if host, _, _ := net.SplitHostPort(bindAddress); host != "" {
		ip := netutils.ParseIPSloppy(host)
		if ip != nil && netutils.IPFamilyOf(ip) == wrongFamily && !ip.IsUnspecified() {
			return true
		}
	}
	return false
}

// createClient creates a kube client from the given config and masterOverride.
// TODO remove masterOverride when CLI flags are removed.
func createClient(ctx context.Context, config componentbaseconfig.ClientConnectionConfiguration, masterOverride string) (clientset.Interface, error) {
	logger := klog.FromContext(ctx)
	var kubeConfig *rest.Config
	var err error

	if len(config.Kubeconfig) == 0 && len(masterOverride) == 0 {
		logger.Info("Neither kubeconfig file nor master URL was specified, falling back to in-cluster config")
		kubeConfig, err = rest.InClusterConfig()
	} else {
		// This creates a client, first loading any specified kubeconfig
		// file, and then overriding the Master flag, if non-empty.
		kubeConfig, err = clientcmd.NewNonInteractiveDeferredLoadingClientConfig(
			&clientcmd.ClientConfigLoadingRules{ExplicitPath: config.Kubeconfig},
			&clientcmd.ConfigOverrides{ClusterInfo: clientcmdapi.Cluster{Server: masterOverride}}).ClientConfig()
	}
	if err != nil {
		return nil, err
	}

	kubeConfig.AcceptContentTypes = config.AcceptContentTypes
	kubeConfig.ContentType = config.ContentType
	kubeConfig.QPS = config.QPS
	kubeConfig.Burst = int(config.Burst)

	client, err := clientset.NewForConfig(kubeConfig)
	if err != nil {
		return nil, err
	}

	return client, nil
}

func serveHealthz(ctx context.Context, hz *healthcheck.ProxierHealthServer, errCh chan error) {
	logger := klog.FromContext(ctx)
	if hz == nil {
		return
	}

	fn := func() {
		err := hz.Run()
		if err != nil {
			logger.Error(err, "Healthz server failed")
			if errCh != nil {
				errCh <- fmt.Errorf("healthz server failed: %v", err)
				// if in hardfail mode, never retry again
				blockCh := make(chan error)
				<-blockCh
			}
		} else {
			logger.Error(nil, "Healthz server returned without error")
		}
	}
	go wait.Until(fn, 5*time.Second, ctx.Done())
}

func serveMetrics(bindAddress string, proxyMode kubeproxyconfig.ProxyMode, enableProfiling bool, errCh chan error) {
	if len(bindAddress) == 0 {
		return
	}

	proxyMux := mux.NewPathRecorderMux("kube-proxy")
	healthz.InstallHandler(proxyMux)
	slis.SLIMetricsWithReset{}.Install(proxyMux)

	proxyMux.HandleFunc("/proxyMode", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/plain; charset=utf-8")
		w.Header().Set("X-Content-Type-Options", "nosniff")
		fmt.Fprintf(w, "%s", proxyMode)
	})

	proxyMux.Handle("/metrics", legacyregistry.Handler())

	if enableProfiling {
		routes.Profiling{}.Install(proxyMux)
		routes.DebugFlags{}.Install(proxyMux, "v", routes.StringFlagPutHandler(logs.GlogSetter))
	}

	configz.InstallHandler(proxyMux)

	fn := func() {
		err := http.ListenAndServe(bindAddress, proxyMux)
		if err != nil {
			err = fmt.Errorf("starting metrics server failed: %v", err)
			utilruntime.HandleError(err)
			if errCh != nil {
				errCh <- err
				// if in hardfail mode, never retry again
				blockCh := make(chan error)
				<-blockCh
			}
		}
	}
	go wait.Until(fn, 5*time.Second, wait.NeverStop)
}

// Run runs the specified ProxyServer.  This should never exit (unless CleanupAndExit is set).
// TODO: At the moment, Run() cannot return a nil error, otherwise it's caller will never exit. Update callers of Run to handle nil errors.
func (s *ProxyServer) Run(ctx context.Context) error {
	logger := klog.FromContext(ctx)
	// To help debugging, immediately log version
	logger.Info("Version info", "version", version.Get())

	logger.Info("Golang settings", "GOGC", os.Getenv("GOGC"), "GOMAXPROCS", os.Getenv("GOMAXPROCS"), "GOTRACEBACK", os.Getenv("GOTRACEBACK"))

	proxymetrics.RegisterMetrics(s.Config.Mode)

	// TODO(vmarmol): Use container config for this.
	var oomAdjuster *oom.OOMAdjuster
	if s.Config.OOMScoreAdj != nil {
		oomAdjuster = oom.NewOOMAdjuster()
		if err := oomAdjuster.ApplyOOMScoreAdj(0, int(*s.Config.OOMScoreAdj)); err != nil {
			logger.V(2).Info("Failed to apply OOMScore", "err", err)
		}
	}

	if s.Broadcaster != nil {
		stopCh := make(chan struct{})
		s.Broadcaster.StartRecordingToSink(stopCh)
	}

	// TODO(thockin): make it possible for healthz and metrics to be on the same port.

	var healthzErrCh, metricsErrCh chan error
	if s.Config.BindAddressHardFail {
		healthzErrCh = make(chan error)
		metricsErrCh = make(chan error)
	}

	// Start up a healthz server if requested
	serveHealthz(ctx, s.HealthzServer, healthzErrCh)

	// Start up a metrics server if requested
	serveMetrics(s.Config.MetricsBindAddress, s.Config.Mode, s.Config.EnableProfiling, metricsErrCh)

	noProxyName, err := labels.NewRequirement(apis.LabelServiceProxyName, selection.DoesNotExist, nil)
	if err != nil {
		return err
	}

	noHeadlessEndpoints, err := labels.NewRequirement(v1.IsHeadlessService, selection.DoesNotExist, nil)
	if err != nil {
		return err
	}

	labelSelector := labels.NewSelector()
	labelSelector = labelSelector.Add(*noProxyName, *noHeadlessEndpoints)

	// Make informers that filter out objects that want a non-default service proxy.
	informerFactory := informers.NewSharedInformerFactoryWithOptions(s.Client, s.Config.ConfigSyncPeriod.Duration,
		informers.WithTweakListOptions(func(options *metav1.ListOptions) {
			options.LabelSelector = labelSelector.String()
		}))

	// Create configs (i.e. Watches for Services, EndpointSlices and ServiceCIDRs)
	// Note: RegisterHandler() calls need to happen before creation of Sources because sources
	// only notify on changes, and the initial update (on process start) may be lost if no handlers
	// are registered yet.
	serviceConfig := config.NewServiceConfig(ctx, informerFactory.Core().V1().Services(), s.Config.ConfigSyncPeriod.Duration)
	serviceConfig.RegisterEventHandler(s.Proxier)
	go serviceConfig.Run(ctx.Done())

	endpointSliceConfig := config.NewEndpointSliceConfig(ctx, informerFactory.Discovery().V1().EndpointSlices(), s.Config.ConfigSyncPeriod.Duration)
	endpointSliceConfig.RegisterEventHandler(s.Proxier)
	go endpointSliceConfig.Run(ctx.Done())

	if utilfeature.DefaultFeatureGate.Enabled(features.MultiCIDRServiceAllocator) {
		serviceCIDRConfig := config.NewServiceCIDRConfig(ctx, informerFactory.Networking().V1alpha1().ServiceCIDRs(), s.Config.ConfigSyncPeriod.Duration)
		serviceCIDRConfig.RegisterEventHandler(s.Proxier)
		go serviceCIDRConfig.Run(wait.NeverStop)
	}
	// This has to start after the calls to NewServiceConfig because that
	// function must configure its shared informer event handlers first.
	informerFactory.Start(wait.NeverStop)

	// Make an informer that selects for our nodename.
	currentNodeInformerFactory := informers.NewSharedInformerFactoryWithOptions(s.Client, s.Config.ConfigSyncPeriod.Duration,
		informers.WithTweakListOptions(func(options *metav1.ListOptions) {
			options.FieldSelector = fields.OneTermEqualSelector("metadata.name", s.NodeRef.Name).String()
		}))
	nodeConfig := config.NewNodeConfig(ctx, currentNodeInformerFactory.Core().V1().Nodes(), s.Config.ConfigSyncPeriod.Duration)
	// https://issues.k8s.io/111321
	if s.Config.DetectLocalMode == kubeproxyconfig.LocalModeNodeCIDR {
		nodeConfig.RegisterEventHandler(proxy.NewNodePodCIDRHandler(ctx, s.podCIDRs))
	}
	if utilfeature.DefaultFeatureGate.Enabled(features.KubeProxyDrainingTerminatingNodes) {
		nodeConfig.RegisterEventHandler(&proxy.NodeEligibleHandler{
			HealthServer: s.HealthzServer,
		})
	}
	nodeConfig.RegisterEventHandler(s.Proxier)

	go nodeConfig.Run(wait.NeverStop)

	// This has to start after the calls to NewNodeConfig because that must
	// configure the shared informer event handler first.
	currentNodeInformerFactory.Start(wait.NeverStop)

	// Birth Cry after the birth is successful
	s.birthCry()

	go s.Proxier.SyncLoop()

	select {
	case err = <-healthzErrCh:
		s.Recorder.Eventf(s.NodeRef, nil, api.EventTypeWarning, "FailedToStartProxierHealthcheck", "StartKubeProxy", err.Error())
	case err = <-metricsErrCh:
		s.Recorder.Eventf(s.NodeRef, nil, api.EventTypeWarning, "FailedToStartMetricServer", "StartKubeProxy", err.Error())
	}
	return err
}

func (s *ProxyServer) birthCry() {
	s.Recorder.Eventf(s.NodeRef, nil, api.EventTypeNormal, "Starting", "StartKubeProxy", "")
}

// detectNodeIPs returns the proxier's "node IP" or IPs, and the IP family to use if the
// node turns out to be incapable of dual-stack. (Note that kube-proxy normally runs as
// dual-stack if the backend is capable of supporting both IP families, regardless of
// whether the node is *actually* configured as dual-stack or not.)

// (Note that on Linux, the node IPs are used only to determine whether a given
// LoadBalancerSourceRanges value matches the node or not. In particular, they are *not*
// used for NodePort handling.)
//
// The order of precedence is:
//  1. if bindAddress is not 0.0.0.0 or ::, then it is used as the primary IP.
//  2. if rawNodeIPs is not empty, then its address(es) is/are used
//  3. otherwise the node IPs are 127.0.0.1 and ::1
func detectNodeIPs(ctx context.Context, rawNodeIPs []net.IP, bindAddress string) (v1.IPFamily, map[v1.IPFamily]net.IP) {
	logger := klog.FromContext(ctx)
	primaryFamily := v1.IPv4Protocol
	nodeIPs := map[v1.IPFamily]net.IP{
		v1.IPv4Protocol: net.IPv4(127, 0, 0, 1),
		v1.IPv6Protocol: net.IPv6loopback,
	}

	if len(rawNodeIPs) > 0 {
		if !netutils.IsIPv4(rawNodeIPs[0]) {
			primaryFamily = v1.IPv6Protocol
		}
		nodeIPs[primaryFamily] = rawNodeIPs[0]
		if len(rawNodeIPs) > 1 {
			// If more than one address is returned, they are guaranteed to be of different families
			family := v1.IPv4Protocol
			if !netutils.IsIPv4(rawNodeIPs[1]) {
				family = v1.IPv6Protocol
			}
			nodeIPs[family] = rawNodeIPs[1]
		}
	}

	// If a bindAddress is passed, override the primary IP
	bindIP := netutils.ParseIPSloppy(bindAddress)
	if bindIP != nil && !bindIP.IsUnspecified() {
		if netutils.IsIPv4(bindIP) {
			primaryFamily = v1.IPv4Protocol
		} else {
			primaryFamily = v1.IPv6Protocol
		}
		nodeIPs[primaryFamily] = bindIP
	}

	if nodeIPs[primaryFamily].IsLoopback() {
		logger.Info("Can't determine this node's IP, assuming loopback; if this is incorrect, please set the --bind-address flag")
	}
	return primaryFamily, nodeIPs
}

// getNodeIP returns IPs for the node with the provided name.  If
// required, it will wait for the node to be created.
func getNodeIPs(ctx context.Context, client clientset.Interface, name string) []net.IP {
	logger := klog.FromContext(ctx)
	var nodeIPs []net.IP
	backoff := wait.Backoff{
		Steps:    6,
		Duration: 1 * time.Second,
		Factor:   2.0,
		Jitter:   0.2,
	}

	err := wait.ExponentialBackoff(backoff, func() (bool, error) {
		node, err := client.CoreV1().Nodes().Get(ctx, name, metav1.GetOptions{})
		if err != nil {
			logger.Error(err, "Failed to retrieve node info")
			return false, nil
		}
		nodeIPs, err = utilnode.GetNodeHostIPs(node)
		if err != nil {
			logger.Error(err, "Failed to retrieve node IPs")
			return false, nil
		}
		return true, nil
	})
	if err == nil {
		logger.Info("Successfully retrieved node IP(s)", "IPs", nodeIPs)
	}
	return nodeIPs
}
