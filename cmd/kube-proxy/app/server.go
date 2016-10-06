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
	"io/ioutil"
	"net"
	"net/http"
	_ "net/http/pprof"
	"runtime"
	"time"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/apis/componentconfig"
	"k8s.io/kubernetes/pkg/apis/componentconfig/v1alpha1"
	"k8s.io/kubernetes/pkg/client/record"
	kubeclient "k8s.io/kubernetes/pkg/client/unversioned"
	"k8s.io/kubernetes/pkg/client/unversioned/clientcmd"
	cmdutil "k8s.io/kubernetes/pkg/kubectl/cmd/util"
	"k8s.io/kubernetes/pkg/proxy"
	proxyconfig "k8s.io/kubernetes/pkg/proxy/config"
	"k8s.io/kubernetes/pkg/proxy/iptables"
	"k8s.io/kubernetes/pkg/proxy/userspace"
	"k8s.io/kubernetes/pkg/types"
	"k8s.io/kubernetes/pkg/util"
	"k8s.io/kubernetes/pkg/util/configz"
	utildbus "k8s.io/kubernetes/pkg/util/dbus"
	"k8s.io/kubernetes/pkg/util/exec"
	utiliptables "k8s.io/kubernetes/pkg/util/iptables"
	utilnet "k8s.io/kubernetes/pkg/util/net"
	nodeutil "k8s.io/kubernetes/pkg/util/node"
	"k8s.io/kubernetes/pkg/util/oom"
	"k8s.io/kubernetes/pkg/util/resourcecontainer"
	utilsysctl "k8s.io/kubernetes/pkg/util/sysctl"
	"k8s.io/kubernetes/pkg/util/wait"

	"github.com/golang/glog"
	"github.com/spf13/cobra"
	"github.com/spf13/pflag"
)

const (
	proxyModeUserspace              = "userspace"
	proxyModeIPTables               = "iptables"
	ExperimentalProxyModeAnnotation = "net.experimental.kubernetes.io/proxy-mode"
	betaProxyModeAnnotation         = "net.beta.kubernetes.io/proxy-mode"
)

func checkKnownProxyMode(proxyMode string) bool {
	switch proxyMode {
	case "", proxyModeUserspace, proxyModeIPTables:
		return true
	}
	return false
}

type KubeProxyOptions struct {
	ConfigFile     util.StringFlag
	CleanupAndExit bool
}

func (o KubeProxyOptions) Complete() error {
	return nil
}

func (o KubeProxyOptions) Validate(args []string) error {
	if len(args) != 0 {
		return errors.New("no arguments are supported")
	}

	if o.CleanupAndExit {
		// we don't use --config for this
		return nil
	}

	if len(o.ConfigFile.String()) == 0 {
		return errors.New("--config must be set")
	}

	return nil
}

func BindKubeProxyOptions(options *KubeProxyOptions, flags *pflag.FlagSet) {
	flags.Var(&options.ConfigFile, "config", "The config file to use")
	flags.BoolVar(&options.CleanupAndExit, "cleanup-iptables", options.CleanupAndExit, "If true cleanup iptables rules and exit.")
}

func BuildSerializableKubeProxyConfiguration() (*componentconfig.KubeProxyConfiguration, error) {
	config := &componentconfig.KubeProxyConfiguration{}

	external, err := api.Scheme.ConvertToVersion(config, v1alpha1.SchemeGroupVersion)
	if err != nil {
		return nil, err
	}

	internal, err := api.Scheme.ConvertToVersion(external, componentconfig.SchemeGroupVersion)
	if err != nil {
		return nil, err
	}

	config = internal.(*componentconfig.KubeProxyConfiguration)

	return config, nil
}

func (o KubeProxyOptions) Run() error {
	data, err := ioutil.ReadFile(o.ConfigFile.String())
	if err != nil {
		return err
	}

	configObj, gvk, err := api.Codecs.UniversalDecoder().Decode(data, nil, nil)
	if err != nil {
		return err
	}
	config, ok := configObj.(*componentconfig.KubeProxyConfiguration)
	if !ok {
		return fmt.Errorf("got unexpected config type: %v", gvk)
	}

	proxyServer, err := o.NewProxyServer(config)
	if err != nil {
		return err
	}

	return proxyServer.Run()
}

// NewProxyCommand creates a *cobra.Command object with default parameters
func NewProxyCommand() *cobra.Command {
	opts := KubeProxyOptions{}

	cmd := &cobra.Command{
		Use: "kube-proxy",
		Long: `The Kubernetes network proxy runs on each node. This
reflects services as defined in the Kubernetes API on each node and can do simple
TCP,UDP stream forwarding or round robin TCP,UDP forwarding across a set of backends.
Service cluster ips and ports are currently found through Docker-links-compatible
environment variables specifying ports opened by the service proxy. There is an optional
addon that provides cluster DNS for these cluster IPs. The user must create a service
with the apiserver API to configure the proxy.`,
		Run: func(c *cobra.Command, args []string) {
			cmdutil.CheckErr(opts.Complete())
			cmdutil.CheckErr(opts.Validate(args))
			cmdutil.CheckErr(opts.Run())
		},
	}

	flags := cmd.Flags()
	BindKubeProxyOptions(&opts, flags)

	cmd.MarkFlagFilename("config", "yaml", "yml")

	return cmd
}

// ProxyServer represents all the parameters required to start the Kuberetes
// proxy server. All fields are required.
type ProxyServer struct {
	Client                 *kubeclient.Client
	IptInterface           utiliptables.Interface
	Proxier                proxy.ProxyProvider
	EndpointsHandler       proxyconfig.EndpointsConfigHandler
	Broadcaster            record.EventBroadcaster
	Recorder               record.EventRecorder
	ConntrackConfiguration componentconfig.KubeProxyConntrackConfiguration
	Conntracker            Conntracker // if nil, ignored
	ProxyMode              string
	NodeRef                *api.ObjectReference
	CleanupAndExit         bool
	HealthzBindAddress     string
	OOMScoreAdj            *int32
	ResourceContainer      string
	ConfigSyncPeriod       time.Duration
}

func createKubeClient(config componentconfig.ClientConnectionConfiguration) (*kubeclient.Client, error) {
	// This creates a client using the specified kubeconfig file
	kubeconfig, err := clientcmd.NewNonInteractiveDeferredLoadingClientConfig(
		&clientcmd.ClientConfigLoadingRules{ExplicitPath: config.KubeConfigFile},
		&clientcmd.ConfigOverrides{}).ClientConfig()
	if err != nil {
		return nil, err
	}

	kubeconfig.AcceptContentTypes = config.AcceptContentTypes
	kubeconfig.ContentType = config.ContentType
	kubeconfig.QPS = config.QPS
	//TODO make config struct use int instead of int32?
	kubeconfig.Burst = int(config.Burst)

	return kubeclient.New(kubeconfig)
}

func (o KubeProxyOptions) NewProxyServer(config *componentconfig.KubeProxyConfiguration) (*ProxyServer, error) {
	if c, err := configz.New("componentconfig"); err == nil {
		c.Set(config)
	} else {
		// TODO should this return the error or is it ok to proceed
		glog.Errorf("unable to register configz: %s", err)
	}

	glog.V(4).Infof("using configuration: %#v", config)

	protocol := utiliptables.ProtocolIpv4
	if net.ParseIP(config.BindAddress).To4() == nil {
		protocol = utiliptables.ProtocolIpv6
	}

	// Create a iptables utils.
	execer := exec.New()
	dbus := utildbus.New()
	iptInterface := utiliptables.New(execer, dbus, protocol)

	proxyServer := &ProxyServer{
		IptInterface: iptInterface,
	}

	// We omit creation of pretty much everything if we run in cleanup mode
	if o.CleanupAndExit {
		return proxyServer, nil
	}

	client, err := createKubeClient(config.ClientConnection)
	if err != nil {
		return nil, err
	}

	// Create event recorder
	hostname := nodeutil.GetHostname(config.HostnameOverride)
	eventBroadcaster := record.NewBroadcaster()
	recorder := eventBroadcaster.NewRecorder(api.EventSource{Component: "kube-proxy", Host: hostname})

	var proxier proxy.ProxyProvider
	var endpointsHandler proxyconfig.EndpointsConfigHandler

	proxyMode := getProxyMode(string(config.Mode), client.Nodes(), hostname, iptInterface, iptables.LinuxKernelCompatTester{})
	if proxyMode == proxyModeIPTables {
		glog.V(0).Info("Using iptables Proxier.")
		if config.IPTablesConfiguration.MasqueradeBit == nil {
			// IPTablesMasqueradeBit must be specified or defaulted.
			return nil, fmt.Errorf("unable to read IPTablesMasqueradeBit from config")
		}
		// TODO this has side effects
		proxierIPTables, err := iptables.NewProxier(iptInterface, utilsysctl.New(), execer, config.IPTablesConfiguration.SyncPeriod.Duration, config.IPTablesConfiguration.MasqueradeAll, int(*config.IPTablesConfiguration.MasqueradeBit), config.ClusterCIDR, hostname, getNodeIP(client, hostname))
		if err != nil {
			return nil, fmt.Errorf("unable to create proxier: %v", err)
		}
		proxier = proxierIPTables
		endpointsHandler = proxierIPTables

		// No turning back. Remove artifacts that might still exist from the userspace Proxier.
		glog.V(0).Info("Tearing down userspace rules.")
		// TODO this has side effects
		userspace.CleanupLeftovers(iptInterface)
	} else {
		glog.V(0).Info("Using userspace Proxier.")
		// This is a proxy.LoadBalancer which NewProxier needs but has methods we don't need for
		// our config.EndpointsConfigHandler.
		loadBalancer := userspace.NewLoadBalancerRR()
		// set EndpointsConfigHandler to our loadBalancer
		endpointsHandler = loadBalancer

		// TODO this has side effects
		proxierUserspace, err := userspace.NewProxier(
			loadBalancer,
			net.ParseIP(config.BindAddress),
			iptInterface,
			*utilnet.ParsePortRangeOrDie(config.PortRange),
			config.IPTablesConfiguration.SyncPeriod.Duration,
			config.UDPIdleTimeout.Duration,
		)
		if err != nil {
			return nil, fmt.Errorf("Unable to create proxier: %v", err)
		}
		proxier = proxierUserspace

		// Remove artifacts from the pure-iptables Proxier.
		glog.V(0).Info("Tearing down pure-iptables proxy rules.")
		// TODO this has side effects
		iptables.CleanupLeftovers(iptInterface)
	}
	iptInterface.AddReloadFunc(proxier.Sync)

	nodeRef := &api.ObjectReference{
		Kind:      "Node",
		Name:      hostname,
		UID:       types.UID(hostname),
		Namespace: "",
	}

	return &ProxyServer{
		Client:                 client,
		IptInterface:           iptInterface,
		Proxier:                proxier,
		EndpointsHandler:       endpointsHandler,
		Broadcaster:            eventBroadcaster,
		Recorder:               recorder,
		ConntrackConfiguration: config.ConntrackConfiguration,
		Conntracker:            realConntracker{},
		ProxyMode:              proxyMode,
		NodeRef:                nodeRef,
		HealthzBindAddress:     config.HealthzBindAddress,
		OOMScoreAdj:            config.OOMScoreAdj,
		ResourceContainer:      config.ResourceContainer,
		ConfigSyncPeriod:       config.ConfigSyncPeriod,
	}, nil
}

// Run runs the specified ProxyServer. This should never exit (unless CleanupAndExit is set).
func (s *ProxyServer) Run() error {
	// remove iptables rules and exit
	if s.CleanupAndExit {
		encounteredError := userspace.CleanupLeftovers(s.IptInterface)
		encounteredError = iptables.CleanupLeftovers(s.IptInterface) || encounteredError
		if encounteredError {
			return errors.New("Encountered an error while tearing down rules.")
		}
		return nil
	}

	// TODO(vmarmol): Use container config for this.
	if s.OOMScoreAdj != nil {
		oomAdjuster := oom.NewOOMAdjuster()
		if err := oomAdjuster.ApplyOOMScoreAdj(0, int(*s.OOMScoreAdj)); err != nil {
			glog.V(2).Info(err)
		}
	}

	if len(s.ResourceContainer) > 0 {
		// Run in its own container.
		if err := resourcecontainer.RunInResourceContainer(s.ResourceContainer); err != nil {
			glog.Warningf("Failed to start in resource-only container %q: %v", s.ResourceContainer, err)
		} else {
			glog.V(2).Infof("Running in resource-only container %q", s.ResourceContainer)
		}
	}

	s.Broadcaster.StartRecordingToSink(s.Client.Events(""))

	// Start up a webserver if requested
	if len(s.HealthzBindAddress) > 0 {
		http.HandleFunc("/proxyMode", func(w http.ResponseWriter, r *http.Request) {
			fmt.Fprintf(w, "%s", s.ProxyMode)
		})
		configz.InstallHandler(http.DefaultServeMux)
		go wait.Until(func() {
			err := http.ListenAndServe(s.HealthzBindAddress, nil)
			if err != nil {
				glog.Errorf("Starting health server failed: %v", err)
			}
		}, 5*time.Second, wait.NeverStop)
	}

	// Tune conntrack, if requested
	if s.Conntracker != nil {
		max, err := getConntrackMax(s.ConntrackConfiguration)
		if err != nil {
			return err
		}
		if max > 0 {
			err := s.Conntracker.SetMax(max)
			if err != nil {
				if err != readOnlySysFSError {
					return err
				}
				// readOnlySysFSError is caused by a known docker issue (https://github.com/docker/docker/issues/24000),
				// the only remediation we know is to restart the docker daemon.
				// Here we'll send an node event with specific reason and message, the
				// administrator should decide whether and how to handle this issue,
				// whether to drain the node and restart docker.
				// TODO(random-liu): Remove this when the docker bug is fixed.
				const message = "DOCKER RESTART NEEDED (docker issue #24000): /sys is read-only: can't raise conntrack limits, problems may arise later."
				s.Recorder.Eventf(s.NodeRef, api.EventTypeWarning, err.Error(), message)
			}
		}
		if s.ConntrackConfiguration.TCPEstablishedTimeout.Duration > 0 {
			if err := s.Conntracker.SetTCPEstablishedTimeout(int(s.ConntrackConfiguration.TCPEstablishedTimeout.Duration / time.Second)); err != nil {
				return err
			}
		}
	}

	// Create configs (i.e. Watches for Services and Endpoints)
	// Note: RegisterHandler() calls need to happen before creation of Sources because sources
	// only notify on changes, and the initial update (on process start) may be lost if no handlers
	// are registered yet.
	serviceConfig := proxyconfig.NewServiceConfig()
	serviceConfig.RegisterHandler(s.Proxier)

	endpointsConfig := proxyconfig.NewEndpointsConfig()
	endpointsConfig.RegisterHandler(s.EndpointsHandler)

	proxyconfig.NewSourceAPI(
		s.Client,
		s.ConfigSyncPeriod,
		serviceConfig.Channel("api"),
		endpointsConfig.Channel("api"),
	)

	// Birth Cry after the birth is successful
	s.birthCry()

	// Just loop forever for now...
	s.Proxier.SyncLoop()
	return nil
}

func getConntrackMax(config componentconfig.KubeProxyConntrackConfiguration) (int, error) {
	if config.Max > 0 {
		if config.MaxPerCore > 0 {
			return -1, fmt.Errorf("invalid config: ConntrackMax and ConntrackMaxPerCore are mutually exclusive")
		}
		glog.V(3).Infof("getConntrackMax: using absolute conntrax-max (deprecated)")
		return int(config.Max), nil
	}
	if config.MaxPerCore > 0 {
		floor := int(config.Min)
		scaled := int(config.MaxPerCore) * runtime.NumCPU()
		if scaled > floor {
			glog.V(3).Infof("getConntrackMax: using scaled conntrax-max-per-core")
			return scaled, nil
		}
		glog.V(3).Infof("getConntrackMax: using conntrax-min")
		return floor, nil
	}
	return 0, nil
}

type nodeGetter interface {
	Get(hostname string) (*api.Node, error)
}

func getProxyMode(proxyMode string, client nodeGetter, hostname string, iptver iptables.IPTablesVersioner, kcompat iptables.KernelCompatTester) string {
	if proxyMode == proxyModeUserspace {
		return proxyModeUserspace
	} else if proxyMode == proxyModeIPTables {
		return tryIPTablesProxy(iptver, kcompat)
	} else if len(proxyMode) > 0 {
		glog.Warningf("proxy-mode=%q unknown, assuming iptables proxy", proxyMode)
		return tryIPTablesProxy(iptver, kcompat)
	}
	// proxyMode == "" - choose the best option.
	if client == nil {
		glog.Errorf("nodeGetter is nil: assuming iptables proxy")
		return tryIPTablesProxy(iptver, kcompat)
	}
	node, err := client.Get(hostname)
	if err != nil {
		glog.Errorf("Can't get Node %q, assuming iptables proxy, err: %v", hostname, err)
		return tryIPTablesProxy(iptver, kcompat)
	}
	if node == nil {
		glog.Errorf("Got nil Node %q, assuming iptables proxy", hostname)
		return tryIPTablesProxy(iptver, kcompat)
	}
	proxyMode, found := node.Annotations[betaProxyModeAnnotation]
	if found {
		glog.V(1).Infof("Found beta annotation %q = %q", betaProxyModeAnnotation, proxyMode)
	} else {
		// We already published some information about this annotation with the "experimental" name, so we will respect it.
		proxyMode, found = node.Annotations[ExperimentalProxyModeAnnotation]
		if found {
			glog.V(1).Infof("Found experimental annotation %q = %q", ExperimentalProxyModeAnnotation, proxyMode)
		}
	}
	if proxyMode == proxyModeUserspace {
		glog.V(1).Infof("Annotation demands userspace proxy")
		return proxyModeUserspace
	}
	return tryIPTablesProxy(iptver, kcompat)
}

func tryIPTablesProxy(iptver iptables.IPTablesVersioner, kcompat iptables.KernelCompatTester) string {
	// guaranteed false on error, error only necessary for debugging
	useIPTablesProxy, err := iptables.CanUseIPTablesProxier(iptver, kcompat)
	if err != nil {
		glog.Errorf("Can't determine whether to use iptables proxy, using userspace proxier: %v", err)
		return proxyModeUserspace
	}
	if useIPTablesProxy {
		return proxyModeIPTables
	}
	// Fallback.
	glog.V(1).Infof("Can't use iptables proxy, using userspace proxier")
	return proxyModeUserspace
}

func (s *ProxyServer) birthCry() {
	s.Recorder.Eventf(s.NodeRef, api.EventTypeNormal, "Starting", "Starting kube-proxy.")
}

func getNodeIP(client *kubeclient.Client, hostname string) net.IP {
	var nodeIP net.IP
	node, err := client.Nodes().Get(hostname)
	if err != nil {
		glog.Warningf("Failed to retrieve node info: %v", err)
		return nil
	}
	nodeIP, err = nodeutil.GetNodeHostIP(node)
	if err != nil {
		glog.Warningf("Failed to retrieve node IP: %v", err)
		return nil
	}
	return nodeIP
}
