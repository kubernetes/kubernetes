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
	"context"
	"errors"
	"fmt"
	"net"
	"strings"
	"time"

	"k8s.io/apimachinery/pkg/watch"

	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/client-go/tools/cache"

	"k8s.io/apimachinery/pkg/fields"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	utilnet "k8s.io/apimachinery/pkg/util/net"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/client-go/tools/record"
	toolswatch "k8s.io/client-go/tools/watch"
	"k8s.io/component-base/configz"
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
	proxyutiliptables "k8s.io/kubernetes/pkg/proxy/util/iptables"
	utilipset "k8s.io/kubernetes/pkg/util/ipset"
	utiliptables "k8s.io/kubernetes/pkg/util/iptables"
	utilipvs "k8s.io/kubernetes/pkg/util/ipvs"
	utilnode "k8s.io/kubernetes/pkg/util/node"
	utilsysctl "k8s.io/kubernetes/pkg/util/sysctl"
	"k8s.io/utils/exec"
	utilsnet "k8s.io/utils/net"

	"k8s.io/klog/v2"
)

// timeoutForNodePodCIDR is the time to wait for allocators to assign a PodCIDR to the
// node after it is registered.
var timeoutForNodePodCIDR = 5 * time.Minute

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

	hostname, err := utilnode.GetHostname(config.HostnameOverride)
	if err != nil {
		return nil, err
	}

	client, eventClient, err := createClients(config.ClientConnection, master)
	if err != nil {
		return nil, err
	}

	nodeIP := detectNodeIP(client, hostname, config.BindAddress)

	protocol := utiliptables.ProtocolIPv4
	if utilsnet.IsIPv6(nodeIP) {
		klog.V(0).Infof("kube-proxy node IP is an IPv6 address (%s), assume IPv6 operation", nodeIP.String())
		protocol = utiliptables.ProtocolIPv6
	} else {
		klog.V(0).Infof("kube-proxy node IP is an IPv4 address (%s), assume IPv4 operation", nodeIP.String())
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
	canUseIPVS, err := ipvs.CanUseIPVSProxier(kernelHandler, ipsetInterface)
	if string(config.Mode) == proxyModeIPVS && err != nil {
		klog.Errorf("Can't use the IPVS proxier: %v", err)
	}

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

	// Create event recorder
	eventBroadcaster := record.NewBroadcaster()
	recorder := eventBroadcaster.NewRecorder(proxyconfigscheme.Scheme, v1.EventSource{Component: "kube-proxy", Host: hostname})

	nodeRef := &v1.ObjectReference{
		Kind:      "Node",
		Name:      hostname,
		UID:       types.UID(hostname),
		Namespace: "",
	}

	var healthzServer healthcheck.ProxierHealthUpdater
	if len(config.HealthzBindAddress) > 0 {
		healthzServer = healthcheck.NewProxierHealthServer(config.HealthzBindAddress, 2*config.IPTables.SyncPeriod.Duration, recorder, nodeRef)
	}

	var proxier proxy.Provider
	var detectLocalMode proxyconfigapi.LocalMode

	proxyMode := getProxyMode(string(config.Mode), canUseIPVS, iptables.LinuxKernelCompatTester{})
	detectLocalMode, err = getDetectLocalMode(config)
	if err != nil {
		return nil, fmt.Errorf("cannot determine detect-local-mode: %v", err)
	}

	var nodeInfo *v1.Node
	if detectLocalMode == proxyconfigapi.LocalModeNodeCIDR {
		klog.Infof("Watching for node %s, awaiting podCIDR allocation", hostname)
		nodeInfo, err = waitForPodCIDR(client, hostname)
		if err != nil {
			return nil, err
		}
		klog.Infof("NodeInfo PodCIDR: %v, PodCIDRs: %v", nodeInfo.Spec.PodCIDR, nodeInfo.Spec.PodCIDRs)
	}

	klog.V(2).Info("DetectLocalMode: '", string(detectLocalMode), "'")

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
			if iptInterface.IsIPv6() {
				ipt[1] = iptInterface
				ipt[0] = utiliptables.New(execer, utiliptables.ProtocolIPv4)
			} else {
				ipt[0] = iptInterface
				ipt[1] = utiliptables.New(execer, utiliptables.ProtocolIPv6)
			}

			// Always ordered to match []ipt
			var localDetectors [2]proxyutiliptables.LocalTrafficDetector
			localDetectors, err = getDualStackLocalDetectorTuple(detectLocalMode, config, ipt, nodeInfo)
			if err != nil {
				return nil, fmt.Errorf("unable to create proxier: %v", err)
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
				localDetectors,
				hostname,
				nodeIPTuple(config.BindAddress),
				recorder,
				healthzServer,
				config.NodePortAddresses,
			)
		} else { // Create a single-stack proxier.
			var localDetector proxyutiliptables.LocalTrafficDetector
			localDetector, err = getLocalDetector(detectLocalMode, config, iptInterface, nodeInfo)
			if err != nil {
				return nil, fmt.Errorf("unable to create proxier: %v", err)
			}

			// TODO this has side effects that should only happen when Run() is invoked.
			proxier, err = iptables.NewProxier(
				iptInterface,
				utilsysctl.New(),
				execer,
				config.IPTables.SyncPeriod.Duration,
				config.IPTables.MinSyncPeriod.Duration,
				config.IPTables.MasqueradeAll,
				int(*config.IPTables.MasqueradeBit),
				localDetector,
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
			if iptInterface.IsIPv6() {
				ipt[1] = iptInterface
				ipt[0] = utiliptables.New(execer, utiliptables.ProtocolIPv4)
			} else {
				ipt[0] = iptInterface
				ipt[1] = utiliptables.New(execer, utiliptables.ProtocolIPv6)
			}

			nodeIPs := nodeIPTuple(config.BindAddress)

			// Always ordered to match []ipt
			var localDetectors [2]proxyutiliptables.LocalTrafficDetector
			localDetectors, err = getDualStackLocalDetectorTuple(detectLocalMode, config, ipt, nodeInfo)
			if err != nil {
				return nil, fmt.Errorf("unable to create proxier: %v", err)
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
				localDetectors,
				hostname,
				nodeIPs,
				recorder,
				healthzServer,
				config.IPVS.Scheduler,
				config.NodePortAddresses,
				kernelHandler,
			)
		} else {
			var localDetector proxyutiliptables.LocalTrafficDetector
			localDetector, err = getLocalDetector(detectLocalMode, config, iptInterface, nodeInfo)
			if err != nil {
				return nil, fmt.Errorf("unable to create proxier: %v", err)
			}

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
				localDetector,
				hostname,
				nodeIP,
				recorder,
				healthzServer,
				config.IPVS.Scheduler,
				config.NodePortAddresses,
				kernelHandler,
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
		BindAddressHardFail:    config.BindAddressHardFail,
		EnableProfiling:        config.EnableProfiling,
		OOMScoreAdj:            config.OOMScoreAdj,
		ConfigSyncPeriod:       config.ConfigSyncPeriod.Duration,
		HealthzServer:          healthzServer,
		UseEndpointSlices:      utilfeature.DefaultFeatureGate.Enabled(features.EndpointSliceProxying),
	}, nil
}

func waitForPodCIDR(client clientset.Interface, nodeName string) (*v1.Node, error) {
	// since allocators can assign the podCIDR after the node registers, we do a watch here to wait
	// for podCIDR to be assigned, instead of assuming that the Get() on startup will have it.
	ctx, cancelFunc := context.WithTimeout(context.TODO(), timeoutForNodePodCIDR)
	defer cancelFunc()

	fieldSelector := fields.OneTermEqualSelector("metadata.name", nodeName).String()
	lw := &cache.ListWatch{
		ListFunc: func(options metav1.ListOptions) (object runtime.Object, e error) {
			options.FieldSelector = fieldSelector
			return client.CoreV1().Nodes().List(ctx, options)
		},
		WatchFunc: func(options metav1.ListOptions) (i watch.Interface, e error) {
			options.FieldSelector = fieldSelector
			return client.CoreV1().Nodes().Watch(ctx, options)
		},
	}
	condition := func(event watch.Event) (bool, error) {
		if n, ok := event.Object.(*v1.Node); ok {
			return n.Spec.PodCIDR != "" && len(n.Spec.PodCIDRs) > 0, nil
		}
		return false, fmt.Errorf("event object not of type Node")
	}

	evt, err := toolswatch.UntilWithSync(ctx, lw, &v1.Node{}, nil, condition)
	if err != nil {
		return nil, fmt.Errorf("timeout waiting for PodCIDR allocation to configure detect-local-mode %v: %v", proxyconfigapi.LocalModeNodeCIDR, err)
	}
	if n, ok := evt.Object.(*v1.Node); ok {
		return n, nil
	}
	return nil, fmt.Errorf("event object not of type node")
}

// detectNodeIP returns the nodeIP used by the proxier
// The order of precedence is:
// 1. config.bindAddress if bindAddress is not 0.0.0.0 or ::
// 2. the primary IP from the Node object, if set
// 3. if no IP is found it defaults to 127.0.0.1 and IPv4
func detectNodeIP(client clientset.Interface, hostname, bindAddress string) net.IP {
	nodeIP := net.ParseIP(bindAddress)
	if nodeIP.IsUnspecified() {
		nodeIP = utilnode.GetNodeIP(client, hostname)
	}
	if nodeIP == nil {
		klog.V(0).Infof("can't determine this node's IP, assuming 127.0.0.1; if this is incorrect, please set the --bind-address flag")
		nodeIP = net.ParseIP("127.0.0.1")
	}
	return nodeIP
}

func getDetectLocalMode(config *proxyconfigapi.KubeProxyConfiguration) (proxyconfigapi.LocalMode, error) {
	mode := config.DetectLocalMode
	switch mode {
	case proxyconfigapi.LocalModeClusterCIDR, proxyconfigapi.LocalModeNodeCIDR:
		return mode, nil
	default:
		if strings.TrimSpace(mode.String()) != "" {
			return mode, fmt.Errorf("unknown detect-local-mode: %v", mode)
		}
		klog.V(4).Info("Defaulting detect-local-mode to ", string(proxyconfigapi.LocalModeClusterCIDR))
		return proxyconfigapi.LocalModeClusterCIDR, nil
	}
}

func getLocalDetector(mode proxyconfigapi.LocalMode, config *proxyconfigapi.KubeProxyConfiguration, ipt utiliptables.Interface, nodeInfo *v1.Node) (proxyutiliptables.LocalTrafficDetector, error) {
	switch mode {
	case proxyconfigapi.LocalModeClusterCIDR:
		if len(strings.TrimSpace(config.ClusterCIDR)) == 0 {
			klog.Warning("detect-local-mode set to ClusterCIDR, but no cluster CIDR defined")
			break
		}
		return proxyutiliptables.NewDetectLocalByCIDR(config.ClusterCIDR, ipt)
	case proxyconfigapi.LocalModeNodeCIDR:
		if len(strings.TrimSpace(nodeInfo.Spec.PodCIDR)) == 0 {
			klog.Warning("detect-local-mode set to NodeCIDR, but no PodCIDR defined at node")
			break
		}
		return proxyutiliptables.NewDetectLocalByCIDR(nodeInfo.Spec.PodCIDR, ipt)
	}
	klog.V(0).Info("detect-local-mode: ", string(mode), " , defaulting to no-op detect-local")
	return proxyutiliptables.NewNoOpLocalDetector(), nil
}

func getDualStackLocalDetectorTuple(mode proxyconfigapi.LocalMode, config *proxyconfigapi.KubeProxyConfiguration, ipt [2]utiliptables.Interface, nodeInfo *v1.Node) ([2]proxyutiliptables.LocalTrafficDetector, error) {
	var err error
	localDetectors := [2]proxyutiliptables.LocalTrafficDetector{proxyutiliptables.NewNoOpLocalDetector(), proxyutiliptables.NewNoOpLocalDetector()}
	switch mode {
	case proxyconfigapi.LocalModeClusterCIDR:
		if len(strings.TrimSpace(config.ClusterCIDR)) == 0 {
			klog.Warning("detect-local-mode set to ClusterCIDR, but no cluster CIDR defined")
			break
		}

		clusterCIDRs := cidrTuple(config.ClusterCIDR)

		if len(strings.TrimSpace(clusterCIDRs[0])) == 0 {
			klog.Warning("detect-local-mode set to ClusterCIDR, but no IPv4 cluster CIDR defined, defaulting to no-op detect-local for IPv4")
		} else {
			localDetectors[0], err = proxyutiliptables.NewDetectLocalByCIDR(clusterCIDRs[0], ipt[0])
			if err != nil { // don't loose the original error
				return localDetectors, err
			}
		}

		if len(strings.TrimSpace(clusterCIDRs[1])) == 0 {
			klog.Warning("detect-local-mode set to ClusterCIDR, but no IPv6 cluster CIDR defined, , defaulting to no-op detect-local for IPv6")
		} else {
			localDetectors[1], err = proxyutiliptables.NewDetectLocalByCIDR(clusterCIDRs[1], ipt[1])
		}
		return localDetectors, err
	case proxyconfigapi.LocalModeNodeCIDR:
		if nodeInfo == nil || len(strings.TrimSpace(nodeInfo.Spec.PodCIDR)) == 0 {
			klog.Warning("No node info available to configure detect-local-mode NodeCIDR")
			break
		}
		// localDetectors, like ipt, need to be of the order [IPv4, IPv6], but PodCIDRs is setup so that PodCIDRs[0] == PodCIDR.
		// so have to handle the case where PodCIDR can be IPv6 and set that to localDetectors[1]
		if utilsnet.IsIPv6CIDRString(nodeInfo.Spec.PodCIDR) {
			localDetectors[1], err = proxyutiliptables.NewDetectLocalByCIDR(nodeInfo.Spec.PodCIDR, ipt[1])
			if err != nil {
				return localDetectors, err
			}
			if len(nodeInfo.Spec.PodCIDRs) > 1 {
				localDetectors[0], err = proxyutiliptables.NewDetectLocalByCIDR(nodeInfo.Spec.PodCIDRs[1], ipt[0])
			}
		} else {
			localDetectors[0], err = proxyutiliptables.NewDetectLocalByCIDR(nodeInfo.Spec.PodCIDR, ipt[0])
			if err != nil {
				return localDetectors, err
			}
			if len(nodeInfo.Spec.PodCIDRs) > 1 {
				localDetectors[1], err = proxyutiliptables.NewDetectLocalByCIDR(nodeInfo.Spec.PodCIDRs[1], ipt[1])
			}
		}
		return localDetectors, err
	default:
		klog.Warningf("unknown detect-local-mode: %v", mode)
	}
	klog.Warning("detect-local-mode: ", string(mode), " , defaulting to no-op detect-local")
	return localDetectors, nil
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

func getProxyMode(proxyMode string, canUseIPVS bool, kcompat iptables.KernelCompatTester) string {
	switch proxyMode {
	case proxyModeUserspace:
		return proxyModeUserspace
	case proxyModeIPTables:
		return tryIPTablesProxy(kcompat)
	case proxyModeIPVS:
		return tryIPVSProxy(canUseIPVS, kcompat)
	}
	klog.Warningf("Unknown proxy mode %q, assuming iptables proxy", proxyMode)
	return tryIPTablesProxy(kcompat)
}

func tryIPVSProxy(canUseIPVS bool, kcompat iptables.KernelCompatTester) string {
	if canUseIPVS {
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
