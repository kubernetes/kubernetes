//go:build !windows
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
	goruntime "runtime"
	"strings"
	"time"

	"github.com/google/cadvisor/machine"
	"github.com/google/cadvisor/utils/sysfs"
	"k8s.io/apimachinery/pkg/watch"

	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/client-go/tools/cache"
	"k8s.io/client-go/tools/events"

	"k8s.io/apimachinery/pkg/fields"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	clientset "k8s.io/client-go/kubernetes"
	toolswatch "k8s.io/client-go/tools/watch"
	"k8s.io/component-base/configz"
	"k8s.io/component-base/metrics"
	nodeutil "k8s.io/component-helpers/node/util"
	utilsysctl "k8s.io/component-helpers/node/util/sysctl"
	"k8s.io/kubernetes/pkg/proxy"
	proxyconfigapi "k8s.io/kubernetes/pkg/proxy/apis/config"
	"k8s.io/kubernetes/pkg/proxy/apis/config/scheme"
	"k8s.io/kubernetes/pkg/proxy/healthcheck"
	"k8s.io/kubernetes/pkg/proxy/iptables"
	"k8s.io/kubernetes/pkg/proxy/ipvs"
	proxymetrics "k8s.io/kubernetes/pkg/proxy/metrics"
	proxyutil "k8s.io/kubernetes/pkg/proxy/util"
	proxyutiliptables "k8s.io/kubernetes/pkg/proxy/util/iptables"
	utilipset "k8s.io/kubernetes/pkg/util/ipset"
	utiliptables "k8s.io/kubernetes/pkg/util/iptables"
	utilipvs "k8s.io/kubernetes/pkg/util/ipvs"
	"k8s.io/utils/exec"
	netutils "k8s.io/utils/net"

	"k8s.io/klog/v2"
)

// timeoutForNodePodCIDR is the time to wait for allocators to assign a PodCIDR to the
// node after it is registered.
var timeoutForNodePodCIDR = 5 * time.Minute

func (o *Options) platformApplyDefaults(config *proxyconfigapi.KubeProxyConfiguration) {
	if config.Mode == "" {
		klog.InfoS("Using iptables proxy")
		config.Mode = proxyconfigapi.ProxyModeIPTables
	}

	if config.DetectLocalMode == "" {
		klog.V(4).InfoS("Defaulting detect-local-mode", "localModeClusterCIDR", string(proxyconfigapi.LocalModeClusterCIDR))
		config.DetectLocalMode = proxyconfigapi.LocalModeClusterCIDR
	}
	klog.V(2).InfoS("DetectLocalMode", "localMode", string(config.DetectLocalMode))
}

// NewProxyServer returns a new ProxyServer.
func NewProxyServer(o *Options) (*ProxyServer, error) {
	return newProxyServer(o.config, o.master)
}

func newProxyServer(
	config *proxyconfigapi.KubeProxyConfiguration,
	master string) (*ProxyServer, error) {

	if c, err := configz.New(proxyconfigapi.GroupName); err == nil {
		c.Set(config)
	} else {
		return nil, fmt.Errorf("unable to register configz: %s", err)
	}

	var ipvsInterface utilipvs.Interface
	var ipsetInterface utilipset.Interface

	if len(config.ShowHiddenMetricsForVersion) > 0 {
		metrics.SetShowHidden()
	}

	hostname, err := nodeutil.GetHostname(config.HostnameOverride)
	if err != nil {
		return nil, err
	}

	client, err := createClient(config.ClientConnection, master)
	if err != nil {
		return nil, err
	}

	nodeIP := detectNodeIP(client, hostname, config.BindAddress)
	klog.InfoS("Detected node IP", "address", nodeIP.String())

	// Create event recorder
	eventBroadcaster := events.NewBroadcaster(&events.EventSinkImpl{Interface: client.EventsV1()})
	recorder := eventBroadcaster.NewRecorder(scheme.Scheme, "kube-proxy")

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

	var nodeInfo *v1.Node
	if config.DetectLocalMode == proxyconfigapi.LocalModeNodeCIDR {
		klog.InfoS("Watching for node, awaiting podCIDR allocation", "hostname", hostname)
		nodeInfo, err = waitForPodCIDR(client, hostname)
		if err != nil {
			return nil, err
		}
		klog.InfoS("NodeInfo", "podCIDR", nodeInfo.Spec.PodCIDR, "podCIDRs", nodeInfo.Spec.PodCIDRs)
	}

	primaryFamily := v1.IPv4Protocol
	primaryProtocol := utiliptables.ProtocolIPv4
	if netutils.IsIPv6(nodeIP) {
		primaryFamily = v1.IPv6Protocol
		primaryProtocol = utiliptables.ProtocolIPv6
	}
	execer := exec.New()
	iptInterface := utiliptables.New(execer, primaryProtocol)

	var ipt [2]utiliptables.Interface
	dualStack := true // While we assume that node supports, we do further checks below

	// Create iptables handlers for both families, one is already created
	// Always ordered as IPv4, IPv6
	if primaryProtocol == utiliptables.ProtocolIPv4 {
		ipt[0] = iptInterface
		ipt[1] = utiliptables.New(execer, utiliptables.ProtocolIPv6)
	} else {
		ipt[0] = utiliptables.New(execer, utiliptables.ProtocolIPv4)
		ipt[1] = iptInterface
	}

	nodePortAddresses := config.NodePortAddresses

	if !ipt[0].Present() {
		return nil, fmt.Errorf("iptables is not supported for primary IP family %q", primaryProtocol)
	} else if !ipt[1].Present() {
		klog.InfoS("kube-proxy running in single-stack mode: secondary ipFamily is not supported", "ipFamily", ipt[1].Protocol())
		dualStack = false

		// Validate NodePortAddresses is single-stack
		npaByFamily := proxyutil.MapCIDRsByIPFamily(config.NodePortAddresses)
		secondaryFamily := proxyutil.OtherIPFamily(primaryFamily)
		badAddrs := npaByFamily[secondaryFamily]
		if len(badAddrs) > 0 {
			klog.InfoS("Ignoring --nodeport-addresses of the wrong family", "ipFamily", secondaryFamily, "addresses", badAddrs)
			nodePortAddresses = npaByFamily[primaryFamily]
		}
	}

	if config.Mode == proxyconfigapi.ProxyModeIPTables {
		klog.InfoS("Using iptables Proxier")

		if dualStack {
			klog.InfoS("kube-proxy running in dual-stack mode", "ipFamily", iptInterface.Protocol())
			klog.InfoS("Creating dualStackProxier for iptables")
			// Always ordered to match []ipt
			var localDetectors [2]proxyutiliptables.LocalTrafficDetector
			localDetectors, err = getDualStackLocalDetectorTuple(config.DetectLocalMode, config, ipt, nodeInfo)
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
				*config.IPTables.LocalhostNodePorts,
				int(*config.IPTables.MasqueradeBit),
				localDetectors,
				hostname,
				nodeIPTuple(config.BindAddress),
				recorder,
				healthzServer,
				nodePortAddresses,
			)
		} else {
			// Create a single-stack proxier if and only if the node does not support dual-stack (i.e, no iptables support).
			var localDetector proxyutiliptables.LocalTrafficDetector
			localDetector, err = getLocalDetector(config.DetectLocalMode, config, iptInterface, nodeInfo)
			if err != nil {
				return nil, fmt.Errorf("unable to create proxier: %v", err)
			}

			// TODO this has side effects that should only happen when Run() is invoked.
			proxier, err = iptables.NewProxier(
				primaryFamily,
				iptInterface,
				utilsysctl.New(),
				execer,
				config.IPTables.SyncPeriod.Duration,
				config.IPTables.MinSyncPeriod.Duration,
				config.IPTables.MasqueradeAll,
				*config.IPTables.LocalhostNodePorts,
				int(*config.IPTables.MasqueradeBit),
				localDetector,
				hostname,
				nodeIP,
				recorder,
				healthzServer,
				nodePortAddresses,
			)
		}

		if err != nil {
			return nil, fmt.Errorf("unable to create proxier: %v", err)
		}
		proxymetrics.RegisterMetrics()
	} else if config.Mode == proxyconfigapi.ProxyModeIPVS {
		kernelHandler := ipvs.NewLinuxKernelHandler()
		ipsetInterface = utilipset.New(execer)
		ipvsInterface = utilipvs.New()
		if err := ipvs.CanUseIPVSProxier(ipvsInterface, ipsetInterface, config.IPVS.Scheduler); err != nil {
			return nil, fmt.Errorf("can't use the IPVS proxier: %v", err)
		}

		klog.InfoS("Using ipvs Proxier")
		if dualStack {
			klog.InfoS("Creating dualStackProxier for ipvs")

			nodeIPs := nodeIPTuple(config.BindAddress)

			// Always ordered to match []ipt
			var localDetectors [2]proxyutiliptables.LocalTrafficDetector
			localDetectors, err = getDualStackLocalDetectorTuple(config.DetectLocalMode, config, ipt, nodeInfo)
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
				nodePortAddresses,
				kernelHandler,
			)
		} else {
			var localDetector proxyutiliptables.LocalTrafficDetector
			localDetector, err = getLocalDetector(config.DetectLocalMode, config, iptInterface, nodeInfo)
			if err != nil {
				return nil, fmt.Errorf("unable to create proxier: %v", err)
			}

			proxier, err = ipvs.NewProxier(
				primaryFamily,
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
				nodePortAddresses,
				kernelHandler,
			)
		}
		if err != nil {
			return nil, fmt.Errorf("unable to create proxier: %v", err)
		}
		proxymetrics.RegisterMetrics()
	}

	return &ProxyServer{
		Config:        config,
		Client:        client,
		Proxier:       proxier,
		Broadcaster:   eventBroadcaster,
		Recorder:      recorder,
		Conntracker:   &realConntracker{},
		NodeRef:       nodeRef,
		HealthzServer: healthzServer,
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
		// don't process delete events
		if event.Type != watch.Modified && event.Type != watch.Added {
			return false, nil
		}

		n, ok := event.Object.(*v1.Node)
		if !ok {
			return false, fmt.Errorf("event object not of type Node")
		}
		// don't consider the node if is going to be deleted and keep waiting
		if !n.DeletionTimestamp.IsZero() {
			return false, nil
		}
		return n.Spec.PodCIDR != "" && len(n.Spec.PodCIDRs) > 0, nil
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

func detectNumCPU() int {
	// try get numCPU from /sys firstly due to a known issue (https://github.com/kubernetes/kubernetes/issues/99225)
	_, numCPU, err := machine.GetTopology(sysfs.NewRealSysFs())
	if err != nil || numCPU < 1 {
		return goruntime.NumCPU()
	}
	return numCPU
}

func getLocalDetector(mode proxyconfigapi.LocalMode, config *proxyconfigapi.KubeProxyConfiguration, ipt utiliptables.Interface, nodeInfo *v1.Node) (proxyutiliptables.LocalTrafficDetector, error) {
	switch mode {
	case proxyconfigapi.LocalModeClusterCIDR:
		// LocalModeClusterCIDR is the default if --detect-local-mode wasn't passed,
		// but --cluster-cidr is optional.
		if len(strings.TrimSpace(config.ClusterCIDR)) == 0 {
			klog.InfoS("Detect-local-mode set to ClusterCIDR, but no cluster CIDR defined")
			break
		}
		return proxyutiliptables.NewDetectLocalByCIDR(config.ClusterCIDR, ipt)
	case proxyconfigapi.LocalModeNodeCIDR:
		if len(strings.TrimSpace(nodeInfo.Spec.PodCIDR)) == 0 {
			klog.InfoS("Detect-local-mode set to NodeCIDR, but no PodCIDR defined at node")
			break
		}
		return proxyutiliptables.NewDetectLocalByCIDR(nodeInfo.Spec.PodCIDR, ipt)
	case proxyconfigapi.LocalModeBridgeInterface:
		return proxyutiliptables.NewDetectLocalByBridgeInterface(config.DetectLocal.BridgeInterface)
	case proxyconfigapi.LocalModeInterfaceNamePrefix:
		return proxyutiliptables.NewDetectLocalByInterfaceNamePrefix(config.DetectLocal.InterfaceNamePrefix)
	}
	klog.InfoS("Defaulting to no-op detect-local", "detectLocalMode", string(mode))
	return proxyutiliptables.NewNoOpLocalDetector(), nil
}

func getDualStackLocalDetectorTuple(mode proxyconfigapi.LocalMode, config *proxyconfigapi.KubeProxyConfiguration, ipt [2]utiliptables.Interface, nodeInfo *v1.Node) ([2]proxyutiliptables.LocalTrafficDetector, error) {
	var err error
	localDetectors := [2]proxyutiliptables.LocalTrafficDetector{proxyutiliptables.NewNoOpLocalDetector(), proxyutiliptables.NewNoOpLocalDetector()}
	switch mode {
	case proxyconfigapi.LocalModeClusterCIDR:
		// LocalModeClusterCIDR is the default if --detect-local-mode wasn't passed,
		// but --cluster-cidr is optional.
		if len(strings.TrimSpace(config.ClusterCIDR)) == 0 {
			klog.InfoS("Detect-local-mode set to ClusterCIDR, but no cluster CIDR defined")
			break
		}

		clusterCIDRs := cidrTuple(config.ClusterCIDR)

		if len(strings.TrimSpace(clusterCIDRs[0])) == 0 {
			klog.InfoS("Detect-local-mode set to ClusterCIDR, but no IPv4 cluster CIDR defined, defaulting to no-op detect-local for IPv4")
		} else {
			localDetectors[0], err = proxyutiliptables.NewDetectLocalByCIDR(clusterCIDRs[0], ipt[0])
			if err != nil { // don't loose the original error
				return localDetectors, err
			}
		}

		if len(strings.TrimSpace(clusterCIDRs[1])) == 0 {
			klog.InfoS("Detect-local-mode set to ClusterCIDR, but no IPv6 cluster CIDR defined, defaulting to no-op detect-local for IPv6")
		} else {
			localDetectors[1], err = proxyutiliptables.NewDetectLocalByCIDR(clusterCIDRs[1], ipt[1])
		}
		return localDetectors, err
	case proxyconfigapi.LocalModeNodeCIDR:
		if len(strings.TrimSpace(nodeInfo.Spec.PodCIDR)) == 0 {
			klog.InfoS("No node info available to configure detect-local-mode NodeCIDR")
			break
		}
		// localDetectors, like ipt, need to be of the order [IPv4, IPv6], but PodCIDRs is setup so that PodCIDRs[0] == PodCIDR.
		// so have to handle the case where PodCIDR can be IPv6 and set that to localDetectors[1]
		if netutils.IsIPv6CIDRString(nodeInfo.Spec.PodCIDR) {
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
	case proxyconfigapi.LocalModeBridgeInterface, proxyconfigapi.LocalModeInterfaceNamePrefix:
		localDetector, err := getLocalDetector(mode, config, ipt[0], nodeInfo)
		if err == nil {
			localDetectors[0] = localDetector
			localDetectors[1] = localDetector
		}
		return localDetectors, err
	default:
		klog.InfoS("Unknown detect-local-mode", "detectLocalMode", mode)
	}
	klog.InfoS("Defaulting to no-op detect-local", "detectLocalMode", string(mode))
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
		if netutils.IsIPv6CIDRString(cidr) && !foundIPv6 {
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

// cleanupAndExit remove iptables rules and ipset/ipvs rules
func cleanupAndExit() error {
	execer := exec.New()

	// cleanup IPv6 and IPv4 iptables rules, regardless of current configuration
	ipts := []utiliptables.Interface{
		utiliptables.New(execer, utiliptables.ProtocolIPv4),
		utiliptables.New(execer, utiliptables.ProtocolIPv6),
	}

	ipsetInterface := utilipset.New(execer)
	ipvsInterface := utilipvs.New()

	var encounteredError bool
	for _, ipt := range ipts {
		encounteredError = iptables.CleanupLeftovers(ipt) || encounteredError
		encounteredError = ipvs.CleanupLeftovers(ipvsInterface, ipt, ipsetInterface) || encounteredError
	}
	if encounteredError {
		return errors.New("encountered an error while tearing down rules")
	}

	return nil
}
