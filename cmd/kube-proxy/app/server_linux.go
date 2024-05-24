//go:build linux
// +build linux

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

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/watch"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/client-go/tools/cache"
	toolswatch "k8s.io/client-go/tools/watch"
	utilsysctl "k8s.io/component-helpers/node/util/sysctl"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/proxy"
	proxyconfigapi "k8s.io/kubernetes/pkg/proxy/apis/config"
	"k8s.io/kubernetes/pkg/proxy/iptables"
	"k8s.io/kubernetes/pkg/proxy/ipvs"
	utilipset "k8s.io/kubernetes/pkg/proxy/ipvs/ipset"
	utilipvs "k8s.io/kubernetes/pkg/proxy/ipvs/util"
	proxymetrics "k8s.io/kubernetes/pkg/proxy/metrics"
	"k8s.io/kubernetes/pkg/proxy/nftables"
	proxyutil "k8s.io/kubernetes/pkg/proxy/util"
	proxyutiliptables "k8s.io/kubernetes/pkg/proxy/util/iptables"
	utiliptables "k8s.io/kubernetes/pkg/util/iptables"
	"k8s.io/utils/exec"
)

// timeoutForNodePodCIDR is the time to wait for allocators to assign a PodCIDR to the
// node after it is registered.
var timeoutForNodePodCIDR = 5 * time.Minute

// platformApplyDefaults is called after parsing command-line flags and/or reading the
// config file, to apply platform-specific default values to config.
func (o *Options) platformApplyDefaults(config *proxyconfigapi.KubeProxyConfiguration) {
	if config.Mode == "" {
		o.logger.Info("Using iptables proxy")
		config.Mode = proxyconfigapi.ProxyModeIPTables
	}

	if config.DetectLocalMode == "" {
		o.logger.V(4).Info("Defaulting detect-local-mode", "localModeClusterCIDR", string(proxyconfigapi.LocalModeClusterCIDR))
		config.DetectLocalMode = proxyconfigapi.LocalModeClusterCIDR
	}
	o.logger.V(2).Info("DetectLocalMode", "localMode", string(config.DetectLocalMode))
}

// platformSetup is called after setting up the ProxyServer, but before creating the
// Proxier. It should fill in any platform-specific fields and perform other
// platform-specific setup.
func (s *ProxyServer) platformSetup() error {
	if s.Config.DetectLocalMode == proxyconfigapi.LocalModeNodeCIDR {
		s.logger.Info("Watching for node, awaiting podCIDR allocation", "hostname", s.Hostname)
		node, err := waitForPodCIDR(s.Client, s.Hostname)
		if err != nil {
			return err
		}
		s.podCIDRs = node.Spec.PodCIDRs
		s.logger.Info("NodeInfo", "podCIDRs", node.Spec.PodCIDRs)
	}

	err := s.setupConntrack()
	if err != nil {
		return err
	}

	proxymetrics.RegisterMetrics()
	return nil
}

// isIPTablesBased checks whether mode is based on iptables rather than nftables
func isIPTablesBased(mode proxyconfigapi.ProxyMode) bool {
	return mode == proxyconfigapi.ProxyModeIPTables || mode == proxyconfigapi.ProxyModeIPVS
}

// getIPTables returns an array of [IPv4, IPv6] utiliptables.Interfaces. If primaryFamily
// is not v1.IPFamilyUnknown then it will also separately return the interface for just
// that family.
func getIPTables(primaryFamily v1.IPFamily) ([2]utiliptables.Interface, utiliptables.Interface) {
	execer := exec.New()

	// Create iptables handlers for both families. Always ordered as IPv4, IPv6
	ipt := [2]utiliptables.Interface{
		utiliptables.New(execer, utiliptables.ProtocolIPv4),
		utiliptables.New(execer, utiliptables.ProtocolIPv6),
	}

	var iptInterface utiliptables.Interface
	if primaryFamily == v1.IPv4Protocol {
		iptInterface = ipt[0]
	} else if primaryFamily == v1.IPv6Protocol {
		iptInterface = ipt[1]
	}

	return ipt, iptInterface
}

// platformCheckSupported is called immediately before creating the Proxier, to check
// what IP families are supported (and whether the configuration is usable at all).
func (s *ProxyServer) platformCheckSupported() (ipv4Supported, ipv6Supported, dualStackSupported bool, err error) {
	if isIPTablesBased(s.Config.Mode) {
		ipt, _ := getIPTables(v1.IPFamilyUnknown)
		ipv4Supported = ipt[0].Present()
		ipv6Supported = ipt[1].Present()

		if !ipv4Supported && !ipv6Supported {
			err = fmt.Errorf("iptables is not available on this host")
		} else if !ipv4Supported {
			s.logger.Info("No iptables support for family", "ipFamily", v1.IPv4Protocol)
		} else if !ipv6Supported {
			s.logger.Info("No iptables support for family", "ipFamily", v1.IPv6Protocol)
		}
	} else {
		// Assume support for both families.
		// FIXME: figure out how to check for kernel IPv6 support using nft
		ipv4Supported, ipv6Supported = true, true
	}

	// The Linux proxies can always support dual-stack if they can support both IPv4
	// and IPv6.
	dualStackSupported = ipv4Supported && ipv6Supported
	return
}

// createProxier creates the proxy.Provider
func (s *ProxyServer) createProxier(config *proxyconfigapi.KubeProxyConfiguration, dualStack, initOnly bool) (proxy.Provider, error) {
	var proxier proxy.Provider
	var localDetectors [2]proxyutiliptables.LocalTrafficDetector
	var localDetector proxyutiliptables.LocalTrafficDetector
	var err error

	if config.Mode == proxyconfigapi.ProxyModeIPTables {
		s.logger.Info("Using iptables Proxier")

		if dualStack {
			ipt, _ := getIPTables(s.PrimaryIPFamily)

			localDetectors, err = getDualStackLocalDetectorTuple(s.logger, config.DetectLocalMode, config, s.podCIDRs)
			if err != nil {
				return nil, fmt.Errorf("unable to create proxier: %v", err)
			}

			// TODO this has side effects that should only happen when Run() is invoked.
			proxier, err = iptables.NewDualStackProxier(
				ipt,
				utilsysctl.New(),
				exec.New(),
				config.IPTables.SyncPeriod.Duration,
				config.IPTables.MinSyncPeriod.Duration,
				config.IPTables.MasqueradeAll,
				*config.IPTables.LocalhostNodePorts,
				int(*config.IPTables.MasqueradeBit),
				localDetectors,
				s.Hostname,
				s.NodeIPs,
				s.Recorder,
				s.HealthzServer,
				config.NodePortAddresses,
				initOnly,
			)
		} else {
			// Create a single-stack proxier if and only if the node does not support dual-stack (i.e, no iptables support).
			_, iptInterface := getIPTables(s.PrimaryIPFamily)
			localDetector, err = getLocalDetector(s.logger, s.PrimaryIPFamily, config.DetectLocalMode, config, s.podCIDRs)
			if err != nil {
				return nil, fmt.Errorf("unable to create proxier: %v", err)
			}

			// TODO this has side effects that should only happen when Run() is invoked.
			proxier, err = iptables.NewProxier(
				s.PrimaryIPFamily,
				iptInterface,
				utilsysctl.New(),
				exec.New(),
				config.IPTables.SyncPeriod.Duration,
				config.IPTables.MinSyncPeriod.Duration,
				config.IPTables.MasqueradeAll,
				*config.IPTables.LocalhostNodePorts,
				int(*config.IPTables.MasqueradeBit),
				localDetector,
				s.Hostname,
				s.NodeIPs[s.PrimaryIPFamily],
				s.Recorder,
				s.HealthzServer,
				config.NodePortAddresses,
				initOnly,
			)
		}

		if err != nil {
			return nil, fmt.Errorf("unable to create proxier: %v", err)
		}
	} else if config.Mode == proxyconfigapi.ProxyModeIPVS {
		execer := exec.New()
		ipsetInterface := utilipset.New(execer)
		ipvsInterface := utilipvs.New()
		if err := ipvs.CanUseIPVSProxier(ipvsInterface, ipsetInterface, config.IPVS.Scheduler); err != nil {
			return nil, fmt.Errorf("can't use the IPVS proxier: %v", err)
		}

		s.logger.Info("Using ipvs Proxier")
		if dualStack {
			ipt, _ := getIPTables(s.PrimaryIPFamily)

			// Always ordered to match []ipt
			localDetectors, err = getDualStackLocalDetectorTuple(s.logger, config.DetectLocalMode, config, s.podCIDRs)
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
				s.Hostname,
				s.NodeIPs,
				s.Recorder,
				s.HealthzServer,
				config.IPVS.Scheduler,
				config.NodePortAddresses,
				initOnly,
			)
		} else {
			_, iptInterface := getIPTables(s.PrimaryIPFamily)
			localDetector, err = getLocalDetector(s.logger, s.PrimaryIPFamily, config.DetectLocalMode, config, s.podCIDRs)
			if err != nil {
				return nil, fmt.Errorf("unable to create proxier: %v", err)
			}

			proxier, err = ipvs.NewProxier(
				s.PrimaryIPFamily,
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
				s.Hostname,
				s.NodeIPs[s.PrimaryIPFamily],
				s.Recorder,
				s.HealthzServer,
				config.IPVS.Scheduler,
				config.NodePortAddresses,
				initOnly,
			)
		}
		if err != nil {
			return nil, fmt.Errorf("unable to create proxier: %v", err)
		}
	} else if config.Mode == proxyconfigapi.ProxyModeNFTables {
		s.logger.Info("Using nftables Proxier")

		if dualStack {
			localDetectors, err = getDualStackLocalDetectorTuple(s.logger, config.DetectLocalMode, config, s.podCIDRs)
			if err != nil {
				return nil, fmt.Errorf("unable to create proxier: %v", err)
			}

			// TODO this has side effects that should only happen when Run() is invoked.
			proxier, err = nftables.NewDualStackProxier(
				utilsysctl.New(),
				config.NFTables.SyncPeriod.Duration,
				config.NFTables.MinSyncPeriod.Duration,
				config.NFTables.MasqueradeAll,
				int(*config.NFTables.MasqueradeBit),
				localDetectors,
				s.Hostname,
				s.NodeIPs,
				s.Recorder,
				s.HealthzServer,
				config.NodePortAddresses,
				initOnly,
			)
		} else {
			// Create a single-stack proxier if and only if the node does not support dual-stack
			localDetector, err = getLocalDetector(s.logger, s.PrimaryIPFamily, config.DetectLocalMode, config, s.podCIDRs)
			if err != nil {
				return nil, fmt.Errorf("unable to create proxier: %v", err)
			}

			// TODO this has side effects that should only happen when Run() is invoked.
			proxier, err = nftables.NewProxier(
				s.PrimaryIPFamily,
				utilsysctl.New(),
				config.NFTables.SyncPeriod.Duration,
				config.NFTables.MinSyncPeriod.Duration,
				config.NFTables.MasqueradeAll,
				int(*config.NFTables.MasqueradeBit),
				localDetector,
				s.Hostname,
				s.NodeIPs[s.PrimaryIPFamily],
				s.Recorder,
				s.HealthzServer,
				config.NodePortAddresses,
				initOnly,
			)
		}

		if err != nil {
			return nil, fmt.Errorf("unable to create proxier: %v", err)
		}
	}

	return proxier, nil
}

func (s *ProxyServer) setupConntrack() error {
	ct := &realConntracker{
		logger: s.logger,
	}

	max, err := getConntrackMax(s.logger, s.Config.Conntrack)
	if err != nil {
		return err
	}
	if max > 0 {
		err := ct.SetMax(max)
		if err != nil {
			if err != errReadOnlySysFS {
				return err
			}
			// errReadOnlySysFS is caused by a known docker issue (https://github.com/docker/docker/issues/24000),
			// the only remediation we know is to restart the docker daemon.
			// Here we'll send an node event with specific reason and message, the
			// administrator should decide whether and how to handle this issue,
			// whether to drain the node and restart docker.  Occurs in other container runtimes
			// as well.
			// TODO(random-liu): Remove this when the docker bug is fixed.
			const message = "CRI error: /sys is read-only: " +
				"cannot modify conntrack limits, problems may arise later (If running Docker, see docker issue #24000)"
			s.Recorder.Eventf(s.NodeRef, nil, v1.EventTypeWarning, err.Error(), "StartKubeProxy", message)
		}
	}

	if s.Config.Conntrack.TCPEstablishedTimeout != nil && s.Config.Conntrack.TCPEstablishedTimeout.Duration > 0 {
		timeout := int(s.Config.Conntrack.TCPEstablishedTimeout.Duration / time.Second)
		if err := ct.SetTCPEstablishedTimeout(timeout); err != nil {
			return err
		}
	}

	if s.Config.Conntrack.TCPCloseWaitTimeout != nil && s.Config.Conntrack.TCPCloseWaitTimeout.Duration > 0 {
		timeout := int(s.Config.Conntrack.TCPCloseWaitTimeout.Duration / time.Second)
		if err := ct.SetTCPCloseWaitTimeout(timeout); err != nil {
			return err
		}
	}

	if s.Config.Conntrack.TCPBeLiberal {
		if err := ct.SetTCPBeLiberal(1); err != nil {
			return err
		}
	}

	if s.Config.Conntrack.UDPTimeout.Duration > 0 {
		timeout := int(s.Config.Conntrack.UDPTimeout.Duration / time.Second)
		if err := ct.SetUDPTimeout(timeout); err != nil {
			return err
		}
	}

	if s.Config.Conntrack.UDPStreamTimeout.Duration > 0 {
		timeout := int(s.Config.Conntrack.UDPStreamTimeout.Duration / time.Second)
		if err := ct.SetUDPStreamTimeout(timeout); err != nil {
			return err
		}
	}

	return nil
}

func getConntrackMax(logger klog.Logger, config proxyconfigapi.KubeProxyConntrackConfiguration) (int, error) {
	if config.MaxPerCore != nil && *config.MaxPerCore > 0 {
		floor := 0
		if config.Min != nil {
			floor = int(*config.Min)
		}
		scaled := int(*config.MaxPerCore) * detectNumCPU()
		if scaled > floor {
			logger.V(3).Info("GetConntrackMax: using scaled conntrack-max-per-core")
			return scaled, nil
		}
		logger.V(3).Info("GetConntrackMax: using conntrack-min")
		return floor, nil
	}
	return 0, nil
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

func getLocalDetector(logger klog.Logger, ipFamily v1.IPFamily, mode proxyconfigapi.LocalMode, config *proxyconfigapi.KubeProxyConfiguration, nodePodCIDRs []string) (proxyutiliptables.LocalTrafficDetector, error) {
	switch mode {
	case proxyconfigapi.LocalModeClusterCIDR:
		// LocalModeClusterCIDR is the default if --detect-local-mode wasn't passed,
		// but --cluster-cidr is optional.
		clusterCIDRs := strings.TrimSpace(config.ClusterCIDR)
		if len(clusterCIDRs) == 0 {
			logger.Info("Detect-local-mode set to ClusterCIDR, but no cluster CIDR defined")
			break
		}

		cidrsByFamily := proxyutil.MapCIDRsByIPFamily(strings.Split(clusterCIDRs, ","))
		if len(cidrsByFamily[ipFamily]) != 0 {
			return proxyutiliptables.NewDetectLocalByCIDR(cidrsByFamily[ipFamily][0].String())
		}

		logger.Info("Detect-local-mode set to ClusterCIDR, but no cluster CIDR for family", "ipFamily", ipFamily)

	case proxyconfigapi.LocalModeNodeCIDR:
		cidrsByFamily := proxyutil.MapCIDRsByIPFamily(nodePodCIDRs)
		if len(cidrsByFamily[ipFamily]) != 0 {
			return proxyutiliptables.NewDetectLocalByCIDR(cidrsByFamily[ipFamily][0].String())
		}

		logger.Info("Detect-local-mode set to NodeCIDR, but no PodCIDR defined at node for family", "ipFamily", ipFamily)

	case proxyconfigapi.LocalModeBridgeInterface:
		return proxyutiliptables.NewDetectLocalByBridgeInterface(config.DetectLocal.BridgeInterface)

	case proxyconfigapi.LocalModeInterfaceNamePrefix:
		return proxyutiliptables.NewDetectLocalByInterfaceNamePrefix(config.DetectLocal.InterfaceNamePrefix)
	}

	logger.Info("Defaulting to no-op detect-local")
	return proxyutiliptables.NewNoOpLocalDetector(), nil
}

func getDualStackLocalDetectorTuple(logger klog.Logger, mode proxyconfigapi.LocalMode, config *proxyconfigapi.KubeProxyConfiguration, nodePodCIDRs []string) ([2]proxyutiliptables.LocalTrafficDetector, error) {
	var localDetectors [2]proxyutiliptables.LocalTrafficDetector
	var err error

	localDetectors[0], err = getLocalDetector(logger, v1.IPv4Protocol, mode, config, nodePodCIDRs)
	if err != nil {
		return localDetectors, err
	}
	localDetectors[1], err = getLocalDetector(logger, v1.IPv6Protocol, mode, config, nodePodCIDRs)
	if err != nil {
		return localDetectors, err
	}
	return localDetectors, nil
}

// platformCleanup removes stale kube-proxy rules that can be safely removed. If
// cleanupAndExit is true, it will attempt to remove rules from all known kube-proxy
// modes. If it is false, it will only remove rules that are definitely not in use by the
// currently-configured mode.
func platformCleanup(mode proxyconfigapi.ProxyMode, cleanupAndExit bool) error {
	var encounteredError bool

	// Clean up iptables and ipvs rules if switching to nftables, or if cleanupAndExit
	if !isIPTablesBased(mode) || cleanupAndExit {
		ipts, _ := getIPTables(v1.IPFamilyUnknown)
		execer := exec.New()
		ipsetInterface := utilipset.New(execer)
		ipvsInterface := utilipvs.New()

		for _, ipt := range ipts {
			encounteredError = iptables.CleanupLeftovers(ipt) || encounteredError
			encounteredError = ipvs.CleanupLeftovers(ipvsInterface, ipt, ipsetInterface) || encounteredError
		}
	}

	if utilfeature.DefaultFeatureGate.Enabled(features.NFTablesProxyMode) {
		// Clean up nftables rules when switching to iptables or ipvs, or if cleanupAndExit
		if isIPTablesBased(mode) || cleanupAndExit {
			encounteredError = nftables.CleanupLeftovers() || encounteredError
		}
	}

	if encounteredError {
		return errors.New("encountered an error while tearing down rules")
	}
	return nil
}
