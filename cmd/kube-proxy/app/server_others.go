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

	"k8s.io/apimachinery/pkg/fields"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	clientset "k8s.io/client-go/kubernetes"
	toolswatch "k8s.io/client-go/tools/watch"
	utilsysctl "k8s.io/component-helpers/node/util/sysctl"
	"k8s.io/kubernetes/pkg/proxy"
	proxyconfigapi "k8s.io/kubernetes/pkg/proxy/apis/config"
	"k8s.io/kubernetes/pkg/proxy/iptables"
	"k8s.io/kubernetes/pkg/proxy/ipvs"
	utilipset "k8s.io/kubernetes/pkg/proxy/ipvs/ipset"
	utilipvs "k8s.io/kubernetes/pkg/proxy/ipvs/util"
	proxymetrics "k8s.io/kubernetes/pkg/proxy/metrics"
	proxyutiliptables "k8s.io/kubernetes/pkg/proxy/util/iptables"
	utiliptables "k8s.io/kubernetes/pkg/util/iptables"
	"k8s.io/utils/exec"
	netutils "k8s.io/utils/net"

	"k8s.io/klog/v2"
)

// timeoutForNodePodCIDR is the time to wait for allocators to assign a PodCIDR to the
// node after it is registered.
var timeoutForNodePodCIDR = 5 * time.Minute

// platformApplyDefaults is called after parsing command-line flags and/or reading the
// config file, to apply platform-specific default values to config.
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

// platformSetup is called after setting up the ProxyServer, but before creating the
// Proxier. It should fill in any platform-specific fields and perform other
// platform-specific setup.
func (s *ProxyServer) platformSetup() error {
	if s.Config.DetectLocalMode == proxyconfigapi.LocalModeNodeCIDR {
		klog.InfoS("Watching for node, awaiting podCIDR allocation", "hostname", s.Hostname)
		node, err := waitForPodCIDR(s.Client, s.Hostname)
		if err != nil {
			return err
		}
		s.podCIDRs = node.Spec.PodCIDRs
		klog.InfoS("NodeInfo", "podCIDRs", node.Spec.PodCIDRs)
	}

	err := s.setupConntrack()
	if err != nil {
		return err
	}

	proxymetrics.RegisterMetrics()
	return nil
}

// platformCheckSupported is called immediately before creating the Proxier, to check
// what IP families are supported (and whether the configuration is usable at all).
func (s *ProxyServer) platformCheckSupported() (ipv4Supported, ipv6Supported, dualStackSupported bool, err error) {
	execer := exec.New()
	ipt := utiliptables.New(execer, utiliptables.ProtocolIPv4)
	ipv4Supported = ipt.Present()
	ipt = utiliptables.New(execer, utiliptables.ProtocolIPv6)
	ipv6Supported = ipt.Present()

	// The Linux proxies can always support dual-stack if they can support both IPv4
	// and IPv6.
	dualStackSupported = ipv4Supported && ipv6Supported

	if !ipv4Supported && !ipv6Supported {
		err = fmt.Errorf("iptables is not available on this host")
	} else if !ipv4Supported {
		klog.InfoS("No iptables support for family", "ipFamily", v1.IPv4Protocol)
	} else if !ipv6Supported {
		klog.InfoS("No iptables support for family", "ipFamily", v1.IPv6Protocol)
	}

	return
}

// createProxier creates the proxy.Provider
func (s *ProxyServer) createProxier(config *proxyconfigapi.KubeProxyConfiguration, dualStack bool) (proxy.Provider, error) {
	var proxier proxy.Provider
	var err error

	primaryProtocol := utiliptables.ProtocolIPv4
	if s.PrimaryIPFamily == v1.IPv6Protocol {
		primaryProtocol = utiliptables.ProtocolIPv6
	}
	execer := exec.New()
	iptInterface := utiliptables.New(execer, primaryProtocol)

	var ipt [2]utiliptables.Interface

	// Create iptables handlers for both families, one is already created
	// Always ordered as IPv4, IPv6
	if primaryProtocol == utiliptables.ProtocolIPv4 {
		ipt[0] = iptInterface
		ipt[1] = utiliptables.New(execer, utiliptables.ProtocolIPv6)
	} else {
		ipt[0] = utiliptables.New(execer, utiliptables.ProtocolIPv4)
		ipt[1] = iptInterface
	}

	if config.Mode == proxyconfigapi.ProxyModeIPTables {
		klog.InfoS("Using iptables Proxier")

		if dualStack {
			// Always ordered to match []ipt
			var localDetectors [2]proxyutiliptables.LocalTrafficDetector
			localDetectors, err = getDualStackLocalDetectorTuple(config.DetectLocalMode, config, ipt, s.podCIDRs)
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
				s.Hostname,
				s.NodeIPs,
				s.Recorder,
				s.HealthzServer,
				config.NodePortAddresses,
			)
		} else {
			// Create a single-stack proxier if and only if the node does not support dual-stack (i.e, no iptables support).
			var localDetector proxyutiliptables.LocalTrafficDetector
			localDetector, err = getLocalDetector(config.DetectLocalMode, config, iptInterface, s.podCIDRs)
			if err != nil {
				return nil, fmt.Errorf("unable to create proxier: %v", err)
			}

			// TODO this has side effects that should only happen when Run() is invoked.
			proxier, err = iptables.NewProxier(
				s.PrimaryIPFamily,
				iptInterface,
				utilsysctl.New(),
				execer,
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
			)
		}

		if err != nil {
			return nil, fmt.Errorf("unable to create proxier: %v", err)
		}
	} else if config.Mode == proxyconfigapi.ProxyModeIPVS {
		kernelHandler := ipvs.NewLinuxKernelHandler()
		ipsetInterface := utilipset.New(execer)
		ipvsInterface := utilipvs.New()
		if err := ipvs.CanUseIPVSProxier(ipvsInterface, ipsetInterface, config.IPVS.Scheduler); err != nil {
			return nil, fmt.Errorf("can't use the IPVS proxier: %v", err)
		}

		klog.InfoS("Using ipvs Proxier")
		if dualStack {
			// Always ordered to match []ipt
			var localDetectors [2]proxyutiliptables.LocalTrafficDetector
			localDetectors, err = getDualStackLocalDetectorTuple(config.DetectLocalMode, config, ipt, s.podCIDRs)
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
				kernelHandler,
			)
		} else {
			var localDetector proxyutiliptables.LocalTrafficDetector
			localDetector, err = getLocalDetector(config.DetectLocalMode, config, iptInterface, s.podCIDRs)
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
				kernelHandler,
			)
		}
		if err != nil {
			return nil, fmt.Errorf("unable to create proxier: %v", err)
		}
	}

	return proxier, nil
}

func (s *ProxyServer) setupConntrack() error {
	ct := &realConntracker{}

	max, err := getConntrackMax(s.Config.Conntrack)
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

	return nil
}

func getConntrackMax(config proxyconfigapi.KubeProxyConntrackConfiguration) (int, error) {
	if config.MaxPerCore != nil && *config.MaxPerCore > 0 {
		floor := 0
		if config.Min != nil {
			floor = int(*config.Min)
		}
		scaled := int(*config.MaxPerCore) * detectNumCPU()
		if scaled > floor {
			klog.V(3).InfoS("GetConntrackMax: using scaled conntrack-max-per-core")
			return scaled, nil
		}
		klog.V(3).InfoS("GetConntrackMax: using conntrack-min")
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

func getLocalDetector(mode proxyconfigapi.LocalMode, config *proxyconfigapi.KubeProxyConfiguration, ipt utiliptables.Interface, nodePodCIDRs []string) (proxyutiliptables.LocalTrafficDetector, error) {
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
		if len(nodePodCIDRs) == 0 {
			klog.InfoS("Detect-local-mode set to NodeCIDR, but no PodCIDR defined at node")
			break
		}
		return proxyutiliptables.NewDetectLocalByCIDR(nodePodCIDRs[0], ipt)
	case proxyconfigapi.LocalModeBridgeInterface:
		return proxyutiliptables.NewDetectLocalByBridgeInterface(config.DetectLocal.BridgeInterface)
	case proxyconfigapi.LocalModeInterfaceNamePrefix:
		return proxyutiliptables.NewDetectLocalByInterfaceNamePrefix(config.DetectLocal.InterfaceNamePrefix)
	}
	klog.InfoS("Defaulting to no-op detect-local", "detectLocalMode", string(mode))
	return proxyutiliptables.NewNoOpLocalDetector(), nil
}

func getDualStackLocalDetectorTuple(mode proxyconfigapi.LocalMode, config *proxyconfigapi.KubeProxyConfiguration, ipt [2]utiliptables.Interface, nodePodCIDRs []string) ([2]proxyutiliptables.LocalTrafficDetector, error) {
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
		if len(nodePodCIDRs) == 0 {
			klog.InfoS("No node info available to configure detect-local-mode NodeCIDR")
			break
		}
		// localDetectors, like ipt, need to be of the order [IPv4, IPv6], but PodCIDRs is setup so that PodCIDRs[0] == PodCIDR.
		// so have to handle the case where PodCIDR can be IPv6 and set that to localDetectors[1]
		if netutils.IsIPv6CIDRString(nodePodCIDRs[0]) {
			localDetectors[1], err = proxyutiliptables.NewDetectLocalByCIDR(nodePodCIDRs[0], ipt[1])
			if err != nil {
				return localDetectors, err
			}
			if len(nodePodCIDRs) > 1 {
				localDetectors[0], err = proxyutiliptables.NewDetectLocalByCIDR(nodePodCIDRs[1], ipt[0])
			}
		} else {
			localDetectors[0], err = proxyutiliptables.NewDetectLocalByCIDR(nodePodCIDRs[0], ipt[0])
			if err != nil {
				return localDetectors, err
			}
			if len(nodePodCIDRs) > 1 {
				localDetectors[1], err = proxyutiliptables.NewDetectLocalByCIDR(nodePodCIDRs[1], ipt[1])
			}
		}
		return localDetectors, err
	case proxyconfigapi.LocalModeBridgeInterface, proxyconfigapi.LocalModeInterfaceNamePrefix:
		localDetector, err := getLocalDetector(mode, config, ipt[0], nodePodCIDRs)
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
