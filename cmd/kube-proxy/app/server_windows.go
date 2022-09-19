//go:build windows
// +build windows

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
	"net"
	goruntime "runtime"
	"strconv"

	// Enable pprof HTTP handlers.
	_ "net/http/pprof"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/types"
	utilnet "k8s.io/apimachinery/pkg/util/net"
	"k8s.io/client-go/tools/events"
	"k8s.io/component-base/configz"
	"k8s.io/component-base/metrics"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/pkg/proxy"
	proxyconfigapi "k8s.io/kubernetes/pkg/proxy/apis/config"
	proxyconfigscheme "k8s.io/kubernetes/pkg/proxy/apis/config/scheme"
	"k8s.io/kubernetes/pkg/proxy/healthcheck"
	"k8s.io/kubernetes/pkg/proxy/winkernel"
	"k8s.io/kubernetes/pkg/proxy/winuserspace"
	utilnetsh "k8s.io/kubernetes/pkg/util/netsh"
	utilnode "k8s.io/kubernetes/pkg/util/node"
	"k8s.io/utils/exec"
	netutils "k8s.io/utils/net"
)

// NewProxyServer returns a new ProxyServer.
func NewProxyServer(o *Options) (*ProxyServer, error) {
	return newProxyServer(o.config, o.master)
}

func newProxyServer(config *proxyconfigapi.KubeProxyConfiguration, master string) (*ProxyServer, error) {
	if config == nil {
		return nil, errors.New("config is required")
	}

	if c, err := configz.New(proxyconfigapi.GroupName); err == nil {
		c.Set(config)
	} else {
		return nil, fmt.Errorf("unable to register configz: %s", err)
	}

	if len(config.ShowHiddenMetricsForVersion) > 0 {
		metrics.SetShowHidden()
	}

	client, eventClient, err := createClients(config.ClientConnection, master)
	if err != nil {
		return nil, err
	}

	// Create event recorder
	hostname, err := utilnode.GetHostname(config.HostnameOverride)
	if err != nil {
		return nil, err
	}
	nodeIP := detectNodeIP(client, hostname, config.BindAddress)
	klog.InfoS("Detected node IP", "IP", nodeIP.String())

	eventBroadcaster := events.NewBroadcaster(&events.EventSinkImpl{Interface: client.EventsV1()})
	recorder := eventBroadcaster.NewRecorder(proxyconfigscheme.Scheme, "kube-proxy")

	nodeRef := &v1.ObjectReference{
		Kind:      "Node",
		Name:      hostname,
		UID:       types.UID(hostname),
		Namespace: "",
	}

	var healthzServer healthcheck.ProxierHealthUpdater
	var healthzPort int
	if len(config.HealthzBindAddress) > 0 {
		healthzServer = healthcheck.NewProxierHealthServer(config.HealthzBindAddress, 2*config.IPTables.SyncPeriod.Duration, recorder, nodeRef)
		_, port, _ := net.SplitHostPort(config.HealthzBindAddress)
		healthzPort, _ = strconv.Atoi(port)
	}

	var proxier proxy.Provider
	proxyMode := getProxyMode(config.Mode, winkernel.WindowsKernelCompatTester{})
	dualStackMode := getDualStackMode(config.Winkernel.NetworkName, winkernel.DualStackCompatTester{})
	if proxyMode == proxyconfigapi.ProxyModeKernelspace {
		klog.V(0).InfoS("Using Kernelspace Proxier.")
		if dualStackMode {
			klog.V(0).InfoS("Creating dualStackProxier for Windows kernel.")

			proxier, err = winkernel.NewDualStackProxier(
				config.IPTables.SyncPeriod.Duration,
				config.IPTables.MinSyncPeriod.Duration,
				config.IPTables.MasqueradeAll,
				int(*config.IPTables.MasqueradeBit),
				config.ClusterCIDR,
				hostname,
				nodeIPTuple(config.BindAddress),
				recorder,
				healthzServer,
				config.Winkernel,
				healthzPort,
			)
		} else {

			proxier, err = winkernel.NewProxier(
				config.IPTables.SyncPeriod.Duration,
				config.IPTables.MinSyncPeriod.Duration,
				config.IPTables.MasqueradeAll,
				int(*config.IPTables.MasqueradeBit),
				config.ClusterCIDR,
				hostname,
				nodeIP,
				recorder,
				healthzServer,
				config.Winkernel,
				healthzPort,
			)

		}

		if err != nil {
			return nil, fmt.Errorf("unable to create proxier: %v", err)
		}

		winkernel.RegisterMetrics()
	} else {
		klog.V(0).InfoS("Using userspace Proxier.")
		klog.V(0).InfoS("The userspace proxier is now deprecated and will be removed in a future release, please use 'kernelspace' instead")
		execer := exec.New()
		var netshInterface utilnetsh.Interface
		netshInterface = utilnetsh.New(execer)

		proxier, err = winuserspace.NewProxier(
			winuserspace.NewLoadBalancerRR(),
			netutils.ParseIPSloppy(config.BindAddress),
			netshInterface,
			*utilnet.ParsePortRangeOrDie(config.PortRange),
			// TODO @pires replace below with default values, if applicable
			config.IPTables.SyncPeriod.Duration,
			config.UDPIdleTimeout.Duration,
		)
		if err != nil {
			return nil, fmt.Errorf("unable to create proxier: %v", err)
		}
	}
	useEndpointSlices := true
	if proxyMode == proxyconfigapi.ProxyModeUserspace {
		// userspace mode doesn't support endpointslice.
		useEndpointSlices = false
	}
	return &ProxyServer{
		Client:              client,
		EventClient:         eventClient,
		Proxier:             proxier,
		Broadcaster:         eventBroadcaster,
		Recorder:            recorder,
		ProxyMode:           proxyMode,
		NodeRef:             nodeRef,
		MetricsBindAddress:  config.MetricsBindAddress,
		BindAddressHardFail: config.BindAddressHardFail,
		EnableProfiling:     config.EnableProfiling,
		OOMScoreAdj:         config.OOMScoreAdj,
		ConfigSyncPeriod:    config.ConfigSyncPeriod.Duration,
		HealthzServer:       healthzServer,
		UseEndpointSlices:   useEndpointSlices,
	}, nil
}

func getDualStackMode(networkname string, compatTester winkernel.StackCompatTester) bool {
	return compatTester.DualStackCompatible(networkname)
}

func getProxyMode(proxyMode proxyconfigapi.ProxyMode, kcompat winkernel.KernelCompatTester) proxyconfigapi.ProxyMode {
	if proxyMode == proxyconfigapi.ProxyModeKernelspace {
		return tryWinKernelSpaceProxy(kcompat)
	}
	return proxyconfigapi.ProxyModeUserspace
}

func detectNumCPU() int {
	return goruntime.NumCPU()
}

func tryWinKernelSpaceProxy(kcompat winkernel.KernelCompatTester) proxyconfigapi.ProxyMode {
	// Check for Windows Kernel Version if we can support Kernel Space proxy
	// Check for Windows Version

	// guaranteed false on error, error only necessary for debugging
	useWinKernelProxy, err := winkernel.CanUseWinKernelProxier(kcompat)
	if err != nil {
		klog.ErrorS(err, "Can't determine whether to use windows kernel proxy, using userspace proxier")
		return proxyconfigapi.ProxyModeUserspace
	}
	if useWinKernelProxy {
		return proxyconfigapi.ProxyModeKernelspace
	}
	// Fallback.
	klog.V(1).InfoS("Can't use winkernel proxy, using userspace proxier")
	return proxyconfigapi.ProxyModeUserspace
}

// cleanupAndExit cleans up after a previous proxy run
func cleanupAndExit() error {
	return errors.New("--cleanup-and-exit is not implemented on Windows")
}
