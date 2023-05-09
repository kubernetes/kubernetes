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
	"k8s.io/client-go/tools/events"
	"k8s.io/component-base/configz"
	"k8s.io/component-base/metrics"
	nodeutil "k8s.io/component-helpers/node/util"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/pkg/proxy"
	proxyconfigapi "k8s.io/kubernetes/pkg/proxy/apis/config"
	proxyconfigscheme "k8s.io/kubernetes/pkg/proxy/apis/config/scheme"
	"k8s.io/kubernetes/pkg/proxy/healthcheck"
	"k8s.io/kubernetes/pkg/proxy/winkernel"
)

func (o *Options) platformApplyDefaults(config *proxyconfigapi.KubeProxyConfiguration) {
	if config.Mode == "" {
		config.Mode = proxyconfigapi.ProxyModeKernelspace
	}
}

// NewProxyServer returns a new ProxyServer.
func NewProxyServer(o *Options) (*ProxyServer, error) {
	return newProxyServer(o.config, o.master)
}

func newProxyServer(config *proxyconfigapi.KubeProxyConfiguration, master string) (*ProxyServer, error) {
	if c, err := configz.New(proxyconfigapi.GroupName); err == nil {
		c.Set(config)
	} else {
		return nil, fmt.Errorf("unable to register configz: %s", err)
	}

	if len(config.ShowHiddenMetricsForVersion) > 0 {
		metrics.SetShowHidden()
	}

	client, err := createClient(config.ClientConnection, master)
	if err != nil {
		return nil, err
	}

	// Create event recorder
	hostname, err := nodeutil.GetHostname(config.HostnameOverride)
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

	// Check if Kernel Space can be used.
	canUseWinKernelProxy, err := winkernel.CanUseWinKernelProxier(winkernel.WindowsKernelCompatTester{})
	if !canUseWinKernelProxy && err != nil {
		return nil, err
	}

	var proxier proxy.Provider
	dualStackMode := getDualStackMode(config.Winkernel.NetworkName, winkernel.DualStackCompatTester{})
	if dualStackMode {
		klog.InfoS("Creating dualStackProxier for Windows kernel.")

		proxier, err = winkernel.NewDualStackProxier(
			config.IPTables.SyncPeriod.Duration,
			config.IPTables.MinSyncPeriod.Duration,
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

	return &ProxyServer{
		Config:        config,
		Client:        client,
		Proxier:       proxier,
		Broadcaster:   eventBroadcaster,
		Recorder:      recorder,
		NodeRef:       nodeRef,
		HealthzServer: healthzServer,
	}, nil
}

func getDualStackMode(networkname string, compatTester winkernel.StackCompatTester) bool {
	return compatTester.DualStackCompatible(networkname)
}

func detectNumCPU() int {
	return goruntime.NumCPU()
}

// cleanupAndExit cleans up after a previous proxy run
func cleanupAndExit() error {
	return errors.New("--cleanup-and-exit is not implemented on Windows")
}
