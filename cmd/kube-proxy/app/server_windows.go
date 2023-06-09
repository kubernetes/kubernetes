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
	"strconv"

	// Enable pprof HTTP handlers.
	_ "net/http/pprof"

	"k8s.io/klog/v2"
	"k8s.io/kubernetes/pkg/proxy"
	proxyconfigapi "k8s.io/kubernetes/pkg/proxy/apis/config"
	"k8s.io/kubernetes/pkg/proxy/winkernel"
)

func (o *Options) platformApplyDefaults(config *proxyconfigapi.KubeProxyConfiguration) {
	if config.Mode == "" {
		config.Mode = proxyconfigapi.ProxyModeKernelspace
	}
}

// createProxier creates the proxy.Provider
func (s *ProxyServer) createProxier(config *proxyconfigapi.KubeProxyConfiguration) (proxy.Provider, error) {
	var healthzPort int
	if len(config.HealthzBindAddress) > 0 {
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
			s.Hostname,
			s.NodeIPs,
			s.Recorder,
			s.HealthzServer,
			config.Winkernel,
			healthzPort,
		)
	} else {
		proxier, err = winkernel.NewProxier(
			config.IPTables.SyncPeriod.Duration,
			config.IPTables.MinSyncPeriod.Duration,
			config.ClusterCIDR,
			s.Hostname,
			s.NodeIPs[s.PrimaryIPFamily],
			s.Recorder,
			s.HealthzServer,
			config.Winkernel,
			healthzPort,
		)
	}
	if err != nil {
		return nil, fmt.Errorf("unable to create proxier: %v", err)
	}

	return proxier, nil
}

func (s *ProxyServer) platformSetup() error {
	winkernel.RegisterMetrics()
	return nil
}

func getDualStackMode(networkname string, compatTester winkernel.StackCompatTester) bool {
	return compatTester.DualStackCompatible(networkname)
}

// cleanupAndExit cleans up after a previous proxy run
func cleanupAndExit() error {
	return errors.New("--cleanup-and-exit is not implemented on Windows")
}
