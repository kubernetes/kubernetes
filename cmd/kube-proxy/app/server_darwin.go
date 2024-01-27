//go:build darwin
// +build darwin

/*
Copyright 2024 The Kubernetes Authors.

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

// This is a MacOS stub

// Package app does all of the work necessary to configure and run a
// Kubernetes app process.
package app

import (
	"k8s.io/kubernetes/pkg/proxy"
	proxyconfigapi "k8s.io/kubernetes/pkg/proxy/apis/config"
)

// platformApplyDefaults is called after parsing command-line flags and/or reading the
// config file, to apply platform-specific default values to config.
// No-op for Darwin (MacOS).
func (o *Options) platformApplyDefaults(config *proxyconfigapi.KubeProxyConfiguration) {}

// platformSetup is called after setting up the ProxyServer, but before creating the
// Proxier. It should fill in any platform-specific fields and perform other
// platform-specific setup.
// No-op for Darwin (MacOS).
func (s *ProxyServer) platformSetup() error {
	return nil
}

// platformCheckSupported is called immediately before creating the Proxier, to check
// what IP families are supported (and whether the configuration is usable at all).
// No-op for Darwin (MacOS).
func (s *ProxyServer) platformCheckSupported() (ipv4Supported, ipv6Supported, dualStackSupported bool, err error) {
	return
}

// createProxier creates the proxy.Provider
// No-op for Darwin (MacOS).
func (s *ProxyServer) createProxier(config *proxyconfigapi.KubeProxyConfiguration, dualStack, initOnly bool) (proxy.Provider, error) {
	return nil, nil
}

// platformCleanup removes stale kube-proxy rules that can be safely removed. If
// cleanupAndExit is true, it will attempt to remove rules from all known kube-proxy
// modes. If it is false, it will only remove rules that are definitely not in use by the
// currently-configured mode.
// No-op for Darwin (MacOS).
func platformCleanup(mode proxyconfigapi.ProxyMode, cleanupAndExit bool) error {
	return nil
}
