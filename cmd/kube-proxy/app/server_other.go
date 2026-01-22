//go:build !windows && !linux

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

// Package app does all of the work necessary to configure and run a
// Kubernetes app process.
package app

import (
	"context"
	"fmt"
	"runtime"

	"k8s.io/kubernetes/pkg/proxy"
	proxyconfigapi "k8s.io/kubernetes/pkg/proxy/apis/config"
)

// platformApplyDefaults is called after parsing command-line flags and/or reading the
// config file, to apply platform-specific default values to config.
func (o *Options) platformApplyDefaults(config *proxyconfigapi.KubeProxyConfiguration) {
}

var unsupportedError = fmt.Errorf(runtime.GOOS + "/" + runtime.GOARCH + "is unsupported")

// platformSetup is called after setting up the ProxyServer, but before creating the
// Proxier. It should fill in any platform-specific fields and perform other
// platform-specific setup.
func (s *ProxyServer) platformSetup(ctx context.Context) error {
	return unsupportedError
}

// platformCheckSupported is called immediately before creating the Proxier, to check
// what IP families are supported (and whether the configuration is usable at all).
func (s *ProxyServer) platformCheckSupported(ctx context.Context) (ipv4Supported, ipv6Supported, dualStackSupported bool, err error) {
	return false, false, false, unsupportedError
}

// createProxier creates the proxy.Provider
func (s *ProxyServer) createProxier(ctx context.Context, config *proxyconfigapi.KubeProxyConfiguration, dualStackMode, initOnly bool) (proxy.Provider, error) {
	return nil, unsupportedError
}

// platformCleanup removes stale kube-proxy rules that can be safely removed.
func platformCleanup(ctx context.Context, mode proxyconfigapi.ProxyMode, cleanupAndExit bool) error {
	return unsupportedError
}
