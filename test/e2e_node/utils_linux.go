//go:build linux
// +build linux

/*
Copyright 2020 The Kubernetes Authors.

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

package e2enode

import (
	"fmt"

	libcontainercgroups "github.com/opencontainers/cgroups"
	"k8s.io/kubernetes/test/e2e_node/criproxy"
)

// IsCgroup2UnifiedMode returns whether we are running in cgroup v2 unified mode.
func IsCgroup2UnifiedMode() bool {
	return libcontainercgroups.IsCgroup2UnifiedMode()
}

// addCRIProxyInjector registers an injector function for the CRIProxy.
func addCRIProxyInjector(proxy *criproxy.RemoteRuntime, injector func(apiName string) error) error {
	if proxy == nil {
		return fmt.Errorf("failed to add injector because the CRI Proxy is undefined")
	}
	proxy.AddInjector(injector)
	return nil
}

// resetCRIProxyInjector resets all injector functions for the CRIProxy.
func resetCRIProxyInjector(proxy *criproxy.RemoteRuntime) error {
	if proxy == nil {
		return fmt.Errorf("failed to reset injector because the CRI Proxy is undefined")
	}
	proxy.ResetInjectors()
	return nil
}
