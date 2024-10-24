//go:build linux
// +build linux

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

package util

import (
	utilsysctl "k8s.io/component-helpers/node/util/sysctl"
)

const sysctlAllDisableIPv6 = "net/ipv6/conf/all/disable_ipv6"

// IsIPv6Supported determines whether the host supports ipv6 by check the global ipv6 support sysctl.
// We use it to determine whether the pod should support ipv6.
// But there are going to be edge cases we don't get right,
// e.g. the user may have IPv6 enabled in the host netns,
// but for some reason disable IPv6 in the pod netns.
// But because kubelet cannot actually know about the loopback interface of the pod,
// we cannot do something completely correct,
// so we can only do simple and basically correct things.
func IsIPv6Supported() (bool, error) {
	// TODO This should be injected by the caller.
	val, err := utilsysctl.New().GetSysctl(sysctlAllDisableIPv6)
	return val == 0, err
}
