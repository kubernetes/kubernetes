//go:build linux
// +build linux

/*
Copyright 2015 The Kubernetes Authors.

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

package nftables

import (
	"fmt"
	"os"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/util/version"
	utilkernel "k8s.io/kubernetes/pkg/util/kernel"
	"sigs.k8s.io/knftables"
)

// Create a knftables.Interface and check if we can use the nftables proxy mode on this host.
func getNFTablesInterface(ipFamily v1.IPFamily) (knftables.Interface, error) {
	var nftablesFamily knftables.Family
	if ipFamily == v1.IPv4Protocol {
		nftablesFamily = knftables.IPv4Family
	} else {
		nftablesFamily = knftables.IPv6Family
	}

	// We require (or rather, knftables.New does) that the nft binary be version 1.0.1
	// or later, because versions before that would always attempt to parse the entire
	// nft ruleset at startup, even if you were only operating on a single table.
	// That's bad, because in some cases, new versions of nft have added new rule
	// types in ways that triggered bugs in older versions of nft, causing them to
	// crash. Thus, if kube-proxy used nft < 1.0.1, it could potentially get locked
	// out of its rules because of something some other component had done in a
	// completely different table.
	nft, err := knftables.New(nftablesFamily, kubeProxyTable)
	if err != nil {
		return nil, err
	}

	// Likewise, we want to ensure that the host filesystem has nft >= 1.0.1, so that
	// it's not possible that *our* rules break *the system's* nft. (In particular, we
	// know that if kube-proxy uses nft >= 1.0.3 and the system has nft <= 0.9.8, that
	// the system nft will become completely unusable.) Unfortunately, we can't easily
	// figure out the version of nft installed on the host filesystem, so instead, we
	// check the kernel version, under the assumption that the distro will have an nft
	// binary that supports the same features as its kernel does, and so kernel 5.13
	// or later implies nft 1.0.1 or later. https://issues.k8s.io/122743
	//
	// However, we allow the user to bypass this check by setting
	// `KUBE_PROXY_NFTABLES_SKIP_KERNEL_VERSION_CHECK` to anything non-empty.
	if os.Getenv("KUBE_PROXY_NFTABLES_SKIP_KERNEL_VERSION_CHECK") == "" {
		kernelVersion, err := utilkernel.GetVersion()
		if err != nil {
			return nil, fmt.Errorf("could not check kernel version: %w", err)
		}
		if kernelVersion.LessThan(version.MustParseGeneric(utilkernel.NFTablesKubeProxyKernelVersion)) {
			return nil, fmt.Errorf("kube-proxy in nftables mode requires kernel %s or later", utilkernel.NFTablesKubeProxyKernelVersion)
		}
	}

	return nft, nil
}
