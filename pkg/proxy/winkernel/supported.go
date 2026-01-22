//go:build windows
// +build windows

/*
Copyright 2017 The Kubernetes Authors.

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

package winkernel

import (
	"fmt"

	"github.com/Microsoft/hnslib"

	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/klog/v2"
	kubefeatures "k8s.io/kubernetes/pkg/features"
)

// KernelCompatTester tests whether the required kernel capabilities are
// present to run the windows kernel proxier.
type KernelCompatTester interface {
	IsCompatible() error
}

// CanUseWinKernelProxier returns true if we should use the Kernel Proxier
// instead of the "classic" userspace Proxier.  This is determined by checking
// the windows kernel version and for the existence of kernel features.
func CanUseWinKernelProxier(kcompat KernelCompatTester) (bool, error) {
	// Check that the kernel supports what we need.
	if err := kcompat.IsCompatible(); err != nil {
		return false, err
	}
	return true, nil
}

type WindowsKernelCompatTester struct{}

// IsCompatible returns true if winkernel can support this mode of proxy
func (lkct WindowsKernelCompatTester) IsCompatible() error {
	_, err := hnslib.HNSListPolicyListRequest()
	if err != nil {
		return fmt.Errorf("Windows kernel is not compatible for Kernel mode")
	}
	return nil
}

// StackCompatTester tests whether the required kernel and network are dualstack capable
type StackCompatTester interface {
	DualStackCompatible(networkName string) bool
}

type DualStackCompatTester struct{}

func (t DualStackCompatTester) DualStackCompatible(networkName string) bool {
	hcnImpl := newHcnImpl()
	// First tag of hnslib that has a proper check for dual stack support is v0.8.22 due to a bug.
	if err := hcnImpl.Ipv6DualStackSupported(); err != nil {
		// Hcn *can* fail the query to grab the version of hcn itself (which this call will do internally before parsing
		// to see if dual stack is supported), but the only time this can happen, at least that can be discerned, is if the host
		// is pre-1803 and hcn didn't exist. hnslib should truthfully return a known error if this happened that we can
		// check against, and the case where 'err != this known error' would be the 'this feature isn't supported' case, as is being
		// used here. For now, seeming as how nothing before ws2019 (1809) is listed as supported for k8s we can pretty much assume
		// any error here isn't because the query failed, it's just that dualstack simply isn't supported on the host. With all
		// that in mind, just log as info and not error to let the user know we're falling back.
		klog.InfoS("This version of Windows does not support dual-stack, falling back to single-stack", "err", err.Error())
		return false
	}

	// check if network is using overlay
	hns, _ := newHostNetworkService(hcnImpl)
	networkName, err := getNetworkName(networkName)
	if err != nil {
		klog.ErrorS(err, "Unable to determine dual-stack status, falling back to single-stack")
		return false
	}
	networkInfo, err := getNetworkInfo(hns, networkName)
	if err != nil {
		klog.ErrorS(err, "Unable to determine dual-stack status, falling back to single-stack")
		return false
	}

	if utilfeature.DefaultFeatureGate.Enabled(kubefeatures.WinOverlay) && isOverlay(networkInfo) {
		// Overlay (VXLAN) networks on Windows do not support dual-stack networking today
		klog.InfoS("Winoverlay does not support dual-stack, falling back to single-stack")
		return false
	}

	return true
}
