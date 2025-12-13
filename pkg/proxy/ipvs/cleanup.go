//go:build linux
// +build linux

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

package ipvs

import (
	"context"
	"os/exec"

	v1 "k8s.io/api/core/v1"
	"k8s.io/klog/v2"
	utilipset "k8s.io/kubernetes/pkg/proxy/ipvs/ipset"
	utilipvs "k8s.io/kubernetes/pkg/proxy/ipvs/util"
	utiliptables "k8s.io/kubernetes/pkg/util/iptables"
)

// CleanupIptablesLeftovers removes all iptables rules and chains created by the Proxier
// It returns true if an error was encountered. Errors are logged.
func cleanupIptablesLeftovers(ctx context.Context, ipt utiliptables.Interface) (encounteredError bool) {
	logger := klog.FromContext(ctx)
	// Unlink the iptables chains created by ipvs Proxier
	for _, jc := range iptablesJumpChain {
		args := []string{
			"-m", "comment", "--comment", jc.comment,
			"-j", string(jc.to),
		}
		if err := ipt.DeleteRule(jc.table, jc.from, args...); err != nil {
			if !utiliptables.IsNotFoundError(err) {
				logger.Error(err, "Error removing iptables rules in ipvs proxier")
				encounteredError = true
			}
		}
	}

	// Flush and remove all of our chains. Flushing all chains before removing them also removes all links between chains first.
	for _, ch := range iptablesCleanupChains {
		if err := ipt.FlushChain(ch.table, ch.chain); err != nil {
			if !utiliptables.IsNotFoundError(err) {
				logger.Error(err, "Error removing iptables rules in ipvs proxier")
				encounteredError = true
			}
		}
	}

	// Remove all of our chains.
	for _, ch := range iptablesCleanupChains {
		if err := ipt.DeleteChain(ch.table, ch.chain); err != nil {
			if !utiliptables.IsNotFoundError(err) {
				logger.Error(err, "Error removing iptables rules in ipvs proxier")
				encounteredError = true
			}
		}
	}

	return encounteredError
}

// CleanupLeftovers clean up all ipvs and iptables rules created by ipvs Proxier.
func CleanupLeftovers(ctx context.Context) (encounteredError bool) {
	// libipvs.New() will log errors if the "ip_vs" kernel module (or the "modprobe"
	// binary) is not available. Logging an extra error is fine if we were actually
	// trying to run the ipvs proxier, but it's confusing to see when just doing
	// best-effort cleanup (eg, when starting the nftables proxier), so we do the same
	// check libipvs does here, and bail out without calling libipvs if it fails.
	if _, err := exec.Command("modprobe", "-va", "ip_vs").CombinedOutput(); err != nil {
		return false
	}

	ipts := utiliptables.NewBestEffort()
	ipsetInterface := utilipset.New()
	ipvsInterface := utilipvs.New()

	return cleanupLeftovers(ctx, ipvsInterface, ipts, ipsetInterface)
}

func cleanupLeftovers(ctx context.Context, ipvs utilipvs.Interface, ipts map[v1.IPFamily]utiliptables.Interface, ipset utilipset.Interface) (encounteredError bool) {
	logger := klog.FromContext(ctx)
	// Clear all ipvs rules
	if ipvs != nil {
		err := ipvs.Flush()
		if err != nil {
			logger.Error(err, "Error flushing ipvs rules")
			encounteredError = true
		}
	}
	// Delete dummy interface created by ipvs Proxier.
	nl := NewNetLinkHandle(false)
	err := nl.DeleteDummyDevice(defaultDummyDevice)
	if err != nil {
		logger.Error(err, "Error deleting dummy device created by ipvs proxier", "device", defaultDummyDevice)
		encounteredError = true
	}

	// Clear iptables created by ipvs Proxier.
	for _, ipt := range ipts {
		encounteredError = cleanupIptablesLeftovers(ctx, ipt) || encounteredError
	}

	// Destroy ip sets created by ipvs Proxier.  We should call it after cleaning up
	// iptables since we can NOT delete ip set which is still referenced by iptables.
	if _, err := ipset.GetVersion(); err == nil {
		for _, set := range ipsetInfo {
			err = ipset.DestroySet(set.name)
			if err != nil {
				if !utilipset.IsNotFoundError(err) {
					logger.Error(err, "Error removing ipset", "ipset", set.name)
					encounteredError = true
				}
			}
		}
	}

	return encounteredError
}
