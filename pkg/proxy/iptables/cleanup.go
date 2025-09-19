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

package iptables

import (
	"bytes"
	"context"

	"k8s.io/klog/v2"
	"k8s.io/kubernetes/pkg/proxy/metrics"
	proxyutil "k8s.io/kubernetes/pkg/proxy/util"
	utiliptables "k8s.io/kubernetes/pkg/util/iptables"
)

// CleanupLeftovers removes all iptables rules and chains created by the Proxier
// It returns true if an error was encountered. Errors are logged.
func CleanupLeftovers(ctx context.Context) (encounteredError bool) {
	ipts := utiliptables.NewBestEffort()
	for _, ipt := range ipts {
		encounteredError = cleanupLeftoversForFamily(ctx, ipt) || encounteredError
	}
	return
}

func cleanupLeftoversForFamily(ctx context.Context, ipt utiliptables.Interface) (encounteredError bool) {
	logger := klog.FromContext(ctx)
	// Unlink our chains
	for _, jump := range append(iptablesJumpChains, iptablesCleanupOnlyChains...) {
		args := append([]string{}, jump.extraArgs...)
		args = append(args,
			"-m", "comment", "--comment", jump.comment,
			"-j", string(jump.dstChain),
		)
		if err := ipt.DeleteRule(jump.table, jump.srcChain, args...); err != nil {
			if !utiliptables.IsNotFoundError(err) {
				logger.Error(err, "Error removing pure-iptables proxy rule")
				encounteredError = true
			}
		}
	}

	// Flush and remove all of our "-t nat" chains.
	iptablesData := bytes.NewBuffer(nil)
	if err := ipt.SaveInto(utiliptables.TableNAT, iptablesData); err != nil {
		logger.Error(err, "Failed to execute iptables-save", "table", utiliptables.TableNAT)
		encounteredError = true
	} else {
		existingNATChains := utiliptables.GetChainsFromTable(iptablesData.Bytes())
		natChains := proxyutil.NewLineBuffer()
		natRules := proxyutil.NewLineBuffer()
		natChains.Write("*nat")
		// Start with chains we know we need to remove.
		for _, chain := range []utiliptables.Chain{kubeServicesChain, kubeNodePortsChain, kubePostroutingChain, kubeMarkMasqChain, kubeProxyCanaryChain} {
			if existingNATChains.Has(chain) {
				chainString := string(chain)
				natChains.Write(utiliptables.MakeChainLine(chain)) // flush
				natRules.Write("-X", chainString)                  // delete
			}
		}
		// Hunt for service and endpoint chains.
		for chain := range existingNATChains {
			chainString := string(chain)
			if isServiceChainName(chainString) {
				natChains.Write(utiliptables.MakeChainLine(chain)) // flush
				natRules.Write("-X", chainString)                  // delete
			}
		}
		natRules.Write("COMMIT")
		natLines := append(natChains.Bytes(), natRules.Bytes()...)
		// Write it.
		err = ipt.Restore(utiliptables.TableNAT, natLines, utiliptables.NoFlushTables, utiliptables.RestoreCounters)
		if err != nil {
			logger.Error(err, "Failed to execute iptables-restore", "table", utiliptables.TableNAT)
			metrics.IPTablesRestoreFailuresTotal.WithLabelValues(string(ipt.Protocol())).Inc()
			encounteredError = true
		}
	}

	// Flush and remove all of our "-t filter" chains.
	iptablesData.Reset()
	if err := ipt.SaveInto(utiliptables.TableFilter, iptablesData); err != nil {
		logger.Error(err, "Failed to execute iptables-save", "table", utiliptables.TableFilter)
		encounteredError = true
	} else {
		existingFilterChains := utiliptables.GetChainsFromTable(iptablesData.Bytes())
		filterChains := proxyutil.NewLineBuffer()
		filterRules := proxyutil.NewLineBuffer()
		filterChains.Write("*filter")
		for _, chain := range []utiliptables.Chain{kubeServicesChain, kubeExternalServicesChain, kubeForwardChain, kubeNodePortsChain, kubeProxyFirewallChain, kubeProxyCanaryChain} {
			if existingFilterChains.Has(chain) {
				chainString := string(chain)
				filterChains.Write(utiliptables.MakeChainLine(chain))
				filterRules.Write("-X", chainString)
			}
		}
		filterRules.Write("COMMIT")
		filterLines := append(filterChains.Bytes(), filterRules.Bytes()...)
		// Write it.
		if err := ipt.Restore(utiliptables.TableFilter, filterLines, utiliptables.NoFlushTables, utiliptables.RestoreCounters); err != nil {
			logger.Error(err, "Failed to execute iptables-restore", "table", utiliptables.TableFilter)
			metrics.IPTablesRestoreFailuresTotal.WithLabelValues(string(ipt.Protocol())).Inc()
			encounteredError = true
		}
	}

	// Remove our "-t mangle" canary chain; ignore errors since it may not exist.
	_ = ipt.DeleteChain(utiliptables.TableMangle, kubeProxyCanaryChain)

	return encounteredError
}
