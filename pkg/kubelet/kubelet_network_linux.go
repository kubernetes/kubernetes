//go:build linux
// +build linux

/*
Copyright 2018 The Kubernetes Authors.

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

package kubelet

import (
	"fmt"
	"time"

	"k8s.io/apimachinery/pkg/util/wait"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/pkg/features"
	utiliptables "k8s.io/kubernetes/pkg/util/iptables"
	utilexec "k8s.io/utils/exec"
)

const (
	// KubeIPTablesHintChain is the chain whose existence in either iptables-legacy
	// or iptables-nft indicates which version of iptables the system is using
	KubeIPTablesHintChain utiliptables.Chain = "KUBE-IPTABLES-HINT"

	// KubeMarkMasqChain is the mark-for-masquerade chain
	// TODO: clean up this logic in kube-proxy
	KubeMarkMasqChain utiliptables.Chain = "KUBE-MARK-MASQ"

	// KubeMarkDropChain is the mark-for-drop chain
	KubeMarkDropChain utiliptables.Chain = "KUBE-MARK-DROP"

	// KubePostroutingChain is kubernetes postrouting rules
	KubePostroutingChain utiliptables.Chain = "KUBE-POSTROUTING"

	// KubeFirewallChain is kubernetes firewall rules
	KubeFirewallChain utiliptables.Chain = "KUBE-FIREWALL"
)

func (kl *Kubelet) initNetworkUtil() {
	exec := utilexec.New()
	iptClients := []utiliptables.Interface{
		utiliptables.New(exec, utiliptables.ProtocolIPv4),
		utiliptables.New(exec, utiliptables.ProtocolIPv6),
	}

	for i := range iptClients {
		iptClient := iptClients[i]
		if kl.syncIPTablesRules(iptClient) {
			klog.InfoS("Initialized iptables rules.", "protocol", iptClient.Protocol())
			go iptClient.Monitor(
				utiliptables.Chain("KUBE-KUBELET-CANARY"),
				[]utiliptables.Table{utiliptables.TableMangle, utiliptables.TableNAT, utiliptables.TableFilter},
				func() { kl.syncIPTablesRules(iptClient) },
				1*time.Minute, wait.NeverStop,
			)
		} else {
			klog.InfoS("Failed to initialize iptables rules; some functionality may be missing.", "protocol", iptClient.Protocol())
		}
	}
}

// syncIPTablesRules ensures the KUBE-IPTABLES-HINT chain exists, and the martian packet
// protection rule is installed. If the IPTablesOwnershipCleanup feature gate is disabled
// it will also synchronize additional deprecated iptables rules.
func (kl *Kubelet) syncIPTablesRules(iptClient utiliptables.Interface) bool {
	// Create hint chain so other components can see whether we are using iptables-legacy
	// or iptables-nft.
	if _, err := iptClient.EnsureChain(utiliptables.TableMangle, KubeIPTablesHintChain); err != nil {
		klog.ErrorS(err, "Failed to ensure that iptables hint chain exists")
		return false
	}

	if !iptClient.IsIPv6() { // ipv6 doesn't have this issue
		// Set up the KUBE-FIREWALL chain and martian packet protection rule.
		// (See below.)

		// NOTE: kube-proxy (in iptables mode) creates an identical copy of this
		// rule. If you want to change this rule in the future, you MUST do so in
		// a way that will interoperate correctly with skewed versions of the rule
		// created by kube-proxy.

		if _, err := iptClient.EnsureChain(utiliptables.TableFilter, KubeFirewallChain); err != nil {
			klog.ErrorS(err, "Failed to ensure that filter table KUBE-FIREWALL chain exists")
			return false
		}

		if _, err := iptClient.EnsureRule(utiliptables.Prepend, utiliptables.TableFilter, utiliptables.ChainOutput, "-j", string(KubeFirewallChain)); err != nil {
			klog.ErrorS(err, "Failed to ensure that OUTPUT chain jumps to KUBE-FIREWALL")
			return false
		}
		if _, err := iptClient.EnsureRule(utiliptables.Prepend, utiliptables.TableFilter, utiliptables.ChainInput, "-j", string(KubeFirewallChain)); err != nil {
			klog.ErrorS(err, "Failed to ensure that INPUT chain jumps to KUBE-FIREWALL")
			return false
		}

		// Kube-proxy's use of `route_localnet` to enable NodePorts on localhost
		// creates a security hole (https://issue.k8s.io/90259) which this
		// iptables rule mitigates. This rule should have been added to
		// kube-proxy, but it mistakenly ended up in kubelet instead, and we are
		// keeping it in kubelet for now in case other third-party components
		// depend on it.
		if _, err := iptClient.EnsureRule(utiliptables.Append, utiliptables.TableFilter, KubeFirewallChain,
			"-m", "comment", "--comment", "block incoming localnet connections",
			"--dst", "127.0.0.0/8",
			"!", "--src", "127.0.0.0/8",
			"-m", "conntrack",
			"!", "--ctstate", "RELATED,ESTABLISHED,DNAT",
			"-j", "DROP"); err != nil {
			klog.ErrorS(err, "Failed to ensure rule to drop invalid localhost packets in filter table KUBE-FIREWALL chain")
			return false
		}
	}

	if !utilfeature.DefaultFeatureGate.Enabled(features.IPTablesOwnershipCleanup) {
		ok := kl.syncIPTablesRulesDeprecated(iptClient)
		if !ok {
			return false
		}
	}

	return true
}

// syncIPTablesRulesDeprecated ensures deprecated iptables rules are present:
//  1. In nat table, KUBE-MARK-DROP rule to mark connections for dropping
//     Marked connection will be drop on INPUT/OUTPUT Chain in filter table
//  2. In nat table, KUBE-MARK-MASQ rule to mark connections for SNAT
//     Marked connection will get SNAT on POSTROUTING Chain in nat table
func (kl *Kubelet) syncIPTablesRulesDeprecated(iptClient utiliptables.Interface) bool {
	// Setup KUBE-MARK-DROP rules
	dropMark := getIPTablesMark(kl.iptablesDropBit)
	if _, err := iptClient.EnsureChain(utiliptables.TableNAT, KubeMarkDropChain); err != nil {
		klog.ErrorS(err, "Failed to ensure that KUBE-MARK-DROP chain exists")
		return false
	}
	if _, err := iptClient.EnsureRule(utiliptables.Append, utiliptables.TableNAT, KubeMarkDropChain, "-j", "MARK", "--or-mark", dropMark); err != nil {
		klog.ErrorS(err, "Failed to ensure that KUBE-MARK-DROP rule exists")
		return false
	}
	if _, err := iptClient.EnsureChain(utiliptables.TableFilter, KubeFirewallChain); err != nil {
		klog.ErrorS(err, "Failed to ensure that KUBE-FIREWALL chain exists")
		return false
	}
	if _, err := iptClient.EnsureRule(utiliptables.Append, utiliptables.TableFilter, KubeFirewallChain,
		"-m", "comment", "--comment", "kubernetes firewall for dropping marked packets",
		"-m", "mark", "--mark", fmt.Sprintf("%s/%s", dropMark, dropMark),
		"-j", "DROP"); err != nil {
		klog.ErrorS(err, "Failed to ensure that KUBE-FIREWALL rule exists")
		return false
	}

	// Setup KUBE-MARK-MASQ rules
	masqueradeMark := getIPTablesMark(kl.iptablesMasqueradeBit)
	if _, err := iptClient.EnsureChain(utiliptables.TableNAT, KubeMarkMasqChain); err != nil {
		klog.ErrorS(err, "Failed to ensure that KUBE-MARK-MASQ chain exists")
		return false
	}
	if _, err := iptClient.EnsureChain(utiliptables.TableNAT, KubePostroutingChain); err != nil {
		klog.ErrorS(err, "Failed to ensure that KUBE-POSTROUTING chain exists")
		return false
	}
	if _, err := iptClient.EnsureRule(utiliptables.Append, utiliptables.TableNAT, KubeMarkMasqChain, "-j", "MARK", "--or-mark", masqueradeMark); err != nil {
		klog.ErrorS(err, "Failed to ensure that KUBE-MARK-MASQ rule exists")
		return false
	}
	if _, err := iptClient.EnsureRule(utiliptables.Prepend, utiliptables.TableNAT, utiliptables.ChainPostrouting,
		"-m", "comment", "--comment", "kubernetes postrouting rules", "-j", string(KubePostroutingChain)); err != nil {
		klog.ErrorS(err, "Failed to ensure that POSTROUTING chain jumps to KUBE-POSTROUTING")
		return false
	}

	// Set up KUBE-POSTROUTING to unmark and masquerade marked packets

	// NOTE: kube-proxy (in iptables and ipvs modes) creates identical copies of these
	// rules. If you want to change these rules in the future, you MUST do so in a way
	// that will interoperate correctly with skewed versions of the rules created by
	// kube-proxy.

	if _, err := iptClient.EnsureRule(utiliptables.Append, utiliptables.TableNAT, KubePostroutingChain,
		"-m", "mark", "!", "--mark", fmt.Sprintf("%s/%s", masqueradeMark, masqueradeMark),
		"-j", "RETURN"); err != nil {
		klog.ErrorS(err, "Failed to ensure first masquerading rule exists")
		return false
	}
	// Clear the mark to avoid re-masquerading if the packet re-traverses the network stack.
	// We know the mark bit is currently set so we can use --xor-mark to clear it (without needing
	// to Sprintf another bitmask).
	if _, err := iptClient.EnsureRule(utiliptables.Append, utiliptables.TableNAT, KubePostroutingChain,
		"-j", "MARK", "--xor-mark", masqueradeMark); err != nil {
		klog.ErrorS(err, "Failed to ensure second masquerading rule exists")
		return false
	}
	masqRule := []string{
		"-m", "comment", "--comment", "kubernetes service traffic requiring SNAT",
		"-j", "MASQUERADE",
	}
	if iptClient.HasRandomFully() {
		masqRule = append(masqRule, "--random-fully")
	}
	if _, err := iptClient.EnsureRule(utiliptables.Append, utiliptables.TableNAT, KubePostroutingChain, masqRule...); err != nil {
		klog.ErrorS(err, "Failed to ensure third masquerading rule exists")
		return false
	}

	return true
}

// getIPTablesMark returns the fwmark given the bit
func getIPTablesMark(bit int) string {
	value := 1 << uint(bit)
	return fmt.Sprintf("%#08x", value)
}
