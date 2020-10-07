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
	utilnet "k8s.io/utils/net"
)

func (kl *Kubelet) initNetworkUtil() {
	exec := utilexec.New()

	// At this point in startup we don't know the actual node IPs, so we configure dual stack iptables
	// rules if the node _might_ be dual-stack, and single-stack based on requested nodeIPs[0] otherwise.
	maybeDualStack := utilfeature.DefaultFeatureGate.Enabled(features.IPv6DualStack)
	ipv6Primary := kl.nodeIPs != nil && utilnet.IsIPv6(kl.nodeIPs[0])

	var iptClients []utiliptables.Interface
	var protocols []utiliptables.Protocol
	if maybeDualStack || !ipv6Primary {
		protocols = append(protocols, utiliptables.ProtocolIPv4)
		iptClients = append(iptClients, utiliptables.New(exec, utiliptables.ProtocolIPv4))
	}
	if maybeDualStack || ipv6Primary {
		protocols = append(protocols, utiliptables.ProtocolIPv6)
		iptClients = append(iptClients, utiliptables.New(exec, utiliptables.ProtocolIPv6))
	}

	for i := range iptClients {
		iptClient := iptClients[i]
		if kl.syncNetworkUtil(iptClient) {
			klog.Infof("Initialized %s iptables rules.", protocols[i])
			go iptClient.Monitor(
				utiliptables.Chain("KUBE-KUBELET-CANARY"),
				[]utiliptables.Table{utiliptables.TableMangle, utiliptables.TableNAT, utiliptables.TableFilter},
				func() { kl.syncNetworkUtil(iptClient) },
				1*time.Minute, wait.NeverStop,
			)
		} else {
			klog.Warningf("Failed to initialize %s iptables rules; some functionality may be missing.", protocols[i])
		}
	}
}

// syncNetworkUtil ensures the network utility are present on host.
// Network util includes:
// 1. 	In nat table, KUBE-MARK-DROP rule to mark connections for dropping
// 	Marked connection will be drop on INPUT/OUTPUT Chain in filter table
// 2. 	In nat table, KUBE-MARK-MASQ rule to mark connections for SNAT
// 	Marked connection will get SNAT on POSTROUTING Chain in nat table
func (kl *Kubelet) syncNetworkUtil(iptClient utiliptables.Interface) bool {
	// Setup KUBE-MARK-DROP rules
	dropMark := getIPTablesMark(kl.iptablesDropBit)
	if _, err := iptClient.EnsureChain(utiliptables.TableNAT, KubeMarkDropChain); err != nil {
		klog.Errorf("Failed to ensure that %s chain %s exists: %v", utiliptables.TableNAT, KubeMarkDropChain, err)
		return false
	}
	if _, err := iptClient.EnsureRule(utiliptables.Append, utiliptables.TableNAT, KubeMarkDropChain, "-j", "MARK", "--or-mark", dropMark); err != nil {
		klog.Errorf("Failed to ensure marking rule for %v: %v", KubeMarkDropChain, err)
		return false
	}
	if _, err := iptClient.EnsureChain(utiliptables.TableFilter, KubeFirewallChain); err != nil {
		klog.Errorf("Failed to ensure that %s chain %s exists: %v", utiliptables.TableFilter, KubeFirewallChain, err)
		return false
	}
	if _, err := iptClient.EnsureRule(utiliptables.Append, utiliptables.TableFilter, KubeFirewallChain,
		"-m", "comment", "--comment", "kubernetes firewall for dropping marked packets",
		"-m", "mark", "--mark", fmt.Sprintf("%s/%s", dropMark, dropMark),
		"-j", "DROP"); err != nil {
		klog.Errorf("Failed to ensure rule to drop packet marked by %v in %v chain %v: %v", KubeMarkDropChain, utiliptables.TableFilter, KubeFirewallChain, err)
		return false
	}

	// drop all non-local packets to localhost if they're not part of an existing
	// forwarded connection. See #90259
	if !iptClient.IsIPv6() { // ipv6 doesn't have this issue
		if _, err := iptClient.EnsureRule(utiliptables.Append, utiliptables.TableFilter, KubeFirewallChain,
			"-m", "comment", "--comment", "block incoming localnet connections",
			"--dst", "127.0.0.0/8",
			"!", "--src", "127.0.0.0/8",
			"-m", "conntrack",
			"!", "--ctstate", "RELATED,ESTABLISHED,DNAT",
			"-j", "DROP"); err != nil {
			klog.Errorf("Failed to ensure rule to drop invalid localhost packets in %v chain %v: %v", utiliptables.TableFilter, KubeFirewallChain, err)
			return false
		}
	}

	if _, err := iptClient.EnsureRule(utiliptables.Prepend, utiliptables.TableFilter, utiliptables.ChainOutput, "-j", string(KubeFirewallChain)); err != nil {
		klog.Errorf("Failed to ensure that %s chain %s jumps to %s: %v", utiliptables.TableFilter, utiliptables.ChainOutput, KubeFirewallChain, err)
		return false
	}
	if _, err := iptClient.EnsureRule(utiliptables.Prepend, utiliptables.TableFilter, utiliptables.ChainInput, "-j", string(KubeFirewallChain)); err != nil {
		klog.Errorf("Failed to ensure that %s chain %s jumps to %s: %v", utiliptables.TableFilter, utiliptables.ChainInput, KubeFirewallChain, err)
		return false
	}

	// Setup KUBE-MARK-MASQ rules
	masqueradeMark := getIPTablesMark(kl.iptablesMasqueradeBit)
	if _, err := iptClient.EnsureChain(utiliptables.TableNAT, KubeMarkMasqChain); err != nil {
		klog.Errorf("Failed to ensure that %s chain %s exists: %v", utiliptables.TableNAT, KubeMarkMasqChain, err)
		return false
	}
	if _, err := iptClient.EnsureChain(utiliptables.TableNAT, KubePostroutingChain); err != nil {
		klog.Errorf("Failed to ensure that %s chain %s exists: %v", utiliptables.TableNAT, KubePostroutingChain, err)
		return false
	}
	if _, err := iptClient.EnsureRule(utiliptables.Append, utiliptables.TableNAT, KubeMarkMasqChain, "-j", "MARK", "--or-mark", masqueradeMark); err != nil {
		klog.Errorf("Failed to ensure marking rule for %v: %v", KubeMarkMasqChain, err)
		return false
	}
	if _, err := iptClient.EnsureRule(utiliptables.Prepend, utiliptables.TableNAT, utiliptables.ChainPostrouting,
		"-m", "comment", "--comment", "kubernetes postrouting rules", "-j", string(KubePostroutingChain)); err != nil {
		klog.Errorf("Failed to ensure that %s chain %s jumps to %s: %v", utiliptables.TableNAT, utiliptables.ChainPostrouting, KubePostroutingChain, err)
		return false
	}

	// Set up KUBE-POSTROUTING to unmark and masquerade marked packets
	// NB: THIS MUST MATCH the corresponding code in the iptables and ipvs
	// modes of kube-proxy
	if _, err := iptClient.EnsureRule(utiliptables.Append, utiliptables.TableNAT, KubePostroutingChain,
		"-m", "mark", "!", "--mark", fmt.Sprintf("%s/%s", masqueradeMark, masqueradeMark),
		"-j", "RETURN"); err != nil {
		klog.Errorf("Failed to ensure filtering rule for %v: %v", KubePostroutingChain, err)
		return false
	}
	// Clear the mark to avoid re-masquerading if the packet re-traverses the network stack.
	// We know the mark bit is currently set so we can use --xor-mark to clear it (without needing
	// to Sprintf another bitmask).
	if _, err := iptClient.EnsureRule(utiliptables.Append, utiliptables.TableNAT, KubePostroutingChain,
		"-j", "MARK", "--xor-mark", masqueradeMark); err != nil {
		klog.Errorf("Failed to ensure unmarking rule for %v: %v", KubePostroutingChain, err)
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
		klog.Errorf("Failed to ensure SNAT rule for packets marked by %v in %v chain %v: %v", KubeMarkMasqChain, utiliptables.TableNAT, KubePostroutingChain, err)
		return false
	}

	return true
}

// getIPTablesMark returns the fwmark given the bit
func getIPTablesMark(bit int) string {
	value := 1 << uint(bit)
	return fmt.Sprintf("%#08x", value)
}
