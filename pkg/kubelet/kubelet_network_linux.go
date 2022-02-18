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
	"k8s.io/klog/v2"
	utiliptables "k8s.io/kubernetes/pkg/util/iptables"
	utilexec "k8s.io/utils/exec"
	utilnet "k8s.io/utils/net"
)

const (
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
	// TODO: @khenidak review when there is no IPv6 iptables exec  what should happen here (note: no error returned from this func)
	ipv6Primary := kl.nodeIPs != nil && utilnet.IsIPv6(kl.nodeIPs[0])

	var iptClients []utiliptables.Interface
	var protocols []utiliptables.Protocol

	// assume 4,6
	protocols = append(protocols, utiliptables.ProtocolIPv4)
	iptClients = append(iptClients, utiliptables.New(exec, utiliptables.ProtocolIPv4))

	protocols = append(protocols, utiliptables.ProtocolIPv6)
	iptClients = append(iptClients, utiliptables.New(exec, utiliptables.ProtocolIPv6))

	// and if they are not
	if ipv6Primary {
		protocols[0], protocols[1] = protocols[1], protocols[0]
		iptClients[0], iptClients[1] = iptClients[1], iptClients[0]
	}

	for i := range iptClients {
		iptClient := iptClients[i]
		if kl.syncNetworkUtil(iptClient) {
			klog.InfoS("Initialized protocol iptables rules.", "protocol", protocols[i])
			go iptClient.Monitor(
				utiliptables.Chain("KUBE-KUBELET-CANARY"),
				[]utiliptables.Table{utiliptables.TableMangle, utiliptables.TableNAT, utiliptables.TableFilter},
				func() { kl.syncNetworkUtil(iptClient) },
				1*time.Minute, wait.NeverStop,
			)
		} else {
			klog.InfoS("Failed to initialize protocol iptables rules; some functionality may be missing.", "protocol", protocols[i])
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
		klog.ErrorS(err, "Failed to ensure that nat chain exists KUBE-MARK-DROP chain")
		return false
	}
	if _, err := iptClient.EnsureRule(utiliptables.Append, utiliptables.TableNAT, KubeMarkDropChain, "-j", "MARK", "--or-mark", dropMark); err != nil {
		klog.ErrorS(err, "Failed to ensure marking rule for KUBE-MARK-DROP chain")
		return false
	}
	if _, err := iptClient.EnsureChain(utiliptables.TableFilter, KubeFirewallChain); err != nil {
		klog.ErrorS(err, "Failed to ensure that filter table exists KUBE-FIREWALL chain")
		return false
	}
	if _, err := iptClient.EnsureRule(utiliptables.Append, utiliptables.TableFilter, KubeFirewallChain,
		"-m", "comment", "--comment", "kubernetes firewall for dropping marked packets",
		"-m", "mark", "--mark", fmt.Sprintf("%s/%s", dropMark, dropMark),
		"-j", "DROP"); err != nil {
		klog.ErrorS(err, "Failed to ensure rule to drop packet marked by the KUBE-MARK-DROP in KUBE-FIREWALL chain")
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
			klog.ErrorS(err, "Failed to ensure rule to drop invalid localhost packets in filter table KUBE-FIREWALL chain")
			return false
		}
	}

	if _, err := iptClient.EnsureRule(utiliptables.Prepend, utiliptables.TableFilter, utiliptables.ChainOutput, "-j", string(KubeFirewallChain)); err != nil {
		klog.ErrorS(err, "Failed to ensure that filter table  from OUTPUT chain jumps to KUBE-FIREWALL chain")
		return false
	}
	if _, err := iptClient.EnsureRule(utiliptables.Prepend, utiliptables.TableFilter, utiliptables.ChainInput, "-j", string(KubeFirewallChain)); err != nil {
		klog.ErrorS(err, "Failed to ensure that filter table INPUT chain jumps to KUBE-FIREWALL chain")
		return false
	}

	// Setup KUBE-MARK-MASQ rules
	masqueradeMark := getIPTablesMark(kl.iptablesMasqueradeBit)
	if _, err := iptClient.EnsureChain(utiliptables.TableNAT, KubeMarkMasqChain); err != nil {
		klog.ErrorS(err, "Failed to ensure that nat table exists KUBE-MARK-MASQ chain")
		return false
	}
	if _, err := iptClient.EnsureChain(utiliptables.TableNAT, KubePostroutingChain); err != nil {
		klog.ErrorS(err, "Failed to ensure that nat table exists kube POSTROUTING chain")
		return false
	}
	if _, err := iptClient.EnsureRule(utiliptables.Append, utiliptables.TableNAT, KubeMarkMasqChain, "-j", "MARK", "--or-mark", masqueradeMark); err != nil {
		klog.ErrorS(err, "Failed to ensure marking rule for KUBE-MARK-MASQ chain")
		return false
	}
	if _, err := iptClient.EnsureRule(utiliptables.Prepend, utiliptables.TableNAT, utiliptables.ChainPostrouting,
		"-m", "comment", "--comment", "kubernetes postrouting rules", "-j", string(KubePostroutingChain)); err != nil {
		klog.ErrorS(err, "Failed to ensure that nat table from POSTROUTING chain jumps to KUBE-POSTROUTING chain")
		return false
	}

	// Set up KUBE-POSTROUTING to unmark and masquerade marked packets
	// NB: THIS MUST MATCH the corresponding code in the iptables and ipvs
	// modes of kube-proxy
	if _, err := iptClient.EnsureRule(utiliptables.Append, utiliptables.TableNAT, KubePostroutingChain,
		"-m", "mark", "!", "--mark", fmt.Sprintf("%s/%s", masqueradeMark, masqueradeMark),
		"-j", "RETURN"); err != nil {
		klog.ErrorS(err, "Failed to ensure filtering rule for KUBE-POSTROUTING chain")
		return false
	}
	// Clear the mark to avoid re-masquerading if the packet re-traverses the network stack.
	// We know the mark bit is currently set so we can use --xor-mark to clear it (without needing
	// to Sprintf another bitmask).
	if _, err := iptClient.EnsureRule(utiliptables.Append, utiliptables.TableNAT, KubePostroutingChain,
		"-j", "MARK", "--xor-mark", masqueradeMark); err != nil {
		klog.ErrorS(err, "Failed to ensure unmarking rule for KUBE-POSTROUTING chain")
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
		klog.ErrorS(err, "Failed to ensure SNAT rule for packets marked by KUBE-MARK-MASQ chain in nat table KUBE-POSTROUTING chain")
		return false
	}

	return true
}

// getIPTablesMark returns the fwmark given the bit
func getIPTablesMark(bit int) string {
	value := 1 << uint(bit)
	return fmt.Sprintf("%#08x", value)
}
