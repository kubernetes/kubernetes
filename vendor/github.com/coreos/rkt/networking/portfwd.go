// Copyright 2015 The rkt Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package networking

import (
	"fmt"
	"net"
	"strconv"

	"github.com/coreos/go-iptables/iptables"

	commonnet "github.com/coreos/rkt/common/networking"
)

type iptablesRule struct {
	Chain string
	Rule  []string
}

// GetForwardableNet iterates through all loaded networks and returns either
// the first network that has masquerading enabled,
// or the last network in case there is no masqueraded one,
// or an error if no network was loaded.
func (n *Networking) GetForwardableNet() (*activeNet, error) {
	numberNets := len(n.nets)
	if numberNets == 0 {
		return nil, fmt.Errorf("no networks found")
	}
	for _, net := range n.nets {
		if net.IPMasq() {
			return &net, nil
		}
	}
	return &n.nets[numberNets-1], nil
}

// GetForwardableNetPodIP uses GetForwardableNet() to determine the default network and then
// returns the Pod's IP of that network.
func (n *Networking) GetForwardableNetPodIP() (net.IP, error) {
	net, err := n.GetForwardableNet()
	if err != nil {
		return nil, err
	}
	return net.runtime.IP, nil
}

// GetForwardableNetHostIP uses GetForwardableNet() to determine the default network and then
// returns the Host's IP of that network.
func (n *Networking) GetForwardableNetHostIP() (net.IP, error) {
	net, err := n.GetForwardableNet()
	if err != nil {
		return nil, err
	}
	return net.runtime.HostIP, nil
}

// setupForwarding creates the iptables chains
func (e *podEnv) setupForwarding() error {
	ipt, err := iptables.New()
	if err != nil {
		return err
	}

	// Create a separate chain for this pod. This helps with debugging
	// and makes it easier to cleanup
	chainDNAT := e.portFwdChain("DNAT")
	chainSNAT := e.portFwdChain("SNAT")

	if err = ipt.NewChain("nat", chainDNAT); err != nil {
		return err
	}

	if err = ipt.NewChain("nat", chainSNAT); err != nil {
		return err
	}

	chainRuleDNAT := e.portFwdChainRuleSpec(chainDNAT, "DNAT")
	chainRuleSNAT := e.portFwdChainRuleSpec(chainSNAT, "SNAT")

	for _, entry := range []struct {
		chain           string
		customChainRule []string
	}{
		{"POSTROUTING", chainRuleSNAT}, // traffic originating from this host from loopback
		{"PREROUTING", chainRuleDNAT},  // outside traffic hitting this host
		{"OUTPUT", chainRuleDNAT},      // traffic originating from this host on non-loopback
	} {
		exists, err := ipt.Exists("nat", entry.chain, entry.customChainRule...)
		if err != nil {
			return err
		}
		if !exists {
			err = ipt.Insert("nat", entry.chain, 1, entry.customChainRule...)
			if err != nil {
				return err
			}
		}
	}
	return nil
}

func (e *podEnv) forwardPorts(fps []commonnet.ForwardedPort, podIP net.IP) error {
	if len(fps) == 0 {
		return nil
	}
	ipt, err := iptables.New()
	if err != nil {
		return err
	}
	chainDNAT := e.portFwdChain("DNAT")
	chainSNAT := e.portFwdChain("SNAT")

	for _, fp := range fps {
		for _, r := range portRules(fp, podIP, chainDNAT, chainSNAT) {
			if err := ipt.AppendUnique("nat", r.Chain, r.Rule...); err != nil {
				return err
			}
		}
	}
	return nil
}

func (e *podEnv) unforwardPorts(fps []commonnet.ForwardedPort, podIP net.IP) error {
	if len(fps) == 0 {
		return nil
	}

	ipt, err := iptables.New()
	if err != nil {
		return err
	}
	chainDNAT := e.portFwdChain("DNAT")
	chainSNAT := e.portFwdChain("SNAT")

	for _, fp := range fps {
		for _, r := range portRules(fp, podIP, chainDNAT, chainSNAT) {
			if err := ipt.Delete("nat", r.Chain, r.Rule...); err != nil {
				return err
			}
		}
	}
	return nil
}

func portRules(fp commonnet.ForwardedPort, podIP net.IP, chainDNAT, chainSNAT string) []iptablesRule {
	socketPod := fmt.Sprintf("%v:%v", podIP, fp.PodPort.Port)
	dstPortHost := strconv.Itoa(int(fp.HostPort.HostPort))
	dstPortPod := strconv.Itoa(int(fp.PodPort.Port))
	dstIPHost := fp.HostPort.HostIP.String()

	if fp.HostPort.HostIP == nil || dstIPHost == "0.0.0.0" {
		dstIPHost = "0.0.0.0/0"
	}

	return []iptablesRule{
		{ // nat the destination
			chainDNAT,
			[]string{
				"-d", dstIPHost,
				"-p", fp.PodPort.Protocol,
				"--dport", dstPortHost,
				"-j", "DNAT",
				"--to-destination", socketPod,
			},
		},
		{ // Rewrite the source for connections to localhost on the host
			chainSNAT,
			[]string{
				"-p", fp.PodPort.Protocol,
				"-s", "127.0.0.1",
				"-d", podIP.String(),
				"--dport", dstPortPod,
				"-j", "MASQUERADE",
			},
		},
	}
}

func (e *podEnv) teardownForwarding() error {
	ipt, err := iptables.New()
	if err != nil {
		return err
	}

	chainDNAT := e.portFwdChain("DNAT")
	chainSNAT := e.portFwdChain("SNAT")

	chainRuleDNAT := e.portFwdChainRuleSpec(chainDNAT, "DNAT")
	chainRuleSNAT := e.portFwdChainRuleSpec(chainSNAT, "SNAT")

	// There's no clean way now to test if a chain exists or
	// even if a rule exists if the chain is not present.
	// So we swallow the errors for now :(
	// TODO(eyakubovich): move to using libiptc for iptable
	// manipulation

	for _, entry := range []struct {
		chain           string
		customChainRule []string
	}{
		{"POSTROUTING", chainRuleSNAT}, // traffic originating on this host
		{"PREROUTING", chainRuleDNAT},  // outside traffic hitting this host
		{"OUTPUT", chainRuleDNAT},      // traffic originating on this host
	} {
		ipt.Delete("nat", entry.chain, entry.customChainRule...)
	}

	for _, entry := range []string{chainDNAT, chainSNAT} {
		ipt.ClearChain("nat", entry)
		ipt.DeleteChain("nat", entry)
	}
	return nil
}

// portFwdChain generates the *name* of the chain for pod port forwarding.
// This name must be stable.
func (e *podEnv) portFwdChain(name string) string {
	return fmt.Sprintf("RKT-PFWD-%s-%s", name, e.podID.String()[0:8])
}

func (e *podEnv) portFwdChainRuleSpec(chain string, name string) []string {
	switch name {
	case "SNAT":
		return []string{"-s", "127.0.0.1", "!", "-d", "127.0.0.1", "-j", chain}
	case "DNAT":
		return []string{"-m", "addrtype", "--dst-type", "LOCAL", "-j", chain}
	default:
		return nil
	}
}
