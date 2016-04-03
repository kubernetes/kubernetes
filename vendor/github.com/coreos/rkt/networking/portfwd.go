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
)

func (e *podEnv) forwardPorts(fps []ForwardedPort, defIP net.IP) error {
	if len(fps) == 0 {
		return nil
	}

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
		{"POSTROUTING", chainRuleSNAT}, // traffic originating from this host
		{"PREROUTING", chainRuleDNAT},  // outside traffic hitting this host
		{"OUTPUT", chainRuleDNAT},      // traffic originating from this host
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

	for _, p := range fps {

		dst := fmt.Sprintf("%v:%v", defIP, p.PodPort)
		dstIP := fmt.Sprintf("%v", defIP)
		dport := strconv.Itoa(int(p.HostPort))

		for _, r := range []struct {
			chain string
			rule  []string
		}{
			{ // Rewrite the destination
				chainDNAT,
				[]string{
					"-p", p.Protocol,
					"--dport", dport,
					"-j", "DNAT",
					"--to-destination", dst,
				},
			},
			{ // Rewrite the source for connections to localhost on the host
				chainSNAT,
				[]string{
					"-p", p.Protocol,
					"-s", "127.0.0.1",
					"-d", dstIP,
					"--dport", dport,
					"-j", "MASQUERADE",
				},
			},
		} {
			if err := ipt.AppendUnique("nat", r.chain, r.rule...); err != nil {
				return err
			}
		}
	}
	return nil
}

func (e *podEnv) unforwardPorts() error {
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
