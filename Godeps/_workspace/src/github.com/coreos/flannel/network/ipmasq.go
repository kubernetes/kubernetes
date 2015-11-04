// Copyright 2015 CoreOS, Inc.
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

package network

import (
	"fmt"
	"strings"

	"github.com/coreos/flannel/Godeps/_workspace/src/github.com/coreos/go-iptables/iptables"
	log "github.com/coreos/flannel/Godeps/_workspace/src/github.com/golang/glog"

	"github.com/coreos/flannel/pkg/ip"
)

func setupIPMasq(ipn ip.IP4Net) error {
	ipt, err := iptables.New()
	if err != nil {
		return fmt.Errorf("failed to setup IP Masquerade. iptables was not found")
	}

	err = ipt.ClearChain("nat", "FLANNEL")
	if err != nil {
		return fmt.Errorf("Failed to create/clear FLANNEL chain in NAT table: %v", err)
	}

	rules := [][]string{
		// This rule makes sure we don't NAT traffic within overlay network (e.g. coming out of docker0)
		{"FLANNEL", "-d", ipn.String(), "-j", "ACCEPT"},
		// NAT if it's not multicast traffic
		{"FLANNEL", "!", "-d", "224.0.0.0/4", "-j", "MASQUERADE"},
		// This rule will take everything coming from overlay and sent it to FLANNEL chain
		{"POSTROUTING", "-s", ipn.String(), "-j", "FLANNEL"},
	}

	for _, rule := range rules {
		log.Info("Adding iptables rule: ", strings.Join(rule, " "))
		chain := rule[0]
		args := rule[1:len(rule)]

		err = ipt.AppendUnique("nat", chain, args...)
		if err != nil {
			return fmt.Errorf("Failed to insert IP masquerade rule: %v", err)
		}
	}

	return nil
}
