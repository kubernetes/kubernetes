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

package ip

import (
	"fmt"
	"net"

	"github.com/coreos/go-iptables/iptables"
)

// SetupIPMasq installs iptables rules to masquerade traffic
// coming from ipn and going outside of it
func SetupIPMasq(ipn *net.IPNet, chain string) error {
	ipt, err := iptables.New()
	if err != nil {
		return fmt.Errorf("failed to locate iptabes: %v", err)
	}

	if err = ipt.NewChain("nat", chain); err != nil {
		if err.(*iptables.Error).ExitStatus() != 1 {
			// TODO(eyakubovich): assumes exit status 1 implies chain exists
			return err
		}
	}

	if err = ipt.AppendUnique("nat", chain, "-d", ipn.String(), "-j", "ACCEPT"); err != nil {
		return err
	}

	if err = ipt.AppendUnique("nat", chain, "!", "-d", "224.0.0.0/4", "-j", "MASQUERADE"); err != nil {
		return err
	}

	return ipt.AppendUnique("nat", "POSTROUTING", "-s", ipn.String(), "-j", chain)
}

// TeardownIPMasq undoes the effects of SetupIPMasq
func TeardownIPMasq(ipn *net.IPNet, chain string) error {
	ipt, err := iptables.New()
	if err != nil {
		return fmt.Errorf("failed to locate iptabes: %v", err)
	}

	if err = ipt.Delete("nat", "POSTROUTING", "-s", ipn.String(), "-j", chain); err != nil {
		return err
	}

	if err = ipt.ClearChain("nat", chain); err != nil {
		return err
	}

	return ipt.DeleteChain("nat", chain)
}
