/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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

package main

import (
	"github.com/golang/glog"
	utildbus "k8s.io/kubernetes/pkg/util/dbus"
	"k8s.io/kubernetes/pkg/util/exec"
	"k8s.io/kubernetes/pkg/util/iptables"
	"math/rand"
	"time"
)

const (
	ChainFirewallName string = "K8S-FIREWALL"
	ChainFirewall            = iptables.Chain(ChainFirewallName)
)

// An injectable interface for running iptables commands.  Implementations must be goroutine-safe.
type IptablesExecutor interface {
	EnsureConnectedFirewallChains() error
	FlushFirewallChains() error
	RestoreAll(lines []byte) error
}

// runner implements Interface in terms of exec("iptables").
type runner struct {
	iptables iptables.Interface
}

func init() {
	rand.Seed(time.Now().UTC().UnixNano())
}

func NewIpTablesExecutor() IptablesExecutor {
	return &runner{
		iptables: iptables.New(exec.New(), utildbus.New(), iptables.ProtocolIpv4),
	}
}

func (runner *runner) ensureConnectedChain(table iptables.Table, parentChain, childChain iptables.Chain) error {
	_, err := runner.iptables.EnsureChain(table, childChain)
	if err != nil {
		return err
	}
	_, err = runner.iptables.EnsureRule(iptables.Append, table, parentChain, "-j", string(childChain))
	if err != nil {
		return err
	}
	return nil
}

func (runner *runner) EnsureConnectedFirewallChains() error {
	err := runner.ensureConnectedChain(iptables.TableMANGLE, iptables.ChainPrerouting, ChainFirewall)
	if err != nil {
		return err
	}
	err = runner.ensureConnectedChain(iptables.TableFILTER, iptables.ChainInput, ChainFirewall)
	if err != nil {
		return err
	}
	return nil
}

func (runner *runner) FlushFirewallChains() error {
	err := runner.iptables.FlushChain(iptables.TableMANGLE, ChainFirewall)
	if err != nil {
		return err
	}
	err = runner.iptables.FlushChain(iptables.TableFILTER, ChainFirewall)
	if err != nil {
		return err
	}
	return nil
}

func (runner *runner) RestoreAll(lines []byte) error {
	glog.V(4).Infof("Restore All:\n:%s", string(lines))
	return runner.iptables.RestoreAll(lines, iptables.NoFlushTables, iptables.RestoreCounters)
}
