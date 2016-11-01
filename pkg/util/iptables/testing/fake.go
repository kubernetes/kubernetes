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

package testing

import (
	"fmt"

	"k8s.io/kubernetes/pkg/util/iptables"
)

const (
	Destination = "-d"
	Source      = "-s"
	DPort       = "--dport"
	Protocol    = "-p"
	Jump        = "-j"
	Reject      = "REJECT"
	ToDest      = "--to-destination"
)

// no-op implementation of iptables Interface
type FakeIPTables struct {
	RestoreLines []byte
	SaveLines    []byte
}

func NewFake() *FakeIPTables {
	return &FakeIPTables{
		SaveLines: make([]byte, 0),
	}
}

func (*FakeIPTables) GetVersion() (string, error) {
	return "0.0.0", nil
}

func (*FakeIPTables) EnsureChain(table iptables.Table, chain iptables.Chain) (bool, error) {
	return true, nil
}

func (*FakeIPTables) FlushChain(table iptables.Table, chain iptables.Chain) error {
	return nil
}

func (*FakeIPTables) DeleteChain(table iptables.Table, chain iptables.Chain) error {
	return nil
}

func (*FakeIPTables) EnsureRule(position iptables.RulePosition, table iptables.Table, chain iptables.Chain, args ...string) (bool, error) {
	return true, nil
}

func (*FakeIPTables) DeleteRule(table iptables.Table, chain iptables.Chain, args ...string) error {
	return nil
}

func (*FakeIPTables) IsIpv6() bool {
	return false
}

func (f *FakeIPTables) Save(table iptables.Table) ([]byte, error) {
	return f.SaveLines, nil
}

func (f *FakeIPTables) SaveAll() ([]byte, error) {
	return f.SaveLines, nil
}

func (*FakeIPTables) Restore(table iptables.Table, data []byte, flush iptables.FlushFlag, counters iptables.RestoreCountersFlag) error {
	return nil
}

func (f *FakeIPTables) RestoreAll(data []byte, flush iptables.FlushFlag, counters iptables.RestoreCountersFlag) error {
	f.RestoreLines = data
	return nil
}
func (*FakeIPTables) AddReloadFunc(reloadFunc func()) {}

func (*FakeIPTables) Destroy() {}

// GetChain returns a list of rules for the given chain.
// The chain name must match exactly.
func (f *FakeIPTables) GetRules(table iptables.Table, chainName iptables.Chain) ([]iptables.Rule, error) {
	chains, err := iptables.ParseTableAddRules(table, nil, nil, f.RestoreLines)
	if err != nil {
		return nil, err
	}
	rules, ok := chains[chainName]
	if !ok {
		return nil, fmt.Errorf("Chain %v not found", chainName)
	}
	return rules, nil
}

var _ = iptables.Interface(&FakeIPTables{})
