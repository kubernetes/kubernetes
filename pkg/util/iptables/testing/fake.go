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
	"bytes"
	"fmt"
	"strings"

	"k8s.io/kubernetes/pkg/util/iptables"
)

const (
	Destination = "-d "
	Source      = "-s "
	DPort       = "--dport "
	Protocol    = "-p "
	Jump        = "-j "
	Reject      = "REJECT"
	ToDest      = "--to-destination "
	Recent      = "recent "
	MatchSet    = "--match-set "
	SrcType     = "--src-type "
)

type Rule map[string]string

// no-op implementation of iptables Interface
type FakeIPTables struct {
	Lines []byte
}

func NewFake() *FakeIPTables {
	return &FakeIPTables{}
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
	lines := make([]byte, len(f.Lines))
	copy(lines, f.Lines)
	return lines, nil
}

func (f *FakeIPTables) SaveInto(table iptables.Table, buffer *bytes.Buffer) error {
	buffer.Write(f.Lines)
	return nil
}

func (*FakeIPTables) Restore(table iptables.Table, data []byte, flush iptables.FlushFlag, counters iptables.RestoreCountersFlag) error {
	return nil
}

func (f *FakeIPTables) RestoreAll(data []byte, flush iptables.FlushFlag, counters iptables.RestoreCountersFlag) error {
	f.Lines = data
	return nil
}
func (*FakeIPTables) AddReloadFunc(reloadFunc func()) {}

func (*FakeIPTables) Destroy() {}

func getToken(line, separator string) string {
	tokens := strings.Split(line, separator)
	if len(tokens) == 2 {
		return strings.Split(tokens[1], " ")[0]
	}
	return ""
}

// GetChain returns a list of rules for the given chain.
// The chain name must match exactly.
// The matching is pretty dumb, don't rely on it for anything but testing.
func (f *FakeIPTables) GetRules(chainName string) (rules []Rule) {
	for _, l := range strings.Split(string(f.Lines), "\n") {
		if strings.Contains(l, fmt.Sprintf("-A %v", chainName)) {
			newRule := Rule(map[string]string{})
			for _, arg := range []string{Destination, Source, DPort, Protocol, Jump, ToDest, Recent, MatchSet, SrcType} {
				tok := getToken(l, arg)
				if tok != "" {
					newRule[arg] = tok
				}
			}
			rules = append(rules, newRule)
		}
	}
	return
}

var _ = iptables.Interface(&FakeIPTables{})
