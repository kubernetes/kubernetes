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
	"time"

	"k8s.io/kubernetes/pkg/util/iptables"
)

const (
	// Destination represents the destination address flag
	Destination = "-d "
	// Source represents the source address flag
	Source = "-s "
	// DPort represents the destination port flag
	DPort = "--dport "
	// Protocol represents the protocol flag
	Protocol = "-p "
	// Jump represents jump flag specifies the jump target
	Jump = "-j "
	// Reject specifies the reject target
	Reject = "REJECT"
	// ToDest represents the flag used to specify the destination address in DNAT
	ToDest = "--to-destination "
	// Recent represents the sub-command recent that allows to dynamically create list of IP address to match against
	Recent = "recent "
	// MatchSet represents the flag which match packets against the specified set
	MatchSet = "--match-set "
	// SrcType represents the --src-type flag which matches if the source address is of given type
	SrcType = "--src-type "
	// Masquerade represents the target that is used in nat table.
	Masquerade = "MASQUERADE "
)

// Rule holds a map of rules.
type Rule map[string]string

// FakeIPTables is no-op implementation of iptables Interface.
type FakeIPTables struct {
	hasRandomFully bool
	Lines          []byte
}

// NewFake returns a no-op iptables.Interface
func NewFake() *FakeIPTables {
	return &FakeIPTables{}
}

// SetHasRandomFully is part of iptables.Interface
func (f *FakeIPTables) SetHasRandomFully(can bool) *FakeIPTables {
	f.hasRandomFully = can
	return f
}

// EnsureChain is part of iptables.Interface
func (*FakeIPTables) EnsureChain(table iptables.Table, chain iptables.Chain) (bool, error) {
	return true, nil
}

// FlushChain is part of iptables.Interface
func (*FakeIPTables) FlushChain(table iptables.Table, chain iptables.Chain) error {
	return nil
}

// DeleteChain is part of iptables.Interface
func (*FakeIPTables) DeleteChain(table iptables.Table, chain iptables.Chain) error {
	return nil
}

// EnsureRule is part of iptables.Interface
func (*FakeIPTables) EnsureRule(position iptables.RulePosition, table iptables.Table, chain iptables.Chain, args ...string) (bool, error) {
	return true, nil
}

// DeleteRule is part of iptables.Interface
func (*FakeIPTables) DeleteRule(table iptables.Table, chain iptables.Chain, args ...string) error {
	return nil
}

// IsIpv6 is part of iptables.Interface
func (*FakeIPTables) IsIpv6() bool {
	return false
}

// Save is part of iptables.Interface
func (f *FakeIPTables) Save(table iptables.Table) ([]byte, error) {
	lines := make([]byte, len(f.Lines))
	copy(lines, f.Lines)
	return lines, nil
}

// SaveInto is part of iptables.Interface
func (f *FakeIPTables) SaveInto(table iptables.Table, buffer *bytes.Buffer) error {
	buffer.Write(f.Lines)
	return nil
}

// Restore is part of iptables.Interface
func (*FakeIPTables) Restore(table iptables.Table, data []byte, flush iptables.FlushFlag, counters iptables.RestoreCountersFlag) error {
	return nil
}

// RestoreAll is part of iptables.Interface
func (f *FakeIPTables) RestoreAll(data []byte, flush iptables.FlushFlag, counters iptables.RestoreCountersFlag) error {
	f.Lines = data
	return nil
}

// Monitor is part of iptables.Interface
func (f *FakeIPTables) Monitor(canary iptables.Chain, tables []iptables.Table, reloadFunc func(), interval time.Duration, stopCh <-chan struct{}) {
}

func getToken(line, separator string) string {
	tokens := strings.Split(line, separator)
	if len(tokens) == 2 {
		return strings.Split(tokens[1], " ")[0]
	}
	return ""
}

// GetRules is part of iptables.Interface
func (f *FakeIPTables) GetRules(chainName string) (rules []Rule) {
	for _, l := range strings.Split(string(f.Lines), "\n") {
		if strings.Contains(l, fmt.Sprintf("-A %v", chainName)) {
			newRule := Rule(map[string]string{})
			for _, arg := range []string{Destination, Source, DPort, Protocol, Jump, ToDest, Recent, MatchSet, SrcType, Masquerade} {
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

// HasRandomFully is part of iptables.Interface
func (f *FakeIPTables) HasRandomFully() bool {
	return f.hasRandomFully
}

var _ = iptables.Interface(&FakeIPTables{})
