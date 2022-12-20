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

	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/kubernetes/pkg/util/iptables"
)

// FakeIPTables is no-op implementation of iptables Interface.
type FakeIPTables struct {
	hasRandomFully bool
	protocol       iptables.Protocol

	Dump *IPTablesDump
}

// NewFake returns a no-op iptables.Interface
func NewFake() *FakeIPTables {
	f := &FakeIPTables{
		protocol: iptables.ProtocolIPv4,
		Dump: &IPTablesDump{
			Tables: []Table{
				{
					Name: iptables.TableNAT,
					Chains: []Chain{
						{Name: iptables.ChainPrerouting},
						{Name: iptables.ChainInput},
						{Name: iptables.ChainOutput},
						{Name: iptables.ChainPostrouting},
					},
				},
				{
					Name: iptables.TableFilter,
					Chains: []Chain{
						{Name: iptables.ChainInput},
						{Name: iptables.ChainForward},
						{Name: iptables.ChainOutput},
					},
				},
				{
					Name:   iptables.TableMangle,
					Chains: []Chain{},
				},
			},
		},
	}

	return f
}

// NewIPv6Fake returns a no-op iptables.Interface with IsIPv6() == true
func NewIPv6Fake() *FakeIPTables {
	f := NewFake()
	f.protocol = iptables.ProtocolIPv6
	return f
}

// SetHasRandomFully sets f's return value for HasRandomFully()
func (f *FakeIPTables) SetHasRandomFully(can bool) *FakeIPTables {
	f.hasRandomFully = can
	return f
}

// EnsureChain is part of iptables.Interface
func (f *FakeIPTables) EnsureChain(table iptables.Table, chain iptables.Chain) (bool, error) {
	t, err := f.Dump.GetTable(table)
	if err != nil {
		return false, err
	}
	if c, _ := f.Dump.GetChain(table, chain); c != nil {
		return true, nil
	}
	t.Chains = append(t.Chains, Chain{Name: chain})
	return false, nil
}

// FlushChain is part of iptables.Interface
func (f *FakeIPTables) FlushChain(table iptables.Table, chain iptables.Chain) error {
	if c, _ := f.Dump.GetChain(table, chain); c != nil {
		c.Rules = nil
	}
	return nil
}

// DeleteChain is part of iptables.Interface
func (f *FakeIPTables) DeleteChain(table iptables.Table, chain iptables.Chain) error {
	t, err := f.Dump.GetTable(table)
	if err != nil {
		return err
	}
	for i := range t.Chains {
		if t.Chains[i].Name == chain {
			t.Chains = append(t.Chains[:i], t.Chains[i+1:]...)
			return nil
		}
	}
	return nil
}

// ChainExists is part of iptables.Interface
func (f *FakeIPTables) ChainExists(table iptables.Table, chain iptables.Chain) (bool, error) {
	if _, err := f.Dump.GetTable(table); err != nil {
		return false, err
	}
	if c, _ := f.Dump.GetChain(table, chain); c != nil {
		return true, nil
	}
	return false, nil
}

// EnsureRule is part of iptables.Interface
func (f *FakeIPTables) EnsureRule(position iptables.RulePosition, table iptables.Table, chain iptables.Chain, args ...string) (bool, error) {
	c, err := f.Dump.GetChain(table, chain)
	if err != nil {
		return false, err
	}

	rule := "-A " + string(chain) + " " + strings.Join(args, " ")
	for _, r := range c.Rules {
		if r.Raw == rule {
			return true, nil
		}
	}

	parsed, err := ParseRule(rule, false)
	if err != nil {
		return false, err
	}

	if position == iptables.Append {
		c.Rules = append(c.Rules, parsed)
	} else {
		c.Rules = append([]*Rule{parsed}, c.Rules...)
	}
	return false, nil
}

// DeleteRule is part of iptables.Interface
func (f *FakeIPTables) DeleteRule(table iptables.Table, chain iptables.Chain, args ...string) error {
	c, err := f.Dump.GetChain(table, chain)
	if err != nil {
		return err
	}

	rule := "-A " + string(chain) + " " + strings.Join(args, " ")
	for i, r := range c.Rules {
		if r.Raw == rule {
			c.Rules = append(c.Rules[:i], c.Rules[i+1:]...)
			break
		}
	}
	return nil
}

// IsIPv6 is part of iptables.Interface
func (f *FakeIPTables) IsIPv6() bool {
	return f.protocol == iptables.ProtocolIPv6
}

// Protocol is part of iptables.Interface
func (f *FakeIPTables) Protocol() iptables.Protocol {
	return f.protocol
}

func (f *FakeIPTables) saveTable(table iptables.Table, buffer *bytes.Buffer) error {
	t, err := f.Dump.GetTable(table)
	if err != nil {
		return err
	}

	fmt.Fprintf(buffer, "*%s\n", table)
	for _, c := range t.Chains {
		fmt.Fprintf(buffer, ":%s - [%d:%d]\n", c.Name, c.Packets, c.Bytes)
	}
	for _, c := range t.Chains {
		for _, r := range c.Rules {
			fmt.Fprintf(buffer, "%s\n", r.Raw)
		}
	}
	fmt.Fprintf(buffer, "COMMIT\n")
	return nil
}

// SaveInto is part of iptables.Interface
func (f *FakeIPTables) SaveInto(table iptables.Table, buffer *bytes.Buffer) error {
	if table == "" {
		// As a secret extension to the API, FakeIPTables treats table="" as
		// meaning "all tables"
		for i := range f.Dump.Tables {
			err := f.saveTable(f.Dump.Tables[i].Name, buffer)
			if err != nil {
				return err
			}
		}
		return nil
	}

	return f.saveTable(table, buffer)
}

// This is not a complete list but it's enough to pass the unit tests
var builtinTargets = sets.NewString("ACCEPT", "DROP", "RETURN", "REJECT", "DNAT", "SNAT", "MASQUERADE", "MARK")

func (f *FakeIPTables) restoreTable(newDump *IPTablesDump, newTable *Table, flush iptables.FlushFlag, counters iptables.RestoreCountersFlag) error {
	oldTable, err := f.Dump.GetTable(newTable.Name)
	if err != nil {
		return err
	}

	backupChains := make([]Chain, len(oldTable.Chains))
	copy(backupChains, oldTable.Chains)

	// Update internal state
	if flush == iptables.FlushTables {
		oldTable.Chains = make([]Chain, 0, len(newTable.Chains))
	}
	for _, newChain := range newTable.Chains {
		oldChain, _ := f.Dump.GetChain(newTable.Name, newChain.Name)
		switch {
		case oldChain == nil && newChain.Deleted:
			// no-op
		case oldChain == nil && !newChain.Deleted:
			oldTable.Chains = append(oldTable.Chains, newChain)
		case oldChain != nil && newChain.Deleted:
			_ = f.DeleteChain(newTable.Name, newChain.Name)
		case oldChain != nil && !newChain.Deleted:
			// replace old data with new
			oldChain.Rules = newChain.Rules
			if counters == iptables.RestoreCounters {
				oldChain.Packets = newChain.Packets
				oldChain.Bytes = newChain.Bytes
			}
		}
	}

	// Now check that all old/new jumps are valid
	for _, chain := range oldTable.Chains {
		for _, rule := range chain.Rules {
			if rule.Jump == nil {
				continue
			}
			if builtinTargets.Has(rule.Jump.Value) {
				continue
			}

			jumpedChain, _ := f.Dump.GetChain(oldTable.Name, iptables.Chain(rule.Jump.Value))
			if jumpedChain == nil {
				newChain, _ := newDump.GetChain(oldTable.Name, iptables.Chain(rule.Jump.Value))
				if newChain != nil {
					// rule is an old rule that jumped to a chain which
					// was deleted by newDump.
					oldTable.Chains = backupChains
					return fmt.Errorf("deleted chain %q is referenced by existing rules", newChain.Name)
				} else {
					// rule is a new rule that jumped to a chain that was
					// neither created nor pre-existing
					oldTable.Chains = backupChains
					return fmt.Errorf("rule %q jumps to a non-existent chain", rule.Raw)
				}
			}
		}
	}

	return nil
}

// Restore is part of iptables.Interface
func (f *FakeIPTables) Restore(table iptables.Table, data []byte, flush iptables.FlushFlag, counters iptables.RestoreCountersFlag) error {
	dump, err := ParseIPTablesDump(string(data))
	if err != nil {
		return err
	}

	newTable, err := dump.GetTable(table)
	if err != nil {
		return err
	}

	return f.restoreTable(dump, newTable, flush, counters)
}

// RestoreAll is part of iptables.Interface
func (f *FakeIPTables) RestoreAll(data []byte, flush iptables.FlushFlag, counters iptables.RestoreCountersFlag) error {
	dump, err := ParseIPTablesDump(string(data))
	if err != nil {
		return err
	}

	for i := range dump.Tables {
		err = f.restoreTable(dump, &dump.Tables[i], flush, counters)
		if err != nil {
			return err
		}
	}
	return nil
}

// Monitor is part of iptables.Interface
func (f *FakeIPTables) Monitor(canary iptables.Chain, tables []iptables.Table, reloadFunc func(), interval time.Duration, stopCh <-chan struct{}) {
}

// HasRandomFully is part of iptables.Interface
func (f *FakeIPTables) HasRandomFully() bool {
	return f.hasRandomFully
}

func (f *FakeIPTables) Present() bool {
	return true
}

var _ = iptables.Interface(&FakeIPTables{})
