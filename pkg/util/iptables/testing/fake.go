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
	"time"

	"k8s.io/kubernetes/pkg/util/iptables"
)

// FakeIPTables is no-op implementation of iptables Interface.
type FakeIPTables struct {
	hasRandomFully bool
	Lines          []byte
	protocol       iptables.Protocol
}

// NewFake returns a no-op iptables.Interface
func NewFake() *FakeIPTables {
	return &FakeIPTables{protocol: iptables.ProtocolIPv4}
}

// NewIPv6Fake returns a no-op iptables.Interface with IsIPv6() == true
func NewIPv6Fake() *FakeIPTables {
	return &FakeIPTables{protocol: iptables.ProtocolIPv6}
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

// ChainExists is part of iptables.Interface
func (*FakeIPTables) ChainExists(table iptables.Table, chain iptables.Chain) (bool, error) {
	return true, nil
}

// EnsureRule is part of iptables.Interface
func (*FakeIPTables) EnsureRule(position iptables.RulePosition, table iptables.Table, chain iptables.Chain, args ...string) (bool, error) {
	return true, nil
}

// DeleteRule is part of iptables.Interface
func (*FakeIPTables) DeleteRule(table iptables.Table, chain iptables.Chain, args ...string) error {
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

// HasRandomFully is part of iptables.Interface
func (f *FakeIPTables) HasRandomFully() bool {
	return f.hasRandomFully
}

func (f *FakeIPTables) Present() bool {
	return true
}

var _ = iptables.Interface(&FakeIPTables{})
