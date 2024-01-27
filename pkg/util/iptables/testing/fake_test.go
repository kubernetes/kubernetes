//go:build linux
// +build linux

/*
Copyright 2022 The Kubernetes Authors.

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
	"strings"
	"testing"

	"github.com/lithammer/dedent"

	"k8s.io/kubernetes/pkg/util/iptables"
)

func TestFakeIPTables(t *testing.T) {
	fake := NewFake()
	buf := bytes.NewBuffer(nil)

	err := fake.SaveInto("", buf)
	if err != nil {
		t.Fatalf("unexpected error from SaveInto: %v", err)
	}
	expected := dedent.Dedent(strings.Trim(`
		*nat
		:PREROUTING - [0:0]
		:INPUT - [0:0]
		:OUTPUT - [0:0]
		:POSTROUTING - [0:0]
		COMMIT
		*filter
		:INPUT - [0:0]
		:FORWARD - [0:0]
		:OUTPUT - [0:0]
		COMMIT
		*mangle
		COMMIT
		`, "\n"))
	if buf.String() != expected {
		t.Fatalf("bad initial dump. expected:\n%s\n\ngot:\n%s\n", expected, buf.Bytes())
	}

	// EnsureChain
	existed, err := fake.EnsureChain(iptables.Table("blah"), iptables.Chain("KUBE-TEST"))
	if err == nil {
		t.Errorf("did not get expected error creating chain in non-existent table")
	} else if existed {
		t.Errorf("wrong return value from EnsureChain with non-existent table")
	}
	existed, err = fake.EnsureChain(iptables.TableNAT, iptables.Chain("KUBE-TEST"))
	if err != nil {
		t.Errorf("unexpected error creating chain: %v", err)
	} else if existed {
		t.Errorf("wrong return value from EnsureChain with non-existent chain")
	}
	existed, err = fake.EnsureChain(iptables.TableNAT, iptables.Chain("KUBE-TEST"))
	if err != nil {
		t.Errorf("unexpected error creating chain: %v", err)
	} else if !existed {
		t.Errorf("wrong return value from EnsureChain with existing chain")
	}

	// ChainExists
	exists, err := fake.ChainExists(iptables.TableNAT, iptables.Chain("KUBE-TEST"))
	if err != nil {
		t.Errorf("unexpected error checking chain: %v", err)
	} else if !exists {
		t.Errorf("wrong return value from ChainExists with existing chain")
	}
	exists, err = fake.ChainExists(iptables.TableNAT, iptables.Chain("KUBE-TEST-NOT"))
	if err != nil {
		t.Errorf("unexpected error checking chain: %v", err)
	} else if exists {
		t.Errorf("wrong return value from ChainExists with non-existent chain")
	}

	// EnsureRule
	existed, err = fake.EnsureRule(iptables.Append, iptables.Table("blah"), iptables.Chain("KUBE-TEST"), "-j", "ACCEPT")
	if err == nil {
		t.Errorf("did not get expected error creating rule in non-existent table")
	} else if existed {
		t.Errorf("wrong return value from EnsureRule with non-existent table")
	}
	existed, err = fake.EnsureRule(iptables.Append, iptables.TableNAT, iptables.Chain("KUBE-TEST-NOT"), "-j", "ACCEPT")
	if err == nil {
		t.Errorf("did not get expected error creating rule in non-existent chain")
	} else if existed {
		t.Errorf("wrong return value from EnsureRule with non-existent chain")
	}
	existed, err = fake.EnsureRule(iptables.Append, iptables.TableNAT, iptables.Chain("KUBE-TEST"), "-j", "ACCEPT")
	if err != nil {
		t.Errorf("unexpected error creating rule: %v", err)
	} else if existed {
		t.Errorf("wrong return value from EnsureRule with non-existent rule")
	}
	existed, err = fake.EnsureRule(iptables.Prepend, iptables.TableNAT, iptables.Chain("KUBE-TEST"), "-j", "DROP")
	if err != nil {
		t.Errorf("unexpected error creating rule: %v", err)
	} else if existed {
		t.Errorf("wrong return value from EnsureRule with non-existent rule")
	}
	existed, err = fake.EnsureRule(iptables.Append, iptables.TableNAT, iptables.Chain("KUBE-TEST"), "-j", "DROP")
	if err != nil {
		t.Errorf("unexpected error creating rule: %v", err)
	} else if !existed {
		t.Errorf("wrong return value from EnsureRule with already-existing rule")
	}

	// Sanity-check...
	buf.Reset()
	err = fake.SaveInto("", buf)
	if err != nil {
		t.Fatalf("unexpected error from SaveInto: %v", err)
	}
	expected = dedent.Dedent(strings.Trim(`
		*nat
		:PREROUTING - [0:0]
		:INPUT - [0:0]
		:OUTPUT - [0:0]
		:POSTROUTING - [0:0]
		:KUBE-TEST - [0:0]
		-A KUBE-TEST -j DROP
		-A KUBE-TEST -j ACCEPT
		COMMIT
		*filter
		:INPUT - [0:0]
		:FORWARD - [0:0]
		:OUTPUT - [0:0]
		COMMIT
		*mangle
		COMMIT
		`, "\n"))
	if buf.String() != expected {
		t.Fatalf("bad sanity-check dump. expected:\n%s\n\ngot:\n%s\n", expected, buf.Bytes())
	}

	// DeleteRule
	err = fake.DeleteRule(iptables.Table("blah"), iptables.Chain("KUBE-TEST"), "-j", "DROP")
	if err == nil {
		t.Errorf("did not get expected error deleting rule in non-existent table")
	}
	err = fake.DeleteRule(iptables.TableNAT, iptables.Chain("KUBE-TEST-NOT"), "-j", "DROP")
	if err == nil {
		t.Errorf("did not get expected error deleting rule in non-existent chain")
	}
	err = fake.DeleteRule(iptables.TableNAT, iptables.Chain("KUBE-TEST"), "-j", "DROPLET")
	if err != nil {
		t.Errorf("unexpected error deleting non-existent rule: %v", err)
	}
	err = fake.DeleteRule(iptables.TableNAT, iptables.Chain("KUBE-TEST"), "-j", "DROP")
	if err != nil {
		t.Errorf("unexpected error deleting rule: %v", err)
	}

	// Restore
	rules := dedent.Dedent(strings.Trim(`
		*nat
		:KUBE-RESTORED - [0:0]
		:KUBE-MISC-CHAIN - [0:0]
		:KUBE-MISC-TWO - [0:0]
		:KUBE-EMPTY - [0:0]
		-A KUBE-RESTORED -m comment --comment "restored chain" -j ACCEPT
		-A KUBE-MISC-CHAIN -s 1.2.3.4 -j KUBE-MISC-TWO
		-A KUBE-MISC-CHAIN -d 5.6.7.8 -j MASQUERADE
		-A KUBE-MISC-TWO -j ACCEPT
		COMMIT
		`, "\n"))
	err = fake.Restore(iptables.TableNAT, []byte(rules), iptables.NoFlushTables, iptables.NoRestoreCounters)
	if err != nil {
		t.Fatalf("unexpected error from Restore: %v", err)
	}

	// We used NoFlushTables, so this should leave KUBE-TEST unchanged
	buf.Reset()
	err = fake.SaveInto("", buf)
	if err != nil {
		t.Fatalf("unexpected error from SaveInto: %v", err)
	}
	expected = dedent.Dedent(strings.Trim(`
		*nat
		:PREROUTING - [0:0]
		:INPUT - [0:0]
		:OUTPUT - [0:0]
		:POSTROUTING - [0:0]
		:KUBE-TEST - [0:0]
		:KUBE-RESTORED - [0:0]
		:KUBE-MISC-CHAIN - [0:0]
		:KUBE-MISC-TWO - [0:0]
		:KUBE-EMPTY - [0:0]
		-A KUBE-TEST -j ACCEPT
		-A KUBE-RESTORED -m comment --comment "restored chain" -j ACCEPT
		-A KUBE-MISC-CHAIN -s 1.2.3.4 -j KUBE-MISC-TWO
		-A KUBE-MISC-CHAIN -d 5.6.7.8 -j MASQUERADE
		-A KUBE-MISC-TWO -j ACCEPT
		COMMIT
		*filter
		:INPUT - [0:0]
		:FORWARD - [0:0]
		:OUTPUT - [0:0]
		COMMIT
		*mangle
		COMMIT
		`, "\n"))
	if buf.String() != expected {
		t.Fatalf("bad post-restore dump. expected:\n%s\n\ngot:\n%s\n", expected, buf.Bytes())
	}

	// Trying to use Restore to delete a chain that another chain jumps to will fail
	rules = dedent.Dedent(strings.Trim(`
		*nat
		:KUBE-MISC-TWO - [0:0]
		-X KUBE-MISC-TWO
		COMMIT
		`, "\n"))
	err = fake.Restore(iptables.TableNAT, []byte(rules), iptables.NoFlushTables, iptables.RestoreCounters)
	if err == nil || !strings.Contains(err.Error(), "referenced by existing rules") {
		t.Fatalf("Expected 'referenced by existing rules' error from Restore, got %v", err)
	}

	// Trying to use Restore to add a jump to a non-existent chain will fail
	rules = dedent.Dedent(strings.Trim(`
		*nat
		:KUBE-MISC-TWO - [0:0]
		-A KUBE-MISC-TWO -j KUBE-MISC-THREE
		COMMIT
		`, "\n"))
	err = fake.Restore(iptables.TableNAT, []byte(rules), iptables.NoFlushTables, iptables.RestoreCounters)
	if err == nil || !strings.Contains(err.Error(), "non-existent chain") {
		t.Fatalf("Expected 'non-existent chain' error from Restore, got %v", err)
	}

	// more Restore; empty out one chain and delete another, but also update its counters
	rules = dedent.Dedent(strings.Trim(`
		*nat
		:KUBE-RESTORED - [0:0]
		:KUBE-TEST - [99:9999]
		-X KUBE-RESTORED
		COMMIT
		`, "\n"))
	err = fake.Restore(iptables.TableNAT, []byte(rules), iptables.NoFlushTables, iptables.RestoreCounters)
	if err != nil {
		t.Fatalf("unexpected error from Restore: %v", err)
	}

	buf.Reset()
	err = fake.SaveInto("", buf)
	if err != nil {
		t.Fatalf("unexpected error from SaveInto: %v", err)
	}
	expected = dedent.Dedent(strings.Trim(`
		*nat
		:PREROUTING - [0:0]
		:INPUT - [0:0]
		:OUTPUT - [0:0]
		:POSTROUTING - [0:0]
		:KUBE-TEST - [99:9999]
		:KUBE-MISC-CHAIN - [0:0]
		:KUBE-MISC-TWO - [0:0]
		:KUBE-EMPTY - [0:0]
		-A KUBE-MISC-CHAIN -s 1.2.3.4 -j KUBE-MISC-TWO
		-A KUBE-MISC-CHAIN -d 5.6.7.8 -j MASQUERADE
		-A KUBE-MISC-TWO -j ACCEPT
		COMMIT
		*filter
		:INPUT - [0:0]
		:FORWARD - [0:0]
		:OUTPUT - [0:0]
		COMMIT
		*mangle
		COMMIT
		`, "\n"))
	if buf.String() != expected {
		t.Fatalf("bad post-second-restore dump. expected:\n%s\n\ngot:\n%s\n", expected, buf.Bytes())
	}

	// RestoreAll, FlushTables
	rules = dedent.Dedent(strings.Trim(`
		*filter
		:INPUT - [0:0]
		:FORWARD - [0:0]
		:OUTPUT - [0:0]
		:KUBE-TEST - [0:0]
		-A KUBE-TEST -m comment --comment "filter table KUBE-TEST" -j ACCEPT
		COMMIT
		*nat
		:PREROUTING - [0:0]
		:INPUT - [0:0]
		:OUTPUT - [0:0]
		:POSTROUTING - [0:0]
		:KUBE-TEST - [88:8888]
		:KUBE-NEW-CHAIN - [0:0]
		-A KUBE-NEW-CHAIN -d 172.30.0.1 -j DNAT --to-destination 10.0.0.1
		-A KUBE-NEW-CHAIN -d 172.30.0.2 -j DNAT --to-destination 10.0.0.2
		-A KUBE-NEW-CHAIN -d 172.30.0.3 -j DNAT --to-destination 10.0.0.3
		COMMIT
		`, "\n"))
	err = fake.RestoreAll([]byte(rules), iptables.FlushTables, iptables.NoRestoreCounters)
	if err != nil {
		t.Fatalf("unexpected error from RestoreAll: %v", err)
	}

	buf.Reset()
	err = fake.SaveInto("", buf)
	if err != nil {
		t.Fatalf("unexpected error from SaveInto: %v", err)
	}
	expected = dedent.Dedent(strings.Trim(`
		*nat
		:PREROUTING - [0:0]
		:INPUT - [0:0]
		:OUTPUT - [0:0]
		:POSTROUTING - [0:0]
		:KUBE-TEST - [88:8888]
		:KUBE-NEW-CHAIN - [0:0]
		-A KUBE-NEW-CHAIN -d 172.30.0.1 -j DNAT --to-destination 10.0.0.1
		-A KUBE-NEW-CHAIN -d 172.30.0.2 -j DNAT --to-destination 10.0.0.2
		-A KUBE-NEW-CHAIN -d 172.30.0.3 -j DNAT --to-destination 10.0.0.3
		COMMIT
		*filter
		:INPUT - [0:0]
		:FORWARD - [0:0]
		:OUTPUT - [0:0]
		:KUBE-TEST - [0:0]
		-A KUBE-TEST -m comment --comment "filter table KUBE-TEST" -j ACCEPT
		COMMIT
		*mangle
		COMMIT
		`, "\n"))
	if buf.String() != expected {
		t.Fatalf("bad post-restore-all dump. expected:\n%s\n\ngot:\n%s\n", expected, buf.Bytes())
	}
}
