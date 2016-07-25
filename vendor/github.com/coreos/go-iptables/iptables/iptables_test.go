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

package iptables

import (
	"crypto/rand"
	"math/big"
	"reflect"
	"testing"
)

func randChain(t *testing.T) string {
	n, err := rand.Int(rand.Reader, big.NewInt(1000000))
	if err != nil {
		t.Fatalf("Failed to generate random chain name: %v", err)
	}

	return "TEST-" + n.String()
}

func TestChain(t *testing.T) {
	chain := randChain(t)

	ipt, err := New()
	if err != nil {
		t.Fatalf("New failed: %v", err)
	}

	// chain shouldn't exist, this will create new
	err = ipt.ClearChain("filter", chain)
	if err != nil {
		t.Fatalf("ClearChain (of missing) failed: %v", err)
	}

	// chain now exists
	err = ipt.ClearChain("filter", chain)
	if err != nil {
		t.Fatalf("ClearChain (of empty) failed: %v", err)
	}

	// put a simple rule in
	err = ipt.Append("filter", chain, "-s", "0.0.0.0/0", "-j", "ACCEPT")
	if err != nil {
		t.Fatalf("Append failed: %v", err)
	}

	// can't delete non-empty chain
	err = ipt.DeleteChain("filter", chain)
	if err == nil {
		t.Fatalf("DeleteChain of non-empty chain did not fail")
	}

	err = ipt.ClearChain("filter", chain)
	if err != nil {
		t.Fatalf("ClearChain (of non-empty) failed: %v", err)
	}

	// chain empty, should be ok
	err = ipt.DeleteChain("filter", chain)
	if err != nil {
		t.Fatalf("DeleteChain of empty chain failed: %v", err)
	}
}

func TestRules(t *testing.T) {
	chain := randChain(t)

	ipt, err := New()
	if err != nil {
		t.Fatalf("New failed: %v", err)
	}

	// chain shouldn't exist, this will create new
	err = ipt.ClearChain("filter", chain)
	if err != nil {
		t.Fatalf("ClearChain (of missing) failed: %v", err)
	}

	err = ipt.Append("filter", chain, "-s", "10.1.0.0/16", "-d", "8.8.8.8/32", "-j", "ACCEPT")
	if err != nil {
		t.Fatalf("Append failed: %v", err)
	}

	err = ipt.AppendUnique("filter", chain, "-s", "10.1.0.0/16", "-d", "8.8.8.8/32", "-j", "ACCEPT")
	if err != nil {
		t.Fatalf("AppendUnique failed: %v", err)
	}

	err = ipt.Append("filter", chain, "-s", "10.2.0.0/16", "-d", "8.8.8.8/32", "-j", "ACCEPT")
	if err != nil {
		t.Fatalf("Append failed: %v", err)
	}

	err = ipt.Insert("filter", chain, 2, "-s", "10.2.0.0/16", "-d", "9.9.9.9/32", "-j", "ACCEPT")
	if err != nil {
		t.Fatalf("Insert failed: %v", err)
	}

	err = ipt.Insert("filter", chain, 1, "-s", "10.1.0.0/16", "-d", "9.9.9.9/32", "-j", "ACCEPT")
	if err != nil {
		t.Fatalf("Insert failed: %v", err)
	}

	err = ipt.Delete("filter", chain, "-s", "10.1.0.0/16", "-d", "9.9.9.9/32", "-j", "ACCEPT")
	if err != nil {
		t.Fatalf("Delete failed: %v", err)
	}

	rules, err := ipt.List("filter", chain)
	if err != nil {
		t.Fatalf("List failed: %v", err)
	}

	expected := []string{
		"-N " + chain,
		"-A " + chain + " -s 10.1.0.0/16 -d 8.8.8.8/32 -j ACCEPT",
		"-A " + chain + " -s 10.2.0.0/16 -d 9.9.9.9/32 -j ACCEPT",
		"-A " + chain + " -s 10.2.0.0/16 -d 8.8.8.8/32 -j ACCEPT",
	}

	if !reflect.DeepEqual(rules, expected) {
		t.Fatalf("List mismatch: \ngot  %#v \nneed %#v", rules, expected)
	}
}
