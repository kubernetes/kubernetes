/*
Copyright 2016 The Kubernetes Authors.

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

package ebtables

import (
	"strings"
	"testing"

	"k8s.io/kubernetes/pkg/util/exec"
)

func TestEnsureChain(t *testing.T) {
	fexec := exec.NewFakeExec(t, nil)
	runner := New(fexec)

	// Does not Exist
	fexec.AddCommand("ebtables", "-t", "filter", "-L", "TEST-CHAIN").
		SetCombinedOutput(nil, &exec.FakeExitError{Status: 1})
	// Success
	fexec.AddCommand("ebtables", "-t", "filter", "-N", "TEST-CHAIN").
		SetCombinedOutput("", nil)

	exists, err := runner.EnsureChain(TableFilter, "TEST-CHAIN")
	if exists {
		t.Errorf("expected exists = false")
	}
	if err != nil {
		t.Errorf("expected err = nil")
	}
	fexec.AssertExpectedCommands()

	// Exists
	fexec.AddCommand("ebtables", "-t", "filter", "-L", "TEST-CHAIN").
		SetCombinedOutput(nil, nil)

	exists, err = runner.EnsureChain(TableFilter, "TEST-CHAIN")
	if !exists {
		t.Errorf("expected exists = true")
	}
	if err != nil {
		t.Errorf("expected err = nil")
	}
	fexec.AssertExpectedCommands()

	// Does not Exist
	fexec.AddCommand("ebtables", "-t", "filter", "-L", "TEST-CHAIN").
		SetCombinedOutput(nil, &exec.FakeExitError{Status: 1})
	// Fail to create chain
	fexec.AddCommand("ebtables", "-t", "filter", "-N", "TEST-CHAIN").
		SetCombinedOutput(nil, &exec.FakeExitError{Status: 2})

	exists, err = runner.EnsureChain(TableFilter, "TEST-CHAIN")
	if exists {
		t.Errorf("expected exists = false")
	}
	errStr := "Failed to ensure TEST-CHAIN chain: exit 2, output:"
	if err == nil || !strings.Contains(err.Error(), errStr) {
		t.Errorf("expected error: %q", errStr)
	}
	fexec.AssertExpectedCommands()
}

func TestEnsureRule(t *testing.T) {
	fexec := exec.NewFakeExec(t, nil)
	runner := New(fexec)

	// Exists
	fexec.AddCommand("ebtables", "-t", "filter", "-L", "OUTPUT", "--Lmac2").
		SetCombinedOutput(`Bridge table: filter

Bridge chain: OUTPUT, entries: 4, policy: ACCEPT
-j TEST
`, nil)

	exists, err := runner.EnsureRule(Append, TableFilter, ChainOutput, "-j", "TEST")
	if !exists {
		t.Errorf("expected exists = true")
	}
	if err != nil {
		t.Errorf("expected err = nil")
	}
	fexec.AssertExpectedCommands()

	// Does not Exist.
	fexec.AddCommand("ebtables", "-t", "filter", "-L", "OUTPUT", "--Lmac2").
		SetCombinedOutput(`Bridge table: filter

Bridge chain: TEST, entries: 0, policy: ACCEPT`, nil)

	// Fail to create
	fexec.AddCommand("ebtables", "-t", "filter", "-A", "OUTPUT", "-j", "NEXT-TEST").
		SetCombinedOutput("", &exec.FakeExitError{Status: 2})

	exists, err = runner.EnsureRule(Append, TableFilter, ChainOutput, "-j", "NEXT-TEST")
	if exists {
		t.Errorf("expected exists = false")
	}
	errStr := "Failed to ensure rule: exit 2, output: "
	if err == nil || err.Error() != errStr {
		t.Errorf("expected error: %q", errStr)
	}
	fexec.AssertExpectedCommands()
}
