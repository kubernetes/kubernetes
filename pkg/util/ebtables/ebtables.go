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
	"fmt"
	"regexp"
	"strings"

	utilexec "k8s.io/utils/exec"
)

const (
	cmdebtables = "ebtables"

	// Flag to show full mac in output. The default representation omits leading zeroes.
	fullMac = "--Lmac2"
)

type RulePosition string

const (
	Prepend RulePosition = "-I"
	Append  RulePosition = "-A"
)

type Table string

const (
	TableNAT    Table = "nat"
	TableFilter Table = "filter"
)

type Chain string

const (
	ChainPostrouting Chain = "POSTROUTING"
	ChainPrerouting  Chain = "PREROUTING"
	ChainOutput      Chain = "OUTPUT"
	ChainInput       Chain = "INPUT"
)

type operation string

const (
	opCreateChain operation = "-N"
	opFlushChain  operation = "-F"
	opDeleteChain operation = "-X"
	opListChain   operation = "-L"
	opAppendRule  operation = "-A"
	opPrependRule operation = "-I"
	opDeleteRule  operation = "-D"
)

// An injectable interface for running ebtables commands.  Implementations must be goroutine-safe.
type Interface interface {
	// GetVersion returns the "X.Y.Z" semver string for ebtables.
	GetVersion() (string, error)
	// EnsureRule checks if the specified rule is present and, if not, creates it.  If the rule existed, return true.
	// WARNING: ebtables does not provide check operation like iptables do. Hence we have to do a string match of args.
	// Input args must follow the format and sequence of ebtables list output. Otherwise, EnsureRule will always create
	// new rules and causing duplicates.
	EnsureRule(position RulePosition, table Table, chain Chain, args ...string) (bool, error)
	// EnsureChain checks if the specified chain is present and, if not, creates it.  If the rule existed, return true.
	EnsureChain(table Table, chain Chain) (bool, error)
	// DeleteChain deletes the specified chain.  If the chain did not exist, return error.
	DeleteChain(table Table, chain Chain) error
	// FlushChain flush the specified chain.  If the chain did not exist, return error.
	FlushChain(table Table, chain Chain) error
}

// runner implements Interface in terms of exec("ebtables").
type runner struct {
	exec utilexec.Interface
}

// New returns a new Interface which will exec ebtables.
func New(exec utilexec.Interface) Interface {
	runner := &runner{
		exec: exec,
	}
	return runner
}

func makeFullArgs(table Table, op operation, chain Chain, args ...string) []string {
	return append([]string{"-t", string(table), string(op), string(chain)}, args...)
}

// getEbtablesVersionString runs "ebtables --version" to get the version string
// in the form "X.X.X"
func getEbtablesVersionString(exec utilexec.Interface) (string, error) {
	// this doesn't access mutable state so we don't need to use the interface / runner
	bytes, err := exec.Command(cmdebtables, "--version").CombinedOutput()
	if err != nil {
		return "", err
	}
	versionMatcher := regexp.MustCompile(`v([0-9]+\.[0-9]+\.[0-9]+)`)
	match := versionMatcher.FindStringSubmatch(string(bytes))
	if match == nil {
		return "", fmt.Errorf("no ebtables version found in string: %s", bytes)
	}
	return match[1], nil
}

func (runner *runner) GetVersion() (string, error) {
	return getEbtablesVersionString(runner.exec)
}

func (runner *runner) EnsureRule(position RulePosition, table Table, chain Chain, args ...string) (bool, error) {
	exist := true
	fullArgs := makeFullArgs(table, opListChain, chain, fullMac)
	out, err := runner.exec.Command(cmdebtables, fullArgs...).CombinedOutput()
	if err != nil {
		exist = false
	} else {
		exist = checkIfRuleExists(string(out), args...)
	}
	if !exist {
		fullArgs = makeFullArgs(table, operation(position), chain, args...)
		out, err := runner.exec.Command(cmdebtables, fullArgs...).CombinedOutput()
		if err != nil {
			return exist, fmt.Errorf("Failed to ensure rule: %v, output: %v", err, string(out))
		}
	}
	return exist, nil
}

func (runner *runner) EnsureChain(table Table, chain Chain) (bool, error) {
	exist := true

	args := makeFullArgs(table, opListChain, chain)
	_, err := runner.exec.Command(cmdebtables, args...).CombinedOutput()
	if err != nil {
		exist = false
	}
	if !exist {
		args = makeFullArgs(table, opCreateChain, chain)
		out, err := runner.exec.Command(cmdebtables, args...).CombinedOutput()
		if err != nil {
			return exist, fmt.Errorf("Failed to ensure %v chain: %v, output: %v", chain, err, string(out))
		}
	}
	return exist, nil
}

// checkIfRuleExists takes the output of ebtables list chain and checks if the input rules exists
// WARNING: checkIfRuleExists expects the input args matches the format and sequence of ebtables list output
func checkIfRuleExists(listChainOutput string, args ...string) bool {
	rule := strings.Join(args, " ")
	for _, line := range strings.Split(listChainOutput, "\n") {
		if strings.TrimSpace(line) == rule {
			return true
		}
	}
	return false
}

func (runner *runner) DeleteChain(table Table, chain Chain) error {
	fullArgs := makeFullArgs(table, opDeleteChain, chain)
	out, err := runner.exec.Command(cmdebtables, fullArgs...).CombinedOutput()
	if err != nil {
		return fmt.Errorf("Failed to delete %v chain %v: %v, output: %v", string(table), string(chain), err, string(out))
	}
	return nil
}

func (runner *runner) FlushChain(table Table, chain Chain) error {
	fullArgs := makeFullArgs(table, opFlushChain, chain)
	out, err := runner.exec.Command(cmdebtables, fullArgs...).CombinedOutput()
	if err != nil {
		return fmt.Errorf("Failed to flush %v chain %v: %v, output: %v", string(table), string(chain), err, string(out))
	}
	return nil
}
