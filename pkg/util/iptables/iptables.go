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

package iptables

import (
	"fmt"
	"regexp"
	"strconv"
	"strings"
	"sync"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
	utilexec "github.com/GoogleCloudPlatform/kubernetes/pkg/util/exec"
	"github.com/golang/glog"
)

type RulePosition string

const (
	Prepend RulePosition = "-I"
	Append  RulePosition = "-A"
)

// An injectable interface for running iptables commands.  Implementations must be goroutine-safe.
type Interface interface {
	// EnsureChain checks if the specified chain exists and, if not, creates it.  If the chain existed, return true.
	EnsureChain(table Table, chain Chain) (bool, error)
	// FlushChain clears the specified chain.  If the chain did not exist, return error.
	FlushChain(table Table, chain Chain) error
	// DeleteChain deletes the specified chain.  If the chain did not exist, return error.
	DeleteChain(table Table, chain Chain) error
	// EnsureRule checks if the specified rule is present and, if not, creates it.  If the rule existed, return true.
	EnsureRule(position RulePosition, table Table, chain Chain, args ...string) (bool, error)
	// DeleteRule checks if the specified rule is present and, if so, deletes it.
	DeleteRule(table Table, chain Chain, args ...string) error
	// IsIpv6 returns true if this is managing ipv6 tables
	IsIpv6() bool
}

type Protocol byte

const (
	ProtocolIpv4 Protocol = iota + 1
	ProtocolIpv6
)

type Table string

const (
	TableNAT Table = "nat"
)

type Chain string

const (
	ChainPostrouting Chain = "POSTROUTING"
	ChainPrerouting  Chain = "PREROUTING"
	ChainOutput      Chain = "OUTPUT"
)

// runner implements Interface in terms of exec("iptables").
type runner struct {
	mu       sync.Mutex
	exec     utilexec.Interface
	protocol Protocol
}

// New returns a new Interface which will exec iptables.
func New(exec utilexec.Interface, protocol Protocol) Interface {
	return &runner{exec: exec, protocol: protocol}
}

// EnsureChain is part of Interface.
func (runner *runner) EnsureChain(table Table, chain Chain) (bool, error) {
	fullArgs := makeFullArgs(table, chain)

	runner.mu.Lock()
	defer runner.mu.Unlock()

	out, err := runner.run(opCreateChain, fullArgs)
	if err != nil {
		if ee, ok := err.(utilexec.ExitError); ok {
			if ee.Exited() && ee.ExitStatus() == 1 {
				return true, nil
			}
		}
		return false, fmt.Errorf("error creating chain %q: %v: %s", chain, err, out)
	}
	return false, nil
}

// FlushChain is part of Interface.
func (runner *runner) FlushChain(table Table, chain Chain) error {
	fullArgs := makeFullArgs(table, chain)

	runner.mu.Lock()
	defer runner.mu.Unlock()

	out, err := runner.run(opFlushChain, fullArgs)
	if err != nil {
		return fmt.Errorf("error flushing chain %q: %v: %s", chain, err, out)
	}
	return nil
}

// DeleteChain is part of Interface.
func (runner *runner) DeleteChain(table Table, chain Chain) error {
	fullArgs := makeFullArgs(table, chain)

	runner.mu.Lock()
	defer runner.mu.Unlock()

	// TODO: we could call iptable -S first, ignore the output and check for non-zero return (more like DeleteRule)
	out, err := runner.run(opDeleteChain, fullArgs)
	if err != nil {
		return fmt.Errorf("error deleting chain %q: %v: %s", chain, err, out)
	}
	return nil
}

// EnsureRule is part of Interface.
func (runner *runner) EnsureRule(position RulePosition, table Table, chain Chain, args ...string) (bool, error) {
	fullArgs := makeFullArgs(table, chain, args...)

	runner.mu.Lock()
	defer runner.mu.Unlock()

	exists, err := runner.checkRule(table, chain, args...)
	if err != nil {
		return false, err
	}
	if exists {
		return true, nil
	}
	out, err := runner.run(operation(position), fullArgs)
	if err != nil {
		return false, fmt.Errorf("error appending rule: %v: %s", err, out)
	}
	return false, nil
}

// DeleteRule is part of Interface.
func (runner *runner) DeleteRule(table Table, chain Chain, args ...string) error {
	fullArgs := makeFullArgs(table, chain, args...)

	runner.mu.Lock()
	defer runner.mu.Unlock()

	exists, err := runner.checkRule(table, chain, args...)
	if err != nil {
		return err
	}
	if !exists {
		return nil
	}
	out, err := runner.run(opDeleteRule, fullArgs)
	if err != nil {
		return fmt.Errorf("error deleting rule: %v: %s", err, out)
	}
	return nil
}

func (runner *runner) IsIpv6() bool {
	return runner.protocol == ProtocolIpv6
}

func (runner *runner) iptablesCommand() string {
	if runner.IsIpv6() {
		return "ip6tables"
	} else {
		return "iptables"
	}
}

func (runner *runner) run(op operation, args []string) ([]byte, error) {
	iptablesCmd := runner.iptablesCommand()

	fullArgs := append([]string{string(op)}, args...)
	glog.V(4).Infof("running iptables %s %v", string(op), args)
	return runner.exec.Command(iptablesCmd, fullArgs...).CombinedOutput()
	// Don't log err here - callers might not think it is an error.
}

// Returns (bool, nil) if it was able to check the existence of the rule, or
// (<undefined>, error) if the process of checking failed.
func (runner *runner) checkRule(table Table, chain Chain, args ...string) (bool, error) {
	checkPresent, err := getIptablesHasCheckCommand(runner.exec)
	if err != nil {
		glog.Warning("Error checking iptables version, assuming version at least 1.4.11: %v", err)
		checkPresent = true
	}
	if checkPresent {
		return runner.checkRuleUsingCheck(makeFullArgs(table, chain, args...))
	} else {
		return runner.checkRuleWithoutCheck(table, chain, args...)
	}
}

// Executes the rule check without using the "-C" flag, instead parsing iptables-save.
// Present for compatibility with <1.4.11 versions of iptables.  This is full
// of hack and half-measures.  We should nix this ASAP.
func (runner *runner) checkRuleWithoutCheck(table Table, chain Chain, args ...string) (bool, error) {
	glog.V(1).Infof("running iptables-save -t %s", string(table))
	out, err := runner.exec.Command("iptables-save", "-t", string(table)).CombinedOutput()
	if err != nil {
		return false, fmt.Errorf("error checking rule: %v", err)
	}

	// Sadly, iptables has inconsistent quoting rules for comments.
	// Just unquote any arg that is wrapped in quotes.
	argsCopy := make([]string, len(args))
	copy(argsCopy, args)
	for i := range argsCopy {
		unquote(&argsCopy[i])
	}
	argset := util.NewStringSet(argsCopy...)

	for _, line := range strings.Split(string(out), "\n") {
		var fields = strings.Fields(line)

		// Check that this is a rule for the correct chain, and that it has
		// the correct number of argument (+2 for "-A <chain name>")
		if !strings.HasPrefix(line, fmt.Sprintf("-A %s", string(chain))) || len(fields) != len(args)+2 {
			continue
		}

		// Sadly, iptables has inconsistent quoting rules for comments.
		// Just unquote any arg that is wrapped in quotes.
		for i := range fields {
			unquote(&fields[i])
		}

		// TODO: This misses reorderings e.g. "-x foo ! -y bar" will match "! -x foo -y bar"
		if util.NewStringSet(fields...).IsSuperset(argset) {
			return true, nil
		}
		glog.V(5).Infof("DBG: fields is not a superset of args: fields=%v  args=%v", fields, args)
	}

	return false, nil
}

func unquote(strp *string) {
	if len(*strp) >= 2 && (*strp)[0] == '"' && (*strp)[len(*strp)-1] == '"' {
		*strp = strings.TrimPrefix(strings.TrimSuffix(*strp, `"`), `"`)
	}
}

// Executes the rule check using the "-C" flag
func (runner *runner) checkRuleUsingCheck(args []string) (bool, error) {
	out, err := runner.run(opCheckRule, args)
	if err == nil {
		return true, nil
	}
	if ee, ok := err.(utilexec.ExitError); ok {
		// iptables uses exit(1) to indicate a failure of the operation,
		// as compared to a malformed commandline, for example.
		if ee.Exited() && ee.ExitStatus() == 1 {
			return false, nil
		}
	}
	return false, fmt.Errorf("error checking rule: %v: %s", err, out)
}

type operation string

const (
	opCreateChain operation = "-N"
	opFlushChain  operation = "-F"
	opDeleteChain operation = "-X"
	opAppendRule  operation = "-A"
	opCheckRule   operation = "-C"
	opDeleteRule  operation = "-D"
)

func makeFullArgs(table Table, chain Chain, args ...string) []string {
	return append([]string{string(chain), "-t", string(table)}, args...)
}

// Checks if iptables has the "-C" flag
func getIptablesHasCheckCommand(exec utilexec.Interface) (bool, error) {
	vstring, err := getIptablesVersionString(exec)
	if err != nil {
		return false, err
	}

	v1, v2, v3, err := extractIptablesVersion(vstring)
	if err != nil {
		return false, err
	}

	return iptablesHasCheckCommand(v1, v2, v3), nil
}

// getIptablesVersion returns the first three components of the iptables version.
// e.g. "iptables v1.3.66" would return (1, 3, 66, nil)
func extractIptablesVersion(str string) (int, int, int, error) {
	versionMatcher := regexp.MustCompile("v([0-9]+)\\.([0-9]+)\\.([0-9]+)")
	result := versionMatcher.FindStringSubmatch(str)
	if result == nil {
		return 0, 0, 0, fmt.Errorf("no iptables version found in string: %s", str)
	}

	v1, err := strconv.Atoi(result[1])
	if err != nil {
		return 0, 0, 0, err
	}

	v2, err := strconv.Atoi(result[2])
	if err != nil {
		return 0, 0, 0, err
	}

	v3, err := strconv.Atoi(result[3])
	if err != nil {
		return 0, 0, 0, err
	}

	return v1, v2, v3, nil
}

// Runs "iptables --version" to get the version string
func getIptablesVersionString(exec utilexec.Interface) (string, error) {
	bytes, err := exec.Command("iptables", "--version").CombinedOutput()
	if err != nil {
		return "", err
	}

	return string(bytes), nil
}

// GetIptablesVersion returns the major minor and patch version of iptables
// which will all be zero in case of error, and any error encountered.
func GetIptablesVersion(exec utilexec.Interface) (int, int, int, error) {
	s, err := getIptablesVersionString(exec)
	if err != nil {
		return 0, 0, 0, err
	}
	return extractIptablesVersion(s)
}

// Checks if an iptables version is after 1.4.11, when --check was added
func iptablesHasCheckCommand(v1 int, v2 int, v3 int) bool {
	if v1 > 1 {
		return true
	}
	if v1 == 1 && v2 > 4 {
		return true
	}
	if v1 == 1 && v2 == 4 && v3 >= 11 {
		return true
	}
	return false
}
