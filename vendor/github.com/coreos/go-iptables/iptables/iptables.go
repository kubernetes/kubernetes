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
	"bytes"
	"fmt"
	"log"
	"os/exec"
	"regexp"
	"strconv"
	"strings"
	"syscall"
)

// Adds the output of stderr to exec.ExitError
type Error struct {
	exec.ExitError
	msg string
}

func (e *Error) ExitStatus() int {
	return e.Sys().(syscall.WaitStatus).ExitStatus()
}

func (e *Error) Error() string {
	return fmt.Sprintf("exit status %v: %v", e.ExitStatus(), e.msg)
}

type IPTables struct {
	path string
}

func New() (*IPTables, error) {
	path, err := exec.LookPath("iptables")
	if err != nil {
		return nil, err
	}

	return &IPTables{path}, nil
}

// Exists checks if given rulespec in specified table/chain exists
func (ipt *IPTables) Exists(table, chain string, rulespec ...string) (bool, error) {
	checkPresent, err := getIptablesHasCheckCommand()
	if err != nil {
		log.Printf("Error checking iptables version, assuming version at least 1.4.11: %v", err)
		checkPresent = true
	}

	if !checkPresent {
		cmd := append([]string{"-A", chain}, rulespec...)
		return existsForOldIpTables(table, strings.Join(cmd, " "))
	} else {
		cmd := append([]string{"-t", table, "-C", chain}, rulespec...)
		err := ipt.run(cmd...)

		switch {
		case err == nil:
			return true, nil
		case err.(*Error).ExitStatus() == 1:
			return false, nil
		default:
			return false, err
		}
	}
}

// Insert inserts rulespec to specified table/chain (in specified pos)
func (ipt *IPTables) Insert(table, chain string, pos int, rulespec ...string) error {
	cmd := append([]string{"-t", table, "-I", chain, strconv.Itoa(pos)}, rulespec...)
	return ipt.run(cmd...)
}

// Append appends rulespec to specified table/chain
func (ipt *IPTables) Append(table, chain string, rulespec ...string) error {
	cmd := append([]string{"-t", table, "-A", chain}, rulespec...)
	return ipt.run(cmd...)
}

// AppendUnique acts like Append except that it won't add a duplicate
func (ipt *IPTables) AppendUnique(table, chain string, rulespec ...string) error {
	exists, err := ipt.Exists(table, chain, rulespec...)
	if err != nil {
		return err
	}

	if !exists {
		return ipt.Append(table, chain, rulespec...)
	}

	return nil
}

// Delete removes rulespec in specified table/chain
func (ipt *IPTables) Delete(table, chain string, rulespec ...string) error {
	cmd := append([]string{"-t", table, "-D", chain}, rulespec...)
	return ipt.run(cmd...)
}

// List rules in specified table/chain
func (ipt *IPTables) List(table, chain string) ([]string, error) {
	var stdout, stderr bytes.Buffer
	cmd := exec.Cmd{
		Path:   ipt.path,
		Args:   []string{ipt.path, "--wait", "-t", table, "-S", chain},
		Stdout: &stdout,
		Stderr: &stderr,
	}

	if err := cmd.Run(); err != nil {
		return nil, &Error{*(err.(*exec.ExitError)), stderr.String()}
	}

	rules := strings.Split(stdout.String(), "\n")
	if len(rules) > 0 && rules[len(rules)-1] == "" {
		rules = rules[:len(rules)-1]
	}

	return rules, nil
}

func (ipt *IPTables) NewChain(table, chain string) error {
	return ipt.run("-t", table, "-N", chain)
}

// ClearChain flushed (deletes all rules) in the specifed table/chain.
// If the chain does not exist, new one will be created
func (ipt *IPTables) ClearChain(table, chain string) error {
	err := ipt.NewChain(table, chain)

	switch {
	case err == nil:
		return nil
	case err.(*Error).ExitStatus() == 1:
		// chain already exists. Flush (clear) it.
		return ipt.run("-t", table, "-F", chain)
	default:
		return err
	}
}

// DeleteChain deletes the chain in the specified table.
// The chain must be empty
func (ipt *IPTables) DeleteChain(table, chain string) error {
	return ipt.run("-t", table, "-X", chain)
}

func (ipt *IPTables) run(args ...string) error {
	var stderr bytes.Buffer
	args = append([]string{"--wait"}, args...)
	cmd := exec.Cmd{
		Path:   ipt.path,
		Args:   append([]string{ipt.path}, args...),
		Stderr: &stderr,
	}

	if err := cmd.Run(); err != nil {
		return &Error{*(err.(*exec.ExitError)), stderr.String()}
	}

	return nil
}

// Checks if iptables has the "-C" flag
func getIptablesHasCheckCommand() (bool, error) {
	vstring, err := getIptablesVersionString()
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
func getIptablesVersionString() (string, error) {
	cmd := exec.Command("iptables", "--version")
	var out bytes.Buffer
	cmd.Stdout = &out
	err := cmd.Run()
	if err != nil {
		return "", err
	}
	return out.String(), nil
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

// Checks if a rule specification exists for a table
func existsForOldIpTables(table string, ruleSpec string) (bool, error) {
	cmd := exec.Command("iptables", "-t", table, "-S")
	var out bytes.Buffer
	cmd.Stdout = &out
	err := cmd.Run()
	if err != nil {
		return false, err
	}
	rules := out.String()
	return strings.Contains(rules, ruleSpec), nil
}
