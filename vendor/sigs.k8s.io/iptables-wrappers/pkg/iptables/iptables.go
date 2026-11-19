//go:build linux

/*
Copyright The Kubernetes Authors.

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
	"bufio"
	"bytes"
	"context"
	"errors"
	"fmt"
	"regexp"
	"strconv"
	"strings"
	"time"

	"k8s.io/klog/v2"
	utilexec "k8s.io/utils/exec"
	utiltrace "k8s.io/utils/trace"

	"sigs.k8s.io/iptables-wrappers/pkg/xtables"
)

// Interface is an injectable interface for running iptables commands.  Implementations must be goroutine-safe.
type Interface interface {
	// IsIPv6 returns true if this is managing ipv6 tables.
	IsIPv6() bool
	// Protocol returns the IP family this instance is managing,
	Protocol() Protocol

	// EnsureChain checks if the specified chain exists and, if not, creates it.  If the chain existed, return true.
	EnsureChain(ctx context.Context, table Table, chain Chain) (bool, error)
	// FlushChain clears the specified chain.  If the chain did not exist, return error.
	FlushChain(ctx context.Context, table Table, chain Chain) error
	// DeleteChain deletes the specified chain.  If the chain did not exist, return error.
	DeleteChain(ctx context.Context, table Table, chain Chain) error
	// ChainExists tests whether the specified chain exists, returning an error if it
	// does not, or if it is unable to check.
	ChainExists(ctx context.Context, table Table, chain Chain) (bool, error)

	// EnsureRule checks if the specified rule is present and, if not, creates it.  If the rule existed, return true.
	EnsureRule(ctx context.Context, position RulePosition, table Table, chain Chain, args ...string) (bool, error)
	// DeleteRule checks if the specified rule is present and, if so, deletes it.
	DeleteRule(ctx context.Context, table Table, chain Chain, args ...string) error

	// SaveInto calls `iptables-save` for table and stores result in a given buffer.
	SaveInto(ctx context.Context, table Table, buffer *bytes.Buffer) error
	// Restore runs `iptables-restore` passing data through []byte.
	// data should be formatted like the output of SaveInto()
	Restore(ctx context.Context, data []byte) error

	// Monitor detects when the given iptables tables have been flushed by an external
	// tool (e.g. a firewall reload) by creating canary chains and polling to see if
	// they have been deleted. (Specifically, it polls tables[0] every interval until
	// the canary has been deleted from there, then waits a short additional time for
	// the canaries to be deleted from the remaining tables as well. You can optimize
	// the polling by listing a relatively empty table in tables[0]). When a flush is
	// detected, this calls the reloadFunc so the caller can reload their own iptables
	// rules. If it is unable to create the canary chains (either initially or after
	// a reload) it will log an error and stop monitoring.
	// (This function should be called from a goroutine.)
	Monitor(ctx context.Context, canary Chain, tables []Table, reloadFunc func(), interval time.Duration)
}

// Mode defines how to use the different iptables modes
type Mode string

const (
	// UseInstalledMode means to just use the "iptables", etc, binaries directly,
	// assuming that they are already linked to the correct mode-specific binary. Use
	// this (a) in uncontainerized binaries like kubelet or CNI plugins that want to
	// just use the system iptables mode, or (b) in pod-network pods that just want to
	// use the iptables binary from their container image and don't need to care what
	// anyone else is doing.
	UseInstalledMode Mode = "installed"

	// DetectHostMode means to use xtables.DetectMode to detect whether kubelet has
	// created rules using LegacyMode or NFTMode. Use this in host-network pods.
	DetectHostMode Mode = "detect"

	// LegacyMode means to use "iptables-legacy", etc (regardless of whether that's
	// what anybody else is using).
	LegacyMode Mode = Mode(xtables.LegacyMode)

	// NFTMode means to use "iptables-nft", etc (regardless of whether that's what
	// anybody else is using).
	NFTMode Mode = Mode(xtables.NFTMode)
)

// RulePosition holds the -I/-A flags for iptable
type RulePosition string

const (
	// Prepend is the insert flag for iptable
	Prepend RulePosition = "-I"
	// Append is the append flag for iptable
	Append RulePosition = "-A"
)

// Protocol defines the ip protocol either ipv4 or ipv6
type Protocol string

const (
	// ProtocolIPv4 represents ipv4 protocol in iptables
	ProtocolIPv4 Protocol = "IPv4"
	// ProtocolIPv6 represents ipv6 protocol in iptables
	ProtocolIPv6 Protocol = "IPv6"
)

// Table represents different iptable like filter,nat, mangle and raw
type Table string

const (
	// TableNAT represents the built-in nat table
	TableNAT Table = "nat"
	// TableFilter represents the built-in filter table
	TableFilter Table = "filter"
	// TableMangle represents the built-in mangle table
	TableMangle Table = "mangle"
)

// Chain represents the different rules
type Chain string

const (
	// ChainPostrouting used for source NAT in nat table
	ChainPostrouting Chain = "POSTROUTING"
	// ChainPrerouting used for DNAT (destination NAT) in nat table
	ChainPrerouting Chain = "PREROUTING"
	// ChainOutput used for the packets going out from local
	ChainOutput Chain = "OUTPUT"
	// ChainInput used for incoming packets
	ChainInput Chain = "INPUT"
	// ChainForward used for the packets for another NIC
	ChainForward Chain = "FORWARD"
)

const (
	cmdIPTablesSave     string = "iptables-save"
	cmdIPTablesRestore  string = "iptables-restore"
	cmdIPTables         string = "iptables"
	cmdIP6TablesRestore string = "ip6tables-restore"
	cmdIP6TablesSave    string = "ip6tables-save"
	cmdIP6Tables        string = "ip6tables"
)

// waitString a constant for specifying the wait flag
const waitString = "-w"

// waitSecondsValue a constant for specifying the default wait seconds
const waitSecondsValue = "5"

// runner implements Interface in terms of exec("iptables").
type runner struct {
	exec     utilexec.Interface
	protocol Protocol
}

// newIPTablesExec returns a utilexec.Interface for mode
func newIPTablesExec(ctx context.Context, mode Mode) utilexec.Interface {
	exec := utilexec.New()
	switch mode {
	case UseInstalledMode:
		return exec
	case DetectHostMode:
		logger := klog.FromContext(ctx)
		if hostMode, err := xtables.DetectMode(ctx, exec); err == nil {
			logger.Info("Auto-detected host iptables mode", "mode", hostMode)
			return xtables.NewExecWrapper(exec, hostMode)
		} else {
			logger.Error(err, "Failed to auto-detect host iptables mode. Will use the container's default iptables binary.")
			// Fall back to UseInstalledMode.
			return exec
		}
	default:
		return xtables.NewExecWrapper(exec, xtables.Mode(mode))
	}
}

// newInternal returns a new Interface which will exec iptables
func newInternal(ctx context.Context, exec utilexec.Interface, protocol Protocol) (*runner, error) {
	ipt := &runner{
		exec:     exec,
		protocol: protocol,
	}
	if _, err := ipt.ChainExists(ctx, TableNAT, ChainPostrouting); err != nil {
		return nil, err
	}
	return ipt, nil
}

// New returns a new Interface which will exec iptables. If the necessary iptables binary
// is not present or doesn't work, this will return an error. For processes running in
// host-network pods, mode should be `DetectHostMode`. For most other cases, it should
// be `UseInstalledMode`. (See the `Mode` documentation.)
func New(ctx context.Context, protocol Protocol, mode Mode) (Interface, error) {
	return newInternal(ctx, newIPTablesExec(ctx, mode), protocol)
}

func newDualStackInternal(ctx context.Context, exec utilexec.Interface) map[Protocol]Interface {
	interfaces := map[Protocol]Interface{}
	if ipt, err := newInternal(ctx, exec, ProtocolIPv4); err == nil {
		interfaces[ProtocolIPv4] = ipt
	}
	if ipt, err := newInternal(ctx, exec, ProtocolIPv6); err == nil {
		interfaces[ProtocolIPv6] = ipt
	}
	return interfaces
}

// NewBestEffort returns a map containing an IPv4 Interface (if IPv4 iptables is
// supported) and an IPv6 Interface (if IPv6 iptables is supported). If iptables is not
// supported, then it just returns an empty map. This function is intended to make things
// simple for callers that just want "best-effort" iptables support, where neither partial
// nor complete lack of iptables support is considered an error.
func NewBestEffort(ctx context.Context, mode Mode) map[Protocol]Interface {
	return newDualStackInternal(ctx, newIPTablesExec(ctx, mode))
}

// IsIPv6 is part of Interface.
func (ipt *runner) IsIPv6() bool {
	return ipt.protocol == ProtocolIPv6
}

// Protocol is part of Interface.
func (ipt *runner) Protocol() Protocol {
	return ipt.protocol
}

// EnsureChain is part of Interface.
func (ipt *runner) EnsureChain(ctx context.Context, table Table, chain Chain) (bool, error) {
	fullArgs := makeFullArgs(table, chain)

	out, err := ipt.run(ctx, opCreateChain, fullArgs)
	if err != nil {
		if ee, ok := errors.AsType[utilexec.ExitError](err); ok {
			if ee.Exited() && ee.ExitStatus() == 1 {
				return true, nil
			}
		}
		return false, fmt.Errorf("error creating chain %q: %w: %s", chain, err, out)
	}
	return false, nil
}

// FlushChain is part of Interface.
func (ipt *runner) FlushChain(ctx context.Context, table Table, chain Chain) error {
	fullArgs := makeFullArgs(table, chain)

	out, err := ipt.run(ctx, opFlushChain, fullArgs)
	if err != nil {
		return fmt.Errorf("error flushing chain %q: %w: %s", chain, err, out)
	}
	return nil
}

// DeleteChain is part of Interface.
func (ipt *runner) DeleteChain(ctx context.Context, table Table, chain Chain) error {
	fullArgs := makeFullArgs(table, chain)

	out, err := ipt.run(ctx, opDeleteChain, fullArgs)
	if err != nil {
		return fmt.Errorf("error deleting chain %q: %w: %s", chain, err, out)
	}
	return nil
}

// ChainExists is part of Interface
func (ipt *runner) ChainExists(ctx context.Context, table Table, chain Chain) (bool, error) {
	fullArgs := makeFullArgs(table, chain)

	trace := utiltrace.New("iptables ChainExists")
	defer trace.LogIfLong(2 * time.Second)

	out, err := ipt.run(ctx, opListChain, fullArgs)
	if err != nil {
		return false, fmt.Errorf("error listing chain %q in table %q: %w: %s", chain, table, err, out)
	}
	return true, nil
}

// EnsureRule is part of Interface.
func (ipt *runner) EnsureRule(ctx context.Context, position RulePosition, table Table, chain Chain, args ...string) (bool, error) {
	fullArgs := makeFullArgs(table, chain, args...)

	exists, err := ipt.checkRule(ctx, fullArgs)
	if err != nil {
		return false, err
	}
	if exists {
		return true, nil
	}
	out, err := ipt.run(ctx, operation(position), fullArgs)
	if err != nil {
		return false, fmt.Errorf("error appending rule: %w: %s", err, out)
	}
	return false, nil
}

// DeleteRule is part of Interface.
func (ipt *runner) DeleteRule(ctx context.Context, table Table, chain Chain, args ...string) error {
	fullArgs := makeFullArgs(table, chain, args...)

	exists, err := ipt.checkRule(ctx, fullArgs)
	if err != nil {
		return err
	}
	if !exists {
		return nil
	}
	out, err := ipt.run(ctx, opDeleteRule, fullArgs)
	if err != nil {
		return fmt.Errorf("error deleting rule: %w: %s", err, out)
	}
	return nil
}

// SaveInto is part of Interface.
func (ipt *runner) SaveInto(ctx context.Context, table Table, buffer *bytes.Buffer) error {
	trace := utiltrace.New("iptables save")
	defer trace.LogIfLong(2 * time.Second)

	// run and return
	iptablesSaveCmd := iptablesSaveCommand(ipt.protocol)
	args := []string{"-t", string(table)}
	klog.V(4).InfoS("Running", "command", iptablesSaveCmd, "arguments", args)
	cmd := ipt.exec.Command(iptablesSaveCmd, args...)
	cmd.SetStdout(buffer)
	stderrBuffer := bytes.NewBuffer(nil)
	cmd.SetStderr(stderrBuffer)

	err := cmd.Run()
	if err != nil {
		_, _ = stderrBuffer.WriteTo(buffer) // ignore error, since we need to return the original error
	}
	return err
}

// Restore is part of Interface.
func (ipt *runner) Restore(ctx context.Context, data []byte) error {
	trace := utiltrace.New("iptables restore")
	defer trace.LogIfLong(2 * time.Second)

	// run the command and return the output or an error including the output and error
	args := []string{"--noflush", waitString, waitSecondsValue}
	iptablesRestoreCmd := iptablesRestoreCommand(ipt.protocol)
	klog.V(4).InfoS("Running", "command", iptablesRestoreCmd, "arguments", args)
	cmd := ipt.exec.CommandContext(ctx, iptablesRestoreCmd, args...)
	cmd.SetStdin(bytes.NewBuffer(data))
	b, err := cmd.CombinedOutput()
	if err != nil {
		pErr, ok := parseRestoreError(string(b))
		if ok {
			return pErr
		}
		return fmt.Errorf("%w: %s", err, b)
	}
	return nil
}

func iptablesSaveCommand(protocol Protocol) string {
	if protocol == ProtocolIPv6 {
		return cmdIP6TablesSave
	}
	return cmdIPTablesSave
}

func iptablesRestoreCommand(protocol Protocol) string {
	if protocol == ProtocolIPv6 {
		return cmdIP6TablesRestore
	}
	return cmdIPTablesRestore
}

func iptablesCommand(protocol Protocol) string {
	if protocol == ProtocolIPv6 {
		return cmdIP6Tables
	}
	return cmdIPTables
}

func (ipt *runner) run(ctx context.Context, op operation, args []string) ([]byte, error) {
	iptablesCmd := iptablesCommand(ipt.protocol)
	fullArgs := append([]string{waitString, waitSecondsValue, string(op)}, args...)
	klog.V(5).InfoS("Running", "command", iptablesCmd, "arguments", fullArgs)
	if ctx == nil {
		return ipt.exec.Command(iptablesCmd, fullArgs...).CombinedOutput()
	}
	return ipt.exec.CommandContext(ctx, iptablesCmd, fullArgs...).CombinedOutput()
	// Don't log err here - callers might not think it is an error.
}

// Returns (bool, nil) if it was able to check the existence of the rule, or
// (<undefined>, error) if the process of checking failed.
func (ipt *runner) checkRule(ctx context.Context, args []string) (bool, error) {
	ctx, cancel := context.WithTimeout(ctx, 5*time.Minute)
	defer cancel()

	out, err := ipt.run(ctx, opCheckRule, args)
	if ctx.Err() == context.DeadlineExceeded {
		return false, fmt.Errorf("timed out while checking rules")
	}
	if err == nil {
		return true, nil
	}
	if ee, ok := errors.AsType[utilexec.ExitError](err); ok {
		// iptables uses exit(1) to indicate a failure of the operation,
		// as compared to a malformed commandline, for example.
		if ee.Exited() && ee.ExitStatus() == 1 {
			return false, nil
		}
	}
	return false, fmt.Errorf("error checking rule: %w: %s", err, out)
}

type operation string

const (
	opCreateChain operation = "-N"
	opFlushChain  operation = "-F"
	opDeleteChain operation = "-X"
	opListChain   operation = "-S"
	opCheckRule   operation = "-C"
	opDeleteRule  operation = "-D"
)

func makeFullArgs(table Table, chain Chain, args ...string) []string {
	return append([]string{string(chain), "-t", string(table)}, args...)
}

var iptablesNotFoundStrings = []string{
	// iptables-legacy [-A|-I] BAD-CHAIN [...]
	// iptables-legacy [-C|-D] GOOD-CHAIN [...non-matching rule...]
	// iptables-legacy [-X|-F|-Z] BAD-CHAIN
	// iptables-nft -X BAD-CHAIN
	// NB: iptables-nft [-F|-Z] BAD-CHAIN exits with no error
	"No chain/target/match by that name",

	// iptables-legacy [...] -j BAD-CHAIN
	// iptables-nft-1.8.0 [-A|-I] BAD-CHAIN [...]
	// iptables-nft-1.8.0 [-A|-I] GOOD-CHAIN -j BAD-CHAIN
	// NB: also matches some other things like "-m BAD-MODULE"
	"No such file or directory",

	// iptables-legacy [-C|-D] BAD-CHAIN [...]
	// iptables-nft [-C|-D] GOOD-CHAIN [...non-matching rule...]
	"does a matching rule exist",

	// iptables-nft-1.8.2 [-A|-C|-D|-I] BAD-CHAIN [...]
	// iptables-nft-1.8.2 [...] -j BAD-CHAIN
	"does not exist",
}

// IsNotFoundError returns true if the error indicates "not found".  It parses
// the error string looking for known values, which is imperfect; beware using
// this function for anything beyond deciding between logging or ignoring an
// error.
func IsNotFoundError(err error) bool {
	es := err.Error()
	for _, str := range iptablesNotFoundStrings {
		if strings.Contains(es, str) {
			return true
		}
	}
	return false
}

const iptablesStatusResourceProblem = 4

// isResourceError returns true if the error indicates that iptables ran into a "resource
// problem" and was unable to attempt the request. In particular, this will be true if it
// times out trying to get the iptables lock.
func isResourceError(err error) bool {
	if ee, isExitError := errors.AsType[utilexec.ExitError](err); isExitError {
		return ee.ExitStatus() == iptablesStatusResourceProblem
	}
	return false
}

// ParseError records the payload when iptables reports an error parsing its input.
type ParseError interface {
	// Line returns the line number on which the parse error was reported.
	// NOTE: First line is 1.
	Line() int
	// Error returns the error message of the parse error, including line number.
	Error() string
}

type parseError struct {
	cmd  string
	line int
}

func (e parseError) Line() int {
	return e.line
}

func (e parseError) Error() string {
	return fmt.Sprintf("%s: input error on line %d: ", e.cmd, e.line)
}

// LineData represents a single numbered line of data.
type LineData struct {
	// Line holds the line number (the first line is 1).
	Line int
	// The data of the line.
	Data string
}

var regexpParseError = regexp.MustCompile("line ([1-9][0-9]*) failed$")

// parseRestoreError extracts the line from the error, if it matches returns parseError
// for example:
// input: iptables-restore: line 51 failed
// output: parseError:  cmd = iptables-restore, line = 51
// NOTE: parseRestoreError depends on the error format of iptables, if it ever changes
// we need to update this function
func parseRestoreError(str string) (ParseError, bool) {
	errors := strings.Split(str, ":")
	if len(errors) != 2 {
		return nil, false
	}
	cmd := errors[0]
	matches := regexpParseError.FindStringSubmatch(errors[1])
	if len(matches) != 2 {
		return nil, false
	}
	line, errMsg := strconv.Atoi(matches[1])
	if errMsg != nil {
		return nil, false
	}
	return parseError{cmd: cmd, line: line}, true
}

// ExtractLines extracts the -count and +count data from the lineNum row of lines and return
// NOTE: lines start from line 1
func ExtractLines(lines []byte, line, count int) []LineData {
	// first line is line 1, so line can't be smaller than 1
	if line < 1 {
		return nil
	}
	start := line - count
	if start <= 0 {
		start = 1
	}
	end := line + count + 1

	offset := 1
	scanner := bufio.NewScanner(bytes.NewBuffer(lines))
	extractLines := make([]LineData, 0, count*2)
	for scanner.Scan() {
		if offset >= start && offset < end {
			extractLines = append(extractLines, LineData{
				Line: offset,
				Data: scanner.Text(),
			})
		}
		if offset == end {
			break
		}
		offset++
	}
	return extractLines
}
