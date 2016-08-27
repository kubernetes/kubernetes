/*
Copyright 2014 The Kubernetes Authors.

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
	"bytes"
	"fmt"
	"regexp"
	"strings"
	"sync"

	"github.com/coreos/go-semver/semver"
	godbus "github.com/godbus/dbus"
	"github.com/golang/glog"
	utildbus "k8s.io/kubernetes/pkg/util/dbus"
	utilexec "k8s.io/kubernetes/pkg/util/exec"
	"k8s.io/kubernetes/pkg/util/sets"
)

type RulePosition string

const (
	Prepend RulePosition = "-I"
	Append  RulePosition = "-A"
)

// An injectable interface for running iptables commands.  Implementations must be goroutine-safe.
type Interface interface {
	// GetVersion returns the "X.Y.Z" semver string for iptables.
	GetVersion() (string, error)
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
	// TODO: (BenTheElder) Unit-Test Save/SaveAll, Restore/RestoreAll
	// Save calls `iptables-save` for table.
	Save(table Table) ([]byte, error)
	// SaveAll calls `iptables-save`.
	SaveAll() ([]byte, error)
	// Restore runs `iptables-restore` passing data through []byte.
	// table is the Table to restore
	// data should be formatted like the output of Save()
	// flush sets the presence of the "--noflush" flag. see: FlushFlag
	// counters sets the "--counters" flag. see: RestoreCountersFlag
	Restore(table Table, data []byte, flush FlushFlag, counters RestoreCountersFlag) error
	// RestoreAll is the same as Restore except that no table is specified.
	RestoreAll(data []byte, flush FlushFlag, counters RestoreCountersFlag) error
	// AddReloadFunc adds a function to call on iptables reload
	AddReloadFunc(reloadFunc func())
	// Destroy cleans up resources used by the Interface
	Destroy()
}

type Protocol byte

const (
	ProtocolIpv4 Protocol = iota + 1
	ProtocolIpv6
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

const (
	cmdIPTablesSave    string = "iptables-save"
	cmdIPTablesRestore string = "iptables-restore"
	cmdIPTables        string = "iptables"
	cmdIp6tables       string = "ip6tables"
)

// Option flag for Restore
type RestoreCountersFlag bool

const RestoreCounters RestoreCountersFlag = true
const NoRestoreCounters RestoreCountersFlag = false

// Option flag for Flush
type FlushFlag bool

const FlushTables FlushFlag = true
const NoFlushTables FlushFlag = false

// Versions of iptables less than this do not support the -C / --check flag
// (test whether a rule exists).
const MinCheckVersion = "1.4.11"

// Minimum iptables versions supporting the -w and -w2 flags
const MinWaitVersion = "1.4.20"
const MinWait2Version = "1.4.22"

// runner implements Interface in terms of exec("iptables").
type runner struct {
	mu       sync.Mutex
	exec     utilexec.Interface
	dbus     utildbus.Interface
	protocol Protocol
	hasCheck bool
	waitFlag []string

	reloadFuncs []func()
	signal      chan *godbus.Signal
}

// New returns a new Interface which will exec iptables.
func New(exec utilexec.Interface, dbus utildbus.Interface, protocol Protocol) Interface {
	vstring, err := getIPTablesVersionString(exec)
	if err != nil {
		glog.Warningf("Error checking iptables version, assuming version at least %s: %v", MinCheckVersion, err)
		vstring = MinCheckVersion
	}
	runner := &runner{
		exec:     exec,
		dbus:     dbus,
		protocol: protocol,
		hasCheck: getIPTablesHasCheckCommand(vstring),
		waitFlag: getIPTablesWaitFlag(vstring),
	}
	runner.connectToFirewallD()
	return runner
}

// Destroy is part of Interface.
func (runner *runner) Destroy() {
	if runner.signal != nil {
		runner.signal <- nil
	}
}

const (
	firewalldName      = "org.fedoraproject.FirewallD1"
	firewalldPath      = "/org/fedoraproject/FirewallD1"
	firewalldInterface = "org.fedoraproject.FirewallD1"
)

// Connects to D-Bus and listens for FirewallD start/restart. (On non-FirewallD-using
// systems, this is effectively a no-op; we listen for the signals, but they will never be
// emitted, so reload() will never be called.)
func (runner *runner) connectToFirewallD() {
	bus, err := runner.dbus.SystemBus()
	if err != nil {
		glog.V(1).Infof("Could not connect to D-Bus system bus: %s", err)
		return
	}

	rule := fmt.Sprintf("type='signal',sender='%s',path='%s',interface='%s',member='Reloaded'", firewalldName, firewalldPath, firewalldInterface)
	bus.BusObject().Call("org.freedesktop.DBus.AddMatch", 0, rule)

	rule = fmt.Sprintf("type='signal',interface='org.freedesktop.DBus',member='NameOwnerChanged',path='/org/freedesktop/DBus',sender='org.freedesktop.DBus',arg0='%s'", firewalldName)
	bus.BusObject().Call("org.freedesktop.DBus.AddMatch", 0, rule)

	runner.signal = make(chan *godbus.Signal, 10)
	bus.Signal(runner.signal)

	go runner.dbusSignalHandler(bus)
}

// GetVersion returns the version string.
func (runner *runner) GetVersion() (string, error) {
	return getIPTablesVersionString(runner.exec)
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

	// TODO: we could call iptables -S first, ignore the output and check for non-zero return (more like DeleteRule)
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

// Save is part of Interface.
func (runner *runner) Save(table Table) ([]byte, error) {
	runner.mu.Lock()
	defer runner.mu.Unlock()

	// run and return
	args := []string{"-t", string(table)}
	glog.V(4).Infof("running iptables-save %v", args)
	return runner.exec.Command(cmdIPTablesSave, args...).CombinedOutput()
}

// SaveAll is part of Interface.
func (runner *runner) SaveAll() ([]byte, error) {
	runner.mu.Lock()
	defer runner.mu.Unlock()

	// run and return
	glog.V(4).Infof("running iptables-save")
	return runner.exec.Command(cmdIPTablesSave, []string{}...).CombinedOutput()
}

// Restore is part of Interface.
func (runner *runner) Restore(table Table, data []byte, flush FlushFlag, counters RestoreCountersFlag) error {
	// setup args
	args := []string{"-T", string(table)}
	return runner.restoreInternal(args, data, flush, counters)
}

// RestoreAll is part of Interface.
func (runner *runner) RestoreAll(data []byte, flush FlushFlag, counters RestoreCountersFlag) error {
	// setup args
	args := make([]string, 0)
	return runner.restoreInternal(args, data, flush, counters)
}

// restoreInternal is the shared part of Restore/RestoreAll
func (runner *runner) restoreInternal(args []string, data []byte, flush FlushFlag, counters RestoreCountersFlag) error {
	runner.mu.Lock()
	defer runner.mu.Unlock()

	if !flush {
		args = append(args, "--noflush")
	}
	if counters {
		args = append(args, "--counters")
	}

	// run the command and return the output or an error including the output and error
	glog.V(4).Infof("running iptables-restore %v", args)
	cmd := runner.exec.Command(cmdIPTablesRestore, args...)
	cmd.SetStdin(bytes.NewBuffer(data))
	b, err := cmd.CombinedOutput()
	if err != nil {
		return fmt.Errorf("%v (%s)", err, b)
	}
	return nil
}

func (runner *runner) iptablesCommand() string {
	if runner.IsIpv6() {
		return cmdIp6tables
	} else {
		return cmdIPTables
	}
}

func (runner *runner) run(op operation, args []string) ([]byte, error) {
	iptablesCmd := runner.iptablesCommand()

	fullArgs := append(runner.waitFlag, string(op))
	fullArgs = append(fullArgs, args...)
	glog.V(4).Infof("running iptables %s %v", string(op), args)
	return runner.exec.Command(iptablesCmd, fullArgs...).CombinedOutput()
	// Don't log err here - callers might not think it is an error.
}

// Returns (bool, nil) if it was able to check the existence of the rule, or
// (<undefined>, error) if the process of checking failed.
func (runner *runner) checkRule(table Table, chain Chain, args ...string) (bool, error) {
	if runner.hasCheck {
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
	out, err := runner.exec.Command(cmdIPTablesSave, "-t", string(table)).CombinedOutput()
	if err != nil {
		return false, fmt.Errorf("error checking rule: %v", err)
	}

	// Sadly, iptables has inconsistent quoting rules for comments. Just remove all quotes.
	// Also, quoted multi-word comments (which are counted as a single arg)
	// will be unpacked into multiple args,
	// in order to compare against iptables-save output (which will be split at whitespace boundary)
	// e.g. a single arg('"this must be before the NodePort rules"') will be unquoted and unpacked into 7 args.
	var argsCopy []string
	for i := range args {
		tmpField := strings.Trim(args[i], "\"")
		argsCopy = append(argsCopy, strings.Fields(tmpField)...)
	}
	argset := sets.NewString(argsCopy...)

	for _, line := range strings.Split(string(out), "\n") {
		var fields = strings.Fields(line)

		// Check that this is a rule for the correct chain, and that it has
		// the correct number of argument (+2 for "-A <chain name>")
		if !strings.HasPrefix(line, fmt.Sprintf("-A %s", string(chain))) || len(fields) != len(argsCopy)+2 {
			continue
		}

		// Sadly, iptables has inconsistent quoting rules for comments.
		// Just remove all quotes.
		for i := range fields {
			fields[i] = strings.Trim(fields[i], "\"")
		}

		// TODO: This misses reorderings e.g. "-x foo ! -y bar" will match "! -x foo -y bar"
		if sets.NewString(fields...).IsSuperset(argset) {
			return true, nil
		}
		glog.V(5).Infof("DBG: fields is not a superset of args: fields=%v  args=%v", fields, args)
	}

	return false, nil
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
func getIPTablesHasCheckCommand(vstring string) bool {
	minVersion, err := semver.NewVersion(MinCheckVersion)
	if err != nil {
		glog.Errorf("MinCheckVersion (%s) is not a valid version string: %v", MinCheckVersion, err)
		return true
	}
	version, err := semver.NewVersion(vstring)
	if err != nil {
		glog.Errorf("vstring (%s) is not a valid version string: %v", vstring, err)
		return true
	}
	if version.LessThan(*minVersion) {
		return false
	}
	return true
}

// Checks if iptables version has a "wait" flag
func getIPTablesWaitFlag(vstring string) []string {
	version, err := semver.NewVersion(vstring)
	if err != nil {
		glog.Errorf("vstring (%s) is not a valid version string: %v", vstring, err)
		return nil
	}

	minVersion, err := semver.NewVersion(MinWaitVersion)
	if err != nil {
		glog.Errorf("MinWaitVersion (%s) is not a valid version string: %v", MinWaitVersion, err)
		return nil
	}
	if version.LessThan(*minVersion) {
		return nil
	}

	minVersion, err = semver.NewVersion(MinWait2Version)
	if err != nil {
		glog.Errorf("MinWait2Version (%s) is not a valid version string: %v", MinWait2Version, err)
		return nil
	}
	if version.LessThan(*minVersion) {
		return []string{"-w"}
	} else {
		return []string{"-w2"}
	}
}

// getIPTablesVersionString runs "iptables --version" to get the version string
// in the form "X.X.X"
func getIPTablesVersionString(exec utilexec.Interface) (string, error) {
	// this doesn't access mutable state so we don't need to use the interface / runner
	bytes, err := exec.Command(cmdIPTables, "--version").CombinedOutput()
	if err != nil {
		return "", err
	}
	versionMatcher := regexp.MustCompile("v([0-9]+\\.[0-9]+\\.[0-9]+)")
	match := versionMatcher.FindStringSubmatch(string(bytes))
	if match == nil {
		return "", fmt.Errorf("no iptables version found in string: %s", bytes)
	}
	return match[1], nil
}

// goroutine to listen for D-Bus signals
func (runner *runner) dbusSignalHandler(bus utildbus.Connection) {
	firewalld := bus.Object(firewalldName, firewalldPath)

	for s := range runner.signal {
		if s == nil {
			// Unregister
			bus.Signal(runner.signal)
			return
		}

		switch s.Name {
		case "org.freedesktop.DBus.NameOwnerChanged":
			name := s.Body[0].(string)
			new_owner := s.Body[2].(string)

			if name != firewalldName || len(new_owner) == 0 {
				continue
			}

			// FirewallD startup (specifically the part where it deletes
			// all existing iptables rules) may not yet be complete when
			// we get this signal, so make a dummy request to it to
			// synchronize.
			firewalld.Call(firewalldInterface+".getDefaultZone", 0)

			runner.reload()
		case firewalldInterface + ".Reloaded":
			runner.reload()
		}
	}
}

// AddReloadFunc is part of Interface
func (runner *runner) AddReloadFunc(reloadFunc func()) {
	runner.reloadFuncs = append(runner.reloadFuncs, reloadFunc)
}

// runs all reload funcs to re-sync iptables rules
func (runner *runner) reload() {
	glog.V(1).Infof("reloading iptables rules")

	for _, f := range runner.reloadFuncs {
		f()
	}
}

// IsNotFoundError returns true if the error indicates "not found".  It parses
// the error string looking for known values, which is imperfect but works in
// practice.
func IsNotFoundError(err error) bool {
	es := err.Error()
	if strings.Contains(es, "No such file or directory") {
		return true
	}
	if strings.Contains(es, "No chain/target/match by that name") {
		return true
	}
	return false
}
