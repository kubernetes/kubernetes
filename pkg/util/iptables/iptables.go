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
	"context"
	"fmt"
	"regexp"
	"strings"
	"sync"
	"time"

	godbus "github.com/godbus/dbus"
	"github.com/golang/glog"
	"k8s.io/apimachinery/pkg/util/sets"
	utiltrace "k8s.io/apiserver/pkg/util/trace"
	utildbus "k8s.io/kubernetes/pkg/util/dbus"
	utilversion "k8s.io/kubernetes/pkg/util/version"
	utilexec "k8s.io/utils/exec"
)

type RulePosition string

const (
	Prepend RulePosition = "-I"
	Append  RulePosition = "-A"
)

// An injectable interface for running iptables commands.  Implementations must be goroutine-safe.
type Interface interface {
	// GetVersion returns the "X.Y.Z" version string for iptables.
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
	// SaveInto calls `iptables-save` for table and stores result in a given buffer.
	SaveInto(table Table, buffer *bytes.Buffer) error
	// Restore runs `iptables-restore` passing data through []byte.
	// table is the Table to restore
	// data should be formatted like the output of SaveInto()
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
	TableMangle Table = "mangle"
)

type Chain string

const (
	ChainPostrouting Chain = "POSTROUTING"
	ChainPrerouting  Chain = "PREROUTING"
	ChainOutput      Chain = "OUTPUT"
	ChainInput       Chain = "INPUT"
	ChainForward     Chain = "FORWARD"
)

const (
	cmdIPTablesSave     string = "iptables-save"
	cmdIPTablesRestore  string = "iptables-restore"
	cmdIPTables         string = "iptables"
	cmdIP6TablesRestore string = "ip6tables-restore"
	cmdIP6TablesSave    string = "ip6tables-save"
	cmdIP6Tables        string = "ip6tables"
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

// Minimum iptables versions supporting the -w and -w<seconds> flags
const WaitMinVersion = "1.4.20"
const WaitSecondsMinVersion = "1.4.22"
const WaitString = "-w"
const WaitSecondsValue = "5"

const LockfilePath16x = "/run/xtables.lock"

// runner implements Interface in terms of exec("iptables").
type runner struct {
	mu              sync.Mutex
	exec            utilexec.Interface
	dbus            utildbus.Interface
	protocol        Protocol
	hasCheck        bool
	hasListener     bool
	waitFlag        []string
	restoreWaitFlag []string
	lockfilePath    string

	reloadFuncs []func()
	signal      chan *godbus.Signal
}

// newInternal returns a new Interface which will exec iptables, and allows the
// caller to change the iptables-restore lockfile path
func newInternal(exec utilexec.Interface, dbus utildbus.Interface, protocol Protocol, lockfilePath string) Interface {
	vstring, err := getIPTablesVersionString(exec, protocol)
	if err != nil {
		glog.Warningf("Error checking iptables version, assuming version at least %s: %v", MinCheckVersion, err)
		vstring = MinCheckVersion
	}

	if lockfilePath == "" {
		lockfilePath = LockfilePath16x
	}

	runner := &runner{
		exec:            exec,
		dbus:            dbus,
		protocol:        protocol,
		hasCheck:        getIPTablesHasCheckCommand(vstring),
		hasListener:     false,
		waitFlag:        getIPTablesWaitFlag(vstring),
		restoreWaitFlag: getIPTablesRestoreWaitFlag(exec, protocol),
		lockfilePath:    lockfilePath,
	}
	return runner
}

// New returns a new Interface which will exec iptables.
func New(exec utilexec.Interface, dbus utildbus.Interface, protocol Protocol) Interface {
	return newInternal(exec, dbus, protocol, "")
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
	runner.hasListener = true

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
	return getIPTablesVersionString(runner.exec, runner.protocol)
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

// SaveInto is part of Interface.
func (runner *runner) SaveInto(table Table, buffer *bytes.Buffer) error {
	runner.mu.Lock()
	defer runner.mu.Unlock()

	trace := utiltrace.New("iptables save")
	defer trace.LogIfLong(2 * time.Second)

	// run and return
	iptablesSaveCmd := iptablesSaveCommand(runner.protocol)
	args := []string{"-t", string(table)}
	glog.V(4).Infof("running %s %v", iptablesSaveCmd, args)
	cmd := runner.exec.Command(iptablesSaveCmd, args...)
	// Since CombinedOutput() doesn't support redirecting it to a buffer,
	// we need to workaround it by redirecting stdout and stderr to buffer
	// and explicitly calling Run() [CombinedOutput() underneath itself
	// creates a new buffer, redirects stdout and stderr to it and also
	// calls Run()].
	cmd.SetStdout(buffer)
	cmd.SetStderr(buffer)
	return cmd.Run()
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

type iptablesLocker interface {
	Close() error
}

// restoreInternal is the shared part of Restore/RestoreAll
func (runner *runner) restoreInternal(args []string, data []byte, flush FlushFlag, counters RestoreCountersFlag) error {
	runner.mu.Lock()
	defer runner.mu.Unlock()

	trace := utiltrace.New("iptables restore")
	defer trace.LogIfLong(2 * time.Second)

	if !flush {
		args = append(args, "--noflush")
	}
	if counters {
		args = append(args, "--counters")
	}

	// Grab the iptables lock to prevent iptables-restore and iptables
	// from stepping on each other.  iptables-restore 1.6.2 will have
	// a --wait option like iptables itself, but that's not widely deployed.
	if len(runner.restoreWaitFlag) == 0 {
		locker, err := grabIptablesLocks(runner.lockfilePath)
		if err != nil {
			return err
		}
		trace.Step("Locks grabbed")
		defer func(locker iptablesLocker) {
			if err := locker.Close(); err != nil {
				glog.Errorf("Failed to close iptables locks: %v", err)
			}
		}(locker)
	}

	// run the command and return the output or an error including the output and error
	fullArgs := append(runner.restoreWaitFlag, args...)
	iptablesRestoreCmd := iptablesRestoreCommand(runner.protocol)
	glog.V(4).Infof("running %s %v", iptablesRestoreCmd, fullArgs)
	cmd := runner.exec.Command(iptablesRestoreCmd, fullArgs...)
	cmd.SetStdin(bytes.NewBuffer(data))
	b, err := cmd.CombinedOutput()
	if err != nil {
		return fmt.Errorf("%v (%s)", err, b)
	}
	return nil
}

func iptablesSaveCommand(protocol Protocol) string {
	if protocol == ProtocolIpv6 {
		return cmdIP6TablesSave
	} else {
		return cmdIPTablesSave
	}
}

func iptablesRestoreCommand(protocol Protocol) string {
	if protocol == ProtocolIpv6 {
		return cmdIP6TablesRestore
	} else {
		return cmdIPTablesRestore
	}
}

func iptablesCommand(protocol Protocol) string {
	if protocol == ProtocolIpv6 {
		return cmdIP6Tables
	} else {
		return cmdIPTables
	}
}

func (runner *runner) run(op operation, args []string) ([]byte, error) {
	return runner.runContext(nil, op, args)
}

func (runner *runner) runContext(ctx context.Context, op operation, args []string) ([]byte, error) {
	iptablesCmd := iptablesCommand(runner.protocol)
	fullArgs := append(runner.waitFlag, string(op))
	fullArgs = append(fullArgs, args...)
	glog.V(5).Infof("running iptables %s %v", string(op), args)
	if ctx == nil {
		return runner.exec.Command(iptablesCmd, fullArgs...).CombinedOutput()
	}
	return runner.exec.CommandContext(ctx, iptablesCmd, fullArgs...).CombinedOutput()
	// Don't log err here - callers might not think it is an error.
}

// Returns (bool, nil) if it was able to check the existence of the rule, or
// (<undefined>, error) if the process of checking failed.
func (runner *runner) checkRule(table Table, chain Chain, args ...string) (bool, error) {
	if runner.hasCheck {
		return runner.checkRuleUsingCheck(makeFullArgs(table, chain, args...))
	}
	return runner.checkRuleWithoutCheck(table, chain, args...)
}

var hexnumRE = regexp.MustCompile("0x0+([0-9])")

func trimhex(s string) string {
	return hexnumRE.ReplaceAllString(s, "0x$1")
}

// Executes the rule check without using the "-C" flag, instead parsing iptables-save.
// Present for compatibility with <1.4.11 versions of iptables.  This is full
// of hack and half-measures.  We should nix this ASAP.
func (runner *runner) checkRuleWithoutCheck(table Table, chain Chain, args ...string) (bool, error) {
	iptablesSaveCmd := iptablesSaveCommand(runner.protocol)
	glog.V(1).Infof("running %s -t %s", iptablesSaveCmd, string(table))
	out, err := runner.exec.Command(iptablesSaveCmd, "-t", string(table)).CombinedOutput()
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
		tmpField = trimhex(tmpField)
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
			fields[i] = trimhex(fields[i])
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
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
	defer cancel()

	out, err := runner.runContext(ctx, opCheckRule, args)
	if ctx.Err() == context.DeadlineExceeded {
		return false, fmt.Errorf("timed out while checking rules")
	}
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
	minVersion, err := utilversion.ParseGeneric(MinCheckVersion)
	if err != nil {
		glog.Errorf("MinCheckVersion (%s) is not a valid version string: %v", MinCheckVersion, err)
		return true
	}
	version, err := utilversion.ParseGeneric(vstring)
	if err != nil {
		glog.Errorf("vstring (%s) is not a valid version string: %v", vstring, err)
		return true
	}
	return version.AtLeast(minVersion)
}

// Checks if iptables version has a "wait" flag
func getIPTablesWaitFlag(vstring string) []string {
	version, err := utilversion.ParseGeneric(vstring)
	if err != nil {
		glog.Errorf("vstring (%s) is not a valid version string: %v", vstring, err)
		return nil
	}

	minVersion, err := utilversion.ParseGeneric(WaitMinVersion)
	if err != nil {
		glog.Errorf("WaitMinVersion (%s) is not a valid version string: %v", WaitMinVersion, err)
		return nil
	}
	if version.LessThan(minVersion) {
		return nil
	}

	minVersion, err = utilversion.ParseGeneric(WaitSecondsMinVersion)
	if err != nil {
		glog.Errorf("WaitSecondsMinVersion (%s) is not a valid version string: %v", WaitSecondsMinVersion, err)
		return nil
	}
	if version.LessThan(minVersion) {
		return []string{WaitString}
	} else {
		return []string{WaitString, WaitSecondsValue}
	}
}

// getIPTablesVersionString runs "iptables --version" to get the version string
// in the form "X.X.X"
func getIPTablesVersionString(exec utilexec.Interface, protocol Protocol) (string, error) {
	// this doesn't access mutable state so we don't need to use the interface / runner
	iptablesCmd := iptablesCommand(protocol)
	bytes, err := exec.Command(iptablesCmd, "--version").CombinedOutput()
	if err != nil {
		return "", err
	}
	versionMatcher := regexp.MustCompile("v([0-9]+(\\.[0-9]+)+)")
	match := versionMatcher.FindStringSubmatch(string(bytes))
	if match == nil {
		return "", fmt.Errorf("no iptables version found in string: %s", bytes)
	}
	return match[1], nil
}

// Checks if iptables-restore has a "wait" flag
// --wait support landed in v1.6.1+ right before --version support, so
// any version of iptables-restore that supports --version will also
// support --wait
func getIPTablesRestoreWaitFlag(exec utilexec.Interface, protocol Protocol) []string {
	vstring, err := getIPTablesRestoreVersionString(exec, protocol)
	if err != nil || vstring == "" {
		glog.V(3).Infof("couldn't get iptables-restore version; assuming it doesn't support --wait")
		return nil
	}
	if _, err := utilversion.ParseGeneric(vstring); err != nil {
		glog.V(3).Infof("couldn't parse iptables-restore version; assuming it doesn't support --wait")
		return nil
	}

	return []string{WaitString, WaitSecondsValue}
}

// getIPTablesRestoreVersionString runs "iptables-restore --version" to get the version string
// in the form "X.X.X"
func getIPTablesRestoreVersionString(exec utilexec.Interface, protocol Protocol) (string, error) {
	// this doesn't access mutable state so we don't need to use the interface / runner

	// iptables-restore hasn't always had --version, and worse complains
	// about unrecognized commands but doesn't exit when it gets them.
	// Work around that by setting stdin to nothing so it exits immediately.
	iptablesRestoreCmd := iptablesRestoreCommand(protocol)
	cmd := exec.Command(iptablesRestoreCmd, "--version")
	cmd.SetStdin(bytes.NewReader([]byte{}))
	bytes, err := cmd.CombinedOutput()
	if err != nil {
		return "", err
	}
	versionMatcher := regexp.MustCompile("v([0-9]+(\\.[0-9]+)+)")
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
	runner.mu.Lock()
	defer runner.mu.Unlock()

	// We only need to listen to firewalld if there are Reload functions, so lazy
	// initialize the listener.
	if !runner.hasListener {
		runner.connectToFirewallD()
	}

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
