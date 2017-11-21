/*
Copyright 2017 The Kubernetes Authors.

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

package ipset

import (
	"bytes"
	"fmt"
	"regexp"
	"strconv"
	"strings"

	utilexec "k8s.io/utils/exec"
)

// Interface is an injectable interface for running ipset commands.  Implementations must be goroutine-safe.
type Interface interface {
	// FlushSet deletes all entries from a named set.
	FlushSet(set string) error
	// DestroySet deletes a named set.
	DestroySet(set string) error
	// DestroyAllSets deletes all sets.
	DestroyAllSets() error
	// CreateSet creates a new set,  it will ignore error when the set already exists if ignoreExistErr=true.
	CreateSet(set *IPSet, ignoreExistErr bool) error
	// AddEntry adds a new entry to the named set.
	AddEntry(entry string, set string, ignoreExistErr bool) error
	// DelEntry deletes one entry from the named set
	DelEntry(entry string, set string) error
	// Test test if an entry exists in the named set
	TestEntry(entry string, set string) (bool, error)
	// ListEntries lists all the entries from a named set
	ListEntries(set string) ([]string, error)
	// ListSets list all set names from kernel
	ListSets() ([]string, error)
	// GetVersion returns the "X.Y" version string for ipset.
	GetVersion() (string, error)
}

// IPSetCmd represents the ipset util.  We use ipset command for ipset execute.
const IPSetCmd = "ipset"

// EntryMemberPattern is the regular expression pattern of ipset member list.
// The raw output of ipset command `ipset list {set}` is similar to,
//Name: foobar
//Type: hash:ip,port
//Revision: 2
//Header: family inet hashsize 1024 maxelem 65536
//Size in memory: 16592
//References: 0
//Members:
//192.168.1.2,tcp:8080
//192.168.1.1,udp:53
var EntryMemberPattern = "(?m)^(.*\n)*Members:\n"

// VersionPattern is the regular expression pattern of ipset version string.
// ipset version output is similar to "v6.10".
var VersionPattern = "v[0-9]+\\.[0-9]+"

// IPSet implements an Interface to an set.
type IPSet struct {
	// Name is the set name.
	Name string
	// SetType specifies the ipset type.
	SetType Type
	// HashFamily specifies the protocol family of the IP addresses to be stored in the set.
	// The default is inet, i.e IPv4.  If users want to use IPv6, they should specify inet6.
	HashFamily string
	// HashSize specifies the hash table size of ipset.
	HashSize int
	// MaxElem specifies the max element number of ipset.
	MaxElem int
	// PortRange specifies the port range of bitmap:port type ipset.
	PortRange string
}

// Entry represents a ipset entry.
type Entry struct {
	// IP is the entry's IP.  The IP address protocol corresponds to the HashFamily of IPSet.
	// All entries' IP addresses in the same ip set has same the protocol, IPv4 or IPv6.
	IP string
	// Port is the entry's Port.
	Port int
	// Protocol is the entry's Protocol.  The protocols of entries in the same ip set are all
	// the same.  The accepted protocols are TCP and UDP.
	Protocol string
	// Net is the entry's IP network address.  Network address with zero prefix size can NOT
	// be stored.
	Net string
	// IP2 is the entry's second IP.  IP2 may not be empty for `hash:ip,port,ip` type ip set.
	IP2 string
	// SetType specifies the type of ip set where the entry exists.
	SetType Type
}

func (e *Entry) String() string {
	switch e.SetType {
	case HashIPPort:
		// Entry{192.168.1.1, udp, 53} -> 192.168.1.1,udp:53
		// Entry{192.168.1.2, tcp, 8080} -> 192.168.1.2,tcp:8080
		return fmt.Sprintf("%s,%s:%s", e.IP, e.Protocol, strconv.Itoa(e.Port))
	case HashIPPortIP:
		// Entry{192.168.1.1, udp, 53, 10.0.0.1} -> 192.168.1.1,udp:53,10.0.0.1
		// Entry{192.168.1.2, tcp, 8080, 192.168.1.2} -> 192.168.1.2,tcp:8080,192.168.1.2
		return fmt.Sprintf("%s,%s:%s,%s", e.IP, e.Protocol, strconv.Itoa(e.Port), e.IP2)
	case HashIPPortNet:
		// Entry{192.168.1.2, udp, 80, 10.0.1.0/24} -> 192.168.1.2,udp:80,10.0.1.0/24
		// Entry{192.168.2,25, tcp, 8080, 10.1.0.0/16} -> 192.168.2,25,tcp:8080,10.1.0.0/16
		return fmt.Sprintf("%s,%s:%s,%s", e.IP, e.Protocol, strconv.Itoa(e.Port), e.Net)
	case BitmapPort:
		// Entry{53} -> 53
		// Entry{8080} -> 8080
		return strconv.Itoa(e.Port)
	}
	return ""
}

type runner struct {
	exec utilexec.Interface
}

// New returns a new Interface which will exec ipset.
func New(exec utilexec.Interface) Interface {
	return &runner{
		exec: exec,
	}
}

// CreateSet creates a new set,  it will ignore error when the set already exists if ignoreExistErr=true.
func (runner *runner) CreateSet(set *IPSet, ignoreExistErr bool) error {
	// Using default values.
	if set.HashSize == 0 {
		set.HashSize = 1024
	}
	if set.MaxElem == 0 {
		set.MaxElem = 65536
	}
	if set.HashFamily == "" {
		set.HashFamily = ProtocolFamilyIPV4
	}
	if len(set.HashFamily) != 0 && set.HashFamily != ProtocolFamilyIPV4 && set.HashFamily != ProtocolFamilyIPV6 {
		return fmt.Errorf("Currently supported protocol families are: %s and %s, %s is not supported", ProtocolFamilyIPV4, ProtocolFamilyIPV6, set.HashFamily)
	}
	// Default ipset type is "hash:ip,port"
	if len(set.SetType) == 0 {
		set.SetType = HashIPPort
	}
	// Check if setType is supported
	if !IsValidIPSetType(set.SetType) {
		return fmt.Errorf("Currently supported ipset types are: %v, %s is not supported", ValidIPSetTypes, set.SetType)
	}

	return runner.createSet(set, ignoreExistErr)
}

// If ignoreExistErr is set to true, then the -exist option of ipset will be specified, ipset ignores the error
// otherwise raised when the same set (setname and create parameters are identical) already exists.
func (runner *runner) createSet(set *IPSet, ignoreExistErr bool) error {
	args := []string{"create", set.Name, string(set.SetType)}
	if set.SetType == HashIPPortIP || set.SetType == HashIPPort {
		args = append(args,
			"family", set.HashFamily,
			"hashsize", strconv.Itoa(set.HashSize),
			"maxelem", strconv.Itoa(set.MaxElem),
		)
	}
	if set.SetType == BitmapPort {
		if len(set.PortRange) == 0 {
			set.PortRange = DefaultPortRange
		}
		if !validatePortRange(set.PortRange) {
			return fmt.Errorf("invalid port range for %s type ip set: %s, expect: a-b", BitmapPort, set.PortRange)
		}
		args = append(args, "range", set.PortRange)
	}
	if ignoreExistErr {
		args = append(args, "-exist")
	}
	if _, err := runner.exec.Command(IPSetCmd, args...).CombinedOutput(); err != nil {
		return fmt.Errorf("error creating ipset %s, error: %v", set.Name, err)
	}
	return nil
}

// AddEntry adds a new entry to the named set.
// If the -exist option is specified, ipset ignores the error otherwise raised when
// the same set (setname and create parameters are identical) already exists.
func (runner *runner) AddEntry(entry string, set string, ignoreExistErr bool) error {
	args := []string{"add", set, entry}
	if ignoreExistErr {
		args = append(args, "-exist")
	}
	if _, err := runner.exec.Command(IPSetCmd, args...).CombinedOutput(); err != nil {
		return fmt.Errorf("error adding entry %s, error: %v", entry, err)
	}
	return nil
}

// DelEntry is used to delete the specified entry from the set.
func (runner *runner) DelEntry(entry string, set string) error {
	if _, err := runner.exec.Command(IPSetCmd, "del", set, entry).CombinedOutput(); err != nil {
		return fmt.Errorf("error deleting entry %s: from set: %s, error: %v", entry, set, err)
	}
	return nil
}

// TestEntry is used to check whether the specified entry is in the set or not.
func (runner *runner) TestEntry(entry string, set string) (bool, error) {
	if out, err := runner.exec.Command(IPSetCmd, "test", set, entry).CombinedOutput(); err == nil {
		reg, e := regexp.Compile("NOT")
		if e == nil && reg.MatchString(string(out)) {
			return false, nil
		} else if e == nil {
			return true, nil
		} else {
			return false, fmt.Errorf("error testing entry: %s, error: %v", entry, e)
		}
	} else {
		return false, fmt.Errorf("error testing entry %s: %v (%s)", entry, err, out)
	}
}

// FlushSet deletes all entries from a named set.
func (runner *runner) FlushSet(set string) error {
	if _, err := runner.exec.Command(IPSetCmd, "flush", set).CombinedOutput(); err != nil {
		return fmt.Errorf("error flushing set: %s, error: %v", set, err)
	}
	return nil
}

// DestroySet is used to destroy a named set.
func (runner *runner) DestroySet(set string) error {
	if _, err := runner.exec.Command(IPSetCmd, "destroy", set).CombinedOutput(); err != nil {
		return fmt.Errorf("error destroying set %s:, error: %v", set, err)
	}
	return nil
}

// DestroyAllSets is used to destroy all sets.
func (runner *runner) DestroyAllSets() error {
	if _, err := runner.exec.Command(IPSetCmd, "destroy").CombinedOutput(); err != nil {
		return fmt.Errorf("error destroying all sets, error: %v", err)
	}
	return nil
}

// ListSets list all set names from kernel
func (runner *runner) ListSets() ([]string, error) {
	out, err := runner.exec.Command(IPSetCmd, "list", "-n").CombinedOutput()
	if err != nil {
		return nil, fmt.Errorf("error listing all sets, error: %v", err)
	}
	return strings.Split(string(out), "\n"), nil
}

// ListEntries lists all the entries from a named set.
func (runner *runner) ListEntries(set string) ([]string, error) {
	if len(set) == 0 {
		return nil, fmt.Errorf("set name can't be nil")
	}
	out, err := runner.exec.Command(IPSetCmd, "list", set).CombinedOutput()
	if err != nil {
		return nil, fmt.Errorf("error listing set: %s, error: %v", set, err)
	}
	memberMatcher := regexp.MustCompile(EntryMemberPattern)
	list := memberMatcher.ReplaceAllString(string(out[:]), "")
	strs := strings.Split(list, "\n")
	results := make([]string, 0)
	for i := range strs {
		if len(strs[i]) > 0 {
			results = append(results, strs[i])
		}
	}
	return results, nil
}

// GetVersion returns the version string.
func (runner *runner) GetVersion() (string, error) {
	return getIPSetVersionString(runner.exec)
}

// getIPSetVersionString runs "ipset --version" to get the version string
// in the form of "X.Y", i.e "6.19"
func getIPSetVersionString(exec utilexec.Interface) (string, error) {
	cmd := exec.Command(IPSetCmd, "--version")
	cmd.SetStdin(bytes.NewReader([]byte{}))
	bytes, err := cmd.CombinedOutput()
	if err != nil {
		return "", err
	}
	versionMatcher := regexp.MustCompile(VersionPattern)
	match := versionMatcher.FindStringSubmatch(string(bytes))
	if match == nil {
		return "", fmt.Errorf("no ipset version found in string: %s", bytes)
	}
	return match[0], nil
}

func validatePortRange(portRange string) bool {
	strs := strings.Split(portRange, "-")
	if len(strs) != 2 {
		return false
	}
	for i := range strs {
		if _, err := strconv.Atoi(strs[i]); err != nil {
			return false
		}
	}
	return true
}

var _ = Interface(&runner{})
