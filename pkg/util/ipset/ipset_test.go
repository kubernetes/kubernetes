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
	"reflect"
	"testing"

	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/utils/exec"
	fakeexec "k8s.io/utils/exec/testing"
)

func TestCheckIPSetVersion(t *testing.T) {
	testCases := []struct {
		vstring string
		Expect  string
		Err     bool
	}{
		{"ipset v4.0, protocol version: 4", "v4.0", false},
		{"ipset v5.1, protocol version: 5", "v5.1", false},
		{"ipset v6.0, protocol version: 6", "v6.0", false},
		{"ipset v6.1, protocol version: 6", "v6.1", false},
		{"ipset v6.19, protocol version: 6", "v6.19", false},
		{"total junk", "", true},
	}

	for i := range testCases {
		fcmd := fakeexec.FakeCmd{
			CombinedOutputScript: []fakeexec.FakeCombinedOutputAction{
				// ipset version response
				func() ([]byte, error) { return []byte(testCases[i].vstring), nil },
			},
		}

		fexec := fakeexec.FakeExec{
			CommandScript: []fakeexec.FakeCommandAction{
				func(cmd string, args ...string) exec.Cmd { return fakeexec.InitFakeCmd(&fcmd, cmd, args...) },
			},
		}

		gotVersion, err := getIPSetVersionString(&fexec)
		if (err != nil) != testCases[i].Err {
			t.Errorf("Expected error: %v, Got error: %v", testCases[i].Err, err)
		}
		if err == nil {
			if testCases[i].Expect != gotVersion {
				t.Errorf("Expected result: %v, Got result: %v", testCases[i].Expect, gotVersion)
			}
		}
	}
}

func TestFlushSet(t *testing.T) {
	fcmd := fakeexec.FakeCmd{
		CombinedOutputScript: []fakeexec.FakeCombinedOutputAction{
			// Success
			func() ([]byte, error) { return []byte{}, nil },
			// Success
			func() ([]byte, error) { return []byte{}, nil },
		},
	}
	fexec := fakeexec.FakeExec{
		CommandScript: []fakeexec.FakeCommandAction{
			func(cmd string, args ...string) exec.Cmd { return fakeexec.InitFakeCmd(&fcmd, cmd, args...) },
			func(cmd string, args ...string) exec.Cmd { return fakeexec.InitFakeCmd(&fcmd, cmd, args...) },
		},
	}
	runner := New(&fexec)
	// Success.
	err := runner.FlushSet("FOOBAR")
	if err != nil {
		t.Errorf("expected success, got %v", err)
	}
	if fcmd.CombinedOutputCalls != 1 {
		t.Errorf("expected 1 CombinedOutput() calls, got %d", fcmd.CombinedOutputCalls)
	}
	if !sets.NewString(fcmd.CombinedOutputLog[0]...).HasAll("ipset", "flush", "FOOBAR") {
		t.Errorf("wrong CombinedOutput() log, got %s", fcmd.CombinedOutputLog[0])
	}
	// Flush again
	err = runner.FlushSet("FOOBAR")
	if err != nil {
		t.Errorf("expected success, got %v", err)
	}
}

func TestDestroySet(t *testing.T) {
	fcmd := fakeexec.FakeCmd{
		CombinedOutputScript: []fakeexec.FakeCombinedOutputAction{
			// Success
			func() ([]byte, error) { return []byte{}, nil },
			// Failure
			func() ([]byte, error) {
				return []byte("ipset v6.19: The set with the given name does not exist"), &fakeexec.FakeExitError{Status: 1}
			},
		},
	}
	fexec := fakeexec.FakeExec{
		CommandScript: []fakeexec.FakeCommandAction{
			func(cmd string, args ...string) exec.Cmd { return fakeexec.InitFakeCmd(&fcmd, cmd, args...) },
			func(cmd string, args ...string) exec.Cmd { return fakeexec.InitFakeCmd(&fcmd, cmd, args...) },
		},
	}
	runner := New(&fexec)
	// Success
	err := runner.DestroySet("FOOBAR")
	if err != nil {
		t.Errorf("expected success, got %v", err)
	}
	if fcmd.CombinedOutputCalls != 1 {
		t.Errorf("expected 1 CombinedOutput() calls, got %d", fcmd.CombinedOutputCalls)
	}
	if !sets.NewString(fcmd.CombinedOutputLog[0]...).HasAll("ipset", "destroy", "FOOBAR") {
		t.Errorf("wrong CombinedOutput() log, got %s", fcmd.CombinedOutputLog[0])
	}
	// Failure
	err = runner.DestroySet("FOOBAR")
	if err == nil {
		t.Errorf("expected failure, got nil")
	}
}

func TestDestroyAllSets(t *testing.T) {
	fcmd := fakeexec.FakeCmd{
		CombinedOutputScript: []fakeexec.FakeCombinedOutputAction{
			// Success
			func() ([]byte, error) { return []byte{}, nil },
			// Success
			func() ([]byte, error) { return []byte{}, nil },
		},
	}
	fexec := fakeexec.FakeExec{
		CommandScript: []fakeexec.FakeCommandAction{
			func(cmd string, args ...string) exec.Cmd { return fakeexec.InitFakeCmd(&fcmd, cmd, args...) },
			func(cmd string, args ...string) exec.Cmd { return fakeexec.InitFakeCmd(&fcmd, cmd, args...) },
		},
	}
	runner := New(&fexec)
	// Success
	err := runner.DestroyAllSets()
	if err != nil {
		t.Errorf("expected success, got %v", err)
	}
	if fcmd.CombinedOutputCalls != 1 {
		t.Errorf("expected 1 CombinedOutput() calls, got %d", fcmd.CombinedOutputCalls)
	}
	if !sets.NewString(fcmd.CombinedOutputLog[0]...).HasAll("ipset", "destroy") {
		t.Errorf("wrong CombinedOutput() log, got %s", fcmd.CombinedOutputLog[0])
	}
	// Success
	err = runner.DestroyAllSets()
	if err != nil {
		t.Errorf("Unexpected failure: %v", err)
	}
}

func TestCreateSet(t *testing.T) {
	testSet := IPSet{
		Name:       "FOOBAR",
		SetType:    HashIPPort,
		HashFamily: ProtocolFamilyIPV4,
	}

	fcmd := fakeexec.FakeCmd{
		CombinedOutputScript: []fakeexec.FakeCombinedOutputAction{
			// Success
			func() ([]byte, error) { return []byte{}, nil },
			// Success
			func() ([]byte, error) { return []byte{}, nil },
			// Failure
			func() ([]byte, error) {
				return []byte("ipset v6.19: Set cannot be created: set with the same name already exists"), &fakeexec.FakeExitError{Status: 1}
			},
		},
	}
	fexec := fakeexec.FakeExec{
		CommandScript: []fakeexec.FakeCommandAction{
			func(cmd string, args ...string) exec.Cmd { return fakeexec.InitFakeCmd(&fcmd, cmd, args...) },
			func(cmd string, args ...string) exec.Cmd { return fakeexec.InitFakeCmd(&fcmd, cmd, args...) },
			func(cmd string, args ...string) exec.Cmd { return fakeexec.InitFakeCmd(&fcmd, cmd, args...) },
		},
	}
	runner := New(&fexec)
	// Create with ignoreExistErr = false, expect success
	err := runner.CreateSet(&testSet, false)
	if err != nil {
		t.Errorf("expected success, got %v", err)
	}
	if fcmd.CombinedOutputCalls != 1 {
		t.Errorf("expected 1 CombinedOutput() calls, got %d", fcmd.CombinedOutputCalls)
	}
	if !sets.NewString(fcmd.CombinedOutputLog[0]...).HasAll("ipset", "create", "FOOBAR", "hash:ip,port", "family", "inet", "hashsize", "1024", "maxelem", "65536") {
		t.Errorf("wrong CombinedOutput() log, got %s", fcmd.CombinedOutputLog[0])
	}
	// Create with ignoreExistErr = true, expect success
	err = runner.CreateSet(&testSet, true)
	if err != nil {
		t.Errorf("expected success, got %v", err)
	}
	if fcmd.CombinedOutputCalls != 2 {
		t.Errorf("expected 2 CombinedOutput() calls, got %d", fcmd.CombinedOutputCalls)
	}
	if !sets.NewString(fcmd.CombinedOutputLog[1]...).HasAll("ipset", "create", "FOOBAR", "hash:ip,port", "family", "inet", "hashsize", "1024", "maxelem", "65536", "-exist") {
		t.Errorf("wrong CombinedOutput() log, got %s", fcmd.CombinedOutputLog[1])
	}
	// Create with ignoreExistErr = false, expect failure
	err = runner.CreateSet(&testSet, false)
	if err == nil {
		t.Errorf("expected failure, got nil")
	}
}

func TestAddEntry(t *testing.T) {
	testEntry := &Entry{
		IP:       "192.168.1.1",
		Port:     53,
		Protocol: ProtocolUDP,
		SetType:  HashIPPort,
	}

	testSet := &IPSet{
		Name: "FOOBAR",
	}

	fcmd := fakeexec.FakeCmd{
		CombinedOutputScript: []fakeexec.FakeCombinedOutputAction{
			// Success
			func() ([]byte, error) { return []byte{}, nil },
			// Success
			func() ([]byte, error) { return []byte{}, nil },
			// Failure
			func() ([]byte, error) {
				return []byte("ipset v6.19: Set cannot be created: set with the same name already exists"), &fakeexec.FakeExitError{Status: 1}
			},
		},
	}
	fexec := fakeexec.FakeExec{
		CommandScript: []fakeexec.FakeCommandAction{
			func(cmd string, args ...string) exec.Cmd { return fakeexec.InitFakeCmd(&fcmd, cmd, args...) },
			func(cmd string, args ...string) exec.Cmd { return fakeexec.InitFakeCmd(&fcmd, cmd, args...) },
			func(cmd string, args ...string) exec.Cmd { return fakeexec.InitFakeCmd(&fcmd, cmd, args...) },
		},
	}
	runner := New(&fexec)
	// Create with ignoreExistErr = false, expect success
	err := runner.AddEntry(testEntry.String(), testSet, false)
	if err != nil {
		t.Errorf("expected success, got %v", err)
	}
	if fcmd.CombinedOutputCalls != 1 {
		t.Errorf("expected 1 CombinedOutput() calls, got %d", fcmd.CombinedOutputCalls)
	}
	if !sets.NewString(fcmd.CombinedOutputLog[0]...).HasAll("ipset", "add", "FOOBAR", "192.168.1.1,udp:53") {
		t.Errorf("wrong CombinedOutput() log, got %s", fcmd.CombinedOutputLog[0])
	}
	// Create with ignoreExistErr = true, expect success
	err = runner.AddEntry(testEntry.String(), testSet, true)
	if err != nil {
		t.Errorf("expected success, got %v", err)
	}
	if fcmd.CombinedOutputCalls != 2 {
		t.Errorf("expected 3 CombinedOutput() calls, got %d", fcmd.CombinedOutputCalls)
	}
	if !sets.NewString(fcmd.CombinedOutputLog[1]...).HasAll("ipset", "add", "FOOBAR", "192.168.1.1,udp:53", "-exist") {
		t.Errorf("wrong CombinedOutput() log, got %s", fcmd.CombinedOutputLog[1])
	}
	// Create with ignoreExistErr = false, expect failure
	err = runner.AddEntry(testEntry.String(), testSet, false)
	if err == nil {
		t.Errorf("expected failure, got nil")
	}
}

func TestDelEntry(t *testing.T) {
	// TODO: Test more set type
	testEntry := &Entry{
		IP:       "192.168.1.1",
		Port:     53,
		Protocol: ProtocolUDP,
		SetType:  HashIPPort,
	}

	fcmd := fakeexec.FakeCmd{
		CombinedOutputScript: []fakeexec.FakeCombinedOutputAction{
			// Success
			func() ([]byte, error) { return []byte{}, nil },
			// Failure
			func() ([]byte, error) {
				return []byte("ipset v6.19: Element cannot be deleted from the set: it's not added"), &fakeexec.FakeExitError{Status: 1}
			},
		},
	}
	fexec := fakeexec.FakeExec{
		CommandScript: []fakeexec.FakeCommandAction{
			func(cmd string, args ...string) exec.Cmd { return fakeexec.InitFakeCmd(&fcmd, cmd, args...) },
			func(cmd string, args ...string) exec.Cmd { return fakeexec.InitFakeCmd(&fcmd, cmd, args...) },
		},
	}
	runner := New(&fexec)
	err := runner.DelEntry(testEntry.String(), "FOOBAR")
	if err != nil {
		t.Errorf("expected success, got %v", err)
	}
	if fcmd.CombinedOutputCalls != 1 {
		t.Errorf("expected 1 CombinedOutput() calls, got %d", fcmd.CombinedOutputCalls)
	}
	if !sets.NewString(fcmd.CombinedOutputLog[0]...).HasAll("ipset", "del", "FOOBAR", "192.168.1.1,udp:53") {
		t.Errorf("wrong CombinedOutput() log, got %s", fcmd.CombinedOutputLog[0])
	}
	err = runner.DelEntry(testEntry.String(), "FOOBAR")
	if err == nil {
		t.Errorf("expected failure, got nil")
	}
}

func TestTestEntry(t *testing.T) {
	// TODO: IPv6?
	testEntry := &Entry{
		IP:       "10.120.7.100",
		Port:     8080,
		Protocol: ProtocolTCP,
		SetType:  HashIPPort,
	}

	fcmd := fakeexec.FakeCmd{
		CombinedOutputScript: []fakeexec.FakeCombinedOutputAction{
			// Success
			func() ([]byte, error) { return []byte("10.120.7.100,tcp:8080 is in set FOOBAR."), nil },
			// Failure
			func() ([]byte, error) {
				return []byte("192.168.1.3,tcp:8080 is NOT in set FOOBAR."), &fakeexec.FakeExitError{Status: 1}
			},
		},
	}
	fexec := fakeexec.FakeExec{
		CommandScript: []fakeexec.FakeCommandAction{
			func(cmd string, args ...string) exec.Cmd { return fakeexec.InitFakeCmd(&fcmd, cmd, args...) },
			func(cmd string, args ...string) exec.Cmd { return fakeexec.InitFakeCmd(&fcmd, cmd, args...) },
		},
	}
	runner := New(&fexec)
	// Success
	ok, err := runner.TestEntry(testEntry.String(), "FOOBAR")
	if err != nil {
		t.Errorf("expected success, got %v", err)
	}
	if fcmd.CombinedOutputCalls != 1 {
		t.Errorf("expected 2 CombinedOutput() calls, got %d", fcmd.CombinedOutputCalls)
	}
	if !sets.NewString(fcmd.CombinedOutputLog[0]...).HasAll("ipset", "test", "FOOBAR", "10.120.7.100,tcp:8080") {
		t.Errorf("wrong CombinedOutput() log, got %s", fcmd.CombinedOutputLog[0])
	}
	if !ok {
		t.Errorf("expect entry exists in test set, got not")
	}
	// Failure
	ok, err = runner.TestEntry(testEntry.String(), "FOOBAR")
	if err == nil || ok {
		t.Errorf("expect entry doesn't exist in test set")
	}
}

func TestListEntries(t *testing.T) {

	output := `Name: foobar
Type: hash:ip,port
Revision: 2
Header: family inet hashsize 1024 maxelem 65536
Size in memory: 16592
References: 0
Members:
192.168.1.2,tcp:8080
192.168.1.1,udp:53`

	emptyOutput := `Name: KUBE-NODE-PORT
Type: bitmap:port
Revision: 1
Header: range 0-65535
Size in memory: 524432
References: 1
Members:

`

	testCases := []struct {
		output   string
		expected []string
	}{
		{
			output:   output,
			expected: []string{"192.168.1.2,tcp:8080", "192.168.1.1,udp:53"},
		},
		{
			output:   emptyOutput,
			expected: []string{},
		},
	}

	for i := range testCases {
		fcmd := fakeexec.FakeCmd{
			CombinedOutputScript: []fakeexec.FakeCombinedOutputAction{
				// Success
				func() ([]byte, error) {
					return []byte(testCases[i].output), nil
				},
			},
		}
		fexec := fakeexec.FakeExec{
			CommandScript: []fakeexec.FakeCommandAction{
				func(cmd string, args ...string) exec.Cmd {
					return fakeexec.InitFakeCmd(&fcmd, cmd, args...)
				},
			},
		}
		runner := New(&fexec)
		// Success
		entries, err := runner.ListEntries("foobar")
		if err != nil {
			t.Errorf("expected success, got: %v", err)
		}
		if fcmd.CombinedOutputCalls != 1 {
			t.Errorf("expected 1 CombinedOutput() calls, got: %d", fcmd.CombinedOutputCalls)
		}
		if !sets.NewString(fcmd.CombinedOutputLog[0]...).HasAll("ipset", "list", "foobar") {
			t.Errorf("wrong CombinedOutput() log, got: %s", fcmd.CombinedOutputLog[0])
		}
		if len(entries) != len(testCases[i].expected) {
			t.Errorf("expected %d ipset entries, got: %d", len(testCases[i].expected), len(entries))
		}
		if !reflect.DeepEqual(entries, testCases[i].expected) {
			t.Errorf("expected entries: %v, got: %v", testCases[i].expected, entries)
		}
	}
}

func TestListSets(t *testing.T) {
	output := `foo
bar
baz`

	expected := []string{"foo", "bar", "baz"}

	fcmd := fakeexec.FakeCmd{
		CombinedOutputScript: []fakeexec.FakeCombinedOutputAction{
			// Success
			func() ([]byte, error) { return []byte(output), nil },
		},
	}
	fexec := fakeexec.FakeExec{
		CommandScript: []fakeexec.FakeCommandAction{
			func(cmd string, args ...string) exec.Cmd { return fakeexec.InitFakeCmd(&fcmd, cmd, args...) },
		},
	}
	runner := New(&fexec)
	// Success
	list, err := runner.ListSets()
	if err != nil {
		t.Errorf("expected success, got: %v", err)
	}
	if fcmd.CombinedOutputCalls != 1 {
		t.Errorf("expected 1 CombinedOutput() calls, got: %d", fcmd.CombinedOutputCalls)
	}
	if !sets.NewString(fcmd.CombinedOutputLog[0]...).HasAll("ipset", "list", "-n") {
		t.Errorf("wrong CombinedOutput() log, got: %s", fcmd.CombinedOutputLog[0])
	}
	if len(list) != len(expected) {
		t.Errorf("expected %d sets, got: %d", len(expected), len(list))
	}
	if !reflect.DeepEqual(list, expected) {
		t.Errorf("expected sets: %v, got: %v", expected, list)
	}
}

func Test_validIPSetType(t *testing.T) {
	testCases := []struct {
		setType Type
		valid   bool
	}{
		{ // case[0]
			setType: Type("foo"),
			valid:   false,
		},
		{ // case[1]
			setType: HashIPPortNet,
			valid:   true,
		},
		{ // case[2]
			setType: HashIPPort,
			valid:   true,
		},
		{ // case[3]
			setType: HashIPPortIP,
			valid:   true,
		},
		{ // case[4]
			setType: BitmapPort,
			valid:   true,
		},
		{ // case[5]
			setType: Type(""),
			valid:   false,
		},
	}
	for i := range testCases {
		valid := validateIPSetType(testCases[i].setType)
		if valid != testCases[i].valid {
			t.Errorf("case [%d]: unexpected mismatch, expect valid[%v], got valid[%v]", i, testCases[i].valid, valid)
		}
	}
}

func Test_validatePortRange(t *testing.T) {
	testCases := []struct {
		portRange string
		valid     bool
		desc      string
	}{
		{ // case[0]
			portRange: "a-b",
			valid:     false,
			desc:      "invalid port number",
		},
		{ // case[1]
			portRange: "1-2",
			valid:     true,
			desc:      "valid",
		},
		{ // case[2]
			portRange: "90-1",
			valid:     true,
			desc:      "ipset util can accept the input of begin port number can be less than end port number",
		},
		{ // case[3]
			portRange: DefaultPortRange,
			valid:     true,
			desc:      "default port range is valid, of course",
		},
		{ // case[4]
			portRange: "12",
			valid:     false,
			desc:      "a single number is invalid",
		},
		{ // case[5]
			portRange: "1-",
			valid:     false,
			desc:      "should specify end port",
		},
		{ // case[6]
			portRange: "-100",
			valid:     false,
			desc:      "should specify begin port",
		},
		{ // case[7]
			portRange: "1:100",
			valid:     false,
			desc:      "delimiter should be -",
		},
		{ // case[8]
			portRange: "1~100",
			valid:     false,
			desc:      "delimiter should be -",
		},
		{ // case[9]
			portRange: "1,100",
			valid:     false,
			desc:      "delimiter should be -",
		},
		{ // case[10]
			portRange: "100-100",
			valid:     true,
			desc:      "begin port number can be equal to end port number",
		},
		{ // case[11]
			portRange: "",
			valid:     false,
			desc:      "empty string is invalid",
		},
		{ // case[12]
			portRange: "-1-12",
			valid:     false,
			desc:      "port number can not be negative value",
		},
		{ // case[13]
			portRange: "-1--8",
			valid:     false,
			desc:      "port number can not be negative value",
		},
	}
	for i := range testCases {
		valid, _ := validatePortRange(testCases[i].portRange)
		if valid != testCases[i].valid {
			t.Errorf("case [%d]: unexpected mismatch, expect valid[%v], got valid[%v], desc: %s", i, testCases[i].valid, valid, testCases[i].desc)
		}
	}
}

func Test_validateFamily(t *testing.T) {
	testCases := []struct {
		family string
		valid  bool
	}{
		{ // case[0]
			family: "foo",
			valid:  false,
		},
		{ // case[1]
			family: ProtocolFamilyIPV4,
			valid:  true,
		},
		{ // case[2]
			family: ProtocolFamilyIPV6,
			valid:  true,
		},
		{ // case[3]
			family: "ipv4",
			valid:  false,
		},
		{ // case[4]
			family: "ipv6",
			valid:  false,
		},
		{ // case[5]
			family: "tcp",
			valid:  false,
		},
		{ // case[6]
			family: "udp",
			valid:  false,
		},
		{ // case[7]
			family: "",
			valid:  false,
		},
	}
	for i := range testCases {
		valid := validateHashFamily(testCases[i].family)
		if valid != testCases[i].valid {
			t.Errorf("case [%d]: unexpected mismatch, expect valid[%v], got valid[%v]", i, testCases[i].valid, valid)
		}
	}
}
