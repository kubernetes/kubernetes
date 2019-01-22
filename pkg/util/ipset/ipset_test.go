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

var testCases = []struct {
	entry                *Entry
	set                  *IPSet
	addCombinedOutputLog [][]string
	delCombinedOutputLog []string
}{
	{ // case 0
		entry: &Entry{
			IP:       "192.168.1.1",
			Port:     53,
			Protocol: ProtocolUDP,
			SetType:  HashIPPort,
		},
		set: &IPSet{
			Name: "ZERO",
		},
		addCombinedOutputLog: [][]string{
			{"ipset", "add", "ZERO", "192.168.1.1,udp:53"},
			{"ipset", "add", "ZERO", "192.168.1.1,udp:53", "-exist"},
		},
		delCombinedOutputLog: []string{"ipset", "del", "ZERO", "192.168.1.1,udp:53"},
	},
	{ // case 1
		entry: &Entry{
			IP:       "192.168.1.2",
			Port:     80,
			Protocol: ProtocolTCP,
			SetType:  HashIPPort,
		},
		set: &IPSet{
			Name: "UN",
		},
		addCombinedOutputLog: [][]string{
			{"ipset", "add", "UN", "192.168.1.2,tcp:80"},
			{"ipset", "add", "UN", "192.168.1.2,tcp:80", "-exist"},
		},
		delCombinedOutputLog: []string{"ipset", "del", "UN", "192.168.1.2,tcp:80"},
	},
	{ // case 2
		entry: &Entry{
			IP:       "192.168.1.3",
			Port:     53,
			Protocol: ProtocolUDP,
			SetType:  HashIPPortIP,
			IP2:      "10.20.30.1",
		},
		set: &IPSet{
			Name: "DEUX",
		},
		addCombinedOutputLog: [][]string{
			{"ipset", "add", "DEUX", "192.168.1.3,udp:53,10.20.30.1"},
			{"ipset", "add", "DEUX", "192.168.1.3,udp:53,10.20.30.1", "-exist"},
		},
		delCombinedOutputLog: []string{"ipset", "del", "DEUX", "192.168.1.3,udp:53,10.20.30.1"},
	},
	{ // case 3
		entry: &Entry{
			IP:       "192.168.1.4",
			Port:     80,
			Protocol: ProtocolTCP,
			SetType:  HashIPPortIP,
			IP2:      "10.20.30.2",
		},
		set: &IPSet{
			Name: "TROIS",
		},
		addCombinedOutputLog: [][]string{
			{"ipset", "add", "TROIS", "192.168.1.4,tcp:80,10.20.30.2"},
			{"ipset", "add", "TROIS", "192.168.1.4,tcp:80,10.20.30.2", "-exist"},
		},
		delCombinedOutputLog: []string{"ipset", "del", "TROIS", "192.168.1.4,tcp:80,10.20.30.2"},
	},
	{ // case 4
		entry: &Entry{
			IP:       "192.168.1.5",
			Port:     53,
			Protocol: ProtocolUDP,
			SetType:  HashIPPortNet,
			Net:      "10.20.30.0/24",
		},
		set: &IPSet{
			Name: "QUATRE",
		},
		addCombinedOutputLog: [][]string{
			{"ipset", "add", "QUATRE", "192.168.1.5,udp:53,10.20.30.0/24"},
			{"ipset", "add", "QUATRE", "192.168.1.5,udp:53,10.20.30.0/24", "-exist"},
		},
		delCombinedOutputLog: []string{"ipset", "del", "QUATRE", "192.168.1.5,udp:53,10.20.30.0/24"},
	},
	{ // case 5
		entry: &Entry{
			IP:       "192.168.1.6",
			Port:     80,
			Protocol: ProtocolTCP,
			SetType:  HashIPPortNet,
			Net:      "10.20.40.0/24",
		},
		set: &IPSet{
			Name: "CINQ",
		},
		addCombinedOutputLog: [][]string{
			{"ipset", "add", "CINQ", "192.168.1.6,tcp:80,10.20.40.0/24"},
			{"ipset", "add", "CINQ", "192.168.1.6,tcp:80,10.20.40.0/24", "-exist"},
		},
		delCombinedOutputLog: []string{"ipset", "del", "CINQ", "192.168.1.6,tcp:80,10.20.40.0/24"},
	},
	{ // case 6
		entry: &Entry{
			Port:     80,
			Protocol: ProtocolTCP,
			SetType:  BitmapPort,
		},
		set: &IPSet{
			Name: "SIX",
		},
		addCombinedOutputLog: [][]string{
			{"ipset", "add", "SIX", "80"},
			{"ipset", "add", "SIX", "80", "-exist"},
		},
		delCombinedOutputLog: []string{"ipset", "del", "SIX", "80"},
	},
	{ // case 7
		entry: &Entry{
			IP:       "192.168.1.2",
			Port:     80,
			Protocol: ProtocolSCTP,
			SetType:  HashIPPort,
		},
		set: &IPSet{
			Name: "SETTE",
		},
		addCombinedOutputLog: [][]string{
			{"ipset", "add", "SETTE", "192.168.1.2,sctp:80"},
			{"ipset", "add", "SETTE", "192.168.1.2,sctp:80", "-exist"},
		},
		delCombinedOutputLog: []string{"ipset", "del", "SETTE", "192.168.1.2,sctp:80"},
	},
}

func TestAddEntry(t *testing.T) {
	for i := range testCases {
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
		err := runner.AddEntry(testCases[i].entry.String(), testCases[i].set, false)
		if err != nil {
			t.Errorf("expected success, got %v", err)
		}
		if fcmd.CombinedOutputCalls != 1 {
			t.Errorf("expected 1 CombinedOutput() calls, got %d", fcmd.CombinedOutputCalls)
		}
		if !sets.NewString(fcmd.CombinedOutputLog[0]...).HasAll(testCases[i].addCombinedOutputLog[0]...) {
			t.Errorf("wrong CombinedOutput() log, got %s", fcmd.CombinedOutputLog[0])
		}
		// Create with ignoreExistErr = true, expect success
		err = runner.AddEntry(testCases[i].entry.String(), testCases[i].set, true)
		if err != nil {
			t.Errorf("expected success, got %v", err)
		}
		if fcmd.CombinedOutputCalls != 2 {
			t.Errorf("expected 3 CombinedOutput() calls, got %d", fcmd.CombinedOutputCalls)
		}
		if !sets.NewString(fcmd.CombinedOutputLog[1]...).HasAll(testCases[i].addCombinedOutputLog[1]...) {
			t.Errorf("wrong CombinedOutput() log, got %s", fcmd.CombinedOutputLog[1])
		}
		// Create with ignoreExistErr = false, expect failure
		err = runner.AddEntry(testCases[i].entry.String(), testCases[i].set, false)
		if err == nil {
			t.Errorf("expected failure, got nil")
		}
	}
}

func TestDelEntry(t *testing.T) {
	for i := range testCases {
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

		err := runner.DelEntry(testCases[i].entry.String(), testCases[i].set.Name)
		if err != nil {
			t.Errorf("expected success, got %v", err)
		}
		if fcmd.CombinedOutputCalls != 1 {
			t.Errorf("expected 1 CombinedOutput() calls, got %d", fcmd.CombinedOutputCalls)
		}
		if !sets.NewString(fcmd.CombinedOutputLog[0]...).HasAll(testCases[i].delCombinedOutputLog...) {
			t.Errorf("wrong CombinedOutput() log, got %s", fcmd.CombinedOutputLog[0])
		}
		err = runner.DelEntry(testCases[i].entry.String(), testCases[i].set.Name)
		if err == nil {
			t.Errorf("expected failure, got nil")
		}
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
		valid := validatePortRange(testCases[i].portRange)
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
		{ // case[8]
			family: "sctp",
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

func Test_validateProtocol(t *testing.T) {
	testCases := []struct {
		protocol string
		valid    bool
		desc     string
	}{
		{ // case[0]
			protocol: "foo",
			valid:    false,
		},
		{ // case[1]
			protocol: ProtocolTCP,
			valid:    true,
		},
		{ // case[2]
			protocol: ProtocolUDP,
			valid:    true,
		},
		{ // case[3]
			protocol: "ipv4",
			valid:    false,
		},
		{ // case[4]
			protocol: "ipv6",
			valid:    false,
		},
		{ // case[5]
			protocol: "TCP",
			valid:    false,
			desc:     "should be low case",
		},
		{ // case[6]
			protocol: "UDP",
			valid:    false,
			desc:     "should be low case",
		},
		{ // case[7]
			protocol: "",
			valid:    false,
		},
		{ // case[8]
			protocol: ProtocolSCTP,
			valid:    true,
		},
	}
	for i := range testCases {
		valid := validateProtocol(testCases[i].protocol)
		if valid != testCases[i].valid {
			t.Errorf("case [%d]: unexpected mismatch, expect valid[%v], got valid[%v], desc: %s", i, testCases[i].valid, valid, testCases[i].desc)
		}
	}
}

func TestValidateIPSet(t *testing.T) {
	testCases := []struct {
		ipset *IPSet
		valid bool
		desc  string
	}{
		{ // case[0]
			ipset: &IPSet{
				Name:       "test",
				SetType:    HashIPPort,
				HashFamily: ProtocolFamilyIPV4,
				HashSize:   1024,
				MaxElem:    1024,
			},
			valid: true,
		},
		{ // case[1]
			ipset: &IPSet{
				Name:       "SET",
				SetType:    BitmapPort,
				HashFamily: ProtocolFamilyIPV6,
				HashSize:   65535,
				MaxElem:    2048,
				PortRange:  DefaultPortRange,
			},
			valid: true,
		},
		{ // case[2]
			ipset: &IPSet{
				Name:       "foo",
				SetType:    BitmapPort,
				HashFamily: ProtocolFamilyIPV6,
				HashSize:   65535,
				MaxElem:    2048,
			},
			valid: false,
			desc:  "should specify right port range for bitmap type set",
		},
		{ // case[3]
			ipset: &IPSet{
				Name:       "bar",
				SetType:    BitmapPort,
				HashFamily: ProtocolFamilyIPV6,
				HashSize:   0,
				MaxElem:    2048,
			},
			valid: false,
			desc:  "wrong hash size number",
		},
		{ // case[4]
			ipset: &IPSet{
				Name:       "baz",
				SetType:    BitmapPort,
				HashFamily: ProtocolFamilyIPV6,
				HashSize:   1024,
				MaxElem:    -1,
			},
			valid: false,
			desc:  "wrong hash max elem number",
		},
		{ // case[5]
			ipset: &IPSet{
				Name:       "baz",
				SetType:    HashIPPortNet,
				HashFamily: "ip",
				HashSize:   1024,
				MaxElem:    1024,
			},
			valid: false,
			desc:  "wrong protocol",
		},
		{ // case[6]
			ipset: &IPSet{
				Name:       "foo-bar",
				SetType:    "xxx",
				HashFamily: ProtocolFamilyIPV4,
				HashSize:   1024,
				MaxElem:    1024,
			},
			valid: false,
			desc:  "wrong set type",
		},
	}
	for i := range testCases {
		valid := testCases[i].ipset.Validate()
		if valid != testCases[i].valid {
			t.Errorf("case [%d]: unexpected mismatch, expect valid[%v], got valid[%v], desc: %s", i, testCases[i].valid, valid, testCases[i].desc)
		}
	}
}

func Test_setIPSetDefaults(t *testing.T) {
	testCases := []struct {
		name   string
		set    *IPSet
		expect *IPSet
	}{
		{
			name: "test all the IPSet fields not present",
			set: &IPSet{
				Name: "test1",
			},
			expect: &IPSet{
				Name:       "test1",
				SetType:    HashIPPort,
				HashFamily: ProtocolFamilyIPV4,
				HashSize:   1024,
				MaxElem:    65536,
				PortRange:  DefaultPortRange,
			},
		},
		{
			name: "test all the IPSet fields present",
			set: &IPSet{
				Name:       "test2",
				SetType:    BitmapPort,
				HashFamily: ProtocolFamilyIPV6,
				HashSize:   65535,
				MaxElem:    2048,
				PortRange:  DefaultPortRange,
			},
			expect: &IPSet{
				Name:       "test2",
				SetType:    BitmapPort,
				HashFamily: ProtocolFamilyIPV6,
				HashSize:   65535,
				MaxElem:    2048,
				PortRange:  DefaultPortRange,
			},
		},
		{
			name: "test part of the IPSet fields present",
			set: &IPSet{
				Name:       "test3",
				SetType:    BitmapPort,
				HashFamily: ProtocolFamilyIPV6,
				HashSize:   65535,
			},
			expect: &IPSet{
				Name:       "test3",
				SetType:    BitmapPort,
				HashFamily: ProtocolFamilyIPV6,
				HashSize:   65535,
				MaxElem:    65536,
				PortRange:  DefaultPortRange,
			},
		},
	}

	for _, test := range testCases {
		t.Run(test.name, func(t *testing.T) {
			test.set.setIPSetDefaults()
			if !reflect.DeepEqual(test.set, test.expect) {
				t.Errorf("expected ipset struct: %v, got ipset struct: %v", test.expect, test.set)
			}
		})
	}
}

func Test_checkIPandProtocol(t *testing.T) {
	testset := &IPSet{
		Name:       "test1",
		SetType:    HashIPPort,
		HashFamily: ProtocolFamilyIPV4,
		HashSize:   1024,
		MaxElem:    65536,
		PortRange:  DefaultPortRange,
	}

	testCases := []struct {
		name  string
		entry *Entry
		valid bool
	}{
		{
			name: "valid IP with ProtocolTCP",
			entry: &Entry{
				SetType:  HashIPPort,
				IP:       "1.2.3.4",
				Protocol: ProtocolTCP,
				Port:     8080,
			},
			valid: true,
		},
		{
			name: "valid IP with ProtocolUDP",
			entry: &Entry{
				SetType:  HashIPPort,
				IP:       "1.2.3.4",
				Protocol: ProtocolUDP,
				Port:     8080,
			},
			valid: true,
		},
		{
			name: "valid IP with nil Protocol",
			entry: &Entry{
				SetType: HashIPPort,
				IP:      "1.2.3.4",
				Port:    8080,
			},
			valid: true,
		},
		{
			name: "valid IP with invalid Protocol",
			entry: &Entry{
				SetType:  HashIPPort,
				IP:       "1.2.3.4",
				Protocol: "invalidProtocol",
				Port:     8080,
			},
			valid: false,
		},
		{
			name: "invalid IP with ProtocolTCP",
			entry: &Entry{
				SetType:  HashIPPort,
				IP:       "1.2.3.423",
				Protocol: ProtocolTCP,
				Port:     8080,
			},
			valid: false,
		},
	}

	for _, test := range testCases {
		t.Run(test.name, func(t *testing.T) {
			result := test.entry.checkIPandProtocol(testset)
			if result != test.valid {
				t.Errorf("expected valid: %v, got valid: %v", test.valid, result)
			}
		})
	}
}

func Test_parsePortRange(t *testing.T) {
	testCases := []struct {
		portRange string
		expectErr bool
		beginPort int
		endPort   int
		desc      string
	}{
		{ // case[0]
			portRange: "1-100",
			expectErr: false,
			beginPort: 1,
			endPort:   100,
		},
		{ // case[1]
			portRange: "0-0",
			expectErr: false,
			beginPort: 0,
			endPort:   0,
		},
		{ // case[2]
			portRange: "100-10",
			expectErr: false,
			beginPort: 10,
			endPort:   100,
		},
		{ // case[3]
			portRange: "1024",
			expectErr: true,
			desc:      "single port number is not allowed",
		},
		{ // case[4]
			portRange: DefaultPortRange,
			expectErr: false,
			beginPort: 0,
			endPort:   65535,
		},
		{ // case[5]
			portRange: "1-",
			expectErr: true,
			desc:      "should specify end port",
		},
		{ // case[6]
			portRange: "-100",
			expectErr: true,
			desc:      "should specify begin port",
		},
		{ // case[7]
			portRange: "1:100",
			expectErr: true,
			desc:      "delimiter should be -",
		},
		{ // case[8]
			portRange: "1~100",
			expectErr: true,
			desc:      "delimiter should be -",
		},
		{ // case[9]
			portRange: "1,100",
			expectErr: true,
			desc:      "delimiter should be -",
		},
		{ // case[10]
			portRange: "100-100",
			expectErr: false,
			desc:      "begin port number can be equal to end port number",
			beginPort: 100,
			endPort:   100,
		},
		{ // case[11]
			portRange: "",
			expectErr: false,
			desc:      "empty string indicates default port range",
			beginPort: 0,
			endPort:   65535,
		},
		{ // case[12]
			portRange: "-1-12",
			expectErr: true,
			desc:      "port number can not be negative value",
		},
		{ // case[13]
			portRange: "-1--8",
			expectErr: true,
			desc:      "port number can not be negative value",
		},
	}
	for i := range testCases {
		begin, end, err := parsePortRange(testCases[i].portRange)
		if err != nil {
			if !testCases[i].expectErr {
				t.Errorf("case [%d]: unexpected err: %v, desc: %s", i, err, testCases[i].desc)
			}
			continue
		}
		if begin != testCases[i].beginPort || end != testCases[i].endPort {
			t.Errorf("case [%d]: unexpected mismatch [beginPort, endPort] pair, expect [%d, %d], got [%d, %d], desc: %s", i, testCases[i].beginPort, testCases[i].endPort, begin, end, testCases[i].desc)
		}
	}
}

// This is a coarse test, but it offers some modicum of confidence as the code is evolved.
func TestValidateEntry(t *testing.T) {
	testCases := []struct {
		entry *Entry
		set   *IPSet
		valid bool
		desc  string
	}{
		{ // case[0]
			entry: &Entry{
				SetType: BitmapPort,
			},
			set: &IPSet{
				PortRange: DefaultPortRange,
			},
			valid: true,
			desc:  "port number can be empty, default is 0. And port number is in the range of its ipset's port range",
		},
		{ // case[1]
			entry: &Entry{
				SetType: BitmapPort,
				Port:    0,
			},
			set: &IPSet{
				PortRange: DefaultPortRange,
			},
			valid: true,
			desc:  "port number can be 0. And port number is in the range of its ipset's port range",
		},
		{ // case[2]
			entry: &Entry{
				SetType: BitmapPort,
				Port:    -1,
			},
			valid: false,
			desc:  "port number can not be negative value",
		},
		{ // case[3]
			entry: &Entry{
				SetType: BitmapPort,
				Port:    1080,
			},
			set: &IPSet{
				Name:      "baz",
				PortRange: DefaultPortRange,
			},
			desc:  "port number is in the range of its ipset's port range",
			valid: true,
		},
		{ // case[4]
			entry: &Entry{
				SetType: BitmapPort,
				Port:    1080,
			},
			set: &IPSet{
				Name:      "foo",
				PortRange: "0-1079",
			},
			desc:  "port number is NOT in the range of its ipset's port range",
			valid: false,
		},
		{ // case[5]
			entry: &Entry{
				SetType:  HashIPPort,
				IP:       "1.2.3.4",
				Protocol: ProtocolTCP,
				Port:     8080,
			},
			set: &IPSet{
				Name: "bar",
			},
			valid: true,
		},
		{ // case[6]
			entry: &Entry{
				SetType:  HashIPPort,
				IP:       "1.2.3.4",
				Protocol: ProtocolUDP,
				Port:     0,
			},
			set: &IPSet{
				Name: "bar",
			},
			valid: true,
		},
		{ // case[7]
			entry: &Entry{
				SetType:  HashIPPort,
				IP:       "FE80:0000:0000:0000:0202:B3FF:FE1E:8329",
				Protocol: ProtocolTCP,
				Port:     1111,
			},
			set: &IPSet{
				Name: "ipv6",
			},
			valid: true,
		},
		{ // case[8]
			entry: &Entry{
				SetType:  HashIPPort,
				IP:       "",
				Protocol: ProtocolTCP,
				Port:     1234,
			},
			set: &IPSet{
				Name: "empty-ip",
			},
			valid: false,
		},
		{ // case[9]
			entry: &Entry{
				SetType:  HashIPPort,
				IP:       "1-2-3-4",
				Protocol: ProtocolTCP,
				Port:     8900,
			},
			set: &IPSet{
				Name: "bad-ip",
			},
			valid: false,
		},
		{ // case[10]
			entry: &Entry{
				SetType:  HashIPPort,
				IP:       "10.20.30.40",
				Protocol: "",
				Port:     8090,
			},
			set: &IPSet{
				Name: "empty-protocol",
			},
			valid: true,
		},
		{ // case[11]
			entry: &Entry{
				SetType:  HashIPPort,
				IP:       "10.20.30.40",
				Protocol: "ICMP",
				Port:     8090,
			},
			set: &IPSet{
				Name: "unsupported-protocol",
			},
			valid: false,
		},
		{ // case[11]
			entry: &Entry{
				SetType:  HashIPPort,
				IP:       "10.20.30.40",
				Protocol: "ICMP",
				Port:     -1,
			},
			set: &IPSet{
				// TODO: set name string with white space?
				Name: "negative-port-number",
			},
			valid: false,
		},
		{ // case[12]
			entry: &Entry{
				SetType:  HashIPPortIP,
				IP:       "10.20.30.40",
				Protocol: ProtocolUDP,
				Port:     53,
				IP2:      "10.20.30.40",
			},
			set: &IPSet{
				Name: "LOOP-BACK",
			},
			valid: true,
		},
		{ // case[13]
			entry: &Entry{
				SetType:  HashIPPortIP,
				IP:       "10.20.30.40",
				Protocol: ProtocolUDP,
				Port:     53,
				IP2:      "",
			},
			set: &IPSet{
				Name: "empty IP2",
			},
			valid: false,
		},
		{ // case[14]
			entry: &Entry{
				SetType:  HashIPPortIP,
				IP:       "10.20.30.40",
				Protocol: ProtocolUDP,
				Port:     53,
				IP2:      "foo",
			},
			set: &IPSet{
				Name: "invalid IP2",
			},
			valid: false,
		},
		{ // case[15]
			entry: &Entry{
				SetType:  HashIPPortIP,
				IP:       "10.20.30.40",
				Protocol: ProtocolTCP,
				Port:     0,
				IP2:      "1.2.3.4",
			},
			set: &IPSet{
				Name: "zero port",
			},
			valid: true,
		},
		{ // case[16]
			entry: &Entry{
				SetType:  HashIPPortIP,
				IP:       "10::40",
				Protocol: ProtocolTCP,
				Port:     10000,
				IP2:      "1::4",
			},
			set: &IPSet{
				Name: "IPV6",
				// TODO: check set's hash family
			},
			valid: true,
		},
		{ // case[17]
			entry: &Entry{
				SetType:  HashIPPortIP,
				IP:       "",
				Protocol: ProtocolTCP,
				Port:     1234,
				IP2:      "1.2.3.4",
			},
			set: &IPSet{
				Name: "empty-ip",
			},
			valid: false,
		},
		{ // case[18]
			entry: &Entry{
				SetType:  HashIPPortIP,
				IP:       "1-2-3-4",
				Protocol: ProtocolTCP,
				Port:     8900,
				IP2:      "10.20.30.41",
			},
			set: &IPSet{
				Name: "bad-ip",
			},
			valid: false,
		},
		{ // case[19]
			entry: &Entry{
				SetType:  HashIPPortIP,
				IP:       "10.20.30.40",
				Protocol: ProtocolSCTP,
				Port:     8090,
				IP2:      "10.20.30.41",
			},
			set: &IPSet{
				Name: "sctp",
			},
			valid: true,
		},
		{ // case[20]
			entry: &Entry{
				SetType:  HashIPPortIP,
				IP:       "10.20.30.40",
				Protocol: "ICMP",
				Port:     -1,
				IP2:      "100.200.30.41",
			},
			set: &IPSet{
				Name: "negative-port-number",
			},
			valid: false,
		},
		{ // case[21]
			entry: &Entry{
				SetType:  HashIPPortNet,
				IP:       "10.20.30.40",
				Protocol: ProtocolTCP,
				Port:     53,
				// TODO: CIDR /32 may not be valid
				Net: "10.20.30.0/24",
			},
			set: &IPSet{
				Name: "abc",
			},
			valid: true,
		},
		{ // case[22]
			entry: &Entry{
				SetType:  HashIPPortNet,
				IP:       "11.21.31.41",
				Protocol: ProtocolUDP,
				Port:     1122,
				Net:      "",
			},
			set: &IPSet{
				Name: "empty Net",
			},
			valid: false,
		},
		{ // case[23]
			entry: &Entry{
				SetType:  HashIPPortNet,
				IP:       "10.20.30.40",
				Protocol: ProtocolUDP,
				Port:     8080,
				Net:      "x-y-z-w",
			},
			set: &IPSet{
				Name: "invalid Net",
			},
			valid: false,
		},
		{ // case[24]
			entry: &Entry{
				SetType:  HashIPPortNet,
				IP:       "10.20.30.40",
				Protocol: ProtocolTCP,
				Port:     0,
				Net:      "10.1.0.0/16",
			},
			set: &IPSet{
				Name: "zero port",
			},
			valid: true,
		},
		{ // case[25]
			entry: &Entry{
				SetType:  HashIPPortNet,
				IP:       "10::40",
				Protocol: ProtocolTCP,
				Port:     80,
				Net:      "2001:db8::/32",
			},
			set: &IPSet{
				Name: "IPV6",
				// TODO: check set's hash family
			},
			valid: true,
		},
		{ // case[26]
			entry: &Entry{
				SetType:  HashIPPortNet,
				IP:       "",
				Protocol: ProtocolTCP,
				Port:     1234,
				Net:      "1.2.3.4/22",
			},
			set: &IPSet{
				Name: "empty-ip",
			},
			valid: false,
		},
		{ // case[27]
			entry: &Entry{
				SetType:  HashIPPortNet,
				IP:       "1-2-3-4",
				Protocol: ProtocolTCP,
				Port:     8900,
				Net:      "10.20.30.41/31",
			},
			set: &IPSet{
				Name: "bad-ip",
			},
			valid: false,
		},
		{ // case[28]
			entry: &Entry{
				SetType:  HashIPPortIP,
				IP:       "10.20.30.40",
				Protocol: "FOO",
				Port:     8090,
				IP2:      "10.20.30.0/10",
			},
			set: &IPSet{
				Name: "unsupported-protocol",
			},
			valid: false,
		},
		{ // case[29]
			entry: &Entry{
				SetType:  HashIPPortIP,
				IP:       "10.20.30.40",
				Protocol: ProtocolUDP,
				Port:     -1,
				IP2:      "100.200.30.0/12",
			},
			set: &IPSet{
				Name: "negative-port-number",
			},
			valid: false,
		},
	}
	for i := range testCases {
		valid := testCases[i].entry.Validate(testCases[i].set)
		if valid != testCases[i].valid {
			t.Errorf("case [%d]: unexpected mismatch, expect valid[%v], got valid[%v], desc: %s", i, testCases[i].valid, valid, testCases[i].entry)
		}
	}
}

func TestEntryString(t *testing.T) {
	testCases := []struct {
		name   string
		entry  *Entry
		expect string
	}{
		{
			name: "test when SetType is HashIPPort",
			entry: &Entry{
				SetType:  HashIPPort,
				IP:       "1.2.3.4",
				Protocol: ProtocolTCP,
				Port:     8080,
			},
			expect: "1.2.3.4,tcp:8080",
		},
		{
			name: "test when SetType is HashIPPortIP",
			entry: &Entry{
				SetType:  HashIPPortIP,
				IP:       "1.2.3.8",
				Protocol: ProtocolUDP,
				Port:     8081,
				IP2:      "1.2.3.8",
			},
			expect: "1.2.3.8,udp:8081,1.2.3.8",
		},
		{
			name: "test when SetType is HashIPPortNet",
			entry: &Entry{
				SetType:  HashIPPortNet,
				IP:       "192.168.1.2",
				Protocol: ProtocolUDP,
				Port:     80,
				Net:      "10.0.1.0/24",
			},
			expect: "192.168.1.2,udp:80,10.0.1.0/24",
		},
		{
			name: "test when SetType is BitmapPort",
			entry: &Entry{
				SetType: BitmapPort,
				Port:    80,
			},
			expect: "80",
		},
		{
			name: "test when SetType is unknown",
			entry: &Entry{
				SetType:  "unknown",
				IP:       "192.168.1.2",
				Protocol: ProtocolUDP,
				Port:     80,
				Net:      "10.0.1.0/24",
			},
			expect: "",
		},
	}

	for _, test := range testCases {
		t.Run(test.name, func(t *testing.T) {
			result := test.entry.String()
			if result != test.expect {
				t.Errorf("Unexpected mismatch, expected: %s, got: %s", test.expect, result)
			}
		})
	}
}
