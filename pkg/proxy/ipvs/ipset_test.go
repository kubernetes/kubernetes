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

package ipvs

import (
	"testing"

	utilipset "k8s.io/kubernetes/pkg/util/ipset"
	fakeipset "k8s.io/kubernetes/pkg/util/ipset/testing"
)

func TestCheckIPSetVersion(t *testing.T) {
	testCases := []struct {
		name    string
		vstring string
		valid   bool
	}{
		{"4.0 < 6.0 is invalid", "4.0", false},
		{"5.1 < 6.0 is invalid", "5.1", false},
		{"5.2 < 6.0 is invalid", "5.1.2", false},
		{"7 is not a valid version", "7", false},
		{"6.0 is valid since >= 6.0", "6.0", true},
		{"6.1 is valid since >= 6.0", "6.1", true},
		{"6.19 is valid since >= 6.0", "6.19", true},
		{"7.0 is valid since >= 6.0", "7.0", true},
		{"8.1.2 is valid since >= 6.0", "8.1.2", true},
		{"9.3.4.0 is valid since >= 6.0", "9.3.4.0", true},
		{"not a valid semantic version", "total junk", false},
	}

	for _, testCase := range testCases {
		t.Run(testCase.name, func(t *testing.T) {
			valid := checkMinVersion(testCase.vstring)
			if testCase.valid != valid {
				t.Errorf("Expected result: %v, Got result: %v", testCase.valid, valid)
			}
		})
	}
}

const testIPSetVersion = "v6.19"

func TestSyncIPSetEntries(t *testing.T) {
	testCases := []struct {
		name            string
		set             *utilipset.IPSet
		setType         utilipset.Type
		ipv6            bool
		activeEntries   []*utilipset.Entry
		currentEntries  []*utilipset.Entry
		expectedEntries []*utilipset.Entry
	}{
		{
			name: "normal ipset sync",
			set: &utilipset.IPSet{
				Name: "foo",
			},
			setType:         utilipset.HashIPPort,
			ipv6:            false,
			activeEntries:   []*utilipset.Entry{{IP: "172.17.0.4", Port: 80, Protocol: utilipset.ProtocolTCP}},
			currentEntries:  nil,
			expectedEntries: []*utilipset.Entry{{IP: "172.17.0.4", Port: 80, Protocol: utilipset.ProtocolTCP}},
		},
		{
			name: "ipset IPv6 sync with no new entries",
			set: &utilipset.IPSet{
				Name: "abz",
			},
			setType:         utilipset.HashIPPort,
			ipv6:            true,
			activeEntries:   []*utilipset.Entry{{IP: "FE80::0202:B3FF:FE1E:8329", Port: 80, Protocol: utilipset.ProtocolTCP}},
			currentEntries:  []*utilipset.Entry{{IP: "FE80::0202:B3FF:FE1E:8329", Port: 80, Protocol: utilipset.ProtocolTCP}},
			expectedEntries: []*utilipset.Entry{{IP: "FE80::0202:B3FF:FE1E:8329", Port: 80, Protocol: utilipset.ProtocolTCP}},
		},
		{
			name: "ipset sync with updated udp->tcp in hash",
			set: &utilipset.IPSet{
				Name: "bca",
			},
			setType: utilipset.HashIPPort,
			ipv6:    false,
			activeEntries: []*utilipset.Entry{
				{IP: "172.17.0.4", Port: 80, Protocol: utilipset.ProtocolTCP},
				{IP: "172.17.0.5", Port: 80, Protocol: utilipset.ProtocolTCP},
			},
			currentEntries: []*utilipset.Entry{{IP: "172.17.0.5", Port: 80, Protocol: utilipset.ProtocolUDP}},
			expectedEntries: []*utilipset.Entry{
				{IP: "172.17.0.4", Port: 80, Protocol: utilipset.ProtocolTCP},
				{IP: "172.17.0.5", Port: 80, Protocol: utilipset.ProtocolTCP},
			},
		},
		{
			name: "ipset sync no updates required",
			set: &utilipset.IPSet{
				Name: "bar",
			},
			setType:         utilipset.HashIPPortIP,
			ipv6:            false,
			activeEntries:   []*utilipset.Entry{{IP: "172.17.0.4", Port: 80, Protocol: utilipset.ProtocolTCP, IP2: "172.17.0.4"}},
			currentEntries:  []*utilipset.Entry{{IP: "172.17.0.4", Port: 80, Protocol: utilipset.ProtocolTCP, IP2: "172.17.0.4"}},
			expectedEntries: []*utilipset.Entry{{IP: "172.17.0.4", Port: 80, Protocol: utilipset.ProtocolTCP, IP2: "172.17.0.4"}},
		},
		{
			name: "ipset IPv6 sync, delete and add new entry",
			set: &utilipset.IPSet{
				Name: "baz",
			},
			setType:         utilipset.HashIPPortIP,
			ipv6:            true,
			activeEntries:   []*utilipset.Entry{{IP: "FE80:0000:0000:0000:0202:B3FF:FE1E:8329", Port: 8080, Protocol: utilipset.ProtocolTCP, IP2: "FE80:0000:0000:0000:0202:B3FF:FE1E:8329"}},
			currentEntries:  []*utilipset.Entry{{IP: "1111:0000:0000:0000:0202:B3FF:FE1E:8329", Port: 8081, Protocol: utilipset.ProtocolTCP, IP2: "1111:0000:0000:0000:0202:B3FF:FE1E:8329"}},
			expectedEntries: []*utilipset.Entry{{IP: "FE80:0000:0000:0000:0202:B3FF:FE1E:8329", Port: 8080, Protocol: utilipset.ProtocolTCP, IP2: "FE80:0000:0000:0000:0202:B3FF:FE1E:8329"}},
		},
		{
			name: "ipset sync, no current entries",
			set: &utilipset.IPSet{
				Name: "NOPE",
			},
			setType: utilipset.HashIPPortIP,
			ipv6:    false,
			activeEntries: []*utilipset.Entry{
				{IP: "172.17.0.4", Port: 80, Protocol: utilipset.ProtocolTCP, IP2: "172.17.0.9"},
				{IP: "172.17.0.5", Port: 80, Protocol: utilipset.ProtocolTCP, IP2: "172.17.0.10"},
			},
			currentEntries: nil,
			expectedEntries: []*utilipset.Entry{
				{IP: "172.17.0.4", Port: 80, Protocol: utilipset.ProtocolTCP, IP2: "172.17.0.9"},
				{IP: "172.17.0.5", Port: 80, Protocol: utilipset.ProtocolTCP, IP2: "172.17.0.10"},
			},
		},
		{
			name: "ipset sync, no current entries with /16 subnet",
			set: &utilipset.IPSet{
				Name: "ABC-DEF",
			},
			setType: utilipset.HashIPPortNet,
			ipv6:    false,
			activeEntries: []*utilipset.Entry{
				{IP: "172.17.0.4", Port: 80, Protocol: utilipset.ProtocolTCP, Net: "172.17.0.0/16"},
				{IP: "172.17.0.5", Port: 80, Protocol: utilipset.ProtocolTCP, Net: "172.17.0.0/16"},
			},
			currentEntries: nil,
			expectedEntries: []*utilipset.Entry{
				{IP: "172.17.0.4", Port: 80, Protocol: utilipset.ProtocolTCP, Net: "172.17.0.0/16"},
				{IP: "172.17.0.5", Port: 80, Protocol: utilipset.ProtocolTCP, Net: "172.17.0.0/16"},
			},
		},
		{
			name: "ipset IPv6 sync, no updates required with /32 subnet",
			set: &utilipset.IPSet{
				Name: "zar",
			},
			setType:         utilipset.HashIPPortNet,
			ipv6:            true,
			activeEntries:   []*utilipset.Entry{{IP: "FE80::8329", Port: 8800, Protocol: utilipset.ProtocolTCP, Net: "2001:db8::/32"}},
			currentEntries:  []*utilipset.Entry{{IP: "FE80::8329", Port: 8800, Protocol: utilipset.ProtocolTCP, Net: "2001:db8::/32"}},
			expectedEntries: []*utilipset.Entry{{IP: "FE80::8329", Port: 8800, Protocol: utilipset.ProtocolTCP, Net: "2001:db8::/32"}},
		},
		{
			name: "ipset IPv6 sync, current entries removed",
			set: &utilipset.IPSet{
				Name: "bbb",
			},
			setType:         utilipset.HashIPPortNet,
			ipv6:            true,
			activeEntries:   nil,
			currentEntries:  []*utilipset.Entry{{IP: "FE80::8329", Port: 8801, Protocol: utilipset.ProtocolUDP, Net: "2001:db8::/32"}},
			expectedEntries: nil,
		},
		{
			name: "ipset sync, port entry removed",
			set: &utilipset.IPSet{
				Name: "AAA",
			},
			setType:         utilipset.BitmapPort,
			activeEntries:   nil,
			currentEntries:  []*utilipset.Entry{{Port: 80}},
			expectedEntries: nil,
		},
		{
			name: "ipset sync, remove old and add new",
			set: &utilipset.IPSet{
				Name: "c-c-c",
			},
			setType:         utilipset.BitmapPort,
			activeEntries:   []*utilipset.Entry{{Port: 8080}, {Port: 9090}},
			currentEntries:  []*utilipset.Entry{{Port: 80}},
			expectedEntries: []*utilipset.Entry{{Port: 8080}, {Port: 9090}},
		},
		{
			name: "ipset sync, remove many stale ports",
			set: &utilipset.IPSet{
				Name: "NODE-PORT",
			},
			setType:         utilipset.BitmapPort,
			activeEntries:   []*utilipset.Entry{{Port: 8080}},
			currentEntries:  []*utilipset.Entry{{Port: 80}, {Port: 9090}},
			expectedEntries: []*utilipset.Entry{{Port: 8080}},
		},
		{
			name: "ipset sync, add sctp entry",
			set: &utilipset.IPSet{
				Name: "sctp-1",
			},
			setType:         utilipset.HashIPPort,
			ipv6:            false,
			activeEntries:   []*utilipset.Entry{{IP: "172.17.0.4", Port: 80, Protocol: utilipset.ProtocolSCTP}},
			currentEntries:  nil,
			expectedEntries: []*utilipset.Entry{{IP: "172.17.0.4", Port: 80, Protocol: utilipset.ProtocolSCTP}},
		},
		{
			name: "ipset sync, add IPV6 sctp entry",
			set: &utilipset.IPSet{
				Name: "sctp-2",
			},
			setType:         utilipset.HashIPPort,
			ipv6:            true,
			activeEntries:   []*utilipset.Entry{{IP: "FE80::0202:B3FF:FE1E:8329", Port: 80, Protocol: utilipset.ProtocolSCTP}},
			currentEntries:  []*utilipset.Entry{{IP: "FE80::0202:B3FF:FE1E:8329", Port: 80, Protocol: utilipset.ProtocolSCTP}},
			expectedEntries: []*utilipset.Entry{{IP: "FE80::0202:B3FF:FE1E:8329", Port: 80, Protocol: utilipset.ProtocolSCTP}},
		},
	}

	for _, testCase := range testCases {
		t.Run(testCase.name, func(t *testing.T) {
			testSetInfoList := []ipsetInfo{
				{testCase.set.Name, testCase.setType, "comment-" + testCase.set.Name},
			}
			setManager := newIPSetManager(testCase.ipv6, testSetInfoList, fakeipset.NewFake(testIPSetVersion))
			set := setManager.getIPSet(testCase.set.Name)
			set.SetIPSetDefaults()

			// build background data, current sets and entries
			setManager.handler.CreateSet(set, true)
			for _, entry := range testCase.currentEntries {
				entry.SetType = set.SetType
				setManager.handler.AddEntry(entry.String(), set, true)
			}

			for _, entry := range testCase.activeEntries {
				entry.SetType = set.SetType
				setManager.insertIPSetEntry(set.Name, entry)
			}

			setManager.syncIPSetEntries()
			data, _ := setManager.handler.SaveAllSets()
			t.Logf("all sets are\n%s", data)
			for _, entry := range testCase.expectedEntries {
				entry.SetType = set.SetType
				found, err := setManager.handler.TestEntry(entry.String(), set.Name)
				if err != nil {
					t.Errorf("Unexpected error: %v", err)
				}
				if !found {
					t.Errorf("Unexpected entry %q not found in setManager foo", entry)
				}
			}
		})
	}
}
