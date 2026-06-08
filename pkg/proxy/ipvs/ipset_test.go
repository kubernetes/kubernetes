//go:build linux

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

	utilipset "k8s.io/kubernetes/pkg/proxy/ipvs/ipset"
	fakeipset "k8s.io/kubernetes/pkg/proxy/ipvs/ipset/testing"
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
		activeEntries   []string
		currentEntries  []string
		expectedEntries []string
	}{
		{
			name: "normal ipset sync",
			set: &utilipset.IPSet{
				Name: "foo",
			},
			setType:         utilipset.HashIPPort,
			ipv6:            false,
			activeEntries:   []string{"172.17.0.4,tcp:80"},
			currentEntries:  nil,
			expectedEntries: []string{"172.17.0.4,tcp:80"},
		},
		{
			name: "ipset IPv6 sync with no new entries",
			set: &utilipset.IPSet{
				Name: "abz",
			},
			setType:         utilipset.HashIPPort,
			ipv6:            true,
			activeEntries:   []string{"FE80::0202:B3FF:FE1E:8329,tcp:80"},
			currentEntries:  []string{"FE80::0202:B3FF:FE1E:8329,tcp:80"},
			expectedEntries: []string{"FE80::0202:B3FF:FE1E:8329,tcp:80"},
		},
		{
			name: "ipset sync with updated udp->tcp in hash",
			set: &utilipset.IPSet{
				Name: "bca",
			},
			setType:         utilipset.HashIPPort,
			ipv6:            false,
			activeEntries:   []string{"172.17.0.4,tcp:80", "172.17.0.5,tcp:80"},
			currentEntries:  []string{"172.17.0.5,udp:53"},
			expectedEntries: []string{"172.17.0.4,tcp:80", "172.17.0.5,tcp:80"},
		},
		{
			name: "ipset sync no updates required",
			set: &utilipset.IPSet{
				Name: "bar",
			},
			setType:         utilipset.HashIPPortIP,
			ipv6:            false,
			activeEntries:   []string{"172.17.0.4,tcp:80:172.17.0.4"},
			currentEntries:  []string{"172.17.0.4,tcp:80:172.17.0.4"},
			expectedEntries: []string{"172.17.0.4,tcp:80:172.17.0.4"},
		},
		{
			name: "ipset IPv6 sync, delete and add new entry",
			set: &utilipset.IPSet{
				Name: "baz",
			},
			setType:         utilipset.HashIPPortIP,
			ipv6:            true,
			activeEntries:   []string{"FE80:0000:0000:0000:0202:B3FF:FE1E:8329,tcp:8080:FE80:0000:0000:0000:0202:B3FF:FE1E:8329"},
			currentEntries:  []string{"1111:0000:0000:0000:0202:B3FF:FE1E:8329,tcp:8081:1111:0000:0000:0000:0202:B3FF:FE1E:8329:8081"},
			expectedEntries: []string{"FE80:0000:0000:0000:0202:B3FF:FE1E:8329,tcp:8080:FE80:0000:0000:0000:0202:B3FF:FE1E:8329"},
		},
		{
			name: "ipset sync, no current entries",
			set: &utilipset.IPSet{
				Name: "NOPE",
			},
			setType:         utilipset.HashIPPortIP,
			ipv6:            false,
			activeEntries:   []string{"172.17.0.4,tcp:80,172.17.0.9", "172.17.0.5,tcp:80,172.17.0.10"},
			currentEntries:  nil,
			expectedEntries: []string{"172.17.0.4,tcp:80,172.17.0.9", "172.17.0.5,tcp:80,172.17.0.10"},
		},
		{
			name: "ipset sync, no current entries with /16 subnet",
			set: &utilipset.IPSet{
				Name: "ABC-DEF",
			},
			setType:         utilipset.HashIPPortNet,
			ipv6:            false,
			activeEntries:   []string{"172.17.0.4,tcp:80,172.17.0.0/16", "172.17.0.5,tcp:80,172.17.0.0/16"},
			currentEntries:  nil,
			expectedEntries: []string{"172.17.0.4,tcp:80,172.17.0.0/16", "172.17.0.5,tcp:80,172.17.0.0/16"},
		},
		{
			name: "ipset IPv6 sync, no updates required with /32 subnet",
			set: &utilipset.IPSet{
				Name: "zar",
			},
			setType:         utilipset.HashIPPortNet,
			ipv6:            true,
			activeEntries:   []string{"FE80::8329,tcp:8800,2001:db8::/32"},
			currentEntries:  []string{"FE80::8329,tcp:8800,2001:db8::/32"},
			expectedEntries: []string{"FE80::8329,tcp:8800,2001:db8::/32"},
		},
		{
			name: "ipset IPv6 sync, current entries removed",
			set: &utilipset.IPSet{
				Name: "bbb",
			},
			setType:         utilipset.HashIPPortNet,
			ipv6:            true,
			activeEntries:   nil,
			currentEntries:  []string{"FE80::8329,udp:8801,2001:db8::/32"},
			expectedEntries: nil,
		},
		{
			name: "ipset sync, port entry removed",
			set: &utilipset.IPSet{
				Name: "AAA",
			},
			setType:         utilipset.BitmapPort,
			activeEntries:   nil,
			currentEntries:  []string{"80"},
			expectedEntries: nil,
		},
		{
			name: "ipset sync, remove old and add new",
			set: &utilipset.IPSet{
				Name: "c-c-c",
			},
			setType:         utilipset.BitmapPort,
			activeEntries:   []string{"8080", "9090"},
			currentEntries:  []string{"80"},
			expectedEntries: []string{"8080", "9090"},
		},
		{
			name: "ipset sync, remove many stale ports",
			set: &utilipset.IPSet{
				Name: "NODE-PORT",
			},
			setType:         utilipset.BitmapPort,
			activeEntries:   []string{"8080"},
			currentEntries:  []string{"80", "9090", "8081", "8082"},
			expectedEntries: []string{"8080"},
		},
		{
			name: "ipset sync, add sctp entry",
			set: &utilipset.IPSet{
				Name: "sctp-1",
			},
			setType:         utilipset.HashIPPort,
			ipv6:            false,
			activeEntries:   []string{"172.17.0.4,sctp:80"},
			currentEntries:  nil,
			expectedEntries: []string{"172.17.0.4,sctp:80"},
		},
		{
			name: "ipset sync, add IPV6 sctp entry",
			set: &utilipset.IPSet{
				Name: "sctp-2",
			},
			setType:         utilipset.HashIPPort,
			ipv6:            true,
			activeEntries:   []string{"FE80::0202:B3FF:FE1E:8329,sctp:80"},
			currentEntries:  []string{"FE80::0202:B3FF:FE1E:8329,sctp:80"},
			expectedEntries: []string{"FE80::0202:B3FF:FE1E:8329,sctp:80"},
		},
	}

	for _, testCase := range testCases {
		t.Run(testCase.name, func(t *testing.T) {
			set := NewIPSet(fakeipset.NewFake(testIPSetVersion), testCase.set.Name, testCase.setType, testCase.ipv6, "comment-"+testCase.set.Name)

			if err := set.handle.CreateSet(&set.IPSet, true); err != nil {
				t.Errorf("Unexpected error: %v", err)
			}

			for _, entry := range testCase.currentEntries {
				set.handle.AddEntry(entry, testCase.set, true)
			}

			set.activeEntries.Insert(testCase.activeEntries...)
			set.syncIPSetEntries()
			for _, entry := range testCase.expectedEntries {
				found, err := set.handle.TestEntry(entry, testCase.set.Name)
				if err != nil {
					t.Errorf("Unexpected error: %v", err)
				}
				if !found {
					t.Errorf("Unexpected entry %q not found in set foo", entry)
				}
			}
		})
	}
}
