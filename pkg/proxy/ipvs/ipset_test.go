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
		vstring string
		valid   bool
	}{
		// version less than "6.0" is not valid.
		{"4.0", false},
		{"5.1", false},
		{"5.1.2", false},
		// "7" is not a valid version string.
		{"7", false},
		{"6.0", true},
		{"6.1", true},
		{"6.19", true},
		{"7.0", true},
		{"8.1.2", true},
		{"9.3.4.0", true},
		{"total junk", false},
	}

	for i := range testCases {
		valid := checkMinVersion(testCases[i].vstring)
		if testCases[i].valid != valid {
			t.Errorf("Expected result: %v, Got result: %v", testCases[i].valid, valid)
		}
	}
}

const testIPSetVersion = "v6.19"

func TestSyncIPSetEntries(t *testing.T) {
	testCases := []struct {
		set             *utilipset.IPSet
		setType         utilipset.Type
		ipv6            bool
		activeEntries   []string
		currentEntries  []string
		expectedEntries []string
	}{
		{ // case 0
			set: &utilipset.IPSet{
				Name: "foo",
			},
			setType:         utilipset.HashIPPort,
			ipv6:            false,
			activeEntries:   []string{"172.17.0.4,tcp:80"},
			currentEntries:  nil,
			expectedEntries: []string{"172.17.0.4,tcp:80"},
		},
		{ // case 1
			set: &utilipset.IPSet{
				Name: "abz",
			},
			setType:         utilipset.HashIPPort,
			ipv6:            true,
			activeEntries:   []string{"FE80::0202:B3FF:FE1E:8329,tcp:80"},
			currentEntries:  []string{"FE80::0202:B3FF:FE1E:8329,tcp:80"},
			expectedEntries: []string{"FE80::0202:B3FF:FE1E:8329,tcp:80"},
		},
		{ // case 2
			set: &utilipset.IPSet{
				Name: "bca",
			},
			setType:         utilipset.HashIPPort,
			ipv6:            false,
			activeEntries:   []string{"172.17.0.4,tcp:80", "172.17.0.5,tcp:80"},
			currentEntries:  []string{"172.17.0.5,udp:53"},
			expectedEntries: []string{"172.17.0.4,tcp:80", "172.17.0.5,tcp:80"},
		},
		{ // case 3
			set: &utilipset.IPSet{
				Name: "bar",
			},
			setType:         utilipset.HashIPPortIP,
			ipv6:            false,
			activeEntries:   []string{"172.17.0.4,tcp:80:172.17.0.4"},
			currentEntries:  []string{"172.17.0.4,tcp:80:172.17.0.4"},
			expectedEntries: []string{"172.17.0.4,tcp:80:172.17.0.4"},
		},
		{ // case 4
			set: &utilipset.IPSet{
				Name: "baz",
			},
			setType:         utilipset.HashIPPortIP,
			ipv6:            true,
			activeEntries:   []string{"FE80:0000:0000:0000:0202:B3FF:FE1E:8329,tcp:8080:FE80:0000:0000:0000:0202:B3FF:FE1E:8329"},
			currentEntries:  []string{"1111:0000:0000:0000:0202:B3FF:FE1E:8329,tcp:8081:1111:0000:0000:0000:0202:B3FF:FE1E:8329:8081"},
			expectedEntries: []string{"FE80:0000:0000:0000:0202:B3FF:FE1E:8329,tcp:8080:FE80:0000:0000:0000:0202:B3FF:FE1E:8329"},
		},
		{ // case 5
			set: &utilipset.IPSet{
				Name: "NOPE",
			},
			setType:         utilipset.HashIPPortIP,
			ipv6:            false,
			activeEntries:   []string{"172.17.0.4,tcp:80,172.17.0.9", "172.17.0.5,tcp:80,172.17.0.10"},
			currentEntries:  nil,
			expectedEntries: []string{"172.17.0.4,tcp:80,172.17.0.9", "172.17.0.5,tcp:80,172.17.0.10"},
		},
		{ // case 6
			set: &utilipset.IPSet{
				Name: "ABC-DEF",
			},
			setType:         utilipset.HashIPPortNet,
			ipv6:            false,
			activeEntries:   []string{"172.17.0.4,tcp:80,172.17.0.0/16", "172.17.0.5,tcp:80,172.17.0.0/16"},
			currentEntries:  nil,
			expectedEntries: []string{"172.17.0.4,tcp:80,172.17.0.0/16", "172.17.0.5,tcp:80,172.17.0.0/16"},
		},
		{ // case 7
			set: &utilipset.IPSet{
				Name: "zar",
			},
			setType:         utilipset.HashIPPortNet,
			ipv6:            true,
			activeEntries:   []string{"FE80::8329,tcp:8800,2001:db8::/32"},
			currentEntries:  []string{"FE80::8329,tcp:8800,2001:db8::/32"},
			expectedEntries: []string{"FE80::8329,tcp:8800,2001:db8::/32"},
		},
		{ // case 8
			set: &utilipset.IPSet{
				Name: "bbb",
			},
			setType:         utilipset.HashIPPortNet,
			ipv6:            true,
			activeEntries:   nil,
			currentEntries:  []string{"FE80::8329,udp:8801,2001:db8::/32"},
			expectedEntries: nil,
		},
		{ // case 9
			set: &utilipset.IPSet{
				Name: "AAA",
			},
			setType:         utilipset.BitmapPort,
			activeEntries:   nil,
			currentEntries:  []string{"80"},
			expectedEntries: nil,
		},
		{ // case 10
			set: &utilipset.IPSet{
				Name: "c-c-c",
			},
			setType:         utilipset.BitmapPort,
			activeEntries:   []string{"8080", "9090"},
			currentEntries:  []string{"80"},
			expectedEntries: []string{"8080", "9090"},
		},
		{ // case 11
			set: &utilipset.IPSet{
				Name: "NODE-PORT",
			},
			setType:         utilipset.BitmapPort,
			activeEntries:   []string{"8080"},
			currentEntries:  []string{"80", "9090", "8081", "8082"},
			expectedEntries: []string{"8080"},
		},
		{ // case 12
			set: &utilipset.IPSet{
				Name: "sctp-1",
			},
			setType:         utilipset.HashIPPort,
			ipv6:            false,
			activeEntries:   []string{"172.17.0.4,sctp:80"},
			currentEntries:  nil,
			expectedEntries: []string{"172.17.0.4,sctp:80"},
		},
		{ // case 1
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

	for i := range testCases {
		set := NewIPSet(fakeipset.NewFake(testIPSetVersion), testCases[i].set.Name, testCases[i].setType, testCases[i].ipv6, "comment-"+testCases[i].set.Name)

		if err := set.handle.CreateSet(&set.IPSet, true); err != nil {
			t.Errorf("Unexpected error: %v", err)
		}
		for _, entry := range testCases[i].expectedEntries {
			set.handle.AddEntry(entry, testCases[i].set, true)
		}

		set.activeEntries.Insert(testCases[i].activeEntries...)
		set.syncIPSetEntries()
		for _, entry := range testCases[i].expectedEntries {
			found, err := set.handle.TestEntry(entry, testCases[i].set.Name)
			if err != nil {
				t.Errorf("Unexpected error: %v", err)
			}
			if !found {
				t.Errorf("Unexpected entry 172.17.0.4,tcp:80 not found in set foo")
			}
		}
	}
}
