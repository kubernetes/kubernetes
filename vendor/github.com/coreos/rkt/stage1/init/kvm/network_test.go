// Copyright 2015 The rkt Authors
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

package kvm

import (
	"net"
	"testing"

	"github.com/containernetworking/cni/pkg/types"
)

type testNetDescriber struct {
	hostIP  net.IP
	guestIP net.IP
	mask    net.IP
	name    string
	ifName  string
	ipMasq  bool
}

func (t testNetDescriber) HostIP() net.IP        { return t.hostIP }
func (t testNetDescriber) GuestIP() net.IP       { return t.guestIP }
func (t testNetDescriber) Mask() net.IP          { return t.mask }
func (t testNetDescriber) IfName() string        { return t.ifName }
func (t testNetDescriber) IPMasq() bool          { return t.ipMasq }
func (t testNetDescriber) Name() string          { return t.name }
func (t testNetDescriber) Gateway() net.IP       { return net.IP{1, 1, 1, 1} }
func (t testNetDescriber) Routes() []types.Route { return []types.Route{} }

func TestGetKVMNetArgs(t *testing.T) {
	tests := []struct {
		netDescriptions []NetDescriber
		expectedLkvm    []string
	}{
		{ // without Masquerading - not gw passed to kernel
			netDescriptions: []NetDescriber{
				testNetDescriber{
					net.ParseIP("1.1.1.1"),
					net.ParseIP("2.2.2.2"),
					net.ParseIP("255.255.255.0"),
					"test-net",
					"fooInt",
					false,
				},
			},
			expectedLkvm: []string{"--network", "mode=tap,tapif=fooInt,host_ip=1.1.1.1,guest_ip=2.2.2.2"},
		},
		{ // extra gw passed to kernel on (third position)
			netDescriptions: []NetDescriber{
				testNetDescriber{
					net.ParseIP("1.1.1.1"),
					net.ParseIP("2.2.2.2"),
					net.ParseIP("255.255.255.0"),
					"test-net",
					"barInt",
					true,
				},
			},
			expectedLkvm: []string{"--network", "mode=tap,tapif=barInt,host_ip=1.1.1.1,guest_ip=2.2.2.2"},
		},
		{ // two networks
			netDescriptions: []NetDescriber{
				testNetDescriber{
					net.ParseIP("1.1.1.1"),
					net.ParseIP("2.2.2.2"),
					net.ParseIP("255.255.255.0"),
					"test-net",
					"fooInt",
					false,
				},
				testNetDescriber{
					net.ParseIP("1.1.1.1"),
					net.ParseIP("2.2.2.2"),
					net.ParseIP("255.255.255.0"),
					"test-net",
					"barInt",
					true,
				},
			},
			expectedLkvm: []string{
				"--network", "mode=tap,tapif=fooInt,host_ip=1.1.1.1,guest_ip=2.2.2.2",
				"--network", "mode=tap,tapif=barInt,host_ip=1.1.1.1,guest_ip=2.2.2.2",
			},
		},
	}

	for i, tt := range tests {
		gotLkvm, err := GetKVMNetArgs(tt.netDescriptions)
		if err != nil {
			t.Errorf("got error: %s", err)
		}
		if len(gotLkvm) != len(tt.expectedLkvm) {
			t.Errorf("#%d: expected lkvm %v elements got %v", i, len(tt.expectedLkvm), len(gotLkvm))
		} else {
			for iarg, argExpected := range tt.expectedLkvm {
				if gotLkvm[iarg] != argExpected {
					t.Errorf("#%d: lkvm arg %d expected `%v` got `%v`", i, iarg, argExpected, gotLkvm[iarg])
				}
			}
		}
	}
}
