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

package testing

import (
	"net"
	"testing"

	utilipvs "k8s.io/kubernetes/pkg/util/ipvs"
)

func TestVirtualServer(t *testing.T) {
	// Initialize
	fake := NewFake()
	// Add a virtual server
	vs1 := &utilipvs.VirtualServer{
		Address:  net.ParseIP("1.2.3.4"),
		Port:     uint16(80),
		Protocol: string("TCP"),
		Flags:    utilipvs.FlagHashed,
	}
	err := fake.AddVirtualServer(vs1)
	if err != nil {
		t.Errorf("Fail to add virtual server, error: %v", err)
	}
	// Get a specific virtual server
	got1, err := fake.GetVirtualServer(vs1)
	if err != nil {
		t.Errorf("Fail to get virtual server, error: %v", err)
	}
	if !vs1.Equal(got1) {
		t.Errorf("Expect virtual server: %v, got: %v", vs1, got1)
	}
	// Update virtual server
	vs12 := &utilipvs.VirtualServer{
		Address:  net.ParseIP("1.2.3.4"),
		Port:     uint16(80),
		Protocol: string("TCP"),
		Flags:    utilipvs.FlagPersistent,
	}
	err = fake.UpdateVirtualServer(vs12)
	if err != nil {
		t.Errorf("Fail to update virtual server, error: %v", err)
	}
	// Check the updated virtual server
	got12, err := fake.GetVirtualServer(vs1)
	if err != nil {
		t.Errorf("Fail to get virtual server, error: %v", err)
	}
	if !got12.Equal(vs12) {
		t.Errorf("Expect virtual server: %v, got: %v", vs12, got12)
	}
	// Add another virtual server
	vs2 := &utilipvs.VirtualServer{
		Address:  net.ParseIP("10::40"),
		Port:     uint16(8080),
		Protocol: string("UDP"),
	}
	err = fake.AddVirtualServer(vs2)
	if err != nil {
		t.Errorf("Unexpected error when add virtual server, error: %v", err)
	}
	// Add another virtual server
	vs3 := &utilipvs.VirtualServer{
		Address:  net.ParseIP("10::40"),
		Port:     uint16(7777),
		Protocol: string("SCTP"),
	}
	err = fake.AddVirtualServer(vs3)
	if err != nil {
		t.Errorf("Unexpected error when add virtual server, error: %v", err)
	}
	// List all virtual servers
	list, err := fake.GetVirtualServers()
	if err != nil {
		t.Errorf("Fail to list virtual servers, error: %v", err)
	}
	if len(list) != 3 {
		t.Errorf("Expect 2 virtual servers, got: %d", len(list))
	}
	// Delete a virtual server
	err = fake.DeleteVirtualServer(vs1)
	if err != nil {
		t.Errorf("Fail to delete virtual server: %v, error: %v", vs1, err)
	}
	// Check the deleted virtual server no longer exists
	got, _ := fake.GetVirtualServer(vs1)
	if got != nil {
		t.Errorf("Expect nil, got: %v", got)
	}
	// Flush all virtual servers
	err = fake.Flush()
	if err != nil {
		t.Errorf("Fail to flush virtual servers, error: %v", err)
	}
	// List all virtual servers
	list, err = fake.GetVirtualServers()
	if err != nil {
		t.Errorf("Fail to list virtual servers, error: %v", err)
	}
	if len(list) != 0 {
		t.Errorf("Expect 0 virtual servers, got: %d", len(list))
	}
}

func TestRealServer(t *testing.T) {
	// Initialize
	fake := NewFake()
	// Add a virtual server
	vs := &utilipvs.VirtualServer{
		Address:  net.ParseIP("10.20.30.40"),
		Port:     uint16(80),
		Protocol: string("TCP"),
	}
	rss := []*utilipvs.RealServer{
		{Address: net.ParseIP("172.16.2.1"), Port: 8080, Weight: 1},
		{Address: net.ParseIP("172.16.2.2"), Port: 8080, Weight: 2},
		{Address: net.ParseIP("172.16.2.3"), Port: 8080, Weight: 3},
	}
	err := fake.AddVirtualServer(vs)
	if err != nil {
		t.Errorf("Fail to add virtual server, error: %v", err)
	}
	// Add real server to the virtual server
	for i := range rss {
		if err = fake.AddRealServer(vs, rss[i]); err != nil {
			t.Errorf("Fail to add real server, error: %v", err)
		}
	}
	// Delete a real server of the virtual server
	// Make sure any position of the list can be real deleted
	rssLen := len(rss)
	for i := range rss {
		// List all real servers of the virtual server
		list, err := fake.GetRealServers(vs)
		if err != nil {
			t.Errorf("Fail to get real servers of the virtual server, error: %v", err)
		}
		if len(list) != rssLen {
			t.Errorf("Expect %d virtual servers, got: %d", len(rss), len(list))
		}
		rsToDel := list[i]
		if err = fake.DeleteRealServer(vs, rsToDel); err != nil {
			t.Errorf("Fail to delete real server of the virtual server, error: %v", err)
		} else {
			dests, err := fake.GetRealServers(vs)
			if err != nil {
				t.Errorf("Fail to get real servers of the virtual server, error: %v", err)
			}
			for _, dest := range dests {
				if toRealServerKey(dest).String() == toRealServerKey(rsToDel).String() {
					t.Errorf("Expect real server %q be deleted.", rsToDel.String())
				}
			}
			if err = fake.AddRealServer(vs, rsToDel); err != nil {
				t.Errorf("Fail to add real server, error: %v", err)
			}
		}
	}
	// Test delete real server that not exist
	rs := &utilipvs.RealServer{
		Address: net.ParseIP("172.16.2.4"),
		Port:    uint16(8080),
		Weight:  1,
	}
	if err = fake.DeleteRealServer(vs, rs); err == nil {
		t.Errorf("Delete real server that not exist, Expect error, got nil")
	}
	// Delete the virtual server
	err = fake.DeleteVirtualServer(vs)
	if err != nil {
		t.Errorf("Fail to delete virtual server, error: %v", err)
	}
	_, err = fake.GetRealServers(vs)
	if err == nil {
		t.Errorf("Expect error, got nil")
	}
}
