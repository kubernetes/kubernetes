/*
Copyright (c) 2017 VMware, Inc. All Rights Reserved.

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

package simulator

import (
	"testing"

	"github.com/vmware/govmomi/simulator/vpx"
)

func compareModel(t *testing.T, m *Model) {
	count := m.Count()

	hosts := (m.Host + (m.ClusterHost * m.Cluster)) * m.Datacenter
	vms := ((m.Host + m.Cluster) * m.Datacenter) * m.Machine
	// child pools + root pools
	pools := (m.Pool * m.Cluster * m.Datacenter) + (m.Host+m.Cluster)*m.Datacenter
	// root folder + Datacenter folders {host,vm,datastore,network} + top-level folders
	folders := 1 + (4 * m.Datacenter) + (5 * m.Folder)
	pgs := m.Portgroup
	if pgs > 0 {
		pgs++ // uplinks
	}

	tests := []struct {
		expect int
		actual int
		kind   string
	}{
		{m.Datacenter, count.Datacenter, "Datacenter"},
		{m.Cluster * m.Datacenter, count.Cluster, "Cluster"},
		{pgs * m.Datacenter, count.Portgroup, "Portgroup"},
		{m.Datastore * m.Datacenter, count.Datastore, "Datastore"},
		{hosts, count.Host, "Host"},
		{vms, count.Machine, "VirtualMachine"},
		{pools, count.Pool, "ResourcePool"},
		{folders, count.Folder, "Folder"},
	}

	for _, test := range tests {
		if test.expect != test.actual {
			t.Errorf("expected %d %s, actual: %d", test.expect, test.kind, test.actual)
		}
	}
}

func TestModelESX(t *testing.T) {
	m := ESX()
	defer m.Remove()

	err := m.Create()
	if err != nil {
		t.Fatal(err)
	}

	// Set these for the compareModel math, and for m.Create to fail below
	m.Datacenter = 1
	m.Host = 1

	compareModel(t, m)

	err = m.Create()
	if err == nil {
		t.Error("expected error")
	}
}

func TestModelVPX(t *testing.T) {
	m := VPX()

	defer m.Remove()

	err := m.Create()
	if err != nil {
		t.Fatal(err)
	}

	compareModel(t, m)
}

func TestModelNoSwitchVPX(t *testing.T) {
	m := VPX()
	m.Portgroup = 0 // disabled DVS creation

	defer m.Remove()

	err := m.Create()
	if err != nil {
		t.Fatal(err)
	}

	compareModel(t, m)
}

func TestModelCustomVPX(t *testing.T) {
	m := &Model{
		ServiceContent: vpx.ServiceContent,
		RootFolder:     vpx.RootFolder,
		Datacenter:     2,
		Cluster:        2,
		Host:           2,
		ClusterHost:    3,
		Datastore:      1,
		Machine:        3,
		Pool:           2,
		Portgroup:      2,
	}

	defer m.Remove()

	err := m.Create()
	if err != nil {
		t.Fatal(err)
	}

	compareModel(t, m)
}

func TestModelWithFolders(t *testing.T) {
	m := VPX()
	m.Datacenter = 3
	m.Folder = 2

	defer m.Remove()

	err := m.Create()
	if err != nil {
		t.Fatal(err)
	}

	compareModel(t, m)
}
