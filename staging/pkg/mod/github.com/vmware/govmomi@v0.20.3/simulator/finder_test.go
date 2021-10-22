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
	"context"
	"errors"
	"fmt"
	"reflect"
	"testing"

	"github.com/vmware/govmomi"
	"github.com/vmware/govmomi/find"
	"github.com/vmware/govmomi/object"
	"github.com/vmware/govmomi/vim25/types"
)

func TestFinderVPX(t *testing.T) {
	ctx := context.Background()

	m := VPX()
	m.Datacenter = 3
	m.Folder = 2
	m.Pool = 1

	defer m.Remove()

	err := m.Create()
	if err != nil {
		t.Fatal(err)
	}

	s := m.Service.NewServer()
	defer s.Close()

	client, err := govmomi.NewClient(ctx, s.URL, true)
	if err != nil {
		t.Fatal(err)
	}

	finder := find.NewFinder(client.Client, false)
	dc, _ := finder.Datacenter(ctx, "/F0/DC1")
	finder.SetDatacenter(dc)

	tests := []struct {
		kind   string
		path   string
		expect int
	}{
		{"ManagedObjectList", "./*", 4},       // /F0/DC1/{vm,host,network,datastore}
		{"ManagedObjectListChildren", ".", 4}, // ""
		{"ManagedObjectList", "/", 1},
		{"ManagedObjectList", "/*", m.Datacenter - m.Folder + m.Folder},
		{"ManagedObjectList", "/DC0", 1},
		{"ManagedObjectList", "/F[01]", 2},
		{"ManagedObjectListChildren", "/*", m.Datacenter + 3},
		{"ManagedObjectListChildren", "/*/*", 19},
		{"ManagedObjectListChildren", "/*/*/*", 25},
		{"FolderList", "/*", m.Folder},
		{"DatacenterList", "/F0/*", 1},
		{"DatacenterList", "/DC0", 1},
		{"VirtualMachineList", "/DC0/vm/*", (m.Host + m.Cluster) * m.Machine},
		{"VirtualMachineList", "F0/DC1_C0_RP0_VM0", 1},
		{"VirtualMachineList", "./DC1_C0_RP0_VM0", 0},
		{"VirtualMachineList", "DC1_C0_RP0_VM0", 1}, // find . -type VirtualMachine -name DC1_C0_RP0_VM0
		{"VirtualAppList", "/DC0/vm/*", 0},
		{"DatastoreList", "/DC0/datastore/*", m.Datastore},
		{"DatastoreList", "./*", 0},
		{"DatastoreList", "./F0/*", m.Datastore},
		{"DatastoreList", "/F*/*/datastore/F*/*", m.Datastore * m.Folder},
		{"DatastoreList", "/F0/DC1/datastore/F0/LocalDS_0", m.Datastore},
		{"DatastoreList", "/F0/DC1/datastore/F0/*", m.Datastore},
		{"DatastoreList", "/F1/DC2/datastore/F1/*", m.Datastore},
		{"DatastoreList", "./LocalDS_0", 0},
		{"DatastoreList", "LocalDS_0", 1}, // find . -type Datastore -name LocalDS_0
		{"DatastoreClusterList", "/DC0/datastore/*", 0},
		{"ComputeResourceList", "/DC0/host/*", m.Host + m.Cluster},
		{"ClusterComputeResourceList", "/DC0/host/*", m.Cluster},
		{"HostSystemList", "/DC0/host/*", m.Host + m.ClusterHost},
		{"HostSystemList", "/F0/DC1/host/F0/*", m.Host + m.ClusterHost},
		{"HostSystemList", "DC1_H0", 1},                    // find . -type HostSystem -name DC1_H0
		{"ComputeResourceList", "DC1_H0", 1},               // find . -type ComputeResource -name DC1_H0
		{"ClusterComputeResourceList", "DC1_C0", 1},        // find . -type ClusterComputeResource -name DC1_H0
		{"NetworkList", "/DC0/network/*", 3 + m.Portgroup}, // VM Network + DSwitch + DSwitch-Uplinks + m.Portgroup
		{"NetworkList", "./*", 1},
		{"NetworkList", "/F0/DC1/network/VM Network", 1},
		{"NetworkList", "/F0/DC1/network/F0/*", 2 + m.Portgroup},
		{"NetworkList", "./F0/DC1_DVPG0", 1},
		{"NetworkList", "./F0/DC1_DVPG0", 1},
		{"NetworkList", "DC1_DVPG0", 1}, // find . -type Network -name DC1_DVPG0
		{"ResourcePoolList", "/F0/DC1/host/F0/*", m.Host + m.Cluster},
		{"ResourcePoolList", "/F0/DC1/host/F0/*/*", m.Host + m.Cluster},
		{"ResourcePoolList", "/DC0/host/*", m.Host + m.Cluster},
		{"ResourcePoolList", "/DC0/host/*/*", m.Host + m.Cluster},
		{"ResourcePoolList", "/DC0/host/DC0_H0/Resources", 1},
		{"ResourcePoolList", "/F1/DC2/host/F1/DC2_C0/Resources", 1},
		{"ResourcePoolList", "Resources", m.Host + m.Cluster},                // find . -type ResourcePool -name Resources
		{"ResourcePoolList", "/F1/DC2/...", m.Pool + m.Host + 1},             // find $path -type ResourcePool
		{"ResourcePoolList", "/F1/DC2/host/...", m.Pool + m.Host + 1},        // find $path -type ResourcePool
		{"ResourcePoolList", "/F1/DC2/host/F1/...", m.Pool + m.Host + 1},     // find $path -type ResourcePool
		{"ResourcePoolList", "/F1/DC2/host/F1/DC2_C0/...", m.Pool + 1},       // find $path -type ResourcePool
		{"ResourcePoolList", "/F1/DC2/host/F1/DC2_C0/Resources/...", m.Pool}, // find $path -type ResourcePool
		{"ResourcePoolList", "F0/DC1_C0", 1},
		{"ResourcePoolList", "DC1_C0_RP1", 1},     // find . -type ResourcePool -name DC1_C0_RP1
		{"", "", 0},                               // unset Datacenter
		{"DatacenterList", "*", m.Datacenter},     // find . -type Datacenter
		{"DatacenterList", "./...", m.Datacenter}, // find . -type Datacenter
		{"DatacenterList", "DC2", 1},              // find . -type Datacenter -name DC2
		{"DatacenterList", "/*", m.Datacenter - m.Folder},
		{"DatacenterList", "/*/*", m.Folder},
		{"DatastoreList", "/F1/DC2/datastore/F1/LocalDS_0", 1},
		{"VirtualMachineList", "DC1_C0_RP0_VM0", 0}, // TODO: recurse all Datacenters?
	}

	f := reflect.ValueOf(finder)
	c := reflect.ValueOf(ctx)

	for i, test := range tests {
		if test.kind == "" {
			finder.SetDatacenter(nil)
			continue
		}

		err = nil

		arg := []reflect.Value{c, reflect.ValueOf(test.path)}
		res := f.MethodByName(test.kind).Call(arg)

		rval := res[0]
		rerr := res[1]

		if rval.Len() != test.expect {
			msg := fmt.Sprintf("expected %d, got %d", test.expect, rval.Len())
			if !rerr.IsNil() {
				msg += fmt.Sprintf(" (%s)", rerr.Interface())
			}
			err = errors.New(msg)

			if !rval.IsNil() {
				for j := 0; j < rval.Len(); j++ {
					t.Logf("%s\n", rval.Index(j).Interface())
				}
			}
		} else if !rerr.IsNil() {
			if test.expect != 0 {
				err = rerr.Interface().(error)
			}
		}

		if err != nil {
			t.Errorf("%d) %s(%s): %s", i, test.kind, test.path, err)
		}
	}
}

func TestFinderESX(t *testing.T) {
	ctx := context.Background()

	m := ESX()

	defer m.Remove()

	err := m.Create()
	if err != nil {
		t.Fatal(err)
	}

	s := m.Service.NewServer()
	defer s.Close()

	client, err := govmomi.NewClient(ctx, s.URL, true)
	if err != nil {
		t.Fatal(err)
	}

	finder := find.NewFinder(client.Client, false)

	dc, err := finder.DefaultDatacenter(ctx)
	if err != nil {
		t.Fatal(err)
	}

	finder.SetDatacenter(dc)

	f := reflect.ValueOf(finder)
	c := reflect.ValueOf(ctx)

	tests := []string{"Folder", "Datastore", "ComputeResource", "HostSystem", "Datastore", "Network"}

	for _, test := range tests {
		res := f.MethodByName("Default" + test).Call([]reflect.Value{c})
		if !res[1].IsNil() {
			t.Fatalf("%s: %s", test, res[1].Interface())
		}

		// test find by moref
		ref := res[0].Interface().(object.Reference).Reference()
		o, err := finder.Element(ctx, ref)
		if err != nil {
			t.Fatal(err)
		}

		if o.Object.Reference() != ref {
			t.Errorf("%s", ref)
		}
	}
}

func TestFinderDefaultHostVPX(t *testing.T) {
	ctx := context.Background()

	m := VPX()
	m.Folder = 1

	defer m.Remove()

	err := m.Create()
	if err != nil {
		t.Fatal(err)
	}

	s := m.Service.NewServer()
	defer s.Close()

	client, err := govmomi.NewClient(ctx, s.URL, true)
	if err != nil {
		t.Fatal(err)
	}

	finder := find.NewFinder(client.Client, false)
	dc, _ := finder.Datacenter(ctx, "/F0/DC0")
	finder.SetDatacenter(dc)

	hostf, _ := finder.Folder(ctx, dc.InventoryPath+"/host/F0")

	folders, err := dc.Folders(ctx)
	if err != nil {
		t.Fatal(err)
	}

	f, err := folders.HostFolder.CreateFolder(ctx, "MyHosts")
	if err != nil {
		t.Fatal(err)
	}

	// 2-levels (MyHosts/F0) deep under the DC host folder: /F0/DC0/host/MyHosts/F0/DC0_C0/DC0_C0_H0
	task, _ := f.MoveInto(ctx, []types.ManagedObjectReference{hostf.Reference()})
	if err = task.Wait(ctx); err != nil {
		t.Fatal(err)
	}

	_, err = finder.HostSystemOrDefault(ctx, "")
	if err == nil {
		t.Fatal("expected error")
	}

	_, ok := err.(*find.DefaultMultipleFoundError)
	if !ok {
		t.Errorf("unexpected error type=%T", err)
	}
}
