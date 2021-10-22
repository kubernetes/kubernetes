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
	"reflect"
	"testing"

	"github.com/vmware/govmomi/find"
	"github.com/vmware/govmomi/object"
	"github.com/vmware/govmomi/task"
	"github.com/vmware/govmomi/vim25/types"
)

func TestDVS(t *testing.T) {
	m := VPX()

	defer m.Remove()

	err := m.Create()
	if err != nil {
		t.Fatal(err)
	}

	ctx := context.Background()
	c := m.Service.client

	finder := find.NewFinder(c, false)
	dc, _ := finder.DatacenterList(ctx, "*")
	finder.SetDatacenter(dc[0])
	folders, _ := dc[0].Folders(ctx)
	hosts, _ := finder.HostSystemList(ctx, "*/*")
	vswitch := Map.Any("DistributedVirtualSwitch").(*DistributedVirtualSwitch)
	dvs0 := object.NewDistributedVirtualSwitch(c, vswitch.Reference())

	if len(vswitch.Summary.HostMember) == 0 {
		t.Fatal("no host member")
	}

	for _, ref := range vswitch.Summary.HostMember {
		host := Map.Get(ref).(*HostSystem)
		if len(host.Network) == 0 {
			t.Fatalf("%s.Network=%v", ref, host.Network)
		}
		parent := hostParent(&host.HostSystem)
		if len(parent.Network) != len(host.Network) {
			t.Fatalf("%s.Network=%v", parent.Reference(), parent.Network)
		}
	}

	var spec types.DVSCreateSpec
	spec.ConfigSpec = &types.VMwareDVSConfigSpec{}
	spec.ConfigSpec.GetDVSConfigSpec().Name = "DVS1"

	dtask, err := folders.NetworkFolder.CreateDVS(ctx, spec)
	if err != nil {
		t.Fatal(err)
	}

	info, err := dtask.WaitForResult(ctx, nil)
	if err != nil {
		t.Fatal(err)
	}

	dvs := object.NewDistributedVirtualSwitch(c, info.Result.(types.ManagedObjectReference))

	config := &types.DVSConfigSpec{}

	for _, host := range hosts {
		config.Host = append(config.Host, types.DistributedVirtualSwitchHostMemberConfigSpec{
			Host: host.Reference(),
		})
	}

	tests := []struct {
		op  types.ConfigSpecOperation
		pg  string
		err types.BaseMethodFault
	}{
		{types.ConfigSpecOperationAdd, "", nil},                               // Add == OK
		{types.ConfigSpecOperationAdd, "", &types.AlreadyExists{}},            // Add == fail (AlreadyExists)
		{types.ConfigSpecOperationEdit, "", &types.NotSupported{}},            // Edit == fail (NotSupported)
		{types.ConfigSpecOperationRemove, "", nil},                            // Remove == OK
		{types.ConfigSpecOperationAdd, "", nil},                               // Add == OK
		{types.ConfigSpecOperationAdd, "DVPG0", nil},                          // Add PG == OK
		{types.ConfigSpecOperationRemove, "", &types.ResourceInUse{}},         // Remove dvs0 == fail (ResourceInUse)
		{types.ConfigSpecOperationRemove, "", nil},                            // Remove dvs1 == OK (no VMs attached)
		{types.ConfigSpecOperationRemove, "", &types.ManagedObjectNotFound{}}, // Remove == fail (ManagedObjectNotFound)
	}

	for x, test := range tests {
		dswitch := dvs

		switch test.err.(type) {
		case *types.ManagedObjectNotFound:
			for i := range config.Host {
				config.Host[i].Host.Value = "enoent"
			}
		case *types.ResourceInUse:
			dswitch = dvs0
		}

		if test.pg == "" {
			for i := range config.Host {
				config.Host[i].Operation = string(test.op)
			}

			dtask, err = dswitch.Reconfigure(ctx, config)
		} else {
			switch test.op {
			case types.ConfigSpecOperationAdd:
				dtask, err = dswitch.AddPortgroup(ctx, []types.DVPortgroupConfigSpec{{Name: test.pg}})
			}
		}

		if err != nil {
			t.Fatal(err)
		}

		err = dtask.Wait(ctx)

		if test.err == nil {
			if err != nil {
				t.Fatalf("%d: %s", x, err)
			}
			continue
		}

		if err == nil {
			t.Errorf("expected error in test %d", x)
		}

		if reflect.TypeOf(test.err) != reflect.TypeOf(err.(task.Error).Fault()) {
			t.Errorf("expected %T fault in test %d", test.err, x)
		}
	}

	ports, err := dvs.FetchDVPorts(ctx, nil)
	if err != nil {
		t.Fatal(err)
	}
	if len(ports) != 2 {
		t.Fatalf("expected 2 ports in DVPorts; got %d", len(ports))
	}

	dtask, err = dvs.Destroy(ctx)
	if err != nil {
		t.Fatal(err)
	}

	err = dtask.Wait(ctx)
	if err != nil {
		t.Fatal(err)
	}
}
