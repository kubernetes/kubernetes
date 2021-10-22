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
	"testing"

	"github.com/vmware/govmomi"
	"github.com/vmware/govmomi/find"
	"github.com/vmware/govmomi/object"
	"github.com/vmware/govmomi/simulator/esx"
)

func TestDefaultESX(t *testing.T) {
	s := New(NewServiceInstance(esx.ServiceContent, esx.RootFolder))

	ts := s.NewServer()
	defer ts.Close()

	ctx := context.Background()

	client, err := govmomi.NewClient(ctx, ts.URL, true)
	if err != nil {
		t.Fatal(err)
	}

	finder := find.NewFinder(client.Client, false)

	dc, err := finder.DatacenterOrDefault(ctx, "")
	if err != nil {
		t.Fatal(err)
	}

	finder.SetDatacenter(dc)

	host, err := finder.HostSystemOrDefault(ctx, "*")
	if err != nil {
		t.Fatal(err)
	}

	if host.Name() != esx.HostSystem.Summary.Config.Name {
		t.Fail()
	}

	pool, err := finder.ResourcePoolOrDefault(ctx, "*")
	if err != nil {
		t.Fatal(err)
	}

	if pool.Name() != "Resources" {
		t.Fail()
	}
}

func TestMaintenanceMode(t *testing.T) {
	ctx := context.Background()
	m := ESX()

	defer m.Remove()

	err := m.Create()
	if err != nil {
		t.Fatal(err)
	}

	s := m.Service.NewServer()
	defer s.Close()

	c := m.Service.client

	hs := Map.Get(esx.HostSystem.Reference()).(*HostSystem)
	host := object.NewHostSystem(c, hs.Self)

	task, err := host.EnterMaintenanceMode(ctx, 1, false, nil)
	if err != nil {
		t.Fatal(err)
	}

	err = task.Wait(ctx)
	if err != nil {
		t.Fatal(err)
	}

	if hs.Runtime.InMaintenanceMode != true {
		t.Fatal("expect InMaintenanceMode is true; got false")
	}

	task, err = host.ExitMaintenanceMode(ctx, 1)
	if err != nil {
		t.Fatal(err)
	}

	err = task.Wait(ctx)
	if err != nil {
		t.Fatal(err)
	}

	if hs.Runtime.InMaintenanceMode != false {
		t.Fatal("expect InMaintenanceMode is false; got true")
	}
}

func TestNewHostSystem(t *testing.T) {
	m := ESX()

	defer m.Remove()

	err := m.Create()
	if err != nil {
		t.Fatal(err)
	}

	hs := NewHostSystem(esx.HostSystem)
	if hs.Summary.Runtime != &hs.Runtime {
		t.Fatal("expected hs.Summary.Runtime == &hs.Runtime; got !=")
	}
}

func TestDestroyHostSystem(t *testing.T) {
	ctx := context.Background()
	m := VPX()

	defer m.Remove()

	err := m.Create()
	if err != nil {
		t.Fatal(err)
	}

	s := m.Service.NewServer()
	defer s.Close()

	c := m.Service.client

	vm := Map.Any("VirtualMachine").(*VirtualMachine)
	hs := Map.Get(*vm.Runtime.Host).(*HostSystem)
	host := object.NewHostSystem(c, hs.Self)

	vms := []*VirtualMachine{}
	for _, vmref := range hs.Vm {
		vms = append(vms, Map.Get(vmref).(*VirtualMachine))
	}

	task, err := host.Destroy(ctx)
	if err != nil {
		t.Fatal(err)
	}

	err = task.Wait(ctx)
	if err == nil {
		t.Fatal("expect err because host with vms cannot be destroyed")
	}

	for _, vmref := range hs.Vm {
		vm := Map.Get(vmref).(*VirtualMachine)
		vmo := object.NewVirtualMachine(c, vm.Self)

		task, err := vmo.PowerOff(ctx)
		if err != nil {
			t.Fatal(err)
		}

		err = task.Wait(ctx)
		if err != nil {
			t.Fatal(err)
		}

		task, err = vmo.Destroy(ctx)
		if err != nil {
			t.Fatal(err)
		}

		err = task.Wait(ctx)
		if err != nil {
			t.Fatal(err)
		}
	}

	task, err = host.Destroy(ctx)
	if err != nil {
		t.Fatal(err)
	}

	err = task.Wait(ctx)
	if err != nil {
		t.Fatal(err)
	}

	hs2 := Map.Get(esx.HostSystem.Reference())
	if hs2 != nil {
		t.Fatal("host should have been destroyed")
	}
}
