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

	"github.com/vmware/govmomi"
	"github.com/vmware/govmomi/find"
	"github.com/vmware/govmomi/object"
	"github.com/vmware/govmomi/simulator/esx"
	"github.com/vmware/govmomi/simulator/vpx"
	"github.com/vmware/govmomi/vim25/methods"
	"github.com/vmware/govmomi/vim25/mo"
	"github.com/vmware/govmomi/vim25/soap"
	"github.com/vmware/govmomi/vim25/types"
)

func addStandaloneHostTask(folder *object.Folder, spec types.HostConnectSpec) (*object.Task, error) {
	// TODO: add govmomi wrapper
	req := types.AddStandaloneHost_Task{
		This:         folder.Reference(),
		Spec:         spec,
		AddConnected: true,
	}

	res, err := methods.AddStandaloneHost_Task(context.TODO(), folder.Client(), &req)
	if err != nil {
		return nil, err
	}

	task := object.NewTask(folder.Client(), res.Returnval)
	return task, nil
}

func TestFolderESX(t *testing.T) {
	content := esx.ServiceContent
	s := New(NewServiceInstance(content, esx.RootFolder))

	ts := s.NewServer()
	defer ts.Close()

	ctx := context.Background()
	c, err := govmomi.NewClient(ctx, ts.URL, true)
	if err != nil {
		t.Fatal(err)
	}

	f := object.NewRootFolder(c.Client)

	_, err = f.CreateFolder(ctx, "foo")
	if err == nil {
		t.Error("expected error")
	}

	_, err = f.CreateDatacenter(ctx, "foo")
	if err == nil {
		t.Error("expected error")
	}

	finder := find.NewFinder(c.Client, false)
	dc, err := finder.DatacenterOrDefault(ctx, "")
	if err != nil {
		t.Fatal(err)
	}

	folders, err := dc.Folders(ctx)
	if err != nil {
		t.Fatal(err)
	}

	spec := types.HostConnectSpec{}
	_, err = addStandaloneHostTask(folders.HostFolder, spec)
	if err == nil {
		t.Fatal("expected error")
	}

	_, err = folders.DatastoreFolder.CreateStoragePod(ctx, "pod")
	if err == nil {
		t.Fatal("expected error")
	}
}

func TestFolderVC(t *testing.T) {
	content := vpx.ServiceContent
	s := New(NewServiceInstance(content, vpx.RootFolder))

	ts := s.NewServer()
	defer ts.Close()

	ctx := context.Background()
	c, err := govmomi.NewClient(ctx, ts.URL, true)
	if err != nil {
		t.Fatal(err)
	}

	f := object.NewRootFolder(c.Client)

	ff, err := f.CreateFolder(ctx, "foo")
	if err != nil {
		t.Error(err)
	}

	dc, err := f.CreateDatacenter(ctx, "bar")
	if err != nil {
		t.Error(err)
	}

	for _, ref := range []object.Reference{ff, dc} {
		o := Map.Get(ref.Reference())
		if o == nil {
			t.Fatalf("failed to find %#v", ref)
		}

		e := o.(mo.Entity).Entity()
		if *e.Parent != f.Reference() {
			t.Fail()
		}
	}

	dc, err = ff.CreateDatacenter(ctx, "biz")
	if err != nil {
		t.Error(err)
	}

	folders, err := dc.Folders(ctx)
	if err != nil {
		t.Fatal(err)
	}

	_, err = folders.VmFolder.CreateStoragePod(ctx, "pod")
	if err == nil {
		t.Error("expected error")
	}

	_, err = folders.DatastoreFolder.CreateStoragePod(ctx, "pod")
	if err != nil {
		t.Error(err)
	}

	tests := []struct {
		name  string
		state types.TaskInfoState
	}{
		{"", types.TaskInfoStateError},
		{"foo.local", types.TaskInfoStateSuccess},
	}

	for _, test := range tests {
		spec := types.HostConnectSpec{
			HostName: test.name,
		}

		task, err := addStandaloneHostTask(folders.HostFolder, spec)
		if err != nil {
			t.Fatal(err)
		}

		res, err := task.WaitForResult(ctx, nil)
		if test.state == types.TaskInfoStateError {
			if err == nil {
				t.Error("expected error")
			}

			if res.Result != nil {
				t.Error("expected nil")
			}
		} else {
			if err != nil {
				t.Fatal(err)
			}

			ref, ok := res.Result.(types.ManagedObjectReference)
			if !ok {
				t.Errorf("expected moref, got type=%T", res.Result)
			}
			host := Map.Get(ref).(*HostSystem)
			if host.Name != test.name {
				t.Fail()
			}

			if ref == esx.HostSystem.Self {
				t.Error("expected new host Self reference")
			}
			if *host.Summary.Host == esx.HostSystem.Self {
				t.Error("expected new host summary Self reference")
			}

			pool := Map.Get(*host.Parent).(*mo.ComputeResource).ResourcePool
			if *pool == esx.ResourcePool.Self {
				t.Error("expected new pool Self reference")
			}
		}

		if res.State != test.state {
			t.Fatalf("%s", res.State)
		}
	}
}

func TestFolderFaults(t *testing.T) {
	f := Folder{}
	f.ChildType = []string{"VirtualMachine"}

	if f.CreateFolder(nil).Fault() == nil {
		t.Error("expected fault")
	}

	if f.CreateDatacenter(nil, nil).Fault() == nil {
		t.Error("expected fault")
	}
}

func TestRegisterVm(t *testing.T) {
	ctx := context.Background()

	for i, model := range []*Model{ESX(), VPX()} {
		match := "*"
		if i == 1 {
			model.App = 1
			match = "*APP*"
		}
		defer model.Remove()
		err := model.Create()
		if err != nil {
			t.Fatal(err)
		}

		s := model.Service.NewServer()
		defer s.Close()

		c, err := govmomi.NewClient(ctx, s.URL, true)
		if err != nil {
			t.Fatal(err)
		}

		finder := find.NewFinder(c.Client, false)
		dc, err := finder.DefaultDatacenter(ctx)
		if err != nil {
			t.Fatal(err)
		}

		finder.SetDatacenter(dc)

		folders, err := dc.Folders(ctx)
		if err != nil {
			t.Fatal(err)
		}

		vmFolder := folders.VmFolder

		vms, err := finder.VirtualMachineList(ctx, match)
		if err != nil {
			t.Fatal(err)
		}

		vm := Map.Get(vms[0].Reference()).(*VirtualMachine)

		req := types.RegisterVM_Task{
			This:       vmFolder.Reference(),
			AsTemplate: true,
		}

		steps := []struct {
			e interface{}
			f func()
		}{
			{
				new(types.InvalidArgument), func() { req.AsTemplate = false },
			},
			{
				new(types.InvalidArgument), func() { req.Pool = vm.ResourcePool },
			},
			{
				new(types.InvalidArgument), func() { req.Path = "enoent" },
			},
			{
				new(types.InvalidDatastorePath), func() { req.Path = vm.Config.Files.VmPathName + "-enoent" },
			},
			{
				new(types.NotFound), func() { req.Path = vm.Config.Files.VmPathName },
			},
			{
				new(types.AlreadyExists), func() { Map.Remove(vm.Reference()) },
			},
			{
				nil, func() {},
			},
		}

		for _, step := range steps {
			res, err := methods.RegisterVM_Task(ctx, c.Client, &req)
			if err != nil {
				t.Fatal(err)
			}

			rt := Map.Get(res.Returnval).(*Task)

			if step.e != nil {
				fault := rt.Info.Error.Fault
				if reflect.TypeOf(fault) != reflect.TypeOf(step.e) {
					t.Errorf("%T != %T", fault, step.e)
				}
			} else {
				if rt.Info.Error != nil {
					t.Errorf("unexpected error: %#v", rt.Info.Error)
				}
			}

			step.f()
		}

		nvm, err := finder.VirtualMachine(ctx, vm.Name)
		if err != nil {
			t.Fatal(err)
		}

		if nvm.Reference() == vm.Reference() {
			t.Error("expected new moref")
		}

		_, _ = nvm.PowerOn(ctx)

		steps = []struct {
			e interface{}
			f func()
		}{
			{
				types.InvalidPowerState{}, func() { _, _ = nvm.PowerOff(ctx) },
			},
			{
				nil, func() {},
			},
			{
				types.ManagedObjectNotFound{}, func() {},
			},
		}

		for _, step := range steps {
			err = nvm.Unregister(ctx)

			if step.e != nil {
				fault := soap.ToSoapFault(err).VimFault()
				if reflect.TypeOf(fault) != reflect.TypeOf(step.e) {
					t.Errorf("%T != %T", fault, step.e)
				}
			} else {
				if err != nil {
					t.Errorf("unexpected error: %#v", err)
				}
			}

			step.f()
		}
	}
}

func TestFolderMoveInto(t *testing.T) {
	ctx := context.Background()
	model := VPX()
	defer model.Remove()
	err := model.Create()
	if err != nil {
		t.Fatal(err)
	}

	s := model.Service.NewServer()
	defer s.Close()

	c, err := govmomi.NewClient(ctx, s.URL, true)
	if err != nil {
		t.Fatal(err)
	}

	finder := find.NewFinder(c.Client, false)

	dc, err := finder.DefaultDatacenter(ctx)
	if err != nil {
		t.Fatal(err)
	}

	finder.SetDatacenter(dc)

	folders, err := dc.Folders(ctx)
	if err != nil {
		t.Fatal(err)
	}

	ds, err := finder.DefaultDatastore(ctx)
	if err != nil {
		t.Fatal(err)
	}

	// Move Datastore into a vm folder should fail
	task, err := folders.VmFolder.MoveInto(ctx, []types.ManagedObjectReference{ds.Reference()})
	if err != nil {
		t.Fatal(err)
	}

	err = task.Wait(ctx)
	if err == nil {
		t.Errorf("expected error")
	}

	// Move Datacenter into a sub folder should pass
	f, err := object.NewRootFolder(c.Client).CreateFolder(ctx, "foo")
	if err != nil {
		t.Error(err)
	}

	task, _ = f.MoveInto(ctx, []types.ManagedObjectReference{dc.Reference()})
	err = task.Wait(ctx)
	if err != nil {
		t.Error(err)
	}

	pod, err := folders.DatastoreFolder.CreateStoragePod(ctx, "pod")
	if err != nil {
		t.Error(err)
	}

	// Moving any type other than Datastore into a StoragePod should fail
	task, _ = pod.MoveInto(ctx, []types.ManagedObjectReference{dc.Reference()})
	err = task.Wait(ctx)
	if err == nil {
		t.Error("expected error")
	}

	// Move DS into a StoragePod
	task, _ = pod.MoveInto(ctx, []types.ManagedObjectReference{ds.Reference()})
	err = task.Wait(ctx)
	if err != nil {
		t.Error(err)
	}
}

func TestFolderCreateDVS(t *testing.T) {
	ctx := context.Background()
	model := VPX()
	defer model.Remove()
	err := model.Create()
	if err != nil {
		t.Fatal(err)
	}

	s := model.Service.NewServer()
	defer s.Close()

	c, err := govmomi.NewClient(ctx, s.URL, true)
	if err != nil {
		t.Fatal(err)
	}

	finder := find.NewFinder(c.Client, false)

	dc, err := finder.DefaultDatacenter(ctx)
	if err != nil {
		t.Fatal(err)
	}

	finder.SetDatacenter(dc)

	folders, err := dc.Folders(ctx)
	if err != nil {
		t.Fatal(err)
	}

	var spec types.DVSCreateSpec
	spec.ConfigSpec = &types.VMwareDVSConfigSpec{}
	spec.ConfigSpec.GetDVSConfigSpec().Name = "foo"

	task, err := folders.NetworkFolder.CreateDVS(ctx, spec)
	if err != nil {
		t.Fatal(err)
	}

	err = task.Wait(ctx)
	if err != nil {
		t.Error(err)
	}

	net, err := finder.Network(ctx, "foo")
	if err != nil {
		t.Error(err)
	}

	dvs, ok := net.(*object.DistributedVirtualSwitch)
	if !ok {
		t.Fatalf("%T is not of type %T", net, dvs)
	}

	task, err = folders.NetworkFolder.CreateDVS(ctx, spec)
	if err != nil {
		t.Fatal(err)
	}

	err = task.Wait(ctx)
	if err == nil {
		t.Error("expected error")
	}

	pspec := types.DVPortgroupConfigSpec{Name: "xnet"}
	task, err = dvs.AddPortgroup(ctx, []types.DVPortgroupConfigSpec{pspec})
	if err != nil {
		t.Fatal(err)
	}

	err = task.Wait(ctx)
	if err != nil {
		t.Error(err)
	}

	net, err = finder.Network(ctx, "xnet")
	if err != nil {
		t.Error(err)
	}

	pg, ok := net.(*object.DistributedVirtualPortgroup)
	if !ok {
		t.Fatalf("%T is not of type %T", net, pg)
	}

	backing, err := net.EthernetCardBackingInfo(ctx)
	if err != nil {
		t.Fatal(err)
	}

	info, ok := backing.(*types.VirtualEthernetCardDistributedVirtualPortBackingInfo)
	if ok {
		if info.Port.SwitchUuid == "" || info.Port.PortgroupKey == "" {
			t.Errorf("invalid port: %#v", info.Port)
		}
	} else {
		t.Fatalf("%T is not of type %T", net, info)
	}

	task, err = dvs.AddPortgroup(ctx, []types.DVPortgroupConfigSpec{pspec})
	if err != nil {
		t.Fatal(err)
	}

	err = task.Wait(ctx)
	if err == nil {
		t.Error("expected error")
	}
}
