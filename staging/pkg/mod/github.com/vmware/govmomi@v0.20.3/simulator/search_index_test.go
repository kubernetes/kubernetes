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
	"strings"
	"testing"

	"github.com/vmware/govmomi"
	"github.com/vmware/govmomi/find"
	"github.com/vmware/govmomi/object"
	"github.com/vmware/govmomi/vim25/soap"
	"github.com/vmware/govmomi/vim25/types"
)

func TestSearchIndex(t *testing.T) {
	ctx := context.Background()

	for _, model := range []*Model{ESX(), VPX()} {
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

		vms, err := finder.VirtualMachineList(ctx, "*")
		if err != nil {
			t.Fatal(err)
		}

		vm := Map.Get(vms[0].Reference()).(*VirtualMachine)

		si := object.NewSearchIndex(c.Client)

		ref, err := si.FindByDatastorePath(ctx, dc, vm.Config.Files.VmPathName)
		if err != nil {
			t.Fatal(err)
		}

		if ref.Reference() != vm.Reference() {
			t.Errorf("moref mismatch %s != %s", ref, vm.Reference())
		}

		ref, err = si.FindByDatastorePath(ctx, dc, vm.Config.Files.VmPathName+"enoent")
		if err != nil {
			t.Fatal(err)
		}

		if ref != nil {
			t.Errorf("ref=%s", ref)
		}

		ref, err = si.FindByUuid(ctx, dc, vm.Config.Uuid, true, nil)
		if err != nil {
			t.Fatal(err)
		}

		if ref.Reference() != vm.Reference() {
			t.Errorf("moref mismatch %s != %s", ref, vm.Reference())
		}

		ref, err = si.FindByUuid(ctx, dc, vm.Config.Uuid, true, types.NewBool(false))
		if err != nil {
			t.Fatal(err)
		}

		if ref.Reference() != vm.Reference() {
			t.Errorf("moref mismatch %s != %s", ref, vm.Reference())
		}

		ref, err = si.FindByUuid(ctx, dc, vm.Config.InstanceUuid, true, types.NewBool(true))
		if err != nil {
			t.Fatal(err)
		}

		if ref.Reference() != vm.Reference() {
			t.Errorf("moref mismatch %s != %s", ref, vm.Reference())
		}

		ref, err = si.FindByUuid(ctx, dc, vm.Config.Uuid, false, nil)
		if err != nil {
			t.Fatal(err)
		}

		if ref != nil {
			t.Error("expected nil")
		}

		host := Map.Any("HostSystem").(*HostSystem)

		ref, err = si.FindByUuid(ctx, dc, host.Summary.Hardware.Uuid, false, nil)
		if err != nil {
			t.Fatal(err)
		}

		if ref.Reference() != host.Reference() {
			t.Errorf("moref mismatch %s != %s", ref, host.Reference())
		}
	}
}

func TestSearchIndexFindChild(t *testing.T) {
	ctx := context.Background()

	model := VPX()
	model.Pool = 3

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

	si := object.NewSearchIndex(c.Client)

	tests := [][]string{
		// Datacenter -> host Folder -> Cluster -> HostSystem
		{"DC0", "host", "DC0_C0", "DC0_C0_H0"},
		// Datacenter -> host Folder -> ComputeResource -> HostSystem
		{"DC0", "host", "DC0_H0", "DC0_H0"},
		// Datacenter -> host Folder -> Cluster -> ResourcePool -> ResourcePool
		{"DC0", "host", "DC0_C0", "Resources", "DC0_C0_RP1"},
		// Datacenter -> host Folder -> Cluster -> ResourcePool -> VirtualMachine
		{"DC0", "host", "DC0_C0", "Resources", "DC0_C0_RP0_VM0"},
		// Datacenter -> vm Folder -> VirtualMachine
		{"DC0", "vm", "DC0_C0_RP0_VM0"},
	}

	root := c.ServiceContent.RootFolder

	for _, path := range tests {
		parent := root
		ipath := []string{""}

		for _, name := range path {
			ref, err := si.FindChild(ctx, parent, name)
			if err != nil {
				t.Fatal(err)
			}

			if ref == nil {
				t.Fatalf("failed to match %s using %s", name, parent)
			}

			parent = ref.Reference()

			ipath = append(ipath, name)

			iref, err := si.FindByInventoryPath(ctx, strings.Join(ipath, "/"))
			if err != nil {
				t.Fatal(err)
			}

			if iref.Reference() != ref.Reference() {
				t.Errorf("%s != %s", iref, ref)
			}
		}
	}

	ref, err := si.FindChild(ctx, root, "enoent")
	if err != nil {
		t.Fatal(err)
	}

	if ref != nil {
		t.Error("unexpected match")
	}

	root.Value = "enoent"
	_, err = si.FindChild(ctx, root, "enoent")
	if err == nil {
		t.Error("expected error")
	}

	if _, ok := soap.ToSoapFault(err).VimFault().(types.ManagedObjectNotFound); !ok {
		t.Error("expected ManagedObjectNotFound fault")
	}

	for _, path := range []string{"", "/", "/enoent"} {
		ref, err := si.FindByInventoryPath(ctx, path)
		if err != nil {
			t.Fatal(err)
		}

		if ref != nil {
			t.Error("unexpected match")
		}
	}
}
