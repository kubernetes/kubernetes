/*
Copyright 2016 The Kubernetes Authors.

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

package vclib

import (
	"context"
	"testing"

	"github.com/vmware/govmomi"
	"github.com/vmware/govmomi/simulator"
)

func TestVirtualMachine(t *testing.T) {
	ctx := context.Background()

	model := simulator.VPX()

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

	vc := &VSphereConnection{Client: c.Client}

	dc, err := GetDatacenter(ctx, vc, TestDefaultDatacenter)
	if err != nil {
		t.Error(err)
	}

	folders, err := dc.Folders(ctx)
	if err != nil {
		t.Fatal(err)
	}

	folder, err := dc.GetFolderByPath(ctx, folders.VmFolder.InventoryPath)
	if err != nil {
		t.Fatal(err)
	}

	vms, err := folder.GetVirtualMachines(ctx)
	if err != nil {
		t.Fatal(err)
	}

	if len(vms) == 0 {
		t.Fatal("no VMs")
	}

	for _, vm := range vms {
		all, err := vm.GetAllAccessibleDatastores(ctx)
		if err != nil {
			t.Error(err)
		}
		if len(all) == 0 {
			t.Error("no accessible datastores")
		}

		_, err = vm.GetResourcePool(ctx)
		if err != nil {
			t.Error(err)
		}

		diskPath, err := vm.GetVirtualDiskPath(ctx)
		if err != nil {
			t.Error(err)
		}

		options := &VolumeOptions{SCSIControllerType: PVSCSIControllerType}

		for _, expect := range []bool{true, false} {
			attached, err := vm.IsDiskAttached(ctx, diskPath)
			if err != nil {
				t.Error(err)
			}

			if attached != expect {
				t.Errorf("attached=%t, expected=%t", attached, expect)
			}

			uuid, err := vm.AttachDisk(ctx, diskPath, options)
			if err != nil {
				t.Error(err)
			}
			if uuid == "" {
				t.Error("missing uuid")
			}

			err = vm.DetachDisk(ctx, diskPath)
			if err != nil {
				t.Error(err)
			}
		}

		for _, turnOff := range []bool{true, false} {
			// Turn off for checking if exist return true
			if turnOff {
				_, _ = vm.PowerOff(ctx)
			}

			exist, err := vm.Exists(ctx)
			if err != nil {
				t.Error(err)
			}
			if !exist {
				t.Errorf("exist=%t, expected=%t", exist, true)
			}

			// Turn back on
			if turnOff {
				_, _ = vm.PowerOn(ctx)
			}
		}

		for _, expect := range []bool{true, false} {
			active, err := vm.IsActive(ctx)
			if err != nil {
				t.Error(err)
			}

			if active != expect {
				t.Errorf("active=%t, expected=%t", active, expect)
			}

			if expect {
				// Expecting to hit the error path since the VM is still powered on
				err = vm.DeleteVM(ctx)
				if err == nil {
					t.Error("expected error")
				}
				_, _ = vm.PowerOff(ctx)
				continue
			}

			// Should be able to delete now that VM power is off
			err = vm.DeleteVM(ctx)
			if err != nil {
				t.Error(err)
			}
		}

		// Expecting Exists func to throw error if VM deleted
		_, err = vm.Exists(ctx)
		if err == nil {
			t.Error("expected error")
		}
	}
}
