/*
Copyright (c) 2015 VMware, Inc. All Rights Reserved.

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

package functional

import (
	"context"
	"fmt"
	"testing"
	"time"

	"github.com/vmware/govmomi/object"
	"github.com/vmware/govmomi/property"
	"github.com/vmware/govmomi/vim25/mo"
	"github.com/vmware/govmomi/vim25/types"
)

func TestIssue242(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	h := NewHelper(t)
	defer h.Teardown()

	h.RequireVirtualCenter()

	df, err := h.Datacenter().Folders(ctx)
	if err != nil {
		t.Fatal(err)
	}

	cr := h.ComputeResource()

	// Get local datastores for compute resource
	dss, err := h.LocalDatastores(ctx, cr)
	if err != nil {
		t.Fatal(err)
	}
	if len(dss) == 0 {
		t.Fatalf("No local datastores")
	}

	// Get root resource pool for compute resource
	rp, err := cr.ResourcePool(ctx)
	if err != nil {
		t.Fatal(err)
	}

	spec := types.VirtualMachineConfigSpec{
		Name:     fmt.Sprintf("govmomi-test-%s", time.Now().Format(time.RFC3339)),
		Files:    &types.VirtualMachineFileInfo{VmPathName: fmt.Sprintf("[%s]", dss[0].Name())},
		NumCPUs:  1,
		MemoryMB: 32,
	}

	// Create new VM
	task, err := df.VmFolder.CreateVM(context.Background(), spec, rp, nil)
	if err != nil {
		t.Fatal(err)
	}

	info, err := task.WaitForResult(context.Background(), nil)
	if err != nil {
		t.Fatal(err)
	}

	vm := object.NewVirtualMachine(h.c, info.Result.(types.ManagedObjectReference))
	defer func() {
		task, err := vm.Destroy(context.Background())
		if err != nil {
			panic(err)
		}
		err = task.Wait(context.Background())
		if err != nil {
			panic(err)
		}
	}()

	// Mark VM as template
	err = vm.MarkAsTemplate(context.Background())
	if err != nil {
		t.Fatal(err)
	}

	// Get "environmentBrowser" property for VM template
	var mvm mo.VirtualMachine
	err = property.DefaultCollector(h.c).RetrieveOne(ctx, vm.Reference(), []string{"environmentBrowser"}, &mvm)
	if err != nil {
		t.Fatal(err)
	}
}
