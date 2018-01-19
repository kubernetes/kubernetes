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

func TestDatacenter(t *testing.T) {
	ctx := context.Background()

	// vCenter model + initial set of objects (cluster, hosts, VMs, network, datastore, etc)
	model := simulator.VPX()

	defer model.Remove()
	err := model.Create()
	if err != nil {
		t.Fatal(err)
	}

	s := model.Service.NewServer()
	defer s.Close()

	avm := simulator.Map.Any("VirtualMachine").(*simulator.VirtualMachine)

	c, err := govmomi.NewClient(ctx, s.URL, true)
	if err != nil {
		t.Fatal(err)
	}

	vc := &VSphereConnection{GoVmomiClient: c}

	_, err = GetDatacenter(ctx, vc, "enoent")
	if err == nil {
		t.Error("expected error")
	}

	dc, err := GetDatacenter(ctx, vc, "DC0")
	if err != nil {
		t.Error(err)
	}

	_, err = dc.GetVMByUUID(ctx, "enoent")
	if err == nil {
		t.Error("expected error")
	}

	_, err = dc.GetVMByUUID(ctx, avm.Summary.Config.Uuid)
	if err != nil {
		t.Error(err)
	}

	_, err = dc.GetVMByPath(ctx, "enoent")
	if err == nil {
		t.Error("expected error")
	}

	vm, err := dc.GetVMByPath(ctx, "/DC0/vm/"+avm.Name)
	if err != nil {
		t.Error(err)
	}

	_, err = dc.GetDatastoreByPath(ctx, "enoent") // invalid format
	if err == nil {
		t.Error("expected error")
	}

	_, err = dc.GetDatastoreByPath(ctx, "[enoent] no/no.vmx")
	if err == nil {
		t.Error("expected error")
	}

	_, err = dc.GetDatastoreByPath(ctx, avm.Summary.Config.VmPathName)
	if err != nil {
		t.Error(err)
	}

	_, err = dc.GetDatastoreByName(ctx, "enoent")
	if err == nil {
		t.Error("expected error")
	}

	ds, err := dc.GetDatastoreByName(ctx, "LocalDS_0")
	if err != nil {
		t.Error(err)
	}

	_, err = dc.GetFolderByPath(ctx, "enoent")
	if err == nil {
		t.Error("expected error")
	}

	_, err = dc.GetFolderByPath(ctx, "/DC0/vm")
	if err != nil {
		t.Error(err)
	}

	_, err = dc.GetVMMoList(ctx, nil, nil)
	if err == nil {
		t.Error("expected error")
	}

	_, err = dc.GetVMMoList(ctx, []*VirtualMachine{vm}, []string{"enoent"}) // invalid property
	if err == nil {
		t.Error("expected error")
	}

	_, err = dc.GetVMMoList(ctx, []*VirtualMachine{vm}, []string{"summary"})
	if err != nil {
		t.Error(err)
	}

	vmdk := ds.Path(avm.Name + "/disk1.vmdk")

	_, err = dc.GetVirtualDiskPage83Data(ctx, vmdk+"-enoent")
	if err == nil {
		t.Error("expected error")
	}

	_, err = dc.GetVirtualDiskPage83Data(ctx, vmdk)
	if err != nil {
		t.Error(err)
	}

	_, err = dc.GetDatastoreMoList(ctx, nil, nil)
	if err == nil {
		t.Error("expected error")
	}

	_, err = dc.GetDatastoreMoList(ctx, []*Datastore{ds}, []string{"enoent"}) // invalid property
	if err == nil {
		t.Error("expected error")
	}

	_, err = dc.GetDatastoreMoList(ctx, []*Datastore{ds}, []string{DatastoreInfoProperty})
	if err != nil {
		t.Error(err)
	}

	nodeVolumes := map[string][]string{
		avm.Name: {"enoent", vmdk},
	}

	attached, err := dc.CheckDisksAttached(ctx, nodeVolumes)
	if err != nil {
		t.Error(err)
	}

	if attached[avm.Name]["enoent"] {
		t.Error("should not be attached")
	}

	if !attached[avm.Name][vmdk] {
		t.Errorf("%s should be attached", vmdk)
	}
}
