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
	"github.com/vmware/govmomi/object"
	"github.com/vmware/govmomi/property"
	"github.com/vmware/govmomi/view"
	"github.com/vmware/govmomi/vim25/mo"
	"github.com/vmware/govmomi/vim25/types"
)

func TestContainerViewVPX(t *testing.T) {
	ctx := context.Background()

	m := VPX()
	m.Datacenter = 3
	m.Folder = 2
	m.Pool = 1
	m.App = 1
	m.Pod = 1

	defer m.Remove()

	err := m.Create()
	if err != nil {
		t.Fatal(err)
	}

	s := m.Service.NewServer()
	defer s.Close()

	c, err := govmomi.NewClient(ctx, s.URL, true)
	if err != nil {
		t.Fatal(err)
	}

	v := view.NewManager(c.Client)
	root := c.Client.ServiceContent.RootFolder

	// test container type validation
	_, err = v.CreateContainerView(ctx, v.Reference(), nil, false)
	if err == nil {
		t.Fatal("expected error")
	}

	// test container value validation
	_, err = v.CreateContainerView(ctx, types.ManagedObjectReference{Value: "enoent"}, nil, false)
	if err == nil {
		t.Fatal("expected error")
	}

	// test types validation
	_, err = v.CreateContainerView(ctx, root, []string{"enoent"}, false)
	if err == nil {
		t.Fatal("expected error")
	}

	vapp := object.NewVirtualApp(c.Client, Map.Any("VirtualApp").Reference())

	count := m.Count()

	tests := []struct {
		root    types.ManagedObjectReference
		recurse bool
		kinds   []string
		expect  int
	}{
		{root, false, nil, m.Datacenter - m.Folder + m.Folder},
		{root, true, nil, count.total - 1},                             // not including the root Folder
		{root, true, []string{"ManagedEntity"}, count.total - 1},       // not including the root Folder
		{root, true, []string{"Folder"}, count.Folder + count.Pod - 1}, // not including the root Folder
		{root, false, []string{"HostSystem"}, 0},
		{root, true, []string{"HostSystem"}, count.Host},
		{root, false, []string{"Datacenter"}, m.Datacenter - m.Folder},
		{root, true, []string{"Datacenter"}, count.Datacenter},
		{root, true, []string{"Datastore"}, count.Datastore},
		{root, true, []string{"VirtualMachine"}, count.Machine},
		{root, true, []string{"ResourcePool"}, count.Pool + count.App},
		{root, true, []string{"VirtualApp"}, count.App},
		{vapp.Reference(), true, []string{"VirtualMachine"}, m.Machine},
		{root, true, []string{"ClusterComputeResource"}, count.Cluster},
		{root, true, []string{"ComputeResource"}, (m.Cluster + m.Host) * m.Datacenter},
		{root, true, []string{"DistributedVirtualSwitch"}, count.Datacenter},
		{root, true, []string{"DistributedVirtualPortgroup"}, count.Portgroup},
		{root, true, []string{"Network"}, count.Portgroup + m.Datacenter},
		{root, true, []string{"OpaqueNetwork"}, 0},
		{root, true, []string{"StoragePod"}, m.Pod * m.Datacenter},
	}

	pc := property.DefaultCollector(c.Client)

	mvm := Map.ViewManager()

	for i, test := range tests {
		cv, err := v.CreateContainerView(ctx, test.root, test.kinds, test.recurse)
		if err != nil {
			t.Fatal(err)
		}

		if len(mvm.ViewList) != 1 {
			t.Errorf("ViewList=%s", mvm.ViewList)
		}

		var mcv mo.ContainerView
		err = pc.RetrieveOne(ctx, cv.Reference(), nil, &mcv)
		if err != nil {
			t.Fatal(err)
		}

		n := len(mcv.View)

		if n != test.expect {
			t.Errorf("%d: %d != %d", i, n, test.expect)
		}

		err = cv.Destroy(ctx)
		if err != nil {
			t.Fatal(err)
		}

		if len(mvm.ViewList) != 0 {
			t.Errorf("ViewList=%s", mvm.ViewList)
		}
	}
}
