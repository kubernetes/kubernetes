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
	"log"
	"reflect"
	"sync"
	"testing"
	"time"

	"github.com/vmware/govmomi"
	"github.com/vmware/govmomi/find"
	"github.com/vmware/govmomi/object"
	"github.com/vmware/govmomi/property"
	"github.com/vmware/govmomi/simulator/esx"
	"github.com/vmware/govmomi/simulator/vpx"
	"github.com/vmware/govmomi/view"
	"github.com/vmware/govmomi/vim25/methods"
	"github.com/vmware/govmomi/vim25/mo"
	"github.com/vmware/govmomi/vim25/soap"
	"github.com/vmware/govmomi/vim25/types"
)

func TestRetrieveProperties(t *testing.T) {
	configs := []struct {
		folder  mo.Folder
		content types.ServiceContent
		dc      *types.ManagedObjectReference
	}{
		{esx.RootFolder, esx.ServiceContent, &esx.Datacenter.Self},
		{vpx.RootFolder, vpx.ServiceContent, nil},
	}

	for _, config := range configs {
		s := New(NewServiceInstance(config.content, config.folder))

		ts := s.NewServer()
		defer ts.Close()

		ctx := context.Background()

		client, err := govmomi.NewClient(ctx, ts.URL, true)
		if err != nil {
			t.Fatal(err)
		}

		if config.dc == nil {
			dc, cerr := object.NewRootFolder(client.Client).CreateDatacenter(ctx, "dc1")
			if cerr != nil {
				t.Fatal(cerr)
			}
			ref := dc.Reference()
			config.dc = &ref
		}

		// Retrieve a specific property
		f := mo.Folder{}
		err = client.RetrieveOne(ctx, config.content.RootFolder, []string{"name"}, &f)
		if err != nil {
			t.Fatal(err)
		}

		if f.Name != config.folder.Name {
			t.Fail()
		}

		// Retrieve all properties
		f = mo.Folder{}
		err = client.RetrieveOne(ctx, config.content.RootFolder, nil, &f)
		if err != nil {
			t.Fatal(err)
		}

		if f.Name != config.folder.Name {
			t.Fatalf("'%s' vs '%s'", f.Name, config.folder.Name)
		}

		// Retrieve an ArrayOf property
		f = mo.Folder{}
		err = client.RetrieveOne(ctx, config.content.RootFolder, []string{"childEntity"}, &f)
		if err != nil {
			t.Fatal(err)
		}

		if len(f.ChildEntity) != 1 {
			t.Fail()
		}

		es, err := mo.Ancestors(ctx, client.Client, config.content.PropertyCollector, config.content.RootFolder)
		if err != nil {
			t.Fatal(err)
		}

		if len(es) != 1 {
			t.Fail()
		}

		finder := find.NewFinder(client.Client, false)
		dc, err := finder.DatacenterOrDefault(ctx, "")
		if err != nil {
			t.Fatal(err)
		}

		if dc.Reference() != *config.dc {
			t.Fail()
		}

		finder.SetDatacenter(dc)

		es, err = mo.Ancestors(ctx, client.Client, config.content.PropertyCollector, dc.Reference())
		if err != nil {
			t.Fatal(err)
		}

		expect := map[string]types.ManagedObjectReference{
			"Folder":     config.folder.Reference(),
			"Datacenter": dc.Reference(),
		}

		if len(es) != len(expect) {
			t.Fail()
		}

		for _, e := range es {
			ref := e.Reference()
			if r, ok := expect[ref.Type]; ok {
				if r != ref {
					t.Errorf("%#v vs %#v", r, ref)
				}
			} else {
				t.Errorf("unexpected object %#v", e.Reference())
			}
		}

		// finder tests
		ls, err := finder.ManagedObjectListChildren(ctx, ".")
		if err != nil {
			t.Error(err)
		}

		folders, err := dc.Folders(ctx)
		if err != nil {
			t.Fatal(err)
		}

		// Validated name properties are recursively retrieved for the datacenter and its folder children
		ipaths := []string{
			folders.VmFolder.InventoryPath,
			folders.HostFolder.InventoryPath,
			folders.DatastoreFolder.InventoryPath,
			folders.NetworkFolder.InventoryPath,
		}

		var lpaths []string
		for _, p := range ls {
			lpaths = append(lpaths, p.Path)
		}

		if !reflect.DeepEqual(ipaths, lpaths) {
			t.Errorf("%#v != %#v\n", ipaths, lpaths)
		}

		// We have no VMs, expect NotFoundError
		_, err = finder.VirtualMachineList(ctx, "*")
		if err == nil {
			t.Error("expected error")
		} else {
			if _, ok := err.(*find.NotFoundError); !ok {
				t.Error(err)
			}
		}

		// Retrieve a missing property
		mdc := mo.Datacenter{}
		err = client.RetrieveOne(ctx, dc.Reference(), []string{"enoent"}, &mdc)
		if err == nil {
			t.Error("expected error")
		} else {
			switch fault := soap.ToVimFault(err).(type) {
			case *types.InvalidProperty:
				// ok
			default:
				t.Errorf("unexpected fault: %#v", fault)
			}
		}

		// Retrieve a nested property
		Map.Get(dc.Reference()).(*Datacenter).Configuration.DefaultHardwareVersionKey = "foo"
		mdc = mo.Datacenter{}
		err = client.RetrieveOne(ctx, dc.Reference(), []string{"configuration.defaultHardwareVersionKey"}, &mdc)
		if err != nil {
			t.Fatal(err)
		}
		if mdc.Configuration.DefaultHardwareVersionKey != "foo" {
			t.Fail()
		}

		// Retrieve a missing nested property
		mdc = mo.Datacenter{}
		err = client.RetrieveOne(ctx, dc.Reference(), []string{"configuration.enoent"}, &mdc)
		if err == nil {
			t.Error("expected error")
		} else {
			switch fault := soap.ToVimFault(err).(type) {
			case *types.InvalidProperty:
				// ok
			default:
				t.Errorf("unexpected fault: %#v", fault)
			}
		}

		// Retrieve an empty property
		err = client.RetrieveOne(ctx, dc.Reference(), []string{""}, &mdc)
		if err != nil {
			t.Error(err)
		}

		// Expect ManagedObjectNotFoundError
		Map.Remove(dc.Reference())
		err = client.RetrieveOne(ctx, dc.Reference(), []string{"name"}, &mdc)
		if err == nil {
			t.Fatal("expected error")
		}
	}
}

func TestWaitForUpdates(t *testing.T) {
	folder := esx.RootFolder
	s := New(NewServiceInstance(esx.ServiceContent, folder))

	ts := s.NewServer()
	defer ts.Close()

	ctx := context.Background()

	c, err := govmomi.NewClient(ctx, ts.URL, true)
	if err != nil {
		t.Fatal(err)
	}

	updates := make(chan bool)
	cb := func(once bool) func([]types.PropertyChange) bool {
		return func(pc []types.PropertyChange) bool {
			if len(pc) != 1 {
				t.Fail()
			}

			c := pc[0]
			if c.Op != types.PropertyChangeOpAssign {
				t.Fail()
			}
			if c.Name != "name" {
				t.Fail()
			}
			if c.Val.(string) != folder.Name {
				t.Fail()
			}

			if once == false {
				updates <- true
			}
			return once
		}
	}

	pc := property.DefaultCollector(c.Client)
	props := []string{"name"}

	err = property.Wait(ctx, pc, folder.Reference(), props, cb(true))
	if err != nil {
		t.Error(err)
	}

	wctx, cancel := context.WithCancel(ctx)
	var wg sync.WaitGroup
	wg.Add(1)
	go func() {
		defer wg.Done()
		_ = property.Wait(wctx, pc, folder.Reference(), props, cb(false))
	}()
	<-updates
	cancel()
	wg.Wait()

	// test object not found
	Map.Remove(folder.Reference())

	err = property.Wait(ctx, pc, folder.Reference(), props, cb(true))
	if err == nil {
		t.Error("expected error")
	}

	// test CancelWaitForUpdates
	p, err := pc.Create(ctx)
	if err != nil {
		t.Fatal(err)
	}

	// test the deprecated WaitForUpdates methods
	_, err = methods.WaitForUpdates(ctx, c.Client, &types.WaitForUpdates{This: p.Reference()})
	if err != nil {
		t.Fatal(err)
	}

	err = p.CancelWaitForUpdates(ctx)
	if err != nil {
		t.Fatal(err)
	}
}

func TestIncrementalWaitForUpdates(t *testing.T) {
	ctx := context.Background()

	m := VPX()

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

	pc := property.DefaultCollector(c.Client)
	obj := Map.Any("VirtualMachine").(*VirtualMachine)
	ref := obj.Reference()
	vm := object.NewVirtualMachine(c.Client, ref)

	tests := []struct {
		name  string
		props []string
	}{
		{"1 field", []string{"runtime.powerState"}},
		{"2 fields", []string{"summary.runtime.powerState", "summary.runtime.bootTime"}},
		{"3 fields", []string{"runtime.powerState", "summary.runtime.powerState", "summary.runtime.bootTime"}},
		{"parent field", []string{"runtime"}},
		{"nested parent field", []string{"summary.runtime"}},
		{"all", nil},
	}

	// toggle power state to generate updates
	state := map[types.VirtualMachinePowerState]func(context.Context) (*object.Task, error){
		types.VirtualMachinePowerStatePoweredOff: vm.PowerOn,
		types.VirtualMachinePowerStatePoweredOn:  vm.PowerOff,
	}

	for i, test := range tests {
		var props []string
		matches := false
		wait := make(chan bool)
		host := obj.Summary.Runtime.Host // add host to filter just to have a different type in the filter
		filter := new(property.WaitFilter).Add(*host, host.Type, nil).Add(ref, ref.Type, test.props)

		go func() {
			perr := property.WaitForUpdates(ctx, pc, filter, func(updates []types.ObjectUpdate) bool {
				if updates[0].Kind == types.ObjectUpdateKindEnter {
					wait <- true
					return false
				}
				for _, update := range updates {
					for _, change := range update.ChangeSet {
						props = append(props, change.Name)
					}
				}

				if test.props == nil {
					// special case to test All flag
					matches = isTrue(filter.Spec.PropSet[0].All) && len(props) > 1

					return matches
				}

				if len(props) > len(test.props) {
					return true
				}

				matches = reflect.DeepEqual(props, test.props)
				return matches
			})

			if perr != nil {
				t.Error(perr)
			}
			wait <- matches
		}()

		<-wait // wait for enter
		_, _ = state[obj.Runtime.PowerState](ctx)
		if !<-wait { // wait for modify
			t.Errorf("%d: updates=%s, expected=%s", i, props, test.props)
		}
	}

	// Test ContainerView + Delete
	v, err := view.NewManager(c.Client).CreateContainerView(ctx, c.Client.ServiceContent.RootFolder, []string{ref.Type}, true)
	if err != nil {
		log.Fatal(err)
	}

	var wg sync.WaitGroup
	wg.Add(1)
	go func() {
		defer wg.Done()

		filter := new(property.WaitFilter).Add(v.Reference(), ref.Type, tests[0].props, v.TraversalSpec())
		perr := property.WaitForUpdates(ctx, pc, filter, func(updates []types.ObjectUpdate) bool {
			for _, update := range updates {
				switch update.Kind {
				case types.ObjectUpdateKindEnter:
					wg.Done()
					return false
				case types.ObjectUpdateKindModify:
				case types.ObjectUpdateKindLeave:
					return update.Obj == vm.Reference()
				}
			}
			return false
		})
		if perr != nil {
			t.Error(perr)
		}
	}()

	wg.Wait() // wait for 1st enter
	wg.Add(1)
	_, _ = vm.PowerOff(ctx)
	_, _ = vm.Destroy(ctx)
	wg.Wait() // wait for Delete to be reported
}

func TestWaitForUpdatesOneUpdateCalculation(t *testing.T) {
	/*
	 * In this test, we use WaitForUpdatesEx in non-blocking way
	 * by setting the MaxWaitSeconds to 0.
	 * We filter on 'runtime.powerState'
	 * Once we get the first 'enter' update, we change the
	 * power state of the VM to generate a 'modify' update.
	 * Once we get the modify, we can stop.
	 */
	ctx := context.Background()

	m := VPX()
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

	wait := make(chan bool)
	pc := property.DefaultCollector(c.Client)
	obj := Map.Any("VirtualMachine").(*VirtualMachine)
	ref := obj.Reference()
	vm := object.NewVirtualMachine(c.Client, ref)
	filter := new(property.WaitFilter).Add(ref, ref.Type, []string{"runtime.powerState"})
	// WaitOptions.maxWaitSeconds:
	// A value of 0 causes WaitForUpdatesEx to do one update calculation and return any results.
	filter.Options = &types.WaitOptions{
		MaxWaitSeconds: types.NewInt32(0),
	}
	// toggle power state to generate updates
	state := map[types.VirtualMachinePowerState]func(context.Context) (*object.Task, error){
		types.VirtualMachinePowerStatePoweredOff: vm.PowerOn,
		types.VirtualMachinePowerStatePoweredOn:  vm.PowerOff,
	}

	err = pc.CreateFilter(ctx, filter.CreateFilter)
	if err != nil {
		t.Fatal(err)
	}

	req := types.WaitForUpdatesEx{
		This:    pc.Reference(),
		Options: filter.Options,
	}

	go func() {
		for {
			res, err := methods.WaitForUpdatesEx(ctx, c.Client, &req)
			if err != nil {
				if ctx.Err() == context.Canceled {
					werr := pc.CancelWaitForUpdates(context.Background())
					t.Error(werr)
					return
				}
				t.Error(err)
				return
			}

			set := res.Returnval
			if set == nil {
				// Retry if the result came back empty
				// Thats a normal case when MaxWaitSeconds is set to 0.
				// It means we have no updates for now
				time.Sleep(500 * time.Millisecond)
				continue
			}

			req.Version = set.Version

			for _, fs := range set.FilterSet {
				// We expect the enter of VM first
				if fs.ObjectSet[0].Kind == types.ObjectUpdateKindEnter {
					wait <- true
					// Keep going
					continue
				}

				// We also expect a modify due to the power state change
				if fs.ObjectSet[0].Kind == types.ObjectUpdateKindModify {
					wait <- true
					// Now we can return to stop the routine
					return
				}
			}
		}
	}()

	// wait for the enter update.
	<-wait

	// Now change the VM power state, to generate a modify update
	_, err = state[obj.Runtime.PowerState](ctx)
	if err != nil {
		t.Error(err)
	}

	// wait for the modify update.
	<-wait
}

func TestPropertyCollectorWithUnsetValues(t *testing.T) {
	ctx := context.Background()

	m := VPX()

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

	pc := property.DefaultCollector(client.Client)

	vm := Map.Any("VirtualMachine")
	vmRef := vm.Reference()

	propSets := [][]string{
		{"parentVApp"},                   // unset string by default (not returned by RetrievePropertiesEx)
		{"rootSnapshot"},                 // unset VirtualMachineSnapshot[] by default (returned by RetrievePropertiesEx)
		{"config.networkShaper.enabled"}, // unset at config.networkShaper level by default (not returned by RetrievePropertiesEx)
		{"parentVApp", "rootSnapshot", "config.networkShaper.enabled"},         // (not returned by RetrievePropertiesEx)
		{"name", "parentVApp", "rootSnapshot", "config.networkShaper.enabled"}, // (only name returned by RetrievePropertiesEx)
		{"name", "config.guestFullName"},                                       // both set (and returned by RetrievePropertiesEx))
	}

	for _, propSet := range propSets {
		// RetrievePropertiesEx
		var objectContents []types.ObjectContent
		err = pc.RetrieveOne(ctx, vmRef, propSet, &objectContents)
		if err != nil {
			t.Error(err)
		}

		if len(objectContents) != 1 {
			t.Fatalf("len(objectContents) %#v != 1", len(objectContents))
		}
		objectContent := objectContents[0]

		if objectContent.Obj != vmRef {
			t.Fatalf("objectContent.Obj %#v != vmRef %#v", objectContent.Obj, vmRef)
		}

		inRetrieveResponseCount := 0
		for _, prop := range propSet {
			_, err := fieldValue(reflect.ValueOf(vm), prop)

			switch err {
			case nil:
				inRetrieveResponseCount++
				found := false
				for _, objProp := range objectContent.PropSet {
					if prop == objProp.Name {
						found = true
						break
					}
				}
				if !found {
					t.Fatalf("prop %#v was not found", prop)
				}
			case errEmptyField:
				continue
			default:
				t.Error(err)
			}
		}

		if len(objectContent.PropSet) != inRetrieveResponseCount {
			t.Fatalf("len(objectContent.PropSet) %#v != inRetrieveResponseCount %#v", len(objectContent.PropSet), inRetrieveResponseCount)
		}

		if len(objectContent.MissingSet) != 0 {
			t.Fatalf("len(objectContent.MissingSet) %#v != 0", len(objectContent.MissingSet))
		}

		// WaitForUpdatesEx
		f := func(once bool) func([]types.PropertyChange) bool {
			return func(pc []types.PropertyChange) bool {
				if len(propSet) != len(pc) {
					t.Fatalf("len(propSet) %#v != len(pc) %#v", len(propSet), len(pc))
				}

				for _, prop := range propSet {
					found := false
					for _, objProp := range pc {
						switch err {
						case nil, errEmptyField:
							if prop == objProp.Name {
								found = true
							}
						default:
							t.Error(err)
						}
						if found {
							break
						}
					}
					if !found {
						t.Fatalf("prop %#v was not found", prop)
					}
				}

				return once
			}
		}

		err = property.Wait(ctx, pc, vmRef, propSet, f(true))
		if err != nil {
			t.Error(err)
		}
	}
}

func TestCollectInterfaceType(t *testing.T) {
	// test that we properly collect an interface type (types.BaseVirtualDevice in this case)
	var config types.VirtualMachineConfigInfo
	config.Hardware.Device = append(config.Hardware.Device, new(types.VirtualFloppy))

	_, err := fieldValue(reflect.ValueOf(&config), "hardware.device")
	if err != nil {
		t.Fatal(err)
	}
}

func TestExtractEmbeddedField(t *testing.T) {
	type YourResourcePool struct {
		mo.ResourcePool
	}

	type MyResourcePool struct {
		YourResourcePool
	}

	x := new(MyResourcePool)

	Map.Put(x)

	obj, ok := getObject(internalContext, x.Reference())
	if !ok {
		t.Error("expected obj")
	}

	if obj.Type() != reflect.ValueOf(new(mo.ResourcePool)).Elem().Type() {
		t.Errorf("unexpected type=%s", obj.Type().Name())
	}
}

func TestPropertyCollectorFold(t *testing.T) {
	ctx := context.Background()

	m := VPX()

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

	cluster := Map.Any("ClusterComputeResource")
	compute := Map.Any("ComputeResource")

	// Test that we fold duplicate properties (rbvmomi depends on this)
	var content []types.ObjectContent
	err = client.Retrieve(ctx, []types.ManagedObjectReference{cluster.Reference()}, []string{"name", "name"}, &content)
	if err != nil {
		t.Fatal(err)
	}
	if len(content) != 1 {
		t.Fatalf("len(content)=%d", len(content))
	}
	if len(content[0].PropSet) != 1 {
		t.Fatalf("len(PropSet)=%d", len(content[0].PropSet))
	}

	// Test that we fold embedded type properties (rbvmomi depends on this)
	req := types.RetrieveProperties{
		SpecSet: []types.PropertyFilterSpec{
			{
				PropSet: []types.PropertySpec{
					{
						DynamicData: types.DynamicData{},
						Type:        "ComputeResource",
						PathSet:     []string{"name"},
					},
					{
						DynamicData: types.DynamicData{},
						Type:        "ClusterComputeResource",
						PathSet:     []string{"name"},
					},
				},
				ObjectSet: []types.ObjectSpec{
					{
						Obj: compute.Reference(),
					},
					{
						Obj: cluster.Reference(),
					},
				},
			},
		},
	}

	pc := client.PropertyCollector()

	res, err := pc.RetrieveProperties(ctx, req)
	if err != nil {
		t.Fatal(err)
	}

	content = res.Returnval

	if len(content) != 2 {
		t.Fatalf("len(content)=%d", len(content))
	}

	for _, oc := range content {
		if len(oc.PropSet) != 1 {
			t.Errorf("%s len(PropSet)=%d", oc.Obj, len(oc.PropSet))
		}
	}
}

func TestPropertyCollectorInvalidSpecName(t *testing.T) {
	obj := Map.Put(new(Folder))
	obj.(*Folder).putChild(new(Folder))

	pc := &PropertyCollector{}

	req := types.RetrievePropertiesEx{
		SpecSet: []types.PropertyFilterSpec{
			{
				PropSet: []types.PropertySpec{
					{
						Type:    obj.Reference().Type,
						PathSet: []string{"name"},
					},
				},
				ObjectSet: []types.ObjectSpec{
					{
						Obj: obj.Reference(),
						SelectSet: []types.BaseSelectionSpec{
							&types.TraversalSpec{
								Type: "Folder",
								Path: "childEntity",
								SelectSet: []types.BaseSelectionSpec{
									&types.SelectionSpec{
										Name: "enoent",
									},
								},
							},
						},
					},
				},
			},
		},
	}

	_, err := pc.collect(internalContext, &req)
	if err == nil {
		t.Fatal("expected error")
	}

	if _, ok := err.(*types.InvalidArgument); !ok {
		t.Errorf("unexpected fault: %#v", err)
	}
}

func TestPropertyCollectorRecursiveSelectSet(t *testing.T) {
	ctx := context.Background()

	m := VPX()

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

	// Capture of PowerCLI's Get-VM spec
	req := types.RetrieveProperties{
		SpecSet: []types.PropertyFilterSpec{
			{
				PropSet: []types.PropertySpec{
					{
						Type: "VirtualMachine",
						PathSet: []string{
							"runtime.host",
							"parent",
							"resourcePool",
							"resourcePool",
							"datastore",
							"config.swapPlacement",
							"config.version",
							"config.instanceUuid",
							"config.guestId",
							"config.annotation",
							"summary.storage.committed",
							"summary.storage.uncommitted",
							"summary.storage.committed",
							"config.template",
						},
					},
				},
				ObjectSet: []types.ObjectSpec{
					{
						Obj:  client.ServiceContent.RootFolder,
						Skip: types.NewBool(false),
						SelectSet: []types.BaseSelectionSpec{
							&types.TraversalSpec{
								SelectionSpec: types.SelectionSpec{
									Name: "traverseFolders",
								},
								Type: "Folder",
								Path: "childEntity",
								Skip: types.NewBool(true),
								SelectSet: []types.BaseSelectionSpec{
									&types.TraversalSpec{
										Type: "HostSystem",
										Path: "vm",
										Skip: types.NewBool(false),
									},
									&types.TraversalSpec{
										Type: "ComputeResource",
										Path: "host",
										Skip: types.NewBool(true),
										SelectSet: []types.BaseSelectionSpec{
											&types.TraversalSpec{
												Type: "HostSystem",
												Path: "vm",
												Skip: types.NewBool(false),
											},
										},
									},
									&types.TraversalSpec{
										Type: "Datacenter",
										Path: "hostFolder",
										Skip: types.NewBool(true),
										SelectSet: []types.BaseSelectionSpec{
											&types.SelectionSpec{
												Name: "traverseFolders",
											},
										},
									},
									&types.SelectionSpec{
										Name: "traverseFolders",
									},
								},
							},
						},
					},
				},
			},
		},
	}

	pc := client.PropertyCollector()

	res, err := pc.RetrieveProperties(ctx, req)
	if err != nil {
		t.Fatal(err)
	}

	content := res.Returnval

	count := m.Count()

	if len(content) != count.Machine {
		t.Fatalf("len(content)=%d", len(content))
	}
}

func TestPropertyCollectorSelectionSpec(t *testing.T) {
	ctx := context.Background()

	m := VPX()

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

	// Capture of PowerCLI's Start-VM spec
	// Differs from the norm in that:
	// 1) Named SelectionSpec before TraversalSpec is defined
	// 2) Skip=false for all mo types, but PropSet only has Type VirtualMachine
	req := types.RetrieveProperties{
		SpecSet: []types.PropertyFilterSpec{
			{
				PropSet: []types.PropertySpec{
					{
						Type:    "VirtualMachine",
						All:     types.NewBool(false),
						PathSet: []string{"name", "parent", "runtime.host", "config.template"},
					},
				},
				ObjectSet: []types.ObjectSpec{
					{
						Obj:  client.ServiceContent.RootFolder,
						Skip: types.NewBool(false),
						SelectSet: []types.BaseSelectionSpec{
							&types.TraversalSpec{
								SelectionSpec: types.SelectionSpec{
									Name: "folderTraversalSpec",
								},
								Type: "Folder",
								Path: "childEntity",
								Skip: types.NewBool(false),
								SelectSet: []types.BaseSelectionSpec{
									&types.SelectionSpec{
										Name: "computeResourceRpTraversalSpec",
									},
									&types.SelectionSpec{
										Name: "computeResourceHostTraversalSpec",
									},
									&types.SelectionSpec{
										Name: "folderTraversalSpec",
									},
									&types.SelectionSpec{
										Name: "datacenterHostTraversalSpec",
									},
									&types.SelectionSpec{
										Name: "hostVmTraversalSpec",
									},
									&types.SelectionSpec{
										Name: "resourcePoolVmTraversalSpec",
									},
									&types.SelectionSpec{
										Name: "hostRpTraversalSpec",
									},
									&types.SelectionSpec{
										Name: "datacenterVmTraversalSpec",
									},
									&types.SelectionSpec{
										Name: "datastoreVmTraversalSpec",
									},
									&types.SelectionSpec{
										Name: "datacenterDatastoreTraversalSpec",
									},
									&types.SelectionSpec{
										Name: "vappTraversalSpec",
									},
									&types.SelectionSpec{
										Name: "datacenterNetworkTraversalSpec",
									},
								},
							},
							&types.TraversalSpec{
								SelectionSpec: types.SelectionSpec{
									Name: "computeResourceHostTraversalSpec",
								},
								Type: "ComputeResource",
								Path: "host",
								Skip: types.NewBool(false),
								SelectSet: []types.BaseSelectionSpec{
									&types.SelectionSpec{
										Name: "hostVmTraversalSpec",
									},
								},
							},
							&types.TraversalSpec{
								SelectionSpec: types.SelectionSpec{
									Name: "computeResourceRpTraversalSpec",
								},
								Type: "ComputeResource",
								Path: "resourcePool",
								Skip: types.NewBool(false),
								SelectSet: []types.BaseSelectionSpec{
									&types.SelectionSpec{
										Name: "resourcePoolTraversalSpec",
									},
									&types.SelectionSpec{
										Name: "vappTraversalSpec",
									},
								},
							},
							&types.TraversalSpec{
								SelectionSpec: types.SelectionSpec{
									Name: "datacenterHostTraversalSpec",
								},
								Type: "Datacenter",
								Path: "hostFolder",
								Skip: types.NewBool(false),
								SelectSet: []types.BaseSelectionSpec{
									&types.SelectionSpec{
										Name: "folderTraversalSpec",
									},
								},
							},
							&types.TraversalSpec{
								SelectionSpec: types.SelectionSpec{
									Name: "resourcePoolTraversalSpec",
								},
								Type: "ResourcePool",
								Path: "resourcePool",
								Skip: types.NewBool(false),
								SelectSet: []types.BaseSelectionSpec{
									&types.SelectionSpec{
										Name: "resourcePoolTraversalSpec",
									},
									&types.SelectionSpec{
										Name: "resourcePoolVmTraversalSpec",
									},
								},
							},
							&types.TraversalSpec{
								SelectionSpec: types.SelectionSpec{
									Name: "hostVmTraversalSpec",
								},
								Type: "HostSystem",
								Path: "vm",
								Skip: types.NewBool(false),
								SelectSet: []types.BaseSelectionSpec{
									&types.SelectionSpec{
										Name: "folderTraversalSpec",
									},
								},
							},
							&types.TraversalSpec{
								SelectionSpec: types.SelectionSpec{
									Name: "datacenterVmTraversalSpec",
								},
								Type: "Datacenter",
								Path: "vmFolder",
								Skip: types.NewBool(false),
								SelectSet: []types.BaseSelectionSpec{
									&types.SelectionSpec{
										Name: "folderTraversalSpec",
									},
								},
							},
							&types.TraversalSpec{
								SelectionSpec: types.SelectionSpec{
									Name: "resourcePoolVmTraversalSpec",
								},
								Type: "ResourcePool",
								Path: "vm",
								Skip: types.NewBool(false),
							},
							&types.TraversalSpec{
								SelectionSpec: types.SelectionSpec{
									Name: "datastoreVmTraversalSpec",
								},
								Type: "Datastore",
								Path: "vm",
								Skip: types.NewBool(false),
							},
							&types.TraversalSpec{
								SelectionSpec: types.SelectionSpec{
									Name: "datacenterDatastoreTraversalSpec",
								},
								Type: "Datacenter",
								Path: "datastoreFolder",
								Skip: types.NewBool(false),
								SelectSet: []types.BaseSelectionSpec{
									&types.SelectionSpec{
										Name: "folderTraversalSpec",
									},
								},
							},
							&types.TraversalSpec{
								SelectionSpec: types.SelectionSpec{
									Name: "vappTraversalSpec",
								},
								Type: "VirtualApp",
								Path: "resourcePool",
								Skip: types.NewBool(false),
								SelectSet: []types.BaseSelectionSpec{
									&types.SelectionSpec{
										Name: "vappTraversalSpec",
									},
									&types.SelectionSpec{
										Name: "resourcePoolTraversalSpec",
									},
									&types.SelectionSpec{
										Name: "resourcePoolVmTraversalSpec",
									},
								},
							},
							&types.TraversalSpec{
								SelectionSpec: types.SelectionSpec{
									Name: "vappVmTraversalSpec",
								},
								Type: "VirtualApp",
								Path: "vm",
								Skip: types.NewBool(false),
							},
							&types.TraversalSpec{
								SelectionSpec: types.SelectionSpec{
									Name: "distributedSwitchHostTraversalSpec",
								},
								Type: "DistributedVirtualSwitch",
								Path: "summary.hostMember",
								Skip: types.NewBool(false),
							},
							&types.TraversalSpec{
								SelectionSpec: types.SelectionSpec{
									Name: "distributedSwitchVmTraversalSpec",
								},
								Type: "DistributedVirtualSwitch",
								Path: "summary.vm",
								Skip: types.NewBool(false),
							},
							&types.TraversalSpec{
								SelectionSpec: types.SelectionSpec{
									Name: "datacenterNetworkTraversalSpec",
								},
								Type: "Datacenter",
								Path: "networkFolder",
								Skip: types.NewBool(false),
								SelectSet: []types.BaseSelectionSpec{
									&types.SelectionSpec{
										Name: "folderTraversalSpec",
									},
								},
							},
							&types.TraversalSpec{
								SelectionSpec: types.SelectionSpec{
									Name: "hostRpTraversalSpec",
								},
								Type: "HostSystem",
								Path: "parent",
								Skip: types.NewBool(false),
								SelectSet: []types.BaseSelectionSpec{
									&types.SelectionSpec{
										Name: "computeResourceRpTraversalSpec",
									},
								},
							},
						},
					},
				},
			},
		},
	}

	pc := client.PropertyCollector()

	res, err := pc.RetrieveProperties(ctx, req)
	if err != nil {
		t.Fatal(err)
	}

	content := res.Returnval
	count := m.Count()

	if len(content) != count.Machine {
		t.Fatalf("len(content)=%d", len(content))
	}
}

func TestIssue945(t *testing.T) {
	// pyvsphere request
	xml := `<?xml version="1.0" encoding="UTF-8"?>
<SOAP-ENV:Envelope xmlns:SOAP-ENV="http://schemas.xmlsoap.org/soap/envelope/" xmlns:SOAP-ENC="http://schemas.xmlsoap.org/soap/encoding/" xmlns:ZSI="http://www.zolera.com/schemas/ZSI/" xmlns:soapenc="http://schemas.xmlsoap.org/soap/encoding/" xmlns:soapenv="http://schemas.xmlsoap.org/soap/envelope/" xmlns:xsd="http://www.w3.org/2001/XMLSchema" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
   <SOAP-ENV:Header />
   <SOAP-ENV:Body xmlns:ns1="urn:vim25">
      <ns1:RetrievePropertiesEx xsi:type="ns1:RetrievePropertiesExRequestType">
         <ns1:_this type="PropertyCollector">propertyCollector</ns1:_this>
         <ns1:specSet>
            <ns1:propSet>
               <ns1:type>VirtualMachine</ns1:type>
               <ns1:pathSet>name</ns1:pathSet>
            </ns1:propSet>
            <ns1:objectSet>
               <ns1:obj type="Folder">group-d1</ns1:obj>
               <ns1:skip>false</ns1:skip>
               <ns1:selectSet xsi:type="ns1:TraversalSpec">
                  <ns1:name>visitFolders</ns1:name>
                  <ns1:type>Folder</ns1:type>
                  <ns1:path>childEntity</ns1:path>
                  <ns1:skip>false</ns1:skip>
                  <ns1:selectSet>
                     <ns1:name>visitFolders</ns1:name>
                  </ns1:selectSet>
                  <ns1:selectSet>
                     <ns1:name>dcToHf</ns1:name>
                  </ns1:selectSet>
                  <ns1:selectSet>
                     <ns1:name>dcToVmf</ns1:name>
                  </ns1:selectSet>
                  <ns1:selectSet>
                     <ns1:name>crToH</ns1:name>
                  </ns1:selectSet>
                  <ns1:selectSet>
                     <ns1:name>crToRp</ns1:name>
                  </ns1:selectSet>
                  <ns1:selectSet>
                     <ns1:name>dcToDs</ns1:name>
                  </ns1:selectSet>
                  <ns1:selectSet>
                     <ns1:name>hToVm</ns1:name>
                  </ns1:selectSet>
                  <ns1:selectSet>
                     <ns1:name>rpToVm</ns1:name>
                  </ns1:selectSet>
               </ns1:selectSet>
               <ns1:selectSet xsi:type="ns1:TraversalSpec">
                  <ns1:name>dcToVmf</ns1:name>
                  <ns1:type>Datacenter</ns1:type>
                  <ns1:path>vmFolder</ns1:path>
                  <ns1:skip>false</ns1:skip>
                  <ns1:selectSet>
                     <ns1:name>visitFolders</ns1:name>
                  </ns1:selectSet>
               </ns1:selectSet>
               <ns1:selectSet xsi:type="ns1:TraversalSpec">
                  <ns1:name>dcToDs</ns1:name>
                  <ns1:type>Datacenter</ns1:type>
                  <ns1:path>datastore</ns1:path>
                  <ns1:skip>false</ns1:skip>
                  <ns1:selectSet>
                     <ns1:name>visitFolders</ns1:name>
                  </ns1:selectSet>
               </ns1:selectSet>
               <ns1:selectSet xsi:type="ns1:TraversalSpec">
                  <ns1:name>dcToHf</ns1:name>
                  <ns1:type>Datacenter</ns1:type>
                  <ns1:path>hostFolder</ns1:path>
                  <ns1:skip>false</ns1:skip>
                  <ns1:selectSet>
                     <ns1:name>visitFolders</ns1:name>
                  </ns1:selectSet>
               </ns1:selectSet>
               <ns1:selectSet xsi:type="ns1:TraversalSpec">
                  <ns1:name>crToH</ns1:name>
                  <ns1:type>ComputeResource</ns1:type>
                  <ns1:path>host</ns1:path>
                  <ns1:skip>false</ns1:skip>
               </ns1:selectSet>
               <ns1:selectSet xsi:type="ns1:TraversalSpec">
                  <ns1:name>crToRp</ns1:name>
                  <ns1:type>ComputeResource</ns1:type>
                  <ns1:path>resourcePool</ns1:path>
                  <ns1:skip>false</ns1:skip>
                  <ns1:selectSet>
                     <ns1:name>rpToRp</ns1:name>
                  </ns1:selectSet>
                  <ns1:selectSet>
                     <ns1:name>rpToVm</ns1:name>
                  </ns1:selectSet>
               </ns1:selectSet>
               <ns1:selectSet xsi:type="ns1:TraversalSpec">
                  <ns1:name>rpToRp</ns1:name>
                  <ns1:type>ResourcePool</ns1:type>
                  <ns1:path>resourcePool</ns1:path>
                  <ns1:skip>false</ns1:skip>
                  <ns1:selectSet>
                     <ns1:name>rpToRp</ns1:name>
                  </ns1:selectSet>
                  <ns1:selectSet>
                     <ns1:name>rpToVm</ns1:name>
                  </ns1:selectSet>
               </ns1:selectSet>
               <ns1:selectSet xsi:type="ns1:TraversalSpec">
                  <ns1:name>hToVm</ns1:name>
                  <ns1:type>HostSystem</ns1:type>
                  <ns1:path>vm</ns1:path>
                  <ns1:skip>false</ns1:skip>
                  <ns1:selectSet>
                     <ns1:name>visitFolders</ns1:name>
                  </ns1:selectSet>
               </ns1:selectSet>
               <ns1:selectSet xsi:type="ns1:TraversalSpec">
                  <ns1:name>rpToVm</ns1:name>
                  <ns1:type>ResourcePool</ns1:type>
                  <ns1:path>vm</ns1:path>
                  <ns1:skip>false</ns1:skip>
               </ns1:selectSet>
            </ns1:objectSet>
         </ns1:specSet>
         <ns1:options />
      </ns1:RetrievePropertiesEx>
   </SOAP-ENV:Body>
</SOAP-ENV:Envelope>`

	method, err := UnmarshalBody(defaultMapType, []byte(xml))
	if err != nil {
		t.Fatal(err)
	}

	req := method.Body.(*types.RetrievePropertiesEx)

	ctx := context.Background()

	m := VPX()

	defer m.Remove()

	err = m.Create()
	if err != nil {
		t.Fatal(err)
	}

	s := m.Service.NewServer()
	defer s.Close()

	client, err := govmomi.NewClient(ctx, s.URL, true)
	if err != nil {
		t.Fatal(err)
	}

	res, err := methods.RetrievePropertiesEx(ctx, client, req)
	if err != nil {
		t.Fatal(err)
	}

	content := res.Returnval.Objects
	count := m.Count()

	if len(content) != count.Machine {
		t.Fatalf("len(content)=%d", len(content))
	}
}

func TestPropertyCollectorSession(t *testing.T) { // aka issue-923
	ctx := context.Background()

	m := VPX()

	defer m.Remove()

	err := m.Create()
	if err != nil {
		t.Fatal(err)
	}

	s := m.Service.NewServer()
	defer s.Close()

	u := s.URL.User
	s.URL.User = nil // skip Login()

	c, err := govmomi.NewClient(ctx, s.URL, true)
	if err != nil {
		t.Fatal(err)
	}

	for i := 0; i < 2; i++ {
		if err = c.Login(ctx, u); err != nil {
			t.Fatal(err)
		}

		if err = c.Login(ctx, u); err == nil {
			t.Error("expected Login failure") // Login fails if we already have a session
		}

		pc := property.DefaultCollector(c.Client)
		filter := new(property.WaitFilter).Add(c.ServiceContent.RootFolder, "Folder", []string{"name"})

		if err = pc.CreateFilter(ctx, filter.CreateFilter); err != nil {
			t.Fatal(err)
		}

		res, err := pc.WaitForUpdates(ctx, "")
		if err != nil {
			t.Error(err)
		}

		if len(res.FilterSet) != 1 {
			t.Errorf("len FilterSet=%d", len(res.FilterSet))
		}

		if err = c.Logout(ctx); err != nil {
			t.Fatal(err)
		}
	}
}

func TestPropertyCollectorNoPathSet(t *testing.T) {
	ctx := context.Background()

	m := VPX()
	m.Datacenter = 3
	m.Folder = 2

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

	// request from https://github.com/vmware/govmomi/issues/1199
	req := &types.RetrieveProperties{
		This: client.ServiceContent.PropertyCollector,
		SpecSet: []types.PropertyFilterSpec{
			{
				PropSet: []types.PropertySpec{
					{
						Type:    "Datacenter",
						All:     types.NewBool(false),
						PathSet: nil,
					},
				},
				ObjectSet: []types.ObjectSpec{
					{
						Obj:  client.ServiceContent.RootFolder,
						Skip: types.NewBool(false),
						SelectSet: []types.BaseSelectionSpec{
							&types.TraversalSpec{
								SelectionSpec: types.SelectionSpec{
									Name: "resourcePoolTraversalSpec",
								},
								Type: "ResourcePool",
								Path: "resourcePool",
								Skip: types.NewBool(false),
								SelectSet: []types.BaseSelectionSpec{
									&types.SelectionSpec{
										Name: "resourcePoolTraversalSpec",
									},
									&types.SelectionSpec{
										Name: "resourcePoolVmTraversalSpec",
									},
								},
							},
							&types.TraversalSpec{
								SelectionSpec: types.SelectionSpec{
									Name: "resourcePoolVmTraversalSpec",
								},
								Type:      "ResourcePool",
								Path:      "vm",
								Skip:      types.NewBool(false),
								SelectSet: nil,
							},
							&types.TraversalSpec{
								SelectionSpec: types.SelectionSpec{
									Name: "computeResourceRpTraversalSpec",
								},
								Type: "ComputeResource",
								Path: "resourcePool",
								Skip: types.NewBool(false),
								SelectSet: []types.BaseSelectionSpec{
									&types.SelectionSpec{
										Name: "resourcePoolTraversalSpec",
									},
									&types.SelectionSpec{
										Name: "resourcePoolVmTraversalSpec",
									},
								},
							},
							&types.TraversalSpec{
								SelectionSpec: types.SelectionSpec{
									Name: "computeResourceHostTraversalSpec",
								},
								Type:      "ComputeResource",
								Path:      "host",
								Skip:      types.NewBool(false),
								SelectSet: nil,
							},
							&types.TraversalSpec{
								SelectionSpec: types.SelectionSpec{
									Name: "datacenterVmTraversalSpec",
								},
								Type: "Datacenter",
								Path: "vmFolder",
								Skip: types.NewBool(false),
								SelectSet: []types.BaseSelectionSpec{
									&types.SelectionSpec{
										Name: "folderTraversalSpec",
									},
								},
							},
							&types.TraversalSpec{
								SelectionSpec: types.SelectionSpec{
									Name: "datacenterHostTraversalSpec",
								},
								Type: "Datacenter",
								Path: "hostFolder",
								Skip: types.NewBool(false),
								SelectSet: []types.BaseSelectionSpec{
									&types.SelectionSpec{
										Name: "folderTraversalSpec",
									},
								},
							},
							&types.TraversalSpec{
								SelectionSpec: types.SelectionSpec{
									Name: "hostVmTraversalSpec",
								},
								Type: "HostSystem",
								Path: "vm",
								Skip: types.NewBool(false),
								SelectSet: []types.BaseSelectionSpec{
									&types.SelectionSpec{
										Name: "folderTraversalSpec",
									},
								},
							},
							&types.TraversalSpec{
								SelectionSpec: types.SelectionSpec{
									Name: "datacenterDatastoreTraversalSpec",
								},
								Type: "Datacenter",
								Path: "datastoreFolder",
								Skip: types.NewBool(false),
								SelectSet: []types.BaseSelectionSpec{
									&types.SelectionSpec{
										Name: "folderTraversalSpec",
									},
								},
							},
							&types.TraversalSpec{
								SelectionSpec: types.SelectionSpec{
									Name: "folderTraversalSpec",
								},
								Type: "Folder",
								Path: "childEntity",
								Skip: types.NewBool(false),
								SelectSet: []types.BaseSelectionSpec{
									&types.SelectionSpec{
										Name: "folderTraversalSpec",
									},
									&types.SelectionSpec{
										Name: "datacenterHostTraversalSpec",
									},
									&types.SelectionSpec{
										Name: "datacenterVmTraversalSpec",
									},
									&types.SelectionSpec{
										Name: "computeResourceRpTraversalSpec",
									},
									&types.SelectionSpec{
										Name: "computeResourceHostTraversalSpec",
									},
									&types.SelectionSpec{
										Name: "hostVmTraversalSpec",
									},
									&types.SelectionSpec{
										Name: "resourcePoolVmTraversalSpec",
									},
									&types.SelectionSpec{
										Name: "datacenterDatastoreTraversalSpec",
									},
								},
							},
						},
					},
				},
				ReportMissingObjectsInResults: (*bool)(nil),
			},
		},
	}

	res, err := methods.RetrieveProperties(ctx, client, req)
	if err != nil {
		t.Fatal(err)
	}

	content := res.Returnval
	count := m.Count()

	if len(content) != count.Datacenter {
		t.Fatalf("len(content)=%d", len(content))
	}
}
