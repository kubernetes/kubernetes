/*
Copyright (c) 2018 VMware, Inc. All Rights Reserved.

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
	"fmt"
	"sync"
	"testing"
	"time"

	"github.com/vmware/govmomi"
	"github.com/vmware/govmomi/event"
	"github.com/vmware/govmomi/find"
	"github.com/vmware/govmomi/property"
	"github.com/vmware/govmomi/view"
	"github.com/vmware/govmomi/vim25/types"
)

func TestRace(t *testing.T) {
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

	content := c.Client.ServiceContent

	wctx, cancel := context.WithCancel(ctx)
	var wg, collectors sync.WaitGroup

	nevents := -1
	em := event.NewManager(c.Client)

	wg.Add(1)
	collectors.Add(1)
	go func() {
		defer collectors.Done()

		werr := em.Events(wctx, []types.ManagedObjectReference{content.RootFolder}, 50, true, false,
			func(_ types.ManagedObjectReference, e []types.BaseEvent) error {
				if nevents == -1 {
					wg.Done() // make sure we are called at least once before cancel() below
					nevents = 0
				}

				nevents += len(e)
				return nil
			})
		if werr != nil {
			t.Error(werr)
		}
	}()

	ntasks := -1
	tv, err := view.NewManager(c.Client).CreateTaskView(ctx, content.TaskManager)
	if err != nil {
		t.Fatal(err)
	}

	wg.Add(1)
	collectors.Add(1)
	go func() {
		defer collectors.Done()

		werr := tv.Collect(ctx, func(tasks []types.TaskInfo) {
			if ntasks == -1 {
				wg.Done() // make sure we are called at least once before cancel() below
				ntasks = 0
			}
			ntasks += len(tasks)
		})
		if werr != nil {
			t.Error(werr)
		}
	}()

	for i := 0; i < 2; i++ {
		spec := types.VirtualMachineConfigSpec{
			Name:    fmt.Sprintf("race-test-%d", i),
			GuestId: string(types.VirtualMachineGuestOsIdentifierOtherGuest),
			Files: &types.VirtualMachineFileInfo{
				VmPathName: "[LocalDS_0]",
			},
		}

		wg.Add(1)
		go func() {
			defer wg.Done()

			finder := find.NewFinder(c.Client, false)
			pc := property.DefaultCollector(c.Client)
			dc, err := finder.DefaultDatacenter(ctx)
			if err != nil {
				t.Error(err)
			}

			finder.SetDatacenter(dc)

			f, err := dc.Folders(ctx)
			if err != nil {
				t.Error(err)
			}

			pool, err := finder.ResourcePool(ctx, "DC0_C0/Resources")
			if err != nil {
				t.Error(err)
			}

			for j := 0; j < 2; j++ {
				cspec := spec // copy spec and give it a unique name
				cspec.Name += fmt.Sprintf("-%d", j)

				wg.Add(1)
				go func() {
					defer wg.Done()

					task, _ := f.VmFolder.CreateVM(ctx, cspec, pool, nil)
					_, terr := task.WaitForResult(ctx, nil)
					if terr != nil {
						t.Error(terr)
					}
				}()
			}

			vms, err := finder.VirtualMachineList(ctx, "*")
			if err != nil {
				t.Error(err)
			}

			for i := range vms {
				props := []string{"runtime.powerState"}
				vm := vms[i]

				wg.Add(1)
				go func() {
					defer wg.Done()

					werr := property.Wait(ctx, pc, vm.Reference(), props, func(changes []types.PropertyChange) bool {
						for _, change := range changes {
							if change.Name != props[0] {
								t.Errorf("unexpected property: %s", change.Name)
							}
							if change.Val == types.VirtualMachinePowerStatePoweredOff {
								return true
							}
						}

						wg.Add(1)
						time.AfterFunc(100*time.Millisecond, func() {
							defer wg.Done()

							task, _ := vm.PowerOff(ctx)
							_ = task.Wait(ctx)
						})

						return false

					})
					if werr != nil {
						if werr != context.Canceled {
							t.Error(werr)
						}
					}
				}()
			}
		}()
	}

	wg.Wait()

	// cancel event and tasks collectors, waiting for them to complete
	cancel()
	collectors.Wait()

	t.Logf("collected %d events, %d tasks", nevents, ntasks)
	if nevents == 0 {
		t.Error("no events collected")
	}
	if ntasks == 0 {
		t.Error("no tasks collected")
	}
}
