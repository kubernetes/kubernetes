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
	"testing"

	"github.com/vmware/govmomi/simulator/esx"
	"github.com/vmware/govmomi/simulator/vpx"
	"github.com/vmware/govmomi/vim25/mo"
	"github.com/vmware/govmomi/vim25/types"
)

func TestDatacenterCreateFolders(t *testing.T) {
	// For this test we only want the RootFolder, 1 Datacenter and its child folders
	models := []Model{
		Model{
			ServiceContent: esx.ServiceContent,
			RootFolder:     esx.RootFolder,
		},
		Model{
			ServiceContent: vpx.ServiceContent,
			RootFolder:     vpx.RootFolder,
			Datacenter:     1,
		},
	}

	for _, model := range models {
		_ = model.Create()

		dc := Map.Any("Datacenter").(*Datacenter)

		folders := []types.ManagedObjectReference{
			dc.VmFolder,
			dc.HostFolder,
			dc.DatastoreFolder,
			dc.NetworkFolder,
		}

		for _, ref := range folders {
			if ref.Type == "" || ref.Value == "" {
				t.Errorf("invalid moref=%#v", ref)
			}

			e := Map.Get(ref).(mo.Entity)

			if e.Entity().Name == "" {
				t.Error("empty name")
			}

			if *e.Entity().Parent != dc.Self {
				t.Fail()
			}

			f, ok := e.(*Folder)
			if !ok {
				t.Fatalf("unexpected type (%T) for %#v", e, ref)
			}

			if Map.IsVPX() {
				if len(f.ChildType) < 2 {
					t.Fail()
				}
			} else {
				if len(f.ChildType) != 1 {
					t.Fail()
				}
			}
		}
	}
}
