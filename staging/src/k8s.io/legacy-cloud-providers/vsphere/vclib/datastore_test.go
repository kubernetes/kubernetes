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
	"github.com/vmware/govmomi/object"
	"github.com/vmware/govmomi/simulator"
)

func TestDatastore(t *testing.T) {
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

	c, err := govmomi.NewClient(ctx, s.URL, true)
	if err != nil {
		t.Fatal(err)
	}

	vc := &VSphereConnection{Client: c.Client}

	dc, err := GetDatacenter(ctx, vc, TestDefaultDatacenter)
	if err != nil {
		t.Error(err)
	}

	all, err := dc.GetAllDatastores(ctx)
	if err != nil {
		t.Fatal(err)
	}

	for _, info := range all {
		ds := info.Datastore
		kind, cerr := ds.GetType(ctx)
		if cerr != nil {
			t.Error(err)
		}
		if kind == "" {
			t.Error("empty Datastore type")
		}

		dir := object.DatastorePath{
			Datastore: info.Info.Name,
			Path:      "kubevols",
		}

		// TODO: test Datastore.IsCompatibleWithStoragePolicy (vcsim needs PBM support)

		for _, fail := range []bool{false, true} {
			cerr = ds.CreateDirectory(ctx, dir.String(), false)
			if fail {
				if cerr != ErrFileAlreadyExist {
					t.Errorf("expected %s, got: %s", ErrFileAlreadyExist, cerr)
				}
				continue
			}

			if cerr != nil {
				t.Error(err)
			}
		}
	}
}
