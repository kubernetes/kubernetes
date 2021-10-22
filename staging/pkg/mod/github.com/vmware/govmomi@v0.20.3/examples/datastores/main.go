/*
Copyright (c) 2015-2017 VMware, Inc. All Rights Reserved.

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

/*
This example program shows how the `view` and `property` packages can
be used to navigate a vSphere inventory structure using govmomi.
*/

package main

import (
	"context"
	"fmt"
	"log"
	"os"
	"text/tabwriter"

	"github.com/vmware/govmomi/examples"
	"github.com/vmware/govmomi/units"
	"github.com/vmware/govmomi/view"
	"github.com/vmware/govmomi/vim25/mo"
)

func main() {
	ctx := context.Background()

	// Connect and log in to ESX or vCenter
	c, err := examples.NewClient(ctx)
	if err != nil {
		log.Fatal(err)
	}

	defer c.Logout(ctx)

	// Create a view of Datastore objects
	m := view.NewManager(c.Client)

	v, err := m.CreateContainerView(ctx, c.ServiceContent.RootFolder, []string{"Datastore"}, true)
	if err != nil {
		log.Fatal(err)
	}

	defer v.Destroy(ctx)

	// Retrieve summary property for all datastores
	// Reference: http://pubs.vmware.com/vsphere-60/topic/com.vmware.wssdk.apiref.doc/vim.Datastore.html
	var dss []mo.Datastore
	err = v.Retrieve(ctx, []string{"Datastore"}, []string{"summary"}, &dss)
	if err != nil {
		log.Fatal(err)
	}

	// Print summary per datastore (see also: govc/datastore/info.go)

	tw := tabwriter.NewWriter(os.Stdout, 2, 0, 2, ' ', 0)
	fmt.Fprintf(tw, "Name:\tType:\tCapacity:\tFree:\n")

	for _, ds := range dss {
		fmt.Fprintf(tw, "%s\t", ds.Summary.Name)
		fmt.Fprintf(tw, "%s\t", ds.Summary.Type)
		fmt.Fprintf(tw, "%s\t", units.ByteSize(ds.Summary.Capacity))
		fmt.Fprintf(tw, "%s\t", units.ByteSize(ds.Summary.FreeSpace))
		fmt.Fprintf(tw, "\n")
	}

	_ = tw.Flush()
}
