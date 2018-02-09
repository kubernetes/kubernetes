/*
Copyright 2018 The Kubernetes Authors.

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

package vsphere

import (
	"github.com/vmware/govmomi"
	"github.com/vmware/govmomi/find"
	"github.com/vmware/govmomi/object"
	"golang.org/x/net/context"
	"strings"
)

// Represents a vSphere instance where one or more kubernetes nodes are running.
type VSphere struct {
	Config *Config
	Client *govmomi.Client
}

// GetDatacenter returns the DataCenter Object for the given datacenterPath
func (vs *VSphere) GetDatacenter(ctx context.Context, datacenterPath string) (*object.Datacenter, error) {
	Connect(ctx, vs)
	finder := find.NewFinder(vs.Client.Client, true)
	return finder.Datacenter(ctx, datacenterPath)
}

// GetAllDatacenter returns all the DataCenter Objects
func (vs *VSphere) GetAllDatacenter(ctx context.Context) ([]*object.Datacenter, error) {
	Connect(ctx, vs)
	finder := find.NewFinder(vs.Client.Client, true)
	return finder.DatacenterList(ctx, "*")
}

// GetVMByUUID gets the VM object from the given vmUUID
func (vs *VSphere) GetVMByUUID(ctx context.Context, vmUUID string, dc object.Reference) (object.Reference, error) {
	Connect(ctx, vs)
	datacenter := object.NewDatacenter(vs.Client.Client, dc.Reference())
	s := object.NewSearchIndex(vs.Client.Client)
	vmUUID = strings.ToLower(strings.TrimSpace(vmUUID))
	return s.FindByUuid(ctx, datacenter, vmUUID, true, nil)
}
