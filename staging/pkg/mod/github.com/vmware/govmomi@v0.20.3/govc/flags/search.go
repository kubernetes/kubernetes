/*
Copyright (c) 2014-2016 VMware, Inc. All Rights Reserved.

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

package flags

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"strings"

	"github.com/vmware/govmomi/find"
	"github.com/vmware/govmomi/object"
	"github.com/vmware/govmomi/vim25"
	"github.com/vmware/govmomi/vim25/soap"
	"github.com/vmware/govmomi/vim25/types"
)

const (
	SearchVirtualMachines = iota + 1
	SearchHosts
	SearchVirtualApps
)

type SearchFlag struct {
	common

	*ClientFlag
	*DatacenterFlag

	t      int
	entity string

	byDatastorePath string
	byDNSName       string
	byInventoryPath string
	byIP            string
	byUUID          string

	isset bool
}

var searchFlagKey = flagKey("search")

func NewSearchFlag(ctx context.Context, t int) (*SearchFlag, context.Context) {
	if v := ctx.Value(searchFlagKey); v != nil {
		return v.(*SearchFlag), ctx
	}

	v := &SearchFlag{
		t: t,
	}

	v.ClientFlag, ctx = NewClientFlag(ctx)
	v.DatacenterFlag, ctx = NewDatacenterFlag(ctx)

	switch t {
	case SearchVirtualMachines:
		v.entity = "VM"
	case SearchHosts:
		v.entity = "host"
	case SearchVirtualApps:
		v.entity = "vapp"
	default:
		panic("invalid search type")
	}

	ctx = context.WithValue(ctx, searchFlagKey, v)
	return v, ctx
}

func (flag *SearchFlag) Register(ctx context.Context, fs *flag.FlagSet) {
	flag.RegisterOnce(func() {
		flag.ClientFlag.Register(ctx, fs)
		flag.DatacenterFlag.Register(ctx, fs)

		register := func(v *string, f string, d string) {
			f = fmt.Sprintf("%s.%s", strings.ToLower(flag.entity), f)
			d = fmt.Sprintf(d, flag.entity)
			fs.StringVar(v, f, "", d)
		}

		switch flag.t {
		case SearchVirtualMachines:
			register(&flag.byDatastorePath, "path", "Find %s by path to .vmx file")
		}

		switch flag.t {
		case SearchVirtualMachines, SearchHosts:
			register(&flag.byDNSName, "dns", "Find %s by FQDN")
			register(&flag.byIP, "ip", "Find %s by IP address")
			register(&flag.byUUID, "uuid", "Find %s by UUID")
		}

		register(&flag.byInventoryPath, "ipath", "Find %s by inventory path")
	})
}

func (flag *SearchFlag) Process(ctx context.Context) error {
	return flag.ProcessOnce(func() error {
		if err := flag.ClientFlag.Process(ctx); err != nil {
			return err
		}
		if err := flag.DatacenterFlag.Process(ctx); err != nil {
			return err
		}

		flags := []string{
			flag.byDatastorePath,
			flag.byDNSName,
			flag.byInventoryPath,
			flag.byIP,
			flag.byUUID,
		}

		flag.isset = false
		for _, f := range flags {
			if f != "" {
				if flag.isset {
					return errors.New("cannot use more than one search flag")
				}
				flag.isset = true
			}
		}

		return nil
	})
}

func (flag *SearchFlag) IsSet() bool {
	return flag.isset
}

func (flag *SearchFlag) searchIndex(c *vim25.Client) *object.SearchIndex {
	return object.NewSearchIndex(c)
}

func (flag *SearchFlag) searchByDatastorePath(c *vim25.Client, dc *object.Datacenter) (object.Reference, error) {
	ctx := context.TODO()
	switch flag.t {
	case SearchVirtualMachines:
		return flag.searchIndex(c).FindByDatastorePath(ctx, dc, flag.byDatastorePath)
	default:
		panic("unsupported type")
	}
}

func (flag *SearchFlag) searchByDNSName(c *vim25.Client, dc *object.Datacenter) (object.Reference, error) {
	ctx := context.TODO()
	switch flag.t {
	case SearchVirtualMachines:
		return flag.searchIndex(c).FindByDnsName(ctx, dc, flag.byDNSName, true)
	case SearchHosts:
		return flag.searchIndex(c).FindByDnsName(ctx, dc, flag.byDNSName, false)
	default:
		panic("unsupported type")
	}
}

func (flag *SearchFlag) searchByInventoryPath(c *vim25.Client, dc *object.Datacenter) (object.Reference, error) {
	// TODO(PN): The datacenter flag should not be set because it is ignored.
	ctx := context.TODO()
	return flag.searchIndex(c).FindByInventoryPath(ctx, flag.byInventoryPath)
}

func (flag *SearchFlag) searchByIP(c *vim25.Client, dc *object.Datacenter) (object.Reference, error) {
	ctx := context.TODO()
	switch flag.t {
	case SearchVirtualMachines:
		return flag.searchIndex(c).FindByIp(ctx, dc, flag.byIP, true)
	case SearchHosts:
		return flag.searchIndex(c).FindByIp(ctx, dc, flag.byIP, false)
	default:
		panic("unsupported type")
	}
}

func (flag *SearchFlag) searchByUUID(c *vim25.Client, dc *object.Datacenter) (object.Reference, error) {
	ctx := context.TODO()
	isVM := false
	switch flag.t {
	case SearchVirtualMachines:
		isVM = true
	case SearchHosts:
	default:
		panic("unsupported type")
	}

	var ref object.Reference
	var err error

	for _, iu := range []*bool{nil, types.NewBool(true)} {
		ref, err = flag.searchIndex(c).FindByUuid(ctx, dc, flag.byUUID, isVM, iu)
		if err != nil {
			if soap.IsSoapFault(err) {
				fault := soap.ToSoapFault(err).VimFault()
				if _, ok := fault.(types.InvalidArgument); ok {
					continue
				}
			}
			return nil, err
		}
		if ref != nil {
			break
		}
	}

	return ref, nil
}

func (flag *SearchFlag) search() (object.Reference, error) {
	ctx := context.TODO()
	var ref object.Reference
	var err error

	c, err := flag.Client()
	if err != nil {
		return nil, err
	}

	dc, err := flag.Datacenter()
	if err != nil {
		return nil, err
	}

	switch {
	case flag.byDatastorePath != "":
		ref, err = flag.searchByDatastorePath(c, dc)
	case flag.byDNSName != "":
		ref, err = flag.searchByDNSName(c, dc)
	case flag.byInventoryPath != "":
		ref, err = flag.searchByInventoryPath(c, dc)
	case flag.byIP != "":
		ref, err = flag.searchByIP(c, dc)
	case flag.byUUID != "":
		ref, err = flag.searchByUUID(c, dc)
	default:
		err = errors.New("no search flag specified")
	}

	if err != nil {
		return nil, err
	}

	if ref == nil {
		return nil, fmt.Errorf("no such %s", flag.entity)
	}

	// set the InventoryPath field
	finder, err := flag.Finder()
	if err != nil {
		return nil, err
	}
	ref, err = finder.ObjectReference(ctx, ref.Reference())
	if err != nil {
		return nil, err
	}

	return ref, nil
}

func (flag *SearchFlag) VirtualMachine() (*object.VirtualMachine, error) {
	ref, err := flag.search()
	if err != nil {
		return nil, err
	}

	vm, ok := ref.(*object.VirtualMachine)
	if !ok {
		return nil, fmt.Errorf("expected VirtualMachine entity, got %s", ref.Reference().Type)
	}

	return vm, nil
}

func (flag *SearchFlag) VirtualMachines(args []string) ([]*object.VirtualMachine, error) {
	ctx := context.TODO()
	var out []*object.VirtualMachine

	if flag.IsSet() {
		vm, err := flag.VirtualMachine()
		if err != nil {
			return nil, err
		}

		out = append(out, vm)
		return out, nil
	}

	// List virtual machines
	if len(args) == 0 {
		return nil, errors.New("no argument")
	}

	finder, err := flag.Finder()
	if err != nil {
		return nil, err
	}

	var nfe error

	// List virtual machines for every argument
	for _, arg := range args {
		vms, err := finder.VirtualMachineList(ctx, arg)
		if err != nil {
			if _, ok := err.(*find.NotFoundError); ok {
				// Let caller decide how to handle NotFoundError
				nfe = err
				continue
			}
			return nil, err
		}

		out = append(out, vms...)
	}

	return out, nfe
}

func (flag *SearchFlag) VirtualApp() (*object.VirtualApp, error) {
	ref, err := flag.search()
	if err != nil {
		return nil, err
	}

	app, ok := ref.(*object.VirtualApp)
	if !ok {
		return nil, fmt.Errorf("expected VirtualApp entity, got %s", ref.Reference().Type)
	}

	return app, nil
}

func (flag *SearchFlag) VirtualApps(args []string) ([]*object.VirtualApp, error) {
	ctx := context.TODO()
	var out []*object.VirtualApp

	if flag.IsSet() {
		app, err := flag.VirtualApp()
		if err != nil {
			return nil, err
		}

		out = append(out, app)
		return out, nil
	}

	// List virtual apps
	if len(args) == 0 {
		return nil, errors.New("no argument")
	}

	finder, err := flag.Finder()
	if err != nil {
		return nil, err
	}

	// List virtual apps for every argument
	for _, arg := range args {
		apps, err := finder.VirtualAppList(ctx, arg)
		if err != nil {
			return nil, err
		}

		out = append(out, apps...)
	}

	return out, nil
}

func (flag *SearchFlag) HostSystem() (*object.HostSystem, error) {
	ref, err := flag.search()
	if err != nil {
		return nil, err
	}

	host, ok := ref.(*object.HostSystem)
	if !ok {
		return nil, fmt.Errorf("expected HostSystem entity, got %s", ref.Reference().Type)
	}

	return host, nil
}

func (flag *SearchFlag) HostSystems(args []string) ([]*object.HostSystem, error) {
	ctx := context.TODO()
	var out []*object.HostSystem

	if flag.IsSet() {
		host, err := flag.HostSystem()
		if err != nil {
			return nil, err
		}

		out = append(out, host)
		return out, nil
	}

	// List host system
	if len(args) == 0 {
		return nil, errors.New("no argument")
	}

	finder, err := flag.Finder()
	if err != nil {
		return nil, err
	}

	// List host systems for every argument
	for _, arg := range args {
		vms, err := finder.HostSystemList(ctx, arg)
		if err != nil {
			return nil, err
		}

		out = append(out, vms...)
	}

	return out, nil
}
