/*
Copyright (c) 2014-2015 VMware, Inc. All Rights Reserved.

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

package importx

import (
	"errors"
	"flag"
	"fmt"
	"path"
	"strings"

	"github.com/vmware/govmomi/govc/cli"
	"github.com/vmware/govmomi/govc/flags"
	"github.com/vmware/govmomi/object"
	"github.com/vmware/govmomi/ovf"
	"github.com/vmware/govmomi/vim25"
	"github.com/vmware/govmomi/vim25/progress"
	"github.com/vmware/govmomi/vim25/soap"
	"github.com/vmware/govmomi/vim25/types"
	"golang.org/x/net/context"
)

type ovfx struct {
	*flags.DatastoreFlag
	*flags.HostSystemFlag
	*flags.OutputFlag
	*flags.ResourcePoolFlag

	*ArchiveFlag
	*OptionsFlag
	*FolderFlag

	Name string

	Client       *vim25.Client
	Datacenter   *object.Datacenter
	Datastore    *object.Datastore
	ResourcePool *object.ResourcePool
}

func init() {
	cli.Register("import.ovf", &ovfx{})
}

func (cmd *ovfx) Register(ctx context.Context, f *flag.FlagSet) {
	cmd.DatastoreFlag, ctx = flags.NewDatastoreFlag(ctx)
	cmd.DatastoreFlag.Register(ctx, f)
	cmd.HostSystemFlag, ctx = flags.NewHostSystemFlag(ctx)
	cmd.HostSystemFlag.Register(ctx, f)
	cmd.OutputFlag, ctx = flags.NewOutputFlag(ctx)
	cmd.OutputFlag.Register(ctx, f)
	cmd.ResourcePoolFlag, ctx = flags.NewResourcePoolFlag(ctx)
	cmd.ResourcePoolFlag.Register(ctx, f)

	cmd.ArchiveFlag, ctx = newArchiveFlag(ctx)
	cmd.ArchiveFlag.Register(ctx, f)
	cmd.OptionsFlag, ctx = newOptionsFlag(ctx)
	cmd.OptionsFlag.Register(ctx, f)
	cmd.FolderFlag, ctx = newFolderFlag(ctx)
	cmd.FolderFlag.Register(ctx, f)

	f.StringVar(&cmd.Name, "name", "", "Name to use for new entity")
}

func (cmd *ovfx) Process(ctx context.Context) error {
	if err := cmd.DatastoreFlag.Process(ctx); err != nil {
		return err
	}
	if err := cmd.HostSystemFlag.Process(ctx); err != nil {
		return err
	}
	if err := cmd.OutputFlag.Process(ctx); err != nil {
		return err
	}
	if err := cmd.ResourcePoolFlag.Process(ctx); err != nil {
		return err
	}
	if err := cmd.ArchiveFlag.Process(ctx); err != nil {
		return err
	}
	if err := cmd.OptionsFlag.Process(ctx); err != nil {
		return err
	}
	if err := cmd.FolderFlag.Process(ctx); err != nil {
		return err
	}
	return nil
}

func (cmd *ovfx) Usage() string {
	return "PATH_TO_OVF"
}

func (cmd *ovfx) Run(ctx context.Context, f *flag.FlagSet) error {
	fpath, err := cmd.Prepare(f)
	if err != nil {
		return err
	}

	cmd.Archive = &FileArchive{fpath}

	moref, err := cmd.Import(fpath)
	if err != nil {
		return err
	}

	vm := object.NewVirtualMachine(cmd.Client, *moref)
	return cmd.Deploy(vm)
}

func (cmd *ovfx) Prepare(f *flag.FlagSet) (string, error) {
	var err error

	args := f.Args()
	if len(args) != 1 {
		return "", errors.New("no file specified")
	}

	cmd.Client, err = cmd.DatastoreFlag.Client()
	if err != nil {
		return "", err
	}

	cmd.Datacenter, err = cmd.DatastoreFlag.Datacenter()
	if err != nil {
		return "", err
	}

	cmd.Datastore, err = cmd.DatastoreFlag.Datastore()
	if err != nil {
		return "", err
	}

	cmd.ResourcePool, err = cmd.ResourcePoolFlag.ResourcePool()
	if err != nil {
		return "", err
	}

	return f.Arg(0), nil
}

func (cmd *ovfx) Deploy(vm *object.VirtualMachine) error {
	if err := cmd.PowerOn(vm); err != nil {
		return err
	}

	if err := cmd.InjectOvfEnv(vm); err != nil {
		return err
	}

	if err := cmd.WaitForIP(vm); err != nil {
		return err
	}

	return nil
}

func (cmd *ovfx) Map(op []Property) (p []types.KeyValue) {
	for _, v := range op {
		p = append(p, v.KeyValue)
	}

	return
}

func (cmd *ovfx) Import(fpath string) (*types.ManagedObjectReference, error) {
	o, err := cmd.ReadOvf(fpath)
	if err != nil {
		return nil, err
	}

	e, err := cmd.ReadEnvelope(fpath)
	if err != nil {
		return nil, fmt.Errorf("failed to parse ovf: %s", err.Error())
	}

	name := "Govc Virtual Appliance"
	if e.VirtualSystem != nil {
		name = e.VirtualSystem.ID
		if e.VirtualSystem.Name != nil {
			name = *e.VirtualSystem.Name
		}
	}

	// Override name from options if specified
	if cmd.Options.Name != nil {
		name = *cmd.Options.Name
	}

	// Override name from arguments if specified
	if cmd.Name != "" {
		name = cmd.Name
	}

	cisp := types.OvfCreateImportSpecParams{
		DiskProvisioning:   cmd.Options.DiskProvisioning,
		EntityName:         name,
		IpAllocationPolicy: cmd.Options.IPAllocationPolicy,
		IpProtocol:         cmd.Options.IPProtocol,
		OvfManagerCommonParams: types.OvfManagerCommonParams{
			DeploymentOption: cmd.Options.Deployment,
			Locale:           "US"},
		PropertyMapping: cmd.Map(cmd.Options.PropertyMapping),
	}

	m := object.NewOvfManager(cmd.Client)
	spec, err := m.CreateImportSpec(context.TODO(), string(o), cmd.ResourcePool, cmd.Datastore, cisp)
	if err != nil {
		return nil, err
	}
	if spec.Error != nil {
		return nil, errors.New(spec.Error[0].LocalizedMessage)
	}
	if spec.Warning != nil {
		for _, w := range spec.Warning {
			_, _ = cmd.Log(fmt.Sprintf("Warning: %s\n", w.LocalizedMessage))
		}
	}

	var host *object.HostSystem
	if cmd.SearchFlag.IsSet() {
		if host, err = cmd.HostSystem(); err != nil {
			return nil, err
		}
	}

	folder, err := cmd.Folder()
	if err != nil {
		return nil, err
	}

	lease, err := cmd.ResourcePool.ImportVApp(context.TODO(), spec.ImportSpec, folder, host)
	if err != nil {
		return nil, err
	}

	info, err := lease.Wait(context.TODO())
	if err != nil {
		return nil, err
	}

	// Build slice of items and URLs first, so that the lease updater can know
	// about every item that needs to be uploaded, and thereby infer progress.
	var items []ovfFileItem

	for _, device := range info.DeviceUrl {
		for _, item := range spec.FileItem {
			if device.ImportKey != item.DeviceId {
				continue
			}

			u, err := cmd.Client.ParseURL(device.Url)
			if err != nil {
				return nil, err
			}

			i := ovfFileItem{
				url:  u,
				item: item,
				ch:   make(chan progress.Report),
			}

			items = append(items, i)
		}
	}

	u := newLeaseUpdater(cmd.Client, lease, items)
	defer u.Done()

	for _, i := range items {
		err = cmd.Upload(lease, i)
		if err != nil {
			return nil, err
		}
	}

	return &info.Entity, lease.HttpNfcLeaseComplete(context.TODO())
}

func (cmd *ovfx) Upload(lease *object.HttpNfcLease, ofi ovfFileItem) error {
	item := ofi.item
	file := item.Path

	f, size, err := cmd.Open(file)
	if err != nil {
		return err
	}
	defer f.Close()

	logger := cmd.ProgressLogger(fmt.Sprintf("Uploading %s... ", path.Base(file)))
	defer logger.Wait()

	opts := soap.Upload{
		ContentLength: size,
		Progress:      progress.Tee(ofi, logger),
	}

	// Non-disk files (such as .iso) use the PUT method.
	// Overwrite: t header is also required in this case (ovftool does the same)
	if item.Create {
		opts.Method = "PUT"
		opts.Headers = map[string]string{
			"Overwrite": "t",
		}
	} else {
		opts.Method = "POST"
		opts.Type = "application/x-vnd.vmware-streamVmdk"
	}

	return cmd.Client.Client.Upload(f, ofi.url, &opts)
}

func (cmd *ovfx) PowerOn(vm *object.VirtualMachine) error {
	if !cmd.Options.PowerOn {
		return nil
	}

	cmd.Log("Powering on VM...\n")

	task, err := vm.PowerOn(context.TODO())
	if err != nil {
		return err
	}

	if _, err = task.WaitForResult(context.TODO(), nil); err != nil {
		return err
	}

	return nil
}

func (cmd *ovfx) InjectOvfEnv(vm *object.VirtualMachine) error {
	if !cmd.Options.PowerOn || !cmd.Options.InjectOvfEnv {
		return nil
	}

	a := cmd.Client.ServiceContent.About
	if strings.EqualFold(a.ProductLineId, "esx") || strings.EqualFold(a.ProductLineId, "embeddedEsx") {
		cmd.Log("Injecting OVF environment...\n")

		// build up Environment in order to marshal to xml
		var epa []ovf.EnvProperty
		for _, p := range cmd.Options.PropertyMapping {
			epa = append(epa, ovf.EnvProperty{
				Key:   p.Key,
				Value: p.Value})
		}
		env := ovf.Env{
			EsxID: vm.Reference().Value,
			Platform: &ovf.PlatformSection{
				Kind:    a.Name,
				Version: a.Version,
				Vendor:  a.Vendor,
				Locale:  "US",
			},
			Property: &ovf.PropertySection{
				Properties: epa},
		}

		xenv := env.MarshalManual()
		vmConfigSpec := types.VirtualMachineConfigSpec{
			ExtraConfig: []types.BaseOptionValue{&types.OptionValue{
				Key:   "guestinfo.ovfEnv",
				Value: xenv}}}

		task, err := vm.Reconfigure(context.TODO(), vmConfigSpec)
		if err != nil {
			return err
		}
		if err := task.Wait(context.TODO()); err != nil {
			return err
		}
	}

	return nil
}

func (cmd *ovfx) WaitForIP(vm *object.VirtualMachine) error {
	if !cmd.Options.PowerOn || !cmd.Options.WaitForIP {
		return nil
	}

	cmd.Log("Waiting for IP address...\n")
	ip, err := vm.WaitForIP(context.TODO())
	if err != nil {
		return err
	}

	cmd.Log(fmt.Sprintf("Received IP address: %s\n", ip))
	return nil
}
