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
	"context"
	"errors"
	"flag"
	"fmt"
	"path"

	"github.com/vmware/govmomi/govc/cli"
	"github.com/vmware/govmomi/govc/flags"
	"github.com/vmware/govmomi/nfc"
	"github.com/vmware/govmomi/object"
	"github.com/vmware/govmomi/ovf"
	"github.com/vmware/govmomi/vim25"
	"github.com/vmware/govmomi/vim25/soap"
	"github.com/vmware/govmomi/vim25/types"
)

type ovfx struct {
	*flags.DatastoreFlag
	*flags.HostSystemFlag
	*flags.OutputFlag
	*flags.ResourcePoolFlag
	*flags.FolderFlag

	*ArchiveFlag
	*OptionsFlag

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
	cmd.FolderFlag, ctx = flags.NewFolderFlag(ctx)
	cmd.FolderFlag.Register(ctx, f)

	cmd.ArchiveFlag, ctx = newArchiveFlag(ctx)
	cmd.ArchiveFlag.Register(ctx, f)
	cmd.OptionsFlag, ctx = newOptionsFlag(ctx)
	cmd.OptionsFlag.Register(ctx, f)

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

	archive := &FileArchive{path: fpath}
	archive.Client = cmd.Client

	cmd.Archive = archive

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

	cmd.ResourcePool, err = cmd.ResourcePoolFlag.ResourcePoolIfSpecified()
	if err != nil {
		return "", err
	}

	return f.Arg(0), nil
}

func (cmd *ovfx) Deploy(vm *object.VirtualMachine) error {
	if err := cmd.InjectOvfEnv(vm); err != nil {
		return err
	}

	if err := cmd.MarkAsTemplate(vm); err != nil {
		return err
	}

	if err := cmd.PowerOn(vm); err != nil {
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

func (cmd *ovfx) NetworkMap(e *ovf.Envelope) (p []types.OvfNetworkMapping) {
	ctx := context.TODO()
	finder, err := cmd.DatastoreFlag.Finder()
	if err != nil {
		return
	}

	networks := map[string]string{}

	if e.Network != nil {
		for _, net := range e.Network.Networks {
			networks[net.Name] = net.Name
		}
	}

	for _, net := range cmd.Options.NetworkMapping {
		networks[net.Name] = net.Network
	}

	for src, dst := range networks {
		if net, err := finder.Network(ctx, dst); err == nil {
			p = append(p, types.OvfNetworkMapping{
				Name:    src,
				Network: net.Reference(),
			})
		}
	}
	return
}

func (cmd *ovfx) Import(fpath string) (*types.ManagedObjectReference, error) {
	ctx := context.TODO()

	o, err := cmd.ReadOvf(fpath)
	if err != nil {
		return nil, err
	}

	e, err := cmd.ReadEnvelope(o)
	if err != nil {
		return nil, fmt.Errorf("failed to parse ovf: %s", err)
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
		NetworkMapping:  cmd.NetworkMap(e),
	}

	host, err := cmd.HostSystemIfSpecified()
	if err != nil {
		return nil, err
	}

	if cmd.ResourcePool == nil {
		if host == nil {
			cmd.ResourcePool, err = cmd.ResourcePoolFlag.ResourcePool()
		} else {
			cmd.ResourcePool, err = host.ResourcePool(ctx)
		}
		if err != nil {
			return nil, err
		}
	}

	m := ovf.NewManager(cmd.Client)
	spec, err := m.CreateImportSpec(ctx, string(o), cmd.ResourcePool, cmd.Datastore, cisp)
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

	if cmd.Options.Annotation != "" {
		switch s := spec.ImportSpec.(type) {
		case *types.VirtualMachineImportSpec:
			s.ConfigSpec.Annotation = cmd.Options.Annotation
		case *types.VirtualAppImportSpec:
			s.VAppConfigSpec.Annotation = cmd.Options.Annotation
		}
	}

	folder, err := cmd.FolderOrDefault("vm")
	if err != nil {
		return nil, err
	}

	lease, err := cmd.ResourcePool.ImportVApp(ctx, spec.ImportSpec, folder, host)
	if err != nil {
		return nil, err
	}

	info, err := lease.Wait(ctx, spec.FileItem)
	if err != nil {
		return nil, err
	}

	u := lease.StartUpdater(ctx, info)
	defer u.Done()

	for _, i := range info.Items {
		err = cmd.Upload(ctx, lease, i)
		if err != nil {
			return nil, err
		}
	}

	return &info.Entity, lease.Complete(ctx)
}

func (cmd *ovfx) Upload(ctx context.Context, lease *nfc.Lease, item nfc.FileItem) error {
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
		Progress:      logger,
	}

	return lease.Upload(ctx, item, f, opts)
}

func (cmd *ovfx) PowerOn(vm *object.VirtualMachine) error {
	ctx := context.TODO()
	if !cmd.Options.PowerOn || cmd.Options.MarkAsTemplate {
		return nil
	}

	cmd.Log("Powering on VM...\n")

	task, err := vm.PowerOn(ctx)
	if err != nil {
		return err
	}

	if _, err = task.WaitForResult(ctx, nil); err != nil {
		return err
	}

	return nil
}

func (cmd *ovfx) MarkAsTemplate(vm *object.VirtualMachine) error {
	ctx := context.TODO()
	if !cmd.Options.MarkAsTemplate {
		return nil
	}

	cmd.Log("Marking VM as template...\n")

	err := vm.MarkAsTemplate(ctx)
	if err != nil {
		return err
	}

	return nil
}

func (cmd *ovfx) InjectOvfEnv(vm *object.VirtualMachine) error {
	if !cmd.Options.InjectOvfEnv {
		return nil
	}

	cmd.Log("Injecting OVF environment...\n")

	var opts []types.BaseOptionValue

	a := cmd.Client.ServiceContent.About

	// build up Environment in order to marshal to xml
	var props []ovf.EnvProperty
	for _, p := range cmd.Options.PropertyMapping {
		props = append(props, ovf.EnvProperty{
			Key:   p.Key,
			Value: p.Value,
		})
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
			Properties: props,
		},
	}

	opts = append(opts, &types.OptionValue{
		Key:   "guestinfo.ovfEnv",
		Value: env.MarshalManual(),
	})

	ctx := context.Background()

	task, err := vm.Reconfigure(ctx, types.VirtualMachineConfigSpec{
		ExtraConfig: opts,
	})

	if err != nil {
		return err
	}

	return task.Wait(ctx)
}

func (cmd *ovfx) WaitForIP(vm *object.VirtualMachine) error {
	ctx := context.TODO()
	if !cmd.Options.PowerOn || !cmd.Options.WaitForIP || cmd.Options.MarkAsTemplate {
		return nil
	}

	cmd.Log("Waiting for IP address...\n")
	ip, err := vm.WaitForIP(ctx)
	if err != nil {
		return err
	}

	cmd.Log(fmt.Sprintf("Received IP address: %s\n", ip))
	return nil
}
