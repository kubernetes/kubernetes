/*
Copyright (c) 2015-2016 VMware, Inc. All Rights Reserved.

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

package datastore

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/vmware/govmomi/govc/cli"
	"github.com/vmware/govmomi/govc/flags"
	"github.com/vmware/govmomi/object"
	"github.com/vmware/govmomi/vim25/soap"
	"github.com/vmware/govmomi/vim25/types"
)

type create struct {
	*flags.HostSystemFlag

	// Generic options
	Type  typeFlag
	Name  string
	Force bool

	// Options for NAS
	RemoteHost string
	RemotePath string
	AccessMode string
	UserName   string
	Password   string

	// Options for VMFS
	DiskCanonicalName string

	// Options for local
	Path string
}

func init() {
	cli.Register("datastore.create", &create{})
}

var nasTypes = []string{
	string(types.HostFileSystemVolumeFileSystemTypeNFS),
	string(types.HostFileSystemVolumeFileSystemTypeNFS41),
	string(types.HostFileSystemVolumeFileSystemTypeCIFS),
}

var vmfsTypes = []string{
	string(types.HostFileSystemVolumeFileSystemTypeVMFS),
}

var localTypes = []string{
	"local",
}

var allTypes = []string{}

func init() {
	allTypes = append(allTypes, nasTypes...)
	allTypes = append(allTypes, vmfsTypes...)
	allTypes = append(allTypes, localTypes...)
}

type typeFlag string

func (t *typeFlag) Set(s string) error {
	s = strings.ToLower(s)
	for _, e := range allTypes {
		if s == strings.ToLower(e) {
			*t = typeFlag(e)
			return nil
		}
	}

	return fmt.Errorf("unknown type")
}

func (t *typeFlag) String() string {
	return string(*t)
}

func (t *typeFlag) partOf(m []string) bool {
	for _, e := range m {
		if t.String() == e {
			return true
		}
	}
	return false
}

func (t *typeFlag) IsNasType() bool {
	return t.partOf(nasTypes)
}

func (t *typeFlag) IsVmfsType() bool {
	return t.partOf(vmfsTypes)
}

func (t *typeFlag) IsLocalType() bool {
	return t.partOf(localTypes)
}

func (cmd *create) Register(ctx context.Context, f *flag.FlagSet) {
	cmd.HostSystemFlag, ctx = flags.NewHostSystemFlag(ctx)
	cmd.HostSystemFlag.Register(ctx, f)

	modes := []string{
		string(types.HostMountModeReadOnly),
		string(types.HostMountModeReadWrite),
	}

	f.StringVar(&cmd.Name, "name", "", "Datastore name")
	f.Var(&cmd.Type, "type", fmt.Sprintf("Datastore type (%s)", strings.Join(allTypes, "|")))
	f.BoolVar(&cmd.Force, "force", false, "Ignore DuplicateName error if datastore is already mounted on a host")

	// Options for NAS
	f.StringVar(&cmd.RemoteHost, "remote-host", "", "Remote hostname of the NAS datastore")
	f.StringVar(&cmd.RemotePath, "remote-path", "", "Remote path of the NFS mount point")
	f.StringVar(&cmd.AccessMode, "mode", modes[0],
		fmt.Sprintf("Access mode for the mount point (%s)", strings.Join(modes, "|")))
	f.StringVar(&cmd.UserName, "username", "", "Username to use when connecting (CIFS only)")
	f.StringVar(&cmd.Password, "password", "", "Password to use when connecting (CIFS only)")

	// Options for VMFS
	f.StringVar(&cmd.DiskCanonicalName, "disk", "", "Canonical name of disk (VMFS only)")

	// Options for Local
	f.StringVar(&cmd.Path, "path", "", "Local directory path for the datastore (local only)")
}

func (cmd *create) Process(ctx context.Context) error {
	if err := cmd.HostSystemFlag.Process(ctx); err != nil {
		return err
	}
	return nil
}

func (cmd *create) Usage() string {
	return "HOST..."
}

func (cmd *create) Description() string {
	return `Create datastore on HOST.

Examples:
  govc datastore.create -type nfs -name nfsDatastore -remote-host 10.143.2.232 -remote-path /share cluster1
  govc datastore.create -type vmfs -name vmfsDatastore -disk=mpx.vmhba0:C0:T0:L0 cluster1
  govc datastore.create -type local -name localDatastore -path /var/datastore host1`
}

func (cmd *create) Run(ctx context.Context, f *flag.FlagSet) error {
	hosts, err := cmd.HostSystems(f.Args())
	if err != nil {
		return err
	}

	switch {
	case cmd.Type.IsNasType():
		return cmd.CreateNasDatastore(ctx, hosts)
	case cmd.Type.IsVmfsType():
		return cmd.CreateVmfsDatastore(ctx, hosts)
	case cmd.Type.IsLocalType():
		return cmd.CreateLocalDatastore(ctx, hosts)
	default:
		return fmt.Errorf("unhandled type %#v", cmd.Type)
	}
}

func (cmd *create) GetHostNasVolumeSpec() types.HostNasVolumeSpec {
	localPath := cmd.Path
	if localPath == "" {
		localPath = cmd.Name
	}

	s := types.HostNasVolumeSpec{
		LocalPath:  localPath,
		Type:       cmd.Type.String(),
		RemoteHost: cmd.RemoteHost,
		RemotePath: cmd.RemotePath,
		AccessMode: cmd.AccessMode,
		UserName:   cmd.UserName,
		Password:   cmd.Password,
	}

	return s
}

func (cmd *create) CreateNasDatastore(ctx context.Context, hosts []*object.HostSystem) error {
	object := types.ManagedObjectReference{
		Type:  "Datastore",
		Value: fmt.Sprintf("%s:%s", cmd.RemoteHost, cmd.RemotePath),
	}

	spec := cmd.GetHostNasVolumeSpec()

	for _, host := range hosts {
		ds, err := host.ConfigManager().DatastoreSystem(ctx)
		if err != nil {
			return err
		}

		_, err = ds.CreateNasDatastore(ctx, spec)
		if err != nil {
			if soap.IsSoapFault(err) {
				switch fault := soap.ToSoapFault(err).VimFault().(type) {
				case types.PlatformConfigFault:
					if len(fault.FaultMessage) != 0 {
						return errors.New(fault.FaultMessage[0].Message)
					}
				case types.DuplicateName:
					if cmd.Force && fault.Object == object {
						fmt.Fprintf(os.Stderr, "%s: '%s' already mounted\n",
							host.InventoryPath, cmd.Name)
						continue
					}
				}
			}

			return fmt.Errorf("%s: %s", host.InventoryPath, err)
		}
	}

	return nil
}

func (cmd *create) CreateVmfsDatastore(ctx context.Context, hosts []*object.HostSystem) error {
	for _, host := range hosts {
		ds, err := host.ConfigManager().DatastoreSystem(ctx)
		if err != nil {
			return err
		}

		// Find the specified disk
		disks, err := ds.QueryAvailableDisksForVmfs(ctx)
		if err != nil {
			return err
		}

		var disk *types.HostScsiDisk
		for _, e := range disks {
			if e.CanonicalName == cmd.DiskCanonicalName {
				disk = &e
				break
			}
		}

		if disk == nil {
			return fmt.Errorf("no eligible disk found for name %#v", cmd.DiskCanonicalName)
		}

		// Query for creation options and pick the right one
		options, err := ds.QueryVmfsDatastoreCreateOptions(ctx, disk.DevicePath)
		if err != nil {
			return err
		}

		var option *types.VmfsDatastoreOption
		for _, e := range options {
			if _, ok := e.Info.(*types.VmfsDatastoreAllExtentOption); ok {
				option = &e
				break
			}
		}

		if option == nil {
			return fmt.Errorf("cannot use entire disk for datastore for name %#v", cmd.DiskCanonicalName)
		}

		spec := *option.Spec.(*types.VmfsDatastoreCreateSpec)
		spec.Vmfs.VolumeName = cmd.Name
		_, err = ds.CreateVmfsDatastore(ctx, spec)
		if err != nil {
			return err
		}
	}

	return nil
}

func (cmd *create) CreateLocalDatastore(ctx context.Context, hosts []*object.HostSystem) error {
	for _, host := range hosts {
		ds, err := host.ConfigManager().DatastoreSystem(ctx)
		if err != nil {
			return err
		}

		if cmd.Path == "" {
			cmd.Path = cmd.Name
		}

		if cmd.Name == "" {
			cmd.Name = filepath.Base(cmd.Path)
		}

		_, err = ds.CreateLocalDatastore(ctx, cmd.Name, cmd.Path)
		if err != nil {
			return err
		}
	}

	return nil
}
