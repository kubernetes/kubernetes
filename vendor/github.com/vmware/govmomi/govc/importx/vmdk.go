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

package importx

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"path"
	"reflect"
	"regexp"

	"github.com/vmware/govmomi/govc/cli"
	"github.com/vmware/govmomi/govc/flags"
	"github.com/vmware/govmomi/object"
	"github.com/vmware/govmomi/property"
	"github.com/vmware/govmomi/vim25"
	"github.com/vmware/govmomi/vim25/mo"
	"github.com/vmware/govmomi/vim25/progress"
	"github.com/vmware/govmomi/vim25/soap"
	"github.com/vmware/govmomi/vim25/types"
)

type vmdk struct {
	*flags.DatastoreFlag
	*flags.ResourcePoolFlag
	*flags.OutputFlag

	upload bool
	force  bool
	keep   bool

	Client       *vim25.Client
	Datacenter   *object.Datacenter
	Datastore    *object.Datastore
	ResourcePool *object.ResourcePool
}

func init() {
	cli.Register("import.vmdk", &vmdk{})
	cli.Alias("import.vmdk", "datastore.import")
}

func (cmd *vmdk) Register(ctx context.Context, f *flag.FlagSet) {
	cmd.DatastoreFlag, ctx = flags.NewDatastoreFlag(ctx)
	cmd.DatastoreFlag.Register(ctx, f)
	cmd.ResourcePoolFlag, ctx = flags.NewResourcePoolFlag(ctx)
	cmd.ResourcePoolFlag.Register(ctx, f)
	cmd.OutputFlag, ctx = flags.NewOutputFlag(ctx)
	cmd.OutputFlag.Register(ctx, f)

	f.BoolVar(&cmd.upload, "upload", true, "Upload specified disk")
	f.BoolVar(&cmd.force, "force", false, "Overwrite existing disk")
	f.BoolVar(&cmd.keep, "keep", false, "Keep uploaded disk after import")
}

func (cmd *vmdk) Process(ctx context.Context) error {
	if err := cmd.DatastoreFlag.Process(ctx); err != nil {
		return err
	}
	if err := cmd.ResourcePoolFlag.Process(ctx); err != nil {
		return err
	}
	if err := cmd.OutputFlag.Process(ctx); err != nil {
		return err
	}
	return nil
}

func (cmd *vmdk) Usage() string {
	return "PATH_TO_VMDK [REMOTE_DIRECTORY]"
}

func (cmd *vmdk) Run(ctx context.Context, f *flag.FlagSet) error {
	var err error

	args := f.Args()
	if len(args) < 1 {
		return errors.New("no file to import")
	}

	file := importable{
		localPath: f.Arg(0),
	}

	// Include remote path if specified
	if len(args) >= 2 {
		file.remotePath = f.Arg(1)
	}

	cmd.Client, err = cmd.DatastoreFlag.Client()
	if err != nil {
		return err
	}

	cmd.Datacenter, err = cmd.DatastoreFlag.Datacenter()
	if err != nil {
		return err
	}

	cmd.Datastore, err = cmd.DatastoreFlag.Datastore()
	if err != nil {
		return err
	}

	cmd.ResourcePool, err = cmd.ResourcePoolFlag.ResourcePool()
	if err != nil {
		return err
	}

	err = cmd.PrepareDestination(file)
	if err != nil {
		return err
	}

	if cmd.upload {
		err = cmd.Upload(file)
		if err != nil {
			return err
		}
	}

	return cmd.Import(file)
}

// PrepareDestination makes sure that the destination VMDK does not yet exist.
// If the force flag is passed, it removes the existing VMDK. This functions
// exists to give a meaningful error if the remote VMDK already exists.
//
// CopyVirtualDisk can return a "<src> file does not exist" error while in fact
// the source file *does* exist and the *destination* file also exist.
//
func (cmd *vmdk) PrepareDestination(i importable) error {
	ctx := context.TODO()
	vmdkPath := i.RemoteDstVMDK()
	res, err := cmd.Datastore.Stat(ctx, vmdkPath)
	if err != nil {
		switch err.(type) {
		case object.DatastoreNoSuchDirectoryError:
			// The base path doesn't exist. Create it.
			dsPath := cmd.Datastore.Path(path.Dir(vmdkPath))
			m := object.NewFileManager(cmd.Client)
			return m.MakeDirectory(ctx, dsPath, cmd.Datacenter, true)
		case object.DatastoreNoSuchFileError:
			// Destination path doesn't exist; all good to continue with import.
			return nil
		}

		return err
	}

	// Check that the returned entry has the right type.
	switch res.(type) {
	case *types.VmDiskFileInfo:
	default:
		expected := "VmDiskFileInfo"
		actual := reflect.TypeOf(res)
		panic(fmt.Sprintf("Expected: %s, actual: %s", expected, actual))
	}

	if !cmd.force {
		dsPath := cmd.Datastore.Path(vmdkPath)
		err = fmt.Errorf("File %s already exists", dsPath)
		return err
	}

	// Delete existing disk.
	err = cmd.DeleteDisk(vmdkPath)
	if err != nil {
		return err
	}

	return nil
}

func (cmd *vmdk) Upload(i importable) error {
	ctx := context.TODO()
	p := soap.DefaultUpload
	if cmd.OutputFlag.TTY {
		logger := cmd.ProgressLogger("Uploading... ")
		p.Progress = logger
		defer logger.Wait()
	}

	return cmd.Datastore.UploadFile(ctx, i.localPath, i.RemoteSrcVMDK(), &p)
}

func (cmd *vmdk) Import(i importable) error {
	err := cmd.Copy(i)
	if err != nil {
		return err
	}

	if !cmd.keep {
		err = cmd.DeleteDisk(i.RemoteSrcVMDK())
		if err != nil {
			return err
		}
	}

	return nil
}

func (cmd *vmdk) Copy(i importable) error {
	var err error

	logger := cmd.ProgressLogger("Importing... ")
	defer logger.Wait()

	agg := progress.NewAggregator(logger)
	defer agg.Done()

	switch p := cmd.Client.ServiceContent.About.ApiType; p {
	case "HostAgent":
		err = cmd.CopyHostAgent(i, agg)
	case "VirtualCenter":
		err = cmd.CopyVirtualCenter(i, agg)
	default:
		return fmt.Errorf("unsupported ApiType: %s", p)
	}

	return err
}

func (cmd *vmdk) CopyHostAgent(i importable, s progress.Sinker) error {
	ctx := context.TODO()
	spec := &types.VirtualDiskSpec{
		AdapterType: "lsiLogic",
		DiskType:    "thin",
	}

	dc := cmd.Datacenter
	src := cmd.Datastore.Path(i.RemoteSrcVMDK())
	dst := cmd.Datastore.Path(i.RemoteDstVMDK())
	vdm := object.NewVirtualDiskManager(cmd.Client)
	task, err := vdm.CopyVirtualDisk(ctx, src, dc, dst, dc, spec, false)
	if err != nil {
		return err
	}

	ps := progress.Prefix(s, "copying disk")
	_, err = task.WaitForResult(ctx, ps)
	if err != nil {
		return err
	}

	return nil
}

func (cmd *vmdk) CopyVirtualCenter(i importable, s progress.Sinker) error {
	var err error

	srcName := i.BaseClean() + "-srcvm"
	dstName := i.BaseClean() + "-dstvm"

	spec := &configSpec{
		Name:    srcName,
		GuestId: "otherGuest",
		Files: &types.VirtualMachineFileInfo{
			VmPathName: fmt.Sprintf("[%s]", cmd.Datastore.Name()),
		},
	}

	spec.AddDisk(cmd.Datastore, i.RemoteSrcVMDK())

	src, err := cmd.CreateVM(spec)
	if err != nil {
		return err
	}

	dst, err := cmd.CloneVM(src, dstName)
	if err != nil {
		return err
	}

	err = cmd.DestroyVM(src)
	if err != nil {
		return err
	}

	vmdk, err := cmd.DetachDisk(dst)
	if err != nil {
		return err
	}

	err = cmd.MoveDisk(vmdk, i.RemoteDstVMDK())
	if err != nil {
		return err
	}

	err = cmd.DestroyVM(dst)
	if err != nil {
		return err
	}

	return nil
}

func (cmd *vmdk) MoveDisk(src, dst string) error {
	ctx := context.TODO()
	dsSrc := cmd.Datastore.Path(src)
	dsDst := cmd.Datastore.Path(dst)
	vdm := object.NewVirtualDiskManager(cmd.Client)
	task, err := vdm.MoveVirtualDisk(ctx, dsSrc, cmd.Datacenter, dsDst, cmd.Datacenter, true)
	if err != nil {
		return err
	}

	return task.Wait(ctx)
}

func (cmd *vmdk) DeleteDisk(path string) error {
	ctx := context.TODO()
	vdm := object.NewVirtualDiskManager(cmd.Client)
	task, err := vdm.DeleteVirtualDisk(ctx, cmd.Datastore.Path(path), cmd.Datacenter)
	if err != nil {
		return err
	}

	return task.Wait(ctx)
}

func (cmd *vmdk) DetachDisk(vm *object.VirtualMachine) (string, error) {
	ctx := context.TODO()
	var mvm mo.VirtualMachine

	pc := property.DefaultCollector(cmd.Client)
	err := pc.RetrieveOne(ctx, vm.Reference(), []string{"config.hardware"}, &mvm)
	if err != nil {
		return "", err
	}

	spec := new(configSpec)
	dsFile := spec.RemoveDisk(&mvm)

	task, err := vm.Reconfigure(ctx, spec.ToSpec())
	if err != nil {
		return "", err
	}

	err = task.Wait(ctx)
	if err != nil {
		return "", err
	}

	return dsFile, nil
}

func (cmd *vmdk) CreateVM(spec *configSpec) (*object.VirtualMachine, error) {
	ctx := context.TODO()
	folders, err := cmd.Datacenter.Folders(ctx)
	if err != nil {
		return nil, err
	}

	task, err := folders.VmFolder.CreateVM(ctx, spec.ToSpec(), cmd.ResourcePool, nil)
	if err != nil {
		return nil, err
	}

	info, err := task.WaitForResult(ctx, nil)
	if err != nil {
		return nil, err
	}

	return object.NewVirtualMachine(cmd.Client, info.Result.(types.ManagedObjectReference)), nil
}

func (cmd *vmdk) CloneVM(vm *object.VirtualMachine, name string) (*object.VirtualMachine, error) {
	ctx := context.TODO()
	folders, err := cmd.Datacenter.Folders(ctx)
	if err != nil {
		return nil, err
	}

	spec := types.VirtualMachineCloneSpec{
		Config:   &types.VirtualMachineConfigSpec{},
		Location: types.VirtualMachineRelocateSpec{},
	}

	task, err := vm.Clone(ctx, folders.VmFolder, name, spec)
	if err != nil {
		return nil, err
	}

	info, err := task.WaitForResult(ctx, nil)
	if err != nil {
		return nil, err
	}

	return object.NewVirtualMachine(cmd.Client, info.Result.(types.ManagedObjectReference)), nil
}

func (cmd *vmdk) DestroyVM(vm *object.VirtualMachine) error {
	ctx := context.TODO()
	_, err := cmd.DetachDisk(vm)
	if err != nil {
		return err
	}

	task, err := vm.Destroy(ctx)
	if err != nil {
		return err
	}

	err = task.Wait(ctx)
	if err != nil {
		return err
	}

	return nil
}

type configSpec types.VirtualMachineConfigSpec

func (c *configSpec) ToSpec() types.VirtualMachineConfigSpec {
	return types.VirtualMachineConfigSpec(*c)
}

func (c *configSpec) AddChange(d types.BaseVirtualDeviceConfigSpec) {
	c.DeviceChange = append(c.DeviceChange, d)
}

func (c *configSpec) AddDisk(ds *object.Datastore, path string) {
	var devices object.VirtualDeviceList

	controller, err := devices.CreateSCSIController("")
	if err != nil {
		panic(err)
	}
	devices = append(devices, controller)

	disk := devices.CreateDisk(controller.(types.BaseVirtualController), ds.Reference(), ds.Path(path))
	devices = append(devices, disk)

	spec, err := devices.ConfigSpec(types.VirtualDeviceConfigSpecOperationAdd)
	if err != nil {
		panic(err)
	}

	c.DeviceChange = append(c.DeviceChange, spec...)
}

var dsPathRegexp = regexp.MustCompile(`^\[.*\] (.*)$`)

func (c *configSpec) RemoveDisk(vm *mo.VirtualMachine) string {
	var file string

	for _, d := range vm.Config.Hardware.Device {
		switch device := d.(type) {
		case *types.VirtualDisk:
			if file != "" {
				panic("expected VM to have only one disk")
			}

			switch backing := device.Backing.(type) {
			case *types.VirtualDiskFlatVer1BackingInfo:
				file = backing.FileName
			case *types.VirtualDiskFlatVer2BackingInfo:
				file = backing.FileName
			case *types.VirtualDiskSeSparseBackingInfo:
				file = backing.FileName
			case *types.VirtualDiskSparseVer1BackingInfo:
				file = backing.FileName
			case *types.VirtualDiskSparseVer2BackingInfo:
				file = backing.FileName
			default:
				name := reflect.TypeOf(device.Backing).String()
				panic(fmt.Sprintf("unexpected backing type: %s", name))
			}

			// Remove [datastore] prefix
			m := dsPathRegexp.FindStringSubmatch(file)
			if len(m) != 2 {
				panic(fmt.Sprintf("expected regexp match for %#v", file))
			}
			file = m[1]

			removeOp := &types.VirtualDeviceConfigSpec{
				Operation: types.VirtualDeviceConfigSpecOperationRemove,
				Device:    device,
			}

			c.AddChange(removeOp)
		}
	}

	return file
}
