/*
Copyright (c) 2017 VMware, Inc. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
nSee the License for the specific language governing permissions and
limitations under the License.
*/

package vmdk

import (
	"bytes"
	"context"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"os"
	"path"
	"path/filepath"
	"strings"
	"text/template"

	"github.com/vmware/govmomi/object"
	"github.com/vmware/govmomi/ovf"
	"github.com/vmware/govmomi/vim25"
	"github.com/vmware/govmomi/vim25/progress"
	"github.com/vmware/govmomi/vim25/soap"
	"github.com/vmware/govmomi/vim25/types"
)

var (
	ErrInvalidFormat = errors.New("vmdk: invalid format (must be streamOptimized)")
)

// info is used to inspect a vmdk and generate an ovf template
type info struct {
	Header struct {
		MagicNumber uint32
		Version     uint32
		Flags       uint32
		Capacity    uint64
	}

	Capacity   uint64
	Size       int64
	Name       string
	ImportName string
}

// stat looks at the vmdk header to make sure the format is streamOptimized and
// extracts the disk capacity required to properly generate the ovf descriptor.
func stat(name string) (*info, error) {
	f, err := os.Open(name)
	if err != nil {
		return nil, err
	}

	var di info

	var buf bytes.Buffer

	_, err = io.CopyN(&buf, f, int64(binary.Size(di.Header)))

	fi, _ := f.Stat()

	_ = f.Close()

	if err != nil {
		return nil, err
	}

	err = binary.Read(&buf, binary.LittleEndian, &di.Header)
	if err != nil {
		return nil, err
	}

	if di.Header.MagicNumber != 0x564d444b { // SPARSE_MAGICNUMBER
		return nil, ErrInvalidFormat
	}

	if di.Header.Flags&(1<<16) == 0 { // SPARSEFLAG_COMPRESSED
		// Needs to be converted, for example:
		//   vmware-vdiskmanager -r src.vmdk -t 5 dst.vmdk
		//   qemu-img convert -O vmdk -o subformat=streamOptimized src.vmdk dst.vmdk
		return nil, ErrInvalidFormat
	}

	di.Capacity = di.Header.Capacity * 512 // VMDK_SECTOR_SIZE
	di.Size = fi.Size()
	di.Name = filepath.Base(name)
	di.ImportName = strings.TrimSuffix(di.Name, ".vmdk")

	return &di, nil
}

// ovfenv is the minimal descriptor template required to import a vmdk
var ovfenv = `<?xml version="1.0" encoding="UTF-8"?>
<Envelope xmlns="http://schemas.dmtf.org/ovf/envelope/1"
          xmlns:ovf="http://schemas.dmtf.org/ovf/envelope/1"
          xmlns:cim="http://schemas.dmtf.org/wbem/wscim/1/common"
          xmlns:rasd="http://schemas.dmtf.org/wbem/wscim/1/cim-schema/2/CIM_ResourceAllocationSettingData"
          xmlns:vmw="http://www.vmware.com/schema/ovf"
          xmlns:vssd="http://schemas.dmtf.org/wbem/wscim/1/cim-schema/2/CIM_VirtualSystemSettingData"
          xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
  <References>
    <File ovf:href="{{ .Name }}" ovf:id="file1" ovf:size="{{ .Size }}"/>
  </References>
  <DiskSection>
    <Info>Virtual disk information</Info>
    <Disk ovf:capacity="{{ .Capacity }}" ovf:capacityAllocationUnits="byte" ovf:diskId="vmdisk1" ovf:fileRef="file1" ovf:format="http://www.vmware.com/interfaces/specifications/vmdk.html#streamOptimized" ovf:populatedSize="0"/>
  </DiskSection>
  <VirtualSystem ovf:id="{{ .ImportName }}">
    <Info>A virtual machine</Info>
    <Name>{{ .ImportName }}</Name>
    <OperatingSystemSection ovf:id="100" vmw:osType="other26xLinux64Guest">
      <Info>The kind of installed guest operating system</Info>
    </OperatingSystemSection>
    <VirtualHardwareSection>
      <Info>Virtual hardware requirements</Info>
      <System>
        <vssd:ElementName>Virtual Hardware Family</vssd:ElementName>
        <vssd:InstanceID>0</vssd:InstanceID>
        <vssd:VirtualSystemIdentifier>{{ .ImportName }}</vssd:VirtualSystemIdentifier>
        <vssd:VirtualSystemType>vmx-07</vssd:VirtualSystemType>
      </System>
      <Item>
        <rasd:AllocationUnits>hertz * 10^6</rasd:AllocationUnits>
        <rasd:Description>Number of Virtual CPUs</rasd:Description>
        <rasd:ElementName>1 virtual CPU(s)</rasd:ElementName>
        <rasd:InstanceID>1</rasd:InstanceID>
        <rasd:ResourceType>3</rasd:ResourceType>
        <rasd:VirtualQuantity>1</rasd:VirtualQuantity>
      </Item>
      <Item>
        <rasd:AllocationUnits>byte * 2^20</rasd:AllocationUnits>
        <rasd:Description>Memory Size</rasd:Description>
        <rasd:ElementName>1024MB of memory</rasd:ElementName>
        <rasd:InstanceID>2</rasd:InstanceID>
        <rasd:ResourceType>4</rasd:ResourceType>
        <rasd:VirtualQuantity>1024</rasd:VirtualQuantity>
      </Item>
      <Item>
        <rasd:Address>0</rasd:Address>
        <rasd:Description>SCSI Controller</rasd:Description>
        <rasd:ElementName>SCSI Controller 0</rasd:ElementName>
        <rasd:InstanceID>3</rasd:InstanceID>
        <rasd:ResourceSubType>VirtualSCSI</rasd:ResourceSubType>
        <rasd:ResourceType>6</rasd:ResourceType>
      </Item>
      <Item>
        <rasd:AddressOnParent>0</rasd:AddressOnParent>
        <rasd:ElementName>Hard Disk 1</rasd:ElementName>
        <rasd:HostResource>ovf:/disk/vmdisk1</rasd:HostResource>
        <rasd:InstanceID>9</rasd:InstanceID>
        <rasd:Parent>3</rasd:Parent>
        <rasd:ResourceType>17</rasd:ResourceType>
        <vmw:Config ovf:required="false" vmw:key="backing.writeThrough" vmw:value="false"/>
      </Item>
    </VirtualHardwareSection>
  </VirtualSystem>
</Envelope>`

// ovf returns an expanded descriptor template
func (di *info) ovf() (string, error) {
	var buf bytes.Buffer

	tmpl, err := template.New("ovf").Parse(ovfenv)
	if err != nil {
		return "", err
	}

	err = tmpl.Execute(&buf, di)
	if err != nil {
		return "", err
	}

	return buf.String(), nil
}

// ImportParams contains the set of optional params to the Import function.
// Note that "optional" may depend on environment, such as ESX or vCenter.
type ImportParams struct {
	Path       string
	Logger     progress.Sinker
	Type       types.VirtualDiskType
	Force      bool
	Datacenter *object.Datacenter
	Pool       *object.ResourcePool
	Folder     *object.Folder
	Host       *object.HostSystem
}

// Import uploads a local vmdk file specified by name to the given datastore.
func Import(ctx context.Context, c *vim25.Client, name string, datastore *object.Datastore, p ImportParams) error {
	m := ovf.NewManager(c)
	fm := datastore.NewFileManager(p.Datacenter, p.Force)

	disk, err := stat(name)
	if err != nil {
		return err
	}

	var rename string

	p.Path = strings.TrimSuffix(p.Path, "/")
	if p.Path != "" {
		disk.ImportName = p.Path
		rename = path.Join(disk.ImportName, disk.Name)
	}

	// "target" is the path that will be created by ImportVApp()
	// ImportVApp uses the same name for the VM and the disk.
	target := fmt.Sprintf("%s/%s.vmdk", disk.ImportName, disk.ImportName)

	if _, err = datastore.Stat(ctx, target); err == nil {
		if p.Force {
			// If we don't delete, the nfc upload adds a file name suffix
			if err = fm.Delete(ctx, target); err != nil {
				return err
			}
		} else {
			return fmt.Errorf("%s: %s", os.ErrExist, datastore.Path(target))
		}
	}

	// If we need to rename at the end, check if the file exists early unless Force.
	if !p.Force && rename != "" {
		if _, err = datastore.Stat(ctx, rename); err == nil {
			return fmt.Errorf("%s: %s", os.ErrExist, datastore.Path(rename))
		}
	}

	// Expand the ovf template
	descriptor, err := disk.ovf()
	if err != nil {
		return err
	}

	pool := p.Pool     // TODO: use datastore to derive a default
	folder := p.Folder // TODO: use datacenter to derive a default

	kind := p.Type
	if kind == "" {
		kind = types.VirtualDiskTypeThin
	}

	params := types.OvfCreateImportSpecParams{
		DiskProvisioning: string(kind),
		EntityName:       disk.ImportName,
	}

	spec, err := m.CreateImportSpec(ctx, descriptor, pool, datastore, params)
	if err != nil {
		return err
	}
	if spec.Error != nil {
		return errors.New(spec.Error[0].LocalizedMessage)
	}

	lease, err := pool.ImportVApp(ctx, spec.ImportSpec, folder, p.Host)
	if err != nil {
		return err
	}

	info, err := lease.Wait(ctx, spec.FileItem)
	if err != nil {
		return err
	}

	f, err := os.Open(name)
	if err != nil {
		return err
	}

	opts := soap.Upload{
		ContentLength: disk.Size,
		Progress:      p.Logger,
	}

	u := lease.StartUpdater(ctx, info)
	defer u.Done()

	item := info.Items[0] // we only have 1 disk to upload

	err = lease.Upload(ctx, item, f, opts)

	_ = f.Close()

	if err != nil {
		return err
	}

	if err = lease.Complete(ctx); err != nil {
		return err
	}

	// ImportVApp created a VM, here we detach the vmdk, then delete the VM.
	vm := object.NewVirtualMachine(c, info.Entity)

	device, err := vm.Device(ctx)
	if err != nil {
		return err
	}

	device = device.SelectByType((*types.VirtualDisk)(nil))

	err = vm.RemoveDevice(ctx, true, device...)
	if err != nil {
		return err
	}

	task, err := vm.Destroy(ctx)
	if err != nil {
		return err
	}

	if err = task.Wait(ctx); err != nil {
		return err
	}

	if rename == "" {
		return nil
	}

	return fm.Move(ctx, target, rename)
}
