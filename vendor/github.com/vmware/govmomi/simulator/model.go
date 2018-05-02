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
	"context"
	"fmt"
	"io/ioutil"
	"os"
	"path"

	"github.com/vmware/govmomi/object"
	"github.com/vmware/govmomi/simulator/esx"
	"github.com/vmware/govmomi/simulator/vpx"
	"github.com/vmware/govmomi/vim25/mo"
	"github.com/vmware/govmomi/vim25/types"
)

// Model is used to populate a Model with an initial set of managed entities.
// This is a simple helper for tests running against a simulator, to populate an inventory
// with commonly used models.
type Model struct {
	Service *Service `json:"-"`

	ServiceContent types.ServiceContent `json:"-"`
	RootFolder     mo.Folder            `json:"-"`

	// Autostart will power on Model created VMs when true
	Autostart bool `json:"-"`

	// Datacenter specifies the number of Datacenter entities to create
	Datacenter int

	// Portgroup specifies the number of DistributedVirtualPortgroup entities to create per Datacenter
	Portgroup int

	// Host specifies the number of standalone HostSystems entities to create per Datacenter
	Host int `json:",omitempty"`

	// Cluster specifies the number of ClusterComputeResource entities to create per Datacenter
	Cluster int

	// ClusterHost specifies the number of HostSystems entities to create within a Cluster
	ClusterHost int `json:",omitempty"`

	// Pool specifies the number of ResourcePool entities to create per Cluster
	Pool int

	// Datastore specifies the number of Datastore entities to create
	// Each Datastore will have temporary local file storage and will be mounted
	// on every HostSystem created by the ModelConfig
	Datastore int

	// Machine specifies the number of VirtualMachine entities to create per ResourcePool
	Machine int

	// Folder specifies the number of Datacenter to place within a Folder.
	// This includes a folder for the Datacenter itself and its host, vm, network and datastore folders.
	// All resources for the Datacenter are placed within these folders, rather than the top-level folders.
	Folder int

	// App specifies the number of VirtualApp to create per Cluster
	App int

	// Pod specifies the number of StoragePod to create per Cluster
	Pod int

	// total number of inventory objects, set by Count()
	total int

	dirs []string
}

// ESX is the default Model for a standalone ESX instance
func ESX() *Model {
	return &Model{
		ServiceContent: esx.ServiceContent,
		RootFolder:     esx.RootFolder,
		Autostart:      true,
		Datastore:      1,
		Machine:        2,
	}
}

// VPX is the default Model for a vCenter instance
func VPX() *Model {
	return &Model{
		ServiceContent: vpx.ServiceContent,
		RootFolder:     vpx.RootFolder,
		Autostart:      true,
		Datacenter:     1,
		Portgroup:      1,
		Host:           1,
		Cluster:        1,
		ClusterHost:    3,
		Datastore:      1,
		Machine:        2,
	}
}

// Count returns a Model with total number of each existing type
func (m *Model) Count() Model {
	count := Model{}

	for ref, obj := range Map.objects {
		if _, ok := obj.(mo.Entity); !ok {
			continue
		}

		count.total++

		switch ref.Type {
		case "Datacenter":
			count.Datacenter++
		case "DistributedVirtualPortgroup":
			count.Portgroup++
		case "ClusterComputeResource":
			count.Cluster++
		case "Datastore":
			count.Datastore++
		case "HostSystem":
			count.Host++
		case "VirtualMachine":
			count.Machine++
		case "ResourcePool":
			count.Pool++
		case "VirtualApp":
			count.App++
		case "Folder":
			count.Folder++
		case "StoragePod":
			count.Pod++
		}
	}

	return count
}

func (*Model) fmtName(prefix string, num int) string {
	return fmt.Sprintf("%s%d", prefix, num)
}

// Create populates the Model with the given ModelConfig
func (m *Model) Create() error {
	m.Service = New(NewServiceInstance(m.ServiceContent, m.RootFolder))

	ctx := context.Background()
	client := m.Service.client
	root := object.NewRootFolder(client)

	// After all hosts are created, this var is used to mount the host datastores.
	var hosts []*object.HostSystem
	hostMap := make(map[string][]*object.HostSystem)

	// We need to defer VM creation until after the datastores are created.
	var vms []func() error
	// 1 DVS per DC, added to all hosts
	var dvs *object.DistributedVirtualSwitch
	// 1 NIC per VM, backed by a DVPG if Model.Portgroup > 0
	vmnet := esx.EthernetCard.Backing

	// addHost adds a cluster host or a stanalone host.
	addHost := func(name string, f func(types.HostConnectSpec) (*object.Task, error)) (*object.HostSystem, error) {
		spec := types.HostConnectSpec{
			HostName: name,
		}

		task, err := f(spec)
		if err != nil {
			return nil, err
		}

		info, err := task.WaitForResult(context.Background(), nil)
		if err != nil {
			return nil, err
		}

		host := object.NewHostSystem(client, info.Result.(types.ManagedObjectReference))
		hosts = append(hosts, host)

		if dvs != nil {
			config := &types.DVSConfigSpec{
				Host: []types.DistributedVirtualSwitchHostMemberConfigSpec{{
					Operation: string(types.ConfigSpecOperationAdd),
					Host:      host.Reference(),
				}},
			}

			_, _ = dvs.Reconfigure(ctx, config)
		}

		return host, nil
	}

	// addMachine returns a func to create a VM.
	addMachine := func(prefix string, host *object.HostSystem, pool *object.ResourcePool, folders *object.DatacenterFolders) {
		nic := esx.EthernetCard
		nic.Backing = vmnet
		ds := types.ManagedObjectReference{}

		f := func() error {
			for i := 0; i < m.Machine; i++ {
				name := m.fmtName(prefix+"_VM", i)

				config := types.VirtualMachineConfigSpec{
					Name:    name,
					GuestId: string(types.VirtualMachineGuestOsIdentifierOtherGuest),
					Files: &types.VirtualMachineFileInfo{
						VmPathName: "[LocalDS_0]",
					},
				}

				if pool == nil {
					pool, _ = host.ResourcePool(ctx)
				}

				var devices object.VirtualDeviceList

				scsi, _ := devices.CreateSCSIController("pvscsi")
				ide, _ := devices.CreateIDEController()
				cdrom, _ := devices.CreateCdrom(ide.(*types.VirtualIDEController))
				disk := devices.CreateDisk(scsi.(types.BaseVirtualController), ds,
					config.Files.VmPathName+" "+path.Join(name, "disk1.vmdk"))
				disk.CapacityInKB = 1024

				devices = append(devices, scsi, cdrom, disk, &nic)

				config.DeviceChange, _ = devices.ConfigSpec(types.VirtualDeviceConfigSpecOperationAdd)

				task, err := folders.VmFolder.CreateVM(ctx, config, pool, host)
				if err != nil {
					return err
				}

				info, err := task.WaitForResult(ctx, nil)
				if err != nil {
					return err
				}

				vm := object.NewVirtualMachine(client, info.Result.(types.ManagedObjectReference))

				if m.Autostart {
					_, _ = vm.PowerOn(ctx)
				}
			}

			return nil
		}

		vms = append(vms, f)
	}

	nfolder := 0

	for ndc := 0; ndc < m.Datacenter; ndc++ {
		dcName := m.fmtName("DC", ndc)
		folder := root
		fName := m.fmtName("F", nfolder)

		// If Datacenter > Folder, don't create folders for the first N DCs.
		if nfolder < m.Folder && ndc >= (m.Datacenter-m.Folder) {
			f, err := folder.CreateFolder(ctx, fName)
			if err != nil {
				return err
			}
			folder = f
		}

		dc, err := folder.CreateDatacenter(ctx, dcName)
		if err != nil {
			return err
		}

		folders, err := dc.Folders(ctx)
		if err != nil {
			return err
		}

		if m.Pod > 0 {
			for pod := 0; pod < m.Pod; pod++ {
				_, _ = folders.DatastoreFolder.CreateStoragePod(ctx, m.fmtName(dcName+"_POD", pod))
			}
		}

		if folder != root {
			// Create sub-folders and use them to create any resources that follow
			subs := []**object.Folder{&folders.DatastoreFolder, &folders.HostFolder, &folders.NetworkFolder, &folders.VmFolder}

			for _, sub := range subs {
				f, err := (*sub).CreateFolder(ctx, fName)
				if err != nil {
					return err
				}

				*sub = f
			}

			nfolder++
		}

		if m.Portgroup > 0 {
			var spec types.DVSCreateSpec
			spec.ConfigSpec = &types.VMwareDVSConfigSpec{}
			spec.ConfigSpec.GetDVSConfigSpec().Name = m.fmtName("DVS", 0)

			task, err := folders.NetworkFolder.CreateDVS(ctx, spec)
			if err != nil {
				return err
			}

			info, err := task.WaitForResult(ctx, nil)
			if err != nil {
				return err
			}

			dvs = object.NewDistributedVirtualSwitch(client, info.Result.(types.ManagedObjectReference))

			for npg := 0; npg < m.Portgroup; npg++ {
				name := m.fmtName(dcName+"_DVPG", npg)

				task, err = dvs.AddPortgroup(ctx, []types.DVPortgroupConfigSpec{{Name: name}})
				if err != nil {
					return err
				}

				err = task.Wait(ctx)
				if err != nil {
					return err
				}

				// Use the 1st DVPG for the VMs eth0 backing
				if npg == 0 {
					// AddPortgroup_Task does not return the moid, so we look it up by name
					net := Map.Get(folders.NetworkFolder.Reference()).(*Folder)
					pg := Map.FindByName(name, net.ChildEntity)

					vmnet, _ = object.NewDistributedVirtualPortgroup(client, pg.Reference()).EthernetCardBackingInfo(ctx)
				}
			}
		}

		for nhost := 0; nhost < m.Host; nhost++ {
			name := m.fmtName(dcName+"_H", nhost)

			host, err := addHost(name, func(spec types.HostConnectSpec) (*object.Task, error) {
				return folders.HostFolder.AddStandaloneHost(ctx, spec, true, nil, nil)
			})
			if err != nil {
				return err
			}

			addMachine(name, host, nil, folders)
		}

		for ncluster := 0; ncluster < m.Cluster; ncluster++ {
			clusterName := m.fmtName(dcName+"_C", ncluster)

			cluster, err := folders.HostFolder.CreateCluster(ctx, clusterName, types.ClusterConfigSpecEx{})
			if err != nil {
				return err
			}

			for nhost := 0; nhost < m.ClusterHost; nhost++ {
				name := m.fmtName(clusterName+"_H", nhost)

				_, err = addHost(name, func(spec types.HostConnectSpec) (*object.Task, error) {
					return cluster.AddHost(ctx, spec, true, nil, nil)
				})
				if err != nil {
					return err
				}
			}

			pool, err := cluster.ResourcePool(ctx)
			if err != nil {
				return err
			}

			prefix := clusterName + "_RP"

			addMachine(prefix+"0", nil, pool, folders)

			for npool := 1; npool <= m.Pool; npool++ {
				spec := types.DefaultResourceConfigSpec()

				_, err = pool.Create(ctx, m.fmtName(prefix, npool), spec)
				if err != nil {
					return err
				}
			}

			prefix = clusterName + "_APP"

			for napp := 0; napp < m.App; napp++ {
				rspec := types.DefaultResourceConfigSpec()
				vspec := NewVAppConfigSpec()
				name := m.fmtName(prefix, napp)

				vapp, err := pool.CreateVApp(ctx, name, rspec, vspec, nil)
				if err != nil {
					return err
				}

				addMachine(name, nil, vapp.ResourcePool, folders)
			}
		}

		hostMap[dcName] = hosts
		hosts = nil
	}

	if m.ServiceContent.RootFolder == esx.RootFolder.Reference() {
		// ESX model
		host := object.NewHostSystem(client, esx.HostSystem.Reference())

		dc := object.NewDatacenter(client, esx.Datacenter.Reference())
		folders, err := dc.Folders(ctx)
		if err != nil {
			return err
		}

		hostMap[dc.Reference().Value] = append(hosts, host)

		addMachine(host.Reference().Value, host, nil, folders)
	}

	for dc, dchosts := range hostMap {
		for i := 0; i < m.Datastore; i++ {
			err := m.createLocalDatastore(dc, m.fmtName("LocalDS_", i), dchosts)
			if err != nil {
				return err
			}
		}
	}

	for _, createVM := range vms {
		err := createVM()
		if err != nil {
			return err
		}
	}

	return nil
}

func (m *Model) createLocalDatastore(dc string, name string, hosts []*object.HostSystem) error {
	ctx := context.Background()
	dir, err := ioutil.TempDir("", fmt.Sprintf("govcsim-%s-%s-", dc, name))
	if err != nil {
		return err
	}

	m.dirs = append(m.dirs, dir)

	for _, host := range hosts {
		dss, err := host.ConfigManager().DatastoreSystem(ctx)
		if err != nil {
			return err
		}

		_, err = dss.CreateLocalDatastore(ctx, name, dir)
		if err != nil {
			return err
		}
	}

	return nil
}

// Remove cleans up items created by the Model, such as local datastore directories
func (m *Model) Remove() {
	for _, dir := range m.dirs {
		_ = os.RemoveAll(dir)
	}
}
