/*
Copyright 2016 The Kubernetes Authors All rights reserved.

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
	"bytes"
	"errors"
	"fmt"
	"io"
	"net/url"
	"os/exec"
	"strings"

	"github.com/vmware/govmomi"
	"github.com/vmware/govmomi/find"
	"github.com/vmware/govmomi/object"
	"github.com/vmware/govmomi/property"
	"github.com/vmware/govmomi/vim25/mo"
	"github.com/vmware/govmomi/vim25/types"
	"golang.org/x/net/context"
	"gopkg.in/gcfg.v1"

	"github.com/golang/glog"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/cloudprovider"
)

const ProviderName = "vsphere"
const ActivePowerState = "poweredOn"
const DefaultDiskController = "scsi"
const DefaultSCSIControllerType = "lsilogic"

var ErrNoDiskUUIDFound = errors.New("No disk UUID found")
var ErrNoDiskIDFound = errors.New("No vSphere disk ID found")
var ErrNoDevicesFound = errors.New("No devices found")

// VSphere is an implementation of cloud provider Interface for VSphere.
type VSphere struct {
	cfg *VSphereConfig
	// InstanceID of the server where this VSphere object is instantiated.
	localInstanceID string
}

type VSphereConfig struct {
	Global struct {
		User         string `gcfg:"user"`
		Password     string `gcfg:"password"`
		VCenterIP    string `gcfg:"server"`
		VCenterPort  string `gcfg:"port"`
		InsecureFlag bool   `gcfg:"insecure-flag"`
		Datacenter   string `gcfg:"datacenter"`
		Datastore    string `gcfg:"datastore"`
	}

	Network struct {
		PublicNetwork string `gcfg:"public-network"`
	}
	Disk struct {
		DiskController     string `dcfg:"diskcontroller"`
		SCSIControllerType string `dcfg:"scsicontrollertype"`
	}
}

func readConfig(config io.Reader) (VSphereConfig, error) {
	if config == nil {
		err := fmt.Errorf("no vSphere cloud provider config file given")
		return VSphereConfig{}, err
	}

	var cfg VSphereConfig
	err := gcfg.ReadInto(&cfg, config)
	return cfg, err
}

func init() {
	cloudprovider.RegisterCloudProvider(ProviderName, func(config io.Reader) (cloudprovider.Interface, error) {
		cfg, err := readConfig(config)
		if err != nil {
			return nil, err
		}
		return newVSphere(cfg)
	})
}

func readInstanceID(cfg *VSphereConfig) (string, error) {
	cmd := exec.Command("bash", "-c", `dmidecode -t 1 | grep UUID | tr -d ' ' | cut -f 2 -d ':'`)
	var out bytes.Buffer
	cmd.Stdout = &out
	err := cmd.Run()
	if err != nil {
		return "", err
	}
	if out.Len() == 0 {
		return "", fmt.Errorf("unable to retrieve Instance ID")
	}

	// Create context
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	// Create vSphere client
	c, err := vsphereLogin(cfg, ctx)
	if err != nil {
		return "", err
	}
	defer c.Logout(ctx)

	// Create a new finder
	f := find.NewFinder(c.Client, true)

	// Fetch and set data center
	dc, err := f.Datacenter(ctx, cfg.Global.Datacenter)
	if err != nil {
		return "", err
	}
	f.SetDatacenter(dc)

	s := object.NewSearchIndex(c.Client)

	svm, err := s.FindByUuid(ctx, dc, strings.ToLower(strings.TrimSpace(out.String())), true, nil)
	var vm mo.VirtualMachine
	err = s.Properties(ctx, svm.Reference(), []string{"name"}, &vm)
	if err != nil {
		return "", err
	}
	return vm.Name, nil
}

func newVSphere(cfg VSphereConfig) (*VSphere, error) {
	id, err := readInstanceID(&cfg)
	if err != nil {
		return nil, err
	}

	if cfg.Disk.DiskController == "" {
		cfg.Disk.DiskController = DefaultDiskController
	}
	if cfg.Disk.SCSIControllerType == "" {
		cfg.Disk.SCSIControllerType = DefaultSCSIControllerType
	}
	vs := VSphere{
		cfg:             &cfg,
		localInstanceID: id,
	}
	return &vs, nil
}

func vsphereLogin(cfg *VSphereConfig, ctx context.Context) (*govmomi.Client, error) {

	// Parse URL from string
	u, err := url.Parse(fmt.Sprintf("https://%s:%s/sdk", cfg.Global.VCenterIP, cfg.Global.VCenterPort))
	if err != nil {
		return nil, err
	}
	// set username and password for the URL
	u.User = url.UserPassword(cfg.Global.User, cfg.Global.Password)

	// Connect and log in to ESX or vCenter
	c, err := govmomi.NewClient(ctx, u, cfg.Global.InsecureFlag)
	if err != nil {
		return nil, err
	}

	return c, nil
}

func getVirtualMachineByName(cfg *VSphereConfig, ctx context.Context, c *govmomi.Client, name string) (*object.VirtualMachine, error) {
	// Create a new finder
	f := find.NewFinder(c.Client, true)

	// Fetch and set data center
	dc, err := f.Datacenter(ctx, cfg.Global.Datacenter)
	if err != nil {
		return nil, err
	}
	f.SetDatacenter(dc)

	// Retrieve vm by name
	//TODO: also look for vm inside subfolders
	vm, err := f.VirtualMachine(ctx, name)
	if err != nil {
		return nil, err
	}

	return vm, nil
}

func getVirtualMachineManagedObjectReference(ctx context.Context, c *govmomi.Client, vm *object.VirtualMachine, field string, dst interface{}) error {
	collector := property.DefaultCollector(c.Client)

	// Retrieve required field from VM object
	err := collector.RetrieveOne(ctx, vm.Reference(), []string{field}, dst)
	if err != nil {
		return err
	}
	return nil
}

func getInstances(cfg *VSphereConfig, ctx context.Context, c *govmomi.Client, filter string) ([]string, error) {
	f := find.NewFinder(c.Client, true)
	dc, err := f.Datacenter(ctx, cfg.Global.Datacenter)
	if err != nil {
		return nil, err
	}

	f.SetDatacenter(dc)

	//TODO: get all vms inside subfolders
	vms, err := f.VirtualMachineList(ctx, filter)
	if err != nil {
		return nil, err
	}

	var vmRef []types.ManagedObjectReference
	for _, vm := range vms {
		vmRef = append(vmRef, vm.Reference())
	}

	pc := property.DefaultCollector(c.Client)

	var vmt []mo.VirtualMachine
	err = pc.Retrieve(ctx, vmRef, []string{"name", "summary"}, &vmt)
	if err != nil {
		return nil, err
	}

	var vmList []string
	for _, vm := range vmt {
		if vm.Summary.Runtime.PowerState == ActivePowerState {
			vmList = append(vmList, vm.Name)
		} else if vm.Summary.Config.Template == false {
			glog.Warningf("VM %s, is not in %s state", vm.Name, ActivePowerState)
		}
	}
	return vmList, nil
}

type Instances struct {
	cfg             *VSphereConfig
	localInstanceID string
}

// Instances returns an implementation of Instances for vSphere.
func (vs *VSphere) Instances() (cloudprovider.Instances, bool) {
	return &Instances{vs.cfg, vs.localInstanceID}, true
}

// List is an implementation of Instances.List.
func (i *Instances) List(filter string) ([]string, error) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	c, err := vsphereLogin(i.cfg, ctx)
	if err != nil {
		return nil, err
	}
	defer c.Logout(ctx)

	vmList, err := getInstances(i.cfg, ctx, c, filter)
	if err != nil {
		return nil, err
	}

	glog.V(3).Infof("found %s instances matching %s: %s",
		len(vmList), filter, vmList)

	return vmList, nil
}

// NodeAddresses is an implementation of Instances.NodeAddresses.
func (i *Instances) NodeAddresses(name string) ([]api.NodeAddress, error) {
	addrs := []api.NodeAddress{}

	// Create context
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	// Create vSphere client
	c, err := vsphereLogin(i.cfg, ctx)
	if err != nil {
		return nil, err
	}
	defer c.Logout(ctx)

	vm, err := getVirtualMachineByName(i.cfg, ctx, c, name)
	if err != nil {
		return nil, err
	}

	var mvm mo.VirtualMachine
	err = getVirtualMachineManagedObjectReference(ctx, c, vm, "guest.net", &mvm)
	if err != nil {
		return nil, err
	}

	// retrieve VM's ip(s)
	for _, v := range mvm.Guest.Net {
		var addressType api.NodeAddressType
		if i.cfg.Network.PublicNetwork == v.Network {
			addressType = api.NodeExternalIP
		} else {
			addressType = api.NodeInternalIP
		}
		for _, ip := range v.IpAddress {
			api.AddToNodeAddresses(&addrs,
				api.NodeAddress{
					Type:    addressType,
					Address: ip,
				},
			)
		}
	}
	return addrs, nil
}

func (i *Instances) AddSSHKeyToAllInstances(user string, keyData []byte) error {
	return errors.New("unimplemented")
}

func (i *Instances) CurrentNodeName(hostname string) (string, error) {
	return i.localInstanceID, nil
}

// ExternalID returns the cloud provider ID of the specified instance (deprecated).
func (i *Instances) ExternalID(name string) (string, error) {
	// Create context
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	// Create vSphere client
	c, err := vsphereLogin(i.cfg, ctx)
	if err != nil {
		return "", err
	}
	defer c.Logout(ctx)

	vm, err := getVirtualMachineByName(i.cfg, ctx, c, name)
	if err != nil {
		return "", err
	}

	var mvm mo.VirtualMachine
	err = getVirtualMachineManagedObjectReference(ctx, c, vm, "summary", &mvm)
	if err != nil {
		return "", err
	}

	if mvm.Summary.Runtime.PowerState == ActivePowerState {
		return vm.InventoryPath, nil
	}

	if mvm.Summary.Config.Template == false {
		glog.Warningf("VM %s, is not in %s state", name, ActivePowerState)
	} else {
		glog.Warningf("VM %s, is a template", name)
	}

	return "", cloudprovider.InstanceNotFound
}

// InstanceID returns the cloud provider ID of the specified instance.
func (i *Instances) InstanceID(name string) (string, error) {
	// Create context
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	// Create vSphere client
	c, err := vsphereLogin(i.cfg, ctx)
	if err != nil {
		return "", err
	}
	defer c.Logout(ctx)

	vm, err := getVirtualMachineByName(i.cfg, ctx, c, name)

	var mvm mo.VirtualMachine
	err = getVirtualMachineManagedObjectReference(ctx, c, vm, "summary", &mvm)
	if err != nil {
		return "", err
	}

	if mvm.Summary.Runtime.PowerState == ActivePowerState {
		return "/" + vm.InventoryPath, nil
	}

	if mvm.Summary.Config.Template == false {
		glog.Warning("VM %s, is not in %s state", name, ActivePowerState)
	} else {
		glog.Warning("VM %s, is a template", name)
	}

	return "", cloudprovider.InstanceNotFound
}

func (i *Instances) InstanceType(name string) (string, error) {
	return "", nil
}

func (vs *VSphere) Clusters() (cloudprovider.Clusters, bool) {
	return nil, true
}

// ProviderName returns the cloud provider ID.
func (vs *VSphere) ProviderName() string {
	return ProviderName
}

// LoadBalancer returns an implementation of LoadBalancer for vSphere.
func (vs *VSphere) LoadBalancer() (cloudprovider.LoadBalancer, bool) {
	return nil, false
}

// Zones returns an implementation of Zones for Google vSphere.
func (vs *VSphere) Zones() (cloudprovider.Zones, bool) {
	glog.V(4).Info("Claiming to support Zones")

	return vs, true
}

func (vs *VSphere) GetZone() (cloudprovider.Zone, error) {
	glog.V(4).Infof("Current zone is %v", vs.cfg.Global.Datacenter)

	return cloudprovider.Zone{Region: vs.cfg.Global.Datacenter}, nil
}

// Routes returns a false since the interface is not supported for vSphere.
func (vs *VSphere) Routes() (cloudprovider.Routes, bool) {
	return nil, false
}

// ScrubDNS filters DNS settings for pods.
func (vs *VSphere) ScrubDNS(nameservers, searches []string) (nsOut, srchOut []string) {
	return nameservers, searches
}

func getVirtualMachineDevices(cfg *VSphereConfig, ctx context.Context, c *govmomi.Client, name string) (*object.VirtualMachine, object.VirtualDeviceList, *object.Datastore, error) {

	// Create a new finder
	f := find.NewFinder(c.Client, true)

	// Fetch and set data center
	dc, err := f.Datacenter(ctx, cfg.Global.Datacenter)
	if err != nil {
		return nil, nil, nil, err
	}
	f.SetDatacenter(dc)

	// Find datastores
	ds, err := f.Datastore(ctx, cfg.Global.Datastore)
	if err != nil {
		return nil, nil, nil, err
	}

	vm, err := f.VirtualMachine(ctx, name)
	if err != nil {
		return nil, nil, nil, err
	}

	// Get devices from VM
	vmDevices, err := vm.Device(ctx)
	if err != nil {
		return nil, nil, nil, err
	}
	return vm, vmDevices, ds, nil
}

//cleaning up the controller
func cleanUpController(newSCSIController types.BaseVirtualDevice, vmDevices object.VirtualDeviceList, vm *object.VirtualMachine, ctx context.Context) error {
	ctls := vmDevices.SelectByType(newSCSIController)
	if len(ctls) < 1 {
		return ErrNoDevicesFound
	}
	newScsi := ctls[len(ctls)-1]
	err := vm.RemoveDevice(ctx, true, newScsi)
	if err != nil {
		return err
	}
	return nil
}

// Attaches given virtual disk volume to the compute running kubelet.
func (vs *VSphere) AttachDisk(vmDiskPath string, nodeName string) (diskID string, diskUUID string, err error) {
	// Create context
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	// Create vSphere client
	c, err := vsphereLogin(vs.cfg, ctx)
	if err != nil {
		return "", "", err
	}
	defer c.Logout(ctx)

	// Find virtual machine to attach disk to
	var vSphereInstance string
	if nodeName == "" {
		vSphereInstance = vs.localInstanceID
	} else {
		vSphereInstance = nodeName
	}

	// Get VM device list
	vm, vmDevices, ds, err := getVirtualMachineDevices(vs.cfg, ctx, c, vSphereInstance)
	if err != nil {
		return "", "", err
	}

	// find SCSI controller to attach the disk
	var newSCSICreated bool = false
	var newSCSIController types.BaseVirtualDevice
	diskController, err := vmDevices.FindDiskController(vs.cfg.Disk.DiskController)
	if err != nil {
		// create a scsi controller if there is not one
		newSCSIController, err := vmDevices.CreateSCSIController(vs.cfg.Disk.SCSIControllerType)
		if err != nil {
			glog.V(3).Infof("cannot create new SCSI controller - %v", err)
			return "", "", err
		}
		configNewSCSIController := newSCSIController.(types.BaseVirtualSCSIController).GetVirtualSCSIController()
		hotAndRemove := true
		configNewSCSIController.HotAddRemove = &hotAndRemove
		configNewSCSIController.SharedBus = types.VirtualSCSISharing(types.VirtualSCSISharingNoSharing)

		// add the scsi controller to virtual machine
		err = vm.AddDevice(context.TODO(), newSCSIController)
		if err != nil {
			glog.V(3).Infof("cannot add SCSI controller to vm - %v", err)
			// attempt clean up of scsi controller
			if vmDevices, err := vm.Device(ctx); err == nil {
				cleanUpController(newSCSIController, vmDevices, vm, ctx)
			}
			return "", "", err
		}

		// verify scsi controller in virtual machine
		vmDevices, err = vm.Device(ctx)
		if err != nil {
			//cannot cleanup if there is no device list
			return "", "", err
		}
		if diskController, err = vmDevices.FindDiskController(vs.cfg.Disk.DiskController); err != nil {
			glog.V(3).Infof("cannot find disk controller - %v", err)
			// attempt clean up of scsi controller
			cleanUpController(newSCSIController, vmDevices, vm, ctx)
			return "", "", err
		}
		newSCSICreated = true
	}

	disk := vmDevices.CreateDisk(diskController, ds.Reference(), vmDiskPath)
	backing := disk.Backing.(*types.VirtualDiskFlatVer2BackingInfo)
	backing.DiskMode = string(types.VirtualDiskModeIndependent_persistent)

	// Attach disk to the VM
	err = vm.AddDevice(ctx, disk)
	if err != nil {
		glog.V(3).Infof("cannot attach disk to the vm - %v", err)
		if newSCSICreated {
			cleanUpController(newSCSIController, vmDevices, vm, ctx)
		}
		return "", "", err
	}

	vmDevices, err = vm.Device(ctx)
	if err != nil {
		if newSCSICreated {
			cleanUpController(newSCSIController, vmDevices, vm, ctx)
		}
		return "", "", err
	}
	devices := vmDevices.SelectByType(disk)
	if len(devices) < 1 {
		if newSCSICreated {
			cleanUpController(newSCSIController, vmDevices, vm, ctx)
		}
		return "", "", ErrNoDevicesFound
	}

	// get new disk id
	newDevice := devices[len(devices)-1]
	deviceName := devices.Name(newDevice)

	// get device uuid
	diskUUID, err = getVirtualDiskUUID(newDevice)
	if err != nil {
		if newSCSICreated {
			cleanUpController(newSCSIController, vmDevices, vm, ctx)
		}
		vs.DetachDisk(deviceName, vSphereInstance)
		return "", "", err
	}

	return deviceName, diskUUID, nil
}

func getVirtualDiskUUID(newDevice types.BaseVirtualDevice) (string, error) {
	vd := newDevice.GetVirtualDevice()

	if b, ok := vd.Backing.(*types.VirtualDiskFlatVer2BackingInfo); ok {
		uuidWithNoHypens := strings.Replace(b.Uuid, "-", "", -1)
		return strings.ToLower(uuidWithNoHypens), nil
	}
	return "", ErrNoDiskUUIDFound
}

func getVirtualDiskID(volPath string, vmDevices object.VirtualDeviceList) (string, error) {
	// filter vm devices to retrieve disk ID for the given vmdk file
	for _, device := range vmDevices {
		if vmDevices.TypeName(device) == "VirtualDisk" {
			d := device.GetVirtualDevice()
			if b, ok := d.Backing.(types.BaseVirtualDeviceFileBackingInfo); ok {
				fileName := b.GetVirtualDeviceFileBackingInfo().FileName
				if fileName == volPath {
					return vmDevices.Name(device), nil
				}
			}
		}
	}
	return "", ErrNoDiskIDFound
}

// Detaches given virtual disk volume from the compute running kubelet.
func (vs *VSphere) DetachDisk(volPath string, nodeName string) error {
	// Create context
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	// Create vSphere client
	c, err := vsphereLogin(vs.cfg, ctx)
	if err != nil {
		return err
	}
	defer c.Logout(ctx)

	// Find VM to detach disk from
	var vSphereInstance string
	if nodeName == "" {
		vSphereInstance = vs.localInstanceID
	} else {
		vSphereInstance = nodeName
	}

	vm, vmDevices, _, err := getVirtualMachineDevices(vs.cfg, ctx, c, vSphereInstance)
	if err != nil {
		return err
	}

	diskID, err := getVirtualDiskID(volPath, vmDevices)
	if err != nil {
		glog.Warningf("disk ID not found for %v ", volPath)
		return err
	}

	// Remove disk from VM
	device := vmDevices.Find(diskID)
	if device == nil {
		return fmt.Errorf("device '%s' not found", diskID)
	}

	err = vm.RemoveDevice(ctx, true, device)
	if err != nil {
		return err
	}

	return nil
}

// Create a volume of given size (in KiB).
func (vs *VSphere) CreateVolume(name string, size int, tags *map[string]string) (volumePath string, err error) {
	// Create context
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	// Create vSphere client
	c, err := vsphereLogin(vs.cfg, ctx)
	if err != nil {
		return "", err
	}
	defer c.Logout(ctx)

	// Create a new finder
	f := find.NewFinder(c.Client, true)

	// Fetch and set data center
	dc, err := f.Datacenter(ctx, vs.cfg.Global.Datacenter)
	f.SetDatacenter(dc)

	// Create a virtual disk manager
	vmDiskPath := "[" + vs.cfg.Global.Datastore + "] " + name + ".vmdk"
	virtualDiskManager := object.NewVirtualDiskManager(c.Client)

	// Create specification for new virtual disk
	vmDiskSpec := &types.FileBackedVirtualDiskSpec{
		VirtualDiskSpec: types.VirtualDiskSpec{
			AdapterType: (*tags)["adapterType"],
			DiskType:    (*tags)["diskType"],
		},
		CapacityKb: int64(size),
	}

	// Create virtual disk
	task, err := virtualDiskManager.CreateVirtualDisk(ctx, vmDiskPath, dc, vmDiskSpec)
	if err != nil {
		return "", err
	}
	err = task.Wait(ctx)
	if err != nil {
		return "", err
	}

	return vmDiskPath, nil
}

// Deletes a volume given volume name.
func (vs *VSphere) DeleteVolume(vmDiskPath string) error {
	// Create context
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	// Create vSphere client
	c, err := vsphereLogin(vs.cfg, ctx)
	if err != nil {
		return err
	}
	defer c.Logout(ctx)

	// Create a new finder
	f := find.NewFinder(c.Client, true)

	// Fetch and set data center
	dc, err := f.Datacenter(ctx, vs.cfg.Global.Datacenter)
	f.SetDatacenter(dc)

	// Create a virtual disk manager
	virtualDiskManager := object.NewVirtualDiskManager(c.Client)

	// Delete virtual disk
	task, err := virtualDiskManager.DeleteVirtualDisk(ctx, vmDiskPath, dc)
	if err != nil {
		return err
	}

	return task.Wait(ctx)
}
