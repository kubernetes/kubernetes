/*
Copyright 2016 The Kubernetes Authors.

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
	"errors"
	"fmt"
	"io"
	"net"
	"net/url"
	"path"
	"path/filepath"
	"runtime"
	"strings"
	"sync"

	"gopkg.in/gcfg.v1"

	"github.com/golang/glog"
	"github.com/vmware/govmomi"
	"github.com/vmware/govmomi/find"
	"github.com/vmware/govmomi/object"
	"github.com/vmware/govmomi/property"
	"github.com/vmware/govmomi/session"
	"github.com/vmware/govmomi/vim25"
	"github.com/vmware/govmomi/vim25/mo"
	"github.com/vmware/govmomi/vim25/soap"
	"github.com/vmware/govmomi/vim25/types"
	"golang.org/x/net/context"

	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/cloudprovider"
	k8stypes "k8s.io/kubernetes/pkg/types"
	k8runtime "k8s.io/kubernetes/pkg/util/runtime"
)

const (
	ProviderName              = "vsphere"
	ActivePowerState          = "poweredOn"
	SCSIControllerType        = "scsi"
	LSILogicControllerType    = "lsiLogic"
	BusLogicControllerType    = "busLogic"
	PVSCSIControllerType      = "pvscsi"
	LSILogicSASControllerType = "lsiLogic-sas"
	SCSIControllerLimit       = 4
	SCSIControllerDeviceLimit = 15
	SCSIDeviceSlots           = 16
	SCSIReservedSlot          = 7
	ThinDiskType              = "thin"
	PreallocatedDiskType      = "preallocated"
	EagerZeroedThickDiskType  = "eagerZeroedThick"
	ZeroedThickDiskType       = "zeroedThick"
	VolDir                    = "kubevols"
	RoundTripperDefaultCount  = 3
)

// Controller types that are currently supported for hot attach of disks
// lsilogic driver type is currently not supported because,when a device gets detached
// it fails to remove the device from the /dev path (which should be manually done)
// making the subsequent attaches to the node to fail.
// TODO: Add support for lsilogic driver type
var supportedSCSIControllerType = []string{strings.ToLower(LSILogicSASControllerType), PVSCSIControllerType}

// Maps user options to API parameters.
// Keeping user options consistent with docker volume plugin for vSphere.
// API: http://pubs.vmware.com/vsphere-60/index.jsp#com.vmware.wssdk.apiref.doc/vim.VirtualDiskManager.VirtualDiskType.html
var diskFormatValidType = map[string]string{
	ThinDiskType:                              ThinDiskType,
	strings.ToLower(EagerZeroedThickDiskType): EagerZeroedThickDiskType,
	strings.ToLower(ZeroedThickDiskType):      PreallocatedDiskType,
}

var DiskformatValidOptions = generateDiskFormatValidOptions()

var ErrNoDiskUUIDFound = errors.New("No disk UUID found")
var ErrNoDiskIDFound = errors.New("No vSphere disk ID found")
var ErrNoDevicesFound = errors.New("No devices found")
var ErrNonSupportedControllerType = errors.New("Disk is attached to non-supported controller type")
var ErrFileAlreadyExist = errors.New("File requested already exist")

var clientLock sync.Mutex

// VSphere is an implementation of cloud provider Interface for VSphere.
type VSphere struct {
	client *govmomi.Client
	cfg    *VSphereConfig
	// InstanceID of the server where this VSphere object is instantiated.
	localInstanceID string
	// Cluster that VirtualMachine belongs to
	clusterName string
}

type VSphereConfig struct {
	Global struct {
		// vCenter username.
		User string `gcfg:"user"`
		// vCenter password in clear text.
		Password string `gcfg:"password"`
		// vCenter IP.
		VCenterIP string `gcfg:"server"`
		// vCenter port.
		VCenterPort string `gcfg:"port"`
		// True if vCenter uses self-signed cert.
		InsecureFlag bool `gcfg:"insecure-flag"`
		// Datacenter in which VMs are located.
		Datacenter string `gcfg:"datacenter"`
		// Datastore in which vmdks are stored.
		Datastore string `gcfg:"datastore"`
		// WorkingDir is path where VMs can be found.
		WorkingDir string `gcfg:"working-dir"`
		// Soap round tripper count (retries = RoundTripper - 1)
		RoundTripperCount uint `gcfg:"soap-roundtrip-count"`
	}

	Network struct {
		// PublicNetwork is name of the network the VMs are joined to.
		PublicNetwork string `gcfg:"public-network"`
	}
	Disk struct {
		// SCSIControllerType defines SCSI controller to be used.
		SCSIControllerType string `dcfg:"scsicontrollertype"`
	}
}

type Volumes interface {
	// AttachDisk attaches given disk to given node. Current node
	// is used when nodeName is empty string.
	AttachDisk(vmDiskPath string, nodeName k8stypes.NodeName) (diskID string, diskUUID string, err error)

	// DetachDisk detaches given disk to given node. Current node
	// is used when nodeName is empty string.
	// Assumption: If node doesn't exist, disk is already detached from node.
	DetachDisk(volPath string, nodeName k8stypes.NodeName) error

	// DiskIsAttached checks if a disk is attached to the given node.
	// Assumption: If node doesn't exist, disk is not attached to the node.
	DiskIsAttached(volPath string, nodeName k8stypes.NodeName) (bool, error)

	// DisksAreAttached checks if a list disks are attached to the given node.
	// Assumption: If node doesn't exist, disks are not attached to the node.
	DisksAreAttached(volPath []string, nodeName k8stypes.NodeName) (map[string]bool, error)

	// CreateVolume creates a new vmdk with specified parameters.
	CreateVolume(volumeOptions *VolumeOptions) (volumePath string, err error)

	// DeleteVolume deletes vmdk.
	DeleteVolume(vmDiskPath string) error
}

// VolumeOptions specifies capacity, tags, name and diskFormat for a volume.
type VolumeOptions struct {
	CapacityKB int
	Tags       map[string]string
	Name       string
	DiskFormat string
}

// Generates Valid Options for Diskformat
func generateDiskFormatValidOptions() string {
	validopts := ""
	for diskformat := range diskFormatValidType {
		validopts += (diskformat + ", ")
	}
	validopts = strings.TrimSuffix(validopts, ", ")
	return validopts
}

// Parses vSphere cloud config file and stores it into VSphereConfig.
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

// Returns the name of the VM and its Cluster on which this code is running.
// This is done by searching for the name of virtual machine by current IP.
// Prerequisite: this code assumes VMWare vmtools or open-vm-tools to be installed in the VM.
func readInstance(client *govmomi.Client, cfg *VSphereConfig) (string, string, error) {
	addrs, err := net.InterfaceAddrs()
	if err != nil {
		return "", "", err
	}

	if len(addrs) == 0 {
		return "", "", fmt.Errorf("unable to retrieve Instance ID")
	}

	// Create context
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	// Create a new finder
	f := find.NewFinder(client.Client, true)

	// Fetch and set data center
	dc, err := f.Datacenter(ctx, cfg.Global.Datacenter)
	if err != nil {
		return "", "", err
	}
	f.SetDatacenter(dc)

	s := object.NewSearchIndex(client.Client)

	var svm object.Reference
	for _, v := range addrs {
		ip, _, err := net.ParseCIDR(v.String())
		if err != nil {
			return "", "", fmt.Errorf("unable to parse cidr from ip")
		}

		// Finds a virtual machine or host by IP address.
		svm, err = s.FindByIp(ctx, dc, ip.String(), true)
		if err == nil && svm != nil {
			break
		}
	}
	if svm == nil {
		return "", "", fmt.Errorf("unable to retrieve vm reference from vSphere")
	}

	var vm mo.VirtualMachine
	err = s.Properties(ctx, svm.Reference(), []string{"name", "resourcePool"}, &vm)
	if err != nil {
		return "", "", err
	}

	var cluster string
	if vm.ResourcePool != nil {
		// Extract the Cluster Name if VM belongs to a ResourcePool
		var rp mo.ResourcePool
		err = s.Properties(ctx, *vm.ResourcePool, []string{"parent"}, &rp)
		if err == nil {
			var ccr mo.ComputeResource
			err = s.Properties(ctx, *rp.Parent, []string{"name"}, &ccr)
			if err == nil {
				cluster = ccr.Name
			} else {
				glog.Warningf("VM %s, does not belong to a vSphere Cluster, will not have FailureDomain label", vm.Name)
			}
		} else {
			glog.Warningf("VM %s, does not belong to a vSphere Cluster, will not have FailureDomain label", vm.Name)
		}
	}
	return vm.Name, cluster, nil
}

func newVSphere(cfg VSphereConfig) (*VSphere, error) {
	if cfg.Disk.SCSIControllerType == "" {
		cfg.Disk.SCSIControllerType = LSILogicSASControllerType
	} else if !checkControllerSupported(cfg.Disk.SCSIControllerType) {
		glog.Errorf("%v is not a supported SCSI Controller type. Please configure 'lsilogic-sas' OR 'pvscsi'", cfg.Disk.SCSIControllerType)
		return nil, errors.New("Controller type not supported. Please configure 'lsilogic-sas' OR 'pvscsi'")
	}
	if cfg.Global.WorkingDir != "" {
		cfg.Global.WorkingDir = path.Clean(cfg.Global.WorkingDir) + "/"
	}
	if cfg.Global.RoundTripperCount == 0 {
		cfg.Global.RoundTripperCount = RoundTripperDefaultCount
	}

	c, err := newClient(context.TODO(), &cfg)
	if err != nil {
		return nil, err
	}

	id, cluster, err := readInstance(c, &cfg)
	if err != nil {
		return nil, err
	}

	vs := VSphere{
		client:          c,
		cfg:             &cfg,
		localInstanceID: id,
		clusterName:     cluster,
	}
	runtime.SetFinalizer(&vs, logout)

	return &vs, nil
}

// Returns if the given controller type is supported by the plugin
func checkControllerSupported(ctrlType string) bool {
	for _, c := range supportedSCSIControllerType {
		if ctrlType == c {
			return true
		}
	}
	return false
}

func logout(vs *VSphere) {
	vs.client.Logout(context.TODO())
}

func newClient(ctx context.Context, cfg *VSphereConfig) (*govmomi.Client, error) {
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

	// Add retry functionality
	c.RoundTripper = vim25.Retry(c.RoundTripper, vim25.TemporaryNetworkError(int(cfg.Global.RoundTripperCount)))

	return c, nil
}

// Returns a client which communicates with vCenter.
// This client can used to perform further vCenter operations.
func vSphereLogin(ctx context.Context, vs *VSphere) error {
	var err error
	clientLock.Lock()
	defer clientLock.Unlock()
	if vs.client == nil {
		vs.client, err = newClient(ctx, vs.cfg)
		if err != nil {
			return err
		}
		return nil
	}

	m := session.NewManager(vs.client.Client)
	// retrieve client's current session
	u, err := m.UserSession(ctx)
	if err == nil && u == nil {
		// current session is valid
		return nil
	}

	glog.Warningf("Creating new client session since the existing session is not valid or not authenticated")
	vs.client.Logout(ctx)
	vs.client, err = newClient(ctx, vs.cfg)
	if err != nil {
		return err
	}

	return nil
}

// Returns vSphere object `virtual machine` by its name.
func getVirtualMachineByName(ctx context.Context, cfg *VSphereConfig, c *govmomi.Client, nodeName k8stypes.NodeName) (*object.VirtualMachine, error) {
	name := nodeNameToVMName(nodeName)

	// Create a new finder
	f := find.NewFinder(c.Client, true)

	// Fetch and set data center
	dc, err := f.Datacenter(ctx, cfg.Global.Datacenter)
	if err != nil {
		return nil, err
	}
	f.SetDatacenter(dc)

	vmRegex := cfg.Global.WorkingDir + name

	// Retrieve vm by name
	//TODO: also look for vm inside subfolders
	vm, err := f.VirtualMachine(ctx, vmRegex)
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

// Returns names of running VMs inside VM folder.
func getInstances(ctx context.Context, cfg *VSphereConfig, c *govmomi.Client, filter string) ([]string, error) {
	f := find.NewFinder(c.Client, true)
	dc, err := f.Datacenter(ctx, cfg.Global.Datacenter)
	if err != nil {
		return nil, err
	}

	f.SetDatacenter(dc)

	vmRegex := cfg.Global.WorkingDir + filter

	//TODO: get all vms inside subfolders
	vms, err := f.VirtualMachineList(ctx, vmRegex)
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
	client          *govmomi.Client
	cfg             *VSphereConfig
	localInstanceID string
}

// Instances returns an implementation of Instances for vSphere.
func (vs *VSphere) Instances() (cloudprovider.Instances, bool) {
	// Ensure client is logged in and session is valid
	err := vSphereLogin(context.TODO(), vs)
	if err != nil {
		glog.Errorf("Failed to login into vCenter - %v", err)
		return nil, false
	}
	return &Instances{vs.client, vs.cfg, vs.localInstanceID}, true
}

// List returns names of VMs (inside vm folder) by applying filter and which are currently running.
func (i *Instances) List(filter string) ([]k8stypes.NodeName, error) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	vmList, err := getInstances(ctx, i.cfg, i.client, filter)
	if err != nil {
		return nil, err
	}

	glog.V(3).Infof("Found %d instances matching %s: %s",
		len(vmList), filter, vmList)

	var nodeNames []k8stypes.NodeName
	for _, n := range vmList {
		nodeNames = append(nodeNames, k8stypes.NodeName(n))
	}
	return nodeNames, nil
}

// NodeAddresses is an implementation of Instances.NodeAddresses.
func (i *Instances) NodeAddresses(nodeName k8stypes.NodeName) ([]v1.NodeAddress, error) {
	addrs := []v1.NodeAddress{}

	// Create context
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	vm, err := getVirtualMachineByName(ctx, i.cfg, i.client, nodeName)
	if err != nil {
		return nil, err
	}

	var mvm mo.VirtualMachine
	err = getVirtualMachineManagedObjectReference(ctx, i.client, vm, "guest.net", &mvm)
	if err != nil {
		return nil, err
	}

	// retrieve VM's ip(s)
	for _, v := range mvm.Guest.Net {
		var addressType v1.NodeAddressType
		if i.cfg.Network.PublicNetwork == v.Network {
			addressType = v1.NodeExternalIP
		} else {
			addressType = v1.NodeInternalIP
		}
		for _, ip := range v.IpAddress {
			v1.AddToNodeAddresses(&addrs,
				v1.NodeAddress{
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

func (i *Instances) CurrentNodeName(hostname string) (k8stypes.NodeName, error) {
	return k8stypes.NodeName(i.localInstanceID), nil
}

// nodeNameToVMName maps a NodeName to the vmware infrastructure name
func nodeNameToVMName(nodeName k8stypes.NodeName) string {
	return string(nodeName)
}

// nodeNameToVMName maps a vmware infrastructure name to a NodeName
func vmNameToNodeName(vmName string) k8stypes.NodeName {
	return k8stypes.NodeName(vmName)
}

// ExternalID returns the cloud provider ID of the node with the specified Name (deprecated).
func (i *Instances) ExternalID(nodeName k8stypes.NodeName) (string, error) {
	// Create context
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	vm, err := getVirtualMachineByName(ctx, i.cfg, i.client, nodeName)
	if err != nil {
		if _, ok := err.(*find.NotFoundError); ok {
			return "", cloudprovider.InstanceNotFound
		}
		return "", err
	}

	var mvm mo.VirtualMachine
	err = getVirtualMachineManagedObjectReference(ctx, i.client, vm, "summary", &mvm)
	if err != nil {
		return "", err
	}

	if mvm.Summary.Runtime.PowerState == ActivePowerState {
		return vm.InventoryPath, nil
	}

	if mvm.Summary.Config.Template == false {
		glog.Warningf("VM %s, is not in %s state", nodeName, ActivePowerState)
	} else {
		glog.Warningf("VM %s, is a template", nodeName)
	}

	return "", cloudprovider.InstanceNotFound
}

// InstanceID returns the cloud provider ID of the node with the specified Name.
func (i *Instances) InstanceID(nodeName k8stypes.NodeName) (string, error) {
	// Create context
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	vm, err := getVirtualMachineByName(ctx, i.cfg, i.client, nodeName)
	if err != nil {
		if _, ok := err.(*find.NotFoundError); ok {
			return "", cloudprovider.InstanceNotFound
		}
		return "", err
	}

	var mvm mo.VirtualMachine
	err = getVirtualMachineManagedObjectReference(ctx, i.client, vm, "summary", &mvm)
	if err != nil {
		return "", err
	}

	if mvm.Summary.Runtime.PowerState == ActivePowerState {
		return "/" + vm.InventoryPath, nil
	}

	if mvm.Summary.Config.Template == false {
		glog.Warningf("VM %s, is not in %s state", nodeName, ActivePowerState)
	} else {
		glog.Warningf("VM %s, is a template", nodeName)
	}

	return "", cloudprovider.InstanceNotFound
}

func (i *Instances) InstanceType(name k8stypes.NodeName) (string, error) {
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
	glog.V(1).Info("Claiming to support Zones")

	return vs, true
}

func (vs *VSphere) GetZone() (cloudprovider.Zone, error) {
	glog.V(1).Infof("Current datacenter is %v, cluster is %v", vs.cfg.Global.Datacenter, vs.clusterName)

	// The clusterName is determined from the VirtualMachine ManagedObjectReference during init
	// If the VM is not created within a Cluster, this will return empty-string
	return cloudprovider.Zone{
		Region:        vs.cfg.Global.Datacenter,
		FailureDomain: vs.clusterName,
	}, nil
}

// Routes returns a false since the interface is not supported for vSphere.
func (vs *VSphere) Routes() (cloudprovider.Routes, bool) {
	return nil, false
}

// ScrubDNS filters DNS settings for pods.
func (vs *VSphere) ScrubDNS(nameservers, searches []string) (nsOut, srchOut []string) {
	return nameservers, searches
}

// Returns vSphere objects virtual machine, virtual device list, datastore and datacenter.
func getVirtualMachineDevices(ctx context.Context, cfg *VSphereConfig, c *govmomi.Client, name string) (*object.VirtualMachine, object.VirtualDeviceList, *object.Datastore, *object.Datacenter, error) {
	// Create a new finder
	f := find.NewFinder(c.Client, true)

	// Fetch and set data center
	dc, err := f.Datacenter(ctx, cfg.Global.Datacenter)
	if err != nil {
		return nil, nil, nil, nil, err
	}
	f.SetDatacenter(dc)

	// Find datastores
	ds, err := f.Datastore(ctx, cfg.Global.Datastore)
	if err != nil {
		return nil, nil, nil, nil, err
	}

	vmRegex := cfg.Global.WorkingDir + name

	vm, err := f.VirtualMachine(ctx, vmRegex)
	if err != nil {
		return nil, nil, nil, nil, err
	}

	// Get devices from VM
	vmDevices, err := vm.Device(ctx)
	if err != nil {
		return nil, nil, nil, nil, err
	}
	return vm, vmDevices, ds, dc, nil
}

// Removes SCSI controller which is latest attached to VM.
func cleanUpController(ctx context.Context, newSCSIController types.BaseVirtualDevice, vmDevices object.VirtualDeviceList, vm *object.VirtualMachine) error {
	if newSCSIController == nil || vmDevices == nil || vm == nil {
		return nil
	}
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
func (vs *VSphere) AttachDisk(vmDiskPath string, nodeName k8stypes.NodeName) (diskID string, diskUUID string, err error) {
	// Create context
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	// Ensure client is logged in and session is valid
	err = vSphereLogin(ctx, vs)
	if err != nil {
		glog.Errorf("Failed to login into vCenter - %v", err)
		return "", "", err
	}

	// Find virtual machine to attach disk to
	var vSphereInstance string
	if nodeName == "" {
		vSphereInstance = vs.localInstanceID
		nodeName = vmNameToNodeName(vSphereInstance)
	} else {
		vSphereInstance = nodeNameToVMName(nodeName)
	}

	// Get VM device list
	vm, vmDevices, ds, dc, err := getVirtualMachineDevices(ctx, vs.cfg, vs.client, vSphereInstance)
	if err != nil {
		return "", "", err
	}

	attached, _ := checkDiskAttached(vmDiskPath, vmDevices, dc, vs.client)
	if attached {
		diskID, _ = getVirtualDiskID(vmDiskPath, vmDevices, dc, vs.client)
		diskUUID, _ = getVirtualDiskUUIDByPath(vmDiskPath, dc, vs.client)
		return diskID, diskUUID, nil
	}

	var diskControllerType = vs.cfg.Disk.SCSIControllerType
	// find SCSI controller of particular type from VM devices
	allSCSIControllers := getSCSIControllers(vmDevices)
	scsiControllersOfRequiredType := getSCSIControllersOfType(vmDevices, diskControllerType)
	scsiController := getAvailableSCSIController(scsiControllersOfRequiredType)

	var newSCSICreated = false
	var newSCSIController types.BaseVirtualDevice

	// creating a scsi controller as there is none found of controller type defined
	if scsiController == nil {
		if len(allSCSIControllers) >= SCSIControllerLimit {
			// we reached the maximum number of controllers we can attach
			return "", "", fmt.Errorf("SCSI Controller Limit of %d has been reached, cannot create another SCSI controller", SCSIControllerLimit)
		}
		glog.V(1).Infof("Creating a SCSI controller of %v type", diskControllerType)
		newSCSIController, err := vmDevices.CreateSCSIController(diskControllerType)
		if err != nil {
			k8runtime.HandleError(fmt.Errorf("error creating new SCSI controller: %v", err))
			return "", "", err
		}
		configNewSCSIController := newSCSIController.(types.BaseVirtualSCSIController).GetVirtualSCSIController()
		hotAndRemove := true
		configNewSCSIController.HotAddRemove = &hotAndRemove
		configNewSCSIController.SharedBus = types.VirtualSCSISharing(types.VirtualSCSISharingNoSharing)

		// add the scsi controller to virtual machine
		err = vm.AddDevice(context.TODO(), newSCSIController)
		if err != nil {
			glog.V(1).Infof("cannot add SCSI controller to vm - %v", err)
			// attempt clean up of scsi controller
			if vmDevices, err := vm.Device(ctx); err == nil {
				cleanUpController(ctx, newSCSIController, vmDevices, vm)
			}
			return "", "", err
		}

		// verify scsi controller in virtual machine
		vmDevices, err = vm.Device(ctx)
		if err != nil {
			// cannot cleanup if there is no device list
			return "", "", err
		}

		scsiController = getSCSIController(vmDevices, vs.cfg.Disk.SCSIControllerType)
		if scsiController == nil {
			glog.Errorf("cannot find SCSI controller in VM")
			// attempt clean up of scsi controller
			cleanUpController(ctx, newSCSIController, vmDevices, vm)
			return "", "", fmt.Errorf("cannot find SCSI controller in VM")
		}
		newSCSICreated = true
	}

	disk := vmDevices.CreateDisk(scsiController, ds.Reference(), vmDiskPath)
	unitNumber, err := getNextUnitNumber(vmDevices, scsiController)
	if err != nil {
		glog.Errorf("cannot attach disk to VM, limit reached - %v.", err)
		return "", "", err
	}
	*disk.UnitNumber = unitNumber

	backing := disk.Backing.(*types.VirtualDiskFlatVer2BackingInfo)
	backing.DiskMode = string(types.VirtualDiskModeIndependent_persistent)

	// Attach disk to the VM
	err = vm.AddDevice(ctx, disk)
	if err != nil {
		glog.Errorf("cannot attach disk to the vm - %v", err)
		if newSCSICreated {
			cleanUpController(ctx, newSCSIController, vmDevices, vm)
		}
		return "", "", err
	}

	vmDevices, err = vm.Device(ctx)
	if err != nil {
		if newSCSICreated {
			cleanUpController(ctx, newSCSIController, vmDevices, vm)
		}
		return "", "", err
	}
	devices := vmDevices.SelectByType(disk)
	if len(devices) < 1 {
		if newSCSICreated {
			cleanUpController(ctx, newSCSIController, vmDevices, vm)
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
			cleanUpController(ctx, newSCSIController, vmDevices, vm)
		}
		vs.DetachDisk(deviceName, nodeName)
		return "", "", err
	}

	return deviceName, diskUUID, nil
}

func getNextUnitNumber(devices object.VirtualDeviceList, c types.BaseVirtualController) (int32, error) {
	// get next available SCSI controller unit number
	var takenUnitNumbers [SCSIDeviceSlots]bool
	takenUnitNumbers[SCSIReservedSlot] = true
	key := c.GetVirtualController().Key

	for _, device := range devices {
		d := device.GetVirtualDevice()
		if d.ControllerKey == key {
			if d.UnitNumber != nil {
				takenUnitNumbers[*d.UnitNumber] = true
			}
		}
	}
	for unitNumber, takenUnitNumber := range takenUnitNumbers {
		if !takenUnitNumber {
			return int32(unitNumber), nil
		}
	}
	return -1, fmt.Errorf("SCSI Controller with key=%d does not have any available slots (LUN).", key)
}

func getSCSIController(vmDevices object.VirtualDeviceList, scsiType string) *types.VirtualController {
	// get virtual scsi controller of passed argument type
	for _, device := range vmDevices {
		devType := vmDevices.Type(device)
		if devType == scsiType {
			if c, ok := device.(types.BaseVirtualController); ok {
				return c.GetVirtualController()
			}
		}
	}
	return nil
}

func getSCSIControllersOfType(vmDevices object.VirtualDeviceList, scsiType string) []*types.VirtualController {
	// get virtual scsi controllers of passed argument type
	var scsiControllers []*types.VirtualController
	for _, device := range vmDevices {
		devType := vmDevices.Type(device)
		if devType == scsiType {
			if c, ok := device.(types.BaseVirtualController); ok {
				scsiControllers = append(scsiControllers, c.GetVirtualController())
			}
		}
	}
	return scsiControllers
}

func getSCSIControllers(vmDevices object.VirtualDeviceList) []*types.VirtualController {
	// get all virtual scsi controllers
	var scsiControllers []*types.VirtualController
	for _, device := range vmDevices {
		devType := vmDevices.Type(device)
		switch devType {
		case SCSIControllerType, strings.ToLower(LSILogicControllerType), strings.ToLower(BusLogicControllerType), PVSCSIControllerType, strings.ToLower(LSILogicSASControllerType):
			if c, ok := device.(types.BaseVirtualController); ok {
				scsiControllers = append(scsiControllers, c.GetVirtualController())
			}
		}
	}
	return scsiControllers
}

func getAvailableSCSIController(scsiControllers []*types.VirtualController) *types.VirtualController {
	// get SCSI controller which has space for adding more devices
	for _, controller := range scsiControllers {
		if len(controller.Device) < SCSIControllerDeviceLimit {
			return controller
		}
	}
	return nil
}

// DiskIsAttached returns if disk is attached to the VM using controllers supported by the plugin.
func (vs *VSphere) DiskIsAttached(volPath string, nodeName k8stypes.NodeName) (bool, error) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	// Ensure client is logged in and session is valid
	err := vSphereLogin(ctx, vs)
	if err != nil {
		glog.Errorf("Failed to login into vCenter - %v", err)
		return false, err
	}

	// Find VM to detach disk from
	var vSphereInstance string
	if nodeName == "" {
		vSphereInstance = vs.localInstanceID
		nodeName = vmNameToNodeName(vSphereInstance)
	} else {
		vSphereInstance = nodeNameToVMName(nodeName)
	}

	nodeExist, err := vs.NodeExists(vs.client, nodeName)
	if err != nil {
		glog.Errorf("Failed to check whether node exist. err: %s.", err)
		return false, err
	}

	if !nodeExist {
		glog.Warningf(
			"Node %q does not exist. DiskIsAttached will assume vmdk %q is not attached to it.",
			vSphereInstance,
			volPath)
		return false, nil
	}

	// Get VM device list
	_, vmDevices, _, dc, err := getVirtualMachineDevices(ctx, vs.cfg, vs.client, vSphereInstance)
	if err != nil {
		glog.Errorf("Failed to get VM devices for VM %#q. err: %s", vSphereInstance, err)
		return false, err
	}

	attached, err := checkDiskAttached(volPath, vmDevices, dc, vs.client)
	return attached, err
}

// DisksAreAttached returns if disks are attached to the VM using controllers supported by the plugin.
func (vs *VSphere) DisksAreAttached(volPaths []string, nodeName k8stypes.NodeName) (map[string]bool, error) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	// Create vSphere client
	attached := make(map[string]bool)
	for _, volPath := range volPaths {
		attached[volPath] = false
	}
	err := vSphereLogin(ctx, vs)
	if err != nil {
		glog.Errorf("Failed to login into vCenter, err: %v", err)
		return attached, err
	}

	// Find VM to detach disk from
	var vSphereInstance string
	if nodeName == "" {
		vSphereInstance = vs.localInstanceID
		nodeName = vmNameToNodeName(vSphereInstance)
	} else {
		vSphereInstance = nodeNameToVMName(nodeName)
	}

	nodeExist, err := vs.NodeExists(vs.client, nodeName)

	if err != nil {
		glog.Errorf("Failed to check whether node exist. err: %s.", err)
		return attached, err
	}

	if !nodeExist {
		glog.Warningf(
			"Node %q does not exist. DisksAreAttached will assume vmdk %v are not attached to it.",
			vSphereInstance,
			volPaths)
		return attached, nil
	}

	// Get VM device list
	_, vmDevices, _, dc, err := getVirtualMachineDevices(ctx, vs.cfg, vs.client, vSphereInstance)
	if err != nil {
		glog.Errorf("Failed to get VM devices for VM %#q. err: %s", vSphereInstance, err)
		return attached, err
	}

	for _, volPath := range volPaths {
		result, _ := checkDiskAttached(volPath, vmDevices, dc, vs.client)
		if result {
			attached[volPath] = true
		}
	}

	return attached, err
}

func checkDiskAttached(volPath string, vmdevices object.VirtualDeviceList, dc *object.Datacenter, client *govmomi.Client) (bool, error) {
	virtualDiskControllerKey, err := getVirtualDiskControllerKey(volPath, vmdevices, dc, client)
	if err != nil {
		if err == ErrNoDevicesFound {
			return false, nil
		}
		glog.Errorf("Failed to check whether disk is attached. err: %s", err)
		return false, err
	}
	for _, controllerType := range supportedSCSIControllerType {
		controllerkey, _ := getControllerKey(controllerType, vmdevices)
		if controllerkey == virtualDiskControllerKey {
			return true, nil
		}
	}
	return false, ErrNonSupportedControllerType

}

// Returns the object key that denotes the controller object to which vmdk is attached.
func getVirtualDiskControllerKey(volPath string, vmDevices object.VirtualDeviceList, dc *object.Datacenter, client *govmomi.Client) (int32, error) {
	volumeUUID, err := getVirtualDiskUUIDByPath(volPath, dc, client)

	if err != nil {
		glog.Errorf("disk uuid not found for %v. err: %s", volPath, err)
		return -1, err
	}

	// filter vm devices to retrieve disk ID for the given vmdk file
	for _, device := range vmDevices {
		if vmDevices.TypeName(device) == "VirtualDisk" {
			diskUUID, _ := getVirtualDiskUUID(device)
			if diskUUID == volumeUUID {
				return device.GetVirtualDevice().ControllerKey, nil
			}
		}
	}
	return -1, ErrNoDevicesFound
}

// Returns key of the controller.
// Key is unique id that distinguishes one device from other devices in the same virtual machine.
func getControllerKey(scsiType string, vmDevices object.VirtualDeviceList) (int32, error) {
	for _, device := range vmDevices {
		devType := vmDevices.Type(device)
		if devType == scsiType {
			if c, ok := device.(types.BaseVirtualController); ok {
				return c.GetVirtualController().Key, nil
			}
		}
	}
	return -1, ErrNoDevicesFound
}

// Returns formatted UUID for a virtual disk device.
func getVirtualDiskUUID(newDevice types.BaseVirtualDevice) (string, error) {
	vd := newDevice.GetVirtualDevice()

	if b, ok := vd.Backing.(*types.VirtualDiskFlatVer2BackingInfo); ok {
		uuid := formatVirtualDiskUUID(b.Uuid)
		return uuid, nil
	}
	return "", ErrNoDiskUUIDFound
}

func formatVirtualDiskUUID(uuid string) string {
	uuidwithNoSpace := strings.Replace(uuid, " ", "", -1)
	uuidWithNoHypens := strings.Replace(uuidwithNoSpace, "-", "", -1)
	return strings.ToLower(uuidWithNoHypens)
}

// Gets virtual disk UUID by datastore (namespace) path
//
// volPath can be namespace path (e.g. "[vsanDatastore] volumes/test.vmdk") or
// uuid path (e.g. "[vsanDatastore] 59427457-6c5a-a917-7997-0200103eedbc/test.vmdk").
// `volumes` in this case would be a symlink to
// `59427457-6c5a-a917-7997-0200103eedbc`.
//
// We want users to use namespace path. It is good for attaching the disk,
// but for detaching the API requires uuid path.  Hence, to detach the right
// device we have to convert the namespace path to uuid path.
func getVirtualDiskUUIDByPath(volPath string, dc *object.Datacenter, client *govmomi.Client) (string, error) {
	if len(volPath) > 0 && filepath.Ext(volPath) != ".vmdk" {
		volPath += ".vmdk"
	}

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	// VirtualDiskManager provides a way to manage and manipulate virtual disks on vmware datastores.
	vdm := object.NewVirtualDiskManager(client.Client)
	// Returns uuid of vmdk virtual disk
	diskUUID, err := vdm.QueryVirtualDiskUuid(ctx, volPath, dc)

	if err != nil {
		return "", ErrNoDiskUUIDFound
	}

	diskUUID = formatVirtualDiskUUID(diskUUID)

	return diskUUID, nil
}

// Returns a device id which is internal vSphere API identifier for the attached virtual disk.
func getVirtualDiskID(volPath string, vmDevices object.VirtualDeviceList, dc *object.Datacenter, client *govmomi.Client) (string, error) {
	volumeUUID, err := getVirtualDiskUUIDByPath(volPath, dc, client)

	if err != nil {
		glog.Warningf("disk uuid not found for %v ", volPath)
		return "", err
	}

	// filter vm devices to retrieve disk ID for the given vmdk file
	for _, device := range vmDevices {
		if vmDevices.TypeName(device) == "VirtualDisk" {
			diskUUID, _ := getVirtualDiskUUID(device)
			if diskUUID == volumeUUID {
				return vmDevices.Name(device), nil
			}
		}
	}
	return "", ErrNoDiskIDFound
}

// DetachDisk detaches given virtual disk volume from the compute running kubelet.
func (vs *VSphere) DetachDisk(volPath string, nodeName k8stypes.NodeName) error {
	// Create context
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	// Ensure client is logged in and session is valid
	err := vSphereLogin(ctx, vs)
	if err != nil {
		glog.Errorf("Failed to login into vCenter - %v", err)
		return err
	}

	// Find virtual machine to attach disk to
	var vSphereInstance string
	if nodeName == "" {
		vSphereInstance = vs.localInstanceID
		nodeName = vmNameToNodeName(vSphereInstance)
	} else {
		vSphereInstance = nodeNameToVMName(nodeName)
	}

	nodeExist, err := vs.NodeExists(vs.client, nodeName)

	if err != nil {
		glog.Errorf("Failed to check whether node exist. err: %s.", err)
		return err
	}

	if !nodeExist {
		glog.Warningf(
			"Node %q does not exist. DetachDisk will assume vmdk %q is not attached to it.",
			nodeName,
			volPath)
		return nil
	}

	vm, vmDevices, _, dc, err := getVirtualMachineDevices(ctx, vs.cfg, vs.client, vSphereInstance)
	if err != nil {
		return err
	}

	diskID, err := getVirtualDiskID(volPath, vmDevices, dc, vs.client)
	if err != nil {
		glog.Warningf("disk ID not found for %v ", volPath)
		return err
	}

	// Gets virtual disk device
	device := vmDevices.Find(diskID)
	if device == nil {
		return fmt.Errorf("device '%s' not found", diskID)
	}

	// Detach disk from VM
	err = vm.RemoveDevice(ctx, true, device)
	if err != nil {
		return err
	}

	return nil
}

// CreateVolume creates a volume of given size (in KiB).
func (vs *VSphere) CreateVolume(volumeOptions *VolumeOptions) (volumePath string, err error) {

	var diskFormat string

	// Default diskformat as 'thin'
	if volumeOptions.DiskFormat == "" {
		volumeOptions.DiskFormat = ThinDiskType
	}

	if _, ok := diskFormatValidType[volumeOptions.DiskFormat]; !ok {
		return "", fmt.Errorf("Cannot create disk. Error diskformat %+q."+
			" Valid options are %s.", volumeOptions.DiskFormat, DiskformatValidOptions)
	}

	diskFormat = diskFormatValidType[volumeOptions.DiskFormat]

	// Create context
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	// Ensure client is logged in and session is valid
	err = vSphereLogin(ctx, vs)
	if err != nil {
		glog.Errorf("Failed to login into vCenter - %v", err)
		return "", err
	}

	// Create a new finder
	f := find.NewFinder(vs.client.Client, true)

	// Fetch and set data center
	dc, err := f.Datacenter(ctx, vs.cfg.Global.Datacenter)
	f.SetDatacenter(dc)

	ds, err := f.Datastore(ctx, vs.cfg.Global.Datastore)
	if err != nil {
		glog.Errorf("Failed while searching for datastore %+q. err %s", vs.cfg.Global.Datastore, err)
		return "", err
	}

	// vmdks will be created inside kubevols directory
	kubeVolsPath := filepath.Clean(ds.Path(VolDir)) + "/"
	err = makeDirectoryInDatastore(vs.client, dc, kubeVolsPath, false)
	if err != nil && err != ErrFileAlreadyExist {
		glog.Errorf("Cannot create dir %#v. err %s", kubeVolsPath, err)
		return "", err
	}
	glog.V(4).Infof("Created dir with path as %+q", kubeVolsPath)

	vmDiskPath := kubeVolsPath + volumeOptions.Name + ".vmdk"

	// Create a virtual disk manager
	virtualDiskManager := object.NewVirtualDiskManager(vs.client.Client)

	// Create specification for new virtual disk
	vmDiskSpec := &types.FileBackedVirtualDiskSpec{
		VirtualDiskSpec: types.VirtualDiskSpec{
			AdapterType: LSILogicControllerType,
			DiskType:    diskFormat,
		},
		CapacityKb: int64(volumeOptions.CapacityKB),
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

// DeleteVolume deletes a volume given volume name.
func (vs *VSphere) DeleteVolume(vmDiskPath string) error {
	// Create context
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	// Ensure client is logged in and session is valid
	err := vSphereLogin(ctx, vs)
	if err != nil {
		glog.Errorf("Failed to login into vCenter - %v", err)
		return err
	}

	// Create a new finder
	f := find.NewFinder(vs.client.Client, true)

	// Fetch and set data center
	dc, err := f.Datacenter(ctx, vs.cfg.Global.Datacenter)
	f.SetDatacenter(dc)

	// Create a virtual disk manager
	virtualDiskManager := object.NewVirtualDiskManager(vs.client.Client)

	// Delete virtual disk
	task, err := virtualDiskManager.DeleteVirtualDisk(ctx, vmDiskPath, dc)
	if err != nil {
		return err
	}

	return task.Wait(ctx)
}

// NodeExists checks if the node with given nodeName exist.
// Returns false if VM doesn't exist or VM is in powerOff state.
func (vs *VSphere) NodeExists(c *govmomi.Client, nodeName k8stypes.NodeName) (bool, error) {
	if nodeName == "" {
		return false, nil
	}

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	vm, err := getVirtualMachineByName(ctx, vs.cfg, c, nodeName)
	if err != nil {
		if _, ok := err.(*find.NotFoundError); ok {
			return false, nil
		}
		glog.Errorf("Failed to get virtual machine object for node %+q. err %s", nodeName, err)
		return false, err
	}

	var mvm mo.VirtualMachine
	err = getVirtualMachineManagedObjectReference(ctx, c, vm, "summary", &mvm)
	if err != nil {
		glog.Errorf("Failed to get virtual machine object reference for node %+q. err %s", nodeName, err)
		return false, err
	}

	if mvm.Summary.Runtime.PowerState == ActivePowerState {
		return true, nil
	}

	if mvm.Summary.Config.Template == false {
		glog.Warningf("VM %s, is not in %s state", nodeName, ActivePowerState)
	} else {
		glog.Warningf("VM %s, is a template", nodeName)
	}

	return false, nil
}

// Creates a folder using the specified name.
// If the intermediate level folders do not exist,
// and the parameter createParents is true,
// all the non-existent folders are created.
func makeDirectoryInDatastore(c *govmomi.Client, dc *object.Datacenter, path string, createParents bool) error {

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	fileManager := object.NewFileManager(c.Client)
	err := fileManager.MakeDirectory(ctx, path, dc, createParents)
	if err != nil {
		if soap.IsSoapFault(err) {
			soapFault := soap.ToSoapFault(err)
			if _, ok := soapFault.VimFault().(types.FileAlreadyExists); ok {
				return ErrFileAlreadyExist
			}
		}
	}

	return err
}
