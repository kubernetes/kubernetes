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
	"io/ioutil"
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

	k8stypes "k8s.io/apimachinery/pkg/types"
	k8runtime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/cloudprovider"
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
	DummyVMName               = "kubernetes-helper-vm"
	VSANDatastoreType         = "vsan"
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
		// VMUUID is the VM Instance UUID of virtual machine which can be retrieved from instanceUuid
		// property in VmConfigInfo, or also set as vc.uuid in VMX file.
		// If not set, will be fetched from the machine via sysfs (requires root)
		VMUUID string `gcfg:"vm-uuid"`
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
	CapacityKB         int
	Tags               map[string]string
	Name               string
	DiskFormat         string
	Datastore          string
	StorageProfileData string
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

// Returns the name of the VM on which this code is running.
// Prerequisite: this code assumes VMWare vmtools or open-vm-tools to be installed in the VM.
// Will attempt to determine the machine's name via it's UUID in this precedence order, failing if neither have a UUID:
// * cloud config value VMUUID
// * sysfs entry
func getVMName(client *govmomi.Client, cfg *VSphereConfig) (string, error) {
	var vmUUID string

	if cfg.Global.VMUUID != "" {
		vmUUID = cfg.Global.VMUUID
	} else {
		// This needs root privileges on the host, and will fail otherwise.
		vmUUIDbytes, err := ioutil.ReadFile("/sys/devices/virtual/dmi/id/product_uuid")
		if err != nil {
			return "", err
		}

		vmUUID = string(vmUUIDbytes)
		cfg.Global.VMUUID = vmUUID
	}

	if vmUUID == "" {
		return "", fmt.Errorf("unable to determine machine ID from cloud configuration or sysfs")
	}

	// Create context
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	// Create a new finder
	f := find.NewFinder(client.Client, true)

	// Fetch and set data center
	dc, err := f.Datacenter(ctx, cfg.Global.Datacenter)
	if err != nil {
		return "", err
	}
	f.SetDatacenter(dc)

	s := object.NewSearchIndex(client.Client)

	svm, err := s.FindByUuid(ctx, dc, strings.ToLower(strings.TrimSpace(vmUUID)), true, nil)
	if err != nil {
		return "", err
	}

	var vm mo.VirtualMachine
	err = s.Properties(ctx, svm.Reference(), []string{"name"}, &vm)
	if err != nil {
		return "", err
	}

	return vm.Name, nil
}

func newVSphere(cfg VSphereConfig) (*VSphere, error) {
	if cfg.Disk.SCSIControllerType == "" {
		cfg.Disk.SCSIControllerType = PVSCSIControllerType
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
	if cfg.Global.VCenterPort != "" {
		glog.Warningf("port is a deprecated field in vsphere.conf and will be removed in future release.")
	}

	c, err := newClient(context.TODO(), &cfg)
	if err != nil {
		return nil, err
	}

	id, err := getVMName(c, &cfg)
	if err != nil {
		return nil, err
	}

	vs := VSphere{
		client:          c,
		cfg:             &cfg,
		localInstanceID: id,
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
	u, err := url.Parse(fmt.Sprintf("https://%s/sdk", cfg.Global.VCenterIP))
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
	if err != nil {
		glog.Errorf("Error while obtaining user session. err: %q", err)
		return err
	}
	if u != nil {
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
func (vs *VSphere) list(filter string) ([]k8stypes.NodeName, error) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	vmList, err := getInstances(ctx, vs.cfg, vs.client, filter)
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
	glog.V(1).Info("The vSphere cloud provider does not support zones")

	return nil, false
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
func getVirtualMachineDevices(ctx context.Context, cfg *VSphereConfig, c *govmomi.Client, name string) (*object.VirtualMachine, object.VirtualDeviceList, *object.Datacenter, error) {
	// Create a new finder
	f := find.NewFinder(c.Client, true)

	// Fetch and set data center
	dc, err := f.Datacenter(ctx, cfg.Global.Datacenter)
	if err != nil {
		return nil, nil, nil, err
	}
	f.SetDatacenter(dc)

	vmRegex := cfg.Global.WorkingDir + name

	vm, err := f.VirtualMachine(ctx, vmRegex)
	if err != nil {
		return nil, nil, nil, err
	}

	// Get devices from VM
	vmDevices, err := vm.Device(ctx)
	if err != nil {
		return nil, nil, nil, err
	}
	return vm, vmDevices, dc, nil
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
	var newSCSIController types.BaseVirtualDevice

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
	vm, vmDevices, dc, err := getVirtualMachineDevices(ctx, vs.cfg, vs.client, vSphereInstance)
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
	scsiControllersOfRequiredType := getSCSIControllersOfType(vmDevices, diskControllerType)
	scsiController := getAvailableSCSIController(scsiControllersOfRequiredType)
	newSCSICreated := false
	if scsiController == nil {
		newSCSIController, err = createAndAttachSCSIControllerToVM(ctx, vm, diskControllerType)
		if err != nil {
			glog.Errorf("Failed to create SCSI controller for VM :%q with err: %+v", vm.Name(), err)
			return "", "", err
		}

		// verify scsi controller in virtual machine
		vmDevices, err := vm.Device(ctx)
		if err != nil {
			return "", "", err
		}

		// Get VM device list
		_, vmDevices, _, err = getVirtualMachineDevices(ctx, vs.cfg, vs.client, vSphereInstance)
		if err != nil {
			glog.Errorf("cannot get vmDevices for VM err=%s", err)
			return "", "", fmt.Errorf("cannot get vmDevices for VM err=%s", err)
		}

		scsiControllersOfRequiredType := getSCSIControllersOfType(vmDevices, diskControllerType)
		scsiController := getAvailableSCSIController(scsiControllersOfRequiredType)
		if scsiController == nil {
			glog.Errorf("cannot find SCSI controller in VM")
			// attempt clean up of scsi controller
			cleanUpController(ctx, newSCSIController, vmDevices, vm)
			return "", "", fmt.Errorf("cannot find SCSI controller in VM")
		}
		newSCSICreated = true
	}

	// Create a new finder
	f := find.NewFinder(vs.client.Client, true)

	// Set data center
	f.SetDatacenter(dc)
	datastorePathObj := new(object.DatastorePath)
	isSuccess := datastorePathObj.FromString(vmDiskPath)
	if !isSuccess {
		glog.Errorf("Failed to parse vmDiskPath: %+q", vmDiskPath)
		return "", "", errors.New("Failed to parse vmDiskPath")
	}
	ds, err := f.Datastore(ctx, datastorePathObj.Datastore)
	if err != nil {
		glog.Errorf("Failed while searching for datastore %+q. err %s", datastorePathObj.Datastore, err)
		return "", "", err
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
		glog.Errorf("DiskIsAttached failed to determine whether disk %q is still attached: node %q does not exist",
			volPath,
			vSphereInstance)
		return false, fmt.Errorf("DiskIsAttached failed to determine whether disk %q is still attached: node %q does not exist",
			volPath,
			vSphereInstance)
	}

	// Get VM device list
	_, vmDevices, dc, err := getVirtualMachineDevices(ctx, vs.cfg, vs.client, vSphereInstance)
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
		glog.Errorf("DisksAreAttached failed to determine whether disks %v are still attached: node %q does not exist",
			volPaths,
			vSphereInstance)
		return attached, fmt.Errorf("DisksAreAttached failed to determine whether disks %v are still attached: node %q does not exist",
			volPaths,
			vSphereInstance)
	}

	// Get VM device list
	_, vmDevices, dc, err := getVirtualMachineDevices(ctx, vs.cfg, vs.client, vSphereInstance)
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
	_, err := getVirtualDiskControllerKey(volPath, vmdevices, dc, client)
	if err != nil {
		if err == ErrNoDevicesFound {
			return false, nil
		}
		glog.Errorf("Failed to check whether disk is attached. err: %s", err)
		return false, err
	}
	return true, nil
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

	vm, vmDevices, dc, err := getVirtualMachineDevices(ctx, vs.cfg, vs.client, vSphereInstance)

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

	var datastore string
	var destVolPath string

	// Default datastore is the datastore in the vSphere config file that is used initialize vSphere cloud provider.
	if volumeOptions.Datastore == "" {
		datastore = vs.cfg.Global.Datastore
	} else {
		datastore = volumeOptions.Datastore
	}

	// Default diskformat as 'thin'
	if volumeOptions.DiskFormat == "" {
		volumeOptions.DiskFormat = ThinDiskType
	}

	if _, ok := diskFormatValidType[volumeOptions.DiskFormat]; !ok {
		return "", fmt.Errorf("Cannot create disk. Error diskformat %+q."+
			" Valid options are %s.", volumeOptions.DiskFormat, DiskformatValidOptions)
	}

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

	ds, err := f.Datastore(ctx, datastore)
	if err != nil {
		glog.Errorf("Failed while searching for datastore %+q. err %s", datastore, err)
		return "", err
	}

	// Create a disk with the VSAN storage capabilities specified in the volumeOptions.StorageProfileData.
	// This is achieved by following steps:
	// 1. Create dummy VM if not already present.
	// 2. Add a new disk to the VM by performing VM reconfigure.
	// 3. Detach the new disk from the dummy VM.
	if volumeOptions.StorageProfileData != "" {
		// Check if the datastore is VSAN if any capability requirements are specified.
		// VSphere cloud provider now only supports VSAN capabilities requirements
		ok, err := checkIfDatastoreTypeIsVSAN(vs.client, ds)
		if err != nil {
			return "", fmt.Errorf("Failed while determining whether the datastore: %q"+
				" is VSAN or not.", datastore)
		}
		if !ok {
			return "", fmt.Errorf("The specified datastore: %q is not a VSAN datastore."+
				" The policy parameters will work only with VSAN Datastore."+
				" So, please specify a valid VSAN datastore in Storage class definition.", datastore)
		}

		// Check if the DummyVM exists in kubernetes cluster folder.
		// The kubernetes cluster folder - vs.cfg.Global.WorkingDir is where all the nodes in the kubernetes cluster are created.
		vmRegex := vs.cfg.Global.WorkingDir + DummyVMName
		dummyVM, err := f.VirtualMachine(ctx, vmRegex)
		if err != nil {
			// 1. Create dummy VM and return the VM reference.
			dummyVM, err = vs.createDummyVM(ctx, dc, ds)
			if err != nil {
				return "", err
			}
		}

		// 2. Reconfigure the VM to attach the disk with the VSAN policy configured.
		vmDiskPath, err := vs.createVirtualDiskWithPolicy(ctx, dc, ds, dummyVM, volumeOptions)
		if err != nil {
			glog.Errorf("Failed to attach the disk to VM: %q with err: %+v", DummyVMName, err)
			return "", err
		}

		dummyVMNodeName := vmNameToNodeName(DummyVMName)
		// 3. Detach the disk from the dummy VM.
		err = vs.DetachDisk(vmDiskPath, dummyVMNodeName)
		if err != nil {
			glog.Errorf("Failed to detach the disk: %q from VM: %q with err: %+v", vmDiskPath, DummyVMName, err)
			return "", fmt.Errorf("Failed to create the volume: %q with err: %+v", volumeOptions.Name, err)
		}
		destVolPath = vmDiskPath
	} else {
		// Create a virtual disk directly if no VSAN storage capabilities are specified by the user.
		destVolPath, err = createVirtualDisk(ctx, vs.client, dc, ds, volumeOptions)
		if err != nil {
			return "", fmt.Errorf("Failed to create the virtual disk having name: %+q with err: %+v", destVolPath, err)
		}
	}
	glog.V(1).Infof("VM Disk path is %+q", destVolPath)
	return destVolPath, nil
}

// DeleteVolume deletes a volume given volume name.
// Also, deletes the folder where the volume resides.
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

	if filepath.Ext(vmDiskPath) != ".vmdk" {
		vmDiskPath += ".vmdk"
	}
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

func (vs *VSphere) createDummyVM(ctx context.Context, datacenter *object.Datacenter, datastore *object.Datastore) (*object.VirtualMachine, error) {
	virtualMachineConfigSpec := types.VirtualMachineConfigSpec{
		Name: DummyVMName,
		Files: &types.VirtualMachineFileInfo{
			VmPathName: "[" + datastore.Name() + "]",
		},
		NumCPUs:  1,
		MemoryMB: 4,
	}

	// Create a new finder
	f := find.NewFinder(vs.client.Client, true)
	f.SetDatacenter(datacenter)

	// Get the folder reference for global working directory where the dummy VM needs to be created.
	vmFolder, err := getFolder(ctx, vs.client, vs.cfg.Global.Datacenter, vs.cfg.Global.WorkingDir)
	if err != nil {
		return nil, fmt.Errorf("Failed to get the folder reference for %q", vs.cfg.Global.WorkingDir)
	}

	vmRegex := vs.cfg.Global.WorkingDir + vs.localInstanceID
	currentVM, err := f.VirtualMachine(ctx, vmRegex)
	if err != nil {
		return nil, err
	}

	currentVMHost, err := currentVM.HostSystem(ctx)
	if err != nil {
		return nil, err
	}

	// Get the resource pool for the current node.
	// We create the dummy VM in the same resource pool as current node.
	resourcePool, err := currentVMHost.ResourcePool(ctx)
	if err != nil {
		return nil, err
	}

	task, err := vmFolder.CreateVM(ctx, virtualMachineConfigSpec, resourcePool, nil)
	if err != nil {
		return nil, err
	}

	dummyVMTaskInfo, err := task.WaitForResult(ctx, nil)
	if err != nil {
		return nil, err
	}

	dummyVM := dummyVMTaskInfo.Result.(*object.VirtualMachine)
	return dummyVM, nil
}

// Creates a virtual disk with the policy configured to the disk.
// A call to this function is made only when a user specifies VSAN storage capabilties in the storage class definition.
func (vs *VSphere) createVirtualDiskWithPolicy(ctx context.Context, datacenter *object.Datacenter, datastore *object.Datastore, virtualMachine *object.VirtualMachine, volumeOptions *VolumeOptions) (string, error) {
	var diskFormat string
	diskFormat = diskFormatValidType[volumeOptions.DiskFormat]

	vmDevices, err := virtualMachine.Device(ctx)
	if err != nil {
		return "", err
	}
	var diskControllerType = vs.cfg.Disk.SCSIControllerType
	// find SCSI controller of particular type from VM devices
	scsiControllersOfRequiredType := getSCSIControllersOfType(vmDevices, diskControllerType)
	scsiController := getAvailableSCSIController(scsiControllersOfRequiredType)
	var newSCSIController types.BaseVirtualDevice
	if scsiController == nil {
		newSCSIController, err = createAndAttachSCSIControllerToVM(ctx, virtualMachine, diskControllerType)
		if err != nil {
			glog.Errorf("Failed to create SCSI controller for VM :%q with err: %+v", virtualMachine.Name(), err)
			return "", err
		}

		// verify scsi controller in virtual machine
		vmDevices, err := virtualMachine.Device(ctx)
		if err != nil {
			return "", err
		}
		scsiController = getSCSIController(vmDevices, diskControllerType)
		if scsiController == nil {
			glog.Errorf("cannot find SCSI controller in VM")
			// attempt clean up of scsi controller
			cleanUpController(ctx, newSCSIController, vmDevices, virtualMachine)
			return "", fmt.Errorf("cannot find SCSI controller in VM")
		}
	}

	kubeVolsPath := filepath.Clean(datastore.Path(VolDir)) + "/"
	// Create a kubevols directory in the datastore if one doesn't exist.
	err = makeDirectoryInDatastore(vs.client, datacenter, kubeVolsPath, false)
	if err != nil && err != ErrFileAlreadyExist {
		glog.Errorf("Cannot create dir %#v. err %s", kubeVolsPath, err)
		return "", err
	}

	glog.V(4).Infof("Created dir with path as %+q", kubeVolsPath)

	vmDiskPath := kubeVolsPath + volumeOptions.Name + ".vmdk"
	disk := vmDevices.CreateDisk(scsiController, datastore.Reference(), vmDiskPath)
	unitNumber, err := getNextUnitNumber(vmDevices, scsiController)
	if err != nil {
		glog.Errorf("cannot attach disk to VM, limit reached - %v.", err)
		return "", err
	}
	*disk.UnitNumber = unitNumber
	disk.CapacityInKB = int64(volumeOptions.CapacityKB)

	backing := disk.Backing.(*types.VirtualDiskFlatVer2BackingInfo)
	backing.DiskMode = string(types.VirtualDiskModeIndependent_persistent)

	switch diskFormat {
	case ThinDiskType:
		backing.ThinProvisioned = types.NewBool(true)
	case EagerZeroedThickDiskType:
		backing.EagerlyScrub = types.NewBool(true)
	default:
		backing.ThinProvisioned = types.NewBool(false)
	}

	// Reconfigure VM
	virtualMachineConfigSpec := types.VirtualMachineConfigSpec{}
	deviceConfigSpec := &types.VirtualDeviceConfigSpec{
		Device:        disk,
		Operation:     types.VirtualDeviceConfigSpecOperationAdd,
		FileOperation: types.VirtualDeviceConfigSpecFileOperationCreate,
	}

	storageProfileSpec := &types.VirtualMachineDefinedProfileSpec{
		ProfileId: "",
		ProfileData: &types.VirtualMachineProfileRawData{
			ExtensionKey: "com.vmware.vim.sps",
			ObjectData:   volumeOptions.StorageProfileData,
		},
	}

	deviceConfigSpec.Profile = append(deviceConfigSpec.Profile, storageProfileSpec)
	virtualMachineConfigSpec.DeviceChange = append(virtualMachineConfigSpec.DeviceChange, deviceConfigSpec)
	task, err := virtualMachine.Reconfigure(ctx, virtualMachineConfigSpec)
	if err != nil {
		glog.Errorf("Failed to reconfigure the VM with the disk with err - %v.", err)
		return "", err
	}

	err = task.Wait(ctx)
	if err != nil {
		glog.Errorf("Failed to reconfigure the VM with the disk with err - %v.", err)
		return "", err
	}

	return vmDiskPath, nil
}

// creating a scsi controller as there is none found.
func createAndAttachSCSIControllerToVM(ctx context.Context, vm *object.VirtualMachine, diskControllerType string) (types.BaseVirtualDevice, error) {
	// Get VM device list
	vmDevices, err := vm.Device(ctx)
	if err != nil {
		return nil, err
	}
	allSCSIControllers := getSCSIControllers(vmDevices)
	if len(allSCSIControllers) >= SCSIControllerLimit {
		// we reached the maximum number of controllers we can attach
		return nil, fmt.Errorf("SCSI Controller Limit of %d has been reached, cannot create another SCSI controller", SCSIControllerLimit)
	}
	newSCSIController, err := vmDevices.CreateSCSIController(diskControllerType)
	if err != nil {
		k8runtime.HandleError(fmt.Errorf("error creating new SCSI controller: %v", err))
		return nil, err
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
		return nil, err
	}
	return newSCSIController, nil
}

// Create a virtual disk.
func createVirtualDisk(ctx context.Context, c *govmomi.Client, dc *object.Datacenter, ds *object.Datastore, volumeOptions *VolumeOptions) (string, error) {
	kubeVolsPath := filepath.Clean(ds.Path(VolDir)) + "/"
	// Create a kubevols directory in the datastore if one doesn't exist.
	err := makeDirectoryInDatastore(c, dc, kubeVolsPath, false)
	if err != nil && err != ErrFileAlreadyExist {
		glog.Errorf("Cannot create dir %#v. err %s", kubeVolsPath, err)
		return "", err
	}

	glog.V(4).Infof("Created dir with path as %+q", kubeVolsPath)
	vmDiskPath := kubeVolsPath + volumeOptions.Name + ".vmdk"

	diskFormat := diskFormatValidType[volumeOptions.DiskFormat]

	// Create a virtual disk manager
	virtualDiskManager := object.NewVirtualDiskManager(c.Client)

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
	return vmDiskPath, task.Wait(ctx)
}

// Check if the provided datastore is VSAN
func checkIfDatastoreTypeIsVSAN(c *govmomi.Client, datastore *object.Datastore) (bool, error) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	pc := property.DefaultCollector(c.Client)

	// Convert datastores into list of references
	var dsRefs []types.ManagedObjectReference
	dsRefs = append(dsRefs, datastore.Reference())

	// Retrieve summary property for the given datastore
	var dsMorefs []mo.Datastore
	err := pc.Retrieve(ctx, dsRefs, []string{"summary"}, &dsMorefs)
	if err != nil {
		return false, err
	}

	for _, ds := range dsMorefs {
		if ds.Summary.Type == VSANDatastoreType {
			return true, nil
		}
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

// Get the folder for a given VM
func getFolder(ctx context.Context, c *govmomi.Client, datacenterName string, folderName string) (*object.Folder, error) {
	f := find.NewFinder(c.Client, true)

	// Fetch and set data center
	dc, err := f.Datacenter(ctx, datacenterName)
	if err != nil {
		return nil, err
	}
	f.SetDatacenter(dc)

	folderName = strings.TrimSuffix(folderName, "/")
	dcFolders, err := dc.Folders(ctx)
	vmFolders, _ := dcFolders.VmFolder.Children(ctx)

	var vmFolderRefs []types.ManagedObjectReference
	for _, vmFolder := range vmFolders {
		vmFolderRefs = append(vmFolderRefs, vmFolder.Reference())
	}

	// Get only references of type folder.
	var folderRefs []types.ManagedObjectReference
	for _, vmFolder := range vmFolderRefs {
		if vmFolder.Type == "Folder" {
			folderRefs = append(folderRefs, vmFolder)
		}
	}

	// Find the specific folder reference matching the folder name.
	var resultFolder *object.Folder
	pc := property.DefaultCollector(c.Client)
	for _, folderRef := range folderRefs {
		var refs []types.ManagedObjectReference
		var folderMorefs []mo.Folder
		refs = append(refs, folderRef)
		err = pc.Retrieve(ctx, refs, []string{"name"}, &folderMorefs)
		for _, fref := range folderMorefs {
			if fref.Name == folderName {
				resultFolder = object.NewFolder(c.Client, folderRef)
			}
		}
	}

	return resultFolder, nil
}
