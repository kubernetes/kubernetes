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
	"net"
	"net/url"
	"path"
	"path/filepath"
	"regexp"
	"runtime"
	"strings"
	"sync"
	"time"

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

	pbm "github.com/vmware/govmomi/pbm"
	k8stypes "k8s.io/apimachinery/pkg/types"
	k8runtime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/kubernetes/pkg/api/v1"
	v1helper "k8s.io/kubernetes/pkg/api/v1/helper"
	"k8s.io/kubernetes/pkg/cloudprovider"
	"k8s.io/kubernetes/pkg/controller"
)

const (
	ProviderName                     = "vsphere"
	ActivePowerState                 = "poweredOn"
	SCSIControllerType               = "scsi"
	LSILogicControllerType           = "lsiLogic"
	BusLogicControllerType           = "busLogic"
	PVSCSIControllerType             = "pvscsi"
	LSILogicSASControllerType        = "lsiLogic-sas"
	SCSIControllerLimit              = 4
	SCSIControllerDeviceLimit        = 15
	SCSIDeviceSlots                  = 16
	SCSIReservedSlot                 = 7
	ThinDiskType                     = "thin"
	PreallocatedDiskType             = "preallocated"
	EagerZeroedThickDiskType         = "eagerZeroedThick"
	ZeroedThickDiskType              = "zeroedThick"
	VolDir                           = "kubevols"
	RoundTripperDefaultCount         = 3
	DummyVMPrefixName                = "vsphere-k8s"
	VSANDatastoreType                = "vsan"
	MAC_OUI_VC                       = "00:50:56"
	MAC_OUI_ESX                      = "00:0c:29"
	DiskNotFoundErrMsg               = "No vSphere disk ID found"
	NoDiskUUIDFoundErrMsg            = "No disk UUID found"
	NoDevicesFoundErrMsg             = "No devices found"
	NonSupportedControllerTypeErrMsg = "Disk is attached to non-supported controller type"
	FileAlreadyExistErrMsg           = "File requested already exist"
	CleanUpDummyVMRoutine_Interval   = 5
	UUIDPath                         = "/sys/class/dmi/id/product_serial"
	UUIDPrefix                       = "VMware-"
	NameProperty                     = "name"
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
var cleanUpRoutineInitialized = false

var ErrNoDiskUUIDFound = errors.New(NoDiskUUIDFoundErrMsg)
var ErrNoDiskIDFound = errors.New(DiskNotFoundErrMsg)
var ErrNoDevicesFound = errors.New(NoDevicesFoundErrMsg)
var ErrNonSupportedControllerType = errors.New(NonSupportedControllerTypeErrMsg)
var ErrFileAlreadyExist = errors.New(FileAlreadyExistErrMsg)

var clientLock sync.Mutex
var cleanUpRoutineInitLock sync.Mutex
var cleanUpDummyVMLock sync.RWMutex

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
		// VMName is the VM name of virtual machine
		// Combining the WorkingDir and VMName can form a unique InstanceID.
		// When vm-name is set, no username/password is required on worker nodes.
		VMName string `gcfg:"vm-name"`
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
	AttachDisk(vmDiskPath string, storagePolicyID string, nodeName k8stypes.NodeName) (diskID string, diskUUID string, err error)

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
	CapacityKB             int
	Tags                   map[string]string
	Name                   string
	DiskFormat             string
	Datastore              string
	VSANStorageProfileData string
	StoragePolicyName      string
	StoragePolicyID        string
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
	registerMetrics()
	cloudprovider.RegisterCloudProvider(ProviderName, func(config io.Reader) (cloudprovider.Interface, error) {
		cfg, err := readConfig(config)
		if err != nil {
			return nil, err
		}
		return newVSphere(cfg)
	})
}

// Initialize passes a Kubernetes clientBuilder interface to the cloud provider
func (vs *VSphere) Initialize(clientBuilder controller.ControllerClientBuilder) {}

// UUID gets the BIOS UUID via the sys interface.  This UUID is known by vsphere
func getvmUUID() (string, error) {
	id, err := ioutil.ReadFile(UUIDPath)
	if err != nil {
		return "", fmt.Errorf("error retrieving vm uuid: %s", err)
	}
	uuidFromFile := string(id[:])
	//strip leading and trailing white space and new line char
	uuid := strings.TrimSpace(uuidFromFile)
	// check the uuid starts with "VMware-"
	if !strings.HasPrefix(uuid, UUIDPrefix) {
		return "", fmt.Errorf("Failed to match Prefix, UUID read from the file is %v", uuidFromFile)
	}
	// Strip the prefix and while spaces and -
	uuid = strings.Replace(uuid[len(UUIDPrefix):(len(uuid))], " ", "", -1)
	uuid = strings.Replace(uuid, "-", "", -1)
	if len(uuid) != 32 {
		return "", fmt.Errorf("Length check failed, UUID read from the file is %v", uuidFromFile)
	}
	// need to add dashes, e.g. "564d395e-d807-e18a-cb25-b79f65eb2b9f"
	uuid = fmt.Sprintf("%s-%s-%s-%s-%s", uuid[0:8], uuid[8:12], uuid[12:16], uuid[16:20], uuid[20:32])
	return uuid, nil
}

// Returns the name of the VM on which this code is running.
// Will attempt to determine the machine's name via it's UUID in this precedence order, failing if neither have a UUID:
// * cloud config value VMUUID
// * sysfs entry
func getVMName(client *govmomi.Client, cfg *VSphereConfig) (string, error) {
	var vmUUID string
	var err error

	if cfg.Global.VMUUID != "" {
		vmUUID = cfg.Global.VMUUID
	} else {
		// This needs root privileges on the host, and will fail otherwise.
		vmUUID, err = getvmUUID()
		if err != nil {
			return "", err
		}
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

	if svm == nil {
		return "", fmt.Errorf("unable to find machine reference by UUID")
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

	var c *govmomi.Client
	var id string
	if cfg.Global.VMName == "" {
		// if VMName is not set in the cloud config file, each nodes (including worker nodes) need credentials to obtain VMName from vCenter
		glog.V(4).Infof("Cannot find VMName from cloud config file, start obtaining it from vCenter")
		c, err := newClient(context.TODO(), &cfg)
		if err != nil {
			return nil, err
		}

		id, err = getVMName(c, &cfg)
		if err != nil {
			return nil, err
		}
	} else {
		id = cfg.Global.VMName
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
	if vs.client != nil {
		vs.client.Logout(context.TODO())
	}
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

// Instances returns an implementation of Instances for vSphere.
func (vs *VSphere) Instances() (cloudprovider.Instances, bool) {
	return vs, true
}

func getLocalIP() ([]v1.NodeAddress, error) {
	addrs := []v1.NodeAddress{}

	ifaces, err := net.Interfaces()
	if err != nil {
		glog.Errorf("net.Interfaces() failed for NodeAddresses - %v", err)
		return nil, err
	}

	for _, i := range ifaces {
		localAddrs, err := i.Addrs()
		if err != nil {
			glog.Warningf("Failed to extract addresses for NodeAddresses - %v", err)
		} else {
			for _, addr := range localAddrs {
				if ipnet, ok := addr.(*net.IPNet); ok && !ipnet.IP.IsLoopback() {
					if ipnet.IP.To4() != nil {
						// Filter external IP by MAC address OUIs from vCenter and from ESX
						var addressType v1.NodeAddressType
						if strings.HasPrefix(i.HardwareAddr.String(), MAC_OUI_VC) ||
							strings.HasPrefix(i.HardwareAddr.String(), MAC_OUI_ESX) {
							v1helper.AddToNodeAddresses(&addrs,
								v1.NodeAddress{
									Type:    v1.NodeExternalIP,
									Address: ipnet.IP.String(),
								},
								v1.NodeAddress{
									Type:    v1.NodeInternalIP,
									Address: ipnet.IP.String(),
								},
							)
						}
						glog.V(4).Infof("Find local IP address %v and set type to %v", ipnet.IP.String(), addressType)
					}
				}
			}
		}
	}
	return addrs, nil
}

// getVMandMO returns the VM object and required field from the VM object
func (vs *VSphere) getVMandMO(ctx context.Context, nodeName k8stypes.NodeName, field string) (vm *object.VirtualMachine, mvm *mo.VirtualMachine, err error) {
	// Ensure client is logged in and session is valid
	err = vSphereLogin(ctx, vs)
	if err != nil {
		glog.Errorf("Failed to login into vCenter - %v", err)
		return nil, nil, err
	}

	vm, err = getVirtualMachineByName(ctx, vs.cfg, vs.client, nodeName)
	if err != nil {
		if _, ok := err.(*find.NotFoundError); ok {
			return nil, nil, cloudprovider.InstanceNotFound
		}
		return nil, nil, err
	}

	// Retrieve required field from VM object
	var movm mo.VirtualMachine
	collector := property.DefaultCollector(vs.client.Client)
	err = collector.RetrieveOne(ctx, vm.Reference(), []string{field}, &movm)
	if err != nil {
		return nil, nil, err
	}

	return vm, &movm, nil
}

// NodeAddresses is an implementation of Instances.NodeAddresses.
func (vs *VSphere) NodeAddresses(nodeName k8stypes.NodeName) ([]v1.NodeAddress, error) {
	if vs.localInstanceID == nodeNameToVMName(nodeName) {
		/* Get local IP addresses if node is local node */
		return getLocalIP()
	}

	addrs := []v1.NodeAddress{}

	// Create context
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	_, mvm, err := vs.getVMandMO(ctx, nodeName, "guest.net")
	if err != nil {
		glog.Errorf("Failed to getVMandMO for NodeAddresses: err %v", err)
		return addrs, err
	}

	// retrieve VM's ip(s)
	for _, v := range mvm.Guest.Net {
		if vs.cfg.Network.PublicNetwork == v.Network {
			for _, ip := range v.IpAddress {
				if net.ParseIP(ip).To4() != nil {
					v1helper.AddToNodeAddresses(&addrs,
						v1.NodeAddress{
							Type:    v1.NodeExternalIP,
							Address: ip,
						}, v1.NodeAddress{
							Type:    v1.NodeInternalIP,
							Address: ip,
						},
					)
				}
			}
		}
	}
	return addrs, nil
}

// NodeAddressesByProviderID returns the node addresses of an instances with the specified unique providerID
// This method will not be called from the node that is requesting this ID. i.e. metadata service
// and other local methods cannot be used here
func (vs *VSphere) NodeAddressesByProviderID(providerID string) ([]v1.NodeAddress, error) {
	return []v1.NodeAddress{}, errors.New("unimplemented")
}

func (vs *VSphere) AddSSHKeyToAllInstances(user string, keyData []byte) error {
	return errors.New("unimplemented")
}

func (vs *VSphere) CurrentNodeName(hostname string) (k8stypes.NodeName, error) {
	return vmNameToNodeName(vs.localInstanceID), nil
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
func (vs *VSphere) ExternalID(nodeName k8stypes.NodeName) (string, error) {
	if vs.localInstanceID == nodeNameToVMName(nodeName) {
		return vs.cfg.Global.WorkingDir + vs.localInstanceID, nil
	}

	// Create context
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	vm, mvm, err := vs.getVMandMO(ctx, nodeName, "summary")
	if err != nil {
		glog.Errorf("Failed to getVMandMO for ExternalID: err %v", err)
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
func (vs *VSphere) InstanceID(nodeName k8stypes.NodeName) (string, error) {
	if vs.localInstanceID == nodeNameToVMName(nodeName) {
		return vs.cfg.Global.WorkingDir + vs.localInstanceID, nil
	}

	// Create context
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	vm, mvm, err := vs.getVMandMO(ctx, nodeName, "summary")
	if err != nil {
		glog.Errorf("Failed to getVMandMO for InstanceID: err %v", err)
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

// InstanceTypeByProviderID returns the cloudprovider instance type of the node with the specified unique providerID
// This method will not be called from the node that is requesting this ID. i.e. metadata service
// and other local methods cannot be used here
func (vs *VSphere) InstanceTypeByProviderID(providerID string) (string, error) {
	return "", errors.New("unimplemented")
}

func (vs *VSphere) InstanceType(name k8stypes.NodeName) (string, error) {
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
func (vs *VSphere) AttachDisk(vmDiskPath string, storagePolicyID string, nodeName k8stypes.NodeName) (diskID string, diskUUID string, err error) {
	attachDiskInternal := func(vmDiskPath string, storagePolicyID string, nodeName k8stypes.NodeName) (diskID string, diskUUID string, err error) {
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

		attached, err := checkDiskAttached(vmDiskPath, vmDevices, dc, vs.client)
		if err != nil {
			return "", "", err
		}
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

			// Get VM device list
			_, vmDevices, _, err := getVirtualMachineDevices(ctx, vs.cfg, vs.client, vSphereInstance)
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
		vmDiskPath = removeClusterFromVDiskPath(vmDiskPath)
		disk := vmDevices.CreateDisk(scsiController, ds.Reference(), vmDiskPath)
		unitNumber, err := getNextUnitNumber(vmDevices, scsiController)
		if err != nil {
			glog.Errorf("cannot attach disk to VM, limit reached - %v.", err)
			return "", "", err
		}
		*disk.UnitNumber = unitNumber

		backing := disk.Backing.(*types.VirtualDiskFlatVer2BackingInfo)
		backing.DiskMode = string(types.VirtualDiskModeIndependent_persistent)

		virtualMachineConfigSpec := types.VirtualMachineConfigSpec{}
		deviceConfigSpec := &types.VirtualDeviceConfigSpec{
			Device:    disk,
			Operation: types.VirtualDeviceConfigSpecOperationAdd,
		}
		// Configure the disk with the SPBM profile only if ProfileID is not empty.
		if storagePolicyID != "" {
			profileSpec := &types.VirtualMachineDefinedProfileSpec{
				ProfileId: storagePolicyID,
			}
			deviceConfigSpec.Profile = append(deviceConfigSpec.Profile, profileSpec)
		}
		virtualMachineConfigSpec.DeviceChange = append(virtualMachineConfigSpec.DeviceChange, deviceConfigSpec)
		requestTime := time.Now()
		task, err := vm.Reconfigure(ctx, virtualMachineConfigSpec)
		if err != nil {
			recordvSphereMetric(api_attachvolume, requestTime, err)
			glog.Errorf("Failed to attach the disk with storagePolicy: %+q with err - %v", storagePolicyID, err)
			if newSCSICreated {
				cleanUpController(ctx, newSCSIController, vmDevices, vm)
			}
			return "", "", err
		}
		err = task.Wait(ctx)
		recordvSphereMetric(api_attachvolume, requestTime, err)
		if err != nil {
			glog.Errorf("Failed to attach the disk with storagePolicy: %+q with err - %v", storagePolicyID, err)
			if newSCSICreated {
				cleanUpController(ctx, newSCSIController, vmDevices, vm)
			}
			return "", "", err
		}

		deviceName, diskUUID, err := getVMDiskInfo(ctx, vm, disk)
		if err != nil {
			if newSCSICreated {
				cleanUpController(ctx, newSCSIController, vmDevices, vm)
			}
			vs.DetachDisk(deviceName, nodeName)
			return "", "", err
		}
		return deviceName, diskUUID, nil
	}
	requestTime := time.Now()
	diskID, diskUUID, err = attachDiskInternal(vmDiskPath, storagePolicyID, nodeName)
	recordvSphereMetric(operation_attachvolume, requestTime, err)
	return diskID, diskUUID, err
}

func getVMDiskInfo(ctx context.Context, vm *object.VirtualMachine, disk *types.VirtualDisk) (string, string, error) {
	vmDevices, err := vm.Device(ctx)
	if err != nil {
		return "", "", err
	}
	devices := vmDevices.SelectByType(disk)
	if len(devices) < 1 {
		return "", "", ErrNoDevicesFound
	}

	// get new disk id
	newDevice := devices[len(devices)-1]
	deviceName := devices.Name(newDevice)

	// get device uuid
	diskUUID, err := getVirtualDiskUUID(newDevice)
	if err != nil {
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
	diskIsAttachedInternal := func(volPath string, nodeName k8stypes.NodeName) (bool, error) {
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

		nodeExist, err := vs.NodeExists(nodeName)
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
	requestTime := time.Now()
	isAttached, err := diskIsAttachedInternal(volPath, nodeName)
	recordvSphereMetric(operation_diskIsAttached, requestTime, err)
	return isAttached, err
}

// DisksAreAttached returns if disks are attached to the VM using controllers supported by the plugin.
func (vs *VSphere) DisksAreAttached(volPaths []string, nodeName k8stypes.NodeName) (map[string]bool, error) {
	disksAreAttachedInternal := func(volPaths []string, nodeName k8stypes.NodeName) (map[string]bool, error) {
		ctx, cancel := context.WithCancel(context.Background())
		defer cancel()

		// Create vSphere client
		err := vSphereLogin(ctx, vs)
		if err != nil {
			glog.Errorf("Failed to login into vCenter, err: %v", err)
			return nil, err
		}

		// Find VM to detach disk from
		var vSphereInstance string
		if nodeName == "" {
			vSphereInstance = vs.localInstanceID
			nodeName = vmNameToNodeName(vSphereInstance)
		} else {
			vSphereInstance = nodeNameToVMName(nodeName)
		}

		nodeExist, err := vs.NodeExists(nodeName)

		if err != nil {
			glog.Errorf("Failed to check whether node exist. err: %s.", err)
			return nil, err
		}

		if !nodeExist {
			glog.Errorf("DisksAreAttached failed to determine whether disks %v are still attached: node %q does not exist",
				volPaths,
				vSphereInstance)
			return nil, fmt.Errorf("DisksAreAttached failed to determine whether disks %v are still attached: node %q does not exist",
				volPaths,
				vSphereInstance)
		}

		// Get VM device list
		_, vmDevices, dc, err := getVirtualMachineDevices(ctx, vs.cfg, vs.client, vSphereInstance)
		if err != nil {
			glog.Errorf("Failed to get VM devices for VM %#q. err: %s", vSphereInstance, err)
			return nil, err
		}

		attached := make(map[string]bool)
		for _, volPath := range volPaths {
			result, err := checkDiskAttached(volPath, vmDevices, dc, vs.client)
			if err == nil {
				if result {
					attached[volPath] = true
				} else {
					attached[volPath] = false
				}
			} else {
				return nil, err
			}
		}
		return attached, nil
	}
	requestTime := time.Now()
	attached, err := disksAreAttachedInternal(volPaths, nodeName)
	recordvSphereMetric(operation_disksAreAttached, requestTime, err)
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
	volPath = removeClusterFromVDiskPath(volPath)
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
	detachDiskInternal := func(volPath string, nodeName k8stypes.NodeName) error {
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
		volPath = removeClusterFromVDiskPath(volPath)
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
		requestTime := time.Now()
		err = vm.RemoveDevice(ctx, true, device)
		recordvSphereMetric(api_detachvolume, requestTime, err)
		if err != nil {
			return err
		}
		return nil
	}
	requestTime := time.Now()
	err := detachDiskInternal(volPath, nodeName)
	recordvSphereMetric(operation_detachvolume, requestTime, nil)
	return err
}

// CreateVolume creates a volume of given size (in KiB).
func (vs *VSphere) CreateVolume(volumeOptions *VolumeOptions) (volumePath string, err error) {
	createVolumeInternal := func(volumeOptions *VolumeOptions) (volumePath string, err error) {
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

		if volumeOptions.StoragePolicyName != "" {
			// Get the pbm client
			pbmClient, err := pbm.NewClient(ctx, vs.client.Client)
			if err != nil {
				return "", err
			}
			volumeOptions.StoragePolicyID, err = pbmClient.ProfileIDByName(ctx, volumeOptions.StoragePolicyName)
			if err != nil {
				recordvSphereMetric(operation_createvolume_with_policy, time.Time{}, err)
				return "", err
			}

			compatibilityResult, err := vs.GetPlacementCompatibilityResult(ctx, pbmClient, volumeOptions.StoragePolicyID)
			if err != nil {
				return "", err
			}
			if len(compatibilityResult) < 1 {
				return "", fmt.Errorf("There are no compatible datastores that satisfy the storage policy: %+q requirements", volumeOptions.StoragePolicyID)
			}

			if volumeOptions.Datastore != "" {
				ok, nonCompatibleDsref := vs.IsUserSpecifiedDatastoreNonCompatible(ctx, compatibilityResult, volumeOptions.Datastore)
				if ok {
					faultMsg := GetNonCompatibleDatastoreFaultMsg(compatibilityResult, *nonCompatibleDsref)
					return "", fmt.Errorf("User specified datastore: %q is not compatible with the storagePolicy: %q. Failed with faults: %+q", volumeOptions.Datastore, volumeOptions.StoragePolicyName, faultMsg)
				}
			} else {
				dsMoList, err := vs.GetCompatibleDatastoresMo(ctx, compatibilityResult)
				if err != nil {
					recordvSphereMetric(operation_createvolume_with_raw_vsan_policy, time.Time{}, err)
					return "", err
				}
				dsMo := GetMostFreeDatastore(dsMoList)
				datastore = dsMo.Info.GetDatastoreInfo().Name
			}
		}
		ds, err := f.Datastore(ctx, datastore)
		if err != nil {
			glog.Errorf("Failed while searching for datastore %+q. err %s", datastore, err)
			return "", err
		}

		if volumeOptions.VSANStorageProfileData != "" {
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
		}
		// Create a disk with the VSAN storage capabilities specified in the volumeOptions.VSANStorageProfileData.
		// This is achieved by following steps:
		// 1. Create dummy VM if not already present.
		// 2. Add a new disk to the VM by performing VM reconfigure.
		// 3. Detach the new disk from the dummy VM.
		// 4. Delete the dummy VM.
		if volumeOptions.VSANStorageProfileData != "" || volumeOptions.StoragePolicyName != "" {
			// Acquire a read lock to ensure multiple PVC requests can be processed simultaneously.
			cleanUpDummyVMLock.RLock()
			defer cleanUpDummyVMLock.RUnlock()

			// Create a new background routine that will delete any dummy VM's that are left stale.
			// This routine will get executed for every 5 minutes and gets initiated only once in its entire lifetime.
			cleanUpRoutineInitLock.Lock()
			if !cleanUpRoutineInitialized {
				go vs.cleanUpDummyVMs(DummyVMPrefixName)
				cleanUpRoutineInitialized = true
			}
			cleanUpRoutineInitLock.Unlock()

			// Check if the VM exists in kubernetes cluster folder.
			// The kubernetes cluster folder - vs.cfg.Global.WorkingDir is where all the nodes in the kubernetes cluster are created.
			dummyVMFullName := DummyVMPrefixName + "-" + volumeOptions.Name
			vmRegex := vs.cfg.Global.WorkingDir + dummyVMFullName
			dummyVM, err := f.VirtualMachine(ctx, vmRegex)
			if err != nil {
				// 1. Create a dummy VM and return the VM reference.
				dummyVM, err = vs.createDummyVM(ctx, dc, ds, dummyVMFullName)
				if err != nil {
					return "", err
				}
			}

			// 2. Reconfigure the VM to attach the disk with the VSAN policy configured.
			vmDiskPath, err := vs.createVirtualDiskWithPolicy(ctx, dc, ds, dummyVM, volumeOptions)
			fileAlreadyExist := false
			if err != nil {
				vmDiskPath = filepath.Clean(ds.Path(VolDir)) + "/" + volumeOptions.Name + ".vmdk"
				errorMessage := fmt.Sprintf("Cannot complete the operation because the file or folder %s already exists", vmDiskPath)
				if errorMessage == err.Error() {
					//Skip error and continue to detach the disk as the disk was already created on the datastore.
					fileAlreadyExist = true
					glog.V(1).Infof("File: %v already exists", vmDiskPath)
				} else {
					glog.Errorf("Failed to attach the disk to VM: %q with err: %+v", dummyVMFullName, err)
					return "", err
				}
			}

			dummyVMNodeName := vmNameToNodeName(dummyVMFullName)
			// 3. Detach the disk from the dummy VM.
			err = vs.DetachDisk(vmDiskPath, dummyVMNodeName)
			if err != nil {
				if DiskNotFoundErrMsg == err.Error() && fileAlreadyExist {
					// Skip error if disk was already detached from the dummy VM but still present on the datastore.
					glog.V(1).Infof("File: %v is already detached", vmDiskPath)
				} else {
					glog.Errorf("Failed to detach the disk: %q from VM: %q with err: %+v", vmDiskPath, dummyVMFullName, err)
					return "", fmt.Errorf("Failed to create the volume: %q with err: %+v", volumeOptions.Name, err)
				}
			}

			// 4. Delete the dummy VM
			err = deleteVM(ctx, dummyVM)
			if err != nil {
				return "", fmt.Errorf("Failed to destroy the vm: %q with err: %+v", dummyVMFullName, err)
			}
			destVolPath = vmDiskPath
		} else {
			// Create a virtual disk directly if no VSAN storage capabilities are specified by the user.
			destVolPath, err = createVirtualDisk(ctx, vs.client, dc, ds, volumeOptions)
			if err != nil {
				return "", fmt.Errorf("Failed to create the virtual disk having name: %+q with err: %+v", destVolPath, err)
			}
		}

		if filepath.Base(datastore) != datastore {
			// If Datastore is within cluster, add cluster path to the destVolPath
			destVolPath = strings.Replace(destVolPath, filepath.Base(datastore), datastore, 1)
		}
		glog.V(1).Infof("VM Disk path is %+q", destVolPath)
		return destVolPath, nil
	}
	requestTime := time.Now()
	volumePath, err = createVolumeInternal(volumeOptions)
	recordCreateVolumeMetric(volumeOptions, requestTime, err)
	if err != nil {
		return "", err
	}
	return volumePath, nil
}

// DeleteVolume deletes a volume given volume name.
// Also, deletes the folder where the volume resides.
func (vs *VSphere) DeleteVolume(vmDiskPath string) error {
	deleteVolumeInternal := func(vmDiskPath string) error {
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

		// Get the vmDisk Name
		diskNameWithExt := path.Base(vmDiskPath)
		diskName := strings.TrimSuffix(diskNameWithExt, filepath.Ext(diskNameWithExt))

		// Search for the dummyVM if present and delete it.
		dummyVMFullName := DummyVMPrefixName + "-" + diskName
		vmRegex := vs.cfg.Global.WorkingDir + dummyVMFullName
		dummyVM, err := f.VirtualMachine(ctx, vmRegex)
		if err == nil {
			err = deleteVM(ctx, dummyVM)
			if err != nil {
				return fmt.Errorf("Failed to destroy the vm: %q with err: %+v", dummyVMFullName, err)
			}
		}

		// Delete virtual disk
		vmDiskPath = removeClusterFromVDiskPath(vmDiskPath)
		requestTime := time.Now()
		task, err := virtualDiskManager.DeleteVirtualDisk(ctx, vmDiskPath, dc)
		if err != nil {
			recordvSphereMetric(api_deletevolume, requestTime, err)
			return err
		}
		err = task.Wait(ctx)
		recordvSphereMetric(api_deletevolume, requestTime, err)
		return err
	}
	requestTime := time.Now()
	err := deleteVolumeInternal(vmDiskPath)
	recordvSphereMetric(operation_deletevolume, requestTime, err)
	return err
}

// NodeExists checks if the node with given nodeName exist.
// Returns false if VM doesn't exist or VM is in powerOff state.
func (vs *VSphere) NodeExists(nodeName k8stypes.NodeName) (bool, error) {
	if nodeName == "" {
		return false, nil
	}

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	_, mvm, err := vs.getVMandMO(ctx, nodeName, "summary")
	if err != nil {
		glog.Errorf("Failed to getVMandMO for NodeExists: err %v", err)
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

// A background routine which will be responsible for deleting stale dummy VM's.
func (vs *VSphere) cleanUpDummyVMs(dummyVMPrefix string) {
	// Create context
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	for {
		time.Sleep(CleanUpDummyVMRoutine_Interval * time.Minute)
		// Ensure client is logged in and session is valid
		err := vSphereLogin(ctx, vs)
		if err != nil {
			glog.V(4).Infof("[cleanUpDummyVMs] Unable to login to vSphere with err: %+v", err)
			continue
		}

		// Create a new finder
		f := find.NewFinder(vs.client.Client, true)

		// Fetch and set data center
		dc, err := f.Datacenter(ctx, vs.cfg.Global.Datacenter)
		if err != nil {
			glog.V(4).Infof("[cleanUpDummyVMs] Unable to fetch the datacenter: %q with err: %+v", vs.cfg.Global.Datacenter, err)
			continue
		}
		f.SetDatacenter(dc)

		// Get the folder reference for global working directory where the dummy VM needs to be created.
		vmFolder, err := f.Folder(ctx, strings.TrimSuffix(vs.cfg.Global.WorkingDir, "/"))
		if err != nil {
			glog.V(4).Infof("[cleanUpDummyVMs] Unable to get the kubernetes folder: %q reference with err: %+v", vs.cfg.Global.WorkingDir, err)
			continue
		}

		// A write lock is acquired to make sure the cleanUp routine doesn't delete any VM's created by ongoing PVC requests.
		cleanUpDummyVMLock.Lock()
		vmMoList, err := vs.GetVMsInsideFolder(ctx, vmFolder, []string{NameProperty})
		if err != nil {
			glog.V(4).Infof("[cleanUpDummyVMs] Unable to get VM list in the kubernetes cluster: %q reference with err: %+v", vs.cfg.Global.WorkingDir, err)
			cleanUpDummyVMLock.Unlock()
			continue
		}
		var dummyVMRefList []*object.VirtualMachine
		for _, vmMo := range vmMoList {
			if strings.HasPrefix(vmMo.Name, dummyVMPrefix) {
				dummyVMRefList = append(dummyVMRefList, object.NewVirtualMachine(vs.client.Client, vmMo.Reference()))
			}
		}

		for _, dummyVMRef := range dummyVMRefList {
			err = deleteVM(ctx, dummyVMRef)
			if err != nil {
				glog.V(4).Infof("[cleanUpDummyVMs] Unable to delete dummy VM: %q with err: %+v", dummyVMRef.Name(), err)
				continue
			}
		}
		cleanUpDummyVMLock.Unlock()
	}
}

func (vs *VSphere) createDummyVM(ctx context.Context, datacenter *object.Datacenter, datastore *object.Datastore, vmName string) (*object.VirtualMachine, error) {
	// Create a virtual machine config spec with 1 SCSI adapter.
	virtualMachineConfigSpec := types.VirtualMachineConfigSpec{
		Name: vmName,
		Files: &types.VirtualMachineFileInfo{
			VmPathName: "[" + datastore.Name() + "]",
		},
		NumCPUs:  1,
		MemoryMB: 4,
		DeviceChange: []types.BaseVirtualDeviceConfigSpec{
			&types.VirtualDeviceConfigSpec{
				Operation: types.VirtualDeviceConfigSpecOperationAdd,
				Device: &types.ParaVirtualSCSIController{
					VirtualSCSIController: types.VirtualSCSIController{
						SharedBus: types.VirtualSCSISharingNoSharing,
						VirtualController: types.VirtualController{
							BusNumber: 0,
							VirtualDevice: types.VirtualDevice{
								Key: 1000,
							},
						},
					},
				},
			},
		},
	}

	// Get the resource pool for current node. This is where dummy VM will be created.
	resourcePool, err := vs.getCurrentNodeResourcePool(ctx, datacenter)
	if err != nil {
		return nil, err
	}
	// Get the folder reference for global working directory where the dummy VM needs to be created.
	f := find.NewFinder(vs.client.Client, true)
	dc, err := f.Datacenter(ctx, vs.cfg.Global.Datacenter)
	f.SetDatacenter(dc)
	vmFolder, err := f.Folder(ctx, strings.TrimSuffix(vs.cfg.Global.WorkingDir, "/"))
	if err != nil {
		return nil, fmt.Errorf("Failed to get the folder reference for %q with err: %+v", vs.cfg.Global.WorkingDir, err)
	}
	task, err := vmFolder.CreateVM(ctx, virtualMachineConfigSpec, resourcePool, nil)
	if err != nil {
		return nil, err
	}

	dummyVMTaskInfo, err := task.WaitForResult(ctx, nil)
	if err != nil {
		return nil, err
	}

	vmRef := dummyVMTaskInfo.Result.(object.Reference)
	dummyVM := object.NewVirtualMachine(vs.client.Client, vmRef.Reference())
	return dummyVM, nil
}

func (vs *VSphere) getCurrentNodeResourcePool(ctx context.Context, datacenter *object.Datacenter) (*object.ResourcePool, error) {
	// Create a new finder
	f := find.NewFinder(vs.client.Client, true)
	f.SetDatacenter(datacenter)

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

	return resourcePool, nil
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
	scsiController := scsiControllersOfRequiredType[0]

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

	storageProfileSpec := &types.VirtualMachineDefinedProfileSpec{}
	// Is PBM storage policy ID is present, set the storage spec profile ID,
	// else, set raw the VSAN policy string.
	if volumeOptions.StoragePolicyID != "" {
		storageProfileSpec.ProfileId = volumeOptions.StoragePolicyID
	} else if volumeOptions.VSANStorageProfileData != "" {
		storageProfileSpec.ProfileId = ""
		storageProfileSpec.ProfileData = &types.VirtualMachineProfileRawData{
			ExtensionKey: "com.vmware.vim.sps",
			ObjectData:   volumeOptions.VSANStorageProfileData,
		}
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
	requestTime := time.Now()
	task, err := virtualDiskManager.CreateVirtualDisk(ctx, vmDiskPath, dc, vmDiskSpec)
	if err != nil {
		recordvSphereMetric(api_createvolume, requestTime, err)
		return "", err
	}
	err = task.Wait(ctx)
	recordvSphereMetric(api_createvolume, requestTime, err)
	return vmDiskPath, err
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

// Delete the VM.
func deleteVM(ctx context.Context, vm *object.VirtualMachine) error {
	destroyTask, err := vm.Destroy(ctx)
	if err != nil {
		return err
	}
	return destroyTask.Wait(ctx)
}

// Remove the cluster or folder path from the vDiskPath
// for vDiskPath [DatastoreCluster/sharedVmfs-0] kubevols/e2e-vmdk-1234.vmdk, return value is [sharedVmfs-0] kubevols/e2e-vmdk-1234.vmdk
// for vDiskPath [sharedVmfs-0] kubevols/e2e-vmdk-1234.vmdk, return value remains same [sharedVmfs-0] kubevols/e2e-vmdk-1234.vmdk

func removeClusterFromVDiskPath(vDiskPath string) string {
	datastore := regexp.MustCompile("\\[(.*?)\\]").FindStringSubmatch(vDiskPath)[1]
	if filepath.Base(datastore) != datastore {
		vDiskPath = strings.Replace(vDiskPath, datastore, filepath.Base(datastore), 1)
	}
	return vDiskPath
}
