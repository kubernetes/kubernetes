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
	"os"
	"path"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"time"

	"gopkg.in/gcfg.v1"

	"github.com/golang/glog"
	"golang.org/x/net/context"
	"k8s.io/api/core/v1"
	k8stypes "k8s.io/apimachinery/pkg/types"
	v1helper "k8s.io/kubernetes/pkg/apis/core/v1/helper"
	"k8s.io/kubernetes/pkg/cloudprovider"
	"k8s.io/kubernetes/pkg/cloudprovider/providers/vsphere/vclib"
	"k8s.io/kubernetes/pkg/cloudprovider/providers/vsphere/vclib/diskmanagers"
	"k8s.io/kubernetes/pkg/controller"
)

// VSphere Cloud Provider constants
const (
	ProviderName                  = "vsphere"
	VolDir                        = "kubevols"
	RoundTripperDefaultCount      = 3
	DummyVMPrefixName             = "vsphere-k8s"
	MacOuiVC                      = "00:50:56"
	MacOuiEsx                     = "00:0c:29"
	CleanUpDummyVMRoutineInterval = 5
	UUIDPath                      = "/sys/class/dmi/id/product_serial"
	UUIDPrefix                    = "VMware-"
)

var cleanUpRoutineInitialized = false
var datastoreFolderIDMap = make(map[string]map[string]string)

var cleanUpRoutineInitLock sync.Mutex
var cleanUpDummyVMLock sync.RWMutex

// VSphere is an implementation of cloud provider Interface for VSphere.
type VSphere struct {
	cfg      *VSphereConfig
	hostName string
	// Maps the VSphere IP address to VSphereInstance
	vsphereInstanceMap map[string]*VSphereInstance
	// Responsible for managing discovery of k8s node, their location etc.
	nodeManager *NodeManager
}

// Represents a vSphere instance where one or more kubernetes nodes are running.
type VSphereInstance struct {
	conn *vclib.VSphereConnection
	cfg  *VirtualCenterConfig
}

// Structure that represents Virtual Center configuration
type VirtualCenterConfig struct {
	// vCenter username.
	User string `gcfg:"user"`
	// vCenter password in clear text.
	Password string `gcfg:"password"`
	// vCenter port.
	VCenterPort string `gcfg:"port"`
	// Datacenter in which VMs are located.
	Datacenters string `gcfg:"datacenters"`
	// Soap round tripper count (retries = RoundTripper - 1)
	RoundTripperCount uint `gcfg:"soap-roundtrip-count"`
}

// Structure that represents the content of vsphere.conf file.
// Users specify the configuration of one or more Virtual Centers in vsphere.conf where
// the Kubernetes master and worker nodes are running.
type VSphereConfig struct {
	Global struct {
		// vCenter username.
		User string `gcfg:"user"`
		// vCenter password in clear text.
		Password string `gcfg:"password"`
		// Deprecated. Use VirtualCenter to specify multiple vCenter Servers.
		// vCenter IP.
		VCenterIP string `gcfg:"server"`
		// vCenter port.
		VCenterPort string `gcfg:"port"`
		// True if vCenter uses self-signed cert.
		InsecureFlag bool `gcfg:"insecure-flag"`
		// Datacenter in which VMs are located.
		// Deprecated. Use "datacenters" instead.
		Datacenter string `gcfg:"datacenter"`
		// Datacenter in which VMs are located.
		Datacenters string `gcfg:"datacenters"`
		// Datastore in which vmdks are stored.
		DefaultDatastore string `gcfg:"datastore"`
		// WorkingDir is path where VMs can be found. Also used to create dummy VMs.
		// Deprecated.
		WorkingDir string `gcfg:"working-dir"`
		// Soap round tripper count (retries = RoundTripper - 1)
		RoundTripperCount uint `gcfg:"soap-roundtrip-count"`
		// Deprecated as the virtual machines will be automatically discovered.
		// VMUUID is the VM Instance UUID of virtual machine which can be retrieved from instanceUuid
		// property in VmConfigInfo, or also set as vc.uuid in VMX file.
		// If not set, will be fetched from the machine via sysfs (requires root)
		VMUUID string `gcfg:"vm-uuid"`
		// Deprecated as virtual machine will be automatically discovered.
		// VMName is the VM name of virtual machine
		// Combining the WorkingDir and VMName can form a unique InstanceID.
		// When vm-name is set, no username/password is required on worker nodes.
		VMName string `gcfg:"vm-name"`
	}

	VirtualCenter map[string]*VirtualCenterConfig

	Network struct {
		// PublicNetwork is name of the network the VMs are joined to.
		PublicNetwork string `gcfg:"public-network"`
	}

	Disk struct {
		// SCSIControllerType defines SCSI controller to be used.
		SCSIControllerType string `dcfg:"scsicontrollertype"`
	}

	Workspace struct {
		VCenterIP  string `gcfg:"server"`
		Datacenter string `gcfg:"datacenter"`
		Folder     string `gcfg:"folder"`
	}
}

type Volumes interface {
	// AttachDisk attaches given disk to given node. Current node
	// is used when nodeName is empty string.
	AttachDisk(vmDiskPath string, storagePolicyID string, nodeName k8stypes.NodeName) (diskUUID string, err error)

	// DetachDisk detaches given disk to given node. Current node
	// is used when nodeName is empty string.
	// Assumption: If node doesn't exist, disk is already detached from node.
	DetachDisk(volPath string, nodeName k8stypes.NodeName) error

	// DiskIsAttached checks if a disk is attached to the given node.
	// Assumption: If node doesn't exist, disk is not attached to the node.
	DiskIsAttached(volPath string, nodeName k8stypes.NodeName) (bool, error)

	// DisksAreAttached checks if a list disks are attached to the given node.
	// Assumption: If node doesn't exist, disks are not attached to the node.
	DisksAreAttached(nodeVolumes map[k8stypes.NodeName][]string) (map[k8stypes.NodeName]map[string]bool, error)

	// CreateVolume creates a new vmdk with specified parameters.
	CreateVolume(volumeOptions *vclib.VolumeOptions) (volumePath string, err error)

	// DeleteVolume deletes vmdk.
	DeleteVolume(vmDiskPath string) error
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
	vclib.RegisterMetrics()
	cloudprovider.RegisterCloudProvider(ProviderName, func(config io.Reader) (cloudprovider.Interface, error) {
		// If vSphere.conf file is not present then it is worker node.
		if config == nil {
			return newWorkerNode()
		}
		cfg, err := readConfig(config)
		if err != nil {
			return nil, err
		}
		return newControllerNode(cfg)
	})
}

// Initialize passes a Kubernetes clientBuilder interface to the cloud provider
func (vs *VSphere) Initialize(clientBuilder controller.ControllerClientBuilder) {}

// Creates new worker node interface and returns
func newWorkerNode() (*VSphere, error) {
	var err error
	vs := VSphere{}
	vs.hostName, err = os.Hostname()
	if err != nil {
		glog.Errorf("Failed to get hostname. err: %+v", err)
		return nil, err
	}

	return &vs, nil
}

func populateVsphereInstanceMap(cfg *VSphereConfig) (map[string]*VSphereInstance, error) {
	vsphereInstanceMap := make(map[string]*VSphereInstance)

	// Check if the vsphere.conf is in old format. In this
	// format the cfg.VirtualCenter will be nil or empty.
	if cfg.VirtualCenter == nil || len(cfg.VirtualCenter) == 0 {
		glog.V(4).Infof("Config is not per virtual center and is in old format.")
		if cfg.Global.User == "" {
			glog.Error("Global.User is empty!")
			return nil, errors.New("Global.User is empty!")
		}
		if cfg.Global.Password == "" {
			glog.Error("Global.Password is empty!")
			return nil, errors.New("Global.Password is empty!")
		}
		if cfg.Global.WorkingDir == "" {
			glog.Error("Global.WorkingDir is empty!")
			return nil, errors.New("Global.WorkingDir is empty!")
		}
		if cfg.Global.VCenterIP == "" {
			glog.Error("Global.VCenterIP is empty!")
			return nil, errors.New("Global.VCenterIP is empty!")
		}
		if cfg.Global.Datacenter == "" {
			glog.Error("Global.Datacenter is empty!")
			return nil, errors.New("Global.Datacenter is empty!")
		}
		cfg.Workspace.VCenterIP = cfg.Global.VCenterIP
		cfg.Workspace.Datacenter = cfg.Global.Datacenter
		cfg.Workspace.Folder = cfg.Global.WorkingDir

		vcConfig := VirtualCenterConfig{
			User:              cfg.Global.User,
			Password:          cfg.Global.Password,
			VCenterPort:       cfg.Global.VCenterPort,
			Datacenters:       cfg.Global.Datacenter,
			RoundTripperCount: cfg.Global.RoundTripperCount,
		}

		vSphereConn := vclib.VSphereConnection{
			Username:          vcConfig.User,
			Password:          vcConfig.Password,
			Hostname:          cfg.Global.VCenterIP,
			Insecure:          cfg.Global.InsecureFlag,
			RoundTripperCount: vcConfig.RoundTripperCount,
			Port:              vcConfig.VCenterPort,
		}
		vsphereIns := VSphereInstance{
			conn: &vSphereConn,
			cfg:  &vcConfig,
		}
		// TODO: Remove this log as it will print VC credentials in log.
		glog.V(4).Infof("vcConfig for VC %s is %+v.", cfg.Global.VCenterIP, vcConfig)
		glog.V(4).Infof("vSphereConn for VC %s is %+v.", cfg.Global.VCenterIP, vSphereConn)

		vsphereInstanceMap[cfg.Global.VCenterIP] = &vsphereIns
	} else {
		// TODO: Remove this log
		glog.V(4).Infof("Workspace is %+v.", cfg.Workspace)
		if cfg.Workspace.VCenterIP == "" || cfg.Workspace.Folder == "" || cfg.Workspace.Datacenter == "" {
			msg := fmt.Sprintf("All fields in workspace are mandatory."+
				" vsphere.conf does not have the workspace specified correctly. cfg.Workspace: %+v", cfg.Workspace)
			glog.Error(msg)
			return nil, errors.New(msg)
		}
		for vcServer, vcConfig := range cfg.VirtualCenter {
			glog.V(4).Infof("Initializing vc server %s and vcConfig %+v", vcServer, vcConfig)
			if vcServer == "" {
				glog.Error("vsphere.conf does not have the VirtualCenter IP address specified")
				return nil, errors.New("vsphere.conf does not have the VirtualCenter IP address specified")
			}
			if vcConfig.User == "" {
				vcConfig.User = cfg.Global.User
			}
			if vcConfig.Password == "" {
				vcConfig.Password = cfg.Global.Password
			}
			if vcConfig.User == "" {
				msg := fmt.Sprintf("vcConfig.User is empty for vc %s!", vcServer)
				glog.Error(msg)
				return nil, errors.New(msg)
			}
			if vcConfig.Password == "" {
				msg := fmt.Sprintf("vcConfig.Password is empty for vc %s!", vcServer)
				glog.Error(msg)
				return nil, errors.New(msg)
			}
			if vcConfig.VCenterPort == "" {
				vcConfig.VCenterPort = cfg.Global.VCenterPort
			}
			if vcConfig.Datacenters == "" {
				if cfg.Global.Datacenters != "" {
					vcConfig.Datacenters = cfg.Global.Datacenters
				} else {
					// cfg.Global.Datacenter is deprecated, so giving it the last preference.
					vcConfig.Datacenters = cfg.Global.Datacenter
				}
			}
			if vcConfig.RoundTripperCount == 0 {
				vcConfig.RoundTripperCount = cfg.Global.RoundTripperCount
			}

			vSphereConn := vclib.VSphereConnection{
				Username:          vcConfig.User,
				Password:          vcConfig.Password,
				Hostname:          vcServer,
				Insecure:          cfg.Global.InsecureFlag,
				RoundTripperCount: vcConfig.RoundTripperCount,
				Port:              vcConfig.VCenterPort,
			}
			vsphereIns := VSphereInstance{
				conn: &vSphereConn,
				cfg:  vcConfig,
			}
			// TODO: Remove this log as it will print VC credentials in log.
			glog.V(4).Infof("vSphereConn for VC %s is %+v.", vcServer, vSphereConn)
			glog.V(4).Infof("vcConfig for VC %s is %+v.", vcServer, vcConfig)

			vsphereInstanceMap[vcServer] = &vsphereIns
		}
	}
	return vsphereInstanceMap, nil
}

// Creates new Contreoller node interface and returns
func newControllerNode(cfg VSphereConfig) (*VSphere, error) {
	var err error

	// TODO: Remove this log as it will print VC credentials in log.
	glog.V(4).Infof("VSphereConfig is %+v", cfg)
	if cfg.Disk.SCSIControllerType == "" {
		cfg.Disk.SCSIControllerType = vclib.PVSCSIControllerType
	} else if !vclib.CheckControllerSupported(cfg.Disk.SCSIControllerType) {
		glog.Errorf("%v is not a supported SCSI Controller type. Please configure 'lsilogic-sas' OR 'pvscsi'", cfg.Disk.SCSIControllerType)
		return nil, errors.New("Controller type not supported. Please configure 'lsilogic-sas' OR 'pvscsi'")
	}
	if cfg.Global.WorkingDir != "" {
		cfg.Global.WorkingDir = path.Clean(cfg.Global.WorkingDir)
	}
	if cfg.Global.RoundTripperCount == 0 {
		cfg.Global.RoundTripperCount = RoundTripperDefaultCount
	}
	if cfg.Global.VCenterPort == "" {
		cfg.Global.VCenterPort = "443"
	}
	if cfg.Global.VMUUID == "" {
		// This needs root privileges on the host, and will fail otherwise.
		cfg.Global.VMUUID, err = getvmUUID()
		if err != nil {
			glog.Errorf("Failed to get VM UUID. err: %+v", err)
			return nil, err
		}
	}
	vsphereInstanceMap, err := populateVsphereInstanceMap(&cfg)
	if err != nil {
		return nil, err
	}

	vs := VSphere{
		vsphereInstanceMap: vsphereInstanceMap,
		nodeManager: &NodeManager{
			vsphereInstanceMap: vsphereInstanceMap,
			nodeInfoMap: make(map[string]*NodeInfo),
			registeredNodes: make(map[string]*v1.Node),
		},
		cfg: &cfg,
	}
	// TODO: Remove this log as it will print VC credentials in log.
	glog.V(4).Infof("VSphereConfig after init is %+v", cfg)

	vs.hostName, err = os.Hostname()
	if err != nil {
		glog.Errorf("Failed to get hostname. err: %+v", err)
		return nil, err
	}
	runtime.SetFinalizer(&vs, logout)
	return &vs, nil
}

func logout(vs *VSphere) {
	for _, vsphereIns := range vs.vsphereInstanceMap {
		if vsphereIns.conn.GoVmomiClient != nil {
			vsphereIns.conn.GoVmomiClient.Logout(context.TODO())
		}
	}

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
						if strings.HasPrefix(i.HardwareAddr.String(), MacOuiVC) ||
							strings.HasPrefix(i.HardwareAddr.String(), MacOuiEsx) {
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

func (vs *VSphere) getVSphereInstance(nodeName k8stypes.NodeName) (*VSphereInstance, error) {
	vsphereIns, err := vs.nodeManager.GetVSphereInstance(nodeName)
	if err != nil {
		glog.Errorf("Cannot find node %q in cache. Node not found!!!", nodeName)
		return nil, errors.New(fmt.Sprintf("Cannot find node %q in vsphere configuration map", nodeName))
	}
	return &vsphereIns, nil
}

// Get the VM Managed Object instance by from the node
func (vs *VSphere) getVMByName(ctx context.Context, nodeName k8stypes.NodeName) (*vclib.VirtualMachine, error) {
	nodeInfo, err := vs.nodeManager.GetNodeInfo(nodeName)
	if err != nil {
		return nil, err
	}
	return nodeInfo.vm, nil
}

// NodeAddresses is an implementation of Instances.NodeAddresses.
func (vs *VSphere) NodeAddresses(nodeName k8stypes.NodeName) ([]v1.NodeAddress, error) {
	// Get local IP addresses if node is local node
	if vs.hostName == convertToString(nodeName) {
		return getLocalIP()
	}

	// TODO: Need to see what to do if nodename and localNodeName are not matching.
	return nil, cloudprovider.InstanceNotFound
}

// NodeAddressesByProviderID returns the node addresses of an instances with the specified unique providerID
// This method will not be called from the node that is requesting this ID. i.e. metadata service
// and other local methods cannot be used here
func (vs *VSphere) NodeAddressesByProviderID(providerID string) ([]v1.NodeAddress, error) {
	vmName := path.Base(providerID)
	return vs.NodeAddresses(convertToK8sType(vmName))
}

// AddSSHKeyToAllInstances add SSH key to all instances
func (vs *VSphere) AddSSHKeyToAllInstances(user string, keyData []byte) error {
	return cloudprovider.NotImplemented
}

// CurrentNodeName gives the current node name
func (vs *VSphere) CurrentNodeName(hostname string) (k8stypes.NodeName, error) {
	return convertToK8sType(vs.hostName), nil
}

func convertToString(nodeName k8stypes.NodeName) string {
	return string(nodeName)
}

func convertToK8sType(vmName string) k8stypes.NodeName {
	return k8stypes.NodeName(vmName)
}

// ExternalID returns the cloud provider ID of the node with the specified Name (deprecated).
func (vs *VSphere) ExternalID(nodeName k8stypes.NodeName) (string, error) {
	return vs.InstanceID(nodeName)
}

// InstanceExistsByProviderID returns true if the instance with the given provider id still exists and is running.
// If false is returned with no error, the instance will be immediately deleted by the cloud controller manager.
func (vs *VSphere) InstanceExistsByProviderID(providerID string) (bool, error) {
	vmName := path.Base(providerID)
	nodeName := vmNameToNodeName(vmName)
	// Create context
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	// Ensure client is logged in and session is valid
	err := vs.conn.Connect(ctx)
	if err != nil {
		return false, err
	}
	vm, err := vs.getVMByName(ctx, nodeName)
	if err != nil {
		if vclib.IsNotFound(err) {
			return false, nil
		}
		glog.Errorf("Failed to get VM object for node: %q. err: +%v", nodeNameToVMName(nodeName), err)
		return false, err
	}

	isActive, err := vm.IsActive(ctx)
	if err != nil {
		glog.Errorf("Failed to check whether node %q is active. err: %+v.", nodeNameToVMName(nodeName), err)
		return false, err
	}
	if !isActive {
		return false, nil
	}

	return true, nil
}

// InstanceID returns the cloud provider ID of the node with the specified Name.
func (vs *VSphere) InstanceID(nodeName k8stypes.NodeName) (string, error) {
	// TODO: Based on hostname, locate the VM in VC inventory and check for following cases.
	// 1. Node Present.
	// 2. Node Powered On/ Powered off.
	// Based on these kubernetes core decides to take specific actions.
	// Also verify if this logic is required only on master or also on worker nodes.
	if vs.hostName == convertToString(nodeName) {
		return vs.hostName, nil
	}

	// TODO: Need to see what to do if nodename and localNodeName are not matching.
	// return "", cloudprovider.InstanceNotFound

	// TODO: Below logic is the existing logic.
	//if vs.localInstanceID == convertToString(nodeName) {
	//	return vs.cfg.Global.WorkingDir + "/" + vs.localInstanceID, nil
	//}

	// Create context
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	vsi, err := vs.getVSphereInstance(nodeName)
	if err != nil {
		return "", err
	}
	// Ensure client is logged in and session is valid
	err = vsi.conn.Connect(ctx)
	if err != nil {
		return "", err
	}
	vm, err := vs.getVMByName(ctx, nodeName)
	if err != nil {
		if vclib.IsNotFound(err) {
			return "", cloudprovider.InstanceNotFound
		}
		glog.Errorf("Failed to get VM object for node: %q. err: +%v", convertToString(nodeName), err)
		return "", err
	}
	isActive, err := vm.IsActive(ctx)
	if err != nil {
		glog.Errorf("Failed to check whether node %q is active. err: %+v.", convertToString(nodeName), err)
		return "", err
	}
	if isActive {
		return "/" + vm.InventoryPath, nil
	}
	return "", fmt.Errorf("The node %q is not active", convertToString(nodeName))
}

// InstanceTypeByProviderID returns the cloudprovider instance type of the node with the specified unique providerID
// This method will not be called from the node that is requesting this ID. i.e. metadata service
// and other local methods cannot be used here
func (vs *VSphere) InstanceTypeByProviderID(providerID string) (string, error) {
	return "", nil
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

// AttachDisk attaches given virtual disk volume to the compute running kubelet.
func (vs *VSphere) AttachDisk(vmDiskPath string, storagePolicyID string, nodeName k8stypes.NodeName) (diskUUID string, err error) {
	attachDiskInternal := func(vmDiskPath string, storagePolicyID string, nodeName k8stypes.NodeName) (diskUUID string, err error) {
		if nodeName == "" {
			nodeName = convertToK8sType(vs.hostName)
		}
		// Create context
		ctx, cancel := context.WithCancel(context.Background())
		defer cancel()
		vsi, err := vs.getVSphereInstance(nodeName)
		if err != nil {
			return "", err
		}
		// Ensure client is logged in and session is valid
		err = vsi.conn.Connect(ctx)
		if err != nil {
			return "", err
		}
		vm, err := vs.getVMByName(ctx, nodeName)
		if err != nil {
			glog.Errorf("Failed to get VM object for node: %q. err: +%v", convertToString(nodeName), err)
			return "", err
		}
		diskUUID, err = vm.AttachDisk(ctx, vmDiskPath, &vclib.VolumeOptions{SCSIControllerType: vclib.PVSCSIControllerType, StoragePolicyID: storagePolicyID})
		if err != nil {
			glog.Errorf("Failed to attach disk: %s for node: %s. err: +%v", vmDiskPath, convertToString(nodeName), err)
			return "", err
		}
		return diskUUID, nil
	}
	requestTime := time.Now()
	diskUUID, err = attachDiskInternal(vmDiskPath, storagePolicyID, nodeName)
	vclib.RecordvSphereMetric(vclib.OperationAttachVolume, requestTime, err)
	return diskUUID, err
}

// DetachDisk detaches given virtual disk volume from the compute running kubelet.
func (vs *VSphere) DetachDisk(volPath string, nodeName k8stypes.NodeName) error {
	detachDiskInternal := func(volPath string, nodeName k8stypes.NodeName) error {
		if nodeName == "" {
			nodeName = convertToK8sType(vs.hostName)
		}
		// Create context
		ctx, cancel := context.WithCancel(context.Background())
		defer cancel()
		vsi, err := vs.getVSphereInstance(nodeName)
		if err != nil {
			return err
		}
		// Ensure client is logged in and session is valid
		err = vsi.conn.Connect(ctx)
		if err != nil {
			return err
		}
		vm, err := vs.getVMByName(ctx, nodeName)
		if err != nil {
			// If node doesn't exist, disk is already detached from node.
			if vclib.IsNotFound(err) {
				glog.Infof("Node %q does not exist, disk %s is already detached from node.", convertToString(nodeName), volPath)
				return nil
			}

			glog.Errorf("Failed to get VM object for node: %q. err: +%v", convertToString(nodeName), err)
			return err
		}
		err = vm.DetachDisk(ctx, volPath)
		if err != nil {
			glog.Errorf("Failed to detach disk: %s for node: %s. err: +%v", volPath, convertToString(nodeName), err)
			return err
		}
		return nil
	}
	requestTime := time.Now()
	err := detachDiskInternal(volPath, nodeName)
	vclib.RecordvSphereMetric(vclib.OperationDetachVolume, requestTime, nil)
	return err
}

// DiskIsAttached returns if disk is attached to the VM using controllers supported by the plugin.
func (vs *VSphere) DiskIsAttached(volPath string, nodeName k8stypes.NodeName) (bool, error) {
	diskIsAttachedInternal := func(volPath string, nodeName k8stypes.NodeName) (bool, error) {
		var vSphereInstance string
		if nodeName == "" {
			vSphereInstance = vs.hostName
			nodeName = convertToK8sType(vSphereInstance)
		} else {
			vSphereInstance = convertToString(nodeName)
		}
		// Create context
		ctx, cancel := context.WithCancel(context.Background())
		defer cancel()
		vsi, err := vs.getVSphereInstance(nodeName)
		if err != nil {
			return false, err
		}
		// Ensure client is logged in and session is valid
		err = vsi.conn.Connect(ctx)
		if err != nil {
			return false, err
		}
		vm, err := vs.getVMByName(ctx, nodeName)
		if err != nil {
			if vclib.IsNotFound(err) {
				glog.Warningf("Node %q does not exist, vsphere CP will assume disk %v is not attached to it.", nodeName, volPath)
				// make the disk as detached and return false without error.
				return false, nil
			}
			glog.Errorf("Failed to get VM object for node: %q. err: +%v", vSphereInstance, err)
			return false, err
		}
		volPath = vclib.RemoveClusterFromVDiskPath(volPath)
		attached, err := vm.IsDiskAttached(ctx, volPath)
		if err != nil {
			glog.Errorf("DiskIsAttached failed to determine whether disk %q is still attached on node %q",
				volPath,
				vSphereInstance)
		}
		return attached, err
	}
	requestTime := time.Now()
	isAttached, err := diskIsAttachedInternal(volPath, nodeName)
	vclib.RecordvSphereMetric(vclib.OperationDiskIsAttached, requestTime, err)
	return isAttached, err
}

// DisksAreAttached returns if disks are attached to the VM using controllers supported by the plugin.
func (vs *VSphere) DisksAreAttached(nodeVolumes map[k8stypes.NodeName][]string) (map[k8stypes.NodeName]map[string]bool, error) {
	disksAreAttachedInternal := func(nodeVolumes map[k8stypes.NodeName][]string) (map[k8stypes.NodeName]map[string]bool, error) {
		attached := make(map[k8stypes.NodeName]map[string]bool)
		if len(nodeVolumes) == 0 {
			return attached, nil
		}
		// Create context
		ctx, cancel := context.WithCancel(context.Background())
		defer cancel()
		vsi, err := vs.getVSphereInstance(nodeName)
		if err != nil {
			return nil, err
		}
		// Ensure client is logged in and session is valid
		err = vsi.conn.Connect(ctx)
		if err != nil {
			return nil, err
		}
		dc, err := vclib.GetDatacenter(ctx, vs.conn, vs.cfg.Global.Datacenter)
		if err != nil {
			return nil, err
		}

		vmVolumes := make(map[string][]string)
		for nodeName, volPaths := range nodeVolumes {
			for i, volPath := range volPaths {
				volPath = vclib.RemoveClusterFromVDiskPath(volPath)
				// Get the canonical volume path for volPath.
				canonicalVolumePath, err := getcanonicalVolumePath(ctx, dc, volPath)
				if err != nil {
					glog.Errorf("Failed to get canonical vsphere volume path for volume: %s. err: %+v", volPath, err)
					return nil, err
				}
				// Check if the volume path contains .vmdk extension. If not, add the extension and update the nodeVolumes Map
				if len(canonicalVolumePath) > 0 && filepath.Ext(canonicalVolumePath) != ".vmdk" {
					canonicalVolumePath += ".vmdk"
				}
				volPaths[i] = canonicalVolumePath
			}
			vmVolumes[nodeNameToVMName(nodeName)] = volPaths
		}
		// Check if the disks are attached to their respective nodes
		disksAttachedList, err := dc.CheckDisksAttached(ctx, vmVolumes)
		if err != nil {
			return nil, err
		}
		for vmName, volPaths := range disksAttachedList {
			attached[vmNameToNodeName(vmName)] = volPaths
		}
		return attached, nil
	}
	requestTime := time.Now()
	attached, err := disksAreAttachedInternal(nodeVolumes)
	vclib.RecordvSphereMetric(vclib.OperationDisksAreAttached, requestTime, err)
	return attached, err
}

// CreateVolume creates a volume of given size (in KiB) and return the volume path.
// If the volumeOptions.Datastore is part of datastore cluster for example - [DatastoreCluster/sharedVmfs-0] then
// return value will be [DatastoreCluster/sharedVmfs-0] kubevols/<volume-name>.vmdk
// else return value will be [sharedVmfs-0] kubevols/<volume-name>.vmdk
func (vs *VSphere) CreateVolume(volumeOptions *vclib.VolumeOptions) (canonicalVolumePath string, err error) {
	glog.V(1).Infof("Starting to create a vSphere volume with volumeOptions: %+v", volumeOptions)
	createVolumeInternal := func(volumeOptions *vclib.VolumeOptions) (canonicalVolumePath string, err error) {
		var datastore string
		// Default datastore is the datastore in the vSphere config file that is used to initialize vSphere cloud provider.
		if volumeOptions.Datastore == "" {
			datastore = vs.cfg.Global.DefaultDatastore
		} else {
			datastore = volumeOptions.Datastore
		}
		datastore = strings.TrimSpace(datastore)
		// Create context
		ctx, cancel := context.WithCancel(context.Background())
		defer cancel()
		vsi, err := vs.getVSphereInstance(convertToK8sType(vs.hostName))
		if err != nil {
			return "", err
		}
		// Ensure client is logged in and session is valid
		err = vsi.conn.Connect(ctx)
		if err != nil {
			return "", err
		}
		dc, err := vclib.GetDatacenter(ctx, vsi.conn, vs.cfg.Global.Datacenter)
		if err != nil {
			return "", err
		}
		var vmOptions *vclib.VMOptions
		if volumeOptions.VSANStorageProfileData != "" || volumeOptions.StoragePolicyName != "" {
			// Acquire a read lock to ensure multiple PVC requests can be processed simultaneously.
			cleanUpDummyVMLock.RLock()
			defer cleanUpDummyVMLock.RUnlock()
			// Create a new background routine that will delete any dummy VM's that are left stale.
			// This routine will get executed for every 5 minutes and gets initiated only once in its entire lifetime.
			cleanUpRoutineInitLock.Lock()
			if !cleanUpRoutineInitialized {
				glog.V(1).Infof("Starting a clean up routine to remove stale dummy VM's")
				go vs.cleanUpDummyVMs(DummyVMPrefixName)
				cleanUpRoutineInitialized = true
			}
			cleanUpRoutineInitLock.Unlock()
			vmOptions, err = vs.setVMOptions(ctx, dc)
			if err != nil {
				glog.Errorf("Failed to set VM options requires to create a vsphere volume. err: %+v", err)
				return "", err
			}
		}
		if volumeOptions.StoragePolicyName != "" && volumeOptions.Datastore == "" {
			datastore, err = getPbmCompatibleDatastore(ctx, dc.Client(), volumeOptions.StoragePolicyName, vmOptions.VMFolder)
			if err != nil {
				glog.Errorf("Failed to get pbm compatible datastore with storagePolicy: %s. err: %+v", volumeOptions.StoragePolicyName, err)
				return "", err
			}
		}
		ds, err := dc.GetDatastoreByName(ctx, datastore)
		if err != nil {
			return "", err
		}
		volumeOptions.Datastore = datastore
		kubeVolsPath := filepath.Clean(ds.Path(VolDir)) + "/"
		err = ds.CreateDirectory(ctx, kubeVolsPath, false)
		if err != nil && err != vclib.ErrFileAlreadyExist {
			glog.Errorf("Cannot create dir %#v. err %s", kubeVolsPath, err)
			return "", err
		}
		volumePath := kubeVolsPath + volumeOptions.Name + ".vmdk"
		disk := diskmanagers.VirtualDisk{
			DiskPath:      volumePath,
			VolumeOptions: volumeOptions,
			VMOptions:     vmOptions,
		}
		volumePath, err = disk.Create(ctx, ds)
		if err != nil {
			glog.Errorf("Failed to create a vsphere volume with volumeOptions: %+v on datastore: %s. err: %+v", volumeOptions, datastore, err)
			return "", err
		}
		// Get the canonical path for the volume path.
		canonicalVolumePath, err = getcanonicalVolumePath(ctx, dc, volumePath)
		if err != nil {
			glog.Errorf("Failed to get canonical vsphere volume path for volume: %s with volumeOptions: %+v on datastore: %s. err: %+v", volumePath, volumeOptions, datastore, err)
			return "", err
		}
		if filepath.Base(datastore) != datastore {
			// If datastore is within cluster, add cluster path to the volumePath
			canonicalVolumePath = strings.Replace(canonicalVolumePath, filepath.Base(datastore), datastore, 1)
		}
		return canonicalVolumePath, nil
	}
	requestTime := time.Now()
	canonicalVolumePath, err = createVolumeInternal(volumeOptions)
	vclib.RecordCreateVolumeMetric(volumeOptions, requestTime, err)
	glog.V(1).Infof("The canonical volume path for the newly created vSphere volume is %q", canonicalVolumePath)
	return canonicalVolumePath, err
}

// DeleteVolume deletes a volume given volume name.
func (vs *VSphere) DeleteVolume(vmDiskPath string) error {
	glog.V(1).Infof("Starting to delete vSphere volume with vmDiskPath: %s", vmDiskPath)
	deleteVolumeInternal := func(vmDiskPath string) error {
		// Create context
		ctx, cancel := context.WithCancel(context.Background())
		defer cancel()
		vsi, err := vs.getVSphereInstance(convertToK8sType(vs.hostName))
		if err != nil {
			return err
		}
		// Ensure client is logged in and session is valid
		err = vsi.conn.Connect(ctx)
		if err != nil {
			return err
		}
		dc, err := vclib.GetDatacenter(ctx, vsi.conn, vs.cfg.Global.Datacenter)
		if err != nil {
			return err
		}
		ds, err := dc.GetDatastoreByName(ctx, vs.cfg.Global.DefaultDatastore)
		if err != nil {
			return err
		}
		disk := diskmanagers.VirtualDisk{
			DiskPath:      vmDiskPath,
			VolumeOptions: &vclib.VolumeOptions{},
			VMOptions:     &vclib.VMOptions{},
		}
		err = disk.Delete(ctx, ds)
		if err != nil {
			glog.Errorf("Failed to delete vsphere volume with vmDiskPath: %s. err: %+v", vmDiskPath, err)
		}
		return err
	}
	requestTime := time.Now()
	err := deleteVolumeInternal(vmDiskPath)
	vclib.RecordvSphereMetric(vclib.OperationDeleteVolume, requestTime, err)
	return err
}

// HasClusterID returns true if the cluster has a clusterID
func (vs *VSphere) HasClusterID() bool {
	return true
}

// Notification handler when node is registered.
func (vs *VSphere) NodeRegistered(node *v1.Node) {
	glog.V(4).Infof("Node Registered: %+v", node)
	vs.nodeManager.RegisterNode(node)
}

// Notification handler when node is unregistered.
func (vs *VSphere) NodeUnregistered(node *v1.Node) {
	glog.V(4).Infof("Node Unregistered: %+v", node)
	vs.nodeManager.UnRegisterNode(node)
}
