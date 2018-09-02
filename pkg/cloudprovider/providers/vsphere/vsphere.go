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
	"context"
	"errors"
	"fmt"
	"io"
	"net"
	"net/url"
	"os"
	"path"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"time"

	"gopkg.in/gcfg.v1"

	"github.com/golang/glog"
	"github.com/vmware/govmomi/vapi/rest"
	"github.com/vmware/govmomi/vapi/tags"
	"github.com/vmware/govmomi/vim25/mo"
	"k8s.io/api/core/v1"
	k8stypes "k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/informers"
	"k8s.io/client-go/tools/cache"
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
)

var cleanUpRoutineInitialized = false
var datastoreFolderIDMap = make(map[string]map[string]string)

var cleanUpRoutineInitLock sync.Mutex
var cleanUpDummyVMLock sync.RWMutex

// Error Messages
const (
	MissingUsernameErrMsg = "Username is missing"
	MissingPasswordErrMsg = "Password is missing"
)

// Error constants
var (
	ErrUsernameMissing = errors.New(MissingUsernameErrMsg)
	ErrPasswordMissing = errors.New(MissingPasswordErrMsg)
)

// VSphere is an implementation of cloud provider Interface for VSphere.
type VSphere struct {
	cfg      *VSphereConfig
	hostName string
	// Maps the VSphere IP address to VSphereInstance
	vsphereInstanceMap map[string]*VSphereInstance
	// Responsible for managing discovery of k8s node, their location etc.
	nodeManager          *NodeManager
	vmUUID               string
	isSecretInfoProvided bool
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
	// Thumbprint of the VCenter's certificate thumbprint
	Thumbprint string `gcfg:"thumbprint"`
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
		// Specifies the path to a CA certificate in PEM format. Optional; if not
		// configured, the system's CA certificates will be used.
		CAFile string `gcfg:"ca-file"`
		// Thumbprint of the VCenter's certificate thumbprint
		Thumbprint string `gcfg:"thumbprint"`
		// Datacenter in which VMs are located.
		// Deprecated. Use "datacenters" instead.
		Datacenter string `gcfg:"datacenter"`
		// Datacenter in which VMs are located.
		Datacenters string `gcfg:"datacenters"`
		// Datastore in which vmdks are stored.
		// Deprecated. See Workspace.DefaultDatastore
		DefaultDatastore string `gcfg:"datastore"`
		// WorkingDir is path where VMs can be found. Also used to create dummy VMs.
		// Deprecated.
		WorkingDir string `gcfg:"working-dir"`
		// Soap round tripper count (retries = RoundTripper - 1)
		RoundTripperCount uint `gcfg:"soap-roundtrip-count"`
		// Is required on the controller-manager if it does not run on a VMware machine
		// VMUUID is the VM Instance UUID of virtual machine which can be retrieved from instanceUuid
		// property in VmConfigInfo, or also set as vc.uuid in VMX file.
		// If not set, will be fetched from the machine via sysfs (requires root)
		VMUUID string `gcfg:"vm-uuid"`
		// Deprecated as virtual machine will be automatically discovered.
		// VMName is the VM name of virtual machine
		// Combining the WorkingDir and VMName can form a unique InstanceID.
		// When vm-name is set, no username/password is required on worker nodes.
		VMName string `gcfg:"vm-name"`
		// Name of the secret were vCenter credentials are present.
		SecretName string `gcfg:"secret-name"`
		// Secret Namespace where secret will be present that has vCenter credentials.
		SecretNamespace string `gcfg:"secret-namespace"`
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

	// Endpoint used to create volumes
	Workspace struct {
		VCenterIP        string `gcfg:"server"`
		Datacenter       string `gcfg:"datacenter"`
		Folder           string `gcfg:"folder"`
		DefaultDatastore string `gcfg:"default-datastore"`
		ResourcePoolPath string `gcfg:"resourcepool-path"`
	}

	// Tag categories and tags which correspond to "built-in node labels: zones and region"
	Labels struct {
		Zone   string `gcfg:"zone"`
		Region string `gcfg:"region"`
	}
}

type Volumes interface {
	// AttachDisk attaches given disk to given node. Current node
	// is used when nodeName is empty string.
	AttachDisk(vmDiskPath string, storagePolicyName string, nodeName k8stypes.NodeName) (diskUUID string, err error)

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
func (vs *VSphere) Initialize(clientBuilder controller.ControllerClientBuilder) {
}

// Initialize Node Informers
func (vs *VSphere) SetInformers(informerFactory informers.SharedInformerFactory) {
	if vs.cfg == nil {
		return
	}

	if vs.isSecretInfoProvided {
		secretCredentialManager := &SecretCredentialManager{
			SecretName:      vs.cfg.Global.SecretName,
			SecretNamespace: vs.cfg.Global.SecretNamespace,
			SecretLister:    informerFactory.Core().V1().Secrets().Lister(),
			Cache: &SecretCache{
				VirtualCenter: make(map[string]*Credential),
			},
		}
		vs.nodeManager.UpdateCredentialManager(secretCredentialManager)
	}

	// Only on controller node it is required to register listeners.
	// Register callbacks for node updates
	glog.V(4).Infof("Setting up node informers for vSphere Cloud Provider")
	nodeInformer := informerFactory.Core().V1().Nodes().Informer()
	nodeInformer.AddEventHandler(cache.ResourceEventHandlerFuncs{
		AddFunc:    vs.NodeAdded,
		DeleteFunc: vs.NodeDeleted,
	})
	glog.V(4).Infof("Node informers in vSphere cloud provider initialized")

}

// Creates new worker node interface and returns
func newWorkerNode() (*VSphere, error) {
	var err error
	vs := VSphere{}
	vs.hostName, err = os.Hostname()
	if err != nil {
		glog.Errorf("Failed to get hostname. err: %+v", err)
		return nil, err
	}
	vs.vmUUID, err = GetVMUUID()
	if err != nil {
		glog.Errorf("Failed to get uuid. err: %+v", err)
		return nil, err
	}
	return &vs, nil
}

func populateVsphereInstanceMap(cfg *VSphereConfig) (map[string]*VSphereInstance, error) {
	vsphereInstanceMap := make(map[string]*VSphereInstance)
	isSecretInfoProvided := true

	if cfg.Global.SecretName == "" || cfg.Global.SecretNamespace == "" {
		glog.Warningf("SecretName and/or SecretNamespace is not provided. " +
			"VCP will use username and password from config file")
		isSecretInfoProvided = false
	}

	if isSecretInfoProvided {
		if cfg.Global.User != "" {
			glog.Warning("Global.User and Secret info provided. VCP will use secret to get credentials")
			cfg.Global.User = ""
		}
		if cfg.Global.Password != "" {
			glog.Warning("Global.Password and Secret info provided. VCP will use secret to get credentials")
			cfg.Global.Password = ""
		}
	}

	// Check if the vsphere.conf is in old format. In this
	// format the cfg.VirtualCenter will be nil or empty.
	if cfg.VirtualCenter == nil || len(cfg.VirtualCenter) == 0 {
		glog.V(4).Infof("Config is not per virtual center and is in old format.")
		if !isSecretInfoProvided {
			if cfg.Global.User == "" {
				glog.Error("Global.User is empty!")
				return nil, ErrUsernameMissing
			}
			if cfg.Global.Password == "" {
				glog.Error("Global.Password is empty!")
				return nil, ErrPasswordMissing
			}
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
		cfg.Workspace.DefaultDatastore = cfg.Global.DefaultDatastore

		vcConfig := VirtualCenterConfig{
			User:              cfg.Global.User,
			Password:          cfg.Global.Password,
			VCenterPort:       cfg.Global.VCenterPort,
			Datacenters:       cfg.Global.Datacenter,
			RoundTripperCount: cfg.Global.RoundTripperCount,
			Thumbprint:        cfg.Global.Thumbprint,
		}

		// Note: If secrets info is provided username and password will be populated
		// once secret is created.
		vSphereConn := vclib.VSphereConnection{
			Username:          vcConfig.User,
			Password:          vcConfig.Password,
			Hostname:          cfg.Global.VCenterIP,
			Insecure:          cfg.Global.InsecureFlag,
			RoundTripperCount: vcConfig.RoundTripperCount,
			Port:              vcConfig.VCenterPort,
			CACert:            cfg.Global.CAFile,
			Thumbprint:        cfg.Global.Thumbprint,
		}

		vsphereIns := VSphereInstance{
			conn: &vSphereConn,
			cfg:  &vcConfig,
		}
		vsphereInstanceMap[cfg.Global.VCenterIP] = &vsphereIns
	} else {
		if cfg.Workspace.VCenterIP == "" || cfg.Workspace.Folder == "" || cfg.Workspace.Datacenter == "" {
			msg := fmt.Sprintf("All fields in workspace are mandatory."+
				" vsphere.conf does not have the workspace specified correctly. cfg.Workspace: %+v", cfg.Workspace)
			glog.Error(msg)
			return nil, errors.New(msg)
		}

		for vcServer, vcConfig := range cfg.VirtualCenter {
			glog.V(4).Infof("Initializing vc server %s", vcServer)
			if vcServer == "" {
				glog.Error("vsphere.conf does not have the VirtualCenter IP address specified")
				return nil, errors.New("vsphere.conf does not have the VirtualCenter IP address specified")
			}

			if !isSecretInfoProvided {
				if vcConfig.User == "" {
					vcConfig.User = cfg.Global.User
					if vcConfig.User == "" {
						glog.Errorf("vcConfig.User is empty for vc %s!", vcServer)
						return nil, ErrUsernameMissing
					}
				}
				if vcConfig.Password == "" {
					vcConfig.Password = cfg.Global.Password
					if vcConfig.Password == "" {
						glog.Errorf("vcConfig.Password is empty for vc %s!", vcServer)
						return nil, ErrPasswordMissing
					}
				}
			} else {
				if vcConfig.User != "" {
					glog.Warningf("vcConfig.User for server %s and Secret info provided. VCP will use secret to get credentials", vcServer)
					vcConfig.User = ""
				}
				if vcConfig.Password != "" {
					glog.Warningf("vcConfig.Password for server %s and Secret info provided. VCP will use secret to get credentials", vcServer)
					vcConfig.Password = ""
				}
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

			// Note: If secrets info is provided username and password will be populated
			// once secret is created.
			vSphereConn := vclib.VSphereConnection{
				Username:          vcConfig.User,
				Password:          vcConfig.Password,
				Hostname:          vcServer,
				Insecure:          cfg.Global.InsecureFlag,
				RoundTripperCount: vcConfig.RoundTripperCount,
				Port:              vcConfig.VCenterPort,
				CACert:            cfg.Global.CAFile,
				Thumbprint:        vcConfig.Thumbprint,
			}
			vsphereIns := VSphereInstance{
				conn: &vSphereConn,
				cfg:  vcConfig,
			}
			vsphereInstanceMap[vcServer] = &vsphereIns
		}
	}
	return vsphereInstanceMap, nil
}

// getVMUUID allows tests to override GetVMUUID
var getVMUUID = GetVMUUID

// Creates new Controller node interface and returns
func newControllerNode(cfg VSphereConfig) (*VSphere, error) {
	vs, err := buildVSphereFromConfig(cfg)
	if err != nil {
		return nil, err
	}
	vs.hostName, err = os.Hostname()
	if err != nil {
		glog.Errorf("Failed to get hostname. err: %+v", err)
		return nil, err
	}
	if cfg.Global.VMUUID != "" {
		vs.vmUUID = cfg.Global.VMUUID
	} else {
		vs.vmUUID, err = getVMUUID()
		if err != nil {
			glog.Errorf("Failed to get uuid. err: %+v", err)
			return nil, err
		}
	}
	runtime.SetFinalizer(vs, logout)
	return vs, nil
}

// Initializes vSphere from vSphere CloudProvider Configuration
func buildVSphereFromConfig(cfg VSphereConfig) (*VSphere, error) {
	isSecretInfoProvided := false
	if cfg.Global.SecretName != "" && cfg.Global.SecretNamespace != "" {
		isSecretInfoProvided = true
	}

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
	vsphereInstanceMap, err := populateVsphereInstanceMap(&cfg)
	if err != nil {
		return nil, err
	}

	vs := VSphere{
		vsphereInstanceMap: vsphereInstanceMap,
		nodeManager: &NodeManager{
			vsphereInstanceMap: vsphereInstanceMap,
			nodeInfoMap:        make(map[string]*NodeInfo),
			registeredNodes:    make(map[string]*v1.Node),
		},
		isSecretInfoProvided: isSecretInfoProvided,
		cfg:                  &cfg,
	}
	return &vs, nil
}

func logout(vs *VSphere) {
	for _, vsphereIns := range vs.vsphereInstanceMap {
		vsphereIns.conn.Logout(context.TODO())
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
		return nil, err
	}
	return &vsphereIns, nil
}

func (vs *VSphere) getVSphereInstanceForServer(vcServer string, ctx context.Context) (*VSphereInstance, error) {
	vsphereIns, ok := vs.vsphereInstanceMap[vcServer]
	if !ok {
		glog.Errorf("cannot find vcServer %q in cache. VC not found!!!", vcServer)
		return nil, errors.New(fmt.Sprintf("Cannot find node %q in vsphere configuration map", vcServer))
	}
	// Ensure client is logged in and session is valid
	err := vs.nodeManager.vcConnect(ctx, vsphereIns)
	if err != nil {
		glog.Errorf("failed connecting to vcServer %q with error %+v", vcServer, err)
		return nil, err
	}

	return vsphereIns, nil
}

// Get the VM Managed Object instance by from the node
func (vs *VSphere) getVMFromNodeName(ctx context.Context, nodeName k8stypes.NodeName) (*vclib.VirtualMachine, error) {
	nodeInfo, err := vs.nodeManager.GetNodeInfo(nodeName)
	if err != nil {
		return nil, err
	}
	return nodeInfo.vm, nil
}

// NodeAddresses is an implementation of Instances.NodeAddresses.
func (vs *VSphere) NodeAddresses(ctx context.Context, nodeName k8stypes.NodeName) ([]v1.NodeAddress, error) {
	// Get local IP addresses if node is local node
	if vs.hostName == convertToString(nodeName) {
		addrs, err := getLocalIP()
		if err != nil {
			return nil, err
		}
		// add the hostname address
		v1helper.AddToNodeAddresses(&addrs, v1.NodeAddress{Type: v1.NodeHostName, Address: vs.hostName})
		return addrs, nil
	}

	if vs.cfg == nil {
		return nil, cloudprovider.InstanceNotFound
	}

	// Below logic can be executed only on master as VC details are present.
	addrs := []v1.NodeAddress{}
	// Create context
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	vsi, err := vs.getVSphereInstance(nodeName)
	if err != nil {
		return nil, err
	}
	// Ensure client is logged in and session is valid
	err = vs.nodeManager.vcConnect(ctx, vsi)
	if err != nil {
		return nil, err
	}

	vm, err := vs.getVMFromNodeName(ctx, nodeName)
	if err != nil {
		glog.Errorf("Failed to get VM object for node: %q. err: +%v", convertToString(nodeName), err)
		return nil, err
	}
	vmMoList, err := vm.Datacenter.GetVMMoList(ctx, []*vclib.VirtualMachine{vm}, []string{"guest.net"})
	if err != nil {
		glog.Errorf("Failed to get VM Managed object with property guest.net for node: %q. err: +%v", convertToString(nodeName), err)
		return nil, err
	}
	// retrieve VM's ip(s)
	for _, v := range vmMoList[0].Guest.Net {
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
func (vs *VSphere) NodeAddressesByProviderID(ctx context.Context, providerID string) ([]v1.NodeAddress, error) {
	return vs.NodeAddresses(ctx, convertToK8sType(providerID))
}

// AddSSHKeyToAllInstances add SSH key to all instances
func (vs *VSphere) AddSSHKeyToAllInstances(ctx context.Context, user string, keyData []byte) error {
	return cloudprovider.NotImplemented
}

// CurrentNodeName gives the current node name
func (vs *VSphere) CurrentNodeName(ctx context.Context, hostname string) (k8stypes.NodeName, error) {
	return convertToK8sType(vs.hostName), nil
}

func convertToString(nodeName k8stypes.NodeName) string {
	return string(nodeName)
}

func convertToK8sType(vmName string) k8stypes.NodeName {
	return k8stypes.NodeName(vmName)
}

// InstanceExistsByProviderID returns true if the instance with the given provider id still exists and is running.
// If false is returned with no error, the instance will be immediately deleted by the cloud controller manager.
func (vs *VSphere) InstanceExistsByProviderID(ctx context.Context, providerID string) (bool, error) {
	nodeName, err := vs.GetNodeNameFromProviderID(providerID)
	if err != nil {
		glog.Errorf("Error while getting nodename for providerID %s", providerID)
		return false, err
	}
	_, err = vs.InstanceID(ctx, convertToK8sType(nodeName))
	if err == nil {
		return true, nil
	}

	return false, err
}

// InstanceShutdownByProviderID returns true if the instance is in safe state to detach volumes
func (vs *VSphere) InstanceShutdownByProviderID(ctx context.Context, providerID string) (bool, error) {
	nodeName, err := vs.GetNodeNameFromProviderID(providerID)
	if err != nil {
		glog.Errorf("Error while getting nodename for providerID %s", providerID)
		return false, err
	}

	vsi, err := vs.getVSphereInstance(convertToK8sType(nodeName))
	if err != nil {
		return false, err
	}
	// Ensure client is logged in and session is valid
	if err := vs.nodeManager.vcConnect(ctx, vsi); err != nil {
		return false, err
	}
	vm, err := vs.getVMFromNodeName(ctx, convertToK8sType(nodeName))
	if err != nil {
		glog.Errorf("Failed to get VM object for node: %q. err: +%v", nodeName, err)
		return false, err
	}
	isActive, err := vm.IsActive(ctx)
	if err != nil {
		glog.Errorf("Failed to check whether node %q is active. err: %+v.", nodeName, err)
		return false, err
	}
	return !isActive, nil
}

// InstanceID returns the cloud provider ID of the node with the specified Name.
func (vs *VSphere) InstanceID(ctx context.Context, nodeName k8stypes.NodeName) (string, error) {

	instanceIDInternal := func() (string, error) {
		if vs.hostName == convertToString(nodeName) {
			return vs.vmUUID, nil
		}

		// Below logic can be performed only on master node where VC details are preset.
		if vs.cfg == nil {
			return "", fmt.Errorf("The current node can't detremine InstanceID for %q", convertToString(nodeName))
		}

		// Create context
		ctx, cancel := context.WithCancel(context.Background())
		defer cancel()
		vsi, err := vs.getVSphereInstance(nodeName)
		if err != nil {
			return "", err
		}
		// Ensure client is logged in and session is valid
		err = vs.nodeManager.vcConnect(ctx, vsi)
		if err != nil {
			return "", err
		}
		vm, err := vs.getVMFromNodeName(ctx, nodeName)
		if err != nil {
			glog.Errorf("Failed to get VM object for node: %q. err: +%v", convertToString(nodeName), err)
			return "", err
		}
		isActive, err := vm.IsActive(ctx)
		if err != nil {
			glog.Errorf("Failed to check whether node %q is active. err: %+v.", convertToString(nodeName), err)
			return "", err
		}
		if isActive {
			return vs.vmUUID, nil
		}
		glog.Warningf("The VM: %s is not in %s state", convertToString(nodeName), vclib.ActivePowerState)
		return "", cloudprovider.InstanceNotFound
	}

	instanceID, err := instanceIDInternal()
	if err != nil {
		if vclib.IsManagedObjectNotFoundError(err) {
			err = vs.nodeManager.RediscoverNode(nodeName)
			if err == nil {
				glog.V(4).Infof("InstanceID: Found node %q", convertToString(nodeName))
				instanceID, err = instanceIDInternal()
			} else if err == vclib.ErrNoVMFound {
				return "", cloudprovider.InstanceNotFound
			}
		}
	}

	return instanceID, err
}

// InstanceTypeByProviderID returns the cloudprovider instance type of the node with the specified unique providerID
// This method will not be called from the node that is requesting this ID. i.e. metadata service
// and other local methods cannot be used here
func (vs *VSphere) InstanceTypeByProviderID(ctx context.Context, providerID string) (string, error) {
	return "", nil
}

func (vs *VSphere) InstanceType(ctx context.Context, name k8stypes.NodeName) (string, error) {
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

// Zones returns an implementation of Zones for vSphere.
func (vs *VSphere) Zones() (cloudprovider.Zones, bool) {
	if vs.cfg == nil {
		glog.V(1).Info("The vSphere cloud provider does not support zones")
		return nil, false
	}
	return vs, true
}

// Routes returns a false since the interface is not supported for vSphere.
func (vs *VSphere) Routes() (cloudprovider.Routes, bool) {
	return nil, false
}

// AttachDisk attaches given virtual disk volume to the compute running kubelet.
func (vs *VSphere) AttachDisk(vmDiskPath string, storagePolicyName string, nodeName k8stypes.NodeName) (diskUUID string, err error) {
	attachDiskInternal := func(vmDiskPath string, storagePolicyName string, nodeName k8stypes.NodeName) (diskUUID string, err error) {
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
		err = vs.nodeManager.vcConnect(ctx, vsi)
		if err != nil {
			return "", err
		}

		vm, err := vs.getVMFromNodeName(ctx, nodeName)
		if err != nil {
			glog.Errorf("Failed to get VM object for node: %q. err: +%v", convertToString(nodeName), err)
			return "", err
		}

		diskUUID, err = vm.AttachDisk(ctx, vmDiskPath, &vclib.VolumeOptions{SCSIControllerType: vclib.PVSCSIControllerType, StoragePolicyName: storagePolicyName})
		if err != nil {
			glog.Errorf("Failed to attach disk: %s for node: %s. err: +%v", vmDiskPath, convertToString(nodeName), err)
			return "", err
		}
		return diskUUID, nil
	}
	requestTime := time.Now()
	diskUUID, err = attachDiskInternal(vmDiskPath, storagePolicyName, nodeName)
	if err != nil {
		if vclib.IsManagedObjectNotFoundError(err) {
			err = vs.nodeManager.RediscoverNode(nodeName)
			if err == nil {
				glog.V(4).Infof("AttachDisk: Found node %q", convertToString(nodeName))
				diskUUID, err = attachDiskInternal(vmDiskPath, storagePolicyName, nodeName)
				glog.V(4).Infof("AttachDisk: Retry: diskUUID %s, err +%v", diskUUID, err)
			}
		}
	}
	glog.V(4).Infof("AttachDisk executed for node %s and volume %s with diskUUID %s. Err: %s", convertToString(nodeName), vmDiskPath, diskUUID, err)
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
			// If node doesn't exist, disk is already detached from node.
			if err == vclib.ErrNoVMFound {
				glog.Infof("Node %q does not exist, disk %s is already detached from node.", convertToString(nodeName), volPath)
				return nil
			}
			return err
		}
		// Ensure client is logged in and session is valid
		err = vs.nodeManager.vcConnect(ctx, vsi)
		if err != nil {
			return err
		}
		vm, err := vs.getVMFromNodeName(ctx, nodeName)
		if err != nil {
			// If node doesn't exist, disk is already detached from node.
			if err == vclib.ErrNoVMFound {
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
	if err != nil {
		if vclib.IsManagedObjectNotFoundError(err) {
			err = vs.nodeManager.RediscoverNode(nodeName)
			if err == nil {
				err = detachDiskInternal(volPath, nodeName)
			}
		}
	}
	vclib.RecordvSphereMetric(vclib.OperationDetachVolume, requestTime, err)
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
		err = vs.nodeManager.vcConnect(ctx, vsi)
		if err != nil {
			return false, err
		}
		vm, err := vs.getVMFromNodeName(ctx, nodeName)
		if err != nil {
			if err == vclib.ErrNoVMFound {
				glog.Warningf("Node %q does not exist, vsphere CP will assume disk %v is not attached to it.", nodeName, volPath)
				// make the disk as detached and return false without error.
				return false, nil
			}
			glog.Errorf("Failed to get VM object for node: %q. err: +%v", vSphereInstance, err)
			return false, err
		}

		volPath = vclib.RemoveStorageClusterORFolderNameFromVDiskPath(volPath)
		attached, err := vm.IsDiskAttached(ctx, volPath)
		if err != nil {
			glog.Errorf("DiskIsAttached failed to determine whether disk %q is still attached on node %q",
				volPath,
				vSphereInstance)
		}
		glog.V(4).Infof("DiskIsAttached result: %v and error: %q, for volume: %q", attached, err, volPath)
		return attached, err
	}
	requestTime := time.Now()
	isAttached, err := diskIsAttachedInternal(volPath, nodeName)
	if err != nil {
		if vclib.IsManagedObjectNotFoundError(err) {
			err = vs.nodeManager.RediscoverNode(nodeName)
			if err == vclib.ErrNoVMFound {
				isAttached, err = false, nil
			} else if err == nil {
				isAttached, err = diskIsAttachedInternal(volPath, nodeName)
			}
		}
	}
	vclib.RecordvSphereMetric(vclib.OperationDiskIsAttached, requestTime, err)
	return isAttached, err
}

// DisksAreAttached returns if disks are attached to the VM using controllers supported by the plugin.
// 1. Converts volPaths into canonical form so that it can be compared with the VM device path.
// 2. Segregates nodes by vCenter and Datacenter they are present in. This reduces calls to VC.
// 3. Creates go routines per VC-DC to find whether disks are attached to the nodes.
// 4. If the some of the VMs are not found or migrated then they are added to a list.
// 5. After successful execution of goroutines,
// 5a. If there are any VMs which needs to be retried, they are rediscovered and the whole operation is initiated again for only rediscovered VMs.
// 5b. If VMs are removed from vSphere inventory they are ignored.
func (vs *VSphere) DisksAreAttached(nodeVolumes map[k8stypes.NodeName][]string) (map[k8stypes.NodeName]map[string]bool, error) {
	disksAreAttachedInternal := func(nodeVolumes map[k8stypes.NodeName][]string) (map[k8stypes.NodeName]map[string]bool, error) {

		// disksAreAttach checks whether disks are attached to the nodes.
		// Returns nodes that need to be retried if retry is true
		// Segregates nodes per VC and DC
		// Creates go routines per VC-DC to find whether disks are attached to the nodes.
		disksAreAttach := func(ctx context.Context, nodeVolumes map[k8stypes.NodeName][]string, attached map[string]map[string]bool, retry bool) ([]k8stypes.NodeName, error) {

			var wg sync.WaitGroup
			var localAttachedMaps []map[string]map[string]bool
			var nodesToRetry []k8stypes.NodeName
			var globalErr error
			globalErr = nil
			globalErrMutex := &sync.Mutex{}
			nodesToRetryMutex := &sync.Mutex{}

			// Segregate nodes according to VC-DC
			dcNodes := make(map[string][]k8stypes.NodeName)
			for nodeName := range nodeVolumes {
				nodeInfo, err := vs.nodeManager.GetNodeInfo(nodeName)
				if err != nil {
					glog.Errorf("Failed to get node info: %+v. err: %+v", nodeInfo.vm, err)
					return nodesToRetry, err
				}
				VC_DC := nodeInfo.vcServer + nodeInfo.dataCenter.String()
				dcNodes[VC_DC] = append(dcNodes[VC_DC], nodeName)
			}

			for _, nodes := range dcNodes {
				localAttachedMap := make(map[string]map[string]bool)
				localAttachedMaps = append(localAttachedMaps, localAttachedMap)
				// Start go routines per VC-DC to check disks are attached
				go func() {
					nodesToRetryLocal, err := vs.checkDiskAttached(ctx, nodes, nodeVolumes, localAttachedMap, retry)
					if err != nil {
						if !vclib.IsManagedObjectNotFoundError(err) {
							globalErrMutex.Lock()
							globalErr = err
							globalErrMutex.Unlock()
							glog.Errorf("Failed to check disk attached for nodes: %+v. err: %+v", nodes, err)
						}
					}
					nodesToRetryMutex.Lock()
					nodesToRetry = append(nodesToRetry, nodesToRetryLocal...)
					nodesToRetryMutex.Unlock()
					wg.Done()
				}()
				wg.Add(1)
			}
			wg.Wait()
			if globalErr != nil {
				return nodesToRetry, globalErr
			}
			for _, localAttachedMap := range localAttachedMaps {
				for key, value := range localAttachedMap {
					attached[key] = value
				}
			}

			return nodesToRetry, nil
		}

		glog.V(4).Infof("Starting DisksAreAttached API for vSphere with nodeVolumes: %+v", nodeVolumes)
		// Create context
		ctx, cancel := context.WithCancel(context.Background())
		defer cancel()

		disksAttached := make(map[k8stypes.NodeName]map[string]bool)
		if len(nodeVolumes) == 0 {
			return disksAttached, nil
		}

		// Convert VolPaths into canonical form so that it can be compared with the VM device path.
		vmVolumes, err := vs.convertVolPathsToDevicePaths(ctx, nodeVolumes)
		if err != nil {
			glog.Errorf("Failed to convert volPaths to devicePaths: %+v. err: %+v", nodeVolumes, err)
			return nil, err
		}
		attached := make(map[string]map[string]bool)
		nodesToRetry, err := disksAreAttach(ctx, vmVolumes, attached, false)
		if err != nil {
			return nil, err
		}

		if len(nodesToRetry) != 0 {
			// Rediscover nodes which are need to be retried
			remainingNodesVolumes := make(map[k8stypes.NodeName][]string)
			for _, nodeName := range nodesToRetry {
				err = vs.nodeManager.RediscoverNode(nodeName)
				if err != nil {
					if err == vclib.ErrNoVMFound {
						glog.V(4).Infof("node %s not found. err: %+v", nodeName, err)
						continue
					}
					glog.Errorf("Failed to rediscover node %s. err: %+v", nodeName, err)
					return nil, err
				}
				remainingNodesVolumes[nodeName] = nodeVolumes[nodeName]
			}

			// If some remaining nodes are still registered
			if len(remainingNodesVolumes) != 0 {
				nodesToRetry, err = disksAreAttach(ctx, remainingNodesVolumes, attached, true)
				if err != nil || len(nodesToRetry) != 0 {
					glog.Errorf("Failed to retry disksAreAttach  for nodes %+v. err: %+v", remainingNodesVolumes, err)
					return nil, err
				}
			}

			for nodeName, volPaths := range attached {
				disksAttached[convertToK8sType(nodeName)] = volPaths
			}
		}
		glog.V(4).Infof("DisksAreAttach successfully executed. result: %+v", attached)
		return disksAttached, nil
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
		// If datastore not specified, then use default datastore
		if volumeOptions.Datastore == "" {
			datastore = vs.cfg.Workspace.DefaultDatastore
		} else {
			datastore = volumeOptions.Datastore
		}
		datastore = strings.TrimSpace(datastore)
		// Create context
		ctx, cancel := context.WithCancel(context.Background())
		defer cancel()
		vsi, err := vs.getVSphereInstanceForServer(vs.cfg.Workspace.VCenterIP, ctx)
		if err != nil {
			return "", err
		}
		dc, err := vclib.GetDatacenter(ctx, vsi.conn, vs.cfg.Workspace.Datacenter)
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
			vmOptions, err = vs.setVMOptions(ctx, dc, vs.cfg.Workspace.ResourcePoolPath)
			if err != nil {
				glog.Errorf("Failed to set VM options requires to create a vsphere volume. err: %+v", err)
				return "", err
			}
		}
		if volumeOptions.StoragePolicyName != "" && volumeOptions.Datastore == "" {
			datastore, err = getPbmCompatibleDatastore(ctx, dc, volumeOptions.StoragePolicyName, vs.nodeManager)
			if err != nil {
				glog.Errorf("Failed to get pbm compatible datastore with storagePolicy: %s. err: %+v", volumeOptions.StoragePolicyName, err)
				return "", err
			}
		} else {
			// Since no storage policy is specified but datastore is specified, check
			// if the given datastore is a shared datastore across all node VMs.
			sharedDsList, err := getSharedDatastoresInK8SCluster(ctx, dc, vs.nodeManager)
			if err != nil {
				glog.Errorf("Failed to get shared datastore: %+v", err)
				return "", err
			}
			found := false
			for _, sharedDs := range sharedDsList {
				if datastore == sharedDs.Info.Name {
					found = true
					break
				}
			}
			if !found {
				msg := fmt.Sprintf("The specified datastore %s is not a shared datastore across node VMs", datastore)
				return "", errors.New(msg)
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
	glog.V(4).Infof("The canonical volume path for the newly created vSphere volume is %q", canonicalVolumePath)
	return canonicalVolumePath, err
}

// DeleteVolume deletes a volume given volume name.
func (vs *VSphere) DeleteVolume(vmDiskPath string) error {
	glog.V(1).Infof("Starting to delete vSphere volume with vmDiskPath: %s", vmDiskPath)
	deleteVolumeInternal := func(vmDiskPath string) error {
		// Create context
		ctx, cancel := context.WithCancel(context.Background())
		defer cancel()
		vsi, err := vs.getVSphereInstanceForServer(vs.cfg.Workspace.VCenterIP, ctx)
		if err != nil {
			return err
		}
		dc, err := vclib.GetDatacenter(ctx, vsi.conn, vs.cfg.Workspace.Datacenter)
		if err != nil {
			return err
		}
		disk := diskmanagers.VirtualDisk{
			DiskPath:      vmDiskPath,
			VolumeOptions: &vclib.VolumeOptions{},
			VMOptions:     &vclib.VMOptions{},
		}
		err = disk.Delete(ctx, dc)
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

// Notification handler when node is added into k8s cluster.
func (vs *VSphere) NodeAdded(obj interface{}) {
	node, ok := obj.(*v1.Node)
	if node == nil || !ok {
		glog.Warningf("NodeAdded: unrecognized object %+v", obj)
		return
	}

	glog.V(4).Infof("Node added: %+v", node)
	vs.nodeManager.RegisterNode(node)
}

// Notification handler when node is removed from k8s cluster.
func (vs *VSphere) NodeDeleted(obj interface{}) {
	node, ok := obj.(*v1.Node)
	if node == nil || !ok {
		glog.Warningf("NodeDeleted: unrecognized object %+v", obj)
		return
	}

	glog.V(4).Infof("Node deleted: %+v", node)
	vs.nodeManager.UnRegisterNode(node)
}

func (vs *VSphere) NodeManager() (nodeManager *NodeManager) {
	if vs == nil {
		return nil
	}
	return vs.nodeManager
}

func withTagsClient(ctx context.Context, connection *vclib.VSphereConnection, f func(c *rest.Client) error) error {
	c := rest.NewClient(connection.Client)
	user := url.UserPassword(connection.Username, connection.Password)
	if err := c.Login(ctx, user); err != nil {
		return err
	}
	defer c.Logout(ctx)
	return f(c)
}

// GetZone implements Zones.GetZone
func (vs *VSphere) GetZone(ctx context.Context) (cloudprovider.Zone, error) {
	nodeName, err := vs.CurrentNodeName(ctx, vs.hostName)
	if err != nil {
		glog.Errorf("Cannot get node name.")
		return cloudprovider.Zone{}, err
	}
	zone := cloudprovider.Zone{}
	vsi, err := vs.getVSphereInstanceForServer(vs.cfg.Workspace.VCenterIP, ctx)
	if err != nil {
		glog.Errorf("Cannot connent to vsphere. Get zone for node %s error", nodeName)
		return cloudprovider.Zone{}, err
	}
	dc, err := vclib.GetDatacenter(ctx, vsi.conn, vs.cfg.Workspace.Datacenter)
	if err != nil {
		glog.Errorf("Cannot connent to datacenter. Get zone for node %s error", nodeName)
		return cloudprovider.Zone{}, err
	}
	vmHost, err := dc.GetHostByVMUUID(ctx, vs.vmUUID)
	if err != nil {
		glog.Errorf("Cannot find VM runtime host. Get zone for node %s error", nodeName)
		return cloudprovider.Zone{}, err
	}

	pc := vsi.conn.Client.ServiceContent.PropertyCollector
	err = withTagsClient(ctx, vsi.conn, func(c *rest.Client) error {
		client := tags.NewManager(c)
		// example result: ["Folder", "Datacenter", "Cluster", "Host"]
		objects, err := mo.Ancestors(ctx, vsi.conn.Client, pc, *vmHost)
		if err != nil {
			return err
		}

		// search the hierarchy, example order: ["Host", "Cluster", "Datacenter", "Folder"]
		for i := range objects {
			obj := objects[len(objects)-1-i]
			tags, err := client.ListAttachedTags(ctx, obj)
			if err != nil {
				glog.Errorf("Cannot list attached tags. Get zone for node %s: %s", nodeName, err)
				return err
			}
			for _, value := range tags {
				tag, err := client.GetTag(ctx, value)
				if err != nil {
					glog.Errorf("Get tag %s: %s", value, err)
					return err
				}
				category, err := client.GetCategory(ctx, tag.CategoryID)
				if err != nil {
					glog.Errorf("Get category %s error", value)
					return err
				}

				found := func() {
					glog.Errorf("Found %q tag (%s) for %s attached to %s", category.Name, tag.Name, vs.vmUUID, obj.Reference())
				}
				switch {
				case category.Name == vs.cfg.Labels.Zone:
					zone.FailureDomain = tag.Name
					found()
				case category.Name == vs.cfg.Labels.Region:
					zone.Region = tag.Name
					found()
				}

				if zone.FailureDomain != "" && zone.Region != "" {
					return nil
				}
			}
		}

		if zone.Region == "" {
			if vs.cfg.Labels.Region != "" {
				return fmt.Errorf("vSphere region category %q does not match any tags for node %s [%s]", vs.cfg.Labels.Region, nodeName, vs.vmUUID)
			}
		}
		if zone.FailureDomain == "" {
			if vs.cfg.Labels.Zone != "" {
				return fmt.Errorf("vSphere zone category %q does not match any tags for node %s [%s]", vs.cfg.Labels.Zone, nodeName, vs.vmUUID)
			}
		}

		return nil
	})
	if err != nil {
		glog.Errorf("Get zone for node %s: %s", nodeName, err)
		return cloudprovider.Zone{}, err
	}
	return zone, nil
}

func (vs *VSphere) GetZoneByNodeName(ctx context.Context, nodeName k8stypes.NodeName) (cloudprovider.Zone, error) {
	return cloudprovider.Zone{}, cloudprovider.NotImplemented
}

func (vs *VSphere) GetZoneByProviderID(ctx context.Context, providerID string) (cloudprovider.Zone, error) {
	return cloudprovider.Zone{}, cloudprovider.NotImplemented
}
