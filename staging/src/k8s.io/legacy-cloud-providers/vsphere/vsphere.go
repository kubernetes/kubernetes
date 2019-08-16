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

	"github.com/vmware/govmomi/object"
	"github.com/vmware/govmomi/vapi/rest"
	"github.com/vmware/govmomi/vapi/tags"
	"github.com/vmware/govmomi/vim25/mo"
	vmwaretypes "github.com/vmware/govmomi/vim25/types"
	v1 "k8s.io/api/core/v1"
	k8stypes "k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/client-go/informers"
	"k8s.io/client-go/tools/cache"
	cloudprovider "k8s.io/cloud-provider"
	nodehelpers "k8s.io/cloud-provider/node/helpers"
	volumehelpers "k8s.io/cloud-provider/volume/helpers"
	"k8s.io/klog"

	"k8s.io/legacy-cloud-providers/vsphere/vclib"
	"k8s.io/legacy-cloud-providers/vsphere/vclib/diskmanagers"
)

// VSphere Cloud Provider constants
const (
	ProviderName                  = "vsphere"
	VolDir                        = "kubevols"
	RoundTripperDefaultCount      = 3
	DummyVMPrefixName             = "vsphere-k8s"
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
	NoZoneTagInVCErrMsg   = "No zone tags found in vCenter"
)

// Error constants
var (
	ErrUsernameMissing = errors.New(MissingUsernameErrMsg)
	ErrPasswordMissing = errors.New(MissingPasswordErrMsg)
	ErrNoZoneTagInVC   = errors.New(NoZoneTagInVCErrMsg)
)

var _ cloudprovider.Interface = (*VSphere)(nil)
var _ cloudprovider.Instances = (*VSphere)(nil)
var _ cloudprovider.Zones = (*VSphere)(nil)
var _ cloudprovider.PVLabeler = (*VSphere)(nil)

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
// NOTE: Cloud config files should follow the same Kubernetes deprecation policy as
// flags or CLIs. Config fields should not change behavior in incompatible ways and
// should be deprecated for at least 2 release prior to removing.
// See https://kubernetes.io/docs/reference/using-api/deprecation-policy/#deprecating-a-flag-or-cli
// for more details.
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
func (vs *VSphere) Initialize(clientBuilder cloudprovider.ControllerClientBuilder, stop <-chan struct{}) {
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
	klog.V(4).Infof("Setting up node informers for vSphere Cloud Provider")
	nodeInformer := informerFactory.Core().V1().Nodes().Informer()
	nodeInformer.AddEventHandler(cache.ResourceEventHandlerFuncs{
		AddFunc:    vs.NodeAdded,
		DeleteFunc: vs.NodeDeleted,
	})
	klog.V(4).Infof("Node informers in vSphere cloud provider initialized")

}

// Creates new worker node interface and returns
func newWorkerNode() (*VSphere, error) {
	var err error
	vs := VSphere{}
	vs.hostName, err = os.Hostname()
	if err != nil {
		klog.Errorf("Failed to get hostname. err: %+v", err)
		return nil, err
	}
	vs.vmUUID, err = GetVMUUID()
	if err != nil {
		klog.Errorf("Failed to get uuid. err: %+v", err)
		return nil, err
	}
	return &vs, nil
}

func populateVsphereInstanceMap(cfg *VSphereConfig) (map[string]*VSphereInstance, error) {
	vsphereInstanceMap := make(map[string]*VSphereInstance)
	isSecretInfoProvided := true

	if cfg.Global.SecretName == "" || cfg.Global.SecretNamespace == "" {
		klog.Warningf("SecretName and/or SecretNamespace is not provided. " +
			"VCP will use username and password from config file")
		isSecretInfoProvided = false
	}

	if isSecretInfoProvided {
		if cfg.Global.User != "" {
			klog.Warning("Global.User and Secret info provided. VCP will use secret to get credentials")
			cfg.Global.User = ""
		}
		if cfg.Global.Password != "" {
			klog.Warning("Global.Password and Secret info provided. VCP will use secret to get credentials")
			cfg.Global.Password = ""
		}
	}

	// Check if the vsphere.conf is in old format. In this
	// format the cfg.VirtualCenter will be nil or empty.
	if cfg.VirtualCenter == nil || len(cfg.VirtualCenter) == 0 {
		klog.V(4).Infof("Config is not per virtual center and is in old format.")
		if !isSecretInfoProvided {
			if cfg.Global.User == "" {
				klog.Error("Global.User is empty!")
				return nil, ErrUsernameMissing
			}
			if cfg.Global.Password == "" {
				klog.Error("Global.Password is empty!")
				return nil, ErrPasswordMissing
			}
		}

		if cfg.Global.WorkingDir == "" {
			klog.Error("Global.WorkingDir is empty!")
			return nil, errors.New("Global.WorkingDir is empty!")
		}
		if cfg.Global.VCenterIP == "" {
			klog.Error("Global.VCenterIP is empty!")
			return nil, errors.New("Global.VCenterIP is empty!")
		}
		if cfg.Global.Datacenter == "" {
			klog.Error("Global.Datacenter is empty!")
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
			klog.Error(msg)
			return nil, errors.New(msg)
		}

		for vcServer, vcConfig := range cfg.VirtualCenter {
			klog.V(4).Infof("Initializing vc server %s", vcServer)
			if vcServer == "" {
				klog.Error("vsphere.conf does not have the VirtualCenter IP address specified")
				return nil, errors.New("vsphere.conf does not have the VirtualCenter IP address specified")
			}

			if !isSecretInfoProvided {
				if vcConfig.User == "" {
					vcConfig.User = cfg.Global.User
					if vcConfig.User == "" {
						klog.Errorf("vcConfig.User is empty for vc %s!", vcServer)
						return nil, ErrUsernameMissing
					}
				}
				if vcConfig.Password == "" {
					vcConfig.Password = cfg.Global.Password
					if vcConfig.Password == "" {
						klog.Errorf("vcConfig.Password is empty for vc %s!", vcServer)
						return nil, ErrPasswordMissing
					}
				}
			} else {
				if vcConfig.User != "" {
					klog.Warningf("vcConfig.User for server %s and Secret info provided. VCP will use secret to get credentials", vcServer)
					vcConfig.User = ""
				}
				if vcConfig.Password != "" {
					klog.Warningf("vcConfig.Password for server %s and Secret info provided. VCP will use secret to get credentials", vcServer)
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
		klog.Errorf("Failed to get hostname. err: %+v", err)
		return nil, err
	}
	if cfg.Global.VMUUID != "" {
		vs.vmUUID = cfg.Global.VMUUID
	} else {
		vs.vmUUID, err = getVMUUID()
		if err != nil {
			klog.Errorf("Failed to get uuid. err: %+v", err)
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
		klog.Errorf("%v is not a supported SCSI Controller type. Please configure 'lsilogic-sas' OR 'pvscsi'", cfg.Disk.SCSIControllerType)
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
	// hashtable with VMware-allocated OUIs for MAC filtering
	// List of official OUIs: http://standards-oui.ieee.org/oui.txt
	vmwareOUI := map[string]bool{
		"00:05:69": true,
		"00:0c:29": true,
		"00:1c:14": true,
		"00:50:56": true,
	}

	addrs := []v1.NodeAddress{}
	ifaces, err := net.Interfaces()
	if err != nil {
		klog.Errorf("net.Interfaces() failed for NodeAddresses - %v", err)
		return nil, err
	}
	for _, i := range ifaces {
		if i.Flags&net.FlagLoopback != 0 {
			continue
		}
		localAddrs, err := i.Addrs()
		if err != nil {
			klog.Warningf("Failed to extract addresses for NodeAddresses - %v", err)
		} else {
			for _, addr := range localAddrs {
				if ipnet, ok := addr.(*net.IPNet); ok {
					if ipnet.IP.To4() != nil {
						// Filter external IP by MAC address OUIs from vCenter and from ESX
						vmMACAddr := strings.ToLower(i.HardwareAddr.String())
						// Making sure that the MAC address is long enough
						if len(vmMACAddr) < 17 {
							klog.V(4).Infof("Skipping invalid MAC address: %q", vmMACAddr)
							continue
						}
						if vmwareOUI[vmMACAddr[:8]] {
							nodehelpers.AddToNodeAddresses(&addrs,
								v1.NodeAddress{
									Type:    v1.NodeExternalIP,
									Address: ipnet.IP.String(),
								},
								v1.NodeAddress{
									Type:    v1.NodeInternalIP,
									Address: ipnet.IP.String(),
								},
							)
							klog.V(4).Infof("Detected local IP address as %q", ipnet.IP.String())
						} else {
							klog.V(4).Infof("Failed to patch IP for interface %q as MAC address %q does not belong to a VMware platform", i.Name, vmMACAddr)
						}
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
		klog.Errorf("Cannot find node %q in cache. Node not found!!!", nodeName)
		return nil, err
	}
	return &vsphereIns, nil
}

func (vs *VSphere) getVSphereInstanceForServer(vcServer string, ctx context.Context) (*VSphereInstance, error) {
	vsphereIns, ok := vs.vsphereInstanceMap[vcServer]
	if !ok {
		klog.Errorf("cannot find vcServer %q in cache. VC not found!!!", vcServer)
		return nil, fmt.Errorf("cannot find node %q in vsphere configuration map", vcServer)
	}
	// Ensure client is logged in and session is valid
	err := vs.nodeManager.vcConnect(ctx, vsphereIns)
	if err != nil {
		klog.Errorf("failed connecting to vcServer %q with error %+v", vcServer, err)
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
		nodehelpers.AddToNodeAddresses(&addrs, v1.NodeAddress{Type: v1.NodeHostName, Address: vs.hostName})
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
		klog.Errorf("Failed to get VM object for node: %q. err: +%v", convertToString(nodeName), err)
		return nil, err
	}
	vmMoList, err := vm.Datacenter.GetVMMoList(ctx, []*vclib.VirtualMachine{vm}, []string{"guest.net"})
	if err != nil {
		klog.Errorf("Failed to get VM Managed object with property guest.net for node: %q. err: +%v", convertToString(nodeName), err)
		return nil, err
	}
	// retrieve VM's ip(s)
	for _, v := range vmMoList[0].Guest.Net {
		if vs.cfg.Network.PublicNetwork == v.Network {
			for _, ip := range v.IpAddress {
				if net.ParseIP(ip).To4() != nil {
					nodehelpers.AddToNodeAddresses(&addrs,
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
		klog.Errorf("Error while getting nodename for providerID %s", providerID)
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
		klog.Errorf("Error while getting nodename for providerID %s", providerID)
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
		klog.Errorf("Failed to get VM object for node: %q. err: +%v", nodeName, err)
		return false, err
	}
	isActive, err := vm.IsActive(ctx)
	if err != nil {
		klog.Errorf("Failed to check whether node %q is active. err: %+v.", nodeName, err)
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
			return "", fmt.Errorf("The current node can't determine InstanceID for %q", convertToString(nodeName))
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
			klog.Errorf("Failed to get VM object for node: %q. err: +%v", convertToString(nodeName), err)
			return "", err
		}
		isActive, err := vm.IsActive(ctx)
		if err != nil {
			klog.Errorf("Failed to check whether node %q is active. err: %+v.", convertToString(nodeName), err)
			return "", err
		}
		if isActive {
			return vs.vmUUID, nil
		}
		klog.Warningf("The VM: %s is not in %s state", convertToString(nodeName), vclib.ActivePowerState)
		return "", cloudprovider.InstanceNotFound
	}

	instanceID, err := instanceIDInternal()
	if err != nil {
		if vclib.IsManagedObjectNotFoundError(err) {
			err = vs.nodeManager.RediscoverNode(nodeName)
			if err == nil {
				klog.V(4).Infof("InstanceID: Found node %q", convertToString(nodeName))
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

func (vs *VSphere) isZoneEnabled() bool {
	return vs.cfg != nil && vs.cfg.Labels.Zone != "" && vs.cfg.Labels.Region != ""
}

// Zones returns an implementation of Zones for vSphere.
func (vs *VSphere) Zones() (cloudprovider.Zones, bool) {
	if vs.isZoneEnabled() {
		return vs, true
	}
	klog.V(1).Info("The vSphere cloud provider does not support zones")
	return nil, false
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
			klog.Errorf("Failed to get VM object for node: %q. err: +%v", convertToString(nodeName), err)
			return "", err
		}

		diskUUID, err = vm.AttachDisk(ctx, vmDiskPath, &vclib.VolumeOptions{SCSIControllerType: vclib.PVSCSIControllerType, StoragePolicyName: storagePolicyName})
		if err != nil {
			klog.Errorf("Failed to attach disk: %s for node: %s. err: +%v", vmDiskPath, convertToString(nodeName), err)
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
				klog.V(4).Infof("AttachDisk: Found node %q", convertToString(nodeName))
				diskUUID, err = attachDiskInternal(vmDiskPath, storagePolicyName, nodeName)
				klog.V(4).Infof("AttachDisk: Retry: diskUUID %s, err +%v", diskUUID, err)
			}
		}
	}
	klog.V(4).Infof("AttachDisk executed for node %s and volume %s with diskUUID %s. Err: %s", convertToString(nodeName), vmDiskPath, diskUUID, err)
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
				klog.Infof("Node %q does not exist, disk %s is already detached from node.", convertToString(nodeName), volPath)
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
				klog.Infof("Node %q does not exist, disk %s is already detached from node.", convertToString(nodeName), volPath)
				return nil
			}

			klog.Errorf("Failed to get VM object for node: %q. err: +%v", convertToString(nodeName), err)
			return err
		}
		err = vm.DetachDisk(ctx, volPath)
		if err != nil {
			klog.Errorf("Failed to detach disk: %s for node: %s. err: +%v", volPath, convertToString(nodeName), err)
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
				klog.Warningf("Node %q does not exist, vsphere CP will assume disk %v is not attached to it.", nodeName, volPath)
				// make the disk as detached and return false without error.
				return false, nil
			}
			klog.Errorf("Failed to get VM object for node: %q. err: +%v", vSphereInstance, err)
			return false, err
		}

		volPath = vclib.RemoveStorageClusterORFolderNameFromVDiskPath(volPath)
		attached, err := vm.IsDiskAttached(ctx, volPath)
		if err != nil {
			klog.Errorf("DiskIsAttached failed to determine whether disk %q is still attached on node %q",
				volPath,
				vSphereInstance)
		}
		klog.V(4).Infof("DiskIsAttached result: %v and error: %q, for volume: %q", attached, err, volPath)
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
					klog.Errorf("Failed to get node info: %+v. err: %+v", nodeInfo.vm, err)
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
							klog.Errorf("Failed to check disk attached for nodes: %+v. err: %+v", nodes, err)
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

		klog.V(4).Infof("Starting DisksAreAttached API for vSphere with nodeVolumes: %+v", nodeVolumes)
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
			klog.Errorf("Failed to convert volPaths to devicePaths: %+v. err: %+v", nodeVolumes, err)
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
						klog.V(4).Infof("node %s not found. err: %+v", nodeName, err)
						continue
					}
					klog.Errorf("Failed to rediscover node %s. err: %+v", nodeName, err)
					return nil, err
				}
				remainingNodesVolumes[nodeName] = nodeVolumes[nodeName]
			}

			// If some remaining nodes are still registered
			if len(remainingNodesVolumes) != 0 {
				nodesToRetry, err = disksAreAttach(ctx, remainingNodesVolumes, attached, true)
				if err != nil || len(nodesToRetry) != 0 {
					klog.Errorf("Failed to retry disksAreAttach  for nodes %+v. err: %+v", remainingNodesVolumes, err)
					return nil, err
				}
			}

			for nodeName, volPaths := range attached {
				disksAttached[convertToK8sType(nodeName)] = volPaths
			}
		}
		klog.V(4).Infof("DisksAreAttach successfully executed. result: %+v", attached)
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
	klog.V(1).Infof("Starting to create a vSphere volume with volumeOptions: %+v", volumeOptions)
	createVolumeInternal := func(volumeOptions *vclib.VolumeOptions) (canonicalVolumePath string, err error) {
		var datastoreInfo *vclib.DatastoreInfo
		var dsList []*vclib.DatastoreInfo

		// Create context
		ctx, cancel := context.WithCancel(context.Background())
		defer cancel()
		vsi, err := vs.getVSphereInstanceForServer(vs.cfg.Workspace.VCenterIP, ctx)
		if err != nil {
			return "", err
		}
		// If datastore not specified, then use default datastore
		datastoreName := strings.TrimSpace(volumeOptions.Datastore)
		if datastoreName == "" {
			datastoreName = strings.TrimSpace(vs.cfg.Workspace.DefaultDatastore)
		}
		// The given datastoreName may be present in more than one datacenter
		candidateDatastoreInfos, err := vs.FindDatastoreByName(ctx, datastoreName)
		if err != nil {
			return "", err
		}
		// Each of the datastores found is a candidate for Volume creation.
		// One of these will be selected based on given policy and/or zone.
		candidateDatastores := make(map[string]*vclib.DatastoreInfo)
		for _, dsInfo := range candidateDatastoreInfos {
			candidateDatastores[dsInfo.Info.Url] = dsInfo
		}

		var vmOptions *vclib.VMOptions
		var zonesToSearch []string

		if volumeOptions.SelectedNode != nil {
			if len(volumeOptions.Zone) > 1 {
				// In waitForFirstConsumer mode, if more than one allowedTopologies is specified, the volume should satisfy all these.
				zonesToSearch = volumeOptions.Zone
			} else {
				// Pick the selectedNode's zone, if available.
				nodeInfo, err := vs.nodeManager.GetNodeInfoWithNodeObject(volumeOptions.SelectedNode)
				if err != nil {
					klog.Errorf("Unable to get node information for %s. err: %+v", volumeOptions.SelectedNode.Name, err)
					return "", err
				}
				klog.V(4).Infof("selectedNode info : %s", nodeInfo)
				if nodeInfo.zone != nil && nodeInfo.zone.FailureDomain != "" {
					zonesToSearch = append(zonesToSearch, nodeInfo.zone.FailureDomain)
				}
			}
		} else {
			// If no selectedNode, pick allowedTopologies, if provided.
			zonesToSearch = volumeOptions.Zone
		}
		klog.V(1).Infof("Volume topology : %s", zonesToSearch)

		if volumeOptions.VSANStorageProfileData != "" || volumeOptions.StoragePolicyName != "" {
			// If datastore and zone are specified, first validate if the datastore is in the provided zone.
			if len(zonesToSearch) != 0 && volumeOptions.Datastore != "" {
				klog.V(4).Infof("Specified zone : %s, datastore : %s", zonesToSearch, volumeOptions.Datastore)
				dsList, err = getDatastoresForZone(ctx, vs.nodeManager, zonesToSearch)
				if err != nil {
					klog.Errorf("Failed to find a shared datastore matching zone %s. err: %+v", zonesToSearch, err)
					return "", err
				}

				// Validate if the datastore provided belongs to the zone. If not, fail the operation.
				found := false
				for _, ds := range dsList {
					if datastoreInfo, found = candidateDatastores[ds.Info.Url]; found {
						break
					}
				}
				if !found {
					err := fmt.Errorf("The specified datastore %s does not match the provided zones : %s", volumeOptions.Datastore, zonesToSearch)
					klog.Error(err)
					return "", err
				}
			}
			// Acquire a read lock to ensure multiple PVC requests can be processed simultaneously.
			cleanUpDummyVMLock.RLock()
			defer cleanUpDummyVMLock.RUnlock()
			// Create a new background routine that will delete any dummy VM's that are left stale.
			// This routine will get executed for every 5 minutes and gets initiated only once in its entire lifetime.
			cleanUpRoutineInitLock.Lock()
			if !cleanUpRoutineInitialized {
				klog.V(1).Infof("Starting a clean up routine to remove stale dummy VM's")
				go vs.cleanUpDummyVMs(DummyVMPrefixName)
				cleanUpRoutineInitialized = true
			}
			cleanUpRoutineInitLock.Unlock()
		}
		if volumeOptions.StoragePolicyName != "" && volumeOptions.Datastore == "" {
			if len(zonesToSearch) == 0 {
				klog.V(4).Infof("Selecting a shared datastore as per the storage policy %s", volumeOptions.StoragePolicyName)
				datastoreInfo, err = getPbmCompatibleDatastore(ctx, vsi.conn.Client, volumeOptions.StoragePolicyName, vs.nodeManager)
			} else {
				// If zone is specified, first get the datastores in the zone.
				dsList, err = getDatastoresForZone(ctx, vs.nodeManager, zonesToSearch)

				if err != nil {
					klog.Errorf("Failed to find a shared datastore matching zone %s. err: %+v", zonesToSearch, err)
					return "", err
				}

				klog.V(4).Infof("Specified zone : %s. Picking a datastore as per the storage policy %s among the zoned datastores : %s", zonesToSearch,
					volumeOptions.StoragePolicyName, dsList)
				// Among the compatible datastores, select the one based on the maximum free space.
				datastoreInfo, err = getPbmCompatibleZonedDatastore(ctx, vsi.conn.Client, volumeOptions.StoragePolicyName, dsList)
			}
			if err != nil {
				klog.Errorf("Failed to get pbm compatible datastore with storagePolicy: %s. err: %+v", volumeOptions.StoragePolicyName, err)
				return "", err
			}
			klog.V(1).Infof("Datastore selected as per policy : %s", datastoreInfo.Info.Name)
		} else {
			// If zone is specified, pick the datastore in the zone with maximum free space within the zone.
			if volumeOptions.Datastore == "" && len(zonesToSearch) != 0 {
				klog.V(4).Infof("Specified zone : %s", zonesToSearch)
				dsList, err = getDatastoresForZone(ctx, vs.nodeManager, zonesToSearch)

				if err != nil {
					klog.Errorf("Failed to find a shared datastore matching zone %s. err: %+v", zonesToSearch, err)
					return "", err
				}
				// If unable to get any datastore, fail the operation
				if len(dsList) == 0 {
					err := fmt.Errorf("Failed to find a shared datastore matching zone %s", zonesToSearch)
					klog.Error(err)
					return "", err
				}

				datastoreInfo, err = getMostFreeDatastore(ctx, nil, dsList)
				if err != nil {
					klog.Errorf("Failed to get shared datastore: %+v", err)
					return "", err
				}
				klog.V(1).Infof("Specified zone : %s. Selected datastore : %s", zonesToSearch, datastoreInfo.Info.Name)
			} else {
				var sharedDsList []*vclib.DatastoreInfo
				var err error
				if len(zonesToSearch) == 0 {
					// If zone is not provided, get the shared datastore across all node VMs.
					klog.V(4).Infof("Validating if datastore %s is shared across all node VMs", datastoreName)
					sharedDsList, err = getSharedDatastoresInK8SCluster(ctx, vs.nodeManager)
					if err != nil {
						klog.Errorf("Failed to get shared datastore: %+v", err)
						return "", err
					}
					// Prepare error msg to be used later, if required.
					err = fmt.Errorf("The specified datastore %s is not a shared datastore across node VMs", datastoreName)
				} else {
					// If zone is provided, get the shared datastores in that zone.
					klog.V(4).Infof("Validating if datastore %s is in zone %s ", datastoreName, zonesToSearch)
					sharedDsList, err = getDatastoresForZone(ctx, vs.nodeManager, zonesToSearch)
					if err != nil {
						klog.Errorf("Failed to find a shared datastore matching zone %s. err: %+v", zonesToSearch, err)
						return "", err
					}
					// Prepare error msg to be used later, if required.
					err = fmt.Errorf("The specified datastore %s does not match the provided zones : %s", datastoreName, zonesToSearch)
				}
				found := false
				// Check if the selected datastore belongs to the list of shared datastores computed.
				for _, sharedDs := range sharedDsList {
					if datastoreInfo, found = candidateDatastores[sharedDs.Info.Url]; found {
						klog.V(4).Infof("Datastore validation succeeded")
						found = true
						break
					}
				}
				if !found {
					klog.Error(err)
					return "", err
				}
			}
		}

		// if datastoreInfo is still not determined, it is an error condition
		if datastoreInfo == nil {
			klog.Errorf("ambiguous datastore name %s, cannot be found among: %v", datastoreName, candidateDatastoreInfos)
			return "", fmt.Errorf("ambiguous datastore name %s", datastoreName)
		}
		ds := datastoreInfo.Datastore
		volumeOptions.Datastore = datastoreInfo.Info.Name
		vmOptions, err = vs.setVMOptions(ctx, vsi.conn, ds)
		if err != nil {
			klog.Errorf("failed to set VM options required to create a vsphere volume. err: %+v", err)
			return "", err
		}
		kubeVolsPath := filepath.Clean(ds.Path(VolDir)) + "/"
		err = ds.CreateDirectory(ctx, kubeVolsPath, false)
		if err != nil && err != vclib.ErrFileAlreadyExist {
			klog.Errorf("Cannot create dir %#v. err %s", kubeVolsPath, err)
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
			klog.Errorf("Failed to create a vsphere volume with volumeOptions: %+v on datastore: %s. err: %+v", volumeOptions, ds, err)
			return "", err
		}
		// Get the canonical path for the volume path.
		canonicalVolumePath, err = getcanonicalVolumePath(ctx, datastoreInfo.Datacenter, volumePath)
		if err != nil {
			klog.Errorf("Failed to get canonical vsphere volume path for volume: %s with volumeOptions: %+v on datastore: %s. err: %+v", volumePath, volumeOptions, ds, err)
			return "", err
		}
		if filepath.Base(datastoreName) != datastoreName {
			// If datastore is within cluster, add cluster path to the volumePath
			canonicalVolumePath = strings.Replace(canonicalVolumePath, filepath.Base(datastoreName), datastoreName, 1)
		}
		return canonicalVolumePath, nil
	}
	requestTime := time.Now()
	canonicalVolumePath, err = createVolumeInternal(volumeOptions)
	vclib.RecordCreateVolumeMetric(volumeOptions, requestTime, err)
	klog.V(4).Infof("The canonical volume path for the newly created vSphere volume is %q", canonicalVolumePath)
	return canonicalVolumePath, err
}

// DeleteVolume deletes a volume given volume name.
func (vs *VSphere) DeleteVolume(vmDiskPath string) error {
	klog.V(1).Infof("Starting to delete vSphere volume with vmDiskPath: %s", vmDiskPath)
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
			klog.Errorf("Failed to delete vsphere volume with vmDiskPath: %s. err: %+v", vmDiskPath, err)
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
		klog.Warningf("NodeAdded: unrecognized object %+v", obj)
		return
	}

	klog.V(4).Infof("Node added: %+v", node)
	if err := vs.nodeManager.RegisterNode(node); err != nil {
		klog.Errorf("failed to add node %+v: %v", node, err)
	}
}

// Notification handler when node is removed from k8s cluster.
func (vs *VSphere) NodeDeleted(obj interface{}) {
	node, ok := obj.(*v1.Node)
	if node == nil || !ok {
		klog.Warningf("NodeDeleted: unrecognized object %+v", obj)
		return
	}

	klog.V(4).Infof("Node deleted: %+v", node)
	if err := vs.nodeManager.UnRegisterNode(node); err != nil {
		klog.Errorf("failed to delete node %s: %v", node.Name, err)
	}
}

func (vs *VSphere) NodeManager() (nodeManager *NodeManager) {
	if vs == nil {
		return nil
	}
	return vs.nodeManager
}

func withTagsClient(ctx context.Context, connection *vclib.VSphereConnection, f func(c *rest.Client) error) error {
	c := rest.NewClient(connection.Client)
	signer, err := connection.Signer(ctx, connection.Client)
	if err != nil {
		return err
	}
	if signer == nil {
		user := url.UserPassword(connection.Username, connection.Password)
		err = c.Login(ctx, user)
	} else {
		err = c.LoginByToken(c.WithSigner(ctx, signer))
	}
	if err != nil {
		return err
	}

	defer func() {
		if err := c.Logout(ctx); err != nil {
			klog.Errorf("failed to logout: %v", err)
		}
	}()
	return f(c)
}

// GetZone implements Zones.GetZone
func (vs *VSphere) GetZone(ctx context.Context) (cloudprovider.Zone, error) {
	nodeName, err := vs.CurrentNodeName(ctx, vs.hostName)
	if err != nil {
		klog.Errorf("Cannot get node name.")
		return cloudprovider.Zone{}, err
	}
	zone := cloudprovider.Zone{}
	vsi, err := vs.getVSphereInstanceForServer(vs.cfg.Workspace.VCenterIP, ctx)
	if err != nil {
		klog.Errorf("Cannot connect to vsphere. Get zone for node %s error", nodeName)
		return cloudprovider.Zone{}, err
	}
	dc, err := vclib.GetDatacenter(ctx, vsi.conn, vs.cfg.Workspace.Datacenter)
	if err != nil {
		klog.Errorf("Cannot connect to datacenter. Get zone for node %s error", nodeName)
		return cloudprovider.Zone{}, err
	}
	vmHost, err := dc.GetHostByVMUUID(ctx, vs.vmUUID)
	if err != nil {
		klog.Errorf("Cannot find VM runtime host. Get zone for node %s error", nodeName)
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
				klog.Errorf("Cannot list attached tags. Get zone for node %s: %s", nodeName, err)
				return err
			}
			for _, value := range tags {
				tag, err := client.GetTag(ctx, value)
				if err != nil {
					klog.Errorf("Get tag %s: %s", value, err)
					return err
				}
				category, err := client.GetCategory(ctx, tag.CategoryID)
				if err != nil {
					klog.Errorf("Get category %s error", value)
					return err
				}

				found := func() {
					klog.Errorf("Found %q tag (%s) for %s attached to %s", category.Name, tag.Name, vs.vmUUID, obj.Reference())
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
			return fmt.Errorf("vSphere region category %q does not match any tags for node %s [%s]", vs.cfg.Labels.Region, nodeName, vs.vmUUID)
		}
		if zone.FailureDomain == "" {
			return fmt.Errorf("vSphere zone category %q does not match any tags for node %s [%s]", vs.cfg.Labels.Zone, nodeName, vs.vmUUID)
		}

		return nil
	})
	if err != nil {
		klog.Errorf("Get zone for node %s: %s", nodeName, err)
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

// GetLabelsForVolume implements the PVLabeler interface for VSphere
// since this interface is used by the PV label admission controller.
func (vs *VSphere) GetLabelsForVolume(ctx context.Context, pv *v1.PersistentVolume) (map[string]string, error) {
	// ignore if zones not enabled
	if !vs.isZoneEnabled() {
		klog.V(4).Infof("Zone labels for volume is not enabled in vsphere.conf")
		return nil, nil
	}
	// ignore if not vSphere volume
	if pv.Spec.VsphereVolume == nil {
		return nil, nil
	}
	return vs.GetVolumeLabels(pv.Spec.VsphereVolume.VolumePath)
}

// GetVolumeLabels returns the well known zone and region labels for given volume
func (vs *VSphere) GetVolumeLabels(volumePath string) (map[string]string, error) {
	// Create context
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	// if zones is not enabled return no labels
	if !vs.isZoneEnabled() {
		klog.V(4).Infof("Volume zone labels is not enabled in vsphere.conf")
		return nil, nil
	}

	// Find the datastore on which this volume resides
	datastorePathObj, err := vclib.GetDatastorePathObjFromVMDiskPath(volumePath)
	if err != nil {
		klog.Errorf("Failed to get datastore for volume: %v: %+v", volumePath, err)
		return nil, err
	}
	dsInfos, err := vs.FindDatastoreByName(ctx, datastorePathObj.Datastore)
	if err != nil {
		klog.Errorf("Failed to get datastore by name: %v: %+v", datastorePathObj.Datastore, err)
		return nil, err
	}
	var datastore *vclib.Datastore
	for _, dsInfo := range dsInfos {
		if dsInfo.Datastore.Exists(ctx, datastorePathObj.Path) {
			datastore = dsInfo.Datastore
		}
	}
	if datastore == nil {
		klog.Errorf("Could not find %s among %v", volumePath, dsInfos)
		return nil, fmt.Errorf("could not find the datastore for volume: %s", volumePath)
	}

	dsZones, err := vs.GetZonesForDatastore(ctx, datastore)
	if err != nil {
		klog.Errorf("Failed to get zones for datastore %v: %+v", datastorePathObj.Datastore, err)
		return nil, err
	}
	dsZones, err = vs.collapseZonesInRegion(ctx, dsZones)
	// FIXME: For now, pick the first zone of datastore as the zone of volume
	labels := make(map[string]string)
	if len(dsZones) > 0 {
		labels[v1.LabelZoneRegion] = dsZones[0].Region
		labels[v1.LabelZoneFailureDomain] = dsZones[0].FailureDomain
	}
	return labels, nil
}

// collapse all zones in same region. Join FailureDomain with well known separator
func (vs *VSphere) collapseZonesInRegion(ctx context.Context, zones []cloudprovider.Zone) ([]cloudprovider.Zone, error) {
	// first create a map of region -> list of zones in that region
	regionToZones := make(map[string][]string)
	for _, zone := range zones {
		fds, exists := regionToZones[zone.Region]
		if !exists {
			fds = make([]string, 0)
		}
		regionToZones[zone.Region] = append(fds, zone.FailureDomain)
	}

	// Join all fds in same region and return Zone instances
	collapsedZones := make([]cloudprovider.Zone, 0)
	for region, fds := range regionToZones {
		fdSet := sets.NewString(fds...)
		appendedZone := volumehelpers.ZonesSetToLabelValue(fdSet)
		collapsedZones = append(collapsedZones, cloudprovider.Zone{FailureDomain: appendedZone, Region: region})
	}
	return collapsedZones, nil
}

// GetZonesForDatastore returns all the zones from which this datastore is visible
func (vs *VSphere) GetZonesForDatastore(ctx context.Context, datastore *vclib.Datastore) ([]cloudprovider.Zone, error) {
	vsi, err := vs.getVSphereInstanceForServer(vs.cfg.Workspace.VCenterIP, ctx)
	if err != nil {
		klog.Errorf("Failed to get vSphere instance: %+v", err)
		return nil, err
	}

	// get the hosts mounted on this datastore
	// datastore -> ["host-1", "host-2", "host-3", ...]
	dsHosts, err := datastore.GetDatastoreHostMounts(ctx)
	if err != nil {
		klog.Errorf("Failed to get datastore host mounts for %v: %+v", datastore, err)
		return nil, err
	}
	klog.V(4).Infof("Got host mounts for datastore: %v: %v", datastore, dsHosts)

	// compute map of zone to list of hosts in that zone across all hosts in vsphere
	// zone -> ["host-i", "host-j", "host-k", ...]
	zoneToHosts, err := vs.GetZoneToHosts(ctx, vsi)
	if err != nil {
		klog.Errorf("Failed to get zones for hosts: %+v", err)
		return nil, err
	}
	klog.V(4).Infof("Got zone to hosts: %v", zoneToHosts)

	// datastore belongs to a zone if all hosts in that zone mount that datastore
	dsZones := make([]cloudprovider.Zone, 0)
	for zone, zoneHosts := range zoneToHosts {
		// if zone is valid and zoneHosts is a subset of dsHosts, then add zone
		if zone.Region != "" && containsAll(dsHosts, zoneHosts) {
			dsZones = append(dsZones, zone)
		}
	}
	klog.V(4).Infof("Datastore %s belongs to zones: %v", datastore, dsZones)
	return dsZones, nil
}

// GetZoneToHosts returns a map of 'zone' -> 'list of hosts in that zone' in given VC
func (vs *VSphere) GetZoneToHosts(ctx context.Context, vsi *VSphereInstance) (map[cloudprovider.Zone][]vmwaretypes.ManagedObjectReference, error) {
	// Approach is to find tags with the category of 'vs.cfg.Labels.Zone'
	zoneToHosts := make(map[cloudprovider.Zone][]vmwaretypes.ManagedObjectReference)

	getHostsInTagCategory := func(ctx context.Context, tagCategoryName string) (map[vmwaretypes.ManagedObjectReference]string, error) {

		hostToTag := make(map[vmwaretypes.ManagedObjectReference]string)
		err := withTagsClient(ctx, vsi.conn, func(c *rest.Client) error {
			// Look whether the zone/region tag is defined in VC
			tagManager := tags.NewManager(c)
			tagsForCat, err := tagManager.GetTagsForCategory(ctx, tagCategoryName)
			if err != nil {
				klog.V(4).Infof("No tags with category %s exists in VC. So ignoring.", tagCategoryName)
				// return a special error so that tag unavailability can be ignored
				return ErrNoZoneTagInVC
			}
			klog.V(4).Infof("List of tags under category %s: %v", tagCategoryName, tagsForCat)

			// Each such tag is a different 'zone' marked in vCenter.
			// Query for objects associated with each tag. Consider Host, Cluster and Datacenter kind of objects.
			tagToObjects := make(map[string][]mo.Reference)
			for _, tag := range tagsForCat {
				klog.V(4).Infof("Getting objects associated with tag %s", tag.Name)
				objects, err := tagManager.ListAttachedObjects(ctx, tag.Name)
				if err != nil {
					klog.Errorf("Error fetching objects associated with zone tag %s: %+v", tag.Name, err)
					return err
				}
				tagToObjects[tag.Name] = objects
			}
			klog.V(4).Infof("Map of tag to objects: %v", tagToObjects)

			// Infer zone for hosts within Datacenter, hosts within clusters and hosts - in this order of increasing priority
			// The below nested for-loops goes over all the objects in tagToObjects three times over.
			for _, moType := range []string{vclib.DatacenterType, vclib.ClusterComputeResourceType, vclib.HostSystemType} {
				for tagName, objects := range tagToObjects {
					for _, obj := range objects {
						if obj.Reference().Type == moType {
							klog.V(4).Infof("Found zone tag %s associated with %s of type %T: %s", tagName, obj, obj, obj.Reference().Value)
							switch moType {
							case "Datacenter":
								// mark that all hosts in this datacenter has tag applied
								dcObjRef := object.NewReference(vsi.conn.Client, obj.Reference())
								klog.V(4).Infof("Converted mo obj %v to govmomi object ref %v", obj, dcObjRef)
								dcObj, ok := dcObjRef.(*object.Datacenter)
								if !ok {
									errMsg := fmt.Sprintf("Not able to convert object to Datacenter %v", obj)
									klog.Errorf(errMsg)
									return errors.New(errMsg)
								}
								klog.V(4).Infof("Converted to object Datacenter %v", dcObj)
								dc := vclib.Datacenter{Datacenter: dcObj}
								hosts, err := dc.GetAllHosts(ctx)
								if err != nil {
									klog.Errorf("Could not get hosts from datacenter %v: %+v", dc, err)
									return err
								}
								for _, host := range hosts {
									hostToTag[host] = tagName
								}
							case "ClusterComputeResource":
								// mark that all hosts in this cluster has tag applied
								clusterObjRef := object.NewReference(vsi.conn.Client, obj.Reference())
								clusterObj, ok := clusterObjRef.(*object.ClusterComputeResource)
								if !ok {
									errMsg := fmt.Sprintf("Not able to convert object ClusterComputeResource %v", obj)
									klog.Errorf(errMsg)
									return errors.New(errMsg)
								}
								hostSystemList, err := clusterObj.Hosts(ctx)
								if err != nil {
									klog.Errorf("Not able to get hosts in cluster %v: %+v", clusterObj, err)
									return err
								}
								for _, host := range hostSystemList {
									hostToTag[host.Reference()] = tagName
								}
							case "HostSystem":
								// mark that this host has tag applied
								hostToTag[obj.Reference()] = tagName
							}
						}
					}
				}
			}
			return nil // no error
		})
		if err != nil {
			klog.Errorf("Error processing tag category %s: %+v", tagCategoryName, err)
			return nil, err
		}
		klog.V(6).Infof("Computed hostToTag: %v", hostToTag)
		return hostToTag, nil
	}

	hostToZone, err := getHostsInTagCategory(ctx, vs.cfg.Labels.Zone)
	if err != nil {
		if err == ErrNoZoneTagInVC {
			return zoneToHosts, nil
		}
		klog.Errorf("Get hosts in tag category %s failed: %+v", vs.cfg.Labels.Zone, err)
		return nil, err
	}

	hostToRegion, err := getHostsInTagCategory(ctx, vs.cfg.Labels.Region)
	if err != nil {
		if err == ErrNoZoneTagInVC {
			return zoneToHosts, nil
		}
		klog.Errorf("Get hosts in tag category %s failed: %+v", vs.cfg.Labels.Region, err)
		return nil, err
	}

	// populate zoneToHosts based on hostToZone and hostToRegion
	klog.V(6).Infof("hostToZone: %v", hostToZone)
	klog.V(6).Infof("hostToRegion: %v", hostToRegion)
	for host, zone := range hostToZone {
		region, regionExists := hostToRegion[host]
		if !regionExists {
			klog.Errorf("Host %s has a zone, but no region. So ignoring.", host)
			continue
		}
		cpZone := cloudprovider.Zone{FailureDomain: zone, Region: region}
		hosts, exists := zoneToHosts[cpZone]
		if !exists {
			hosts = make([]vmwaretypes.ManagedObjectReference, 0)
		}
		zoneToHosts[cpZone] = append(hosts, host)
	}
	klog.V(4).Infof("Final zoneToHosts: %v", zoneToHosts)
	return zoneToHosts, nil
}

// returns true if s1 contains all elements from s2; false otherwise
func containsAll(s1 []vmwaretypes.ManagedObjectReference, s2 []vmwaretypes.ManagedObjectReference) bool {
	// put all elements of s1 into a map
	s1Map := make(map[vmwaretypes.ManagedObjectReference]bool)
	for _, mor := range s1 {
		s1Map[mor] = true
	}
	// verify if all elements of s2 are present in s1Map
	for _, mor := range s2 {
		if _, found := s1Map[mor]; !found {
			return false
		}
	}
	return true
}
