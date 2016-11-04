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

// This version of Photon cloud provider supports the disk interface
// for Photon persistent disk volume plugin. LoadBalancer, Routes, and
// Zones are currently not supported.
// The use of Photon cloud provider requires to start kubelet, kube-apiserver,
// and kube-controller-manager with config flag: '--cloud-provider=photon
// --cloud-config=[path_to_config_file]'. When running multi-node kubernetes
// using docker, the config file should be located inside /etc/kubernetes.
package photon

import (
	"errors"
	"fmt"
	"github.com/golang/glog"
	"github.com/vmware/photon-controller-go-sdk/photon"
	"gopkg.in/gcfg.v1"
	"io"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/cloudprovider"
	k8stypes "k8s.io/kubernetes/pkg/types"
	"log"
	"os/exec"
	"strings"
)

const (
	ProviderName = "photon"
	DiskSpecKind = "persistent-disk"
)

// Global variable pointing to photon client
var photonClient *photon.Client
var logger *log.Logger = nil

// overrideIP indicates if the hostname is overriden by IP address, such as when
// running multi-node kubernetes using docker. In this case the user should set
// overrideIP = true in cloud config file. Default value is false.
var overrideIP bool = false

// Photon is an implementation of the cloud provider interface for Photon Controller.
type PCCloud struct {
	cfg *PCConfig
	// InstanceID of the server where this PCCloud object is instantiated.
	localInstanceID string
	// local $HOSTNAME
	localHostname string
	// hostname from K8S, could be overridden
	localK8sHostname string
	// Photon project ID. We assume that there is only one Photon Controller project
	// in the environment per current Photon Controller deployment methodology.
	projID string
	cloudprovider.Zone
}

type PCConfig struct {
	Global struct {
		// the Photon Controller endpoint IP address
		CloudTarget string `gcfg:"target"`
		// when the Photon Controller authentication is enabled, set to true;
		// otherwise, set to false.
		IgnoreCertificate bool `gcfg:"ignoreCertificate"`
		// Photon Controller tenant name
		Tenant string `gcfg:"tenant"`
		// Photon Controller project name
		Project string `gcfg:"project"`
		// when kubelet is started with '--hostname-override=${IP_ADDRESS}', set to true;
		// otherwise, set to false.
		OverrideIP bool `gcfg:"overrideIP"`
	}
}

// Disks is interface for manipulation with PhotonController Persistent Disks.
type Disks interface {
	// AttachDisk attaches given disk to given node. Current node
	// is used when nodeName is empty string.
	AttachDisk(pdID string, nodeName k8stypes.NodeName) error

	// DetachDisk detaches given disk to given node. Current node
	// is used when nodeName is empty string.
	DetachDisk(pdID string, nodeName k8stypes.NodeName) error

	// DiskIsAttached checks if a disk is attached to the given node.
	DiskIsAttached(pdID string, nodeName k8stypes.NodeName) (bool, error)

	// DisksAreAttached is a batch function to check if a list of disks are attached
	// to the node with the specified NodeName.
	DisksAreAttached(pdIDs []string, nodeName k8stypes.NodeName) (map[string]bool, error)

	// CreateDisk creates a new PD with given properties.
	CreateDisk(volumeOptions *VolumeOptions) (pdID string, err error)

	// DeleteDisk deletes PD.
	DeleteDisk(pdID string) error
}

// VolumeOptions specifies capacity, tags, name and flavorID for a volume.
type VolumeOptions struct {
	CapacityGB int
	Tags       map[string]string
	Name       string
	Flavor     string
}

func readConfig(config io.Reader) (PCConfig, error) {
	if config == nil {
		err := fmt.Errorf("cloud provider config file is missing. Please restart kubelet with --cloud-provider=photon --cloud-config=[path_to_config_file]")
		return PCConfig{}, err
	}

	var cfg PCConfig
	err := gcfg.ReadInto(&cfg, config)
	return cfg, err
}

func init() {
	cloudprovider.RegisterCloudProvider(ProviderName, func(config io.Reader) (cloudprovider.Interface, error) {
		cfg, err := readConfig(config)
		if err != nil {
			glog.Errorf("Photon Cloud Provider: failed to read in cloud provider config file. Error[%v]", err)
			return nil, err
		}
		return newPCCloud(cfg)
	})
}

// Retrieve the Photon VM ID from the Photon Controller endpoint based on the node name
func getVMIDbyNodename(project string, nodeName string) (string, error) {
	vmList, err := photonClient.Projects.GetVMs(project, nil)
	if err != nil {
		glog.Errorf("Photon Cloud Provider: Failed to GetVMs from project %s with nodeName %s, error: [%v]", project, nodeName, err)
		return "", err
	}

	for _, vm := range vmList.Items {
		if vm.Name == nodeName {
			return vm.ID, nil
		}
	}

	return "", fmt.Errorf("No matching started VM is found with name %s", nodeName)
}

// Retrieve the Photon VM ID from the Photon Controller endpoint based on the IP address
func getVMIDbyIP(project string, IPAddress string) (string, error) {
	vmList, err := photonClient.Projects.GetVMs(project, nil)
	if err != nil {
		glog.Errorf("Photon Cloud Provider: Failed to GetVMs for project %s. error: [%v]", project, err)
		return "", err
	}

	for _, vm := range vmList.Items {
		task, err := photonClient.VMs.GetNetworks(vm.ID)
		if err != nil {
			glog.Warningf("Photon Cloud Provider: GetNetworks failed for vm.ID %s, error [%v]", vm.ID, err)
		} else {
			task, err = photonClient.Tasks.Wait(task.ID)
			if err != nil {
				glog.Warning("Photon Cloud Provider: Wait task for GetNetworks failed for vm.ID %s, error [%v]", vm.ID, err)
			} else {
				networkConnections := task.ResourceProperties.(map[string]interface{})
				networks := networkConnections["networkConnections"].([]interface{})
				for _, nt := range networks {
					network := nt.(map[string]interface{})
					if val, ok := network["ipAddress"]; ok && val != nil {
						ipAddr := val.(string)
						if ipAddr == IPAddress {
							return vm.ID, nil
						}
					}
				}
			}
		}
	}

	return "", fmt.Errorf("No matching VM is found with IP %s", IPAddress)
}

// Retrieve the the Photon project ID from the Photon Controller endpoint based on the project name
func getProjIDbyName(tenantName, projName string) (string, error) {
	tenants, err := photonClient.Tenants.GetAll()
	if err != nil {
		glog.Errorf("Photon Cloud Provider: GetAll tenants failed with error [%v].", err)
		return "", err
	}

	for _, tenant := range tenants.Items {
		if tenant.Name == tenantName {
			projects, err := photonClient.Tenants.GetProjects(tenant.ID, nil)
			if err != nil {
				glog.Errorf("Photon Cloud Provider: Failed to GetProjects for tenant %s. error [%v]", tenantName, err)
				return "", err
			}

			for _, project := range projects.Items {
				if project.Name == projName {
					return project.ID, nil
				}
			}
		}
	}

	return "", fmt.Errorf("No matching tenant/project name is found with %s/%s", tenantName, projName)
}

func newPCCloud(cfg PCConfig) (*PCCloud, error) {
	if len(cfg.Global.CloudTarget) == 0 {
		return nil, fmt.Errorf("Photon Controller endpoint was not specified.")
	}

	// Currently we support Photon Controller endpoint with authentication disabled.
	options := &photon.ClientOptions{
		IgnoreCertificate: cfg.Global.IgnoreCertificate,
	}

	photonClient = photon.NewClient(cfg.Global.CloudTarget, options, logger)
	status, err := photonClient.Status.Get()
	if err != nil {
		glog.Errorf("Photon Cloud Provider: new client creation failed. Error[%v]", err)
		return nil, err
	}
	glog.V(2).Info("Photon Cloud Provider: Status of the new photon controller client: %v", status)

	// Get Photon Controller project ID for future use
	projID, err := getProjIDbyName(cfg.Global.Tenant, cfg.Global.Project)
	if err != nil {
		glog.Errorf("Photon Cloud Provider: getProjIDbyName failed when creating new Photon Controller client. Error[%v]", err)
		return nil, err
	}

	// Get local hostname for localInstanceID
	cmd := exec.Command("bash", "-c", `echo $HOSTNAME`)
	out, err := cmd.CombinedOutput()
	if err != nil {
		glog.Errorf("Photon Cloud Provider: get local hostname bash command failed. Error[%v]", err)
		return nil, err
	}
	if len(out) == 0 {
		glog.Errorf("unable to retrieve hostname for Instance ID")
		return nil, fmt.Errorf("unable to retrieve hostname for Instance ID")
	}
	hostname := strings.TrimRight(string(out), "\n")
	vmID, err := getVMIDbyNodename(projID, hostname)
	if err != nil {
		glog.Errorf("Photon Cloud Provider: getVMIDbyNodename failed when creating new Photon Controller client. Error[%v]", err)
		return nil, err
	}

	pc := PCCloud{
		cfg:              &cfg,
		localInstanceID:  vmID,
		localHostname:    hostname,
		localK8sHostname: "",
		projID:           projID,
	}

	overrideIP = cfg.Global.OverrideIP

	return &pc, nil
}

// Instances returns an implementation of Instances for Photon Controller.
func (pc *PCCloud) Instances() (cloudprovider.Instances, bool) {
	return pc, true
}

// List is an implementation of Instances.List.
func (pc *PCCloud) List(filter string) ([]k8stypes.NodeName, error) {
	return nil, nil
}

// NodeAddresses is an implementation of Instances.NodeAddresses.
func (pc *PCCloud) NodeAddresses(nodeName k8stypes.NodeName) ([]api.NodeAddress, error) {
	addrs := []api.NodeAddress{}
	name := string(nodeName)

	var vmID string
	var err error
	if name == pc.localK8sHostname {
		vmID = pc.localInstanceID
	} else {
		vmID, err = getInstanceID(name, pc.projID)
		if err != nil {
			glog.Errorf("Photon Cloud Provider: getInstanceID failed for NodeAddresses. Error[%v]", err)
			return addrs, err
		}
	}

	// Retrieve the Photon VM's IP addresses from the Photon Controller endpoint based on the VM ID
	vmList, err := photonClient.Projects.GetVMs(pc.projID, nil)
	if err != nil {
		glog.Errorf("Photon Cloud Provider: Failed to GetVMs for project %s. Error[%v]", pc.projID, err)
		return addrs, err
	}

	for _, vm := range vmList.Items {
		if vm.ID == vmID {
			task, err := photonClient.VMs.GetNetworks(vm.ID)
			if err != nil {
				glog.Errorf("Photon Cloud Provider: GetNetworks failed for node %s with vm.ID %s. Error[%v]", name, vm.ID, err)
				return addrs, err
			} else {
				task, err = photonClient.Tasks.Wait(task.ID)
				if err != nil {
					glog.Errorf("Photon Cloud Provider: Wait task for GetNetworks failed for node %s with vm.ID %s. Error[%v]", name, vm.ID, err)
					return addrs, err
				} else {
					networkConnections := task.ResourceProperties.(map[string]interface{})
					networks := networkConnections["networkConnections"].([]interface{})
					for _, nt := range networks {
						network := nt.(map[string]interface{})
						if val, ok := network["ipAddress"]; ok && val != nil {
							ipAddr := val.(string)
							if ipAddr != "-" {
								api.AddToNodeAddresses(&addrs,
									api.NodeAddress{
										// TODO: figure out the type of the IP
										Type:    api.NodeInternalIP,
										Address: ipAddr,
									},
								)
							}
						}
					}
					return addrs, nil
				}
			}
		}
	}

	glog.Errorf("Failed to find the node %s from Photon Controller endpoint", name)
	return addrs, fmt.Errorf("Failed to find the node %s from Photon Controller endpoint", name)
}

func (pc *PCCloud) AddSSHKeyToAllInstances(user string, keyData []byte) error {
	return errors.New("unimplemented")
}

func (pc *PCCloud) CurrentNodeName(hostname string) (k8stypes.NodeName, error) {
	pc.localK8sHostname = hostname
	return k8stypes.NodeName(hostname), nil
}

func getInstanceID(name string, projID string) (string, error) {
	var vmID string
	var err error

	if overrideIP == true {
		vmID, err = getVMIDbyIP(projID, name)
	} else {
		vmID, err = getVMIDbyNodename(projID, name)
	}
	if err != nil {
		return "", err
	}

	if vmID == "" {
		err = cloudprovider.InstanceNotFound
	}

	return vmID, err
}

// ExternalID returns the cloud provider ID of the specified instance (deprecated).
func (pc *PCCloud) ExternalID(nodeName k8stypes.NodeName) (string, error) {
	name := string(nodeName)
	if name == pc.localK8sHostname {
		return pc.localInstanceID, nil
	} else {
		ID, err := getInstanceID(name, pc.projID)
		if err != nil {
			glog.Errorf("Photon Cloud Provider: getInstanceID failed for ExternalID. Error[%v]", err)
			return ID, err
		} else {
			return ID, nil
		}
	}
}

// InstanceID returns the cloud provider ID of the specified instance.
func (pc *PCCloud) InstanceID(nodeName k8stypes.NodeName) (string, error) {
	name := string(nodeName)
	if name == pc.localK8sHostname {
		return pc.localInstanceID, nil
	} else {
		ID, err := getInstanceID(name, pc.projID)
		if err != nil {
			glog.Errorf("Photon Cloud Provider: getInstanceID failed for InstanceID. Error[%v]", err)
			return ID, err
		} else {
			return ID, nil
		}
	}
}

func (pc *PCCloud) InstanceType(nodeName k8stypes.NodeName) (string, error) {
	return "", nil
}

func (pc *PCCloud) Clusters() (cloudprovider.Clusters, bool) {
	return nil, true
}

// ProviderName returns the cloud provider ID.
func (pc *PCCloud) ProviderName() string {
	return ProviderName
}

// LoadBalancer returns an implementation of LoadBalancer for Photon Controller.
func (pc *PCCloud) LoadBalancer() (cloudprovider.LoadBalancer, bool) {
	return nil, false
}

// Zones returns an implementation of Zones for Photon Controller.
func (pc *PCCloud) Zones() (cloudprovider.Zones, bool) {
	return pc, true
}

func (pc *PCCloud) GetZone() (cloudprovider.Zone, error) {
	return pc.Zone, nil
}

// Routes returns a false since the interface is not supported for photon controller.
func (pc *PCCloud) Routes() (cloudprovider.Routes, bool) {
	return nil, false
}

// ScrubDNS filters DNS settings for pods.
func (pc *PCCloud) ScrubDNS(nameservers, searches []string) (nsOut, srchOut []string) {
	return nameservers, searches
}

// Attaches given virtual disk volume to the compute running kubelet.
func (pc *PCCloud) AttachDisk(pdID string, nodeName k8stypes.NodeName) error {
	operation := &photon.VmDiskOperation{
		DiskID: pdID,
	}

	vmID, err := pc.InstanceID(nodeName)
	if err != nil {
		glog.Errorf("Photon Cloud Provider: pc.InstanceID failed for AttachDisk. Error[%v]", err)
		return err
	}

	task, err := photonClient.VMs.AttachDisk(vmID, operation)
	if err != nil {
		glog.Errorf("Photon Cloud Provider: Failed to attach disk with pdID %s. Error[%v]", pdID, err)
		return err
	}

	_, err = photonClient.Tasks.Wait(task.ID)
	if err != nil {
		glog.Errorf("Photon Cloud Provider: Failed to wait for task to attach disk with pdID %s. Error[%v]", pdID, err)
		return err
	}

	return nil
}

// Detaches given virtual disk volume from the compute running kubelet.
func (pc *PCCloud) DetachDisk(pdID string, nodeName k8stypes.NodeName) error {
	operation := &photon.VmDiskOperation{
		DiskID: pdID,
	}

	vmID, err := pc.InstanceID(nodeName)
	if err != nil {
		glog.Errorf("Photon Cloud Provider: pc.InstanceID failed for DetachDisk. Error[%v]", err)
		return err
	}

	task, err := photonClient.VMs.DetachDisk(vmID, operation)
	if err != nil {
		glog.Errorf("Photon Cloud Provider: Failed to detach disk with pdID %s. Error[%v]", pdID, err)
		return err
	}

	_, err = photonClient.Tasks.Wait(task.ID)
	if err != nil {
		glog.Errorf("Photon Cloud Provider: Failed to wait for task to detach disk with pdID %s. Error[%v]", pdID, err)
		return err
	}

	return nil
}

// DiskIsAttached returns if disk is attached to the VM using controllers supported by the plugin.
func (pc *PCCloud) DiskIsAttached(pdID string, nodeName k8stypes.NodeName) (bool, error) {
	disk, err := photonClient.Disks.Get(pdID)
	if err != nil {
		glog.Errorf("Photon Cloud Provider: Failed to Get disk with pdID %s. Error[%v]", pdID, err)
		return false, err
	}

	vmID, err := pc.InstanceID(nodeName)
	if err != nil {
		glog.Errorf("Photon Cloud Provider: pc.InstanceID failed for DiskIsAttached. Error[%v]", err)
		return false, err
	}

	for _, vm := range disk.VMs {
		if strings.Compare(vm, vmID) == 0 {
			return true, nil
		}
	}

	return false, nil
}

// DisksAreAttached returns if disks are attached to the VM using controllers supported by the plugin.
func (pc *PCCloud) DisksAreAttached(pdIDs []string, nodeName k8stypes.NodeName) (map[string]bool, error) {
	attached := make(map[string]bool)
	for _, pdID := range pdIDs {
		attached[pdID] = false
	}

	vmID, err := pc.InstanceID(nodeName)
	if err != nil {
		glog.Errorf("Photon Cloud Provider: pc.InstanceID failed for DiskIsAttached. Error[%v]", err)
		return attached, err
	}

	for _, pdID := range pdIDs {
		disk, err := photonClient.Disks.Get(pdID)
		if err != nil {
			glog.Warningf("Photon Cloud Provider: failed to get VMs for persistent disk %s, err [%v]", pdID, err)
		} else {
			for _, vm := range disk.VMs {
				if vm == vmID {
					attached[pdID] = true
				}
			}
		}
	}

	return attached, nil
}

// Create a volume of given size (in GB).
func (pc *PCCloud) CreateDisk(volumeOptions *VolumeOptions) (pdID string, err error) {
	diskSpec := photon.DiskCreateSpec{}
	diskSpec.Name = volumeOptions.Name
	diskSpec.Flavor = volumeOptions.Flavor
	diskSpec.CapacityGB = volumeOptions.CapacityGB
	diskSpec.Kind = DiskSpecKind

	task, err := photonClient.Projects.CreateDisk(pc.projID, &diskSpec)
	if err != nil {
		glog.Errorf("Photon Cloud Provider: Failed to CreateDisk. Error[%v]", err)
		return "", err
	}

	waitTask, err := photonClient.Tasks.Wait(task.ID)
	if err != nil {
		glog.Errorf("Photon Cloud Provider: Failed to wait for task to CreateDisk. Error[%v]", err)
		return "", err
	}

	return waitTask.Entity.ID, nil
}

// Deletes a volume given volume name.
func (pc *PCCloud) DeleteDisk(pdID string) error {
	task, err := photonClient.Disks.Delete(pdID)
	if err != nil {
		glog.Errorf("Photon Cloud Provider: Failed to DeleteDisk. Error[%v]", err)
		return err
	}

	_, err = photonClient.Tasks.Wait(task.ID)
	if err != nil {
		glog.Errorf("Photon Cloud Provider: Failed to wait for task to DeleteDisk. Error[%v]", err)
		return err
	}

	return nil
}
