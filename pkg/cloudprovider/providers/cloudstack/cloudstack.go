/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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

package cloudstack

import (
	"fmt"
	"github.com/golang/glog"
	"github.com/xanzy/go-cloudstack/cloudstack"
	"gopkg.in/gcfg.v1"
	"io"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/cloudprovider"
)

const ProviderName = "cloudstack"

type Config struct {
	Global struct {
		APIUrl    string `gcfg:"api-url"`
		APIKey    string `gcfg:"api-key"`
		SecretKey string `gcfg:"secret-key"`
		VerifySSL bool   `gcfg:"verify-ssl"`
	}
}

// CSCloud is an implementation of cloud provider Interface for CloudStack.
type CSCloud struct {
	client *cloudstack.CloudStackClient
	// InstanceID of the server where this CloudStack object is instantiated.
	localInstanceID string
}

func init() {
	cloudprovider.RegisterCloudProvider(ProviderName, func(config io.Reader) (cloudprovider.Interface, error) {
		cfg, err := readConfig(config)
		if err != nil {
			return nil, err
		}
		return newCSCloud(cfg)
	})
}

func readConfig(config io.Reader) (Config, error) {
	if config == nil {
		err := fmt.Errorf("no cloud provider config given")
		return Config{}, err
	}

	cfg := Config{}
	if err := gcfg.ReadInto(&cfg, config); err != nil {
		glog.Errorf("Couldn't parse config: %v", err)
		return Config{}, err
	}

	return cfg, nil
}

// newCSCloud creates a new instance of CSCloud
func newCSCloud(cfg Config) (*CSCloud, error) {
	client := cloudstack.NewAsyncClient(cfg.Global.APIUrl, cfg.Global.APIKey, cfg.Global.SecretKey, cfg.Global.VerifySSL)

	id, err := readInstanceID()
	if err != nil {
		return nil, err
	}

	cs := CSCloud{
		client:          client,
		localInstanceID: id,
	}

	return &cs, nil
}

func readInstanceID() (string, error) {
	// TODO: get instanceID from virtual router metadata
	return "", nil
}

// LoadBalancer returns an implementation of LoadBalancer for CloudStack.
func (cs *CSCloud) LoadBalancer() (cloudprovider.LoadBalancer, bool) {
	glog.V(4).Info("cloudstack.LoadBalancer() called")
	return &LoadBalancer{cs}, true
}

func (cs *CSCloud) Clusters() (cloudprovider.Clusters, bool) {
	return nil, false
}

func (cs *CSCloud) Instances() (cloudprovider.Instances, bool) {
	return &Instances{cs}, true
}

func (cs *CSCloud) Routes() (cloudprovider.Routes, bool) {
	return nil, false
}

func (cs *CSCloud) Zones() (cloudprovider.Zones, bool) {
	return cs, true
}

func (cs *CSCloud) ProviderName() string {
	return ProviderName
}

// ScrubDNS filters DNS settings for pods.
func (cs *CSCloud) ScrubDNS(nameservers, searches []string) (nsOut, srchOut []string) {
	return nameservers, searches
}

func (i *Instances) AddSSHKeyToAllInstances(user string, keyData []byte) error {
	return fmt.Errorf("unimplemented")
}

func (cs *CSCloud) GetZone() (cloudprovider.Zone, error) {
	glog.V(1).Infof("Current zone is null")

	return cloudprovider.Zone{Region: ""}, nil
}

func (i *Instances) CurrentNodeName(hostname string) (string, error) {
	return hostname, nil
}

// ExternalID returns the cloud provider ID of the specified instance (deprecated).
func (i *Instances) ExternalID(name string) (string, error) {
	var lb LoadBalancer
	var hosts []string
	hosts = append(hosts, name)
	vmIDs, err := lb.getVirtualMachineIds(hosts)
	if err != nil {
		return "", err
	}
	return vmIDs[0], nil
}

// InstanceID returns the cloud provider ID of the specified instance.
// Note that if the instance does not exist or is no longer running, we must return ("", cloudprovider.InstanceNotFound)
func (i *Instances) InstanceID(name string) (string, error) {
	var lb LoadBalancer
	var hosts []string
	hosts = append(hosts, name)
	vmIDs, err := lb.getVirtualMachineIds(hosts)
	if err != nil {
		return "", cloudprovider.InstanceNotFound
	}
	return vmIDs[0], nil
}

// InstanceType returns the type of the specified instance.
func (i *Instances) InstanceType(name string) (string, error) {
	return "", nil
}

// List lists instances that match 'filter' which is a regular expression which must match the entire instance name (fqdn)
func (i *Instances) List(name_filter string) ([]string, error) {
	vmParams := i.cs.client.VirtualMachine.NewListVirtualMachinesParams()
	vmParams.SetName(name_filter)
	vmParamsResponse, err := i.cs.client.VirtualMachine.ListVirtualMachines(vmParams)
	if err != nil {
		return nil, err
	}
	var vms []string
	for _, vm := range vmParamsResponse.VirtualMachines {
		vms = append(vms, vm.Name)
	}
	return vms, nil
}

// NodeAddresses returns the addresses of the specified instance.
// TODO(roberthbailey): This currently is only used in such a way that it
// returns the address of the calling instance. We should do a rename to
// make this clearer.
func (i *Instances) NodeAddresses(name string) ([]api.NodeAddress, error) {
	vmParams := i.cs.client.VirtualMachine.NewListVirtualMachinesParams()
	vmParams.SetName(name)
	vmParamsResponse, err := i.cs.client.VirtualMachine.ListVirtualMachines(vmParams)
	if err != nil {
		return nil, err
	}

	addrs := []api.NodeAddress{}
	publicIP := vmParamsResponse.VirtualMachines[0].Publicip
	addrs = append(addrs, api.NodeAddress{Type: api.NodeExternalIP, Address: publicIP})

	for _, nic := range vmParamsResponse.VirtualMachines[0].Nic {
		addrs = append(addrs, api.NodeAddress{Type: api.NodeInternalIP, Address: nic.Ipaddress})
		addrs = append(addrs, api.NodeAddress{Type: api.NodeLegacyHostIP, Address: nic.Ipaddress})
	}

	return addrs, nil
}

type LoadBalancer struct {
	cs *CSCloud
}

type Instances struct {
	cs *CSCloud
}

func (lb *LoadBalancer) GetLoadBalancer(apiService *api.Service) (*api.LoadBalancerStatus, bool, error) {
	loadBalancerName := cloudprovider.GetLoadBalancerName(apiService)
	loadBalancer, _, err := lb.cs.client.LoadBalancer.GetLoadBalancerByName(loadBalancerName)

	if err != nil {
		return nil, false, nil
	}

	vip := loadBalancer.Sourceipaddress
	status := &api.LoadBalancerStatus{}
	status.Ingress = []api.LoadBalancerIngress{{IP: vip}}

	return status, true, err
}

func (lb *LoadBalancer) EnsureLoadBalancer(apiService *api.Service, hosts []string) (*api.LoadBalancerStatus, error) {
	glog.V(4).Infof("EnsureLoadBalancer(%v, %v, %v, %v, %v, %v)", apiService.Namespace, apiService.Name, apiService.Spec.LoadBalancerIP, apiService.Spec.Ports, hosts)

	glog.V(2).Infof("Checking if CloudStack load balancer already exists: %s", cloudprovider.GetLoadBalancerName(apiService))
	_, exists, err := lb.GetLoadBalancer(apiService)
	if err != nil {
		return nil, fmt.Errorf("error checking if CloudStack load balancer already exists: %v", err)
	}

	// TODO: Implement a more efficient update strategy for common changes than delete & create
	if exists {
		err := lb.EnsureLoadBalancerDeleted(apiService)
		if err != nil {
			return nil, fmt.Errorf("error deleting existing CloudStack load balancer: %v", err)
		}
	}

	//Config algorithm for the new LB
	var algorithm string
	switch apiService.Spec.SessionAffinity {
	case api.ServiceAffinityNone:
		algorithm = "roundrobin"
	case api.ServiceAffinityClientIP:
		algorithm = "source"
	default:
		return nil, fmt.Errorf("unsupported load balancer affinity: %v", apiService.Spec.SessionAffinity)
	}

	//Get public IP address will be associated to the new LB
	lbIpAddr := apiService.Spec.LoadBalancerIP
	if lbIpAddr == "" {
		return nil, fmt.Errorf("unsupported service without predefined Load Balancer IPaddress")
	}
	publicIpId, err := lb.getPublicIpId(lbIpAddr)
	if err != nil {
		return nil, fmt.Errorf("error getting public IP address information for creating CloudStack load balancer")
	}

	//Config name for new LB
	lbName := apiService.ObjectMeta.Name
	if lbName == "" {
		return nil, fmt.Errorf("name is a required field for a CloudStack load balancer")
	}

	ports := apiService.Spec.Ports
	if len(ports) == 0 {
		return nil, fmt.Errorf("no ports provided to CloudStack load balancer")
	}

	//support multiple ports
	for _, port := range ports {
		//Init a new LB configuration
		lbParams := lb.cs.client.LoadBalancer.NewCreateLoadBalancerRuleParams(
			algorithm,
			lbName,
			int(port.NodePort),
			int(port.Port),
		)

		//Config protocol for new LB
		switch port.Protocol {
		case api.ProtocolTCP:
			lbParams.SetProtocol("TCP")
		case api.ProtocolUDP:
			lbParams.SetProtocol("UDP")
		}

		//Config LB IP
		lbParams.SetPublicipid(publicIpId)

		//Do not create corresponding firewall rule
		lbParams.SetOpenfirewall(false)

		// create a Load Balancer rule
		createLBRuleResponse, err := lb.cs.client.LoadBalancer.CreateLoadBalancerRule(lbParams)
		if err != nil {
			return nil, err
		}

		// associate vms to new LB
		assignLbParams := lb.cs.client.LoadBalancer.NewAssignToLoadBalancerRuleParams(createLBRuleResponse.Id)
		vmIds, err := lb.getVirtualMachineIds(hosts)
		if err != nil {
			return nil, fmt.Errorf("error getting list of vms associated with CloudStack load balancer")
		}
		assignLbParams.SetVirtualmachineids(vmIds)
		assignLBRuleResponse, err := lb.cs.client.LoadBalancer.AssignToLoadBalancerRule(assignLbParams)

		if err != nil || !assignLBRuleResponse.Success {
			return nil, err
		}
	}

	status := &api.LoadBalancerStatus{}
	status.Ingress = []api.LoadBalancerIngress{{IP: lbIpAddr}}

	return status, nil
}

func (lb *LoadBalancer) UpdateLoadBalancer(apiService *api.Service, hosts []string) error {
	loadBalancerName := cloudprovider.GetLoadBalancerName(apiService)
	glog.V(4).Infof("UpdateLoadBalancer(%v, %v)", loadBalancerName, hosts)

	lbParams := lb.cs.client.LoadBalancer.NewListLoadBalancerRulesParams()

	//Get new list of vms associated with LB of service
	//Set of member (addresses) that _should_ exist
	vmIds, err := lb.getVirtualMachineIds(hosts)
	if err != nil {
		return fmt.Errorf("error getting list of vms associated with CloudStack load balancer")
	}
	vms := map[string]bool{}
	for _, vmId := range vmIds {
		vms[vmId] = true
	}

	//Now get the current list of vms. And then make comparison to update the list.
	//Public IPaddress associated with LB of service
	lbIpAddr := apiService.Spec.LoadBalancerIP
	if lbIpAddr == "" {
		return fmt.Errorf("unsupported service without predefined Load Balancer IPaddress")
	}

	//list all LB rules associated with this public IPaddress
	publicIpId, err := lb.getPublicIpId(lbIpAddr)
	if err != nil {
		return fmt.Errorf("error getting public IP address information for creating CloudStack load balancer")
	}
	lbParams.SetPublicipid(publicIpId)
	lbRulesResponse, err := lb.cs.client.LoadBalancer.ListLoadBalancerRules(lbParams)
	if err != nil {
		return err
	}
	lbRuleId := lbRulesResponse.LoadBalancerRules[0].Id
	lbInstancesParams := lb.cs.client.LoadBalancer.NewListLoadBalancerRuleInstancesParams(lbRuleId)
	lbInstancesParams.SetLbvmips(true)

	//list out all VMs currently associated to this LB
	lbInstancesResponse, err := lb.cs.client.LoadBalancer.ListLoadBalancerRuleInstances(lbInstancesParams)
	if err != nil {
		return err
	}

	var oldvmIds []string
	for _, lbInstance := range lbInstancesResponse.LoadBalancerRuleInstances {
		oldvmIds = append(oldvmIds, lbInstance.Loadbalancerruleinstance.Id)
	}

	//Compare two list of vms to thus update LB
	var removedVmIds []string
	for _, oldvmId := range oldvmIds {
		if _, found := vms[oldvmId]; found {
			delete(vms, oldvmId)
		} else {
			removedVmIds = append(removedVmIds, oldvmId)
		}
	}

	//remove old vms from all LB rules associated with the public IP
	for _, lbRule := range lbRulesResponse.LoadBalancerRules {
		removeFromLbRuleParams := lb.cs.client.LoadBalancer.NewRemoveFromLoadBalancerRuleParams(lbRule.Id)
		removeFromLbRuleParams.SetVirtualmachineids(removedVmIds)
		_, err := lb.cs.client.LoadBalancer.RemoveFromLoadBalancerRule(removeFromLbRuleParams)
		if err != nil {
			return err
		}
	}

	//assign new vms (the rest of vms map) to all LB rules associated with the public IP
	var assignVmIds []string
	for vm := range vms {
		assignVmIds = append(assignVmIds, vm)
	}

	for _, lbRule := range lbRulesResponse.LoadBalancerRules {
		assignToLbRuleParams := lb.cs.client.LoadBalancer.NewAssignToLoadBalancerRuleParams(lbRule.Id)
		assignToLbRuleParams.SetVirtualmachineids(assignVmIds)
		_, err := lb.cs.client.LoadBalancer.AssignToLoadBalancerRule(assignToLbRuleParams)
		if err != nil {
			return err
		}
	}
	return nil
}

func (lb *LoadBalancer) EnsureLoadBalancerDeleted(apiService *api.Service) error {
	loadBalancerName := cloudprovider.GetLoadBalancerName(apiService)
	glog.V(4).Infof("EnsureLoadBalancerDeleted(%v)", loadBalancerName)

	lbIpAddr := apiService.Spec.LoadBalancerIP
	if lbIpAddr != "" {
		//list all LB rules associated to this public ipaddress.
		listLBParams := lb.cs.client.LoadBalancer.NewListLoadBalancerRulesParams()
		publicIpId, err := lb.getPublicIpId(lbIpAddr)
		if err != nil {
			return fmt.Errorf("error getting public IP address information for creating CloudStack load balancer")
		}
		listLBParams.SetPublicipid(publicIpId)
		listLoadBalancerResponse, err := lb.cs.client.LoadBalancer.ListLoadBalancerRules(listLBParams)
		if err != nil {
			return err
		}
		lbRules := listLoadBalancerResponse.LoadBalancerRules

		//delete all found load balancer rules associated to this public ipaddress.
		for _, lbRule := range lbRules {
			lbParams := lb.cs.client.LoadBalancer.NewDeleteLoadBalancerRuleParams(lbRule.Id)
			_, err := lb.cs.client.LoadBalancer.DeleteLoadBalancerRule(lbParams)
			if err != nil {
				return err
			}
		}
	} else {
		//only support delete load balancer with existing IP address
		return nil
	}

	return nil
}

func (lb *LoadBalancer) getPublicIpId(lbIP string) (string, error) {
	addressParams := lb.cs.client.Address.NewListPublicIpAddressesParams()
	addressParams.SetIpaddress(lbIP)
	addressResponse, err := lb.cs.client.Address.ListPublicIpAddresses(addressParams)
	if err != nil {
		return "", err
	}

	if addressResponse.Count > 1 {
		return "", fmt.Errorf("Found more than one address objects with IP = %s", lbIP)
	} else if addressResponse.Count == 0 {
		//TODO: acquire new IP address with lbIP from CloudStack
	}

	return addressResponse.PublicIpAddresses[0].Id, nil
}

func (lb *LoadBalancer) getVirtualMachineIds(hosts []string) ([]string, error) {
	var vmIDs []string
	ipAddrs := map[string]bool{}
	for _, host := range hosts {
		ipAddrs[host] = true
	}

	//list all vms
	listVMParams := lb.cs.client.VirtualMachine.NewListVirtualMachinesParams()
	listVMParams.SetListall(true)
	listVMResponse, err := lb.cs.client.VirtualMachine.ListVirtualMachines(listVMParams)
	if err != nil {
		return nil, err
	}

	//check if ipaddress belongs to the hosts slice, then add the corresponding vmid
	for i := 0; i < listVMResponse.Count; i++ {
		//check only the first Nic
		ipAddr := listVMResponse.VirtualMachines[i].Nic[0].Ipaddress
		if _, found := ipAddrs[ipAddr]; found {
			vmIDs = append(vmIDs, listVMResponse.VirtualMachines[i].Id)
		}
	}

	return vmIDs, nil
}
