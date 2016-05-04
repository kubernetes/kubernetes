package cloudstack

import (
	"fmt"
	"io"
	"github.com/scalingdata/gcfg"
	"github.com/kubernetes/kubernetes/pkg/cloudprovider"
	"github.com/xanzy/go-cloudstack/cloudstack"
	"k8s.io/kubernetes/pkg/api"
	//"github.com/kubernetes/kubernetes/pkg/api"
	"k8s.io/kubernetes/kubernetes/pkg/api/service"
	//"github.com/kubernetes/kubernetes/pkg/api/service"
	"github.com/golang/glog"
)

const ProviderName = "cloudstack"

type Config struct {
	Global struct {
		       APIUrl     string `gcfg:"api-url"`
		       APIKey     string `gcfg:"api-key"`
		       SecretKey  string `gcfg:"secret-key"`
		       VerifySSL  string `gcfg:"verify-ssl"`
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
		return nil, err
	}

	cfg := &Config{}
	if err := gcfg.ReadInto(cfg, config); err != nil {
		glog.Errorf("Couldn't parse config: %v", err)
		return nil, err
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
		client:        client,
		localInstanceID: id,
	}

	return &cs, nil
}

func readInstanceID() (string, error) {
	// TODO: get instanceID from virtual router metadata
	return nil, nil
}

// LoadBalancer returns an implementation of LoadBalancer for CloudStack.
func (cs *CSCloud) LoadBalancer() (cloudprovider.LoadBalancer, bool) {
	glog.V(4).Info("cloudstack.LoadBalancer() called")
	return cs, true
}

// Instances returns an implementation of Instances for CloudStack.
func (cs *CSCloud) Instances() (cloudprovider.Instances, bool) {
	glog.V(4).Info("cloudstack.Instances() called")
	return cs, true
}

// Routes returns an implementation of Routes for CloudStack.
func (cs *CSCloud) Routes() (cloudprovider.Routes, bool) {
	glog.V(4).Info("cloudstack.Routes() called")
	return cs, true
}

// Zones returns an implementation of Zones for CloudStack.
func (cs *CSCloud) Zones() (cloudprovider.Zones, bool) {
	glog.V(4).Info("cloudstack.Zones() called")
	return cs, true
}

type LoadBalancer struct {
	cs	*CSCloud
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

func (lb *LoadBalancer) EnsureLoadBalancer(apiService *api.Service, hosts []string, annotations map[string]string) (*api.LoadBalancerStatus, error) {
	glog.V(4).Infof("EnsureLoadBalancer(%v, %v, %v, %v, %v, %v)", apiService.Namespace, apiService.Name, apiService.Spec.LoadBalancerIP, apiService.Spec.Ports, hosts, annotations)

	sourceRanges, err := service.GetLoadBalancerSourceRanges(annotations)
	if err != nil {
		return nil, err
	}

	if !service.IsAllowAll(sourceRanges) {
		return nil, fmt.Errorf("Source range restrictions are not supported for CloudStack load balancers")
	}

	glog.V(2).Infof("Checking if CloudStack load balancer already exists: %s", cloudprovider.GetLoadBalancerName(apiService))
	_, exists, err := lb.GetLoadBalancer(apiService)
	if err != nil {
		return nil, fmt.Errorf("error checking if CloudStack load balancer already exists: %v", err)
	}

	// TODO: Implement a more efficient update strategy for common changes than delete & create
	// In particular, if we implement hosts update, we can get rid of UpdateHosts
	if exists {
		err := lb.EnsureLoadBalancerDeleted(apiService)
		if err != nil {
			return nil, fmt.Errorf("error deleting existing openstack load balancer: %v", err)
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
	if lbIpAddr == nil {
		return nil, fmt.Errorf("unsupported service without predefined Load Balancer IPaddress")
	}
	publicIpId := getPublicIpId(lbIpAddr)

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
			port.NodePort,
			port.Port,
		)

		//Config protocol for new LB
		lbParams.SetProtocol(port.Protocol)

		//Config LB IP
		lbParams.SetPublicipid(publicIpId)

		// create a Load Balancer rule
		createLBRuleResponse, err := lb.cs.client.LoadBalancer.CreateLoadBalancerRule(lbParams)
		if err != nil {
			return nil, err
		}

		// associate vms to new LB
		assignLbParams := lb.cs.client.LoadBalancer.NewAssignToLoadBalancerRuleParams(createLBRuleResponse.Id)
		assignLbParams.SetVirtualmachineids(getVirtualMachineIds(hosts))
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
	vmIds := getVirtualMachineIds(hosts)
	vms := map[string]bool{}
	for _, vmId := range vmIds {
		vms[vmId] = true
	}

	//Now get the current list of vms. And then make comparison to update the list.
	//Public IPaddress associated with LB of service
	lbIpAddr := apiService.Spec.LoadBalancerIP
	if lbIpAddr == nil {
		return nil, fmt.Errorf("unsupported service without predefined Load Balancer IPaddress")
	}

	//list all LB rules associated with this public IPaddress
	lbParams.SetPublicipid(getPublicIpId(lbIpAddr))
	lbRulesResponse, err := lb.cs.client.LoadBalancer.ListLoadBalancerRules(lbParams)
	if err != nil {
		return err
	}
	lbRuleId := lbRulesResponse.LoadBalancerRules[0].Id
	lbInstancesParams := lb.cs.client.LoadBalancer.NewListLoadBalancerRuleInstancesParams(lbRuleId)
	lbInstancesParams.SetLbvmips(true)

	//TODO - lbInstancesResponse object doesn't fit to real output when making the API call on terminal
	//lbInstancesResponse.LoadBalancerRuleInstances[0].
	//OUTPUT: oldvmIds
	lbInstancesResponse, err := lb.cs.client.LoadBalancer.ListLoadBalancerRuleInstances(lbInstancesParams)
	if err != nil {
		return err
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
	if lbIpAddr != nil {
		//list all LB rules associated to this public ipaddress.
		listLBParams := lb.cs.client.LoadBalancer.NewListLoadBalancerRulesParams()
		listLBParams.SetPublicipid(getPublicIpId(lbIpAddr))
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

func getPublicIpId(lbIp string) string {
	//TODO
}

func getVirtualMachineIds(hosts []string) []string {
	//TODO
}