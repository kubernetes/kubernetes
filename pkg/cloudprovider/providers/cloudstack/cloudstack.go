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
		       AuthUrl    string `gcfg:"auth-url"`
		       ApiKey     string `gcfg:"api-key"`
		       SecretKey  string `gcfg:"secret-key"`
		       VerifySSL  string `gcfg:"verify-ssl"`
		       Region     string
	       }
	LoadBalancer LoadBalancerOpts
}

type LoadBalancerOpts struct {
	Name			string		`gcfg:"name"`
	Algorithm		string		`gcfg:"algorithm"`
	InstancePort		int		`gcfg:"instance-port"`
	NetworkID		string		`gcfg:"network-id"`
	Scheme			string 		`gcfg:"scheme"`
	SourceIPNetworkID	string		`gcfg:"source-ip-network-id"`
	SourcePort		int		`gcfg:"source-port"`
}

// CloudStack is an implementation of cloud provider Interface for CloudStack.
type CloudStack struct {
	provider *cloudstack.CloudStackClient
	region   string
	lbOpts   LoadBalancerOpts
	// InstanceID of the server where this CloudStack object is instantiated.
	localInstanceID string
}

func init() {
	cloudprovider.RegisterCloudProvider(ProviderName, func(config io.Reader) (cloudprovider.Interface, error) {
		cfg, err := readConfig(config)
		if err != nil {
			return nil, err
		}
		return newCloudStack(cfg)
	})
}

func readConfig(config io.Reader) (Config, error) {
	if config == nil {
		err := fmt.Errorf("no CloudStack cloud provider config file given")
		return Config{}, err
	}

	var cfg Config
	err := gcfg.ReadInto(&cfg, config)
	return cfg, err
}

func newCloudStack(cfg Config) (*CloudStack, error) {
	provider := cloudstack.NewClient(cfg.Global.AuthUrl, cfg.Global.ApiKey, cfg.Global.SecretKey, cfg.Global.VerifySSL)
	if provider == nil {
		return nil, nil
	}
	fmt.Println(provider)

	id, err := readInstanceID()
	if err != nil {
		return nil, err
	}

	cs := CloudStack{
		provider:        provider,
		region:          cfg.Global.Region,
		lbOpts:          cfg.LoadBalancer,
		localInstanceID: id,
	}

	return &cs, nil
}

func readInstanceID() (string, error) {
	// TODO: get instanceID from virtual router metadata
	return nil, nil
}

// LoadBalancer returns an implementation of LoadBalancer for CloudStack.
func (cs *CloudStack) LoadBalancer() (cloudprovider.LoadBalancer, bool) {
	glog.V(4).Info("cloudstack.LoadBalancer() called")
	return cs, true
}

// Instances returns an implementation of Instances for CloudStack.
func (cs *CloudStack) Instances() (cloudprovider.Instances, bool) {
	glog.V(4).Info("cloudstack.Instances() called")
	return cs, true
}

// Routes returns an implementation of Routes for CloudStack.
func (cs *CloudStack) Routes() (cloudprovider.Routes, bool) {
	glog.V(4).Info("cloudstack.Routes() called")
	return cs, true
}

// Zones returns an implementation of Zones for CloudStack.
func (cs *CloudStack) Zones() (cloudprovider.Zones, bool) {
	glog.V(4).Info("cloudstack.Zones() called")
	return cs, true
}

type LoadBalancer struct {
	cs	*CloudStack
}

func (lb *LoadBalancer) GetLoadBalancer(service *api.Service) (*api.LoadBalancerStatus, bool, error) {
	loadBalancerName := cloudprovider.GetLoadBalancerName(service)
	loadBalancerService := cloudstack.LoadBalancerService{lb.cs.provider}
	loadBalancer, _, err := loadBalancerService.GetLoadBalancerByName(loadBalancerName)

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
		return nil, fmt.Errorf("Source range restrictions are not supported for cloudstack load balancers")
	}

	glog.V(2).Infof("Checking if cloudstack load balancer already exists: %s", cloudprovider.GetLoadBalancerName(apiService))
	_, exists, err := lb.GetLoadBalancer(apiService)
	if err != nil {
		return nil, fmt.Errorf("error checking if cloudstack load balancer already exists: %v", err)
	}

	// TODO: Implement a more efficient update strategy for common changes than delete & create
	// In particular, if we implement hosts update, we can get rid of UpdateHosts
	if exists {
		err := lb.EnsureLoadBalancerDeleted(apiService)
		if err != nil {
			return nil, fmt.Errorf("error deleting existing openstack load balancer: %v", err)
		}
	}

	//init Cloudstack LB client
	lbService := cloudstack.LoadBalancerService{}

	//Init a new LB configuration
	lbParams := cloudstack.CreateLoadBalancerRuleParams{}

	//Config LB IP
	lbIpAddr := apiService.Spec.LoadBalancerIP
	if lbIpAddr == nil {
		return nil, fmt.Errorf("unsupported service without predefined Load Balancer IPaddress")
	}
	publicIpId := getPublicIpId(lbIpAddr)
	lbParams.SetPublicipid(publicIpId)

	//Config algorithm for new LB
	affinity := apiService.Spec.SessionAffinity
	switch affinity {
	case api.ServiceAffinityNone:
		lbParams.SetAlgorithm("roundrobin")
	case api.ServiceAffinityClientIP:
	// TODO
	default:
		return nil, fmt.Errorf("unsupported load balancer affinity: %v", affinity)
	}

	//Config name for new LB
	if apiService.ObjectMeta.Name != nil {
		lbParams.SetName(apiService.ObjectMeta.Name)
	}

	ports := apiService.Spec.Ports
	if len(ports) == 0 {
		return nil, fmt.Errorf("no ports provided to cloudstack load balancer")
	}

	//support multiple ports
	for _, port := range ports {
		//Config protocol for new LB
		lbParams.SetProtocol(port.Protocol)

		//Config ports for new LB
		lbParams.SetPublicport(port.Port)
		lbParams.SetPrivateport(port.NodePort)

		// create a Load Balancer rule
		createLBRuleResponse, err := lbService.CreateLoadBalancerRule(lbParams)
		if err != nil {
			return nil, err
		}

		// associate vms to new LB
		vmIds := getVirtualMachineIds(hosts)
		lbRuleId := createLBRuleResponse.Id

		assignLbParams := cloudstack.AssignToLoadBalancerRuleParams{}
		assignLbParams.SetId(lbRuleId)
		assignLbParams.SetVirtualmachineids(vmIds)
		assignLBRuleResponse, err := lbService.AssignToLoadBalancerRule(assignLbParams)

		if err != nil {
			return nil, err
		}

		if assignLBRuleResponse.Success == false {
			return nil, err
		}
	}

	status := &api.LoadBalancerStatus{}
	status.Ingress = []api.LoadBalancerIngress{{IP: lbIpAddr}}

	return status, nil
}

func (lb *LoadBalancer) UpdateLoadBalancer(service *api.Service, hosts []string) error {
	loadBalancerName := cloudprovider.GetLoadBalancerName(service)
	glog.V(4).Infof("UpdateLoadBalancer(%v, %v)", loadBalancerName, hosts)

	//init Cloudstack LB client
	lbService := cloudstack.LoadBalancerService{}

	//Init a new LB configuration
	lbParams := cloudstack.ListLoadBalancerRulesParams{}

	//Get new list of vms associated with LB of service
	//Set of member (addresses) that _should_ exist
	vmIds := getVirtualMachineIds(hosts)
	vms := map[string]bool{}
	for _, vmId := range vmIds {
		vms[vmId] = true
	}

	//Now get the current list of vms. And then make comparison to update the list.
	//Public IPaddress associated with LB of service
	lbIpAddr := service.Spec.LoadBalancerIP
	if lbIpAddr == nil {
		return nil, fmt.Errorf("unsupported service without predefined Load Balancer IPaddress")
	}
	publicIpId := getPublicIpId(lbIpAddr)

	//list all LB rules associated with this public IPaddress
	lbParams.SetPublicipid(publicIpId)
	lbRulesResponse, err := lbService.ListLoadBalancerRules(lbParams)
	if err != nil {
		return err
	}
	lbRuleId := lbRulesResponse.LoadBalancerRules[0].Id
	lbInstancesParams := cloudstack.ListLoadBalancerRuleInstancesParams{}
	lbInstancesParams.SetId(lbRuleId)
	lbInstancesParams.SetLbvmips(true)

	//TODO - lbInstancesResponse object doesn't fit to real output when making the API call on terminal
	//lbInstancesResponse.LoadBalancerRuleInstances[0].
	//OUTPUT: oldvmIds
	lbInstancesResponse, err := lbService.ListLoadBalancerRuleInstances(lbInstancesParams)
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
	removeFromLbRuleParams := cloudstack.RemoveFromLoadBalancerRuleParams{}
	for _, lbRule := range lbRulesResponse.LoadBalancerRules {
		removeFromLbRuleParams.SetId(lbRule.Id)
		removeFromLbRuleParams.SetVirtualmachineids(removedVmIds)
		_, err := lbService.RemoveFromLoadBalancerRule(removeFromLbRuleParams)
		if err != nil {
			return err
		}
	}

	//assign new vms (the rest of vms map) to all LB rules associated with the public IP
	var assignVmIds []string
	for vm := range vms {
		assignVmIds = append(assignVmIds, vm)
	}
	assignToLbRuleParams := cloudstack.AssignToLoadBalancerRuleParams{}
	for _, lbRule := range lbRulesResponse.LoadBalancerRules {
		assignToLbRuleParams.SetId(lbRule.Id)
		assignToLbRuleParams.SetVirtualmachineids(assignVmIds)
		_, err := lbService.AssignToLoadBalancerRule(assignToLbRuleParams)
		if err != nil {
			return err
		}
	}
	return nil
}

func (lb *LoadBalancer) EnsureLoadBalancerDeleted(service *api.Service) error {
	loadBalancerName := cloudprovider.GetLoadBalancerName(service)
	glog.V(4).Infof("EnsureLoadBalancerDeleted(%v)", loadBalancerName)

	//init cloudstack LB client
	lbService := cloudstack.LoadBalancerService{}

	lbParams := cloudstack.DeleteLoadBalancerRuleParams{}
	lbIpAddr := service.Spec.LoadBalancerIP
	if lbIpAddr != nil {
		//get ID of public ipaddress
		publicIpId := getPublicIpId(lbIpAddr)

		//list all LB rules associated to this public ipaddress.
		listLBParams := cloudstack.ListLoadBalancerRulesParams{}
		listLBParams.SetPublicipid(publicIpId)
		listLoadBalancerResponse, err := lbService.ListLoadBalancerRules(listLBParams)
		if err != nil {
			return err
		}
		lbRules := listLoadBalancerResponse.LoadBalancerRules

		//delete all found load balancer rules associated to this public ipaddress.
		for _, lbRule := range lbRules {
			lbParams.SetId(lbRule.Id)
			_, err := lbService.DeleteLoadBalancerRule(lbParams)
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