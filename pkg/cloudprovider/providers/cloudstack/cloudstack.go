package cloudstack

import (
	"fmt"
	"io"
	"github.com/scalingdata/gcfg"
	"github.com/kubernetes/kubernetes/pkg/cloudprovider"
	"github.com/xanzy/go-cloudstack/cloudstack"
	"k8s.io/kubernetes/pkg/api"
	//"github.com/kubernetes/kubernetes/pkg/api"
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

	//TODO
}

func (lb *LoadBalancer) UpdateLoadBalancer(service *api.Service, hosts []string) error {
	loadBalancerName := cloudprovider.GetLoadBalancerName(service)
	glog.V(4).Infof("UpdateLoadBalancer(%v, %v)", loadBalancerName, hosts)

	//TODO
}

func (lb *LoadBalancer) EnsureLoadBalancerDeleted(service *api.Service) error {
	loadBalancerName := cloudprovider.GetLoadBalancerName(service)
	glog.V(4).Infof("EnsureLoadBalancerDeleted(%v)", loadBalancerName)

	//TODO
}