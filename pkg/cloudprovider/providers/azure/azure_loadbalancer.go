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

package azure

import (
	"fmt"
	"math"
	"strconv"
	"strings"

	"k8s.io/api/core/v1"
	utilerrors "k8s.io/apimachinery/pkg/util/errors"
	"k8s.io/apimachinery/pkg/util/sets"
	serviceapi "k8s.io/kubernetes/pkg/api/v1/service"

	"github.com/Azure/azure-sdk-for-go/arm/compute"
	"github.com/Azure/azure-sdk-for-go/arm/network"
	"github.com/Azure/go-autorest/autorest/to"
	"github.com/golang/glog"
	"k8s.io/apimachinery/pkg/types"
)

// ServiceAnnotationLoadBalancerInternal is the annotation used on the service
const ServiceAnnotationLoadBalancerInternal = "service.beta.kubernetes.io/azure-load-balancer-internal"

// ServiceAnnotationLoadBalancerInternalSubnet is the annotation used on the service
// to specify what subnet it is exposed on
const ServiceAnnotationLoadBalancerInternalSubnet = "service.beta.kubernetes.io/azure-load-balancer-internal-subnet"

// ServiceAnnotationLoadBalancerMode is the annotation used on the service to specify the
// Azure load balancer selection based on availability sets
// There are currently three possible load balancer selection modes :
// 1. Default mode - service has no annotation ("service.beta.kubernetes.io/azure-load-balancer-mode")
//	  In this case the Loadbalancer of the primary Availability set is selected
// 2. "__auto__" mode - service is annotated with __auto__ value, this when loadbalancer from any availability set
//    is selected which has the miinimum rules associated with it.
// 3. "as1,as2" mode - this is when the laod balancer from the specified availability sets is selected that has the
//    miinimum rules associated with it.
const ServiceAnnotationLoadBalancerMode = "service.beta.kubernetes.io/azure-load-balancer-mode"

// ServiceAnnotationLoadBalancerAutoModeValue the annotation used on the service to specify the
// Azure load balancer auto selection from the availability sets
const ServiceAnnotationLoadBalancerAutoModeValue = "__auto__"

// ServiceAnnotationDNSLabelName annotation speficying the DNS label name for the service.
const ServiceAnnotationDNSLabelName = "service.beta.kubernetes.io/azure-dns-label-name"

// GetLoadBalancer returns whether the specified load balancer exists, and
// if so, what its status is.
func (az *Cloud) GetLoadBalancer(clusterName string, service *v1.Service) (status *v1.LoadBalancerStatus, exists bool, err error) {
	_, status, exists, err = az.getServiceLoadBalancer(service, clusterName, nil, false)
	if err != nil {
		return nil, false, err
	}
	if exists == false {
		serviceName := getServiceName(service)
		glog.V(5).Infof("getloadbalancer (cluster:%s) (service:%s)- IP doesn't exist in any of the lbs", clusterName, serviceName)
		return nil, false, fmt.Errorf("Service(%s) - Loadbalancer not found", serviceName)
	}
	return status, true, nil
}

func getPublicIPLabel(service *v1.Service) string {
	if labelName, found := service.Annotations[ServiceAnnotationDNSLabelName]; found {
		return labelName
	}
	return ""
}

// EnsureLoadBalancer creates a new load balancer 'name', or updates the existing one. Returns the status of the balancer
func (az *Cloud) EnsureLoadBalancer(clusterName string, service *v1.Service, nodes []*v1.Node) (*v1.LoadBalancerStatus, error) {
	// When a client updates the internal load balancer annotation,
	// the service may be switched from an internal LB to a public one, or vise versa.
	// Here we'll firstly ensure service do not lie in the opposite LB.
	serviceName := getServiceName(service)
	glog.V(5).Infof("ensureloadbalancer(%s): START clusterName=%q", serviceName, clusterName)
	flippedService := flipServiceInternalAnnotation(service)
	if _, err := az.reconcileLoadBalancer(clusterName, flippedService, nil, false /* wantLb */); err != nil {
		return nil, err
	}

	if _, err := az.reconcilePublicIP(clusterName, service, true /* wantLb */); err != nil {
		return nil, err
	}

	lb, err := az.reconcileLoadBalancer(clusterName, service, nodes, true /* wantLb */)
	if err != nil {
		return nil, err
	}

	lbStatus, err := az.getServiceLoadBalancerStatus(service, lb)
	if err != nil {
		return nil, err
	}

	if _, err := az.reconcileSecurityGroup(clusterName, service, lbStatus, true /* wantLb */); err != nil {
		return nil, err
	}

	return lbStatus, nil
}

// UpdateLoadBalancer updates hosts under the specified load balancer.
func (az *Cloud) UpdateLoadBalancer(clusterName string, service *v1.Service, nodes []*v1.Node) error {
	_, err := az.EnsureLoadBalancer(clusterName, service, nodes)
	return err
}

// EnsureLoadBalancerDeleted deletes the specified load balancer if it
// exists, returning nil if the load balancer specified either didn't exist or
// was successfully deleted.
// This construction is useful because many cloud providers' load balancers
// have multiple underlying components, meaning a Get could say that the LB
// doesn't exist even if some part of it is still laying around.
func (az *Cloud) EnsureLoadBalancerDeleted(clusterName string, service *v1.Service) error {
	serviceName := getServiceName(service)
	glog.V(5).Infof("delete(%s): START clusterName=%q", serviceName, clusterName)
	if _, err := az.reconcileSecurityGroup(clusterName, service, nil, false /* wantLb */); err != nil {
		return err
	}

	if _, err := az.reconcileLoadBalancer(clusterName, service, nil, false /* wantLb */); err != nil {
		return err
	}

	if _, err := az.reconcilePublicIP(clusterName, service, false /* wantLb */); err != nil {
		return err
	}

	glog.V(2).Infof("delete(%s): FINISH", serviceName)
	return nil
}

// getServiceLoadBalancer gets the loadbalancer for the service if it already exists
// If wantLb is TRUE then -it selects a new load balancer
// In case the selected load balancer does not exists it returns network.LoadBalancer struct
// with added metadata (such as name, location) and existsLB set to FALSE
// By default - cluster default LB is returned
func (az *Cloud) getServiceLoadBalancer(service *v1.Service, clusterName string, nodes []*v1.Node, wantLb bool) (lb *network.LoadBalancer, status *v1.LoadBalancerStatus, exists bool, err error) {
	isInternal := requiresInternalLoadBalancer(service)
	var defaultLB *network.LoadBalancer
	defaultLBName := az.getLoadBalancerName(clusterName, az.Config.PrimaryAvailabilitySetName, isInternal)

	existingLBs, err := az.ListLBWithRetry()
	if err != nil {
		return nil, nil, false, err
	}

	// check if the service already has a load balancer
	if existingLBs != nil {
		for _, existingLB := range existingLBs {
			if strings.EqualFold(*existingLB.Name, defaultLBName) {
				defaultLB = &existingLB
			}
			if isInternalLoadBalancer(&existingLB) != isInternal {
				continue
			}
			status, err = az.getServiceLoadBalancerStatus(service, &existingLB)
			if err != nil {
				return nil, nil, false, err
			}
			if status == nil {
				// service is not om this load balancer
				continue
			}

			return &existingLB, status, true, nil
		}
	}

	// service does not have a load balancer, select one
	if wantLb {
		// select new load balancer for service
		selectedLB, exists, err := az.selectLoadBalancer(clusterName, service, &existingLBs, nodes)
		if err != nil {
			return nil, nil, false, err
		}

		return selectedLB, nil, exists, err
	}

	// create a default LB with meta data if not present
	if defaultLB == nil {
		defaultLB = &network.LoadBalancer{
			Name:                         &defaultLBName,
			Location:                     &az.Location,
			LoadBalancerPropertiesFormat: &network.LoadBalancerPropertiesFormat{},
		}
	}

	return defaultLB, nil, false, nil
}

// select load balancer for the service in the cluster
// the selection algorithm selectes the the load balancer with currently has
// the minimum lb rules, there there are multiple LB's with same number of rules
// it selects the first one (sorted based on name)
func (az *Cloud) selectLoadBalancer(clusterName string, service *v1.Service, existingLBs *[]network.LoadBalancer, nodes []*v1.Node) (selectedLB *network.LoadBalancer, existsLb bool, err error) {
	isInternal := requiresInternalLoadBalancer(service)
	serviceName := getServiceName(service)
	glog.V(3).Infof("selectLoadBalancer(%s): isInternal(%s) - start", serviceName, isInternal)
	availabilitySetNames, err := az.getLoadBalancerAvailabilitySetNames(service, nodes)
	if err != nil {
		glog.Errorf("az.selectLoadBalancer: cluster(%s) service(%s) isInternal(%t) - az.getLoadBalancerAvailabilitySetNames failed, err=(%v)", clusterName, serviceName, isInternal, err)
		return nil, false, err
	}
	glog.Infof("selectLoadBalancer: cluster(%s) service(%s) isInternal(%t) - availabilitysetsnames %v", clusterName, serviceName, isInternal, *availabilitySetNames)
	mapExistingLBs := map[string]network.LoadBalancer{}
	for _, lb := range *existingLBs {
		mapExistingLBs[*lb.Name] = lb
	}
	selectedLBRuleCount := math.MaxInt32
	for _, currASName := range *availabilitySetNames {
		currLBName := az.getLoadBalancerName(clusterName, currASName, isInternal)
		lb, exists := mapExistingLBs[currLBName]
		if !exists {
			// select this LB as this is a new LB and will have minimum rules
			// create tmp lb struct to hold metadata for the new load-balancer
			selectedLB = &network.LoadBalancer{
				Name:                         &currLBName,
				Location:                     &az.Location,
				LoadBalancerPropertiesFormat: &network.LoadBalancerPropertiesFormat{},
			}

			return selectedLB, false, nil
		}

		lbRules := *lb.LoadBalancingRules
		currLBRuleCount := 0
		if lbRules != nil {
			currLBRuleCount = len(lbRules)
		}
		if currLBRuleCount < selectedLBRuleCount {
			selectedLBRuleCount = currLBRuleCount
			selectedLB = &lb
		}
	}

	if selectedLB == nil {
		err = fmt.Errorf("selectLoadBalancer: cluster(%s) service(%s) isInternal(%t) - unable to find load balancer for selected availability sets %v", clusterName, serviceName, isInternal, *availabilitySetNames)
		glog.Error(err)
		return nil, false, err
	}
	// validate if the selected LB has not exceeded the MaximumLoadBalancerRuleCount
	if az.Config.MaximumLoadBalancerRuleCount != 0 && selectedLBRuleCount >= az.Config.MaximumLoadBalancerRuleCount {
		err = fmt.Errorf("selectLoadBalancer: cluster(%s) service(%s) isInternal(%t) -  all available load balancers have exceeded maximum rule limit %d, availabilitysetnames (%v)", clusterName, serviceName, isInternal, selectedLBRuleCount, *availabilitySetNames)
		glog.Error(err)
		return selectedLB, existsLb, err
	}

	return selectedLB, existsLb, nil
}

func (az *Cloud) getServiceLoadBalancerStatus(service *v1.Service, lb *network.LoadBalancer) (status *v1.LoadBalancerStatus, err error) {
	if lb == nil {
		glog.V(10).Infof("getServiceLoadBalancerStatus lb is nil")
		return nil, nil
	}
	if lb.FrontendIPConfigurations == nil || *lb.FrontendIPConfigurations == nil {
		return nil, nil
	}
	isInternal := requiresInternalLoadBalancer(service)
	lbFrontendIPConfigName := getFrontendIPConfigName(service, subnet(service))
	serviceName := getServiceName(service)
	for _, ipConfiguration := range *lb.FrontendIPConfigurations {
		if lbFrontendIPConfigName == *ipConfiguration.Name {
			var lbIP *string
			if isInternal {
				lbIP = ipConfiguration.PrivateIPAddress
			} else {
				if ipConfiguration.PublicIPAddress == nil {
					return nil, fmt.Errorf("get(%s): lb(%s) - failed to get LB PublicIPAddress is Nil", serviceName, *lb.Name)
				}
				pipID := ipConfiguration.PublicIPAddress.ID
				if pipID == nil {
					return nil, fmt.Errorf("get(%s): lb(%s) - failed to get LB PublicIPAddress ID is Nil", serviceName, *lb.Name)
				}
				pipName, err := getLastSegment(*pipID)
				if err != nil {
					return nil, fmt.Errorf("get(%s): lb(%s) - failed to get LB PublicIPAddress Name from ID(%s)", serviceName, *lb.Name, *pipID)
				}
				pip, existsPip, err := az.getPublicIPAddress(pipName)
				if err != nil {
					return nil, err
				}
				if existsPip {
					lbIP = pip.IPAddress
				}
			}

			return &v1.LoadBalancerStatus{Ingress: []v1.LoadBalancerIngress{{IP: *lbIP}}}, nil
		}
	}

	return nil, nil
}

func (az *Cloud) determinePublicIPName(clusterName string, service *v1.Service) (string, error) {
	loadBalancerIP := service.Spec.LoadBalancerIP
	if len(loadBalancerIP) == 0 {
		return getPublicIPName(clusterName, service), nil
	}

	pips, err := az.ListPIPWithRetry()
	if err != nil {
		return "", err
	}

	for _, pip := range pips {
		if pip.PublicIPAddressPropertiesFormat.IPAddress != nil &&
			*pip.PublicIPAddressPropertiesFormat.IPAddress == loadBalancerIP {
			return *pip.Name, nil
		}
	}
	return "", fmt.Errorf("user supplied IP Address %s was not found", loadBalancerIP)
}

func flipServiceInternalAnnotation(service *v1.Service) *v1.Service {
	copyService := service.DeepCopy()
	if v, ok := copyService.Annotations[ServiceAnnotationLoadBalancerInternal]; ok && v == "true" {
		// If it is internal now, we make it external by remove the annotation
		delete(copyService.Annotations, ServiceAnnotationLoadBalancerInternal)
	} else {
		// If it is external now, we make it internal
		copyService.Annotations[ServiceAnnotationLoadBalancerInternal] = "true"
	}
	return copyService
}

func (az *Cloud) ensurePublicIPExists(serviceName, pipName, domainNameLabel string) (*network.PublicIPAddress, error) {
	pip, existsPip, err := az.getPublicIPAddress(pipName)
	if err != nil {
		return nil, err
	}
	if existsPip {
		return &pip, nil
	}

	pip.Name = to.StringPtr(pipName)
	pip.Location = to.StringPtr(az.Location)
	pip.PublicIPAddressPropertiesFormat = &network.PublicIPAddressPropertiesFormat{
		PublicIPAllocationMethod: network.Static,
	}
	if len(domainNameLabel) > 0 {
		pip.PublicIPAddressPropertiesFormat.DNSSettings = &network.PublicIPAddressDNSSettings{
			DomainNameLabel: &domainNameLabel,
		}
	}
	pip.Tags = &map[string]*string{"service": &serviceName}
	glog.V(3).Infof("ensure(%s): pip(%s) - creating", serviceName, *pip.Name)
	az.operationPollRateLimiter.Accept()
	glog.V(10).Infof("PublicIPAddressesClient.CreateOrUpdate(%q): start", *pip.Name)
	respChan, errChan := az.PublicIPAddressesClient.CreateOrUpdate(az.ResourceGroup, *pip.Name, pip, nil)
	resp := <-respChan
	err = <-errChan
	glog.V(10).Infof("PublicIPAddressesClient.CreateOrUpdate(%q): end", *pip.Name)
	if az.CloudProviderBackoff && shouldRetryAPIRequest(resp.Response, err) {
		glog.V(2).Infof("ensure(%s) backing off: pip(%s) - creating", serviceName, *pip.Name)
		retryErr := az.CreateOrUpdatePIPWithRetry(pip)
		if retryErr != nil {
			glog.V(2).Infof("ensure(%s) abort backoff: pip(%s) - creating", serviceName, *pip.Name)
			err = retryErr
		}
	}
	if err != nil {
		return nil, err
	}

	az.operationPollRateLimiter.Accept()
	glog.V(10).Infof("PublicIPAddressesClient.Get(%q): start", *pip.Name)
	pip, err = az.PublicIPAddressesClient.Get(az.ResourceGroup, *pip.Name, "")
	glog.V(10).Infof("PublicIPAddressesClient.Get(%q): end", *pip.Name)
	if err != nil {
		return nil, err
	}

	return &pip, nil
}

// This ensures load balancer exists and the frontend ip config is setup.
// This also reconciles the Service's Ports  with the LoadBalancer config.
// This entails adding rules/probes for expected Ports and removing stale rules/ports.
// nodes only used if wantLb is true
func (az *Cloud) reconcileLoadBalancer(clusterName string, service *v1.Service, nodes []*v1.Node, wantLb bool) (*network.LoadBalancer, error) {
	isInternal := requiresInternalLoadBalancer(service)
	serviceName := getServiceName(service)
	glog.V(2).Infof("reconcileLoadBalancer(%s) - wantLb(%t): started", serviceName, wantLb)
	lb, _, _, err := az.getServiceLoadBalancer(service, clusterName, nodes, wantLb)
	if err != nil {
		return nil, err
	}
	lbName := *lb.Name
	glog.V(2).Infof("reconcileLoadBalancer(%s): lb(%s) wantLb(%t) resolved load balancer name", serviceName, lbName, wantLb)
	lbFrontendIPConfigName := getFrontendIPConfigName(service, subnet(service))
	lbFrontendIPConfigID := az.getFrontendIPConfigID(lbName, lbFrontendIPConfigName)
	lbBackendPoolName := getBackendPoolName(clusterName)
	lbBackendPoolID := az.getBackendPoolID(lbName, lbBackendPoolName)

	dirtyLb := false

	// Ensure LoadBalancer's Backend Pool Configuration
	if wantLb {
		newBackendPools := []network.BackendAddressPool{}
		if lb.BackendAddressPools != nil {
			newBackendPools = *lb.BackendAddressPools
		}

		foundBackendPool := false
		for _, bp := range newBackendPools {
			if strings.EqualFold(*bp.Name, lbBackendPoolName) {
				glog.V(10).Infof("reconcile(%s)(%t): lb backendpool - found wanted backendpool. not adding anything", serviceName, wantLb)
				foundBackendPool = true
				break
			} else {
				glog.V(10).Infof("reconcile(%s)(%t): lb backendpool - found other backendpool %s", serviceName, wantLb, *bp.Name)
			}
		}
		if !foundBackendPool {
			newBackendPools = append(newBackendPools, network.BackendAddressPool{
				Name: to.StringPtr(lbBackendPoolName),
			})
			glog.V(10).Infof("reconcile(%s)(%t): lb backendpool - adding backendpool", serviceName, wantLb)

			dirtyLb = true
			lb.BackendAddressPools = &newBackendPools
		}
	}

	// Ensure LoadBalancer's Frontend IP Configurations
	dirtyConfigs := false
	newConfigs := []network.FrontendIPConfiguration{}
	if lb.FrontendIPConfigurations != nil {
		newConfigs = *lb.FrontendIPConfigurations
	}

	if !wantLb {
		for i := len(newConfigs) - 1; i >= 0; i-- {
			config := newConfigs[i]
			if serviceOwnsFrontendIP(config, service) {
				glog.V(3).Infof("reconcile(%s)(%t): lb frontendconfig(%s) - dropping", serviceName, wantLb, lbFrontendIPConfigName)
				newConfigs = append(newConfigs[:i], newConfigs[i+1:]...)
				dirtyConfigs = true
			}
		}
	} else {
		if isInternal {
			for i := len(newConfigs) - 1; i >= 0; i-- {
				config := newConfigs[i]
				if serviceOwnsFrontendIP(config, service) && !strings.EqualFold(*config.Name, lbFrontendIPConfigName) {
					glog.V(3).Infof("reconcile(%s)(%t): lb frontendconfig(%s) - dropping", serviceName, wantLb, *config.Name)
					newConfigs = append(newConfigs[:i], newConfigs[i+1:]...)
					dirtyConfigs = true
				}
			}
		}
		foundConfig := false
		for _, config := range newConfigs {
			if strings.EqualFold(*config.Name, lbFrontendIPConfigName) {
				foundConfig = true
				break
			}
		}
		if !foundConfig {
			// construct FrontendIPConfigurationPropertiesFormat
			var fipConfigurationProperties *network.FrontendIPConfigurationPropertiesFormat
			if isInternal {
				subnetName := subnet(service)
				if subnetName == nil {
					subnetName = &az.SubnetName
				}
				subnet, existsSubnet, err := az.getSubnet(az.VnetName, *subnetName)
				if err != nil {
					return nil, err
				}

				if !existsSubnet {
					return nil, fmt.Errorf("ensure(%s): lb(%s) - failed to get subnet: %s/%s", serviceName, lbName, az.VnetName, az.SubnetName)
				}

				configProperties := network.FrontendIPConfigurationPropertiesFormat{
					Subnet: &subnet,
				}

				loadBalancerIP := service.Spec.LoadBalancerIP
				if loadBalancerIP != "" {
					configProperties.PrivateIPAllocationMethod = network.Static
					configProperties.PrivateIPAddress = &loadBalancerIP
				} else {
					// We'll need to call GetLoadBalancer later to retrieve allocated IP.
					configProperties.PrivateIPAllocationMethod = network.Dynamic
				}

				fipConfigurationProperties = &configProperties
			} else {
				pipName, err := az.determinePublicIPName(clusterName, service)
				if err != nil {
					return nil, err
				}
				domainNameLabel := getPublicIPLabel(service)
				pip, err := az.ensurePublicIPExists(serviceName, pipName, domainNameLabel)
				if err != nil {
					return nil, err
				}
				fipConfigurationProperties = &network.FrontendIPConfigurationPropertiesFormat{
					PublicIPAddress: &network.PublicIPAddress{ID: pip.ID},
				}
			}

			newConfigs = append(newConfigs,
				network.FrontendIPConfiguration{
					Name: to.StringPtr(lbFrontendIPConfigName),
					FrontendIPConfigurationPropertiesFormat: fipConfigurationProperties,
				})
			glog.V(10).Infof("reconcile(%s)(%t): lb frontendconfig(%s) - adding", serviceName, wantLb, lbFrontendIPConfigName)
			dirtyConfigs = true
		}
	}
	if dirtyConfigs {
		dirtyLb = true
		lb.FrontendIPConfigurations = &newConfigs
	}

	// update probes/rules
	var ports []v1.ServicePort
	if wantLb {
		ports = service.Spec.Ports
	} else {
		ports = []v1.ServicePort{}
	}

	var expectedProbes []network.Probe
	var expectedRules []network.LoadBalancingRule
	for _, port := range ports {
		lbRuleName := getLoadBalancerRuleName(service, port, subnet(service))

		transportProto, _, probeProto, err := getProtocolsFromKubernetesProtocol(port.Protocol)
		if err != nil {
			return nil, err
		}

		if serviceapi.NeedsHealthCheck(service) {
			if port.Protocol == v1.ProtocolUDP {
				// ERROR: this isn't supported
				// health check (aka source ip preservation) is not
				// compatible with UDP (it uses an HTTP check)
				return nil, fmt.Errorf("services requiring health checks are incompatible with UDP ports")
			}

			podPresencePath, podPresencePort := serviceapi.GetServiceHealthCheckPathPort(service)

			expectedProbes = append(expectedProbes, network.Probe{
				Name: &lbRuleName,
				ProbePropertiesFormat: &network.ProbePropertiesFormat{
					RequestPath:       to.StringPtr(podPresencePath),
					Protocol:          network.ProbeProtocolHTTP,
					Port:              to.Int32Ptr(podPresencePort),
					IntervalInSeconds: to.Int32Ptr(5),
					NumberOfProbes:    to.Int32Ptr(2),
				},
			})
		} else if port.Protocol != v1.ProtocolUDP {
			// we only add the expected probe if we're doing TCP
			expectedProbes = append(expectedProbes, network.Probe{
				Name: &lbRuleName,
				ProbePropertiesFormat: &network.ProbePropertiesFormat{
					Protocol:          *probeProto,
					Port:              to.Int32Ptr(port.NodePort),
					IntervalInSeconds: to.Int32Ptr(5),
					NumberOfProbes:    to.Int32Ptr(2),
				},
			})
		}

		loadDistribution := network.Default
		if service.Spec.SessionAffinity == v1.ServiceAffinityClientIP {
			loadDistribution = network.SourceIP
		}

		expectedRule := network.LoadBalancingRule{
			Name: &lbRuleName,
			LoadBalancingRulePropertiesFormat: &network.LoadBalancingRulePropertiesFormat{
				Protocol: *transportProto,
				FrontendIPConfiguration: &network.SubResource{
					ID: to.StringPtr(lbFrontendIPConfigID),
				},
				BackendAddressPool: &network.SubResource{
					ID: to.StringPtr(lbBackendPoolID),
				},
				LoadDistribution: loadDistribution,
				FrontendPort:     to.Int32Ptr(port.Port),
				BackendPort:      to.Int32Ptr(port.Port),
				EnableFloatingIP: to.BoolPtr(true),
			},
		}

		// we didn't construct the probe objects for UDP because they're not used/needed/allowed
		if port.Protocol != v1.ProtocolUDP {
			expectedRule.Probe = &network.SubResource{
				ID: to.StringPtr(az.getLoadBalancerProbeID(lbName, lbRuleName)),
			}
		}

		expectedRules = append(expectedRules, expectedRule)
	}

	// remove unwanted probes
	dirtyProbes := false
	var updatedProbes []network.Probe
	if lb.Probes != nil {
		updatedProbes = *lb.Probes
	}
	for i := len(updatedProbes) - 1; i >= 0; i-- {
		existingProbe := updatedProbes[i]
		if serviceOwnsRule(service, *existingProbe.Name) {
			glog.V(10).Infof("reconcile(%s)(%t): lb probe(%s) - considering evicting", serviceName, wantLb, *existingProbe.Name)
			keepProbe := false
			if findProbe(expectedProbes, existingProbe) {
				glog.V(10).Infof("reconcile(%s)(%t): lb probe(%s) - keeping", serviceName, wantLb, *existingProbe.Name)
				keepProbe = true
			}
			if !keepProbe {
				updatedProbes = append(updatedProbes[:i], updatedProbes[i+1:]...)
				glog.V(10).Infof("reconcile(%s)(%t): lb probe(%s) - dropping", serviceName, wantLb, *existingProbe.Name)
				dirtyProbes = true
			}
		}
	}
	// add missing, wanted probes
	for _, expectedProbe := range expectedProbes {
		foundProbe := false
		if findProbe(updatedProbes, expectedProbe) {
			glog.V(10).Infof("reconcile(%s)(%t): lb probe(%s) - already exists", serviceName, wantLb, *expectedProbe.Name)
			foundProbe = true
		}
		if !foundProbe {
			glog.V(10).Infof("reconcile(%s)(%t): lb probe(%s) - adding", serviceName, wantLb, *expectedProbe.Name)
			updatedProbes = append(updatedProbes, expectedProbe)
			dirtyProbes = true
		}
	}
	if dirtyProbes {
		dirtyLb = true
		lb.Probes = &updatedProbes
	}

	// update rules
	dirtyRules := false
	var updatedRules []network.LoadBalancingRule
	if lb.LoadBalancingRules != nil {
		updatedRules = *lb.LoadBalancingRules
	}
	// update rules: remove unwanted
	for i := len(updatedRules) - 1; i >= 0; i-- {
		existingRule := updatedRules[i]
		if serviceOwnsRule(service, *existingRule.Name) {
			keepRule := false
			glog.V(10).Infof("reconcile(%s)(%t): lb rule(%s) - considering evicting", serviceName, wantLb, *existingRule.Name)
			if findRule(expectedRules, existingRule) {
				glog.V(10).Infof("reconcile(%s)(%t): lb rule(%s) - keeping", serviceName, wantLb, *existingRule.Name)
				keepRule = true
			}
			if !keepRule {
				glog.V(3).Infof("reconcile(%s)(%t): lb rule(%s) - dropping", serviceName, wantLb, *existingRule.Name)
				updatedRules = append(updatedRules[:i], updatedRules[i+1:]...)
				dirtyRules = true
			}
		}
	}
	// update rules: add needed
	for _, expectedRule := range expectedRules {
		foundRule := false
		if findRule(updatedRules, expectedRule) {
			glog.V(10).Infof("reconcile(%s)(%t): lb rule(%s) - already exists", serviceName, wantLb, *expectedRule.Name)
			foundRule = true
		}
		if !foundRule {
			glog.V(10).Infof("reconcile(%s)(%t): lb rule(%s) adding", serviceName, wantLb, *expectedRule.Name)
			updatedRules = append(updatedRules, expectedRule)
			dirtyRules = true
		}
	}
	if dirtyRules {
		dirtyLb = true
		lb.LoadBalancingRules = &updatedRules
	}

	// We don't care if the LB exists or not
	// We only care about if there is any change in the LB, which means dirtyLB
	// If it is not exist, and no change to that, we don't CreateOrUpdate LB
	if dirtyLb {
		if lb.FrontendIPConfigurations == nil || len(*lb.FrontendIPConfigurations) == 0 {
			// When FrontendIPConfigurations is empty, we need to delete the Azure load balancer resource itself,
			// because an Azure load balancer cannot have an empty FrontendIPConfigurations collection
			glog.V(3).Infof("delete(%s): lb(%s) - deleting; no remaining frontendipconfigs", serviceName, lbName)

			az.operationPollRateLimiter.Accept()
			glog.V(10).Infof("LoadBalancerClient.Delete(%q): start", lbName)
			respChan, errChan := az.LoadBalancerClient.Delete(az.ResourceGroup, lbName, nil)
			resp := <-respChan
			err := <-errChan
			glog.V(10).Infof("LoadBalancerClient.Delete(%q): end", lbName)
			if az.CloudProviderBackoff && shouldRetryAPIRequest(resp, err) {
				glog.V(2).Infof("delete(%s) backing off: lb(%s) - deleting; no remaining frontendipconfigs", serviceName, lbName)
				retryErr := az.DeleteLBWithRetry(lbName)
				if retryErr != nil {
					err = retryErr
					glog.V(2).Infof("delete(%s) abort backoff: lb(%s) - deleting; no remaining frontendipconfigs", serviceName, lbName)
				}
			}
			if err != nil {
				return nil, err
			}

		} else {
			glog.V(3).Infof("ensure(%s): lb(%s) - updating", serviceName, lbName)
			az.operationPollRateLimiter.Accept()
			glog.V(10).Infof("LoadBalancerClient.CreateOrUpdate(%q): start", lbName)
			respChan, errChan := az.LoadBalancerClient.CreateOrUpdate(az.ResourceGroup, lbName, *lb, nil)
			resp := <-respChan
			err := <-errChan
			glog.V(10).Infof("LoadBalancerClient.CreateOrUpdate(%q): end", lbName)
			if az.CloudProviderBackoff && shouldRetryAPIRequest(resp.Response, err) {
				glog.V(2).Infof("ensure(%s) backing off: lb(%s) - updating", serviceName, lbName)
				retryErr := az.CreateOrUpdateLBWithRetry(*lb)
				if retryErr != nil {
					glog.V(2).Infof("ensure(%s) abort backoff: lb(%s) - updating", serviceName, lbName)
					return nil, retryErr
				}
			}
			if err != nil {
				return nil, err
			}
		}
	}

	if wantLb && nodes != nil {
		// Add the machines to the backend pool if they're not already
		availabilitySetName := az.mapLoadBalancerNameToAvailabilitySet(lbName, clusterName)
		hostUpdates := make([]func() error, len(nodes))
		for i, node := range nodes {
			localNodeName := node.Name
			f := func() error {
				err := az.ensureHostInPool(serviceName, types.NodeName(localNodeName), lbBackendPoolID, availabilitySetName)
				if err != nil {
					return fmt.Errorf("ensure(%s): lb(%s) - failed to ensure host in pool: %q", serviceName, lbName, err)
				}
				return nil
			}
			hostUpdates[i] = f
		}

		errs := utilerrors.AggregateGoroutines(hostUpdates...)
		if errs != nil {
			return nil, utilerrors.Flatten(errs)
		}
	}

	glog.V(2).Infof("ensure(%s): lb(%s) finished", serviceName, lbName)
	return lb, nil
}

// This reconciles the Network Security Group similar to how the LB is reconciled.
// This entails adding required, missing SecurityRules and removing stale rules.
func (az *Cloud) reconcileSecurityGroup(clusterName string, service *v1.Service, lbStatus *v1.LoadBalancerStatus, wantLb bool) (*network.SecurityGroup, error) {
	serviceName := getServiceName(service)
	glog.V(5).Infof("ensure(%s): START clusterName=%q lbName=%q", serviceName, clusterName)

	var ports []v1.ServicePort
	if wantLb {
		ports = service.Spec.Ports
	} else {
		ports = []v1.ServicePort{}
	}
	az.operationPollRateLimiter.Accept()
	glog.V(10).Infof("SecurityGroupsClient.Get(%q): start", az.SecurityGroupName)
	sg, err := az.SecurityGroupsClient.Get(az.ResourceGroup, az.SecurityGroupName, "")
	glog.V(10).Infof("SecurityGroupsClient.Get(%q): end", az.SecurityGroupName)
	if err != nil {
		return nil, err
	}

	destinationIPAddress := ""
	if wantLb {
		// Get lbIP since we make up NSG rules based on ingress IP
		lbIP := &lbStatus.Ingress[0].IP
		if lbIP == nil {
			return nil, fmt.Errorf("No load balancer IP for setting up security rules for service %s", service.Name)
		}
		destinationIPAddress = *lbIP
	}
	if destinationIPAddress == "" {
		destinationIPAddress = "*"
	}

	sourceRanges, err := serviceapi.GetLoadBalancerSourceRanges(service)
	if err != nil {
		return nil, err
	}
	var sourceAddressPrefixes []string
	if sourceRanges == nil || serviceapi.IsAllowAll(sourceRanges) {
		if !requiresInternalLoadBalancer(service) {
			sourceAddressPrefixes = []string{"Internet"}
		}
	} else {
		for _, ip := range sourceRanges {
			sourceAddressPrefixes = append(sourceAddressPrefixes, ip.String())
		}
	}
	expectedSecurityRules := make([]network.SecurityRule, len(ports)*len(sourceAddressPrefixes))

	for i, port := range ports {
		_, securityProto, _, err := getProtocolsFromKubernetesProtocol(port.Protocol)
		if err != nil {
			return nil, err
		}
		for j := range sourceAddressPrefixes {
			ix := i*len(sourceAddressPrefixes) + j
			securityRuleName := getSecurityRuleName(service, port, sourceAddressPrefixes[j])
			expectedSecurityRules[ix] = network.SecurityRule{
				Name: to.StringPtr(securityRuleName),
				SecurityRulePropertiesFormat: &network.SecurityRulePropertiesFormat{
					Protocol:                 *securityProto,
					SourcePortRange:          to.StringPtr("*"),
					DestinationPortRange:     to.StringPtr(strconv.Itoa(int(port.Port))),
					SourceAddressPrefix:      to.StringPtr(sourceAddressPrefixes[j]),
					DestinationAddressPrefix: to.StringPtr(destinationIPAddress),
					Access:    network.SecurityRuleAccessAllow,
					Direction: network.SecurityRuleDirectionInbound,
				},
			}
		}
	}

	// update security rules
	dirtySg := false
	var updatedRules []network.SecurityRule
	if sg.SecurityRules != nil {
		updatedRules = *sg.SecurityRules
	}
	// update security rules: remove unwanted
	for i := len(updatedRules) - 1; i >= 0; i-- {
		existingRule := updatedRules[i]
		if serviceOwnsRule(service, *existingRule.Name) {
			glog.V(10).Infof("reconcile(%s)(%t): sg rule(%s) - considering evicting", serviceName, wantLb, *existingRule.Name)
			keepRule := false
			if findSecurityRule(expectedSecurityRules, existingRule) {
				glog.V(10).Infof("reconcile(%s)(%t): sg rule(%s) - keeping", serviceName, wantLb, *existingRule.Name)
				keepRule = true
			}
			if !keepRule {
				glog.V(10).Infof("reconcile(%s)(%t): sg rule(%s) - dropping", serviceName, wantLb, *existingRule.Name)
				updatedRules = append(updatedRules[:i], updatedRules[i+1:]...)
				dirtySg = true
			}
		}
	}
	// update security rules: add needed
	for _, expectedRule := range expectedSecurityRules {
		foundRule := false
		if findSecurityRule(updatedRules, expectedRule) {
			glog.V(10).Infof("reconcile(%s)(%t): sg rule(%s) - already exists", serviceName, wantLb, *expectedRule.Name)
			foundRule = true
		}
		if !foundRule {
			glog.V(10).Infof("reconcile(%s)(%t): sg rule(%s) - adding", serviceName, wantLb, *expectedRule.Name)

			nextAvailablePriority, err := getNextAvailablePriority(updatedRules)
			if err != nil {
				return nil, err
			}

			expectedRule.Priority = to.Int32Ptr(nextAvailablePriority)
			updatedRules = append(updatedRules, expectedRule)
			dirtySg = true
		}
	}
	if dirtySg {
		sg.SecurityRules = &updatedRules
		glog.V(3).Infof("ensure(%s): sg(%s) - updating", serviceName, *sg.Name)
		az.operationPollRateLimiter.Accept()
		glog.V(10).Infof("SecurityGroupsClient.CreateOrUpdate(%q): start", *sg.Name)
		respChan, errChan := az.SecurityGroupsClient.CreateOrUpdate(az.ResourceGroup, *sg.Name, sg, nil)
		resp := <-respChan
		err := <-errChan
		glog.V(10).Infof("SecurityGroupsClient.CreateOrUpdate(%q): end", *sg.Name)
		if az.CloudProviderBackoff && shouldRetryAPIRequest(resp.Response, err) {
			glog.V(2).Infof("ensure(%s) backing off: sg(%s) - updating", serviceName, *sg.Name)
			retryErr := az.CreateOrUpdateSGWithRetry(sg)
			if retryErr != nil {
				glog.V(2).Infof("ensure(%s) abort backoff: sg(%s) - updating", serviceName, *sg.Name)
				return nil, retryErr
			}
		}
		if err != nil {
			return nil, err
		}
	}
	return &sg, nil
}

// This reconciles the PublicIP resources similar to how the LB is reconciled.
func (az *Cloud) reconcilePublicIP(clusterName string, service *v1.Service, wantLb bool) (*network.PublicIPAddress, error) {
	isInternal := requiresInternalLoadBalancer(service)
	serviceName := getServiceName(service)
	desiredPipName, err := az.determinePublicIPName(clusterName, service)
	if err != nil {
		return nil, err
	}

	pips, err := az.ListPIPWithRetry()
	if err != nil {
		return nil, err
	}

	for _, pip := range pips {
		if pip.Tags != nil &&
			(*pip.Tags)["service"] != nil &&
			*(*pip.Tags)["service"] == serviceName {
			// We need to process for pips belong to this service
			pipName := *pip.Name
			if wantLb && !isInternal && pipName == desiredPipName {
				// This is the only case we should preserve the
				// Public ip resource with match service tag
			} else {
				glog.V(2).Infof("ensure(%s): pip(%s) - deleting", serviceName, pipName)
				az.operationPollRateLimiter.Accept()
				glog.V(10).Infof("PublicIPAddressesClient.Delete(%q): start", pipName)
				resp, deleteErrChan := az.PublicIPAddressesClient.Delete(az.ResourceGroup, pipName, nil)
				deleteErr := <-deleteErrChan
				glog.V(10).Infof("PublicIPAddressesClient.Delete(%q): end", pipName) // response not read yet...
				if az.CloudProviderBackoff && shouldRetryAPIRequest(<-resp, deleteErr) {
					glog.V(2).Infof("ensure(%s) backing off: pip(%s) - deleting", serviceName, pipName)
					retryErr := az.DeletePublicIPWithRetry(pipName)
					if retryErr != nil {
						glog.V(2).Infof("ensure(%s) abort backoff: pip(%s) - deleting", serviceName, pipName)
						return nil, retryErr
					}
				}

				deleteErr = ignoreStatusNotFoundFromError(deleteErr)
				if deleteErr != nil {
					return nil, deleteErr
				}
				glog.V(2).Infof("ensure(%s): pip(%s) - finished", serviceName, pipName)
			}
		}

	}

	if !isInternal && wantLb {
		// Confirm desired public ip resource exists
		var pip *network.PublicIPAddress
		domainNameLabel := getPublicIPLabel(service)
		if pip, err = az.ensurePublicIPExists(serviceName, desiredPipName, domainNameLabel); err != nil {
			return nil, err
		}
		return pip, nil
	}
	return nil, nil
}

func findProbe(probes []network.Probe, probe network.Probe) bool {
	for _, existingProbe := range probes {
		if strings.EqualFold(*existingProbe.Name, *probe.Name) {
			return true
		}
	}
	return false
}

func findRule(rules []network.LoadBalancingRule, rule network.LoadBalancingRule) bool {
	for _, existingRule := range rules {
		if strings.EqualFold(*existingRule.Name, *rule.Name) {
			return true
		}
	}
	return false
}

func findSecurityRule(rules []network.SecurityRule, rule network.SecurityRule) bool {
	for _, existingRule := range rules {
		if strings.EqualFold(*existingRule.Name, *rule.Name) {
			return true
		}
	}
	return false
}

// This ensures the given VM's Primary NIC's Primary IP Configuration is
// participating in the specified LoadBalancer Backend Pool.
func (az *Cloud) ensureHostInPool(serviceName string, nodeName types.NodeName, backendPoolID string, availabilitySetName string) error {
	var machine compute.VirtualMachine
	vmName := mapNodeNameToVMName(nodeName)
	az.operationPollRateLimiter.Accept()
	glog.V(10).Infof("VirtualMachinesClient.Get(%q): start", vmName)
	machine, err := az.VirtualMachinesClient.Get(az.ResourceGroup, vmName, "")
	glog.V(10).Infof("VirtualMachinesClient.Get(%q): end", vmName)
	if err != nil {
		if az.CloudProviderBackoff {
			glog.V(2).Infof("ensureHostInPool(%s, %s, %s) backing off", serviceName, nodeName, backendPoolID)
			machine, err = az.VirtualMachineClientGetWithRetry(az.ResourceGroup, vmName, "")
			if err != nil {
				glog.V(2).Infof("ensureHostInPool(%s, %s, %s) abort backoff", serviceName, nodeName, backendPoolID)
				return err
			}
		} else {
			return err
		}
	}

	primaryNicID, err := getPrimaryInterfaceID(machine)
	if err != nil {
		return err
	}
	nicName, err := getLastSegment(primaryNicID)
	if err != nil {
		return err
	}

	// Check availability set
	if availabilitySetName != "" {
		expectedAvailabilitySetName := az.getAvailabilitySetID(availabilitySetName)
		if machine.AvailabilitySet == nil || !strings.EqualFold(*machine.AvailabilitySet.ID, expectedAvailabilitySetName) {
			glog.V(3).Infof(
				"nicupdate(%s): skipping nic (%s) since it is not in the availabilitySet(%s)",
				serviceName, nicName, availabilitySetName)
			return nil
		}
	}

	az.operationPollRateLimiter.Accept()
	glog.V(10).Infof("InterfacesClient.Get(%q): start", nicName)
	nic, err := az.InterfacesClient.Get(az.ResourceGroup, nicName, "")
	glog.V(10).Infof("InterfacesClient.Get(%q): end", nicName)
	if err != nil {
		return err
	}

	var primaryIPConfig *network.InterfaceIPConfiguration
	primaryIPConfig, err = getPrimaryIPConfig(nic)
	if err != nil {
		return err
	}

	foundPool := false
	newBackendPools := []network.BackendAddressPool{}
	if primaryIPConfig.LoadBalancerBackendAddressPools != nil {
		newBackendPools = *primaryIPConfig.LoadBalancerBackendAddressPools
	}
	for _, existingPool := range newBackendPools {
		if strings.EqualFold(backendPoolID, *existingPool.ID) {
			foundPool = true
			break
		}
	}
	if !foundPool {
		newBackendPools = append(newBackendPools,
			network.BackendAddressPool{
				ID: to.StringPtr(backendPoolID),
			})

		primaryIPConfig.LoadBalancerBackendAddressPools = &newBackendPools

		glog.V(3).Infof("nicupdate(%s): nic(%s) - updating", serviceName, nicName)
		az.operationPollRateLimiter.Accept()
		glog.V(10).Infof("InterfacesClient.CreateOrUpdate(%q): start", *nic.Name)
		respChan, errChan := az.InterfacesClient.CreateOrUpdate(az.ResourceGroup, *nic.Name, nic, nil)
		resp := <-respChan
		err := <-errChan
		glog.V(10).Infof("InterfacesClient.CreateOrUpdate(%q): end", *nic.Name)
		if az.CloudProviderBackoff && shouldRetryAPIRequest(resp.Response, err) {
			glog.V(2).Infof("nicupdate(%s) backing off: nic(%s) - updating, err=%v", serviceName, nicName, err)
			retryErr := az.CreateOrUpdateInterfaceWithRetry(nic)
			if retryErr != nil {
				err = retryErr
				glog.V(2).Infof("nicupdate(%s) abort backoff: nic(%s) - updating", serviceName, nicName)
			}
		}
		if err != nil {
			return err
		}
	}
	return nil
}

// Check if service requires an internal load balancer.
func requiresInternalLoadBalancer(service *v1.Service) bool {
	if l, ok := service.Annotations[ServiceAnnotationLoadBalancerInternal]; ok {
		return l == "true"
	}

	return false
}

func subnet(service *v1.Service) *string {
	if requiresInternalLoadBalancer(service) {
		if l, ok := service.Annotations[ServiceAnnotationLoadBalancerInternalSubnet]; ok {
			return &l
		}
	}

	return nil
}

// getServiceLoadBalancerMode parses the mode value
// if the value is __auto__ it returns isAuto = TRUE
// if anything else it returns the unique availability set names after triming spaces
func getServiceLoadBalancerMode(service *v1.Service) (hasMode bool, isAuto bool, availabilitySetNames []string) {
	mode, hasMode := service.Annotations[ServiceAnnotationLoadBalancerMode]
	mode = strings.TrimSpace(mode)
	isAuto = strings.EqualFold(mode, ServiceAnnotationLoadBalancerAutoModeValue)
	if !isAuto {
		// Break up list of "AS1,AS2"
		availabilitySetParsedList := strings.Split(mode, ",")

		// Trim the availability set names and remove duplicates
		//  e.g. {"AS1"," AS2", "AS3", "AS3"} => {"AS1", "AS2", "AS3"}
		availabilitySetNameSet := sets.NewString()
		for _, v := range availabilitySetParsedList {
			availabilitySetNameSet.Insert(strings.TrimSpace(v))
		}

		availabilitySetNames = availabilitySetNameSet.List()
	}

	return hasMode, isAuto, availabilitySetNames
}
