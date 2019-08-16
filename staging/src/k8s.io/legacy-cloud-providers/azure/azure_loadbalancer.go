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
	"context"
	"fmt"
	"math"
	"reflect"
	"strconv"
	"strings"

	"github.com/Azure/azure-sdk-for-go/services/network/mgmt/2018-08-01/network"
	"github.com/Azure/go-autorest/autorest/to"

	v1 "k8s.io/api/core/v1"
	utilerrors "k8s.io/apimachinery/pkg/util/errors"
	"k8s.io/apimachinery/pkg/util/sets"
	cloudprovider "k8s.io/cloud-provider"
	servicehelpers "k8s.io/cloud-provider/service/helpers"
	"k8s.io/klog"
)

const (
	// ServiceAnnotationLoadBalancerInternal is the annotation used on the service
	ServiceAnnotationLoadBalancerInternal = "service.beta.kubernetes.io/azure-load-balancer-internal"

	// ServiceAnnotationLoadBalancerInternalSubnet is the annotation used on the service
	// to specify what subnet it is exposed on
	ServiceAnnotationLoadBalancerInternalSubnet = "service.beta.kubernetes.io/azure-load-balancer-internal-subnet"

	// ServiceAnnotationLoadBalancerMode is the annotation used on the service to specify the
	// Azure load balancer selection based on availability sets
	// There are currently three possible load balancer selection modes :
	// 1. Default mode - service has no annotation ("service.beta.kubernetes.io/azure-load-balancer-mode")
	//	  In this case the Loadbalancer of the primary Availability set is selected
	// 2. "__auto__" mode - service is annotated with __auto__ value, this when loadbalancer from any availability set
	//    is selected which has the minimum rules associated with it.
	// 3. "as1,as2" mode - this is when the load balancer from the specified availability sets is selected that has the
	//    minimum rules associated with it.
	ServiceAnnotationLoadBalancerMode = "service.beta.kubernetes.io/azure-load-balancer-mode"

	// ServiceAnnotationLoadBalancerAutoModeValue is the annotation used on the service to specify the
	// Azure load balancer auto selection from the availability sets
	ServiceAnnotationLoadBalancerAutoModeValue = "__auto__"

	// ServiceAnnotationDNSLabelName is the annotation used on the service
	// to specify the DNS label name for the service.
	ServiceAnnotationDNSLabelName = "service.beta.kubernetes.io/azure-dns-label-name"

	// ServiceAnnotationSharedSecurityRule is the annotation used on the service
	// to specify that the service should be exposed using an Azure security rule
	// that may be shared with other service, trading specificity of rules for an
	// increase in the number of services that can be exposed. This relies on the
	// Azure "augmented security rules" feature.
	ServiceAnnotationSharedSecurityRule = "service.beta.kubernetes.io/azure-shared-securityrule"

	// ServiceAnnotationLoadBalancerResourceGroup is the annotation used on the service
	// to specify the resource group of load balancer objects that are not in the same resource group as the cluster.
	ServiceAnnotationLoadBalancerResourceGroup = "service.beta.kubernetes.io/azure-load-balancer-resource-group"

	// ServiceAnnotationPIPName specifies the pip that will be applied to load balancer
	ServiceAnnotationPIPName = "service.beta.kubernetes.io/azure-pip-name"

	// ServiceAnnotationAllowedServiceTag is the annotation used on the service
	// to specify a list of allowed service tags separated by comma
	// Refer https://docs.microsoft.com/en-us/azure/virtual-network/security-overview#service-tags for all supported service tags.
	ServiceAnnotationAllowedServiceTag = "service.beta.kubernetes.io/azure-allowed-service-tags"

	// ServiceAnnotationLoadBalancerIdleTimeout is the annotation used on the service
	// to specify the idle timeout for connections on the load balancer in minutes.
	ServiceAnnotationLoadBalancerIdleTimeout = "service.beta.kubernetes.io/azure-load-balancer-tcp-idle-timeout"

	// ServiceAnnotationLoadBalancerMixedProtocols is the annotation used on the service
	// to create both TCP and UDP protocols when creating load balancer rules.
	ServiceAnnotationLoadBalancerMixedProtocols = "service.beta.kubernetes.io/azure-load-balancer-mixed-protocols"

	// ServiceAnnotationLoadBalancerDisableTCPReset is the annotation used on the service
	// to set enableTcpReset to false in load balancer rule. This only works for Azure standard load balancer backed service.
	ServiceAnnotationLoadBalancerDisableTCPReset = "service.beta.kubernetes.io/azure-load-balancer-disable-tcp-reset"

	// serviceTagKey is the service key applied for public IP tags.
	serviceTagKey = "service"
	// clusterNameKey is the cluster name key applied for public IP tags.
	clusterNameKey = "kubernetes-cluster-name"
)

// GetLoadBalancer returns whether the specified load balancer exists, and
// if so, what its status is.
func (az *Cloud) GetLoadBalancer(ctx context.Context, clusterName string, service *v1.Service) (status *v1.LoadBalancerStatus, exists bool, err error) {
	_, status, exists, err = az.getServiceLoadBalancer(service, clusterName, nil, false)
	if err != nil {
		return nil, false, err
	}
	if !exists {
		serviceName := getServiceName(service)
		klog.V(5).Infof("getloadbalancer (cluster:%s) (service:%s) - doesn't exist", clusterName, serviceName)
		return nil, false, nil
	}
	return status, true, nil
}

func getPublicIPDomainNameLabel(service *v1.Service) string {
	if labelName, found := service.Annotations[ServiceAnnotationDNSLabelName]; found {
		return labelName
	}
	return ""
}

// EnsureLoadBalancer creates a new load balancer 'name', or updates the existing one. Returns the status of the balancer
func (az *Cloud) EnsureLoadBalancer(ctx context.Context, clusterName string, service *v1.Service, nodes []*v1.Node) (*v1.LoadBalancerStatus, error) {
	// When a client updates the internal load balancer annotation,
	// the service may be switched from an internal LB to a public one, or vise versa.
	// Here we'll firstly ensure service do not lie in the opposite LB.
	serviceName := getServiceName(service)
	klog.V(5).Infof("ensureloadbalancer(%s): START clusterName=%q", serviceName, clusterName)

	lb, err := az.reconcileLoadBalancer(clusterName, service, nodes, true /* wantLb */)
	if err != nil {
		return nil, err
	}

	lbStatus, err := az.getServiceLoadBalancerStatus(service, lb)
	if err != nil {
		return nil, err
	}

	var serviceIP *string
	if lbStatus != nil && len(lbStatus.Ingress) > 0 {
		serviceIP = &lbStatus.Ingress[0].IP
	}
	klog.V(2).Infof("EnsureLoadBalancer: reconciling security group for service %q with IP %q, wantLb = true", serviceName, logSafe(serviceIP))
	if _, err := az.reconcileSecurityGroup(clusterName, service, serviceIP, true /* wantLb */); err != nil {
		return nil, err
	}

	updateService := updateServiceLoadBalancerIP(service, to.String(serviceIP))
	flippedService := flipServiceInternalAnnotation(updateService)
	if _, err := az.reconcileLoadBalancer(clusterName, flippedService, nil, false /* wantLb */); err != nil {
		return nil, err
	}

	// lb is not reused here because the ETAG may be changed in above operations, hence reconcilePublicIP() would get lb again from cache.
	if _, err := az.reconcilePublicIP(clusterName, updateService, to.String(lb.Name), true /* wantLb */); err != nil {
		return nil, err
	}

	return lbStatus, nil
}

// UpdateLoadBalancer updates hosts under the specified load balancer.
func (az *Cloud) UpdateLoadBalancer(ctx context.Context, clusterName string, service *v1.Service, nodes []*v1.Node) error {
	_, err := az.EnsureLoadBalancer(ctx, clusterName, service, nodes)
	return err
}

// EnsureLoadBalancerDeleted deletes the specified load balancer if it
// exists, returning nil if the load balancer specified either didn't exist or
// was successfully deleted.
// This construction is useful because many cloud providers' load balancers
// have multiple underlying components, meaning a Get could say that the LB
// doesn't exist even if some part of it is still laying around.
func (az *Cloud) EnsureLoadBalancerDeleted(ctx context.Context, clusterName string, service *v1.Service) error {
	isInternal := requiresInternalLoadBalancer(service)
	serviceName := getServiceName(service)
	klog.V(5).Infof("Delete service (%s): START clusterName=%q", serviceName, clusterName)

	ignoreErrors := func(err error) error {
		if ignoreStatusNotFoundFromError(err) == nil {
			klog.V(5).Infof("EnsureLoadBalancerDeleted: ignoring StatusNotFound error because the resource doesn't exist (%v)", err)
			return nil
		}

		if ignoreStatusForbiddenFromError(err) == nil {
			klog.V(5).Infof("EnsureLoadBalancerDeleted: ignoring StatusForbidden error (%v). This may be caused by wrong configuration via service annotations", err)
			return nil
		}

		return err
	}

	serviceIPToCleanup, err := az.findServiceIPAddress(ctx, clusterName, service, isInternal)
	if ignoreErrors(err) != nil {
		return err
	}

	klog.V(2).Infof("EnsureLoadBalancerDeleted: reconciling security group for service %q with IP %q, wantLb = false", serviceName, serviceIPToCleanup)
	if _, err := az.reconcileSecurityGroup(clusterName, service, &serviceIPToCleanup, false /* wantLb */); err != nil {
		if ignoreErrors(err) != nil {
			return err
		}
	}

	if _, err := az.reconcileLoadBalancer(clusterName, service, nil, false /* wantLb */); err != nil {
		if ignoreErrors(err) != nil {
			return err
		}
	}

	if _, err := az.reconcilePublicIP(clusterName, service, "", false /* wantLb */); err != nil {
		if ignoreErrors(err) != nil {
			return err
		}
	}

	klog.V(2).Infof("Delete service (%s): FINISH", serviceName)
	return nil
}

// GetLoadBalancerName returns the LoadBalancer name.
func (az *Cloud) GetLoadBalancerName(ctx context.Context, clusterName string, service *v1.Service) string {
	return cloudprovider.DefaultLoadBalancerName(service)
}

func (az *Cloud) getLoadBalancerResourceGroup() string {
	if az.LoadBalancerResourceGroup != "" {
		return az.LoadBalancerResourceGroup
	}

	return az.ResourceGroup
}

// getServiceLoadBalancer gets the loadbalancer for the service if it already exists.
// If wantLb is TRUE then -it selects a new load balancer.
// In case the selected load balancer does not exist it returns network.LoadBalancer struct
// with added metadata (such as name, location) and existsLB set to FALSE.
// By default - cluster default LB is returned.
func (az *Cloud) getServiceLoadBalancer(service *v1.Service, clusterName string, nodes []*v1.Node, wantLb bool) (lb *network.LoadBalancer, status *v1.LoadBalancerStatus, exists bool, err error) {
	isInternal := requiresInternalLoadBalancer(service)
	var defaultLB *network.LoadBalancer
	primaryVMSetName := az.vmSet.GetPrimaryVMSetName()
	defaultLBName := az.getAzureLoadBalancerName(clusterName, primaryVMSetName, isInternal)

	existingLBs, err := az.ListLB(service)
	if err != nil {
		return nil, nil, false, err
	}

	// check if the service already has a load balancer
	if existingLBs != nil {
		for i := range existingLBs {
			existingLB := existingLBs[i]
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
				// service is not on this load balancer
				continue
			}

			return &existingLB, status, true, nil
		}
	}

	hasMode, _, _ := getServiceLoadBalancerMode(service)
	if az.useStandardLoadBalancer() && hasMode {
		return nil, nil, false, fmt.Errorf("standard load balancer doesn't work with annotation %q", ServiceAnnotationLoadBalancerMode)
	}

	// service does not have a basic load balancer, select one.
	// Standard load balancer doesn't need this because all backends nodes should be added to same LB.
	if wantLb && !az.useStandardLoadBalancer() {
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
		if az.useStandardLoadBalancer() {
			defaultLB.Sku = &network.LoadBalancerSku{
				Name: network.LoadBalancerSkuNameStandard,
			}
		}
	}

	return defaultLB, nil, false, nil
}

// selectLoadBalancer selects load balancer for the service in the cluster.
// The selection algorithm selects the load balancer which currently has
// the minimum lb rules. If there are multiple LBs with same number of rules,
// then selects the first one (sorted based on name).
func (az *Cloud) selectLoadBalancer(clusterName string, service *v1.Service, existingLBs *[]network.LoadBalancer, nodes []*v1.Node) (selectedLB *network.LoadBalancer, existsLb bool, err error) {
	isInternal := requiresInternalLoadBalancer(service)
	serviceName := getServiceName(service)
	klog.V(2).Infof("selectLoadBalancer for service (%s): isInternal(%v) - start", serviceName, isInternal)
	vmSetNames, err := az.vmSet.GetVMSetNames(service, nodes)
	if err != nil {
		klog.Errorf("az.selectLoadBalancer: cluster(%s) service(%s) isInternal(%t) - az.GetVMSetNames failed, err=(%v)", clusterName, serviceName, isInternal, err)
		return nil, false, err
	}
	klog.Infof("selectLoadBalancer: cluster(%s) service(%s) isInternal(%t) - vmSetNames %v", clusterName, serviceName, isInternal, *vmSetNames)

	mapExistingLBs := map[string]network.LoadBalancer{}
	for _, lb := range *existingLBs {
		mapExistingLBs[*lb.Name] = lb
	}
	selectedLBRuleCount := math.MaxInt32
	for _, currASName := range *vmSetNames {
		currLBName := az.getAzureLoadBalancerName(clusterName, currASName, isInternal)
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
		err = fmt.Errorf("selectLoadBalancer: cluster(%s) service(%s) isInternal(%t) - unable to find load balancer for selected VM sets %v", clusterName, serviceName, isInternal, *vmSetNames)
		klog.Error(err)
		return nil, false, err
	}
	// validate if the selected LB has not exceeded the MaximumLoadBalancerRuleCount
	if az.Config.MaximumLoadBalancerRuleCount != 0 && selectedLBRuleCount >= az.Config.MaximumLoadBalancerRuleCount {
		err = fmt.Errorf("selectLoadBalancer: cluster(%s) service(%s) isInternal(%t) -  all available load balancers have exceeded maximum rule limit %d, vmSetNames (%v)", clusterName, serviceName, isInternal, selectedLBRuleCount, *vmSetNames)
		klog.Error(err)
		return selectedLB, existsLb, err
	}

	return selectedLB, existsLb, nil
}

func (az *Cloud) getServiceLoadBalancerStatus(service *v1.Service, lb *network.LoadBalancer) (status *v1.LoadBalancerStatus, err error) {
	if lb == nil {
		klog.V(10).Info("getServiceLoadBalancerStatus: lb is nil")
		return nil, nil
	}
	if lb.FrontendIPConfigurations == nil || *lb.FrontendIPConfigurations == nil {
		klog.V(10).Info("getServiceLoadBalancerStatus: lb.FrontendIPConfigurations is nil")
		return nil, nil
	}
	isInternal := requiresInternalLoadBalancer(service)
	lbFrontendIPConfigName := az.getFrontendIPConfigName(service, subnet(service))
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
				pip, existsPip, err := az.getPublicIPAddress(az.getPublicIPAddressResourceGroup(service), pipName)
				if err != nil {
					return nil, err
				}
				if existsPip {
					lbIP = pip.IPAddress
				}
			}

			klog.V(2).Infof("getServiceLoadBalancerStatus gets ingress IP %q from frontendIPConfiguration %q for service %q", to.String(lbIP), lbFrontendIPConfigName, serviceName)
			return &v1.LoadBalancerStatus{Ingress: []v1.LoadBalancerIngress{{IP: to.String(lbIP)}}}, nil
		}
	}

	return nil, nil
}

func (az *Cloud) determinePublicIPName(clusterName string, service *v1.Service) (string, bool, error) {
	var shouldPIPExisted bool
	if name, found := service.Annotations[ServiceAnnotationPIPName]; found && name != "" {
		shouldPIPExisted = true
		return name, shouldPIPExisted, nil
	}

	loadBalancerIP := service.Spec.LoadBalancerIP
	if len(loadBalancerIP) == 0 {
		return az.getPublicIPName(clusterName, service), shouldPIPExisted, nil
	}

	pipResourceGroup := az.getPublicIPAddressResourceGroup(service)

	pips, err := az.ListPIP(service, pipResourceGroup)
	if err != nil {
		return "", shouldPIPExisted, err
	}

	for _, pip := range pips {
		if pip.PublicIPAddressPropertiesFormat.IPAddress != nil &&
			*pip.PublicIPAddressPropertiesFormat.IPAddress == loadBalancerIP {
			return *pip.Name, shouldPIPExisted, nil
		}
	}
	return "", shouldPIPExisted, fmt.Errorf("user supplied IP Address %s was not found in resource group %s", loadBalancerIP, pipResourceGroup)
}

func flipServiceInternalAnnotation(service *v1.Service) *v1.Service {
	copyService := service.DeepCopy()
	if copyService.Annotations == nil {
		copyService.Annotations = map[string]string{}
	}
	if v, ok := copyService.Annotations[ServiceAnnotationLoadBalancerInternal]; ok && v == "true" {
		// If it is internal now, we make it external by remove the annotation
		delete(copyService.Annotations, ServiceAnnotationLoadBalancerInternal)
	} else {
		// If it is external now, we make it internal
		copyService.Annotations[ServiceAnnotationLoadBalancerInternal] = "true"
	}
	return copyService
}

func updateServiceLoadBalancerIP(service *v1.Service, serviceIP string) *v1.Service {
	copyService := service.DeepCopy()
	if len(serviceIP) > 0 && copyService != nil {
		copyService.Spec.LoadBalancerIP = serviceIP
	}
	return copyService
}

func (az *Cloud) findServiceIPAddress(ctx context.Context, clusterName string, service *v1.Service, isInternalLb bool) (string, error) {
	if len(service.Spec.LoadBalancerIP) > 0 {
		return service.Spec.LoadBalancerIP, nil
	}

	lbStatus, existsLb, err := az.GetLoadBalancer(ctx, clusterName, service)
	if err != nil {
		return "", err
	}
	if !existsLb {
		klog.V(2).Infof("Expected to find an IP address for service %s but did not. Assuming it has been removed", service.Name)
		return "", nil
	}
	if len(lbStatus.Ingress) < 1 {
		klog.V(2).Infof("Expected to find an IP address for service %s but it had no ingresses. Assuming it has been removed", service.Name)
		return "", nil
	}

	return lbStatus.Ingress[0].IP, nil
}

func (az *Cloud) ensurePublicIPExists(service *v1.Service, pipName string, domainNameLabel, clusterName string, shouldPIPExisted bool) (*network.PublicIPAddress, error) {
	pipResourceGroup := az.getPublicIPAddressResourceGroup(service)
	pip, existsPip, err := az.getPublicIPAddress(pipResourceGroup, pipName)
	if err != nil {
		return nil, err
	}

	serviceName := getServiceName(service)

	if existsPip {
		// return if pip exist and dns label is the same
		if getDomainNameLabel(&pip) == domainNameLabel {
			return &pip, nil
		}
		klog.V(2).Infof("ensurePublicIPExists for service(%s): pip(%s) - updating", serviceName, *pip.Name)
		if pip.PublicIPAddressPropertiesFormat == nil {
			pip.PublicIPAddressPropertiesFormat = &network.PublicIPAddressPropertiesFormat{
				PublicIPAllocationMethod: network.Static,
			}
		}
	} else {
		if shouldPIPExisted {
			return nil, fmt.Errorf("PublicIP from annotation azure-pip-name=%s for service %s doesn't exist", pipName, serviceName)
		}
		pip.Name = to.StringPtr(pipName)
		pip.Location = to.StringPtr(az.Location)
		pip.PublicIPAddressPropertiesFormat = &network.PublicIPAddressPropertiesFormat{
			PublicIPAllocationMethod: network.Static,
		}
		pip.Tags = map[string]*string{
			serviceTagKey:  &serviceName,
			clusterNameKey: &clusterName,
		}
		if az.useStandardLoadBalancer() {
			pip.Sku = &network.PublicIPAddressSku{
				Name: network.PublicIPAddressSkuNameStandard,
			}
		}
		klog.V(2).Infof("ensurePublicIPExists for service(%s): pip(%s) - creating", serviceName, *pip.Name)
	}
	if len(domainNameLabel) == 0 {
		pip.PublicIPAddressPropertiesFormat.DNSSettings = nil
	} else {
		pip.PublicIPAddressPropertiesFormat.DNSSettings = &network.PublicIPAddressDNSSettings{
			DomainNameLabel: &domainNameLabel,
		}
	}

	klog.V(10).Infof("CreateOrUpdatePIP(%s, %q): start", pipResourceGroup, *pip.Name)
	err = az.CreateOrUpdatePIP(service, pipResourceGroup, pip)
	if err != nil {
		klog.V(2).Infof("ensure(%s) abort backoff: pip(%s)", serviceName, *pip.Name)
		return nil, err
	}
	klog.V(10).Infof("CreateOrUpdatePIP(%s, %q): end", pipResourceGroup, *pip.Name)

	ctx, cancel := getContextWithCancel()
	defer cancel()
	pip, err = az.PublicIPAddressesClient.Get(ctx, pipResourceGroup, *pip.Name, "")
	if err != nil {
		return nil, err
	}
	return &pip, nil
}

func getDomainNameLabel(pip *network.PublicIPAddress) string {
	if pip == nil || pip.PublicIPAddressPropertiesFormat == nil || pip.PublicIPAddressPropertiesFormat.DNSSettings == nil {
		return ""
	}
	return to.String(pip.PublicIPAddressPropertiesFormat.DNSSettings.DomainNameLabel)
}

func getIdleTimeout(s *v1.Service) (*int32, error) {
	const (
		min = 4
		max = 30
	)

	val, ok := s.Annotations[ServiceAnnotationLoadBalancerIdleTimeout]
	if !ok {
		// Return a nil here as this will set the value to the azure default
		return nil, nil
	}

	errInvalidTimeout := fmt.Errorf("idle timeout value must be a whole number representing minutes between %d and %d", min, max)
	to, err := strconv.Atoi(val)
	if err != nil {
		return nil, fmt.Errorf("error parsing idle timeout value: %v: %v", err, errInvalidTimeout)
	}
	to32 := int32(to)

	if to32 < min || to32 > max {
		return nil, errInvalidTimeout
	}
	return &to32, nil
}

func (az *Cloud) isFrontendIPChanged(clusterName string, config network.FrontendIPConfiguration, service *v1.Service, lbFrontendIPConfigName string) (bool, error) {
	if az.serviceOwnsFrontendIP(config, service) && !strings.EqualFold(to.String(config.Name), lbFrontendIPConfigName) {
		return true, nil
	}
	if !strings.EqualFold(to.String(config.Name), lbFrontendIPConfigName) {
		return false, nil
	}
	loadBalancerIP := service.Spec.LoadBalancerIP
	isInternal := requiresInternalLoadBalancer(service)
	if isInternal {
		// Judge subnet
		subnetName := subnet(service)
		if subnetName != nil {
			subnet, existsSubnet, err := az.getSubnet(az.VnetName, *subnetName)
			if err != nil {
				return false, err
			}
			if !existsSubnet {
				return false, fmt.Errorf("failed to get subnet")
			}
			if config.Subnet != nil && !strings.EqualFold(to.String(config.Subnet.Name), to.String(subnet.Name)) {
				return true, nil
			}
		}
		if loadBalancerIP == "" {
			return config.PrivateIPAllocationMethod == network.Static, nil
		}
		return config.PrivateIPAllocationMethod != network.Static || !strings.EqualFold(loadBalancerIP, to.String(config.PrivateIPAddress)), nil
	}
	if loadBalancerIP == "" {
		return false, nil
	}
	pipName, _, err := az.determinePublicIPName(clusterName, service)
	if err != nil {
		return false, err
	}
	pipResourceGroup := az.getPublicIPAddressResourceGroup(service)
	pip, existsPip, err := az.getPublicIPAddress(pipResourceGroup, pipName)
	if err != nil {
		return false, err
	}
	if !existsPip {
		return true, nil
	}
	return config.PublicIPAddress != nil && !strings.EqualFold(to.String(pip.ID), to.String(config.PublicIPAddress.ID)), nil
}

// This ensures load balancer exists and the frontend ip config is setup.
// This also reconciles the Service's Ports  with the LoadBalancer config.
// This entails adding rules/probes for expected Ports and removing stale rules/ports.
// nodes only used if wantLb is true
func (az *Cloud) reconcileLoadBalancer(clusterName string, service *v1.Service, nodes []*v1.Node, wantLb bool) (*network.LoadBalancer, error) {
	isInternal := requiresInternalLoadBalancer(service)
	serviceName := getServiceName(service)
	klog.V(2).Infof("reconcileLoadBalancer for service(%s) - wantLb(%t): started", serviceName, wantLb)
	lb, _, _, err := az.getServiceLoadBalancer(service, clusterName, nodes, wantLb)
	if err != nil {
		klog.Errorf("reconcileLoadBalancer: failed to get load balancer for service %q, error: %v", serviceName, err)
		return nil, err
	}
	lbName := *lb.Name
	klog.V(2).Infof("reconcileLoadBalancer for service(%s): lb(%s) wantLb(%t) resolved load balancer name", serviceName, lbName, wantLb)
	lbFrontendIPConfigName := az.getFrontendIPConfigName(service, subnet(service))
	lbFrontendIPConfigID := az.getFrontendIPConfigID(lbName, lbFrontendIPConfigName)
	lbBackendPoolName := getBackendPoolName(clusterName)
	lbBackendPoolID := az.getBackendPoolID(lbName, lbBackendPoolName)

	lbIdleTimeout, err := getIdleTimeout(service)
	if wantLb && err != nil {
		return nil, err
	}

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
				klog.V(10).Infof("reconcileLoadBalancer for service (%s)(%t): lb backendpool - found wanted backendpool. not adding anything", serviceName, wantLb)
				foundBackendPool = true
				break
			} else {
				klog.V(10).Infof("reconcileLoadBalancer for service (%s)(%t): lb backendpool - found other backendpool %s", serviceName, wantLb, *bp.Name)
			}
		}
		if !foundBackendPool {
			newBackendPools = append(newBackendPools, network.BackendAddressPool{
				Name: to.StringPtr(lbBackendPoolName),
			})
			klog.V(10).Infof("reconcileLoadBalancer for service (%s)(%t): lb backendpool - adding backendpool", serviceName, wantLb)

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
			if az.serviceOwnsFrontendIP(config, service) {
				klog.V(2).Infof("reconcileLoadBalancer for service (%s)(%t): lb frontendconfig(%s) - dropping", serviceName, wantLb, lbFrontendIPConfigName)
				newConfigs = append(newConfigs[:i], newConfigs[i+1:]...)
				dirtyConfigs = true
			}
		}
	} else {
		for i := len(newConfigs) - 1; i >= 0; i-- {
			config := newConfigs[i]
			isFipChanged, err := az.isFrontendIPChanged(clusterName, config, service, lbFrontendIPConfigName)
			if err != nil {
				return nil, err
			}
			if isFipChanged {
				klog.V(2).Infof("reconcileLoadBalancer for service (%s)(%t): lb frontendconfig(%s) - dropping", serviceName, wantLb, *config.Name)
				newConfigs = append(newConfigs[:i], newConfigs[i+1:]...)
				dirtyConfigs = true
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
				pipName, shouldPIPExisted, err := az.determinePublicIPName(clusterName, service)
				if err != nil {
					return nil, err
				}
				domainNameLabel := getPublicIPDomainNameLabel(service)
				pip, err := az.ensurePublicIPExists(service, pipName, domainNameLabel, clusterName, shouldPIPExisted)
				if err != nil {
					return nil, err
				}
				fipConfigurationProperties = &network.FrontendIPConfigurationPropertiesFormat{
					PublicIPAddress: &network.PublicIPAddress{ID: pip.ID},
				}
			}

			newConfigs = append(newConfigs,
				network.FrontendIPConfiguration{
					Name:                                    to.StringPtr(lbFrontendIPConfigName),
					FrontendIPConfigurationPropertiesFormat: fipConfigurationProperties,
				})
			klog.V(10).Infof("reconcileLoadBalancer for service (%s)(%t): lb frontendconfig(%s) - adding", serviceName, wantLb, lbFrontendIPConfigName)
			dirtyConfigs = true
		}
	}
	if dirtyConfigs {
		dirtyLb = true
		lb.FrontendIPConfigurations = &newConfigs
	}

	// update probes/rules
	expectedProbes, expectedRules, err := az.reconcileLoadBalancerRule(service, wantLb, lbFrontendIPConfigID, lbBackendPoolID, lbName, lbIdleTimeout)
	if err != nil {
		return nil, err
	}

	// remove unwanted probes
	dirtyProbes := false
	var updatedProbes []network.Probe
	if lb.Probes != nil {
		updatedProbes = *lb.Probes
	}
	for i := len(updatedProbes) - 1; i >= 0; i-- {
		existingProbe := updatedProbes[i]
		if az.serviceOwnsRule(service, *existingProbe.Name) {
			klog.V(10).Infof("reconcileLoadBalancer for service (%s)(%t): lb probe(%s) - considering evicting", serviceName, wantLb, *existingProbe.Name)
			keepProbe := false
			if findProbe(expectedProbes, existingProbe) {
				klog.V(10).Infof("reconcileLoadBalancer for service (%s)(%t): lb probe(%s) - keeping", serviceName, wantLb, *existingProbe.Name)
				keepProbe = true
			}
			if !keepProbe {
				updatedProbes = append(updatedProbes[:i], updatedProbes[i+1:]...)
				klog.V(10).Infof("reconcileLoadBalancer for service (%s)(%t): lb probe(%s) - dropping", serviceName, wantLb, *existingProbe.Name)
				dirtyProbes = true
			}
		}
	}
	// add missing, wanted probes
	for _, expectedProbe := range expectedProbes {
		foundProbe := false
		if findProbe(updatedProbes, expectedProbe) {
			klog.V(10).Infof("reconcileLoadBalancer for service (%s)(%t): lb probe(%s) - already exists", serviceName, wantLb, *expectedProbe.Name)
			foundProbe = true
		}
		if !foundProbe {
			klog.V(10).Infof("reconcileLoadBalancer for service (%s)(%t): lb probe(%s) - adding", serviceName, wantLb, *expectedProbe.Name)
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
		if az.serviceOwnsRule(service, *existingRule.Name) {
			keepRule := false
			klog.V(10).Infof("reconcileLoadBalancer for service (%s)(%t): lb rule(%s) - considering evicting", serviceName, wantLb, *existingRule.Name)
			if findRule(expectedRules, existingRule, wantLb) {
				klog.V(10).Infof("reconcileLoadBalancer for service (%s)(%t): lb rule(%s) - keeping", serviceName, wantLb, *existingRule.Name)
				keepRule = true
			}
			if !keepRule {
				klog.V(2).Infof("reconcileLoadBalancer for service (%s)(%t): lb rule(%s) - dropping", serviceName, wantLb, *existingRule.Name)
				updatedRules = append(updatedRules[:i], updatedRules[i+1:]...)
				dirtyRules = true
			}
		}
	}
	// update rules: add needed
	for _, expectedRule := range expectedRules {
		foundRule := false
		if findRule(updatedRules, expectedRule, wantLb) {
			klog.V(10).Infof("reconcileLoadBalancer for service (%s)(%t): lb rule(%s) - already exists", serviceName, wantLb, *expectedRule.Name)
			foundRule = true
		}
		if !foundRule {
			klog.V(10).Infof("reconcileLoadBalancer for service (%s)(%t): lb rule(%s) adding", serviceName, wantLb, *expectedRule.Name)
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
			klog.V(2).Infof("reconcileLoadBalancer for service(%s): lb(%s) - deleting; no remaining frontendIPConfigurations", serviceName, lbName)

			// Remove backend pools from vmSets. This is required for virtual machine scale sets before removing the LB.
			vmSetName := az.mapLoadBalancerNameToVMSet(lbName, clusterName)
			klog.V(10).Infof("EnsureBackendPoolDeleted(%s,%s) for service %s: start", lbBackendPoolID, vmSetName, serviceName)
			err := az.vmSet.EnsureBackendPoolDeleted(service, lbBackendPoolID, vmSetName, lb.BackendAddressPools)
			if err != nil {
				klog.Errorf("EnsureBackendPoolDeleted(%s) for service %s failed: %v", lbBackendPoolID, serviceName, err)
				return nil, err
			}
			klog.V(10).Infof("EnsureBackendPoolDeleted(%s) for service %s: end", lbBackendPoolID, serviceName)

			// Remove the LB.
			klog.V(10).Infof("reconcileLoadBalancer: az.DeleteLB(%q): start", lbName)
			err = az.DeleteLB(service, lbName)
			if err != nil {
				klog.V(2).Infof("reconcileLoadBalancer for service(%s) abort backoff: lb(%s) - deleting; no remaining frontendIPConfigurations", serviceName, lbName)
				return nil, err
			}
			klog.V(10).Infof("az.DeleteLB(%q): end", lbName)
		} else {
			klog.V(2).Infof("reconcileLoadBalancer: reconcileLoadBalancer for service(%s): lb(%s) - updating", serviceName, lbName)
			err := az.CreateOrUpdateLB(service, *lb)
			if err != nil {
				klog.V(2).Infof("reconcileLoadBalancer for service(%s) abort backoff: lb(%s) - updating", serviceName, lbName)
				return nil, err
			}

			if isInternal {
				// Refresh updated lb which will be used later in other places.
				newLB, exist, err := az.getAzureLoadBalancer(lbName)
				if err != nil {
					klog.V(2).Infof("reconcileLoadBalancer for service(%s): getAzureLoadBalancer(%s) failed: %v", serviceName, lbName, err)
					return nil, err
				}
				if !exist {
					return nil, fmt.Errorf("load balancer %q not found", lbName)
				}
				lb = &newLB
			}
		}
	}

	if wantLb && nodes != nil {
		// Add the machines to the backend pool if they're not already
		vmSetName := az.mapLoadBalancerNameToVMSet(lbName, clusterName)
		// Etag would be changed when updating backend pools, so invalidate lbCache after it.
		defer az.lbCache.Delete(lbName)
		err := az.vmSet.EnsureHostsInPool(service, nodes, lbBackendPoolID, vmSetName, isInternal)
		if err != nil {
			return nil, err
		}
	}

	klog.V(2).Infof("reconcileLoadBalancer for service(%s): lb(%s) finished", serviceName, lbName)
	return lb, nil
}

func (az *Cloud) reconcileLoadBalancerRule(
	service *v1.Service,
	wantLb bool,
	lbFrontendIPConfigID string,
	lbBackendPoolID string,
	lbName string,
	lbIdleTimeout *int32) ([]network.Probe, []network.LoadBalancingRule, error) {

	var ports []v1.ServicePort
	if wantLb {
		ports = service.Spec.Ports
	} else {
		ports = []v1.ServicePort{}
	}

	var enableTCPReset *bool
	if az.useStandardLoadBalancer() {
		enableTCPReset = to.BoolPtr(true)
		if v, ok := service.Annotations[ServiceAnnotationLoadBalancerDisableTCPReset]; ok {
			klog.V(2).Infof("reconcileLoadBalancerRule lb name (%s) flag(%s) is set to %s", lbName, ServiceAnnotationLoadBalancerDisableTCPReset, v)
			enableTCPReset = to.BoolPtr(!strings.EqualFold(v, "true"))
		}
	}

	var expectedProbes []network.Probe
	var expectedRules []network.LoadBalancingRule
	for _, port := range ports {
		protocols := []v1.Protocol{port.Protocol}
		if v, ok := service.Annotations[ServiceAnnotationLoadBalancerMixedProtocols]; ok && v == "true" {
			klog.V(2).Infof("reconcileLoadBalancerRule lb name (%s) flag(%s) is set", lbName, ServiceAnnotationLoadBalancerMixedProtocols)
			if port.Protocol == v1.ProtocolTCP {
				protocols = append(protocols, v1.ProtocolUDP)
			} else if port.Protocol == v1.ProtocolUDP {
				protocols = append(protocols, v1.ProtocolTCP)
			}
		}

		for _, protocol := range protocols {
			lbRuleName := az.getLoadBalancerRuleName(service, protocol, port.Port, subnet(service))
			klog.V(2).Infof("reconcileLoadBalancerRule lb name (%s) rule name (%s)", lbName, lbRuleName)

			transportProto, _, probeProto, err := getProtocolsFromKubernetesProtocol(protocol)
			if err != nil {
				return expectedProbes, expectedRules, err
			}

			if servicehelpers.NeedsHealthCheck(service) {
				podPresencePath, podPresencePort := servicehelpers.GetServiceHealthCheckPathPort(service)

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
			} else if protocol != v1.ProtocolUDP && protocol != v1.ProtocolSCTP {
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
					LoadDistribution:    loadDistribution,
					FrontendPort:        to.Int32Ptr(port.Port),
					BackendPort:         to.Int32Ptr(port.Port),
					EnableFloatingIP:    to.BoolPtr(true),
					DisableOutboundSnat: to.BoolPtr(az.disableLoadBalancerOutboundSNAT()),
					EnableTCPReset:      enableTCPReset,
				},
			}
			if protocol == v1.ProtocolTCP {
				expectedRule.LoadBalancingRulePropertiesFormat.IdleTimeoutInMinutes = lbIdleTimeout
			}

			// we didn't construct the probe objects for UDP or SCTP because they're not used/needed/allowed
			if protocol != v1.ProtocolUDP && protocol != v1.ProtocolSCTP {
				expectedRule.Probe = &network.SubResource{
					ID: to.StringPtr(az.getLoadBalancerProbeID(lbName, lbRuleName)),
				}
			}

			expectedRules = append(expectedRules, expectedRule)
		}
	}

	return expectedProbes, expectedRules, nil
}

// This reconciles the Network Security Group similar to how the LB is reconciled.
// This entails adding required, missing SecurityRules and removing stale rules.
func (az *Cloud) reconcileSecurityGroup(clusterName string, service *v1.Service, lbIP *string, wantLb bool) (*network.SecurityGroup, error) {
	serviceName := getServiceName(service)
	klog.V(5).Infof("reconcileSecurityGroup(%s): START clusterName=%q", serviceName, clusterName)

	ports := service.Spec.Ports
	if ports == nil {
		if useSharedSecurityRule(service) {
			klog.V(2).Infof("Attempting to reconcile security group for service %s, but service uses shared rule and we don't know which port it's for", service.Name)
			return nil, fmt.Errorf("No port info for reconciling shared rule for service %s", service.Name)
		}
		ports = []v1.ServicePort{}
	}

	sg, err := az.getSecurityGroup()
	if err != nil {
		return nil, err
	}

	destinationIPAddress := ""
	if wantLb && lbIP == nil {
		return nil, fmt.Errorf("No load balancer IP for setting up security rules for service %s", service.Name)
	}
	if lbIP != nil {
		destinationIPAddress = *lbIP
	}
	if destinationIPAddress == "" {
		destinationIPAddress = "*"
	}

	sourceRanges, err := servicehelpers.GetLoadBalancerSourceRanges(service)
	if err != nil {
		return nil, err
	}
	serviceTags := getServiceTags(service)
	var sourceAddressPrefixes []string
	if (sourceRanges == nil || servicehelpers.IsAllowAll(sourceRanges)) && len(serviceTags) == 0 {
		if !requiresInternalLoadBalancer(service) {
			sourceAddressPrefixes = []string{"Internet"}
		}
	} else {
		for _, ip := range sourceRanges {
			sourceAddressPrefixes = append(sourceAddressPrefixes, ip.String())
		}
		sourceAddressPrefixes = append(sourceAddressPrefixes, serviceTags...)
	}
	expectedSecurityRules := []network.SecurityRule{}

	if wantLb {
		expectedSecurityRules = make([]network.SecurityRule, len(ports)*len(sourceAddressPrefixes))

		for i, port := range ports {
			_, securityProto, _, err := getProtocolsFromKubernetesProtocol(port.Protocol)
			if err != nil {
				return nil, err
			}
			for j := range sourceAddressPrefixes {
				ix := i*len(sourceAddressPrefixes) + j
				securityRuleName := az.getSecurityRuleName(service, port, sourceAddressPrefixes[j])
				expectedSecurityRules[ix] = network.SecurityRule{
					Name: to.StringPtr(securityRuleName),
					SecurityRulePropertiesFormat: &network.SecurityRulePropertiesFormat{
						Protocol:                 *securityProto,
						SourcePortRange:          to.StringPtr("*"),
						DestinationPortRange:     to.StringPtr(strconv.Itoa(int(port.Port))),
						SourceAddressPrefix:      to.StringPtr(sourceAddressPrefixes[j]),
						DestinationAddressPrefix: to.StringPtr(destinationIPAddress),
						Access:                   network.SecurityRuleAccessAllow,
						Direction:                network.SecurityRuleDirectionInbound,
					},
				}
			}
		}
	}

	for _, r := range expectedSecurityRules {
		klog.V(10).Infof("Expecting security rule for %s: %s:%s -> %s:%s", service.Name, *r.SourceAddressPrefix, *r.SourcePortRange, *r.DestinationAddressPrefix, *r.DestinationPortRange)
	}

	// update security rules
	dirtySg := false
	var updatedRules []network.SecurityRule
	if sg.SecurityGroupPropertiesFormat != nil && sg.SecurityGroupPropertiesFormat.SecurityRules != nil {
		updatedRules = *sg.SecurityGroupPropertiesFormat.SecurityRules
	}

	for _, r := range updatedRules {
		klog.V(10).Infof("Existing security rule while processing %s: %s:%s -> %s:%s", service.Name, logSafe(r.SourceAddressPrefix), logSafe(r.SourcePortRange), logSafeCollection(r.DestinationAddressPrefix, r.DestinationAddressPrefixes), logSafe(r.DestinationPortRange))
	}

	// update security rules: remove unwanted rules that belong privately
	// to this service
	for i := len(updatedRules) - 1; i >= 0; i-- {
		existingRule := updatedRules[i]
		if az.serviceOwnsRule(service, *existingRule.Name) {
			klog.V(10).Infof("reconcile(%s)(%t): sg rule(%s) - considering evicting", serviceName, wantLb, *existingRule.Name)
			keepRule := false
			if findSecurityRule(expectedSecurityRules, existingRule) {
				klog.V(10).Infof("reconcile(%s)(%t): sg rule(%s) - keeping", serviceName, wantLb, *existingRule.Name)
				keepRule = true
			}
			if !keepRule {
				klog.V(10).Infof("reconcile(%s)(%t): sg rule(%s) - dropping", serviceName, wantLb, *existingRule.Name)
				updatedRules = append(updatedRules[:i], updatedRules[i+1:]...)
				dirtySg = true
			}
		}
	}
	// update security rules: if the service uses a shared rule and is being deleted,
	// then remove it from the shared rule
	if useSharedSecurityRule(service) && !wantLb {
		for _, port := range ports {
			for _, sourceAddressPrefix := range sourceAddressPrefixes {
				sharedRuleName := az.getSecurityRuleName(service, port, sourceAddressPrefix)
				sharedIndex, sharedRule, sharedRuleFound := findSecurityRuleByName(updatedRules, sharedRuleName)
				if !sharedRuleFound {
					klog.V(4).Infof("Expected to find shared rule %s for service %s being deleted, but did not", sharedRuleName, service.Name)
					return nil, fmt.Errorf("Expected to find shared rule %s for service %s being deleted, but did not", sharedRuleName, service.Name)
				}
				if sharedRule.DestinationAddressPrefixes == nil {
					klog.V(4).Infof("Expected to have array of destinations in shared rule for service %s being deleted, but did not", service.Name)
					return nil, fmt.Errorf("Expected to have array of destinations in shared rule for service %s being deleted, but did not", service.Name)
				}
				existingPrefixes := *sharedRule.DestinationAddressPrefixes
				addressIndex, found := findIndex(existingPrefixes, destinationIPAddress)
				if !found {
					klog.V(4).Infof("Expected to find destination address %s in shared rule %s for service %s being deleted, but did not", destinationIPAddress, sharedRuleName, service.Name)
					return nil, fmt.Errorf("Expected to find destination address %s in shared rule %s for service %s being deleted, but did not", destinationIPAddress, sharedRuleName, service.Name)
				}
				if len(existingPrefixes) == 1 {
					updatedRules = append(updatedRules[:sharedIndex], updatedRules[sharedIndex+1:]...)
				} else {
					newDestinations := append(existingPrefixes[:addressIndex], existingPrefixes[addressIndex+1:]...)
					sharedRule.DestinationAddressPrefixes = &newDestinations
					updatedRules[sharedIndex] = sharedRule
				}
				dirtySg = true
			}
		}
	}

	// update security rules: prepare rules for consolidation
	for index, rule := range updatedRules {
		if allowsConsolidation(rule) {
			updatedRules[index] = makeConsolidatable(rule)
		}
	}
	for index, rule := range expectedSecurityRules {
		if allowsConsolidation(rule) {
			expectedSecurityRules[index] = makeConsolidatable(rule)
		}
	}
	// update security rules: add needed
	for _, expectedRule := range expectedSecurityRules {
		foundRule := false
		if findSecurityRule(updatedRules, expectedRule) {
			klog.V(10).Infof("reconcile(%s)(%t): sg rule(%s) - already exists", serviceName, wantLb, *expectedRule.Name)
			foundRule = true
		}
		if foundRule && allowsConsolidation(expectedRule) {
			index, _ := findConsolidationCandidate(updatedRules, expectedRule)
			updatedRules[index] = consolidate(updatedRules[index], expectedRule)
			dirtySg = true
		}
		if !foundRule {
			klog.V(10).Infof("reconcile(%s)(%t): sg rule(%s) - adding", serviceName, wantLb, *expectedRule.Name)

			nextAvailablePriority, err := getNextAvailablePriority(updatedRules)
			if err != nil {
				return nil, err
			}

			expectedRule.Priority = to.Int32Ptr(nextAvailablePriority)
			updatedRules = append(updatedRules, expectedRule)
			dirtySg = true
		}
	}

	for _, r := range updatedRules {
		klog.V(10).Infof("Updated security rule while processing %s: %s:%s -> %s:%s", service.Name, logSafe(r.SourceAddressPrefix), logSafe(r.SourcePortRange), logSafeCollection(r.DestinationAddressPrefix, r.DestinationAddressPrefixes), logSafe(r.DestinationPortRange))
	}

	if dirtySg {
		sg.SecurityRules = &updatedRules
		klog.V(2).Infof("reconcileSecurityGroup for service(%s): sg(%s) - updating", serviceName, *sg.Name)
		klog.V(10).Infof("CreateOrUpdateSecurityGroup(%q): start", *sg.Name)
		err := az.CreateOrUpdateSecurityGroup(service, sg)
		if err != nil {
			klog.V(2).Infof("ensure(%s) abort backoff: sg(%s) - updating", serviceName, *sg.Name)
			return nil, err
		}
		klog.V(10).Infof("CreateOrUpdateSecurityGroup(%q): end", *sg.Name)
	}
	return &sg, nil
}

func logSafe(s *string) string {
	if s == nil {
		return "(nil)"
	}
	return *s
}

func logSafeCollection(s *string, strs *[]string) string {
	if s == nil {
		if strs == nil {
			return "(nil)"
		}
		return "[" + strings.Join(*strs, ",") + "]"
	}
	return *s
}

func findSecurityRuleByName(rules []network.SecurityRule, ruleName string) (int, network.SecurityRule, bool) {
	for index, rule := range rules {
		if rule.Name != nil && strings.EqualFold(*rule.Name, ruleName) {
			return index, rule, true
		}
	}
	return 0, network.SecurityRule{}, false
}

func findIndex(strs []string, s string) (int, bool) {
	for index, str := range strs {
		if strings.EqualFold(str, s) {
			return index, true
		}
	}
	return 0, false
}

func allowsConsolidation(rule network.SecurityRule) bool {
	return strings.HasPrefix(to.String(rule.Name), "shared")
}

func findConsolidationCandidate(rules []network.SecurityRule, rule network.SecurityRule) (int, bool) {
	for index, r := range rules {
		if allowsConsolidation(r) {
			if strings.EqualFold(to.String(r.Name), to.String(rule.Name)) {
				return index, true
			}
		}
	}

	return 0, false
}

func makeConsolidatable(rule network.SecurityRule) network.SecurityRule {
	return network.SecurityRule{
		Name: rule.Name,
		SecurityRulePropertiesFormat: &network.SecurityRulePropertiesFormat{
			Priority:                   rule.Priority,
			Protocol:                   rule.Protocol,
			SourcePortRange:            rule.SourcePortRange,
			SourcePortRanges:           rule.SourcePortRanges,
			DestinationPortRange:       rule.DestinationPortRange,
			DestinationPortRanges:      rule.DestinationPortRanges,
			SourceAddressPrefix:        rule.SourceAddressPrefix,
			SourceAddressPrefixes:      rule.SourceAddressPrefixes,
			DestinationAddressPrefixes: collectionOrSingle(rule.DestinationAddressPrefixes, rule.DestinationAddressPrefix),
			Access:                     rule.Access,
			Direction:                  rule.Direction,
		},
	}
}

func consolidate(existingRule network.SecurityRule, newRule network.SecurityRule) network.SecurityRule {
	destinations := appendElements(existingRule.SecurityRulePropertiesFormat.DestinationAddressPrefixes, newRule.DestinationAddressPrefix, newRule.DestinationAddressPrefixes)
	destinations = deduplicate(destinations) // there are transient conditions during controller startup where it tries to add a service that is already added

	return network.SecurityRule{
		Name: existingRule.Name,
		SecurityRulePropertiesFormat: &network.SecurityRulePropertiesFormat{
			Priority:                   existingRule.Priority,
			Protocol:                   existingRule.Protocol,
			SourcePortRange:            existingRule.SourcePortRange,
			SourcePortRanges:           existingRule.SourcePortRanges,
			DestinationPortRange:       existingRule.DestinationPortRange,
			DestinationPortRanges:      existingRule.DestinationPortRanges,
			SourceAddressPrefix:        existingRule.SourceAddressPrefix,
			SourceAddressPrefixes:      existingRule.SourceAddressPrefixes,
			DestinationAddressPrefixes: destinations,
			Access:                     existingRule.Access,
			Direction:                  existingRule.Direction,
		},
	}
}

func collectionOrSingle(collection *[]string, s *string) *[]string {
	if collection != nil && len(*collection) > 0 {
		return collection
	}
	if s == nil {
		return &[]string{}
	}
	return &[]string{*s}
}

func appendElements(collection *[]string, appendString *string, appendStrings *[]string) *[]string {
	newCollection := []string{}

	if collection != nil {
		newCollection = append(newCollection, *collection...)
	}
	if appendString != nil {
		newCollection = append(newCollection, *appendString)
	}
	if appendStrings != nil {
		newCollection = append(newCollection, *appendStrings...)
	}

	return &newCollection
}

func deduplicate(collection *[]string) *[]string {
	if collection == nil {
		return nil
	}

	seen := map[string]bool{}
	result := make([]string, 0, len(*collection))

	for _, v := range *collection {
		if seen[v] == true {
			// skip this element
		} else {
			seen[v] = true
			result = append(result, v)
		}
	}

	return &result
}

// This reconciles the PublicIP resources similar to how the LB is reconciled.
func (az *Cloud) reconcilePublicIP(clusterName string, service *v1.Service, lbName string, wantLb bool) (*network.PublicIPAddress, error) {
	isInternal := requiresInternalLoadBalancer(service)
	serviceName := getServiceName(service)
	var lb *network.LoadBalancer
	var desiredPipName string
	var err error
	var shouldPIPExisted bool
	if !isInternal && wantLb {
		desiredPipName, shouldPIPExisted, err = az.determinePublicIPName(clusterName, service)
		if err != nil {
			return nil, err
		}
	}

	if lbName != "" {
		loadBalancer, _, err := az.getAzureLoadBalancer(lbName)
		if err != nil {
			return nil, err
		}
		lb = &loadBalancer
	}

	pipResourceGroup := az.getPublicIPAddressResourceGroup(service)

	pips, err := az.ListPIP(service, pipResourceGroup)
	if err != nil {
		return nil, err
	}

	var found bool
	var pipsToBeDeleted []*network.PublicIPAddress
	for i := range pips {
		pip := pips[i]
		pipName := *pip.Name
		if serviceOwnsPublicIP(&pip, clusterName, serviceName) {
			// We need to process for pips belong to this service
			if wantLb && !isInternal && pipName == desiredPipName {
				// This is the only case we should preserve the
				// Public ip resource with match service tag
				found = true
			} else {
				pipsToBeDeleted = append(pipsToBeDeleted, &pip)
			}
		} else if wantLb && !isInternal && pipName == desiredPipName {
			found = true
		}
	}
	if !isInternal && shouldPIPExisted && !found && wantLb {
		return nil, fmt.Errorf("reconcilePublicIP for service(%s): pip(%s) not found", serviceName, desiredPipName)
	}
	var deleteFuncs []func() error
	for _, pip := range pipsToBeDeleted {
		pipCopy := *pip
		deleteFuncs = append(deleteFuncs, func() error {
			klog.V(2).Infof("reconcilePublicIP for service(%s): pip(%s) - deleting", serviceName, *pip.Name)
			return az.safeDeletePublicIP(service, pipResourceGroup, &pipCopy, lb)
		})
	}
	errs := utilerrors.AggregateGoroutines(deleteFuncs...)
	if errs != nil {
		return nil, utilerrors.Flatten(errs)
	}

	if !isInternal && wantLb {
		// Confirm desired public ip resource exists
		var pip *network.PublicIPAddress
		domainNameLabel := getPublicIPDomainNameLabel(service)
		if pip, err = az.ensurePublicIPExists(service, desiredPipName, domainNameLabel, clusterName, shouldPIPExisted); err != nil {
			return nil, err
		}
		return pip, nil
	}
	return nil, nil
}

// safeDeletePublicIP deletes public IP by removing its reference first.
func (az *Cloud) safeDeletePublicIP(service *v1.Service, pipResourceGroup string, pip *network.PublicIPAddress, lb *network.LoadBalancer) error {
	// Remove references if pip.IPConfiguration is not nil.
	if pip.PublicIPAddressPropertiesFormat != nil &&
		pip.PublicIPAddressPropertiesFormat.IPConfiguration != nil &&
		lb != nil && lb.LoadBalancerPropertiesFormat != nil &&
		lb.LoadBalancerPropertiesFormat.FrontendIPConfigurations != nil {
		referencedLBRules := []network.SubResource{}
		frontendIPConfigUpdated := false
		loadBalancerRuleUpdated := false

		// Check whether there are still frontend IP configurations referring to it.
		ipConfigurationID := to.String(pip.PublicIPAddressPropertiesFormat.IPConfiguration.ID)
		if ipConfigurationID != "" {
			lbFrontendIPConfigs := *lb.LoadBalancerPropertiesFormat.FrontendIPConfigurations
			for i := len(lbFrontendIPConfigs) - 1; i >= 0; i-- {
				config := lbFrontendIPConfigs[i]
				if strings.EqualFold(ipConfigurationID, to.String(config.ID)) {
					if config.FrontendIPConfigurationPropertiesFormat != nil &&
						config.FrontendIPConfigurationPropertiesFormat.LoadBalancingRules != nil {
						referencedLBRules = *config.FrontendIPConfigurationPropertiesFormat.LoadBalancingRules
					}

					frontendIPConfigUpdated = true
					lbFrontendIPConfigs = append(lbFrontendIPConfigs[:i], lbFrontendIPConfigs[i+1:]...)
					break
				}
			}

			if frontendIPConfigUpdated {
				lb.LoadBalancerPropertiesFormat.FrontendIPConfigurations = &lbFrontendIPConfigs
			}
		}

		// Check whether there are still load balancer rules referring to it.
		if len(referencedLBRules) > 0 {
			referencedLBRuleIDs := sets.NewString()
			for _, refer := range referencedLBRules {
				referencedLBRuleIDs.Insert(to.String(refer.ID))
			}

			if lb.LoadBalancerPropertiesFormat.LoadBalancingRules != nil {
				lbRules := *lb.LoadBalancerPropertiesFormat.LoadBalancingRules
				for i := len(lbRules) - 1; i >= 0; i-- {
					ruleID := to.String(lbRules[i].ID)
					if ruleID != "" && referencedLBRuleIDs.Has(ruleID) {
						loadBalancerRuleUpdated = true
						lbRules = append(lbRules[:i], lbRules[i+1:]...)
					}
				}

				if loadBalancerRuleUpdated {
					lb.LoadBalancerPropertiesFormat.LoadBalancingRules = &lbRules
				}
			}
		}

		// Update load balancer when frontendIPConfigUpdated or loadBalancerRuleUpdated.
		if frontendIPConfigUpdated || loadBalancerRuleUpdated {
			err := az.CreateOrUpdateLB(service, *lb)
			if err != nil {
				klog.Errorf("safeDeletePublicIP for service(%s) failed with error: %v", getServiceName(service), err)
				return err
			}
		}
	}

	pipName := to.String(pip.Name)
	klog.V(10).Infof("DeletePublicIP(%s, %q): start", pipResourceGroup, pipName)
	err := az.DeletePublicIP(service, pipResourceGroup, pipName)
	if err != nil {
		if err = ignoreStatusNotFoundFromError(err); err != nil {
			return err
		}
	}
	klog.V(10).Infof("DeletePublicIP(%s, %q): end", pipResourceGroup, pipName)

	return nil
}

func findProbe(probes []network.Probe, probe network.Probe) bool {
	for _, existingProbe := range probes {
		if strings.EqualFold(to.String(existingProbe.Name), to.String(probe.Name)) && to.Int32(existingProbe.Port) == to.Int32(probe.Port) {
			return true
		}
	}
	return false
}

func findRule(rules []network.LoadBalancingRule, rule network.LoadBalancingRule, wantLB bool) bool {
	for _, existingRule := range rules {
		if strings.EqualFold(to.String(existingRule.Name), to.String(rule.Name)) &&
			equalLoadBalancingRulePropertiesFormat(existingRule.LoadBalancingRulePropertiesFormat, rule.LoadBalancingRulePropertiesFormat, wantLB) {
			return true
		}
	}
	return false
}

// equalLoadBalancingRulePropertiesFormat checks whether the provided LoadBalancingRulePropertiesFormat are equal.
// Note: only fields used in reconcileLoadBalancer are considered.
func equalLoadBalancingRulePropertiesFormat(s *network.LoadBalancingRulePropertiesFormat, t *network.LoadBalancingRulePropertiesFormat, wantLB bool) bool {
	if s == nil || t == nil {
		return false
	}

	properties := reflect.DeepEqual(s.Protocol, t.Protocol) &&
		reflect.DeepEqual(s.FrontendIPConfiguration, t.FrontendIPConfiguration) &&
		reflect.DeepEqual(s.BackendAddressPool, t.BackendAddressPool) &&
		reflect.DeepEqual(s.LoadDistribution, t.LoadDistribution) &&
		reflect.DeepEqual(s.FrontendPort, t.FrontendPort) &&
		reflect.DeepEqual(s.BackendPort, t.BackendPort) &&
		reflect.DeepEqual(s.EnableFloatingIP, t.EnableFloatingIP) &&
		reflect.DeepEqual(s.EnableTCPReset, t.EnableTCPReset) &&
		reflect.DeepEqual(s.DisableOutboundSnat, t.DisableOutboundSnat)

	if wantLB {
		return properties && reflect.DeepEqual(s.IdleTimeoutInMinutes, t.IdleTimeoutInMinutes)
	}
	return properties
}

// This compares rule's Name, Protocol, SourcePortRange, DestinationPortRange, SourceAddressPrefix, Access, and Direction.
// Note that it compares rule's DestinationAddressPrefix only when it's not consolidated rule as such rule does not have DestinationAddressPrefix defined.
// We intentionally do not compare DestinationAddressPrefixes in consolidated case because reconcileSecurityRule has to consider the two rules equal,
// despite different DestinationAddressPrefixes, in order to give it a chance to consolidate the two rules.
func findSecurityRule(rules []network.SecurityRule, rule network.SecurityRule) bool {
	for _, existingRule := range rules {
		if !strings.EqualFold(to.String(existingRule.Name), to.String(rule.Name)) {
			continue
		}
		if existingRule.Protocol != rule.Protocol {
			continue
		}
		if !strings.EqualFold(to.String(existingRule.SourcePortRange), to.String(rule.SourcePortRange)) {
			continue
		}
		if !strings.EqualFold(to.String(existingRule.DestinationPortRange), to.String(rule.DestinationPortRange)) {
			continue
		}
		if !strings.EqualFold(to.String(existingRule.SourceAddressPrefix), to.String(rule.SourceAddressPrefix)) {
			continue
		}
		if !allowsConsolidation(existingRule) && !allowsConsolidation(rule) {
			if !strings.EqualFold(to.String(existingRule.DestinationAddressPrefix), to.String(rule.DestinationAddressPrefix)) {
				continue
			}
		}
		if existingRule.Access != rule.Access {
			continue
		}
		if existingRule.Direction != rule.Direction {
			continue
		}
		return true
	}
	return false
}

func (az *Cloud) getPublicIPAddressResourceGroup(service *v1.Service) string {
	if resourceGroup, found := service.Annotations[ServiceAnnotationLoadBalancerResourceGroup]; found {
		resourceGroupName := strings.TrimSpace(resourceGroup)
		if len(resourceGroupName) > 0 {
			return resourceGroupName
		}
	}

	return az.ResourceGroup
}

// Check if service requires an internal load balancer.
func requiresInternalLoadBalancer(service *v1.Service) bool {
	if l, found := service.Annotations[ServiceAnnotationLoadBalancerInternal]; found {
		return l == "true"
	}

	return false
}

func subnet(service *v1.Service) *string {
	if requiresInternalLoadBalancer(service) {
		if l, found := service.Annotations[ServiceAnnotationLoadBalancerInternalSubnet]; found && strings.TrimSpace(l) != "" {
			return &l
		}
	}

	return nil
}

// getServiceLoadBalancerMode parses the mode value.
// if the value is __auto__ it returns isAuto = TRUE.
// if anything else it returns the unique VM set names after triming spaces.
func getServiceLoadBalancerMode(service *v1.Service) (hasMode bool, isAuto bool, vmSetNames []string) {
	mode, hasMode := service.Annotations[ServiceAnnotationLoadBalancerMode]
	mode = strings.TrimSpace(mode)
	isAuto = strings.EqualFold(mode, ServiceAnnotationLoadBalancerAutoModeValue)
	if !isAuto {
		// Break up list of "AS1,AS2"
		vmSetParsedList := strings.Split(mode, ",")

		// Trim the VM set names and remove duplicates
		//  e.g. {"AS1"," AS2", "AS3", "AS3"} => {"AS1", "AS2", "AS3"}
		vmSetNameSet := sets.NewString()
		for _, v := range vmSetParsedList {
			vmSetNameSet.Insert(strings.TrimSpace(v))
		}

		vmSetNames = vmSetNameSet.List()
	}

	return hasMode, isAuto, vmSetNames
}

func useSharedSecurityRule(service *v1.Service) bool {
	if l, ok := service.Annotations[ServiceAnnotationSharedSecurityRule]; ok {
		return l == "true"
	}

	return false
}

func getServiceTags(service *v1.Service) []string {
	if service == nil {
		return nil
	}

	if serviceTags, found := service.Annotations[ServiceAnnotationAllowedServiceTag]; found {
		result := []string{}
		tags := strings.Split(strings.TrimSpace(serviceTags), ",")
		for _, tag := range tags {
			serviceTag := strings.TrimSpace(tag)
			if serviceTag != "" {
				result = append(result, serviceTag)
			}
		}

		return result
	}

	return nil
}

func serviceOwnsPublicIP(pip *network.PublicIPAddress, clusterName, serviceName string) bool {
	if pip != nil && pip.Tags != nil {
		serviceTag := pip.Tags[serviceTagKey]
		clusterTag := pip.Tags[clusterNameKey]

		if serviceTag != nil && *serviceTag == serviceName {
			// Backward compatible for clusters upgraded from old releases.
			// In such case, only "service" tag is set.
			if clusterTag == nil {
				return true
			}

			// If cluster name tag is set, then return true if it matches.
			if *clusterTag == clusterName {
				return true
			}
		}
	}
	return false
}
