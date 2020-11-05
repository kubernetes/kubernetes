// +build !providerless

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
	"sort"
	"strconv"
	"strings"

	"github.com/Azure/azure-sdk-for-go/services/network/mgmt/2019-06-01/network"
	"github.com/Azure/go-autorest/autorest/to"

	v1 "k8s.io/api/core/v1"
	utilerrors "k8s.io/apimachinery/pkg/util/errors"
	"k8s.io/apimachinery/pkg/util/sets"
	cloudprovider "k8s.io/cloud-provider"
	servicehelpers "k8s.io/cloud-provider/service/helpers"
	"k8s.io/klog/v2"
	azcache "k8s.io/legacy-cloud-providers/azure/cache"
	"k8s.io/legacy-cloud-providers/azure/metrics"
	"k8s.io/legacy-cloud-providers/azure/retry"
	utilnet "k8s.io/utils/net"
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

	// ServiceAnnotationIPTagsForPublicIP specifies the iptags used when dynamically creating a public ip
	ServiceAnnotationIPTagsForPublicIP = "service.beta.kubernetes.io/azure-pip-ip-tags"

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
	// TODO(feiskyer): disable-tcp-reset annotations has been depracated since v1.18, it would removed on v1.20.
	ServiceAnnotationLoadBalancerDisableTCPReset = "service.beta.kubernetes.io/azure-load-balancer-disable-tcp-reset"

	// serviceTagKey is the service key applied for public IP tags.
	serviceTagKey = "service"
	// clusterNameKey is the cluster name key applied for public IP tags.
	clusterNameKey = "kubernetes-cluster-name"
	// serviceUsingDNSKey is the service name consuming the DNS label on the public IP
	serviceUsingDNSKey = "kubernetes-dns-label-service"

	defaultLoadBalancerSourceRanges = "0.0.0.0/0"
)

// GetLoadBalancer returns whether the specified load balancer and its components exist, and
// if so, what its status is.
func (az *Cloud) GetLoadBalancer(ctx context.Context, clusterName string, service *v1.Service) (status *v1.LoadBalancerStatus, exists bool, err error) {
	// Since public IP is not a part of the load balancer on Azure,
	// there is a chance that we could orphan public IP resources while we delete the load blanacer (kubernetes/kubernetes#80571).
	// We need to make sure the existence of the load balancer depends on the load balancer resource and public IP resource on Azure.
	existsPip := func() bool {
		pipName, _, err := az.determinePublicIPName(clusterName, service)
		if err != nil {
			return false
		}
		pipResourceGroup := az.getPublicIPAddressResourceGroup(service)
		_, existsPip, err := az.getPublicIPAddress(pipResourceGroup, pipName)
		if err != nil {
			return false
		}
		return existsPip
	}()

	_, status, existsLb, err := az.getServiceLoadBalancer(service, clusterName, nil, false)
	if err != nil {
		return nil, existsPip, err
	}

	// Return exists = false only if the load balancer and the public IP are not found on Azure
	if !existsLb && !existsPip {
		serviceName := getServiceName(service)
		klog.V(5).Infof("getloadbalancer (cluster:%s) (service:%s) - doesn't exist", clusterName, serviceName)
		return nil, false, nil
	}

	// Return exists = true if either the load balancer or the public IP (or both) exists
	return status, true, nil
}

func getPublicIPDomainNameLabel(service *v1.Service) (string, bool) {
	if labelName, found := service.Annotations[ServiceAnnotationDNSLabelName]; found {
		return labelName, found
	}
	return "", false
}

// EnsureLoadBalancer creates a new load balancer 'name', or updates the existing one. Returns the status of the balancer
func (az *Cloud) EnsureLoadBalancer(ctx context.Context, clusterName string, service *v1.Service, nodes []*v1.Node) (*v1.LoadBalancerStatus, error) {
	// When a client updates the internal load balancer annotation,
	// the service may be switched from an internal LB to a public one, or vise versa.
	// Here we'll firstly ensure service do not lie in the opposite LB.
	serviceName := getServiceName(service)
	klog.V(5).Infof("ensureloadbalancer(%s): START clusterName=%q", serviceName, clusterName)

	mc := metrics.NewMetricContext("services", "ensure_loadbalancer", az.ResourceGroup, az.SubscriptionID, serviceName)
	isOperationSucceeded := false
	defer func() {
		mc.ObserveOperationWithResult(isOperationSucceeded)
	}()

	lb, err := az.reconcileLoadBalancer(clusterName, service, nodes, true /* wantLb */)
	if err != nil {
		klog.Errorf("reconcileLoadBalancer(%s) failed: %v", serviceName, err)
		return nil, err
	}

	lbStatus, err := az.getServiceLoadBalancerStatus(service, lb)
	if err != nil {
		klog.Errorf("getServiceLoadBalancerStatus(%s) failed: %v", serviceName, err)
		return nil, err
	}

	var serviceIP *string
	if lbStatus != nil && len(lbStatus.Ingress) > 0 {
		serviceIP = &lbStatus.Ingress[0].IP
	}
	klog.V(2).Infof("EnsureLoadBalancer: reconciling security group for service %q with IP %q, wantLb = true", serviceName, logSafe(serviceIP))
	if _, err := az.reconcileSecurityGroup(clusterName, service, serviceIP, true /* wantLb */); err != nil {
		klog.Errorf("reconcileSecurityGroup(%s) failed: %#v", serviceName, err)
		return nil, err
	}

	updateService := updateServiceLoadBalancerIP(service, to.String(serviceIP))
	flippedService := flipServiceInternalAnnotation(updateService)
	if _, err := az.reconcileLoadBalancer(clusterName, flippedService, nil, false /* wantLb */); err != nil {
		klog.Errorf("reconcileLoadBalancer(%s) failed: %#v", serviceName, err)
		return nil, err
	}

	// lb is not reused here because the ETAG may be changed in above operations, hence reconcilePublicIP() would get lb again from cache.
	if _, err := az.reconcilePublicIP(clusterName, updateService, to.String(lb.Name), true /* wantLb */); err != nil {
		klog.Errorf("reconcilePublicIP(%s) failed: %#v", serviceName, err)
		return nil, err
	}

	isOperationSucceeded = true

	return lbStatus, nil
}

// UpdateLoadBalancer updates hosts under the specified load balancer.
func (az *Cloud) UpdateLoadBalancer(ctx context.Context, clusterName string, service *v1.Service, nodes []*v1.Node) error {
	if !az.shouldUpdateLoadBalancer(clusterName, service) {
		klog.V(2).Infof("UpdateLoadBalancer: skipping service %s because it is either being deleted or does not exist anymore", service.Name)
		return nil
	}
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

	mc := metrics.NewMetricContext("services", "ensure_loadbalancer_deleted", az.ResourceGroup, az.SubscriptionID, serviceName)
	isOperationSucceeded := false
	defer func() {
		mc.ObserveOperationWithResult(isOperationSucceeded)
	}()

	serviceIPToCleanup, err := az.findServiceIPAddress(ctx, clusterName, service, isInternal)
	if err != nil && !retry.HasStatusForbiddenOrIgnoredError(err) {
		return err
	}

	klog.V(2).Infof("EnsureLoadBalancerDeleted: reconciling security group for service %q with IP %q, wantLb = false", serviceName, serviceIPToCleanup)
	if _, err := az.reconcileSecurityGroup(clusterName, service, &serviceIPToCleanup, false /* wantLb */); err != nil {
		return err
	}

	if _, err := az.reconcileLoadBalancer(clusterName, service, nil, false /* wantLb */); err != nil && !retry.HasStatusForbiddenOrIgnoredError(err) {
		return err
	}

	if _, err := az.reconcilePublicIP(clusterName, service, "", false /* wantLb */); err != nil {
		return err
	}

	klog.V(2).Infof("Delete service (%s): FINISH", serviceName)
	isOperationSucceeded = true

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

// cleanBackendpoolForPrimarySLB decouples the unwanted nodes from the standard load balancer.
// This is needed because when migrating from single SLB to multiple SLBs, The existing
// SLB's backend pool contains nodes from different agent pools, while we only want the
// nodes from the primary agent pool to join the backend pool.
func (az *Cloud) cleanBackendpoolForPrimarySLB(primarySLB *network.LoadBalancer, service *v1.Service, clusterName string) (*network.LoadBalancer, error) {
	lbBackendPoolName := getBackendPoolName(clusterName, service)
	lbResourceGroup := az.getLoadBalancerResourceGroup()
	lbBackendPoolID := az.getBackendPoolID(to.String(primarySLB.Name), lbResourceGroup, lbBackendPoolName)
	newBackendPools := make([]network.BackendAddressPool, 0)
	if primarySLB.LoadBalancerPropertiesFormat != nil && primarySLB.BackendAddressPools != nil {
		newBackendPools = *primarySLB.BackendAddressPools
	}
	vmSetNameToBackendIPConfigurationsToBeDeleted := make(map[string][]network.InterfaceIPConfiguration)
	for j, bp := range newBackendPools {
		if strings.EqualFold(to.String(bp.Name), lbBackendPoolName) {
			klog.V(2).Infof("cleanBackendpoolForPrimarySLB: checking the backend pool %s from standard load balancer %s", to.String(bp.Name), to.String(primarySLB.Name))
			if bp.BackendAddressPoolPropertiesFormat != nil && bp.BackendIPConfigurations != nil {
				for i := len(*bp.BackendIPConfigurations) - 1; i >= 0; i-- {
					ipConf := (*bp.BackendIPConfigurations)[i]
					ipConfigID := to.String(ipConf.ID)
					_, vmSetName, err := az.VMSet.GetNodeNameByIPConfigurationID(ipConfigID)
					if err != nil {
						return nil, err
					}
					primaryVMSetName := az.VMSet.GetPrimaryVMSetName()
					if !strings.EqualFold(primaryVMSetName, vmSetName) {
						klog.V(2).Infof("cleanBackendpoolForPrimarySLB: found unwanted vmSet %s, decouple it from the LB", vmSetName)
						// construct a backendPool that only contains the IP config of the node to be deleted
						interfaceIPConfigToBeDeleted := network.InterfaceIPConfiguration{
							ID: to.StringPtr(ipConfigID),
						}
						vmSetNameToBackendIPConfigurationsToBeDeleted[vmSetName] = append(vmSetNameToBackendIPConfigurationsToBeDeleted[vmSetName], interfaceIPConfigToBeDeleted)
						*bp.BackendIPConfigurations = append((*bp.BackendIPConfigurations)[:i], (*bp.BackendIPConfigurations)[i+1:]...)
					}
				}
			}
			newBackendPools[j] = bp
			break
		}
	}
	for vmSetName, backendIPConfigurationsToBeDeleted := range vmSetNameToBackendIPConfigurationsToBeDeleted {
		backendpoolToBeDeleted := &[]network.BackendAddressPool{
			{
				ID: to.StringPtr(lbBackendPoolID),
				BackendAddressPoolPropertiesFormat: &network.BackendAddressPoolPropertiesFormat{
					BackendIPConfigurations: &backendIPConfigurationsToBeDeleted,
				},
			},
		}
		// decouple the backendPool from the node
		err := az.VMSet.EnsureBackendPoolDeleted(service, lbBackendPoolID, vmSetName, backendpoolToBeDeleted)
		if err != nil {
			return nil, err
		}
		primarySLB.BackendAddressPools = &newBackendPools
	}
	return primarySLB, nil
}

// getServiceLoadBalancer gets the loadbalancer for the service if it already exists.
// If wantLb is TRUE then -it selects a new load balancer.
// In case the selected load balancer does not exist it returns network.LoadBalancer struct
// with added metadata (such as name, location) and existsLB set to FALSE.
// By default - cluster default LB is returned.
func (az *Cloud) getServiceLoadBalancer(service *v1.Service, clusterName string, nodes []*v1.Node, wantLb bool) (lb *network.LoadBalancer, status *v1.LoadBalancerStatus, exists bool, err error) {
	isInternal := requiresInternalLoadBalancer(service)
	var defaultLB *network.LoadBalancer
	primaryVMSetName := az.VMSet.GetPrimaryVMSetName()
	defaultLBName := az.getAzureLoadBalancerName(clusterName, primaryVMSetName, isInternal)
	useMultipleSLBs := az.useStandardLoadBalancer() && az.EnableMultipleStandardLoadBalancers

	existingLBs, err := az.ListLB(service)
	if err != nil {
		return nil, nil, false, err
	}

	// check if the service already has a load balancer
	for i := range existingLBs {
		existingLB := existingLBs[i]
		if strings.EqualFold(to.String(existingLB.Name), clusterName) && useMultipleSLBs {
			cleanedLB, err := az.cleanBackendpoolForPrimarySLB(&existingLB, service, clusterName)
			if err != nil {
				return nil, nil, false, err
			}
			existingLB = *cleanedLB
		}
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

	hasMode, _, _ := getServiceLoadBalancerMode(service)
	useSingleSLB := az.useStandardLoadBalancer() && !az.EnableMultipleStandardLoadBalancers
	if useSingleSLB && hasMode {
		klog.Warningf("single standard load balancer doesn't work with annotation %q, would ignore it", ServiceAnnotationLoadBalancerMode)
	}

	// Service does not have a load balancer, select one.
	// Single standard load balancer doesn't need this because
	// all backends nodes should be added to same LB.
	if wantLb && !useSingleSLB {
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
	vmSetNames, err := az.VMSet.GetVMSetNames(service, nodes)
	if err != nil {
		klog.Errorf("az.selectLoadBalancer: cluster(%s) service(%s) isInternal(%t) - az.GetVMSetNames failed, err=(%v)", clusterName, serviceName, isInternal, err)
		return nil, false, err
	}
	klog.V(2).Infof("selectLoadBalancer: cluster(%s) service(%s) isInternal(%t) - vmSetNames %v", clusterName, serviceName, isInternal, *vmSetNames)

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
			var loadBalancerSKU network.LoadBalancerSkuName
			if az.useStandardLoadBalancer() {
				loadBalancerSKU = network.LoadBalancerSkuNameStandard
			} else {
				loadBalancerSKU = network.LoadBalancerSkuNameBasic
			}
			selectedLB = &network.LoadBalancer{
				Name:                         &currLBName,
				Location:                     &az.Location,
				Sku:                          &network.LoadBalancerSku{Name: loadBalancerSKU},
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
	serviceName := getServiceName(service)
	for _, ipConfiguration := range *lb.FrontendIPConfigurations {
		owns, isPrimaryService, err := az.serviceOwnsFrontendIP(ipConfiguration, service)
		if err != nil {
			return nil, fmt.Errorf("get(%s): lb(%s) - failed to filter frontend IP configs with error: %v", serviceName, to.String(lb.Name), err)
		}
		if owns {
			klog.V(2).Infof("get(%s): lb(%s) - found frontend IP config, primary service: %v", serviceName, to.String(lb.Name), isPrimaryService)

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
				pipName, err := getLastSegment(*pipID, "/")
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

			klog.V(2).Infof("getServiceLoadBalancerStatus gets ingress IP %q from frontendIPConfiguration %q for service %q", to.String(lbIP), to.String(ipConfiguration.Name), serviceName)
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

	pipResourceGroup := az.getPublicIPAddressResourceGroup(service)
	loadBalancerIP := service.Spec.LoadBalancerIP

	// Assume that the service without loadBalancerIP set is a primary service.
	// If a secondary service doesn't set the loadBalancerIP, it is not allowed to share the IP.
	if len(loadBalancerIP) == 0 {
		return az.getPublicIPName(clusterName, service), shouldPIPExisted, nil
	}

	// For the services with loadBalancerIP set, an existing public IP is required, primary
	// or secondary, or a public IP not found error would be reported.
	pip, err := az.findMatchedPIPByLoadBalancerIP(service, loadBalancerIP, pipResourceGroup)
	if err != nil {
		return "", shouldPIPExisted, err
	}

	if pip != nil && pip.Name != nil {
		return *pip.Name, shouldPIPExisted, nil
	}

	return "", shouldPIPExisted, fmt.Errorf("user supplied IP Address %s was not found in resource group %s", loadBalancerIP, pipResourceGroup)
}

func (az *Cloud) findMatchedPIPByLoadBalancerIP(service *v1.Service, loadBalancerIP, pipResourceGroup string) (*network.PublicIPAddress, error) {
	pips, err := az.ListPIP(service, pipResourceGroup)
	if err != nil {
		return nil, err
	}

	for _, pip := range pips {
		if pip.PublicIPAddressPropertiesFormat.IPAddress != nil &&
			*pip.PublicIPAddressPropertiesFormat.IPAddress == loadBalancerIP {
			return &pip, nil
		}
	}

	return nil, fmt.Errorf("findMatchedPIPByLoadBalancerIP: cannot find public IP with IP address %s in resource group %s", loadBalancerIP, pipResourceGroup)
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

	if len(service.Status.LoadBalancer.Ingress) > 0 && len(service.Status.LoadBalancer.Ingress[0].IP) > 0 {
		return service.Status.LoadBalancer.Ingress[0].IP, nil
	}

	_, lbStatus, existsLb, err := az.getServiceLoadBalancer(service, clusterName, nil, false)
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

func (az *Cloud) ensurePublicIPExists(service *v1.Service, pipName string, domainNameLabel, clusterName string, shouldPIPExisted, foundDNSLabelAnnotation bool) (*network.PublicIPAddress, error) {
	pipResourceGroup := az.getPublicIPAddressResourceGroup(service)
	pip, existsPip, err := az.getPublicIPAddress(pipResourceGroup, pipName)
	if err != nil {
		return nil, err
	}

	serviceName := getServiceName(service)

	if existsPip {
		// ensure that the service tag is good
		changed, err := bindServicesToPIP(&pip, []string{serviceName}, false)
		if err != nil {
			return nil, err
		}

		// return if pip exist and dns label is the same
		if strings.EqualFold(getDomainNameLabel(&pip), domainNameLabel) {
			if existingServiceName, ok := pip.Tags[serviceUsingDNSKey]; ok &&
				strings.EqualFold(*existingServiceName, serviceName) {
				klog.V(6).Infof("ensurePublicIPExists for service(%s): pip(%s) - "+
					"the service is using the DNS label on the public IP", serviceName, pipName)

				var rerr *retry.Error
				if changed {
					klog.V(2).Infof("ensurePublicIPExists: updating the PIP %s for the incoming service %s", pipName, serviceName)
					err = az.CreateOrUpdatePIP(service, pipResourceGroup, pip)
					if err != nil {
						return nil, err
					}

					ctx, cancel := getContextWithCancel()
					defer cancel()
					pip, rerr = az.PublicIPAddressesClient.Get(ctx, pipResourceGroup, *pip.Name, "")
					if rerr != nil {
						return nil, rerr.Error()
					}
				}

				return &pip, nil
			}
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
			IPTags:                   getServiceIPTagRequestForPublicIP(service).IPTags,
		}
		pip.Tags = map[string]*string{
			serviceTagKey:  to.StringPtr(""),
			clusterNameKey: &clusterName,
		}
		if _, err = bindServicesToPIP(&pip, []string{serviceName}, false); err != nil {
			return nil, err
		}

		if az.useStandardLoadBalancer() {
			pip.Sku = &network.PublicIPAddressSku{
				Name: network.PublicIPAddressSkuNameStandard,
			}
		}
		klog.V(2).Infof("ensurePublicIPExists for service(%s): pip(%s) - creating", serviceName, *pip.Name)
	}
	if foundDNSLabelAnnotation {
		if existingServiceName, ok := pip.Tags[serviceUsingDNSKey]; ok {
			if !strings.EqualFold(to.String(existingServiceName), serviceName) {
				return nil, fmt.Errorf("ensurePublicIPExists for service(%s): pip(%s) - there is an existing service %s consuming the DNS label on the public IP, so the service cannot set the DNS label annotation with this value", serviceName, pipName, *existingServiceName)
			}
		}

		if len(domainNameLabel) == 0 {
			pip.PublicIPAddressPropertiesFormat.DNSSettings = nil
		} else {
			if pip.PublicIPAddressPropertiesFormat.DNSSettings == nil ||
				pip.PublicIPAddressPropertiesFormat.DNSSettings.DomainNameLabel == nil {
				klog.V(6).Infof("ensurePublicIPExists for service(%s): pip(%s) - no existing DNS label on the public IP, create one", serviceName, pipName)
				pip.PublicIPAddressPropertiesFormat.DNSSettings = &network.PublicIPAddressDNSSettings{
					DomainNameLabel: &domainNameLabel,
				}
			} else {
				existingDNSLabel := pip.PublicIPAddressPropertiesFormat.DNSSettings.DomainNameLabel
				if !strings.EqualFold(to.String(existingDNSLabel), domainNameLabel) {
					return nil, fmt.Errorf("ensurePublicIPExists for service(%s): pip(%s) - there is an existing DNS label %s on the public IP", serviceName, pipName, *existingDNSLabel)
				}
			}
			pip.Tags[serviceUsingDNSKey] = &serviceName
		}
	}

	// use the same family as the clusterIP as we support IPv6 single stack as well
	// as dual-stack clusters
	ipv6 := utilnet.IsIPv6String(service.Spec.ClusterIP)
	if ipv6 {
		pip.PublicIPAddressVersion = network.IPv6
		klog.V(2).Infof("service(%s): pip(%s) - creating as ipv6 for clusterIP:%v", serviceName, *pip.Name, service.Spec.ClusterIP)

		pip.PublicIPAddressPropertiesFormat.PublicIPAllocationMethod = network.Dynamic
		if az.useStandardLoadBalancer() {
			// standard sku must have static allocation method for ipv6
			pip.PublicIPAddressPropertiesFormat.PublicIPAllocationMethod = network.Static
		}
	} else {
		pip.PublicIPAddressVersion = network.IPv4
		klog.V(2).Infof("service(%s): pip(%s) - creating as ipv4 for clusterIP:%v", serviceName, *pip.Name, service.Spec.ClusterIP)
	}

	klog.V(2).Infof("CreateOrUpdatePIP(%s, %q): start", pipResourceGroup, *pip.Name)
	err = az.CreateOrUpdatePIP(service, pipResourceGroup, pip)
	if err != nil {
		klog.V(2).Infof("ensure(%s) abort backoff: pip(%s)", serviceName, *pip.Name)
		return nil, err
	}
	klog.V(10).Infof("CreateOrUpdatePIP(%s, %q): end", pipResourceGroup, *pip.Name)

	ctx, cancel := getContextWithCancel()
	defer cancel()
	pip, rerr := az.PublicIPAddressesClient.Get(ctx, pipResourceGroup, *pip.Name, "")
	if rerr != nil {
		return nil, rerr.Error()
	}
	return &pip, nil
}

type serviceIPTagRequest struct {
	IPTagsRequestedByAnnotation bool
	IPTags                      *[]network.IPTag
}

// Get the ip tag Request for the public ip from service annotations.
func getServiceIPTagRequestForPublicIP(service *v1.Service) serviceIPTagRequest {
	if service != nil {
		if ipTagString, found := service.Annotations[ServiceAnnotationIPTagsForPublicIP]; found {
			return serviceIPTagRequest{
				IPTagsRequestedByAnnotation: true,
				IPTags:                      convertIPTagMapToSlice(getIPTagMap(ipTagString)),
			}
		}
	}

	return serviceIPTagRequest{
		IPTagsRequestedByAnnotation: false,
		IPTags:                      nil,
	}
}

func getIPTagMap(ipTagString string) map[string]string {
	outputMap := make(map[string]string)
	commaDelimitedPairs := strings.Split(strings.TrimSpace(ipTagString), ",")
	for _, commaDelimitedPair := range commaDelimitedPairs {
		splitKeyValue := strings.Split(commaDelimitedPair, "=")

		// Include only valid pairs in the return value
		// Last Write wins.
		if len(splitKeyValue) == 2 {
			tagKey := strings.TrimSpace(splitKeyValue[0])
			tagValue := strings.TrimSpace(splitKeyValue[1])

			outputMap[tagKey] = tagValue
		}
	}

	return outputMap
}

func sortIPTags(ipTags *[]network.IPTag) {
	if ipTags != nil {
		sort.Slice(*ipTags, func(i, j int) bool {
			ipTag := *ipTags
			return to.String(ipTag[i].IPTagType) < to.String(ipTag[j].IPTagType) ||
				to.String(ipTag[i].Tag) < to.String(ipTag[j].Tag)
		})
	}
}

func areIPTagsEquivalent(ipTags1 *[]network.IPTag, ipTags2 *[]network.IPTag) bool {
	sortIPTags(ipTags1)
	sortIPTags(ipTags2)

	if ipTags1 == nil {
		ipTags1 = &[]network.IPTag{}
	}

	if ipTags2 == nil {
		ipTags2 = &[]network.IPTag{}
	}

	return reflect.DeepEqual(ipTags1, ipTags2)
}

func convertIPTagMapToSlice(ipTagMap map[string]string) *[]network.IPTag {
	if ipTagMap == nil {
		return nil
	}

	if len(ipTagMap) == 0 {
		return &[]network.IPTag{}
	}

	outputTags := []network.IPTag{}
	for k, v := range ipTagMap {
		ipTag := network.IPTag{
			IPTagType: to.StringPtr(k),
			Tag:       to.StringPtr(v),
		}
		outputTags = append(outputTags, ipTag)
	}

	return &outputTags
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
	isServiceOwnsFrontendIP, isPrimaryService, err := az.serviceOwnsFrontendIP(config, service)
	if err != nil {
		return false, err
	}
	if isServiceOwnsFrontendIP && isPrimaryService && !strings.EqualFold(to.String(config.Name), lbFrontendIPConfigName) {
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

// isFrontendIPConfigUnsafeToDelete checks if a frontend IP config is safe to be deleted.
// It is safe to be deleted if and only if there is no reference from other
// loadBalancing resources, including loadBalancing rules, outbound rules, inbound NAT rules
// and inbound NAT pools.
func (az *Cloud) isFrontendIPConfigUnsafeToDelete(
	lb *network.LoadBalancer,
	service *v1.Service,
	fipConfigID *string,
) (bool, error) {
	if lb == nil || fipConfigID == nil || *fipConfigID == "" {
		return false, fmt.Errorf("isFrontendIPConfigUnsafeToDelete: incorrect parameters")
	}

	var (
		lbRules         []network.LoadBalancingRule
		outboundRules   []network.OutboundRule
		inboundNatRules []network.InboundNatRule
		inboundNatPools []network.InboundNatPool
		unsafe          bool
	)

	if lb.LoadBalancerPropertiesFormat != nil {
		if lb.LoadBalancingRules != nil {
			lbRules = *lb.LoadBalancingRules
		}
		if lb.OutboundRules != nil {
			outboundRules = *lb.OutboundRules
		}
		if lb.InboundNatRules != nil {
			inboundNatRules = *lb.InboundNatRules
		}
		if lb.InboundNatPools != nil {
			inboundNatPools = *lb.InboundNatPools
		}
	}

	// check if there are load balancing rules from other services
	// referencing this frontend IP configuration
	for _, lbRule := range lbRules {
		if lbRule.LoadBalancingRulePropertiesFormat != nil &&
			lbRule.FrontendIPConfiguration != nil &&
			lbRule.FrontendIPConfiguration.ID != nil &&
			strings.EqualFold(*lbRule.FrontendIPConfiguration.ID, *fipConfigID) {
			if !az.serviceOwnsRule(service, *lbRule.Name) {
				warningMsg := fmt.Sprintf("isFrontendIPConfigUnsafeToDelete: frontend IP configuration with ID %s on LB %s cannot be deleted because it is being referenced by load balancing rules of other services", *fipConfigID, *lb.Name)
				klog.Warning(warningMsg)
				az.Event(service, v1.EventTypeWarning, "DeletingFrontendIPConfiguration", warningMsg)
				unsafe = true
				break
			}
		}
	}

	// check if there are outbound rules
	// referencing this frontend IP configuration
	for _, outboundRule := range outboundRules {
		if outboundRule.OutboundRulePropertiesFormat != nil && outboundRule.FrontendIPConfigurations != nil {
			outboundRuleFIPConfigs := *outboundRule.FrontendIPConfigurations
			if found := findMatchedOutboundRuleFIPConfig(fipConfigID, outboundRuleFIPConfigs); found {
				warningMsg := fmt.Sprintf("isFrontendIPConfigUnsafeToDelete: frontend IP configuration with ID %s on LB %s cannot be deleted because it is being referenced by the outbound rule %s", *fipConfigID, *lb.Name, *outboundRule.Name)
				klog.Warning(warningMsg)
				az.Event(service, v1.EventTypeWarning, "DeletingFrontendIPConfiguration", warningMsg)
				unsafe = true
				break
			}
		}
	}

	// check if there are inbound NAT rules
	// referencing this frontend IP configuration
	for _, inboundNatRule := range inboundNatRules {
		if inboundNatRule.InboundNatRulePropertiesFormat != nil &&
			inboundNatRule.FrontendIPConfiguration != nil &&
			inboundNatRule.FrontendIPConfiguration.ID != nil &&
			strings.EqualFold(*inboundNatRule.FrontendIPConfiguration.ID, *fipConfigID) {
			warningMsg := fmt.Sprintf("isFrontendIPConfigUnsafeToDelete: frontend IP configuration with ID %s on LB %s cannot be deleted because it is being referenced by the inbound NAT rule %s", *fipConfigID, *lb.Name, *inboundNatRule.Name)
			klog.Warning(warningMsg)
			az.Event(service, v1.EventTypeWarning, "DeletingFrontendIPConfiguration", warningMsg)
			unsafe = true
			break
		}
	}

	// check if there are inbound NAT pools
	// referencing this frontend IP configuration
	for _, inboundNatPool := range inboundNatPools {
		if inboundNatPool.InboundNatPoolPropertiesFormat != nil &&
			inboundNatPool.FrontendIPConfiguration != nil &&
			inboundNatPool.FrontendIPConfiguration.ID != nil &&
			strings.EqualFold(*inboundNatPool.FrontendIPConfiguration.ID, *fipConfigID) {
			warningMsg := fmt.Sprintf("isFrontendIPConfigUnsafeToDelete: frontend IP configuration with ID %s on LB %s cannot be deleted because it is being referenced by the inbound NAT pool %s", *fipConfigID, *lb.Name, *inboundNatPool.Name)
			klog.Warning(warningMsg)
			az.Event(service, v1.EventTypeWarning, "DeletingFrontendIPConfiguration", warningMsg)
			unsafe = true
			break
		}
	}

	return unsafe, nil
}

func findMatchedOutboundRuleFIPConfig(fipConfigID *string, outboundRuleFIPConfigs []network.SubResource) bool {
	var found bool
	for _, config := range outboundRuleFIPConfigs {
		if config.ID != nil && strings.EqualFold(*config.ID, *fipConfigID) {
			found = true
		}
	}
	return found
}

func (az *Cloud) findFrontendIPConfigOfService(
	fipConfigs *[]network.FrontendIPConfiguration,
	service *v1.Service,
) (*network.FrontendIPConfiguration, bool, error) {
	for _, config := range *fipConfigs {
		owns, isPrimaryService, err := az.serviceOwnsFrontendIP(config, service)
		if err != nil {
			return nil, false, err
		}
		if owns {
			return &config, isPrimaryService, nil
		}
	}

	return nil, false, nil
}

func nodeNameInNodes(nodeName string, nodes []*v1.Node) bool {
	for _, node := range nodes {
		if strings.EqualFold(nodeName, node.Name) {
			return true
		}
	}
	return false
}

// reconcileLoadBalancer ensures load balancer exists and the frontend ip config is setup.
// This also reconciles the Service's Ports  with the LoadBalancer config.
// This entails adding rules/probes for expected Ports and removing stale rules/ports.
// nodes only used if wantLb is true
func (az *Cloud) reconcileLoadBalancer(clusterName string, service *v1.Service, nodes []*v1.Node, wantLb bool) (*network.LoadBalancer, error) {
	isInternal := requiresInternalLoadBalancer(service)
	isBackendPoolPreConfigured := az.isBackendPoolPreConfigured(service)
	serviceName := getServiceName(service)
	klog.V(2).Infof("reconcileLoadBalancer for service(%s) - wantLb(%t): started", serviceName, wantLb)
	lb, _, _, err := az.getServiceLoadBalancer(service, clusterName, nodes, wantLb)
	if err != nil {
		klog.Errorf("reconcileLoadBalancer: failed to get load balancer for service %q, error: %v", serviceName, err)
		return nil, err
	}
	lbName := *lb.Name
	lbResourceGroup := az.getLoadBalancerResourceGroup()
	klog.V(2).Infof("reconcileLoadBalancer for service(%s): lb(%s/%s) wantLb(%t) resolved load balancer name", serviceName, lbResourceGroup, lbName, wantLb)
	defaultLBFrontendIPConfigName := az.getDefaultFrontendIPConfigName(service)
	defaultLBFrontendIPConfigID := az.getFrontendIPConfigID(lbName, lbResourceGroup, defaultLBFrontendIPConfigName)
	lbBackendPoolName := getBackendPoolName(clusterName, service)
	lbBackendPoolID := az.getBackendPoolID(lbName, lbResourceGroup, lbBackendPoolName)

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

				var backendIPConfigurationsToBeDeleted []network.InterfaceIPConfiguration
				if bp.BackendAddressPoolPropertiesFormat != nil && bp.BackendIPConfigurations != nil {
					for _, ipConf := range *bp.BackendIPConfigurations {
						ipConfID := to.String(ipConf.ID)
						nodeName, _, err := az.VMSet.GetNodeNameByIPConfigurationID(ipConfID)
						if err != nil {
							return nil, err
						}
						// If a node is not supposed to be included in the LB, it
						// would not be in the `nodes` slice. We need to check the nodes that
						// have been added to the LB's backendpool, find the unwanted ones and
						// delete them from the pool.
						if !nodeNameInNodes(nodeName, nodes) {
							klog.V(2).Infof("reconcileLoadBalancer for service (%s)(%t): lb backendpool - found unwanted node %s, decouple it from the LB", serviceName, wantLb, nodeName)
							// construct a backendPool that only contains the IP config of the node to be deleted
							backendIPConfigurationsToBeDeleted = append(backendIPConfigurationsToBeDeleted, network.InterfaceIPConfiguration{ID: to.StringPtr(ipConfID)})
						}
					}
					if len(backendIPConfigurationsToBeDeleted) > 0 {
						backendpoolToBeDeleted := &[]network.BackendAddressPool{
							{
								ID: to.StringPtr(lbBackendPoolID),
								BackendAddressPoolPropertiesFormat: &network.BackendAddressPoolPropertiesFormat{
									BackendIPConfigurations: &backendIPConfigurationsToBeDeleted,
								},
							},
						}
						vmSetName := az.mapLoadBalancerNameToVMSet(lbName, clusterName)
						// decouple the backendPool from the node
						err = az.VMSet.EnsureBackendPoolDeleted(service, lbBackendPoolID, vmSetName, backendpoolToBeDeleted)
						if err != nil {
							return nil, err
						}
					}
				}
				break
			} else {
				klog.V(10).Infof("reconcileLoadBalancer for service (%s)(%t): lb backendpool - found other backendpool %s", serviceName, wantLb, *bp.Name)
			}
		}
		if !foundBackendPool {
			if isBackendPoolPreConfigured {
				klog.V(2).Infof("reconcileLoadBalancer for service (%s)(%t): lb backendpool - PreConfiguredBackendPoolLoadBalancerTypes %s has been set but can not find corresponding backend pool, ignoring it",
					serviceName,
					wantLb,
					az.PreConfiguredBackendPoolLoadBalancerTypes)
				isBackendPoolPreConfigured = false
			}

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

	var ownedFIPConfig *network.FrontendIPConfiguration
	if !wantLb {
		for i := len(newConfigs) - 1; i >= 0; i-- {
			config := newConfigs[i]
			isServiceOwnsFrontendIP, _, err := az.serviceOwnsFrontendIP(config, service)
			if err != nil {
				return nil, err
			}
			if isServiceOwnsFrontendIP {
				unsafe, err := az.isFrontendIPConfigUnsafeToDelete(lb, service, config.ID)
				if err != nil {
					return nil, err
				}

				// If the frontend IP configuration is not being referenced by:
				// 1. loadBalancing rules of other services with different ports;
				// 2. outbound rules;
				// 3. inbound NAT rules;
				// 4. inbound NAT pools,
				// do the deletion, or skip it.
				if !unsafe {
					var configNameToBeDeleted string
					if newConfigs[i].Name != nil {
						configNameToBeDeleted = *newConfigs[i].Name
						klog.V(2).Infof("reconcileLoadBalancer for service (%s)(%t): lb frontendconfig(%s) - dropping", serviceName, wantLb, configNameToBeDeleted)
					} else {
						klog.V(2).Infof("reconcileLoadBalancer for service (%s)(%t): nil name of lb frontendconfig", serviceName, wantLb)
					}

					newConfigs = append(newConfigs[:i], newConfigs[i+1:]...)
					dirtyConfigs = true
				}
			}
		}
	} else {
		for i := len(newConfigs) - 1; i >= 0; i-- {
			config := newConfigs[i]
			isFipChanged, err := az.isFrontendIPChanged(clusterName, config, service, defaultLBFrontendIPConfigName)
			if err != nil {
				return nil, err
			}
			if isFipChanged {
				klog.V(2).Infof("reconcileLoadBalancer for service (%s)(%t): lb frontendconfig(%s) - dropping", serviceName, wantLb, *config.Name)
				newConfigs = append(newConfigs[:i], newConfigs[i+1:]...)
				dirtyConfigs = true
			}
		}

		ownedFIPConfig, _, err = az.findFrontendIPConfigOfService(&newConfigs, service)
		if err != nil {
			return nil, err
		}

		if ownedFIPConfig == nil {
			klog.V(4).Infof("ensure(%s): lb(%s) - creating a new frontend IP config", serviceName, lbName)

			// construct FrontendIPConfigurationPropertiesFormat
			var fipConfigurationProperties *network.FrontendIPConfigurationPropertiesFormat
			if isInternal {
				// azure does not support ILB for IPv6 yet.
				// TODO: remove this check when ILB supports IPv6 *and* the SDK
				// have been rev'ed to 2019* version
				if utilnet.IsIPv6String(service.Spec.ClusterIP) {
					return nil, fmt.Errorf("ensure(%s): lb(%s) - internal load balancers does not support IPv6", serviceName, lbName)
				}

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
				domainNameLabel, found := getPublicIPDomainNameLabel(service)
				pip, err := az.ensurePublicIPExists(service, pipName, domainNameLabel, clusterName, shouldPIPExisted, found)
				if err != nil {
					return nil, err
				}
				fipConfigurationProperties = &network.FrontendIPConfigurationPropertiesFormat{
					PublicIPAddress: &network.PublicIPAddress{ID: pip.ID},
				}
			}

			newConfigs = append(newConfigs,
				network.FrontendIPConfiguration{
					Name:                                    to.StringPtr(defaultLBFrontendIPConfigName),
					ID:                                      to.StringPtr(fmt.Sprintf(frontendIPConfigIDTemplate, az.SubscriptionID, az.ResourceGroup, *lb.Name, defaultLBFrontendIPConfigName)),
					FrontendIPConfigurationPropertiesFormat: fipConfigurationProperties,
				})
			klog.V(2).Infof("reconcileLoadBalancer for service (%s)(%t): lb frontendconfig(%s) - adding", serviceName, wantLb, defaultLBFrontendIPConfigName)
			dirtyConfigs = true
		}
	}
	if dirtyConfigs {
		dirtyLb = true
		lb.FrontendIPConfigurations = &newConfigs
	}

	// update probes/rules
	if ownedFIPConfig != nil {
		if ownedFIPConfig.ID != nil {
			defaultLBFrontendIPConfigID = *ownedFIPConfig.ID
		} else {
			return nil, fmt.Errorf("reconcileLoadBalancer for service (%s)(%t): nil ID for frontend IP config", serviceName, wantLb)
		}
	}

	if wantLb {
		err = az.checkLoadBalancerResourcesConflicted(lb, defaultLBFrontendIPConfigID, service)
		if err != nil {
			return nil, err
		}
	}

	expectedProbes, expectedRules, err := az.reconcileLoadBalancerRule(service, wantLb, defaultLBFrontendIPConfigID, lbBackendPoolID, lbName, lbIdleTimeout)
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
			if isBackendPoolPreConfigured {
				klog.V(2).Infof("reconcileLoadBalancer for service(%s): lb(%s) - ignore cleanup of dirty lb because the lb is pre-configured", serviceName, lbName)
			} else {
				// When FrontendIPConfigurations is empty, we need to delete the Azure load balancer resource itself,
				// because an Azure load balancer cannot have an empty FrontendIPConfigurations collection
				klog.V(2).Infof("reconcileLoadBalancer for service(%s): lb(%s) - deleting; no remaining frontendIPConfigurations", serviceName, lbName)

				// Remove backend pools from vmSets. This is required for virtual machine scale sets before removing the LB.
				vmSetName := az.mapLoadBalancerNameToVMSet(lbName, clusterName)
				klog.V(10).Infof("EnsureBackendPoolDeleted(%s,%s) for service %s: start", lbBackendPoolID, vmSetName, serviceName)
				if _, ok := az.VMSet.(*availabilitySet); ok {
					// do nothing for availability set
					lb.BackendAddressPools = nil
				}
				err := az.VMSet.EnsureBackendPoolDeleted(service, lbBackendPoolID, vmSetName, lb.BackendAddressPools)
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
			}
		} else {
			klog.V(2).Infof("reconcileLoadBalancer: reconcileLoadBalancer for service(%s): lb(%s) - updating", serviceName, lbName)
			err := az.CreateOrUpdateLB(service, *lb)
			if err != nil {
				klog.V(2).Infof("reconcileLoadBalancer for service(%s) abort backoff: lb(%s) - updating", serviceName, lbName)
				return nil, err
			}

			if isInternal {
				// Refresh updated lb which will be used later in other places.
				newLB, exist, err := az.getAzureLoadBalancer(lbName, azcache.CacheReadTypeDefault)
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

	if wantLb && nodes != nil && !isBackendPoolPreConfigured {
		// Add the machines to the backend pool if they're not already
		vmSetName := az.mapLoadBalancerNameToVMSet(lbName, clusterName)
		// Etag would be changed when updating backend pools, so invalidate lbCache after it.
		defer az.lbCache.Delete(lbName)
		err := az.VMSet.EnsureHostsInPool(service, nodes, lbBackendPoolID, vmSetName, isInternal)
		if err != nil {
			return nil, err
		}
	}

	klog.V(2).Infof("reconcileLoadBalancer for service(%s): lb(%s) finished", serviceName, lbName)
	return lb, nil
}

// checkLoadBalancerResourcesConflicted checks if the service is consuming
// ports which are conflicted with the existing loadBalancer resources,
// including inbound NAT rule, inbound NAT pools and loadBalancing rules
func (az *Cloud) checkLoadBalancerResourcesConflicted(
	lb *network.LoadBalancer,
	frontendIPConfigID string,
	service *v1.Service,
) error {
	if service.Spec.Ports == nil {
		return nil
	}
	ports := service.Spec.Ports

	for _, port := range ports {
		if lb.LoadBalancingRules != nil {
			for _, rule := range *lb.LoadBalancingRules {
				if rule.LoadBalancingRulePropertiesFormat != nil &&
					rule.FrontendIPConfiguration != nil &&
					rule.FrontendIPConfiguration.ID != nil &&
					strings.EqualFold(*rule.FrontendIPConfiguration.ID, frontendIPConfigID) &&
					strings.EqualFold(string(rule.Protocol), string(port.Protocol)) &&
					rule.FrontendPort != nil &&
					*rule.FrontendPort == port.Port {
					// ignore self-owned rules for unit test
					if rule.Name != nil && az.serviceOwnsRule(service, *rule.Name) {
						continue
					}
					return fmt.Errorf("checkLoadBalancerResourcesConflicted: service port %s is trying to "+
						"consume the port %d which is being referenced by an existing loadBalancing rule %s with "+
						"the same protocol %s and frontend IP config with ID %s",
						port.Name,
						*rule.FrontendPort,
						*rule.Name,
						rule.Protocol,
						*rule.FrontendIPConfiguration.ID)
				}
			}
		}

		if lb.InboundNatRules != nil {
			for _, inboundNatRule := range *lb.InboundNatRules {
				if inboundNatRule.InboundNatRulePropertiesFormat != nil &&
					inboundNatRule.FrontendIPConfiguration != nil &&
					inboundNatRule.FrontendIPConfiguration.ID != nil &&
					strings.EqualFold(*inboundNatRule.FrontendIPConfiguration.ID, frontendIPConfigID) &&
					strings.EqualFold(string(inboundNatRule.Protocol), string(port.Protocol)) &&
					inboundNatRule.FrontendPort != nil &&
					*inboundNatRule.FrontendPort == port.Port {
					return fmt.Errorf("checkLoadBalancerResourcesConflicted: service port %s is trying to "+
						"consume the port %d which is being referenced by an existing inbound NAT rule %s with "+
						"the same protocol %s and frontend IP config with ID %s",
						port.Name,
						*inboundNatRule.FrontendPort,
						*inboundNatRule.Name,
						inboundNatRule.Protocol,
						*inboundNatRule.FrontendIPConfiguration.ID)
				}
			}
		}

		if lb.InboundNatPools != nil {
			for _, pool := range *lb.InboundNatPools {
				if pool.InboundNatPoolPropertiesFormat != nil &&
					pool.FrontendIPConfiguration != nil &&
					pool.FrontendIPConfiguration.ID != nil &&
					strings.EqualFold(*pool.FrontendIPConfiguration.ID, frontendIPConfigID) &&
					strings.EqualFold(string(pool.Protocol), string(port.Protocol)) &&
					pool.FrontendPortRangeStart != nil &&
					pool.FrontendPortRangeEnd != nil &&
					*pool.FrontendPortRangeStart <= port.Port &&
					*pool.FrontendPortRangeEnd >= port.Port {
					return fmt.Errorf("checkLoadBalancerResourcesConflicted: service port %s is trying to "+
						"consume the port %d which is being in the range (%d-%d) of an existing "+
						"inbound NAT pool %s with the same protocol %s and frontend IP config with ID %s",
						port.Name,
						port.Port,
						*pool.FrontendPortRangeStart,
						*pool.FrontendPortRangeEnd,
						*pool.Name,
						pool.Protocol,
						*pool.FrontendIPConfiguration.ID)
				}
			}
		}
	}

	return nil
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
		if _, ok := service.Annotations[ServiceAnnotationLoadBalancerDisableTCPReset]; ok {
			klog.Warning("annotation service.beta.kubernetes.io/azure-load-balancer-disable-tcp-reset has been removed as of Kubernetes 1.20. TCP Resets are always enabled on Standard SKU load balancers.")
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
			lbRuleName := az.getLoadBalancerRuleName(service, protocol, port.Port)
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

			loadDistribution := network.LoadDistributionDefault
			if service.Spec.SessionAffinity == v1.ServiceAffinityClientIP {
				loadDistribution = network.LoadDistributionSourceIP
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
					DisableOutboundSnat: to.BoolPtr(az.disableLoadBalancerOutboundSNAT()),
					EnableTCPReset:      enableTCPReset,
					EnableFloatingIP:    to.BoolPtr(true),
				},
			}

			if protocol == v1.ProtocolTCP {
				expectedRule.LoadBalancingRulePropertiesFormat.IdleTimeoutInMinutes = lbIdleTimeout
			}

			// we didn't construct the probe objects for UDP or SCTP because they're not allowed on Azure.
			// However, when externalTrafficPolicy is Local, Kubernetes HTTP health check would be used for probing.
			if servicehelpers.NeedsHealthCheck(service) || (protocol != v1.ProtocolUDP && protocol != v1.ProtocolSCTP) {
				expectedRule.Probe = &network.SubResource{
					ID: to.StringPtr(az.getLoadBalancerProbeID(lbName, az.getLoadBalancerResourceGroup(), lbRuleName)),
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
			return nil, fmt.Errorf("no port info for reconciling shared rule for service %s", service.Name)
		}
		ports = []v1.ServicePort{}
	}

	sg, err := az.getSecurityGroup(azcache.CacheReadTypeDefault)
	if err != nil {
		return nil, err
	}

	destinationIPAddress := ""
	if wantLb && lbIP == nil {
		return nil, fmt.Errorf("no load balancer IP for setting up security rules for service %s", service.Name)
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
	if len(serviceTags) != 0 {
		if _, ok := sourceRanges[defaultLoadBalancerSourceRanges]; ok {
			delete(sourceRanges, defaultLoadBalancerSourceRanges)
		}
	}

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
					return nil, fmt.Errorf("expected to find shared rule %s for service %s being deleted, but did not", sharedRuleName, service.Name)
				}
				if sharedRule.DestinationAddressPrefixes == nil {
					klog.V(4).Infof("Expected to have array of destinations in shared rule for service %s being deleted, but did not", service.Name)
					return nil, fmt.Errorf("expected to have array of destinations in shared rule for service %s being deleted, but did not", service.Name)
				}
				existingPrefixes := *sharedRule.DestinationAddressPrefixes
				addressIndex, found := findIndex(existingPrefixes, destinationIPAddress)
				if !found {
					klog.V(4).Infof("Expected to find destination address %s in shared rule %s for service %s being deleted, but did not", destinationIPAddress, sharedRuleName, service.Name)
					return nil, fmt.Errorf("expected to find destination address %s in shared rule %s for service %s being deleted, but did not", destinationIPAddress, sharedRuleName, service.Name)
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

func (az *Cloud) shouldUpdateLoadBalancer(clusterName string, service *v1.Service) bool {
	_, _, existsLb, _ := az.getServiceLoadBalancer(service, clusterName, nil, false)
	return existsLb && service.ObjectMeta.DeletionTimestamp == nil
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

// Determine if we should release existing owned public IPs
func shouldReleaseExistingOwnedPublicIP(existingPip *network.PublicIPAddress, lbShouldExist, lbIsInternal bool, desiredPipName, svcName string, ipTagRequest serviceIPTagRequest) bool {
	// Latch some variables for readability purposes.
	pipName := *(*existingPip).Name

	// Assume the current IP Tags are empty by default unless properties specify otherwise.
	currentIPTags := &[]network.IPTag{}
	pipPropertiesFormat := (*existingPip).PublicIPAddressPropertiesFormat
	if pipPropertiesFormat != nil {
		currentIPTags = (*pipPropertiesFormat).IPTags
	}

	// Check whether the public IP is being referenced by other service.
	// The owned public IP can be released only when there is not other service using it.
	if existingPip.Tags[serviceTagKey] != nil {
		// case 1: there is at least one reference when deleting the PIP
		if !lbShouldExist && len(parsePIPServiceTag(existingPip.Tags[serviceTagKey])) > 0 {
			return false
		}

		// case 2: there is at least one reference from other service
		if lbShouldExist && len(parsePIPServiceTag(existingPip.Tags[serviceTagKey])) > 1 {
			return false
		}
	}

	// Release the ip under the following criteria -
	// #1 - If we don't actually want a load balancer,
	return !lbShouldExist ||
		// #2 - If the load balancer is internal, and thus doesn't require public exposure
		lbIsInternal ||
		// #3 - If the name of this public ip does not match the desired name,
		(pipName != desiredPipName) ||
		// #4 If the service annotations have specified the ip tags that the public ip must have, but they do not match the ip tags of the existing instance
		(ipTagRequest.IPTagsRequestedByAnnotation && !areIPTagsEquivalent(currentIPTags, ipTagRequest.IPTags))
}

// This reconciles the PublicIP resources similar to how the LB is reconciled.
func (az *Cloud) reconcilePublicIP(clusterName string, service *v1.Service, lbName string, wantLb bool) (*network.PublicIPAddress, error) {
	isInternal := requiresInternalLoadBalancer(service)
	serviceName := getServiceName(service)
	serviceIPTagRequest := getServiceIPTagRequestForPublicIP(service)

	var (
		lb               *network.LoadBalancer
		desiredPipName   string
		err              error
		shouldPIPExisted bool
	)

	if !isInternal && wantLb {
		desiredPipName, shouldPIPExisted, err = az.determinePublicIPName(clusterName, service)
		if err != nil {
			return nil, err
		}
	}

	if lbName != "" {
		loadBalancer, _, err := az.getAzureLoadBalancer(lbName, azcache.CacheReadTypeDefault)
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

	var (
		serviceAnnotationRequestsNamedPublicIP = shouldPIPExisted
		discoveredDesiredPublicIP              bool
		deletedDesiredPublicIP                 bool
		pipsToBeDeleted                        []*network.PublicIPAddress
		pipsToBeUpdated                        []*network.PublicIPAddress
	)

	for i := range pips {
		pip := pips[i]
		pipName := *pip.Name

		// If we've been told to use a specific public ip by the client, let's track whether or not it actually existed
		// when we inspect the set in Azure.
		discoveredDesiredPublicIP = discoveredDesiredPublicIP || wantLb && !isInternal && pipName == desiredPipName

		// Now, let's perform additional analysis to determine if we should release the public ips we have found.
		// We can only let them go if (a) they are owned by this service and (b) they meet the criteria for deletion.
		if serviceOwnsPublicIP(&pip, clusterName, serviceName) {
			var dirtyPIP bool
			if !wantLb {
				klog.V(2).Infof("reconcilePublicIP for service(%s): unbinding the service from pip %s", serviceName, *pip.Name)
				err = unbindServiceFromPIP(&pip, serviceName)
				if err != nil {
					return nil, err
				}
				dirtyPIP = true
			}
			if shouldReleaseExistingOwnedPublicIP(&pip, wantLb, isInternal, desiredPipName, serviceName, serviceIPTagRequest) {
				// Then, release the public ip
				pipsToBeDeleted = append(pipsToBeDeleted, &pip)

				// Flag if we deleted the desired public ip
				deletedDesiredPublicIP = deletedDesiredPublicIP || pipName == desiredPipName

				// An aside: It would be unusual, but possible, for us to delete a public ip referred to explicitly by name
				// in Service annotations (which is usually reserved for non-service-owned externals), if that IP is tagged as
				// having been owned by a particular Kubernetes cluster.
			}

			// Update tags of PIP only instead of deleting it.
			if dirtyPIP {
				pipsToBeUpdated = append(pipsToBeUpdated, &pip)
			}
		}
	}

	if !isInternal && serviceAnnotationRequestsNamedPublicIP && !discoveredDesiredPublicIP && wantLb {
		return nil, fmt.Errorf("reconcilePublicIP for service(%s): pip(%s) not found", serviceName, desiredPipName)
	}

	var deleteFuncs, updateFuncs []func() error
	for _, pip := range pipsToBeUpdated {
		pipCopy := *pip
		updateFuncs = append(updateFuncs, func() error {
			klog.V(2).Infof("reconcilePublicIP for service(%s): pip(%s) - updating", serviceName, *pip.Name)
			return az.CreateOrUpdatePIP(service, pipResourceGroup, pipCopy)
		})
	}
	errs := utilerrors.AggregateGoroutines(updateFuncs...)
	if errs != nil {
		return nil, utilerrors.Flatten(errs)
	}

	for _, pip := range pipsToBeDeleted {
		pipCopy := *pip
		deleteFuncs = append(deleteFuncs, func() error {
			klog.V(2).Infof("reconcilePublicIP for service(%s): pip(%s) - deleting", serviceName, *pip.Name)
			return az.safeDeletePublicIP(service, pipResourceGroup, &pipCopy, lb)
		})
	}
	errs = utilerrors.AggregateGoroutines(deleteFuncs...)
	if errs != nil {
		return nil, utilerrors.Flatten(errs)
	}

	if !isInternal && wantLb {
		// Confirm desired public ip resource exists
		var pip *network.PublicIPAddress
		domainNameLabel, found := getPublicIPDomainNameLabel(service)
		errorIfPublicIPDoesNotExist := serviceAnnotationRequestsNamedPublicIP && discoveredDesiredPublicIP && !deletedDesiredPublicIP
		if pip, err = az.ensurePublicIPExists(service, desiredPipName, domainNameLabel, clusterName, errorIfPublicIPDoesNotExist, found); err != nil {
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
		return err
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
		reflect.DeepEqual(to.Bool(s.EnableTCPReset), to.Bool(t.EnableTCPReset)) &&
		reflect.DeepEqual(to.Bool(s.DisableOutboundSnat), to.Bool(t.DisableOutboundSnat))

	if wantLB && s.IdleTimeoutInMinutes != nil && t.IdleTimeoutInMinutes != nil {
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

func (az *Cloud) isBackendPoolPreConfigured(service *v1.Service) bool {
	preConfigured := false
	isInternal := requiresInternalLoadBalancer(service)

	if az.PreConfiguredBackendPoolLoadBalancerTypes == PreConfiguredBackendPoolLoadBalancerTypesAll {
		preConfigured = true
	}
	if (az.PreConfiguredBackendPoolLoadBalancerTypes == PreConfiguredBackendPoolLoadBalancerTypesInternal) && isInternal {
		preConfigured = true
	}
	if (az.PreConfiguredBackendPoolLoadBalancerTypes == PreConfiguredBackendPoolLoadBalancerTypesExternal) && !isInternal {
		preConfigured = true
	}

	return preConfigured
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
// if anything else it returns the unique VM set names after trimming spaces.
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

		if serviceTag != nil && isSVCNameInPIPTag(*serviceTag, serviceName) {
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

func isSVCNameInPIPTag(tag, svcName string) bool {
	svcNames := parsePIPServiceTag(&tag)

	for _, name := range svcNames {
		if strings.EqualFold(name, svcName) {
			return true
		}
	}

	return false
}

func parsePIPServiceTag(serviceTag *string) []string {
	if serviceTag == nil {
		return []string{}
	}

	serviceNames := strings.FieldsFunc(*serviceTag, func(r rune) bool {
		return r == ','
	})
	for i, name := range serviceNames {
		serviceNames[i] = strings.TrimSpace(name)
	}

	return serviceNames
}

// bindServicesToPIP add the incoming service name to the PIP's tag
// parameters: public IP address to be updated and incoming service names
// return values:
// 1. a bool flag to indicate if there is a new service added
// 2. an error when the pip is nil
// example:
// "ns1/svc1" + ["ns1/svc1", "ns2/svc2"] = "ns1/svc1,ns2/svc2"
func bindServicesToPIP(pip *network.PublicIPAddress, incomingServiceNames []string, replace bool) (bool, error) {
	if pip == nil {
		return false, fmt.Errorf("nil public IP")
	}

	if pip.Tags == nil {
		pip.Tags = map[string]*string{serviceTagKey: to.StringPtr("")}
	}

	serviceTagValue := pip.Tags[serviceTagKey]
	serviceTagValueSet := make(map[string]struct{})
	existingServiceNames := parsePIPServiceTag(serviceTagValue)
	addedNew := false

	// replace is used when unbinding the service from PIP so addedNew remains false all the time
	if replace {
		serviceTagValue = to.StringPtr(strings.Join(incomingServiceNames, ","))
		pip.Tags[serviceTagKey] = serviceTagValue

		return false, nil
	}

	for _, name := range existingServiceNames {
		if _, ok := serviceTagValueSet[name]; !ok {
			serviceTagValueSet[name] = struct{}{}
		}
	}

	for _, serviceName := range incomingServiceNames {
		if serviceTagValue == nil || *serviceTagValue == "" {
			serviceTagValue = to.StringPtr(serviceName)
			addedNew = true
		} else {
			// detect duplicates
			if _, ok := serviceTagValueSet[serviceName]; !ok {
				*serviceTagValue += fmt.Sprintf(",%s", serviceName)
				addedNew = true
			} else {
				klog.V(10).Infof("service %s has been bound to the pip already", serviceName)
			}
		}
	}
	pip.Tags[serviceTagKey] = serviceTagValue

	return addedNew, nil
}

func unbindServiceFromPIP(pip *network.PublicIPAddress, serviceName string) error {
	if pip == nil || pip.Tags == nil {
		return fmt.Errorf("nil public IP or tags")
	}

	serviceTagValue := pip.Tags[serviceTagKey]
	existingServiceNames := parsePIPServiceTag(serviceTagValue)
	var found bool
	for i := len(existingServiceNames) - 1; i >= 0; i-- {
		if strings.EqualFold(existingServiceNames[i], serviceName) {
			existingServiceNames = append(existingServiceNames[:i], existingServiceNames[i+1:]...)
			found = true
		}
	}
	if !found {
		klog.Warningf("cannot find the service %s in the corresponding PIP", serviceName)
	}

	_, err := bindServicesToPIP(pip, existingServiceNames, true)
	if err != nil {
		return err
	}

	if existingServiceName, ok := pip.Tags[serviceUsingDNSKey]; ok {
		if strings.EqualFold(*existingServiceName, serviceName) {
			pip.Tags[serviceUsingDNSKey] = to.StringPtr("")
		}
	}

	return nil
}
