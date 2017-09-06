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
	"strconv"
	"strings"

	"k8s.io/api/core/v1"
	utilerrors "k8s.io/apimachinery/pkg/util/errors"
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

// GetLoadBalancer returns whether the specified load balancer exists, and
// if so, what its status is.
func (az *Cloud) GetLoadBalancer(clusterName string, service *v1.Service) (status *v1.LoadBalancerStatus, exists bool, err error) {
	isInternal := requiresInternalLoadBalancer(service)
	lbName := getLoadBalancerName(clusterName, isInternal)
	serviceName := getServiceName(service)

	lb, existsLb, err := az.getAzureLoadBalancer(lbName)
	if err != nil {
		return nil, false, err
	}
	if !existsLb {
		glog.V(5).Infof("get(%s): lb(%s) - doesn't exist", serviceName, lbName)
		return nil, false, nil
	}

	var lbIP *string

	if isInternal {
		lbFrontendIPConfigName := getFrontendIPConfigName(service, subnet(service))
		for _, ipConfiguration := range *lb.FrontendIPConfigurations {
			if lbFrontendIPConfigName == *ipConfiguration.Name {
				lbIP = ipConfiguration.PrivateIPAddress
				break
			}
		}
	} else {
		// TODO: Consider also read address from lb's FrontendIPConfigurations
		pipName, err := az.determinePublicIPName(clusterName, service)
		if err != nil {
			return nil, false, err
		}
		pip, existsPip, err := az.getPublicIPAddress(pipName)
		if err != nil {
			return nil, false, err
		}
		if existsPip {
			lbIP = pip.IPAddress
		}
	}

	if lbIP == nil {
		glog.V(5).Infof("get(%s): lb(%s) - IP doesn't exist", serviceName, lbName)
		return nil, false, nil
	}

	return &v1.LoadBalancerStatus{
		Ingress: []v1.LoadBalancerIngress{{IP: *lbIP}},
	}, true, nil
}

func (az *Cloud) determinePublicIPName(clusterName string, service *v1.Service) (string, error) {
	loadBalancerIP := service.Spec.LoadBalancerIP
	if len(loadBalancerIP) == 0 {
		return getPublicIPName(clusterName, service), nil
	}

	az.operationPollRateLimiter.Accept()
	glog.V(10).Infof("PublicIPAddressesClient.List(%v): start", az.ResourceGroup)
	list, err := az.PublicIPAddressesClient.List(az.ResourceGroup)
	glog.V(10).Infof("PublicIPAddressesClient.List(%v): end", az.ResourceGroup)
	if err != nil {
		return "", err
	}

	if list.Value != nil {
		for ix := range *list.Value {
			ip := &(*list.Value)[ix]
			if ip.PublicIPAddressPropertiesFormat.IPAddress != nil &&
				*ip.PublicIPAddressPropertiesFormat.IPAddress == loadBalancerIP {
				return *ip.Name, nil
			}
		}
	}
	// TODO: follow next link here? Will there really ever be that many public IPs?

	return "", fmt.Errorf("user supplied IP Address %s was not found", loadBalancerIP)
}

// EnsureLoadBalancer creates a new load balancer 'name', or updates the existing one. Returns the status of the balancer
func (az *Cloud) EnsureLoadBalancer(clusterName string, service *v1.Service, nodes []*v1.Node) (*v1.LoadBalancerStatus, error) {
	isInternal := requiresInternalLoadBalancer(service)
	lbName := getLoadBalancerName(clusterName, isInternal)

	// When a client updates the internal load balancer annotation,
	// the service may be switched from an internal LB to a public one, or vise versa.
	// Here we'll firstly ensure service do not lie in the opposite LB.
	err := az.cleanupLoadBalancer(clusterName, service, !isInternal)
	if err != nil {
		return nil, err
	}

	serviceName := getServiceName(service)
	glog.V(5).Infof("ensure(%s): START clusterName=%q lbName=%q", serviceName, clusterName, lbName)

	az.operationPollRateLimiter.Accept()
	glog.V(10).Infof("SecurityGroupsClient.Get(%q): start", az.SecurityGroupName)
	sg, err := az.SecurityGroupsClient.Get(az.ResourceGroup, az.SecurityGroupName, "")
	glog.V(10).Infof("SecurityGroupsClient.Get(%q): end", az.SecurityGroupName)
	if err != nil {
		return nil, err
	}
	sg, sgNeedsUpdate, err := az.reconcileSecurityGroup(sg, clusterName, service, true /* wantLb */)
	if err != nil {
		return nil, err
	}
	if sgNeedsUpdate {
		glog.V(3).Infof("ensure(%s): sg(%s) - updating", serviceName, *sg.Name)
		// azure-sdk-for-go introduced contraint validation which breaks the updating here if we don't set these
		// to nil. This is a workaround until https://github.com/Azure/go-autorest/issues/112 is fixed
		sg.SecurityGroupPropertiesFormat.NetworkInterfaces = nil
		sg.SecurityGroupPropertiesFormat.Subnets = nil
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

	lb, existsLb, err := az.getAzureLoadBalancer(lbName)
	if err != nil {
		return nil, err
	}
	if !existsLb {
		lb = network.LoadBalancer{
			Name:                         &lbName,
			Location:                     &az.Location,
			LoadBalancerPropertiesFormat: &network.LoadBalancerPropertiesFormat{},
		}
	}

	var lbIP *string
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
			Subnet: &network.Subnet{
				ID: subnet.ID,
			},
		}

		loadBalancerIP := service.Spec.LoadBalancerIP
		if loadBalancerIP != "" {
			configProperties.PrivateIPAllocationMethod = network.Static
			configProperties.PrivateIPAddress = &loadBalancerIP
			lbIP = &loadBalancerIP
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
		pip, err := az.ensurePublicIPExists(serviceName, pipName)
		if err != nil {
			return nil, err
		}

		lbIP = pip.IPAddress
		fipConfigurationProperties = &network.FrontendIPConfigurationPropertiesFormat{
			PublicIPAddress: &network.PublicIPAddress{ID: pip.ID},
		}
	}

	lb, lbNeedsUpdate, err := az.reconcileLoadBalancer(lb, fipConfigurationProperties, clusterName, service, nodes)
	if err != nil {
		return nil, err
	}
	if !existsLb || lbNeedsUpdate {
		glog.V(3).Infof("ensure(%s): lb(%s) - updating", serviceName, lbName)
		az.operationPollRateLimiter.Accept()
		glog.V(10).Infof("LoadBalancerClient.CreateOrUpdate(%q): start", *lb.Name)
		respChan, errChan := az.LoadBalancerClient.CreateOrUpdate(az.ResourceGroup, *lb.Name, lb, nil)
		resp := <-respChan
		err := <-errChan
		glog.V(10).Infof("LoadBalancerClient.CreateOrUpdate(%q): end", *lb.Name)
		if az.CloudProviderBackoff && shouldRetryAPIRequest(resp.Response, err) {
			glog.V(2).Infof("ensure(%s) backing off: lb(%s) - updating", serviceName, lbName)
			retryErr := az.CreateOrUpdateLBWithRetry(lb)
			if retryErr != nil {
				glog.V(2).Infof("ensure(%s) abort backoff: lb(%s) - updating", serviceName, lbName)
				return nil, retryErr
			}
		}
		if err != nil {
			return nil, err
		}
	}

	// Add the machines to the backend pool if they're not already
	lbBackendName := getBackendPoolName(clusterName)
	lbBackendPoolID := az.getBackendPoolID(lbName, lbBackendName)
	hostUpdates := make([]func() error, len(nodes))
	for i, node := range nodes {
		localNodeName := node.Name
		f := func() error {
			err := az.ensureHostInPool(serviceName, types.NodeName(localNodeName), lbBackendPoolID)
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

	glog.V(2).Infof("ensure(%s): lb(%s) finished", serviceName, lbName)

	if lbIP == nil {
		lbStatus, exists, err := az.GetLoadBalancer(clusterName, service)
		if err != nil {
			return nil, err
		}
		if !exists {
			return nil, fmt.Errorf("ensure(%s): lb(%s) - failed to get back load balancer", serviceName, lbName)
		}
		return lbStatus, nil
	}

	return &v1.LoadBalancerStatus{
		Ingress: []v1.LoadBalancerIngress{{IP: *lbIP}},
	}, nil
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
	isInternal := requiresInternalLoadBalancer(service)
	lbName := getLoadBalancerName(clusterName, isInternal)
	serviceName := getServiceName(service)

	glog.V(5).Infof("delete(%s): START clusterName=%q lbName=%q", serviceName, clusterName, lbName)

	err := az.cleanupLoadBalancer(clusterName, service, isInternal)
	if err != nil {
		return err
	}

	sg, existsSg, err := az.getSecurityGroup()
	if err != nil {
		return err
	}
	if existsSg {
		reconciledSg, sgNeedsUpdate, reconcileErr := az.reconcileSecurityGroup(sg, clusterName, service, false /* wantLb */)
		if reconcileErr != nil {
			return reconcileErr
		}
		if sgNeedsUpdate {
			glog.V(3).Infof("delete(%s): sg(%s) - updating", serviceName, az.SecurityGroupName)
			// azure-sdk-for-go introduced contraint validation which breaks the updating here if we don't set these
			// to nil. This is a workaround until https://github.com/Azure/go-autorest/issues/112 is fixed
			sg.SecurityGroupPropertiesFormat.NetworkInterfaces = nil
			sg.SecurityGroupPropertiesFormat.Subnets = nil
			az.operationPollRateLimiter.Accept()
			glog.V(10).Infof("SecurityGroupsClient.CreateOrUpdate(%q): start", *reconciledSg.Name)
			respChan, errChan := az.SecurityGroupsClient.CreateOrUpdate(az.ResourceGroup, *reconciledSg.Name, reconciledSg, nil)
			resp := <-respChan
			err := <-errChan
			glog.V(10).Infof("SecurityGroupsClient.CreateOrUpdate(%q): end", *reconciledSg.Name)
			if az.CloudProviderBackoff && shouldRetryAPIRequest(resp.Response, err) {
				glog.V(2).Infof("delete(%s) backing off: sg(%s) - updating", serviceName, az.SecurityGroupName)
				retryErr := az.CreateOrUpdateSGWithRetry(reconciledSg)
				if retryErr != nil {
					err = retryErr
					glog.V(2).Infof("delete(%s) abort backoff: sg(%s) - updating", serviceName, az.SecurityGroupName)
				}
			}
			if err != nil {
				return err
			}
		}
	}

	glog.V(2).Infof("delete(%s): FINISH", serviceName)
	return nil
}

func (az *Cloud) cleanupLoadBalancer(clusterName string, service *v1.Service, isInternalLb bool) error {
	lbName := getLoadBalancerName(clusterName, isInternalLb)
	serviceName := getServiceName(service)

	glog.V(10).Infof("ensure lb deleted: clusterName=%q, serviceName=%s, lbName=%q", clusterName, serviceName, lbName)

	lb, existsLb, err := az.getAzureLoadBalancer(lbName)
	if err != nil {
		return err
	}
	if existsLb {
		var publicIPToCleanup *string

		if !isInternalLb {
			// Find public ip resource to clean up from IP configuration
			lbFrontendIPConfigName := getFrontendIPConfigName(service, nil)
			for _, config := range *lb.FrontendIPConfigurations {
				if strings.EqualFold(*config.Name, lbFrontendIPConfigName) {
					if config.PublicIPAddress != nil {
						// Only ID property is available
						publicIPToCleanup = config.PublicIPAddress.ID
					}
					break
				}
			}
		}

		lb, lbNeedsUpdate, reconcileErr := az.reconcileLoadBalancer(lb, nil, clusterName, service, []*v1.Node{})
		if reconcileErr != nil {
			return reconcileErr
		}
		if lbNeedsUpdate {
			if len(*lb.FrontendIPConfigurations) > 0 {
				glog.V(3).Infof("delete(%s): lb(%s) - updating", serviceName, lbName)
				az.operationPollRateLimiter.Accept()
				glog.V(10).Infof("LoadBalancerClient.CreateOrUpdate(%q): start", *lb.Name)
				respChan, errChan := az.LoadBalancerClient.CreateOrUpdate(az.ResourceGroup, *lb.Name, lb, nil)
				resp := <-respChan
				err := <-errChan
				glog.V(10).Infof("LoadBalancerClient.CreateOrUpdate(%q): end", *lb.Name)
				if az.CloudProviderBackoff && shouldRetryAPIRequest(resp.Response, err) {
					glog.V(2).Infof("delete(%s) backing off: sg(%s) - updating", serviceName, az.SecurityGroupName)
					retryErr := az.CreateOrUpdateLBWithRetry(lb)
					if retryErr != nil {
						err = retryErr
						glog.V(2).Infof("delete(%s) abort backoff: sg(%s) - updating", serviceName, az.SecurityGroupName)
					}
				}
				if err != nil {
					return err
				}
			} else {
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
					return err
				}
			}
		}

		// Public IP can be deleted after frontend ip configuration rule deleted.
		if publicIPToCleanup != nil {
			// Only delete an IP address if we created it, deducing by name.
			if index := strings.LastIndex(*publicIPToCleanup, "/"); index != -1 {
				managedPipName := getPublicIPName(clusterName, service)
				pipName := (*publicIPToCleanup)[index+1:]
				if strings.EqualFold(managedPipName, pipName) {
					glog.V(5).Infof("Deleting public IP resource %q.", pipName)
					err = az.ensurePublicIPDeleted(serviceName, pipName)
					if err != nil {
						return err
					}
				} else {
					glog.V(5).Infof("Public IP resource %q found, but it does not match managed name %q, skip deleting.", pipName, managedPipName)
				}
			}
		}
	}

	return nil
}

func (az *Cloud) ensurePublicIPExists(serviceName, pipName string) (*network.PublicIPAddress, error) {
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

func (az *Cloud) ensurePublicIPDeleted(serviceName, pipName string) error {
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
			return retryErr
		}
	}
	_, realErr := checkResourceExistsFromError(deleteErr)
	if realErr != nil {
		return nil
	}
	return nil
}

// This ensures load balancer exists and the frontend ip config is setup.
// This also reconciles the Service's Ports  with the LoadBalancer config.
// This entails adding rules/probes for expected Ports and removing stale rules/ports.
func (az *Cloud) reconcileLoadBalancer(lb network.LoadBalancer, fipConfigurationProperties *network.FrontendIPConfigurationPropertiesFormat, clusterName string, service *v1.Service, nodes []*v1.Node) (network.LoadBalancer, bool, error) {
	isInternal := requiresInternalLoadBalancer(service)
	lbName := getLoadBalancerName(clusterName, isInternal)
	serviceName := getServiceName(service)
	lbFrontendIPConfigName := getFrontendIPConfigName(service, subnet(service))
	lbFrontendIPConfigID := az.getFrontendIPConfigID(lbName, lbFrontendIPConfigName)
	lbBackendPoolName := getBackendPoolName(clusterName)
	lbBackendPoolID := az.getBackendPoolID(lbName, lbBackendPoolName)

	wantLb := fipConfigurationProperties != nil
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
			return lb, false, err
		}

		if serviceapi.NeedsHealthCheck(service) {
			if port.Protocol == v1.ProtocolUDP {
				// ERROR: this isn't supported
				// health check (aka source ip preservation) is not
				// compatible with UDP (it uses an HTTP check)
				return lb, false, fmt.Errorf("services requiring health checks are incompatible with UDP ports")
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

	return lb, dirtyLb, nil
}

// This reconciles the Network Security Group similar to how the LB is reconciled.
// This entails adding required, missing SecurityRules and removing stale rules.
func (az *Cloud) reconcileSecurityGroup(sg network.SecurityGroup, clusterName string, service *v1.Service, wantLb bool) (network.SecurityGroup, bool, error) {
	serviceName := getServiceName(service)
	var ports []v1.ServicePort
	if wantLb {
		ports = service.Spec.Ports
	} else {
		ports = []v1.ServicePort{}
	}

	sourceRanges, err := serviceapi.GetLoadBalancerSourceRanges(service)
	if err != nil {
		return sg, false, err
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
			return sg, false, err
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
					DestinationAddressPrefix: to.StringPtr("*"),
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
				return sg, false, err
			}

			expectedRule.Priority = to.Int32Ptr(nextAvailablePriority)
			updatedRules = append(updatedRules, expectedRule)
			dirtySg = true
		}
	}
	if dirtySg {
		sg.SecurityRules = &updatedRules
	}
	return sg, dirtySg, nil
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
func (az *Cloud) ensureHostInPool(serviceName string, nodeName types.NodeName, backendPoolID string) error {
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
	if az.PrimaryAvailabilitySetName != "" {
		expectedAvailabilitySetName := az.getAvailabilitySetID(az.PrimaryAvailabilitySetName)
		if !strings.EqualFold(*machine.AvailabilitySet.ID, expectedAvailabilitySetName) {
			glog.V(3).Infof(
				"nicupdate(%s): skipping nic (%s) since it is not in the primaryAvailabilitSet(%s)",
				serviceName, nicName, az.PrimaryAvailabilitySetName)
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
