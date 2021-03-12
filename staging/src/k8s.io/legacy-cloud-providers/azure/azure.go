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
	"fmt"
	"io"
	"io/ioutil"
	"net/http"
	"strings"
	"sync"
	"time"

	"github.com/Azure/go-autorest/autorest"
	"github.com/Azure/go-autorest/autorest/adal"
	"github.com/Azure/go-autorest/autorest/azure"
	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/wait"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/client-go/informers"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/client-go/kubernetes/scheme"
	v1core "k8s.io/client-go/kubernetes/typed/core/v1"
	"k8s.io/client-go/tools/cache"
	"k8s.io/client-go/tools/record"
	cloudprovider "k8s.io/cloud-provider"
	"k8s.io/klog/v2"
	"k8s.io/legacy-cloud-providers/azure/auth"
	azcache "k8s.io/legacy-cloud-providers/azure/cache"
	azclients "k8s.io/legacy-cloud-providers/azure/clients"
	"k8s.io/legacy-cloud-providers/azure/clients/diskclient"
	"k8s.io/legacy-cloud-providers/azure/clients/fileclient"
	"k8s.io/legacy-cloud-providers/azure/clients/interfaceclient"
	"k8s.io/legacy-cloud-providers/azure/clients/loadbalancerclient"
	"k8s.io/legacy-cloud-providers/azure/clients/publicipclient"
	"k8s.io/legacy-cloud-providers/azure/clients/routeclient"
	"k8s.io/legacy-cloud-providers/azure/clients/routetableclient"
	"k8s.io/legacy-cloud-providers/azure/clients/securitygroupclient"
	"k8s.io/legacy-cloud-providers/azure/clients/snapshotclient"
	"k8s.io/legacy-cloud-providers/azure/clients/storageaccountclient"
	"k8s.io/legacy-cloud-providers/azure/clients/subnetclient"
	"k8s.io/legacy-cloud-providers/azure/clients/vmclient"
	"k8s.io/legacy-cloud-providers/azure/clients/vmsizeclient"
	"k8s.io/legacy-cloud-providers/azure/clients/vmssclient"
	"k8s.io/legacy-cloud-providers/azure/clients/vmssvmclient"
	"k8s.io/legacy-cloud-providers/azure/retry"

	// ensure the newly added package from azure-sdk-for-go is in vendor/
	_ "k8s.io/legacy-cloud-providers/azure/clients/containerserviceclient"
	// ensure the newly added package from azure-sdk-for-go is in vendor/
	_ "k8s.io/legacy-cloud-providers/azure/clients/deploymentclient"

	"sigs.k8s.io/yaml"
)

const (
	// CloudProviderName is the value used for the --cloud-provider flag
	CloudProviderName = "azure"
	// AzureStackCloudName is the cloud name of Azure Stack
	AzureStackCloudName    = "AZURESTACKCLOUD"
	rateLimitQPSDefault    = 1.0
	rateLimitBucketDefault = 5
	backoffRetriesDefault  = 6
	backoffExponentDefault = 1.5
	backoffDurationDefault = 5 // in seconds
	backoffJitterDefault   = 1.0
	// According to https://docs.microsoft.com/en-us/azure/azure-subscription-service-limits#load-balancer.
	maximumLoadBalancerRuleCount = 250

	vmTypeVMSS     = "vmss"
	vmTypeStandard = "standard"

	loadBalancerSkuBasic    = "basic"
	loadBalancerSkuStandard = "standard"

	externalResourceGroupLabel = "kubernetes.azure.com/resource-group"
	managedByAzureLabel        = "kubernetes.azure.com/managed"

	// LabelFailureDomainBetaZone refer to https://github.com/kubernetes/api/blob/8519c5ea46199d57724725d5b969c5e8e0533692/core/v1/well_known_labels.go#L22-L23
	LabelFailureDomainBetaZone = "failure-domain.beta.kubernetes.io/zone"

	// LabelFailureDomainBetaRegion failure-domain region label
	LabelFailureDomainBetaRegion = "failure-domain.beta.kubernetes.io/region"
)

const (
	// PreConfiguredBackendPoolLoadBalancerTypesNone means that the load balancers are not pre-configured
	PreConfiguredBackendPoolLoadBalancerTypesNone = ""
	// PreConfiguredBackendPoolLoadBalancerTypesInternal means that the `internal` load balancers are pre-configured
	PreConfiguredBackendPoolLoadBalancerTypesInternal = "internal"
	// PreConfiguredBackendPoolLoadBalancerTypesExternal means that the `external` load balancers are pre-configured
	PreConfiguredBackendPoolLoadBalancerTypesExternal = "external"
	// PreConfiguredBackendPoolLoadBalancerTypesAll means that all load balancers are pre-configured
	PreConfiguredBackendPoolLoadBalancerTypesAll = "all"
)

var (
	// Master nodes are not added to standard load balancer by default.
	defaultExcludeMasterFromStandardLB = true
	// Outbound SNAT is enabled by default.
	defaultDisableOutboundSNAT = false
)

// Config holds the configuration parsed from the --cloud-config flag
// All fields are required unless otherwise specified
// NOTE: Cloud config files should follow the same Kubernetes deprecation policy as
// flags or CLIs. Config fields should not change behavior in incompatible ways and
// should be deprecated for at least 2 release prior to removing.
// See https://kubernetes.io/docs/reference/using-api/deprecation-policy/#deprecating-a-flag-or-cli
// for more details.
type Config struct {
	auth.AzureAuthConfig
	CloudProviderRateLimitConfig

	// The name of the resource group that the cluster is deployed in
	ResourceGroup string `json:"resourceGroup,omitempty" yaml:"resourceGroup,omitempty"`
	// The location of the resource group that the cluster is deployed in
	Location string `json:"location,omitempty" yaml:"location,omitempty"`
	// The name of the VNet that the cluster is deployed in
	VnetName string `json:"vnetName,omitempty" yaml:"vnetName,omitempty"`
	// The name of the resource group that the Vnet is deployed in
	VnetResourceGroup string `json:"vnetResourceGroup,omitempty" yaml:"vnetResourceGroup,omitempty"`
	// The name of the subnet that the cluster is deployed in
	SubnetName string `json:"subnetName,omitempty" yaml:"subnetName,omitempty"`
	// The name of the security group attached to the cluster's subnet
	SecurityGroupName string `json:"securityGroupName,omitempty" yaml:"securityGroupName,omitempty"`
	// The name of the resource group that the security group is deployed in
	SecurityGroupResourceGroup string `json:"securityGroupResourceGroup,omitempty" yaml:"securityGroupResourceGroup,omitempty"`
	// (Optional in 1.6) The name of the route table attached to the subnet that the cluster is deployed in
	RouteTableName string `json:"routeTableName,omitempty" yaml:"routeTableName,omitempty"`
	// The name of the resource group that the RouteTable is deployed in
	RouteTableResourceGroup string `json:"routeTableResourceGroup,omitempty" yaml:"routeTableResourceGroup,omitempty"`
	// (Optional) The name of the availability set that should be used as the load balancer backend
	// If this is set, the Azure cloudprovider will only add nodes from that availability set to the load
	// balancer backend pool. If this is not set, and multiple agent pools (availability sets) are used, then
	// the cloudprovider will try to add all nodes to a single backend pool which is forbidden.
	// In other words, if you use multiple agent pools (availability sets), you MUST set this field.
	PrimaryAvailabilitySetName string `json:"primaryAvailabilitySetName,omitempty" yaml:"primaryAvailabilitySetName,omitempty"`
	// The type of azure nodes. Candidate values are: vmss and standard.
	// If not set, it will be default to standard.
	VMType string `json:"vmType,omitempty" yaml:"vmType,omitempty"`
	// The name of the scale set that should be used as the load balancer backend.
	// If this is set, the Azure cloudprovider will only add nodes from that scale set to the load
	// balancer backend pool. If this is not set, and multiple agent pools (scale sets) are used, then
	// the cloudprovider will try to add all nodes to a single backend pool which is forbidden.
	// In other words, if you use multiple agent pools (scale sets), you MUST set this field.
	PrimaryScaleSetName string `json:"primaryScaleSetName,omitempty" yaml:"primaryScaleSetName,omitempty"`
	// Enable exponential backoff to manage resource request retries
	CloudProviderBackoff bool `json:"cloudProviderBackoff,omitempty" yaml:"cloudProviderBackoff,omitempty"`
	// Backoff retry limit
	CloudProviderBackoffRetries int `json:"cloudProviderBackoffRetries,omitempty" yaml:"cloudProviderBackoffRetries,omitempty"`
	// Backoff exponent
	CloudProviderBackoffExponent float64 `json:"cloudProviderBackoffExponent,omitempty" yaml:"cloudProviderBackoffExponent,omitempty"`
	// Backoff duration
	CloudProviderBackoffDuration int `json:"cloudProviderBackoffDuration,omitempty" yaml:"cloudProviderBackoffDuration,omitempty"`
	// Backoff jitter
	CloudProviderBackoffJitter float64 `json:"cloudProviderBackoffJitter,omitempty" yaml:"cloudProviderBackoffJitter,omitempty"`
	// Use instance metadata service where possible
	UseInstanceMetadata bool `json:"useInstanceMetadata,omitempty" yaml:"useInstanceMetadata,omitempty"`

	// Sku of Load Balancer and Public IP. Candidate values are: basic and standard.
	// If not set, it will be default to basic.
	LoadBalancerSku string `json:"loadBalancerSku,omitempty" yaml:"loadBalancerSku,omitempty"`
	// ExcludeMasterFromStandardLB excludes master nodes from standard load balancer.
	// If not set, it will be default to true.
	ExcludeMasterFromStandardLB *bool `json:"excludeMasterFromStandardLB,omitempty" yaml:"excludeMasterFromStandardLB,omitempty"`
	// DisableOutboundSNAT disables the outbound SNAT for public load balancer rules.
	// It should only be set when loadBalancerSku is standard. If not set, it will be default to false.
	DisableOutboundSNAT *bool `json:"disableOutboundSNAT,omitempty" yaml:"disableOutboundSNAT,omitempty"`

	// Maximum allowed LoadBalancer Rule Count is the limit enforced by Azure Load balancer
	MaximumLoadBalancerRuleCount int `json:"maximumLoadBalancerRuleCount,omitempty" yaml:"maximumLoadBalancerRuleCount,omitempty"`

	// The cloud configure type for Azure cloud provider. Supported values are file, secret and merge.
	CloudConfigType cloudConfigType `json:"cloudConfigType,omitempty" yaml:"cloudConfigType,omitempty"`

	// LoadBalancerName determines the specific name of the load balancer user want to use, working with
	// LoadBalancerResourceGroup
	LoadBalancerName string `json:"loadBalancerName,omitempty" yaml:"loadBalancerName,omitempty"`
	// LoadBalancerResourceGroup determines the specific resource group of the load balancer user want to use, working
	// with LoadBalancerName
	LoadBalancerResourceGroup string `json:"loadBalancerResourceGroup,omitempty" yaml:"loadBalancerResourceGroup,omitempty"`
	// PreConfiguredBackendPoolLoadBalancerTypes determines whether the LoadBalancer BackendPool has been preconfigured.
	// Candidate values are:
	//   "": exactly with today (not pre-configured for any LBs)
	//   "internal": for internal LoadBalancer
	//   "external": for external LoadBalancer
	//   "all": for both internal and external LoadBalancer
	PreConfiguredBackendPoolLoadBalancerTypes string `json:"preConfiguredBackendPoolLoadBalancerTypes,omitempty" yaml:"preConfiguredBackendPoolLoadBalancerTypes,omitempty"`
	// EnableMultipleStandardLoadBalancers determines the behavior of the standard load balancer. If set to true
	// there would be one standard load balancer per VMAS or VMSS, which is similar with the behavior of the basic
	// load balancer. Users could select the specific standard load balancer for their service by the service
	// annotation `service.beta.kubernetes.io/azure-load-balancer-mode`, If set to false, the same standard load balancer
	// would be shared by all services in the cluster. In this case, the mode selection annotation would be ignored.
	EnableMultipleStandardLoadBalancers bool `json:"enableMultipleStandardLoadBalancers,omitempty" yaml:"enableMultipleStandardLoadBalancers,omitempty"`

	// AvailabilitySetNodesCacheTTLInSeconds sets the Cache TTL for availabilitySetNodesCache
	// if not set, will use default value
	AvailabilitySetNodesCacheTTLInSeconds int `json:"availabilitySetNodesCacheTTLInSeconds,omitempty" yaml:"availabilitySetNodesCacheTTLInSeconds,omitempty"`
	// VmssCacheTTLInSeconds sets the cache TTL for VMSS
	VmssCacheTTLInSeconds int `json:"vmssCacheTTLInSeconds,omitempty" yaml:"vmssCacheTTLInSeconds,omitempty"`
	// VmssVirtualMachinesCacheTTLInSeconds sets the cache TTL for vmssVirtualMachines
	VmssVirtualMachinesCacheTTLInSeconds int `json:"vmssVirtualMachinesCacheTTLInSeconds,omitempty" yaml:"vmssVirtualMachinesCacheTTLInSeconds,omitempty"`
	// VmCacheTTLInSeconds sets the cache TTL for vm
	VMCacheTTLInSeconds int `json:"vmCacheTTLInSeconds,omitempty" yaml:"vmCacheTTLInSeconds,omitempty"`
	// LoadBalancerCacheTTLInSeconds sets the cache TTL for load balancer
	LoadBalancerCacheTTLInSeconds int `json:"loadBalancerCacheTTLInSeconds,omitempty" yaml:"loadBalancerCacheTTLInSeconds,omitempty"`
	// NsgCacheTTLInSeconds sets the cache TTL for network security group
	NsgCacheTTLInSeconds int `json:"nsgCacheTTLInSeconds,omitempty" yaml:"nsgCacheTTLInSeconds,omitempty"`
	// RouteTableCacheTTLInSeconds sets the cache TTL for route table
	RouteTableCacheTTLInSeconds int `json:"routeTableCacheTTLInSeconds,omitempty" yaml:"routeTableCacheTTLInSeconds,omitempty"`

	// DisableAvailabilitySetNodes disables VMAS nodes support when "VMType" is set to "vmss".
	DisableAvailabilitySetNodes bool `json:"disableAvailabilitySetNodes,omitempty" yaml:"disableAvailabilitySetNodes,omitempty"`

	// Tags determines what tags shall be applied to the shared resources managed by controller manager, which
	// includes load balancer, security group and route table. The supported format is `a=b,c=d,...`. After updated
	// this config, the old tags would be replaced by the new ones.
	Tags string `json:"tags,omitempty" yaml:"tags,omitempty"`
}

var (
	_ cloudprovider.Interface    = (*Cloud)(nil)
	_ cloudprovider.Instances    = (*Cloud)(nil)
	_ cloudprovider.LoadBalancer = (*Cloud)(nil)
	_ cloudprovider.Routes       = (*Cloud)(nil)
	_ cloudprovider.Zones        = (*Cloud)(nil)
	_ cloudprovider.PVLabeler    = (*Cloud)(nil)
)

// Cloud holds the config and clients
type Cloud struct {
	Config
	Environment azure.Environment

	RoutesClient                    routeclient.Interface
	SubnetsClient                   subnetclient.Interface
	InterfacesClient                interfaceclient.Interface
	RouteTablesClient               routetableclient.Interface
	LoadBalancerClient              loadbalancerclient.Interface
	PublicIPAddressesClient         publicipclient.Interface
	SecurityGroupsClient            securitygroupclient.Interface
	VirtualMachinesClient           vmclient.Interface
	StorageAccountClient            storageaccountclient.Interface
	DisksClient                     diskclient.Interface
	SnapshotsClient                 snapshotclient.Interface
	FileClient                      fileclient.Interface
	VirtualMachineScaleSetsClient   vmssclient.Interface
	VirtualMachineScaleSetVMsClient vmssvmclient.Interface
	VirtualMachineSizesClient       vmsizeclient.Interface

	ResourceRequestBackoff wait.Backoff
	metadata               *InstanceMetadataService
	VMSet                  VMSet

	// ipv6DualStack allows overriding for unit testing.  It's normally initialized from featuregates
	ipv6DualStackEnabled bool
	// Lock for access to node caches, includes nodeZones, nodeResourceGroups, and unmanagedNodes.
	nodeCachesLock sync.RWMutex
	// nodeZones is a mapping from Zone to a sets.String of Node's names in the Zone
	// it is updated by the nodeInformer
	nodeZones map[string]sets.String
	// nodeResourceGroups holds nodes external resource groups
	nodeResourceGroups map[string]string
	// unmanagedNodes holds a list of nodes not managed by Azure cloud provider.
	unmanagedNodes sets.String
	// nodeInformerSynced is for determining if the informer has synced.
	nodeInformerSynced cache.InformerSynced

	// routeCIDRsLock holds lock for routeCIDRs cache.
	routeCIDRsLock sync.Mutex
	// routeCIDRs holds cache for route CIDRs.
	routeCIDRs map[string]string

	KubeClient       clientset.Interface
	eventBroadcaster record.EventBroadcaster
	eventRecorder    record.EventRecorder
	routeUpdater     *delayedRouteUpdater

	vmCache  *azcache.TimedCache
	lbCache  *azcache.TimedCache
	nsgCache *azcache.TimedCache
	rtCache  *azcache.TimedCache

	*BlobDiskController
	*ManagedDiskController
	*controllerCommon
}

func init() {
	// In go-autorest SDK https://github.com/Azure/go-autorest/blob/master/autorest/sender.go#L258-L287,
	// if ARM returns http.StatusTooManyRequests, the sender doesn't increase the retry attempt count,
	// hence the Azure clients will keep retrying forever until it get a status code other than 429.
	// So we explicitly removes http.StatusTooManyRequests from autorest.StatusCodesForRetry.
	// Refer https://github.com/Azure/go-autorest/issues/398.
	// TODO(feiskyer): Use autorest.SendDecorator to customize the retry policy when new Azure SDK is available.
	statusCodesForRetry := make([]int, 0)
	for _, code := range autorest.StatusCodesForRetry {
		if code != http.StatusTooManyRequests {
			statusCodesForRetry = append(statusCodesForRetry, code)
		}
	}
	autorest.StatusCodesForRetry = statusCodesForRetry

	cloudprovider.RegisterCloudProvider(CloudProviderName, NewCloud)
}

// NewCloud returns a Cloud with initialized clients
func NewCloud(configReader io.Reader) (cloudprovider.Interface, error) {
	az, err := NewCloudWithoutFeatureGates(configReader)
	if err != nil {
		return nil, err
	}
	az.ipv6DualStackEnabled = utilfeature.DefaultFeatureGate.Enabled(IPv6DualStack)

	return az, nil
}

// NewCloudWithoutFeatureGates returns a Cloud without trying to wire the feature gates.  This is used by the unit tests
// that don't load the actual features being used in the cluster.
func NewCloudWithoutFeatureGates(configReader io.Reader) (*Cloud, error) {
	config, err := parseConfig(configReader)
	if err != nil {
		return nil, err
	}

	az := &Cloud{
		nodeZones:          map[string]sets.String{},
		nodeResourceGroups: map[string]string{},
		unmanagedNodes:     sets.NewString(),
		routeCIDRs:         map[string]string{},
	}

	err = az.InitializeCloudFromConfig(config, false)
	if err != nil {
		return nil, err
	}

	return az, nil
}

// InitializeCloudFromConfig initializes the Cloud from config.
func (az *Cloud) InitializeCloudFromConfig(config *Config, fromSecret bool) error {
	// cloud-config not set, return nil so that it would be initialized from secret.
	if config == nil {
		klog.Warning("cloud-config is not provided, Azure cloud provider would be initialized from secret")
		return nil
	}

	if config.RouteTableResourceGroup == "" {
		config.RouteTableResourceGroup = config.ResourceGroup
	}

	if config.SecurityGroupResourceGroup == "" {
		config.SecurityGroupResourceGroup = config.ResourceGroup
	}

	if config.VMType == "" {
		// default to standard vmType if not set.
		config.VMType = vmTypeStandard
	}

	if config.DisableAvailabilitySetNodes && config.VMType != vmTypeVMSS {
		return fmt.Errorf("disableAvailabilitySetNodes %v is only supported when vmType is 'vmss'", config.DisableAvailabilitySetNodes)
	}

	if config.CloudConfigType == "" {
		// The default cloud config type is cloudConfigTypeMerge.
		config.CloudConfigType = cloudConfigTypeMerge
	} else {
		supportedCloudConfigTypes := sets.NewString(
			string(cloudConfigTypeMerge),
			string(cloudConfigTypeFile),
			string(cloudConfigTypeSecret))
		if !supportedCloudConfigTypes.Has(string(config.CloudConfigType)) {
			return fmt.Errorf("cloudConfigType %v is not supported, supported values are %v", config.CloudConfigType, supportedCloudConfigTypes.List())
		}
	}

	env, err := auth.ParseAzureEnvironment(config.Cloud, config.ResourceManagerEndpoint, config.IdentitySystem)
	if err != nil {
		return err
	}

	servicePrincipalToken, err := auth.GetServicePrincipalToken(&config.AzureAuthConfig, env)
	if err == auth.ErrorNoAuth {
		// Only controller-manager would lazy-initialize from secret, and credentials are required for such case.
		if fromSecret {
			err := fmt.Errorf("no credentials provided for Azure cloud provider")
			klog.Fatalf("%v", err)
			return err
		}

		// No credentials provided, useInstanceMetadata should be enabled for Kubelet.
		// TODO(feiskyer): print different error message for Kubelet and controller-manager, as they're
		// requiring different credential settings.
		if !config.UseInstanceMetadata && config.CloudConfigType == cloudConfigTypeFile {
			return fmt.Errorf("useInstanceMetadata must be enabled without Azure credentials")
		}

		klog.V(2).Infof("Azure cloud provider is starting without credentials")
	} else if err != nil {
		return err
	}

	// Initialize rate limiting config options.
	InitializeCloudProviderRateLimitConfig(&config.CloudProviderRateLimitConfig)

	// Conditionally configure resource request backoff
	resourceRequestBackoff := wait.Backoff{
		Steps: 1,
	}
	if config.CloudProviderBackoff {
		// Assign backoff defaults if no configuration was passed in
		if config.CloudProviderBackoffRetries == 0 {
			config.CloudProviderBackoffRetries = backoffRetriesDefault
		}
		if config.CloudProviderBackoffDuration == 0 {
			config.CloudProviderBackoffDuration = backoffDurationDefault
		}
		if config.CloudProviderBackoffExponent == 0 {
			config.CloudProviderBackoffExponent = backoffExponentDefault
		}

		if config.CloudProviderBackoffJitter == 0 {
			config.CloudProviderBackoffJitter = backoffJitterDefault
		}

		resourceRequestBackoff = wait.Backoff{
			Steps:    config.CloudProviderBackoffRetries,
			Factor:   config.CloudProviderBackoffExponent,
			Duration: time.Duration(config.CloudProviderBackoffDuration) * time.Second,
			Jitter:   config.CloudProviderBackoffJitter,
		}
		klog.V(2).Infof("Azure cloudprovider using try backoff: retries=%d, exponent=%f, duration=%d, jitter=%f",
			config.CloudProviderBackoffRetries,
			config.CloudProviderBackoffExponent,
			config.CloudProviderBackoffDuration,
			config.CloudProviderBackoffJitter)
	} else {
		// CloudProviderBackoffRetries will be set to 1 by default as the requirements of Azure SDK.
		config.CloudProviderBackoffRetries = 1
		config.CloudProviderBackoffDuration = backoffDurationDefault
	}

	if strings.EqualFold(config.LoadBalancerSku, loadBalancerSkuStandard) {
		// Do not add master nodes to standard LB by default.
		if config.ExcludeMasterFromStandardLB == nil {
			config.ExcludeMasterFromStandardLB = &defaultExcludeMasterFromStandardLB
		}

		// Enable outbound SNAT by default.
		if config.DisableOutboundSNAT == nil {
			config.DisableOutboundSNAT = &defaultDisableOutboundSNAT
		}
	} else {
		if config.DisableOutboundSNAT != nil && *config.DisableOutboundSNAT {
			return fmt.Errorf("disableOutboundSNAT should only set when loadBalancerSku is standard")
		}
	}

	az.Config = *config
	az.Environment = *env
	az.ResourceRequestBackoff = resourceRequestBackoff
	az.metadata, err = NewInstanceMetadataService(metadataURL)
	if err != nil {
		return err
	}

	// No credentials provided, InstanceMetadataService would be used for getting Azure resources.
	// Note that this only applies to Kubelet, controller-manager should configure credentials for managing Azure resources.
	if servicePrincipalToken == nil {
		return nil
	}

	// If uses network resources in different AAD Tenant, then prepare corresponding Service Principal Token for VM/VMSS client and network resources client
	var multiTenantServicePrincipalToken *adal.MultiTenantServicePrincipalToken
	var networkResourceServicePrincipalToken *adal.ServicePrincipalToken
	if az.Config.UsesNetworkResourceInDifferentTenant() {
		multiTenantServicePrincipalToken, err = auth.GetMultiTenantServicePrincipalToken(&az.Config.AzureAuthConfig, &az.Environment)
		if err != nil {
			return err
		}
		networkResourceServicePrincipalToken, err = auth.GetNetworkResourceServicePrincipalToken(&az.Config.AzureAuthConfig, &az.Environment)
		if err != nil {
			return err
		}
	}

	az.configAzureClients(servicePrincipalToken, multiTenantServicePrincipalToken, networkResourceServicePrincipalToken)

	if az.MaximumLoadBalancerRuleCount == 0 {
		az.MaximumLoadBalancerRuleCount = maximumLoadBalancerRuleCount
	}

	if strings.EqualFold(vmTypeVMSS, az.Config.VMType) {
		az.VMSet, err = newScaleSet(az)
		if err != nil {
			return err
		}
	} else {
		az.VMSet = newAvailabilitySet(az)
	}

	az.vmCache, err = az.newVMCache()
	if err != nil {
		return err
	}

	az.lbCache, err = az.newLBCache()
	if err != nil {
		return err
	}

	az.nsgCache, err = az.newNSGCache()
	if err != nil {
		return err
	}

	az.rtCache, err = az.newRouteTableCache()
	if err != nil {
		return err
	}

	if err := initDiskControllers(az); err != nil {
		return err
	}

	// start delayed route updater.
	az.routeUpdater = newDelayedRouteUpdater(az, routeUpdateInterval)
	go az.routeUpdater.run()

	return nil
}

func (az *Cloud) configAzureClients(
	servicePrincipalToken *adal.ServicePrincipalToken,
	multiTenantServicePrincipalToken *adal.MultiTenantServicePrincipalToken,
	networkResourceServicePrincipalToken *adal.ServicePrincipalToken) {
	azClientConfig := az.getAzureClientConfig(servicePrincipalToken)

	// Prepare AzureClientConfig for all azure clients
	interfaceClientConfig := azClientConfig.WithRateLimiter(az.Config.InterfaceRateLimit)
	vmSizeClientConfig := azClientConfig.WithRateLimiter(az.Config.VirtualMachineSizeRateLimit)
	snapshotClientConfig := azClientConfig.WithRateLimiter(az.Config.SnapshotRateLimit)
	storageAccountClientConfig := azClientConfig.WithRateLimiter(az.Config.StorageAccountRateLimit)
	diskClientConfig := azClientConfig.WithRateLimiter(az.Config.DiskRateLimit)
	vmClientConfig := azClientConfig.WithRateLimiter(az.Config.VirtualMachineRateLimit)
	vmssClientConfig := azClientConfig.WithRateLimiter(az.Config.VirtualMachineScaleSetRateLimit)
	// Error "not an active Virtual Machine Scale Set VM" is not retriable for VMSS VM.
	// But http.StatusNotFound is retriable because of ARM replication latency.
	vmssVMClientConfig := azClientConfig.WithRateLimiter(az.Config.VirtualMachineScaleSetRateLimit)
	vmssVMClientConfig.Backoff = vmssVMClientConfig.Backoff.WithNonRetriableErrors([]string{vmssVMNotActiveErrorMessage}).WithRetriableHTTPStatusCodes([]int{http.StatusNotFound})
	routeClientConfig := azClientConfig.WithRateLimiter(az.Config.RouteRateLimit)
	subnetClientConfig := azClientConfig.WithRateLimiter(az.Config.SubnetsRateLimit)
	routeTableClientConfig := azClientConfig.WithRateLimiter(az.Config.RouteTableRateLimit)
	loadBalancerClientConfig := azClientConfig.WithRateLimiter(az.Config.LoadBalancerRateLimit)
	securityGroupClientConfig := azClientConfig.WithRateLimiter(az.Config.SecurityGroupRateLimit)
	publicIPClientConfig := azClientConfig.WithRateLimiter(az.Config.PublicIPAddressRateLimit)
	// TODO(ZeroMagic): add azurefileRateLimit
	fileClientConfig := azClientConfig.WithRateLimiter(nil)

	// If uses network resources in different AAD Tenant, update Authorizer for VM/VMSS client config
	if multiTenantServicePrincipalToken != nil {
		multiTenantServicePrincipalTokenAuthorizer := autorest.NewMultiTenantServicePrincipalTokenAuthorizer(multiTenantServicePrincipalToken)
		vmClientConfig.Authorizer = multiTenantServicePrincipalTokenAuthorizer
		vmssClientConfig.Authorizer = multiTenantServicePrincipalTokenAuthorizer
		vmssVMClientConfig.Authorizer = multiTenantServicePrincipalTokenAuthorizer
	}

	// If uses network resources in different AAD Tenant, update SubscriptionID and Authorizer for network resources client config
	if networkResourceServicePrincipalToken != nil {
		networkResourceServicePrincipalTokenAuthorizer := autorest.NewBearerAuthorizer(networkResourceServicePrincipalToken)
		routeClientConfig.Authorizer = networkResourceServicePrincipalTokenAuthorizer
		subnetClientConfig.Authorizer = networkResourceServicePrincipalTokenAuthorizer
		routeTableClientConfig.Authorizer = networkResourceServicePrincipalTokenAuthorizer
		loadBalancerClientConfig.Authorizer = networkResourceServicePrincipalTokenAuthorizer
		securityGroupClientConfig.Authorizer = networkResourceServicePrincipalTokenAuthorizer
		publicIPClientConfig.Authorizer = networkResourceServicePrincipalTokenAuthorizer

		routeClientConfig.SubscriptionID = az.Config.NetworkResourceSubscriptionID
		subnetClientConfig.SubscriptionID = az.Config.NetworkResourceSubscriptionID
		routeTableClientConfig.SubscriptionID = az.Config.NetworkResourceSubscriptionID
		loadBalancerClientConfig.SubscriptionID = az.Config.NetworkResourceSubscriptionID
		securityGroupClientConfig.SubscriptionID = az.Config.NetworkResourceSubscriptionID
		publicIPClientConfig.SubscriptionID = az.Config.NetworkResourceSubscriptionID
	}

	// Initialize all azure clients based on client config
	az.InterfacesClient = interfaceclient.New(interfaceClientConfig)
	az.VirtualMachineSizesClient = vmsizeclient.New(vmSizeClientConfig)
	az.SnapshotsClient = snapshotclient.New(snapshotClientConfig)
	az.StorageAccountClient = storageaccountclient.New(storageAccountClientConfig)
	az.DisksClient = diskclient.New(diskClientConfig)
	az.VirtualMachinesClient = vmclient.New(vmClientConfig)
	az.VirtualMachineScaleSetsClient = vmssclient.New(vmssClientConfig)
	az.VirtualMachineScaleSetVMsClient = vmssvmclient.New(vmssVMClientConfig)
	az.RoutesClient = routeclient.New(routeClientConfig)
	az.SubnetsClient = subnetclient.New(subnetClientConfig)
	az.RouteTablesClient = routetableclient.New(routeTableClientConfig)
	az.LoadBalancerClient = loadbalancerclient.New(loadBalancerClientConfig)
	az.SecurityGroupsClient = securitygroupclient.New(securityGroupClientConfig)
	az.PublicIPAddressesClient = publicipclient.New(publicIPClientConfig)
	az.FileClient = fileclient.New(fileClientConfig)
}

func (az *Cloud) getAzureClientConfig(servicePrincipalToken *adal.ServicePrincipalToken) *azclients.ClientConfig {
	azClientConfig := &azclients.ClientConfig{
		CloudName:               az.Config.Cloud,
		Location:                az.Config.Location,
		SubscriptionID:          az.Config.SubscriptionID,
		ResourceManagerEndpoint: az.Environment.ResourceManagerEndpoint,
		Authorizer:              autorest.NewBearerAuthorizer(servicePrincipalToken),
		Backoff:                 &retry.Backoff{Steps: 1},
	}

	if az.Config.CloudProviderBackoff {
		azClientConfig.Backoff = &retry.Backoff{
			Steps:    az.Config.CloudProviderBackoffRetries,
			Factor:   az.Config.CloudProviderBackoffExponent,
			Duration: time.Duration(az.Config.CloudProviderBackoffDuration) * time.Second,
			Jitter:   az.Config.CloudProviderBackoffJitter,
		}
	}

	return azClientConfig
}

// parseConfig returns a parsed configuration for an Azure cloudprovider config file
func parseConfig(configReader io.Reader) (*Config, error) {
	var config Config
	if configReader == nil {
		return nil, nil
	}

	configContents, err := ioutil.ReadAll(configReader)
	if err != nil {
		return nil, err
	}

	err = yaml.Unmarshal(configContents, &config)
	if err != nil {
		return nil, err
	}

	// The resource group name may be in different cases from different Azure APIs, hence it is converted to lower here.
	// See more context at https://github.com/kubernetes/kubernetes/issues/71994.
	config.ResourceGroup = strings.ToLower(config.ResourceGroup)
	return &config, nil
}

// Initialize passes a Kubernetes clientBuilder interface to the cloud provider
func (az *Cloud) Initialize(clientBuilder cloudprovider.ControllerClientBuilder, stop <-chan struct{}) {
	az.KubeClient = clientBuilder.ClientOrDie("azure-cloud-provider")
	az.eventBroadcaster = record.NewBroadcaster()
	az.eventBroadcaster.StartRecordingToSink(&v1core.EventSinkImpl{Interface: az.KubeClient.CoreV1().Events("")})
	az.eventRecorder = az.eventBroadcaster.NewRecorder(scheme.Scheme, v1.EventSource{Component: "azure-cloud-provider"})
	az.InitializeCloudFromSecret()
}

// LoadBalancer returns a balancer interface. Also returns true if the interface is supported, false otherwise.
func (az *Cloud) LoadBalancer() (cloudprovider.LoadBalancer, bool) {
	return az, true
}

// Instances returns an instances interface. Also returns true if the interface is supported, false otherwise.
func (az *Cloud) Instances() (cloudprovider.Instances, bool) {
	return az, true
}

// InstancesV2 returns an instancesV2 interface. Also returns true if the interface is supported, false otherwise.
// TODO: implement ONLY for external cloud provider
func (az *Cloud) InstancesV2() (cloudprovider.InstancesV2, bool) {
	return nil, false
}

// Zones returns a zones interface. Also returns true if the interface is supported, false otherwise.
func (az *Cloud) Zones() (cloudprovider.Zones, bool) {
	return az, true
}

// Clusters returns a clusters interface.  Also returns true if the interface is supported, false otherwise.
func (az *Cloud) Clusters() (cloudprovider.Clusters, bool) {
	return nil, false
}

// Routes returns a routes interface along with whether the interface is supported.
func (az *Cloud) Routes() (cloudprovider.Routes, bool) {
	return az, true
}

// HasClusterID returns true if the cluster has a clusterID
func (az *Cloud) HasClusterID() bool {
	return true
}

// ProviderName returns the cloud provider ID.
func (az *Cloud) ProviderName() string {
	return CloudProviderName
}

func initDiskControllers(az *Cloud) error {
	// Common controller contains the function
	// needed by both blob disk and managed disk controllers

	common := &controllerCommon{
		location:              az.Location,
		storageEndpointSuffix: az.Environment.StorageEndpointSuffix,
		resourceGroup:         az.ResourceGroup,
		subscriptionID:        az.SubscriptionID,
		cloud:                 az,
		vmLockMap:             newLockMap(),
	}

	az.BlobDiskController = &BlobDiskController{common: common}
	az.ManagedDiskController = &ManagedDiskController{common: common}
	az.controllerCommon = common

	return nil
}

// SetInformers sets informers for Azure cloud provider.
func (az *Cloud) SetInformers(informerFactory informers.SharedInformerFactory) {
	klog.Infof("Setting up informers for Azure cloud provider")
	nodeInformer := informerFactory.Core().V1().Nodes().Informer()
	nodeInformer.AddEventHandler(cache.ResourceEventHandlerFuncs{
		AddFunc: func(obj interface{}) {
			node := obj.(*v1.Node)
			az.updateNodeCaches(nil, node)
		},
		UpdateFunc: func(prev, obj interface{}) {
			prevNode := prev.(*v1.Node)
			newNode := obj.(*v1.Node)
			if newNode.Labels[LabelFailureDomainBetaZone] ==
				prevNode.Labels[LabelFailureDomainBetaZone] {
				return
			}
			az.updateNodeCaches(prevNode, newNode)
		},
		DeleteFunc: func(obj interface{}) {
			node, isNode := obj.(*v1.Node)
			// We can get DeletedFinalStateUnknown instead of *v1.Node here
			// and we need to handle that correctly.
			if !isNode {
				deletedState, ok := obj.(cache.DeletedFinalStateUnknown)
				if !ok {
					klog.Errorf("Received unexpected object: %v", obj)
					return
				}
				node, ok = deletedState.Obj.(*v1.Node)
				if !ok {
					klog.Errorf("DeletedFinalStateUnknown contained non-Node object: %v", deletedState.Obj)
					return
				}
			}
			az.updateNodeCaches(node, nil)
		},
	})
	az.nodeInformerSynced = nodeInformer.HasSynced
}

// updateNodeCaches updates local cache for node's zones and external resource groups.
func (az *Cloud) updateNodeCaches(prevNode, newNode *v1.Node) {
	az.nodeCachesLock.Lock()
	defer az.nodeCachesLock.Unlock()

	if prevNode != nil {
		// Remove from nodeZones cache.
		prevZone, ok := prevNode.ObjectMeta.Labels[LabelFailureDomainBetaZone]
		if ok && az.isAvailabilityZone(prevZone) {
			az.nodeZones[prevZone].Delete(prevNode.ObjectMeta.Name)
			if az.nodeZones[prevZone].Len() == 0 {
				az.nodeZones[prevZone] = nil
			}
		}

		// Remove from nodeResourceGroups cache.
		_, ok = prevNode.ObjectMeta.Labels[externalResourceGroupLabel]
		if ok {
			delete(az.nodeResourceGroups, prevNode.ObjectMeta.Name)
		}

		// Remove from unmanagedNodes cache.
		managed, ok := prevNode.ObjectMeta.Labels[managedByAzureLabel]
		if ok && managed == "false" {
			az.unmanagedNodes.Delete(prevNode.ObjectMeta.Name)
		}
	}

	if newNode != nil {
		// Add to nodeZones cache.
		newZone, ok := newNode.ObjectMeta.Labels[LabelFailureDomainBetaZone]
		if ok && az.isAvailabilityZone(newZone) {
			if az.nodeZones[newZone] == nil {
				az.nodeZones[newZone] = sets.NewString()
			}
			az.nodeZones[newZone].Insert(newNode.ObjectMeta.Name)
		}

		// Add to nodeResourceGroups cache.
		newRG, ok := newNode.ObjectMeta.Labels[externalResourceGroupLabel]
		if ok && len(newRG) > 0 {
			az.nodeResourceGroups[newNode.ObjectMeta.Name] = strings.ToLower(newRG)
		}

		// Add to unmanagedNodes cache.
		managed, ok := newNode.ObjectMeta.Labels[managedByAzureLabel]
		if ok && managed == "false" {
			az.unmanagedNodes.Insert(newNode.ObjectMeta.Name)
		}
	}
}

// GetActiveZones returns all the zones in which k8s nodes are currently running.
func (az *Cloud) GetActiveZones() (sets.String, error) {
	if az.nodeInformerSynced == nil {
		return nil, fmt.Errorf("azure cloud provider doesn't have informers set")
	}

	az.nodeCachesLock.RLock()
	defer az.nodeCachesLock.RUnlock()
	if !az.nodeInformerSynced() {
		return nil, fmt.Errorf("node informer is not synced when trying to GetActiveZones")
	}

	zones := sets.NewString()
	for zone, nodes := range az.nodeZones {
		if len(nodes) > 0 {
			zones.Insert(zone)
		}
	}
	return zones, nil
}

// GetLocation returns the location in which k8s cluster is currently running.
func (az *Cloud) GetLocation() string {
	return az.Location
}

// GetNodeResourceGroup gets resource group for given node.
func (az *Cloud) GetNodeResourceGroup(nodeName string) (string, error) {
	// Kubelet won't set az.nodeInformerSynced, always return configured resourceGroup.
	if az.nodeInformerSynced == nil {
		return az.ResourceGroup, nil
	}

	az.nodeCachesLock.RLock()
	defer az.nodeCachesLock.RUnlock()
	if !az.nodeInformerSynced() {
		return "", fmt.Errorf("node informer is not synced when trying to GetNodeResourceGroup")
	}

	// Return external resource group if it has been cached.
	if cachedRG, ok := az.nodeResourceGroups[nodeName]; ok {
		return cachedRG, nil
	}

	// Return resource group from cloud provider options.
	return az.ResourceGroup, nil
}

// GetResourceGroups returns a set of resource groups that all nodes are running on.
func (az *Cloud) GetResourceGroups() (sets.String, error) {
	// Kubelet won't set az.nodeInformerSynced, always return configured resourceGroup.
	if az.nodeInformerSynced == nil {
		return sets.NewString(az.ResourceGroup), nil
	}

	az.nodeCachesLock.RLock()
	defer az.nodeCachesLock.RUnlock()
	if !az.nodeInformerSynced() {
		return nil, fmt.Errorf("node informer is not synced when trying to GetResourceGroups")
	}

	resourceGroups := sets.NewString(az.ResourceGroup)
	for _, rg := range az.nodeResourceGroups {
		resourceGroups.Insert(rg)
	}

	return resourceGroups, nil
}

// GetUnmanagedNodes returns a list of nodes not managed by Azure cloud provider (e.g. on-prem nodes).
func (az *Cloud) GetUnmanagedNodes() (sets.String, error) {
	// Kubelet won't set az.nodeInformerSynced, always return nil.
	if az.nodeInformerSynced == nil {
		return nil, nil
	}

	az.nodeCachesLock.RLock()
	defer az.nodeCachesLock.RUnlock()
	if !az.nodeInformerSynced() {
		return nil, fmt.Errorf("node informer is not synced when trying to GetUnmanagedNodes")
	}

	return sets.NewString(az.unmanagedNodes.List()...), nil
}

// ShouldNodeExcludedFromLoadBalancer returns true if node is unmanaged or in external resource group.
func (az *Cloud) ShouldNodeExcludedFromLoadBalancer(node *v1.Node) bool {
	labels := node.ObjectMeta.Labels
	if rg, ok := labels[externalResourceGroupLabel]; ok && !strings.EqualFold(rg, az.ResourceGroup) {
		return true
	}

	if managed, ok := labels[managedByAzureLabel]; ok && managed == "false" {
		return true
	}

	return false
}
