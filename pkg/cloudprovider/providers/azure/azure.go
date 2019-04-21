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
	"strings"
	"sync"
	"time"

	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/client-go/informers"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/client-go/kubernetes/scheme"
	v1core "k8s.io/client-go/kubernetes/typed/core/v1"
	"k8s.io/client-go/pkg/version"
	"k8s.io/client-go/tools/cache"
	"k8s.io/client-go/tools/record"
	"k8s.io/client-go/util/flowcontrol"
	cloudprovider "k8s.io/cloud-provider"
	"k8s.io/kubernetes/pkg/cloudprovider/providers/azure/auth"

	"github.com/Azure/azure-sdk-for-go/services/compute/mgmt/2019-03-01/compute"
	"github.com/Azure/go-autorest/autorest"
	"github.com/Azure/go-autorest/autorest/azure"
	"k8s.io/klog"
	"sigs.k8s.io/yaml"
)

const (
	// CloudProviderName is the value used for the --cloud-provider flag
	CloudProviderName      = "azure"
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

	backoffModeDefault = "default"
	backoffModeV2      = "v2"

	loadBalancerSkuBasic    = "basic"
	loadBalancerSkuStandard = "standard"

	externalResourceGroupLabel = "kubernetes.azure.com/resource-group"
	managedByAzureLabel        = "kubernetes.azure.com/managed"
)

var (
	// Master nodes are not added to standard load balancer by default.
	defaultExcludeMasterFromStandardLB = true
	// Outbound SNAT is enabled by default.
	defaultDisableOutboundSNAT = false
)

// Config holds the configuration parsed from the --cloud-config flag
// All fields are required unless otherwise specified
type Config struct {
	auth.AzureAuthConfig

	// The name of the resource group that the cluster is deployed in
	ResourceGroup string `json:"resourceGroup" yaml:"resourceGroup"`
	// The location of the resource group that the cluster is deployed in
	Location string `json:"location" yaml:"location"`
	// The name of the VNet that the cluster is deployed in
	VnetName string `json:"vnetName" yaml:"vnetName"`
	// The name of the resource group that the Vnet is deployed in
	VnetResourceGroup string `json:"vnetResourceGroup" yaml:"vnetResourceGroup"`
	// The name of the subnet that the cluster is deployed in
	SubnetName string `json:"subnetName" yaml:"subnetName"`
	// The name of the security group attached to the cluster's subnet
	SecurityGroupName string `json:"securityGroupName" yaml:"securityGroupName"`
	// (Optional in 1.6) The name of the route table attached to the subnet that the cluster is deployed in
	RouteTableName string `json:"routeTableName" yaml:"routeTableName"`
	// (Optional) The name of the availability set that should be used as the load balancer backend
	// If this is set, the Azure cloudprovider will only add nodes from that availability set to the load
	// balancer backend pool. If this is not set, and multiple agent pools (availability sets) are used, then
	// the cloudprovider will try to add all nodes to a single backend pool which is forbidden.
	// In other words, if you use multiple agent pools (availability sets), you MUST set this field.
	PrimaryAvailabilitySetName string `json:"primaryAvailabilitySetName" yaml:"primaryAvailabilitySetName"`
	// The type of azure nodes. Candidate values are: vmss and standard.
	// If not set, it will be default to standard.
	VMType string `json:"vmType" yaml:"vmType"`
	// The name of the scale set that should be used as the load balancer backend.
	// If this is set, the Azure cloudprovider will only add nodes from that scale set to the load
	// balancer backend pool. If this is not set, and multiple agent pools (scale sets) are used, then
	// the cloudprovider will try to add all nodes to a single backend pool which is forbidden.
	// In other words, if you use multiple agent pools (scale sets), you MUST set this field.
	PrimaryScaleSetName string `json:"primaryScaleSetName" yaml:"primaryScaleSetName"`
	// Enable exponential backoff to manage resource request retries
	CloudProviderBackoff bool `json:"cloudProviderBackoff" yaml:"cloudProviderBackoff"`
	// Backoff retry limit
	CloudProviderBackoffRetries int `json:"cloudProviderBackoffRetries" yaml:"cloudProviderBackoffRetries"`
	// Backoff exponent
	CloudProviderBackoffExponent float64 `json:"cloudProviderBackoffExponent" yaml:"cloudProviderBackoffExponent"`
	// Backoff duration
	CloudProviderBackoffDuration int `json:"cloudProviderBackoffDuration" yaml:"cloudProviderBackoffDuration"`
	// Backoff jitter
	CloudProviderBackoffJitter float64 `json:"cloudProviderBackoffJitter" yaml:"cloudProviderBackoffJitter"`
	// Backoff mode, options are v2 and default.
	// * default means two-layer backoff retrying, one in the cloud provider and the other in the Azure SDK.
	// * v2 means only backoff in the Azure SDK is used. In such mode, CloudProviderBackoffDuration and
	//   CloudProviderBackoffJitter are omitted.
	// "default" will be used if not specified.
	CloudProviderBackoffMode string `json:"cloudProviderBackoffMode" yaml:"cloudProviderBackoffMode"`
	// Enable rate limiting
	CloudProviderRateLimit bool `json:"cloudProviderRateLimit" yaml:"cloudProviderRateLimit"`
	// Rate limit QPS (Read)
	CloudProviderRateLimitQPS float32 `json:"cloudProviderRateLimitQPS" yaml:"cloudProviderRateLimitQPS"`
	// Rate limit Bucket Size
	CloudProviderRateLimitBucket int `json:"cloudProviderRateLimitBucket" yaml:"cloudProviderRateLimitBucket"`
	// Rate limit QPS (Write)
	CloudProviderRateLimitQPSWrite float32 `json:"cloudProviderRateLimitQPSWrite" yaml:"cloudProviderRateLimitQPSWrite"`
	// Rate limit Bucket Size
	CloudProviderRateLimitBucketWrite int `json:"cloudProviderRateLimitBucketWrite" yaml:"cloudProviderRateLimitBucketWrite"`

	// Use instance metadata service where possible
	UseInstanceMetadata bool `json:"useInstanceMetadata" yaml:"useInstanceMetadata"`

	// Sku of Load Balancer and Public IP. Candidate values are: basic and standard.
	// If not set, it will be default to basic.
	LoadBalancerSku string `json:"loadBalancerSku" yaml:"loadBalancerSku"`
	// ExcludeMasterFromStandardLB excludes master nodes from standard load balancer.
	// If not set, it will be default to true.
	ExcludeMasterFromStandardLB *bool `json:"excludeMasterFromStandardLB" yaml:"excludeMasterFromStandardLB"`
	// DisableOutboundSNAT disables the outbound SNAT for public load balancer rules.
	// It should only be set when loadBalancerSku is standard. If not set, it will be default to false.
	DisableOutboundSNAT *bool `json:"disableOutboundSNAT" yaml:"disableOutboundSNAT"`

	// Maximum allowed LoadBalancer Rule Count is the limit enforced by Azure Load balancer
	MaximumLoadBalancerRuleCount int `json:"maximumLoadBalancerRuleCount" yaml:"maximumLoadBalancerRuleCount"`
}

var _ cloudprovider.Interface = (*Cloud)(nil)
var _ cloudprovider.Instances = (*Cloud)(nil)
var _ cloudprovider.LoadBalancer = (*Cloud)(nil)
var _ cloudprovider.Routes = (*Cloud)(nil)
var _ cloudprovider.Zones = (*Cloud)(nil)
var _ cloudprovider.PVLabeler = (*Cloud)(nil)

// Cloud holds the config and clients
type Cloud struct {
	Config
	Environment             azure.Environment
	RoutesClient            RoutesClient
	SubnetsClient           SubnetsClient
	InterfacesClient        InterfacesClient
	RouteTablesClient       RouteTablesClient
	LoadBalancerClient      LoadBalancersClient
	PublicIPAddressesClient PublicIPAddressesClient
	SecurityGroupsClient    SecurityGroupsClient
	VirtualMachinesClient   VirtualMachinesClient
	StorageAccountClient    StorageAccountClient
	DisksClient             DisksClient
	SnapshotsClient         *compute.SnapshotsClient
	FileClient              FileClient
	resourceRequestBackoff  wait.Backoff
	metadata                *InstanceMetadataService
	vmSet                   VMSet

	// Lock for access to node caches, includes nodeZones, nodeResourceGroups, and unmanagedNodes.
	nodeCachesLock sync.Mutex
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

	// Clients for vmss.
	VirtualMachineScaleSetsClient   VirtualMachineScaleSetsClient
	VirtualMachineScaleSetVMsClient VirtualMachineScaleSetVMsClient

	// client for vm sizes list
	VirtualMachineSizesClient VirtualMachineSizesClient

	kubeClient       clientset.Interface
	eventBroadcaster record.EventBroadcaster
	eventRecorder    record.EventRecorder

	vmCache  *timedCache
	lbCache  *timedCache
	nsgCache *timedCache
	rtCache  *timedCache

	*BlobDiskController
	*ManagedDiskController
	*controllerCommon
}

func init() {
	cloudprovider.RegisterCloudProvider(CloudProviderName, NewCloud)
}

// NewCloud returns a Cloud with initialized clients
func NewCloud(configReader io.Reader) (cloudprovider.Interface, error) {
	config, err := parseConfig(configReader)
	if err != nil {
		return nil, err
	}

	if config.VMType == "" {
		// default to standard vmType if not set.
		config.VMType = vmTypeStandard
	}

	env, err := auth.ParseAzureEnvironment(config.Cloud)
	if err != nil {
		return nil, err
	}

	servicePrincipalToken, err := auth.GetServicePrincipalToken(&config.AzureAuthConfig, env)
	if err != nil {
		return nil, err
	}

	// operationPollRateLimiter.Accept() is a no-op if rate limits are configured off.
	operationPollRateLimiter := flowcontrol.NewFakeAlwaysRateLimiter()
	operationPollRateLimiterWrite := flowcontrol.NewFakeAlwaysRateLimiter()

	// If reader is provided (and no writer) we will
	// use the same value for both.
	if config.CloudProviderRateLimit {
		// Assign rate limit defaults if no configuration was passed in
		if config.CloudProviderRateLimitQPS == 0 {
			config.CloudProviderRateLimitQPS = rateLimitQPSDefault
		}
		if config.CloudProviderRateLimitBucket == 0 {
			config.CloudProviderRateLimitBucket = rateLimitBucketDefault
		}
		if config.CloudProviderRateLimitQPSWrite == 0 {
			config.CloudProviderRateLimitQPSWrite = rateLimitQPSDefault
		}
		if config.CloudProviderRateLimitBucketWrite == 0 {
			config.CloudProviderRateLimitBucketWrite = rateLimitBucketDefault
		}

		operationPollRateLimiter = flowcontrol.NewTokenBucketRateLimiter(
			config.CloudProviderRateLimitQPS,
			config.CloudProviderRateLimitBucket)

		operationPollRateLimiterWrite = flowcontrol.NewTokenBucketRateLimiter(
			config.CloudProviderRateLimitQPSWrite,
			config.CloudProviderRateLimitBucketWrite)

		klog.V(2).Infof("Azure cloudprovider (read ops) using rate limit config: QPS=%g, bucket=%d",
			config.CloudProviderRateLimitQPS,
			config.CloudProviderRateLimitBucket)

		klog.V(2).Infof("Azure cloudprovider (write ops) using rate limit config: QPS=%g, bucket=%d",
			config.CloudProviderRateLimitQPSWrite,
			config.CloudProviderRateLimitBucketWrite)
	}

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
		} else if config.shouldOmitCloudProviderBackoff() {
			klog.Warning("Azure cloud provider config 'cloudProviderBackoffExponent' has been deprecated for 'v2' backoff mode. 2 is always used as the backoff exponent.")
		}
		if config.CloudProviderBackoffJitter == 0 {
			config.CloudProviderBackoffJitter = backoffJitterDefault
		} else if config.shouldOmitCloudProviderBackoff() {
			klog.Warning("Azure cloud provider config 'cloudProviderBackoffJitter' has been deprecated for 'v2' backoff mode.")
		}

		if !config.shouldOmitCloudProviderBackoff() {
			resourceRequestBackoff = wait.Backoff{
				Steps:    config.CloudProviderBackoffRetries,
				Factor:   config.CloudProviderBackoffExponent,
				Duration: time.Duration(config.CloudProviderBackoffDuration) * time.Second,
				Jitter:   config.CloudProviderBackoffJitter,
			}
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
			return nil, fmt.Errorf("disableOutboundSNAT should only set when loadBalancerSku is standard")
		}
	}

	azClientConfig := &azClientConfig{
		subscriptionID:                 config.SubscriptionID,
		resourceManagerEndpoint:        env.ResourceManagerEndpoint,
		servicePrincipalToken:          servicePrincipalToken,
		rateLimiterReader:              operationPollRateLimiter,
		rateLimiterWriter:              operationPollRateLimiterWrite,
		CloudProviderBackoffRetries:    config.CloudProviderBackoffRetries,
		CloudProviderBackoffDuration:   config.CloudProviderBackoffDuration,
		ShouldOmitCloudProviderBackoff: config.shouldOmitCloudProviderBackoff(),
	}
	az := Cloud{
		Config:                 *config,
		Environment:            *env,
		nodeZones:              map[string]sets.String{},
		nodeResourceGroups:     map[string]string{},
		unmanagedNodes:         sets.NewString(),
		routeCIDRs:             map[string]string{},
		resourceRequestBackoff: resourceRequestBackoff,

		DisksClient:                     newAzDisksClient(azClientConfig),
		SnapshotsClient:                 newSnapshotsClient(azClientConfig),
		RoutesClient:                    newAzRoutesClient(azClientConfig),
		SubnetsClient:                   newAzSubnetsClient(azClientConfig),
		InterfacesClient:                newAzInterfacesClient(azClientConfig),
		RouteTablesClient:               newAzRouteTablesClient(azClientConfig),
		LoadBalancerClient:              newAzLoadBalancersClient(azClientConfig),
		SecurityGroupsClient:            newAzSecurityGroupsClient(azClientConfig),
		StorageAccountClient:            newAzStorageAccountClient(azClientConfig),
		VirtualMachinesClient:           newAzVirtualMachinesClient(azClientConfig),
		PublicIPAddressesClient:         newAzPublicIPAddressesClient(azClientConfig),
		VirtualMachineSizesClient:       newAzVirtualMachineSizesClient(azClientConfig),
		VirtualMachineScaleSetsClient:   newAzVirtualMachineScaleSetsClient(azClientConfig),
		VirtualMachineScaleSetVMsClient: newAzVirtualMachineScaleSetVMsClient(azClientConfig),
		FileClient:                      &azureFileClient{env: *env},
	}

	az.metadata, err = NewInstanceMetadataService(metadataURL)
	if err != nil {
		return nil, err
	}

	if az.MaximumLoadBalancerRuleCount == 0 {
		az.MaximumLoadBalancerRuleCount = maximumLoadBalancerRuleCount
	}

	if strings.EqualFold(vmTypeVMSS, az.Config.VMType) {
		az.vmSet, err = newScaleSet(&az)
		if err != nil {
			return nil, err
		}
	} else {
		az.vmSet = newAvailabilitySet(&az)
	}

	az.vmCache, err = az.newVMCache()
	if err != nil {
		return nil, err
	}

	az.lbCache, err = az.newLBCache()
	if err != nil {
		return nil, err
	}

	az.nsgCache, err = az.newNSGCache()
	if err != nil {
		return nil, err
	}

	az.rtCache, err = az.newRouteTableCache()
	if err != nil {
		return nil, err
	}

	if err := initDiskControllers(&az); err != nil {
		return nil, err
	}
	return &az, nil
}

// parseConfig returns a parsed configuration for an Azure cloudprovider config file
func parseConfig(configReader io.Reader) (*Config, error) {
	var config Config

	if configReader == nil {
		return &config, nil
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
	az.kubeClient = clientBuilder.ClientOrDie("azure-cloud-provider")
	az.eventBroadcaster = record.NewBroadcaster()
	az.eventBroadcaster.StartRecordingToSink(&v1core.EventSinkImpl{Interface: az.kubeClient.CoreV1().Events("")})
	az.eventRecorder = az.eventBroadcaster.NewRecorder(scheme.Scheme, v1.EventSource{Component: "azure-cloud-provider"})
}

// LoadBalancer returns a balancer interface. Also returns true if the interface is supported, false otherwise.
func (az *Cloud) LoadBalancer() (cloudprovider.LoadBalancer, bool) {
	return az, true
}

// Instances returns an instances interface. Also returns true if the interface is supported, false otherwise.
func (az *Cloud) Instances() (cloudprovider.Instances, bool) {
	return az, true
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

// configureUserAgent configures the autorest client with a user agent that
// includes "kubernetes" and the full kubernetes git version string
// example:
// Azure-SDK-for-Go/7.0.1-beta arm-network/2016-09-01; kubernetes-cloudprovider/v1.7.0-alpha.2.711+a2fadef8170bb0-dirty;
func configureUserAgent(client *autorest.Client) {
	k8sVersion := version.Get().GitVersion
	client.UserAgent = fmt.Sprintf("%s; kubernetes-cloudprovider/%s", client.UserAgent, k8sVersion)
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
			if newNode.Labels[v1.LabelZoneFailureDomain] ==
				prevNode.Labels[v1.LabelZoneFailureDomain] {
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
		prevZone, ok := prevNode.ObjectMeta.Labels[v1.LabelZoneFailureDomain]
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
		newZone, ok := newNode.ObjectMeta.Labels[v1.LabelZoneFailureDomain]
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
		return nil, fmt.Errorf("Azure cloud provider doesn't have informers set")
	}

	az.nodeCachesLock.Lock()
	defer az.nodeCachesLock.Unlock()
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

	az.nodeCachesLock.Lock()
	defer az.nodeCachesLock.Unlock()
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

	az.nodeCachesLock.Lock()
	defer az.nodeCachesLock.Unlock()
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

	az.nodeCachesLock.Lock()
	defer az.nodeCachesLock.Unlock()
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
