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
	"time"

	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/client-go/util/flowcontrol"
	"k8s.io/kubernetes/pkg/cloudprovider"
	"k8s.io/kubernetes/pkg/cloudprovider/providers/azure/auth"
	"k8s.io/kubernetes/pkg/controller"
	"k8s.io/kubernetes/pkg/version"

	"github.com/Azure/go-autorest/autorest"
	"github.com/Azure/go-autorest/autorest/azure"
	"github.com/ghodss/yaml"
	"github.com/golang/glog"
)

const (
	// CloudProviderName is the value used for the --cloud-provider flag
	CloudProviderName            = "azure"
	rateLimitQPSDefault          = 1.0
	rateLimitBucketDefault       = 5
	backoffRetriesDefault        = 6
	backoffExponentDefault       = 1.5
	backoffDurationDefault       = 5 // in seconds
	backoffJitterDefault         = 1.0
	maximumLoadBalancerRuleCount = 148 // According to Azure LB rule default limit

	vmTypeVMSS     = "vmss"
	vmTypeStandard = "standard"
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

	// Use managed service identity for the virtual machine to access Azure ARM APIs
	UseManagedIdentityExtension bool `json:"useManagedIdentityExtension"`

	// Maximum allowed LoadBalancer Rule Count is the limit enforced by Azure Load balancer
	MaximumLoadBalancerRuleCount int `json:"maximumLoadBalancerRuleCount"`
}

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
	FileClient              FileClient
	resourceRequestBackoff  wait.Backoff
	metadata                *InstanceMetadata
	vmSet                   VMSet

	// Clients for vmss.
	VirtualMachineScaleSetsClient   VirtualMachineScaleSetsClient
	VirtualMachineScaleSetVMsClient VirtualMachineScaleSetVMsClient

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

		glog.V(2).Infof("Azure cloudprovider (read ops) using rate limit config: QPS=%g, bucket=%d",
			config.CloudProviderRateLimitQPS,
			config.CloudProviderRateLimitBucket)

		glog.V(2).Infof("Azure cloudprovider (write ops) using rate limit config: QPS=%g, bucket=%d",
			config.CloudProviderRateLimitQPSWrite,
			config.CloudProviderRateLimitBucketWrite)

	}

	azClientConfig := &azClientConfig{
		subscriptionID:          config.SubscriptionID,
		resourceManagerEndpoint: env.ResourceManagerEndpoint,
		servicePrincipalToken:   servicePrincipalToken,
		rateLimiterReader:       operationPollRateLimiter,
		rateLimiterWriter:       operationPollRateLimiterWrite,
	}
	az := Cloud{
		Config:      *config,
		Environment: *env,

		DisksClient:                     newAzDisksClient(azClientConfig),
		RoutesClient:                    newAzRoutesClient(azClientConfig),
		SubnetsClient:                   newAzSubnetsClient(azClientConfig),
		InterfacesClient:                newAzInterfacesClient(azClientConfig),
		RouteTablesClient:               newAzRouteTablesClient(azClientConfig),
		LoadBalancerClient:              newAzLoadBalancersClient(azClientConfig),
		SecurityGroupsClient:            newAzSecurityGroupsClient(azClientConfig),
		StorageAccountClient:            newAzStorageAccountClient(azClientConfig),
		VirtualMachinesClient:           newAzVirtualMachinesClient(azClientConfig),
		PublicIPAddressesClient:         newAzPublicIPAddressesClient(azClientConfig),
		VirtualMachineScaleSetsClient:   newAzVirtualMachineScaleSetsClient(azClientConfig),
		VirtualMachineScaleSetVMsClient: newAzVirtualMachineScaleSetVMsClient(azClientConfig),
		FileClient:                      &azureFileClient{env: *env},
	}

	// Conditionally configure resource request backoff
	if az.CloudProviderBackoff {
		// Assign backoff defaults if no configuration was passed in
		if az.CloudProviderBackoffRetries == 0 {
			az.CloudProviderBackoffRetries = backoffRetriesDefault
		}
		if az.CloudProviderBackoffExponent == 0 {
			az.CloudProviderBackoffExponent = backoffExponentDefault
		}
		if az.CloudProviderBackoffDuration == 0 {
			az.CloudProviderBackoffDuration = backoffDurationDefault
		}
		if az.CloudProviderBackoffJitter == 0 {
			az.CloudProviderBackoffJitter = backoffJitterDefault
		}
		az.resourceRequestBackoff = wait.Backoff{
			Steps:    az.CloudProviderBackoffRetries,
			Factor:   az.CloudProviderBackoffExponent,
			Duration: time.Duration(az.CloudProviderBackoffDuration) * time.Second,
			Jitter:   az.CloudProviderBackoffJitter,
		}
		glog.V(2).Infof("Azure cloudprovider using retry backoff: retries=%d, exponent=%f, duration=%d, jitter=%f",
			az.CloudProviderBackoffRetries,
			az.CloudProviderBackoffExponent,
			az.CloudProviderBackoffDuration,
			az.CloudProviderBackoffJitter)
	}

	az.metadata = NewInstanceMetadata()

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

	return &config, nil
}

// Initialize passes a Kubernetes clientBuilder interface to the cloud provider
func (az *Cloud) Initialize(clientBuilder controller.ControllerClientBuilder) {}

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

	// BlobDiskController: contains the function needed to
	// create/attach/detach/delete blob based (unmanaged disks)
	blobController, err := newBlobDiskController(common)
	if err != nil {
		return fmt.Errorf("AzureDisk -  failed to init Blob Disk Controller with error (%s)", err.Error())
	}

	// ManagedDiskController: contains the functions needed to
	// create/attach/detach/delete managed disks
	managedController, err := newManagedDiskController(common)
	if err != nil {
		return fmt.Errorf("AzureDisk -  failed to init Managed  Disk Controller with error (%s)", err.Error())
	}

	az.BlobDiskController = blobController
	az.ManagedDiskController = managedController
	az.controllerCommon = common

	return nil
}
