/*
Copyright 2014 The Kubernetes Authors.

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

package gce

import (
	"fmt"
	"io"
	"net/http"
	"regexp"
	"strconv"
	"strings"
	"sync"
	"time"

	gcfg "gopkg.in/gcfg.v1"

	"cloud.google.com/go/compute/metadata"

	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apiserver/pkg/server/options/encryptionconfig"
	"k8s.io/apiserver/pkg/storage/value/encrypt/envelope"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/client-go/kubernetes/scheme"
	v1core "k8s.io/client-go/kubernetes/typed/core/v1"
	"k8s.io/client-go/tools/record"
	"k8s.io/client-go/util/flowcontrol"
	"k8s.io/kubernetes/pkg/cloudprovider"
	"k8s.io/kubernetes/pkg/controller"

	"github.com/golang/glog"
	"golang.org/x/oauth2"
	"golang.org/x/oauth2/google"
	cloudkms "google.golang.org/api/cloudkms/v1"
	computealpha "google.golang.org/api/compute/v0.alpha"
	computebeta "google.golang.org/api/compute/v0.beta"
	compute "google.golang.org/api/compute/v1"
	container "google.golang.org/api/container/v1"
)

const (
	ProviderName = "gce"

	k8sNodeRouteTag = "k8s-node-route"

	// AffinityTypeNone - no session affinity.
	gceAffinityTypeNone = "NONE"
	// AffinityTypeClientIP - affinity based on Client IP.
	gceAffinityTypeClientIP = "CLIENT_IP"
	// AffinityTypeClientIPProto - affinity based on Client IP and port.
	gceAffinityTypeClientIPProto = "CLIENT_IP_PROTO"

	operationPollInterval = 3 * time.Second
	// Creating Route in very large clusters, may take more than half an hour.
	operationPollTimeoutDuration = time.Hour

	// Each page can have 500 results, but we cap how many pages
	// are iterated through to prevent infinite loops if the API
	// were to continuously return a nextPageToken.
	maxPages = 25

	maxTargetPoolCreateInstances = 200

	// HTTP Load Balancer parameters
	// Configure 2 second period for external health checks.
	gceHcCheckIntervalSeconds = int64(2)
	gceHcTimeoutSeconds       = int64(1)
	// Start sending requests as soon as a pod is found on the node.
	gceHcHealthyThreshold = int64(1)
	// Defaults to 5 * 2 = 10 seconds before the LB will steer traffic away
	gceHcUnhealthyThreshold = int64(5)

	gceComputeAPIEndpoint      = "https://www.googleapis.com/compute/v1/"
	gceComputeAPIEndpointAlpha = "https://www.googleapis.com/compute/alpha/"
)

// gceObject is an abstraction of all GCE API object in go client
type gceObject interface {
	MarshalJSON() ([]byte, error)
}

// GCECloud is an implementation of Interface, LoadBalancer and Instances for Google Compute Engine.
type GCECloud struct {
	// ClusterID contains functionality for getting (and initializing) the ingress-uid. Call GCECloud.Initialize()
	// for the cloudprovider to start watching the configmap.
	ClusterID ClusterID

	service                  *compute.Service
	serviceBeta              *computebeta.Service
	serviceAlpha             *computealpha.Service
	containerService         *container.Service
	cloudkmsService          *cloudkms.Service
	client                   clientset.Interface
	clientBuilder            controller.ControllerClientBuilder
	eventBroadcaster         record.EventBroadcaster
	eventRecorder            record.EventRecorder
	projectID                string
	region                   string
	localZone                string   // The zone in which we are running
	managedZones             []string // List of zones we are spanning (for multi-AZ clusters, primarily when running on master)
	networkURL               string
	subnetworkURL            string
	secondaryRangeName       string
	networkProjectID         string
	onXPN                    bool
	nodeTags                 []string    // List of tags to use on firewall rules for load balancers
	lastComputedNodeTags     []string    // List of node tags calculated in GetHostTags()
	lastKnownNodeNames       sets.String // List of hostnames used to calculate lastComputedHostTags in GetHostTags(names)
	computeNodeTagLock       sync.Mutex  // Lock for computing and setting node tags
	nodeInstancePrefix       string      // If non-"", an advisory prefix for all nodes in the cluster
	useMetadataServer        bool
	operationPollRateLimiter flowcontrol.RateLimiter
	manager                  diskServiceManager
	// sharedResourceLock is used to serialize GCE operations that may mutate shared state to
	// prevent inconsistencies. For example, load balancers manipulation methods will take the
	// lock to prevent shared resources from being prematurely deleted while the operation is
	// in progress.
	sharedResourceLock sync.Mutex
	// AlphaFeatureGate gates gce alpha features in GCECloud instance.
	// Related wrapper functions that interacts with gce alpha api should examine whether
	// the corresponding api is enabled.
	// If not enabled, it should return error.
	AlphaFeatureGate *AlphaFeatureGate
}

type ConfigGlobal struct {
	TokenURL  string `gcfg:"token-url"`
	TokenBody string `gcfg:"token-body"`
	// ProjectID and NetworkProjectID can either be the numeric or string-based
	// unique identifier that starts with [a-z].
	ProjectID string `gcfg:"project-id"`
	// NetworkProjectID refers to the project which owns the network being used.
	NetworkProjectID string `gcfg:"network-project-id"`
	NetworkName      string `gcfg:"network-name"`
	SubnetworkName   string `gcfg:"subnetwork-name"`
	// SecondaryRangeName is the name of the secondary range to allocate IP
	// aliases. The secondary range must be present on the subnetwork the
	// cluster is attached to.
	SecondaryRangeName string   `gcfg:"secondary-range-name"`
	NodeTags           []string `gcfg:"node-tags"`
	NodeInstancePrefix string   `gcfg:"node-instance-prefix"`
	Multizone          bool     `gcfg:"multizone"`
	// ApiEndpoint is the GCE compute API endpoint to use. If this is blank,
	// then the default endpoint is used.
	ApiEndpoint string `gcfg:"api-endpoint"`
	// LocalZone specifies the GCE zone that gce cloud client instance is
	// located in (i.e. where the controller will be running). If this is
	// blank, then the local zone will be discovered via the metadata server.
	LocalZone string `gcfg:"local-zone"`
	// Possible values: List of api names separated by comma. Default to none.
	// For example: MyFeatureFlag
	AlphaFeatures []string `gcfg:"alpha-features"`
}

// ConfigFile is the struct used to parse the /etc/gce.conf configuration file.
type ConfigFile struct {
	Global ConfigGlobal `gcfg:"global"`
}

// CloudConfig includes all the necessary configuration for creating GCECloud
type CloudConfig struct {
	ApiEndpoint        string
	ProjectID          string
	NetworkProjectID   string
	Region             string
	Zone               string
	ManagedZones       []string
	NetworkName        string
	NetworkURL         string
	SubnetworkName     string
	SubnetworkURL      string
	SecondaryRangeName string
	NodeTags           []string
	NodeInstancePrefix string
	TokenSource        oauth2.TokenSource
	UseMetadataServer  bool
	AlphaFeatureGate   *AlphaFeatureGate
}

// kmsPluginRegisterOnce prevents the cloudprovider from registering its KMS plugin
// more than once in the KMS plugin registry.
var kmsPluginRegisterOnce sync.Once

func init() {
	cloudprovider.RegisterCloudProvider(
		ProviderName,
		func(config io.Reader) (cloudprovider.Interface, error) {
			return newGCECloud(config)
		})
}

// Raw access to the underlying GCE service, probably should only be used for e2e tests
func (g *GCECloud) GetComputeService() *compute.Service {
	return g.service
}

// Raw access to the cloudkmsService of GCE cloud. Required for encryption of etcd using Google KMS.
func (g *GCECloud) GetKMSService() *cloudkms.Service {
	return g.cloudkmsService
}

// newGCECloud creates a new instance of GCECloud.
func newGCECloud(config io.Reader) (gceCloud *GCECloud, err error) {
	var cloudConfig *CloudConfig
	var configFile *ConfigFile

	if config != nil {
		configFile, err = readConfig(config)
		if err != nil {
			return nil, err
		}
		glog.Infof("Using GCE provider config %+v", configFile)
	}

	cloudConfig, err = generateCloudConfig(configFile)
	if err != nil {
		return nil, err
	}
	return CreateGCECloud(cloudConfig)

}

func readConfig(reader io.Reader) (*ConfigFile, error) {
	cfg := &ConfigFile{}
	if err := gcfg.FatalOnly(gcfg.ReadInto(cfg, reader)); err != nil {
		glog.Errorf("Couldn't read config: %v", err)
		return nil, err
	}
	return cfg, nil
}

func generateCloudConfig(configFile *ConfigFile) (cloudConfig *CloudConfig, err error) {
	cloudConfig = &CloudConfig{}
	// By default, fetch token from GCE metadata server
	cloudConfig.TokenSource = google.ComputeTokenSource("")
	cloudConfig.UseMetadataServer = true

	if configFile != nil {
		if configFile.Global.ApiEndpoint != "" {
			cloudConfig.ApiEndpoint = configFile.Global.ApiEndpoint
		}

		if configFile.Global.TokenURL != "" {
			// if tokenURL is nil, set tokenSource to nil. This will force the OAuth client to fall
			// back to use DefaultTokenSource. This allows running gceCloud remotely.
			if configFile.Global.TokenURL == "nil" {
				cloudConfig.TokenSource = nil
			} else {
				cloudConfig.TokenSource = NewAltTokenSource(configFile.Global.TokenURL, configFile.Global.TokenBody)
			}
		}

		cloudConfig.NodeTags = configFile.Global.NodeTags
		cloudConfig.NodeInstancePrefix = configFile.Global.NodeInstancePrefix

		alphaFeatureGate, err := NewAlphaFeatureGate(configFile.Global.AlphaFeatures)
		if err != nil {
			glog.Errorf("Encountered error for creating alpha feature gate: %v", err)
		}
		cloudConfig.AlphaFeatureGate = alphaFeatureGate
	}

	// retrieve projectID and zone
	if configFile == nil || configFile.Global.ProjectID == "" || configFile.Global.LocalZone == "" {
		cloudConfig.ProjectID, cloudConfig.Zone, err = getProjectAndZone()
		if err != nil {
			return nil, err
		}
	}

	if configFile != nil {
		if configFile.Global.ProjectID != "" {
			cloudConfig.ProjectID = configFile.Global.ProjectID
		}
		if configFile.Global.LocalZone != "" {
			cloudConfig.Zone = configFile.Global.LocalZone
		}
		if configFile.Global.NetworkProjectID != "" {
			cloudConfig.NetworkProjectID = configFile.Global.NetworkProjectID
		}
	}

	// retrieve region
	cloudConfig.Region, err = GetGCERegion(cloudConfig.Zone)
	if err != nil {
		return nil, err
	}

	// generate managedZones
	cloudConfig.ManagedZones = []string{cloudConfig.Zone}
	if configFile != nil && configFile.Global.Multizone {
		cloudConfig.ManagedZones = nil // Use all zones in region
	}

	// Determine if network parameter is URL or Name
	if configFile != nil && configFile.Global.NetworkName != "" {
		if strings.Contains(configFile.Global.NetworkName, "/") {
			cloudConfig.NetworkURL = configFile.Global.NetworkName
		} else {
			cloudConfig.NetworkName = configFile.Global.NetworkName
		}
	} else {
		cloudConfig.NetworkName, err = getNetworkNameViaMetadata()
		if err != nil {
			return nil, err
		}
	}

	// Determine if subnetwork parameter is URL or Name
	// If cluster is on a GCP network of mode=custom, then `SubnetName` must be specified in config file.
	if configFile != nil && configFile.Global.SubnetworkName != "" {
		if strings.Contains(configFile.Global.SubnetworkName, "/") {
			cloudConfig.SubnetworkURL = configFile.Global.SubnetworkName
		} else {
			cloudConfig.SubnetworkName = configFile.Global.SubnetworkName
		}
	}

	if configFile != nil {
		cloudConfig.SecondaryRangeName = configFile.Global.SecondaryRangeName
	}

	return cloudConfig, err
}

// CreateGCECloud creates a GCECloud object using the specified parameters.
// If no networkUrl is specified, loads networkName via rest call.
// If no tokenSource is specified, uses oauth2.DefaultTokenSource.
// If managedZones is nil / empty all zones in the region will be managed.
func CreateGCECloud(config *CloudConfig) (*GCECloud, error) {
	// Use ProjectID for NetworkProjectID, if it wasn't explicitly set.
	if config.NetworkProjectID == "" {
		config.NetworkProjectID = config.ProjectID
	}

	client, err := newOauthClient(config.TokenSource)
	if err != nil {
		return nil, err
	}
	service, err := compute.New(client)
	if err != nil {
		return nil, err
	}

	client, err = newOauthClient(config.TokenSource)
	if err != nil {
		return nil, err
	}
	serviceBeta, err := computebeta.New(client)
	if err != nil {
		return nil, err
	}

	client, err = newOauthClient(config.TokenSource)
	if err != nil {
		return nil, err
	}
	serviceAlpha, err := computealpha.New(client)
	if err != nil {
		return nil, err
	}

	// Expect override api endpoint to always be v1 api and follows the same pattern as prod.
	// Generate alpha and beta api endpoints based on override v1 api endpoint.
	// For example,
	// staging API endpoint: https://www.googleapis.com/compute/staging_v1/
	if config.ApiEndpoint != "" {
		service.BasePath = fmt.Sprintf("%sprojects/", config.ApiEndpoint)
		serviceBeta.BasePath = fmt.Sprintf("%sprojects/", strings.Replace(config.ApiEndpoint, "v1", "beta", -1))
		serviceAlpha.BasePath = fmt.Sprintf("%sprojects/", strings.Replace(config.ApiEndpoint, "v1", "alpha", -1))
	}

	containerService, err := container.New(client)
	if err != nil {
		return nil, err
	}

	cloudkmsService, err := cloudkms.New(client)
	if err != nil {
		return nil, err
	}

	// ProjectID and.NetworkProjectID may be project number or name.
	projID, netProjID := tryConvertToProjectNames(config.ProjectID, config.NetworkProjectID, service)

	onXPN := projID != netProjID

	var networkURL string
	var subnetURL string

	if config.NetworkName == "" && config.NetworkURL == "" {
		// TODO: Stop using this call and return an error.
		// This function returns the first network in a list of networks for a project. The project
		// should be set via configuration instead of randomly taking the first.
		networkName, err := getNetworkNameViaAPICall(service, config.NetworkProjectID)
		if err != nil {
			return nil, err
		}
		networkURL = gceNetworkURL(config.ApiEndpoint, netProjID, networkName)
	} else if config.NetworkURL != "" {
		networkURL = config.NetworkURL
	} else {
		networkURL = gceNetworkURL(config.ApiEndpoint, netProjID, config.NetworkName)
	}

	if config.SubnetworkURL != "" {
		subnetURL = config.SubnetworkURL
	} else if config.SubnetworkName != "" {
		subnetURL = gceSubnetworkURL(config.ApiEndpoint, netProjID, config.Region, config.SubnetworkName)
	}

	if len(config.ManagedZones) == 0 {
		config.ManagedZones, err = getZonesForRegion(service, config.ProjectID, config.Region)
		if err != nil {
			return nil, err
		}
	}
	if len(config.ManagedZones) != 1 {
		glog.Infof("managing multiple zones: %v", config.ManagedZones)
	}

	operationPollRateLimiter := flowcontrol.NewTokenBucketRateLimiter(10, 100) // 10 qps, 100 bucket size.

	gce := &GCECloud{
		service:                  service,
		serviceAlpha:             serviceAlpha,
		serviceBeta:              serviceBeta,
		containerService:         containerService,
		cloudkmsService:          cloudkmsService,
		projectID:                projID,
		networkProjectID:         netProjID,
		onXPN:                    onXPN,
		region:                   config.Region,
		localZone:                config.Zone,
		managedZones:             config.ManagedZones,
		networkURL:               networkURL,
		subnetworkURL:            subnetURL,
		secondaryRangeName:       config.SecondaryRangeName,
		nodeTags:                 config.NodeTags,
		nodeInstancePrefix:       config.NodeInstancePrefix,
		useMetadataServer:        config.UseMetadataServer,
		operationPollRateLimiter: operationPollRateLimiter,
		AlphaFeatureGate:         config.AlphaFeatureGate,
	}

	gce.manager = &gceServiceManager{gce}

	// Registering the KMS plugin only the first time.
	kmsPluginRegisterOnce.Do(func() {
		// Register the Google Cloud KMS based service in the KMS plugin registry.
		encryptionconfig.KMSPluginRegistry.Register(KMSServiceName, func(config io.Reader) (envelope.Service, error) {
			return gce.getGCPCloudKMSService(config)
		})
	})

	return gce, nil
}

func tryConvertToProjectNames(configProject, configNetworkProject string, service *compute.Service) (projID, netProjID string) {
	projID = configProject
	if isProjectNumber(projID) {
		projName, err := getProjectID(service, projID)
		if err != nil {
			glog.Warningf("Failed to retrieve project %v while trying to retrieve its name. err %v", projID, err)
		} else {
			projID = projName
		}
	}

	netProjID = projID
	if configNetworkProject != configProject {
		netProjID = configNetworkProject
	}
	if isProjectNumber(netProjID) {
		netProjName, err := getProjectID(service, netProjID)
		if err != nil {
			glog.Warningf("Failed to retrieve network project %v while trying to retrieve its name. err %v", netProjID, err)
		} else {
			netProjID = netProjName
		}
	}

	return projID, netProjID
}

// Initialize takes in a clientBuilder and spawns a goroutine for watching the clusterid configmap.
// This must be called before utilizing the funcs of gce.ClusterID
func (gce *GCECloud) Initialize(clientBuilder controller.ControllerClientBuilder) {
	gce.clientBuilder = clientBuilder
	gce.client = clientBuilder.ClientOrDie("cloud-provider")

	if gce.OnXPN() {
		gce.eventBroadcaster = record.NewBroadcaster()
		gce.eventBroadcaster.StartRecordingToSink(&v1core.EventSinkImpl{Interface: v1core.New(gce.client.Core().RESTClient()).Events("")})
		gce.eventRecorder = gce.eventBroadcaster.NewRecorder(scheme.Scheme, v1.EventSource{Component: "gce-cloudprovider"})
	}

	go gce.watchClusterID()
}

// LoadBalancer returns an implementation of LoadBalancer for Google Compute Engine.
func (gce *GCECloud) LoadBalancer() (cloudprovider.LoadBalancer, bool) {
	return gce, true
}

// Instances returns an implementation of Instances for Google Compute Engine.
func (gce *GCECloud) Instances() (cloudprovider.Instances, bool) {
	return gce, true
}

// Zones returns an implementation of Zones for Google Compute Engine.
func (gce *GCECloud) Zones() (cloudprovider.Zones, bool) {
	return gce, true
}

func (gce *GCECloud) Clusters() (cloudprovider.Clusters, bool) {
	return gce, true
}

// Routes returns an implementation of Routes for Google Compute Engine.
func (gce *GCECloud) Routes() (cloudprovider.Routes, bool) {
	return gce, true
}

// ProviderName returns the cloud provider ID.
func (gce *GCECloud) ProviderName() string {
	return ProviderName
}

// ProjectID returns the ProjectID corresponding to the project this cloud is in.
func (g *GCECloud) ProjectID() string {
	return g.projectID
}

// NetworkProjectID returns the ProjectID corresponding to the project this cluster's network is in.
func (g *GCECloud) NetworkProjectID() string {
	return g.networkProjectID
}

// Region returns the region
func (gce *GCECloud) Region() string {
	return gce.region
}

// OnXPN returns true if the cluster is running on a cross project network (XPN)
func (gce *GCECloud) OnXPN() bool {
	return gce.onXPN
}

// NetworkURL returns the network url
func (gce *GCECloud) NetworkURL() string {
	return gce.networkURL
}

// SubnetworkURL returns the subnetwork url
func (gce *GCECloud) SubnetworkURL() string {
	return gce.subnetworkURL
}

// Known-useless DNS search path.
var uselessDNSSearchRE = regexp.MustCompile(`^[0-9]+.google.internal.$`)

// ScrubDNS filters DNS settings for pods.
func (gce *GCECloud) ScrubDNS(nameservers, searches []string) (nsOut, srchOut []string) {
	// GCE has too many search paths by default. Filter the ones we know are useless.
	for _, s := range searches {
		if !uselessDNSSearchRE.MatchString(s) {
			srchOut = append(srchOut, s)
		}
	}
	return nameservers, srchOut
}

// HasClusterID returns true if the cluster has a clusterID
func (gce *GCECloud) HasClusterID() bool {
	return true
}

// Project IDs cannot have a digit for the first characeter. If the id contains a digit,
// then it must be a project number.
func isProjectNumber(idOrNumber string) bool {
	_, err := strconv.ParseUint(idOrNumber, 10, 64)
	return err == nil
}

// GCECloud implements cloudprovider.Interface.
var _ cloudprovider.Interface = (*GCECloud)(nil)

func gceNetworkURL(apiEndpoint, project, network string) string {
	if apiEndpoint == "" {
		apiEndpoint = gceComputeAPIEndpoint
	}
	return apiEndpoint + strings.Join([]string{"projects", project, "global", "networks", network}, "/")
}

func gceSubnetworkURL(apiEndpoint, project, region, subnetwork string) string {
	if apiEndpoint == "" {
		apiEndpoint = gceComputeAPIEndpoint
	}
	return apiEndpoint + strings.Join([]string{"projects", project, "regions", region, "subnetworks", subnetwork}, "/")
}

// getProjectIDInURL parses typical full resource URLS and shorter URLS
// https://www.googleapis.com/compute/v1/projects/myproject/global/networks/mycustom
// projects/myproject/global/networks/mycustom
// All return "myproject"
func getProjectIDInURL(urlStr string) (string, error) {
	fields := strings.Split(urlStr, "/")
	for i, v := range fields {
		if v == "projects" && i < len(fields)-1 {
			return fields[i+1], nil
		}
	}
	return "", fmt.Errorf("could not find project field in url: %v", urlStr)
}

func getNetworkNameViaMetadata() (string, error) {
	result, err := metadata.Get("instance/network-interfaces/0/network")
	if err != nil {
		return "", err
	}
	parts := strings.Split(result, "/")
	if len(parts) != 4 {
		return "", fmt.Errorf("unexpected response: %s", result)
	}
	return parts[3], nil
}

func getNetworkNameViaAPICall(svc *compute.Service, projectID string) (string, error) {
	// TODO: use PageToken to list all not just the first 500
	networkList, err := svc.Networks.List(projectID).Do()
	if err != nil {
		return "", err
	}

	if networkList == nil || len(networkList.Items) <= 0 {
		return "", fmt.Errorf("GCE Network List call returned no networks for project %q", projectID)
	}

	return networkList.Items[0].Name, nil
}

// getProjectID returns the project's string ID given a project number or string
func getProjectID(svc *compute.Service, projectNumberOrID string) (string, error) {
	proj, err := svc.Projects.Get(projectNumberOrID).Do()
	if err != nil {
		return "", err
	}

	return proj.Name, nil
}

func getZonesForRegion(svc *compute.Service, projectID, region string) ([]string, error) {
	// TODO: use PageToken to list all not just the first 500
	listCall := svc.Zones.List(projectID)

	// Filtering by region doesn't seem to work
	// (tested in https://cloud.google.com/compute/docs/reference/latest/zones/list)
	// listCall = listCall.Filter("region eq " + region)

	res, err := listCall.Do()
	if err != nil {
		return nil, fmt.Errorf("unexpected response listing zones: %v", err)
	}
	zones := []string{}
	for _, zone := range res.Items {
		regionName := lastComponent(zone.Region)
		if regionName == region {
			zones = append(zones, zone.Name)
		}
	}
	return zones, nil
}

func newOauthClient(tokenSource oauth2.TokenSource) (*http.Client, error) {
	if tokenSource == nil {
		var err error
		tokenSource, err = google.DefaultTokenSource(
			oauth2.NoContext,
			compute.CloudPlatformScope,
			compute.ComputeScope)
		glog.Infof("Using DefaultTokenSource %#v", tokenSource)
		if err != nil {
			return nil, err
		}
	} else {
		glog.Infof("Using existing Token Source %#v", tokenSource)
	}

	if err := wait.PollImmediate(5*time.Second, 30*time.Second, func() (bool, error) {
		if _, err := tokenSource.Token(); err != nil {
			glog.Errorf("error fetching initial token: %v", err)
			return false, nil
		}
		return true, nil
	}); err != nil {
		return nil, err
	}

	return oauth2.NewClient(oauth2.NoContext, tokenSource), nil
}

func (manager *gceServiceManager) getProjectsAPIEndpoint() string {
	projectsApiEndpoint := gceComputeAPIEndpoint + "projects/"
	if manager.gce.service != nil {
		projectsApiEndpoint = manager.gce.service.BasePath
	}

	return projectsApiEndpoint
}

func (manager *gceServiceManager) getProjectsAPIEndpointAlpha() string {
	projectsApiEndpoint := gceComputeAPIEndpointAlpha + "projects/"
	if manager.gce.service != nil {
		projectsApiEndpoint = manager.gce.serviceAlpha.BasePath
	}

	return projectsApiEndpoint
}
