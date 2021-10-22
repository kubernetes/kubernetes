// Package clients contains functions for creating OpenStack service clients
// for use in acceptance tests. It also manages the required environment
// variables to run the tests.
package clients

import (
	"fmt"
	"net/http"
	"os"
	"strings"

	"github.com/gophercloud/gophercloud"
	"github.com/gophercloud/gophercloud/openstack"
	baremetalNoAuth "github.com/gophercloud/gophercloud/openstack/baremetal/noauth"
	blockstorageNoAuth "github.com/gophercloud/gophercloud/openstack/blockstorage/noauth"
)

// AcceptanceTestChoices contains image and flavor selections for use by the acceptance tests.
type AcceptanceTestChoices struct {
	// ImageID contains the ID of a valid image.
	ImageID string

	// FlavorID contains the ID of a valid flavor.
	FlavorID string

	// FlavorIDResize contains the ID of a different flavor available on the same OpenStack installation, that is distinct
	// from FlavorID.
	FlavorIDResize string

	// FloatingIPPool contains the name of the pool from where to obtain floating IPs.
	FloatingIPPoolName string

	// MagnumKeypair contains the ID of a valid key pair.
	MagnumKeypair string

	// MagnumImageID contains the ID of a valid magnum image.
	MagnumImageID string

	// NetworkName is the name of a network to launch the instance on.
	NetworkName string

	// NetworkID is the ID of a network to launch the instance on.
	NetworkID string

	// SubnetID is the ID of a subnet to launch the instance on.
	SubnetID string

	// ExternalNetworkID is the network ID of the external network.
	ExternalNetworkID string

	// DBDatastoreType is the datastore type for DB tests.
	DBDatastoreType string

	// DBDatastoreTypeID is the datastore type version for DB tests.
	DBDatastoreVersion string
}

// AcceptanceTestChoicesFromEnv populates a ComputeChoices struct from environment variables.
// If any required state is missing, an `error` will be returned that enumerates the missing properties.
func AcceptanceTestChoicesFromEnv() (*AcceptanceTestChoices, error) {
	imageID := os.Getenv("OS_IMAGE_ID")
	flavorID := os.Getenv("OS_FLAVOR_ID")
	flavorIDResize := os.Getenv("OS_FLAVOR_ID_RESIZE")
	magnumImageID := os.Getenv("OS_MAGNUM_IMAGE_ID")
	magnumKeypair := os.Getenv("OS_MAGNUM_KEYPAIR")
	networkName := os.Getenv("OS_NETWORK_NAME")
	networkID := os.Getenv("OS_NETWORK_ID")
	subnetID := os.Getenv("OS_SUBNET_ID")
	floatingIPPoolName := os.Getenv("OS_POOL_NAME")
	externalNetworkID := os.Getenv("OS_EXTGW_ID")
	dbDatastoreType := os.Getenv("OS_DB_DATASTORE_TYPE")
	dbDatastoreVersion := os.Getenv("OS_DB_DATASTORE_VERSION")

	missing := make([]string, 0, 3)
	if imageID == "" {
		missing = append(missing, "OS_IMAGE_ID")
	}
	if flavorID == "" {
		missing = append(missing, "OS_FLAVOR_ID")
	}
	if flavorIDResize == "" {
		missing = append(missing, "OS_FLAVOR_ID_RESIZE")
	}
	if floatingIPPoolName == "" {
		missing = append(missing, "OS_POOL_NAME")
	}
	if externalNetworkID == "" {
		missing = append(missing, "OS_EXTGW_ID")
	}

	/* // Temporarily disabled, see https://github.com/gophercloud/gophercloud/issues/1345
	if networkID == "" {
		missing = append(missing, "OS_NETWORK_ID")
	}
	if subnetID == "" {
		missing = append(missing, "OS_SUBNET_ID")
	}
	*/

	if networkName == "" {
		networkName = "private"
	}
	notDistinct := ""
	if flavorID == flavorIDResize {
		notDistinct = "OS_FLAVOR_ID and OS_FLAVOR_ID_RESIZE must be distinct."
	}

	if len(missing) > 0 || notDistinct != "" {
		text := "You're missing some important setup:\n"
		if len(missing) > 0 {
			text += " * These environment variables must be provided: " + strings.Join(missing, ", ") + "\n"
		}
		if notDistinct != "" {
			text += " * " + notDistinct + "\n"
		}

		return nil, fmt.Errorf(text)
	}

	return &AcceptanceTestChoices{
		ImageID:            imageID,
		FlavorID:           flavorID,
		FlavorIDResize:     flavorIDResize,
		FloatingIPPoolName: floatingIPPoolName,
		MagnumImageID:      magnumImageID,
		MagnumKeypair:      magnumKeypair,
		NetworkName:        networkName,
		NetworkID:          networkID,
		SubnetID:           subnetID,
		ExternalNetworkID:  externalNetworkID,
		DBDatastoreType:    dbDatastoreType,
		DBDatastoreVersion: dbDatastoreVersion,
	}, nil
}

// NewBlockStorageV1Client returns a *ServiceClient for making calls
// to the OpenStack Block Storage v1 API. An error will be returned
// if authentication or client creation was not possible.
func NewBlockStorageV1Client() (*gophercloud.ServiceClient, error) {
	ao, err := openstack.AuthOptionsFromEnv()
	if err != nil {
		return nil, err
	}

	client, err := openstack.AuthenticatedClient(ao)
	if err != nil {
		return nil, err
	}

	client = configureDebug(client)

	return openstack.NewBlockStorageV1(client, gophercloud.EndpointOpts{
		Region: os.Getenv("OS_REGION_NAME"),
	})
}

// NewBlockStorageV2Client returns a *ServiceClient for making calls
// to the OpenStack Block Storage v2 API. An error will be returned
// if authentication or client creation was not possible.
func NewBlockStorageV2Client() (*gophercloud.ServiceClient, error) {
	ao, err := openstack.AuthOptionsFromEnv()
	if err != nil {
		return nil, err
	}

	client, err := openstack.AuthenticatedClient(ao)
	if err != nil {
		return nil, err
	}

	client = configureDebug(client)

	return openstack.NewBlockStorageV2(client, gophercloud.EndpointOpts{
		Region: os.Getenv("OS_REGION_NAME"),
	})
}

// NewBlockStorageV3Client returns a *ServiceClient for making calls
// to the OpenStack Block Storage v3 API. An error will be returned
// if authentication or client creation was not possible.
func NewBlockStorageV3Client() (*gophercloud.ServiceClient, error) {
	ao, err := openstack.AuthOptionsFromEnv()
	if err != nil {
		return nil, err
	}

	client, err := openstack.AuthenticatedClient(ao)
	if err != nil {
		return nil, err
	}

	client = configureDebug(client)

	return openstack.NewBlockStorageV3(client, gophercloud.EndpointOpts{
		Region: os.Getenv("OS_REGION_NAME"),
	})
}

// NewBlockStorageV2NoAuthClient returns a noauth *ServiceClient for
// making calls to the OpenStack Block Storage v2 API. An error will be
// returned if client creation was not possible.
func NewBlockStorageV2NoAuthClient() (*gophercloud.ServiceClient, error) {
	client, err := blockstorageNoAuth.NewClient(gophercloud.AuthOptions{
		Username:   os.Getenv("OS_USERNAME"),
		TenantName: os.Getenv("OS_TENANT_NAME"),
	})
	if err != nil {
		return nil, err
	}

	client = configureDebug(client)

	return blockstorageNoAuth.NewBlockStorageNoAuth(client, blockstorageNoAuth.EndpointOpts{
		CinderEndpoint: os.Getenv("CINDER_ENDPOINT"),
	})
}

// NewBlockStorageV3NoAuthClient returns a noauth *ServiceClient for
// making calls to the OpenStack Block Storage v2 API. An error will be
// returned if client creation was not possible.
func NewBlockStorageV3NoAuthClient() (*gophercloud.ServiceClient, error) {
	client, err := blockstorageNoAuth.NewClient(gophercloud.AuthOptions{
		Username:   os.Getenv("OS_USERNAME"),
		TenantName: os.Getenv("OS_TENANT_NAME"),
	})
	if err != nil {
		return nil, err
	}

	client = configureDebug(client)

	return blockstorageNoAuth.NewBlockStorageNoAuth(client, blockstorageNoAuth.EndpointOpts{
		CinderEndpoint: os.Getenv("CINDER_ENDPOINT"),
	})
}

// NewComputeV2Client returns a *ServiceClient for making calls
// to the OpenStack Compute v2 API. An error will be returned
// if authentication or client creation was not possible.
func NewComputeV2Client() (*gophercloud.ServiceClient, error) {
	ao, err := openstack.AuthOptionsFromEnv()
	if err != nil {
		return nil, err
	}

	client, err := openstack.AuthenticatedClient(ao)
	if err != nil {
		return nil, err
	}

	client = configureDebug(client)

	return openstack.NewComputeV2(client, gophercloud.EndpointOpts{
		Region: os.Getenv("OS_REGION_NAME"),
	})
}

// NewBareMetalV1Client returns a *ServiceClient for making calls
// to the OpenStack Bare Metal v1 API. An error will be returned
// if authentication or client creation was not possible.
func NewBareMetalV1Client() (*gophercloud.ServiceClient, error) {
	ao, err := openstack.AuthOptionsFromEnv()
	if err != nil {
		return nil, err
	}

	client, err := openstack.AuthenticatedClient(ao)
	if err != nil {
		return nil, err
	}

	client = configureDebug(client)

	return openstack.NewBareMetalV1(client, gophercloud.EndpointOpts{
		Region: os.Getenv("OS_REGION_NAME"),
	})
}

// NewBareMetalV1NoAuthClient returns a *ServiceClient for making calls
// to the OpenStack Bare Metal v1 API. An error will be returned
// if authentication or client creation was not possible.
func NewBareMetalV1NoAuthClient() (*gophercloud.ServiceClient, error) {
	return baremetalNoAuth.NewBareMetalNoAuth(baremetalNoAuth.EndpointOpts{
		IronicEndpoint: os.Getenv("IRONIC_ENDPOINT"),
	})
}

// NewBareMetalIntrospectionV1Client returns a *ServiceClient for making calls
// to the OpenStack Bare Metal Introspection v1 API. An error will be returned
// if authentication or client creation was not possible.
func NewBareMetalIntrospectionV1Client() (*gophercloud.ServiceClient, error) {
	ao, err := openstack.AuthOptionsFromEnv()
	if err != nil {
		return nil, err
	}

	client, err := openstack.AuthenticatedClient(ao)
	if err != nil {
		return nil, err
	}

	client = configureDebug(client)

	return openstack.NewBareMetalIntrospectionV1(client, gophercloud.EndpointOpts{
		Region: os.Getenv("OS_REGION_NAME"),
	})
}

// NewDBV1Client returns a *ServiceClient for making calls
// to the OpenStack Database v1 API. An error will be returned
// if authentication or client creation was not possible.
func NewDBV1Client() (*gophercloud.ServiceClient, error) {
	ao, err := openstack.AuthOptionsFromEnv()
	if err != nil {
		return nil, err
	}

	client, err := openstack.AuthenticatedClient(ao)
	if err != nil {
		return nil, err
	}

	client = configureDebug(client)

	return openstack.NewDBV1(client, gophercloud.EndpointOpts{
		Region: os.Getenv("OS_REGION_NAME"),
	})
}

// NewDNSV2Client returns a *ServiceClient for making calls
// to the OpenStack Compute v2 API. An error will be returned
// if authentication or client creation was not possible.
func NewDNSV2Client() (*gophercloud.ServiceClient, error) {
	ao, err := openstack.AuthOptionsFromEnv()
	if err != nil {
		return nil, err
	}

	client, err := openstack.AuthenticatedClient(ao)
	if err != nil {
		return nil, err
	}

	client = configureDebug(client)

	return openstack.NewDNSV2(client, gophercloud.EndpointOpts{
		Region: os.Getenv("OS_REGION_NAME"),
	})
}

// NewIdentityV2Client returns a *ServiceClient for making calls
// to the OpenStack Identity v2 API. An error will be returned
// if authentication or client creation was not possible.
func NewIdentityV2Client() (*gophercloud.ServiceClient, error) {
	ao, err := openstack.AuthOptionsFromEnv()
	if err != nil {
		return nil, err
	}

	client, err := openstack.AuthenticatedClient(ao)
	if err != nil {
		return nil, err
	}

	client = configureDebug(client)

	return openstack.NewIdentityV2(client, gophercloud.EndpointOpts{
		Region: os.Getenv("OS_REGION_NAME"),
	})
}

// NewIdentityV2AdminClient returns a *ServiceClient for making calls
// to the Admin Endpoint of the OpenStack Identity v2 API. An error
// will be returned if authentication or client creation was not possible.
func NewIdentityV2AdminClient() (*gophercloud.ServiceClient, error) {
	ao, err := openstack.AuthOptionsFromEnv()
	if err != nil {
		return nil, err
	}

	client, err := openstack.AuthenticatedClient(ao)
	if err != nil {
		return nil, err
	}

	client = configureDebug(client)

	return openstack.NewIdentityV2(client, gophercloud.EndpointOpts{
		Region:       os.Getenv("OS_REGION_NAME"),
		Availability: gophercloud.AvailabilityAdmin,
	})
}

// NewIdentityV2UnauthenticatedClient returns an unauthenticated *ServiceClient
// for the OpenStack Identity v2 API. An error  will be returned if
// authentication or client creation was not possible.
func NewIdentityV2UnauthenticatedClient() (*gophercloud.ServiceClient, error) {
	ao, err := openstack.AuthOptionsFromEnv()
	if err != nil {
		return nil, err
	}

	client, err := openstack.NewClient(ao.IdentityEndpoint)
	if err != nil {
		return nil, err
	}

	client = configureDebug(client)

	return openstack.NewIdentityV2(client, gophercloud.EndpointOpts{})
}

// NewIdentityV3Client returns a *ServiceClient for making calls
// to the OpenStack Identity v3 API. An error will be returned
// if authentication or client creation was not possible.
func NewIdentityV3Client() (*gophercloud.ServiceClient, error) {
	ao, err := openstack.AuthOptionsFromEnv()
	if err != nil {
		return nil, err
	}

	client, err := openstack.AuthenticatedClient(ao)
	if err != nil {
		return nil, err
	}

	client = configureDebug(client)

	return openstack.NewIdentityV3(client, gophercloud.EndpointOpts{
		Region: os.Getenv("OS_REGION_NAME"),
	})
}

// NewIdentityV3UnauthenticatedClient returns an unauthenticated *ServiceClient
// for the OpenStack Identity v3 API. An error  will be returned if
// authentication or client creation was not possible.
func NewIdentityV3UnauthenticatedClient() (*gophercloud.ServiceClient, error) {
	ao, err := openstack.AuthOptionsFromEnv()
	if err != nil {
		return nil, err
	}

	client, err := openstack.NewClient(ao.IdentityEndpoint)
	if err != nil {
		return nil, err
	}

	client = configureDebug(client)

	return openstack.NewIdentityV3(client, gophercloud.EndpointOpts{})
}

// NewImageServiceV2Client returns a *ServiceClient for making calls to the
// OpenStack Image v2 API. An error will be returned if authentication or
// client creation was not possible.
func NewImageServiceV2Client() (*gophercloud.ServiceClient, error) {
	ao, err := openstack.AuthOptionsFromEnv()
	if err != nil {
		return nil, err
	}

	client, err := openstack.AuthenticatedClient(ao)
	if err != nil {
		return nil, err
	}

	client = configureDebug(client)

	return openstack.NewImageServiceV2(client, gophercloud.EndpointOpts{
		Region: os.Getenv("OS_REGION_NAME"),
	})
}

// NewNetworkV2Client returns a *ServiceClient for making calls to the
// OpenStack Networking v2 API. An error will be returned if authentication
// or client creation was not possible.
func NewNetworkV2Client() (*gophercloud.ServiceClient, error) {
	ao, err := openstack.AuthOptionsFromEnv()
	if err != nil {
		return nil, err
	}

	client, err := openstack.AuthenticatedClient(ao)
	if err != nil {
		return nil, err
	}

	client = configureDebug(client)

	return openstack.NewNetworkV2(client, gophercloud.EndpointOpts{
		Region: os.Getenv("OS_REGION_NAME"),
	})
}

// NewObjectStorageV1Client returns a *ServiceClient for making calls to the
// OpenStack Object Storage v1 API. An error will be returned if authentication
// or client creation was not possible.
func NewObjectStorageV1Client() (*gophercloud.ServiceClient, error) {
	ao, err := openstack.AuthOptionsFromEnv()
	if err != nil {
		return nil, err
	}

	client, err := openstack.AuthenticatedClient(ao)
	if err != nil {
		return nil, err
	}

	client = configureDebug(client)

	return openstack.NewObjectStorageV1(client, gophercloud.EndpointOpts{
		Region: os.Getenv("OS_REGION_NAME"),
	})
}

// NewSharedFileSystemV2Client returns a *ServiceClient for making calls
// to the OpenStack Shared File System v2 API. An error will be returned
// if authentication or client creation was not possible.
func NewSharedFileSystemV2Client() (*gophercloud.ServiceClient, error) {
	ao, err := openstack.AuthOptionsFromEnv()
	if err != nil {
		return nil, err
	}

	client, err := openstack.AuthenticatedClient(ao)
	if err != nil {
		return nil, err
	}

	client = configureDebug(client)

	return openstack.NewSharedFileSystemV2(client, gophercloud.EndpointOpts{
		Region: os.Getenv("OS_REGION_NAME"),
	})
}

// NewLoadBalancerV2Client returns a *ServiceClient for making calls to the
// OpenStack Octavia v2 API. An error will be returned if authentication
// or client creation was not possible.
func NewLoadBalancerV2Client() (*gophercloud.ServiceClient, error) {
	ao, err := openstack.AuthOptionsFromEnv()
	if err != nil {
		return nil, err
	}

	client, err := openstack.AuthenticatedClient(ao)
	if err != nil {
		return nil, err
	}

	client = configureDebug(client)

	return openstack.NewLoadBalancerV2(client, gophercloud.EndpointOpts{
		Region: os.Getenv("OS_REGION_NAME"),
	})
}

// NewClusteringV1Client returns a *ServiceClient for making calls
// to the OpenStack Clustering v1 API. An error will be returned
// if authentication or client creation was not possible.
func NewClusteringV1Client() (*gophercloud.ServiceClient, error) {
	ao, err := openstack.AuthOptionsFromEnv()
	if err != nil {
		return nil, err
	}

	client, err := openstack.AuthenticatedClient(ao)
	if err != nil {
		return nil, err
	}

	client = configureDebug(client)

	return openstack.NewClusteringV1(client, gophercloud.EndpointOpts{
		Region: os.Getenv("OS_REGION_NAME"),
	})
}

// NewMessagingV2Client returns a *ServiceClient for making calls
// to the OpenStack Messaging (Zaqar) v2 API. An error will be returned
// if authentication or client creation was not possible.
func NewMessagingV2Client(clientID string) (*gophercloud.ServiceClient, error) {
	ao, err := openstack.AuthOptionsFromEnv()
	if err != nil {
		return nil, err
	}

	client, err := openstack.AuthenticatedClient(ao)
	if err != nil {
		return nil, err
	}

	client = configureDebug(client)

	return openstack.NewMessagingV2(client, clientID, gophercloud.EndpointOpts{
		Region: os.Getenv("OS_REGION_NAME"),
	})
}

// NewContainerV1Client returns a *ServiceClient for making calls
// to the OpenStack Container V1 API. An error will be returned
// if authentication or client creation was not possible.
func NewContainerV1Client() (*gophercloud.ServiceClient, error) {
	ao, err := openstack.AuthOptionsFromEnv()
	if err != nil {
		return nil, err
	}

	client, err := openstack.AuthenticatedClient(ao)
	if err != nil {
		return nil, err
	}

	client = configureDebug(client)

	return openstack.NewContainerV1(client, gophercloud.EndpointOpts{
		Region: os.Getenv("OS_REGION_NAME"),
	})
}

// NewKeyManagerV1Client returns a *ServiceClient for making calls
// to the OpenStack Key Manager (Barbican) v1 API. An error will be
// returned if authentication or client creation was not possible.
func NewKeyManagerV1Client() (*gophercloud.ServiceClient, error) {
	ao, err := openstack.AuthOptionsFromEnv()
	if err != nil {
		return nil, err
	}

	client, err := openstack.AuthenticatedClient(ao)
	if err != nil {
		return nil, err
	}

	client = configureDebug(client)

	return openstack.NewKeyManagerV1(client, gophercloud.EndpointOpts{
		Region: os.Getenv("OS_REGION_NAME"),
	})
}

// configureDebug will configure the provider client to print the API
// requests and responses if OS_DEBUG is enabled.
func configureDebug(client *gophercloud.ProviderClient) *gophercloud.ProviderClient {
	if os.Getenv("OS_DEBUG") != "" {
		client.HTTPClient = http.Client{
			Transport: &LogRoundTripper{
				Rt: &http.Transport{},
			},
		}
	}

	return client
}

// NewContainerInfraV1Client returns a *ServiceClient for making calls
// to the OpenStack Container Infra Management v1 API. An error will be returned
// if authentication or client creation was not possible.
func NewContainerInfraV1Client() (*gophercloud.ServiceClient, error) {
	ao, err := openstack.AuthOptionsFromEnv()
	if err != nil {
		return nil, err
	}

	client, err := openstack.AuthenticatedClient(ao)
	if err != nil {
		return nil, err
	}

	client = configureDebug(client)

	return openstack.NewContainerInfraV1(client, gophercloud.EndpointOpts{
		Region: os.Getenv("OS_REGION_NAME"),
	})
}

// NewWorkflowV2Client returns a *ServiceClient for making calls
// to the OpenStack Workflow v2 API (Mistral). An error will be returned if
// authentication or client creation failed.
func NewWorkflowV2Client() (*gophercloud.ServiceClient, error) {
	ao, err := openstack.AuthOptionsFromEnv()
	if err != nil {
		return nil, err
	}

	client, err := openstack.AuthenticatedClient(ao)
	if err != nil {
		return nil, err
	}

	client = configureDebug(client)

	return openstack.NewWorkflowV2(client, gophercloud.EndpointOpts{
		Region: os.Getenv("OS_REGION_NAME"),
	})
}

// NewOrchestrationV1Client returns a *ServiceClient for making calls
// to the OpenStack Orchestration v1 API. An error will be returned
// if authentication or client creation was not possible.
func NewOrchestrationV1Client() (*gophercloud.ServiceClient, error) {
	ao, err := openstack.AuthOptionsFromEnv()
	if err != nil {
		return nil, err
	}

	client, err := openstack.AuthenticatedClient(ao)
	if err != nil {
		return nil, err
	}

	client = configureDebug(client)

	return openstack.NewOrchestrationV1(client, gophercloud.EndpointOpts{
		Region: os.Getenv("OS_REGION_NAME"),
	})
}
