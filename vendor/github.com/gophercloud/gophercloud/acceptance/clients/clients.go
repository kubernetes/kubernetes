// Package clients contains functions for creating OpenStack service clients
// for use in acceptance tests. It also manages the required environment
// variables to run the tests.
package clients

import (
	"fmt"
	"os"
	"strings"

	"github.com/gophercloud/gophercloud"
	"github.com/gophercloud/gophercloud/openstack"
	"github.com/gophercloud/gophercloud/openstack/blockstorage/noauth"
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

	// NetworkName is the name of a network to launch the instance on.
	NetworkName string

	// ExternalNetworkID is the network ID of the external network.
	ExternalNetworkID string

	// ShareNetworkID is the Manila Share network ID
	ShareNetworkID string

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
	networkName := os.Getenv("OS_NETWORK_NAME")
	floatingIPPoolName := os.Getenv("OS_POOL_NAME")
	externalNetworkID := os.Getenv("OS_EXTGW_ID")
	shareNetworkID := os.Getenv("OS_SHARE_NETWORK_ID")
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
	if networkName == "" {
		networkName = "private"
	}
	if shareNetworkID == "" {
		missing = append(missing, "OS_SHARE_NETWORK_ID")
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
		NetworkName:        networkName,
		ExternalNetworkID:  externalNetworkID,
		ShareNetworkID:     shareNetworkID,
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

	return openstack.NewBlockStorageV3(client, gophercloud.EndpointOpts{
		Region: os.Getenv("OS_REGION_NAME"),
	})
}

// NewBlockStorageV2NoAuthClient returns a noauth *ServiceClient for
// making calls to the OpenStack Block Storage v2 API. An error will be
// returned if client creation was not possible.
func NewBlockStorageV2NoAuthClient() (*gophercloud.ServiceClient, error) {
	client, err := noauth.NewClient(gophercloud.AuthOptions{
		Username:   os.Getenv("OS_USERNAME"),
		TenantName: os.Getenv("OS_TENANT_NAME"),
	})
	if err != nil {
		return nil, err
	}

	return noauth.NewBlockStorageNoAuth(client, noauth.EndpointOpts{
		CinderEndpoint: os.Getenv("CINDER_ENDPOINT"),
	})
}

// NewBlockStorageV3NoAuthClient returns a noauth *ServiceClient for
// making calls to the OpenStack Block Storage v2 API. An error will be
// returned if client creation was not possible.
func NewBlockStorageV3NoAuthClient() (*gophercloud.ServiceClient, error) {
	client, err := noauth.NewClient(gophercloud.AuthOptions{
		Username:   os.Getenv("OS_USERNAME"),
		TenantName: os.Getenv("OS_TENANT_NAME"),
	})
	if err != nil {
		return nil, err
	}

	return noauth.NewBlockStorageNoAuth(client, noauth.EndpointOpts{
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

	return openstack.NewComputeV2(client, gophercloud.EndpointOpts{
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

	return openstack.NewSharedFileSystemV2(client, gophercloud.EndpointOpts{
		Region: os.Getenv("OS_REGION_NAME"),
	})
}
