package rackspace

import (
	"fmt"

	"github.com/rackspace/gophercloud"
	os "github.com/rackspace/gophercloud/openstack"
	"github.com/rackspace/gophercloud/openstack/utils"
	tokens2 "github.com/rackspace/gophercloud/rackspace/identity/v2/tokens"
)

const (
	// RackspaceUSIdentity is an identity endpoint located in the United States.
	RackspaceUSIdentity = "https://identity.api.rackspacecloud.com/v2.0/"

	// RackspaceUKIdentity is an identity endpoint located in the UK.
	RackspaceUKIdentity = "https://lon.identity.api.rackspacecloud.com/v2.0/"
)

const (
	v20 = "v2.0"
)

// NewClient creates a client that's prepared to communicate with the Rackspace API, but is not
// yet authenticated. Most users will probably prefer using the AuthenticatedClient function
// instead.
//
// Provide the base URL of the identity endpoint you wish to authenticate against as "endpoint".
// Often, this will be either RackspaceUSIdentity or RackspaceUKIdentity.
func NewClient(endpoint string) (*gophercloud.ProviderClient, error) {
	if endpoint == "" {
		return os.NewClient(RackspaceUSIdentity)
	}
	return os.NewClient(endpoint)
}

// AuthenticatedClient logs in to Rackspace with the provided credentials and constructs a
// ProviderClient that's ready to operate.
//
// If the provided AuthOptions does not specify an explicit IdentityEndpoint, it will default to
// the canonical, production Rackspace US identity endpoint.
func AuthenticatedClient(options gophercloud.AuthOptions) (*gophercloud.ProviderClient, error) {
	client, err := NewClient(options.IdentityEndpoint)
	if err != nil {
		return nil, err
	}

	err = Authenticate(client, options)
	if err != nil {
		return nil, err
	}
	return client, nil
}

// Authenticate or re-authenticate against the most recent identity service supported at the
// provided endpoint.
func Authenticate(client *gophercloud.ProviderClient, options gophercloud.AuthOptions) error {
	versions := []*utils.Version{
		&utils.Version{ID: v20, Priority: 20, Suffix: "/v2.0/"},
	}

	chosen, endpoint, err := utils.ChooseVersion(client, versions)
	if err != nil {
		return err
	}

	switch chosen.ID {
	case v20:
		return v2auth(client, endpoint, options)
	default:
		// The switch statement must be out of date from the versions list.
		return fmt.Errorf("Unrecognized identity version: %s", chosen.ID)
	}
}

// AuthenticateV2 explicitly authenticates with v2 of the identity service.
func AuthenticateV2(client *gophercloud.ProviderClient, options gophercloud.AuthOptions) error {
	return v2auth(client, "", options)
}

func v2auth(client *gophercloud.ProviderClient, endpoint string, options gophercloud.AuthOptions) error {
	v2Client := NewIdentityV2(client)
	if endpoint != "" {
		v2Client.Endpoint = endpoint
	}

	result := tokens2.Create(v2Client, tokens2.WrapOptions(options))

	token, err := result.ExtractToken()
	if err != nil {
		return err
	}

	catalog, err := result.ExtractServiceCatalog()
	if err != nil {
		return err
	}

	if options.AllowReauth {
		client.ReauthFunc = func() error {
			return AuthenticateV2(client, options)
		}
	}
	client.TokenID = token.ID
	client.EndpointLocator = func(opts gophercloud.EndpointOpts) (string, error) {
		return os.V2EndpointURL(catalog, opts)
	}

	return nil
}

// NewIdentityV2 creates a ServiceClient that may be used to access the v2 identity service.
func NewIdentityV2(client *gophercloud.ProviderClient) *gophercloud.ServiceClient {
	v2Endpoint := client.IdentityBase + "v2.0/"

	return &gophercloud.ServiceClient{
		ProviderClient: client,
		Endpoint:       v2Endpoint,
	}
}

// NewComputeV2 creates a ServiceClient that may be used to access the v2 compute service.
func NewComputeV2(client *gophercloud.ProviderClient, eo gophercloud.EndpointOpts) (*gophercloud.ServiceClient, error) {
	eo.ApplyDefaults("compute")
	url, err := client.EndpointLocator(eo)
	if err != nil {
		return nil, err
	}

	return &gophercloud.ServiceClient{
		ProviderClient: client,
		Endpoint:       url,
	}, nil
}

// NewObjectCDNV1 creates a ServiceClient that may be used with the Rackspace v1 CDN.
func NewObjectCDNV1(client *gophercloud.ProviderClient, eo gophercloud.EndpointOpts) (*gophercloud.ServiceClient, error) {
	eo.ApplyDefaults("rax:object-cdn")
	url, err := client.EndpointLocator(eo)
	if err != nil {
		return nil, err
	}
	return &gophercloud.ServiceClient{ProviderClient: client, Endpoint: url}, nil
}

// NewObjectStorageV1 creates a ServiceClient that may be used with the Rackspace v1 object storage package.
func NewObjectStorageV1(client *gophercloud.ProviderClient, eo gophercloud.EndpointOpts) (*gophercloud.ServiceClient, error) {
	return os.NewObjectStorageV1(client, eo)
}

// NewBlockStorageV1 creates a ServiceClient that can be used to access the
// Rackspace Cloud Block Storage v1 API.
func NewBlockStorageV1(client *gophercloud.ProviderClient, eo gophercloud.EndpointOpts) (*gophercloud.ServiceClient, error) {
	eo.ApplyDefaults("volume")
	url, err := client.EndpointLocator(eo)
	if err != nil {
		return nil, err
	}

	return &gophercloud.ServiceClient{ProviderClient: client, Endpoint: url}, nil
}

// NewLBV1 creates a ServiceClient that can be used to access the Rackspace
// Cloud Load Balancer v1 API.
func NewLBV1(client *gophercloud.ProviderClient, eo gophercloud.EndpointOpts) (*gophercloud.ServiceClient, error) {
	eo.ApplyDefaults("rax:load-balancer")
	url, err := client.EndpointLocator(eo)
	if err != nil {
		return nil, err
	}
	return &gophercloud.ServiceClient{ProviderClient: client, Endpoint: url}, nil
}

// NewNetworkV2 creates a ServiceClient that can be used to access the Rackspace
// Networking v2 API.
func NewNetworkV2(client *gophercloud.ProviderClient, eo gophercloud.EndpointOpts) (*gophercloud.ServiceClient, error) {
	eo.ApplyDefaults("network")
	url, err := client.EndpointLocator(eo)
	if err != nil {
		return nil, err
	}
	return &gophercloud.ServiceClient{ProviderClient: client, Endpoint: url}, nil
}

// NewCDNV1 creates a ServiceClient that may be used to access the Rackspace v1
// CDN service.
func NewCDNV1(client *gophercloud.ProviderClient, eo gophercloud.EndpointOpts) (*gophercloud.ServiceClient, error) {
	eo.ApplyDefaults("rax:cdn")
	url, err := client.EndpointLocator(eo)
	if err != nil {
		return nil, err
	}
	return &gophercloud.ServiceClient{ProviderClient: client, Endpoint: url}, nil
}

// NewOrchestrationV1 creates a ServiceClient that may be used to access the v1 orchestration service.
func NewOrchestrationV1(client *gophercloud.ProviderClient, eo gophercloud.EndpointOpts) (*gophercloud.ServiceClient, error) {
	eo.ApplyDefaults("orchestration")
	url, err := client.EndpointLocator(eo)
	if err != nil {
		return nil, err
	}
	return &gophercloud.ServiceClient{ProviderClient: client, Endpoint: url}, nil
}

// NewRackConnectV3 creates a ServiceClient that may be used to access the v3 RackConnect service.
func NewRackConnectV3(client *gophercloud.ProviderClient, eo gophercloud.EndpointOpts) (*gophercloud.ServiceClient, error) {
	eo.ApplyDefaults("rax:rackconnect")
	url, err := client.EndpointLocator(eo)
	if err != nil {
		return nil, err
	}
	return &gophercloud.ServiceClient{ProviderClient: client, Endpoint: url}, nil
}

// NewDBV1 creates a ServiceClient that may be used to access the v1 DB service.
func NewDBV1(client *gophercloud.ProviderClient, eo gophercloud.EndpointOpts) (*gophercloud.ServiceClient, error) {
	eo.ApplyDefaults("rax:database")
	url, err := client.EndpointLocator(eo)
	if err != nil {
		return nil, err
	}
	return &gophercloud.ServiceClient{ProviderClient: client, Endpoint: url}, nil
}

// NewAutoScaleV1 creates a ServiceClient that may be used to access the v1 Auto Scale service.
func NewAutoScaleV1(client *gophercloud.ProviderClient, eo gophercloud.EndpointOpts) (*gophercloud.ServiceClient, error) {
	eo.ApplyDefaults("rax:autoscale")
	url, err := client.EndpointLocator(eo)
	if err != nil {
		return nil, err
	}
	return &gophercloud.ServiceClient{ProviderClient: client, Endpoint: url}, nil
}
