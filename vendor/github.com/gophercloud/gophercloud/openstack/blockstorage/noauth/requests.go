package noauth

import (
	"fmt"
	"strings"

	"github.com/gophercloud/gophercloud"
)

// EndpointOpts specifies a "noauth" Cinder Endpoint.
type EndpointOpts struct {
	// CinderEndpoint [required] is currently only used with "noauth" Cinder.
	// A cinder endpoint with "auth_strategy=noauth" is necessary, for example:
	// http://example.com:8776/v2.
	CinderEndpoint string
}

// NewClient prepares an unauthenticated ProviderClient instance.
func NewClient(options gophercloud.AuthOptions) (*gophercloud.ProviderClient, error) {
	if options.Username == "" {
		options.Username = "admin"
	}
	if options.TenantName == "" {
		options.TenantName = "admin"
	}

	client := &gophercloud.ProviderClient{
		TokenID: fmt.Sprintf("%s:%s", options.Username, options.TenantName),
	}

	return client, nil
}

func initClientOpts(client *gophercloud.ProviderClient, eo EndpointOpts) (*gophercloud.ServiceClient, error) {
	sc := new(gophercloud.ServiceClient)
	if eo.CinderEndpoint == "" {
		return nil, fmt.Errorf("CinderEndpoint is required")
	}

	token := strings.Split(client.TokenID, ":")
	if len(token) != 2 {
		return nil, fmt.Errorf("Malformed noauth token")
	}

	endpoint := fmt.Sprintf("%s%s", gophercloud.NormalizeURL(eo.CinderEndpoint), token[1])
	sc.Endpoint = gophercloud.NormalizeURL(endpoint)
	sc.ProviderClient = client
	return sc, nil
}

// NewBlockStorageNoAuth creates a ServiceClient that may be used to access a
// "noauth" block storage service (V2 or V3 Cinder API).
func NewBlockStorageNoAuth(client *gophercloud.ProviderClient, eo EndpointOpts) (*gophercloud.ServiceClient, error) {
	return initClientOpts(client, eo)
}
