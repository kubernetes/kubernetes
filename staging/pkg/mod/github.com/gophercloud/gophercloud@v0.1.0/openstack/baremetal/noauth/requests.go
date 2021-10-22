package noauth

import (
	"fmt"

	"github.com/gophercloud/gophercloud"
)

// EndpointOpts specifies a "noauth" Ironic Endpoint.
type EndpointOpts struct {
	// IronicEndpoint [required] is currently only used with "noauth" Ironic.
	// An Ironic endpoint with "auth_strategy=noauth" is necessary, for example:
	// http://ironic.example.com:6385/v1.
	IronicEndpoint string
}

func initClientOpts(client *gophercloud.ProviderClient, eo EndpointOpts) (*gophercloud.ServiceClient, error) {
	sc := new(gophercloud.ServiceClient)
	if eo.IronicEndpoint == "" {
		return nil, fmt.Errorf("IronicEndpoint is required")
	}

	sc.Endpoint = gophercloud.NormalizeURL(eo.IronicEndpoint)
	sc.ProviderClient = client
	return sc, nil
}

// NewBareMetalNoAuth creates a ServiceClient that may be used to access a
// "noauth" bare metal service.
func NewBareMetalNoAuth(eo EndpointOpts) (*gophercloud.ServiceClient, error) {
	sc, err := initClientOpts(&gophercloud.ProviderClient{}, eo)

	sc.Type = "baremetal"

	return sc, err
}
