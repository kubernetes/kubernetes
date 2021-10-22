package noauth

import (
	"fmt"

	"github.com/gophercloud/gophercloud"
)

// EndpointOpts specifies a "noauth" Ironic Inspector Endpoint.
type EndpointOpts struct {
	// IronicInspectorEndpoint [required] is currently only used with "noauth" Ironic introspection.
	// An Ironic inspector endpoint with "auth_strategy=noauth" is necessary, for example:
	// http://ironic.example.com:5050/v1.
	IronicInspectorEndpoint string
}

func initClientOpts(client *gophercloud.ProviderClient, eo EndpointOpts) (*gophercloud.ServiceClient, error) {
	sc := new(gophercloud.ServiceClient)
	if eo.IronicInspectorEndpoint == "" {
		return nil, fmt.Errorf("IronicInspectorEndpoint is required")
	}

	sc.Endpoint = gophercloud.NormalizeURL(eo.IronicInspectorEndpoint)
	sc.ProviderClient = client
	return sc, nil
}

// NewBareMetalIntrospectionNoAuth creates a ServiceClient that may be used to access a
// "noauth" bare metal introspection service.
func NewBareMetalIntrospectionNoAuth(eo EndpointOpts) (*gophercloud.ServiceClient, error) {
	sc, err := initClientOpts(&gophercloud.ProviderClient{}, eo)

	sc.Type = "baremetal-inspector"

	return sc, err
}
