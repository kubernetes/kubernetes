package servers

import (
	"github.com/mitchellh/mapstructure"
	"github.com/rackspace/gophercloud"
	"github.com/rackspace/gophercloud/pagination"
)

type serverResult struct {
	gophercloud.Result
}

// Extract interprets any serverResult as a Server, if possible.
func (r serverResult) Extract() (*Server, error) {
	if r.Err != nil {
		return nil, r.Err
	}

	var response struct {
		Server Server `mapstructure:"server"`
	}

	err := mapstructure.Decode(r.Body, &response)
	return &response.Server, err
}

// CreateResult temporarily contains the response from a Create call.
type CreateResult struct {
	serverResult
}

// GetResult temporarily contains the response from a Get call.
type GetResult struct {
	serverResult
}

// UpdateResult temporarily contains the response from an Update call.
type UpdateResult struct {
	serverResult
}

// DeleteResult temporarily contains the response from an Delete call.
type DeleteResult struct {
	gophercloud.ErrResult
}

// RebuildResult temporarily contains the response from a Rebuild call.
type RebuildResult struct {
	serverResult
}

// ActionResult represents the result of server action operations, like reboot
type ActionResult struct {
	gophercloud.ErrResult
}

// Server exposes only the standard OpenStack fields corresponding to a given server on the user's account.
type Server struct {
	// ID uniquely identifies this server amongst all other servers, including those not accessible to the current tenant.
	ID string

	// TenantID identifies the tenant owning this server resource.
	TenantID string `mapstructure:"tenant_id"`

	// UserID uniquely identifies the user account owning the tenant.
	UserID string `mapstructure:"user_id"`

	// Name contains the human-readable name for the server.
	Name string

	// Updated and Created contain ISO-8601 timestamps of when the state of the server last changed, and when it was created.
	Updated string
	Created string

	HostID string

	// Status contains the current operational status of the server, such as IN_PROGRESS or ACTIVE.
	Status string

	// Progress ranges from 0..100.
	// A request made against the server completes only once Progress reaches 100.
	Progress int

	// AccessIPv4 and AccessIPv6 contain the IP addresses of the server, suitable for remote access for administration.
	AccessIPv4, AccessIPv6 string

	// Image refers to a JSON object, which itself indicates the OS image used to deploy the server.
	Image map[string]interface{}

	// Flavor refers to a JSON object, which itself indicates the hardware configuration of the deployed server.
	Flavor map[string]interface{}

	// Addresses includes a list of all IP addresses assigned to the server, keyed by pool.
	Addresses map[string]interface{}

	// Metadata includes a list of all user-specified key-value pairs attached to the server.
	Metadata map[string]interface{}

	// Links includes HTTP references to the itself, useful for passing along to other APIs that might want a server reference.
	Links []interface{}

	// KeyName indicates which public key was injected into the server on launch.
	KeyName string `json:"key_name" mapstructure:"key_name"`

	// AdminPass will generally be empty ("").  However, it will contain the administrative password chosen when provisioning a new server without a set AdminPass setting in the first place.
	// Note that this is the ONLY time this field will be valid.
	AdminPass string `json:"adminPass" mapstructure:"adminPass"`
}

// ServerPage abstracts the raw results of making a List() request against the API.
// As OpenStack extensions may freely alter the response bodies of structures returned to the client, you may only safely access the
// data provided through the ExtractServers call.
type ServerPage struct {
	pagination.LinkedPageBase
}

// IsEmpty returns true if a page contains no Server results.
func (page ServerPage) IsEmpty() (bool, error) {
	servers, err := ExtractServers(page)
	if err != nil {
		return true, err
	}
	return len(servers) == 0, nil
}

// NextPageURL uses the response's embedded link reference to navigate to the next page of results.
func (page ServerPage) NextPageURL() (string, error) {
	type resp struct {
		Links []gophercloud.Link `mapstructure:"servers_links"`
	}

	var r resp
	err := mapstructure.Decode(page.Body, &r)
	if err != nil {
		return "", err
	}

	return gophercloud.ExtractNextURL(r.Links)
}

// ExtractServers interprets the results of a single page from a List() call, producing a slice of Server entities.
func ExtractServers(page pagination.Page) ([]Server, error) {
	casted := page.(ServerPage).Body

	var response struct {
		Servers []Server `mapstructure:"servers"`
	}
	err := mapstructure.Decode(casted, &response)
	return response.Servers, err
}
