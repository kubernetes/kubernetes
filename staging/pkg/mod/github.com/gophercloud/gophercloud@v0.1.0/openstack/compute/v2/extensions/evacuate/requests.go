package evacuate

import (
	"github.com/gophercloud/gophercloud"
)

// EvacuateOptsBuilder allows extensions to add additional parameters to the
// the Evacuate request.
type EvacuateOptsBuilder interface {
	ToEvacuateMap() (map[string]interface{}, error)
}

// EvacuateOpts specifies Evacuate action parameters.
type EvacuateOpts struct {
	// The name of the host to which the server is evacuated
	Host string `json:"host,omitempty"`

	// Indicates whether server is on shared storage
	OnSharedStorage bool `json:"onSharedStorage"`

	// An administrative password to access the evacuated server
	AdminPass string `json:"adminPass,omitempty"`
}

// ToServerGroupCreateMap constructs a request body from CreateOpts.
func (opts EvacuateOpts) ToEvacuateMap() (map[string]interface{}, error) {
	return gophercloud.BuildRequestBody(opts, "evacuate")
}

// Evacuate will Evacuate a failed instance to another host.
func Evacuate(client *gophercloud.ServiceClient, id string, opts EvacuateOptsBuilder) (r EvacuateResult) {
	b, err := opts.ToEvacuateMap()
	if err != nil {
		r.Err = err
		return
	}
	_, r.Err = client.Post(actionURL(client, id), b, &r.Body, &gophercloud.RequestOpts{
		OkCodes: []int{200},
	})
	return
}
