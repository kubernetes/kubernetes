package rescueunrescue

import "github.com/gophercloud/gophercloud"

// RescueOptsBuilder is an interface that allows extensions to override the
// default structure of a Rescue request.
type RescueOptsBuilder interface {
	ToServerRescueMap() (map[string]interface{}, error)
}

// RescueOpts represents the configuration options used to control a Rescue
// option.
type RescueOpts struct {
	// AdminPass is the desired administrative password for the instance in
	// RESCUE mode.
	// If it's left blank, the server will generate a password.
	AdminPass string `json:"adminPass,omitempty"`

	// RescueImageRef contains reference on an image that needs to be used as
	// rescue image.
	// If it's left blank, the server will be rescued with the default image.
	RescueImageRef string `json:"rescue_image_ref,omitempty"`
}

// ToServerRescueMap formats a RescueOpts as a map that can be used as a JSON
// request body for the Rescue request.
func (opts RescueOpts) ToServerRescueMap() (map[string]interface{}, error) {
	return gophercloud.BuildRequestBody(opts, "rescue")
}

// Rescue instructs the provider to place the server into RESCUE mode.
func Rescue(client *gophercloud.ServiceClient, id string, opts RescueOptsBuilder) (r RescueResult) {
	b, err := opts.ToServerRescueMap()
	if err != nil {
		r.Err = err
		return
	}
	_, r.Err = client.Post(actionURL(client, id), b, &r.Body, &gophercloud.RequestOpts{
		OkCodes: []int{200},
	})
	return
}

// Unrescue instructs the provider to return the server from RESCUE mode.
func Unrescue(client *gophercloud.ServiceClient, id string) (r UnrescueResult) {
	_, r.Err = client.Post(actionURL(client, id), map[string]interface{}{"unrescue": nil}, nil, nil)
	return
}
