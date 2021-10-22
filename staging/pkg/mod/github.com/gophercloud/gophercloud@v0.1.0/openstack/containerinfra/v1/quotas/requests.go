package quotas

import (
	"net/http"

	"github.com/gophercloud/gophercloud"
)

// CreateOptsBuilder Builder.
type CreateOptsBuilder interface {
	ToQuotaCreateMap() (map[string]interface{}, error)
}

// CreateOpts params
type CreateOpts struct {
	ProjectID string `json:"project_id"`
	Resource  string `json:"resource"`
	HardLimit int    `json:"hard_limit"`
}

// ToQuotaCreateMap constructs a request body from CreateOpts.
func (opts CreateOpts) ToQuotaCreateMap() (map[string]interface{}, error) {
	return gophercloud.BuildRequestBody(opts, "")
}

// Create requests the creation of a new quota.
func Create(client *gophercloud.ServiceClient, opts CreateOptsBuilder) (r CreateResult) {
	b, err := opts.ToQuotaCreateMap()
	if err != nil {
		r.Err = err
		return
	}
	var result *http.Response
	result, r.Err = client.Post(createURL(client), b, &r.Body, &gophercloud.RequestOpts{
		OkCodes: []int{201},
	})

	if r.Err == nil {
		r.Header = result.Header
	}

	return
}
