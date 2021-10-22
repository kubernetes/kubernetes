package imageimport

import "github.com/gophercloud/gophercloud"

// ImportMethod represents valid Import API method.
type ImportMethod string

const (
	// GlanceDirectMethod represents glance-direct Import API method.
	GlanceDirectMethod ImportMethod = "glance-direct"

	// WebDownloadMethod represents web-download Import API method.
	WebDownloadMethod ImportMethod = "web-download"
)

// Get retrieves Import API information data.
func Get(c *gophercloud.ServiceClient) (r GetResult) {
	_, r.Err = c.Get(infoURL(c), &r.Body, nil)
	return
}

// CreateOptsBuilder allows to add additional parameters to the Create request.
type CreateOptsBuilder interface {
	ToImportCreateMap() (map[string]interface{}, error)
}

// CreateOpts specifies parameters of a new image import.
type CreateOpts struct {
	Name ImportMethod `json:"name"`
	URI  string       `json:"uri"`
}

// ToImportCreateMap constructs a request body from CreateOpts.
func (opts CreateOpts) ToImportCreateMap() (map[string]interface{}, error) {
	b, err := gophercloud.BuildRequestBody(opts, "")
	if err != nil {
		return nil, err
	}
	return map[string]interface{}{"method": b}, nil
}

// Create requests the creation of a new image import on the server.
func Create(client *gophercloud.ServiceClient, imageID string, opts CreateOptsBuilder) (r CreateResult) {
	b, err := opts.ToImportCreateMap()
	if err != nil {
		r.Err = err
		return
	}
	_, r.Err = client.Post(importURL(client, imageID), b, nil, &gophercloud.RequestOpts{
		OkCodes: []int{202},
	})
	return
}
