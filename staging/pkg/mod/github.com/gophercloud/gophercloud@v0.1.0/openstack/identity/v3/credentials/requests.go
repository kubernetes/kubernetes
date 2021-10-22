package credentials

import (
	"github.com/gophercloud/gophercloud"
	"github.com/gophercloud/gophercloud/pagination"
)

// ListOptsBuilder allows extensions to add additional parameters to
// the List request
type ListOptsBuilder interface {
	ToCredentialListQuery() (string, error)
}

// ListOpts provides options to filter the List results.
type ListOpts struct {
	// UserID filters the response by a credential user_id
	UserID string `q:"user_id"`
	// Type filters the response by a credential type
	Type string `q:"type"`
}

// ToCredentialListQuery formats a ListOpts into a query string.
func (opts ListOpts) ToCredentialListQuery() (string, error) {
	q, err := gophercloud.BuildQueryString(opts)
	return q.String(), err
}

// List enumerates the Credentials to which the current token has access.
func List(client *gophercloud.ServiceClient, opts ListOptsBuilder) pagination.Pager {
	url := listURL(client)
	if opts != nil {
		query, err := opts.ToCredentialListQuery()
		if err != nil {
			return pagination.Pager{Err: err}
		}
		url += query
	}
	return pagination.NewPager(client, url, func(r pagination.PageResult) pagination.Page {
		return CredentialPage{pagination.LinkedPageBase{PageResult: r}}
	})
}

// Get retrieves details on a single user, by ID.
func Get(client *gophercloud.ServiceClient, id string) (r GetResult) {
	_, r.Err = client.Get(getURL(client, id), &r.Body, nil)
	return
}

// CreateOptsBuilder allows extensions to add additional parameters to
// the Create request.
type CreateOptsBuilder interface {
	ToCredentialCreateMap() (map[string]interface{}, error)
}

// CreateOpts provides options used to create a credential.
type CreateOpts struct {
	// Serialized blob containing the credentials
	Blob string `json:"blob" required:"true"`
	// ID of the project.
	ProjectID string `json:"project_id,omitempty"`
	// The type of the credential.
	Type string `json:"type" required:"true"`
	// ID of the user who owns the credential.
	UserID string `json:"user_id" required:"true"`
}

// ToCredentialCreateMap formats a CreateOpts into a create request.
func (opts CreateOpts) ToCredentialCreateMap() (map[string]interface{}, error) {
	return gophercloud.BuildRequestBody(opts, "credential")
}

// Create creates a new Credential.
func Create(client *gophercloud.ServiceClient, opts CreateOptsBuilder) (r CreateResult) {
	b, err := opts.ToCredentialCreateMap()
	if err != nil {
		r.Err = err
		return
	}
	_, r.Err = client.Post(createURL(client), &b, &r.Body, &gophercloud.RequestOpts{
		OkCodes: []int{201},
	})
	return
}

// Delete deletes a credential.
func Delete(client *gophercloud.ServiceClient, id string) (r DeleteResult) {
	_, r.Err = client.Delete(deleteURL(client, id), nil)
	return
}

// UpdateOptsBuilder allows extensions to add additional parameters to
// the Update request.
type UpdateOptsBuilder interface {
	ToCredentialsUpdateMap() (map[string]interface{}, error)
}

// UpdateOpts represents parameters to update a credential.
type UpdateOpts struct {
	// Serialized blob containing the credentials.
	Blob string `json:"blob,omitempty"`
	// ID of the project.
	ProjectID string `json:"project_id,omitempty"`
	// The type of the credential.
	Type string `json:"type,omitempty"`
	// ID of the user who owns the credential.
	UserID string `json:"user_id,omitempty"`
}

// ToUpdateCreateMap formats a UpdateOpts into an update request.
func (opts UpdateOpts) ToCredentialsUpdateMap() (map[string]interface{}, error) {
	return gophercloud.BuildRequestBody(opts, "credential")
}

// Update modifies the attributes of a Credential.
func Update(client *gophercloud.ServiceClient, id string, opts UpdateOptsBuilder) (r UpdateResult) {
	b, err := opts.ToCredentialsUpdateMap()
	if err != nil {
		r.Err = err
		return
	}
	_, r.Err = client.Patch(updateURL(client, id), b, &r.Body, &gophercloud.RequestOpts{
		OkCodes: []int{200},
	})
	return
}
