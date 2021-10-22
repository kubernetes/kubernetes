package ikepolicies

import (
	"github.com/gophercloud/gophercloud"
	"github.com/gophercloud/gophercloud/pagination"
)

type AuthAlgorithm string
type EncryptionAlgorithm string
type PFS string
type Unit string
type IKEVersion string
type Phase1NegotiationMode string

const (
	AuthAlgorithmSHA1         AuthAlgorithm         = "sha1"
	AuthAlgorithmSHA256       AuthAlgorithm         = "sha256"
	AuthAlgorithmSHA384       AuthAlgorithm         = "sha384"
	AuthAlgorithmSHA512       AuthAlgorithm         = "sha512"
	EncryptionAlgorithm3DES   EncryptionAlgorithm   = "3des"
	EncryptionAlgorithmAES128 EncryptionAlgorithm   = "aes-128"
	EncryptionAlgorithmAES256 EncryptionAlgorithm   = "aes-256"
	EncryptionAlgorithmAES192 EncryptionAlgorithm   = "aes-192"
	UnitSeconds               Unit                  = "seconds"
	UnitKilobytes             Unit                  = "kilobytes"
	PFSGroup2                 PFS                   = "group2"
	PFSGroup5                 PFS                   = "group5"
	PFSGroup14                PFS                   = "group14"
	IKEVersionv1              IKEVersion            = "v1"
	IKEVersionv2              IKEVersion            = "v2"
	Phase1NegotiationModeMain Phase1NegotiationMode = "main"
)

// CreateOptsBuilder allows extensions to add additional parameters to the
// Create request.
type CreateOptsBuilder interface {
	ToPolicyCreateMap() (map[string]interface{}, error)
}

// CreateOpts contains all the values needed to create a new IKE policy
type CreateOpts struct {
	// TenantID specifies a tenant to own the IKE policy. The caller must have
	// an admin role in order to set this. Otherwise, this field is left unset
	// and the caller will be the owner.
	TenantID string `json:"tenant_id,omitempty"`

	// Description is the human readable description of the policy.
	Description string `json:"description,omitempty"`

	// Name is the human readable name of the policy.
	// Does not have to be unique.
	Name string `json:"name,omitempty"`

	// AuthAlgorithm is the authentication hash algorithm.
	// Valid values are sha1, sha256, sha384, sha512.
	// The default is sha1.
	AuthAlgorithm AuthAlgorithm `json:"auth_algorithm,omitempty"`

	// EncryptionAlgorithm is the encryption algorithm.
	// A valid value is 3des, aes-128, aes-192, aes-256, and so on.
	// Default is aes-128.
	EncryptionAlgorithm EncryptionAlgorithm `json:"encryption_algorithm,omitempty"`

	// PFS is the Perfect forward secrecy mode.
	// A valid value is Group2, Group5, Group14, and so on.
	// Default is Group5.
	PFS PFS `json:"pfs,omitempty"`

	// The IKE mode.
	// A valid value is main, which is the default.
	Phase1NegotiationMode Phase1NegotiationMode `json:"phase1_negotiation_mode,omitempty"`

	// The IKE version.
	// A valid value is v1 or v2.
	// Default is v1.
	IKEVersion IKEVersion `json:"ike_version,omitempty"`

	//Lifetime is the lifetime of the security association
	Lifetime *LifetimeCreateOpts `json:"lifetime,omitempty"`
}

// The lifetime consists of a unit and integer value
// You can omit either the unit or value portion of the lifetime
type LifetimeCreateOpts struct {
	// Units is the units for the lifetime of the security association
	// Default unit is seconds
	Units Unit `json:"units,omitempty"`

	// The lifetime value.
	// Must be a positive integer.
	// Default value is 3600.
	Value int `json:"value,omitempty"`
}

// ToPolicyCreateMap casts a CreateOpts struct to a map.
func (opts CreateOpts) ToPolicyCreateMap() (map[string]interface{}, error) {
	return gophercloud.BuildRequestBody(opts, "ikepolicy")
}

// Create accepts a CreateOpts struct and uses the values to create a new
// IKE policy
func Create(c *gophercloud.ServiceClient, opts CreateOptsBuilder) (r CreateResult) {
	b, err := opts.ToPolicyCreateMap()
	if err != nil {
		r.Err = err
		return
	}
	_, r.Err = c.Post(rootURL(c), b, &r.Body, nil)
	return
}

// Get retrieves a particular IKE policy based on its unique ID.
func Get(c *gophercloud.ServiceClient, id string) (r GetResult) {
	_, r.Err = c.Get(resourceURL(c, id), &r.Body, nil)
	return
}

// Delete will permanently delete a particular IKE policy based on its
// unique ID.
func Delete(c *gophercloud.ServiceClient, id string) (r DeleteResult) {
	_, r.Err = c.Delete(resourceURL(c, id), nil)
	return
}

// ListOptsBuilder allows extensions to add additional parameters to the
// List request.
type ListOptsBuilder interface {
	ToPolicyListQuery() (string, error)
}

// ListOpts allows the filtering of paginated collections through
// the API. Filtering is achieved by passing in struct field values that map to
// the IKE policy attributes you want to see returned.
type ListOpts struct {
	TenantID              string `q:"tenant_id"`
	Name                  string `q:"name"`
	Description           string `q:"description"`
	ProjectID             string `q:"project_id"`
	AuthAlgorithm         string `q:"auth_algorithm"`
	EncapsulationMode     string `q:"encapsulation_mode"`
	EncryptionAlgorithm   string `q:"encryption_algorithm"`
	PFS                   string `q:"pfs"`
	Phase1NegotiationMode string `q:"phase_1_negotiation_mode"`
	IKEVersion            string `q:"ike_version"`
}

// ToPolicyListQuery formats a ListOpts into a query string.
func (opts ListOpts) ToPolicyListQuery() (string, error) {
	q, err := gophercloud.BuildQueryString(opts)
	return q.String(), err
}

// List returns a Pager which allows you to iterate over a collection of
// IKE policies. It accepts a ListOpts struct, which allows you to filter
// the returned collection for greater efficiency.
func List(c *gophercloud.ServiceClient, opts ListOptsBuilder) pagination.Pager {
	url := rootURL(c)
	if opts != nil {
		query, err := opts.ToPolicyListQuery()
		if err != nil {
			return pagination.Pager{Err: err}
		}
		url += query
	}
	return pagination.NewPager(c, url, func(r pagination.PageResult) pagination.Page {
		return PolicyPage{pagination.LinkedPageBase{PageResult: r}}
	})
}

// UpdateOptsBuilder allows extensions to add additional parameters to the
// Update request.
type UpdateOptsBuilder interface {
	ToPolicyUpdateMap() (map[string]interface{}, error)
}

type LifetimeUpdateOpts struct {
	Units Unit `json:"units,omitempty"`
	Value int  `json:"value,omitempty"`
}

// UpdateOpts contains the values used when updating an IKE policy
type UpdateOpts struct {
	Description           *string               `json:"description,omitempty"`
	Name                  *string               `json:"name,omitempty"`
	AuthAlgorithm         AuthAlgorithm         `json:"auth_algorithm,omitempty"`
	EncryptionAlgorithm   EncryptionAlgorithm   `json:"encryption_algorithm,omitempty"`
	PFS                   PFS                   `json:"pfs,omitempty"`
	Lifetime              *LifetimeUpdateOpts   `json:"lifetime,omitempty"`
	Phase1NegotiationMode Phase1NegotiationMode `json:"phase_1_negotiation_mode,omitempty"`
	IKEVersion            IKEVersion            `json:"ike_version,omitempty"`
}

// ToPolicyUpdateMap casts an UpdateOpts struct to a map.
func (opts UpdateOpts) ToPolicyUpdateMap() (map[string]interface{}, error) {
	return gophercloud.BuildRequestBody(opts, "ikepolicy")
}

// Update allows IKE policies to be updated.
func Update(c *gophercloud.ServiceClient, id string, opts UpdateOptsBuilder) (r UpdateResult) {
	b, err := opts.ToPolicyUpdateMap()
	if err != nil {
		r.Err = err
		return
	}
	_, r.Err = c.Put(resourceURL(c, id), b, &r.Body, &gophercloud.RequestOpts{
		OkCodes: []int{200},
	})
	return
}
