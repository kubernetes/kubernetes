package flavors

import (
	"github.com/gophercloud/gophercloud"
	"github.com/gophercloud/gophercloud/pagination"
)

// ListOptsBuilder allows extensions to add additional parameters to the
// List request.
type ListOptsBuilder interface {
	ToFlavorListQuery() (string, error)
}

/*
	AccessType maps to OpenStack's Flavor.is_public field. Although the is_public
	field is boolean, the request options are ternary, which is why AccessType is
	a string. The following values are allowed:

	The AccessType arguement is optional, and if it is not supplied, OpenStack
	returns the PublicAccess flavors.
*/
type AccessType string

const (
	// PublicAccess returns public flavors and private flavors associated with
	// that project.
	PublicAccess AccessType = "true"

	// PrivateAccess (admin only) returns private flavors, across all projects.
	PrivateAccess AccessType = "false"

	// AllAccess (admin only) returns public and private flavors across all
	// projects.
	AllAccess AccessType = "None"
)

/*
	ListOpts filters the results returned by the List() function.
	For example, a flavor with a minDisk field of 10 will not be returned if you
	specify MinDisk set to 20.

	Typically, software will use the last ID of the previous call to List to set
	the Marker for the current call.
*/
type ListOpts struct {
	// ChangesSince, if provided, instructs List to return only those things which
	// have changed since the timestamp provided.
	ChangesSince string `q:"changes-since"`

	// MinDisk and MinRAM, if provided, elides flavors which do not meet your
	// criteria.
	MinDisk int `q:"minDisk"`
	MinRAM  int `q:"minRam"`

	// Marker and Limit control paging.
	// Marker instructs List where to start listing from.
	Marker string `q:"marker"`

	// Limit instructs List to refrain from sending excessively large lists of
	// flavors.
	Limit int `q:"limit"`

	// AccessType, if provided, instructs List which set of flavors to return.
	// If IsPublic not provided, flavors for the current project are returned.
	AccessType AccessType `q:"is_public"`
}

// ToFlavorListQuery formats a ListOpts into a query string.
func (opts ListOpts) ToFlavorListQuery() (string, error) {
	q, err := gophercloud.BuildQueryString(opts)
	return q.String(), err
}

// ListDetail instructs OpenStack to provide a list of flavors.
// You may provide criteria by which List curtails its results for easier
// processing.
func ListDetail(client *gophercloud.ServiceClient, opts ListOptsBuilder) pagination.Pager {
	url := listURL(client)
	if opts != nil {
		query, err := opts.ToFlavorListQuery()
		if err != nil {
			return pagination.Pager{Err: err}
		}
		url += query
	}
	return pagination.NewPager(client, url, func(r pagination.PageResult) pagination.Page {
		return FlavorPage{pagination.LinkedPageBase{PageResult: r}}
	})
}

type CreateOptsBuilder interface {
	ToFlavorCreateMap() (map[string]interface{}, error)
}

// CreateOpts specifies parameters used for creating a flavor.
type CreateOpts struct {
	// Name is the name of the flavor.
	Name string `json:"name" required:"true"`

	// RAM is the memory of the flavor, measured in MB.
	RAM int `json:"ram" required:"true"`

	// VCPUs is the number of vcpus for the flavor.
	VCPUs int `json:"vcpus" required:"true"`

	// Disk the amount of root disk space, measured in GB.
	Disk *int `json:"disk" required:"true"`

	// ID is a unique ID for the flavor.
	ID string `json:"id,omitempty"`

	// Swap is the amount of swap space for the flavor, measured in MB.
	Swap *int `json:"swap,omitempty"`

	// RxTxFactor alters the network bandwidth of a flavor.
	RxTxFactor float64 `json:"rxtx_factor,omitempty"`

	// IsPublic flags a flavor as being available to all projects or not.
	IsPublic *bool `json:"os-flavor-access:is_public,omitempty"`

	// Ephemeral is the amount of ephemeral disk space, measured in GB.
	Ephemeral *int `json:"OS-FLV-EXT-DATA:ephemeral,omitempty"`
}

// ToFlavorCreateMap constructs a request body from CreateOpts.
func (opts CreateOpts) ToFlavorCreateMap() (map[string]interface{}, error) {
	return gophercloud.BuildRequestBody(opts, "flavor")
}

// Create requests the creation of a new flavor.
func Create(client *gophercloud.ServiceClient, opts CreateOptsBuilder) (r CreateResult) {
	b, err := opts.ToFlavorCreateMap()
	if err != nil {
		r.Err = err
		return
	}
	_, r.Err = client.Post(createURL(client), b, &r.Body, &gophercloud.RequestOpts{
		OkCodes: []int{200, 201},
	})
	return
}

// Get retrieves details of a single flavor. Use ExtractFlavor to convert its
// result into a Flavor.
func Get(client *gophercloud.ServiceClient, id string) (r GetResult) {
	_, r.Err = client.Get(getURL(client, id), &r.Body, nil)
	return
}

// Delete deletes the specified flavor ID.
func Delete(client *gophercloud.ServiceClient, id string) (r DeleteResult) {
	_, r.Err = client.Delete(deleteURL(client, id), nil)
	return
}

// ListAccesses retrieves the tenants which have access to a flavor.
func ListAccesses(client *gophercloud.ServiceClient, id string) pagination.Pager {
	url := accessURL(client, id)

	return pagination.NewPager(client, url, func(r pagination.PageResult) pagination.Page {
		return AccessPage{pagination.SinglePageBase(r)}
	})
}

// AddAccessOptsBuilder allows extensions to add additional parameters to the
// AddAccess requests.
type AddAccessOptsBuilder interface {
	ToAddAccessMap() (map[string]interface{}, error)
}

// AddAccessOpts represents options for adding access to a flavor.
type AddAccessOpts struct {
	// Tenant is the project/tenant ID to grant access.
	Tenant string `json:"tenant"`
}

// ToAddAccessMap constructs a request body from AddAccessOpts.
func (opts AddAccessOpts) ToAddAccessMap() (map[string]interface{}, error) {
	return gophercloud.BuildRequestBody(opts, "addTenantAccess")
}

// AddAccess grants a tenant/project access to a flavor.
func AddAccess(client *gophercloud.ServiceClient, id string, opts AddAccessOptsBuilder) (r AddAccessResult) {
	b, err := opts.ToAddAccessMap()
	if err != nil {
		r.Err = err
		return
	}
	_, r.Err = client.Post(accessActionURL(client, id), b, &r.Body, &gophercloud.RequestOpts{
		OkCodes: []int{200},
	})
	return
}

// ExtraSpecs requests all the extra-specs for the given flavor ID.
func ListExtraSpecs(client *gophercloud.ServiceClient, flavorID string) (r ListExtraSpecsResult) {
	_, r.Err = client.Get(extraSpecsListURL(client, flavorID), &r.Body, nil)
	return
}

func GetExtraSpec(client *gophercloud.ServiceClient, flavorID string, key string) (r GetExtraSpecResult) {
	_, r.Err = client.Get(extraSpecsGetURL(client, flavorID, key), &r.Body, nil)
	return
}

// CreateExtraSpecsOptsBuilder allows extensions to add additional parameters to the
// CreateExtraSpecs requests.
type CreateExtraSpecsOptsBuilder interface {
	ToExtraSpecsCreateMap() (map[string]interface{}, error)
}

// ExtraSpecsOpts is a map that contains key-value pairs.
type ExtraSpecsOpts map[string]string

// ToExtraSpecsCreateMap assembles a body for a Create request based on the
// contents of a ExtraSpecsOpts
func (opts ExtraSpecsOpts) ToExtraSpecsCreateMap() (map[string]interface{}, error) {
	return map[string]interface{}{"extra_specs": opts}, nil
}

// CreateExtraSpecs will create or update the extra-specs key-value pairs for the specified Flavor
func CreateExtraSpecs(client *gophercloud.ServiceClient, flavorID string, opts CreateExtraSpecsOptsBuilder) (r CreateExtraSpecsResult) {
	b, err := opts.ToExtraSpecsCreateMap()
	if err != nil {
		r.Err = err
		return
	}
	_, r.Err = client.Post(extraSpecsCreateURL(client, flavorID), b, &r.Body, &gophercloud.RequestOpts{
		OkCodes: []int{200},
	})
	return
}

// IDFromName is a convienience function that returns a flavor's ID given its
// name.
func IDFromName(client *gophercloud.ServiceClient, name string) (string, error) {
	count := 0
	id := ""
	allPages, err := ListDetail(client, nil).AllPages()
	if err != nil {
		return "", err
	}

	all, err := ExtractFlavors(allPages)
	if err != nil {
		return "", err
	}

	for _, f := range all {
		if f.Name == name {
			count++
			id = f.ID
		}
	}

	switch count {
	case 0:
		err := &gophercloud.ErrResourceNotFound{}
		err.ResourceType = "flavor"
		err.Name = name
		return "", err
	case 1:
		return id, nil
	default:
		err := &gophercloud.ErrMultipleResourcesFound{}
		err.ResourceType = "flavor"
		err.Name = name
		err.Count = count
		return "", err
	}
}
