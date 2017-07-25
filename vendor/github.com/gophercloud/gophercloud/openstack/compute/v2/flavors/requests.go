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

// AccessType maps to OpenStack's Flavor.is_public field. Although the is_public field is boolean, the
// request options are ternary, which is why AccessType is a string. The following values are
// allowed:
//
//      PublicAccess (the default):  Returns public flavors and private flavors associated with that project.
//      PrivateAccess (admin only):  Returns private flavors, across all projects.
//      AllAccess (admin only):      Returns public and private flavors across all projects.
//
// The AccessType arguement is optional, and if it is not supplied, OpenStack returns the PublicAccess
// flavors.
type AccessType string

const (
	PublicAccess  AccessType = "true"
	PrivateAccess AccessType = "false"
	AllAccess     AccessType = "None"
)

// ListOpts helps control the results returned by the List() function.
// For example, a flavor with a minDisk field of 10 will not be returned if you specify MinDisk set to 20.
// Typically, software will use the last ID of the previous call to List to set the Marker for the current call.
type ListOpts struct {

	// ChangesSince, if provided, instructs List to return only those things which have changed since the timestamp provided.
	ChangesSince string `q:"changes-since"`

	// MinDisk and MinRAM, if provided, elides flavors which do not meet your criteria.
	MinDisk int `q:"minDisk"`
	MinRAM  int `q:"minRam"`

	// Marker and Limit control paging.
	// Marker instructs List where to start listing from.
	Marker string `q:"marker"`

	// Limit instructs List to refrain from sending excessively large lists of flavors.
	Limit int `q:"limit"`

	// AccessType, if provided, instructs List which set of flavors to return. If IsPublic not provided,
	// flavors for the current project are returned.
	AccessType AccessType `q:"is_public"`
}

// ToFlavorListQuery formats a ListOpts into a query string.
func (opts ListOpts) ToFlavorListQuery() (string, error) {
	q, err := gophercloud.BuildQueryString(opts)
	return q.String(), err
}

// ListDetail instructs OpenStack to provide a list of flavors.
// You may provide criteria by which List curtails its results for easier processing.
// See ListOpts for more details.
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

// CreateOpts is passed to Create to create a flavor
// Source:
// https://github.com/openstack/nova/blob/stable/newton/nova/api/openstack/compute/schemas/flavor_manage.py#L20
type CreateOpts struct {
	Name string `json:"name" required:"true"`
	// memory size, in MBs
	RAM   int `json:"ram" required:"true"`
	VCPUs int `json:"vcpus" required:"true"`
	// disk size, in GBs
	Disk *int   `json:"disk" required:"true"`
	ID   string `json:"id,omitempty"`
	// non-zero, positive
	Swap       *int    `json:"swap,omitempty"`
	RxTxFactor float64 `json:"rxtx_factor,omitempty"`
	IsPublic   *bool   `json:"os-flavor-access:is_public,omitempty"`
	// ephemeral disk size, in GBs, non-zero, positive
	Ephemeral *int `json:"OS-FLV-EXT-DATA:ephemeral,omitempty"`
}

// ToFlavorCreateMap satisfies the CreateOptsBuilder interface
func (opts *CreateOpts) ToFlavorCreateMap() (map[string]interface{}, error) {
	return gophercloud.BuildRequestBody(opts, "flavor")
}

// Create a flavor
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

// Get instructs OpenStack to provide details on a single flavor, identified by its ID.
// Use ExtractFlavor to convert its result into a Flavor.
func Get(client *gophercloud.ServiceClient, id string) (r GetResult) {
	_, r.Err = client.Get(getURL(client, id), &r.Body, nil)
	return
}

// IDFromName is a convienience function that returns a flavor's ID given its name.
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
