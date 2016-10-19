package groups

import (
	"fmt"

	"github.com/rackspace/gophercloud"
	"github.com/rackspace/gophercloud/pagination"
)

// ListOpts allows the filtering and sorting of paginated collections through
// the API. Filtering is achieved by passing in struct field values that map to
// the floating IP attributes you want to see returned. SortKey allows you to
// sort by a particular network attribute. SortDir sets the direction, and is
// either `asc' or `desc'. Marker and Limit are used for pagination.
type ListOpts struct {
	ID       string `q:"id"`
	Name     string `q:"name"`
	TenantID string `q:"tenant_id"`
	Limit    int    `q:"limit"`
	Marker   string `q:"marker"`
	SortKey  string `q:"sort_key"`
	SortDir  string `q:"sort_dir"`
}

// List returns a Pager which allows you to iterate over a collection of
// security groups. It accepts a ListOpts struct, which allows you to filter
// and sort the returned collection for greater efficiency.
func List(c *gophercloud.ServiceClient, opts ListOpts) pagination.Pager {
	q, err := gophercloud.BuildQueryString(&opts)
	if err != nil {
		return pagination.Pager{Err: err}
	}
	u := rootURL(c) + q.String()
	return pagination.NewPager(c, u, func(r pagination.PageResult) pagination.Page {
		return SecGroupPage{pagination.LinkedPageBase{PageResult: r}}
	})
}

var (
	errNameRequired = fmt.Errorf("Name is required")
)

// CreateOpts contains all the values needed to create a new security group.
type CreateOpts struct {
	// Required. Human-readable name for the VIP. Does not have to be unique.
	Name string

	// Required for admins. Indicates the owner of the VIP.
	TenantID string

	// Optional. Describes the security group.
	Description string
}

// Create is an operation which provisions a new security group with default
// security group rules for the IPv4 and IPv6 ether types.
func Create(c *gophercloud.ServiceClient, opts CreateOpts) CreateResult {
	var res CreateResult

	// Validate required opts
	if opts.Name == "" {
		res.Err = errNameRequired
		return res
	}

	type secgroup struct {
		Name        string `json:"name"`
		TenantID    string `json:"tenant_id,omitempty"`
		Description string `json:"description,omitempty"`
	}

	type request struct {
		SecGroup secgroup `json:"security_group"`
	}

	reqBody := request{SecGroup: secgroup{
		Name:        opts.Name,
		TenantID:    opts.TenantID,
		Description: opts.Description,
	}}

	_, res.Err = c.Post(rootURL(c), reqBody, &res.Body, nil)
	return res
}

// Get retrieves a particular security group based on its unique ID.
func Get(c *gophercloud.ServiceClient, id string) GetResult {
	var res GetResult
	_, res.Err = c.Get(resourceURL(c, id), &res.Body, nil)
	return res
}

// Delete will permanently delete a particular security group based on its unique ID.
func Delete(c *gophercloud.ServiceClient, id string) DeleteResult {
	var res DeleteResult
	_, res.Err = c.Delete(resourceURL(c, id), nil)
	return res
}

// IDFromName is a convenience function that returns a security group's ID given its name.
func IDFromName(client *gophercloud.ServiceClient, name string) (string, error) {
	securityGroupCount := 0
	securityGroupID := ""
	if name == "" {
		return "", fmt.Errorf("A security group name must be provided.")
	}
	pager := List(client, ListOpts{})
	pager.EachPage(func(page pagination.Page) (bool, error) {
		securityGroupList, err := ExtractGroups(page)
		if err != nil {
			return false, err
		}

		for _, s := range securityGroupList {
			if s.Name == name {
				securityGroupCount++
				securityGroupID = s.ID
			}
		}
		return true, nil
	})

	switch securityGroupCount {
	case 0:
		return "", fmt.Errorf("Unable to find security group: %s", name)
	case 1:
		return securityGroupID, nil
	default:
		return "", fmt.Errorf("Found %d security groups matching %s", securityGroupCount, name)
	}
}
