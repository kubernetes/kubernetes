package servergroups

import (
	"errors"

	"github.com/rackspace/gophercloud"
	"github.com/rackspace/gophercloud/pagination"
)

// List returns a Pager that allows you to iterate over a collection of ServerGroups.
func List(client *gophercloud.ServiceClient) pagination.Pager {
	return pagination.NewPager(client, listURL(client), func(r pagination.PageResult) pagination.Page {
		return ServerGroupsPage{pagination.SinglePageBase(r)}
	})
}

// CreateOptsBuilder describes struct types that can be accepted by the Create call. Notably, the
// CreateOpts struct in this package does.
type CreateOptsBuilder interface {
	ToServerGroupCreateMap() (map[string]interface{}, error)
}

// CreateOpts specifies a Server Group allocation request
type CreateOpts struct {
	// Name is the name of the server group
	Name string

	// Policies are the server group policies
	Policies []string
}

// ToServerGroupCreateMap constructs a request body from CreateOpts.
func (opts CreateOpts) ToServerGroupCreateMap() (map[string]interface{}, error) {
	if opts.Name == "" {
		return nil, errors.New("Missing field required for server group creation: Name")
	}

	if len(opts.Policies) < 1 {
		return nil, errors.New("Missing field required for server group creation: Policies")
	}

	serverGroup := make(map[string]interface{})
	serverGroup["name"] = opts.Name
	serverGroup["policies"] = opts.Policies

	return map[string]interface{}{"server_group": serverGroup}, nil
}

// Create requests the creation of a new Server Group
func Create(client *gophercloud.ServiceClient, opts CreateOptsBuilder) CreateResult {
	var res CreateResult

	reqBody, err := opts.ToServerGroupCreateMap()
	if err != nil {
		res.Err = err
		return res
	}

	_, res.Err = client.Post(createURL(client), reqBody, &res.Body, &gophercloud.RequestOpts{
		OkCodes: []int{200},
	})
	return res
}

// Get returns data about a previously created ServerGroup.
func Get(client *gophercloud.ServiceClient, id string) GetResult {
	var res GetResult
	_, res.Err = client.Get(getURL(client, id), &res.Body, nil)
	return res
}

// Delete requests the deletion of a previously allocated ServerGroup.
func Delete(client *gophercloud.ServiceClient, id string) DeleteResult {
	var res DeleteResult
	_, res.Err = client.Delete(deleteURL(client, id), nil)
	return res
}
