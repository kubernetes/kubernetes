package ports

import (
	"fmt"

	"github.com/gophercloud/gophercloud"
	"github.com/gophercloud/gophercloud/pagination"
)

// ListOptsBuilder allows extensions to add additional parameters to the
// List request.
type ListOptsBuilder interface {
	ToPortListQuery() (string, error)
	ToPortListDetailQuery() (string, error)
}

// ListOpts allows the filtering and sorting of paginated collections through
// the API. Filtering is achieved by passing in struct field values that map to
// the node attributes you want to see returned. Marker and Limit are used
// for pagination.
type ListOpts struct {
	// Filter the list by the name or uuid of the Node
	Node string `q:"node"`

	// Filter the list by the Node uuid
	NodeUUID string `q:"node_uuid"`

	// Filter the list with the specified Portgroup (name or UUID)
	PortGroup string `q:"portgroup"`

	// Filter the list with the specified physical hardware address, typically MAC
	Address string `q:"address"`

	// One or more fields to be returned in the response.
	Fields []string `q:"fields"`

	// Requests a page size of items.
	Limit int `q:"limit"`

	// The ID of the last-seen item
	Marker string `q:"marker"`

	// Sorts the response by the requested sort direction.
	// Valid value is asc (ascending) or desc (descending). Default is asc.
	SortDir string `q:"sort_dir"`

	// Sorts the response by the this attribute value. Default is id.
	SortKey string `q:"sort_key"`
}

// ToPortListQuery formats a ListOpts into a query string.
func (opts ListOpts) ToPortListQuery() (string, error) {
	q, err := gophercloud.BuildQueryString(opts)
	return q.String(), err
}

// List makes a request against the API to list ports accessible to you.
func List(client *gophercloud.ServiceClient, opts ListOptsBuilder) pagination.Pager {
	url := listURL(client)
	if opts != nil {
		query, err := opts.ToPortListQuery()
		if err != nil {
			return pagination.Pager{Err: err}
		}
		url += query
	}
	return pagination.NewPager(client, url, func(r pagination.PageResult) pagination.Page {
		return PortPage{pagination.LinkedPageBase{PageResult: r}}
	})
}

// ToPortListDetailQuery formats a ListOpts into a query string for the list details API.
func (opts ListOpts) ToPortListDetailQuery() (string, error) {
	// Detail endpoint can't filter by Fields
	if len(opts.Fields) > 0 {
		return "", fmt.Errorf("fields is not a valid option when getting a detailed listing of ports")
	}

	q, err := gophercloud.BuildQueryString(opts)
	return q.String(), err
}

// ListDetail - Return a list ports with complete details.
// Some filtering is possible by passing in flags in "ListOpts",
// but you cannot limit by the fields returned.
func ListDetail(client *gophercloud.ServiceClient, opts ListOptsBuilder) pagination.Pager {
	url := listDetailURL(client)
	if opts != nil {
		query, err := opts.ToPortListDetailQuery()
		if err != nil {
			return pagination.Pager{Err: err}
		}
		url += query
	}
	return pagination.NewPager(client, url, func(r pagination.PageResult) pagination.Page {
		return PortPage{pagination.LinkedPageBase{PageResult: r}}
	})
}

// Get - requests the details off a port, by ID.
func Get(client *gophercloud.ServiceClient, id string) (r GetResult) {
	_, r.Err = client.Get(getURL(client, id), &r.Body, &gophercloud.RequestOpts{
		OkCodes: []int{200},
	})
	return
}

// CreateOptsBuilder allows extensions to add additional parameters to the
// Create request.
type CreateOptsBuilder interface {
	ToPortCreateMap() (map[string]interface{}, error)
}

// CreateOpts specifies port creation parameters.
type CreateOpts struct {
	// UUID of the Node this resource belongs to.
	NodeUUID string `json:"node_uuid,omitempty"`

	// Physical hardware address of this network Port,
	// typically the hardware MAC address.
	Address string `json:"address,omitempty"`

	// UUID of the Portgroup this resource belongs to.
	PortGroupUUID string `json:"portgroup_uuid,omitempty"`

	// The Port binding profile. If specified, must contain switch_id (only a MAC
	// address or an OpenFlow based datapath_id of the switch are accepted in this
	// field) and port_id (identifier of the physical port on the switch to which
	// node’s port is connected to) fields. switch_info is an optional string
	// field to be used to store any vendor-specific information.
	LocalLinkConnection map[string]interface{} `json:"local_link_connection,omitempty"`

	// Indicates whether PXE is enabled or disabled on the Port.
	PXEEnabled *bool `json:"pxe_enabled,omitempty"`

	// The name of the physical network to which a port is connected. May be empty.
	PhysicalNetwork string `json:"physical_network,omitempty"`

	// A set of one or more arbitrary metadata key and value pairs.
	Extra map[string]interface{} `json:"extra,omitempty"`

	// Indicates whether the Port is a Smart NIC port.
	IsSmartNIC *bool `json:"is_smartnic,omitempty"`
}

// ToPortCreateMap assembles a request body based on the contents of a CreateOpts.
func (opts CreateOpts) ToPortCreateMap() (map[string]interface{}, error) {
	body, err := gophercloud.BuildRequestBody(opts, "")
	if err != nil {
		return nil, err
	}

	return body, nil
}

// Create - requests the creation of a port
func Create(client *gophercloud.ServiceClient, opts CreateOptsBuilder) (r CreateResult) {
	reqBody, err := opts.ToPortCreateMap()
	if err != nil {
		r.Err = err
		return
	}

	_, r.Err = client.Post(createURL(client), reqBody, &r.Body, nil)
	return
}

// TODO Update
type Patch interface {
	ToPortUpdateMap() map[string]interface{}
}

// UpdateOpts is a slice of Patches used to update a port
type UpdateOpts []Patch

type UpdateOp string

const (
	ReplaceOp UpdateOp = "replace"
	AddOp     UpdateOp = "add"
	RemoveOp  UpdateOp = "remove"
)

type UpdateOperation struct {
	Op    UpdateOp    `json:"op,required"`
	Path  string      `json:"path,required"`
	Value interface{} `json:"value,omitempty"`
}

func (opts UpdateOperation) ToPortUpdateMap() map[string]interface{} {
	return map[string]interface{}{
		"op":    opts.Op,
		"path":  opts.Path,
		"value": opts.Value,
	}
}

// Update - requests the update of a port
func Update(client *gophercloud.ServiceClient, id string, opts UpdateOpts) (r UpdateResult) {
	body := make([]map[string]interface{}, len(opts))
	for i, patch := range opts {
		body[i] = patch.ToPortUpdateMap()
	}

	_, r.Err = client.Patch(updateURL(client, id), body, &r.Body, &gophercloud.RequestOpts{
		JSONBody: &body,
		OkCodes:  []int{200},
	})
	return
}

// Delete - requests the deletion of a port
func Delete(client *gophercloud.ServiceClient, id string) (r DeleteResult) {
	_, r.Err = client.Delete(deleteURL(client, id), nil)
	return
}
