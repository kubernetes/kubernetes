package trunks

import (
	"github.com/gophercloud/gophercloud"
	"github.com/gophercloud/gophercloud/pagination"
)

// CreateOptsBuilder allows extensions to add additional parameters to the
// Create request.
type CreateOptsBuilder interface {
	ToTrunkCreateMap() (map[string]interface{}, error)
}

// CreateOpts represents the attributes used when creating a new trunk.
type CreateOpts struct {
	TenantID     string    `json:"tenant_id,omitempty"`
	ProjectID    string    `json:"project_id,omitempty"`
	PortID       string    `json:"port_id" required:"true"`
	Name         string    `json:"name,omitempty"`
	Description  string    `json:"description,omitempty"`
	AdminStateUp *bool     `json:"admin_state_up,omitempty"`
	Subports     []Subport `json:"sub_ports"`
}

// ToTrunkCreateMap builds a request body from CreateOpts.
func (opts CreateOpts) ToTrunkCreateMap() (map[string]interface{}, error) {
	if opts.Subports == nil {
		opts.Subports = []Subport{}
	}
	return gophercloud.BuildRequestBody(opts, "trunk")
}

func Create(c *gophercloud.ServiceClient, opts CreateOptsBuilder) (r CreateResult) {
	body, err := opts.ToTrunkCreateMap()
	if err != nil {
		r.Err = err
		return
	}

	_, r.Err = c.Post(createURL(c), body, &r.Body, nil)
	return
}

// Delete accepts a unique ID and deletes the trunk associated with it.
func Delete(c *gophercloud.ServiceClient, id string) (r DeleteResult) {
	_, r.Err = c.Delete(deleteURL(c, id), nil)
	return
}

// ListOptsBuilder allows extensions to add additional parameters to the
// List request.
type ListOptsBuilder interface {
	ToTrunkListQuery() (string, error)
}

// ListOpts allows the filtering and sorting of paginated collections through
// the API. Filtering is achieved by passing in struct field values that map to
// the trunk attributes you want to see returned. SortKey allows you to sort
// by a particular trunk attribute. SortDir sets the direction, and is either
// `asc' or `desc'. Marker and Limit are used for pagination.
type ListOpts struct {
	AdminStateUp   *bool  `q:"admin_state_up"`
	Description    string `q:"description"`
	ID             string `q:"id"`
	Name           string `q:"name"`
	PortID         string `q:"port_id"`
	RevisionNumber string `q:"revision_number"`
	Status         string `q:"status"`
	TenantID       string `q:"tenant_id"`
	ProjectID      string `q:"project_id"`
	SortDir        string `q:"sort_dir"`
	SortKey        string `q:"sort_key"`
	Tags           string `q:"tags"`
	TagsAny        string `q:"tags-any"`
	NotTags        string `q:"not-tags"`
	NotTagsAny     string `q:"not-tags-any"`
}

// ToTrunkListQuery formats a ListOpts into a query string.
func (opts ListOpts) ToTrunkListQuery() (string, error) {
	q, err := gophercloud.BuildQueryString(opts)
	return q.String(), err
}

// List returns a Pager which allows you to iterate over a collection of
// trunks. It accepts a ListOpts struct, which allows you to filter and sort
// the returned collection for greater efficiency.
//
// Default policy settings return only those trunks that are owned by the tenant
// who submits the request, unless the request is submitted by a user with
// administrative rights.
func List(c *gophercloud.ServiceClient, opts ListOptsBuilder) pagination.Pager {
	url := listURL(c)
	if opts != nil {
		query, err := opts.ToTrunkListQuery()
		if err != nil {
			return pagination.Pager{Err: err}
		}
		url += query
	}
	return pagination.NewPager(c, url, func(r pagination.PageResult) pagination.Page {
		return TrunkPage{pagination.LinkedPageBase{PageResult: r}}
	})
}

// Get retrieves a specific trunk based on its unique ID.
func Get(c *gophercloud.ServiceClient, id string) (r GetResult) {
	_, r.Err = c.Get(getURL(c, id), &r.Body, nil)
	return
}

type UpdateOptsBuilder interface {
	ToTrunkUpdateMap() (map[string]interface{}, error)
}

type UpdateOpts struct {
	AdminStateUp *bool   `json:"admin_state_up,omitempty"`
	Name         *string `json:"name,omitempty"`
	Description  *string `json:"description,omitempty"`
}

func (opts UpdateOpts) ToTrunkUpdateMap() (map[string]interface{}, error) {
	return gophercloud.BuildRequestBody(opts, "trunk")
}

func Update(c *gophercloud.ServiceClient, id string, opts UpdateOptsBuilder) (r UpdateResult) {
	body, err := opts.ToTrunkUpdateMap()
	if err != nil {
		r.Err = err
		return
	}
	_, r.Err = c.Put(updateURL(c, id), body, &r.Body, &gophercloud.RequestOpts{
		OkCodes: []int{200},
	})
	return
}

func GetSubports(c *gophercloud.ServiceClient, id string) (r GetSubportsResult) {
	_, r.Err = c.Get(getSubportsURL(c, id), &r.Body, &gophercloud.RequestOpts{
		OkCodes: []int{200},
	})
	return
}

type AddSubportsOpts struct {
	Subports []Subport `json:"sub_ports" required:"true"`
}

type AddSubportsOptsBuilder interface {
	ToTrunkAddSubportsMap() (map[string]interface{}, error)
}

func (opts AddSubportsOpts) ToTrunkAddSubportsMap() (map[string]interface{}, error) {
	return gophercloud.BuildRequestBody(opts, "")
}

func AddSubports(c *gophercloud.ServiceClient, id string, opts AddSubportsOptsBuilder) (r UpdateSubportsResult) {
	body, err := opts.ToTrunkAddSubportsMap()
	if err != nil {
		r.Err = err
		return
	}
	_, r.Err = c.Put(addSubportsURL(c, id), body, &r.Body, &gophercloud.RequestOpts{
		OkCodes: []int{200},
	})
	return
}

type RemoveSubport struct {
	PortID string `json:"port_id" required:"true"`
}

type RemoveSubportsOpts struct {
	Subports []RemoveSubport `json:"sub_ports"`
}

type RemoveSubportsOptsBuilder interface {
	ToTrunkRemoveSubportsMap() (map[string]interface{}, error)
}

func (opts RemoveSubportsOpts) ToTrunkRemoveSubportsMap() (map[string]interface{}, error) {
	return gophercloud.BuildRequestBody(opts, "")
}

func RemoveSubports(c *gophercloud.ServiceClient, id string, opts RemoveSubportsOptsBuilder) (r UpdateSubportsResult) {
	body, err := opts.ToTrunkRemoveSubportsMap()
	if err != nil {
		r.Err = err
		return
	}
	_, r.Err = c.Put(removeSubportsURL(c, id), body, &r.Body, &gophercloud.RequestOpts{
		OkCodes: []int{200},
	})
	return
}
