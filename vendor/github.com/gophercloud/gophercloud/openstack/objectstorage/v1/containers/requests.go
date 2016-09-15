package containers

import (
	"github.com/gophercloud/gophercloud"
	"github.com/gophercloud/gophercloud/pagination"
)

// ListOptsBuilder allows extensions to add additional parameters to the List
// request.
type ListOptsBuilder interface {
	ToContainerListParams() (bool, string, error)
}

// ListOpts is a structure that holds options for listing containers.
type ListOpts struct {
	Full      bool
	Limit     int    `q:"limit"`
	Marker    string `q:"marker"`
	EndMarker string `q:"end_marker"`
	Format    string `q:"format"`
	Prefix    string `q:"prefix"`
	Delimiter string `q:"delimiter"`
}

// ToContainerListParams formats a ListOpts into a query string and boolean
// representing whether to list complete information for each container.
func (opts ListOpts) ToContainerListParams() (bool, string, error) {
	q, err := gophercloud.BuildQueryString(opts)
	return opts.Full, q.String(), err
}

// List is a function that retrieves containers associated with the account as
// well as account metadata. It returns a pager which can be iterated with the
// EachPage function.
func List(c *gophercloud.ServiceClient, opts ListOptsBuilder) pagination.Pager {
	headers := map[string]string{"Accept": "text/plain", "Content-Type": "text/plain"}

	url := listURL(c)
	if opts != nil {
		full, query, err := opts.ToContainerListParams()
		if err != nil {
			return pagination.Pager{Err: err}
		}
		url += query

		if full {
			headers = map[string]string{"Accept": "application/json", "Content-Type": "application/json"}
		}
	}

	pager := pagination.NewPager(c, url, func(r pagination.PageResult) pagination.Page {
		p := ContainerPage{pagination.MarkerPageBase{PageResult: r}}
		p.MarkerPageBase.Owner = p
		return p
	})
	pager.Headers = headers
	return pager
}

// CreateOptsBuilder allows extensions to add additional parameters to the
// Create request.
type CreateOptsBuilder interface {
	ToContainerCreateMap() (map[string]string, error)
}

// CreateOpts is a structure that holds parameters for creating a container.
type CreateOpts struct {
	Metadata          map[string]string
	ContainerRead     string `h:"X-Container-Read"`
	ContainerSyncTo   string `h:"X-Container-Sync-To"`
	ContainerSyncKey  string `h:"X-Container-Sync-Key"`
	ContainerWrite    string `h:"X-Container-Write"`
	ContentType       string `h:"Content-Type"`
	DetectContentType bool   `h:"X-Detect-Content-Type"`
	IfNoneMatch       string `h:"If-None-Match"`
	VersionsLocation  string `h:"X-Versions-Location"`
}

// ToContainerCreateMap formats a CreateOpts into a map of headers.
func (opts CreateOpts) ToContainerCreateMap() (map[string]string, error) {
	h, err := gophercloud.BuildHeaders(opts)
	if err != nil {
		return nil, err
	}
	for k, v := range opts.Metadata {
		h["X-Container-Meta-"+k] = v
	}
	return h, nil
}

// Create is a function that creates a new container.
func Create(c *gophercloud.ServiceClient, containerName string, opts CreateOptsBuilder) (r CreateResult) {
	h := make(map[string]string)
	if opts != nil {
		headers, err := opts.ToContainerCreateMap()
		if err != nil {
			r.Err = err
			return
		}
		for k, v := range headers {
			h[k] = v
		}
	}
	resp, err := c.Request("PUT", createURL(c, containerName), &gophercloud.RequestOpts{
		MoreHeaders: h,
		OkCodes:     []int{201, 202, 204},
	})
	if resp != nil {
		r.Header = resp.Header
	}
	r.Err = err
	return
}

// Delete is a function that deletes a container.
func Delete(c *gophercloud.ServiceClient, containerName string) (r DeleteResult) {
	_, r.Err = c.Delete(deleteURL(c, containerName), nil)
	return
}

// UpdateOptsBuilder allows extensions to add additional parameters to the
// Update request.
type UpdateOptsBuilder interface {
	ToContainerUpdateMap() (map[string]string, error)
}

// UpdateOpts is a structure that holds parameters for updating, creating, or
// deleting a container's metadata.
type UpdateOpts struct {
	Metadata               map[string]string
	ContainerRead          string `h:"X-Container-Read"`
	ContainerSyncTo        string `h:"X-Container-Sync-To"`
	ContainerSyncKey       string `h:"X-Container-Sync-Key"`
	ContainerWrite         string `h:"X-Container-Write"`
	ContentType            string `h:"Content-Type"`
	DetectContentType      bool   `h:"X-Detect-Content-Type"`
	RemoveVersionsLocation string `h:"X-Remove-Versions-Location"`
	VersionsLocation       string `h:"X-Versions-Location"`
}

// ToContainerUpdateMap formats a CreateOpts into a map of headers.
func (opts UpdateOpts) ToContainerUpdateMap() (map[string]string, error) {
	h, err := gophercloud.BuildHeaders(opts)
	if err != nil {
		return nil, err
	}
	for k, v := range opts.Metadata {
		h["X-Container-Meta-"+k] = v
	}
	return h, nil
}

// Update is a function that creates, updates, or deletes a container's
// metadata.
func Update(c *gophercloud.ServiceClient, containerName string, opts UpdateOptsBuilder) (r UpdateResult) {
	h := make(map[string]string)
	if opts != nil {
		headers, err := opts.ToContainerUpdateMap()
		if err != nil {
			r.Err = err
			return
		}

		for k, v := range headers {
			h[k] = v
		}
	}
	resp, err := c.Request("POST", updateURL(c, containerName), &gophercloud.RequestOpts{
		MoreHeaders: h,
		OkCodes:     []int{201, 202, 204},
	})
	if resp != nil {
		r.Header = resp.Header
	}
	r.Err = err
	return
}

// Get is a function that retrieves the metadata of a container. To extract just
// the custom metadata, pass the GetResult response to the ExtractMetadata
// function.
func Get(c *gophercloud.ServiceClient, containerName string) (r GetResult) {
	resp, err := c.Request("HEAD", getURL(c, containerName), &gophercloud.RequestOpts{
		OkCodes: []int{200, 204},
	})
	if resp != nil {
		r.Header = resp.Header
	}
	r.Err = err
	return
}
