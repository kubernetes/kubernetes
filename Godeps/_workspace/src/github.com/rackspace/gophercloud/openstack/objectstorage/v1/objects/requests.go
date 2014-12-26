package objects

import (
	"fmt"
	"io"
	"time"

	"github.com/racker/perigee"
	"github.com/rackspace/gophercloud"
	"github.com/rackspace/gophercloud/pagination"
)

// ListOptsBuilder allows extensions to add additional parameters to the List
// request.
type ListOptsBuilder interface {
	ToObjectListParams() (bool, string, error)
}

// ListOpts is a structure that holds parameters for listing objects.
type ListOpts struct {
	Full      bool
	Limit     int    `q:"limit"`
	Marker    string `q:"marker"`
	EndMarker string `q:"end_marker"`
	Format    string `q:"format"`
	Prefix    string `q:"prefix"`
	Delimiter string `q:"delimiter"`
	Path      string `q:"path"`
}

// ToObjectListParams formats a ListOpts into a query string and boolean
// representing whether to list complete information for each object.
func (opts ListOpts) ToObjectListParams() (bool, string, error) {
	q, err := gophercloud.BuildQueryString(opts)
	if err != nil {
		return false, "", err
	}
	return opts.Full, q.String(), nil
}

// List is a function that retrieves all objects in a container. It also returns the details
// for the container. To extract only the object information or names, pass the ListResult
// response to the ExtractInfo or ExtractNames function, respectively.
func List(c *gophercloud.ServiceClient, containerName string, opts ListOptsBuilder) pagination.Pager {
	headers := map[string]string{"Accept": "text/plain", "Content-Type": "text/plain"}

	url := listURL(c, containerName)
	if opts != nil {
		full, query, err := opts.ToObjectListParams()
		if err != nil {
			return pagination.Pager{Err: err}
		}
		url += query

		if full {
			headers = map[string]string{"Accept": "application/json", "Content-Type": "application/json"}
		}
	}

	createPage := func(r pagination.PageResult) pagination.Page {
		p := ObjectPage{pagination.MarkerPageBase{PageResult: r}}
		p.MarkerPageBase.Owner = p
		return p
	}

	pager := pagination.NewPager(c, url, createPage)
	pager.Headers = headers
	return pager
}

// DownloadOptsBuilder allows extensions to add additional parameters to the
// Download request.
type DownloadOptsBuilder interface {
	ToObjectDownloadParams() (map[string]string, string, error)
}

// DownloadOpts is a structure that holds parameters for downloading an object.
type DownloadOpts struct {
	IfMatch           string    `h:"If-Match"`
	IfModifiedSince   time.Time `h:"If-Modified-Since"`
	IfNoneMatch       string    `h:"If-None-Match"`
	IfUnmodifiedSince time.Time `h:"If-Unmodified-Since"`
	Range             string    `h:"Range"`
	Expires           string    `q:"expires"`
	MultipartManifest string    `q:"multipart-manifest"`
	Signature         string    `q:"signature"`
}

// ToObjectDownloadParams formats a DownloadOpts into a query string and map of
// headers.
func (opts ListOpts) ToObjectDownloadParams() (map[string]string, string, error) {
	q, err := gophercloud.BuildQueryString(opts)
	if err != nil {
		return nil, "", err
	}
	h, err := gophercloud.BuildHeaders(opts)
	if err != nil {
		return nil, q.String(), err
	}
	return h, q.String(), nil
}

// Download is a function that retrieves the content and metadata for an object.
// To extract just the content, pass the DownloadResult response to the
// ExtractContent function.
func Download(c *gophercloud.ServiceClient, containerName, objectName string, opts DownloadOptsBuilder) DownloadResult {
	var res DownloadResult

	url := downloadURL(c, containerName, objectName)
	h := c.AuthenticatedHeaders()

	if opts != nil {
		headers, query, err := opts.ToObjectDownloadParams()
		if err != nil {
			res.Err = err
			return res
		}

		for k, v := range headers {
			h[k] = v
		}

		url += query
	}

	resp, err := perigee.Request("GET", url, perigee.Options{
		MoreHeaders: h,
		OkCodes:     []int{200},
	})

	res.Body = resp.HttpResponse.Body
	res.Err = err
	res.Header = resp.HttpResponse.Header

	return res
}

// CreateOptsBuilder allows extensions to add additional parameters to the
// Create request.
type CreateOptsBuilder interface {
	ToObjectCreateParams() (map[string]string, string, error)
}

// CreateOpts is a structure that holds parameters for creating an object.
type CreateOpts struct {
	Metadata           map[string]string
	ContentDisposition string `h:"Content-Disposition"`
	ContentEncoding    string `h:"Content-Encoding"`
	ContentLength      int    `h:"Content-Length"`
	ContentType        string `h:"Content-Type"`
	CopyFrom           string `h:"X-Copy-From"`
	DeleteAfter        int    `h:"X-Delete-After"`
	DeleteAt           int    `h:"X-Delete-At"`
	DetectContentType  string `h:"X-Detect-Content-Type"`
	ETag               string `h:"ETag"`
	IfNoneMatch        string `h:"If-None-Match"`
	ObjectManifest     string `h:"X-Object-Manifest"`
	TransferEncoding   string `h:"Transfer-Encoding"`
	Expires            string `q:"expires"`
	MultipartManifest  string `q:"multiple-manifest"`
	Signature          string `q:"signature"`
}

// ToObjectCreateParams formats a CreateOpts into a query string and map of
// headers.
func (opts CreateOpts) ToObjectCreateParams() (map[string]string, string, error) {
	q, err := gophercloud.BuildQueryString(opts)
	if err != nil {
		return nil, "", err
	}
	h, err := gophercloud.BuildHeaders(opts)
	if err != nil {
		return nil, q.String(), err
	}

	for k, v := range opts.Metadata {
		h["X-Object-Meta-"+k] = v
	}

	return h, q.String(), nil
}

// Create is a function that creates a new object or replaces an existing object.
func Create(c *gophercloud.ServiceClient, containerName, objectName string, content io.Reader, opts CreateOptsBuilder) CreateResult {
	var res CreateResult

	url := createURL(c, containerName, objectName)
	h := c.AuthenticatedHeaders()

	if opts != nil {
		headers, query, err := opts.ToObjectCreateParams()
		if err != nil {
			res.Err = err
			return res
		}

		for k, v := range headers {
			h[k] = v
		}

		url += query
	}

	contentType := h["Content-Type"]

	resp, err := perigee.Request("PUT", url, perigee.Options{
		ContentType: contentType,
		ReqBody:     content,
		MoreHeaders: h,
		OkCodes:     []int{201, 202},
	})
	res.Header = resp.HttpResponse.Header
	res.Err = err
	return res
}

// CopyOptsBuilder allows extensions to add additional parameters to the
// Copy request.
type CopyOptsBuilder interface {
	ToObjectCopyMap() (map[string]string, error)
}

// CopyOpts is a structure that holds parameters for copying one object to
// another.
type CopyOpts struct {
	Metadata           map[string]string
	ContentDisposition string `h:"Content-Disposition"`
	ContentEncoding    string `h:"Content-Encoding"`
	ContentType        string `h:"Content-Type"`
	Destination        string `h:"Destination,required"`
}

// ToObjectCopyMap formats a CopyOpts into a map of headers.
func (opts CopyOpts) ToObjectCopyMap() (map[string]string, error) {
	if opts.Destination == "" {
		return nil, fmt.Errorf("Required CopyOpts field 'Destination' not set.")
	}
	h, err := gophercloud.BuildHeaders(opts)
	if err != nil {
		return nil, err
	}
	for k, v := range opts.Metadata {
		h["X-Object-Meta-"+k] = v
	}
	return h, nil
}

// Copy is a function that copies one object to another.
func Copy(c *gophercloud.ServiceClient, containerName, objectName string, opts CopyOptsBuilder) CopyResult {
	var res CopyResult
	h := c.AuthenticatedHeaders()

	headers, err := opts.ToObjectCopyMap()
	if err != nil {
		res.Err = err
		return res
	}

	for k, v := range headers {
		h[k] = v
	}

	url := copyURL(c, containerName, objectName)
	resp, err := perigee.Request("COPY", url, perigee.Options{
		MoreHeaders: h,
		OkCodes:     []int{201},
	})
	res.Header = resp.HttpResponse.Header
	res.Err = err
	return res
}

// DeleteOptsBuilder allows extensions to add additional parameters to the
// Delete request.
type DeleteOptsBuilder interface {
	ToObjectDeleteQuery() (string, error)
}

// DeleteOpts is a structure that holds parameters for deleting an object.
type DeleteOpts struct {
	MultipartManifest string `q:"multipart-manifest"`
}

// ToObjectDeleteQuery formats a DeleteOpts into a query string.
func (opts DeleteOpts) ToObjectDeleteQuery() (string, error) {
	q, err := gophercloud.BuildQueryString(opts)
	if err != nil {
		return "", err
	}
	return q.String(), nil
}

// Delete is a function that deletes an object.
func Delete(c *gophercloud.ServiceClient, containerName, objectName string, opts DeleteOptsBuilder) DeleteResult {
	var res DeleteResult
	url := deleteURL(c, containerName, objectName)

	if opts != nil {
		query, err := opts.ToObjectDeleteQuery()
		if err != nil {
			res.Err = err
			return res
		}
		url += query
	}

	resp, err := perigee.Request("DELETE", url, perigee.Options{
		MoreHeaders: c.AuthenticatedHeaders(),
		OkCodes:     []int{204},
	})
	res.Header = resp.HttpResponse.Header
	res.Err = err
	return res
}

// GetOptsBuilder allows extensions to add additional parameters to the
// Get request.
type GetOptsBuilder interface {
	ToObjectGetQuery() (string, error)
}

// GetOpts is a structure that holds parameters for getting an object's metadata.
type GetOpts struct {
	Expires   string `q:"expires"`
	Signature string `q:"signature"`
}

// ToObjectGetQuery formats a GetOpts into a query string.
func (opts GetOpts) ToObjectGetQuery() (string, error) {
	q, err := gophercloud.BuildQueryString(opts)
	if err != nil {
		return "", err
	}
	return q.String(), nil
}

// Get is a function that retrieves the metadata of an object. To extract just the custom
// metadata, pass the GetResult response to the ExtractMetadata function.
func Get(c *gophercloud.ServiceClient, containerName, objectName string, opts GetOptsBuilder) GetResult {
	var res GetResult
	url := getURL(c, containerName, objectName)

	if opts != nil {
		query, err := opts.ToObjectGetQuery()
		if err != nil {
			res.Err = err
			return res
		}
		url += query
	}

	resp, err := perigee.Request("HEAD", url, perigee.Options{
		MoreHeaders: c.AuthenticatedHeaders(),
		OkCodes:     []int{200, 204},
	})
	res.Header = resp.HttpResponse.Header
	res.Err = err
	return res
}

// UpdateOptsBuilder allows extensions to add additional parameters to the
// Update request.
type UpdateOptsBuilder interface {
	ToObjectUpdateMap() (map[string]string, error)
}

// UpdateOpts is a structure that holds parameters for updating, creating, or deleting an
// object's metadata.
type UpdateOpts struct {
	Metadata           map[string]string
	ContentDisposition string `h:"Content-Disposition"`
	ContentEncoding    string `h:"Content-Encoding"`
	ContentType        string `h:"Content-Type"`
	DeleteAfter        int    `h:"X-Delete-After"`
	DeleteAt           int    `h:"X-Delete-At"`
	DetectContentType  bool   `h:"X-Detect-Content-Type"`
}

// ToObjectUpdateMap formats a UpdateOpts into a map of headers.
func (opts UpdateOpts) ToObjectUpdateMap() (map[string]string, error) {
	h, err := gophercloud.BuildHeaders(opts)
	if err != nil {
		return nil, err
	}
	for k, v := range opts.Metadata {
		h["X-Object-Meta-"+k] = v
	}
	return h, nil
}

// Update is a function that creates, updates, or deletes an object's metadata.
func Update(c *gophercloud.ServiceClient, containerName, objectName string, opts UpdateOptsBuilder) UpdateResult {
	var res UpdateResult
	h := c.AuthenticatedHeaders()

	if opts != nil {
		headers, err := opts.ToObjectUpdateMap()
		if err != nil {
			res.Err = err
			return res
		}

		for k, v := range headers {
			h[k] = v
		}
	}

	url := updateURL(c, containerName, objectName)
	resp, err := perigee.Request("POST", url, perigee.Options{
		MoreHeaders: h,
		OkCodes:     []int{202},
	})
	res.Header = resp.HttpResponse.Header
	res.Err = err
	return res
}
