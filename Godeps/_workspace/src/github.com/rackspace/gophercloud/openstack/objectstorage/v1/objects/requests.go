package objects

import (
	"bytes"
	"crypto/hmac"
	"crypto/md5"
	"crypto/sha1"
	"fmt"
	"io"
	"strings"
	"time"

	"github.com/rackspace/gophercloud"
	"github.com/rackspace/gophercloud/openstack/objectstorage/v1/accounts"
	"github.com/rackspace/gophercloud/pagination"
)

// ListOptsBuilder allows extensions to add additional parameters to the List
// request.
type ListOptsBuilder interface {
	ToObjectListParams() (bool, string, error)
}

// ListOpts is a structure that holds parameters for listing objects.
type ListOpts struct {
	// Full is a true/false value that represents the amount of object information
	// returned. If Full is set to true, then the content-type, number of bytes, hash
	// date last modified, and name are returned. If set to false or not set, then
	// only the object names are returned.
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
func (opts DownloadOpts) ToObjectDownloadParams() (map[string]string, string, error) {
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

	resp, err := c.Request("GET", url, gophercloud.RequestOpts{
		MoreHeaders: h,
		OkCodes:     []int{200, 304},
	})
	if resp != nil {
		res.Header = resp.Header
		res.Body = resp.Body
	}
	res.Err = err

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
	ContentLength      int64  `h:"Content-Length"`
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

// Create is a function that creates a new object or replaces an existing object. If the returned response's ETag
// header fails to match the local checksum, the failed request will automatically be retried up to a maximum of 3 times.
func Create(c *gophercloud.ServiceClient, containerName, objectName string, content io.ReadSeeker, opts CreateOptsBuilder) CreateResult {
	var res CreateResult

	url := createURL(c, containerName, objectName)
	h := make(map[string]string)

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

	hash := md5.New()

	contentBuffer := bytes.NewBuffer([]byte{})
	_, err := io.Copy(contentBuffer, io.TeeReader(content, hash))
	if err != nil {
		res.Err = err
		return res
	}

	localChecksum := hash.Sum(nil)
	h["ETag"] = fmt.Sprintf("%x", localChecksum)

	ropts := gophercloud.RequestOpts{
		RawBody:     strings.NewReader(contentBuffer.String()),
		MoreHeaders: h,
	}

	resp, err := c.Request("PUT", url, ropts)
	if err != nil {
		res.Err = err
		return res
	}
	if resp != nil {
		res.Header = resp.Header
		if resp.Header.Get("ETag") == fmt.Sprintf("%x", localChecksum) {
			res.Err = err
			return res
		}
		res.Err = fmt.Errorf("Local checksum does not match API ETag header")
	}

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
	resp, err := c.Request("COPY", url, gophercloud.RequestOpts{
		MoreHeaders: h,
		OkCodes:     []int{201},
	})
	if resp != nil {
		res.Header = resp.Header
	}
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

	resp, err := c.Delete(url, nil)
	if resp != nil {
		res.Header = resp.Header
	}
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

	resp, err := c.Request("HEAD", url, gophercloud.RequestOpts{
		OkCodes: []int{200, 204},
	})
	if resp != nil {
		res.Header = resp.Header
	}
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
	resp, err := c.Request("POST", url, gophercloud.RequestOpts{
		MoreHeaders: h,
	})
	if resp != nil {
		res.Header = resp.Header
	}
	res.Err = err
	return res
}

// HTTPMethod represents an HTTP method string (e.g. "GET").
type HTTPMethod string

var (
	// GET represents an HTTP "GET" method.
	GET HTTPMethod = "GET"
	// POST represents an HTTP "POST" method.
	POST HTTPMethod = "POST"
)

// CreateTempURLOpts are options for creating a temporary URL for an object.
type CreateTempURLOpts struct {
	// (REQUIRED) Method is the HTTP method to allow for users of the temp URL. Valid values
	// are "GET" and "POST".
	Method HTTPMethod
	// (REQUIRED) TTL is the number of seconds the temp URL should be active.
	TTL int
	// (Optional) Split is the string on which to split the object URL. Since only
	// the object path is used in the hash, the object URL needs to be parsed. If
	// empty, the default OpenStack URL split point will be used ("/v1/").
	Split string
}

// CreateTempURL is a function for creating a temporary URL for an object. It
// allows users to have "GET" or "POST" access to a particular tenant's object
// for a limited amount of time.
func CreateTempURL(c *gophercloud.ServiceClient, containerName, objectName string, opts CreateTempURLOpts) (string, error) {
	if opts.Split == "" {
		opts.Split = "/v1/"
	}
	duration := time.Duration(opts.TTL) * time.Second
	expiry := time.Now().Add(duration).Unix()
	getHeader, err := accounts.Get(c, nil).Extract()
	if err != nil {
		return "", err
	}
	secretKey := []byte(getHeader.TempURLKey)
	url := getURL(c, containerName, objectName)
	splitPath := strings.Split(url, opts.Split)
	baseURL, objectPath := splitPath[0], splitPath[1]
	objectPath = opts.Split + objectPath
	body := fmt.Sprintf("%s\n%d\n%s", opts.Method, expiry, objectPath)
	hash := hmac.New(sha1.New, secretKey)
	hash.Write([]byte(body))
	hexsum := fmt.Sprintf("%x", hash.Sum(nil))
	return fmt.Sprintf("%s%s?temp_url_sig=%s&temp_url_expires=%d", baseURL, objectPath, hexsum, expiry), nil
}
