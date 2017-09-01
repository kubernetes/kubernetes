package storage

import (
	"encoding/xml"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strconv"
	"strings"
	"time"
)

// Container represents an Azure container.
type Container struct {
	bsc        *BlobStorageClient
	Name       string              `xml:"Name"`
	Properties ContainerProperties `xml:"Properties"`
	Metadata   map[string]string
}

func (c *Container) buildPath() string {
	return fmt.Sprintf("/%s", c.Name)
}

// ContainerProperties contains various properties of a container returned from
// various endpoints like ListContainers.
type ContainerProperties struct {
	LastModified  string `xml:"Last-Modified"`
	Etag          string `xml:"Etag"`
	LeaseStatus   string `xml:"LeaseStatus"`
	LeaseState    string `xml:"LeaseState"`
	LeaseDuration string `xml:"LeaseDuration"`
}

// ContainerListResponse contains the response fields from
// ListContainers call.
//
// See https://msdn.microsoft.com/en-us/library/azure/dd179352.aspx
type ContainerListResponse struct {
	XMLName    xml.Name    `xml:"EnumerationResults"`
	Xmlns      string      `xml:"xmlns,attr"`
	Prefix     string      `xml:"Prefix"`
	Marker     string      `xml:"Marker"`
	NextMarker string      `xml:"NextMarker"`
	MaxResults int64       `xml:"MaxResults"`
	Containers []Container `xml:"Containers>Container"`
}

// BlobListResponse contains the response fields from ListBlobs call.
//
// See https://msdn.microsoft.com/en-us/library/azure/dd135734.aspx
type BlobListResponse struct {
	XMLName    xml.Name `xml:"EnumerationResults"`
	Xmlns      string   `xml:"xmlns,attr"`
	Prefix     string   `xml:"Prefix"`
	Marker     string   `xml:"Marker"`
	NextMarker string   `xml:"NextMarker"`
	MaxResults int64    `xml:"MaxResults"`
	Blobs      []Blob   `xml:"Blobs>Blob"`

	// BlobPrefix is used to traverse blobs as if it were a file system.
	// It is returned if ListBlobsParameters.Delimiter is specified.
	// The list here can be thought of as "folders" that may contain
	// other folders or blobs.
	BlobPrefixes []string `xml:"Blobs>BlobPrefix>Name"`

	// Delimiter is used to traverse blobs as if it were a file system.
	// It is returned if ListBlobsParameters.Delimiter is specified.
	Delimiter string `xml:"Delimiter"`
}

// IncludeBlobDataset has options to include in a list blobs operation
type IncludeBlobDataset struct {
	Snapshots        bool
	Metadata         bool
	UncommittedBlobs bool
	Copy             bool
}

// ListBlobsParameters defines the set of customizable
// parameters to make a List Blobs call.
//
// See https://msdn.microsoft.com/en-us/library/azure/dd135734.aspx
type ListBlobsParameters struct {
	Prefix     string
	Delimiter  string
	Marker     string
	Include    *IncludeBlobDataset
	MaxResults uint
	Timeout    uint
	RequestID  string
}

func (p ListBlobsParameters) getParameters() url.Values {
	out := url.Values{}

	if p.Prefix != "" {
		out.Set("prefix", p.Prefix)
	}
	if p.Delimiter != "" {
		out.Set("delimiter", p.Delimiter)
	}
	if p.Marker != "" {
		out.Set("marker", p.Marker)
	}
	if p.Include != nil {
		include := []string{}
		include = addString(include, p.Include.Snapshots, "snapshots")
		include = addString(include, p.Include.Metadata, "metadata")
		include = addString(include, p.Include.UncommittedBlobs, "uncommittedblobs")
		include = addString(include, p.Include.Copy, "copy")
		fullInclude := strings.Join(include, ",")
		out.Set("include", fullInclude)
	}
	if p.MaxResults != 0 {
		out.Set("maxresults", strconv.FormatUint(uint64(p.MaxResults), 10))
	}
	if p.Timeout != 0 {
		out.Set("timeout", strconv.FormatUint(uint64(p.Timeout), 10))
	}

	return out
}

func addString(datasets []string, include bool, text string) []string {
	if include {
		datasets = append(datasets, text)
	}
	return datasets
}

// ContainerAccessType defines the access level to the container from a public
// request.
//
// See https://msdn.microsoft.com/en-us/library/azure/dd179468.aspx and "x-ms-
// blob-public-access" header.
type ContainerAccessType string

// Access options for containers
const (
	ContainerAccessTypePrivate   ContainerAccessType = ""
	ContainerAccessTypeBlob      ContainerAccessType = "blob"
	ContainerAccessTypeContainer ContainerAccessType = "container"
)

// ContainerAccessPolicy represents each access policy in the container ACL.
type ContainerAccessPolicy struct {
	ID         string
	StartTime  time.Time
	ExpiryTime time.Time
	CanRead    bool
	CanWrite   bool
	CanDelete  bool
}

// ContainerPermissions represents the container ACLs.
type ContainerPermissions struct {
	AccessType     ContainerAccessType
	AccessPolicies []ContainerAccessPolicy
}

// ContainerAccessHeader references header used when setting/getting container ACL
const (
	ContainerAccessHeader string = "x-ms-blob-public-access"
)

// GetBlobReference returns a Blob object for the specified blob name.
func (c *Container) GetBlobReference(name string) *Blob {
	return &Blob{
		Container: c,
		Name:      name,
	}
}

// CreateContainerOptions includes the options for a create container operation
type CreateContainerOptions struct {
	Timeout   uint
	Access    ContainerAccessType `header:"x-ms-blob-public-access"`
	RequestID string              `header:"x-ms-client-request-id"`
}

// Create creates a blob container within the storage account
// with given name and access level. Returns error if container already exists.
//
// See https://docs.microsoft.com/en-us/rest/api/storageservices/fileservices/Create-Container
func (c *Container) Create(options *CreateContainerOptions) error {
	resp, err := c.create(options)
	if err != nil {
		return err
	}
	readAndCloseBody(resp.body)
	return checkRespCode(resp.statusCode, []int{http.StatusCreated})
}

// CreateIfNotExists creates a blob container if it does not exist. Returns
// true if container is newly created or false if container already exists.
func (c *Container) CreateIfNotExists(options *CreateContainerOptions) (bool, error) {
	resp, err := c.create(options)
	if resp != nil {
		defer readAndCloseBody(resp.body)
		if resp.statusCode == http.StatusCreated || resp.statusCode == http.StatusConflict {
			return resp.statusCode == http.StatusCreated, nil
		}
	}
	return false, err
}

func (c *Container) create(options *CreateContainerOptions) (*storageResponse, error) {
	query := url.Values{"restype": {"container"}}
	headers := c.bsc.client.getStandardHeaders()
	headers = c.bsc.client.addMetadataToHeaders(headers, c.Metadata)

	if options != nil {
		query = addTimeout(query, options.Timeout)
		headers = mergeHeaders(headers, headersFromStruct(*options))
	}
	uri := c.bsc.client.getEndpoint(blobServiceName, c.buildPath(), query)

	return c.bsc.client.exec(http.MethodPut, uri, headers, nil, c.bsc.auth)
}

// Exists returns true if a container with given name exists
// on the storage account, otherwise returns false.
func (c *Container) Exists() (bool, error) {
	uri := c.bsc.client.getEndpoint(blobServiceName, c.buildPath(), url.Values{"restype": {"container"}})
	headers := c.bsc.client.getStandardHeaders()

	resp, err := c.bsc.client.exec(http.MethodHead, uri, headers, nil, c.bsc.auth)
	if resp != nil {
		defer readAndCloseBody(resp.body)
		if resp.statusCode == http.StatusOK || resp.statusCode == http.StatusNotFound {
			return resp.statusCode == http.StatusOK, nil
		}
	}
	return false, err
}

// SetContainerPermissionOptions includes options for a set container permissions operation
type SetContainerPermissionOptions struct {
	Timeout           uint
	LeaseID           string     `header:"x-ms-lease-id"`
	IfModifiedSince   *time.Time `header:"If-Modified-Since"`
	IfUnmodifiedSince *time.Time `header:"If-Unmodified-Since"`
	RequestID         string     `header:"x-ms-client-request-id"`
}

// SetPermissions sets up container permissions
// See https://docs.microsoft.com/en-us/rest/api/storageservices/fileservices/Set-Container-ACL
func (c *Container) SetPermissions(permissions ContainerPermissions, options *SetContainerPermissionOptions) error {
	body, length, err := generateContainerACLpayload(permissions.AccessPolicies)
	if err != nil {
		return err
	}
	params := url.Values{
		"restype": {"container"},
		"comp":    {"acl"},
	}
	headers := c.bsc.client.getStandardHeaders()
	headers = addToHeaders(headers, ContainerAccessHeader, string(permissions.AccessType))
	headers["Content-Length"] = strconv.Itoa(length)

	if options != nil {
		params = addTimeout(params, options.Timeout)
		headers = mergeHeaders(headers, headersFromStruct(*options))
	}
	uri := c.bsc.client.getEndpoint(blobServiceName, c.buildPath(), params)

	resp, err := c.bsc.client.exec(http.MethodPut, uri, headers, body, c.bsc.auth)
	if err != nil {
		return err
	}
	defer readAndCloseBody(resp.body)

	if err := checkRespCode(resp.statusCode, []int{http.StatusOK}); err != nil {
		return errors.New("Unable to set permissions")
	}

	return nil
}

// GetContainerPermissionOptions includes options for a get container permissions operation
type GetContainerPermissionOptions struct {
	Timeout   uint
	LeaseID   string `header:"x-ms-lease-id"`
	RequestID string `header:"x-ms-client-request-id"`
}

// GetPermissions gets the container permissions as per https://msdn.microsoft.com/en-us/library/azure/dd179469.aspx
// If timeout is 0 then it will not be passed to Azure
// leaseID will only be passed to Azure if populated
func (c *Container) GetPermissions(options *GetContainerPermissionOptions) (*ContainerPermissions, error) {
	params := url.Values{
		"restype": {"container"},
		"comp":    {"acl"},
	}
	headers := c.bsc.client.getStandardHeaders()

	if options != nil {
		params = addTimeout(params, options.Timeout)
		headers = mergeHeaders(headers, headersFromStruct(*options))
	}
	uri := c.bsc.client.getEndpoint(blobServiceName, c.buildPath(), params)

	resp, err := c.bsc.client.exec(http.MethodGet, uri, headers, nil, c.bsc.auth)
	if err != nil {
		return nil, err
	}
	defer resp.body.Close()

	var ap AccessPolicy
	err = xmlUnmarshal(resp.body, &ap.SignedIdentifiersList)
	if err != nil {
		return nil, err
	}
	return buildAccessPolicy(ap, &resp.headers), nil
}

func buildAccessPolicy(ap AccessPolicy, headers *http.Header) *ContainerPermissions {
	// containerAccess. Blob, Container, empty
	containerAccess := headers.Get(http.CanonicalHeaderKey(ContainerAccessHeader))
	permissions := ContainerPermissions{
		AccessType:     ContainerAccessType(containerAccess),
		AccessPolicies: []ContainerAccessPolicy{},
	}

	for _, policy := range ap.SignedIdentifiersList.SignedIdentifiers {
		capd := ContainerAccessPolicy{
			ID:         policy.ID,
			StartTime:  policy.AccessPolicy.StartTime,
			ExpiryTime: policy.AccessPolicy.ExpiryTime,
		}
		capd.CanRead = updatePermissions(policy.AccessPolicy.Permission, "r")
		capd.CanWrite = updatePermissions(policy.AccessPolicy.Permission, "w")
		capd.CanDelete = updatePermissions(policy.AccessPolicy.Permission, "d")

		permissions.AccessPolicies = append(permissions.AccessPolicies, capd)
	}
	return &permissions
}

// DeleteContainerOptions includes options for a delete container operation
type DeleteContainerOptions struct {
	Timeout           uint
	LeaseID           string     `header:"x-ms-lease-id"`
	IfModifiedSince   *time.Time `header:"If-Modified-Since"`
	IfUnmodifiedSince *time.Time `header:"If-Unmodified-Since"`
	RequestID         string     `header:"x-ms-client-request-id"`
}

// Delete deletes the container with given name on the storage
// account. If the container does not exist returns error.
//
// See https://docs.microsoft.com/en-us/rest/api/storageservices/fileservices/delete-container
func (c *Container) Delete(options *DeleteContainerOptions) error {
	resp, err := c.delete(options)
	if err != nil {
		return err
	}
	readAndCloseBody(resp.body)
	return checkRespCode(resp.statusCode, []int{http.StatusAccepted})
}

// DeleteIfExists deletes the container with given name on the storage
// account if it exists. Returns true if container is deleted with this call, or
// false if the container did not exist at the time of the Delete Container
// operation.
//
// See https://docs.microsoft.com/en-us/rest/api/storageservices/fileservices/delete-container
func (c *Container) DeleteIfExists(options *DeleteContainerOptions) (bool, error) {
	resp, err := c.delete(options)
	if resp != nil {
		defer readAndCloseBody(resp.body)
		if resp.statusCode == http.StatusAccepted || resp.statusCode == http.StatusNotFound {
			return resp.statusCode == http.StatusAccepted, nil
		}
	}
	return false, err
}

func (c *Container) delete(options *DeleteContainerOptions) (*storageResponse, error) {
	query := url.Values{"restype": {"container"}}
	headers := c.bsc.client.getStandardHeaders()

	if options != nil {
		query = addTimeout(query, options.Timeout)
		headers = mergeHeaders(headers, headersFromStruct(*options))
	}
	uri := c.bsc.client.getEndpoint(blobServiceName, c.buildPath(), query)

	return c.bsc.client.exec(http.MethodDelete, uri, headers, nil, c.bsc.auth)
}

// ListBlobs returns an object that contains list of blobs in the container,
// pagination token and other information in the response of List Blobs call.
//
// See https://docs.microsoft.com/en-us/rest/api/storageservices/fileservices/List-Blobs
func (c *Container) ListBlobs(params ListBlobsParameters) (BlobListResponse, error) {
	q := mergeParams(params.getParameters(), url.Values{
		"restype": {"container"},
		"comp":    {"list"}},
	)
	uri := c.bsc.client.getEndpoint(blobServiceName, c.buildPath(), q)

	headers := c.bsc.client.getStandardHeaders()
	headers = addToHeaders(headers, "x-ms-client-request-id", params.RequestID)

	var out BlobListResponse
	resp, err := c.bsc.client.exec(http.MethodGet, uri, headers, nil, c.bsc.auth)
	if err != nil {
		return out, err
	}
	defer resp.body.Close()

	err = xmlUnmarshal(resp.body, &out)
	for i := range out.Blobs {
		out.Blobs[i].Container = c
	}
	return out, err
}

func generateContainerACLpayload(policies []ContainerAccessPolicy) (io.Reader, int, error) {
	sil := SignedIdentifiers{
		SignedIdentifiers: []SignedIdentifier{},
	}
	for _, capd := range policies {
		permission := capd.generateContainerPermissions()
		signedIdentifier := convertAccessPolicyToXMLStructs(capd.ID, capd.StartTime, capd.ExpiryTime, permission)
		sil.SignedIdentifiers = append(sil.SignedIdentifiers, signedIdentifier)
	}
	return xmlMarshal(sil)
}

func (capd *ContainerAccessPolicy) generateContainerPermissions() (permissions string) {
	// generate the permissions string (rwd).
	// still want the end user API to have bool flags.
	permissions = ""

	if capd.CanRead {
		permissions += "r"
	}

	if capd.CanWrite {
		permissions += "w"
	}

	if capd.CanDelete {
		permissions += "d"
	}

	return permissions
}
