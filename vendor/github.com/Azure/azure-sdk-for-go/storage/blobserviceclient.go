package storage

import (
	"net/http"
	"net/url"
	"strconv"
)

// BlobStorageClient contains operations for Microsoft Azure Blob Storage
// Service.
type BlobStorageClient struct {
	client Client
	auth   authentication
}

// GetServiceProperties gets the properties of your storage account's blob service.
// See: https://docs.microsoft.com/en-us/rest/api/storageservices/fileservices/get-blob-service-properties
func (b *BlobStorageClient) GetServiceProperties() (*ServiceProperties, error) {
	return b.client.getServiceProperties(blobServiceName, b.auth)
}

// SetServiceProperties sets the properties of your storage account's blob service.
// See: https://docs.microsoft.com/en-us/rest/api/storageservices/fileservices/set-blob-service-properties
func (b *BlobStorageClient) SetServiceProperties(props ServiceProperties) error {
	return b.client.setServiceProperties(props, blobServiceName, b.auth)
}

// ListContainersParameters defines the set of customizable parameters to make a
// List Containers call.
//
// See https://msdn.microsoft.com/en-us/library/azure/dd179352.aspx
type ListContainersParameters struct {
	Prefix     string
	Marker     string
	Include    string
	MaxResults uint
	Timeout    uint
}

// GetContainerReference returns a Container object for the specified container name.
func (b *BlobStorageClient) GetContainerReference(name string) *Container {
	return &Container{
		bsc:  b,
		Name: name,
	}
}

// ListContainers returns the list of containers in a storage account along with
// pagination token and other response details.
//
// See https://msdn.microsoft.com/en-us/library/azure/dd179352.aspx
func (b BlobStorageClient) ListContainers(params ListContainersParameters) (*ContainerListResponse, error) {
	q := mergeParams(params.getParameters(), url.Values{"comp": {"list"}})
	uri := b.client.getEndpoint(blobServiceName, "", q)
	headers := b.client.getStandardHeaders()

	var out ContainerListResponse
	resp, err := b.client.exec(http.MethodGet, uri, headers, nil, b.auth)
	if err != nil {
		return nil, err
	}
	defer resp.body.Close()
	err = xmlUnmarshal(resp.body, &out)
	if err != nil {
		return nil, err
	}

	// assign our client to the newly created Container objects
	for i := range out.Containers {
		out.Containers[i].bsc = &b
	}
	return &out, err
}

func (p ListContainersParameters) getParameters() url.Values {
	out := url.Values{}

	if p.Prefix != "" {
		out.Set("prefix", p.Prefix)
	}
	if p.Marker != "" {
		out.Set("marker", p.Marker)
	}
	if p.Include != "" {
		out.Set("include", p.Include)
	}
	if p.MaxResults != 0 {
		out.Set("maxresults", strconv.FormatUint(uint64(p.MaxResults), 10))
	}
	if p.Timeout != 0 {
		out.Set("timeout", strconv.FormatUint(uint64(p.Timeout), 10))
	}

	return out
}
