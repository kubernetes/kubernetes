package storage

// Copyright 2017 Microsoft Corporation
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.

import (
	"encoding/xml"
	"fmt"
	"net/http"
	"net/url"
	"strconv"
	"strings"
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

// GetContainerReferenceFromSASURI returns a Container object for the specified
// container SASURI
func GetContainerReferenceFromSASURI(sasuri url.URL) (*Container, error) {
	path := strings.Split(sasuri.Path, "/")
	if len(path) <= 1 {
		return nil, fmt.Errorf("could not find a container in URI: %s", sasuri.String())
	}
	cli := newSASClient().GetBlobService()
	return &Container{
		bsc:    &cli,
		Name:   path[1],
		sasuri: sasuri,
	}, nil
}

// ListContainers returns the list of containers in a storage account along with
// pagination token and other response details.
//
// See https://msdn.microsoft.com/en-us/library/azure/dd179352.aspx
func (b BlobStorageClient) ListContainers(params ListContainersParameters) (*ContainerListResponse, error) {
	q := mergeParams(params.getParameters(), url.Values{"comp": {"list"}})
	uri := b.client.getEndpoint(blobServiceName, "", q)
	headers := b.client.getStandardHeaders()

	type ContainerAlias struct {
		bsc        *BlobStorageClient
		Name       string              `xml:"Name"`
		Properties ContainerProperties `xml:"Properties"`
		Metadata   BlobMetadata
		sasuri     url.URL
	}
	type ContainerListResponseAlias struct {
		XMLName    xml.Name         `xml:"EnumerationResults"`
		Xmlns      string           `xml:"xmlns,attr"`
		Prefix     string           `xml:"Prefix"`
		Marker     string           `xml:"Marker"`
		NextMarker string           `xml:"NextMarker"`
		MaxResults int64            `xml:"MaxResults"`
		Containers []ContainerAlias `xml:"Containers>Container"`
	}

	var outAlias ContainerListResponseAlias
	resp, err := b.client.exec(http.MethodGet, uri, headers, nil, b.auth)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	err = xmlUnmarshal(resp.Body, &outAlias)
	if err != nil {
		return nil, err
	}

	out := ContainerListResponse{
		XMLName:    outAlias.XMLName,
		Xmlns:      outAlias.Xmlns,
		Prefix:     outAlias.Prefix,
		Marker:     outAlias.Marker,
		NextMarker: outAlias.NextMarker,
		MaxResults: outAlias.MaxResults,
		Containers: make([]Container, len(outAlias.Containers)),
	}
	for i, cnt := range outAlias.Containers {
		out.Containers[i] = Container{
			bsc:        &b,
			Name:       cnt.Name,
			Properties: cnt.Properties,
			Metadata:   map[string]string(cnt.Metadata),
			sasuri:     cnt.sasuri,
		}
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

func writeMetadata(h http.Header) map[string]string {
	metadata := make(map[string]string)
	for k, v := range h {
		// Can't trust CanonicalHeaderKey() to munge case
		// reliably. "_" is allowed in identifiers:
		// https://msdn.microsoft.com/en-us/library/azure/dd179414.aspx
		// https://msdn.microsoft.com/library/aa664670(VS.71).aspx
		// http://tools.ietf.org/html/rfc7230#section-3.2
		// ...but "_" is considered invalid by
		// CanonicalMIMEHeaderKey in
		// https://golang.org/src/net/textproto/reader.go?s=14615:14659#L542
		// so k can be "X-Ms-Meta-Lol" or "x-ms-meta-lol_rofl".
		k = strings.ToLower(k)
		if len(v) == 0 || !strings.HasPrefix(k, strings.ToLower(userDefinedMetadataHeaderPrefix)) {
			continue
		}
		// metadata["lol"] = content of the last X-Ms-Meta-Lol header
		k = k[len(userDefinedMetadataHeaderPrefix):]
		metadata[k] = v[len(v)-1]
	}
	return metadata
}
