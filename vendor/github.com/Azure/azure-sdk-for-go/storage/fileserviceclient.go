package storage

// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See License.txt in the project root for license information.

import (
	"encoding/xml"
	"fmt"
	"net/http"
	"net/url"
	"strconv"
)

// FileServiceClient contains operations for Microsoft Azure File Service.
type FileServiceClient struct {
	client Client
	auth   authentication
}

// ListSharesParameters defines the set of customizable parameters to make a
// List Shares call.
//
// See https://docs.microsoft.com/en-us/rest/api/storageservices/fileservices/List-Shares
type ListSharesParameters struct {
	Prefix     string
	Marker     string
	Include    string
	MaxResults uint
	Timeout    uint
}

// ShareListResponse contains the response fields from
// ListShares call.
//
// See https://docs.microsoft.com/en-us/rest/api/storageservices/fileservices/List-Shares
type ShareListResponse struct {
	XMLName    xml.Name `xml:"EnumerationResults"`
	Xmlns      string   `xml:"xmlns,attr"`
	Prefix     string   `xml:"Prefix"`
	Marker     string   `xml:"Marker"`
	NextMarker string   `xml:"NextMarker"`
	MaxResults int64    `xml:"MaxResults"`
	Shares     []Share  `xml:"Shares>Share"`
}

type compType string

const (
	compNone       compType = ""
	compList       compType = "list"
	compMetadata   compType = "metadata"
	compProperties compType = "properties"
	compRangeList  compType = "rangelist"
)

func (ct compType) String() string {
	return string(ct)
}

type resourceType string

const (
	resourceDirectory resourceType = "directory"
	resourceFile      resourceType = ""
	resourceShare     resourceType = "share"
)

func (rt resourceType) String() string {
	return string(rt)
}

func (p ListSharesParameters) getParameters() url.Values {
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

func (p ListDirsAndFilesParameters) getParameters() url.Values {
	out := url.Values{}

	if p.Prefix != "" {
		out.Set("prefix", p.Prefix)
	}
	if p.Marker != "" {
		out.Set("marker", p.Marker)
	}
	if p.MaxResults != 0 {
		out.Set("maxresults", strconv.FormatUint(uint64(p.MaxResults), 10))
	}
	out = addTimeout(out, p.Timeout)

	return out
}

// returns url.Values for the specified types
func getURLInitValues(comp compType, res resourceType) url.Values {
	values := url.Values{}
	if comp != compNone {
		values.Set("comp", comp.String())
	}
	if res != resourceFile {
		values.Set("restype", res.String())
	}
	return values
}

// GetShareReference returns a Share object for the specified share name.
func (f *FileServiceClient) GetShareReference(name string) *Share {
	return &Share{
		fsc:  f,
		Name: name,
		Properties: ShareProperties{
			Quota: -1,
		},
	}
}

// ListShares returns the list of shares in a storage account along with
// pagination token and other response details.
//
// See https://docs.microsoft.com/en-us/rest/api/storageservices/fileservices/list-shares
func (f FileServiceClient) ListShares(params ListSharesParameters) (*ShareListResponse, error) {
	q := mergeParams(params.getParameters(), url.Values{"comp": {"list"}})

	var out ShareListResponse
	resp, err := f.listContent("", q, nil)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	err = xmlUnmarshal(resp.Body, &out)

	// assign our client to the newly created Share objects
	for i := range out.Shares {
		out.Shares[i].fsc = &f
	}
	return &out, err
}

// GetServiceProperties gets the properties of your storage account's file service.
// File service does not support logging
// See: https://docs.microsoft.com/en-us/rest/api/storageservices/fileservices/get-file-service-properties
func (f *FileServiceClient) GetServiceProperties() (*ServiceProperties, error) {
	return f.client.getServiceProperties(fileServiceName, f.auth)
}

// SetServiceProperties sets the properties of your storage account's file service.
// File service does not support logging
// See: https://docs.microsoft.com/en-us/rest/api/storageservices/fileservices/set-file-service-properties
func (f *FileServiceClient) SetServiceProperties(props ServiceProperties) error {
	return f.client.setServiceProperties(props, fileServiceName, f.auth)
}

// retrieves directory or share content
func (f FileServiceClient) listContent(path string, params url.Values, extraHeaders map[string]string) (*http.Response, error) {
	if err := f.checkForStorageEmulator(); err != nil {
		return nil, err
	}

	uri := f.client.getEndpoint(fileServiceName, path, params)
	extraHeaders = f.client.protectUserAgent(extraHeaders)
	headers := mergeHeaders(f.client.getStandardHeaders(), extraHeaders)

	resp, err := f.client.exec(http.MethodGet, uri, headers, nil, f.auth)
	if err != nil {
		return nil, err
	}

	if err = checkRespCode(resp, []int{http.StatusOK}); err != nil {
		drainRespBody(resp)
		return nil, err
	}

	return resp, nil
}

// returns true if the specified resource exists
func (f FileServiceClient) resourceExists(path string, res resourceType) (bool, http.Header, error) {
	if err := f.checkForStorageEmulator(); err != nil {
		return false, nil, err
	}

	uri := f.client.getEndpoint(fileServiceName, path, getURLInitValues(compNone, res))
	headers := f.client.getStandardHeaders()

	resp, err := f.client.exec(http.MethodHead, uri, headers, nil, f.auth)
	if resp != nil {
		defer drainRespBody(resp)
		if resp.StatusCode == http.StatusOK || resp.StatusCode == http.StatusNotFound {
			return resp.StatusCode == http.StatusOK, resp.Header, nil
		}
	}
	return false, nil, err
}

// creates a resource depending on the specified resource type
func (f FileServiceClient) createResource(path string, res resourceType, urlParams url.Values, extraHeaders map[string]string, expectedResponseCodes []int) (http.Header, error) {
	resp, err := f.createResourceNoClose(path, res, urlParams, extraHeaders)
	if err != nil {
		return nil, err
	}
	defer drainRespBody(resp)
	return resp.Header, checkRespCode(resp, expectedResponseCodes)
}

// creates a resource depending on the specified resource type, doesn't close the response body
func (f FileServiceClient) createResourceNoClose(path string, res resourceType, urlParams url.Values, extraHeaders map[string]string) (*http.Response, error) {
	if err := f.checkForStorageEmulator(); err != nil {
		return nil, err
	}

	values := getURLInitValues(compNone, res)
	combinedParams := mergeParams(values, urlParams)
	uri := f.client.getEndpoint(fileServiceName, path, combinedParams)
	extraHeaders = f.client.protectUserAgent(extraHeaders)
	headers := mergeHeaders(f.client.getStandardHeaders(), extraHeaders)

	return f.client.exec(http.MethodPut, uri, headers, nil, f.auth)
}

// returns HTTP header data for the specified directory or share
func (f FileServiceClient) getResourceHeaders(path string, comp compType, res resourceType, params url.Values, verb string) (http.Header, error) {
	resp, err := f.getResourceNoClose(path, comp, res, params, verb, nil)
	if err != nil {
		return nil, err
	}
	defer drainRespBody(resp)

	if err = checkRespCode(resp, []int{http.StatusOK}); err != nil {
		return nil, err
	}

	return resp.Header, nil
}

// gets the specified resource, doesn't close the response body
func (f FileServiceClient) getResourceNoClose(path string, comp compType, res resourceType, params url.Values, verb string, extraHeaders map[string]string) (*http.Response, error) {
	if err := f.checkForStorageEmulator(); err != nil {
		return nil, err
	}

	params = mergeParams(params, getURLInitValues(comp, res))
	uri := f.client.getEndpoint(fileServiceName, path, params)
	headers := mergeHeaders(f.client.getStandardHeaders(), extraHeaders)

	return f.client.exec(verb, uri, headers, nil, f.auth)
}

// deletes the resource and returns the response
func (f FileServiceClient) deleteResource(path string, res resourceType, options *FileRequestOptions) error {
	resp, err := f.deleteResourceNoClose(path, res, options)
	if err != nil {
		return err
	}
	defer drainRespBody(resp)
	return checkRespCode(resp, []int{http.StatusAccepted})
}

// deletes the resource and returns the response, doesn't close the response body
func (f FileServiceClient) deleteResourceNoClose(path string, res resourceType, options *FileRequestOptions) (*http.Response, error) {
	if err := f.checkForStorageEmulator(); err != nil {
		return nil, err
	}

	values := mergeParams(getURLInitValues(compNone, res), prepareOptions(options))
	uri := f.client.getEndpoint(fileServiceName, path, values)
	return f.client.exec(http.MethodDelete, uri, f.client.getStandardHeaders(), nil, f.auth)
}

// merges metadata into extraHeaders and returns extraHeaders
func mergeMDIntoExtraHeaders(metadata, extraHeaders map[string]string) map[string]string {
	if metadata == nil && extraHeaders == nil {
		return nil
	}
	if extraHeaders == nil {
		extraHeaders = make(map[string]string)
	}
	for k, v := range metadata {
		extraHeaders[userDefinedMetadataHeaderPrefix+k] = v
	}
	return extraHeaders
}

// sets extra header data for the specified resource
func (f FileServiceClient) setResourceHeaders(path string, comp compType, res resourceType, extraHeaders map[string]string, options *FileRequestOptions) (http.Header, error) {
	if err := f.checkForStorageEmulator(); err != nil {
		return nil, err
	}

	params := mergeParams(getURLInitValues(comp, res), prepareOptions(options))
	uri := f.client.getEndpoint(fileServiceName, path, params)
	extraHeaders = f.client.protectUserAgent(extraHeaders)
	headers := mergeHeaders(f.client.getStandardHeaders(), extraHeaders)

	resp, err := f.client.exec(http.MethodPut, uri, headers, nil, f.auth)
	if err != nil {
		return nil, err
	}
	defer drainRespBody(resp)

	return resp.Header, checkRespCode(resp, []int{http.StatusOK})
}

//checkForStorageEmulator determines if the client is setup for use with
//Azure Storage Emulator, and returns a relevant error
func (f FileServiceClient) checkForStorageEmulator() error {
	if f.client.accountName == StorageEmulatorAccountName {
		return fmt.Errorf("Error: File service is not currently supported by Azure Storage Emulator")
	}
	return nil
}
