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
	"fmt"
	"net/http"
	"net/url"
	"strconv"
)

// Share represents an Azure file share.
type Share struct {
	fsc        *FileServiceClient
	Name       string          `xml:"Name"`
	Properties ShareProperties `xml:"Properties"`
	Metadata   map[string]string
}

// ShareProperties contains various properties of a share.
type ShareProperties struct {
	LastModified string `xml:"Last-Modified"`
	Etag         string `xml:"Etag"`
	Quota        int    `xml:"Quota"`
}

// builds the complete path for this share object.
func (s *Share) buildPath() string {
	return fmt.Sprintf("/%s", s.Name)
}

// Create this share under the associated account.
// If a share with the same name already exists, the operation fails.
//
// See https://docs.microsoft.com/en-us/rest/api/storageservices/fileservices/Create-Share
func (s *Share) Create(options *FileRequestOptions) error {
	extraheaders := map[string]string{}
	if s.Properties.Quota > 0 {
		extraheaders["x-ms-share-quota"] = strconv.Itoa(s.Properties.Quota)
	}

	params := prepareOptions(options)
	headers, err := s.fsc.createResource(s.buildPath(), resourceShare, params, mergeMDIntoExtraHeaders(s.Metadata, extraheaders), []int{http.StatusCreated})
	if err != nil {
		return err
	}

	s.updateEtagAndLastModified(headers)
	return nil
}

// CreateIfNotExists creates this share under the associated account if
// it does not exist. Returns true if the share is newly created or false if
// the share already exists.
//
// See https://docs.microsoft.com/en-us/rest/api/storageservices/fileservices/Create-Share
func (s *Share) CreateIfNotExists(options *FileRequestOptions) (bool, error) {
	extraheaders := map[string]string{}
	if s.Properties.Quota > 0 {
		extraheaders["x-ms-share-quota"] = strconv.Itoa(s.Properties.Quota)
	}

	params := prepareOptions(options)
	resp, err := s.fsc.createResourceNoClose(s.buildPath(), resourceShare, params, extraheaders)
	if resp != nil {
		defer readAndCloseBody(resp.body)
		if resp.statusCode == http.StatusCreated || resp.statusCode == http.StatusConflict {
			if resp.statusCode == http.StatusCreated {
				s.updateEtagAndLastModified(resp.headers)
				return true, nil
			}
			return false, s.FetchAttributes(nil)
		}
	}

	return false, err
}

// Delete marks this share for deletion. The share along with any files
// and directories contained within it are later deleted during garbage
// collection.  If the share does not exist the operation fails
//
// See https://docs.microsoft.com/en-us/rest/api/storageservices/fileservices/Delete-Share
func (s *Share) Delete(options *FileRequestOptions) error {
	return s.fsc.deleteResource(s.buildPath(), resourceShare, options)
}

// DeleteIfExists operation marks this share for deletion if it exists.
//
// See https://docs.microsoft.com/en-us/rest/api/storageservices/fileservices/Delete-Share
func (s *Share) DeleteIfExists(options *FileRequestOptions) (bool, error) {
	resp, err := s.fsc.deleteResourceNoClose(s.buildPath(), resourceShare, options)
	if resp != nil {
		defer readAndCloseBody(resp.body)
		if resp.statusCode == http.StatusAccepted || resp.statusCode == http.StatusNotFound {
			return resp.statusCode == http.StatusAccepted, nil
		}
	}
	return false, err
}

// Exists returns true if this share already exists
// on the storage account, otherwise returns false.
func (s *Share) Exists() (bool, error) {
	exists, headers, err := s.fsc.resourceExists(s.buildPath(), resourceShare)
	if exists {
		s.updateEtagAndLastModified(headers)
		s.updateQuota(headers)
	}
	return exists, err
}

// FetchAttributes retrieves metadata and properties for this share.
// See https://docs.microsoft.com/en-us/rest/api/storageservices/fileservices/get-share-properties
func (s *Share) FetchAttributes(options *FileRequestOptions) error {
	params := prepareOptions(options)
	headers, err := s.fsc.getResourceHeaders(s.buildPath(), compNone, resourceShare, params, http.MethodHead)
	if err != nil {
		return err
	}

	s.updateEtagAndLastModified(headers)
	s.updateQuota(headers)
	s.Metadata = getMetadataFromHeaders(headers)

	return nil
}

// GetRootDirectoryReference returns a Directory object at the root of this share.
func (s *Share) GetRootDirectoryReference() *Directory {
	return &Directory{
		fsc:   s.fsc,
		share: s,
	}
}

// ServiceClient returns the FileServiceClient associated with this share.
func (s *Share) ServiceClient() *FileServiceClient {
	return s.fsc
}

// SetMetadata replaces the metadata for this share.
//
// Some keys may be converted to Camel-Case before sending. All keys
// are returned in lower case by GetShareMetadata. HTTP header names
// are case-insensitive so case munging should not matter to other
// applications either.
//
// See https://docs.microsoft.com/en-us/rest/api/storageservices/fileservices/set-share-metadata
func (s *Share) SetMetadata(options *FileRequestOptions) error {
	headers, err := s.fsc.setResourceHeaders(s.buildPath(), compMetadata, resourceShare, mergeMDIntoExtraHeaders(s.Metadata, nil), options)
	if err != nil {
		return err
	}

	s.updateEtagAndLastModified(headers)
	return nil
}

// SetProperties sets system properties for this share.
//
// Some keys may be converted to Camel-Case before sending. All keys
// are returned in lower case by SetShareProperties. HTTP header names
// are case-insensitive so case munging should not matter to other
// applications either.
//
// See https://docs.microsoft.com/en-us/rest/api/storageservices/fileservices/Set-Share-Properties
func (s *Share) SetProperties(options *FileRequestOptions) error {
	extraheaders := map[string]string{}
	if s.Properties.Quota > 0 {
		if s.Properties.Quota > 5120 {
			return fmt.Errorf("invalid value %v for quota, valid values are [1, 5120]", s.Properties.Quota)
		}
		extraheaders["x-ms-share-quota"] = strconv.Itoa(s.Properties.Quota)
	}

	headers, err := s.fsc.setResourceHeaders(s.buildPath(), compProperties, resourceShare, extraheaders, options)
	if err != nil {
		return err
	}

	s.updateEtagAndLastModified(headers)
	return nil
}

// updates Etag and last modified date
func (s *Share) updateEtagAndLastModified(headers http.Header) {
	s.Properties.Etag = headers.Get("Etag")
	s.Properties.LastModified = headers.Get("Last-Modified")
}

// updates quota value
func (s *Share) updateQuota(headers http.Header) {
	quota, err := strconv.Atoi(headers.Get("x-ms-share-quota"))
	if err == nil {
		s.Properties.Quota = quota
	}
}

// URL gets the canonical URL to this share. This method does not create a publicly accessible
// URL if the share is private and this method does not check if the share exists.
func (s *Share) URL() string {
	return s.fsc.client.getEndpoint(fileServiceName, s.buildPath(), url.Values{})
}
