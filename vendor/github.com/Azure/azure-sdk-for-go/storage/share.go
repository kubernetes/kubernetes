package storage

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
// See https://msdn.microsoft.com/en-us/library/azure/dn167008.aspx
func (s *Share) Create() error {
	headers, err := s.fsc.createResource(s.buildPath(), resourceShare, mergeMDIntoExtraHeaders(s.Metadata, nil))
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
// See https://msdn.microsoft.com/en-us/library/azure/dn167008.aspx
func (s *Share) CreateIfNotExists() (bool, error) {
	resp, err := s.fsc.createResourceNoClose(s.buildPath(), resourceShare, nil)
	if resp != nil {
		defer resp.body.Close()
		if resp.statusCode == http.StatusCreated || resp.statusCode == http.StatusConflict {
			if resp.statusCode == http.StatusCreated {
				s.updateEtagAndLastModified(resp.headers)
				return true, nil
			}
			return false, s.FetchAttributes()
		}
	}

	return false, err
}

// Delete marks this share for deletion. The share along with any files
// and directories contained within it are later deleted during garbage
// collection.  If the share does not exist the operation fails
//
// See https://msdn.microsoft.com/en-us/library/azure/dn689090.aspx
func (s *Share) Delete() error {
	return s.fsc.deleteResource(s.buildPath(), resourceShare)
}

// DeleteIfExists operation marks this share for deletion if it exists.
//
// See https://msdn.microsoft.com/en-us/library/azure/dn689090.aspx
func (s *Share) DeleteIfExists() (bool, error) {
	resp, err := s.fsc.deleteResourceNoClose(s.buildPath(), resourceShare)
	if resp != nil {
		defer resp.body.Close()
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
func (s *Share) FetchAttributes() error {
	headers, err := s.fsc.getResourceHeaders(s.buildPath(), compNone, resourceShare, http.MethodHead)
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
// See https://msdn.microsoft.com/en-us/library/azure/dd179414.aspx
func (s *Share) SetMetadata() error {
	headers, err := s.fsc.setResourceHeaders(s.buildPath(), compMetadata, resourceShare, mergeMDIntoExtraHeaders(s.Metadata, nil))
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
// See https://msdn.microsoft.com/en-us/library/azure/mt427368.aspx
func (s *Share) SetProperties() error {
	if s.Properties.Quota < 1 || s.Properties.Quota > 5120 {
		return fmt.Errorf("invalid value %v for quota, valid values are [1, 5120]", s.Properties.Quota)
	}

	headers, err := s.fsc.setResourceHeaders(s.buildPath(), compProperties, resourceShare, map[string]string{
		"x-ms-share-quota": strconv.Itoa(s.Properties.Quota),
	})
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
