package storage

import (
	"encoding/xml"
	"net/http"
	"net/url"
)

// Directory represents a directory on a share.
type Directory struct {
	fsc        *FileServiceClient
	Metadata   map[string]string
	Name       string `xml:"Name"`
	parent     *Directory
	Properties DirectoryProperties
	share      *Share
}

// DirectoryProperties contains various properties of a directory.
type DirectoryProperties struct {
	LastModified string `xml:"Last-Modified"`
	Etag         string `xml:"Etag"`
}

// ListDirsAndFilesParameters defines the set of customizable parameters to
// make a List Files and Directories call.
//
// See https://msdn.microsoft.com/en-us/library/azure/dn166980.aspx
type ListDirsAndFilesParameters struct {
	Marker     string
	MaxResults uint
	Timeout    uint
}

// DirsAndFilesListResponse contains the response fields from
// a List Files and Directories call.
//
// See https://msdn.microsoft.com/en-us/library/azure/dn166980.aspx
type DirsAndFilesListResponse struct {
	XMLName     xml.Name    `xml:"EnumerationResults"`
	Xmlns       string      `xml:"xmlns,attr"`
	Marker      string      `xml:"Marker"`
	MaxResults  int64       `xml:"MaxResults"`
	Directories []Directory `xml:"Entries>Directory"`
	Files       []File      `xml:"Entries>File"`
	NextMarker  string      `xml:"NextMarker"`
}

// builds the complete directory path for this directory object.
func (d *Directory) buildPath() string {
	path := ""
	current := d
	for current.Name != "" {
		path = "/" + current.Name + path
		current = current.parent
	}
	return d.share.buildPath() + path
}

// Create this directory in the associated share.
// If a directory with the same name already exists, the operation fails.
//
// See https://msdn.microsoft.com/en-us/library/azure/dn166993.aspx
func (d *Directory) Create() error {
	// if this is the root directory exit early
	if d.parent == nil {
		return nil
	}

	headers, err := d.fsc.createResource(d.buildPath(), resourceDirectory, mergeMDIntoExtraHeaders(d.Metadata, nil))
	if err != nil {
		return err
	}

	d.updateEtagAndLastModified(headers)
	return nil
}

// CreateIfNotExists creates this directory under the associated share if the
// directory does not exists. Returns true if the directory is newly created or
// false if the directory already exists.
//
// See https://msdn.microsoft.com/en-us/library/azure/dn166993.aspx
func (d *Directory) CreateIfNotExists() (bool, error) {
	// if this is the root directory exit early
	if d.parent == nil {
		return false, nil
	}

	resp, err := d.fsc.createResourceNoClose(d.buildPath(), resourceDirectory, nil)
	if resp != nil {
		defer resp.body.Close()
		if resp.statusCode == http.StatusCreated || resp.statusCode == http.StatusConflict {
			if resp.statusCode == http.StatusCreated {
				d.updateEtagAndLastModified(resp.headers)
				return true, nil
			}

			return false, d.FetchAttributes()
		}
	}

	return false, err
}

// Delete removes this directory.  It must be empty in order to be deleted.
// If the directory does not exist the operation fails.
//
// See https://msdn.microsoft.com/en-us/library/azure/dn166969.aspx
func (d *Directory) Delete() error {
	return d.fsc.deleteResource(d.buildPath(), resourceDirectory)
}

// DeleteIfExists removes this directory if it exists.
//
// See https://msdn.microsoft.com/en-us/library/azure/dn166969.aspx
func (d *Directory) DeleteIfExists() (bool, error) {
	resp, err := d.fsc.deleteResourceNoClose(d.buildPath(), resourceDirectory)
	if resp != nil {
		defer resp.body.Close()
		if resp.statusCode == http.StatusAccepted || resp.statusCode == http.StatusNotFound {
			return resp.statusCode == http.StatusAccepted, nil
		}
	}
	return false, err
}

// Exists returns true if this directory exists.
func (d *Directory) Exists() (bool, error) {
	exists, headers, err := d.fsc.resourceExists(d.buildPath(), resourceDirectory)
	if exists {
		d.updateEtagAndLastModified(headers)
	}
	return exists, err
}

// FetchAttributes retrieves metadata for this directory.
func (d *Directory) FetchAttributes() error {
	headers, err := d.fsc.getResourceHeaders(d.buildPath(), compNone, resourceDirectory, http.MethodHead)
	if err != nil {
		return err
	}

	d.updateEtagAndLastModified(headers)
	d.Metadata = getMetadataFromHeaders(headers)

	return nil
}

// GetDirectoryReference returns a child Directory object for this directory.
func (d *Directory) GetDirectoryReference(name string) *Directory {
	return &Directory{
		fsc:    d.fsc,
		Name:   name,
		parent: d,
		share:  d.share,
	}
}

// GetFileReference returns a child File object for this directory.
func (d *Directory) GetFileReference(name string) *File {
	return &File{
		fsc:    d.fsc,
		Name:   name,
		parent: d,
		share:  d.share,
	}
}

// ListDirsAndFiles returns a list of files and directories under this directory.
// It also contains a pagination token and other response details.
//
// See https://msdn.microsoft.com/en-us/library/azure/dn166980.aspx
func (d *Directory) ListDirsAndFiles(params ListDirsAndFilesParameters) (*DirsAndFilesListResponse, error) {
	q := mergeParams(params.getParameters(), getURLInitValues(compList, resourceDirectory))

	resp, err := d.fsc.listContent(d.buildPath(), q, nil)
	if err != nil {
		return nil, err
	}

	defer resp.body.Close()
	var out DirsAndFilesListResponse
	err = xmlUnmarshal(resp.body, &out)
	return &out, err
}

// SetMetadata replaces the metadata for this directory.
//
// Some keys may be converted to Camel-Case before sending. All keys
// are returned in lower case by GetDirectoryMetadata. HTTP header names
// are case-insensitive so case munging should not matter to other
// applications either.
//
// See https://msdn.microsoft.com/en-us/library/azure/mt427370.aspx
func (d *Directory) SetMetadata() error {
	headers, err := d.fsc.setResourceHeaders(d.buildPath(), compMetadata, resourceDirectory, mergeMDIntoExtraHeaders(d.Metadata, nil))
	if err != nil {
		return err
	}

	d.updateEtagAndLastModified(headers)
	return nil
}

// updates Etag and last modified date
func (d *Directory) updateEtagAndLastModified(headers http.Header) {
	d.Properties.Etag = headers.Get("Etag")
	d.Properties.LastModified = headers.Get("Last-Modified")
}

// URL gets the canonical URL to this directory.
// This method does not create a publicly accessible URL if the directory
// is private and this method does not check if the directory exists.
func (d *Directory) URL() string {
	return d.fsc.client.getEndpoint(fileServiceName, d.buildPath(), url.Values{})
}
