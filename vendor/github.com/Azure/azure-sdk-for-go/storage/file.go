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
)

// FileServiceClient contains operations for Microsoft Azure File Service.
type FileServiceClient struct {
	client Client
}

// A Share is an entry in ShareListResponse.
type Share struct {
	Name       string          `xml:"Name"`
	Properties ShareProperties `xml:"Properties"`
}

// A Directory is an entry in DirsAndFilesListResponse.
type Directory struct {
	Name string `xml:"Name"`
}

// A File is an entry in DirsAndFilesListResponse.
type File struct {
	Name       string         `xml:"Name"`
	Properties FileProperties `xml:"Properties"`
}

// ShareProperties contains various properties of a share returned from
// various endpoints like ListShares.
type ShareProperties struct {
	LastModified string `xml:"Last-Modified"`
	Etag         string `xml:"Etag"`
	Quota        string `xml:"Quota"`
}

// DirectoryProperties contains various properties of a directory returned
// from various endpoints like GetDirectoryProperties.
type DirectoryProperties struct {
	LastModified string `xml:"Last-Modified"`
	Etag         string `xml:"Etag"`
}

// FileProperties contains various properties of a file returned from
// various endpoints like ListDirsAndFiles.
type FileProperties struct {
	CacheControl       string `header:"x-ms-cache-control"`
	ContentLength      uint64 `xml:"Content-Length"`
	ContentType        string `header:"x-ms-content-type"`
	CopyCompletionTime string
	CopyID             string
	CopySource         string
	CopyProgress       string
	CopyStatusDesc     string
	CopyStatus         string
	Disposition        string `header:"x-ms-content-disposition"`
	Encoding           string `header:"x-ms-content-encoding"`
	Etag               string
	Language           string `header:"x-ms-content-language"`
	LastModified       string
	MD5                string `header:"x-ms-content-md5"`
}

// FileStream contains file data returned from a call to GetFile.
type FileStream struct {
	Body       io.ReadCloser
	Properties *FileProperties
	Metadata   map[string]string
}

// ShareListResponse contains the response fields from
// ListShares call.
//
// See https://msdn.microsoft.com/en-us/library/azure/dn167009.aspx
type ShareListResponse struct {
	XMLName    xml.Name `xml:"EnumerationResults"`
	Xmlns      string   `xml:"xmlns,attr"`
	Prefix     string   `xml:"Prefix"`
	Marker     string   `xml:"Marker"`
	NextMarker string   `xml:"NextMarker"`
	MaxResults int64    `xml:"MaxResults"`
	Shares     []Share  `xml:"Shares>Share"`
}

// ListSharesParameters defines the set of customizable parameters to make a
// List Shares call.
//
// See https://msdn.microsoft.com/en-us/library/azure/dn167009.aspx
type ListSharesParameters struct {
	Prefix     string
	Marker     string
	Include    string
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

// FileRanges contains a list of file range information for a file.
//
// See https://msdn.microsoft.com/en-us/library/azure/dn166984.aspx
type FileRanges struct {
	ContentLength uint64
	LastModified  string
	ETag          string
	FileRanges    []FileRange `xml:"Range"`
}

// FileRange contains range information for a file.
//
// See https://msdn.microsoft.com/en-us/library/azure/dn166984.aspx
type FileRange struct {
	Start uint64 `xml:"Start"`
	End   uint64 `xml:"End"`
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

// ShareHeaders contains various properties of a file and is an entry
// in SetShareProperties
type ShareHeaders struct {
	Quota string `header:"x-ms-share-quota"`
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
		out.Set("maxresults", fmt.Sprintf("%v", p.MaxResults))
	}
	if p.Timeout != 0 {
		out.Set("timeout", fmt.Sprintf("%v", p.Timeout))
	}

	return out
}

func (p ListDirsAndFilesParameters) getParameters() url.Values {
	out := url.Values{}

	if p.Marker != "" {
		out.Set("marker", p.Marker)
	}
	if p.MaxResults != 0 {
		out.Set("maxresults", fmt.Sprintf("%v", p.MaxResults))
	}
	if p.Timeout != 0 {
		out.Set("timeout", fmt.Sprintf("%v", p.Timeout))
	}

	return out
}

func (fr FileRange) String() string {
	return fmt.Sprintf("bytes=%d-%d", fr.Start, fr.End)
}

// ToPathSegment returns the URL path segment for the specified values
func ToPathSegment(parts ...string) string {
	join := strings.Join(parts, "/")
	if join[0] != '/' {
		join = fmt.Sprintf("/%s", join)
	}
	return join
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

// ListDirsAndFiles returns a list of files or directories under the specified share or
// directory.  It also contains a pagination token and other response details.
//
// See https://msdn.microsoft.com/en-us/library/azure/dn166980.aspx
func (f FileServiceClient) ListDirsAndFiles(path string, params ListDirsAndFilesParameters) (DirsAndFilesListResponse, error) {
	q := mergeParams(params.getParameters(), getURLInitValues(compList, resourceDirectory))

	var out DirsAndFilesListResponse
	resp, err := f.listContent(path, q, nil)
	if err != nil {
		return out, err
	}

	defer resp.body.Close()
	err = xmlUnmarshal(resp.body, &out)
	return out, err
}

// ListFileRanges returns the list of valid ranges for a file.
//
// See https://msdn.microsoft.com/en-us/library/azure/dn166984.aspx
func (f FileServiceClient) ListFileRanges(path string, listRange *FileRange) (FileRanges, error) {
	params := url.Values{"comp": {"rangelist"}}

	// add optional range to list
	var headers map[string]string
	if listRange != nil {
		headers = make(map[string]string)
		headers["Range"] = listRange.String()
	}

	var out FileRanges
	resp, err := f.listContent(path, params, headers)
	if err != nil {
		return out, err
	}

	defer resp.body.Close()
	var cl uint64
	cl, err = strconv.ParseUint(resp.headers.Get("x-ms-content-length"), 10, 64)
	if err != nil {
		return out, err
	}

	out.ContentLength = cl
	out.ETag = resp.headers.Get("ETag")
	out.LastModified = resp.headers.Get("Last-Modified")

	err = xmlUnmarshal(resp.body, &out)
	return out, err
}

// ListShares returns the list of shares in a storage account along with
// pagination token and other response details.
//
// See https://msdn.microsoft.com/en-us/library/azure/dd179352.aspx
func (f FileServiceClient) ListShares(params ListSharesParameters) (ShareListResponse, error) {
	q := mergeParams(params.getParameters(), url.Values{"comp": {"list"}})

	var out ShareListResponse
	resp, err := f.listContent("", q, nil)
	if err != nil {
		return out, err
	}

	defer resp.body.Close()
	err = xmlUnmarshal(resp.body, &out)
	return out, err
}

// retrieves directory or share content
func (f FileServiceClient) listContent(path string, params url.Values, extraHeaders map[string]string) (*storageResponse, error) {
	if err := f.checkForStorageEmulator(); err != nil {
		return nil, err
	}

	uri := f.client.getEndpoint(fileServiceName, path, params)
	headers := mergeHeaders(f.client.getStandardHeaders(), extraHeaders)

	resp, err := f.client.exec(http.MethodGet, uri, headers, nil)
	if err != nil {
		return nil, err
	}

	if err = checkRespCode(resp.statusCode, []int{http.StatusOK}); err != nil {
		resp.body.Close()
		return nil, err
	}

	return resp, nil
}

// CreateDirectory operation creates a new directory with optional metadata in the
// specified share. If a directory with the same name already exists, the operation fails.
//
// See https://msdn.microsoft.com/en-us/library/azure/dn166993.aspx
func (f FileServiceClient) CreateDirectory(path string, metadata map[string]string) error {
	return f.createResource(path, resourceDirectory, mergeMDIntoExtraHeaders(metadata, nil))
}

// CreateFile operation creates a new file with optional metadata or replaces an existing one.
// Note that this only initializes the file, call PutRange to add content.
//
// See https://msdn.microsoft.com/en-us/library/azure/dn194271.aspx
func (f FileServiceClient) CreateFile(path string, maxSize uint64, metadata map[string]string) error {
	extraHeaders := map[string]string{
		"x-ms-content-length": strconv.FormatUint(maxSize, 10),
		"x-ms-type":           "file",
	}
	return f.createResource(path, resourceFile, mergeMDIntoExtraHeaders(metadata, extraHeaders))
}

// ClearRange releases the specified range of space in storage.
//
// See https://msdn.microsoft.com/en-us/library/azure/dn194276.aspx
func (f FileServiceClient) ClearRange(path string, fileRange FileRange) error {
	return f.modifyRange(path, nil, fileRange)
}

// PutRange writes a range of bytes to a file.  Note that the length of bytes must
// match (rangeEnd - rangeStart) + 1 with a maximum size of 4MB.
//
// See https://msdn.microsoft.com/en-us/library/azure/dn194276.aspx
func (f FileServiceClient) PutRange(path string, bytes io.Reader, fileRange FileRange) error {
	return f.modifyRange(path, bytes, fileRange)
}

// modifies a range of bytes in the specified file
func (f FileServiceClient) modifyRange(path string, bytes io.Reader, fileRange FileRange) error {
	if err := f.checkForStorageEmulator(); err != nil {
		return err
	}
	if fileRange.End < fileRange.Start {
		return errors.New("the value for rangeEnd must be greater than or equal to rangeStart")
	}
	if bytes != nil && fileRange.End-fileRange.Start > 4194304 {
		return errors.New("range cannot exceed 4MB in size")
	}

	uri := f.client.getEndpoint(fileServiceName, path, url.Values{"comp": {"range"}})

	// default to clear
	write := "clear"
	cl := uint64(0)

	// if bytes is not nil then this is an update operation
	if bytes != nil {
		write = "update"
		cl = (fileRange.End - fileRange.Start) + 1
	}

	extraHeaders := map[string]string{
		"Content-Length": strconv.FormatUint(cl, 10),
		"Range":          fileRange.String(),
		"x-ms-write":     write,
	}

	headers := mergeHeaders(f.client.getStandardHeaders(), extraHeaders)
	resp, err := f.client.exec(http.MethodPut, uri, headers, bytes)
	if err != nil {
		return err
	}

	defer resp.body.Close()
	return checkRespCode(resp.statusCode, []int{http.StatusCreated})
}

// GetFile operation reads or downloads a file from the system, including its
// metadata and properties.
//
// See https://docs.microsoft.com/en-us/rest/api/storageservices/fileservices/get-file
func (f FileServiceClient) GetFile(path string, fileRange *FileRange) (*FileStream, error) {
	var extraHeaders map[string]string
	if fileRange != nil {
		extraHeaders = map[string]string{
			"Range": fileRange.String(),
		}
	}

	resp, err := f.getResourceNoClose(path, compNone, resourceFile, http.MethodGet, extraHeaders)
	if err != nil {
		return nil, err
	}

	if err = checkRespCode(resp.statusCode, []int{http.StatusOK, http.StatusPartialContent}); err != nil {
		resp.body.Close()
		return nil, err
	}

	props, err := getFileProps(resp.headers)
	md := getFileMDFromHeaders(resp.headers)
	return &FileStream{Body: resp.body, Properties: props, Metadata: md}, nil
}

// CreateShare operation creates a new share with optional metadata under the specified account.
// If the share with the same name already exists, the operation fails.
//
// See https://msdn.microsoft.com/en-us/library/azure/dn167008.aspx
func (f FileServiceClient) CreateShare(name string, metadata map[string]string) error {
	return f.createResource(ToPathSegment(name), resourceShare, mergeMDIntoExtraHeaders(metadata, nil))
}

// DirectoryExists returns true if the specified directory exists on the specified share.
func (f FileServiceClient) DirectoryExists(path string) (bool, error) {
	return f.resourceExists(path, resourceDirectory)
}

// FileExists returns true if the specified file exists.
func (f FileServiceClient) FileExists(path string) (bool, error) {
	return f.resourceExists(path, resourceFile)
}

// ShareExists returns true if a share with given name exists
// on the storage account, otherwise returns false.
func (f FileServiceClient) ShareExists(name string) (bool, error) {
	return f.resourceExists(ToPathSegment(name), resourceShare)
}

// returns true if the specified directory or share exists
func (f FileServiceClient) resourceExists(path string, res resourceType) (bool, error) {
	if err := f.checkForStorageEmulator(); err != nil {
		return false, err
	}

	uri := f.client.getEndpoint(fileServiceName, path, getURLInitValues(compNone, res))
	headers := f.client.getStandardHeaders()

	resp, err := f.client.exec(http.MethodHead, uri, headers, nil)
	if resp != nil {
		defer resp.body.Close()
		if resp.statusCode == http.StatusOK || resp.statusCode == http.StatusNotFound {
			return resp.statusCode == http.StatusOK, nil
		}
	}
	return false, err
}

// GetDirectoryURL gets the canonical URL to the directory with the specified name
// in the specified share. This method does not create a publicly accessible URL if
// the file is private and this method does not check if the directory exists.
func (f FileServiceClient) GetDirectoryURL(path string) string {
	return f.client.getEndpoint(fileServiceName, path, url.Values{})
}

// GetShareURL gets the canonical URL to the share with the specified name in the
// specified container. This method does not create a publicly accessible URL if
// the file is private and this method does not check if the share exists.
func (f FileServiceClient) GetShareURL(name string) string {
	return f.client.getEndpoint(fileServiceName, ToPathSegment(name), url.Values{})
}

// CreateDirectoryIfNotExists creates a new directory on the specified share
// if it does not exist. Returns true if directory is newly created or false
// if the directory already exists.
//
// See https://msdn.microsoft.com/en-us/library/azure/dn166993.aspx
func (f FileServiceClient) CreateDirectoryIfNotExists(path string) (bool, error) {
	resp, err := f.createResourceNoClose(path, resourceDirectory, nil)
	if resp != nil {
		defer resp.body.Close()
		if resp.statusCode == http.StatusCreated || resp.statusCode == http.StatusConflict {
			return resp.statusCode == http.StatusCreated, nil
		}
	}
	return false, err
}

// CreateShareIfNotExists creates a new share under the specified account if
// it does not exist. Returns true if container is newly created or false if
// container already exists.
//
// See https://msdn.microsoft.com/en-us/library/azure/dn167008.aspx
func (f FileServiceClient) CreateShareIfNotExists(name string) (bool, error) {
	resp, err := f.createResourceNoClose(ToPathSegment(name), resourceShare, nil)
	if resp != nil {
		defer resp.body.Close()
		if resp.statusCode == http.StatusCreated || resp.statusCode == http.StatusConflict {
			return resp.statusCode == http.StatusCreated, nil
		}
	}
	return false, err
}

// creates a resource depending on the specified resource type
func (f FileServiceClient) createResource(path string, res resourceType, extraHeaders map[string]string) error {
	resp, err := f.createResourceNoClose(path, res, extraHeaders)
	if err != nil {
		return err
	}
	defer resp.body.Close()
	return checkRespCode(resp.statusCode, []int{http.StatusCreated})
}

// creates a resource depending on the specified resource type, doesn't close the response body
func (f FileServiceClient) createResourceNoClose(path string, res resourceType, extraHeaders map[string]string) (*storageResponse, error) {
	if err := f.checkForStorageEmulator(); err != nil {
		return nil, err
	}

	values := getURLInitValues(compNone, res)
	uri := f.client.getEndpoint(fileServiceName, path, values)
	headers := mergeHeaders(f.client.getStandardHeaders(), extraHeaders)

	return f.client.exec(http.MethodPut, uri, headers, nil)
}

// GetDirectoryProperties provides various information about the specified directory.
//
// See https://msdn.microsoft.com/en-us/library/azure/dn194272.aspx
func (f FileServiceClient) GetDirectoryProperties(path string) (*DirectoryProperties, error) {
	headers, err := f.getResourceHeaders(path, compNone, resourceDirectory, http.MethodHead)
	if err != nil {
		return nil, err
	}

	return &DirectoryProperties{
		LastModified: headers.Get("Last-Modified"),
		Etag:         headers.Get("Etag"),
	}, nil
}

// GetFileProperties provides various information about the specified file.
//
// See https://msdn.microsoft.com/en-us/library/azure/dn166971.aspx
func (f FileServiceClient) GetFileProperties(path string) (*FileProperties, error) {
	headers, err := f.getResourceHeaders(path, compNone, resourceFile, http.MethodHead)
	if err != nil {
		return nil, err
	}
	return getFileProps(headers)
}

// returns file properties from the specified HTTP header
func getFileProps(header http.Header) (*FileProperties, error) {
	size, err := strconv.ParseUint(header.Get("Content-Length"), 10, 64)
	if err != nil {
		return nil, err
	}

	return &FileProperties{
		CacheControl:       header.Get("Cache-Control"),
		ContentLength:      size,
		ContentType:        header.Get("Content-Type"),
		CopyCompletionTime: header.Get("x-ms-copy-completion-time"),
		CopyID:             header.Get("x-ms-copy-id"),
		CopyProgress:       header.Get("x-ms-copy-progress"),
		CopySource:         header.Get("x-ms-copy-source"),
		CopyStatus:         header.Get("x-ms-copy-status"),
		CopyStatusDesc:     header.Get("x-ms-copy-status-description"),
		Disposition:        header.Get("Content-Disposition"),
		Encoding:           header.Get("Content-Encoding"),
		Etag:               header.Get("ETag"),
		Language:           header.Get("Content-Language"),
		LastModified:       header.Get("Last-Modified"),
		MD5:                header.Get("Content-MD5"),
	}, nil
}

// GetShareProperties provides various information about the specified
// file. See https://msdn.microsoft.com/en-us/library/azure/dn689099.aspx
func (f FileServiceClient) GetShareProperties(name string) (*ShareProperties, error) {
	headers, err := f.getResourceHeaders(ToPathSegment(name), compNone, resourceShare, http.MethodHead)
	if err != nil {
		return nil, err
	}
	return &ShareProperties{
		LastModified: headers.Get("Last-Modified"),
		Etag:         headers.Get("Etag"),
		Quota:        headers.Get("x-ms-share-quota"),
	}, nil
}

// returns HTTP header data for the specified directory or share
func (f FileServiceClient) getResourceHeaders(path string, comp compType, res resourceType, verb string) (http.Header, error) {
	resp, err := f.getResourceNoClose(path, comp, res, verb, nil)
	if err != nil {
		return nil, err
	}
	defer resp.body.Close()

	if err = checkRespCode(resp.statusCode, []int{http.StatusOK}); err != nil {
		return nil, err
	}

	return resp.headers, nil
}

// gets the specified resource, doesn't close the response body
func (f FileServiceClient) getResourceNoClose(path string, comp compType, res resourceType, verb string, extraHeaders map[string]string) (*storageResponse, error) {
	if err := f.checkForStorageEmulator(); err != nil {
		return nil, err
	}

	params := getURLInitValues(comp, res)
	uri := f.client.getEndpoint(fileServiceName, path, params)
	headers := mergeHeaders(f.client.getStandardHeaders(), extraHeaders)

	return f.client.exec(verb, uri, headers, nil)
}

// SetFileProperties operation sets system properties on the specified file.
//
// Some keys may be converted to Camel-Case before sending. All keys
// are returned in lower case by SetFileProperties. HTTP header names
// are case-insensitive so case munging should not matter to other
// applications either.
//
// See https://msdn.microsoft.com/en-us/library/azure/dn166975.aspx
func (f FileServiceClient) SetFileProperties(path string, props FileProperties) error {
	return f.setResourceHeaders(path, compProperties, resourceFile, headersFromStruct(props))
}

// SetShareProperties replaces the ShareHeaders for the specified file.
//
// Some keys may be converted to Camel-Case before sending. All keys
// are returned in lower case by SetShareProperties. HTTP header names
// are case-insensitive so case munging should not matter to other
// applications either.
//
// See https://msdn.microsoft.com/en-us/library/azure/mt427368.aspx
func (f FileServiceClient) SetShareProperties(name string, shareHeaders ShareHeaders) error {
	return f.setResourceHeaders(ToPathSegment(name), compProperties, resourceShare, headersFromStruct(shareHeaders))
}

// DeleteDirectory operation removes the specified empty directory.
//
// See https://msdn.microsoft.com/en-us/library/azure/dn166969.aspx
func (f FileServiceClient) DeleteDirectory(path string) error {
	return f.deleteResource(path, resourceDirectory)
}

// DeleteFile operation immediately removes the file from the storage account.
//
// See https://msdn.microsoft.com/en-us/library/azure/dn689085.aspx
func (f FileServiceClient) DeleteFile(path string) error {
	return f.deleteResource(path, resourceFile)
}

// DeleteShare operation marks the specified share for deletion. The share
// and any files contained within it are later deleted during garbage
// collection.
//
// See https://msdn.microsoft.com/en-us/library/azure/dn689090.aspx
func (f FileServiceClient) DeleteShare(name string) error {
	return f.deleteResource(ToPathSegment(name), resourceShare)
}

// DeleteShareIfExists operation marks the specified share for deletion if it
// exists. The share and any files contained within it are later deleted during
// garbage collection. Returns true if share existed and deleted with this call,
// false otherwise.
//
// See https://msdn.microsoft.com/en-us/library/azure/dn689090.aspx
func (f FileServiceClient) DeleteShareIfExists(name string) (bool, error) {
	resp, err := f.deleteResourceNoClose(ToPathSegment(name), resourceShare)
	if resp != nil {
		defer resp.body.Close()
		if resp.statusCode == http.StatusAccepted || resp.statusCode == http.StatusNotFound {
			return resp.statusCode == http.StatusAccepted, nil
		}
	}
	return false, err
}

// deletes the resource and returns the response
func (f FileServiceClient) deleteResource(path string, res resourceType) error {
	resp, err := f.deleteResourceNoClose(path, res)
	if err != nil {
		return err
	}
	defer resp.body.Close()
	return checkRespCode(resp.statusCode, []int{http.StatusAccepted})
}

// deletes the resource and returns the response, doesn't close the response body
func (f FileServiceClient) deleteResourceNoClose(path string, res resourceType) (*storageResponse, error) {
	if err := f.checkForStorageEmulator(); err != nil {
		return nil, err
	}

	values := getURLInitValues(compNone, res)
	uri := f.client.getEndpoint(fileServiceName, path, values)
	return f.client.exec(http.MethodDelete, uri, f.client.getStandardHeaders(), nil)
}

// SetDirectoryMetadata replaces the metadata for the specified directory.
//
// Some keys may be converted to Camel-Case before sending. All keys
// are returned in lower case by GetDirectoryMetadata. HTTP header names
// are case-insensitive so case munging should not matter to other
// applications either.
//
// See https://msdn.microsoft.com/en-us/library/azure/mt427370.aspx
func (f FileServiceClient) SetDirectoryMetadata(path string, metadata map[string]string) error {
	return f.setResourceHeaders(path, compMetadata, resourceDirectory, mergeMDIntoExtraHeaders(metadata, nil))
}

// SetFileMetadata replaces the metadata for the specified file.
//
// Some keys may be converted to Camel-Case before sending. All keys
// are returned in lower case by GetFileMetadata. HTTP header names
// are case-insensitive so case munging should not matter to other
// applications either.
//
// See https://msdn.microsoft.com/en-us/library/azure/dn689097.aspx
func (f FileServiceClient) SetFileMetadata(path string, metadata map[string]string) error {
	return f.setResourceHeaders(path, compMetadata, resourceFile, mergeMDIntoExtraHeaders(metadata, nil))
}

// SetShareMetadata replaces the metadata for the specified Share.
//
// Some keys may be converted to Camel-Case before sending. All keys
// are returned in lower case by GetShareMetadata. HTTP header names
// are case-insensitive so case munging should not matter to other
// applications either.
//
// See https://msdn.microsoft.com/en-us/library/azure/dd179414.aspx
func (f FileServiceClient) SetShareMetadata(name string, metadata map[string]string) error {
	return f.setResourceHeaders(ToPathSegment(name), compMetadata, resourceShare, mergeMDIntoExtraHeaders(metadata, nil))
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

// merges extraHeaders into headers and returns headers
func mergeHeaders(headers, extraHeaders map[string]string) map[string]string {
	for k, v := range extraHeaders {
		headers[k] = v
	}
	return headers
}

// sets extra header data for the specified resource
func (f FileServiceClient) setResourceHeaders(path string, comp compType, res resourceType, extraHeaders map[string]string) error {
	if err := f.checkForStorageEmulator(); err != nil {
		return err
	}

	params := getURLInitValues(comp, res)
	uri := f.client.getEndpoint(fileServiceName, path, params)
	headers := mergeHeaders(f.client.getStandardHeaders(), extraHeaders)

	resp, err := f.client.exec(http.MethodPut, uri, headers, nil)
	if err != nil {
		return err
	}
	defer resp.body.Close()

	return checkRespCode(resp.statusCode, []int{http.StatusOK})
}

// GetDirectoryMetadata returns all user-defined metadata for the specified directory.
//
// All metadata keys will be returned in lower case. (HTTP header
// names are case-insensitive.)
//
// See https://msdn.microsoft.com/en-us/library/azure/mt427371.aspx
func (f FileServiceClient) GetDirectoryMetadata(path string) (map[string]string, error) {
	return f.getMetadata(path, resourceDirectory)
}

// GetFileMetadata returns all user-defined metadata for the specified file.
//
// All metadata keys will be returned in lower case. (HTTP header
// names are case-insensitive.)
//
// See https://msdn.microsoft.com/en-us/library/azure/dn689098.aspx
func (f FileServiceClient) GetFileMetadata(path string) (map[string]string, error) {
	return f.getMetadata(path, resourceFile)
}

// GetShareMetadata returns all user-defined metadata for the specified share.
//
// All metadata keys will be returned in lower case. (HTTP header
// names are case-insensitive.)
//
// See https://msdn.microsoft.com/en-us/library/azure/dd179414.aspx
func (f FileServiceClient) GetShareMetadata(name string) (map[string]string, error) {
	return f.getMetadata(ToPathSegment(name), resourceShare)
}

// gets metadata for the specified resource
func (f FileServiceClient) getMetadata(path string, res resourceType) (map[string]string, error) {
	if err := f.checkForStorageEmulator(); err != nil {
		return nil, err
	}

	headers, err := f.getResourceHeaders(path, compMetadata, res, http.MethodGet)
	if err != nil {
		return nil, err
	}

	return getFileMDFromHeaders(headers), nil
}

// returns a map of custom metadata values from the specified HTTP header
func getFileMDFromHeaders(header http.Header) map[string]string {
	metadata := make(map[string]string)
	for k, v := range header {
		// Can't trust CanonicalHeaderKey() to munge case
		// reliably. "_" is allowed in identifiers:
		// https://msdn.microsoft.com/en-us/library/azure/dd179414.aspx
		// https://msdn.microsoft.com/library/aa664670(VS.71).aspx
		// http://tools.ietf.org/html/rfc7230#section-3.2
		// ...but "_" is considered invalid by
		// CanonicalMIMEHeaderKey in
		// https://golang.org/src/net/textproto/reader.go?s=14615:14659#L542
		// so k can be "X-Ms-Meta-Foo" or "x-ms-meta-foo_bar".
		k = strings.ToLower(k)
		if len(v) == 0 || !strings.HasPrefix(k, strings.ToLower(userDefinedMetadataHeaderPrefix)) {
			continue
		}
		// metadata["foo"] = content of the last X-Ms-Meta-Foo header
		k = k[len(userDefinedMetadataHeaderPrefix):]
		metadata[k] = v[len(v)-1]
	}
	return metadata
}

//checkForStorageEmulator determines if the client is setup for use with
//Azure Storage Emulator, and returns a relevant error
func (f FileServiceClient) checkForStorageEmulator() error {
	if f.client.accountName == StorageEmulatorAccountName {
		return fmt.Errorf("Error: File service is not currently supported by Azure Storage Emulator")
	}
	return nil
}
