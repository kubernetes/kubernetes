package storage

// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See License.txt in the project root for license information.

import (
	"errors"
	"fmt"
	"io"
	"io/ioutil"
	"net/http"
	"net/url"
	"strconv"
	"sync"
)

const fourMB = uint64(4194304)
const oneTB = uint64(1099511627776)

// Export maximum range and file sizes

// MaxRangeSize defines the maximum size in bytes for a file range.
const MaxRangeSize = fourMB

// MaxFileSize defines the maximum size in bytes for a file.
const MaxFileSize = oneTB

// File represents a file on a share.
type File struct {
	fsc                *FileServiceClient
	Metadata           map[string]string
	Name               string `xml:"Name"`
	parent             *Directory
	Properties         FileProperties `xml:"Properties"`
	share              *Share
	FileCopyProperties FileCopyState
	mutex              *sync.Mutex
}

// FileProperties contains various properties of a file.
type FileProperties struct {
	CacheControl string `header:"x-ms-cache-control"`
	Disposition  string `header:"x-ms-content-disposition"`
	Encoding     string `header:"x-ms-content-encoding"`
	Etag         string
	Language     string `header:"x-ms-content-language"`
	LastModified string
	Length       uint64 `xml:"Content-Length" header:"x-ms-content-length"`
	MD5          string `header:"x-ms-content-md5"`
	Type         string `header:"x-ms-content-type"`
}

// FileCopyState contains various properties of a file copy operation.
type FileCopyState struct {
	CompletionTime string
	ID             string `header:"x-ms-copy-id"`
	Progress       string
	Source         string
	Status         string `header:"x-ms-copy-status"`
	StatusDesc     string
}

// FileStream contains file data returned from a call to GetFile.
type FileStream struct {
	Body       io.ReadCloser
	ContentMD5 string
}

// FileRequestOptions will be passed to misc file operations.
// Currently just Timeout (in seconds) but could expand.
type FileRequestOptions struct {
	Timeout uint // timeout duration in seconds.
}

func prepareOptions(options *FileRequestOptions) url.Values {
	params := url.Values{}
	if options != nil {
		params = addTimeout(params, options.Timeout)
	}
	return params
}

// FileRanges contains a list of file range information for a file.
//
// See https://docs.microsoft.com/en-us/rest/api/storageservices/fileservices/List-Ranges
type FileRanges struct {
	ContentLength uint64
	LastModified  string
	ETag          string
	FileRanges    []FileRange `xml:"Range"`
}

// FileRange contains range information for a file.
//
// See https://docs.microsoft.com/en-us/rest/api/storageservices/fileservices/List-Ranges
type FileRange struct {
	Start uint64 `xml:"Start"`
	End   uint64 `xml:"End"`
}

func (fr FileRange) String() string {
	return fmt.Sprintf("bytes=%d-%d", fr.Start, fr.End)
}

// builds the complete file path for this file object
func (f *File) buildPath() string {
	return f.parent.buildPath() + "/" + f.Name
}

// ClearRange releases the specified range of space in a file.
//
// See https://docs.microsoft.com/en-us/rest/api/storageservices/fileservices/Put-Range
func (f *File) ClearRange(fileRange FileRange, options *FileRequestOptions) error {
	var timeout *uint
	if options != nil {
		timeout = &options.Timeout
	}
	headers, err := f.modifyRange(nil, fileRange, timeout, nil)
	if err != nil {
		return err
	}

	f.updateEtagAndLastModified(headers)
	return nil
}

// Create creates a new file or replaces an existing one.
//
// See https://docs.microsoft.com/en-us/rest/api/storageservices/fileservices/Create-File
func (f *File) Create(maxSize uint64, options *FileRequestOptions) error {
	if maxSize > oneTB {
		return fmt.Errorf("max file size is 1TB")
	}
	params := prepareOptions(options)
	headers := headersFromStruct(f.Properties)
	headers["x-ms-content-length"] = strconv.FormatUint(maxSize, 10)
	headers["x-ms-type"] = "file"

	outputHeaders, err := f.fsc.createResource(f.buildPath(), resourceFile, params, mergeMDIntoExtraHeaders(f.Metadata, headers), []int{http.StatusCreated})
	if err != nil {
		return err
	}

	f.Properties.Length = maxSize
	f.updateEtagAndLastModified(outputHeaders)
	return nil
}

// CopyFile operation copied a file/blob from the sourceURL to the path provided.
//
// See https://docs.microsoft.com/en-us/rest/api/storageservices/fileservices/copy-file
func (f *File) CopyFile(sourceURL string, options *FileRequestOptions) error {
	extraHeaders := map[string]string{
		"x-ms-type":        "file",
		"x-ms-copy-source": sourceURL,
	}
	params := prepareOptions(options)

	headers, err := f.fsc.createResource(f.buildPath(), resourceFile, params, mergeMDIntoExtraHeaders(f.Metadata, extraHeaders), []int{http.StatusAccepted})
	if err != nil {
		return err
	}

	f.updateEtagAndLastModified(headers)
	f.FileCopyProperties.ID = headers.Get("X-Ms-Copy-Id")
	f.FileCopyProperties.Status = headers.Get("X-Ms-Copy-Status")
	return nil
}

// Delete immediately removes this file from the storage account.
//
// See https://docs.microsoft.com/en-us/rest/api/storageservices/fileservices/Delete-File2
func (f *File) Delete(options *FileRequestOptions) error {
	return f.fsc.deleteResource(f.buildPath(), resourceFile, options)
}

// DeleteIfExists removes this file if it exists.
//
// See https://docs.microsoft.com/en-us/rest/api/storageservices/fileservices/Delete-File2
func (f *File) DeleteIfExists(options *FileRequestOptions) (bool, error) {
	resp, err := f.fsc.deleteResourceNoClose(f.buildPath(), resourceFile, options)
	if resp != nil {
		defer drainRespBody(resp)
		if resp.StatusCode == http.StatusAccepted || resp.StatusCode == http.StatusNotFound {
			return resp.StatusCode == http.StatusAccepted, nil
		}
	}
	return false, err
}

// GetFileOptions includes options for a get file operation
type GetFileOptions struct {
	Timeout       uint
	GetContentMD5 bool
}

// DownloadToStream operation downloads the file.
//
// See: https://docs.microsoft.com/en-us/rest/api/storageservices/fileservices/get-file
func (f *File) DownloadToStream(options *FileRequestOptions) (io.ReadCloser, error) {
	params := prepareOptions(options)
	resp, err := f.fsc.getResourceNoClose(f.buildPath(), compNone, resourceFile, params, http.MethodGet, nil)
	if err != nil {
		return nil, err
	}

	if err = checkRespCode(resp, []int{http.StatusOK}); err != nil {
		drainRespBody(resp)
		return nil, err
	}
	return resp.Body, nil
}

// DownloadRangeToStream operation downloads the specified range of this file with optional MD5 hash.
//
// See https://docs.microsoft.com/en-us/rest/api/storageservices/fileservices/get-file
func (f *File) DownloadRangeToStream(fileRange FileRange, options *GetFileOptions) (fs FileStream, err error) {
	extraHeaders := map[string]string{
		"Range": fileRange.String(),
	}
	params := url.Values{}
	if options != nil {
		if options.GetContentMD5 {
			if isRangeTooBig(fileRange) {
				return fs, fmt.Errorf("must specify a range less than or equal to 4MB when getContentMD5 is true")
			}
			extraHeaders["x-ms-range-get-content-md5"] = "true"
		}
		params = addTimeout(params, options.Timeout)
	}

	resp, err := f.fsc.getResourceNoClose(f.buildPath(), compNone, resourceFile, params, http.MethodGet, extraHeaders)
	if err != nil {
		return fs, err
	}

	if err = checkRespCode(resp, []int{http.StatusOK, http.StatusPartialContent}); err != nil {
		drainRespBody(resp)
		return fs, err
	}

	fs.Body = resp.Body
	if options != nil && options.GetContentMD5 {
		fs.ContentMD5 = resp.Header.Get("Content-MD5")
	}
	return fs, nil
}

// Exists returns true if this file exists.
func (f *File) Exists() (bool, error) {
	exists, headers, err := f.fsc.resourceExists(f.buildPath(), resourceFile)
	if exists {
		f.updateEtagAndLastModified(headers)
		f.updateProperties(headers)
	}
	return exists, err
}

// FetchAttributes updates metadata and properties for this file.
// See https://docs.microsoft.com/en-us/rest/api/storageservices/fileservices/get-file-properties
func (f *File) FetchAttributes(options *FileRequestOptions) error {
	params := prepareOptions(options)
	headers, err := f.fsc.getResourceHeaders(f.buildPath(), compNone, resourceFile, params, http.MethodHead)
	if err != nil {
		return err
	}

	f.updateEtagAndLastModified(headers)
	f.updateProperties(headers)
	f.Metadata = getMetadataFromHeaders(headers)
	return nil
}

// returns true if the range is larger than 4MB
func isRangeTooBig(fileRange FileRange) bool {
	if fileRange.End-fileRange.Start > fourMB {
		return true
	}

	return false
}

// ListRangesOptions includes options for a list file ranges operation
type ListRangesOptions struct {
	Timeout   uint
	ListRange *FileRange
}

// ListRanges returns the list of valid ranges for this file.
//
// See https://docs.microsoft.com/en-us/rest/api/storageservices/fileservices/List-Ranges
func (f *File) ListRanges(options *ListRangesOptions) (*FileRanges, error) {
	params := url.Values{"comp": {"rangelist"}}

	// add optional range to list
	var headers map[string]string
	if options != nil {
		params = addTimeout(params, options.Timeout)
		if options.ListRange != nil {
			headers = make(map[string]string)
			headers["Range"] = options.ListRange.String()
		}
	}

	resp, err := f.fsc.listContent(f.buildPath(), params, headers)
	if err != nil {
		return nil, err
	}

	defer resp.Body.Close()
	var cl uint64
	cl, err = strconv.ParseUint(resp.Header.Get("x-ms-content-length"), 10, 64)
	if err != nil {
		ioutil.ReadAll(resp.Body)
		return nil, err
	}

	var out FileRanges
	out.ContentLength = cl
	out.ETag = resp.Header.Get("ETag")
	out.LastModified = resp.Header.Get("Last-Modified")

	err = xmlUnmarshal(resp.Body, &out)
	return &out, err
}

// modifies a range of bytes in this file
func (f *File) modifyRange(bytes io.Reader, fileRange FileRange, timeout *uint, contentMD5 *string) (http.Header, error) {
	if err := f.fsc.checkForStorageEmulator(); err != nil {
		return nil, err
	}
	if fileRange.End < fileRange.Start {
		return nil, errors.New("the value for rangeEnd must be greater than or equal to rangeStart")
	}
	if bytes != nil && isRangeTooBig(fileRange) {
		return nil, errors.New("range cannot exceed 4MB in size")
	}

	params := url.Values{"comp": {"range"}}
	if timeout != nil {
		params = addTimeout(params, *timeout)
	}

	uri := f.fsc.client.getEndpoint(fileServiceName, f.buildPath(), params)

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

	if contentMD5 != nil {
		extraHeaders["Content-MD5"] = *contentMD5
	}

	headers := mergeHeaders(f.fsc.client.getStandardHeaders(), extraHeaders)
	resp, err := f.fsc.client.exec(http.MethodPut, uri, headers, bytes, f.fsc.auth)
	if err != nil {
		return nil, err
	}
	defer drainRespBody(resp)
	return resp.Header, checkRespCode(resp, []int{http.StatusCreated})
}

// SetMetadata replaces the metadata for this file.
//
// Some keys may be converted to Camel-Case before sending. All keys
// are returned in lower case by GetFileMetadata. HTTP header names
// are case-insensitive so case munging should not matter to other
// applications either.
//
// See https://docs.microsoft.com/en-us/rest/api/storageservices/fileservices/Set-File-Metadata
func (f *File) SetMetadata(options *FileRequestOptions) error {
	headers, err := f.fsc.setResourceHeaders(f.buildPath(), compMetadata, resourceFile, mergeMDIntoExtraHeaders(f.Metadata, nil), options)
	if err != nil {
		return err
	}

	f.updateEtagAndLastModified(headers)
	return nil
}

// SetProperties sets system properties on this file.
//
// Some keys may be converted to Camel-Case before sending. All keys
// are returned in lower case by SetFileProperties. HTTP header names
// are case-insensitive so case munging should not matter to other
// applications either.
//
// See https://docs.microsoft.com/en-us/rest/api/storageservices/fileservices/Set-File-Properties
func (f *File) SetProperties(options *FileRequestOptions) error {
	headers, err := f.fsc.setResourceHeaders(f.buildPath(), compProperties, resourceFile, headersFromStruct(f.Properties), options)
	if err != nil {
		return err
	}

	f.updateEtagAndLastModified(headers)
	return nil
}

// updates Etag and last modified date
func (f *File) updateEtagAndLastModified(headers http.Header) {
	f.Properties.Etag = headers.Get("Etag")
	f.Properties.LastModified = headers.Get("Last-Modified")
}

// updates file properties from the specified HTTP header
func (f *File) updateProperties(header http.Header) {
	size, err := strconv.ParseUint(header.Get("Content-Length"), 10, 64)
	if err == nil {
		f.Properties.Length = size
	}

	f.updateEtagAndLastModified(header)
	f.Properties.CacheControl = header.Get("Cache-Control")
	f.Properties.Disposition = header.Get("Content-Disposition")
	f.Properties.Encoding = header.Get("Content-Encoding")
	f.Properties.Language = header.Get("Content-Language")
	f.Properties.MD5 = header.Get("Content-MD5")
	f.Properties.Type = header.Get("Content-Type")
}

// URL gets the canonical URL to this file.
// This method does not create a publicly accessible URL if the file
// is private and this method does not check if the file exists.
func (f *File) URL() string {
	return f.fsc.client.getEndpoint(fileServiceName, f.buildPath(), nil)
}

// WriteRangeOptions includes options for a write file range operation
type WriteRangeOptions struct {
	Timeout    uint
	ContentMD5 string
}

// WriteRange writes a range of bytes to this file with an optional MD5 hash of the content (inside
// options parameter). Note that the length of bytes must match (rangeEnd - rangeStart) + 1 with
// a maximum size of 4MB.
//
// See https://docs.microsoft.com/en-us/rest/api/storageservices/fileservices/Put-Range
func (f *File) WriteRange(bytes io.Reader, fileRange FileRange, options *WriteRangeOptions) error {
	if bytes == nil {
		return errors.New("bytes cannot be nil")
	}
	var timeout *uint
	var md5 *string
	if options != nil {
		timeout = &options.Timeout
		md5 = &options.ContentMD5
	}

	headers, err := f.modifyRange(bytes, fileRange, timeout, md5)
	if err != nil {
		return err
	}
	// it's perfectly legal for multiple go routines to call WriteRange
	// on the same *File (e.g. concurrently writing non-overlapping ranges)
	// so we must take the file mutex before updating our properties.
	f.mutex.Lock()
	f.updateEtagAndLastModified(headers)
	f.mutex.Unlock()
	return nil
}
