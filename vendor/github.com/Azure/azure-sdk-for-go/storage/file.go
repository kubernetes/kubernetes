package storage

import (
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strconv"
)

const fourMB = uint64(4194304)
const oneTB = uint64(1099511627776)

// File represents a file on a share.
type File struct {
	fsc        *FileServiceClient
	Metadata   map[string]string
	Name       string `xml:"Name"`
	parent     *Directory
	Properties FileProperties `xml:"Properties"`
	share      *Share
}

// FileProperties contains various properties of a file.
type FileProperties struct {
	CacheControl string `header:"x-ms-cache-control"`
	Disposition  string `header:"x-ms-content-disposition"`
	Encoding     string `header:"x-ms-content-encoding"`
	Etag         string
	Language     string `header:"x-ms-content-language"`
	LastModified string
	Length       uint64 `xml:"Content-Length"`
	MD5          string `header:"x-ms-content-md5"`
	Type         string `header:"x-ms-content-type"`
}

// FileCopyState contains various properties of a file copy operation.
type FileCopyState struct {
	CompletionTime string
	ID             string
	Progress       string
	Source         string
	Status         string
	StatusDesc     string
}

// FileStream contains file data returned from a call to GetFile.
type FileStream struct {
	Body       io.ReadCloser
	ContentMD5 string
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

func (fr FileRange) String() string {
	return fmt.Sprintf("bytes=%d-%d", fr.Start, fr.End)
}

// builds the complete file path for this file object
func (f *File) buildPath() string {
	return f.parent.buildPath() + "/" + f.Name
}

// ClearRange releases the specified range of space in a file.
//
// See https://msdn.microsoft.com/en-us/library/azure/dn194276.aspx
func (f *File) ClearRange(fileRange FileRange) error {
	headers, err := f.modifyRange(nil, fileRange, nil)
	if err != nil {
		return err
	}

	f.updateEtagAndLastModified(headers)
	return nil
}

// Create creates a new file or replaces an existing one.
//
// See https://msdn.microsoft.com/en-us/library/azure/dn194271.aspx
func (f *File) Create(maxSize uint64) error {
	if maxSize > oneTB {
		return fmt.Errorf("max file size is 1TB")
	}

	extraHeaders := map[string]string{
		"x-ms-content-length": strconv.FormatUint(maxSize, 10),
		"x-ms-type":           "file",
	}

	headers, err := f.fsc.createResource(f.buildPath(), resourceFile, mergeMDIntoExtraHeaders(f.Metadata, extraHeaders))
	if err != nil {
		return err
	}

	f.Properties.Length = maxSize
	f.updateEtagAndLastModified(headers)
	return nil
}

// Delete immediately removes this file from the storage account.
//
// See https://msdn.microsoft.com/en-us/library/azure/dn689085.aspx
func (f *File) Delete() error {
	return f.fsc.deleteResource(f.buildPath(), resourceFile)
}

// DeleteIfExists removes this file if it exists.
//
// See https://msdn.microsoft.com/en-us/library/azure/dn689085.aspx
func (f *File) DeleteIfExists() (bool, error) {
	resp, err := f.fsc.deleteResourceNoClose(f.buildPath(), resourceFile)
	if resp != nil {
		defer resp.body.Close()
		if resp.statusCode == http.StatusAccepted || resp.statusCode == http.StatusNotFound {
			return resp.statusCode == http.StatusAccepted, nil
		}
	}
	return false, err
}

// DownloadRangeToStream operation downloads the specified range of this file with optional MD5 hash.
//
// See https://docs.microsoft.com/en-us/rest/api/storageservices/fileservices/get-file
func (f *File) DownloadRangeToStream(fileRange FileRange, getContentMD5 bool) (fs FileStream, err error) {
	if getContentMD5 && isRangeTooBig(fileRange) {
		return fs, fmt.Errorf("must specify a range less than or equal to 4MB when getContentMD5 is true")
	}

	extraHeaders := map[string]string{
		"Range": fileRange.String(),
	}
	if getContentMD5 == true {
		extraHeaders["x-ms-range-get-content-md5"] = "true"
	}

	resp, err := f.fsc.getResourceNoClose(f.buildPath(), compNone, resourceFile, http.MethodGet, extraHeaders)
	if err != nil {
		return fs, err
	}

	if err = checkRespCode(resp.statusCode, []int{http.StatusOK, http.StatusPartialContent}); err != nil {
		resp.body.Close()
		return fs, err
	}

	fs.Body = resp.body
	if getContentMD5 {
		fs.ContentMD5 = resp.headers.Get("Content-MD5")
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
func (f *File) FetchAttributes() error {
	headers, err := f.fsc.getResourceHeaders(f.buildPath(), compNone, resourceFile, http.MethodHead)
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

// ListRanges returns the list of valid ranges for this file.
//
// See https://msdn.microsoft.com/en-us/library/azure/dn166984.aspx
func (f *File) ListRanges(listRange *FileRange) (*FileRanges, error) {
	params := url.Values{"comp": {"rangelist"}}

	// add optional range to list
	var headers map[string]string
	if listRange != nil {
		headers = make(map[string]string)
		headers["Range"] = listRange.String()
	}

	resp, err := f.fsc.listContent(f.buildPath(), params, headers)
	if err != nil {
		return nil, err
	}

	defer resp.body.Close()
	var cl uint64
	cl, err = strconv.ParseUint(resp.headers.Get("x-ms-content-length"), 10, 64)
	if err != nil {
		return nil, err
	}

	var out FileRanges
	out.ContentLength = cl
	out.ETag = resp.headers.Get("ETag")
	out.LastModified = resp.headers.Get("Last-Modified")

	err = xmlUnmarshal(resp.body, &out)
	return &out, err
}

// modifies a range of bytes in this file
func (f *File) modifyRange(bytes io.Reader, fileRange FileRange, contentMD5 *string) (http.Header, error) {
	if err := f.fsc.checkForStorageEmulator(); err != nil {
		return nil, err
	}
	if fileRange.End < fileRange.Start {
		return nil, errors.New("the value for rangeEnd must be greater than or equal to rangeStart")
	}
	if bytes != nil && isRangeTooBig(fileRange) {
		return nil, errors.New("range cannot exceed 4MB in size")
	}

	uri := f.fsc.client.getEndpoint(fileServiceName, f.buildPath(), url.Values{"comp": {"range"}})

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
	defer resp.body.Close()
	return resp.headers, checkRespCode(resp.statusCode, []int{http.StatusCreated})
}

// SetMetadata replaces the metadata for this file.
//
// Some keys may be converted to Camel-Case before sending. All keys
// are returned in lower case by GetFileMetadata. HTTP header names
// are case-insensitive so case munging should not matter to other
// applications either.
//
// See https://msdn.microsoft.com/en-us/library/azure/dn689097.aspx
func (f *File) SetMetadata() error {
	headers, err := f.fsc.setResourceHeaders(f.buildPath(), compMetadata, resourceFile, mergeMDIntoExtraHeaders(f.Metadata, nil))
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
// See https://msdn.microsoft.com/en-us/library/azure/dn166975.aspx
func (f *File) SetProperties() error {
	headers, err := f.fsc.setResourceHeaders(f.buildPath(), compProperties, resourceFile, headersFromStruct(f.Properties))
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
	return f.fsc.client.getEndpoint(fileServiceName, f.buildPath(), url.Values{})
}

// WriteRange writes a range of bytes to this file with an optional MD5 hash of the content.
// Note that the length of bytes must match (rangeEnd - rangeStart) + 1 with a maximum size of 4MB.
//
// See https://msdn.microsoft.com/en-us/library/azure/dn194276.aspx
func (f *File) WriteRange(bytes io.Reader, fileRange FileRange, contentMD5 *string) error {
	if bytes == nil {
		return errors.New("bytes cannot be nil")
	}

	headers, err := f.modifyRange(bytes, fileRange, contentMD5)
	if err != nil {
		return err
	}

	f.updateEtagAndLastModified(headers)
	return nil
}
