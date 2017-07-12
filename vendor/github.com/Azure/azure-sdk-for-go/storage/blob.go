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

// A Blob is an entry in BlobListResponse.
type Blob struct {
	Container  *Container
	Name       string         `xml:"Name"`
	Snapshot   time.Time      `xml:"Snapshot"`
	Properties BlobProperties `xml:"Properties"`
	Metadata   BlobMetadata   `xml:"Metadata"`
}

// PutBlobOptions includes the options any put blob operation
// (page, block, append)
type PutBlobOptions struct {
	Timeout           uint
	LeaseID           string     `header:"x-ms-lease-id"`
	Origin            string     `header:"Origin"`
	IfModifiedSince   *time.Time `header:"If-Modified-Since"`
	IfUnmodifiedSince *time.Time `header:"If-Unmodified-Since"`
	IfMatch           string     `header:"If-Match"`
	IfNoneMatch       string     `header:"If-None-Match"`
	RequestID         string     `header:"x-ms-client-request-id"`
}

// BlobMetadata is a set of custom name/value pairs.
//
// See https://msdn.microsoft.com/en-us/library/azure/dd179404.aspx
type BlobMetadata map[string]string

type blobMetadataEntries struct {
	Entries []blobMetadataEntry `xml:",any"`
}
type blobMetadataEntry struct {
	XMLName xml.Name
	Value   string `xml:",chardata"`
}

// UnmarshalXML converts the xml:Metadata into Metadata map
func (bm *BlobMetadata) UnmarshalXML(d *xml.Decoder, start xml.StartElement) error {
	var entries blobMetadataEntries
	if err := d.DecodeElement(&entries, &start); err != nil {
		return err
	}
	for _, entry := range entries.Entries {
		if *bm == nil {
			*bm = make(BlobMetadata)
		}
		(*bm)[strings.ToLower(entry.XMLName.Local)] = entry.Value
	}
	return nil
}

// MarshalXML implements the xml.Marshaler interface. It encodes
// metadata name/value pairs as they would appear in an Azure
// ListBlobs response.
func (bm BlobMetadata) MarshalXML(enc *xml.Encoder, start xml.StartElement) error {
	entries := make([]blobMetadataEntry, 0, len(bm))
	for k, v := range bm {
		entries = append(entries, blobMetadataEntry{
			XMLName: xml.Name{Local: http.CanonicalHeaderKey(k)},
			Value:   v,
		})
	}
	return enc.EncodeElement(blobMetadataEntries{
		Entries: entries,
	}, start)
}

// BlobProperties contains various properties of a blob
// returned in various endpoints like ListBlobs or GetBlobProperties.
type BlobProperties struct {
	LastModified          TimeRFC1123 `xml:"Last-Modified"`
	Etag                  string      `xml:"Etag"`
	ContentMD5            string      `xml:"Content-MD5" header:"x-ms-blob-content-md5"`
	ContentLength         int64       `xml:"Content-Length"`
	ContentType           string      `xml:"Content-Type" header:"x-ms-blob-content-type"`
	ContentEncoding       string      `xml:"Content-Encoding" header:"x-ms-blob-content-encoding"`
	CacheControl          string      `xml:"Cache-Control" header:"x-ms-blob-cache-control"`
	ContentLanguage       string      `xml:"Cache-Language" header:"x-ms-blob-content-language"`
	ContentDisposition    string      `xml:"Content-Disposition" header:"x-ms-blob-content-disposition"`
	BlobType              BlobType    `xml:"x-ms-blob-blob-type"`
	SequenceNumber        int64       `xml:"x-ms-blob-sequence-number"`
	CopyID                string      `xml:"CopyId"`
	CopyStatus            string      `xml:"CopyStatus"`
	CopySource            string      `xml:"CopySource"`
	CopyProgress          string      `xml:"CopyProgress"`
	CopyCompletionTime    TimeRFC1123 `xml:"CopyCompletionTime"`
	CopyStatusDescription string      `xml:"CopyStatusDescription"`
	LeaseStatus           string      `xml:"LeaseStatus"`
	LeaseState            string      `xml:"LeaseState"`
	LeaseDuration         string      `xml:"LeaseDuration"`
	ServerEncrypted       bool        `xml:"ServerEncrypted"`
	IncrementalCopy       bool        `xml:"IncrementalCopy"`
}

// BlobType defines the type of the Azure Blob.
type BlobType string

// Types of page blobs
const (
	BlobTypeBlock  BlobType = "BlockBlob"
	BlobTypePage   BlobType = "PageBlob"
	BlobTypeAppend BlobType = "AppendBlob"
)

func (b *Blob) buildPath() string {
	return b.Container.buildPath() + "/" + b.Name
}

// Exists returns true if a blob with given name exists on the specified
// container of the storage account.
func (b *Blob) Exists() (bool, error) {
	uri := b.Container.bsc.client.getEndpoint(blobServiceName, b.buildPath(), nil)
	headers := b.Container.bsc.client.getStandardHeaders()
	resp, err := b.Container.bsc.client.exec(http.MethodHead, uri, headers, nil, b.Container.bsc.auth)
	if resp != nil {
		defer readAndCloseBody(resp.body)
		if resp.statusCode == http.StatusOK || resp.statusCode == http.StatusNotFound {
			return resp.statusCode == http.StatusOK, nil
		}
	}
	return false, err
}

// GetURL gets the canonical URL to the blob with the specified name in the
// specified container. If name is not specified, the canonical URL for the entire
// container is obtained.
// This method does not create a publicly accessible URL if the blob or container
// is private and this method does not check if the blob exists.
func (b *Blob) GetURL() string {
	container := b.Container.Name
	if container == "" {
		container = "$root"
	}
	return b.Container.bsc.client.getEndpoint(blobServiceName, pathForResource(container, b.Name), nil)
}

// GetBlobRangeOptions includes the options for a get blob range operation
type GetBlobRangeOptions struct {
	Range              *BlobRange
	GetRangeContentMD5 bool
	*GetBlobOptions
}

// GetBlobOptions includes the options for a get blob operation
type GetBlobOptions struct {
	Timeout           uint
	Snapshot          *time.Time
	LeaseID           string     `header:"x-ms-lease-id"`
	Origin            string     `header:"Origin"`
	IfModifiedSince   *time.Time `header:"If-Modified-Since"`
	IfUnmodifiedSince *time.Time `header:"If-Unmodified-Since"`
	IfMatch           string     `header:"If-Match"`
	IfNoneMatch       string     `header:"If-None-Match"`
	RequestID         string     `header:"x-ms-client-request-id"`
}

// BlobRange represents the bytes range to be get
type BlobRange struct {
	Start uint64
	End   uint64
}

func (br BlobRange) String() string {
	return fmt.Sprintf("bytes=%d-%d", br.Start, br.End)
}

// Get returns a stream to read the blob. Caller must call both Read and Close()
// to correctly close the underlying connection.
// See https://docs.microsoft.com/en-us/rest/api/storageservices/fileservices/Get-Blob
func (b *Blob) Get(options *GetBlobOptions) (io.ReadCloser, error) {
	rangeOptions := GetBlobRangeOptions{
		GetBlobOptions: options,
	}
	resp, err := b.getRange(&rangeOptions)
	if err != nil {
		return nil, err
	}

	if err := checkRespCode(resp.statusCode, []int{http.StatusOK}); err != nil {
		return nil, err
	}
	if err := b.writePropoerties(resp.headers); err != nil {
		return resp.body, err
	}
	return resp.body, nil
}

// GetRange reads the specified range of a blob to a stream. The bytesRange
// string must be in a format like "0-", "10-100" as defined in HTTP 1.1 spec.
// Caller must call both Read and Close()// to correctly close the underlying
// connection.
// See https://docs.microsoft.com/en-us/rest/api/storageservices/fileservices/Get-Blob
func (b *Blob) GetRange(options *GetBlobRangeOptions) (io.ReadCloser, error) {
	resp, err := b.getRange(options)
	if err != nil {
		return nil, err
	}

	if err := checkRespCode(resp.statusCode, []int{http.StatusPartialContent}); err != nil {
		return nil, err
	}
	if err := b.writePropoerties(resp.headers); err != nil {
		return resp.body, err
	}
	return resp.body, nil
}

func (b *Blob) getRange(options *GetBlobRangeOptions) (*storageResponse, error) {
	params := url.Values{}
	headers := b.Container.bsc.client.getStandardHeaders()

	if options != nil {
		if options.Range != nil {
			headers["Range"] = options.Range.String()
			headers["x-ms-range-get-content-md5"] = fmt.Sprintf("%v", options.GetRangeContentMD5)
		}
		if options.GetBlobOptions != nil {
			headers = mergeHeaders(headers, headersFromStruct(*options.GetBlobOptions))
			params = addTimeout(params, options.Timeout)
			params = addSnapshot(params, options.Snapshot)
		}
	}
	uri := b.Container.bsc.client.getEndpoint(blobServiceName, b.buildPath(), params)

	resp, err := b.Container.bsc.client.exec(http.MethodGet, uri, headers, nil, b.Container.bsc.auth)
	if err != nil {
		return nil, err
	}
	return resp, err
}

// SnapshotOptions includes the options for a snapshot blob operation
type SnapshotOptions struct {
	Timeout           uint
	LeaseID           string     `header:"x-ms-lease-id"`
	IfModifiedSince   *time.Time `header:"If-Modified-Since"`
	IfUnmodifiedSince *time.Time `header:"If-Unmodified-Since"`
	IfMatch           string     `header:"If-Match"`
	IfNoneMatch       string     `header:"If-None-Match"`
	RequestID         string     `header:"x-ms-client-request-id"`
}

// CreateSnapshot creates a snapshot for a blob
// See https://msdn.microsoft.com/en-us/library/azure/ee691971.aspx
func (b *Blob) CreateSnapshot(options *SnapshotOptions) (snapshotTimestamp *time.Time, err error) {
	params := url.Values{"comp": {"snapshot"}}
	headers := b.Container.bsc.client.getStandardHeaders()
	headers = b.Container.bsc.client.addMetadataToHeaders(headers, b.Metadata)

	if options != nil {
		params = addTimeout(params, options.Timeout)
		headers = mergeHeaders(headers, headersFromStruct(*options))
	}
	uri := b.Container.bsc.client.getEndpoint(blobServiceName, b.buildPath(), params)

	resp, err := b.Container.bsc.client.exec(http.MethodPut, uri, headers, nil, b.Container.bsc.auth)
	if err != nil || resp == nil {
		return nil, err
	}
	defer readAndCloseBody(resp.body)

	if err := checkRespCode(resp.statusCode, []int{http.StatusCreated}); err != nil {
		return nil, err
	}

	snapshotResponse := resp.headers.Get(http.CanonicalHeaderKey("x-ms-snapshot"))
	if snapshotResponse != "" {
		snapshotTimestamp, err := time.Parse(time.RFC3339, snapshotResponse)
		if err != nil {
			return nil, err
		}
		return &snapshotTimestamp, nil
	}

	return nil, errors.New("Snapshot not created")
}

// GetBlobPropertiesOptions includes the options for a get blob properties operation
type GetBlobPropertiesOptions struct {
	Timeout           uint
	Snapshot          *time.Time
	LeaseID           string     `header:"x-ms-lease-id"`
	IfModifiedSince   *time.Time `header:"If-Modified-Since"`
	IfUnmodifiedSince *time.Time `header:"If-Unmodified-Since"`
	IfMatch           string     `header:"If-Match"`
	IfNoneMatch       string     `header:"If-None-Match"`
	RequestID         string     `header:"x-ms-client-request-id"`
}

// GetProperties provides various information about the specified blob.
// See https://msdn.microsoft.com/en-us/library/azure/dd179394.aspx
func (b *Blob) GetProperties(options *GetBlobPropertiesOptions) error {
	params := url.Values{}
	headers := b.Container.bsc.client.getStandardHeaders()

	if options != nil {
		params = addTimeout(params, options.Timeout)
		params = addSnapshot(params, options.Snapshot)
		headers = mergeHeaders(headers, headersFromStruct(*options))
	}
	uri := b.Container.bsc.client.getEndpoint(blobServiceName, b.buildPath(), params)

	resp, err := b.Container.bsc.client.exec(http.MethodHead, uri, headers, nil, b.Container.bsc.auth)
	if err != nil {
		return err
	}
	defer readAndCloseBody(resp.body)

	if err = checkRespCode(resp.statusCode, []int{http.StatusOK}); err != nil {
		return err
	}
	return b.writePropoerties(resp.headers)
}

func (b *Blob) writePropoerties(h http.Header) error {
	var err error

	var contentLength int64
	contentLengthStr := h.Get("Content-Length")
	if contentLengthStr != "" {
		contentLength, err = strconv.ParseInt(contentLengthStr, 0, 64)
		if err != nil {
			return err
		}
	}

	var sequenceNum int64
	sequenceNumStr := h.Get("x-ms-blob-sequence-number")
	if sequenceNumStr != "" {
		sequenceNum, err = strconv.ParseInt(sequenceNumStr, 0, 64)
		if err != nil {
			return err
		}
	}

	lastModified, err := getTimeFromHeaders(h, "Last-Modified")
	if err != nil {
		return err
	}

	copyCompletionTime, err := getTimeFromHeaders(h, "x-ms-copy-completion-time")
	if err != nil {
		return err
	}

	b.Properties = BlobProperties{
		LastModified:          TimeRFC1123(*lastModified),
		Etag:                  h.Get("Etag"),
		ContentMD5:            h.Get("Content-MD5"),
		ContentLength:         contentLength,
		ContentEncoding:       h.Get("Content-Encoding"),
		ContentType:           h.Get("Content-Type"),
		ContentDisposition:    h.Get("Content-Disposition"),
		CacheControl:          h.Get("Cache-Control"),
		ContentLanguage:       h.Get("Content-Language"),
		SequenceNumber:        sequenceNum,
		CopyCompletionTime:    TimeRFC1123(*copyCompletionTime),
		CopyStatusDescription: h.Get("x-ms-copy-status-description"),
		CopyID:                h.Get("x-ms-copy-id"),
		CopyProgress:          h.Get("x-ms-copy-progress"),
		CopySource:            h.Get("x-ms-copy-source"),
		CopyStatus:            h.Get("x-ms-copy-status"),
		BlobType:              BlobType(h.Get("x-ms-blob-type")),
		LeaseStatus:           h.Get("x-ms-lease-status"),
		LeaseState:            h.Get("x-ms-lease-state"),
	}
	b.writeMetadata(h)
	return nil
}

// SetBlobPropertiesOptions contains various properties of a blob and is an entry
// in SetProperties
type SetBlobPropertiesOptions struct {
	Timeout              uint
	LeaseID              string     `header:"x-ms-lease-id"`
	Origin               string     `header:"Origin"`
	IfModifiedSince      *time.Time `header:"If-Modified-Since"`
	IfUnmodifiedSince    *time.Time `header:"If-Unmodified-Since"`
	IfMatch              string     `header:"If-Match"`
	IfNoneMatch          string     `header:"If-None-Match"`
	SequenceNumberAction *SequenceNumberAction
	RequestID            string `header:"x-ms-client-request-id"`
}

// SequenceNumberAction defines how the blob's sequence number should be modified
type SequenceNumberAction string

// Options for sequence number action
const (
	SequenceNumberActionMax       SequenceNumberAction = "max"
	SequenceNumberActionUpdate    SequenceNumberAction = "update"
	SequenceNumberActionIncrement SequenceNumberAction = "increment"
)

// SetProperties replaces the BlobHeaders for the specified blob.
//
// Some keys may be converted to Camel-Case before sending. All keys
// are returned in lower case by GetBlobProperties. HTTP header names
// are case-insensitive so case munging should not matter to other
// applications either.
//
// See https://docs.microsoft.com/en-us/rest/api/storageservices/fileservices/Set-Blob-Properties
func (b *Blob) SetProperties(options *SetBlobPropertiesOptions) error {
	params := url.Values{"comp": {"properties"}}
	headers := b.Container.bsc.client.getStandardHeaders()
	headers = mergeHeaders(headers, headersFromStruct(b.Properties))

	if options != nil {
		params = addTimeout(params, options.Timeout)
		headers = mergeHeaders(headers, headersFromStruct(*options))
	}
	uri := b.Container.bsc.client.getEndpoint(blobServiceName, b.buildPath(), params)

	if b.Properties.BlobType == BlobTypePage {
		headers = addToHeaders(headers, "x-ms-blob-content-length", fmt.Sprintf("byte %v", b.Properties.ContentLength))
		if options != nil || options.SequenceNumberAction != nil {
			headers = addToHeaders(headers, "x-ms-sequence-number-action", string(*options.SequenceNumberAction))
			if *options.SequenceNumberAction != SequenceNumberActionIncrement {
				headers = addToHeaders(headers, "x-ms-blob-sequence-number", fmt.Sprintf("%v", b.Properties.SequenceNumber))
			}
		}
	}

	resp, err := b.Container.bsc.client.exec(http.MethodPut, uri, headers, nil, b.Container.bsc.auth)
	if err != nil {
		return err
	}
	readAndCloseBody(resp.body)
	return checkRespCode(resp.statusCode, []int{http.StatusOK})
}

// SetBlobMetadataOptions includes the options for a set blob metadata operation
type SetBlobMetadataOptions struct {
	Timeout           uint
	LeaseID           string     `header:"x-ms-lease-id"`
	IfModifiedSince   *time.Time `header:"If-Modified-Since"`
	IfUnmodifiedSince *time.Time `header:"If-Unmodified-Since"`
	IfMatch           string     `header:"If-Match"`
	IfNoneMatch       string     `header:"If-None-Match"`
	RequestID         string     `header:"x-ms-client-request-id"`
}

// SetMetadata replaces the metadata for the specified blob.
//
// Some keys may be converted to Camel-Case before sending. All keys
// are returned in lower case by GetBlobMetadata. HTTP header names
// are case-insensitive so case munging should not matter to other
// applications either.
//
// See https://msdn.microsoft.com/en-us/library/azure/dd179414.aspx
func (b *Blob) SetMetadata(options *SetBlobMetadataOptions) error {
	params := url.Values{"comp": {"metadata"}}
	headers := b.Container.bsc.client.getStandardHeaders()
	headers = b.Container.bsc.client.addMetadataToHeaders(headers, b.Metadata)

	if options != nil {
		params = addTimeout(params, options.Timeout)
		headers = mergeHeaders(headers, headersFromStruct(*options))
	}
	uri := b.Container.bsc.client.getEndpoint(blobServiceName, b.buildPath(), params)

	resp, err := b.Container.bsc.client.exec(http.MethodPut, uri, headers, nil, b.Container.bsc.auth)
	if err != nil {
		return err
	}
	readAndCloseBody(resp.body)
	return checkRespCode(resp.statusCode, []int{http.StatusOK})
}

// GetBlobMetadataOptions includes the options for a get blob metadata operation
type GetBlobMetadataOptions struct {
	Timeout           uint
	Snapshot          *time.Time
	LeaseID           string     `header:"x-ms-lease-id"`
	IfModifiedSince   *time.Time `header:"If-Modified-Since"`
	IfUnmodifiedSince *time.Time `header:"If-Unmodified-Since"`
	IfMatch           string     `header:"If-Match"`
	IfNoneMatch       string     `header:"If-None-Match"`
	RequestID         string     `header:"x-ms-client-request-id"`
}

// GetMetadata returns all user-defined metadata for the specified blob.
//
// All metadata keys will be returned in lower case. (HTTP header
// names are case-insensitive.)
//
// See https://msdn.microsoft.com/en-us/library/azure/dd179414.aspx
func (b *Blob) GetMetadata(options *GetBlobMetadataOptions) error {
	params := url.Values{"comp": {"metadata"}}
	headers := b.Container.bsc.client.getStandardHeaders()

	if options != nil {
		params = addTimeout(params, options.Timeout)
		params = addSnapshot(params, options.Snapshot)
		headers = mergeHeaders(headers, headersFromStruct(*options))
	}
	uri := b.Container.bsc.client.getEndpoint(blobServiceName, b.buildPath(), params)

	resp, err := b.Container.bsc.client.exec(http.MethodGet, uri, headers, nil, b.Container.bsc.auth)
	if err != nil {
		return err
	}
	defer readAndCloseBody(resp.body)

	if err := checkRespCode(resp.statusCode, []int{http.StatusOK}); err != nil {
		return err
	}

	b.writeMetadata(resp.headers)
	return nil
}

func (b *Blob) writeMetadata(h http.Header) {
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

	b.Metadata = BlobMetadata(metadata)
}

// DeleteBlobOptions includes the options for a delete blob operation
type DeleteBlobOptions struct {
	Timeout           uint
	Snapshot          *time.Time
	LeaseID           string `header:"x-ms-lease-id"`
	DeleteSnapshots   *bool
	IfModifiedSince   *time.Time `header:"If-Modified-Since"`
	IfUnmodifiedSince *time.Time `header:"If-Unmodified-Since"`
	IfMatch           string     `header:"If-Match"`
	IfNoneMatch       string     `header:"If-None-Match"`
	RequestID         string     `header:"x-ms-client-request-id"`
}

// Delete deletes the given blob from the specified container.
// If the blob does not exists at the time of the Delete Blob operation, it
// returns error.
// See https://docs.microsoft.com/en-us/rest/api/storageservices/fileservices/Delete-Blob
func (b *Blob) Delete(options *DeleteBlobOptions) error {
	resp, err := b.delete(options)
	if err != nil {
		return err
	}
	readAndCloseBody(resp.body)
	return checkRespCode(resp.statusCode, []int{http.StatusAccepted})
}

// DeleteIfExists deletes the given blob from the specified container If the
// blob is deleted with this call, returns true. Otherwise returns false.
//
// See https://docs.microsoft.com/en-us/rest/api/storageservices/fileservices/Delete-Blob
func (b *Blob) DeleteIfExists(options *DeleteBlobOptions) (bool, error) {
	resp, err := b.delete(options)
	if resp != nil {
		defer readAndCloseBody(resp.body)
		if resp.statusCode == http.StatusAccepted || resp.statusCode == http.StatusNotFound {
			return resp.statusCode == http.StatusAccepted, nil
		}
	}
	return false, err
}

func (b *Blob) delete(options *DeleteBlobOptions) (*storageResponse, error) {
	params := url.Values{}
	headers := b.Container.bsc.client.getStandardHeaders()

	if options != nil {
		params = addTimeout(params, options.Timeout)
		params = addSnapshot(params, options.Snapshot)
		headers = mergeHeaders(headers, headersFromStruct(*options))
		if options.DeleteSnapshots != nil {
			if *options.DeleteSnapshots {
				headers["x-ms-delete-snapshots"] = "include"
			} else {
				headers["x-ms-delete-snapshots"] = "only"
			}
		}
	}
	uri := b.Container.bsc.client.getEndpoint(blobServiceName, b.buildPath(), params)
	return b.Container.bsc.client.exec(http.MethodDelete, uri, headers, nil, b.Container.bsc.auth)
}

// helper method to construct the path to either a blob or container
func pathForResource(container, name string) string {
	if name != "" {
		return fmt.Sprintf("/%s/%s", container, name)
	}
	return fmt.Sprintf("/%s", container)
}
