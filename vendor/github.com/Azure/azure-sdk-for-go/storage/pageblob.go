package storage

import (
	"encoding/xml"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"time"
)

// GetPageRangesResponse contains the response fields from
// Get Page Ranges call.
//
// See https://msdn.microsoft.com/en-us/library/azure/ee691973.aspx
type GetPageRangesResponse struct {
	XMLName  xml.Name    `xml:"PageList"`
	PageList []PageRange `xml:"PageRange"`
}

// PageRange contains information about a page of a page blob from
// Get Pages Range call.
//
// See https://msdn.microsoft.com/en-us/library/azure/ee691973.aspx
type PageRange struct {
	Start int64 `xml:"Start"`
	End   int64 `xml:"End"`
}

var (
	errBlobCopyAborted    = errors.New("storage: blob copy is aborted")
	errBlobCopyIDMismatch = errors.New("storage: blob copy id is a mismatch")
)

// PutPageOptions includes the options for a put page operation
type PutPageOptions struct {
	Timeout                           uint
	LeaseID                           string     `header:"x-ms-lease-id"`
	IfSequenceNumberLessThanOrEqualTo *int       `header:"x-ms-if-sequence-number-le"`
	IfSequenceNumberLessThan          *int       `header:"x-ms-if-sequence-number-lt"`
	IfSequenceNumberEqualTo           *int       `header:"x-ms-if-sequence-number-eq"`
	IfModifiedSince                   *time.Time `header:"If-Modified-Since"`
	IfUnmodifiedSince                 *time.Time `header:"If-Unmodified-Since"`
	IfMatch                           string     `header:"If-Match"`
	IfNoneMatch                       string     `header:"If-None-Match"`
	RequestID                         string     `header:"x-ms-client-request-id"`
}

// WriteRange writes a range of pages to a page blob.
// Ranges must be aligned with 512-byte boundaries and chunk must be of size
// multiplies by 512.
//
// See https://docs.microsoft.com/en-us/rest/api/storageservices/fileservices/Put-Page
func (b *Blob) WriteRange(blobRange BlobRange, bytes io.Reader, options *PutPageOptions) error {
	if bytes == nil {
		return errors.New("bytes cannot be nil")
	}
	return b.modifyRange(blobRange, bytes, options)
}

// ClearRange clears the given range in a page blob.
// Ranges must be aligned with 512-byte boundaries and chunk must be of size
// multiplies by 512.
//
// See https://docs.microsoft.com/en-us/rest/api/storageservices/fileservices/Put-Page
func (b *Blob) ClearRange(blobRange BlobRange, options *PutPageOptions) error {
	return b.modifyRange(blobRange, nil, options)
}

func (b *Blob) modifyRange(blobRange BlobRange, bytes io.Reader, options *PutPageOptions) error {
	if blobRange.End < blobRange.Start {
		return errors.New("the value for rangeEnd must be greater than or equal to rangeStart")
	}
	if blobRange.Start%512 != 0 {
		return errors.New("the value for rangeStart must be a modulus of 512")
	}
	if blobRange.End%512 != 511 {
		return errors.New("the value for rangeEnd must be a modulus of 511")
	}

	params := url.Values{"comp": {"page"}}

	// default to clear
	write := "clear"
	var cl uint64

	// if bytes is not nil then this is an update operation
	if bytes != nil {
		write = "update"
		cl = (blobRange.End - blobRange.Start) + 1
	}

	headers := b.Container.bsc.client.getStandardHeaders()
	headers["x-ms-blob-type"] = string(BlobTypePage)
	headers["x-ms-page-write"] = write
	headers["x-ms-range"] = blobRange.String()
	headers["Content-Length"] = fmt.Sprintf("%v", cl)

	if options != nil {
		params = addTimeout(params, options.Timeout)
		headers = mergeHeaders(headers, headersFromStruct(*options))
	}
	uri := b.Container.bsc.client.getEndpoint(blobServiceName, b.buildPath(), params)

	resp, err := b.Container.bsc.client.exec(http.MethodPut, uri, headers, bytes, b.Container.bsc.auth)
	if err != nil {
		return err
	}
	readAndCloseBody(resp.body)

	return checkRespCode(resp.statusCode, []int{http.StatusCreated})
}

// GetPageRangesOptions includes the options for a get page ranges operation
type GetPageRangesOptions struct {
	Timeout          uint
	Snapshot         *time.Time
	PreviousSnapshot *time.Time
	Range            *BlobRange
	LeaseID          string `header:"x-ms-lease-id"`
	RequestID        string `header:"x-ms-client-request-id"`
}

// GetPageRanges returns the list of valid page ranges for a page blob.
//
// See https://docs.microsoft.com/en-us/rest/api/storageservices/fileservices/Get-Page-Ranges
func (b *Blob) GetPageRanges(options *GetPageRangesOptions) (GetPageRangesResponse, error) {
	params := url.Values{"comp": {"pagelist"}}
	headers := b.Container.bsc.client.getStandardHeaders()

	if options != nil {
		params = addTimeout(params, options.Timeout)
		params = addSnapshot(params, options.Snapshot)
		if options.PreviousSnapshot != nil {
			params.Add("prevsnapshot", timeRfc1123Formatted(*options.PreviousSnapshot))
		}
		if options.Range != nil {
			headers["Range"] = options.Range.String()
		}
		headers = mergeHeaders(headers, headersFromStruct(*options))
	}
	uri := b.Container.bsc.client.getEndpoint(blobServiceName, b.buildPath(), params)

	var out GetPageRangesResponse
	resp, err := b.Container.bsc.client.exec(http.MethodGet, uri, headers, nil, b.Container.bsc.auth)
	if err != nil {
		return out, err
	}
	defer resp.body.Close()

	if err = checkRespCode(resp.statusCode, []int{http.StatusOK}); err != nil {
		return out, err
	}
	err = xmlUnmarshal(resp.body, &out)
	return out, err
}

// PutPageBlob initializes an empty page blob with specified name and maximum
// size in bytes (size must be aligned to a 512-byte boundary). A page blob must
// be created using this method before writing pages.
//
// See CreateBlockBlobFromReader for more info on creating blobs.
//
// See https://docs.microsoft.com/en-us/rest/api/storageservices/fileservices/Put-Blob
func (b *Blob) PutPageBlob(options *PutBlobOptions) error {
	if b.Properties.ContentLength%512 != 0 {
		return errors.New("Content length must be aligned to a 512-byte boundary")
	}

	params := url.Values{}
	headers := b.Container.bsc.client.getStandardHeaders()
	headers["x-ms-blob-type"] = string(BlobTypePage)
	headers["x-ms-blob-content-length"] = fmt.Sprintf("%v", b.Properties.ContentLength)
	headers["x-ms-blob-sequence-number"] = fmt.Sprintf("%v", b.Properties.SequenceNumber)
	headers = mergeHeaders(headers, headersFromStruct(b.Properties))
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
	return checkRespCode(resp.statusCode, []int{http.StatusCreated})
}
