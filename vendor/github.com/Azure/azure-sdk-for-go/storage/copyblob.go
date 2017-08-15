package storage

import (
	"errors"
	"fmt"
	"net/http"
	"net/url"
	"strings"
	"time"
)

const (
	blobCopyStatusPending = "pending"
	blobCopyStatusSuccess = "success"
	blobCopyStatusAborted = "aborted"
	blobCopyStatusFailed  = "failed"
)

// CopyOptions includes the options for a copy blob operation
type CopyOptions struct {
	Timeout   uint
	Source    CopyOptionsConditions
	Destiny   CopyOptionsConditions
	RequestID string
}

// IncrementalCopyOptions includes the options for an incremental copy blob operation
type IncrementalCopyOptions struct {
	Timeout     uint
	Destination IncrementalCopyOptionsConditions
	RequestID   string
}

// CopyOptionsConditions includes some conditional options in a copy blob operation
type CopyOptionsConditions struct {
	LeaseID           string
	IfModifiedSince   *time.Time
	IfUnmodifiedSince *time.Time
	IfMatch           string
	IfNoneMatch       string
}

// IncrementalCopyOptionsConditions includes some conditional options in a copy blob operation
type IncrementalCopyOptionsConditions struct {
	IfModifiedSince   *time.Time
	IfUnmodifiedSince *time.Time
	IfMatch           string
	IfNoneMatch       string
}

// Copy starts a blob copy operation and waits for the operation to
// complete. sourceBlob parameter must be a canonical URL to the blob (can be
// obtained using the GetURL method.) There is no SLA on blob copy and therefore
// this helper method works faster on smaller files.
//
// See https://docs.microsoft.com/en-us/rest/api/storageservices/fileservices/Copy-Blob
func (b *Blob) Copy(sourceBlob string, options *CopyOptions) error {
	copyID, err := b.StartCopy(sourceBlob, options)
	if err != nil {
		return err
	}

	return b.WaitForCopy(copyID)
}

// StartCopy starts a blob copy operation.
// sourceBlob parameter must be a canonical URL to the blob (can be
// obtained using the GetURL method.)
//
// See https://docs.microsoft.com/en-us/rest/api/storageservices/fileservices/Copy-Blob
func (b *Blob) StartCopy(sourceBlob string, options *CopyOptions) (string, error) {
	params := url.Values{}
	headers := b.Container.bsc.client.getStandardHeaders()
	headers["x-ms-copy-source"] = sourceBlob
	headers = b.Container.bsc.client.addMetadataToHeaders(headers, b.Metadata)

	if options != nil {
		params = addTimeout(params, options.Timeout)
		headers = addToHeaders(headers, "x-ms-client-request-id", options.RequestID)
		// source
		headers = addToHeaders(headers, "x-ms-source-lease-id", options.Source.LeaseID)
		headers = addTimeToHeaders(headers, "x-ms-source-if-modified-since", options.Source.IfModifiedSince)
		headers = addTimeToHeaders(headers, "x-ms-source-if-unmodified-since", options.Source.IfUnmodifiedSince)
		headers = addToHeaders(headers, "x-ms-source-if-match", options.Source.IfMatch)
		headers = addToHeaders(headers, "x-ms-source-if-none-match", options.Source.IfNoneMatch)
		//destiny
		headers = addToHeaders(headers, "x-ms-lease-id", options.Destiny.LeaseID)
		headers = addTimeToHeaders(headers, "x-ms-if-modified-since", options.Destiny.IfModifiedSince)
		headers = addTimeToHeaders(headers, "x-ms-if-unmodified-since", options.Destiny.IfUnmodifiedSince)
		headers = addToHeaders(headers, "x-ms-if-match", options.Destiny.IfMatch)
		headers = addToHeaders(headers, "x-ms-if-none-match", options.Destiny.IfNoneMatch)
	}
	uri := b.Container.bsc.client.getEndpoint(blobServiceName, b.buildPath(), params)

	resp, err := b.Container.bsc.client.exec(http.MethodPut, uri, headers, nil, b.Container.bsc.auth)
	if err != nil {
		return "", err
	}
	defer readAndCloseBody(resp.body)

	if err := checkRespCode(resp.statusCode, []int{http.StatusAccepted, http.StatusCreated}); err != nil {
		return "", err
	}

	copyID := resp.headers.Get("x-ms-copy-id")
	if copyID == "" {
		return "", errors.New("Got empty copy id header")
	}
	return copyID, nil
}

// AbortCopyOptions includes the options for an abort blob operation
type AbortCopyOptions struct {
	Timeout   uint
	LeaseID   string `header:"x-ms-lease-id"`
	RequestID string `header:"x-ms-client-request-id"`
}

// AbortCopy aborts a BlobCopy which has already been triggered by the StartBlobCopy function.
// copyID is generated from StartBlobCopy function.
// currentLeaseID is required IF the destination blob has an active lease on it.
// See https://docs.microsoft.com/en-us/rest/api/storageservices/fileservices/Abort-Copy-Blob
func (b *Blob) AbortCopy(copyID string, options *AbortCopyOptions) error {
	params := url.Values{
		"comp":   {"copy"},
		"copyid": {copyID},
	}
	headers := b.Container.bsc.client.getStandardHeaders()
	headers["x-ms-copy-action"] = "abort"

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
	return checkRespCode(resp.statusCode, []int{http.StatusNoContent})
}

// WaitForCopy loops until a BlobCopy operation is completed (or fails with error)
func (b *Blob) WaitForCopy(copyID string) error {
	for {
		err := b.GetProperties(nil)
		if err != nil {
			return err
		}

		if b.Properties.CopyID != copyID {
			return errBlobCopyIDMismatch
		}

		switch b.Properties.CopyStatus {
		case blobCopyStatusSuccess:
			return nil
		case blobCopyStatusPending:
			continue
		case blobCopyStatusAborted:
			return errBlobCopyAborted
		case blobCopyStatusFailed:
			return fmt.Errorf("storage: blob copy failed. Id=%s Description=%s", b.Properties.CopyID, b.Properties.CopyStatusDescription)
		default:
			return fmt.Errorf("storage: unhandled blob copy status: '%s'", b.Properties.CopyStatus)
		}
	}
}

// IncrementalCopyBlob copies a snapshot of a source blob and copies to referring blob
// sourceBlob parameter must be a valid snapshot URL of the original blob.
// THe original blob mut be public, or use a Shared Access Signature.
//
// See https://docs.microsoft.com/en-us/rest/api/storageservices/fileservices/incremental-copy-blob .
func (b *Blob) IncrementalCopyBlob(sourceBlobURL string, snapshotTime time.Time, options *IncrementalCopyOptions) (string, error) {
	params := url.Values{"comp": {"incrementalcopy"}}

	// need formatting to 7 decimal places so it's friendly to Windows and *nix
	snapshotTimeFormatted := snapshotTime.Format("2006-01-02T15:04:05.0000000Z")
	u, err := url.Parse(sourceBlobURL)
	if err != nil {
		return "", err
	}
	query := u.Query()
	query.Add("snapshot", snapshotTimeFormatted)
	encodedQuery := query.Encode()
	encodedQuery = strings.Replace(encodedQuery, "%3A", ":", -1)
	u.RawQuery = encodedQuery
	snapshotURL := u.String()

	headers := b.Container.bsc.client.getStandardHeaders()
	headers["x-ms-copy-source"] = snapshotURL

	if options != nil {
		addTimeout(params, options.Timeout)
		headers = addToHeaders(headers, "x-ms-client-request-id", options.RequestID)
		headers = addTimeToHeaders(headers, "x-ms-if-modified-since", options.Destination.IfModifiedSince)
		headers = addTimeToHeaders(headers, "x-ms-if-unmodified-since", options.Destination.IfUnmodifiedSince)
		headers = addToHeaders(headers, "x-ms-if-match", options.Destination.IfMatch)
		headers = addToHeaders(headers, "x-ms-if-none-match", options.Destination.IfNoneMatch)
	}

	// get URI of destination blob
	uri := b.Container.bsc.client.getEndpoint(blobServiceName, b.buildPath(), params)

	resp, err := b.Container.bsc.client.exec(http.MethodPut, uri, headers, nil, b.Container.bsc.auth)
	if err != nil {
		return "", err
	}
	defer readAndCloseBody(resp.body)

	if err := checkRespCode(resp.statusCode, []int{http.StatusAccepted}); err != nil {
		return "", err
	}

	copyID := resp.headers.Get("x-ms-copy-id")
	if copyID == "" {
		return "", errors.New("Got empty copy id header")
	}
	return copyID, nil
}
