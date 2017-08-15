package storage

import (
	"bytes"
	"fmt"
	"net/http"
	"net/url"
	"time"
)

// PutAppendBlob initializes an empty append blob with specified name. An
// append blob must be created using this method before appending blocks.
//
// See CreateBlockBlobFromReader for more info on creating blobs.
//
// See https://docs.microsoft.com/en-us/rest/api/storageservices/fileservices/Put-Blob
func (b *Blob) PutAppendBlob(options *PutBlobOptions) error {
	params := url.Values{}
	headers := b.Container.bsc.client.getStandardHeaders()
	headers["x-ms-blob-type"] = string(BlobTypeAppend)
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

// AppendBlockOptions includes the options for an append block operation
type AppendBlockOptions struct {
	Timeout           uint
	LeaseID           string     `header:"x-ms-lease-id"`
	MaxSize           *uint      `header:"x-ms-blob-condition-maxsize"`
	AppendPosition    *uint      `header:"x-ms-blob-condition-appendpos"`
	IfModifiedSince   *time.Time `header:"If-Modified-Since"`
	IfUnmodifiedSince *time.Time `header:"If-Unmodified-Since"`
	IfMatch           string     `header:"If-Match"`
	IfNoneMatch       string     `header:"If-None-Match"`
	RequestID         string     `header:"x-ms-client-request-id"`
}

// AppendBlock appends a block to an append blob.
//
// See https://docs.microsoft.com/en-us/rest/api/storageservices/fileservices/Append-Block
func (b *Blob) AppendBlock(chunk []byte, options *AppendBlockOptions) error {
	params := url.Values{"comp": {"appendblock"}}
	headers := b.Container.bsc.client.getStandardHeaders()
	headers["x-ms-blob-type"] = string(BlobTypeAppend)
	headers["Content-Length"] = fmt.Sprintf("%v", len(chunk))

	if options != nil {
		params = addTimeout(params, options.Timeout)
		headers = mergeHeaders(headers, headersFromStruct(*options))
	}
	uri := b.Container.bsc.client.getEndpoint(blobServiceName, b.buildPath(), params)

	resp, err := b.Container.bsc.client.exec(http.MethodPut, uri, headers, bytes.NewReader(chunk), b.Container.bsc.auth)
	if err != nil {
		return err
	}
	readAndCloseBody(resp.body)
	return checkRespCode(resp.statusCode, []int{http.StatusCreated})
}
