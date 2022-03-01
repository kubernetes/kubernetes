package storage

// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See License.txt in the project root for license information.

import (
	"bytes"
	"crypto/md5"
	"encoding/base64"
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
	return b.respondCreation(resp, BlobTypeAppend)
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
	ContentMD5        bool
}

// AppendBlock appends a block to an append blob.
//
// See https://docs.microsoft.com/en-us/rest/api/storageservices/fileservices/Append-Block
func (b *Blob) AppendBlock(chunk []byte, options *AppendBlockOptions) error {
	params := url.Values{"comp": {"appendblock"}}
	headers := b.Container.bsc.client.getStandardHeaders()
	headers["Content-Length"] = fmt.Sprintf("%v", len(chunk))

	if options != nil {
		params = addTimeout(params, options.Timeout)
		headers = mergeHeaders(headers, headersFromStruct(*options))
		if options.ContentMD5 {
			md5sum := md5.Sum(chunk)
			headers[headerContentMD5] = base64.StdEncoding.EncodeToString(md5sum[:])
		}
	}
	uri := b.Container.bsc.client.getEndpoint(blobServiceName, b.buildPath(), params)

	resp, err := b.Container.bsc.client.exec(http.MethodPut, uri, headers, bytes.NewReader(chunk), b.Container.bsc.auth)
	if err != nil {
		return err
	}
	return b.respondCreation(resp, BlobTypeAppend)
}
