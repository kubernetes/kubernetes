package storage

// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See License.txt in the project root for license information.

import (
	"encoding/xml"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strconv"
	"time"
)

const (
	// casing is per Golang's http.Header canonicalizing the header names.
	approximateMessagesCountHeader = "X-Ms-Approximate-Messages-Count"
)

// QueueAccessPolicy represents each access policy in the queue ACL.
type QueueAccessPolicy struct {
	ID         string
	StartTime  time.Time
	ExpiryTime time.Time
	CanRead    bool
	CanAdd     bool
	CanUpdate  bool
	CanProcess bool
}

// QueuePermissions represents the queue ACLs.
type QueuePermissions struct {
	AccessPolicies []QueueAccessPolicy
}

// SetQueuePermissionOptions includes options for a set queue permissions operation
type SetQueuePermissionOptions struct {
	Timeout   uint
	RequestID string `header:"x-ms-client-request-id"`
}

// Queue represents an Azure queue.
type Queue struct {
	qsc               *QueueServiceClient
	Name              string
	Metadata          map[string]string
	AproxMessageCount uint64
}

func (q *Queue) buildPath() string {
	return fmt.Sprintf("/%s", q.Name)
}

func (q *Queue) buildPathMessages() string {
	return fmt.Sprintf("%s/messages", q.buildPath())
}

// QueueServiceOptions includes options for some queue service operations
type QueueServiceOptions struct {
	Timeout   uint
	RequestID string `header:"x-ms-client-request-id"`
}

// Create operation creates a queue under the given account.
//
// See https://docs.microsoft.com/en-us/rest/api/storageservices/fileservices/Create-Queue4
func (q *Queue) Create(options *QueueServiceOptions) error {
	params := url.Values{}
	headers := q.qsc.client.getStandardHeaders()
	headers = q.qsc.client.addMetadataToHeaders(headers, q.Metadata)

	if options != nil {
		params = addTimeout(params, options.Timeout)
		headers = mergeHeaders(headers, headersFromStruct(*options))
	}
	uri := q.qsc.client.getEndpoint(queueServiceName, q.buildPath(), params)

	resp, err := q.qsc.client.exec(http.MethodPut, uri, headers, nil, q.qsc.auth)
	if err != nil {
		return err
	}
	defer drainRespBody(resp)
	return checkRespCode(resp, []int{http.StatusCreated})
}

// Delete operation permanently deletes the specified queue.
//
// See https://docs.microsoft.com/en-us/rest/api/storageservices/fileservices/Delete-Queue3
func (q *Queue) Delete(options *QueueServiceOptions) error {
	params := url.Values{}
	headers := q.qsc.client.getStandardHeaders()

	if options != nil {
		params = addTimeout(params, options.Timeout)
		headers = mergeHeaders(headers, headersFromStruct(*options))
	}
	uri := q.qsc.client.getEndpoint(queueServiceName, q.buildPath(), params)
	resp, err := q.qsc.client.exec(http.MethodDelete, uri, headers, nil, q.qsc.auth)
	if err != nil {
		return err
	}
	defer drainRespBody(resp)
	return checkRespCode(resp, []int{http.StatusNoContent})
}

// Exists returns true if a queue with given name exists.
func (q *Queue) Exists() (bool, error) {
	uri := q.qsc.client.getEndpoint(queueServiceName, q.buildPath(), url.Values{"comp": {"metadata"}})
	resp, err := q.qsc.client.exec(http.MethodGet, uri, q.qsc.client.getStandardHeaders(), nil, q.qsc.auth)
	if resp != nil {
		defer drainRespBody(resp)
		if resp.StatusCode == http.StatusOK || resp.StatusCode == http.StatusNotFound {
			return resp.StatusCode == http.StatusOK, nil
		}
		err = getErrorFromResponse(resp)
	}
	return false, err
}

// SetMetadata operation sets user-defined metadata on the specified queue.
// Metadata is associated with the queue as name-value pairs.
//
// See https://docs.microsoft.com/en-us/rest/api/storageservices/fileservices/Set-Queue-Metadata
func (q *Queue) SetMetadata(options *QueueServiceOptions) error {
	params := url.Values{"comp": {"metadata"}}
	headers := q.qsc.client.getStandardHeaders()
	headers = q.qsc.client.addMetadataToHeaders(headers, q.Metadata)

	if options != nil {
		params = addTimeout(params, options.Timeout)
		headers = mergeHeaders(headers, headersFromStruct(*options))
	}
	uri := q.qsc.client.getEndpoint(queueServiceName, q.buildPath(), params)

	resp, err := q.qsc.client.exec(http.MethodPut, uri, headers, nil, q.qsc.auth)
	if err != nil {
		return err
	}
	defer drainRespBody(resp)
	return checkRespCode(resp, []int{http.StatusNoContent})
}

// GetMetadata operation retrieves user-defined metadata and queue
// properties on the specified queue. Metadata is associated with
// the queue as name-values pairs.
//
// See https://docs.microsoft.com/en-us/rest/api/storageservices/fileservices/Set-Queue-Metadata
//
// Because the way Golang's http client (and http.Header in particular)
// canonicalize header names, the returned metadata names would always
// be all lower case.
func (q *Queue) GetMetadata(options *QueueServiceOptions) error {
	params := url.Values{"comp": {"metadata"}}
	headers := q.qsc.client.getStandardHeaders()

	if options != nil {
		params = addTimeout(params, options.Timeout)
		headers = mergeHeaders(headers, headersFromStruct(*options))
	}
	uri := q.qsc.client.getEndpoint(queueServiceName, q.buildPath(), params)

	resp, err := q.qsc.client.exec(http.MethodGet, uri, headers, nil, q.qsc.auth)
	if err != nil {
		return err
	}
	defer drainRespBody(resp)

	if err := checkRespCode(resp, []int{http.StatusOK}); err != nil {
		return err
	}

	aproxMessagesStr := resp.Header.Get(http.CanonicalHeaderKey(approximateMessagesCountHeader))
	if aproxMessagesStr != "" {
		aproxMessages, err := strconv.ParseUint(aproxMessagesStr, 10, 64)
		if err != nil {
			return err
		}
		q.AproxMessageCount = aproxMessages
	}

	q.Metadata = getMetadataFromHeaders(resp.Header)
	return nil
}

// GetMessageReference returns a message object with the specified text.
func (q *Queue) GetMessageReference(text string) *Message {
	return &Message{
		Queue: q,
		Text:  text,
	}
}

// GetMessagesOptions is the set of options can be specified for Get
// Messsages operation. A zero struct does not use any preferences for the
// request.
type GetMessagesOptions struct {
	Timeout           uint
	NumOfMessages     int
	VisibilityTimeout int
	RequestID         string `header:"x-ms-client-request-id"`
}

type messages struct {
	XMLName  xml.Name  `xml:"QueueMessagesList"`
	Messages []Message `xml:"QueueMessage"`
}

// GetMessages operation retrieves one or more messages from the front of the
// queue.
//
// See https://docs.microsoft.com/en-us/rest/api/storageservices/fileservices/Get-Messages
func (q *Queue) GetMessages(options *GetMessagesOptions) ([]Message, error) {
	query := url.Values{}
	headers := q.qsc.client.getStandardHeaders()

	if options != nil {
		if options.NumOfMessages != 0 {
			query.Set("numofmessages", strconv.Itoa(options.NumOfMessages))
		}
		if options.VisibilityTimeout != 0 {
			query.Set("visibilitytimeout", strconv.Itoa(options.VisibilityTimeout))
		}
		query = addTimeout(query, options.Timeout)
		headers = mergeHeaders(headers, headersFromStruct(*options))
	}
	uri := q.qsc.client.getEndpoint(queueServiceName, q.buildPathMessages(), query)

	resp, err := q.qsc.client.exec(http.MethodGet, uri, headers, nil, q.qsc.auth)
	if err != nil {
		return []Message{}, err
	}
	defer resp.Body.Close()

	var out messages
	err = xmlUnmarshal(resp.Body, &out)
	if err != nil {
		return []Message{}, err
	}
	for i := range out.Messages {
		out.Messages[i].Queue = q
	}
	return out.Messages, err
}

// PeekMessagesOptions is the set of options can be specified for Peek
// Messsage operation. A zero struct does not use any preferences for the
// request.
type PeekMessagesOptions struct {
	Timeout       uint
	NumOfMessages int
	RequestID     string `header:"x-ms-client-request-id"`
}

// PeekMessages retrieves one or more messages from the front of the queue, but
// does not alter the visibility of the message.
//
// See https://docs.microsoft.com/en-us/rest/api/storageservices/fileservices/Peek-Messages
func (q *Queue) PeekMessages(options *PeekMessagesOptions) ([]Message, error) {
	query := url.Values{"peekonly": {"true"}} // Required for peek operation
	headers := q.qsc.client.getStandardHeaders()

	if options != nil {
		if options.NumOfMessages != 0 {
			query.Set("numofmessages", strconv.Itoa(options.NumOfMessages))
		}
		query = addTimeout(query, options.Timeout)
		headers = mergeHeaders(headers, headersFromStruct(*options))
	}
	uri := q.qsc.client.getEndpoint(queueServiceName, q.buildPathMessages(), query)

	resp, err := q.qsc.client.exec(http.MethodGet, uri, headers, nil, q.qsc.auth)
	if err != nil {
		return []Message{}, err
	}
	defer resp.Body.Close()

	var out messages
	err = xmlUnmarshal(resp.Body, &out)
	if err != nil {
		return []Message{}, err
	}
	for i := range out.Messages {
		out.Messages[i].Queue = q
	}
	return out.Messages, err
}

// ClearMessages operation deletes all messages from the specified queue.
//
// See https://docs.microsoft.com/en-us/rest/api/storageservices/fileservices/Clear-Messages
func (q *Queue) ClearMessages(options *QueueServiceOptions) error {
	params := url.Values{}
	headers := q.qsc.client.getStandardHeaders()

	if options != nil {
		params = addTimeout(params, options.Timeout)
		headers = mergeHeaders(headers, headersFromStruct(*options))
	}
	uri := q.qsc.client.getEndpoint(queueServiceName, q.buildPathMessages(), params)

	resp, err := q.qsc.client.exec(http.MethodDelete, uri, headers, nil, q.qsc.auth)
	if err != nil {
		return err
	}
	defer drainRespBody(resp)
	return checkRespCode(resp, []int{http.StatusNoContent})
}

// SetPermissions sets up queue permissions
// See https://docs.microsoft.com/en-us/rest/api/storageservices/fileservices/set-queue-acl
func (q *Queue) SetPermissions(permissions QueuePermissions, options *SetQueuePermissionOptions) error {
	body, length, err := generateQueueACLpayload(permissions.AccessPolicies)
	if err != nil {
		return err
	}

	params := url.Values{
		"comp": {"acl"},
	}
	headers := q.qsc.client.getStandardHeaders()
	headers["Content-Length"] = strconv.Itoa(length)

	if options != nil {
		params = addTimeout(params, options.Timeout)
		headers = mergeHeaders(headers, headersFromStruct(*options))
	}
	uri := q.qsc.client.getEndpoint(queueServiceName, q.buildPath(), params)
	resp, err := q.qsc.client.exec(http.MethodPut, uri, headers, body, q.qsc.auth)
	if err != nil {
		return err
	}
	defer drainRespBody(resp)
	return checkRespCode(resp, []int{http.StatusNoContent})
}

func generateQueueACLpayload(policies []QueueAccessPolicy) (io.Reader, int, error) {
	sil := SignedIdentifiers{
		SignedIdentifiers: []SignedIdentifier{},
	}
	for _, qapd := range policies {
		permission := qapd.generateQueuePermissions()
		signedIdentifier := convertAccessPolicyToXMLStructs(qapd.ID, qapd.StartTime, qapd.ExpiryTime, permission)
		sil.SignedIdentifiers = append(sil.SignedIdentifiers, signedIdentifier)
	}
	return xmlMarshal(sil)
}

func (qapd *QueueAccessPolicy) generateQueuePermissions() (permissions string) {
	// generate the permissions string (raup).
	// still want the end user API to have bool flags.
	permissions = ""

	if qapd.CanRead {
		permissions += "r"
	}

	if qapd.CanAdd {
		permissions += "a"
	}

	if qapd.CanUpdate {
		permissions += "u"
	}

	if qapd.CanProcess {
		permissions += "p"
	}

	return permissions
}

// GetQueuePermissionOptions includes options for a get queue permissions operation
type GetQueuePermissionOptions struct {
	Timeout   uint
	RequestID string `header:"x-ms-client-request-id"`
}

// GetPermissions gets the queue permissions as per https://docs.microsoft.com/en-us/rest/api/storageservices/fileservices/get-queue-acl
// If timeout is 0 then it will not be passed to Azure
func (q *Queue) GetPermissions(options *GetQueuePermissionOptions) (*QueuePermissions, error) {
	params := url.Values{
		"comp": {"acl"},
	}
	headers := q.qsc.client.getStandardHeaders()

	if options != nil {
		params = addTimeout(params, options.Timeout)
		headers = mergeHeaders(headers, headersFromStruct(*options))
	}
	uri := q.qsc.client.getEndpoint(queueServiceName, q.buildPath(), params)
	resp, err := q.qsc.client.exec(http.MethodGet, uri, headers, nil, q.qsc.auth)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	var ap AccessPolicy
	err = xmlUnmarshal(resp.Body, &ap.SignedIdentifiersList)
	if err != nil {
		return nil, err
	}
	return buildQueueAccessPolicy(ap, &resp.Header), nil
}

func buildQueueAccessPolicy(ap AccessPolicy, headers *http.Header) *QueuePermissions {
	permissions := QueuePermissions{
		AccessPolicies: []QueueAccessPolicy{},
	}

	for _, policy := range ap.SignedIdentifiersList.SignedIdentifiers {
		qapd := QueueAccessPolicy{
			ID:         policy.ID,
			StartTime:  policy.AccessPolicy.StartTime,
			ExpiryTime: policy.AccessPolicy.ExpiryTime,
		}
		qapd.CanRead = updatePermissions(policy.AccessPolicy.Permission, "r")
		qapd.CanAdd = updatePermissions(policy.AccessPolicy.Permission, "a")
		qapd.CanUpdate = updatePermissions(policy.AccessPolicy.Permission, "u")
		qapd.CanProcess = updatePermissions(policy.AccessPolicy.Permission, "p")

		permissions.AccessPolicies = append(permissions.AccessPolicies, qapd)
	}
	return &permissions
}
