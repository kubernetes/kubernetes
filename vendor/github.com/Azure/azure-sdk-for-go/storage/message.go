package storage

// Copyright 2017 Microsoft Corporation
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.

import (
	"encoding/xml"
	"fmt"
	"net/http"
	"net/url"
	"strconv"
	"time"
)

// Message represents an Azure message.
type Message struct {
	Queue        *Queue
	Text         string      `xml:"MessageText"`
	ID           string      `xml:"MessageId"`
	Insertion    TimeRFC1123 `xml:"InsertionTime"`
	Expiration   TimeRFC1123 `xml:"ExpirationTime"`
	PopReceipt   string      `xml:"PopReceipt"`
	NextVisible  TimeRFC1123 `xml:"TimeNextVisible"`
	DequeueCount int         `xml:"DequeueCount"`
}

func (m *Message) buildPath() string {
	return fmt.Sprintf("%s/%s", m.Queue.buildPathMessages(), m.ID)
}

// PutMessageOptions is the set of options can be specified for Put Messsage
// operation. A zero struct does not use any preferences for the request.
type PutMessageOptions struct {
	Timeout           uint
	VisibilityTimeout int
	MessageTTL        int
	RequestID         string `header:"x-ms-client-request-id"`
}

// Put operation adds a new message to the back of the message queue.
//
// See https://docs.microsoft.com/en-us/rest/api/storageservices/fileservices/Put-Message
func (m *Message) Put(options *PutMessageOptions) error {
	query := url.Values{}
	headers := m.Queue.qsc.client.getStandardHeaders()

	req := putMessageRequest{MessageText: m.Text}
	body, nn, err := xmlMarshal(req)
	if err != nil {
		return err
	}
	headers["Content-Length"] = strconv.Itoa(nn)

	if options != nil {
		if options.VisibilityTimeout != 0 {
			query.Set("visibilitytimeout", strconv.Itoa(options.VisibilityTimeout))
		}
		if options.MessageTTL != 0 {
			query.Set("messagettl", strconv.Itoa(options.MessageTTL))
		}
		query = addTimeout(query, options.Timeout)
		headers = mergeHeaders(headers, headersFromStruct(*options))
	}

	uri := m.Queue.qsc.client.getEndpoint(queueServiceName, m.Queue.buildPathMessages(), query)
	resp, err := m.Queue.qsc.client.exec(http.MethodPost, uri, headers, body, m.Queue.qsc.auth)
	if err != nil {
		return err
	}
	defer readAndCloseBody(resp.body)

	err = xmlUnmarshal(resp.body, m)
	if err != nil {
		return err
	}
	return checkRespCode(resp.statusCode, []int{http.StatusCreated})
}

// UpdateMessageOptions is the set of options can be specified for Update Messsage
// operation. A zero struct does not use any preferences for the request.
type UpdateMessageOptions struct {
	Timeout           uint
	VisibilityTimeout int
	RequestID         string `header:"x-ms-client-request-id"`
}

// Update operation updates the specified message.
//
// See https://docs.microsoft.com/en-us/rest/api/storageservices/fileservices/Update-Message
func (m *Message) Update(options *UpdateMessageOptions) error {
	query := url.Values{}
	if m.PopReceipt != "" {
		query.Set("popreceipt", m.PopReceipt)
	}

	headers := m.Queue.qsc.client.getStandardHeaders()
	req := putMessageRequest{MessageText: m.Text}
	body, nn, err := xmlMarshal(req)
	if err != nil {
		return err
	}
	headers["Content-Length"] = strconv.Itoa(nn)

	if options != nil {
		if options.VisibilityTimeout != 0 {
			query.Set("visibilitytimeout", strconv.Itoa(options.VisibilityTimeout))
		}
		query = addTimeout(query, options.Timeout)
		headers = mergeHeaders(headers, headersFromStruct(*options))
	}
	uri := m.Queue.qsc.client.getEndpoint(queueServiceName, m.buildPath(), query)

	resp, err := m.Queue.qsc.client.exec(http.MethodPut, uri, headers, body, m.Queue.qsc.auth)
	if err != nil {
		return err
	}
	defer readAndCloseBody(resp.body)

	m.PopReceipt = resp.headers.Get("x-ms-popreceipt")
	nextTimeStr := resp.headers.Get("x-ms-time-next-visible")
	if nextTimeStr != "" {
		nextTime, err := time.Parse(time.RFC1123, nextTimeStr)
		if err != nil {
			return err
		}
		m.NextVisible = TimeRFC1123(nextTime)
	}

	return checkRespCode(resp.statusCode, []int{http.StatusNoContent})
}

// Delete operation deletes the specified message.
//
// See https://msdn.microsoft.com/en-us/library/azure/dd179347.aspx
func (m *Message) Delete(options *QueueServiceOptions) error {
	params := url.Values{"popreceipt": {m.PopReceipt}}
	headers := m.Queue.qsc.client.getStandardHeaders()

	if options != nil {
		params = addTimeout(params, options.Timeout)
		headers = mergeHeaders(headers, headersFromStruct(*options))
	}
	uri := m.Queue.qsc.client.getEndpoint(queueServiceName, m.buildPath(), params)

	resp, err := m.Queue.qsc.client.exec(http.MethodDelete, uri, headers, nil, m.Queue.qsc.auth)
	if err != nil {
		return err
	}
	readAndCloseBody(resp.body)
	return checkRespCode(resp.statusCode, []int{http.StatusNoContent})
}

type putMessageRequest struct {
	XMLName     xml.Name `xml:"QueueMessage"`
	MessageText string   `xml:"MessageText"`
}
