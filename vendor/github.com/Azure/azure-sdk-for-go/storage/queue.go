package storage

import (
	"encoding/xml"
	"fmt"
	"net/http"
	"net/url"
	"strconv"
	"strings"
)

const (
	// casing is per Golang's http.Header canonicalizing the header names.
	approximateMessagesCountHeader  = "X-Ms-Approximate-Messages-Count"
	userDefinedMetadataHeaderPrefix = "X-Ms-Meta-"
)

// QueueServiceClient contains operations for Microsoft Azure Queue Storage
// Service.
type QueueServiceClient struct {
	client Client
}

func pathForQueue(queue string) string         { return fmt.Sprintf("/%s", queue) }
func pathForQueueMessages(queue string) string { return fmt.Sprintf("/%s/messages", queue) }
func pathForMessage(queue, name string) string { return fmt.Sprintf("/%s/messages/%s", queue, name) }

type putMessageRequest struct {
	XMLName     xml.Name `xml:"QueueMessage"`
	MessageText string   `xml:"MessageText"`
}

// PutMessageParameters is the set of options can be specified for Put Messsage
// operation. A zero struct does not use any preferences for the request.
type PutMessageParameters struct {
	VisibilityTimeout int
	MessageTTL        int
}

func (p PutMessageParameters) getParameters() url.Values {
	out := url.Values{}
	if p.VisibilityTimeout != 0 {
		out.Set("visibilitytimeout", strconv.Itoa(p.VisibilityTimeout))
	}
	if p.MessageTTL != 0 {
		out.Set("messagettl", strconv.Itoa(p.MessageTTL))
	}
	return out
}

// GetMessagesParameters is the set of options can be specified for Get
// Messsages operation. A zero struct does not use any preferences for the
// request.
type GetMessagesParameters struct {
	NumOfMessages     int
	VisibilityTimeout int
}

func (p GetMessagesParameters) getParameters() url.Values {
	out := url.Values{}
	if p.NumOfMessages != 0 {
		out.Set("numofmessages", strconv.Itoa(p.NumOfMessages))
	}
	if p.VisibilityTimeout != 0 {
		out.Set("visibilitytimeout", strconv.Itoa(p.VisibilityTimeout))
	}
	return out
}

// PeekMessagesParameters is the set of options can be specified for Peek
// Messsage operation. A zero struct does not use any preferences for the
// request.
type PeekMessagesParameters struct {
	NumOfMessages int
}

func (p PeekMessagesParameters) getParameters() url.Values {
	out := url.Values{"peekonly": {"true"}} // Required for peek operation
	if p.NumOfMessages != 0 {
		out.Set("numofmessages", strconv.Itoa(p.NumOfMessages))
	}
	return out
}

// GetMessagesResponse represents a response returned from Get Messages
// operation.
type GetMessagesResponse struct {
	XMLName           xml.Name             `xml:"QueueMessagesList"`
	QueueMessagesList []GetMessageResponse `xml:"QueueMessage"`
}

// GetMessageResponse represents a QueueMessage object returned from Get
// Messages operation response.
type GetMessageResponse struct {
	MessageID       string `xml:"MessageId"`
	InsertionTime   string `xml:"InsertionTime"`
	ExpirationTime  string `xml:"ExpirationTime"`
	PopReceipt      string `xml:"PopReceipt"`
	TimeNextVisible string `xml:"TimeNextVisible"`
	DequeueCount    int    `xml:"DequeueCount"`
	MessageText     string `xml:"MessageText"`
}

// PeekMessagesResponse represents a response returned from Get Messages
// operation.
type PeekMessagesResponse struct {
	XMLName           xml.Name              `xml:"QueueMessagesList"`
	QueueMessagesList []PeekMessageResponse `xml:"QueueMessage"`
}

// PeekMessageResponse represents a QueueMessage object returned from Peek
// Messages operation response.
type PeekMessageResponse struct {
	MessageID      string `xml:"MessageId"`
	InsertionTime  string `xml:"InsertionTime"`
	ExpirationTime string `xml:"ExpirationTime"`
	DequeueCount   int    `xml:"DequeueCount"`
	MessageText    string `xml:"MessageText"`
}

// QueueMetadataResponse represents user defined metadata and queue
// properties on a specific queue.
//
// See https://msdn.microsoft.com/en-us/library/azure/dd179384.aspx
type QueueMetadataResponse struct {
	ApproximateMessageCount int
	UserDefinedMetadata     map[string]string
}

// SetMetadata operation sets user-defined metadata on the specified queue.
// Metadata is associated with the queue as name-value pairs.
//
// See https://msdn.microsoft.com/en-us/library/azure/dd179348.aspx
func (c QueueServiceClient) SetMetadata(name string, metadata map[string]string) error {
	uri := c.client.getEndpoint(queueServiceName, pathForQueue(name), url.Values{"comp": []string{"metadata"}})
	headers := c.client.getStandardHeaders()
	for k, v := range metadata {
		headers[userDefinedMetadataHeaderPrefix+k] = v
	}

	resp, err := c.client.exec("PUT", uri, headers, nil)
	if err != nil {
		return err
	}
	defer resp.body.Close()

	return checkRespCode(resp.statusCode, []int{http.StatusNoContent})
}

// GetMetadata operation retrieves user-defined metadata and queue
// properties on the specified queue. Metadata is associated with
// the queue as name-values pairs.
//
// See https://msdn.microsoft.com/en-us/library/azure/dd179384.aspx
//
// Because the way Golang's http client (and http.Header in particular)
// canonicalize header names, the returned metadata names would always
// be all lower case.
func (c QueueServiceClient) GetMetadata(name string) (QueueMetadataResponse, error) {
	qm := QueueMetadataResponse{}
	qm.UserDefinedMetadata = make(map[string]string)
	uri := c.client.getEndpoint(queueServiceName, pathForQueue(name), url.Values{"comp": []string{"metadata"}})
	headers := c.client.getStandardHeaders()
	resp, err := c.client.exec("GET", uri, headers, nil)
	if err != nil {
		return qm, err
	}
	defer resp.body.Close()

	for k, v := range resp.headers {
		if len(v) != 1 {
			return qm, fmt.Errorf("Unexpected number of values (%d) in response header '%s'", len(v), k)
		}

		value := v[0]

		if k == approximateMessagesCountHeader {
			qm.ApproximateMessageCount, err = strconv.Atoi(value)
			if err != nil {
				return qm, fmt.Errorf("Unexpected value in response header '%s': '%s' ", k, value)
			}
		} else if strings.HasPrefix(k, userDefinedMetadataHeaderPrefix) {
			name := strings.TrimPrefix(k, userDefinedMetadataHeaderPrefix)
			qm.UserDefinedMetadata[strings.ToLower(name)] = value
		}
	}

	return qm, checkRespCode(resp.statusCode, []int{http.StatusOK})
}

// CreateQueue operation creates a queue under the given account.
//
// See https://msdn.microsoft.com/en-us/library/azure/dd179342.aspx
func (c QueueServiceClient) CreateQueue(name string) error {
	uri := c.client.getEndpoint(queueServiceName, pathForQueue(name), url.Values{})
	headers := c.client.getStandardHeaders()
	resp, err := c.client.exec("PUT", uri, headers, nil)
	if err != nil {
		return err
	}
	defer resp.body.Close()
	return checkRespCode(resp.statusCode, []int{http.StatusCreated})
}

// DeleteQueue operation permanently deletes the specified queue.
//
// See https://msdn.microsoft.com/en-us/library/azure/dd179436.aspx
func (c QueueServiceClient) DeleteQueue(name string) error {
	uri := c.client.getEndpoint(queueServiceName, pathForQueue(name), url.Values{})
	resp, err := c.client.exec("DELETE", uri, c.client.getStandardHeaders(), nil)
	if err != nil {
		return err
	}
	defer resp.body.Close()
	return checkRespCode(resp.statusCode, []int{http.StatusNoContent})
}

// QueueExists returns true if a queue with given name exists.
func (c QueueServiceClient) QueueExists(name string) (bool, error) {
	uri := c.client.getEndpoint(queueServiceName, pathForQueue(name), url.Values{"comp": {"metadata"}})
	resp, err := c.client.exec("GET", uri, c.client.getStandardHeaders(), nil)
	if resp != nil && (resp.statusCode == http.StatusOK || resp.statusCode == http.StatusNotFound) {
		return resp.statusCode == http.StatusOK, nil
	}

	return false, err
}

// PutMessage operation adds a new message to the back of the message queue.
//
// See https://msdn.microsoft.com/en-us/library/azure/dd179346.aspx
func (c QueueServiceClient) PutMessage(queue string, message string, params PutMessageParameters) error {
	uri := c.client.getEndpoint(queueServiceName, pathForQueueMessages(queue), params.getParameters())
	req := putMessageRequest{MessageText: message}
	body, nn, err := xmlMarshal(req)
	if err != nil {
		return err
	}
	headers := c.client.getStandardHeaders()
	headers["Content-Length"] = strconv.Itoa(nn)
	resp, err := c.client.exec("POST", uri, headers, body)
	if err != nil {
		return err
	}
	defer resp.body.Close()
	return checkRespCode(resp.statusCode, []int{http.StatusCreated})
}

// ClearMessages operation deletes all messages from the specified queue.
//
// See https://msdn.microsoft.com/en-us/library/azure/dd179454.aspx
func (c QueueServiceClient) ClearMessages(queue string) error {
	uri := c.client.getEndpoint(queueServiceName, pathForQueueMessages(queue), url.Values{})
	resp, err := c.client.exec("DELETE", uri, c.client.getStandardHeaders(), nil)
	if err != nil {
		return err
	}
	defer resp.body.Close()
	return checkRespCode(resp.statusCode, []int{http.StatusNoContent})
}

// GetMessages operation retrieves one or more messages from the front of the
// queue.
//
// See https://msdn.microsoft.com/en-us/library/azure/dd179474.aspx
func (c QueueServiceClient) GetMessages(queue string, params GetMessagesParameters) (GetMessagesResponse, error) {
	var r GetMessagesResponse
	uri := c.client.getEndpoint(queueServiceName, pathForQueueMessages(queue), params.getParameters())
	resp, err := c.client.exec("GET", uri, c.client.getStandardHeaders(), nil)
	if err != nil {
		return r, err
	}
	defer resp.body.Close()
	err = xmlUnmarshal(resp.body, &r)
	return r, err
}

// PeekMessages retrieves one or more messages from the front of the queue, but
// does not alter the visibility of the message.
//
// See https://msdn.microsoft.com/en-us/library/azure/dd179472.aspx
func (c QueueServiceClient) PeekMessages(queue string, params PeekMessagesParameters) (PeekMessagesResponse, error) {
	var r PeekMessagesResponse
	uri := c.client.getEndpoint(queueServiceName, pathForQueueMessages(queue), params.getParameters())
	resp, err := c.client.exec("GET", uri, c.client.getStandardHeaders(), nil)
	if err != nil {
		return r, err
	}
	defer resp.body.Close()
	err = xmlUnmarshal(resp.body, &r)
	return r, err
}

// DeleteMessage operation deletes the specified message.
//
// See https://msdn.microsoft.com/en-us/library/azure/dd179347.aspx
func (c QueueServiceClient) DeleteMessage(queue, messageID, popReceipt string) error {
	uri := c.client.getEndpoint(queueServiceName, pathForMessage(queue, messageID), url.Values{
		"popreceipt": {popReceipt}})
	resp, err := c.client.exec("DELETE", uri, c.client.getStandardHeaders(), nil)
	if err != nil {
		return err
	}
	defer resp.body.Close()
	return checkRespCode(resp.statusCode, []int{http.StatusNoContent})
}
