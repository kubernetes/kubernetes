// Copyright 2015 Google Inc. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package gcl

import (
	"io"
	"io/ioutil"
	"net/http"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	sink_api "k8s.io/heapster/sinks/api"
	"k8s.io/heapster/util/gce"
	kube_api "k8s.io/kubernetes/pkg/api"
	kube_api_unv "k8s.io/kubernetes/pkg/api/unversioned"
)

type fakeHttpClient struct {
	capturedSendRequests []*http.Request
}

func NewFakeHttpClient() *fakeHttpClient {
	return &fakeHttpClient{[]*http.Request{}}
}

type fakeResponseBody struct {
	io.Reader
}

func (body fakeResponseBody) Close() error {
	// No-op
	return nil
}

func NewFakeResponseBody(responseBody string) io.ReadCloser {
	return fakeResponseBody{strings.NewReader(responseBody)}
}

func (client *fakeHttpClient) Do(req *http.Request) (resp *http.Response, err error) {
	client.capturedSendRequests = append(client.capturedSendRequests, req)
	resp = &http.Response{
		Status:     "200 OK",
		StatusCode: 200,
		Proto:      "HTTP/1.0",
		ProtoMajor: 1,
		ProtoMinor: 0,
		Body:       NewFakeResponseBody(""),
	}
	return resp, err
}

type fakeGCLSink struct {
	sink_api.ExternalSink
	fakeHttpClient *fakeHttpClient
}

// Returns a fake GCL sink.
func NewFakeSink() fakeGCLSink {
	fakeHttpClient := NewFakeHttpClient()
	return fakeGCLSink{
		&gclSink{
			projectId:  "fakeProjectId",
			httpClient: fakeHttpClient,
			token:      gce.NewFakeAuthTokenProvider("fakeToken"),
		},
		fakeHttpClient,
	}
}

func TestStoreEventsNilInput(t *testing.T) {
	// Arrange
	fakeSink := NewFakeSink()

	// Act
	err := fakeSink.StoreEvents(nil /*events*/)

	// Assert
	assert.NoError(t, err)
	assert.Equal(t, 0 /* expected */, len(fakeSink.fakeHttpClient.capturedSendRequests) /* actual */)
}

func TestStoreEventsEmptyInput(t *testing.T) {
	// Arrange
	fakeSink := NewFakeSink()

	// Act
	err := fakeSink.StoreEvents([]kube_api.Event{})

	// Assert
	assert.NoError(t, err)
	assert.Equal(t, 0 /* expected */, len(fakeSink.fakeHttpClient.capturedSendRequests) /* actual */)
}

func TestStoreEventsSingleEventInput(t *testing.T) {
	// Arrange
	fakeSink := NewFakeSink()
	eventTime := kube_api_unv.Unix(12345, 0)
	events := []kube_api.Event{
		{
			Reason:         "event1",
			FirstTimestamp: eventTime,
			LastTimestamp:  eventTime,
			Source: kube_api.EventSource{
				Host: "event1HostName",
			},
		},
	}

	// Act
	err := fakeSink.StoreEvents(events)

	// Assert
	assert.NoError(t, err)
	assert.Equal(t, 1 /* expected */, len(fakeSink.fakeHttpClient.capturedSendRequests) /* actual */)
	assert.Equal(t, "POST" /* expected */, fakeSink.fakeHttpClient.capturedSendRequests[0].Method /* actual */)
	assert.Equal(t, "https" /* expected */, fakeSink.fakeHttpClient.capturedSendRequests[0].URL.Scheme /* actual */)
	assert.Equal(t, "logging.googleapis.com" /* expected */, fakeSink.fakeHttpClient.capturedSendRequests[0].URL.Host /* actual */)
	assert.Equal(t, "/v1beta3/projects/fakeProjectId/logs/kubernetes.io%2Fevents/entries:write" /* expected */, fakeSink.fakeHttpClient.capturedSendRequests[0].URL.Opaque /* actual */)
	assert.Equal(t, 2 /* expected */, len(fakeSink.fakeHttpClient.capturedSendRequests[0].Header) /* actual */)
	assert.Equal(t, 1 /* expected */, len(fakeSink.fakeHttpClient.capturedSendRequests[0].Header["Content-Type"]) /* actual */)
	assert.Equal(t, "application/json" /* expected */, fakeSink.fakeHttpClient.capturedSendRequests[0].Header["Content-Type"][0] /* actual */)
	assert.Equal(t, 1 /* expected */, len(fakeSink.fakeHttpClient.capturedSendRequests[0].Header["Authorization"]) /* actual */)
	assert.Equal(t, "Bearer fakeToken" /* expected */, fakeSink.fakeHttpClient.capturedSendRequests[0].Header["Authorization"][0] /* actual */)
	reqBodyBytes, err := ioutil.ReadAll(fakeSink.fakeHttpClient.capturedSendRequests[0].Body)
	assert.NoError(t, err)
	expectedJson := "{\"entries\":[{\"metadata\":{\"timestamp\":\"1970-01-01T03:25:45Z\",\"severity\":\"NOTICE\",\"projectId\":\"fakeProjectId\",\"serviceName\":\"custom.googleapis.com\"},\"structPayload\":{\"metadata\":{\"creationTimestamp\":null},\"involvedObject\":{},\"reason\":\"event1\",\"source\":{\"host\":\"event1HostName\"},\"firstTimestamp\":\"1970-01-01T03:25:45Z\",\"lastTimestamp\":\"1970-01-01T03:25:45Z\"}}]}"
	assert.Equal(t, expectedJson /* expected */, string(reqBodyBytes) /* actual */)
}

func TestStoreEventsMultipleEventInput(t *testing.T) {
	// Arrange
	fakeSink := NewFakeSink()
	event1Time := kube_api_unv.Unix(12345, 0)
	event2Time := kube_api_unv.Unix(12366, 0)
	events := []kube_api.Event{
		{
			Reason:         "event1",
			FirstTimestamp: event1Time,
			LastTimestamp:  event1Time,
			Source: kube_api.EventSource{
				Host: "event1HostName",
			},
		},
		{
			Reason:         "event2",
			FirstTimestamp: event2Time,
			LastTimestamp:  event2Time,
			Source: kube_api.EventSource{
				Host: "event2HostName",
			},
		},
	}

	// Act
	err := fakeSink.StoreEvents(events)

	// Assert
	assert.NoError(t, err)
	assert.Equal(t, 1 /* expected */, len(fakeSink.fakeHttpClient.capturedSendRequests) /* actual */)
	assert.Equal(t, "POST" /* expected */, fakeSink.fakeHttpClient.capturedSendRequests[0].Method /* actual */)
	assert.Equal(t, "https" /* expected */, fakeSink.fakeHttpClient.capturedSendRequests[0].URL.Scheme /* actual */)
	assert.Equal(t, "logging.googleapis.com" /* expected */, fakeSink.fakeHttpClient.capturedSendRequests[0].URL.Host /* actual */)
	assert.Equal(t, "/v1beta3/projects/fakeProjectId/logs/kubernetes.io%2Fevents/entries:write" /* expected */, fakeSink.fakeHttpClient.capturedSendRequests[0].URL.Opaque /* actual */)
	assert.Equal(t, 2 /* expected */, len(fakeSink.fakeHttpClient.capturedSendRequests[0].Header) /* actual */)
	assert.Equal(t, 1 /* expected */, len(fakeSink.fakeHttpClient.capturedSendRequests[0].Header["Content-Type"]) /* actual */)
	assert.Equal(t, "application/json" /* expected */, fakeSink.fakeHttpClient.capturedSendRequests[0].Header["Content-Type"][0] /* actual */)
	assert.Equal(t, 1 /* expected */, len(fakeSink.fakeHttpClient.capturedSendRequests[0].Header["Authorization"]) /* actual */)
	assert.Equal(t, "Bearer fakeToken" /* expected */, fakeSink.fakeHttpClient.capturedSendRequests[0].Header["Authorization"][0] /* actual */)
	reqBodyBytes, err := ioutil.ReadAll(fakeSink.fakeHttpClient.capturedSendRequests[0].Body)
	assert.NoError(t, err)
	expectedJson := "{\"entries\":[{\"metadata\":{\"timestamp\":\"1970-01-01T03:25:45Z\",\"severity\":\"NOTICE\",\"projectId\":\"fakeProjectId\",\"serviceName\":\"custom.googleapis.com\"},\"structPayload\":{\"metadata\":{\"creationTimestamp\":null},\"involvedObject\":{},\"reason\":\"event1\",\"source\":{\"host\":\"event1HostName\"},\"firstTimestamp\":\"1970-01-01T03:25:45Z\",\"lastTimestamp\":\"1970-01-01T03:25:45Z\"}},{\"metadata\":{\"timestamp\":\"1970-01-01T03:26:06Z\",\"severity\":\"NOTICE\",\"projectId\":\"fakeProjectId\",\"serviceName\":\"custom.googleapis.com\"},\"structPayload\":{\"metadata\":{\"creationTimestamp\":null},\"involvedObject\":{},\"reason\":\"event2\",\"source\":{\"host\":\"event2HostName\"},\"firstTimestamp\":\"1970-01-01T03:26:06Z\",\"lastTimestamp\":\"1970-01-01T03:26:06Z\"}}]}"
	assert.Equal(t, expectedJson /* expected */, string(reqBodyBytes) /* actual */)
}
