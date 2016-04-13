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

package api

import (
	"io"
	"net/http"
	"reflect"
	"testing"

	"github.com/google/cadvisor/events"
	info "github.com/google/cadvisor/info/v1"

	"github.com/stretchr/testify/assert"
)

// returns an http.Request pointer for an input url test string
func makeHTTPRequest(requestURL string, t *testing.T) *http.Request {
	dummyReader, _ := io.Pipe()
	r, err := http.NewRequest("GET", requestURL, dummyReader)
	assert.Nil(t, err)
	return r
}

func TestGetEventRequestBasicRequest(t *testing.T) {
	r := makeHTTPRequest("http://localhost:8080/api/v1.3/events?oom_events=true&stream=false&max_events=20", t)
	expectedQuery := events.NewRequest()
	expectedQuery.EventType = map[info.EventType]bool{
		info.EventOom: true,
	}
	expectedQuery.MaxEventsReturned = 20

	receivedQuery, stream, err := getEventRequest(r)

	if !reflect.DeepEqual(expectedQuery, receivedQuery) {
		t.Errorf("expected %#v but received %#v", expectedQuery, receivedQuery)
	}
	assert.False(t, stream)
	assert.Nil(t, err)
}

func TestGetEventEmptyRequest(t *testing.T) {
	r := makeHTTPRequest("", t)
	expectedQuery := events.NewRequest()

	receivedQuery, stream, err := getEventRequest(r)

	if !reflect.DeepEqual(expectedQuery, receivedQuery) {
		t.Errorf("expected %#v but received %#v", expectedQuery, receivedQuery)
	}
	assert.False(t, stream)
	assert.Nil(t, err)
}

func TestGetEventRequestDoubleArgument(t *testing.T) {
	r := makeHTTPRequest("http://localhost:8080/api/v1.3/events?stream=true&oom_events=true&oom_events=false", t)
	expectedQuery := events.NewRequest()
	expectedQuery.EventType = map[info.EventType]bool{
		info.EventOom: true,
	}

	receivedQuery, stream, err := getEventRequest(r)

	if !reflect.DeepEqual(expectedQuery, receivedQuery) {
		t.Errorf("expected %#v but received %#v", expectedQuery, receivedQuery)
	}
	assert.True(t, stream)
	assert.Nil(t, err)
}
