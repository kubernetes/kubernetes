// Copyright 2014 Google Inc. All Rights Reserved.
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

// Package api provides a handler for /api/
package api

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"path"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/google/cadvisor/events"
	httpmux "github.com/google/cadvisor/http/mux"
	info "github.com/google/cadvisor/info/v1"
	"github.com/google/cadvisor/manager"

	"github.com/golang/glog"
)

const (
	apiResource = "/api/"
)

func RegisterHandlers(mux httpmux.Mux, m manager.Manager) error {
	apiVersions := getApiVersions()
	supportedApiVersions := make(map[string]ApiVersion, len(apiVersions))
	for _, v := range apiVersions {
		supportedApiVersions[v.Version()] = v
	}

	mux.HandleFunc(apiResource, func(w http.ResponseWriter, r *http.Request) {
		err := handleRequest(supportedApiVersions, m, w, r)
		if err != nil {
			http.Error(w, err.Error(), 500)
		}
	})
	return nil
}

// Captures the API version, requestType [optional], and remaining request [optional].
var apiRegexp = regexp.MustCompile(`/api/([^/]+)/?([^/]+)?(.*)`)

const (
	apiVersion = iota + 1
	apiRequestType
	apiRequestArgs
)

func handleRequest(supportedApiVersions map[string]ApiVersion, m manager.Manager, w http.ResponseWriter, r *http.Request) error {
	start := time.Now()
	defer func() {
		glog.V(4).Infof("Request took %s", time.Since(start))
	}()

	request := r.URL.Path

	const apiPrefix = "/api"
	if !strings.HasPrefix(request, apiPrefix) {
		return fmt.Errorf("incomplete API request %q", request)
	}

	// If the request doesn't have an API version, list those.
	if request == apiPrefix || request == apiResource {
		versions := make([]string, 0, len(supportedApiVersions))
		for v := range supportedApiVersions {
			versions = append(versions, v)
		}
		sort.Strings(versions)
		http.Error(w, fmt.Sprintf("Supported API versions: %s", strings.Join(versions, ",")), http.StatusBadRequest)
		return nil
	}

	// Verify that we have all the elements we expect:
	// /<version>/<request type>[/<args...>]
	requestElements := apiRegexp.FindStringSubmatch(request)
	if len(requestElements) == 0 {
		return fmt.Errorf("malformed request %q", request)
	}
	version := requestElements[apiVersion]
	requestType := requestElements[apiRequestType]
	requestArgs := strings.Split(requestElements[apiRequestArgs], "/")

	// Check supported versions.
	versionHandler, ok := supportedApiVersions[version]
	if !ok {
		return fmt.Errorf("unsupported API version %q", version)
	}

	// If no request type, list possible request types.
	if requestType == "" {
		requestTypes := versionHandler.SupportedRequestTypes()
		sort.Strings(requestTypes)
		http.Error(w, fmt.Sprintf("Supported request types: %q", strings.Join(requestTypes, ",")), http.StatusBadRequest)
		return nil
	}

	// Trim the first empty element from the request.
	if len(requestArgs) > 0 && requestArgs[0] == "" {
		requestArgs = requestArgs[1:]
	}

	return versionHandler.HandleRequest(requestType, requestArgs, m, w, r)

}

func writeResult(res interface{}, w http.ResponseWriter) error {
	out, err := json.Marshal(res)
	if err != nil {
		return fmt.Errorf("failed to marshall response %+v with error: %s", res, err)
	}

	w.Header().Set("Content-Type", "application/json")
	w.Write(out)
	return nil

}

func streamResults(eventChannel *events.EventChannel, w http.ResponseWriter, r *http.Request, m manager.Manager) error {
	cn, ok := w.(http.CloseNotifier)
	if !ok {
		return errors.New("could not access http.CloseNotifier")
	}
	flusher, ok := w.(http.Flusher)
	if !ok {
		return errors.New("could not access http.Flusher")
	}

	w.Header().Set("Transfer-Encoding", "chunked")
	w.WriteHeader(http.StatusOK)
	flusher.Flush()

	enc := json.NewEncoder(w)
	for {
		select {
		case <-cn.CloseNotify():
			m.CloseEventChannel(eventChannel.GetWatchId())
			return nil
		case ev := <-eventChannel.GetChannel():
			err := enc.Encode(ev)
			if err != nil {
				glog.Errorf("error encoding message %+v for result stream: %v", ev, err)
			}
			flusher.Flush()
		}
	}
}

func getContainerInfoRequest(body io.ReadCloser) (*info.ContainerInfoRequest, error) {
	query := info.DefaultContainerInfoRequest()
	decoder := json.NewDecoder(body)
	err := decoder.Decode(&query)
	if err != nil && err != io.EOF {
		return nil, fmt.Errorf("unable to decode the json value: %s", err)
	}

	return &query, nil
}

// The user can set any or none of the following arguments in any order
// with any twice defined arguments being assigned the first value.
// If the value type for the argument is wrong the field will be assumed to be
// unassigned
// bools: stream, subcontainers, oom_events, creation_events, deletion_events
// ints: max_events, start_time (unix timestamp), end_time (unix timestamp)
// example r.URL: http://localhost:8080/api/v1.3/events?oom_events=true&stream=true
func getEventRequest(r *http.Request) (*events.Request, bool, error) {
	query := events.NewRequest()
	stream := false

	urlMap := r.URL.Query()

	if val, ok := urlMap["stream"]; ok {
		newBool, err := strconv.ParseBool(val[0])
		if err == nil {
			stream = newBool
		}
	}
	if val, ok := urlMap["subcontainers"]; ok {
		newBool, err := strconv.ParseBool(val[0])
		if err == nil {
			query.IncludeSubcontainers = newBool
		}
	}
	eventTypes := map[string]info.EventType{
		"oom_events":      info.EventOom,
		"oom_kill_events": info.EventOomKill,
		"creation_events": info.EventContainerCreation,
		"deletion_events": info.EventContainerDeletion,
	}
	allEventTypes := false
	if val, ok := urlMap["all_events"]; ok {
		newBool, err := strconv.ParseBool(val[0])
		if err == nil {
			allEventTypes = newBool
		}
	}
	for opt, eventType := range eventTypes {
		if allEventTypes {
			query.EventType[eventType] = true
		} else if val, ok := urlMap[opt]; ok {
			newBool, err := strconv.ParseBool(val[0])
			if err == nil {
				query.EventType[eventType] = newBool
			}
		}
	}
	if val, ok := urlMap["max_events"]; ok {
		newInt, err := strconv.Atoi(val[0])
		if err == nil {
			query.MaxEventsReturned = int(newInt)
		}
	}
	if val, ok := urlMap["start_time"]; ok {
		newTime, err := time.Parse(time.RFC3339, val[0])
		if err == nil {
			query.StartTime = newTime
		}
	}
	if val, ok := urlMap["end_time"]; ok {
		newTime, err := time.Parse(time.RFC3339, val[0])
		if err == nil {
			query.EndTime = newTime
		}
	}

	return query, stream, nil
}

func getContainerName(request []string) string {
	return path.Join("/", strings.Join(request, "/"))
}
