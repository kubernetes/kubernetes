// Copyright 2018 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package proxy

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io/ioutil"
	"log"
	"net/http"
	"reflect"
	"sync"

	"github.com/google/martian/martianlog"
)

// ForReplaying returns a Proxy configured to replay.
func ForReplaying(filename string, port int) (*Proxy, error) {
	p, err := newProxy(filename)
	if err != nil {
		return nil, err
	}
	lg, err := readLog(filename)
	if err != nil {
		return nil, err
	}
	calls, err := constructCalls(lg)
	if err != nil {
		return nil, err
	}
	p.Initial = lg.Initial
	p.mproxy.SetRoundTripper(&replayRoundTripper{
		calls:         calls,
		ignoreHeaders: p.ignoreHeaders,
		conv:          lg.Converter,
	})

	// Debug logging.
	// TODO(jba): factor out from here and ForRecording.
	logger := martianlog.NewLogger()
	logger.SetDecode(true)
	p.mproxy.SetRequestModifier(logger)
	p.mproxy.SetResponseModifier(logger)

	if err := p.start(port); err != nil {
		return nil, err
	}
	return p, nil
}

func readLog(filename string) (*Log, error) {
	bytes, err := ioutil.ReadFile(filename)
	if err != nil {
		return nil, err
	}
	var lg Log
	if err := json.Unmarshal(bytes, &lg); err != nil {
		return nil, fmt.Errorf("%s: %v", filename, err)
	}
	if lg.Version != LogVersion {
		return nil, fmt.Errorf(
			"httpreplay: read log version %s but current version is %s; re-record the log",
			lg.Version, LogVersion)
	}
	return &lg, nil
}

// A call is an HTTP request and its matching response.
type call struct {
	req *Request
	res *Response
}

func constructCalls(lg *Log) ([]*call, error) {
	ignoreIDs := map[string]bool{} // IDs of requests to ignore
	callsByID := map[string]*call{}
	var calls []*call
	for _, e := range lg.Entries {
		if ignoreIDs[e.ID] {
			continue
		}
		c, ok := callsByID[e.ID]
		switch {
		case !ok:
			if e.Request == nil {
				return nil, fmt.Errorf("first entry for ID %s does not have a request", e.ID)
			}
			if e.Request.Method == "CONNECT" {
				// Ignore CONNECT methods.
				ignoreIDs[e.ID] = true
			} else {
				c := &call{e.Request, e.Response}
				calls = append(calls, c)
				callsByID[e.ID] = c
			}
		case e.Request != nil:
			if e.Response != nil {
				return nil, errors.New("entry has both request and response")
			}
			c.req = e.Request
		case e.Response != nil:
			c.res = e.Response
		default:
			return nil, errors.New("entry has neither request nor response")
		}
	}
	for _, c := range calls {
		if c.req == nil || c.res == nil {
			return nil, fmt.Errorf("missing request or response: %+v", c)
		}
	}
	return calls, nil
}

type replayRoundTripper struct {
	mu            sync.Mutex
	calls         []*call
	ignoreHeaders map[string]bool
	conv          *Converter
}

func (r *replayRoundTripper) RoundTrip(req *http.Request) (*http.Response, error) {
	if req.Body != nil {
		defer req.Body.Close()
	}
	creq, err := r.conv.convertRequest(req)
	if err != nil {
		return nil, err
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	for i, call := range r.calls {
		if call == nil {
			continue
		}
		if requestsMatch(creq, call.req, r.ignoreHeaders) {
			r.calls[i] = nil // nil out this call so we don't reuse it
			return toHTTPResponse(call.res, req), nil
		}
	}
	return nil, fmt.Errorf("no matching request for %+v", req)
}

// Report whether the incoming request in matches the candidate request cand.
func requestsMatch(in, cand *Request, ignoreHeaders map[string]bool) bool {
	if in.Method != cand.Method {
		return false
	}
	if in.URL != cand.URL {
		return false
	}
	if in.MediaType != cand.MediaType {
		return false
	}
	if len(in.BodyParts) != len(cand.BodyParts) {
		return false
	}
	for i, p1 := range in.BodyParts {
		if !bytes.Equal(p1, cand.BodyParts[i]) {
			return false
		}
	}
	// Check headers last. See DebugHeaders.
	return headersMatch(in.Header, cand.Header, ignoreHeaders)
}

// DebugHeaders helps to determine whether a header should be ignored.
// When true, if requests have the same method, URL and body but differ
// in a header, the first mismatched header is logged.
var DebugHeaders = false

func headersMatch(in, cand http.Header, ignores map[string]bool) bool {
	for k1, v1 := range in {
		if ignores[k1] {
			continue
		}
		v2 := cand[k1]
		if v2 == nil {
			if DebugHeaders {
				log.Printf("header %s: present in incoming request but not candidate", k1)
			}
			return false
		}
		if !reflect.DeepEqual(v1, v2) {
			if DebugHeaders {
				log.Printf("header %s: incoming %v, candidate %v", k1, v1, v2)
			}
			return false
		}
	}
	for k2 := range cand {
		if ignores[k2] {
			continue
		}
		if in[k2] == nil {
			if DebugHeaders {
				log.Printf("header %s: not in incoming request but present in candidate", k2)
			}
			return false
		}
	}
	return true
}
