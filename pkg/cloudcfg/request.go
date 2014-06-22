/*
Copyright 2014 Google Inc. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package cloudcfg

import (
	"bytes"
	"io"
	"io/ioutil"
	"net/http"
	"net/url"
	"path"
	"time"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/labels"
)

// Server contains info locating a kubernetes api server.
// Example usage:
// auth, err := LoadAuth(filename)
// s := New(url, auth)
// resp, err := s.Verb("GET").
//	Path("api/v1beta1").
//	Path("pods").
//	Selector("area=staging").
//	Timeout(10*time.Second).
//	Do()
// list, ok := resp.(api.PodList)
type Server struct {
	auth   *client.AuthInfo
	rawUrl string
}

// Create a new server object.
func New(serverUrl string, auth *client.AuthInfo) *Server {
	return &Server{
		auth:   auth,
		rawUrl: serverUrl,
	}
}

// Begin a request with a verb (GET, POST, PUT, DELETE)
func (s *Server) Verb(verb string) *Request {
	return &Request{
		verb: verb,
		s:    s,
		path: "/",
	}
}

// Request allows for building up a request to a server in a chained fashion.
type Request struct {
	s        *Server
	err      error
	verb     string
	path     string
	body     interface{}
	selector labels.Selector
	timeout  time.Duration
}

// Append an item to the request path. You must call Path at least once.
func (r *Request) Path(item string) *Request {
	if r.err != nil {
		return r
	}
	r.path = path.Join(r.path, item)
	return r
}

// Use the given item as a resource label selector. Optional.
func (r *Request) Selector(item string) *Request {
	if r.err != nil {
		return r
	}
	r.selector, r.err = labels.ParseSelector(item)
	return r
}

// Use the given duration as a timeout. Optional.
func (r *Request) Timeout(d time.Duration) *Request {
	if r.err != nil {
		return r
	}
	r.timeout = d
	return r
}

// Use obj as the body of the request. Optional.
// If obj is a string, try to read a file of that name.
// If obj is a []byte, send it directly.
// Otherwise, assume obj is an api type and marshall it correctly.
func (r *Request) Body(obj interface{}) *Request {
	if r.err != nil {
		return r
	}
	r.body = obj
	return r
}

// Format and xecute the request. Returns the API object received, or an error.
func (r *Request) Do() (interface{}, error) {
	if r.err != nil {
		return nil, r.err
	}
	finalUrl := r.s.rawUrl + r.path
	query := url.Values{}
	if r.selector != nil {
		query.Add("labels", r.selector.String())
	}
	if r.timeout != 0 {
		query.Add("timeout", r.timeout.String())
	}
	finalUrl += "?" + query.Encode()
	var body io.Reader
	if r.body != nil {
		switch t := r.body.(type) {
		case string:
			data, err := ioutil.ReadFile(t)
			if err != nil {
				return nil, err
			}
			body = bytes.NewBuffer(data)
		case []byte:
			body = bytes.NewBuffer(t)
		default:
			data, err := api.Encode(r.body)
			if err != nil {
				return nil, err
			}
			body = bytes.NewBuffer(data)
		}
	}
	req, err := http.NewRequest(r.verb, finalUrl, body)
	if err != nil {
		return nil, err
	}
	str, err := doRequest(req, r.s.auth)
	if err != nil {
		return nil, err
	}
	return api.Decode([]byte(str))
}
