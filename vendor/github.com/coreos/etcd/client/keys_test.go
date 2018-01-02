// Copyright 2015 The etcd Authors
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

package client

import (
	"errors"
	"fmt"
	"io/ioutil"
	"net/http"
	"net/url"
	"reflect"
	"testing"
	"time"

	"golang.org/x/net/context"
)

func TestV2KeysURLHelper(t *testing.T) {
	tests := []struct {
		endpoint url.URL
		prefix   string
		key      string
		want     url.URL
	}{
		// key is empty, no problem
		{
			endpoint: url.URL{Scheme: "http", Host: "example.com", Path: "/v2/keys"},
			prefix:   "",
			key:      "",
			want:     url.URL{Scheme: "http", Host: "example.com", Path: "/v2/keys"},
		},

		// key is joined to path
		{
			endpoint: url.URL{Scheme: "http", Host: "example.com", Path: "/v2/keys"},
			prefix:   "",
			key:      "/foo/bar",
			want:     url.URL{Scheme: "http", Host: "example.com", Path: "/v2/keys/foo/bar"},
		},

		// key is joined to path when path is empty
		{
			endpoint: url.URL{Scheme: "http", Host: "example.com", Path: ""},
			prefix:   "",
			key:      "/foo/bar",
			want:     url.URL{Scheme: "http", Host: "example.com", Path: "/foo/bar"},
		},

		// Host field carries through with port
		{
			endpoint: url.URL{Scheme: "http", Host: "example.com:8080", Path: "/v2/keys"},
			prefix:   "",
			key:      "",
			want:     url.URL{Scheme: "http", Host: "example.com:8080", Path: "/v2/keys"},
		},

		// Scheme carries through
		{
			endpoint: url.URL{Scheme: "https", Host: "example.com", Path: "/v2/keys"},
			prefix:   "",
			key:      "",
			want:     url.URL{Scheme: "https", Host: "example.com", Path: "/v2/keys"},
		},
		// Prefix is applied
		{
			endpoint: url.URL{Scheme: "https", Host: "example.com", Path: "/foo"},
			prefix:   "/bar",
			key:      "/baz",
			want:     url.URL{Scheme: "https", Host: "example.com", Path: "/foo/bar/baz"},
		},
		// Prefix is joined to path
		{
			endpoint: url.URL{Scheme: "https", Host: "example.com", Path: "/foo"},
			prefix:   "/bar",
			key:      "",
			want:     url.URL{Scheme: "https", Host: "example.com", Path: "/foo/bar"},
		},
		// Keep trailing slash
		{
			endpoint: url.URL{Scheme: "https", Host: "example.com", Path: "/foo"},
			prefix:   "/bar",
			key:      "/baz/",
			want:     url.URL{Scheme: "https", Host: "example.com", Path: "/foo/bar/baz/"},
		},
	}

	for i, tt := range tests {
		got := v2KeysURL(tt.endpoint, tt.prefix, tt.key)
		if tt.want != *got {
			t.Errorf("#%d: want=%#v, got=%#v", i, tt.want, *got)
		}
	}
}

func TestGetAction(t *testing.T) {
	ep := url.URL{Scheme: "http", Host: "example.com", Path: "/v2/keys"}
	baseWantURL := &url.URL{
		Scheme: "http",
		Host:   "example.com",
		Path:   "/v2/keys/foo/bar",
	}
	wantHeader := http.Header{}

	tests := []struct {
		recursive bool
		sorted    bool
		quorum    bool
		wantQuery string
	}{
		{
			recursive: false,
			sorted:    false,
			quorum:    false,
			wantQuery: "quorum=false&recursive=false&sorted=false",
		},
		{
			recursive: true,
			sorted:    false,
			quorum:    false,
			wantQuery: "quorum=false&recursive=true&sorted=false",
		},
		{
			recursive: false,
			sorted:    true,
			quorum:    false,
			wantQuery: "quorum=false&recursive=false&sorted=true",
		},
		{
			recursive: true,
			sorted:    true,
			quorum:    false,
			wantQuery: "quorum=false&recursive=true&sorted=true",
		},
		{
			recursive: false,
			sorted:    false,
			quorum:    true,
			wantQuery: "quorum=true&recursive=false&sorted=false",
		},
	}

	for i, tt := range tests {
		f := getAction{
			Key:       "/foo/bar",
			Recursive: tt.recursive,
			Sorted:    tt.sorted,
			Quorum:    tt.quorum,
		}
		got := *f.HTTPRequest(ep)

		wantURL := baseWantURL
		wantURL.RawQuery = tt.wantQuery

		err := assertRequest(got, "GET", wantURL, wantHeader, nil)
		if err != nil {
			t.Errorf("#%d: %v", i, err)
		}
	}
}

func TestWaitAction(t *testing.T) {
	ep := url.URL{Scheme: "http", Host: "example.com", Path: "/v2/keys"}
	baseWantURL := &url.URL{
		Scheme: "http",
		Host:   "example.com",
		Path:   "/v2/keys/foo/bar",
	}
	wantHeader := http.Header{}

	tests := []struct {
		waitIndex uint64
		recursive bool
		wantQuery string
	}{
		{
			recursive: false,
			waitIndex: uint64(0),
			wantQuery: "recursive=false&wait=true&waitIndex=0",
		},
		{
			recursive: false,
			waitIndex: uint64(12),
			wantQuery: "recursive=false&wait=true&waitIndex=12",
		},
		{
			recursive: true,
			waitIndex: uint64(12),
			wantQuery: "recursive=true&wait=true&waitIndex=12",
		},
	}

	for i, tt := range tests {
		f := waitAction{
			Key:       "/foo/bar",
			WaitIndex: tt.waitIndex,
			Recursive: tt.recursive,
		}
		got := *f.HTTPRequest(ep)

		wantURL := baseWantURL
		wantURL.RawQuery = tt.wantQuery

		err := assertRequest(got, "GET", wantURL, wantHeader, nil)
		if err != nil {
			t.Errorf("#%d: unexpected error: %#v", i, err)
		}
	}
}

func TestSetAction(t *testing.T) {
	wantHeader := http.Header(map[string][]string{
		"Content-Type": {"application/x-www-form-urlencoded"},
	})

	tests := []struct {
		act      setAction
		wantURL  string
		wantBody string
	}{
		// default prefix
		{
			act: setAction{
				Prefix: defaultV2KeysPrefix,
				Key:    "foo",
			},
			wantURL:  "http://example.com/v2/keys/foo",
			wantBody: "value=",
		},

		// non-default prefix
		{
			act: setAction{
				Prefix: "/pfx",
				Key:    "foo",
			},
			wantURL:  "http://example.com/pfx/foo",
			wantBody: "value=",
		},

		// no prefix
		{
			act: setAction{
				Key: "foo",
			},
			wantURL:  "http://example.com/foo",
			wantBody: "value=",
		},

		// Key with path separators
		{
			act: setAction{
				Prefix: defaultV2KeysPrefix,
				Key:    "foo/bar/baz",
			},
			wantURL:  "http://example.com/v2/keys/foo/bar/baz",
			wantBody: "value=",
		},

		// Key with leading slash, Prefix with trailing slash
		{
			act: setAction{
				Prefix: "/foo/",
				Key:    "/bar",
			},
			wantURL:  "http://example.com/foo/bar",
			wantBody: "value=",
		},

		// Key with trailing slash
		{
			act: setAction{
				Key: "/foo/",
			},
			wantURL:  "http://example.com/foo/",
			wantBody: "value=",
		},

		// Value is set
		{
			act: setAction{
				Key:   "foo",
				Value: "baz",
			},
			wantURL:  "http://example.com/foo",
			wantBody: "value=baz",
		},

		// PrevExist set, but still ignored
		{
			act: setAction{
				Key:       "foo",
				PrevExist: PrevIgnore,
			},
			wantURL:  "http://example.com/foo",
			wantBody: "value=",
		},

		// PrevExist set to true
		{
			act: setAction{
				Key:       "foo",
				PrevExist: PrevExist,
			},
			wantURL:  "http://example.com/foo?prevExist=true",
			wantBody: "value=",
		},

		// PrevExist set to false
		{
			act: setAction{
				Key:       "foo",
				PrevExist: PrevNoExist,
			},
			wantURL:  "http://example.com/foo?prevExist=false",
			wantBody: "value=",
		},

		// PrevValue is urlencoded
		{
			act: setAction{
				Key:       "foo",
				PrevValue: "bar baz",
			},
			wantURL:  "http://example.com/foo?prevValue=bar+baz",
			wantBody: "value=",
		},

		// PrevIndex is set
		{
			act: setAction{
				Key:       "foo",
				PrevIndex: uint64(12),
			},
			wantURL:  "http://example.com/foo?prevIndex=12",
			wantBody: "value=",
		},

		// TTL is set
		{
			act: setAction{
				Key: "foo",
				TTL: 3 * time.Minute,
			},
			wantURL:  "http://example.com/foo",
			wantBody: "ttl=180&value=",
		},

		// Refresh is set
		{
			act: setAction{
				Key:     "foo",
				TTL:     3 * time.Minute,
				Refresh: true,
			},
			wantURL:  "http://example.com/foo",
			wantBody: "refresh=true&ttl=180&value=",
		},

		// Dir is set
		{
			act: setAction{
				Key: "foo",
				Dir: true,
			},
			wantURL:  "http://example.com/foo?dir=true",
			wantBody: "",
		},
		// Dir is set with a value
		{
			act: setAction{
				Key:   "foo",
				Value: "bar",
				Dir:   true,
			},
			wantURL:  "http://example.com/foo?dir=true",
			wantBody: "",
		},
		// Dir is set with PrevExist set to true
		{
			act: setAction{
				Key:       "foo",
				PrevExist: PrevExist,
				Dir:       true,
			},
			wantURL:  "http://example.com/foo?dir=true&prevExist=true",
			wantBody: "",
		},
		// Dir is set with PrevValue
		{
			act: setAction{
				Key:       "foo",
				PrevValue: "bar",
				Dir:       true,
			},
			wantURL:  "http://example.com/foo?dir=true",
			wantBody: "",
		},
		// NoValueOnSuccess is set
		{
			act: setAction{
				Key:              "foo",
				NoValueOnSuccess: true,
			},
			wantURL:  "http://example.com/foo?noValueOnSuccess=true",
			wantBody: "value=",
		},
	}

	for i, tt := range tests {
		u, err := url.Parse(tt.wantURL)
		if err != nil {
			t.Errorf("#%d: unable to use wantURL fixture: %v", i, err)
		}

		got := tt.act.HTTPRequest(url.URL{Scheme: "http", Host: "example.com"})
		if err := assertRequest(*got, "PUT", u, wantHeader, []byte(tt.wantBody)); err != nil {
			t.Errorf("#%d: %v", i, err)
		}
	}
}

func TestCreateInOrderAction(t *testing.T) {
	wantHeader := http.Header(map[string][]string{
		"Content-Type": {"application/x-www-form-urlencoded"},
	})

	tests := []struct {
		act      createInOrderAction
		wantURL  string
		wantBody string
	}{
		// default prefix
		{
			act: createInOrderAction{
				Prefix: defaultV2KeysPrefix,
				Dir:    "foo",
			},
			wantURL:  "http://example.com/v2/keys/foo",
			wantBody: "value=",
		},

		// non-default prefix
		{
			act: createInOrderAction{
				Prefix: "/pfx",
				Dir:    "foo",
			},
			wantURL:  "http://example.com/pfx/foo",
			wantBody: "value=",
		},

		// no prefix
		{
			act: createInOrderAction{
				Dir: "foo",
			},
			wantURL:  "http://example.com/foo",
			wantBody: "value=",
		},

		// Key with path separators
		{
			act: createInOrderAction{
				Prefix: defaultV2KeysPrefix,
				Dir:    "foo/bar/baz",
			},
			wantURL:  "http://example.com/v2/keys/foo/bar/baz",
			wantBody: "value=",
		},

		// Key with leading slash, Prefix with trailing slash
		{
			act: createInOrderAction{
				Prefix: "/foo/",
				Dir:    "/bar",
			},
			wantURL:  "http://example.com/foo/bar",
			wantBody: "value=",
		},

		// Key with trailing slash
		{
			act: createInOrderAction{
				Dir: "/foo/",
			},
			wantURL:  "http://example.com/foo/",
			wantBody: "value=",
		},

		// Value is set
		{
			act: createInOrderAction{
				Dir:   "foo",
				Value: "baz",
			},
			wantURL:  "http://example.com/foo",
			wantBody: "value=baz",
		},
		// TTL is set
		{
			act: createInOrderAction{
				Dir: "foo",
				TTL: 3 * time.Minute,
			},
			wantURL:  "http://example.com/foo",
			wantBody: "ttl=180&value=",
		},
	}

	for i, tt := range tests {
		u, err := url.Parse(tt.wantURL)
		if err != nil {
			t.Errorf("#%d: unable to use wantURL fixture: %v", i, err)
		}

		got := tt.act.HTTPRequest(url.URL{Scheme: "http", Host: "example.com"})
		if err := assertRequest(*got, "POST", u, wantHeader, []byte(tt.wantBody)); err != nil {
			t.Errorf("#%d: %v", i, err)
		}
	}
}

func TestDeleteAction(t *testing.T) {
	wantHeader := http.Header(map[string][]string{
		"Content-Type": {"application/x-www-form-urlencoded"},
	})

	tests := []struct {
		act     deleteAction
		wantURL string
	}{
		// default prefix
		{
			act: deleteAction{
				Prefix: defaultV2KeysPrefix,
				Key:    "foo",
			},
			wantURL: "http://example.com/v2/keys/foo",
		},

		// non-default prefix
		{
			act: deleteAction{
				Prefix: "/pfx",
				Key:    "foo",
			},
			wantURL: "http://example.com/pfx/foo",
		},

		// no prefix
		{
			act: deleteAction{
				Key: "foo",
			},
			wantURL: "http://example.com/foo",
		},

		// Key with path separators
		{
			act: deleteAction{
				Prefix: defaultV2KeysPrefix,
				Key:    "foo/bar/baz",
			},
			wantURL: "http://example.com/v2/keys/foo/bar/baz",
		},

		// Key with leading slash, Prefix with trailing slash
		{
			act: deleteAction{
				Prefix: "/foo/",
				Key:    "/bar",
			},
			wantURL: "http://example.com/foo/bar",
		},

		// Key with trailing slash
		{
			act: deleteAction{
				Key: "/foo/",
			},
			wantURL: "http://example.com/foo/",
		},

		// Recursive set to true
		{
			act: deleteAction{
				Key:       "foo",
				Recursive: true,
			},
			wantURL: "http://example.com/foo?recursive=true",
		},

		// PrevValue is urlencoded
		{
			act: deleteAction{
				Key:       "foo",
				PrevValue: "bar baz",
			},
			wantURL: "http://example.com/foo?prevValue=bar+baz",
		},

		// PrevIndex is set
		{
			act: deleteAction{
				Key:       "foo",
				PrevIndex: uint64(12),
			},
			wantURL: "http://example.com/foo?prevIndex=12",
		},
	}

	for i, tt := range tests {
		u, err := url.Parse(tt.wantURL)
		if err != nil {
			t.Errorf("#%d: unable to use wantURL fixture: %v", i, err)
		}

		got := tt.act.HTTPRequest(url.URL{Scheme: "http", Host: "example.com"})
		if err := assertRequest(*got, "DELETE", u, wantHeader, nil); err != nil {
			t.Errorf("#%d: %v", i, err)
		}
	}
}

func assertRequest(got http.Request, wantMethod string, wantURL *url.URL, wantHeader http.Header, wantBody []byte) error {
	if wantMethod != got.Method {
		return fmt.Errorf("want.Method=%#v got.Method=%#v", wantMethod, got.Method)
	}

	if !reflect.DeepEqual(wantURL, got.URL) {
		return fmt.Errorf("want.URL=%#v got.URL=%#v", wantURL, got.URL)
	}

	if !reflect.DeepEqual(wantHeader, got.Header) {
		return fmt.Errorf("want.Header=%#v got.Header=%#v", wantHeader, got.Header)
	}

	if got.Body == nil {
		if wantBody != nil {
			return fmt.Errorf("want.Body=%v got.Body=%v", wantBody, got.Body)
		}
	} else {
		if wantBody == nil {
			return fmt.Errorf("want.Body=%v got.Body=%s", wantBody, got.Body)
		}
		gotBytes, err := ioutil.ReadAll(got.Body)
		if err != nil {
			return err
		}

		if !reflect.DeepEqual(wantBody, gotBytes) {
			return fmt.Errorf("want.Body=%s got.Body=%s", wantBody, gotBytes)
		}
	}

	return nil
}

func TestUnmarshalSuccessfulResponse(t *testing.T) {
	var expiration time.Time
	expiration.UnmarshalText([]byte("2015-04-07T04:40:23.044979686Z"))

	tests := []struct {
		indexHdr     string
		clusterIDHdr string
		body         string
		wantRes      *Response
		wantErr      bool
	}{
		// Neither PrevNode or Node
		{
			indexHdr: "1",
			body:     `{"action":"delete"}`,
			wantRes:  &Response{Action: "delete", Index: 1},
			wantErr:  false,
		},

		// PrevNode
		{
			indexHdr: "15",
			body:     `{"action":"delete", "prevNode": {"key": "/foo", "value": "bar", "modifiedIndex": 12, "createdIndex": 10}}`,
			wantRes: &Response{
				Action: "delete",
				Index:  15,
				Node:   nil,
				PrevNode: &Node{
					Key:           "/foo",
					Value:         "bar",
					ModifiedIndex: 12,
					CreatedIndex:  10,
				},
			},
			wantErr: false,
		},

		// Node
		{
			indexHdr: "15",
			body:     `{"action":"get", "node": {"key": "/foo", "value": "bar", "modifiedIndex": 12, "createdIndex": 10, "ttl": 10, "expiration": "2015-04-07T04:40:23.044979686Z"}}`,
			wantRes: &Response{
				Action: "get",
				Index:  15,
				Node: &Node{
					Key:           "/foo",
					Value:         "bar",
					ModifiedIndex: 12,
					CreatedIndex:  10,
					TTL:           10,
					Expiration:    &expiration,
				},
				PrevNode: nil,
			},
			wantErr: false,
		},

		// Node Dir
		{
			indexHdr:     "15",
			clusterIDHdr: "abcdef",
			body:         `{"action":"get", "node": {"key": "/foo", "dir": true, "modifiedIndex": 12, "createdIndex": 10}}`,
			wantRes: &Response{
				Action: "get",
				Index:  15,
				Node: &Node{
					Key:           "/foo",
					Dir:           true,
					ModifiedIndex: 12,
					CreatedIndex:  10,
				},
				PrevNode:  nil,
				ClusterID: "abcdef",
			},
			wantErr: false,
		},

		// PrevNode and Node
		{
			indexHdr: "15",
			body:     `{"action":"update", "prevNode": {"key": "/foo", "value": "baz", "modifiedIndex": 10, "createdIndex": 10}, "node": {"key": "/foo", "value": "bar", "modifiedIndex": 12, "createdIndex": 10}}`,
			wantRes: &Response{
				Action: "update",
				Index:  15,
				PrevNode: &Node{
					Key:           "/foo",
					Value:         "baz",
					ModifiedIndex: 10,
					CreatedIndex:  10,
				},
				Node: &Node{
					Key:           "/foo",
					Value:         "bar",
					ModifiedIndex: 12,
					CreatedIndex:  10,
				},
			},
			wantErr: false,
		},

		// Garbage in body
		{
			indexHdr: "",
			body:     `garbage`,
			wantRes:  nil,
			wantErr:  true,
		},

		// non-integer index
		{
			indexHdr: "poo",
			body:     `{}`,
			wantRes:  nil,
			wantErr:  true,
		},
	}

	for i, tt := range tests {
		h := make(http.Header)
		h.Add("X-Etcd-Index", tt.indexHdr)
		res, err := unmarshalSuccessfulKeysResponse(h, []byte(tt.body))
		if tt.wantErr != (err != nil) {
			t.Errorf("#%d: wantErr=%t, err=%v", i, tt.wantErr, err)
		}

		if (res == nil) != (tt.wantRes == nil) {
			t.Errorf("#%d: received res=%#v, but expected res=%#v", i, res, tt.wantRes)
			continue
		} else if tt.wantRes == nil {
			// expected and successfully got nil response
			continue
		}

		if res.Action != tt.wantRes.Action {
			t.Errorf("#%d: Action=%s, expected %s", i, res.Action, tt.wantRes.Action)
		}
		if res.Index != tt.wantRes.Index {
			t.Errorf("#%d: Index=%d, expected %d", i, res.Index, tt.wantRes.Index)
		}
		if !reflect.DeepEqual(res.Node, tt.wantRes.Node) {
			t.Errorf("#%d: Node=%v, expected %v", i, res.Node, tt.wantRes.Node)
		}
	}
}

func TestUnmarshalFailedKeysResponse(t *testing.T) {
	body := []byte(`{"errorCode":100,"message":"Key not found","cause":"/foo","index":18}`)

	wantErr := Error{
		Code:    100,
		Message: "Key not found",
		Cause:   "/foo",
		Index:   uint64(18),
	}

	gotErr := unmarshalFailedKeysResponse(body)
	if !reflect.DeepEqual(wantErr, gotErr) {
		t.Errorf("unexpected error: want=%#v got=%#v", wantErr, gotErr)
	}
}

func TestUnmarshalFailedKeysResponseBadJSON(t *testing.T) {
	err := unmarshalFailedKeysResponse([]byte(`{"er`))
	if err == nil {
		t.Errorf("got nil error")
	} else if _, ok := err.(Error); ok {
		t.Errorf("error is of incorrect type *Error: %#v", err)
	}
}

func TestHTTPWatcherNextWaitAction(t *testing.T) {
	initAction := waitAction{
		Prefix:    "/pants",
		Key:       "/foo/bar",
		Recursive: true,
		WaitIndex: 19,
	}

	client := &actionAssertingHTTPClient{
		t:   t,
		act: &initAction,
		resp: http.Response{
			StatusCode: http.StatusOK,
			Header:     http.Header{"X-Etcd-Index": []string{"42"}},
		},
		body: []byte(`{"action":"update","node":{"key":"/pants/foo/bar/baz","value":"snarf","modifiedIndex":21,"createdIndex":19},"prevNode":{"key":"/pants/foo/bar/baz","value":"snazz","modifiedIndex":20,"createdIndex":19}}`),
	}

	wantResponse := &Response{
		Action:   "update",
		Node:     &Node{Key: "/pants/foo/bar/baz", Value: "snarf", CreatedIndex: uint64(19), ModifiedIndex: uint64(21)},
		PrevNode: &Node{Key: "/pants/foo/bar/baz", Value: "snazz", CreatedIndex: uint64(19), ModifiedIndex: uint64(20)},
		Index:    uint64(42),
	}

	wantNextWait := waitAction{
		Prefix:    "/pants",
		Key:       "/foo/bar",
		Recursive: true,
		WaitIndex: 22,
	}

	watcher := &httpWatcher{
		client:   client,
		nextWait: initAction,
	}

	resp, err := watcher.Next(context.Background())
	if err != nil {
		t.Errorf("non-nil error: %#v", err)
	}

	if !reflect.DeepEqual(wantResponse, resp) {
		t.Errorf("received incorrect Response: want=%#v got=%#v", wantResponse, resp)
	}

	if !reflect.DeepEqual(wantNextWait, watcher.nextWait) {
		t.Errorf("nextWait incorrect: want=%#v got=%#v", wantNextWait, watcher.nextWait)
	}
}

func TestHTTPWatcherNextFail(t *testing.T) {
	tests := []httpClient{
		// generic HTTP client failure
		&staticHTTPClient{
			err: errors.New("fail!"),
		},

		// unusable status code
		&staticHTTPClient{
			resp: http.Response{
				StatusCode: http.StatusTeapot,
			},
		},

		// etcd Error response
		&staticHTTPClient{
			resp: http.Response{
				StatusCode: http.StatusNotFound,
			},
			body: []byte(`{"errorCode":100,"message":"Key not found","cause":"/foo","index":18}`),
		},
	}

	for i, tt := range tests {
		act := waitAction{
			Prefix:    "/pants",
			Key:       "/foo/bar",
			Recursive: true,
			WaitIndex: 19,
		}

		watcher := &httpWatcher{
			client:   tt,
			nextWait: act,
		}

		resp, err := watcher.Next(context.Background())
		if err == nil {
			t.Errorf("#%d: expected non-nil error", i)
		}
		if resp != nil {
			t.Errorf("#%d: expected nil Response, got %#v", i, resp)
		}
		if !reflect.DeepEqual(act, watcher.nextWait) {
			t.Errorf("#%d: nextWait changed: want=%#v got=%#v", i, act, watcher.nextWait)
		}
	}
}

func TestHTTPKeysAPIWatcherAction(t *testing.T) {
	tests := []struct {
		key  string
		opts *WatcherOptions
		want waitAction
	}{
		{
			key:  "/foo",
			opts: nil,
			want: waitAction{
				Key:       "/foo",
				Recursive: false,
				WaitIndex: 0,
			},
		},

		{
			key: "/foo",
			opts: &WatcherOptions{
				Recursive:  false,
				AfterIndex: 0,
			},
			want: waitAction{
				Key:       "/foo",
				Recursive: false,
				WaitIndex: 0,
			},
		},

		{
			key: "/foo",
			opts: &WatcherOptions{
				Recursive:  true,
				AfterIndex: 0,
			},
			want: waitAction{
				Key:       "/foo",
				Recursive: true,
				WaitIndex: 0,
			},
		},

		{
			key: "/foo",
			opts: &WatcherOptions{
				Recursive:  false,
				AfterIndex: 19,
			},
			want: waitAction{
				Key:       "/foo",
				Recursive: false,
				WaitIndex: 20,
			},
		},
	}

	for i, tt := range tests {
		kAPI := &httpKeysAPI{
			client: &staticHTTPClient{err: errors.New("fail!")},
		}

		want := &httpWatcher{
			client:   &staticHTTPClient{err: errors.New("fail!")},
			nextWait: tt.want,
		}

		got := kAPI.Watcher(tt.key, tt.opts)
		if !reflect.DeepEqual(want, got) {
			t.Errorf("#%d: incorrect watcher: want=%#v got=%#v", i, want, got)
		}
	}
}

func TestHTTPKeysAPISetAction(t *testing.T) {
	tests := []struct {
		key        string
		value      string
		opts       *SetOptions
		wantAction httpAction
	}{
		// nil SetOptions
		{
			key:   "/foo",
			value: "bar",
			opts:  nil,
			wantAction: &setAction{
				Key:       "/foo",
				Value:     "bar",
				PrevValue: "",
				PrevIndex: 0,
				PrevExist: PrevIgnore,
				TTL:       0,
			},
		},
		// empty SetOptions
		{
			key:   "/foo",
			value: "bar",
			opts:  &SetOptions{},
			wantAction: &setAction{
				Key:       "/foo",
				Value:     "bar",
				PrevValue: "",
				PrevIndex: 0,
				PrevExist: PrevIgnore,
				TTL:       0,
			},
		},
		// populated SetOptions
		{
			key:   "/foo",
			value: "bar",
			opts: &SetOptions{
				PrevValue: "baz",
				PrevIndex: 13,
				PrevExist: PrevExist,
				TTL:       time.Minute,
				Dir:       true,
			},
			wantAction: &setAction{
				Key:       "/foo",
				Value:     "bar",
				PrevValue: "baz",
				PrevIndex: 13,
				PrevExist: PrevExist,
				TTL:       time.Minute,
				Dir:       true,
			},
		},
	}

	for i, tt := range tests {
		client := &actionAssertingHTTPClient{t: t, num: i, act: tt.wantAction}
		kAPI := httpKeysAPI{client: client}
		kAPI.Set(context.Background(), tt.key, tt.value, tt.opts)
	}
}

func TestHTTPKeysAPISetError(t *testing.T) {
	tests := []httpClient{
		// generic HTTP client failure
		&staticHTTPClient{
			err: errors.New("fail!"),
		},

		// unusable status code
		&staticHTTPClient{
			resp: http.Response{
				StatusCode: http.StatusTeapot,
			},
		},

		// etcd Error response
		&staticHTTPClient{
			resp: http.Response{
				StatusCode: http.StatusInternalServerError,
			},
			body: []byte(`{"errorCode":300,"message":"Raft internal error","cause":"/foo","index":18}`),
		},
	}

	for i, tt := range tests {
		kAPI := httpKeysAPI{client: tt}
		resp, err := kAPI.Set(context.Background(), "/foo", "bar", nil)
		if err == nil {
			t.Errorf("#%d: received nil error", i)
		}
		if resp != nil {
			t.Errorf("#%d: received non-nil Response: %#v", i, resp)
		}
	}
}

func TestHTTPKeysAPISetResponse(t *testing.T) {
	client := &staticHTTPClient{
		resp: http.Response{
			StatusCode: http.StatusOK,
			Header:     http.Header{"X-Etcd-Index": []string{"21"}},
		},
		body: []byte(`{"action":"set","node":{"key":"/pants/foo/bar/baz","value":"snarf","modifiedIndex":21,"createdIndex":21},"prevNode":{"key":"/pants/foo/bar/baz","value":"snazz","modifiedIndex":20,"createdIndex":19}}`),
	}

	wantResponse := &Response{
		Action:   "set",
		Node:     &Node{Key: "/pants/foo/bar/baz", Value: "snarf", CreatedIndex: uint64(21), ModifiedIndex: uint64(21)},
		PrevNode: &Node{Key: "/pants/foo/bar/baz", Value: "snazz", CreatedIndex: uint64(19), ModifiedIndex: uint64(20)},
		Index:    uint64(21),
	}

	kAPI := &httpKeysAPI{client: client, prefix: "/pants"}
	resp, err := kAPI.Set(context.Background(), "/foo/bar/baz", "snarf", nil)
	if err != nil {
		t.Errorf("non-nil error: %#v", err)
	}
	if !reflect.DeepEqual(wantResponse, resp) {
		t.Errorf("incorrect Response: want=%#v got=%#v", wantResponse, resp)
	}
}

func TestHTTPKeysAPIGetAction(t *testing.T) {
	tests := []struct {
		key        string
		opts       *GetOptions
		wantAction httpAction
	}{
		// nil GetOptions
		{
			key:  "/foo",
			opts: nil,
			wantAction: &getAction{
				Key:       "/foo",
				Sorted:    false,
				Recursive: false,
			},
		},
		// empty GetOptions
		{
			key:  "/foo",
			opts: &GetOptions{},
			wantAction: &getAction{
				Key:       "/foo",
				Sorted:    false,
				Recursive: false,
			},
		},
		// populated GetOptions
		{
			key: "/foo",
			opts: &GetOptions{
				Sort:      true,
				Recursive: true,
				Quorum:    true,
			},
			wantAction: &getAction{
				Key:       "/foo",
				Sorted:    true,
				Recursive: true,
				Quorum:    true,
			},
		},
	}

	for i, tt := range tests {
		client := &actionAssertingHTTPClient{t: t, num: i, act: tt.wantAction}
		kAPI := httpKeysAPI{client: client}
		kAPI.Get(context.Background(), tt.key, tt.opts)
	}
}

func TestHTTPKeysAPIGetError(t *testing.T) {
	tests := []httpClient{
		// generic HTTP client failure
		&staticHTTPClient{
			err: errors.New("fail!"),
		},

		// unusable status code
		&staticHTTPClient{
			resp: http.Response{
				StatusCode: http.StatusTeapot,
			},
		},

		// etcd Error response
		&staticHTTPClient{
			resp: http.Response{
				StatusCode: http.StatusInternalServerError,
			},
			body: []byte(`{"errorCode":300,"message":"Raft internal error","cause":"/foo","index":18}`),
		},
	}

	for i, tt := range tests {
		kAPI := httpKeysAPI{client: tt}
		resp, err := kAPI.Get(context.Background(), "/foo", nil)
		if err == nil {
			t.Errorf("#%d: received nil error", i)
		}
		if resp != nil {
			t.Errorf("#%d: received non-nil Response: %#v", i, resp)
		}
	}
}

func TestHTTPKeysAPIGetResponse(t *testing.T) {
	client := &staticHTTPClient{
		resp: http.Response{
			StatusCode: http.StatusOK,
			Header:     http.Header{"X-Etcd-Index": []string{"42"}},
		},
		body: []byte(`{"action":"get","node":{"key":"/pants/foo/bar","modifiedIndex":25,"createdIndex":19,"nodes":[{"key":"/pants/foo/bar/baz","value":"snarf","createdIndex":21,"modifiedIndex":25}]}}`),
	}

	wantResponse := &Response{
		Action: "get",
		Node: &Node{
			Key: "/pants/foo/bar",
			Nodes: []*Node{
				{Key: "/pants/foo/bar/baz", Value: "snarf", CreatedIndex: 21, ModifiedIndex: 25},
			},
			CreatedIndex:  uint64(19),
			ModifiedIndex: uint64(25),
		},
		Index: uint64(42),
	}

	kAPI := &httpKeysAPI{client: client, prefix: "/pants"}
	resp, err := kAPI.Get(context.Background(), "/foo/bar", &GetOptions{Recursive: true})
	if err != nil {
		t.Errorf("non-nil error: %#v", err)
	}
	if !reflect.DeepEqual(wantResponse, resp) {
		t.Errorf("incorrect Response: want=%#v got=%#v", wantResponse, resp)
	}
}

func TestHTTPKeysAPIDeleteAction(t *testing.T) {
	tests := []struct {
		key        string
		opts       *DeleteOptions
		wantAction httpAction
	}{
		// nil DeleteOptions
		{
			key:  "/foo",
			opts: nil,
			wantAction: &deleteAction{
				Key:       "/foo",
				PrevValue: "",
				PrevIndex: 0,
				Recursive: false,
			},
		},
		// empty DeleteOptions
		{
			key:  "/foo",
			opts: &DeleteOptions{},
			wantAction: &deleteAction{
				Key:       "/foo",
				PrevValue: "",
				PrevIndex: 0,
				Recursive: false,
			},
		},
		// populated DeleteOptions
		{
			key: "/foo",
			opts: &DeleteOptions{
				PrevValue: "baz",
				PrevIndex: 13,
				Recursive: true,
			},
			wantAction: &deleteAction{
				Key:       "/foo",
				PrevValue: "baz",
				PrevIndex: 13,
				Recursive: true,
			},
		},
	}

	for i, tt := range tests {
		client := &actionAssertingHTTPClient{t: t, num: i, act: tt.wantAction}
		kAPI := httpKeysAPI{client: client}
		kAPI.Delete(context.Background(), tt.key, tt.opts)
	}
}

func TestHTTPKeysAPIDeleteError(t *testing.T) {
	tests := []httpClient{
		// generic HTTP client failure
		&staticHTTPClient{
			err: errors.New("fail!"),
		},

		// unusable status code
		&staticHTTPClient{
			resp: http.Response{
				StatusCode: http.StatusTeapot,
			},
		},

		// etcd Error response
		&staticHTTPClient{
			resp: http.Response{
				StatusCode: http.StatusInternalServerError,
			},
			body: []byte(`{"errorCode":300,"message":"Raft internal error","cause":"/foo","index":18}`),
		},
	}

	for i, tt := range tests {
		kAPI := httpKeysAPI{client: tt}
		resp, err := kAPI.Delete(context.Background(), "/foo", nil)
		if err == nil {
			t.Errorf("#%d: received nil error", i)
		}
		if resp != nil {
			t.Errorf("#%d: received non-nil Response: %#v", i, resp)
		}
	}
}

func TestHTTPKeysAPIDeleteResponse(t *testing.T) {
	client := &staticHTTPClient{
		resp: http.Response{
			StatusCode: http.StatusOK,
			Header:     http.Header{"X-Etcd-Index": []string{"22"}},
		},
		body: []byte(`{"action":"delete","node":{"key":"/pants/foo/bar/baz","value":"snarf","modifiedIndex":22,"createdIndex":19},"prevNode":{"key":"/pants/foo/bar/baz","value":"snazz","modifiedIndex":20,"createdIndex":19}}`),
	}

	wantResponse := &Response{
		Action:   "delete",
		Node:     &Node{Key: "/pants/foo/bar/baz", Value: "snarf", CreatedIndex: uint64(19), ModifiedIndex: uint64(22)},
		PrevNode: &Node{Key: "/pants/foo/bar/baz", Value: "snazz", CreatedIndex: uint64(19), ModifiedIndex: uint64(20)},
		Index:    uint64(22),
	}

	kAPI := &httpKeysAPI{client: client, prefix: "/pants"}
	resp, err := kAPI.Delete(context.Background(), "/foo/bar/baz", nil)
	if err != nil {
		t.Errorf("non-nil error: %#v", err)
	}
	if !reflect.DeepEqual(wantResponse, resp) {
		t.Errorf("incorrect Response: want=%#v got=%#v", wantResponse, resp)
	}
}

func TestHTTPKeysAPICreateAction(t *testing.T) {
	act := &setAction{
		Key:       "/foo",
		Value:     "bar",
		PrevExist: PrevNoExist,
		PrevIndex: 0,
		PrevValue: "",
		TTL:       0,
	}

	kAPI := httpKeysAPI{client: &actionAssertingHTTPClient{t: t, act: act}}
	kAPI.Create(context.Background(), "/foo", "bar")
}

func TestHTTPKeysAPICreateInOrderAction(t *testing.T) {
	act := &createInOrderAction{
		Dir:   "/foo",
		Value: "bar",
		TTL:   0,
	}
	kAPI := httpKeysAPI{client: &actionAssertingHTTPClient{t: t, act: act}}
	kAPI.CreateInOrder(context.Background(), "/foo", "bar", nil)
}

func TestHTTPKeysAPIUpdateAction(t *testing.T) {
	act := &setAction{
		Key:       "/foo",
		Value:     "bar",
		PrevExist: PrevExist,
		PrevIndex: 0,
		PrevValue: "",
		TTL:       0,
	}

	kAPI := httpKeysAPI{client: &actionAssertingHTTPClient{t: t, act: act}}
	kAPI.Update(context.Background(), "/foo", "bar")
}

func TestNodeTTLDuration(t *testing.T) {
	tests := []struct {
		node *Node
		want time.Duration
	}{
		{
			node: &Node{TTL: 0},
			want: 0,
		},
		{
			node: &Node{TTL: 97},
			want: 97 * time.Second,
		},
	}

	for i, tt := range tests {
		got := tt.node.TTLDuration()
		if tt.want != got {
			t.Errorf("#%d: incorrect duration: want=%v got=%v", i, tt.want, got)
		}
	}
}
