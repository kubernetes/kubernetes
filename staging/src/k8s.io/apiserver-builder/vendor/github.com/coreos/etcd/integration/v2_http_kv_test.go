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

package integration

import (
	"encoding/json"
	"fmt"
	"io"
	"io/ioutil"
	"net/http"
	"net/url"
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/coreos/etcd/pkg/testutil"
	"github.com/coreos/etcd/pkg/transport"
	"github.com/coreos/pkg/capnslog"
)

func init() {
	capnslog.SetGlobalLogLevel(capnslog.CRITICAL)
}

func TestV2Set(t *testing.T) {
	defer testutil.AfterTest(t)
	cl := NewCluster(t, 1)
	cl.Launch(t)
	defer cl.Terminate(t)

	u := cl.URL(0)
	tc := NewTestClient()
	v := url.Values{}
	v.Set("value", "bar")

	tests := []struct {
		relativeURL string
		value       url.Values
		wStatus     int
		w           string
	}{
		{
			"/v2/keys/foo/bar",
			v,
			http.StatusCreated,
			`{"action":"set","node":{"key":"/foo/bar","value":"bar","modifiedIndex":4,"createdIndex":4}}`,
		},
		{
			"/v2/keys/foodir?dir=true",
			url.Values{},
			http.StatusCreated,
			`{"action":"set","node":{"key":"/foodir","dir":true,"modifiedIndex":5,"createdIndex":5}}`,
		},
		{
			"/v2/keys/fooempty",
			url.Values(map[string][]string{"value": {""}}),
			http.StatusCreated,
			`{"action":"set","node":{"key":"/fooempty","value":"","modifiedIndex":6,"createdIndex":6}}`,
		},
	}

	for i, tt := range tests {
		resp, err := tc.PutForm(fmt.Sprintf("%s%s", u, tt.relativeURL), tt.value)
		if err != nil {
			t.Errorf("#%d: err = %v, want nil", i, err)
		}
		g := string(tc.ReadBody(resp))
		w := tt.w + "\n"
		if g != w {
			t.Errorf("#%d: body = %v, want %v", i, g, w)
		}
		if resp.StatusCode != tt.wStatus {
			t.Errorf("#%d: status = %d, want %d", i, resp.StatusCode, tt.wStatus)
		}
	}
}

func TestV2CreateUpdate(t *testing.T) {
	defer testutil.AfterTest(t)
	cl := NewCluster(t, 1)
	cl.Launch(t)
	defer cl.Terminate(t)

	u := cl.URL(0)
	tc := NewTestClient()

	tests := []struct {
		relativeURL string
		value       url.Values
		wStatus     int
		w           map[string]interface{}
	}{
		// key with ttl
		{
			"/v2/keys/ttl/foo",
			url.Values(map[string][]string{"value": {"XXX"}, "ttl": {"20"}}),
			http.StatusCreated,
			map[string]interface{}{
				"node": map[string]interface{}{
					"value": "XXX",
					"ttl":   float64(20),
				},
			},
		},
		// key with bad ttl
		{
			"/v2/keys/ttl/foo",
			url.Values(map[string][]string{"value": {"XXX"}, "ttl": {"bad_ttl"}}),
			http.StatusBadRequest,
			map[string]interface{}{
				"errorCode": float64(202),
				"message":   "The given TTL in POST form is not a number",
			},
		},
		// create key
		{
			"/v2/keys/create/foo",
			url.Values(map[string][]string{"value": {"XXX"}, "prevExist": {"false"}}),
			http.StatusCreated,
			map[string]interface{}{
				"node": map[string]interface{}{
					"value": "XXX",
				},
			},
		},
		// created key failed
		{
			"/v2/keys/create/foo",
			url.Values(map[string][]string{"value": {"XXX"}, "prevExist": {"false"}}),
			http.StatusPreconditionFailed,
			map[string]interface{}{
				"errorCode": float64(105),
				"message":   "Key already exists",
				"cause":     "/create/foo",
			},
		},
		// update the newly created key with ttl
		{
			"/v2/keys/create/foo",
			url.Values(map[string][]string{"value": {"YYY"}, "prevExist": {"true"}, "ttl": {"20"}}),
			http.StatusOK,
			map[string]interface{}{
				"node": map[string]interface{}{
					"value": "YYY",
					"ttl":   float64(20),
				},
				"action": "update",
			},
		},
		// update the ttl to none
		{
			"/v2/keys/create/foo",
			url.Values(map[string][]string{"value": {"ZZZ"}, "prevExist": {"true"}}),
			http.StatusOK,
			map[string]interface{}{
				"node": map[string]interface{}{
					"value": "ZZZ",
				},
				"action": "update",
			},
		},
		// update on a non-existing key
		{
			"/v2/keys/nonexist",
			url.Values(map[string][]string{"value": {"XXX"}, "prevExist": {"true"}}),
			http.StatusNotFound,
			map[string]interface{}{
				"errorCode": float64(100),
				"message":   "Key not found",
				"cause":     "/nonexist",
			},
		},
	}

	for i, tt := range tests {
		resp, err := tc.PutForm(fmt.Sprintf("%s%s", u, tt.relativeURL), tt.value)
		if err != nil {
			t.Fatalf("#%d: put err = %v, want nil", i, err)
		}
		if resp.StatusCode != tt.wStatus {
			t.Errorf("#%d: status = %d, want %d", i, resp.StatusCode, tt.wStatus)
		}
		if err := checkBody(tc.ReadBodyJSON(resp), tt.w); err != nil {
			t.Errorf("#%d: %v", i, err)
		}
	}
}

func TestV2CAS(t *testing.T) {
	defer testutil.AfterTest(t)
	cl := NewCluster(t, 1)
	cl.Launch(t)
	defer cl.Terminate(t)

	u := cl.URL(0)
	tc := NewTestClient()

	tests := []struct {
		relativeURL string
		value       url.Values
		wStatus     int
		w           map[string]interface{}
	}{
		{
			"/v2/keys/cas/foo",
			url.Values(map[string][]string{"value": {"XXX"}}),
			http.StatusCreated,
			nil,
		},
		{
			"/v2/keys/cas/foo",
			url.Values(map[string][]string{"value": {"YYY"}, "prevIndex": {"4"}}),
			http.StatusOK,
			map[string]interface{}{
				"node": map[string]interface{}{
					"value":         "YYY",
					"modifiedIndex": float64(5),
				},
				"action": "compareAndSwap",
			},
		},
		{
			"/v2/keys/cas/foo",
			url.Values(map[string][]string{"value": {"YYY"}, "prevIndex": {"10"}}),
			http.StatusPreconditionFailed,
			map[string]interface{}{
				"errorCode": float64(101),
				"message":   "Compare failed",
				"cause":     "[10 != 5]",
				"index":     float64(5),
			},
		},
		{
			"/v2/keys/cas/foo",
			url.Values(map[string][]string{"value": {"YYY"}, "prevIndex": {"bad_index"}}),
			http.StatusBadRequest,
			map[string]interface{}{
				"errorCode": float64(203),
				"message":   "The given index in POST form is not a number",
			},
		},
		{
			"/v2/keys/cas/foo",
			url.Values(map[string][]string{"value": {"ZZZ"}, "prevValue": {"YYY"}}),
			http.StatusOK,
			map[string]interface{}{
				"node": map[string]interface{}{
					"value": "ZZZ",
				},
				"action": "compareAndSwap",
			},
		},
		{
			"/v2/keys/cas/foo",
			url.Values(map[string][]string{"value": {"XXX"}, "prevValue": {"bad_value"}}),
			http.StatusPreconditionFailed,
			map[string]interface{}{
				"errorCode": float64(101),
				"message":   "Compare failed",
				"cause":     "[bad_value != ZZZ]",
			},
		},
		// prevValue is required
		{
			"/v2/keys/cas/foo",
			url.Values(map[string][]string{"value": {"XXX"}, "prevValue": {""}}),
			http.StatusBadRequest,
			map[string]interface{}{
				"errorCode": float64(201),
			},
		},
		{
			"/v2/keys/cas/foo",
			url.Values(map[string][]string{"value": {"XXX"}, "prevValue": {"bad_value"}, "prevIndex": {"100"}}),
			http.StatusPreconditionFailed,
			map[string]interface{}{
				"errorCode": float64(101),
				"message":   "Compare failed",
				"cause":     "[bad_value != ZZZ] [100 != 6]",
			},
		},
		{
			"/v2/keys/cas/foo",
			url.Values(map[string][]string{"value": {"XXX"}, "prevValue": {"ZZZ"}, "prevIndex": {"100"}}),
			http.StatusPreconditionFailed,
			map[string]interface{}{
				"errorCode": float64(101),
				"message":   "Compare failed",
				"cause":     "[100 != 6]",
			},
		},
		{
			"/v2/keys/cas/foo",
			url.Values(map[string][]string{"value": {"XXX"}, "prevValue": {"bad_value"}, "prevIndex": {"6"}}),
			http.StatusPreconditionFailed,
			map[string]interface{}{
				"errorCode": float64(101),
				"message":   "Compare failed",
				"cause":     "[bad_value != ZZZ]",
			},
		},
	}

	for i, tt := range tests {
		resp, err := tc.PutForm(fmt.Sprintf("%s%s", u, tt.relativeURL), tt.value)
		if err != nil {
			t.Fatalf("#%d: put err = %v, want nil", i, err)
		}
		if resp.StatusCode != tt.wStatus {
			t.Errorf("#%d: status = %d, want %d", i, resp.StatusCode, tt.wStatus)
		}
		if err := checkBody(tc.ReadBodyJSON(resp), tt.w); err != nil {
			t.Errorf("#%d: %v", i, err)
		}
	}
}

func TestV2Delete(t *testing.T) {
	defer testutil.AfterTest(t)
	cl := NewCluster(t, 1)
	cl.Launch(t)
	defer cl.Terminate(t)

	u := cl.URL(0)
	tc := NewTestClient()

	v := url.Values{}
	v.Set("value", "XXX")
	r, err := tc.PutForm(fmt.Sprintf("%s%s", u, "/v2/keys/foo"), v)
	if err != nil {
		t.Error(err)
	}
	r.Body.Close()
	r, err = tc.PutForm(fmt.Sprintf("%s%s", u, "/v2/keys/emptydir?dir=true"), v)
	if err != nil {
		t.Error(err)
	}
	r.Body.Close()
	r, err = tc.PutForm(fmt.Sprintf("%s%s", u, "/v2/keys/foodir/bar?dir=true"), v)
	if err != nil {
		t.Error(err)
	}
	r.Body.Close()

	tests := []struct {
		relativeURL string
		wStatus     int
		w           map[string]interface{}
	}{
		{
			"/v2/keys/foo",
			http.StatusOK,
			map[string]interface{}{
				"node": map[string]interface{}{
					"key": "/foo",
				},
				"prevNode": map[string]interface{}{
					"key":   "/foo",
					"value": "XXX",
				},
				"action": "delete",
			},
		},
		{
			"/v2/keys/emptydir",
			http.StatusForbidden,
			map[string]interface{}{
				"errorCode": float64(102),
				"message":   "Not a file",
				"cause":     "/emptydir",
			},
		},
		{
			"/v2/keys/emptydir?dir=true",
			http.StatusOK,
			nil,
		},
		{
			"/v2/keys/foodir?dir=true",
			http.StatusForbidden,
			map[string]interface{}{
				"errorCode": float64(108),
				"message":   "Directory not empty",
				"cause":     "/foodir",
			},
		},
		{
			"/v2/keys/foodir?recursive=true",
			http.StatusOK,
			map[string]interface{}{
				"node": map[string]interface{}{
					"key": "/foodir",
					"dir": true,
				},
				"prevNode": map[string]interface{}{
					"key": "/foodir",
					"dir": true,
				},
				"action": "delete",
			},
		},
	}

	for i, tt := range tests {
		resp, err := tc.DeleteForm(fmt.Sprintf("%s%s", u, tt.relativeURL), nil)
		if err != nil {
			t.Fatalf("#%d: delete err = %v, want nil", i, err)
		}
		if resp.StatusCode != tt.wStatus {
			t.Errorf("#%d: status = %d, want %d", i, resp.StatusCode, tt.wStatus)
		}
		if err := checkBody(tc.ReadBodyJSON(resp), tt.w); err != nil {
			t.Errorf("#%d: %v", i, err)
		}
	}
}

func TestV2CAD(t *testing.T) {
	defer testutil.AfterTest(t)
	cl := NewCluster(t, 1)
	cl.Launch(t)
	defer cl.Terminate(t)

	u := cl.URL(0)
	tc := NewTestClient()

	v := url.Values{}
	v.Set("value", "XXX")
	r, err := tc.PutForm(fmt.Sprintf("%s%s", u, "/v2/keys/foo"), v)
	if err != nil {
		t.Error(err)
	}
	r.Body.Close()

	r, err = tc.PutForm(fmt.Sprintf("%s%s", u, "/v2/keys/foovalue"), v)
	if err != nil {
		t.Error(err)
	}
	r.Body.Close()

	tests := []struct {
		relativeURL string
		wStatus     int
		w           map[string]interface{}
	}{
		{
			"/v2/keys/foo?prevIndex=100",
			http.StatusPreconditionFailed,
			map[string]interface{}{
				"errorCode": float64(101),
				"message":   "Compare failed",
				"cause":     "[100 != 4]",
			},
		},
		{
			"/v2/keys/foo?prevIndex=bad_index",
			http.StatusBadRequest,
			map[string]interface{}{
				"errorCode": float64(203),
				"message":   "The given index in POST form is not a number",
			},
		},
		{
			"/v2/keys/foo?prevIndex=4",
			http.StatusOK,
			map[string]interface{}{
				"node": map[string]interface{}{
					"key":           "/foo",
					"modifiedIndex": float64(6),
				},
				"action": "compareAndDelete",
			},
		},
		{
			"/v2/keys/foovalue?prevValue=YYY",
			http.StatusPreconditionFailed,
			map[string]interface{}{
				"errorCode": float64(101),
				"message":   "Compare failed",
				"cause":     "[YYY != XXX]",
			},
		},
		{
			"/v2/keys/foovalue?prevValue=",
			http.StatusBadRequest,
			map[string]interface{}{
				"errorCode": float64(201),
				"cause":     `"prevValue" cannot be empty`,
			},
		},
		{
			"/v2/keys/foovalue?prevValue=XXX",
			http.StatusOK,
			map[string]interface{}{
				"node": map[string]interface{}{
					"key":           "/foovalue",
					"modifiedIndex": float64(7),
				},
				"action": "compareAndDelete",
			},
		},
	}

	for i, tt := range tests {
		resp, err := tc.DeleteForm(fmt.Sprintf("%s%s", u, tt.relativeURL), nil)
		if err != nil {
			t.Fatalf("#%d: delete err = %v, want nil", i, err)
		}
		if resp.StatusCode != tt.wStatus {
			t.Errorf("#%d: status = %d, want %d", i, resp.StatusCode, tt.wStatus)
		}
		if err := checkBody(tc.ReadBodyJSON(resp), tt.w); err != nil {
			t.Errorf("#%d: %v", i, err)
		}
	}
}

func TestV2Unique(t *testing.T) {
	defer testutil.AfterTest(t)
	cl := NewCluster(t, 1)
	cl.Launch(t)
	defer cl.Terminate(t)

	u := cl.URL(0)
	tc := NewTestClient()

	tests := []struct {
		relativeURL string
		value       url.Values
		wStatus     int
		w           map[string]interface{}
	}{
		{
			"/v2/keys/foo",
			url.Values(map[string][]string{"value": {"XXX"}}),
			http.StatusCreated,
			map[string]interface{}{
				"node": map[string]interface{}{
					"key":   "/foo/00000000000000000004",
					"value": "XXX",
				},
				"action": "create",
			},
		},
		{
			"/v2/keys/foo",
			url.Values(map[string][]string{"value": {"XXX"}}),
			http.StatusCreated,
			map[string]interface{}{
				"node": map[string]interface{}{
					"key":   "/foo/00000000000000000005",
					"value": "XXX",
				},
				"action": "create",
			},
		},
		{
			"/v2/keys/bar",
			url.Values(map[string][]string{"value": {"XXX"}}),
			http.StatusCreated,
			map[string]interface{}{
				"node": map[string]interface{}{
					"key":   "/bar/00000000000000000006",
					"value": "XXX",
				},
				"action": "create",
			},
		},
	}

	for i, tt := range tests {
		resp, err := tc.PostForm(fmt.Sprintf("%s%s", u, tt.relativeURL), tt.value)
		if err != nil {
			t.Fatalf("#%d: post err = %v, want nil", i, err)
		}
		if resp.StatusCode != tt.wStatus {
			t.Errorf("#%d: status = %d, want %d", i, resp.StatusCode, tt.wStatus)
		}
		if err := checkBody(tc.ReadBodyJSON(resp), tt.w); err != nil {
			t.Errorf("#%d: %v", i, err)
		}
	}
}

func TestV2Get(t *testing.T) {
	defer testutil.AfterTest(t)
	cl := NewCluster(t, 1)
	cl.Launch(t)
	defer cl.Terminate(t)

	u := cl.URL(0)
	tc := NewTestClient()

	v := url.Values{}
	v.Set("value", "XXX")
	r, err := tc.PutForm(fmt.Sprintf("%s%s", u, "/v2/keys/foo/bar/zar"), v)
	if err != nil {
		t.Error(err)
	}
	r.Body.Close()

	tests := []struct {
		relativeURL string
		wStatus     int
		w           map[string]interface{}
	}{
		{
			"/v2/keys/foo/bar/zar",
			http.StatusOK,
			map[string]interface{}{
				"node": map[string]interface{}{
					"key":   "/foo/bar/zar",
					"value": "XXX",
				},
				"action": "get",
			},
		},
		{
			"/v2/keys/foo",
			http.StatusOK,
			map[string]interface{}{
				"node": map[string]interface{}{
					"key": "/foo",
					"dir": true,
					"nodes": []interface{}{
						map[string]interface{}{
							"key":           "/foo/bar",
							"dir":           true,
							"createdIndex":  float64(4),
							"modifiedIndex": float64(4),
						},
					},
				},
				"action": "get",
			},
		},
		{
			"/v2/keys/foo?recursive=true",
			http.StatusOK,
			map[string]interface{}{
				"node": map[string]interface{}{
					"key": "/foo",
					"dir": true,
					"nodes": []interface{}{
						map[string]interface{}{
							"key":           "/foo/bar",
							"dir":           true,
							"createdIndex":  float64(4),
							"modifiedIndex": float64(4),
							"nodes": []interface{}{
								map[string]interface{}{
									"key":           "/foo/bar/zar",
									"value":         "XXX",
									"createdIndex":  float64(4),
									"modifiedIndex": float64(4),
								},
							},
						},
					},
				},
				"action": "get",
			},
		},
	}

	for i, tt := range tests {
		resp, err := tc.Get(fmt.Sprintf("%s%s", u, tt.relativeURL))
		if err != nil {
			t.Fatalf("#%d: get err = %v, want nil", i, err)
		}
		if resp.StatusCode != tt.wStatus {
			t.Errorf("#%d: status = %d, want %d", i, resp.StatusCode, tt.wStatus)
		}
		if resp.Header.Get("Content-Type") != "application/json" {
			t.Errorf("#%d: header = %v, want %v", i, resp.Header.Get("Content-Type"), "application/json")
		}
		if err := checkBody(tc.ReadBodyJSON(resp), tt.w); err != nil {
			t.Errorf("#%d: %v", i, err)
		}
	}
}

func TestV2QuorumGet(t *testing.T) {
	defer testutil.AfterTest(t)
	cl := NewCluster(t, 1)
	cl.Launch(t)
	defer cl.Terminate(t)

	u := cl.URL(0)
	tc := NewTestClient()

	v := url.Values{}
	v.Set("value", "XXX")
	r, err := tc.PutForm(fmt.Sprintf("%s%s", u, "/v2/keys/foo/bar/zar?quorum=true"), v)
	if err != nil {
		t.Error(err)
	}
	r.Body.Close()

	tests := []struct {
		relativeURL string
		wStatus     int
		w           map[string]interface{}
	}{
		{
			"/v2/keys/foo/bar/zar",
			http.StatusOK,
			map[string]interface{}{
				"node": map[string]interface{}{
					"key":   "/foo/bar/zar",
					"value": "XXX",
				},
				"action": "get",
			},
		},
		{
			"/v2/keys/foo",
			http.StatusOK,
			map[string]interface{}{
				"node": map[string]interface{}{
					"key": "/foo",
					"dir": true,
					"nodes": []interface{}{
						map[string]interface{}{
							"key":           "/foo/bar",
							"dir":           true,
							"createdIndex":  float64(4),
							"modifiedIndex": float64(4),
						},
					},
				},
				"action": "get",
			},
		},
		{
			"/v2/keys/foo?recursive=true",
			http.StatusOK,
			map[string]interface{}{
				"node": map[string]interface{}{
					"key": "/foo",
					"dir": true,
					"nodes": []interface{}{
						map[string]interface{}{
							"key":           "/foo/bar",
							"dir":           true,
							"createdIndex":  float64(4),
							"modifiedIndex": float64(4),
							"nodes": []interface{}{
								map[string]interface{}{
									"key":           "/foo/bar/zar",
									"value":         "XXX",
									"createdIndex":  float64(4),
									"modifiedIndex": float64(4),
								},
							},
						},
					},
				},
				"action": "get",
			},
		},
	}

	for i, tt := range tests {
		resp, err := tc.Get(fmt.Sprintf("%s%s", u, tt.relativeURL))
		if err != nil {
			t.Fatalf("#%d: get err = %v, want nil", i, err)
		}
		if resp.StatusCode != tt.wStatus {
			t.Errorf("#%d: status = %d, want %d", i, resp.StatusCode, tt.wStatus)
		}
		if resp.Header.Get("Content-Type") != "application/json" {
			t.Errorf("#%d: header = %v, want %v", i, resp.Header.Get("Content-Type"), "application/json")
		}
		if err := checkBody(tc.ReadBodyJSON(resp), tt.w); err != nil {
			t.Errorf("#%d: %v", i, err)
		}
	}
}

func TestV2Watch(t *testing.T) {
	defer testutil.AfterTest(t)
	cl := NewCluster(t, 1)
	cl.Launch(t)
	defer cl.Terminate(t)

	u := cl.URL(0)
	tc := NewTestClient()

	watchResp, err := tc.Get(fmt.Sprintf("%s%s", u, "/v2/keys/foo/bar?wait=true"))
	if err != nil {
		t.Fatalf("watch err = %v, want nil", err)
	}

	// Set a value.
	v := url.Values{}
	v.Set("value", "XXX")
	resp, err := tc.PutForm(fmt.Sprintf("%s%s", u, "/v2/keys/foo/bar"), v)
	if err != nil {
		t.Fatalf("put err = %v, want nil", err)
	}
	resp.Body.Close()

	body := tc.ReadBodyJSON(watchResp)
	w := map[string]interface{}{
		"node": map[string]interface{}{
			"key":           "/foo/bar",
			"value":         "XXX",
			"modifiedIndex": float64(4),
		},
		"action": "set",
	}

	if err := checkBody(body, w); err != nil {
		t.Error(err)
	}
}

func TestV2WatchWithIndex(t *testing.T) {
	defer testutil.AfterTest(t)
	cl := NewCluster(t, 1)
	cl.Launch(t)
	defer cl.Terminate(t)

	u := cl.URL(0)
	tc := NewTestClient()

	var body map[string]interface{}
	c := make(chan bool, 1)
	go func() {
		resp, err := tc.Get(fmt.Sprintf("%s%s", u, "/v2/keys/foo/bar?wait=true&waitIndex=5"))
		if err != nil {
			t.Fatalf("watch err = %v, want nil", err)
		}
		body = tc.ReadBodyJSON(resp)
		c <- true
	}()

	select {
	case <-c:
		t.Fatal("should not get the watch result")
	case <-time.After(time.Millisecond):
	}

	// Set a value (before given index).
	v := url.Values{}
	v.Set("value", "XXX")
	resp, err := tc.PutForm(fmt.Sprintf("%s%s", u, "/v2/keys/foo/bar"), v)
	if err != nil {
		t.Fatalf("put err = %v, want nil", err)
	}
	resp.Body.Close()

	select {
	case <-c:
		t.Fatal("should not get the watch result")
	case <-time.After(time.Millisecond):
	}

	// Set a value (before given index).
	resp, err = tc.PutForm(fmt.Sprintf("%s%s", u, "/v2/keys/foo/bar"), v)
	if err != nil {
		t.Fatalf("put err = %v, want nil", err)
	}
	resp.Body.Close()

	select {
	case <-c:
	case <-time.After(time.Second):
		t.Fatal("cannot get watch result")
	}

	w := map[string]interface{}{
		"node": map[string]interface{}{
			"key":           "/foo/bar",
			"value":         "XXX",
			"modifiedIndex": float64(5),
		},
		"action": "set",
	}
	if err := checkBody(body, w); err != nil {
		t.Error(err)
	}
}

func TestV2WatchKeyInDir(t *testing.T) {
	defer testutil.AfterTest(t)
	cl := NewCluster(t, 1)
	cl.Launch(t)
	defer cl.Terminate(t)

	u := cl.URL(0)
	tc := NewTestClient()

	var body map[string]interface{}
	c := make(chan bool)

	// Create an expiring directory
	v := url.Values{}
	v.Set("dir", "true")
	v.Set("ttl", "1")
	resp, err := tc.PutForm(fmt.Sprintf("%s%s", u, "/v2/keys/keyindir"), v)
	if err != nil {
		t.Fatalf("put err = %v, want nil", err)
	}
	resp.Body.Close()

	// Create a permanent node within the directory
	v = url.Values{}
	v.Set("value", "XXX")
	resp, err = tc.PutForm(fmt.Sprintf("%s%s", u, "/v2/keys/keyindir/bar"), v)
	if err != nil {
		t.Fatalf("put err = %v, want nil", err)
	}
	resp.Body.Close()

	go func() {
		// Expect a notification when watching the node
		resp, err := tc.Get(fmt.Sprintf("%s%s", u, "/v2/keys/keyindir/bar?wait=true"))
		if err != nil {
			t.Fatalf("watch err = %v, want nil", err)
		}
		body = tc.ReadBodyJSON(resp)
		c <- true
	}()

	select {
	case <-c:
	// 1s ttl + 0.5s sync delay + 1.5s disk and network delay
	// We set that long disk and network delay because travis may be slow
	// when do system calls.
	case <-time.After(3 * time.Second):
		t.Fatal("timed out waiting for watch result")
	}

	w := map[string]interface{}{
		"node": map[string]interface{}{
			"key": "/keyindir",
		},
		"action": "expire",
	}
	if err := checkBody(body, w); err != nil {
		t.Error(err)
	}
}

func TestV2Head(t *testing.T) {
	defer testutil.AfterTest(t)
	cl := NewCluster(t, 1)
	cl.Launch(t)
	defer cl.Terminate(t)

	u := cl.URL(0)
	tc := NewTestClient()

	v := url.Values{}
	v.Set("value", "XXX")
	fullURL := fmt.Sprintf("%s%s", u, "/v2/keys/foo/bar")
	resp, err := tc.Head(fullURL)
	if err != nil {
		t.Fatalf("head err = %v, want nil", err)
	}
	resp.Body.Close()
	if resp.StatusCode != http.StatusNotFound {
		t.Errorf("status = %d, want %d", resp.StatusCode, http.StatusNotFound)
	}
	if resp.ContentLength <= 0 {
		t.Errorf("ContentLength = %d, want > 0", resp.ContentLength)
	}

	resp, err = tc.PutForm(fullURL, v)
	if err != nil {
		t.Fatalf("put err = %v, want nil", err)
	}
	resp.Body.Close()

	resp, err = tc.Head(fullURL)
	if err != nil {
		t.Fatalf("head err = %v, want nil", err)
	}
	resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Errorf("status = %d, want %d", resp.StatusCode, http.StatusOK)
	}
	if resp.ContentLength <= 0 {
		t.Errorf("ContentLength = %d, want > 0", resp.ContentLength)
	}
}

func checkBody(body map[string]interface{}, w map[string]interface{}) error {
	if body["node"] != nil {
		if w["node"] != nil {
			wn := w["node"].(map[string]interface{})
			n := body["node"].(map[string]interface{})
			for k := range n {
				if wn[k] == nil {
					delete(n, k)
				}
			}
			body["node"] = n
		}
		if w["prevNode"] != nil {
			wn := w["prevNode"].(map[string]interface{})
			n := body["prevNode"].(map[string]interface{})
			for k := range n {
				if wn[k] == nil {
					delete(n, k)
				}
			}
			body["prevNode"] = n
		}
	}
	for k, v := range w {
		g := body[k]
		if !reflect.DeepEqual(g, v) {
			return fmt.Errorf("%v = %+v, want %+v", k, g, v)
		}
	}
	return nil
}

type testHttpClient struct {
	*http.Client
}

// Creates a new HTTP client with KeepAlive disabled.
func NewTestClient() *testHttpClient {
	tr, _ := transport.NewTransport(transport.TLSInfo{}, time.Second)
	tr.DisableKeepAlives = true
	return &testHttpClient{&http.Client{Transport: tr}}
}

// Reads the body from the response and closes it.
func (t *testHttpClient) ReadBody(resp *http.Response) []byte {
	if resp == nil {
		return []byte{}
	}
	body, _ := ioutil.ReadAll(resp.Body)
	resp.Body.Close()
	return body
}

// Reads the body from the response and parses it as JSON.
func (t *testHttpClient) ReadBodyJSON(resp *http.Response) map[string]interface{} {
	m := make(map[string]interface{})
	b := t.ReadBody(resp)
	if err := json.Unmarshal(b, &m); err != nil {
		panic(fmt.Sprintf("HTTP body JSON parse error: %v: %s", err, string(b)))
	}
	return m
}

func (t *testHttpClient) Head(url string) (*http.Response, error) {
	return t.send("HEAD", url, "application/json", nil)
}

func (t *testHttpClient) Get(url string) (*http.Response, error) {
	return t.send("GET", url, "application/json", nil)
}

func (t *testHttpClient) Post(url string, bodyType string, body io.Reader) (*http.Response, error) {
	return t.send("POST", url, bodyType, body)
}

func (t *testHttpClient) PostForm(url string, data url.Values) (*http.Response, error) {
	return t.Post(url, "application/x-www-form-urlencoded", strings.NewReader(data.Encode()))
}

func (t *testHttpClient) Put(url string, bodyType string, body io.Reader) (*http.Response, error) {
	return t.send("PUT", url, bodyType, body)
}

func (t *testHttpClient) PutForm(url string, data url.Values) (*http.Response, error) {
	return t.Put(url, "application/x-www-form-urlencoded", strings.NewReader(data.Encode()))
}

func (t *testHttpClient) Delete(url string, bodyType string, body io.Reader) (*http.Response, error) {
	return t.send("DELETE", url, bodyType, body)
}

func (t *testHttpClient) DeleteForm(url string, data url.Values) (*http.Response, error) {
	return t.Delete(url, "application/x-www-form-urlencoded", strings.NewReader(data.Encode()))
}

func (t *testHttpClient) send(method string, url string, bodyType string, body io.Reader) (*http.Response, error) {
	req, err := http.NewRequest(method, url, body)
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", bodyType)
	return t.Do(req)
}
