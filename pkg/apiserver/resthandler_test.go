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

package apiserver

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
)

func TestFinishReq(t *testing.T) {
	handler := &RESTHandler{codec: api.Codec}
	op := &Operation{finished: &time.Time{}, result: RESTResult{Object: &api.Status{Code: http.StatusNotFound}}}
	resp := httptest.NewRecorder()
	handler.finishReq(op, nil, resp)
	status := &api.Status{}
	if err := json.Unmarshal([]byte(resp.Body.String()), status); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if resp.Code != http.StatusNotFound || status.Code != http.StatusNotFound {
		t.Errorf("unexpected status: %#v", status)
	}
}

func TestFinishReqUnwrap(t *testing.T) {
	handler := &RESTHandler{codec: api.Codec}
	op := &Operation{finished: &time.Time{}, result: RESTResult{Created: true, Object: &api.Pod{ObjectMeta: api.ObjectMeta{Name: "foo"}}}}
	resp := httptest.NewRecorder()
	handler.finishReq(op, nil, resp)
	obj := &api.Pod{}
	if err := json.Unmarshal([]byte(resp.Body.String()), obj); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if resp.Code != http.StatusCreated || obj.Name != "foo" {
		t.Errorf("unexpected object: %#v", obj)
	}
}

func TestFinishReqUnwrapStatus(t *testing.T) {
	handler := &RESTHandler{codec: api.Codec}
	op := &Operation{finished: &time.Time{}, result: RESTResult{Created: true, Object: &api.Status{Code: http.StatusNotFound}}}
	resp := httptest.NewRecorder()
	handler.finishReq(op, nil, resp)
	obj := &api.Status{}
	if err := json.Unmarshal([]byte(resp.Body.String()), obj); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if resp.Code != http.StatusNotFound || obj.Code != http.StatusNotFound {
		t.Errorf("unexpected object: %#v", obj)
	}
}
