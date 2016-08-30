package agent

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/hashicorp/consul/consul/structs"
)

func makeTestACL(t *testing.T, srv *HTTPServer) string {
	body := bytes.NewBuffer(nil)
	enc := json.NewEncoder(body)
	raw := map[string]interface{}{
		"Name":  "User Token",
		"Type":  "client",
		"Rules": "",
	}
	enc.Encode(raw)

	req, err := http.NewRequest("PUT", "/v1/acl/create?token=root", body)
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	resp := httptest.NewRecorder()
	obj, err := srv.ACLCreate(resp, req)
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	aclResp := obj.(aclCreateResponse)
	return aclResp.ID
}

func TestACLUpdate(t *testing.T) {
	httpTest(t, func(srv *HTTPServer) {
		id := makeTestACL(t, srv)

		body := bytes.NewBuffer(nil)
		enc := json.NewEncoder(body)
		raw := map[string]interface{}{
			"ID":    id,
			"Name":  "User Token 2",
			"Type":  "client",
			"Rules": "",
		}
		enc.Encode(raw)

		req, err := http.NewRequest("PUT", "/v1/acl/update?token=root", body)
		if err != nil {
			t.Fatalf("err: %v", err)
		}
		resp := httptest.NewRecorder()
		obj, err := srv.ACLUpdate(resp, req)
		if err != nil {
			t.Fatalf("err: %v", err)
		}
		aclResp := obj.(aclCreateResponse)
		if aclResp.ID != id {
			t.Fatalf("bad: %v", aclResp)
		}
	})
}

func TestACLUpdate_Upsert(t *testing.T) {
	httpTest(t, func(srv *HTTPServer) {
		body := bytes.NewBuffer(nil)
		enc := json.NewEncoder(body)
		raw := map[string]interface{}{
			"ID":    "my-old-id",
			"Name":  "User Token 2",
			"Type":  "client",
			"Rules": "",
		}
		enc.Encode(raw)

		req, err := http.NewRequest("PUT", "/v1/acl/update?token=root", body)
		if err != nil {
			t.Fatalf("err: %v", err)
		}
		resp := httptest.NewRecorder()
		obj, err := srv.ACLUpdate(resp, req)
		if err != nil {
			t.Fatalf("err: %v", err)
		}
		aclResp := obj.(aclCreateResponse)
		if aclResp.ID != "my-old-id" {
			t.Fatalf("bad: %v", aclResp)
		}
	})
}

func TestACLDestroy(t *testing.T) {
	httpTest(t, func(srv *HTTPServer) {
		id := makeTestACL(t, srv)
		req, err := http.NewRequest("PUT", "/v1/acl/destroy/"+id+"?token=root", nil)
		resp := httptest.NewRecorder()
		obj, err := srv.ACLDestroy(resp, req)
		if err != nil {
			t.Fatalf("err: %v", err)
		}
		if resp, ok := obj.(bool); !ok || !resp {
			t.Fatalf("should work")
		}

		req, err = http.NewRequest("GET",
			"/v1/acl/info/"+id, nil)
		resp = httptest.NewRecorder()
		obj, err = srv.ACLGet(resp, req)
		if err != nil {
			t.Fatalf("err: %v", err)
		}
		respObj, ok := obj.(structs.ACLs)
		if !ok {
			t.Fatalf("should work")
		}
		if len(respObj) != 0 {
			t.Fatalf("bad: %v", respObj)
		}
	})
}

func TestACLClone(t *testing.T) {
	httpTest(t, func(srv *HTTPServer) {
		id := makeTestACL(t, srv)

		req, err := http.NewRequest("PUT",
			"/v1/acl/clone/"+id+"?token=root", nil)
		resp := httptest.NewRecorder()
		obj, err := srv.ACLClone(resp, req)
		if err != nil {
			t.Fatalf("err: %v", err)
		}
		aclResp, ok := obj.(aclCreateResponse)
		if !ok {
			t.Fatalf("should work: %#v %#v", obj, resp)
		}
		if aclResp.ID == id {
			t.Fatalf("bad id")
		}

		req, err = http.NewRequest("GET",
			"/v1/acl/info/"+aclResp.ID, nil)
		resp = httptest.NewRecorder()
		obj, err = srv.ACLGet(resp, req)
		if err != nil {
			t.Fatalf("err: %v", err)
		}
		respObj, ok := obj.(structs.ACLs)
		if !ok {
			t.Fatalf("should work")
		}
		if len(respObj) != 1 {
			t.Fatalf("bad: %v", respObj)
		}
	})
}

func TestACLGet(t *testing.T) {
	httpTest(t, func(srv *HTTPServer) {
		req, err := http.NewRequest("GET", "/v1/acl/info/nope", nil)
		resp := httptest.NewRecorder()
		obj, err := srv.ACLGet(resp, req)
		if err != nil {
			t.Fatalf("err: %v", err)
		}
		respObj, ok := obj.(structs.ACLs)
		if !ok {
			t.Fatalf("should work")
		}
		if respObj == nil || len(respObj) != 0 {
			t.Fatalf("bad: %v", respObj)
		}
	})

	httpTest(t, func(srv *HTTPServer) {
		id := makeTestACL(t, srv)

		req, err := http.NewRequest("GET",
			"/v1/acl/info/"+id, nil)
		resp := httptest.NewRecorder()
		obj, err := srv.ACLGet(resp, req)
		if err != nil {
			t.Fatalf("err: %v", err)
		}
		respObj, ok := obj.(structs.ACLs)
		if !ok {
			t.Fatalf("should work")
		}
		if len(respObj) != 1 {
			t.Fatalf("bad: %v", respObj)
		}
	})
}

func TestACLList(t *testing.T) {
	httpTest(t, func(srv *HTTPServer) {
		var ids []string
		for i := 0; i < 10; i++ {
			ids = append(ids, makeTestACL(t, srv))
		}

		req, err := http.NewRequest("GET", "/v1/acl/list?token=root", nil)
		resp := httptest.NewRecorder()
		obj, err := srv.ACLList(resp, req)
		if err != nil {
			t.Fatalf("err: %v", err)
		}
		respObj, ok := obj.(structs.ACLs)
		if !ok {
			t.Fatalf("should work")
		}

		// 10 + anonymous + master
		if len(respObj) != 12 {
			t.Fatalf("bad: %v", respObj)
		}
	})
}
