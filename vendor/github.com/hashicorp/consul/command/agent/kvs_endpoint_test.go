package agent

import (
	"bytes"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"reflect"
	"testing"

	"github.com/hashicorp/consul/consul/structs"
	"github.com/hashicorp/consul/testutil"
)

func TestKVSEndpoint_PUT_GET_DELETE(t *testing.T) {
	dir, srv := makeHTTPServer(t)
	defer os.RemoveAll(dir)
	defer srv.Shutdown()
	defer srv.agent.Shutdown()

	testutil.WaitForLeader(t, srv.agent.RPC, "dc1")

	keys := []string{
		"baz",
		"bar",
		"foo/sub1",
		"foo/sub2",
		"zip",
	}

	for _, key := range keys {
		buf := bytes.NewBuffer([]byte("test"))
		req, err := http.NewRequest("PUT", "/v1/kv/"+key, buf)
		if err != nil {
			t.Fatalf("err: %v", err)
		}

		resp := httptest.NewRecorder()
		obj, err := srv.KVSEndpoint(resp, req)
		if err != nil {
			t.Fatalf("err: %v", err)
		}

		if res := obj.(bool); !res {
			t.Fatalf("should work")
		}
	}

	for _, key := range keys {
		req, err := http.NewRequest("GET", "/v1/kv/"+key, nil)
		if err != nil {
			t.Fatalf("err: %v", err)
		}

		resp := httptest.NewRecorder()
		obj, err := srv.KVSEndpoint(resp, req)
		if err != nil {
			t.Fatalf("err: %v", err)
		}
		assertIndex(t, resp)

		res, ok := obj.(structs.DirEntries)
		if !ok {
			t.Fatalf("should work")
		}

		if len(res) != 1 {
			t.Fatalf("bad: %v", res)
		}

		if res[0].Key != key {
			t.Fatalf("bad: %v", res)
		}
	}

	for _, key := range keys {
		req, err := http.NewRequest("DELETE", "/v1/kv/"+key, nil)
		if err != nil {
			t.Fatalf("err: %v", err)
		}

		resp := httptest.NewRecorder()
		_, err = srv.KVSEndpoint(resp, req)
		if err != nil {
			t.Fatalf("err: %v", err)
		}
	}
}

func TestKVSEndpoint_Recurse(t *testing.T) {
	dir, srv := makeHTTPServer(t)
	defer os.RemoveAll(dir)
	defer srv.Shutdown()
	defer srv.agent.Shutdown()

	testutil.WaitForLeader(t, srv.agent.RPC, "dc1")

	keys := []string{
		"bar",
		"baz",
		"foo/sub1",
		"foo/sub2",
		"zip",
	}

	for _, key := range keys {
		buf := bytes.NewBuffer([]byte("test"))
		req, err := http.NewRequest("PUT", "/v1/kv/"+key, buf)
		if err != nil {
			t.Fatalf("err: %v", err)
		}

		resp := httptest.NewRecorder()
		obj, err := srv.KVSEndpoint(resp, req)
		if err != nil {
			t.Fatalf("err: %v", err)
		}

		if res := obj.(bool); !res {
			t.Fatalf("should work")
		}
	}

	{
		// Get all the keys
		req, err := http.NewRequest("GET", "/v1/kv/?recurse", nil)
		if err != nil {
			t.Fatalf("err: %v", err)
		}

		resp := httptest.NewRecorder()
		obj, err := srv.KVSEndpoint(resp, req)
		if err != nil {
			t.Fatalf("err: %v", err)
		}
		assertIndex(t, resp)

		res, ok := obj.(structs.DirEntries)
		if !ok {
			t.Fatalf("should work")
		}

		if len(res) != len(keys) {
			t.Fatalf("bad: %v", res)
		}

		for idx, key := range keys {
			if res[idx].Key != key {
				t.Fatalf("bad: %v %v", res[idx].Key, key)
			}
		}
	}

	{
		req, err := http.NewRequest("DELETE", "/v1/kv/?recurse", nil)
		if err != nil {
			t.Fatalf("err: %v", err)
		}

		resp := httptest.NewRecorder()
		_, err = srv.KVSEndpoint(resp, req)
		if err != nil {
			t.Fatalf("err: %v", err)
		}
	}

	{
		// Get all the keys
		req, err := http.NewRequest("GET", "/v1/kv/?recurse", nil)
		if err != nil {
			t.Fatalf("err: %v", err)
		}

		resp := httptest.NewRecorder()
		obj, err := srv.KVSEndpoint(resp, req)
		if err != nil {
			t.Fatalf("err: %v", err)
		}

		if obj != nil {
			t.Fatalf("bad: %v", obj)
		}
	}
}

func TestKVSEndpoint_DELETE_CAS(t *testing.T) {
	dir, srv := makeHTTPServer(t)
	defer os.RemoveAll(dir)
	defer srv.Shutdown()
	defer srv.agent.Shutdown()

	testutil.WaitForLeader(t, srv.agent.RPC, "dc1")

	{
		buf := bytes.NewBuffer([]byte("test"))
		req, err := http.NewRequest("PUT", "/v1/kv/test", buf)
		if err != nil {
			t.Fatalf("err: %v", err)
		}

		resp := httptest.NewRecorder()
		obj, err := srv.KVSEndpoint(resp, req)
		if err != nil {
			t.Fatalf("err: %v", err)
		}

		if res := obj.(bool); !res {
			t.Fatalf("should work")
		}
	}

	req, err := http.NewRequest("GET", "/v1/kv/test", nil)
	if err != nil {
		t.Fatalf("err: %v", err)
	}

	resp := httptest.NewRecorder()
	obj, err := srv.KVSEndpoint(resp, req)
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	d := obj.(structs.DirEntries)[0]

	// Create a CAS request, bad index
	{
		buf := bytes.NewBuffer([]byte("zip"))
		req, err := http.NewRequest("DELETE",
			fmt.Sprintf("/v1/kv/test?cas=%d", d.ModifyIndex-1), buf)
		if err != nil {
			t.Fatalf("err: %v", err)
		}

		resp := httptest.NewRecorder()
		obj, err := srv.KVSEndpoint(resp, req)
		if err != nil {
			t.Fatalf("err: %v", err)
		}

		if res := obj.(bool); res {
			t.Fatalf("should NOT work")
		}
	}

	// Create a CAS request, good index
	{
		buf := bytes.NewBuffer([]byte("zip"))
		req, err := http.NewRequest("DELETE",
			fmt.Sprintf("/v1/kv/test?cas=%d", d.ModifyIndex), buf)
		if err != nil {
			t.Fatalf("err: %v", err)
		}

		resp := httptest.NewRecorder()
		obj, err := srv.KVSEndpoint(resp, req)
		if err != nil {
			t.Fatalf("err: %v", err)
		}

		if res := obj.(bool); !res {
			t.Fatalf("should work")
		}
	}

	// Verify the delete
	req, _ = http.NewRequest("GET", "/v1/kv/test", nil)
	resp = httptest.NewRecorder()
	obj, _ = srv.KVSEndpoint(resp, req)
	if obj != nil {
		t.Fatalf("should be destroyed")
	}
}

func TestKVSEndpoint_CAS(t *testing.T) {
	dir, srv := makeHTTPServer(t)
	defer os.RemoveAll(dir)
	defer srv.Shutdown()
	defer srv.agent.Shutdown()

	testutil.WaitForLeader(t, srv.agent.RPC, "dc1")

	{
		buf := bytes.NewBuffer([]byte("test"))
		req, err := http.NewRequest("PUT", "/v1/kv/test?flags=50", buf)
		if err != nil {
			t.Fatalf("err: %v", err)
		}

		resp := httptest.NewRecorder()
		obj, err := srv.KVSEndpoint(resp, req)
		if err != nil {
			t.Fatalf("err: %v", err)
		}

		if res := obj.(bool); !res {
			t.Fatalf("should work")
		}
	}

	req, err := http.NewRequest("GET", "/v1/kv/test", nil)
	if err != nil {
		t.Fatalf("err: %v", err)
	}

	resp := httptest.NewRecorder()
	obj, err := srv.KVSEndpoint(resp, req)
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	d := obj.(structs.DirEntries)[0]

	// Check the flags
	if d.Flags != 50 {
		t.Fatalf("bad: %v", d)
	}

	// Create a CAS request, bad index
	{
		buf := bytes.NewBuffer([]byte("zip"))
		req, err := http.NewRequest("PUT",
			fmt.Sprintf("/v1/kv/test?flags=42&cas=%d", d.ModifyIndex-1), buf)
		if err != nil {
			t.Fatalf("err: %v", err)
		}

		resp := httptest.NewRecorder()
		obj, err := srv.KVSEndpoint(resp, req)
		if err != nil {
			t.Fatalf("err: %v", err)
		}

		if res := obj.(bool); res {
			t.Fatalf("should NOT work")
		}
	}

	// Create a CAS request, good index
	{
		buf := bytes.NewBuffer([]byte("zip"))
		req, err := http.NewRequest("PUT",
			fmt.Sprintf("/v1/kv/test?flags=42&cas=%d", d.ModifyIndex), buf)
		if err != nil {
			t.Fatalf("err: %v", err)
		}

		resp := httptest.NewRecorder()
		obj, err := srv.KVSEndpoint(resp, req)
		if err != nil {
			t.Fatalf("err: %v", err)
		}

		if res := obj.(bool); !res {
			t.Fatalf("should work")
		}
	}

	// Verify the update
	req, _ = http.NewRequest("GET", "/v1/kv/test", nil)
	resp = httptest.NewRecorder()
	obj, _ = srv.KVSEndpoint(resp, req)
	d = obj.(structs.DirEntries)[0]

	if d.Flags != 42 {
		t.Fatalf("bad: %v", d)
	}
	if string(d.Value) != "zip" {
		t.Fatalf("bad: %v", d)
	}
}

func TestKVSEndpoint_ListKeys(t *testing.T) {
	dir, srv := makeHTTPServer(t)
	defer os.RemoveAll(dir)
	defer srv.Shutdown()
	defer srv.agent.Shutdown()

	testutil.WaitForLeader(t, srv.agent.RPC, "dc1")

	keys := []string{
		"bar",
		"baz",
		"foo/sub1",
		"foo/sub2",
		"zip",
	}

	for _, key := range keys {
		buf := bytes.NewBuffer([]byte("test"))
		req, err := http.NewRequest("PUT", "/v1/kv/"+key, buf)
		if err != nil {
			t.Fatalf("err: %v", err)
		}

		resp := httptest.NewRecorder()
		obj, err := srv.KVSEndpoint(resp, req)
		if err != nil {
			t.Fatalf("err: %v", err)
		}

		if res := obj.(bool); !res {
			t.Fatalf("should work")
		}
	}

	{
		// Get all the keys
		req, err := http.NewRequest("GET", "/v1/kv/?keys&seperator=/", nil)
		if err != nil {
			t.Fatalf("err: %v", err)
		}

		resp := httptest.NewRecorder()
		obj, err := srv.KVSEndpoint(resp, req)
		if err != nil {
			t.Fatalf("err: %v", err)
		}
		assertIndex(t, resp)

		res, ok := obj.([]string)
		if !ok {
			t.Fatalf("should work")
		}

		expect := []string{"bar", "baz", "foo/", "zip"}
		if !reflect.DeepEqual(res, expect) {
			t.Fatalf("bad: %v", res)
		}
	}
}

func TestKVSEndpoint_AcquireRelease(t *testing.T) {
	httpTest(t, func(srv *HTTPServer) {
		// Acquire the lock
		id := makeTestSession(t, srv)
		req, err := http.NewRequest("PUT",
			"/v1/kv/test?acquire="+id, bytes.NewReader(nil))
		if err != nil {
			t.Fatalf("err: %v", err)
		}

		resp := httptest.NewRecorder()
		obj, err := srv.KVSEndpoint(resp, req)
		if err != nil {
			t.Fatalf("err: %v", err)
		}
		if res := obj.(bool); !res {
			t.Fatalf("should work")
		}

		// Verify we have the lock
		req, err = http.NewRequest("GET", "/v1/kv/test", nil)
		if err != nil {
			t.Fatalf("err: %v", err)
		}
		resp = httptest.NewRecorder()
		obj, err = srv.KVSEndpoint(resp, req)
		if err != nil {
			t.Fatalf("err: %v", err)
		}
		d := obj.(structs.DirEntries)[0]

		// Check the flags
		if d.Session != id {
			t.Fatalf("bad: %v", d)
		}

		// Release the lock
		req, err = http.NewRequest("PUT",
			"/v1/kv/test?release="+id, bytes.NewReader(nil))
		if err != nil {
			t.Fatalf("err: %v", err)
		}
		resp = httptest.NewRecorder()
		obj, err = srv.KVSEndpoint(resp, req)
		if err != nil {
			t.Fatalf("err: %v", err)
		}
		if res := obj.(bool); !res {
			t.Fatalf("should work")
		}

		// Verify we do not have the lock
		req, err = http.NewRequest("GET", "/v1/kv/test", nil)
		if err != nil {
			t.Fatalf("err: %v", err)
		}
		resp = httptest.NewRecorder()
		obj, err = srv.KVSEndpoint(resp, req)
		if err != nil {
			t.Fatalf("err: %v", err)
		}
		d = obj.(structs.DirEntries)[0]

		// Check the flags
		if d.Session != "" {
			t.Fatalf("bad: %v", d)
		}
	})
}

func TestKVSEndpoint_GET_Raw(t *testing.T) {
	httpTest(t, func(srv *HTTPServer) {
		buf := bytes.NewBuffer([]byte("test"))
		req, err := http.NewRequest("PUT", "/v1/kv/test", buf)
		if err != nil {
			t.Fatalf("err: %v", err)
		}
		resp := httptest.NewRecorder()
		obj, err := srv.KVSEndpoint(resp, req)
		if err != nil {
			t.Fatalf("err: %v", err)
		}
		if res := obj.(bool); !res {
			t.Fatalf("should work")
		}

		req, err = http.NewRequest("GET", "/v1/kv/test?raw", nil)
		if err != nil {
			t.Fatalf("err: %v", err)
		}
		resp = httptest.NewRecorder()
		obj, err = srv.KVSEndpoint(resp, req)
		if err != nil {
			t.Fatalf("err: %v", err)
		}
		assertIndex(t, resp)

		// Check the body
		if !bytes.Equal(resp.Body.Bytes(), []byte("test")) {
			t.Fatalf("bad: %s", resp.Body.Bytes())
		}
	})
}

func TestKVSEndpoint_PUT_ConflictingFlags(t *testing.T) {
	httpTest(t, func(srv *HTTPServer) {
		req, err := http.NewRequest("PUT", "/v1/kv/test?cas=0&acquire=xxx", nil)
		if err != nil {
			t.Fatalf("err: %v", err)
		}

		resp := httptest.NewRecorder()
		if _, err := srv.KVSEndpoint(resp, req); err != nil {
			t.Fatalf("err: %v", err)
		}

		if resp.Code != 400 {
			t.Fatalf("expected 400, got %d", resp.Code)
		}
		if !bytes.Contains(resp.Body.Bytes(), []byte("Conflicting")) {
			t.Fatalf("expected conflicting args error")
		}
	})
}

func TestKVSEndpoint_DELETE_ConflictingFlags(t *testing.T) {
	httpTest(t, func(srv *HTTPServer) {
		req, err := http.NewRequest("DELETE", "/v1/kv/test?recurse&cas=0", nil)
		if err != nil {
			t.Fatalf("err: %v", err)
		}

		resp := httptest.NewRecorder()
		if _, err := srv.KVSEndpoint(resp, req); err != nil {
			t.Fatalf("err: %v", err)
		}

		if resp.Code != 400 {
			t.Fatalf("expected 400, got %d", resp.Code)
		}
		if !bytes.Contains(resp.Body.Bytes(), []byte("Conflicting")) {
			t.Fatalf("expected conflicting args error")
		}
	})
}
