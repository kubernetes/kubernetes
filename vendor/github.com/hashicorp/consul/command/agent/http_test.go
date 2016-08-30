package agent

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"io/ioutil"
	"log"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/hashicorp/consul/consul/structs"
	"github.com/hashicorp/consul/testutil"
	"github.com/hashicorp/go-cleanhttp"
)

func makeHTTPServer(t *testing.T) (string, *HTTPServer) {
	return makeHTTPServerWithConfig(t, nil)
}

func makeHTTPServerWithConfig(t *testing.T, cb func(c *Config)) (string, *HTTPServer) {
	conf := nextConfig()
	if cb != nil {
		cb(conf)
	}

	dir, agent := makeAgent(t, conf)
	servers, err := NewHTTPServers(agent, conf, agent.logOutput)
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	if len(servers) == 0 {
		t.Fatalf(fmt.Sprintf("Failed to make HTTP server"))
	}
	return dir, servers[0]
}

func encodeReq(obj interface{}) io.ReadCloser {
	buf := bytes.NewBuffer(nil)
	enc := json.NewEncoder(buf)
	enc.Encode(obj)
	return ioutil.NopCloser(buf)
}

func TestHTTPServer_UnixSocket(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.SkipNow()
	}

	tempDir, err := ioutil.TempDir("", "consul")
	if err != nil {
		t.Fatalf("err: %s", err)
	}
	defer os.RemoveAll(tempDir)
	socket := filepath.Join(tempDir, "test.sock")

	dir, srv := makeHTTPServerWithConfig(t, func(c *Config) {
		c.Addresses.HTTP = "unix://" + socket

		// Only testing mode, since uid/gid might not be settable
		// from test environment.
		c.UnixSockets = UnixSocketConfig{}
		c.UnixSockets.Perms = "0777"
	})
	defer os.RemoveAll(dir)
	defer srv.Shutdown()
	defer srv.agent.Shutdown()

	// Ensure the socket was created
	if _, err := os.Stat(socket); err != nil {
		t.Fatalf("err: %s", err)
	}

	// Ensure the mode was set properly
	fi, err := os.Stat(socket)
	if err != nil {
		t.Fatalf("err: %s", err)
	}
	if fi.Mode().String() != "Srwxrwxrwx" {
		t.Fatalf("bad permissions: %s", fi.Mode())
	}

	// Ensure we can get a response from the socket.
	path, _ := unixSocketAddr(srv.agent.config.Addresses.HTTP)
	trans := cleanhttp.DefaultTransport()
	trans.Dial = func(_, _ string) (net.Conn, error) {
		return net.Dial("unix", path)
	}
	client := &http.Client{
		Transport: trans,
	}

	// This URL doesn't look like it makes sense, but the scheme (http://) and
	// the host (127.0.0.1) are required by the HTTP client library. In reality
	// this will just use the custom dialer and talk to the socket.
	resp, err := client.Get("http://127.0.0.1/v1/agent/self")
	if err != nil {
		t.Fatalf("err: %s", err)
	}
	defer resp.Body.Close()

	if body, err := ioutil.ReadAll(resp.Body); err != nil || len(body) == 0 {
		t.Fatalf("bad: %s %v", body, err)
	}
}

func TestHTTPServer_UnixSocket_FileExists(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.SkipNow()
	}

	tempDir, err := ioutil.TempDir("", "consul")
	if err != nil {
		t.Fatalf("err: %s", err)
	}
	defer os.RemoveAll(tempDir)
	socket := filepath.Join(tempDir, "test.sock")

	// Create a regular file at the socket path
	if err := ioutil.WriteFile(socket, []byte("hello world"), 0644); err != nil {
		t.Fatalf("err: %s", err)
	}
	fi, err := os.Stat(socket)
	if err != nil {
		t.Fatalf("err: %s", err)
	}
	if !fi.Mode().IsRegular() {
		t.Fatalf("not a regular file: %s", socket)
	}

	conf := nextConfig()
	conf.Addresses.HTTP = "unix://" + socket

	dir, agent := makeAgent(t, conf)
	defer os.RemoveAll(dir)

	// Try to start the server with the same path anyways.
	if _, err := NewHTTPServers(agent, conf, agent.logOutput); err != nil {
		t.Fatalf("err: %s", err)
	}

	// Ensure the file was replaced by the socket
	fi, err = os.Stat(socket)
	if err != nil {
		t.Fatalf("err: %s", err)
	}
	if fi.Mode()&os.ModeSocket == 0 {
		t.Fatalf("expected socket to replace file")
	}
}

func TestSetIndex(t *testing.T) {
	resp := httptest.NewRecorder()
	setIndex(resp, 1000)
	header := resp.Header().Get("X-Consul-Index")
	if header != "1000" {
		t.Fatalf("Bad: %v", header)
	}
	setIndex(resp, 2000)
	if v := resp.Header()["X-Consul-Index"]; len(v) != 1 {
		t.Fatalf("bad: %#v", v)
	}
}

func TestSetKnownLeader(t *testing.T) {
	resp := httptest.NewRecorder()
	setKnownLeader(resp, true)
	header := resp.Header().Get("X-Consul-KnownLeader")
	if header != "true" {
		t.Fatalf("Bad: %v", header)
	}
	resp = httptest.NewRecorder()
	setKnownLeader(resp, false)
	header = resp.Header().Get("X-Consul-KnownLeader")
	if header != "false" {
		t.Fatalf("Bad: %v", header)
	}
}

func TestSetLastContact(t *testing.T) {
	resp := httptest.NewRecorder()
	setLastContact(resp, 123456*time.Microsecond)
	header := resp.Header().Get("X-Consul-LastContact")
	if header != "123" {
		t.Fatalf("Bad: %v", header)
	}
}

func TestSetMeta(t *testing.T) {
	meta := structs.QueryMeta{
		Index:       1000,
		KnownLeader: true,
		LastContact: 123456 * time.Microsecond,
	}
	resp := httptest.NewRecorder()
	setMeta(resp, &meta)
	header := resp.Header().Get("X-Consul-Index")
	if header != "1000" {
		t.Fatalf("Bad: %v", header)
	}
	header = resp.Header().Get("X-Consul-KnownLeader")
	if header != "true" {
		t.Fatalf("Bad: %v", header)
	}
	header = resp.Header().Get("X-Consul-LastContact")
	if header != "123" {
		t.Fatalf("Bad: %v", header)
	}
}

func TestHTTPAPIResponseHeaders(t *testing.T) {
	dir, srv := makeHTTPServer(t)
	srv.agent.config.HTTPAPIResponseHeaders = map[string]string{
		"Access-Control-Allow-Origin": "*",
		"X-XSS-Protection":            "1; mode=block",
	}

	defer os.RemoveAll(dir)
	defer srv.Shutdown()
	defer srv.agent.Shutdown()

	resp := httptest.NewRecorder()

	handler := func(resp http.ResponseWriter, req *http.Request) (interface{}, error) {
		return nil, nil
	}

	req, _ := http.NewRequest("GET", "/v1/agent/self", nil)
	srv.wrap(handler)(resp, req)

	origin := resp.Header().Get("Access-Control-Allow-Origin")
	if origin != "*" {
		t.Fatalf("bad Access-Control-Allow-Origin: expected %q, got %q", "*", origin)
	}

	xss := resp.Header().Get("X-XSS-Protection")
	if xss != "1; mode=block" {
		t.Fatalf("bad X-XSS-Protection header: expected %q, got %q", "1; mode=block", xss)
	}
}

func TestContentTypeIsJSON(t *testing.T) {
	dir, srv := makeHTTPServer(t)

	defer os.RemoveAll(dir)
	defer srv.Shutdown()
	defer srv.agent.Shutdown()

	resp := httptest.NewRecorder()

	handler := func(resp http.ResponseWriter, req *http.Request) (interface{}, error) {
		// stub out a DirEntry so that it will be encoded as JSON
		return &structs.DirEntry{Key: "key"}, nil
	}

	req, _ := http.NewRequest("GET", "/v1/kv/key", nil)
	srv.wrap(handler)(resp, req)

	contentType := resp.Header().Get("Content-Type")

	if contentType != "application/json" {
		t.Fatalf("Content-Type header was not 'application/json'")
	}
}

func TestHTTP_wrap_obfuscateLog(t *testing.T) {
	dir, srv := makeHTTPServer(t)
	defer os.RemoveAll(dir)
	defer srv.Shutdown()
	defer srv.agent.Shutdown()

	// Attach a custom logger so we can inspect it
	buf := &bytes.Buffer{}
	srv.logger = log.New(buf, "", log.LstdFlags)

	resp := httptest.NewRecorder()
	req, _ := http.NewRequest("GET", "/some/url?token=secret1&token=secret2", nil)

	handler := func(resp http.ResponseWriter, req *http.Request) (interface{}, error) {
		return nil, nil
	}
	srv.wrap(handler)(resp, req)

	// Make sure no tokens from the URL show up in the log
	if strings.Contains(buf.String(), "secret") {
		t.Fatalf("bad: %s", buf.String())
	}
}

func TestPrettyPrint(t *testing.T) {
	testPrettyPrint("pretty=1", t)
}

func TestPrettyPrintBare(t *testing.T) {
	testPrettyPrint("pretty", t)
}

func testPrettyPrint(pretty string, t *testing.T) {
	dir, srv := makeHTTPServer(t)
	defer os.RemoveAll(dir)
	defer srv.Shutdown()
	defer srv.agent.Shutdown()

	r := &structs.DirEntry{Key: "key"}

	resp := httptest.NewRecorder()
	handler := func(resp http.ResponseWriter, req *http.Request) (interface{}, error) {
		return r, nil
	}

	urlStr := "/v1/kv/key?" + pretty
	req, _ := http.NewRequest("GET", urlStr, nil)
	srv.wrap(handler)(resp, req)

	expected, _ := json.MarshalIndent(r, "", "    ")
	actual, err := ioutil.ReadAll(resp.Body)
	if err != nil {
		t.Fatalf("err: %s", err)
	}

	if !bytes.Equal(expected, actual) {
		t.Fatalf("bad: %q", string(actual))
	}
}

func TestParseSource(t *testing.T) {
	dir, srv := makeHTTPServer(t)
	defer os.RemoveAll(dir)
	defer srv.Shutdown()
	defer srv.agent.Shutdown()

	// Default is agent's DC and no node (since the user didn't care, then
	// just give them the cheapest possible query).
	req, err := http.NewRequest("GET",
		"/v1/catalog/nodes", nil)
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	source := structs.QuerySource{}
	srv.parseSource(req, &source)
	if source.Datacenter != "dc1" || source.Node != "" {
		t.Fatalf("bad: %v", source)
	}

	// Adding the source parameter should set that node.
	req, err = http.NewRequest("GET",
		"/v1/catalog/nodes?near=bob", nil)
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	source = structs.QuerySource{}
	srv.parseSource(req, &source)
	if source.Datacenter != "dc1" || source.Node != "bob" {
		t.Fatalf("bad: %v", source)
	}

	// We should follow whatever dc parameter was given so that the node is
	// looked up correctly on the receiving end.
	req, err = http.NewRequest("GET",
		"/v1/catalog/nodes?near=bob&dc=foo", nil)
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	source = structs.QuerySource{}
	srv.parseSource(req, &source)
	if source.Datacenter != "foo" || source.Node != "bob" {
		t.Fatalf("bad: %v", source)
	}

	// The magic "_agent" node name will use the agent's local node name.
	req, err = http.NewRequest("GET",
		"/v1/catalog/nodes?near=_agent", nil)
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	source = structs.QuerySource{}
	srv.parseSource(req, &source)
	if source.Datacenter != "dc1" || source.Node != srv.agent.config.NodeName {
		t.Fatalf("bad: %v", source)
	}
}

func TestParseWait(t *testing.T) {
	resp := httptest.NewRecorder()
	var b structs.QueryOptions

	req, err := http.NewRequest("GET",
		"/v1/catalog/nodes?wait=60s&index=1000", nil)
	if err != nil {
		t.Fatalf("err: %v", err)
	}

	if d := parseWait(resp, req, &b); d {
		t.Fatalf("unexpected done")
	}

	if b.MinQueryIndex != 1000 {
		t.Fatalf("Bad: %v", b)
	}
	if b.MaxQueryTime != 60*time.Second {
		t.Fatalf("Bad: %v", b)
	}
}

func TestParseWait_InvalidTime(t *testing.T) {
	resp := httptest.NewRecorder()
	var b structs.QueryOptions

	req, err := http.NewRequest("GET",
		"/v1/catalog/nodes?wait=60foo&index=1000", nil)
	if err != nil {
		t.Fatalf("err: %v", err)
	}

	if d := parseWait(resp, req, &b); !d {
		t.Fatalf("expected done")
	}

	if resp.Code != 400 {
		t.Fatalf("bad code: %v", resp.Code)
	}
}

func TestParseWait_InvalidIndex(t *testing.T) {
	resp := httptest.NewRecorder()
	var b structs.QueryOptions

	req, err := http.NewRequest("GET",
		"/v1/catalog/nodes?wait=60s&index=foo", nil)
	if err != nil {
		t.Fatalf("err: %v", err)
	}

	if d := parseWait(resp, req, &b); !d {
		t.Fatalf("expected done")
	}

	if resp.Code != 400 {
		t.Fatalf("bad code: %v", resp.Code)
	}
}

func TestParseConsistency(t *testing.T) {
	resp := httptest.NewRecorder()
	var b structs.QueryOptions

	req, err := http.NewRequest("GET",
		"/v1/catalog/nodes?stale", nil)
	if err != nil {
		t.Fatalf("err: %v", err)
	}

	if d := parseConsistency(resp, req, &b); d {
		t.Fatalf("unexpected done")
	}

	if !b.AllowStale {
		t.Fatalf("Bad: %v", b)
	}
	if b.RequireConsistent {
		t.Fatalf("Bad: %v", b)
	}

	b = structs.QueryOptions{}
	req, err = http.NewRequest("GET",
		"/v1/catalog/nodes?consistent", nil)
	if err != nil {
		t.Fatalf("err: %v", err)
	}

	if d := parseConsistency(resp, req, &b); d {
		t.Fatalf("unexpected done")
	}

	if b.AllowStale {
		t.Fatalf("Bad: %v", b)
	}
	if !b.RequireConsistent {
		t.Fatalf("Bad: %v", b)
	}
}

func TestParseConsistency_Invalid(t *testing.T) {
	resp := httptest.NewRecorder()
	var b structs.QueryOptions

	req, err := http.NewRequest("GET",
		"/v1/catalog/nodes?stale&consistent", nil)
	if err != nil {
		t.Fatalf("err: %v", err)
	}

	if d := parseConsistency(resp, req, &b); !d {
		t.Fatalf("expected done")
	}

	if resp.Code != 400 {
		t.Fatalf("bad code: %v", resp.Code)
	}
}

// Test ACL token is resolved in correct order
func TestACLResolution(t *testing.T) {
	var token string
	// Request without token
	req, err := http.NewRequest("GET",
		"/v1/catalog/nodes", nil)
	if err != nil {
		t.Fatalf("err: %v", err)
	}

	// Request with explicit token
	reqToken, err := http.NewRequest("GET",
		"/v1/catalog/nodes?token=foo", nil)
	if err != nil {
		t.Fatalf("err: %v", err)
	}

	// Request with header token only
	reqHeaderToken, err := http.NewRequest("GET",
		"/v1/catalog/nodes", nil)
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	reqHeaderToken.Header.Add("X-Consul-Token", "bar")

	// Request with header and querystring tokens
	reqBothTokens, err := http.NewRequest("GET",
		"/v1/catalog/nodes?token=baz", nil)
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	reqBothTokens.Header.Add("X-Consul-Token", "zap")

	httpTest(t, func(srv *HTTPServer) {
		// Check when no token is set
		srv.agent.config.ACLToken = ""
		srv.parseToken(req, &token)
		if token != "" {
			t.Fatalf("bad: %s", token)
		}

		// Check when ACLToken set
		srv.agent.config.ACLToken = "agent"
		srv.parseToken(req, &token)
		if token != "agent" {
			t.Fatalf("bad: %s", token)
		}

		// Check when AtlasACLToken set, wrong server
		srv.agent.config.AtlasACLToken = "atlas"
		srv.parseToken(req, &token)
		if token != "agent" {
			t.Fatalf("bad: %s", token)
		}

		// Check when AtlasACLToken set, correct server
		srv.addr = scadaHTTPAddr
		srv.parseToken(req, &token)
		if token != "atlas" {
			t.Fatalf("bad: %s", token)
		}

		// Check when AtlasACLToken not, correct server
		srv.agent.config.AtlasACLToken = ""
		srv.parseToken(req, &token)
		if token != "agent" {
			t.Fatalf("bad: %s", token)
		}

		// Explicit token has highest precedence
		srv.parseToken(reqToken, &token)
		if token != "foo" {
			t.Fatalf("bad: %s", token)
		}

		// Header token has precedence over agent token
		srv.parseToken(reqHeaderToken, &token)
		if token != "bar" {
			t.Fatalf("bad: %s", token)
		}

		// Querystring token has precendence over header and agent tokens
		srv.parseToken(reqBothTokens, &token)
		if token != "baz" {
			t.Fatalf("bad: %s", token)
		}
	})
}

func TestScadaHTTP(t *testing.T) {
	// Create the agent
	dir, agent := makeAgent(t, nextConfig())
	defer os.RemoveAll(dir)
	defer agent.Shutdown()

	// Create a generic listener
	list, err := net.Listen("tcp", ":0")
	if err != nil {
		t.Fatalf("err: %s", err)
	}
	defer list.Close()

	// Create the SCADA HTTP server
	scadaHttp := newScadaHttp(agent, list)

	// Returned server uses the listener and scada addr
	if scadaHttp.listener != list {
		t.Fatalf("bad listener: %#v", scadaHttp)
	}
	if scadaHttp.addr != scadaHTTPAddr {
		t.Fatalf("expected %v, got: %v", scadaHttp.addr, scadaHTTPAddr)
	}

	// Check that debug endpoints were not enabled. This will cause
	// the serve mux to panic if the routes are already handled.
	mockFn := func(w http.ResponseWriter, r *http.Request) {}
	scadaHttp.mux.HandleFunc("/debug/pprof/", mockFn)
	scadaHttp.mux.HandleFunc("/debug/pprof/cmdline", mockFn)
	scadaHttp.mux.HandleFunc("/debug/pprof/profile", mockFn)
	scadaHttp.mux.HandleFunc("/debug/pprof/symbol", mockFn)
}

func TestEnableWebUI(t *testing.T) {
	httpTestWithConfig(t, func(s *HTTPServer) {
		req, err := http.NewRequest("GET", "/ui/", nil)
		if err != nil {
			t.Fatalf("err: %v", err)
		}

		// Perform the request
		resp := httptest.NewRecorder()
		s.mux.ServeHTTP(resp, req)

		// Check the result
		if resp.Code != 200 {
			t.Fatalf("should handle ui")
		}
	}, func(c *Config) {
		c.EnableUi = true
	})
}

// assertIndex tests that X-Consul-Index is set and non-zero
func assertIndex(t *testing.T, resp *httptest.ResponseRecorder) {
	header := resp.Header().Get("X-Consul-Index")
	if header == "" || header == "0" {
		t.Fatalf("Bad: %v", header)
	}
}

// checkIndex is like assertIndex but returns an error
func checkIndex(resp *httptest.ResponseRecorder) error {
	header := resp.Header().Get("X-Consul-Index")
	if header == "" || header == "0" {
		return fmt.Errorf("Bad: %v", header)
	}
	return nil
}

// getIndex parses X-Consul-Index
func getIndex(t *testing.T, resp *httptest.ResponseRecorder) uint64 {
	header := resp.Header().Get("X-Consul-Index")
	if header == "" {
		t.Fatalf("Bad: %v", header)
	}
	val, err := strconv.Atoi(header)
	if err != nil {
		t.Fatalf("Bad: %v", header)
	}
	return uint64(val)
}

func httpTest(t *testing.T, f func(srv *HTTPServer)) {
	httpTestWithConfig(t, f, nil)
}

func httpTestWithConfig(t *testing.T, f func(srv *HTTPServer), cb func(c *Config)) {
	dir, srv := makeHTTPServerWithConfig(t, cb)
	defer os.RemoveAll(dir)
	defer srv.Shutdown()
	defer srv.agent.Shutdown()
	testutil.WaitForLeader(t, srv.agent.RPC, "dc1")
	f(srv)
}
