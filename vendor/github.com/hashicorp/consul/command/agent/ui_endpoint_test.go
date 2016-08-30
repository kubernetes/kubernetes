package agent

import (
	"bytes"
	"fmt"
	"io"
	"io/ioutil"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"reflect"
	"testing"

	"github.com/hashicorp/consul/consul/structs"
	"github.com/hashicorp/consul/testutil"
	"github.com/hashicorp/go-cleanhttp"
)

func TestUiIndex(t *testing.T) {
	// Make a test dir to serve UI files
	uiDir, err := ioutil.TempDir("", "consul")
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	defer os.RemoveAll(uiDir)

	// Make the server
	dir, srv := makeHTTPServerWithConfig(t, func(c *Config) {
		c.UiDir = uiDir
	})
	defer os.RemoveAll(dir)
	defer srv.Shutdown()
	defer srv.agent.Shutdown()

	// Create file
	path := filepath.Join(srv.uiDir, "my-file")
	if err := ioutil.WriteFile(path, []byte("test"), 777); err != nil {
		t.Fatalf("err: %v", err)
	}

	// Register node
	req, err := http.NewRequest("GET", "/ui/my-file", nil)
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	req.URL.Scheme = "http"
	req.URL.Host = srv.listener.Addr().String()

	// Make the request
	client := cleanhttp.DefaultClient()
	resp, err := client.Do(req)
	if err != nil {
		t.Fatalf("err: %v", err)
	}

	// Verify the response
	if resp.StatusCode != 200 {
		t.Fatalf("bad: %v", resp)
	}

	// Verify the body
	out := bytes.NewBuffer(nil)
	io.Copy(out, resp.Body)
	if string(out.Bytes()) != "test" {
		t.Fatalf("bad: %s", out.Bytes())
	}
}

func TestUiNodes(t *testing.T) {
	dir, srv := makeHTTPServer(t)
	defer os.RemoveAll(dir)
	defer srv.Shutdown()
	defer srv.agent.Shutdown()

	testutil.WaitForLeader(t, srv.agent.RPC, "dc1")

	args := &structs.RegisterRequest{
		Datacenter: "dc1",
		Node:       "test",
		Address:    "127.0.0.1",
	}

	var out struct{}
	if err := srv.agent.RPC("Catalog.Register", args, &out); err != nil {
		t.Fatalf("err: %v", err)
	}

	req, err := http.NewRequest("GET", "/v1/internal/ui/nodes/dc1", nil)
	if err != nil {
		t.Fatalf("err: %v", err)
	}

	resp := httptest.NewRecorder()
	obj, err := srv.UINodes(resp, req)
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	assertIndex(t, resp)

	// Should be 2 nodes, and all the empty lists should be non-nil
	nodes := obj.(structs.NodeDump)
	if len(nodes) != 2 ||
		nodes[0].Node != srv.agent.config.NodeName ||
		nodes[0].Services == nil || len(nodes[0].Services) != 1 ||
		nodes[0].Checks == nil || len(nodes[0].Checks) != 1 ||
		nodes[1].Node != "test" ||
		nodes[1].Services == nil || len(nodes[1].Services) != 0 ||
		nodes[1].Checks == nil || len(nodes[1].Checks) != 0 {
		t.Fatalf("bad: %v", obj)
	}
}

func TestUiNodeInfo(t *testing.T) {
	dir, srv := makeHTTPServer(t)
	defer os.RemoveAll(dir)
	defer srv.Shutdown()
	defer srv.agent.Shutdown()

	testutil.WaitForLeader(t, srv.agent.RPC, "dc1")

	req, err := http.NewRequest("GET",
		fmt.Sprintf("/v1/internal/ui/node/%s", srv.agent.config.NodeName), nil)
	if err != nil {
		t.Fatalf("err: %v", err)
	}

	resp := httptest.NewRecorder()
	obj, err := srv.UINodeInfo(resp, req)
	if err != nil {
		t.Fatalf("err: %v", err)
	}

	assertIndex(t, resp)

	// Should be 1 node for the server
	node := obj.(*structs.NodeInfo)
	if node.Node != srv.agent.config.NodeName {
		t.Fatalf("bad: %v", node)
	}

	args := &structs.RegisterRequest{
		Datacenter: "dc1",
		Node:       "test",
		Address:    "127.0.0.1",
	}

	var out struct{}
	if err := srv.agent.RPC("Catalog.Register", args, &out); err != nil {
		t.Fatalf("err: %v", err)
	}

	req, err = http.NewRequest("GET", "/v1/internal/ui/node/test", nil)
	if err != nil {
		t.Fatalf("err: %v", err)
	}

	resp = httptest.NewRecorder()
	obj, err = srv.UINodeInfo(resp, req)
	if err != nil {
		t.Fatalf("err: %v", err)
	}

	assertIndex(t, resp)

	// Should be non-nil empty lists for services and checks
	node = obj.(*structs.NodeInfo)
	if node.Node != "test" ||
		node.Services == nil || len(node.Services) != 0 ||
		node.Checks == nil || len(node.Checks) != 0 {
		t.Fatalf("bad: %v", node)
	}
}

func TestSummarizeServices(t *testing.T) {
	dump := structs.NodeDump{
		&structs.NodeInfo{
			Node:    "foo",
			Address: "127.0.0.1",
			Services: []*structs.NodeService{
				&structs.NodeService{
					Service: "api",
				},
				&structs.NodeService{
					Service: "web",
				},
			},
			Checks: []*structs.HealthCheck{
				&structs.HealthCheck{
					Status:      structs.HealthPassing,
					ServiceName: "",
				},
				&structs.HealthCheck{
					Status:      structs.HealthPassing,
					ServiceName: "web",
				},
				&structs.HealthCheck{
					Status:      structs.HealthWarning,
					ServiceName: "api",
				},
			},
		},
		&structs.NodeInfo{
			Node:    "bar",
			Address: "127.0.0.2",
			Services: []*structs.NodeService{
				&structs.NodeService{
					Service: "web",
				},
			},
			Checks: []*structs.HealthCheck{
				&structs.HealthCheck{
					Status:      structs.HealthCritical,
					ServiceName: "web",
				},
			},
		},
		&structs.NodeInfo{
			Node:    "zip",
			Address: "127.0.0.3",
			Services: []*structs.NodeService{
				&structs.NodeService{
					Service: "cache",
				},
			},
		},
	}

	summary := summarizeServices(dump)
	if len(summary) != 3 {
		t.Fatalf("bad: %v", summary)
	}

	expectAPI := &ServiceSummary{
		Name:           "api",
		Nodes:          []string{"foo"},
		ChecksPassing:  1,
		ChecksWarning:  1,
		ChecksCritical: 0,
	}
	if !reflect.DeepEqual(summary[0], expectAPI) {
		t.Fatalf("bad: %v", summary[0])
	}

	expectCache := &ServiceSummary{
		Name:           "cache",
		Nodes:          []string{"zip"},
		ChecksPassing:  0,
		ChecksWarning:  0,
		ChecksCritical: 0,
	}
	if !reflect.DeepEqual(summary[1], expectCache) {
		t.Fatalf("bad: %v", summary[1])
	}

	expectWeb := &ServiceSummary{
		Name:           "web",
		Nodes:          []string{"bar", "foo"},
		ChecksPassing:  2,
		ChecksWarning:  0,
		ChecksCritical: 1,
	}
	if !reflect.DeepEqual(summary[2], expectWeb) {
		t.Fatalf("bad: %v", summary[2])
	}
}
