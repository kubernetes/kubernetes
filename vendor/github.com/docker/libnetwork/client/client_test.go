package client

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"
	"testing"

	_ "github.com/docker/libnetwork/testutils"
)

// nopCloser is used to provide a dummy CallFunc for Cmd()
type nopCloser struct {
	io.Reader
}

func (nopCloser) Close() error { return nil }

func TestMain(m *testing.M) {
	setupMockHTTPCallback()
	os.Exit(m.Run())
}

var callbackFunc func(method, path string, data interface{}, headers map[string][]string) (io.ReadCloser, http.Header, int, error)
var mockNwJSON, mockNwListJSON, mockServiceJSON, mockServiceListJSON, mockSbJSON, mockSbListJSON []byte
var mockNwName = "test"
var mockNwID = "2a3456789"
var mockServiceName = "testSrv"
var mockServiceID = "2a3456789"
var mockContainerID = "2a3456789"
var mockSandboxID = "2b3456789"

func setupMockHTTPCallback() {
	var list []networkResource
	nw := networkResource{Name: mockNwName, ID: mockNwID}
	mockNwJSON, _ = json.Marshal(nw)
	list = append(list, nw)
	mockNwListJSON, _ = json.Marshal(list)

	var srvList []serviceResource
	ep := serviceResource{Name: mockServiceName, ID: mockServiceID, Network: mockNwName}
	mockServiceJSON, _ = json.Marshal(ep)
	srvList = append(srvList, ep)
	mockServiceListJSON, _ = json.Marshal(srvList)

	var sbxList []SandboxResource
	sb := SandboxResource{ID: mockSandboxID, ContainerID: mockContainerID}
	mockSbJSON, _ = json.Marshal(sb)
	sbxList = append(sbxList, sb)
	mockSbListJSON, _ = json.Marshal(sbxList)

	dummyHTTPHdr := http.Header{}

	callbackFunc = func(method, path string, data interface{}, headers map[string][]string) (io.ReadCloser, http.Header, int, error) {
		var rsp string
		switch method {
		case "GET":
			if strings.Contains(path, fmt.Sprintf("networks?name=%s", mockNwName)) {
				rsp = string(mockNwListJSON)
			} else if strings.Contains(path, "networks?name=") {
				rsp = "[]"
			} else if strings.Contains(path, fmt.Sprintf("networks?partial-id=%s", mockNwID)) {
				rsp = string(mockNwListJSON)
			} else if strings.Contains(path, "networks?partial-id=") {
				rsp = "[]"
			} else if strings.HasSuffix(path, "networks") {
				rsp = string(mockNwListJSON)
			} else if strings.HasSuffix(path, "networks/"+mockNwID) {
				rsp = string(mockNwJSON)
			} else if strings.Contains(path, fmt.Sprintf("services?name=%s", mockServiceName)) {
				rsp = string(mockServiceListJSON)
			} else if strings.Contains(path, "services?name=") {
				rsp = "[]"
			} else if strings.Contains(path, fmt.Sprintf("services?partial-id=%s", mockServiceID)) {
				rsp = string(mockServiceListJSON)
			} else if strings.Contains(path, "services?partial-id=") {
				rsp = "[]"
			} else if strings.HasSuffix(path, "services") {
				rsp = string(mockServiceListJSON)
			} else if strings.HasSuffix(path, "services/"+mockServiceID) {
				rsp = string(mockServiceJSON)
			} else if strings.Contains(path, "containers") {
				return nopCloser{bytes.NewBufferString("")}, dummyHTTPHdr, 400, fmt.Errorf("Bad Request")
			} else if strings.Contains(path, fmt.Sprintf("sandboxes?container-id=%s", mockContainerID)) {
				rsp = string(mockSbListJSON)
			} else if strings.Contains(path, fmt.Sprintf("sandboxes?partial-container-id=%s", mockContainerID)) {
				rsp = string(mockSbListJSON)
			}
		case "POST":
			var data []byte
			if strings.HasSuffix(path, "networks") {
				data, _ = json.Marshal(mockNwID)
			} else if strings.HasSuffix(path, "services") {
				data, _ = json.Marshal(mockServiceID)
			} else if strings.HasSuffix(path, "backend") {
				data, _ = json.Marshal(mockSandboxID)
			}
			rsp = string(data)
		case "PUT":
		case "DELETE":
			rsp = ""
		}
		return nopCloser{bytes.NewBufferString(rsp)}, dummyHTTPHdr, 200, nil
	}
}

func TestClientDummyCommand(t *testing.T) {
	var out, errOut bytes.Buffer
	cli := NewNetworkCli(&out, &errOut, callbackFunc)

	err := cli.Cmd("docker", "dummy")
	if err == nil {
		t.Fatal("Incorrect Command must fail")
	}
}

func TestClientNetworkInvalidCommand(t *testing.T) {
	var out, errOut bytes.Buffer
	cli := NewNetworkCli(&out, &errOut, callbackFunc)

	err := cli.Cmd("docker", "network", "invalid")
	if err == nil {
		t.Fatal("Passing invalid commands must fail")
	}
}

func TestClientNetworkCreate(t *testing.T) {
	var out, errOut bytes.Buffer
	cli := NewNetworkCli(&out, &errOut, callbackFunc)

	err := cli.Cmd("docker", "network", "create", mockNwName)
	if err != nil {
		t.Fatal(err.Error())
	}
}

func TestClientNetworkCreateWithDriver(t *testing.T) {
	var out, errOut bytes.Buffer
	cli := NewNetworkCli(&out, &errOut, callbackFunc)

	err := cli.Cmd("docker", "network", "create", "-f=dummy", mockNwName)
	if err == nil {
		t.Fatal("Passing incorrect flags to the create command must fail")
	}

	err = cli.Cmd("docker", "network", "create", "-d=dummy", mockNwName)
	if err != nil {
		t.Fatalf(err.Error())
	}
}

func TestClientNetworkRm(t *testing.T) {
	var out, errOut bytes.Buffer
	cli := NewNetworkCli(&out, &errOut, callbackFunc)

	err := cli.Cmd("docker", "network", "rm", mockNwName)
	if err != nil {
		t.Fatal(err.Error())
	}
}

func TestClientNetworkLs(t *testing.T) {
	var out, errOut bytes.Buffer
	cli := NewNetworkCli(&out, &errOut, callbackFunc)

	err := cli.Cmd("docker", "network", "ls")
	if err != nil {
		t.Fatal(err.Error())
	}
}

func TestClientNetworkInfo(t *testing.T) {
	var out, errOut bytes.Buffer
	cli := NewNetworkCli(&out, &errOut, callbackFunc)

	err := cli.Cmd("docker", "network", "info", mockNwName)
	if err != nil {
		t.Fatal(err.Error())
	}
}

func TestClientNetworkInfoById(t *testing.T) {
	var out, errOut bytes.Buffer
	cli := NewNetworkCli(&out, &errOut, callbackFunc)

	err := cli.Cmd("docker", "network", "info", mockNwID)
	if err != nil {
		t.Fatal(err.Error())
	}
}

// Docker Flag processing in flag.go uses os.Exit() frequently, even for --help
// TODO : Handle the --help test-case in the IT when CLI is available
/*
func TestClientNetworkServiceCreateHelp(t *testing.T) {
	var out, errOut bytes.Buffer
	cFunc := func(method, path string, data interface{}, headers map[string][]string) (io.ReadCloser, int, error) {
		return nil, 0, nil
	}
	cli := NewNetworkCli(&out, &errOut, callbackFunc)

	err := cli.Cmd("docker", "network", "create", "--help")
	if err != nil {
		t.Fatalf(err.Error())
	}
}
*/

// Docker flag processing in flag.go uses os.Exit(1) for incorrect parameter case.
// TODO : Handle the missing argument case in the IT when CLI is available
/*
func TestClientNetworkServiceCreateMissingArgument(t *testing.T) {
	var out, errOut bytes.Buffer
	cFunc := func(method, path string, data interface{}, headers map[string][]string) (io.ReadCloser, int, error) {
		return nil, 0, nil
	}
	cli := NewNetworkCli(&out, &errOut, callbackFunc)

	err := cli.Cmd("docker", "network", "create")
	if err != nil {
		t.Fatal(err.Error())
	}
}
*/
