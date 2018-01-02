package main

import (
	"archive/tar"
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"io/ioutil"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
	"time"

	"github.com/docker/docker/api/types"
	containertypes "github.com/docker/docker/api/types/container"
	mounttypes "github.com/docker/docker/api/types/mount"
	networktypes "github.com/docker/docker/api/types/network"
	"github.com/docker/docker/integration-cli/checker"
	"github.com/docker/docker/integration-cli/cli"
	"github.com/docker/docker/integration-cli/cli/build"
	"github.com/docker/docker/integration-cli/request"
	"github.com/docker/docker/pkg/ioutils"
	"github.com/docker/docker/pkg/mount"
	"github.com/docker/docker/pkg/stringid"
	"github.com/docker/docker/pkg/testutil"
	"github.com/docker/docker/volume"
	"github.com/go-check/check"
)

func (s *DockerSuite) TestContainerAPIGetAll(c *check.C) {
	startCount := getContainerCount(c)
	name := "getall"
	dockerCmd(c, "run", "--name", name, "busybox", "true")

	status, body, err := request.SockRequest("GET", "/containers/json?all=1", nil, daemonHost())
	c.Assert(err, checker.IsNil)
	c.Assert(status, checker.Equals, http.StatusOK)

	var inspectJSON []struct {
		Names []string
	}
	err = json.Unmarshal(body, &inspectJSON)
	c.Assert(err, checker.IsNil, check.Commentf("unable to unmarshal response body"))

	c.Assert(inspectJSON, checker.HasLen, startCount+1)

	actual := inspectJSON[0].Names[0]
	c.Assert(actual, checker.Equals, "/"+name)
}

// regression test for empty json field being omitted #13691
func (s *DockerSuite) TestContainerAPIGetJSONNoFieldsOmitted(c *check.C) {
	dockerCmd(c, "run", "busybox", "true")

	status, body, err := request.SockRequest("GET", "/containers/json?all=1", nil, daemonHost())
	c.Assert(err, checker.IsNil)
	c.Assert(status, checker.Equals, http.StatusOK)

	// empty Labels field triggered this bug, make sense to check for everything
	// cause even Ports for instance can trigger this bug
	// better safe than sorry..
	fields := []string{
		"Id",
		"Names",
		"Image",
		"Command",
		"Created",
		"Ports",
		"Labels",
		"Status",
		"NetworkSettings",
	}

	// decoding into types.Container do not work since it eventually unmarshal
	// and empty field to an empty go map, so we just check for a string
	for _, f := range fields {
		if !strings.Contains(string(body), f) {
			c.Fatalf("Field %s is missing and it shouldn't", f)
		}
	}
}

type containerPs struct {
	Names []string
	Ports []map[string]interface{}
}

// regression test for non-empty fields from #13901
func (s *DockerSuite) TestContainerAPIPsOmitFields(c *check.C) {
	// Problematic for Windows porting due to networking not yet being passed back
	testRequires(c, DaemonIsLinux)
	name := "pstest"
	port := 80
	runSleepingContainer(c, "--name", name, "--expose", strconv.Itoa(port))

	status, body, err := request.SockRequest("GET", "/containers/json?all=1", nil, daemonHost())
	c.Assert(err, checker.IsNil)
	c.Assert(status, checker.Equals, http.StatusOK)

	var resp []containerPs
	err = json.Unmarshal(body, &resp)
	c.Assert(err, checker.IsNil)

	var foundContainer *containerPs
	for _, container := range resp {
		for _, testName := range container.Names {
			if "/"+name == testName {
				foundContainer = &container
				break
			}
		}
	}

	c.Assert(foundContainer.Ports, checker.HasLen, 1)
	c.Assert(foundContainer.Ports[0]["PrivatePort"], checker.Equals, float64(port))
	_, ok := foundContainer.Ports[0]["PublicPort"]
	c.Assert(ok, checker.Not(checker.Equals), true)
	_, ok = foundContainer.Ports[0]["IP"]
	c.Assert(ok, checker.Not(checker.Equals), true)
}

func (s *DockerSuite) TestContainerAPIGetExport(c *check.C) {
	// Not supported on Windows as Windows does not support docker export
	testRequires(c, DaemonIsLinux)
	name := "exportcontainer"
	dockerCmd(c, "run", "--name", name, "busybox", "touch", "/test")

	status, body, err := request.SockRequest("GET", "/containers/"+name+"/export", nil, daemonHost())
	c.Assert(err, checker.IsNil)
	c.Assert(status, checker.Equals, http.StatusOK)

	found := false
	for tarReader := tar.NewReader(bytes.NewReader(body)); ; {
		h, err := tarReader.Next()
		if err != nil && err == io.EOF {
			break
		}
		if h.Name == "test" {
			found = true
			break
		}
	}
	c.Assert(found, checker.True, check.Commentf("The created test file has not been found in the exported image"))
}

func (s *DockerSuite) TestContainerAPIGetChanges(c *check.C) {
	// Not supported on Windows as Windows does not support docker diff (/containers/name/changes)
	testRequires(c, DaemonIsLinux)
	name := "changescontainer"
	dockerCmd(c, "run", "--name", name, "busybox", "rm", "/etc/passwd")

	status, body, err := request.SockRequest("GET", "/containers/"+name+"/changes", nil, daemonHost())
	c.Assert(err, checker.IsNil)
	c.Assert(status, checker.Equals, http.StatusOK)

	changes := []struct {
		Kind int
		Path string
	}{}
	c.Assert(json.Unmarshal(body, &changes), checker.IsNil, check.Commentf("unable to unmarshal response body"))

	// Check the changelog for removal of /etc/passwd
	success := false
	for _, elem := range changes {
		if elem.Path == "/etc/passwd" && elem.Kind == 2 {
			success = true
		}
	}
	c.Assert(success, checker.True, check.Commentf("/etc/passwd has been removed but is not present in the diff"))
}

func (s *DockerSuite) TestGetContainerStats(c *check.C) {
	var (
		name = "statscontainer"
	)
	runSleepingContainer(c, "--name", name)

	type b struct {
		status int
		body   []byte
		err    error
	}
	bc := make(chan b, 1)
	go func() {
		status, body, err := request.SockRequest("GET", "/containers/"+name+"/stats", nil, daemonHost())
		bc <- b{status, body, err}
	}()

	// allow some time to stream the stats from the container
	time.Sleep(4 * time.Second)
	dockerCmd(c, "rm", "-f", name)

	// collect the results from the stats stream or timeout and fail
	// if the stream was not disconnected.
	select {
	case <-time.After(2 * time.Second):
		c.Fatal("stream was not closed after container was removed")
	case sr := <-bc:
		c.Assert(sr.err, checker.IsNil)
		c.Assert(sr.status, checker.Equals, http.StatusOK)

		dec := json.NewDecoder(bytes.NewBuffer(sr.body))
		var s *types.Stats
		// decode only one object from the stream
		c.Assert(dec.Decode(&s), checker.IsNil)
	}
}

func (s *DockerSuite) TestGetContainerStatsRmRunning(c *check.C) {
	out := runSleepingContainer(c)
	id := strings.TrimSpace(out)

	buf := &testutil.ChannelBuffer{C: make(chan []byte, 1)}
	defer buf.Close()

	_, body, err := request.Get("/containers/"+id+"/stats?stream=1", request.JSON)
	c.Assert(err, checker.IsNil)
	defer body.Close()

	chErr := make(chan error, 1)
	go func() {
		_, err = io.Copy(buf, body)
		chErr <- err
	}()

	b := make([]byte, 32)
	// make sure we've got some stats
	_, err = buf.ReadTimeout(b, 2*time.Second)
	c.Assert(err, checker.IsNil)

	// Now remove without `-f` and make sure we are still pulling stats
	_, _, err = dockerCmdWithError("rm", id)
	c.Assert(err, checker.Not(checker.IsNil), check.Commentf("rm should have failed but didn't"))
	_, err = buf.ReadTimeout(b, 2*time.Second)
	c.Assert(err, checker.IsNil)

	dockerCmd(c, "rm", "-f", id)
	c.Assert(<-chErr, checker.IsNil)
}

// regression test for gh13421
// previous test was just checking one stat entry so it didn't fail (stats with
// stream false always return one stat)
func (s *DockerSuite) TestGetContainerStatsStream(c *check.C) {
	name := "statscontainer"
	runSleepingContainer(c, "--name", name)

	type b struct {
		status int
		body   io.ReadCloser
		err    error
	}
	bc := make(chan b, 1)
	go func() {
		status, body, err := request.Get("/containers/" + name + "/stats")
		bc <- b{status.StatusCode, body, err}
	}()

	// allow some time to stream the stats from the container
	time.Sleep(4 * time.Second)
	dockerCmd(c, "rm", "-f", name)

	// collect the results from the stats stream or timeout and fail
	// if the stream was not disconnected.
	select {
	case <-time.After(2 * time.Second):
		c.Fatal("stream was not closed after container was removed")
	case sr := <-bc:
		c.Assert(sr.err, checker.IsNil)
		c.Assert(sr.status, checker.Equals, http.StatusOK)

		b, err := ioutil.ReadAll(sr.body)
		c.Assert(err, checker.IsNil)
		s := string(b)
		// count occurrences of "read" of types.Stats
		if l := strings.Count(s, "read"); l < 2 {
			c.Fatalf("Expected more than one stat streamed, got %d", l)
		}
	}
}

func (s *DockerSuite) TestGetContainerStatsNoStream(c *check.C) {
	name := "statscontainer"
	runSleepingContainer(c, "--name", name)

	type b struct {
		status int
		body   []byte
		err    error
	}
	bc := make(chan b, 1)
	go func() {
		status, body, err := request.SockRequest("GET", "/containers/"+name+"/stats?stream=0", nil, daemonHost())
		bc <- b{status, body, err}
	}()

	// allow some time to stream the stats from the container
	time.Sleep(4 * time.Second)
	dockerCmd(c, "rm", "-f", name)

	// collect the results from the stats stream or timeout and fail
	// if the stream was not disconnected.
	select {
	case <-time.After(2 * time.Second):
		c.Fatal("stream was not closed after container was removed")
	case sr := <-bc:
		c.Assert(sr.err, checker.IsNil)
		c.Assert(sr.status, checker.Equals, http.StatusOK)

		s := string(sr.body)
		// count occurrences of `"read"` of types.Stats
		c.Assert(strings.Count(s, `"read"`), checker.Equals, 1, check.Commentf("Expected only one stat streamed, got %d", strings.Count(s, `"read"`)))
	}
}

func (s *DockerSuite) TestGetStoppedContainerStats(c *check.C) {
	name := "statscontainer"
	dockerCmd(c, "create", "--name", name, "busybox", "ps")

	type stats struct {
		status int
		err    error
	}
	chResp := make(chan stats)

	// We expect an immediate response, but if it's not immediate, the test would hang, so put it in a goroutine
	// below we'll check this on a timeout.
	go func() {
		resp, body, err := request.Get("/containers/" + name + "/stats")
		body.Close()
		chResp <- stats{resp.StatusCode, err}
	}()

	select {
	case r := <-chResp:
		c.Assert(r.err, checker.IsNil)
		c.Assert(r.status, checker.Equals, http.StatusOK)
	case <-time.After(10 * time.Second):
		c.Fatal("timeout waiting for stats response for stopped container")
	}
}

func (s *DockerSuite) TestContainerAPIPause(c *check.C) {
	// Problematic on Windows as Windows does not support pause
	testRequires(c, DaemonIsLinux)

	getPaused := func(c *check.C) []string {
		return strings.Fields(cli.DockerCmd(c, "ps", "-f", "status=paused", "-q", "-a").Combined())
	}

	out := cli.DockerCmd(c, "run", "-d", "busybox", "sleep", "30").Combined()
	ContainerID := strings.TrimSpace(out)

	resp, _, err := request.Post("/containers/" + ContainerID + "/pause")
	c.Assert(err, checker.IsNil)
	c.Assert(resp.StatusCode, checker.Equals, http.StatusNoContent)

	pausedContainers := getPaused(c)

	if len(pausedContainers) != 1 || stringid.TruncateID(ContainerID) != pausedContainers[0] {
		c.Fatalf("there should be one paused container and not %d", len(pausedContainers))
	}

	resp, _, err = request.Post("/containers/" + ContainerID + "/unpause")
	c.Assert(err, checker.IsNil)
	c.Assert(resp.StatusCode, checker.Equals, http.StatusNoContent)

	pausedContainers = getPaused(c)
	c.Assert(pausedContainers, checker.HasLen, 0, check.Commentf("There should be no paused container."))
}

func (s *DockerSuite) TestContainerAPITop(c *check.C) {
	testRequires(c, DaemonIsLinux)
	out, _ := dockerCmd(c, "run", "-d", "busybox", "/bin/sh", "-c", "top")
	id := strings.TrimSpace(string(out))
	c.Assert(waitRun(id), checker.IsNil)

	type topResp struct {
		Titles    []string
		Processes [][]string
	}
	var top topResp
	status, b, err := request.SockRequest("GET", "/containers/"+id+"/top?ps_args=aux", nil, daemonHost())
	c.Assert(err, checker.IsNil)
	c.Assert(status, checker.Equals, http.StatusOK)
	c.Assert(json.Unmarshal(b, &top), checker.IsNil)
	c.Assert(top.Titles, checker.HasLen, 11, check.Commentf("expected 11 titles, found %d: %v", len(top.Titles), top.Titles))

	if top.Titles[0] != "USER" || top.Titles[10] != "COMMAND" {
		c.Fatalf("expected `USER` at `Titles[0]` and `COMMAND` at Titles[10]: %v", top.Titles)
	}
	c.Assert(top.Processes, checker.HasLen, 2, check.Commentf("expected 2 processes, found %d: %v", len(top.Processes), top.Processes))
	c.Assert(top.Processes[0][10], checker.Equals, "/bin/sh -c top")
	c.Assert(top.Processes[1][10], checker.Equals, "top")
}

func (s *DockerSuite) TestContainerAPITopWindows(c *check.C) {
	testRequires(c, DaemonIsWindows)
	out := runSleepingContainer(c, "-d")
	id := strings.TrimSpace(string(out))
	c.Assert(waitRun(id), checker.IsNil)

	type topResp struct {
		Titles    []string
		Processes [][]string
	}
	var top topResp
	status, b, err := request.SockRequest("GET", "/containers/"+id+"/top", nil, daemonHost())
	c.Assert(err, checker.IsNil)
	c.Assert(status, checker.Equals, http.StatusOK)
	c.Assert(json.Unmarshal(b, &top), checker.IsNil)
	c.Assert(top.Titles, checker.HasLen, 4, check.Commentf("expected 4 titles, found %d: %v", len(top.Titles), top.Titles))

	if top.Titles[0] != "Name" || top.Titles[3] != "Private Working Set" {
		c.Fatalf("expected `Name` at `Titles[0]` and `Private Working Set` at Titles[3]: %v", top.Titles)
	}
	c.Assert(len(top.Processes), checker.GreaterOrEqualThan, 2, check.Commentf("expected at least 2 processes, found %d: %v", len(top.Processes), top.Processes))

	foundProcess := false
	expectedProcess := "busybox.exe"
	for _, process := range top.Processes {
		if process[0] == expectedProcess {
			foundProcess = true
			break
		}
	}

	c.Assert(foundProcess, checker.Equals, true, check.Commentf("expected to find %s: %v", expectedProcess, top.Processes))
}

func (s *DockerSuite) TestContainerAPICommit(c *check.C) {
	cName := "testapicommit"
	dockerCmd(c, "run", "--name="+cName, "busybox", "/bin/sh", "-c", "touch /test")

	name := "testcontainerapicommit"
	status, b, err := request.SockRequest("POST", "/commit?repo="+name+"&testtag=tag&container="+cName, nil, daemonHost())
	c.Assert(err, checker.IsNil)
	c.Assert(status, checker.Equals, http.StatusCreated)

	type resp struct {
		ID string
	}
	var img resp
	c.Assert(json.Unmarshal(b, &img), checker.IsNil)

	cmd := inspectField(c, img.ID, "Config.Cmd")
	c.Assert(cmd, checker.Equals, "[/bin/sh -c touch /test]", check.Commentf("got wrong Cmd from commit: %q", cmd))

	// sanity check, make sure the image is what we think it is
	dockerCmd(c, "run", img.ID, "ls", "/test")
}

func (s *DockerSuite) TestContainerAPICommitWithLabelInConfig(c *check.C) {
	cName := "testapicommitwithconfig"
	dockerCmd(c, "run", "--name="+cName, "busybox", "/bin/sh", "-c", "touch /test")

	config := map[string]interface{}{
		"Labels": map[string]string{"key1": "value1", "key2": "value2"},
	}

	name := "testcontainerapicommitwithconfig"
	status, b, err := request.SockRequest("POST", "/commit?repo="+name+"&container="+cName, config, daemonHost())
	c.Assert(err, checker.IsNil)
	c.Assert(status, checker.Equals, http.StatusCreated)

	type resp struct {
		ID string
	}
	var img resp
	c.Assert(json.Unmarshal(b, &img), checker.IsNil)

	label1 := inspectFieldMap(c, img.ID, "Config.Labels", "key1")
	c.Assert(label1, checker.Equals, "value1")

	label2 := inspectFieldMap(c, img.ID, "Config.Labels", "key2")
	c.Assert(label2, checker.Equals, "value2")

	cmd := inspectField(c, img.ID, "Config.Cmd")
	c.Assert(cmd, checker.Equals, "[/bin/sh -c touch /test]", check.Commentf("got wrong Cmd from commit: %q", cmd))

	// sanity check, make sure the image is what we think it is
	dockerCmd(c, "run", img.ID, "ls", "/test")
}

func (s *DockerSuite) TestContainerAPIBadPort(c *check.C) {
	// TODO Windows to Windows CI - Port this test
	testRequires(c, DaemonIsLinux)
	config := map[string]interface{}{
		"Image": "busybox",
		"Cmd":   []string{"/bin/sh", "-c", "echo test"},
		"PortBindings": map[string]interface{}{
			"8080/tcp": []map[string]interface{}{
				{
					"HostIP":   "",
					"HostPort": "aa80",
				},
			},
		},
	}

	jsonData := bytes.NewBuffer(nil)
	json.NewEncoder(jsonData).Encode(config)

	status, body, err := request.SockRequest("POST", "/containers/create", config, daemonHost())
	c.Assert(err, checker.IsNil)
	c.Assert(status, checker.Equals, http.StatusInternalServerError)
	c.Assert(getErrorMessage(c, body), checker.Equals, `invalid port specification: "aa80"`, check.Commentf("Incorrect error msg: %s", body))
}

func (s *DockerSuite) TestContainerAPICreate(c *check.C) {
	config := map[string]interface{}{
		"Image": "busybox",
		"Cmd":   []string{"/bin/sh", "-c", "touch /test && ls /test"},
	}

	status, b, err := request.SockRequest("POST", "/containers/create", config, daemonHost())
	c.Assert(err, checker.IsNil)
	c.Assert(status, checker.Equals, http.StatusCreated)

	type createResp struct {
		ID string
	}
	var container createResp
	c.Assert(json.Unmarshal(b, &container), checker.IsNil)

	out, _ := dockerCmd(c, "start", "-a", container.ID)
	c.Assert(strings.TrimSpace(out), checker.Equals, "/test")
}

func (s *DockerSuite) TestContainerAPICreateEmptyConfig(c *check.C) {
	config := map[string]interface{}{}

	status, body, err := request.SockRequest("POST", "/containers/create", config, daemonHost())
	c.Assert(err, checker.IsNil)
	c.Assert(status, checker.Equals, http.StatusInternalServerError)

	expected := "Config cannot be empty in order to create a container"
	c.Assert(getErrorMessage(c, body), checker.Equals, expected)
}

func (s *DockerSuite) TestContainerAPICreateMultipleNetworksConfig(c *check.C) {
	// Container creation must fail if client specified configurations for more than one network
	config := map[string]interface{}{
		"Image": "busybox",
		"NetworkingConfig": networktypes.NetworkingConfig{
			EndpointsConfig: map[string]*networktypes.EndpointSettings{
				"net1": {},
				"net2": {},
				"net3": {},
			},
		},
	}

	status, body, err := request.SockRequest("POST", "/containers/create", config, daemonHost())
	c.Assert(err, checker.IsNil)
	c.Assert(status, checker.Equals, http.StatusBadRequest)
	msg := getErrorMessage(c, body)
	// network name order in error message is not deterministic
	c.Assert(msg, checker.Contains, "Container cannot be connected to network endpoints")
	c.Assert(msg, checker.Contains, "net1")
	c.Assert(msg, checker.Contains, "net2")
	c.Assert(msg, checker.Contains, "net3")
}

func (s *DockerSuite) TestContainerAPICreateWithHostName(c *check.C) {
	domainName := "test-domain"
	hostName := "test-hostname"
	config := map[string]interface{}{
		"Image":      "busybox",
		"Hostname":   hostName,
		"Domainname": domainName,
	}

	status, body, err := request.SockRequest("POST", "/containers/create", config, daemonHost())
	c.Assert(err, checker.IsNil)
	c.Assert(status, checker.Equals, http.StatusCreated)

	var container containertypes.ContainerCreateCreatedBody
	c.Assert(json.Unmarshal(body, &container), checker.IsNil)

	status, body, err = request.SockRequest("GET", "/containers/"+container.ID+"/json", nil, daemonHost())
	c.Assert(err, checker.IsNil)
	c.Assert(status, checker.Equals, http.StatusOK)

	var containerJSON types.ContainerJSON
	c.Assert(json.Unmarshal(body, &containerJSON), checker.IsNil)
	c.Assert(containerJSON.Config.Hostname, checker.Equals, hostName, check.Commentf("Mismatched Hostname"))
	c.Assert(containerJSON.Config.Domainname, checker.Equals, domainName, check.Commentf("Mismatched Domainname"))
}

func (s *DockerSuite) TestContainerAPICreateBridgeNetworkMode(c *check.C) {
	// Windows does not support bridge
	testRequires(c, DaemonIsLinux)
	UtilCreateNetworkMode(c, "bridge")
}

func (s *DockerSuite) TestContainerAPICreateOtherNetworkModes(c *check.C) {
	// Windows does not support these network modes
	testRequires(c, DaemonIsLinux, NotUserNamespace)
	UtilCreateNetworkMode(c, "host")
	UtilCreateNetworkMode(c, "container:web1")
}

func UtilCreateNetworkMode(c *check.C, networkMode string) {
	config := map[string]interface{}{
		"Image":      "busybox",
		"HostConfig": map[string]interface{}{"NetworkMode": networkMode},
	}

	status, body, err := request.SockRequest("POST", "/containers/create", config, daemonHost())
	c.Assert(err, checker.IsNil)
	c.Assert(status, checker.Equals, http.StatusCreated)

	var container containertypes.ContainerCreateCreatedBody
	c.Assert(json.Unmarshal(body, &container), checker.IsNil)

	status, body, err = request.SockRequest("GET", "/containers/"+container.ID+"/json", nil, daemonHost())
	c.Assert(err, checker.IsNil)
	c.Assert(status, checker.Equals, http.StatusOK)

	var containerJSON types.ContainerJSON
	c.Assert(json.Unmarshal(body, &containerJSON), checker.IsNil)
	c.Assert(containerJSON.HostConfig.NetworkMode, checker.Equals, containertypes.NetworkMode(networkMode), check.Commentf("Mismatched NetworkMode"))
}

func (s *DockerSuite) TestContainerAPICreateWithCpuSharesCpuset(c *check.C) {
	// TODO Windows to Windows CI. The CpuShares part could be ported.
	testRequires(c, DaemonIsLinux)
	config := map[string]interface{}{
		"Image":      "busybox",
		"CpuShares":  512,
		"CpusetCpus": "0",
	}

	status, body, err := request.SockRequest("POST", "/containers/create", config, daemonHost())
	c.Assert(err, checker.IsNil)
	c.Assert(status, checker.Equals, http.StatusCreated)

	var container containertypes.ContainerCreateCreatedBody
	c.Assert(json.Unmarshal(body, &container), checker.IsNil)

	status, body, err = request.SockRequest("GET", "/containers/"+container.ID+"/json", nil, daemonHost())
	c.Assert(err, checker.IsNil)
	c.Assert(status, checker.Equals, http.StatusOK)

	var containerJSON types.ContainerJSON

	c.Assert(json.Unmarshal(body, &containerJSON), checker.IsNil)

	out := inspectField(c, containerJSON.ID, "HostConfig.CpuShares")
	c.Assert(out, checker.Equals, "512")

	outCpuset := inspectField(c, containerJSON.ID, "HostConfig.CpusetCpus")
	c.Assert(outCpuset, checker.Equals, "0")
}

func (s *DockerSuite) TestContainerAPIVerifyHeader(c *check.C) {
	config := map[string]interface{}{
		"Image": "busybox",
	}

	create := func(ct string) (*http.Response, io.ReadCloser, error) {
		jsonData := bytes.NewBuffer(nil)
		c.Assert(json.NewEncoder(jsonData).Encode(config), checker.IsNil)
		return request.Post("/containers/create", request.RawContent(ioutil.NopCloser(jsonData)), request.ContentType(ct))
	}

	// Try with no content-type
	res, body, err := create("")
	c.Assert(err, checker.IsNil)
	c.Assert(res.StatusCode, checker.Equals, http.StatusInternalServerError)
	body.Close()

	// Try with wrong content-type
	res, body, err = create("application/xml")
	c.Assert(err, checker.IsNil)
	c.Assert(res.StatusCode, checker.Equals, http.StatusInternalServerError)
	body.Close()

	// now application/json
	res, body, err = create("application/json")
	c.Assert(err, checker.IsNil)
	c.Assert(res.StatusCode, checker.Equals, http.StatusCreated)
	body.Close()
}

//Issue 14230. daemon should return 500 for invalid port syntax
func (s *DockerSuite) TestContainerAPIInvalidPortSyntax(c *check.C) {
	config := `{
				  "Image": "busybox",
				  "HostConfig": {
					"NetworkMode": "default",
					"PortBindings": {
					  "19039;1230": [
						{}
					  ]
					}
				  }
				}`

	res, body, err := request.Post("/containers/create", request.RawString(config), request.JSON)
	c.Assert(err, checker.IsNil)
	c.Assert(res.StatusCode, checker.Equals, http.StatusInternalServerError)

	b, err := testutil.ReadBody(body)
	c.Assert(err, checker.IsNil)
	c.Assert(string(b[:]), checker.Contains, "invalid port")
}

func (s *DockerSuite) TestContainerAPIRestartPolicyInvalidPolicyName(c *check.C) {
	config := `{
		"Image": "busybox",
		"HostConfig": {
			"RestartPolicy": {
				"Name": "something",
				"MaximumRetryCount": 0
			}
		}
	}`

	res, body, err := request.Post("/containers/create", request.RawString(config), request.JSON)
	c.Assert(err, checker.IsNil)
	c.Assert(res.StatusCode, checker.Equals, http.StatusInternalServerError)

	b, err := testutil.ReadBody(body)
	c.Assert(err, checker.IsNil)
	c.Assert(string(b[:]), checker.Contains, "invalid restart policy")
}

func (s *DockerSuite) TestContainerAPIRestartPolicyRetryMismatch(c *check.C) {
	config := `{
		"Image": "busybox",
		"HostConfig": {
			"RestartPolicy": {
				"Name": "always",
				"MaximumRetryCount": 2
			}
		}
	}`

	res, body, err := request.Post("/containers/create", request.RawString(config), request.JSON)
	c.Assert(err, checker.IsNil)
	c.Assert(res.StatusCode, checker.Equals, http.StatusInternalServerError)

	b, err := testutil.ReadBody(body)
	c.Assert(err, checker.IsNil)
	c.Assert(string(b[:]), checker.Contains, "maximum retry count cannot be used with restart policy")
}

func (s *DockerSuite) TestContainerAPIRestartPolicyNegativeRetryCount(c *check.C) {
	config := `{
		"Image": "busybox",
		"HostConfig": {
			"RestartPolicy": {
				"Name": "on-failure",
				"MaximumRetryCount": -2
			}
		}
	}`

	res, body, err := request.Post("/containers/create", request.RawString(config), request.JSON)
	c.Assert(err, checker.IsNil)
	c.Assert(res.StatusCode, checker.Equals, http.StatusInternalServerError)

	b, err := testutil.ReadBody(body)
	c.Assert(err, checker.IsNil)
	c.Assert(string(b[:]), checker.Contains, "maximum retry count cannot be negative")
}

func (s *DockerSuite) TestContainerAPIRestartPolicyDefaultRetryCount(c *check.C) {
	config := `{
		"Image": "busybox",
		"HostConfig": {
			"RestartPolicy": {
				"Name": "on-failure",
				"MaximumRetryCount": 0
			}
		}
	}`

	res, _, err := request.Post("/containers/create", request.RawString(config), request.JSON)
	c.Assert(err, checker.IsNil)
	c.Assert(res.StatusCode, checker.Equals, http.StatusCreated)
}

// Issue 7941 - test to make sure a "null" in JSON is just ignored.
// W/o this fix a null in JSON would be parsed into a string var as "null"
func (s *DockerSuite) TestContainerAPIPostCreateNull(c *check.C) {
	config := `{
		"Hostname":"",
		"Domainname":"",
		"Memory":0,
		"MemorySwap":0,
		"CpuShares":0,
		"Cpuset":null,
		"AttachStdin":true,
		"AttachStdout":true,
		"AttachStderr":true,
		"ExposedPorts":{},
		"Tty":true,
		"OpenStdin":true,
		"StdinOnce":true,
		"Env":[],
		"Cmd":"ls",
		"Image":"busybox",
		"Volumes":{},
		"WorkingDir":"",
		"Entrypoint":null,
		"NetworkDisabled":false,
		"OnBuild":null}`

	res, body, err := request.Post("/containers/create", request.RawString(config), request.JSON)
	c.Assert(err, checker.IsNil)
	c.Assert(res.StatusCode, checker.Equals, http.StatusCreated)

	b, err := testutil.ReadBody(body)
	c.Assert(err, checker.IsNil)
	type createResp struct {
		ID string
	}
	var container createResp
	c.Assert(json.Unmarshal(b, &container), checker.IsNil)
	out := inspectField(c, container.ID, "HostConfig.CpusetCpus")
	c.Assert(out, checker.Equals, "")

	outMemory := inspectField(c, container.ID, "HostConfig.Memory")
	c.Assert(outMemory, checker.Equals, "0")
	outMemorySwap := inspectField(c, container.ID, "HostConfig.MemorySwap")
	c.Assert(outMemorySwap, checker.Equals, "0")
}

func (s *DockerSuite) TestCreateWithTooLowMemoryLimit(c *check.C) {
	// TODO Windows: Port once memory is supported
	testRequires(c, DaemonIsLinux)
	config := `{
		"Image":     "busybox",
		"Cmd":       "ls",
		"OpenStdin": true,
		"CpuShares": 100,
		"Memory":    524287
	}`

	res, body, err := request.Post("/containers/create", request.RawString(config), request.JSON)
	c.Assert(err, checker.IsNil)
	b, err2 := testutil.ReadBody(body)
	c.Assert(err2, checker.IsNil)

	c.Assert(res.StatusCode, checker.Equals, http.StatusInternalServerError)
	c.Assert(string(b), checker.Contains, "Minimum memory limit allowed is 4MB")
}

func (s *DockerSuite) TestContainerAPIRename(c *check.C) {
	out, _ := dockerCmd(c, "run", "--name", "TestContainerAPIRename", "-d", "busybox", "sh")

	containerID := strings.TrimSpace(out)
	newName := "TestContainerAPIRenameNew"
	statusCode, _, err := request.SockRequest("POST", "/containers/"+containerID+"/rename?name="+newName, nil, daemonHost())
	c.Assert(err, checker.IsNil)
	// 204 No Content is expected, not 200
	c.Assert(statusCode, checker.Equals, http.StatusNoContent)

	name := inspectField(c, containerID, "Name")
	c.Assert(name, checker.Equals, "/"+newName, check.Commentf("Failed to rename container"))
}

func (s *DockerSuite) TestContainerAPIKill(c *check.C) {
	name := "test-api-kill"
	runSleepingContainer(c, "-i", "--name", name)

	status, _, err := request.SockRequest("POST", "/containers/"+name+"/kill", nil, daemonHost())
	c.Assert(err, checker.IsNil)
	c.Assert(status, checker.Equals, http.StatusNoContent)

	state := inspectField(c, name, "State.Running")
	c.Assert(state, checker.Equals, "false", check.Commentf("got wrong State from container %s: %q", name, state))
}

func (s *DockerSuite) TestContainerAPIRestart(c *check.C) {
	name := "test-api-restart"
	runSleepingContainer(c, "-di", "--name", name)

	status, _, err := request.SockRequest("POST", "/containers/"+name+"/restart?t=1", nil, daemonHost())
	c.Assert(err, checker.IsNil)
	c.Assert(status, checker.Equals, http.StatusNoContent)
	c.Assert(waitInspect(name, "{{ .State.Restarting  }} {{ .State.Running  }}", "false true", 15*time.Second), checker.IsNil)
}

func (s *DockerSuite) TestContainerAPIRestartNotimeoutParam(c *check.C) {
	name := "test-api-restart-no-timeout-param"
	out := runSleepingContainer(c, "-di", "--name", name)
	id := strings.TrimSpace(out)
	c.Assert(waitRun(id), checker.IsNil)

	status, _, err := request.SockRequest("POST", "/containers/"+name+"/restart", nil, daemonHost())
	c.Assert(err, checker.IsNil)
	c.Assert(status, checker.Equals, http.StatusNoContent)
	c.Assert(waitInspect(name, "{{ .State.Restarting  }} {{ .State.Running  }}", "false true", 15*time.Second), checker.IsNil)
}

func (s *DockerSuite) TestContainerAPIStart(c *check.C) {
	name := "testing-start"
	config := map[string]interface{}{
		"Image":     "busybox",
		"Cmd":       append([]string{"/bin/sh", "-c"}, sleepCommandForDaemonPlatform()...),
		"OpenStdin": true,
	}

	status, _, err := request.SockRequest("POST", "/containers/create?name="+name, config, daemonHost())
	c.Assert(err, checker.IsNil)
	c.Assert(status, checker.Equals, http.StatusCreated)

	status, _, err = request.SockRequest("POST", "/containers/"+name+"/start", nil, daemonHost())
	c.Assert(err, checker.IsNil)
	c.Assert(status, checker.Equals, http.StatusNoContent)

	// second call to start should give 304
	status, _, err = request.SockRequest("POST", "/containers/"+name+"/start", nil, daemonHost())
	c.Assert(err, checker.IsNil)

	// TODO(tibor): figure out why this doesn't work on windows
	if testEnv.LocalDaemon() {
		c.Assert(status, checker.Equals, http.StatusNotModified)
	}
}

func (s *DockerSuite) TestContainerAPIStop(c *check.C) {
	name := "test-api-stop"
	runSleepingContainer(c, "-i", "--name", name)

	status, _, err := request.SockRequest("POST", "/containers/"+name+"/stop?t=30", nil, daemonHost())
	c.Assert(err, checker.IsNil)
	c.Assert(status, checker.Equals, http.StatusNoContent)
	c.Assert(waitInspect(name, "{{ .State.Running  }}", "false", 60*time.Second), checker.IsNil)

	// second call to start should give 304
	status, _, err = request.SockRequest("POST", "/containers/"+name+"/stop?t=30", nil, daemonHost())
	c.Assert(err, checker.IsNil)
	c.Assert(status, checker.Equals, http.StatusNotModified)
}

func (s *DockerSuite) TestContainerAPIWait(c *check.C) {
	name := "test-api-wait"

	sleepCmd := "/bin/sleep"
	if testEnv.DaemonPlatform() == "windows" {
		sleepCmd = "sleep"
	}
	dockerCmd(c, "run", "--name", name, "busybox", sleepCmd, "2")

	status, body, err := request.SockRequest("POST", "/containers/"+name+"/wait", nil, daemonHost())
	c.Assert(err, checker.IsNil)
	c.Assert(status, checker.Equals, http.StatusOK)
	c.Assert(waitInspect(name, "{{ .State.Running  }}", "false", 60*time.Second), checker.IsNil)

	var waitres containertypes.ContainerWaitOKBody
	c.Assert(json.Unmarshal(body, &waitres), checker.IsNil)
	c.Assert(waitres.StatusCode, checker.Equals, int64(0))
}

func (s *DockerSuite) TestContainerAPICopyNotExistsAnyMore(c *check.C) {
	name := "test-container-api-copy"
	dockerCmd(c, "run", "--name", name, "busybox", "touch", "/test.txt")

	postData := types.CopyConfig{
		Resource: "/test.txt",
	}

	status, _, err := request.SockRequest("POST", "/containers/"+name+"/copy", postData, daemonHost())
	c.Assert(err, checker.IsNil)
	c.Assert(status, checker.Equals, http.StatusNotFound)
}

func (s *DockerSuite) TestContainerAPICopyPre124(c *check.C) {
	testRequires(c, DaemonIsLinux) // Windows only supports 1.25 or later
	name := "test-container-api-copy"
	dockerCmd(c, "run", "--name", name, "busybox", "touch", "/test.txt")

	postData := types.CopyConfig{
		Resource: "/test.txt",
	}

	status, body, err := request.SockRequest("POST", "/v1.23/containers/"+name+"/copy", postData, daemonHost())
	c.Assert(err, checker.IsNil)
	c.Assert(status, checker.Equals, http.StatusOK)

	found := false
	for tarReader := tar.NewReader(bytes.NewReader(body)); ; {
		h, err := tarReader.Next()
		if err != nil {
			if err == io.EOF {
				break
			}
			c.Fatal(err)
		}
		if h.Name == "test.txt" {
			found = true
			break
		}
	}
	c.Assert(found, checker.True)
}

func (s *DockerSuite) TestContainerAPICopyResourcePathEmptyPr124(c *check.C) {
	testRequires(c, DaemonIsLinux) // Windows only supports 1.25 or later
	name := "test-container-api-copy-resource-empty"
	dockerCmd(c, "run", "--name", name, "busybox", "touch", "/test.txt")

	postData := types.CopyConfig{
		Resource: "",
	}

	status, body, err := request.SockRequest("POST", "/v1.23/containers/"+name+"/copy", postData, daemonHost())
	c.Assert(err, checker.IsNil)
	c.Assert(status, checker.Equals, http.StatusInternalServerError)
	c.Assert(string(body), checker.Matches, "Path cannot be empty\n")
}

func (s *DockerSuite) TestContainerAPICopyResourcePathNotFoundPre124(c *check.C) {
	testRequires(c, DaemonIsLinux) // Windows only supports 1.25 or later
	name := "test-container-api-copy-resource-not-found"
	dockerCmd(c, "run", "--name", name, "busybox")

	postData := types.CopyConfig{
		Resource: "/notexist",
	}

	status, body, err := request.SockRequest("POST", "/v1.23/containers/"+name+"/copy", postData, daemonHost())
	c.Assert(err, checker.IsNil)
	c.Assert(status, checker.Equals, http.StatusInternalServerError)
	c.Assert(string(body), checker.Matches, "Could not find the file /notexist in container "+name+"\n")
}

func (s *DockerSuite) TestContainerAPICopyContainerNotFoundPr124(c *check.C) {
	testRequires(c, DaemonIsLinux) // Windows only supports 1.25 or later
	postData := types.CopyConfig{
		Resource: "/something",
	}

	status, _, err := request.SockRequest("POST", "/v1.23/containers/notexists/copy", postData, daemonHost())
	c.Assert(err, checker.IsNil)
	c.Assert(status, checker.Equals, http.StatusNotFound)
}

func (s *DockerSuite) TestContainerAPIDelete(c *check.C) {
	out := runSleepingContainer(c)

	id := strings.TrimSpace(out)
	c.Assert(waitRun(id), checker.IsNil)

	dockerCmd(c, "stop", id)

	status, _, err := request.SockRequest("DELETE", "/containers/"+id, nil, daemonHost())
	c.Assert(err, checker.IsNil)
	c.Assert(status, checker.Equals, http.StatusNoContent)
}

func (s *DockerSuite) TestContainerAPIDeleteNotExist(c *check.C) {
	status, body, err := request.SockRequest("DELETE", "/containers/doesnotexist", nil, daemonHost())
	c.Assert(err, checker.IsNil)
	c.Assert(status, checker.Equals, http.StatusNotFound)
	c.Assert(getErrorMessage(c, body), checker.Matches, "No such container: doesnotexist")
}

func (s *DockerSuite) TestContainerAPIDeleteForce(c *check.C) {
	out := runSleepingContainer(c)

	id := strings.TrimSpace(out)
	c.Assert(waitRun(id), checker.IsNil)

	status, _, err := request.SockRequest("DELETE", "/containers/"+id+"?force=1", nil, daemonHost())
	c.Assert(err, checker.IsNil)
	c.Assert(status, checker.Equals, http.StatusNoContent)
}

func (s *DockerSuite) TestContainerAPIDeleteRemoveLinks(c *check.C) {
	// Windows does not support links
	testRequires(c, DaemonIsLinux)
	out, _ := dockerCmd(c, "run", "-d", "--name", "tlink1", "busybox", "top")

	id := strings.TrimSpace(out)
	c.Assert(waitRun(id), checker.IsNil)

	out, _ = dockerCmd(c, "run", "--link", "tlink1:tlink1", "--name", "tlink2", "-d", "busybox", "top")

	id2 := strings.TrimSpace(out)
	c.Assert(waitRun(id2), checker.IsNil)

	links := inspectFieldJSON(c, id2, "HostConfig.Links")
	c.Assert(links, checker.Equals, "[\"/tlink1:/tlink2/tlink1\"]", check.Commentf("expected to have links between containers"))

	status, b, err := request.SockRequest("DELETE", "/containers/tlink2/tlink1?link=1", nil, daemonHost())
	c.Assert(err, check.IsNil)
	c.Assert(status, check.Equals, http.StatusNoContent, check.Commentf(string(b)))

	linksPostRm := inspectFieldJSON(c, id2, "HostConfig.Links")
	c.Assert(linksPostRm, checker.Equals, "null", check.Commentf("call to api deleteContainer links should have removed the specified links"))
}

func (s *DockerSuite) TestContainerAPIDeleteConflict(c *check.C) {
	out := runSleepingContainer(c)

	id := strings.TrimSpace(out)
	c.Assert(waitRun(id), checker.IsNil)

	status, _, err := request.SockRequest("DELETE", "/containers/"+id, nil, daemonHost())
	c.Assert(err, checker.IsNil)
	c.Assert(status, checker.Equals, http.StatusConflict)
}

func (s *DockerSuite) TestContainerAPIDeleteRemoveVolume(c *check.C) {
	testRequires(c, SameHostDaemon)

	vol := "/testvolume"
	if testEnv.DaemonPlatform() == "windows" {
		vol = `c:\testvolume`
	}

	out := runSleepingContainer(c, "-v", vol)

	id := strings.TrimSpace(out)
	c.Assert(waitRun(id), checker.IsNil)

	source, err := inspectMountSourceField(id, vol)
	_, err = os.Stat(source)
	c.Assert(err, checker.IsNil)

	status, _, err := request.SockRequest("DELETE", "/containers/"+id+"?v=1&force=1", nil, daemonHost())
	c.Assert(err, checker.IsNil)
	c.Assert(status, checker.Equals, http.StatusNoContent)
	_, err = os.Stat(source)
	c.Assert(os.IsNotExist(err), checker.True, check.Commentf("expected to get ErrNotExist error, got %v", err))
}

// Regression test for https://github.com/docker/docker/issues/6231
func (s *DockerSuite) TestContainerAPIChunkedEncoding(c *check.C) {

	config := map[string]interface{}{
		"Image":     "busybox",
		"Cmd":       append([]string{"/bin/sh", "-c"}, sleepCommandForDaemonPlatform()...),
		"OpenStdin": true,
	}

	resp, _, err := request.Post("/containers/create", request.JSONBody(config), func(req *http.Request) error {
		// This is a cheat to make the http request do chunked encoding
		// Otherwise (just setting the Content-Encoding to chunked) net/http will overwrite
		// https://golang.org/src/pkg/net/http/request.go?s=11980:12172
		req.ContentLength = -1
		return nil
	})
	c.Assert(err, checker.IsNil, check.Commentf("error creating container with chunked encoding"))
	defer resp.Body.Close()
	c.Assert(resp.StatusCode, checker.Equals, http.StatusCreated)
}

func (s *DockerSuite) TestContainerAPIPostContainerStop(c *check.C) {
	out := runSleepingContainer(c)

	containerID := strings.TrimSpace(out)
	c.Assert(waitRun(containerID), checker.IsNil)

	statusCode, _, err := request.SockRequest("POST", "/containers/"+containerID+"/stop", nil, daemonHost())
	c.Assert(err, checker.IsNil)
	// 204 No Content is expected, not 200
	c.Assert(statusCode, checker.Equals, http.StatusNoContent)
	c.Assert(waitInspect(containerID, "{{ .State.Running  }}", "false", 60*time.Second), checker.IsNil)
}

// #14170
func (s *DockerSuite) TestPostContainerAPICreateWithStringOrSliceEntrypoint(c *check.C) {
	config := struct {
		Image      string
		Entrypoint string
		Cmd        []string
	}{"busybox", "echo", []string{"hello", "world"}}
	_, _, err := request.SockRequest("POST", "/containers/create?name=echotest", config, daemonHost())
	c.Assert(err, checker.IsNil)
	out, _ := dockerCmd(c, "start", "-a", "echotest")
	c.Assert(strings.TrimSpace(out), checker.Equals, "hello world")

	config2 := struct {
		Image      string
		Entrypoint []string
		Cmd        []string
	}{"busybox", []string{"echo"}, []string{"hello", "world"}}
	_, _, err = request.SockRequest("POST", "/containers/create?name=echotest2", config2, daemonHost())
	c.Assert(err, checker.IsNil)
	out, _ = dockerCmd(c, "start", "-a", "echotest2")
	c.Assert(strings.TrimSpace(out), checker.Equals, "hello world")
}

// #14170
func (s *DockerSuite) TestPostContainersCreateWithStringOrSliceCmd(c *check.C) {
	config := struct {
		Image      string
		Entrypoint string
		Cmd        string
	}{"busybox", "echo", "hello world"}
	_, _, err := request.SockRequest("POST", "/containers/create?name=echotest", config, daemonHost())
	c.Assert(err, checker.IsNil)
	out, _ := dockerCmd(c, "start", "-a", "echotest")
	c.Assert(strings.TrimSpace(out), checker.Equals, "hello world")

	config2 := struct {
		Image string
		Cmd   []string
	}{"busybox", []string{"echo", "hello", "world"}}
	_, _, err = request.SockRequest("POST", "/containers/create?name=echotest2", config2, daemonHost())
	c.Assert(err, checker.IsNil)
	out, _ = dockerCmd(c, "start", "-a", "echotest2")
	c.Assert(strings.TrimSpace(out), checker.Equals, "hello world")
}

// regression #14318
func (s *DockerSuite) TestPostContainersCreateWithStringOrSliceCapAddDrop(c *check.C) {
	// Windows doesn't support CapAdd/CapDrop
	testRequires(c, DaemonIsLinux)
	config := struct {
		Image   string
		CapAdd  string
		CapDrop string
	}{"busybox", "NET_ADMIN", "SYS_ADMIN"}
	status, _, err := request.SockRequest("POST", "/containers/create?name=capaddtest0", config, daemonHost())
	c.Assert(err, checker.IsNil)
	c.Assert(status, checker.Equals, http.StatusCreated)

	config2 := struct {
		Image   string
		CapAdd  []string
		CapDrop []string
	}{"busybox", []string{"NET_ADMIN", "SYS_ADMIN"}, []string{"SETGID"}}
	status, _, err = request.SockRequest("POST", "/containers/create?name=capaddtest1", config2, daemonHost())
	c.Assert(err, checker.IsNil)
	c.Assert(status, checker.Equals, http.StatusCreated)
}

// #14915
func (s *DockerSuite) TestContainerAPICreateNoHostConfig118(c *check.C) {
	testRequires(c, DaemonIsLinux) // Windows only support 1.25 or later
	config := struct {
		Image string
	}{"busybox"}
	status, _, err := request.SockRequest("POST", "/v1.18/containers/create", config, daemonHost())
	c.Assert(err, checker.IsNil)
	c.Assert(status, checker.Equals, http.StatusCreated)
}

// Ensure an error occurs when you have a container read-only rootfs but you
// extract an archive to a symlink in a writable volume which points to a
// directory outside of the volume.
func (s *DockerSuite) TestPutContainerArchiveErrSymlinkInVolumeToReadOnlyRootfs(c *check.C) {
	// Windows does not support read-only rootfs
	// Requires local volume mount bind.
	// --read-only + userns has remount issues
	testRequires(c, SameHostDaemon, NotUserNamespace, DaemonIsLinux)

	testVol := getTestDir(c, "test-put-container-archive-err-symlink-in-volume-to-read-only-rootfs-")
	defer os.RemoveAll(testVol)

	makeTestContentInDir(c, testVol)

	cID := makeTestContainer(c, testContainerOptions{
		readOnly: true,
		volumes:  defaultVolumes(testVol), // Our bind mount is at /vol2
	})

	// Attempt to extract to a symlink in the volume which points to a
	// directory outside the volume. This should cause an error because the
	// rootfs is read-only.
	query := make(url.Values, 1)
	query.Set("path", "/vol2/symlinkToAbsDir")
	urlPath := fmt.Sprintf("/v1.20/containers/%s/archive?%s", cID, query.Encode())

	statusCode, body, err := request.SockRequest("PUT", urlPath, nil, daemonHost())
	c.Assert(err, checker.IsNil)

	if !isCpCannotCopyReadOnly(fmt.Errorf(string(body))) {
		c.Fatalf("expected ErrContainerRootfsReadonly error, but got %d: %s", statusCode, string(body))
	}
}

func (s *DockerSuite) TestContainerAPIGetContainersJSONEmpty(c *check.C) {
	status, body, err := request.SockRequest("GET", "/containers/json?all=1", nil, daemonHost())
	c.Assert(err, checker.IsNil)
	c.Assert(status, checker.Equals, http.StatusOK)
	c.Assert(string(body), checker.Equals, "[]\n")
}

func (s *DockerSuite) TestPostContainersCreateWithWrongCpusetValues(c *check.C) {
	// Not supported on Windows
	testRequires(c, DaemonIsLinux)

	c1 := struct {
		Image      string
		CpusetCpus string
	}{"busybox", "1-42,,"}
	name := "wrong-cpuset-cpus"
	status, body, err := request.SockRequest("POST", "/containers/create?name="+name, c1, daemonHost())
	c.Assert(err, checker.IsNil)
	c.Assert(status, checker.Equals, http.StatusInternalServerError)
	expected := "Invalid value 1-42,, for cpuset cpus"
	c.Assert(getErrorMessage(c, body), checker.Equals, expected)

	c2 := struct {
		Image      string
		CpusetMems string
	}{"busybox", "42-3,1--"}
	name = "wrong-cpuset-mems"
	status, body, err = request.SockRequest("POST", "/containers/create?name="+name, c2, daemonHost())
	c.Assert(err, checker.IsNil)
	c.Assert(status, checker.Equals, http.StatusInternalServerError)
	expected = "Invalid value 42-3,1-- for cpuset mems"
	c.Assert(getErrorMessage(c, body), checker.Equals, expected)
}

func (s *DockerSuite) TestPostContainersCreateShmSizeNegative(c *check.C) {
	// ShmSize is not supported on Windows
	testRequires(c, DaemonIsLinux)
	config := map[string]interface{}{
		"Image":      "busybox",
		"HostConfig": map[string]interface{}{"ShmSize": -1},
	}

	status, body, err := request.SockRequest("POST", "/containers/create", config, daemonHost())
	c.Assert(err, check.IsNil)
	c.Assert(status, check.Equals, http.StatusInternalServerError)
	c.Assert(getErrorMessage(c, body), checker.Contains, "SHM size can not be less than 0")
}

func (s *DockerSuite) TestPostContainersCreateShmSizeHostConfigOmitted(c *check.C) {
	// ShmSize is not supported on Windows
	testRequires(c, DaemonIsLinux)
	var defaultSHMSize int64 = 67108864
	config := map[string]interface{}{
		"Image": "busybox",
		"Cmd":   "mount",
	}

	status, body, err := request.SockRequest("POST", "/containers/create", config, daemonHost())
	c.Assert(err, check.IsNil)
	c.Assert(status, check.Equals, http.StatusCreated)

	var container containertypes.ContainerCreateCreatedBody
	c.Assert(json.Unmarshal(body, &container), check.IsNil)

	status, body, err = request.SockRequest("GET", "/containers/"+container.ID+"/json", nil, daemonHost())
	c.Assert(err, check.IsNil)
	c.Assert(status, check.Equals, http.StatusOK)

	var containerJSON types.ContainerJSON
	c.Assert(json.Unmarshal(body, &containerJSON), check.IsNil)

	c.Assert(containerJSON.HostConfig.ShmSize, check.Equals, defaultSHMSize)

	out, _ := dockerCmd(c, "start", "-i", containerJSON.ID)
	shmRegexp := regexp.MustCompile(`shm on /dev/shm type tmpfs(.*)size=65536k`)
	if !shmRegexp.MatchString(out) {
		c.Fatalf("Expected shm of 64MB in mount command, got %v", out)
	}
}

func (s *DockerSuite) TestPostContainersCreateShmSizeOmitted(c *check.C) {
	// ShmSize is not supported on Windows
	testRequires(c, DaemonIsLinux)
	config := map[string]interface{}{
		"Image":      "busybox",
		"HostConfig": map[string]interface{}{},
		"Cmd":        "mount",
	}

	status, body, err := request.SockRequest("POST", "/containers/create", config, daemonHost())
	c.Assert(err, check.IsNil)
	c.Assert(status, check.Equals, http.StatusCreated)

	var container containertypes.ContainerCreateCreatedBody
	c.Assert(json.Unmarshal(body, &container), check.IsNil)

	status, body, err = request.SockRequest("GET", "/containers/"+container.ID+"/json", nil, daemonHost())
	c.Assert(err, check.IsNil)
	c.Assert(status, check.Equals, http.StatusOK)

	var containerJSON types.ContainerJSON
	c.Assert(json.Unmarshal(body, &containerJSON), check.IsNil)

	c.Assert(containerJSON.HostConfig.ShmSize, check.Equals, int64(67108864))

	out, _ := dockerCmd(c, "start", "-i", containerJSON.ID)
	shmRegexp := regexp.MustCompile(`shm on /dev/shm type tmpfs(.*)size=65536k`)
	if !shmRegexp.MatchString(out) {
		c.Fatalf("Expected shm of 64MB in mount command, got %v", out)
	}
}

func (s *DockerSuite) TestPostContainersCreateWithShmSize(c *check.C) {
	// ShmSize is not supported on Windows
	testRequires(c, DaemonIsLinux)
	config := map[string]interface{}{
		"Image":      "busybox",
		"Cmd":        "mount",
		"HostConfig": map[string]interface{}{"ShmSize": 1073741824},
	}

	status, body, err := request.SockRequest("POST", "/containers/create", config, daemonHost())
	c.Assert(err, check.IsNil)
	c.Assert(status, check.Equals, http.StatusCreated)

	var container containertypes.ContainerCreateCreatedBody
	c.Assert(json.Unmarshal(body, &container), check.IsNil)

	status, body, err = request.SockRequest("GET", "/containers/"+container.ID+"/json", nil, daemonHost())
	c.Assert(err, check.IsNil)
	c.Assert(status, check.Equals, http.StatusOK)

	var containerJSON types.ContainerJSON
	c.Assert(json.Unmarshal(body, &containerJSON), check.IsNil)

	c.Assert(containerJSON.HostConfig.ShmSize, check.Equals, int64(1073741824))

	out, _ := dockerCmd(c, "start", "-i", containerJSON.ID)
	shmRegex := regexp.MustCompile(`shm on /dev/shm type tmpfs(.*)size=1048576k`)
	if !shmRegex.MatchString(out) {
		c.Fatalf("Expected shm of 1GB in mount command, got %v", out)
	}
}

func (s *DockerSuite) TestPostContainersCreateMemorySwappinessHostConfigOmitted(c *check.C) {
	// Swappiness is not supported on Windows
	testRequires(c, DaemonIsLinux)
	config := map[string]interface{}{
		"Image": "busybox",
	}

	status, body, err := request.SockRequest("POST", "/containers/create", config, daemonHost())
	c.Assert(err, check.IsNil)
	c.Assert(status, check.Equals, http.StatusCreated)

	var container containertypes.ContainerCreateCreatedBody
	c.Assert(json.Unmarshal(body, &container), check.IsNil)

	status, body, err = request.SockRequest("GET", "/containers/"+container.ID+"/json", nil, daemonHost())
	c.Assert(err, check.IsNil)
	c.Assert(status, check.Equals, http.StatusOK)

	var containerJSON types.ContainerJSON
	c.Assert(json.Unmarshal(body, &containerJSON), check.IsNil)

	c.Assert(containerJSON.HostConfig.MemorySwappiness, check.IsNil)
}

// check validation is done daemon side and not only in cli
func (s *DockerSuite) TestPostContainersCreateWithOomScoreAdjInvalidRange(c *check.C) {
	// OomScoreAdj is not supported on Windows
	testRequires(c, DaemonIsLinux)

	config := struct {
		Image       string
		OomScoreAdj int
	}{"busybox", 1001}
	name := "oomscoreadj-over"
	status, b, err := request.SockRequest("POST", "/containers/create?name="+name, config, daemonHost())
	c.Assert(err, check.IsNil)
	c.Assert(status, check.Equals, http.StatusInternalServerError)

	expected := "Invalid value 1001, range for oom score adj is [-1000, 1000]"
	msg := getErrorMessage(c, b)
	if !strings.Contains(msg, expected) {
		c.Fatalf("Expected output to contain %q, got %q", expected, msg)
	}

	config = struct {
		Image       string
		OomScoreAdj int
	}{"busybox", -1001}
	name = "oomscoreadj-low"
	status, b, err = request.SockRequest("POST", "/containers/create?name="+name, config, daemonHost())
	c.Assert(err, check.IsNil)
	c.Assert(status, check.Equals, http.StatusInternalServerError)
	expected = "Invalid value -1001, range for oom score adj is [-1000, 1000]"
	msg = getErrorMessage(c, b)
	if !strings.Contains(msg, expected) {
		c.Fatalf("Expected output to contain %q, got %q", expected, msg)
	}
}

// test case for #22210 where an empty container name caused panic.
func (s *DockerSuite) TestContainerAPIDeleteWithEmptyName(c *check.C) {
	status, out, err := request.SockRequest("DELETE", "/containers/", nil, daemonHost())
	c.Assert(err, checker.IsNil)
	c.Assert(status, checker.Equals, http.StatusBadRequest)
	c.Assert(string(out), checker.Contains, "No container name or ID supplied")
}

func (s *DockerSuite) TestContainerAPIStatsWithNetworkDisabled(c *check.C) {
	// Problematic on Windows as Windows does not support stats
	testRequires(c, DaemonIsLinux)

	name := "testing-network-disabled"
	config := map[string]interface{}{
		"Image":           "busybox",
		"Cmd":             []string{"top"},
		"NetworkDisabled": true,
	}

	status, _, err := request.SockRequest("POST", "/containers/create?name="+name, config, daemonHost())
	c.Assert(err, checker.IsNil)
	c.Assert(status, checker.Equals, http.StatusCreated)

	status, _, err = request.SockRequest("POST", "/containers/"+name+"/start", nil, daemonHost())
	c.Assert(err, checker.IsNil)
	c.Assert(status, checker.Equals, http.StatusNoContent)

	c.Assert(waitRun(name), check.IsNil)

	type b struct {
		status int
		body   []byte
		err    error
	}
	bc := make(chan b, 1)
	go func() {
		status, body, err := request.SockRequest("GET", "/containers/"+name+"/stats", nil, daemonHost())
		bc <- b{status, body, err}
	}()

	// allow some time to stream the stats from the container
	time.Sleep(4 * time.Second)
	dockerCmd(c, "rm", "-f", name)

	// collect the results from the stats stream or timeout and fail
	// if the stream was not disconnected.
	select {
	case <-time.After(2 * time.Second):
		c.Fatal("stream was not closed after container was removed")
	case sr := <-bc:
		c.Assert(sr.err, checker.IsNil)
		c.Assert(sr.status, checker.Equals, http.StatusOK)

		// decode only one object from the stream
		var s *types.Stats
		dec := json.NewDecoder(bytes.NewBuffer(sr.body))
		c.Assert(dec.Decode(&s), checker.IsNil)
	}
}

func (s *DockerSuite) TestContainersAPICreateMountsValidation(c *check.C) {
	type m mounttypes.Mount
	type hc struct{ Mounts []m }
	type cfg struct {
		Image      string
		HostConfig hc
	}
	type testCase struct {
		config cfg
		status int
		msg    string
	}

	prefix, slash := getPrefixAndSlashFromDaemonPlatform()
	destPath := prefix + slash + "foo"
	notExistPath := prefix + slash + "notexist"

	cases := []testCase{
		{
			config: cfg{
				Image: "busybox",
				HostConfig: hc{
					Mounts: []m{{
						Type:   "notreal",
						Target: destPath}}}},
			status: http.StatusBadRequest,
			msg:    "mount type unknown",
		},
		{
			config: cfg{
				Image: "busybox",
				HostConfig: hc{
					Mounts: []m{{
						Type: "bind"}}}},
			status: http.StatusBadRequest,
			msg:    "Target must not be empty",
		},
		{
			config: cfg{
				Image: "busybox",
				HostConfig: hc{
					Mounts: []m{{
						Type:   "bind",
						Target: destPath}}}},
			status: http.StatusBadRequest,
			msg:    "Source must not be empty",
		},
		{
			config: cfg{
				Image: "busybox",
				HostConfig: hc{
					Mounts: []m{{
						Type:   "bind",
						Source: notExistPath,
						Target: destPath}}}},
			status: http.StatusBadRequest,
			msg:    "bind source path does not exist",
		},
		{
			config: cfg{
				Image: "busybox",
				HostConfig: hc{
					Mounts: []m{{
						Type: "volume"}}}},
			status: http.StatusBadRequest,
			msg:    "Target must not be empty",
		},
		{
			config: cfg{
				Image: "busybox",
				HostConfig: hc{
					Mounts: []m{{
						Type:   "volume",
						Source: "hello",
						Target: destPath}}}},
			status: http.StatusCreated,
			msg:    "",
		},
		{
			config: cfg{
				Image: "busybox",
				HostConfig: hc{
					Mounts: []m{{
						Type:   "volume",
						Source: "hello2",
						Target: destPath,
						VolumeOptions: &mounttypes.VolumeOptions{
							DriverConfig: &mounttypes.Driver{
								Name: "local"}}}}}},
			status: http.StatusCreated,
			msg:    "",
		},
	}

	if SameHostDaemon() {
		tmpDir, err := ioutils.TempDir("", "test-mounts-api")
		c.Assert(err, checker.IsNil)
		defer os.RemoveAll(tmpDir)
		cases = append(cases, []testCase{
			{
				config: cfg{
					Image: "busybox",
					HostConfig: hc{
						Mounts: []m{{
							Type:   "bind",
							Source: tmpDir,
							Target: destPath}}}},
				status: http.StatusCreated,
				msg:    "",
			},
			{
				config: cfg{
					Image: "busybox",
					HostConfig: hc{
						Mounts: []m{{
							Type:          "bind",
							Source:        tmpDir,
							Target:        destPath,
							VolumeOptions: &mounttypes.VolumeOptions{}}}}},
				status: http.StatusBadRequest,
				msg:    "VolumeOptions must not be specified",
			},
		}...)
	}

	if DaemonIsLinux() {
		cases = append(cases, []testCase{
			{
				config: cfg{
					Image: "busybox",
					HostConfig: hc{
						Mounts: []m{{
							Type:   "volume",
							Source: "hello3",
							Target: destPath,
							VolumeOptions: &mounttypes.VolumeOptions{
								DriverConfig: &mounttypes.Driver{
									Name:    "local",
									Options: map[string]string{"o": "size=1"}}}}}}},
				status: http.StatusCreated,
				msg:    "",
			},
			{
				config: cfg{
					Image: "busybox",
					HostConfig: hc{
						Mounts: []m{{
							Type:   "tmpfs",
							Target: destPath}}}},
				status: http.StatusCreated,
				msg:    "",
			},
			{
				config: cfg{
					Image: "busybox",
					HostConfig: hc{
						Mounts: []m{{
							Type:   "tmpfs",
							Target: destPath,
							TmpfsOptions: &mounttypes.TmpfsOptions{
								SizeBytes: 4096 * 1024,
								Mode:      0700,
							}}}}},
				status: http.StatusCreated,
				msg:    "",
			},

			{
				config: cfg{
					Image: "busybox",
					HostConfig: hc{
						Mounts: []m{{
							Type:   "tmpfs",
							Source: "/shouldnotbespecified",
							Target: destPath}}}},
				status: http.StatusBadRequest,
				msg:    "Source must not be specified",
			},
		}...)

	}

	for i, x := range cases {
		c.Logf("case %d", i)
		status, b, err := request.SockRequest("POST", "/containers/create", x.config, daemonHost())
		c.Assert(err, checker.IsNil)
		c.Assert(status, checker.Equals, x.status, check.Commentf("%s\n%v", string(b), cases[i].config))
		if len(x.msg) > 0 {
			c.Assert(string(b), checker.Contains, x.msg, check.Commentf("%v", cases[i].config))
		}
	}
}

func (s *DockerSuite) TestContainerAPICreateMountsBindRead(c *check.C) {
	testRequires(c, NotUserNamespace, SameHostDaemon)
	// also with data in the host side
	prefix, slash := getPrefixAndSlashFromDaemonPlatform()
	destPath := prefix + slash + "foo"
	tmpDir, err := ioutil.TempDir("", "test-mounts-api-bind")
	c.Assert(err, checker.IsNil)
	defer os.RemoveAll(tmpDir)
	err = ioutil.WriteFile(filepath.Join(tmpDir, "bar"), []byte("hello"), 666)
	c.Assert(err, checker.IsNil)

	data := map[string]interface{}{
		"Image":      "busybox",
		"Cmd":        []string{"/bin/sh", "-c", "cat /foo/bar"},
		"HostConfig": map[string]interface{}{"Mounts": []map[string]interface{}{{"Type": "bind", "Source": tmpDir, "Target": destPath}}},
	}
	status, resp, err := request.SockRequest("POST", "/containers/create?name=test", data, daemonHost())
	c.Assert(err, checker.IsNil, check.Commentf(string(resp)))
	c.Assert(status, checker.Equals, http.StatusCreated, check.Commentf(string(resp)))

	out, _ := dockerCmd(c, "start", "-a", "test")
	c.Assert(out, checker.Equals, "hello")
}

// Test Mounts comes out as expected for the MountPoint
func (s *DockerSuite) TestContainersAPICreateMountsCreate(c *check.C) {
	prefix, slash := getPrefixAndSlashFromDaemonPlatform()
	destPath := prefix + slash + "foo"

	var (
		testImg string
	)
	if testEnv.DaemonPlatform() != "windows" {
		testImg = "test-mount-config"
		buildImageSuccessfully(c, testImg, build.WithDockerfile(`
	FROM busybox
	RUN mkdir `+destPath+` && touch `+destPath+slash+`bar
	CMD cat `+destPath+slash+`bar
	`))
	} else {
		testImg = "busybox"
	}

	type testCase struct {
		cfg      mounttypes.Mount
		expected types.MountPoint
	}

	cases := []testCase{
		// use literal strings here for `Type` instead of the defined constants in the volume package to keep this honest
		// Validation of the actual `Mount` struct is done in another test is not needed here
		{mounttypes.Mount{Type: "volume", Target: destPath}, types.MountPoint{Driver: volume.DefaultDriverName, Type: "volume", RW: true, Destination: destPath}},
		{mounttypes.Mount{Type: "volume", Target: destPath + slash}, types.MountPoint{Driver: volume.DefaultDriverName, Type: "volume", RW: true, Destination: destPath}},
		{mounttypes.Mount{Type: "volume", Target: destPath, Source: "test1"}, types.MountPoint{Type: "volume", Name: "test1", RW: true, Destination: destPath}},
		{mounttypes.Mount{Type: "volume", Target: destPath, ReadOnly: true, Source: "test2"}, types.MountPoint{Type: "volume", Name: "test2", RW: false, Destination: destPath}},
		{mounttypes.Mount{Type: "volume", Target: destPath, Source: "test3", VolumeOptions: &mounttypes.VolumeOptions{DriverConfig: &mounttypes.Driver{Name: volume.DefaultDriverName}}}, types.MountPoint{Driver: volume.DefaultDriverName, Type: "volume", Name: "test3", RW: true, Destination: destPath}},
	}

	if SameHostDaemon() {
		// setup temp dir for testing binds
		tmpDir1, err := ioutil.TempDir("", "test-mounts-api-1")
		c.Assert(err, checker.IsNil)
		defer os.RemoveAll(tmpDir1)
		cases = append(cases, []testCase{
			{mounttypes.Mount{Type: "bind", Source: tmpDir1, Target: destPath}, types.MountPoint{Type: "bind", RW: true, Destination: destPath, Source: tmpDir1}},
			{mounttypes.Mount{Type: "bind", Source: tmpDir1, Target: destPath, ReadOnly: true}, types.MountPoint{Type: "bind", RW: false, Destination: destPath, Source: tmpDir1}},
		}...)

		// for modes only supported on Linux
		if DaemonIsLinux() {
			tmpDir3, err := ioutils.TempDir("", "test-mounts-api-3")
			c.Assert(err, checker.IsNil)
			defer os.RemoveAll(tmpDir3)

			c.Assert(mount.Mount(tmpDir3, tmpDir3, "none", "bind,rw"), checker.IsNil)
			c.Assert(mount.ForceMount("", tmpDir3, "none", "shared"), checker.IsNil)

			cases = append(cases, []testCase{
				{mounttypes.Mount{Type: "bind", Source: tmpDir3, Target: destPath}, types.MountPoint{Type: "bind", RW: true, Destination: destPath, Source: tmpDir3}},
				{mounttypes.Mount{Type: "bind", Source: tmpDir3, Target: destPath, ReadOnly: true}, types.MountPoint{Type: "bind", RW: false, Destination: destPath, Source: tmpDir3}},
				{mounttypes.Mount{Type: "bind", Source: tmpDir3, Target: destPath, ReadOnly: true, BindOptions: &mounttypes.BindOptions{Propagation: "shared"}}, types.MountPoint{Type: "bind", RW: false, Destination: destPath, Source: tmpDir3, Propagation: "shared"}},
			}...)
		}
	}

	if testEnv.DaemonPlatform() != "windows" { // Windows does not support volume populate
		cases = append(cases, []testCase{
			{mounttypes.Mount{Type: "volume", Target: destPath, VolumeOptions: &mounttypes.VolumeOptions{NoCopy: true}}, types.MountPoint{Driver: volume.DefaultDriverName, Type: "volume", RW: true, Destination: destPath}},
			{mounttypes.Mount{Type: "volume", Target: destPath + slash, VolumeOptions: &mounttypes.VolumeOptions{NoCopy: true}}, types.MountPoint{Driver: volume.DefaultDriverName, Type: "volume", RW: true, Destination: destPath}},
			{mounttypes.Mount{Type: "volume", Target: destPath, Source: "test4", VolumeOptions: &mounttypes.VolumeOptions{NoCopy: true}}, types.MountPoint{Type: "volume", Name: "test4", RW: true, Destination: destPath}},
			{mounttypes.Mount{Type: "volume", Target: destPath, Source: "test5", ReadOnly: true, VolumeOptions: &mounttypes.VolumeOptions{NoCopy: true}}, types.MountPoint{Type: "volume", Name: "test5", RW: false, Destination: destPath}},
		}...)
	}

	type wrapper struct {
		containertypes.Config
		HostConfig containertypes.HostConfig
	}
	type createResp struct {
		ID string `json:"Id"`
	}
	for i, x := range cases {
		c.Logf("case %d - config: %v", i, x.cfg)
		status, data, err := request.SockRequest("POST", "/containers/create", wrapper{containertypes.Config{Image: testImg}, containertypes.HostConfig{Mounts: []mounttypes.Mount{x.cfg}}}, daemonHost())
		c.Assert(err, checker.IsNil, check.Commentf(string(data)))
		c.Assert(status, checker.Equals, http.StatusCreated, check.Commentf(string(data)))

		var resp createResp
		err = json.Unmarshal(data, &resp)
		c.Assert(err, checker.IsNil, check.Commentf(string(data)))
		id := resp.ID

		var mps []types.MountPoint
		err = json.NewDecoder(strings.NewReader(inspectFieldJSON(c, id, "Mounts"))).Decode(&mps)
		c.Assert(err, checker.IsNil)
		c.Assert(mps, checker.HasLen, 1)
		c.Assert(mps[0].Destination, checker.Equals, x.expected.Destination)

		if len(x.expected.Source) > 0 {
			c.Assert(mps[0].Source, checker.Equals, x.expected.Source)
		}
		if len(x.expected.Name) > 0 {
			c.Assert(mps[0].Name, checker.Equals, x.expected.Name)
		}
		if len(x.expected.Driver) > 0 {
			c.Assert(mps[0].Driver, checker.Equals, x.expected.Driver)
		}
		c.Assert(mps[0].RW, checker.Equals, x.expected.RW)
		c.Assert(mps[0].Type, checker.Equals, x.expected.Type)
		c.Assert(mps[0].Mode, checker.Equals, x.expected.Mode)
		if len(x.expected.Propagation) > 0 {
			c.Assert(mps[0].Propagation, checker.Equals, x.expected.Propagation)
		}

		out, _, err := dockerCmdWithError("start", "-a", id)
		if (x.cfg.Type != "volume" || (x.cfg.VolumeOptions != nil && x.cfg.VolumeOptions.NoCopy)) && testEnv.DaemonPlatform() != "windows" {
			c.Assert(err, checker.NotNil, check.Commentf("%s\n%v", out, mps[0]))
		} else {
			c.Assert(err, checker.IsNil, check.Commentf("%s\n%v", out, mps[0]))
		}

		dockerCmd(c, "rm", "-fv", id)
		if x.cfg.Type == "volume" && len(x.cfg.Source) > 0 {
			// This should still exist even though we removed the container
			dockerCmd(c, "volume", "inspect", mps[0].Name)
		} else {
			// This should be removed automatically when we removed the container
			out, _, err := dockerCmdWithError("volume", "inspect", mps[0].Name)
			c.Assert(err, checker.NotNil, check.Commentf(out))
		}
	}
}

func (s *DockerSuite) TestContainersAPICreateMountsTmpfs(c *check.C) {
	testRequires(c, DaemonIsLinux)
	type testCase struct {
		cfg             map[string]interface{}
		expectedOptions []string
	}
	target := "/foo"
	cases := []testCase{
		{
			cfg: map[string]interface{}{
				"Type":   "tmpfs",
				"Target": target},
			expectedOptions: []string{"rw", "nosuid", "nodev", "noexec", "relatime"},
		},
		{
			cfg: map[string]interface{}{
				"Type":   "tmpfs",
				"Target": target,
				"TmpfsOptions": map[string]interface{}{
					"SizeBytes": 4096 * 1024, "Mode": 0700}},
			expectedOptions: []string{"rw", "nosuid", "nodev", "noexec", "relatime", "size=4096k", "mode=700"},
		},
	}

	for i, x := range cases {
		cName := fmt.Sprintf("test-tmpfs-%d", i)
		data := map[string]interface{}{
			"Image": "busybox",
			"Cmd": []string{"/bin/sh", "-c",
				fmt.Sprintf("mount | grep 'tmpfs on %s'", target)},
			"HostConfig": map[string]interface{}{"Mounts": []map[string]interface{}{x.cfg}},
		}
		status, resp, err := request.SockRequest("POST", "/containers/create?name="+cName, data, daemonHost())
		c.Assert(err, checker.IsNil, check.Commentf(string(resp)))
		c.Assert(status, checker.Equals, http.StatusCreated, check.Commentf(string(resp)))
		out, _ := dockerCmd(c, "start", "-a", cName)
		for _, option := range x.expectedOptions {
			c.Assert(out, checker.Contains, option)
		}
	}
}

// Regression test for #33334
// Makes sure that when a container which has a custom stop signal + restart=always
// gets killed (with SIGKILL) by the kill API, that the restart policy is cancelled.
func (s *DockerSuite) TestContainerKillCustomStopSignal(c *check.C) {
	id := strings.TrimSpace(runSleepingContainer(c, "--stop-signal=SIGTERM", "--restart=always"))
	res, _, err := request.Post("/containers/" + id + "/kill")
	c.Assert(err, checker.IsNil)
	defer res.Body.Close()

	b, err := ioutil.ReadAll(res.Body)
	c.Assert(res.StatusCode, checker.Equals, http.StatusNoContent, check.Commentf(string(b)))
	err = waitInspect(id, "{{.State.Running}} {{.State.Restarting}}", "false false", 30*time.Second)
	c.Assert(err, checker.IsNil)
}
